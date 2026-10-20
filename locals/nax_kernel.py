"""Small-M 4-bit matmul on the M5's neural accelerators (NAX), built from MLX's own pieces.

Why: MLX's NAX quantized matmul (affine_qmm_t_nax in quantized_nax.h) tiles 64 rows at a time (BM = 64), and
for fewer rows MLX uses its vector kernels, so a decode step of 6 to 32 requests either leaves the neural
accelerators idle or computes 64 rows for 16. The NAX fragment is 16 x 16 (BaseNAXFrag in steel/gemm/nax.h),
so BM = 16 is the smallest tile: every weight is read and dequantized once and applied to up to 16 rows.
This mirrors locals/mma_kernel.py (the same idea with 8 x 8 simdgroup matrices, BM = 8) and MLX's
qmm_t_nax_tgp_impl, with K, N, M as plain ints (custom kernels pass them as template constants), the output
type U (float32 for split-K partials), and a split-K slice per threadgroup z.

The register variant (configs starting with "r", locals_qmm_naxr) is what wins on an M5 Pro: no weight tile
in threadgroup memory, each thread dequantizes its own fragment elements, and the accelerators' operands
and accumulators stay in their cooperative tensors for the whole K loop. Qwen3.5-4B, one decode step's
4-bit matmuls: about 12 ms at 4 to 16 rows (MLX's kernels 13-32 ms, frag 14.5-24 ms), 16 ms at 24-32.

Contract with locals/kernels.py and the scripts of the note:
  available() -> bool            compiles and checks a tiny case once (False on GPUs without NAX)
  CANDIDATES                     configs (BM, BN, BK, WM, WN, SPLIT) or ("r", SN, WN, WK, TK, SPLIT)
  launch(M, K, N, dtype, *cfg)   f(x, w, scales, biases), template and grid computed once
  supported(M, K, N, cfg) -> bool
  qmm_nax(x, w, scales, biases, *cfg) -> y = x @ dequant(w).T (4-bit affine, group 64), x: (M, K)
"""

from __future__ import annotations

import re
from pathlib import Path

import mlx.core as mx

_INC = Path(mx.__file__).parent / "include"
_INCLUDE_RE = re.compile(r'^\s*#include\s+"(mlx/[^"]+)"\s*$', re.M)


def _inline(rel: str, seen: set) -> str:
    if rel in seen:
        return ""
    seen.add(rel)
    text = (_INC / rel).read_text()
    text = text.replace("#pragma once", "")
    return _INCLUDE_RE.sub(lambda m: _inline(m.group(1), seen), text)


def _header() -> str:
    # utils.h (and what it pulls in) is already part of every custom kernel (as in mma_kernel.py).
    seen = {"mlx/backend/metal/kernels/utils.h", "mlx/backend/metal/kernels/bf16.h",
            "mlx/backend/metal/kernels/bf16_math.h", "mlx/backend/metal/kernels/complex.h",
            "mlx/backend/metal/kernels/defines.h", "mlx/backend/metal/kernels/logging.h"}
    parts = [
        _inline("mlx/backend/metal/kernels/steel/gemm/gemm.h", seen),  # mlx::steel, dispatch_bool
        _inline("mlx/backend/metal/kernels/steel/gemm/nax.h", seen),  # NAXTile, tile_matmad_nax (Metal 4)
        _inline("mlx/backend/metal/kernels/quantized_nax.h", seen),  # QuantizedBlockLoader, pack factors
    ]
    return "\n".join(parts) + _IMPL


_IMPL = r"""
// qmm_t_nax_tgp_impl from MLX's quantized_nax.h with K, N, M as ints, an output type U, and split-K.
template <typename T, typename U, int group_size, int bits, int BM, int BK, int BN, int WM, int WN>
METAL_FUNC void locals_qmm_t_nax(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device U* y,
    threadgroup T* Ws,
    const int K,
    const int N,
    const int M,
    const int K_eff,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const int k0 = tid.z * K_eff;  // split-K slice

  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K) + k0;
  wl += y_col * K_w + k0 * bytes_per_pack / pack_factor;
  scales += y_col * K_g + k0 / group_size;
  biases += y_col * K_g + k0 / group_size;
  y += tid.z * static_cast<int64_t>(M) * N + y_row * static_cast<int64_t>(N) + y_col;

  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);
  const short sgp_sm = min(int(SM), M - (y_row + tm));  // rows this simdgroup has (fewer than 16 at the end)

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();
  x += tm * K;

  dispatch_bool(sgp_sm == SM, [&](auto kAlignedM) {
    for (int k = 0; k < K_eff; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_w.load_unsafe();  // N % BN == 0 (supported() checks it)
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < BK; kk1 += SK) {
        NAXTile<T, TM, TK> Atile;
        NAXTile<T, TN, TK> Btile;
        volatile int compiler_barrier;
        if constexpr (kAlignedM.value) {
          Atile.load(x + kk1, K);
        } else {
          Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
        }
        Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
        tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
        (void)compiler_barrier;
      }
      x += BK;
      loader_w.next();
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if constexpr (kAlignedM.value) {
      Dtile.store(y + tm * N + tn, N);
    } else {
      Dtile.store_safe(y + tm * N + tn, N, short2(SN, sgp_sm));
    }
  });
}

// Weights dequantized in registers, no threadgroup tile. A fragment's thread holds 2 rows x 4 consecutive
// columns; the dot product does not depend on the order of K, so fragment kk's columns of the thread with
// column quad q stand for K positions q * 4 * TK + kk * 4 + j of a 16 * TK chunk (the same for x and w).
// Each thread then reads 4 * TK consecutive 4-bit weights per row at once (8 or 16 bytes, one group),
// and nothing waits on a barrier until the WK simdgroups splitting K add up their partial sums.
template <typename T, typename U, int SN, int WN, int WK, int TK, int K, int N, int M, int SPLIT>
METAL_FUNC void locals_qmm_naxr(
    const device T* x,
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    device U* y,
    threadgroup float* red,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  using Frag = mlx::steel::BaseNAXFrag;
  using FragC = typename Frag::template dtype_frag_t<float>;
  constexpr int TN = SN / 16;
  constexpr int TM = (M + 15) / 16;
  constexpr int KC = 16 * TK;
  constexpr int K_eff = K / SPLIT;
  constexpr int K_g = K / 64;
  constexpr int E = TM * TN * 8;

  const int wn = simd_gid % WN;
  const int wk = simd_gid / WN;
  const int n0 = (int(tid.x) * WN + wn) * SN;
  const int kz = int(tid.z) * K_eff;
  const short2 sc = Frag::get_coord();
  const short fn = sc.x;
  const short fm = sc.y;
  const short q = fn / 4;

  FragC C[TM][TN];
  for (short tm = 0; tm < TM; tm++)
    for (short t = 0; t < TN; t++)
      C[tm][t] = FragC(0);

  const int nr = n0 + fm;  // this thread's first weight row; the others are 8, 16, 24 ... further
  const device uint8_t* wb = (const device uint8_t*)w + nr * (K / 2);
  const device T* sb = scales + nr * K_g;
  const device T* bb = biases + nr * K_g;
  const device T* xb = x + fm * K;  // rows fm + 8 i + 16 tm of x

  // The accelerators' 16 x 32 x 16 product, its operands and accumulators kept in their own registers for
  // the whole loop (MLX's NAXFrag::mma copies the accumulator in and out at every call). A cooperative
  // tensor's elements are ordered as NAXFrag's: ct_b holds two 16 x 16 fragments, ct_c two.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  using CtA = decltype(op.template get_left_input_cooperative_tensor<T, T, float>());
  using CtB = decltype(op.template get_right_input_cooperative_tensor<T, T, float>());
  constexpr int P = TN / 2;  // (cooperative tensors cannot form arrays: up to 2 x 2 named ones)
  static_assert(TM <= 2 && P <= 2, "up to 32 rows and 64 columns per simdgroup");
  auto cc00 = op.template get_destination_cooperative_tensor<CtA, CtB, float>();
  auto cc01 = op.template get_destination_cooperative_tensor<CtA, CtB, float>();
  auto cc10 = op.template get_destination_cooperative_tensor<CtA, CtB, float>();
  auto cc11 = op.template get_destination_cooperative_tensor<CtA, CtB, float>();
  STEEL_PRAGMA_UNROLL
  for (short e = 0; e < 16; e++) {
    cc00[e] = 0.0f;
    cc01[e] = 0.0f;
    cc10[e] = 0.0f;
    cc11[e] = 0.0f;
  }

  for (int k = kz + wk * KC; k < kz + K_eff; k += WK * KC) {
    const int kq = k + q * 4 * TK;
    uint32_t wv[TN][2][TK / 2];
    float s[TN][2], b[TN][2];
    STEEL_PRAGMA_UNROLL
    for (short t = 0; t < TN; t++)
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 2; i++) {
        const int r = t * 16 + i * 8;
        const device uint8_t* src = wb + r * (K / 2) + kq / 2;
        if constexpr (TK == 4) {
          uint2 v = *(const device uint2*)src;
          wv[t][i][0] = v.x;
          wv[t][i][1] = v.y;
        } else {
          STEEL_PRAGMA_UNROLL
          for (short h = 0; h < TK / 8; h++) {
            uint4 v = *(const device uint4*)(src + 16 * h);
            wv[t][i][4 * h] = v.x;
            wv[t][i][4 * h + 1] = v.y;
            wv[t][i][4 * h + 2] = v.z;
            wv[t][i][4 * h + 3] = v.w;
          }
        }
        s[t][i] = float(sb[r * K_g + kq / 64]);
        b[t][i] = float(bb[r * K_g + kq / 64]);
      }
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < TK; kk++) {
      auto load_a = [&](thread auto& ca, short tm) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
          const int r = tm * 16 + i * 8;
          vec<T, 4> v = (fm + r < M) ? *(const device vec<T, 4>*)(xb + r * K + kq + kk * 4) : vec<T, 4>(0);
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < 4; j++)
            ca[i * 4 + j] = v[j];
        }
      };
      auto load_b = [&](thread auto& cb, short p) {
        STEEL_PRAGMA_UNROLL
        for (short h = 0; h < 2; h++)
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < 2; i++) {
            const short t = 2 * p + h;
            const uint32_t word = wv[t][i][kk / 2];
            const short base = (kk % 2) * 16;
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++)
              cb[h * 8 + i * 4 + j] = T(s[t][i] * float((word >> (base + 4 * j)) & 0xF) + b[t][i]);
          }
      };
      auto ca0 = op.template get_left_input_cooperative_tensor<T, T, float>();
      auto ca1 = op.template get_left_input_cooperative_tensor<T, T, float>();
      load_a(ca0, 0);
      if constexpr (TM > 1)
        load_a(ca1, 1);
      {
        auto cb = op.template get_right_input_cooperative_tensor<T, T, float>();
        load_b(cb, 0);
        op.run(ca0, cb, cc00);
        if constexpr (TM > 1)
          op.run(ca1, cb, cc10);
      }
      if constexpr (P > 1) {
        auto cb = op.template get_right_input_cooperative_tensor<T, T, float>();
        load_b(cb, 1);
        op.run(ca0, cb, cc01);
        if constexpr (TM > 1)
          op.run(ca1, cb, cc11);
      }
    }
  }
  STEEL_PRAGMA_UNROLL
  for (short e = 0; e < 8; e++) {
    C[0][0][e] = cc00[e];
    C[0][1][e] = cc00[8 + e];
    if constexpr (P > 1) {
      C[0][2][e] = cc01[e];
      C[0][3][e] = cc01[8 + e];
    }
    if constexpr (TM > 1) {
      C[TM - 1][0][e] = cc10[e];
      C[TM - 1][1][e] = cc10[8 + e];
      if constexpr (P > 1) {
        C[TM - 1][2][e] = cc11[e];
        C[TM - 1][3][e] = cc11[8 + e];
      }
    }
  }

  if constexpr (WK > 1) {
    if (wk > 0) {
      for (short tm = 0; tm < TM; tm++)
        for (short t = 0; t < TN; t++)
          for (short e = 0; e < 8; e++)
            red[(((wk - 1) * WN + wn) * E + (tm * TN + t) * 8 + e) * 32 + simd_lid] = C[tm][t][e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (wk != 0)
      return;
    for (int r = 1; r < WK; r++)
      for (short tm = 0; tm < TM; tm++)
        for (short t = 0; t < TN; t++)
          for (short e = 0; e < 8; e++)
            C[tm][t][e] += red[(((r - 1) * WN + wn) * E + (tm * TN + t) * 8 + e) * 32 + simd_lid];
  }

  device U* yz = y + int(tid.z) * M * N;
  for (short tm = 0; tm < TM; tm++)
    for (short i = 0; i < 2; i++) {
      const int m = tm * 16 + fm + i * 8;
      if (m < M)
        for (short t = 0; t < TN; t++)
          for (short j = 0; j < 4; j++)
            yz[m * N + n0 + t * 16 + fn + j] = U(C[tm][t][i * 4 + j]);
    }
}
"""

_SOURCE_R = r"""
    constexpr int RED = (WK - 1) * WN * ((M + 15) / 16) * (SN / 16) * 8 * 32;
    threadgroup float red[RED > 0 ? RED : 1];
    locals_qmm_naxr<T, U, SN, WN, WK, TK, K, N, M, SPLIT>(
        x, w, scales, biases, y, red,
        threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_SOURCE = r"""
    constexpr int BK_padded = (BK + 16 / sizeof(T));
    threadgroup T Ws[BN * BK_padded];
    locals_qmm_t_nax<T, U, 64, 4, BM, BK, BN, WM, WN>(
        w, scales, biases, x, y, Ws, K, N, M, K / SPLIT,
        threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

# (BM, BN, BK, WM, WN, SPLIT). BM = 16 is one NAX fragment of rows; SN = BN / WN and SM = BM / WM must be
# multiples of 16, and one of them a multiple of 32: the accelerators multiply 16 x 32 at a time, and
# tile_matmad_nax computes nothing for a simdgroup tile of one fragment by one. SPLIT > 1 gives small
# matrices more threadgroups (float32 partials summed afterwards).
CANDIDATES = [
    (16, 64, 32, 1, 2, 1), (16, 64, 64, 1, 2, 1), (16, 128, 32, 1, 4, 1),
    (16, 128, 64, 1, 4, 1), (16, 64, 32, 1, 2, 2), (16, 64, 32, 1, 2, 4),
    (32, 64, 32, 2, 2, 1), (32, 128, 32, 2, 4, 1),
]
# ("r", SN, WN, WK, TK, SPLIT): weights dequantized in registers (locals_qmm_naxr). SN output columns per
# simdgroup (32 or 64), WN simdgroups along N and WK along K per threadgroup, chunks of 16 * TK of K
# (TK 4, 8 or 16: 8, 16 or 32 bytes of weights per row and thread), SPLIT threadgroups along K. Up to 32 rows.
CANDIDATES += [
    ("r", 32, 1, 4, 8, 1), ("r", 32, 2, 4, 8, 1), ("r", 32, 1, 8, 8, 1), ("r", 32, 2, 2, 8, 1),
    ("r", 32, 4, 1, 8, 1), ("r", 32, 1, 4, 4, 1), ("r", 32, 2, 4, 4, 1), ("r", 64, 1, 4, 8, 1),
    ("r", 64, 2, 2, 8, 1), ("r", 32, 1, 4, 8, 2), ("r", 32, 1, 4, 16, 1), ("r", 32, 1, 2, 16, 1),
    ("r", 32, 2, 2, 16, 1),
]

_KERNEL = None
_KERNEL_R = None
_AVAILABLE = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="locals_qmm_nax",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_SOURCE,
            header=_header(),
        )
    return _KERNEL


def _kernel_r():
    global _KERNEL_R
    if _KERNEL_R is None:
        _KERNEL_R = mx.fast.metal_kernel(
            name="locals_qmm_naxr",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_SOURCE_R,
            header=_header(),
        )
    return _KERNEL_R


def supported(M: int, K: int, N: int, cfg) -> bool:
    if cfg[0] == "r":
        _, SN, WN, WK, TK, split = cfg
        red = (WK - 1) * WN * ((M + 15) // 16) * (SN // 16) * 8 * 32 * 4
        return (1 <= M <= 32 and SN in (32, 64) and TK in (4, 8, 16) and N % (SN * WN) == 0
                and K % (16 * TK * split) == 0 and red <= 32768 and 32 * WN * WK <= 1024)
    BM, BN, BK, WM, WN, split = cfg
    return (N % BN == 0 and K % (BK * split) == 0 and (K // split) % 64 == 0 and BK % 32 == 0
            and (BN // WN) % 16 == 0 and (BM // WM) % 16 == 0 and ((BN // WN) % 32 == 0 or (BM // WM) % 32 == 0)
            and (BM == 16 or M > 16))


def launch(M: int, K: int, N: int, dtype, *cfg):
    """f(x, w, scales, biases) for (M, K) inputs of this shape, its template and grid computed once."""
    if cfg[0] == "r":
        _, SN, WN, WK, TK, split = cfg
        kern = _kernel_r()
        template = [("SN", SN), ("WN", WN), ("WK", WK), ("TK", TK)]
        grid, tg = (N // (SN * WN) * 32 * WN * WK, 1, split), (32 * WN * WK, 1, 1)
    else:
        BM, BN, BK, WM, WN, split = cfg
        kern = _kernel()
        template = [("BM", BM), ("BN", BN), ("BK", BK), ("WM", WM), ("WN", WN)]
        grid, tg = (N // BN * 32 * WM * WN, (M + BM - 1) // BM, split), (32 * WM * WN, 1, 1)
    out_dtype = dtype if split == 1 else mx.float32
    template = [("T", dtype), ("U", out_dtype)] + template + [("K", K), ("N", N), ("M", M), ("SPLIT", split)]
    shapes = [(split * M, N)]

    def f(x, w, scales, biases):
        (y,) = kern(inputs=[x, w, scales, biases], template=template, grid=grid, threadgroup=tg,
                    output_shapes=shapes, output_dtypes=[out_dtype])
        return y.reshape(split, M, N).sum(axis=0).astype(dtype) if split > 1 else y

    return f


def qmm_nax(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array,
            BM=16, BN: int = 64, BK: int = 32, WM: int = 1, WN: int = 2, split: int = 1) -> mx.array:
    """y = x @ dequant(w).T, 4-bit affine, group 64. x: (M, K). A config starting with "r" is the register
    variant: qmm_nax(x, w, s, b, "r", SN, WN, WK, TK, SPLIT)."""
    M, K = x.shape
    N = w.shape[0]
    cfg = (BM, BN, BK, WM, WN, split)
    if not supported(M, K, N, cfg):
        raise ValueError("unsupported tiling for this shape")
    return launch(M, K, N, x.dtype, *cfg)(x, w, scales, biases)


def available() -> bool:
    """Whether this GPU compiles and runs the kernel correctly (checked once on a small case)."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            w = mx.random.normal((128, 256), key=mx.random.key(1)).astype(mx.bfloat16)
            wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
            x = mx.random.normal((12, 256), key=mx.random.key(2)).astype(mx.bfloat16)
            ref = mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=4).astype(mx.float32)
            tol = 0.05 * (1 + float(mx.max(mx.abs(ref))))
            _AVAILABLE = all(float(mx.max(mx.abs(qmm_nax(x, wq, sc, bi, *cfg).astype(mx.float32) - ref))) <= tol
                             for cfg in ((16, 64, 32, 1, 2, 1), ("r", 32, 1, 4, 8, 1)))
        except Exception as e:  # noqa: BLE001  (no NAX on this GPU, or a compile error to read)
            import os

            if os.environ.get("LOCALS_NAX_DEBUG"):
                print(f"nax_kernel unavailable: {e}")
            _AVAILABLE = False
    return _AVAILABLE
