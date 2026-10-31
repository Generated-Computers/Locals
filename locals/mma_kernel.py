"""Small-M 4-bit matmul on Apple's matrix units, built from MLX's own pieces.

MLX's tiled quantized matmul uses BM = 32 rows with a 2 x 2 simdgroup layout,
so a pass over m = 6..32 tokens costs the same as 32 rows. This kernel reuses
MLX's ``QuantizedBlockLoader`` (dequantize a weight tile into threadgroup
memory) and ``steel::BlockMMA`` (simdgroup_matrix 8 x 8 multiply-accumulate)
with BM = 8 and a 1 x 4 simdgroup layout: every weight is read and
dequantized once and then applied to up to 8 rows.

MLX custom kernels get no include path, so the needed MLX headers (shipped in
the wheel) are inlined here.
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

    def sub(m):
        return _inline(m.group(1), seen)

    return _INCLUDE_RE.sub(sub, text)


def _header() -> str:
    # utils.h (and what it pulls in) is already part of every custom kernel.
    seen = {"mlx/backend/metal/kernels/utils.h", "mlx/backend/metal/kernels/bf16.h",
            "mlx/backend/metal/kernels/bf16_math.h", "mlx/backend/metal/kernels/complex.h",
            "mlx/backend/metal/kernels/defines.h", "mlx/backend/metal/kernels/logging.h"}
    parts = [
        _inline("mlx/backend/metal/kernels/steel/gemm/gemm.h", seen),
        _inline("mlx/backend/metal/kernels/quantized.h", seen),
    ]
    return "\n".join(parts) + _IMPL


_IMPL = r"""
// qmm_t_impl from MLX with a WM x WN simdgroup layout as template parameters.
template <typename T, typename U, int group_size, int bits, int BM, int BK, int BN, int WM, int WN>
METAL_FUNC void locals_qmm_t_small(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device U* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const int K,
    const int N,
    const int M,
    const int K_eff,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<T, U, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  auto wl = (const device uint8_t*)w;
  const int k0 = tid.z * K_eff;  // split-K slice
  x += y_row * static_cast<int64_t>(K) + k0;
  wl += y_col * K_w + k0 * bytes_per_pack / pack_factor;
  scales += y_col * K_g + k0 / group_size;
  biases += y_col * K_g + k0 / group_size;
  y += tid.z * static_cast<int64_t>(M) * N + y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    for (int k = 0; k < K_eff; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  } else {
    for (int k = 0; k < K_eff; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe();
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM) {
    mma_op.store_result_safe(y, N, short2(BN, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}
"""

_SOURCE = r"""
    constexpr int BK_padded = (BK + 16 / sizeof(T));
    threadgroup T Xs[BM * BK_padded];
    threadgroup T Ws[BN * BK_padded];
    locals_qmm_t_small<T, U, 64, BITS, BM, BK, BN, WM, WN>(
        w, scales, biases, x, y, Xs, Ws, K, N, M, K / SPLIT,
        threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="locals_qmm_mma",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_SOURCE,
            header=_header(),
        )
    return _KERNEL


def qmm_mma(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array,
            BM: int = 8, BN: int = 64, BK: int = 32, WM: int = 1, WN: int = 4, split: int = 1, bits: int = 4) -> mx.array:
    """y = x @ dequant(w).T, affine, group 64, ``bits`` 4 or 8. x: (M, K).

    ``split`` > 1 partitions K across threadgroups (split-K) and sums the
    float32 partials, which helps small matrices fill the GPU.
    """
    M, K = x.shape
    N = w.shape[0]
    if N % BN or K % (BK * split) or (K // split) % 64:
        raise ValueError("unsupported tiling for this shape")
    m_tiles = (M + BM - 1) // BM
    out_dtype = x.dtype if split == 1 else mx.float32
    (y,) = _kernel()(
        inputs=[x, w, scales, biases],
        template=[("T", x.dtype), ("U", out_dtype), ("BM", BM), ("BN", BN), ("BK", BK), ("WM", WM), ("WN", WN),
                  ("K", K), ("N", N), ("M", M), ("SPLIT", split), ("BITS", bits)],
        grid=(N // BN * 32 * WM * WN, m_tiles, split),
        threadgroup=(32 * WM * WN, 1, 1),
        output_shapes=[(split * M, N)],
        output_dtypes=[out_dtype],
    )
    if split > 1:
        y = y.reshape(split, M, N).sum(axis=0).astype(x.dtype)
    return y
