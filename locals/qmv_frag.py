"""Small-M quantized matmul with weights dequantized straight into MMA fragments.

y = x @ W^T for x (M, K), M <= 16, W 4-bit affine with group size 64. We
compute the transposed product Y^T = W X^T with simdgroup 8x8 matrices:
A = an 8x8 tile of W (rows n, columns k), B = an 8x8 tile of X^T (rows k,
columns m). No threadgroup memory and no barriers: each lane loads its own
weights from device memory into its A-fragment registers.

The trick is the order of k inside a 128-wide chunk. A lane of an 8x8
fragment holds row fm and columns fn, fn+1 (fn even). A dot product does not
care about the order of its terms, so we are free to choose which physical k
each (fragment j, logical column) stands for, as long as A and B agree. With

    physical k = 32 * (col / 2) + 16 * (col % 2) + j        (j = 0..15)

lane (fm, fn) finds all 32 of its weights for the chunk in one uint4 of row
fm (bytes 16c..16c+15, c = fn/2), and its B elements are two contiguous runs of
16 inputs (rows fn and fn+1 of x). Every weight is read exactly once, and the
scale/bias of a lane is constant over the chunk (its 32 k's lie in one group).
"""

from __future__ import annotations

import mlx.core as mx

_HEADER = (
    "#include <metal_simdgroup_matrix>\nusing namespace metal;\n"
    "#define UNROLL _Pragma(\"clang loop unroll(full)\")\n"
    "template <typename T> inline float2 unpack2(uint u);\n"
    "template <> inline float2 unpack2<half>(uint u) { return float2(as_type<half2>(u)); }\n"
    "template <> inline float2 unpack2<bfloat16_t>(uint u) { return float2(as_type<float>(u << 16), as_type<float>(u & 0xffff0000u)); }\n"
)

_SOURCE = r"""
    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const short c = fn / 2;
    const int M = int(mcount[0]);  // rows: a runtime value, so M = 5..8 share one compiled kernel
    constexpr int Kw = K / 8;
    constexpr int Kg = K / 64;
    constexpr int KS = K / SPLIT;
    const int split = int(threadgroup_position_in_grid.y);
    const int n0 = (int(threadgroup_position_in_grid.x) * NSG + int(sg)) * (8 * TN);
    if (n0 >= N) {
        return;
    }
    const int kbeg = split * KS;

    simdgroup_matrix<float, 8, 8> acc[TN][MT];
    UNROLL
    for (int t = 0; t < TN; t++) {
        UNROLL
        for (int mt = 0; mt < MT; mt++) {
            acc[t][mt] = simdgroup_matrix<float, 8, 8>(0);
        }
    }

    const device uint32_t* wp[TN];
    const device T* sp[TN];
    const device T* bp[TN];
    UNROLL
    for (int t = 0; t < TN; t++) {
        const int row = n0 + t * 8 + fm;
        wp[t] = w + row * Kw + kbeg / 8 + c * 4;
        sp[t] = scales + row * Kg + kbeg / 64 + c / 2;
        bp[t] = biases + row * Kg + kbeg / 64 + c / 2;
    }
    const int xoff = kbeg + 32 * (fm / 2) + 16 * (fm % 2);
    const device T* xr[MT][2];
    UNROLL
    for (int mt = 0; mt < MT; mt++) {
        xr[mt][0] = x + min(mt * 8 + fn, M - 1) * K + xoff;
        xr[mt][1] = x + min(mt * 8 + fn + 1, M - 1) * K + xoff;
    }

    uint4 wv[TN];
    UNROLL
    for (int t = 0; t < TN; t++) {
        wv[t] = *((const device uint4*)wp[t]);
    }
    for (int k = 0; k < KS; k += 128) {
        simdgroup_matrix<float, 8, 8> bf[MT][16];
        UNROLL
        for (int mt = 0; mt < MT; mt++) {
            UNROLL
            for (int e = 0; e < 2; e++) {
                const device uint4* xv = (const device uint4*)(xr[mt][e] + k);
                const uint4 u0 = xv[0];
                const uint4 u1 = xv[1];
                UNROLL
                for (int q = 0; q < 4; q++) {
                    const float2 lo = unpack2<T>(u0[q]);
                    const float2 hi = unpack2<T>(u1[q]);
                    bf[mt][2 * q].thread_elements()[e] = lo.x;
                    bf[mt][2 * q + 1].thread_elements()[e] = lo.y;
                    bf[mt][8 + 2 * q].thread_elements()[e] = hi.x;
                    bf[mt][8 + 2 * q + 1].thread_elements()[e] = hi.y;
                }
            }
        }
        UNROLL
        for (int t = 0; t < TN; t++) {
            const uint4 cur = wv[t];
            if (k + 128 < KS) {
                wv[t] = *((const device uint4*)(wp[t] + (k + 128) / 8));
            }
            const float s = float(sp[t][k / 64]);
            const float b = float(bp[t][k / 64]);
            UNROLL
            for (int j = 0; j < 16; j++) {
                const uint sh = 4 * (j % 8);
                const uint w0 = (cur[j / 8] >> sh) & 0xf;
                const uint w1 = (cur[2 + j / 8] >> sh) & 0xf;
                simdgroup_matrix<float, 8, 8> a;
                a.thread_elements()[0] = float(w0) * s + b;
                a.thread_elements()[1] = float(w1) * s + b;
                UNROLL
                for (int mt = 0; mt < MT; mt++) {
                    simdgroup_multiply_accumulate(acc[t][mt], a, bf[mt][j], acc[t][mt]);
                }
            }
        }
    }

    UNROLL

    for (int t = 0; t < TN; t++) {
        const int row = n0 + t * 8 + fm;
        UNROLL
        for (int mt = 0; mt < MT; mt++) {
            const int m = mt * 8 + fn;
            auto e = acc[t][mt].thread_elements();
            if (m < M) {
                y[(split * M + m) * N + row] = static_cast<U>(e[0]);
            }
            if (m + 1 < M) {
                y[(split * M + m + 1) * N + row] = static_cast<U>(e[1]);
            }
        }
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="locals_qmv_frag_v3",
            input_names=["x", "w", "scales", "biases", "mcount"],
            output_names=["y"],
            source=_SOURCE,
            header=_HEADER,
        )
    return _KERNEL


def supported(M: int, K: int, N: int, TN: int = 2, SPLIT: int = 1) -> bool:
    return 1 <= M <= 32 and K % (128 * SPLIT) == 0 and N % (8 * TN) == 0


def launch(M: int, K: int, N: int, dtype, TN: int = 2, NSG: int = 2, SPLIT: int = 1):
    """A closure (x, w, scales, biases) -> y with the launch precomputed."""
    out_dtype = dtype if SPLIT == 1 else mx.float32
    template = [("T", dtype), ("U", out_dtype), ("MT", (M + 7) // 8), ("K", K), ("N", N), ("TN", TN), ("NSG", NSG), ("SPLIT", SPLIT)]
    mcount = mx.array([M], dtype=mx.int32)
    n_tg = (N + 8 * TN * NSG - 1) // (8 * TN * NSG)
    grid, tg, shapes = (n_tg * 32 * NSG, SPLIT, 1), (32 * NSG, 1, 1), [(SPLIT * M, N)]
    kern = _kernel()

    def f(x, w, scales, biases):
        (y,) = kern(inputs=[x, w, scales, biases, mcount], template=template, grid=grid, threadgroup=tg,
                    output_shapes=shapes, output_dtypes=[out_dtype])
        if SPLIT > 1:
            y = y.reshape(SPLIT, M, N).sum(axis=0).astype(dtype)
        return y

    return f


def qmv_frag(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, TN: int = 2, NSG: int = 2, SPLIT: int = 1) -> mx.array:
    M, K = x.shape
    return launch(M, K, w.shape[0], x.dtype, TN, NSG, SPLIT)(x, w, scales, biases)


# --- half-precision fragments --------------------------------------------------
#
# Same scheme with half A/B fragments (float accumulator) and a cheaper
# dequantization. Within the lane's uint4, element e of fragment j is nibble
# (j % 4) + 4e of word j / 4, i.e. bits 4p and 4p + 16 of one word, so one
# shift and mask yields both nibbles as the low bits of a half2 with the
# exponent of 1024: (v | 0x64006400) is exactly 1024 + q, and after an exact
# subtraction one half2 fma applies scale and bias to two weights at once.
# Physical k of (j, e) inside the lane's 32 values: 8 * (j / 4) + (j % 4) + 4e,
# so a lane's B elements are 4 runs of 4 consecutive inputs.
# Activations are converted to half with saturation; Qwen-class models stay
# orders of magnitude below the half range at every matmul input.

_SOURCE_H = r"""
    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const short c = fn / 2;
    const int M = int(mcount[0]);  // rows: a runtime value, so M = 5..8 share one compiled kernel
    constexpr int Kw = K / 8;
    constexpr int Kg = K / 64;
    constexpr int KS = K / SPLIT;
    const int split = int(threadgroup_position_in_grid.y);
    const int n0 = (int(threadgroup_position_in_grid.x) * NSG + int(sg)) * (8 * TN);
    if (n0 >= N) {
        return;
    }
    const int kbeg = split * KS;

    simdgroup_matrix<float, 8, 8> acc[TN][MT];
    UNROLL
    for (int t = 0; t < TN; t++) {
        UNROLL
        for (int mt = 0; mt < MT; mt++) {
            acc[t][mt] = simdgroup_matrix<float, 8, 8>(0);
        }
    }

    const device uint32_t* wp[TN];
    const device T* sp[TN];
    const device T* bp[TN];
    UNROLL
    for (int t = 0; t < TN; t++) {
        const int row = n0 + t * 8 + fm;
        wp[t] = w + row * Kw + kbeg / 8 + c * 4;
        sp[t] = scales + row * Kg + kbeg / 64 + c / 2;
        bp[t] = biases + row * Kg + kbeg / 64 + c / 2;
    }
    const int xoff = kbeg + 32 * (fm / 2) + 4 * (fm % 2);
    const device T* xr[MT][2];
    UNROLL
    for (int mt = 0; mt < MT; mt++) {
        xr[mt][0] = x + min(mt * 8 + fn, M - 1) * K + xoff;
        xr[mt][1] = x + min(mt * 8 + fn + 1, M - 1) * K + xoff;
    }

    uint4 wv[TN];
    UNROLL
    for (int t = 0; t < TN; t++) {
        wv[t] = *((const device uint4*)wp[t]);
    }
    for (int k = 0; k < KS; k += 128) {
        simdgroup_matrix<half, 8, 8> bf[MT][16];
        UNROLL
        for (int mt = 0; mt < MT; mt++) {
            UNROLL
            for (int e = 0; e < 2; e++) {
                const device uint2* xv = (const device uint2*)(xr[mt][e] + k);
                UNROLL
                for (int r = 0; r < 4; r++) {
                    const uint2 u = xv[2 * r];
                    const float2 lo = clamp(unpack2<T>(u.x), -65504.0f, 65504.0f);
                    const float2 hi = clamp(unpack2<T>(u.y), -65504.0f, 65504.0f);
                    bf[mt][4 * r].thread_elements()[e] = half(lo.x);
                    bf[mt][4 * r + 1].thread_elements()[e] = half(lo.y);
                    bf[mt][4 * r + 2].thread_elements()[e] = half(hi.x);
                    bf[mt][4 * r + 3].thread_elements()[e] = half(hi.y);
                }
            }
        }
        UNROLL
        for (int t = 0; t < TN; t++) {
            const uint4 cur = wv[t];
            if (k + 128 < KS) {
                wv[t] = *((const device uint4*)(wp[t] + (k + 128) / 8));
            }
            const half2 s2 = half2(half(float(sp[t][k / 64])));
            const half2 b2 = half2(half(float(bp[t][k / 64])));
            UNROLL
            for (int j = 0; j < 16; j++) {
                const uint v = (cur[j / 4] >> (4 * (j % 4))) & 0x000f000fu;
                const half2 q = as_type<half2>(v | 0x64006400u) - half2(1024.0h);
                simdgroup_matrix<half, 8, 8> a;
                reinterpret_cast<thread half2&>(a.thread_elements()) = fma(q, s2, b2);
                UNROLL
                for (int mt = 0; mt < MT; mt++) {
                    simdgroup_multiply_accumulate(acc[t][mt], a, bf[mt][j], acc[t][mt]);
                }
            }
        }
    }

    UNROLL
    for (int t = 0; t < TN; t++) {
        const int row = n0 + t * 8 + fm;
        UNROLL
        for (int mt = 0; mt < MT; mt++) {
            const int m = mt * 8 + fn;
            auto e = acc[t][mt].thread_elements();
            if (m < M) {
                y[(split * M + m) * N + row] = static_cast<U>(e[0]);
            }
            if (m + 1 < M) {
                y[(split * M + m + 1) * N + row] = static_cast<U>(e[1]);
            }
        }
    }
"""

_KERNEL_H = None


def _kernel_h():
    global _KERNEL_H
    if _KERNEL_H is None:
        _KERNEL_H = mx.fast.metal_kernel(
            name="locals_qmv_fragh_v2",
            input_names=["x", "w", "scales", "biases", "mcount"],
            output_names=["y"],
            source=_SOURCE_H,
            header=_HEADER,
        )
    return _KERNEL_H


def launch_h(M: int, K: int, N: int, dtype, TN: int = 2, NSG: int = 2, SPLIT: int = 1):
    """Like ``launch`` but with half fragments and half2 dequantization."""
    out_dtype = dtype if SPLIT == 1 else mx.float32
    template = [("T", dtype), ("U", out_dtype), ("MT", (M + 7) // 8), ("K", K), ("N", N), ("TN", TN), ("NSG", NSG), ("SPLIT", SPLIT)]
    mcount = mx.array([M], dtype=mx.int32)
    n_tg = (N + 8 * TN * NSG - 1) // (8 * TN * NSG)
    grid, tg, shapes = (n_tg * 32 * NSG, SPLIT, 1), (32 * NSG, 1, 1), [(SPLIT * M, N)]
    kern = _kernel_h()

    def f(x, w, scales, biases):
        (y,) = kern(inputs=[x, w, scales, biases, mcount], template=template, grid=grid, threadgroup=tg,
                    output_shapes=shapes, output_dtypes=[out_dtype])
        if SPLIT > 1:
            y = y.reshape(SPLIT, M, N).sum(axis=0).astype(dtype)
        return y

    return f

