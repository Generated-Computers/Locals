"""Multi-row quantized matrix-vector kernel (4-bit affine, group size 64).

MLX's qmv reads the weights once per input row, so M rows cost about M times
one row. This kernel follows qmv_fast's layout (a simdgroup owns R output rows,
each lane VPT consecutive K values per block) but keeps M input rows in
registers, so each weight word is loaded and converted once and then applied
to all M rows. The nibbles are used in place with MLX's mask trick: x is
pre-divided by 1, 16, 256, 4096, so ``(w & 0xf0) * (x / 16)`` needs no shift.
(MLX: MIT, Apple; see THIRD_PARTY_NOTICES.md.)
"""

from __future__ import annotations

import mlx.core as mx

_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int out_row = (int(threadgroup_position_in_grid.x) * NSG + int(sg)) * R;
    if (out_row >= N) {
        return;
    }
    constexpr int PPT = VPT / 8;
    constexpr int BLOCK = VPT * 32;
    constexpr int Kw = K / 8;
    constexpr int Kg = K / GS;
    const device uint32_t* wp = w + out_row * Kw + lane * PPT;
    const device T* sc = scales + out_row * Kg + (lane * VPT) / GS;
    const device T* bi = biases + out_row * Kg + (lane * VPT) / GS;
    const device T* xp = x + lane * VPT;

    float acc[M][R];
    for (int m = 0; m < M; m++) {
        for (int r = 0; r < R; r++) {
            acc[m][r] = 0.0f;
        }
    }

    for (int k = 0; k < K; k += BLOCK) {
        float wq[R][VPT];
        float s[R];
        float b[R];
        for (int r = 0; r < R; r++) {
            s[r] = float(sc[r * Kg]);
            b[r] = float(bi[r * Kg]);
            for (int p = 0; p < PPT; p++) {
                const uint32_t word = wp[r * Kw + p];
                for (int h = 0; h < 2; h++) {
                    const uint32_t ws = (word >> (16 * h)) & 0xffff;
                    wq[r][p * 8 + h * 4 + 0] = float(ws & 0x000f);
                    wq[r][p * 8 + h * 4 + 1] = float(ws & 0x00f0);
                    wq[r][p * 8 + h * 4 + 2] = float(ws & 0x0f00);
                    wq[r][p * 8 + h * 4 + 3] = float(ws & 0xf000);
                }
            }
        }
        for (int m = 0; m < M; m++) {
            const device T* xm = xp + m * K;
            float xt[VPT];
            float xs = 0.0f;
            for (int i = 0; i < VPT; i += 4) {
                float a0 = float(xm[i]), a1 = float(xm[i + 1]), a2 = float(xm[i + 2]), a3 = float(xm[i + 3]);
                xs += a0 + a1 + a2 + a3;
                xt[i] = a0;
                xt[i + 1] = a1 * (1.0f / 16.0f);
                xt[i + 2] = a2 * (1.0f / 256.0f);
                xt[i + 3] = a3 * (1.0f / 4096.0f);
            }
            for (int r = 0; r < R; r++) {
                float d = 0.0f;
                for (int i = 0; i < VPT; i++) {
                    d += xt[i] * wq[r][i];
                }
                acc[m][r] += s[r] * d + b[r] * xs;
            }
        }
        wp += BLOCK / 8;
        sc += BLOCK / GS;
        bi += BLOCK / GS;
        xp += BLOCK;
    }

    for (int m = 0; m < M; m++) {
        for (int r = 0; r < R; r++) {
            float v = simd_sum(acc[m][r]);
            if (lane == 0) {
                y[m * N + out_row + r] = static_cast<T>(v);
            }
        }
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="locals_qmv_multi2",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_SOURCE,
        )
    return _KERNEL


def supported(M: int, K: int, N: int, VPT: int = 8, R: int = 4) -> bool:
    return K % (VPT * 32) == 0 and N % R == 0 and 64 % VPT == 0 and M >= 1


def launch_params(M: int, K: int, N: int, dtype, VPT: int = 8, R: int = 4, NSG: int = 2):
    n_tg = (N + NSG * R - 1) // (NSG * R)
    template = [("T", dtype), ("M", M), ("K", K), ("N", N), ("GS", 64), ("VPT", VPT), ("R", R), ("NSG", NSG)]
    return template, (n_tg * 32 * NSG, 1, 1), (32 * NSG, 1, 1)


def qmv_multi(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, VPT: int = 8, R: int = 4, NSG: int = 2) -> mx.array:
    M, K = x.shape
    N = w.shape[0]
    template, grid, tg = launch_params(M, K, N, x.dtype, VPT, R, NSG)
    (y,) = _kernel()(inputs=[x, w, scales, biases], template=template, grid=grid, threadgroup=tg,
                     output_shapes=[(M, N)], output_dtypes=[x.dtype])
    return y


# --- v3: no int -> float conversions ---------------------------------------------
#
# Converting integers to floats is a slow instruction on Apple GPUs. A nibble
# masked in place, OR-ed into the mantissa of 2^23, gives the float 2^23 + q 16^p
# exactly; subtracting 2^23 leaves q 16^p with full-rate instructions only, and
# x is pre-divided by 16^p (as in MLX's qdot). Nibbles 5 to 7 of a word would
# overflow the 23-bit mantissa, so they come from word >> 12. The M input rows
# stay in registers for the whole block and every weight feeds M FMAs.

_SOURCE_V3 = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int out_row = (int(threadgroup_position_in_grid.x) * NSG + int(sg)) * R;
    if (out_row >= N) {
        return;
    }
    constexpr int PPT = VPT / 8;
    constexpr int BLOCK = VPT * 32;
    constexpr int Kw = K / 8;
    constexpr int Kg = K / 64;
    const device uint32_t* wp = w + out_row * Kw + lane * PPT;
    const device T* sc = scales + out_row * Kg + (lane * VPT) / 64;
    const device T* bi = biases + out_row * Kg + (lane * VPT) / 64;
    const device T* xp = x + lane * VPT;

    float acc[M][R];
    UNROLL
    for (int m = 0; m < M; m++) {
        UNROLL
        for (int r = 0; r < R; r++) {
            acc[m][r] = 0.0f;
        }
    }

    for (int k = 0; k < K; k += BLOCK) {
        float xt[M][VPT];
        float xs[M];
        UNROLL
        for (int m = 0; m < M; m++) {
            const device T* xm = xp + m * K + k;
            float sum = 0.0f;
            UNROLL
            for (int p = 0; p < PPT; p++) {
                UNROLL
                for (int i = 0; i < 8; i++) {
                    const float a = float(xm[p * 8 + i]);
                    sum += a;
                    xt[m][p * 8 + i] = a * (i < 5 ? 1.0f / float(1u << (4 * i)) : 1.0f / float(1u << (4 * i - 12)));
                }
            }
            xs[m] = sum;
        }
        UNROLL
        for (int r = 0; r < R; r++) {
            const float s = float(sc[r * Kg + k / 64]);
            const float b = float(bi[r * Kg + k / 64]);
            float d[M];
            UNROLL
            for (int m = 0; m < M; m++) {
                d[m] = 0.0f;
            }
            UNROLL
            for (int p = 0; p < PPT; p++) {
                const uint32_t word = wp[r * Kw + k / 8 + p];
                const uint32_t hi = word >> 12;
                UNROLL
                for (int i = 0; i < 8; i++) {
                    const uint32_t bits = i < 5 ? (word & (0xfu << (4 * i))) : (hi & (0xfu << (4 * i - 12)));
                    const float q = as_type<float>(0x4B000000u | bits) - 8388608.0f;
                    UNROLL
                    for (int m = 0; m < M; m++) {
                        d[m] = fma(xt[m][p * 8 + i], q, d[m]);
                    }
                }
            }
            UNROLL
            for (int m = 0; m < M; m++) {
                acc[m][r] += s * d[m] + b * xs[m];
            }
        }
    }

    UNROLL
    for (int m = 0; m < M; m++) {
        UNROLL
        for (int r = 0; r < R; r++) {
            const float v = simd_sum(acc[m][r]);
            if (lane == 0) {
                y[m * N + out_row + r] = static_cast<T>(v);
            }
        }
    }
"""

_KERNEL_V3 = None


def _kernel_v3():
    global _KERNEL_V3
    if _KERNEL_V3 is None:
        _KERNEL_V3 = mx.fast.metal_kernel(
            name="locals_qmv_v3",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=_SOURCE_V3,
            header="#define UNROLL _Pragma(\"clang loop unroll(full)\")\n",
        )
    return _KERNEL_V3


def launch_v3(M: int, K: int, N: int, dtype, VPT: int = 16, R: int = 4, NSG: int = 2):
    n_tg = (N + NSG * R - 1) // (NSG * R)
    template = [("T", dtype), ("M", M), ("K", K), ("N", N), ("VPT", VPT), ("R", R), ("NSG", NSG)]
    grid, tg = (n_tg * 32 * NSG, 1, 1), (32 * NSG, 1, 1)
    kern = _kernel_v3()

    def f(x, w, scales, biases):
        (y,) = kern(inputs=[x, w, scales, biases], template=template, grid=grid, threadgroup=tg,
                    output_shapes=[(M, N)], output_dtypes=[dtype])
        return y

    return f
