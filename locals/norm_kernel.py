"""Fused residual add + RMS norm: (x, r, w) -> (h = x + r, rms_norm(h) * w).

A pre-norm decoder layer adds a residual and immediately normalizes the sum,
twice per layer (after the mixer, and after the MLP into the next layer's
input norm). Doing both in one kernel halves those launches. Rounding follows
mlx: h is rounded to the activation dtype, and the normalized value is rounded
before the weight multiply, as in mx.fast.rms_norm.
"""

from __future__ import annotations

import mlx.core as mx

_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    constexpr int PER = D / NT;
    threadgroup float partial[NT / 32];

    const device T* xr = x + row * D;
    const device T* rr = r + row * D;
    device T* hr = h + row * D;
    device T* nr = n + row * D;

    float hv[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; i++) {
        const int d = i * NT + tid;
        const T s = static_cast<T>(float(xr[d]) + float(rr[d]));
        hr[d] = s;
        hv[i] = float(s);
        ss += hv[i] * hv[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) {
        partial[sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < NT / 32; i++) {
        tot += partial[i];
    }
    const float inv = metal::rsqrt(tot / float(D) + __EPS__);
    for (int i = 0; i < PER; i++) {
        const int d = i * NT + tid;
        nr[d] = static_cast<T>(float(w[d]) * float(static_cast<T>(hv[i] * inv)));
    }
"""

_KERNELS = {}


def _kernel(eps: float):
    k = _KERNELS.get(eps)
    if k is None:
        k = _KERNELS[eps] = mx.fast.metal_kernel(
            name="locals_add_rms_%d" % len(_KERNELS),
            input_names=["x", "r", "w"],
            output_names=["h", "n"],
            source=_SOURCE.replace("__EPS__", "%.9ef" % eps),
        )
    return k


def supported(D: int) -> bool:
    return D % 256 == 0


def add_rms(x: mx.array, r: mx.array, w: mx.array, eps: float) -> tuple:
    D = x.shape[-1]
    rows = x.size // D
    h, n = _kernel(float(eps))(
        inputs=[x, r, w],
        template=[("T", x.dtype), ("D", D), ("NT", 256)],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
    return h, n
