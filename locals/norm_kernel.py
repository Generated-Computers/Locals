"""Fused residual add + RMS norm: (x, r, w) -> (h = x + r, rms_norm(h) * w).

A pre-norm decoder layer adds a residual and immediately normalizes the sum,
twice per layer (after the mixer, and after the MLP into the next layer's
input norm). Doing both in one kernel halves those launches. The kernel is
laid out as MLX's own RMS norm for one row (4 values a thread, a simdgroup
sum, the threadgroup's partial sums summed by one simdgroup, a precise
rsqrt), and h is added as MLX's add adds: its outputs are the same bits as
``x + r`` and ``mx.fast.rms_norm(x + r, w, eps)`` (checked on 42M values).
"""

from __future__ import annotations

import mlx.core as mx

_SOURCE = r"""
    constexpr int N_READS = 4;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    const uint row = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const device T* xr = x + row * size_t(D) + lid * N_READS;
    const device T* rr = r + row * size_t(D) + lid * N_READS;
    device T* hr = h + row * size_t(D) + lid * N_READS;
    T hv[N_READS];
    float acc = 0;
    for (int i = 0; i < N_READS; i++) {
        hv[i] = xr[i] + rr[i];
        hr[i] = hv[i];
        float xi = hv[i];
        acc += xi * xi;
    }
    acc = simd_sum(acc);
    if (sg == 0) local_sums[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) local_sums[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / D + __EPS__);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device T* nr = n + row * size_t(D) + lid * N_READS;
    const device T* wr = w + lid * N_READS;
    for (int i = 0; i < N_READS; i++) nr[i] = wr[i] * static_cast<T>(hv[i] * local_inv_mean[0]);
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
    """A row in one threadgroup of D / 4 threads, whole simdgroups (MLX's single-row norm goes to 4,096)."""
    return D % 128 == 0 and D <= 4096


def add_rms(x: mx.array, r: mx.array, w: mx.array, eps: float) -> tuple:
    D = x.shape[-1]
    rows = x.size // D
    h, n = _kernel(float(eps))(
        inputs=[x, r, w],
        template=[("T", x.dtype), ("D", D)],
        grid=(rows * (D // 4), 1, 1),
        threadgroup=(D // 4, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
    return h, n
