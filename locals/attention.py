"""Ragged batch decode attention: one kernel launch for all rows of all requests.

Adapted from MLX's ``sdpa_vector`` (MIT, Apple; see THIRD_PARTY_NOTICES.md). Row r of the packed batch belongs to KV
slot ``slots[r]`` and may see the first ``nkeys[r]`` keys of that slot
(its prefix plus the earlier new tokens of its own segment, which gives the
causal mask for multi-token segments). K/V live in slot-major buffers of shape
(n_slots, n_kv_heads, cap, D), so a row reads only its own valid keys.
"""

from __future__ import annotations

import mlx.core as mx

_SOURCE = r"""
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    typedef float U;

    thread U qv[qk_per_thread];
    thread U kv[qk_per_thread];
    thread U o[qk_per_thread];
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];

    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int rh = int(threadgroup_position_in_grid.x);
    const int row = rh / H;
    const int head = rh - row * H;
    const int slot = slots[row];
    const int N = nkeys[row];
    const int kvh = head / (H / HKV);
    const float sc = scale[0];

    const device T* qp = q + (row * H + head) * D + simd_lid * qk_per_thread;
    const size_t base = ((size_t)(slot * HKV + kvh) * CAP) * D;
    const device T* kp = keys + base + simd_gid * D + simd_lid * qk_per_thread;
    const device T* vp = values + base + simd_gid * D + simd_lid * qk_per_thread;
    device T* op = out + (row * H + head) * D + simd_gid * qk_per_thread;

    for (int i = 0; i < qk_per_thread; i++) {
        qv[i] = sc * static_cast<U>(qp[i]);
        o[i] = 0;
    }
    U max_score = -1e30f;
    U sum_exp = 0;
    for (int i = simd_gid; i < N; i += BN) {
        for (int j = 0; j < qk_per_thread; j++) {
            kv[j] = kp[j];
        }
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
            score += qv[j] * kv[j];
        }
        score = simd_sum(score);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U e = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + e;
        for (int j = 0; j < qk_per_thread; j++) {
            o[j] = o[j] * factor + e * static_cast<U>(vp[j]);
        }
        kp += BN * D;
        vp += BN * D;
    }
    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    sum_exp = simd_sum(sum_exp_scores[simd_lid] * factor);
    for (int i = 0; i < qk_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp == 0 ? o[i] : (o[i] / sum_exp);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lid == 0) {
        for (int i = 0; i < qk_per_thread; i++) {
            op[i] = static_cast<T>(o[i]);
        }
    }
"""

_KERNEL = mx.fast.metal_kernel(
    name="locals_ragged_sdpa",
    input_names=["q", "keys", "values", "slots", "nkeys", "scale"],
    output_names=["out"],
    source=_SOURCE,
)


# GQA variant: one threadgroup per (row, kv head) computes all G = H / HKV
# query heads of that group, so each key/value vector is read once.
_SOURCE_GQA = r"""
    constexpr int BD = 32;
    constexpr int qk = D / BD;
    constexpr int G = H / HKV;
    typedef float U;

    thread U qv[G][qk];
    thread U o[G][qk];
    thread U mx_[G];
    thread U se[G];
    threadgroup U outputs[BD * BD];
    threadgroup U max_scores[G][BD];
    threadgroup U sum_scores[G][BD];

    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int rk = int(threadgroup_position_in_grid.x);
    const int row = rk / HKV;
    const int kvh = rk - row * HKV;
    const int slot = slots[row];
    const int N = nkeys[row];
    const float sc = scale[0];

    const size_t base = ((size_t)(slot * HKV + kvh) * CAP) * D;
    const device T* kp = keys + base + simd_gid * D + simd_lid * qk;
    const device T* vp = values + base + simd_gid * D + simd_lid * qk;
    for (int g = 0; g < G; g++) {
        const device T* qp = q + (row * H + kvh * G + g) * D + simd_lid * qk;
        for (int i = 0; i < qk; i++) {
            qv[g][i] = sc * static_cast<U>(qp[i]);
            o[g][i] = 0;
        }
        mx_[g] = -1e30f;
        se[g] = 0;
    }
    for (int i = simd_gid; i < N; i += BN) {
        U kk[qk];
        U vv[qk];
        for (int j = 0; j < qk; j++) {
            kk[j] = kp[j];
            vv[j] = vp[j];
        }
        for (int g = 0; g < G; g++) {
            U score = 0;
            for (int j = 0; j < qk; j++) {
                score += qv[g][j] * kk[j];
            }
            score = simd_sum(score);
            U new_max = max(mx_[g], score);
            U factor = fast::exp(mx_[g] - new_max);
            U e = fast::exp(score - new_max);
            mx_[g] = new_max;
            se[g] = se[g] * factor + e;
            for (int j = 0; j < qk; j++) {
                o[g][j] = o[g][j] * factor + e * vv[j];
            }
        }
        kp += BN * D;
        vp += BN * D;
    }
    for (int g = 0; g < G; g++) {
        if (simd_lid == 0) {
            max_scores[g][simd_gid] = mx_[g];
            sum_scores[g][simd_gid] = se[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Lane l stands for simdgroup l in the reductions (only l < BN is valid).
    for (int g = 0; g < G; g++) {
        U m = simd_lid < BN ? max_scores[g][simd_lid] : -1e30f;
        U new_max = simd_max(m);
        U factor = simd_lid < BN ? fast::exp(m - new_max) : 0.0f;
        U total = simd_sum((simd_lid < BN ? sum_scores[g][simd_lid] : 0.0f) * factor);
        device T* op = out + (row * H + kvh * G + g) * D;
        for (int i = 0; i < qk; i++) {
            outputs[simd_lid * BD + simd_gid] = o[g][i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int dg = simd_gid; dg < BD; dg += BN) {
                U part = simd_lid < BN ? outputs[dg * BD + simd_lid] : 0.0f;
                U val = simd_sum(part * factor);
                val = total == 0 ? val : val / total;
                if (simd_lid == 0) {
                    op[dg * qk + i] = static_cast<T>(val);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
"""

_KERNEL_GQA = mx.fast.metal_kernel(
    name="locals_ragged_sdpa_gqa",
    input_names=["q", "keys", "values", "slots", "nkeys", "scale"],
    output_names=["out"],
    source=_SOURCE_GQA,
)


def ragged_attention(q: mx.array, keys: mx.array, values: mx.array, slots: mx.array, nkeys: mx.array, scale: mx.array) -> mx.array:
    """q: (R, H, D); keys/values: (n_slots, Hkv, cap, D); slots/nkeys: (R,) int32;
    scale: (1,) float32. Returns (R, H, D)."""
    R, H, D = q.shape
    _, Hkv, cap, _ = keys.shape
    if H // Hkv > 1:
        BN = 16
        (out,) = _KERNEL_GQA(
            inputs=[q, keys, values, slots, nkeys, scale],
            template=[("T", q.dtype), ("D", D), ("H", H), ("HKV", Hkv), ("CAP", cap), ("BN", BN)],
            grid=(R * Hkv * 32 * BN, 1, 1),
            threadgroup=(32 * BN, 1, 1),
            output_shapes=[(R, H, D)],
            output_dtypes=[q.dtype],
        )
        return out
    (out,) = _KERNEL(
        inputs=[q, keys, values, slots, nkeys, scale],
        template=[("T", q.dtype), ("D", D), ("H", H), ("HKV", Hkv), ("CAP", cap)],
        grid=(R * H * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(R, H, D)],
        output_dtypes=[q.dtype],
    )
    return out
