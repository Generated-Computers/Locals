"""Mixture-of-experts layers with fewer kernel launches (Flint).

A Flint decode step is mostly launches, not arithmetic: one token reads 6 of
256 experts in each of 47 layers, about 0.3 GB, which an M4 Pro streams in a
millisecond, yet stock mlx-lm takes 12 ms. Per layer the router alone was
seven kernels (cast the input, cast the bf16 router weight to float32, matmul,
sigmoid, top-k, gather, normalize), and the experts' gate and up projections
two gathers of the same rows. Here:

- the router's logits are one kernel, ``router_logits``, which reads the bf16
  weight and sums in float32 across many small threadgroups (MLX's float32
  matrix-vector product took 42 us a layer, on a weight it had to cast first),
  and the choice after it (sigmoid, top-k, renormalize) another, ``route_topk``;
- gate and up become one gathered matmul, as ``fuse.py`` does for dense MLPs
  (quantization groups run along the input, so stacking outputs is exact);
- the shared expert's gate and up are fused the same way.

The experts each token gets and their weights are the stock computation's,
up to the order the k experts are summed in.
"""

from __future__ import annotations

import os
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu

# One simdgroup per token: lane i holds experts i, i+32, ...; k rounds of a simdgroup argmax pick the experts in
# order of score (ties to the lower index, as a stable sort), then the chosen scores are renormalized to sum to 1.
# With ORDER, one threadgroup holds every token, and its threads then rank the slots (token, k) by expert: the
# expert kernels' order (expert_order) without a sort kernel of its own.
_ROUTE_SRC = """
    const uint t = thread_position_in_grid.x / 32;
    const uint lane = thread_index_in_simdgroup;
    threadgroup uint chosen[ORDER ? ORDER_ROWS * K : 1];
    if (!ORDER && t >= nrows[0]) return;
    constexpr int PER = (E + 31) / 32;
    float s[PER];
    for (int j = 0; j < PER; j++) {
        const int e = lane + 32 * j;
        const float v = 1.0f / (1.0f + metal::precise::exp(-float(logits[t * E + e])));
        s[j] = e < E ? (isnan(v) ? 0.0f : v) : -1.0f;  // (a NaN logit never beat -1: no expert won, and expert E was named)
    }
    float picked[K];
    uint which[K];
    for (int r = 0; r < K; r++) {
        float best = -1.0f;
        uint best_e = E;
        for (int j = 0; j < PER; j++) {
            const uint e = lane + 32 * j;
            if (s[j] > best) { best = s[j]; best_e = e; }
        }
        const float top = simd_max(best);
        const uint win = min(simd_min(best == top ? best_e : uint(E)), uint(E - 1));
        if (win % 32 == lane) s[win / 32] = -1.0f;
        picked[r] = top;
        which[r] = win;
    }
    if (lane == 0) {
        float total = 0.0f;
        for (int r = 0; r < K; r++) total += picked[r];
        for (int r = 0; r < K; r++) {
            inds[t * K + r] = which[r];
            weights[t * K + r] = picked[r] / total;
            if (ORDER) chosen[t * K + r] = which[r];
        }
    }
    if (ORDER) {  // (the grid is exactly the tokens' simdgroups: every thread reaches the barrier)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint S = nrows[0] * K, i = thread_position_in_threadgroup.x;
        if (i < S) {
            const uint e = chosen[i];
            uint rank = 0;
            for (uint j = 0; j < S; j++) rank += (chosen[j] < e || (chosen[j] == e && j < i)) ? 1u : 0u;
            order[rank] = i;
        }
    }
"""

_route_kernel = None

# Router logits for a few rows: one simdgroup per expert, each lane 8 consecutive weights at a time, every row's dot
# product in float32 (the bf16 weights and activations are exact in float32, as in x.astype(f32) @ W.astype(f32).T).
# Many small threadgroups: at one row MLX's float32 matrix-vector product ran 42 us a layer against a 2 MB weight
# (the bf16 one cast to float32 first), on a handful of the GPU's cores.
_LOGITS_SRC = """
    const uint e = thread_position_in_grid.x / 32;  // (a partial last threadgroup is smaller: not its size times its index)
    const uint lane = thread_index_in_simdgroup;
    if (e >= E) return;
    const int T = nrows[0];  // at most TMAX, the register tile's size (a power of two: few kernels to compile)
    float acc[TMAX];
    for (int t = 0; t < TMAX; t++) acc[t] = 0.0f;
    const device auto* wrow = w + size_t(e) * D;
    for (uint d = lane * 8; d < D; d += 256) {
        float wv[8];
        for (int i = 0; i < 8; i++) wv[i] = float(wrow[d + i]);
        for (int t = 0; t < TMAX && t < T; t++) {
            const device auto* xr = x + size_t(t) * D + d;
            float s = 0.0f;
            for (int i = 0; i < 8; i++) s += wv[i] * float(xr[i]);
            acc[t] += s;
        }
    }
    for (int t = 0; t < TMAX && t < T; t++) {
        const float v = simd_sum(acc[t]);
        if (lane == 0) logits[t * E + e] = v;
    }
"""

_logits_kernel = None
ROUTER_MAX_ROWS = 16  # more rows (a prompt): MLX's matmul, which reads the weight once for many rows


def router_logits(x: mx.array, w: mx.array) -> mx.array:
    """float32 ``x @ w.T`` for a router: ``x`` (..., D), ``w`` (E, D), both bf16 or fp16, at most ROUTER_MAX_ROWS rows."""
    global _logits_kernel
    if _logits_kernel is None:
        _logits_kernel = mx.fast.metal_kernel(name="locals_router_logits", input_names=["x", "w", "nrows"],
                                              output_names=["logits"], source=_LOGITS_SRC)
    E, D = w.shape
    T = x.size // D
    sg = 4  # simdgroups (experts) per threadgroup: 64 threadgroups for 256 experts
    tmax = 1 << max(0, (T - 1).bit_length())
    (out,) = _logits_kernel(inputs=[x, w, mx.array([T], dtype=mx.uint32)], template=[("TMAX", tmax), ("E", E), ("D", D)],
                            grid=((E + sg - 1) // sg * sg * 32, 1, 1), threadgroup=(sg * 32, 1, 1),
                            output_shapes=[(*x.shape[:-1], E)], output_dtypes=[mx.float32])
    return out


def router(x: mx.array, w: mx.array) -> mx.array:
    """A router's float32 logits: the few-rows kernel where it applies, else the stock computation."""
    D = w.shape[1]
    if x.size // D <= ROUTER_MAX_ROWS and D % 256 == 0 and x.dtype == w.dtype and x.dtype in (mx.bfloat16, mx.float16):
        return router_logits(x, w)
    return x.astype(mx.float32) @ w.astype(mx.float32).T


ORDER_MAX_ROWS = 16  # tokens whose slots route_topk orders itself (one threadgroup of a simdgroup a token)


def route_topk(logits: mx.array, k: int, order: bool = False):
    """(experts, weights) of each row of router ``logits`` (..., E): the ``k`` largest sigmoid scores, as uint32
    expert indices and float32 weights that sum to 1. ``flint.route`` in one kernel. With ``order`` also the slots in
    the order of their experts, for experts_up/experts_down (as expert_order, up to the order of one expert's slots,
    which the kernels do not depend on)."""
    global _route_kernel
    if _route_kernel is None:
        _route_kernel = mx.fast.metal_kernel(name="locals_route_topk", input_names=["logits", "nrows"],
                                             output_names=["inds", "weights", "order"], source=_ROUTE_SRC)
    E = logits.shape[-1]
    T = logits.size // E
    fused = order and T <= ORDER_MAX_ROWS
    inds, weights, slots = _route_kernel(
        inputs=[logits, mx.array([T], dtype=mx.uint32)],
        template=[("E", E), ("K", k), ("ORDER", int(fused)), ("ORDER_ROWS", ORDER_MAX_ROWS)], grid=(T * 32, 1, 1),
        threadgroup=(T * 32 if fused else min(256, T * 32), 1, 1),
        output_shapes=[(*logits.shape[:-1], k)] * 2 + [(T * k if fused else 1,)],
        output_dtypes=[mx.uint32, mx.float32, mx.uint32])
    if not order:
        return inds, weights
    return inds, weights, slots if fused else expert_order(inds.reshape(-1, k))


# Qwen3-Next/3.5's choice (mlx-lm's Qwen3NextSparseMoeBlock: softmax(precise=True) of the router's logits in their
# dtype, argpartition's top k, the k renormalized) for a simdgroup a token. Lane i holds experts 128 g + 4 i .. + 3 of
# each group g of 128, the layout of MLX's softmax kernel (4 a thread, a simdgroup per 128), so the normalizer is summed
# in its order and each probability is its bits, rounded to T. argpartition is a stable ascending sort here (ties: the
# higher index ends among the last k), so the k are chosen largest first with ties to the higher index and stored in
# ascending order, the order the block sums them in; the weights are its running sum in T and its division.
_ROUTE_SOFTMAX_SRC = """
    #define UNROLL _Pragma("clang loop unroll(full)")
    const uint t = thread_position_in_grid.x / 32;
    const uint lane = thread_index_in_simdgroup;
    threadgroup uint chosen[ORDER ? ORDER_ROWS * K : 1];
    if (!ORDER && t >= nrows[0]) return;
    constexpr int G = (E + 127) / 128, Q = 4 * G;  // a lane's values: q = 4 g + j is expert 128 g + 4 lane + j
    const uint LS = logits_shape[1];  // (the logits' row: the router's E, then what was fused after them)
    float p[Q];
    float vmax = -INFINITY;
    UNROLL for (int q = 0; q < Q; q++) {
        const int e = (q / 4) * 128 + int(lane) * 4 + q % 4;
        const float v = e < E ? float(logits[t * LS + e]) : -INFINITY;
        p[q] = v;
        vmax = vmax < v ? v : vmax;
    }
    vmax = simd_max(vmax);
    float mine = 0.0f;
    UNROLL for (int g = 0; g < G; g++) {
        float part = 0.0f;
        UNROLL for (int j = 0; j < 4; j++) {
            const float ex = metal::fast::exp(p[4 * g + j] - vmax);
            p[4 * g + j] = ex;
            part += ex;
        }
        part = simd_sum(part);
        if (int(lane) == g) mine = part;
    }
    const float inv = 1.0f / simd_sum(mine);
    UNROLL for (int q = 0; q < Q; q++) {
        const int e = (q / 4) * 128 + int(lane) * 4 + q % 4;
        const float pr = float(T(p[q] * inv));
        p[q] = e < E ? (isnan(pr) ? -1.0f : pr) : -2.0f;  // (a NaN is never chosen over a number; past E never)
    }
    // (every loop unrolled, the arrays indexed by constants: they stay in registers)
    float picked[K];
    uint which[K];
    UNROLL for (int r = 0; r < K; r++) {
        float best = -3.0f;
        uint best_e = 0;
        UNROLL for (int q = 0; q < Q; q++) {  // (ascending experts: >= keeps the higher of a tie)
            if (p[q] >= best) { best = p[q]; best_e = (q / 4) * 128 + lane * 4 + q % 4; }
        }
        const float top = simd_max(best);
        const uint win = min(simd_max(best == top ? best_e : 0u), uint(E - 1));
        UNROLL for (int q = 0; q < Q; q++) {
            if ((q / 4) * 128 + lane * 4 + q % 4 == win) p[q] = -3.0f;
        }
        picked[r] = top;
        which[r] = win;
    }
    if (lane == 0) {
        T total = T(picked[K - 1]);
        UNROLL for (int j = 1; j < K; j++) total = T(float(total) + picked[K - 1 - j]);
        UNROLL for (int j = 0; j < K; j++) {
            const float s = picked[K - 1 - j];
            inds[t * K + j] = which[K - 1 - j];
            weights[t * K + j] = NORM ? float(T(s / float(total))) : s;
            if (ORDER) chosen[t * K + j] = which[K - 1 - j];
        }
    }
    if (ORDER) {  // (the grid is exactly the tokens' simdgroups: every thread reaches the barrier)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint S = nrows[0] * K, i = thread_position_in_threadgroup.x;
        if (i < S) {
            const uint e = chosen[i];
            uint rank = 0;
            for (uint j = 0; j < S; j++) rank += (chosen[j] < e || (chosen[j] == e && j < i)) ? 1u : 0u;
            order[rank] = i;
        }
    }
"""

_route_softmax_kernel = None


def route_softmax(logits: mx.array, E: int, k: int, norm: bool = True, order: bool = False):
    """(experts, weights[, slots in expert order]) as mlx-lm's Qwen3NextSparseMoeBlock chooses them from router
    ``logits`` (..., L) whose first ``E`` are the router's (L > E: more fused after them): uint32 experts in ascending
    order of their probability (ties: ascending index) and float32 weights that hold the block's values in the
    logits' dtype: the block's experts, order and weights bit for bit, ties at the k-th place included."""
    global _route_softmax_kernel
    if _route_softmax_kernel is None:
        _route_softmax_kernel = mx.fast.metal_kernel(name="locals_route_softmax", input_names=["logits", "nrows"],
                                                     output_names=["inds", "weights", "order"], source=_ROUTE_SOFTMAX_SRC)
    L = logits.shape[-1]
    T = logits.size // L
    fused = order and T <= ORDER_MAX_ROWS
    inds, weights, slots = _route_softmax_kernel(
        inputs=[logits.reshape(T, L), mx.array([T], dtype=mx.uint32)],
        template=[("T", logits.dtype), ("E", E), ("K", k), ("NORM", int(norm)), ("ORDER", int(fused)),
                  ("ORDER_ROWS", ORDER_MAX_ROWS)],
        grid=(T * 32, 1, 1), threadgroup=(T * 32 if fused else min(256, T * 32), 1, 1),
        output_shapes=[(*logits.shape[:-1], k)] * 2 + [(T * k if fused else 1,)],
        output_dtypes=[mx.uint32, mx.float32, mx.uint32])
    if not order:
        return inds, weights
    return inds, weights, slots if fused else expert_order(inds.reshape(-1, k))


# --- the routed experts of a few tokens -----------------------------------------------------------------------------
#
# gather_qmm, the gate/up slices, swiglu, gather_qmm, the weighted sum over the k experts, its cast and the shared
# expert's add were eight dependent kernels a layer, 70 us at one token (the weights they read take 39 us): every
# dependent launch costs about 5 us on an M4 Pro. ``experts_up`` writes swiglu(gate, up) of each (token, expert)
# pair, ``experts_down`` each pair's down projection, then their sum with the routing weights plus the shared expert.
# Both take the pairs in the order of their experts (expert_order), so rows of a batch that chose one expert read it
# once: 8 rows of Flint wrote 287 tokens/s instead of 265, 16 rows 315 instead of 288. From 4 rows the kernels are
# bound by their arithmetic as much as by the weights they read (8 rows: 15.7 ms a step, the loads alone 11.8, the
# arithmetic alone 13.7), so a simdgroup of experts_up takes 2 hidden units and a lane of experts_down 4 rows, each
# input chunk converted once for all of them, and 3-bit values become floats without int-to-float (qdot8): 14.2 ms,
# the same bits. Rounding is where the stock graph rounds: each projection to the activation dtype, swiglu's steps,
# the float32 weighted sum.
# Weights are MLX's affine format with groups of 64: a group is a little-endian bit stream of 2 * BITS 32-bit words.

_QDOT_HEADER = """
// MLX's sigmoid in the type T, each step rounded as its kernel rounds it (1 / (1 + exp|x|), and 1 minus that for x >= 0):
// in bf16 the bits of mx.sigmoid and of mlx-lm's swiglu, where 1 / (1 + exp(-x)) rounded once differs in 17% of values.
template <typename T>
inline T mlx_sigmoid(T x) {
    const T e = T(metal::precise::exp(metal::abs(float(x))));
    const T y = T(1.0f / float(T(1.0f + float(e))));
    return float(x) < 0.0f ? y : T(1.0f - float(y));
}

// Values 8c..8c+7 of a group as one integer: value i at bits BITS*i (3 bits: four 24-bit chunks in three words).
// A chunk kept in a register is a uint up to 4 bits and a ulong from 5 (40 to 64 bits).
template <bool WIDE> struct chunk_type { typedef uint type; };
template <> struct chunk_type<true> { typedef ulong type; };
template <int BITS>
inline ulong chunk8(const device uint32_t* wp, int c) {
    if (BITS == 4) return wp[c];
    if (BITS == 3) {
        const device uint32_t* b = wp + (c / 4) * 3;
        switch (c % 4) {
            case 0: return b[0] & 0xffffffu;
            case 1: return (b[0] >> 24) | ((b[1] & 0xffffu) << 8);
            case 2: return (b[1] >> 16) | ((b[2] & 0xffu) << 16);
            default: return b[2] >> 8;
        }
    }
    const uint bit = 8 * BITS * c, wi = bit / 32, sh = bit % 32;  // any other width: a 64-bit window
    ulong v = ulong(wp[wi]) >> sh;
    if (sh + 8 * BITS > 32) v |= ulong(wp[wi + 1]) << (32 - sh);
    return v;  // (5 to 8 bits: 40 to 64 bits, which a uint cut short)
}

// The 8 inputs of a chunk divided by 2^(BITS*i), so a value is masked in place, not shifted (as MLX's qmv does);
// a power of two scales a float exactly. BITS * 8 > 32 (5 to 8 bits) does not fit a uint: shifted there.
// 3 bits: a masked value (below 2^23) becomes a float without an int-to-float conversion, ORed into the mantissa of
// 2^23 and 2^23 subtracted (exact); value 7 (bits 21 to 23) is shifted down 3 first and its input divided by 2^18
// instead of 2^21, the same product.
template <int BITS>
inline float qdot8(ulong v, thread const float* xs) {
    constexpr uint mask = (1u << BITS) - 1;
    float s = 0.0f;
    if (BITS == 3) {
        const uint w = uint(v);
        for (int i = 0; i < 7; i++) s += (as_type<float>((w & (7u << (3 * i))) | 0x4B000000u) - 8388608.0f) * xs[i];
        s += (as_type<float>(((w >> 3) & (7u << 18)) | 0x4B000000u) - 8388608.0f) * xs[7];
    } else if (BITS * 8 <= 32) {
        const uint w = uint(v);
        for (int i = 0; i < 8; i++) s += float(w & (mask << (BITS * i))) * xs[i];
    } else {
        for (int i = 0; i < 8; i++) s += float(uint(v >> (BITS * i)) & mask) * xs[i];
    }
    return s;
}

template <int BITS>
inline void prescale8(thread float* xs) {
    if (BITS == 3) {
        for (int i = 1; i < 7; i++) xs[i] *= 1.0f / float(1u << (3 * i));
        xs[7] *= 1.0f / float(1u << 18);
    } else if (BITS * 8 <= 32) {
        for (int i = 1; i < 8; i++) xs[i] *= 1.0f / float(1u << (BITS * i));
    }
}
"""

# The slots (token, k) sorted by expert (expert_order): the simdgroup at the start of each run of one expert loads its
# weight words once and serves every slot of the run (rows sharing an expert read it once); the others return at
# once. One simdgroup per (slot in that order, R hidden units), a lane per group of 64 inputs (D <= 2048): each chunk
# of the input is converted and prescaled once for the R units' gate and up rows.
_UP_SRC = """
    constexpr int NG = D / 64, WPG = 2 * BITS;
    constexpr uint ROWW = NG * WPG, NB = I / R;
    const uint sgid = thread_position_in_grid.x / 32;  // (a partial last threadgroup is smaller: not its size times its index)
    const uint lane = thread_index_in_simdgroup;
    const uint S = nslots[0];
    const uint j0 = (sgid % NB) * R, p = sgid / NB;
    if (p >= S || lane >= NG) return;
    const uint e = inds[order[p]];
    if (p > 0 && inds[order[p - 1]] == e) return;  // not its run's first slot: that one's simdgroup serves it
    uint q_end = p + 1;
    while (q_end < S && inds[order[q_end]] == e) q_end++;
    const uint g = lane;
    const size_t row0 = size_t(e) * 2 * I + j0;  // the units' gate rows; their up rows are I further
    typename chunk_type<(BITS * 8 > 32)>::type wg[R][8], wu[R][8];
    float sg[R], bg[R], su[R], bu[R];
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < 8; c++) {
            wg[r][c] = chunk8<BITS>(w + (row0 + r) * ROWW + g * WPG, c);
            wu[r][c] = chunk8<BITS>(w + (row0 + r + I) * ROWW + g * WPG, c);
        }
        sg[r] = float(scales[(row0 + r) * NG + g]);
        bg[r] = float(biases[(row0 + r) * NG + g]);
        su[r] = float(scales[(row0 + r + I) * NG + g]);
        bu[r] = float(biases[(row0 + r + I) * NG + g]);
    }
    for (uint q = p; q < q_end; q++) {
        const uint slot = order[q], t = slot / K;
        const device T* xr = x + size_t(t) * D + g * 64;
        float qg[R], qu[R], xs = 0.0f;
        for (int r = 0; r < R; r++) { qg[r] = 0.0f; qu[r] = 0.0f; }
        for (int c = 0; c < 8; c++) {
            float xv[8];
            for (int i = 0; i < 8; i++) { xv[i] = float(xr[c * 8 + i]); xs += xv[i]; }
            prescale8<BITS>(xv);
            for (int r = 0; r < R; r++) {
                qg[r] += qdot8<BITS>(wg[r][c], xv);
                qu[r] += qdot8<BITS>(wu[r][c], xv);
            }
        }
        for (int r = 0; r < R; r++) {  // (one group a lane: the simd_sum is each row's whole sum)
            const T gt = T(simd_sum(sg[r] * qg[r] + bg[r] * xs)), ut = T(simd_sum(su[r] * qu[r] + bu[r] * xs));
            if (lane == 0) {
                const T sig = mlx_sigmoid<T>(gt);
                const T act = T(float(gt) * float(sig));
                h[size_t(slot) * I + j0 + r] = T(float(act) * float(ut));
            }
        }
    }
"""

# One simdgroup per (slot in expert order, 32 / (I / 64) * R output rows): a lane takes one of a row's I / 64 groups in
# R rows, the run's weight words loaded once, each chunk of h converted and prescaled once for the R rows. Each slot's
# down projection, rounded to T as gather_qmm rounds it, goes to ``ys``, and _COMBINE_SRC sums each token's slots in
# their order with the routing weights (float32), plus the shared expert.
_DOWN_SRC = """
    constexpr int NG = I / 64, WPG = 2 * BITS, RPS = 32 / NG;
    constexpr uint ROWW = NG * WPG, ROWS = RPS * R;
    const uint sgid = thread_position_in_grid.x / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint S = nslots[0];
    const uint rb = sgid % (D / ROWS), p = sgid / (D / ROWS);
    const uint g = lane % NG, row0 = rb * ROWS + lane / NG;  // this lane's rows: row0 + r * RPS
    if (p >= S) return;
    const uint e = inds[order[p]];
    if (p > 0 && inds[order[p - 1]] == e) return;
    uint q_end = p + 1;
    while (q_end < S && inds[order[q_end]] == e) q_end++;
    typename chunk_type<(BITS * 8 > 32)>::type wd[R][8];
    float sc[R], bi[R];
    for (int r = 0; r < R; r++) {
        const size_t wrow = size_t(e) * D + row0 + r * RPS;
        const device uint32_t* wp = w + wrow * ROWW + g * WPG;
        for (int c = 0; c < 8; c++) wd[r][c] = chunk8<BITS>(wp, c);
        sc[r] = float(scales[wrow * NG + g]);
        bi[r] = float(biases[wrow * NG + g]);
    }
    for (uint q = p; q < q_end; q++) {
        const uint slot = order[q];
        const device T* hr = h + size_t(slot) * I + g * 64;
        float qv[R], xs = 0.0f;
        for (int r = 0; r < R; r++) qv[r] = 0.0f;
        for (int c = 0; c < 8; c++) {
            float xv[8];
            for (int i = 0; i < 8; i++) { xv[i] = float(hr[c * 8 + i]); xs += xv[i]; }
            prescale8<BITS>(xv);
            for (int r = 0; r < R; r++) qv[r] += qdot8<BITS>(wd[r][c], xv);
        }
        for (int r = 0; r < R; r++) {
            float pv = sc[r] * qv[r] + bi[r] * xs;
            for (uint off = NG / 2; off > 0; off /= 2) pv += simd_shuffle_xor(pv, off);
            if (g == 0) ys[size_t(slot) * D + row0 + r * RPS] = T(pv);
        }
    }
"""

_COMBINE_SRC = """
    const uint i = thread_position_in_grid.x;
    const uint t = i / D, row = i % D;
    if (t >= ntok[0]) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += float(ys[size_t(t * K + k) * D + row]) * weights[t * K + k];
    T yv = T(acc);
    if (HAS_SHARED) yv = T(float(yv) + float(shared[size_t(t) * D + row]));
    if (HAS_RES) yv = T(float(res[size_t(t) * D + row]) + float(yv));  // the layer's residual add, as h + mlp(...)
    y[size_t(t) * D + row] = yv;
"""

# The combine, the layer's residual add and the next layer's input norm in one kernel: a threadgroup per token, laid
# out as norm_kernel.add_rms (MLX's RMS norm), so its two outputs are the bits of the combine and of rms_norm.
_COMBINE_NORM_SRC = """
    constexpr int N_READS = 4;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    const uint t = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    T hv[N_READS];
    float sq = 0;
    for (int i = 0; i < N_READS; i++) {
        const uint row = lid * N_READS + i;
        float acc = 0.0f;
        for (int k = 0; k < K; k++) acc += float(ys[size_t(t * K + k) * D + row]) * weights[t * K + k];
        T yv = T(acc);
        if (HAS_SHARED) yv = T(float(yv) + float(shared[size_t(t) * D + row]));
        yv = T(float(res[size_t(t) * D + row]) + float(yv));
        y[size_t(t) * D + row] = yv;
        hv[i] = yv;
        float xi = hv[i];
        sq += xi * xi;
    }
    sq = simd_sum(sq);
    if (sg == 0) local_sums[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) local_sums[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        sq = simd_sum(local_sums[lane]);
        if (lane == 0) local_inv_mean[0] = metal::precise::rsqrt(sq / D + __EPS__);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_READS; i++) {
        const uint row = lid * N_READS + i;
        n[size_t(t) * D + row] = nw[row] * static_cast<T>(hv[i] * local_inv_mean[0]);
    }
"""

# The combine as Qwen3-Next/3.5's block computes it in T (mlx-lm's Qwen3NextSparseMoeBlock): each slot's output times
# its weight rounded, summed in slot order with each partial sum rounded (an MLX sum of a 16-bit axis of 8 is that
# running sum), then the shared expert times MLX's sigmoid of its gate's logit (column GATE_COL of ``gate``), and the
# layer's residual add.
_COMBINE_GATED_SRC = """
    const uint i = thread_position_in_grid.x;
    const uint t = i / D, row = i % D;
    if (t >= ntok[0]) return;
    T acc = T(float(ys[size_t(t * K) * D + row]) * weights[t * K]);
    for (int k = 1; k < K; k++) acc = T(float(acc) + float(T(float(ys[size_t(t * K + k) * D + row]) * weights[t * K + k])));
    const T g = mlx_sigmoid<T>(T(gate[size_t(t) * gate_shape[1] + GATE_COL]));
    acc = T(float(acc) + float(T(float(g) * float(shared[size_t(t) * D + row]))));
    if (HAS_RES) acc = T(float(res[size_t(t) * D + row]) + float(acc));
    y[size_t(t) * D + row] = acc;
"""

# The same, then the next layer's input norm of the layer's output, laid out as _COMBINE_NORM_SRC (MLX's RMS norm).
_COMBINE_GATED_NORM_SRC = """
    constexpr int N_READS = 4;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    const uint t = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const T g = mlx_sigmoid<T>(T(gate[size_t(t) * gate_shape[1] + GATE_COL]));
    T hv[N_READS];
    float sq = 0;
    for (int i = 0; i < N_READS; i++) {
        const uint row = lid * N_READS + i;
        T acc = T(float(ys[size_t(t * K) * D + row]) * weights[t * K]);
        for (int k = 1; k < K; k++) acc = T(float(acc) + float(T(float(ys[size_t(t * K + k) * D + row]) * weights[t * K + k])));
        acc = T(float(acc) + float(T(float(g) * float(shared[size_t(t) * D + row]))));
        acc = T(float(res[size_t(t) * D + row]) + float(acc));
        y[size_t(t) * D + row] = acc;
        hv[i] = acc;
        float xi = hv[i];
        sq += xi * xi;
    }
    sq = simd_sum(sq);
    if (sg == 0) local_sums[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) local_sums[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        sq = simd_sum(local_sums[lane]);
        if (lane == 0) local_inv_mean[0] = metal::precise::rsqrt(sq / D + __EPS__);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_READS; i++) {
        const uint row = lid * N_READS + i;
        n[size_t(t) * D + row] = nw[row] * static_cast<T>(hv[i] * local_inv_mean[0]);
    }
"""

_up_kernel = _down_kernel = _combine_kernel = _combine_gated_kernel = None
_combine_gated_norm_kernels: dict = {}
_combine_norm_kernels: dict = {}
USE_EXPERT_KERNELS = os.environ.get("LOCALS_EXPERT_KERNELS", "1") != "0"
EXPERT_MAX_ROWS = 16  # more tokens (a prompt): gather_qmm, sorted, reads each expert once for all its tokens
# From this many tokens a simdgroup of experts_up takes 2 hidden units and a lane of experts_down 4 output rows; fewer
# tokens have too few simdgroups to spread over the GPU that way.
EXPERT_ROWS_FROM = 4


def experts_fit(x: mx.array, gate_up, down) -> bool:
    """Whether experts_up/experts_down take these: few rows of a 16-bit dtype, affine groups of 64, input sizes the
    kernels split evenly (the input in at most 32 groups, one a lane; the down projection's in at most 32, a power
    of 2)."""
    D, I = x.shape[-1], down.input_dims
    ng = I // 64
    return (x.size // D <= EXPERT_MAX_ROWS and x.dtype in (mx.bfloat16, mx.float16)
            and all(m.group_size == 64 and m.mode == "affine" and m.get("biases") is not None and m.bits in (2, 3, 4, 5, 6, 8)
                    for m in (gate_up, down))
            and D % 64 == 0 and D // 64 <= 32 and I % 64 == 0 and 0 < ng <= 32 and ng & (ng - 1) == 0
            and D % (32 // ng) == 0 and I % 2 == 0
            and gate_up["scales"].dtype == x.dtype and down["scales"].dtype == x.dtype)


def _experts_buffers(q):
    """(weight, scales, biases) of ``q``'s first expert, views of their buffers, for a kernel that reads the experts
    ``inds`` names from those buffers. MLX commits a command buffer once its inputs pass MLX_MAX_MB_PER_BUFFER (50 MB
    on an M4 Pro), counting each input at its size; a layer's whole expert tensors (Flint's: 350 MB, of which a token
    reads 8) made that every expert kernel, about 94 commits a step: 10.2 ms a Flint step, 8.75 with the views (the
    same tokens). Kept on ``q`` while its arrays are the same (a view made each call is a graph node each call)."""
    src = (q["weight"], q["scales"], q["biases"])
    c = getattr(q, "_one_expert", None)
    if c is not None and c[0] is src[0] and c[1] is src[1] and c[2] is src[2]:
        return c[3:]
    views = tuple(a[:1] for a in src)
    mx.eval(views)
    try:
        object.__setattr__(q, "_one_expert", src + views)  # (not a parameter: outside the module's mapping)
    except AttributeError:
        pass
    return views


def expert_order(inds: mx.array) -> mx.array:
    """The slots of ``inds`` (T, K), flattened, in the order of their experts (a token's own k are distinct: one token
    needs no sort)."""
    flat = inds.reshape(-1)
    if inds.shape[0] == 1:
        return mx.arange(flat.size, dtype=mx.uint32)
    return mx.argsort(flat).astype(mx.uint32)


def experts_up(x: mx.array, inds: mx.array, gate_up, order: Optional[mx.array] = None, r: Optional[int] = None) -> mx.array:
    """swiglu(gate, up) of each token for each of its experts: ``x`` (T, D), ``inds`` (T, K) -> (T * K, I). Rows that
    share an expert read it once (``order``: expert_order(inds)): at 8 and 16 rows of Flint's routing 10 to 12% faster
    than a simdgroup a slot, at one row 2%. ``r`` hidden units a simdgroup (default: 2 from EXPERT_ROWS_FROM tokens)
    give the same bits."""
    global _up_kernel
    if _up_kernel is None:
        _up_kernel = mx.fast.metal_kernel(name="locals_experts_up_units",
                                          input_names=["x", "inds", "order", "w", "scales", "biases", "nslots"],
                                          output_names=["h"], source=_UP_SRC, header=_QDOT_HEADER)
    T_, D = x.shape
    K = inds.shape[-1]
    I = gate_up["weight"].shape[1] // 2
    S = T_ * K
    r = r or (2 if T_ >= EXPERT_ROWS_FROM else 1)
    order = expert_order(inds) if order is None else order
    w, sc, bi = _experts_buffers(gate_up)
    (h,) = _up_kernel(inputs=[x, inds.reshape(-1), order, w, sc, bi, mx.array([S], dtype=mx.uint32)],
                      template=[("T", x.dtype), ("BITS", gate_up.bits), ("D", D), ("I", I), ("K", K), ("R", r)],
                      grid=(S * (I // r) * 32, 1, 1), threadgroup=(64, 1, 1), output_shapes=[(S, I)], output_dtypes=[x.dtype])
    return h


def experts_down(h: mx.array, inds: mx.array, weights: mx.array, down, shared=None, order: Optional[mx.array] = None,
                 r: Optional[int] = None, residual=None, norm=None, gate=None, gate_col: int = 0):
    """sum_k weights[t, k] * down_{inds[t, k]}(h[t, k]) (+ ``shared``) (+ ``residual``, added last as the decoder layer
    adds it): ``h`` (T * K, I) -> (T, D). Each slot's projection (rows sharing an expert read it once), then a combine
    that sums a token's slots in their order. ``r`` output rows a lane (default: 4 from EXPERT_ROWS_FROM tokens) give
    the same bits. With ``residual`` and ``norm`` (an RMSNorm, the next layer's input norm): (that sum, its norm).
    With ``gate`` (Qwen3-Next/3.5's block): the sum rounded as that block rounds it, and ``shared`` times the sigmoid of
    ``gate[:, gate_col]`` (_COMBINE_GATED_SRC)."""
    global _down_kernel, _combine_kernel
    if _down_kernel is None:
        _down_kernel = mx.fast.metal_kernel(name="locals_experts_down_rows",
                                            input_names=["h", "inds", "order", "w", "scales", "biases", "nslots"],
                                            output_names=["ys"], source=_DOWN_SRC, header=_QDOT_HEADER)
        _combine_kernel = mx.fast.metal_kernel(name="locals_experts_combine",
                                               input_names=["ys", "weights", "shared", "res", "ntok"],
                                               output_names=["y"], source=_COMBINE_SRC)
    NT, K = inds.shape
    D, I = down["weight"].shape[1], h.shape[-1]
    S = NT * K
    rps = 32 // (I // 64)
    r = r or (4 if NT >= EXPERT_ROWS_FROM and D % (4 * rps) == 0 else 1)
    order = expert_order(inds) if order is None else order
    w, sc, bi = _experts_buffers(down)
    (ys,) = _down_kernel(inputs=[h, inds.reshape(-1), order, w, sc, bi, mx.array([S], dtype=mx.uint32)],
                         template=[("T", h.dtype), ("BITS", down.bits), ("D", D), ("I", I), ("K", K), ("R", r)],
                         grid=(S * (D // (rps * r)) * 32, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(S, D)],
                         output_dtypes=[h.dtype])
    if gate is not None:
        return combine_gated(ys, weights, shared, gate, gate_col, residual, norm)
    has, res = shared is not None, residual is not None
    none = mx.zeros((1,), dtype=h.dtype)
    if res and norm is not None:
        from .norm_kernel import supported

        if supported(D):
            eps = float(norm.eps)
            kern = _combine_norm_kernels.get(eps)
            if kern is None:
                kern = _combine_norm_kernels[eps] = mx.fast.metal_kernel(
                    name="locals_experts_combine_norm_%d" % len(_combine_norm_kernels),
                    input_names=["ys", "weights", "shared", "res", "nw"], output_names=["y", "n"],
                    source=_COMBINE_NORM_SRC.replace("__EPS__", "%.9ef" % eps))
            y, n = kern(inputs=[ys, weights.astype(mx.float32), shared if has else none, residual, norm.weight],
                        template=[("T", h.dtype), ("D", D), ("K", K), ("HAS_SHARED", int(has))],
                        grid=(NT * (D // 4), 1, 1), threadgroup=(D // 4, 1, 1), output_shapes=[(NT, D)] * 2,
                        output_dtypes=[h.dtype] * 2)
            return y, n
    (y,) = _combine_kernel(inputs=[ys, weights.astype(mx.float32), shared if has else none, residual if res else none,
                                   mx.array([NT], dtype=mx.uint32)],
                           template=[("T", h.dtype), ("D", D), ("K", K), ("HAS_SHARED", int(has)), ("HAS_RES", int(res))],
                           grid=(NT * D, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(NT, D)], output_dtypes=[h.dtype])
    return y if norm is None else (y, norm(y))


def combine_gated(ys: mx.array, weights: mx.array, shared: mx.array, gate: mx.array, gate_col: int = 0, residual=None,
                  norm=None):
    """Qwen3-Next/3.5's combine (_COMBINE_GATED_SRC): ``ys`` (T * K, D), each slot's expert output, with the routing
    ``weights`` (T, K), plus ``shared`` (T, D) times the sigmoid of ``gate[:, gate_col]`` (+ ``residual``); with
    ``residual`` and ``norm`` (the next layer's input norm): (that sum, its norm). The bits of mlx-lm's block."""
    global _combine_gated_kernel
    NT, K = weights.shape
    D = ys.shape[-1]
    res = residual is not None
    if res and norm is not None:
        from .norm_kernel import supported

        if supported(D):
            eps = float(norm.eps)
            kern = _combine_gated_norm_kernels.get(eps)
            if kern is None:
                kern = _combine_gated_norm_kernels[eps] = mx.fast.metal_kernel(
                    name="locals_experts_combine_gated_norm_%d" % len(_combine_gated_norm_kernels),
                    input_names=["ys", "weights", "shared", "gate", "res", "nw"], output_names=["y", "n"],
                    source=_COMBINE_GATED_NORM_SRC.replace("__EPS__", "%.9ef" % eps), header=_QDOT_HEADER)
            y, n = kern(inputs=[ys, weights.astype(mx.float32), shared, gate, residual, norm.weight],
                        template=[("T", ys.dtype), ("D", D), ("K", K), ("GATE_COL", gate_col)],
                        grid=(NT * (D // 4), 1, 1), threadgroup=(D // 4, 1, 1), output_shapes=[(NT, D)] * 2,
                        output_dtypes=[ys.dtype] * 2)
            return y, n
    if _combine_gated_kernel is None:
        _combine_gated_kernel = mx.fast.metal_kernel(name="locals_experts_combine_gated",
                                                     input_names=["ys", "weights", "shared", "gate", "res", "ntok"],
                                                     output_names=["y"], source=_COMBINE_GATED_SRC, header=_QDOT_HEADER)
    (y,) = _combine_gated_kernel(
        inputs=[ys, weights.astype(mx.float32), shared, gate, residual if res else mx.zeros((1,), dtype=ys.dtype),
                mx.array([NT], dtype=mx.uint32)],
        template=[("T", ys.dtype), ("D", D), ("K", K), ("GATE_COL", gate_col), ("HAS_RES", int(res))],
        grid=(NT * D, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(NT, D)], output_dtypes=[ys.dtype])
    return y if norm is None else (y, norm(y))


def _concat_switch(mods):
    """One QuantizedSwitchLinear for ``mods`` stacked along each expert's output (the same quantization)."""
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    first = mods[0]
    if any(not isinstance(m, QuantizedSwitchLinear) for m in mods) or len({(m.bits, m.group_size, m.mode) for m in mods}) != 1:
        raise ValueError("expert projections with different quantization do not fuse")
    if any(m.get("biases") is None or "bias" in m or m.get("global_scale") is not None for m in mods):
        raise ValueError("only affine expert projections without a bias fuse")
    q = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)  # (the constructor quantizes random weights)
    nn.Module.__init__(q)
    q.group_size, q.bits, q.mode = first.group_size, first.bits, first.mode
    q.weight = mx.concatenate([m["weight"] for m in mods], axis=1)
    q.scales = mx.concatenate([m["scales"] for m in mods], axis=1)
    q.biases = mx.concatenate([m["biases"] for m in mods], axis=1)
    q.global_scale = None
    q.freeze()
    return q


class FusedSwitchGLU(nn.Module):
    """SwitchGLU with gate and up as one gathered matmul."""

    def __call__(self, x, indices) -> mx.array:
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        x = mx.expand_dims(x, (-2, -3))
        indices = mx.stop_gradient(indices)
        do_sort = indices.size >= 64  # as SwitchGLU: many tokens read the experts in order
        idx, inv_order = indices, None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        gu = self.gate_up_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(swiglu(gu[..., : self.hidden], gu[..., self.hidden :]), idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


def fuse_switch_glu(sw) -> bool:
    """Fuse a SwitchGLU's gate and up in place; False (unchanged) when they do not fuse or the activation is not
    SwiGLU."""
    from mlx_lm.models.switch_layers import SwiGLU

    if type(sw).__name__ != "SwitchGLU" or not isinstance(getattr(sw, "activation", None), SwiGLU):
        return False
    try:
        gu = _concat_switch([sw.gate_proj, sw.up_proj])
    except ValueError:
        return False
    sw.hidden = sw.gate_proj.output_dims
    sw.gate_up_proj = gu
    del sw.gate_proj, sw.up_proj
    sw.__class__ = type("Fused" + type(sw).__name__, (FusedSwitchGLU, type(sw)), {})
    return True


class FusedFlintMoe(nn.Module):
    """Flint's SparseMoeBlock: the router's float32 logits from its bf16 weight in one kernel, the choice (and the
    slots' order for the expert kernels) in another. With ``residual``, returns ``residual`` + the block's output, the
    add inside the last kernel (FusedFlintLayer)."""

    def __call__(self, x: mx.array, residual: Optional[mx.array] = None, norm=None):
        """With ``norm`` (and ``residual``): (residual + the block's output, ``norm`` of it)."""
        sw, D, k = self.switch_mlp, x.shape[-1], self.top_k
        if USE_EXPERT_KERNELS and hasattr(sw, "gate_up_proj") and experts_fit(x, sw.gate_up_proj, sw.down_proj):
            inds, weights, order = route_topk(router(x, self.gate.weight), k, order=True)
            inds2 = inds.reshape(-1, k)
            h = experts_up(x.reshape(-1, D), inds2, sw.gate_up_proj, order)
            out = experts_down(h, inds2, weights.reshape(-1, k), sw.down_proj, self.shared_expert(x).reshape(-1, D), order,
                               residual=None if residual is None else residual.reshape(-1, D), norm=norm)
            if norm is None:
                return out.reshape(x.shape)
            return out[0].reshape(x.shape), out[1].reshape(x.shape)
        inds, weights = route_topk(router(x, self.gate.weight), k)
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        y = y + self.shared_expert(x)
        if norm is not None:
            return _add_norm(residual, y, norm)
        return y if residual is None else residual + y


FLINT_ADD_RMS = os.environ.get("LOCALS_FLINT_ADD_RMS", "1") != "0"


def _add_norm(x: mx.array, r: mx.array, norm):
    """(x + r, norm(x + r)): one kernel where norm_kernel.add_rms takes them (the same bits as MLX's add and norm)."""
    from .norm_kernel import add_rms, supported

    if FLINT_ADD_RMS and supported(x.shape[-1]) and x.dtype in (mx.bfloat16, mx.float16) and r.dtype == x.dtype:
        return add_rms(x, r, norm.weight, norm.eps)
    h = x + r
    return h, norm(h)


class FusedFlintLayer(nn.Module):
    """Flint's decoder layer with the attention's residual add in the post-attention norm's kernel (_add_norm), and a
    MoE block whose residual add is the expert combine's (FusedFlintMoe). ``step`` also computes the next layer's
    input norm with the layer's last kernel (FusedFlintBody)."""

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache=None) -> mx.array:
        h, n = _add_norm(x, self.self_attn(self.input_layernorm(x), mask, cache), self.post_attention_layernorm)
        return self.mlp(n, residual=h) if isinstance(self.mlp, FusedFlintMoe) else h + self.mlp(n)

    def step(self, x: mx.array, nx: mx.array, mask, cache, norm):
        """(this layer's output, ``norm`` of it), from its input ``x`` and its input norm ``nx``."""
        h, n = _add_norm(x, self.self_attn(nx, mask, cache), self.post_attention_layernorm)
        if isinstance(self.mlp, FusedFlintMoe):
            return self.mlp(n, residual=h, norm=norm)
        return _add_norm(h, self.mlp(n), norm)


class FusedFlintBody(nn.Module):
    """Flint's layer stack with each layer's input norm computed by the last kernel of the layer before it (the expert
    combine, or the add after a dense MLP): one kernel a layer fewer, the same bits."""

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None) -> mx.array:
        from mlx_lm.models.base import create_attention_mask

        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        swa_mask = create_attention_mask(h, cache[self.swa_idx], window_size=self.args.sliding_window)
        layers = self.layers
        n = layers[0].input_layernorm(h)
        for i, (layer, c) in enumerate(zip(layers, cache)):
            nxt = layers[i + 1].input_layernorm if i + 1 < len(layers) else self.norm
            h, n = layer.step(h, n, swa_mask if layer.use_sliding else fa_mask, c, nxt)
        return n


def fuse_flint_moe(block) -> bool:
    """Flint's SparseMoeBlock (Locals's or mlx-lm's), in place: router, experts and shared expert; False if it is
    another kind of block."""
    from .fuse import _fuse_mlp, _fusable, _swiglu_mlp

    if type(block).__name__ != "SparseMoeBlock" or type(block).__module__.split(".")[-1] != "flint":
        return False
    if getattr(block, "sharding_group", None) is not None or not isinstance(block.gate, nn.Linear) or "bias" in block.gate:
        return False
    if isinstance(block.gate, nn.QuantizedLinear):
        return False
    fuse_switch_glu(block.switch_mlp)
    se = block.shared_expert
    if isinstance(getattr(se, "gate_proj", None), nn.QuantizedLinear) and _fusable([se.gate_proj, se.up_proj]) and _swiglu_mlp(se):
        _fuse_mlp(se)
    block.__class__ = type("Fused" + type(block).__name__, (FusedFlintMoe, type(block)), {})
    return True


class FusedQwenMoe(nn.Module):
    """Qwen3-Next/3.5's SparseMoeBlock (mlx-lm's Qwen3NextSparseMoeBlock: Qwen3.5-MoE, Clover's causal pass). The
    router and the shared expert's gate are one matmul (``gates``: the E logits, then the gate's), the choice one kernel
    (route_softmax); a few tokens' experts go through experts_up / experts_down, whose combine adds the gated shared
    expert and, with ``residual``, the layer's residual add (FusedQwenLayer). Rounded where mlx-lm's block rounds."""

    def _logits(self, x):
        """(router logits, the array holding the shared expert's gate logit, its column)."""
        if "gates" in self:
            g = self.gates(x)
            return g, g, self.num_experts
        return self.gate(x), self.shared_expert_gate(x), 0

    def __call__(self, x: mx.array, residual: Optional[mx.array] = None, norm=None):
        """With ``norm`` (and ``residual``): (residual + the block's output, ``norm`` of it), both from the combine."""
        sw, D, k, E = self.switch_mlp, x.shape[-1], self.top_k, self.num_experts
        logits, gl, col = self._logits(x)
        if USE_EXPERT_KERNELS and hasattr(sw, "gate_up_proj") and experts_fit(x, sw.gate_up_proj, sw.down_proj):
            inds, weights, order = route_softmax(logits, E, k, self.norm_topk_prob, order=True)
            x2, inds2 = x.reshape(-1, D), inds.reshape(-1, k)
            h = experts_up(x2, inds2, sw.gate_up_proj, order)
            y = experts_down(h, inds2, weights.reshape(-1, k), sw.down_proj, self.shared_expert(x).reshape(-1, D), order,
                             residual=None if residual is None else residual.reshape(-1, D), norm=norm,
                             gate=gl.reshape(-1, gl.shape[-1]), gate_col=col)
            if norm is None:
                return y.reshape(x.shape)
            return y[0].reshape(x.shape), y[1].reshape(x.shape)
        gates = mx.softmax(logits[..., :E], axis=-1, precise=True)  # mlx-lm's block (a prompt)
        inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        y = (self.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2)
        y = y + mx.sigmoid(gl[..., col : col + 1]) * self.shared_expert(x)
        if residual is not None:
            y = residual + y
        return y if norm is None else (y, norm(y))


class FusedQwenLayer(nn.Module):
    """Qwen3.5's decoder layer with a FusedQwenMoe block, the block's residual add in its last kernel. ``step`` also
    has the mixer's residual add and the post-mixer norm in one kernel (_add_norm), and the next layer's input norm
    computed by the combine (fuse.FusedHybridBody, with ``_moe_steps``)."""

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache=None) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        return self.mlp(self.post_attention_layernorm(h), residual=h)

    def step(self, x: mx.array, nx: mx.array, mask, cache, norm):
        """(this layer's output, ``norm`` of it), from its input ``x`` and its input norm ``nx``."""
        r = self.linear_attn(nx, mask, cache) if self.is_linear else self.self_attn(nx, mask, cache)
        h, n = _add_norm(x, r, self.post_attention_layernorm)
        return self.mlp(n, residual=h, norm=norm)


def fuse_qwen_moe(block) -> bool:
    """Qwen3-Next/3.5's SparseMoeBlock (mlx-lm's), in place: expert gate/up, the shared expert's gate/up, the router
    with the shared expert's gate; False if it is another kind of block."""
    from .fuse import _concat_quantized, _fusable, _fuse_mlp, _swiglu_mlp

    if type(block).__name__ != "Qwen3NextSparseMoeBlock" or type(block).__module__.split(".")[-1] != "qwen3_next":
        return False
    if getattr(block, "sharding_group", None) is not None:
        return False
    fuse_switch_glu(block.switch_mlp)
    se = block.shared_expert
    if isinstance(getattr(se, "gate_proj", None), nn.QuantizedLinear) and _fusable([se.gate_proj, se.up_proj]) and _swiglu_mlp(se):
        _fuse_mlp(se)
    g, sg = block.gate, block.shared_expert_gate
    if all(isinstance(m, nn.QuantizedLinear) for m in (g, sg)) and _fusable([g, sg]):
        block.gates = _concat_quantized([g, sg])
        del block.gate, block.shared_expert_gate
    block.__class__ = type("Fused" + type(block).__name__, (FusedQwenMoe, type(block)), {})
    return True
