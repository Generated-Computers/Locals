"""Inkling 2's one-token step, a visit in a few kernels.

A token passes 48 layer visits (models/inkling2_loop.py), and around each visit's weight reads (attention's
projections, eight experts) the model does a great many small things to vectors of 2304: norms, attention over
depth, the loop's gates, four short convolutions, clips, a bias by distance, a softcapped SwiGLU. As MLX operations
that is about 125 a visit, 6000 a token: 34 ms of a 55 ms step on an M4 Pro, where the weight reads take 21.
Here each stretch between two weight reads is one kernel:

* ``mix``: a norm, attention over depth, the loop's gate and the layer norm; after attention also the convolution
  of its output and the residual. It makes the inputs of attention, of the experts and of the output layer, and
  the messages of the pass before.
* ``kv_prep``: from the projections to the clipped queries, the bias of every distance, and the new key and value
  (their convolutions, clip and norm).
* ``attend``: attention of the one query over a window with its bias, sink and output gate.
* ``route``: the router's scores and its choice of experts.
* ``act``: the softcapped SwiGLU on the experts' interleaved gate and up rows.
* ``combine``: the weighted sum of the experts, its convolution and the residual.

The weight reads stay MLX's (8-bit matrix-vector products, gather_qmm on NVFP4): on one row they run at the memory's
speed already. A step is taken this way for one token a row; a prompt, and a batch's rows still padded, run the reference, as
does a global layer's attention past 1024 tokens (MLX's own), and both leave the same cache.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, BatchKVCache, BatchRotatingKVCache, KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

from . import inkling_experts as once
from .models import inkling2_loop as ink

_F32 = mx.float32
_KV = (KVCache, RotatingKVCache, BatchKVCache, BatchRotatingKVCache)
SEQ_MAX = 32  # tokens of one text a pass takes in kernels (a forced span, a draft); longer passes: the reference

# Sums over a threadgroup: each lane's value summed over its simdgroup, the simdgroups' sums added by every thread.
_HEADER = """
#define REDUCE(n, vals, tots) \\
    threadgroup_barrier(mem_flags::mem_threadgroup); \\
    for (int i_ = 0; i_ < (n); i_++) { const float s_ = simd_sum((vals)[i_]); if (lane == 0) buf[i_][sg] = s_; } \\
    threadgroup_barrier(mem_flags::mem_threadgroup); \\
    for (int i_ = 0; i_ < (n); i_++) { float t_ = 0.0f; for (int g_ = 0; g_ < NS; g_++) t_ += buf[i_][g_]; (tots)[i_] = t_; }
#define REDUCE2(n, vals, tots) \\
    threadgroup_barrier(mem_flags::mem_threadgroup); \\
    for (int i_ = 0; i_ < (n); i_++) { const float s_ = simd_sum((vals)[i_]); if (lane == 0) buf[i_][sg] = s_; } \\
    threadgroup_barrier(mem_flags::mem_threadgroup); \\
    if (sg == 0) for (int i_ = 0; i_ < (n); i_++) { \\
        const float s_ = simd_sum(lane < NS ? buf[i_][lane] : 0.0f); if (lane == 0) red[i_] = s_; } \\
    threadgroup_barrier(mem_flags::mem_threadgroup); \\
    for (int i_ = 0; i_ < (n); i_++) (tots)[i_] = red[i_];
inline float sigmoid_(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }
"""

# A GPU thread takes 25 to 45 ns an iteration of a loop, and a step is one chain of kernels, each waiting for the
# one before: a kernel whose threads loop 2000 times costs 70 us however little it computes. So the work is spread
# over a threadgroup of 1024 (the most there is), a few hundred iterations a thread at most.

_MIX_SRC = """
    constexpr int NS = TG / 32;
    constexpr int DPT = (D + TG - 1) / TG;
    constexpr int NSRC = K + 1;
    constexpr int NCH = TG / RANK;  // (the gate: a thread a rank and a chunk of the dimensions)
    constexpr int CD = (D + NCH - 1) / NCH;
    const uint j = threadgroup_position_in_grid.x;  // an output a threadgroup: the loop's three gates side by side
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int M = x_shape[0];
    const float eps = prm[0];
    const int jq = Q == 1 ? 0 : j;
    threadgroup float buf[NRED][NS];
    threadgroup float red[NRED];

    // the state this reads: x, or after attention the convolution of x plus the residual
    float v[DPT];
    for (int c = 0; c < DPT; c++) {
        const int d = tid + c * TG;
        float val = 0.0f;
        if (d < D) {
            val = float(x[size_t(row) * D + d]);
            if (CONV) {
                const size_t so = size_t(row) * 3 * D + d, to = size_t(d) * 4;
                float s0, s1, s2;
                if (SEQ) {  // rows are one text's tokens: a token's earlier inputs are the rows before it
                    const int r = int(row);
                    s0 = r >= 3 ? float(x[size_t(r - 3) * D + d]) : state[size_t(r) * D + d];
                    s1 = r >= 2 ? float(x[size_t(r - 2) * D + d]) : state[size_t(r + 1) * D + d];
                    s2 = r >= 1 ? float(x[size_t(r - 1) * D + d]) : state[size_t(r + 2) * D + d];
                } else {
                    s0 = state[so]; s1 = state[so + D]; s2 = state[so + 2 * D];
                }
                float h = (taps[to] * s0 + taps[to + 1] * s1 + taps[to + 2] * s2 + taps[to + 3] * val + val) * 0.4472135955f;
                h = float(T(h));
                new_state[so] = s1; new_state[so + D] = s2; new_state[so + 2 * D] = val;
                val = ADD ? float(res[size_t(row) * D + d]) + h : h;
                resid[size_t(row) * D + d] = val;
            }
        }
        v[c] = val;
    }
    float one[1], tot1[1];
    one[0] = 0.0f;
    for (int c = 0; c < DPT; c++) one[0] += v[c] * v[c];
    REDUCE2(1, one, tot1)
    const float inv = metal::precise::rsqrt(tot1[0] / D + 1e-6f);
    for (int c = 0; c < DPT; c++) {
        const int d = tid + c * TG;
        v[c] *= inv;
        if (d < D && j == 0) partial[size_t(row) * D + d] = v[c];
    }

    // attention over depth: the block states and this state, by this output's query
    float acc[NSRC], w[NSRC];
    for (int i = 0; i < NSRC; i++) acc[i] = 0.0f;
    for (int c = 0; c < DPT; c++) {
        const int d = tid + c * TG;
        if (d >= D) continue;
        const float qv = queries[size_t(jq) * D + d];
        for (int i = 0; i < NSRC; i++) acc[i] += (i < K ? blocks[(size_t(i) * M + row) * D + d] : v[c]) * qv;
    }
    REDUCE2(NSRC, acc, w)
    const float sc = metal::precise::rsqrt(float(D));
    float mw = -INFINITY, zw = 0.0f;
    for (int i = 0; i < NSRC; i++) mw = max(mw, w[i] * sc);
    for (int i = 0; i < NSRC; i++) { w[i] = metal::precise::exp(w[i] * sc - mw); zw += w[i]; }
    float o[DPT];
    for (int c = 0; c < DPT; c++) {
        const int d = tid + c * TG;
        float s = 0.0f;
        if (d < D) for (int i = 0; i < NSRC; i++) s += w[i] / zw * (i < K ? blocks[(size_t(i) * M + row) * D + d] : v[c]);
        o[c] = s;
    }

    // the loop's gate: some of the state forgotten, some of the pass before's message added
    if (GATE) {
        threadgroup float on[D];
        threadgroup float mn[D];
        threadgroup float pg[TG];
        threadgroup float low[RANK];
        float mm[DPT], two[2], tt[2];
        two[0] = 0.0f; two[1] = 0.0f;
        for (int c = 0; c < DPT; c++) {
            const int d = tid + c * TG;
            mm[c] = d < D ? messages[(size_t(row) * 4 + MOFF + j) * D + d] : 0.0f;
            two[0] += o[c] * o[c]; two[1] += mm[c] * mm[c];
        }
        REDUCE2(2, two, tt)
        const float io = metal::precise::rsqrt(tt[0] / D + 1e-6f), im = metal::precise::rsqrt(tt[1] / D + 1e-6f);
        for (int c = 0; c < DPT; c++) {
            const int d = tid + c * TG;
            if (d < D) { on[d] = float(GT(o[c] * io)); mn[d] = float(GT(mm[c] * im)); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int r = tid % RANK, chunk = tid / RANK;  // the low-rank projection of both: a rank's row, a chunk a thread
        const size_t base = (size_t(j) * RANK + r) * 2 * D;
        float gp = 0.0f;
        for (int d = chunk * CD; d < min(D, (chunk + 1) * CD); d++)
            gp += float(down[base + d]) * on[d] + float(down[base + D + d]) * mn[d];
        pg[tid] = gp;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (chunk == 0) {
            for (int c = 1; c < NCH; c++) gp += pg[c * RANK + r];
            low[r] = float(GT(gp));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = 0; c < DPT; c++) {
            const int d = tid + c * TG;
            if (d >= D) continue;
            const size_t fb = (size_t(j) * D + d) * RANK;
            float f = 0.0f, gn = 0.0f;
            for (int k = 0; k < RANK; k++) { f += forget[fb + k] * low[k]; gn += gain[fb + k] * low[k]; }
            o[c] = (1.0f - metal::precise::tanh(f)) * o[c] + metal::precise::tanh(gn) * mm[c];
        }
    }

    if (NORM) {
        float n2[1], nt[1];
        n2[0] = 0.0f;
        for (int c = 0; c < DPT; c++) n2[0] += o[c] * o[c];
        REDUCE2(1, n2, nt)
        const float invj = metal::precise::rsqrt(nt[0] / D + eps);
        for (int c = 0; c < DPT; c++) {
            const int d = tid + c * TG;
            if (d < D) out[(size_t(row) * QO + j) * D + d] = OT(o[c] * invj * float(norm_w[size_t(j % QN) * D + d]));
        }
    } else {
        for (int c = 0; c < DPT; c++) {
            const int d = tid + c * TG;
            if (d < D) out[(size_t(row) * QO + j) * D + d] = OT(o[c]);
        }
    }
"""

_KV_SRC = """
    constexpr int NS = TG / 32;
    constexpr int C = HKV * HD;
    const uint h = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const float qk_thr = prm[0], rel_thr = prm[1], eps = prm[2];
    const float tau_ = tau[row];
    threadgroup float buf[9][NS];
    threadgroup float rr[DR];

    // the head's sink logit, and in a global layer its key and value taps: small projections of the inputs
    float nine[9], dots[9];
    for (int i = 0; i < 9; i++) nine[i] = 0.0f;
    for (int d = t; d < D; d += TG) {
        nine[0] += float(xq[size_t(row) * D + d]) * float(sink_w[size_t(h) * D + d]);
        if (DYN && h < HKV) {
            const float kx = float(xk[size_t(row) * D + d]), vx = float(xv[size_t(row) * D + d]);
            for (int lag = 0; lag < 4; lag++) {
                nine[1 + lag] += kx * float(kdyn_w[size_t(h * 4 + lag) * D + d]);
                nine[5 + lag] += vx * float(vdyn_w[size_t(h * 4 + lag) * D + d]);
            }
        }
    }
    REDUCE(DYN ? 9 : 1, nine, dots)
    if (t == 0) sinks[size_t(row) * HQ + h] = T(dots[0]);

    float four[4], tot[4];
    const float qy = t < HD ? float(q[size_t(row) * QW + QOFF + h * HD + t]) * float(qscale[h * HD + t]) : 0.0f;
    const float ry = t < DR ? float(r[size_t(row) * RW + ROFF + h * DR + t]) * float(rscale[h * DR + t]) : 0.0f;
    float ky = 0.0f, vy = 0.0f;
    if (h < HKV && t < HD) {  // this key/value head's channel: the short convolutions over the last 4 tokens
        const int c = h * HD + t;
        const size_t so = size_t(row) * 3 * C + c;
        const float kx = float(k[size_t(row) * KW + KOFF + c]), vx = float(v[size_t(row) * VW + VOFF + c]);
        float k0, k1, k2, v0, v1, v2;
        if (SEQ) {  // rows are one text's tokens: a token's earlier inputs are the rows before it
            const int r = int(row);
            k0 = r >= 3 ? float(k[size_t(r - 3) * KW + KOFF + c]) : kstate[size_t(r) * C + c];
            k1 = r >= 2 ? float(k[size_t(r - 2) * KW + KOFF + c]) : kstate[size_t(r + 1) * C + c];
            k2 = r >= 1 ? float(k[size_t(r - 1) * KW + KOFF + c]) : kstate[size_t(r + 2) * C + c];
            v0 = r >= 3 ? float(v[size_t(r - 3) * VW + VOFF + c]) : vstate[size_t(r) * C + c];
            v1 = r >= 2 ? float(v[size_t(r - 2) * VW + VOFF + c]) : vstate[size_t(r + 1) * C + c];
            v2 = r >= 1 ? float(v[size_t(r - 1) * VW + VOFF + c]) : vstate[size_t(r + 2) * C + c];
        } else {
            k0 = kstate[so]; k1 = kstate[so + C]; k2 = kstate[so + 2 * C];
            v0 = vstate[so]; v1 = vstate[so + C]; v2 = vstate[so + 2 * C];
        }
        float kc, vc;
        if (DYN) {  // taps by lag (0: the token), learned ones plus this token's own, per head
            kc = ((float(ktaps[c]) + float(T(dots[1]))) * kx + (float(ktaps[C + c]) + float(T(dots[2]))) * k2
                + (float(ktaps[2 * C + c]) + float(T(dots[3]))) * k1 + (float(ktaps[3 * C + c]) + float(T(dots[4]))) * k0
                + kx) * 0.3333333333f;
            vc = ((float(vtaps[c]) + float(T(dots[5]))) * vx + (float(vtaps[C + c]) + float(T(dots[6]))) * v2
                + (float(vtaps[2 * C + c]) + float(T(dots[7]))) * v1 + (float(vtaps[3 * C + c]) + float(T(dots[8]))) * v0
                + vx) * 0.3333333333f;
        } else {
            const size_t to = size_t(c) * 4;
            kc = (float(ktaps[to]) * k0 + float(ktaps[to + 1]) * k1 + float(ktaps[to + 2]) * k2 + float(ktaps[to + 3]) * kx + kx)
                * 0.4472135955f;
            vc = (float(vtaps[to]) * v0 + float(vtaps[to + 1]) * v1 + float(vtaps[to + 2]) * v2 + float(vtaps[to + 3]) * vx + vx)
                * 0.4472135955f;
        }
        ky = float(T(kc)); vy = float(T(vc));
        kstate_new[so] = k1; kstate_new[so + C] = k2; kstate_new[so + 2 * C] = kx;
        vstate_new[so] = v1; vstate_new[so + C] = v2; vstate_new[so + 2 * C] = vx;
    }
    four[0] = qy * qy; four[1] = ry * ry; four[2] = ky * ky; four[3] = vy * vy;
    REDUCE(4, four, tot)

    if (t < HD) {
        float qv = float(T(qy * min(qk_thr / (metal::precise::sqrt(tot[0]) + 1e-6f), 1.0f)));
        if (TAU) qv = float(T(qv * tau_));
        qo[(size_t(row) * HQ + h) * HD + t] = T(qv);
        if (h < HKV) {
            const size_t xo = (size_t(row) * HKV + h) * HD + t;
            ko[xo] = T(ky * min(qk_thr / (metal::precise::sqrt(tot[2]) + 1e-6f), 1.0f));
            vo[xo] = T(vy * metal::precise::rsqrt(tot[3] / HD + eps));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < DR) {
        float rv = float(T(ry * min(rel_thr / (metal::precise::sqrt(tot[1]) + 1e-6f), 1.0f)));
        if (TAU) rv = float(T(rv * tau_));
        rr[t] = rv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = t; d < R; d += TG) {  // the bias of each distance
        float b = 0.0f;
        for (int i = 0; i < DR; i++) b += rr[i] * float(proj[size_t(i) * R + d]);
        bias[(size_t(row) * HQ + h) * R + d] = T(b);
    }
"""

_ATTEND_SRC = """
    constexpr int NS = TG / 32;
    constexpr int G = HQ / HKV;  // query heads of one key/value head: a threadgroup, the keys and values read once
    constexpr int CH = TG / HD;  // the chunks of keys a value's sum is made in
    constexpr int CK = (TG + CH - 1) / CH;
    const uint kvh = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int S = keys_shape[2], M = q_shape[0];
    const int brow = SEQ ? 0 : int(row);  // (rows of one text share its keys)
    const int idx = meta[0], count = counts[brow];
    const float scale = prm[0];
    threadgroup float p[G][TG];
    threadgroup float part[G][TG];
    threadgroup float buf[G][NS];
    threadgroup float red[G];
    const device T* kp = keys + size_t(brow) * keys_strides[0] + size_t(kvh) * keys_strides[1];
    const device T* vp = values + size_t(brow) * values_strides[0] + size_t(kvh) * values_strides[1];
    const size_t qb = (size_t(row) * HQ + kvh * G) * HD;  // the first of the G query heads

    // Keys by their place in time: the query's own is ``last`` (a text's tokens: the keys after it are later
    // tokens of the pass), a thread takes the key ``first + t``, in slot (place + idx) % S of a window written
    // round and round
    const int last = SEQ ? S - M + int(row) : S - 1;
    const int first = max(0, last - (TG - 1));
    const int place = first + int(t);
    const bool seen = place <= last && place >= S - count && (WINDOW == 0 || last - place < WINDOW);
    float sc[G], m[G], z[G];
    for (int g = 0; g < G; g++) sc[g] = -INFINITY;
    if (seen) {
        const device T* ks = kp + size_t((place + idx) % S) * keys_strides[2];
        for (int g = 0; g < G; g++) sc[g] = 0.0f;
        for (int d = 0; d < HD; d++) {
            const float kd = float(ks[d]);
            for (int g = 0; g < G; g++) sc[g] += float(q[qb + g * HD + d]) * kd;
        }
        const int dist = last - place;
        for (int g = 0; g < G; g++)
            sc[g] = sc[g] * scale + (dist < R ? float(bias[(size_t(row) * HQ + kvh * G + g) * R + dist]) : 0.0f);
    }
    for (int g = 0; g < G; g++) { const float v = simd_max(sc[g]); if (lane == 0) buf[g][sg] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) for (int g = 0; g < G; g++) {
        const float v = simd_max(lane < NS ? buf[g][lane] : -INFINITY);
        if (lane == 0) red[g] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = 0; g < G; g++) {
        m[g] = max(red[g], float(sinks[size_t(row) * HQ + kvh * G + g]));
        const float e = seen ? metal::precise::exp(sc[g] - m[g]) : 0.0f;
        p[g][t] = e;
        z[g] = simd_sum(e);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = 0; g < G; g++) if (lane == 0) buf[g][sg] = z[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) for (int g = 0; g < G; g++) {
        const float v = simd_sum(lane < NS ? buf[g][lane] : 0.0f);
        if (lane == 0) red[g] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // (the sink takes weight and has no value)
    for (int g = 0; g < G; g++) z[g] = red[g] + metal::precise::exp(float(sinks[size_t(row) * HQ + kvh * G + g]) - m[g]);

    const int d = t % HD, ch = t / HD;
    float o[G];
    for (int g = 0; g < G; g++) o[g] = 0.0f;
    for (int i = ch * CK; i < min(last - first + 1, (ch + 1) * CK); i++) {
        const float vd = float(vp[size_t((first + i + idx) % S) * values_strides[2] + d]);
        for (int g = 0; g < G; g++) o[g] += p[g][i] * vd;
    }
    for (int g = 0; g < G; g++) part[g][t] = o[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ch == 0) for (int g = 0; g < G; g++) {
        float v = o[g];
        for (int c = 1; c < CH; c++) v += part[g][c * HD + d];
        v = float(T(v / z[g]));
        const size_t at = (kvh * G + g) * HD + d;
        out[size_t(row) * HQ * HD + at] = T(v * sigmoid_(float(gate[size_t(row) * GW + GOFF + at])));
    }
"""

_ROUTE_SRC = """
    constexpr int CHUNK = (D + PARTS - 1) / PARTS;
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const int e = tid / PARTS, part = tid % PARTS;
    threadgroup float partial[E * PARTS];
    threadgroup float score[E];
    threadgroup float chosen[K];
    float dot = 0.0f;  // an expert's score, PARTS threads to it
    for (int d = part * CHUNK; d < min(D, (part + 1) * CHUNK); d++) dot += float(x[size_t(row) * D + d]) * w[size_t(e) * D + d];
    partial[tid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0) {
        for (int i = 1; i < PARTS; i++) dot += partial[e * PARTS + i];
        score[e] = sigmoid_(dot);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0) {  // its place among score + bias (the lower expert first at a tie); the K first are chosen
        const float mine = score[e] + bias[e];
        int rank = 0;
        for (int i = 0; i < E; i++) {
            const float other = score[i] + bias[i];
            rank += (other > mine || (other == mine && i < e)) ? 1 : 0;
        }
        if (rank < K) { inds[size_t(row) * K + rank] = uint(e); chosen[rank] = score[e]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < K) {  // weights without the bias
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += chosen[k];
        weights[size_t(row) * K + tid] = chosen[tid] / (sum + 1e-20f) * prm[0];
    }
"""

_ACT_SRC = """
    const uint i = thread_position_in_grid.x;
    const float g = float(gu[size_t(i) * 2]), u = float(gu[size_t(i) * 2 + 1]);
    const float gc = prm[0], uc = prm[1];
    out[i] = T(metal::precise::tanh(g / gc) * gc * sigmoid_(g) * metal::precise::tanh(u / uc) * uc);
"""

_COMBINE_SRC = """
    const uint d = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    float ysum[4];  // the experts' sum of this token, and (rows one text's tokens) of the three before it
    for (int j = 0; j < 4; j++) {
        const int r = int(row) - 3 + j;
        ysum[j] = 0.0f;
        if (r < 0 || (j < 3 && !SEQ)) continue;
        float y = 0.0f;
        for (int k = 0; k < K; k++) y += float(ys[(size_t(r) * K + k) * D + d]) * weights[size_t(r) * K + k];
        ysum[j] = float(T(y));
    }
    const float y = ysum[3];
    const size_t so = size_t(row) * 3 * D + d, to = size_t(d) * 4;
    float s0, s1, s2;
    if (SEQ) {
        const int r = int(row);
        s0 = r >= 3 ? ysum[0] : state[size_t(r) * D + d];
        s1 = r >= 2 ? ysum[1] : state[size_t(r + 1) * D + d];
        s2 = r >= 1 ? ysum[2] : state[size_t(r + 2) * D + d];
    } else {
        s0 = state[so]; s1 = state[so + D]; s2 = state[so + 2 * D];
    }
    float h = (taps[to] * s0 + taps[to + 1] * s1 + taps[to + 2] * s2 + taps[to + 3] * y + y) * 0.4472135955f;
    h = float(T(h));
    new_state[so] = s1; new_state[so + D] = s2; new_state[so + 2 * D] = y;
    out[size_t(row) * D + d] = T(resid[size_t(row) * D + d] + h);
"""

# combine for a build with an adapter: the experts' sum without its part along a direction (ink.without), which is a
# sum over all of D, so a threadgroup of 1,024 takes a token and a thread the dimensions 1,024 apart. As a kernel of
# its own before combine it was one more link in the step's chain, 1.5 ms a step over the 48 visits.
_COMBINE_WITHOUT_SRC = """
    constexpr int NS = TG / 32;
    constexpr int P = (D + TG - 1) / TG;
    const uint t = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float buf[4][NS];
    threadgroup float red[4];
    float ysum[4][P];
    float part[4], along[4];
    for (int j = 0; j < 4; j++) {
        const int r = int(row) - 3 + j;
        part[j] = 0.0f;
        for (int c = 0; c < P; c++) {
            const int d = int(t) + c * TG;
            ysum[j][c] = 0.0f;
            if (d >= D || r < 0 || (j < 3 && !SEQ)) continue;
            float y = 0.0f;
            for (int k = 0; k < K; k++) y += float(ys[(size_t(r) * K + k) * D + d]) * weights[size_t(r) * K + k];
            ysum[j][c] = y;
            part[j] += y * dir[d];
        }
    }
    REDUCE2(4, part, along)
    for (int c = 0; c < P; c++) {
        const int d = int(t) + c * TG;
        if (d >= D) continue;
        float yv[4];
        for (int j = 0; j < 4; j++) yv[j] = float(T(ysum[j][c] - along[j] * dir[d]));
        const float y = yv[3];
        const size_t so = size_t(row) * 3 * D + d, to = size_t(d) * 4;
        float s0, s1, s2;
        if (SEQ) {
            const int r = int(row);
            s0 = r >= 3 ? yv[0] : state[size_t(r) * D + d];
            s1 = r >= 2 ? yv[1] : state[size_t(r + 1) * D + d];
            s2 = r >= 1 ? yv[2] : state[size_t(r + 2) * D + d];
        } else {
            s0 = state[so]; s1 = state[so + D]; s2 = state[so + 2 * D];
        }
        float h = (taps[to] * s0 + taps[to + 1] * s1 + taps[to + 2] * s2 + taps[to + 3] * y + y) * 0.4472135955f;
        h = float(T(h));
        new_state[so] = s1; new_state[so + D] = s2; new_state[so + 2 * D] = y;
        out[size_t(row) * D + d] = T(resid[size_t(row) * D + d] + h);
    }
"""

_kernels: dict = {}


def _kernel(name: str, inputs: List[str], outputs: List[str], source: str, contiguous: bool = True):
    if name not in _kernels:
        _kernels[name] = mx.fast.metal_kernel(name="locals_inkling_" + name, input_names=inputs, output_names=outputs,
                                              source=source, header=_HEADER, ensure_row_contiguous=contiguous)
    return _kernels[name]


_consts: dict = {}


def _f(*vals: float) -> mx.array:
    if vals not in _consts:
        _consts[vals] = mx.array(vals, dtype=_F32)
    return _consts[vals]


_SMALL: dict = {}


def _small(dtype, vals: tuple) -> mx.array:
    """A few numbers as an array, kept: a step makes the same ones again and again (a window's place, a position's
    scale), and each new array is a trip to the GPU."""
    key = (dtype, vals)
    out = _SMALL.get(key)
    if out is None:
        if len(_SMALL) > 8192:
            _SMALL.clear()
        out = _SMALL[key] = mx.array(vals, dtype=dtype)
    return out


_ZERO = {}


def _zero(dtype) -> mx.array:
    if dtype not in _ZERO:
        _ZERO[dtype] = mx.zeros((1,), dtype=dtype)
    return _ZERO[dtype]


def mix(x: mx.array, blocks: mx.array, queries: mx.array, *, norm_w: Optional[mx.array] = None, eps: float = 1e-6,
        conv=None, messages: Optional[mx.array] = None, gates=None, moff: int = 0, outputs: int = 0, shape=None,
        seq: bool = False):
    """The inputs a visit reads over depth. ``x`` (M, D): the state (after attention, with ``conv``, attention's
    output). ``blocks`` (K, M, D) float32: the block states. ``queries`` (Q, D). Returns (out, partial[, resid,
    new_state]): ``out`` (M, outputs or Q, D), each query's attention over the block states and the normalized
    state ``partial`` (M, D), then:

    * with ``messages`` (M, 4, D) and ``gates`` = (down (n, rank, 2D), forget (n, D, rank), gain (n, D, rank)), the
      loop's gate of output j with message ``moff + j`` (one query's output gated ``outputs`` ways);
    * with ``norm_w`` (1 or n, D), the layer norm of each, in x's dtype; float32 without.

    ``conv`` = (state (M, 3, D), taps (D, 4), residual (M, D) or None): the state is first the short convolution
    of x, plus the residual; ``resid`` (M, D) float32 is that sum and ``new_state`` the convolution's. ``shape``:
    another shape for ``out`` (a kernel's output costs nothing to shape; a reshape after it is one more node of a
    step's chain, 9 us each). ``seq``: the rows are one text's consecutive tokens, so the convolution of a row reads
    the rows before it and ``state`` (1, 3, D) only at the start; ``new_state`` (M, 3, D) is then the state after
    each token."""
    M, D = x.shape
    K, Q = blocks.shape[0], queries.shape[0]
    QO = outputs or Q
    gate = messages is not None
    rank = gates[0].shape[1] if gate else 1
    TG = 1024 if D >= 1024 else 256 if D >= 256 else 32
    assert TG % rank == 0, (TG, rank)
    state, taps, res = conv if conv is not None else (_zero(_F32), _zero(_F32), None)
    down, forget, gain = gates if gate else (_zero(x.dtype), _zero(_F32), _zero(_F32))
    T, OT = x.dtype, x.dtype if norm_w is not None else _F32
    out = _kernel("mix", ["x", "blocks", "queries", "state", "taps", "res", "messages", "down", "forget", "gain", "norm_w", "prm"],
                  ["out", "partial", "resid", "new_state"], _MIX_SRC)(
        inputs=[x, blocks, queries, state, taps, res if res is not None else _zero(T), messages if gate else _zero(_F32),
                down, forget, gain, norm_w if norm_w is not None else _zero(T), _f(eps)],
        template=[("T", T), ("OT", OT), ("GT", down.dtype), ("D", D), ("K", K), ("Q", Q), ("QO", QO),
                  ("QN", norm_w.shape[0] if norm_w is not None else 1), ("TG", TG), ("CONV", conv is not None),
                  ("ADD", res is not None), ("GATE", gate), ("NORM", norm_w is not None), ("MOFF", moff), ("RANK", rank),
                  ("NRED", max(K + 1, 2)), ("SEQ", seq)],
        grid=(QO * TG, M, 1), threadgroup=(TG, 1, 1),
        output_shapes=[shape or (M, QO, D), (M, D), (M, D) if conv is not None else (1,),
                       (M, 3, D) if conv is not None else (1,)],
        output_dtypes=[OT, _F32, _F32, _F32])
    return out if conv is not None else out[:2]


def kv_prep(q, r, k, v, at, xq, xk, xv, sink_w, kdyn_w, vdyn_w, kstate, vstate, ktaps, vtaps, qscale, rscale, proj, tau, *,
            heads: int, kv_heads: int, hd: int, qk_threshold: float, rel_threshold: float, eps: float, seq: bool = False):
    """From a token's projections to what attention takes. ``q``, ``r``, ``k``, ``v``: (M, ...) arrays holding the
    query, relative, key and value projections at the offsets ``at`` (four ints; one array may hold several:
    projections of one input are one matrix). ``xq``, ``xk``, ``xv`` (M, D): the inputs, for the sink logit of each
    head (``sink_w`` (heads, D)) and, in a global layer, the dynamic taps of the key and value convolutions
    (``kdyn_w``, ``vdyn_w`` (kv_heads * 4, D), over learned taps (4, C); else None and plain taps (C, 4)).
    Returns the clipped, scaled queries (M, heads, hd), the bias of every distance (M, heads, R), the new key and
    value (M, kv_heads, 1, hd) after their convolutions, clip and norm, the sink logits (M, heads) and the
    convolutions' new states. ``tau`` (M,) scales queries and bias, or None. ``seq``: the rows are one text's
    consecutive tokens (see mix)."""
    M, D = xq.shape
    dr, R = rscale.shape[-1], proj.shape[-1]
    T = xq.dtype
    C = kv_heads * hd
    TG = max(32, -(-hd // 32) * 32)
    dyn = kdyn_w is not None
    z = _zero(T)
    return _kernel("kv", ["q", "r", "k", "v", "xq", "xk", "xv", "sink_w", "kdyn_w", "vdyn_w", "kstate", "vstate", "ktaps", "vtaps",
                          "qscale", "rscale", "proj", "tau", "prm"],
                   ["qo", "bias", "ko", "vo", "sinks", "kstate_new", "vstate_new"], _KV_SRC)(
        inputs=[q, r, k, v, xq, xk if dyn else z, xv if dyn else z, sink_w, kdyn_w if dyn else z, vdyn_w if dyn else z, kstate,
                vstate, ktaps, vtaps, qscale, rscale, proj, tau if tau is not None else _zero(_F32),
                _f(qk_threshold, rel_threshold, eps)],
        template=[("T", T), ("D", D), ("HQ", heads), ("HKV", kv_heads), ("HD", hd), ("DR", dr), ("R", R), ("TG", TG), ("DYN", dyn),
                  ("TAU", tau is not None), ("QW", q.shape[-1]), ("RW", r.shape[-1]), ("KW", k.shape[-1]), ("VW", v.shape[-1]),
                  ("QOFF", at[0]), ("ROFF", at[1]), ("KOFF", at[2]), ("VOFF", at[3]), ("SEQ", seq)],
        grid=(heads * TG, M, 1), threadgroup=(TG, 1, 1),
        output_shapes=[(M, heads, hd), (M, heads, R), (M, kv_heads, 1, hd), (M, kv_heads, 1, hd), (M, heads), (M, 3, C), (M, 3, C)],
        output_dtypes=[T, T, T, T, T, _F32, _F32])


def window(hd: int) -> int:
    """The most keys attend() takes for heads of ``hd``: a thread a key, in one threadgroup."""
    return min(1024, -(-hd * 8 // 32) * 32)


def attend(q, keys, values, bias, sinks, gate, idx: int, counts, scale: float, gate_at: int = 0, window_size: int = 0,
           seq: bool = False):
    """Attention of each row's one query (M, heads, hd) over its window, ``keys`` and ``values`` (M, kv_heads, S,
    hd) with S <= window(hd), as the cache holds them: ``idx`` is the slot after the newest key (0: the keys are in order) and
    ``counts`` (M,) int32 each row's number of real keys, its newest (a batch's shorter rows are padded). With the bias by distance (M, heads, R), the sink logits (M, heads) and the sigmoid of
    ``gate`` on the output (M, heads * hd, or wider with the gate at ``gate_at``): (M, heads * hd). ``seq``: the
    M queries are one text's consecutive tokens over its keys (1, kv_heads, S, hd), which end with theirs, in
    order (``idx`` 0): each sees the keys up to its own, the last ``window_size`` of them in a sliding layer."""
    M, heads, hd = q.shape
    T = q.dtype
    TG = window(hd)
    return _kernel("attend", ["q", "keys", "values", "bias", "sinks", "gate", "meta", "counts", "prm"], ["out"], _ATTEND_SRC, False)(
        inputs=[q, keys, values, bias, sinks, gate, _small(mx.int32, (idx,)), counts, _f(scale)],
        template=[("T", T), ("HQ", heads), ("HKV", keys.shape[1]), ("HD", hd), ("R", bias.shape[-1]), ("TG", TG), ("GW", gate.shape[-1]),
                  ("GOFF", gate_at), ("WINDOW", window_size), ("SEQ", seq)],
        grid=(keys.shape[1] * TG, M, 1), threadgroup=(TG, 1, 1), output_shapes=[(M, heads * hd)], output_dtypes=[T])[0]


_SOFT_SRC = """
    const uint lane = thread_index_in_simdgroup;  // (a threadgroup is one of them: a query of a head, 32 keys apart)
    const uint id = thread_position_in_grid.y;
    const int S = scores_shape[4], L = scores_shape[3];
    const int l = int(id) % L, h = (int(id) / L) % HQ, b = int(id) / (L * HQ);
    const int qp = pos[2 * b] + l, k0 = pos[2 * b + 1];
    const float scale = prm[0];
    const device float* sc = scores + size_t(id) * S;
    const device T* bs = bias + (size_t(b * L + l) * HQ + h) * R;
    const float sink = float(sinks[size_t(b * L + l) * HQ + h]);
    float m = -INFINITY;
    for (int j = int(lane); j < S; j += 32) {
        const int dist = qp - (k0 + j);
        if (dist >= 0 && k0 + j >= 0 && (WINDOW == 0 || dist < WINDOW))
            m = max(m, sc[j] * scale + (dist < R ? float(bs[dist]) : 0.0f));
    }
    m = max(simd_max(m), sink);
    float z = 0.0f;
    for (int j = int(lane); j < S; j += 32) {
        const int dist = qp - (k0 + j);
        float e = 0.0f;
        if (dist >= 0 && k0 + j >= 0 && (WINDOW == 0 || dist < WINDOW))
            e = metal::precise::exp(sc[j] * scale + (dist < R ? float(bs[dist]) : 0.0f) - m);
        shares[size_t(id) * S + j] = O(e);
        z += e;
    }
    z = simd_sum(z) + metal::precise::exp(sink - m);  // (the sink takes weight and has no value)
    if (lane == 0) sums[id] = z;
"""


def soft(scores, bias, sinks, q0, k0, scale: float, window_size: int, dtype):
    """Attention's weights from its scores, for the reference's passes over several tokens (ink.attend): of scores
    (B, kv_heads, G, L, S) in float32, each query's first at position ``q0`` and the keys consecutive from ``k0``
    (numbers, or a row's each; a key before 0 is padding), with the bias by distance (B, L, heads, R) and the sink
    logits (B, L, heads): each key's share, exp(logit - the row's largest), in ``dtype``, and the rows' sums with
    the sink's share, (B, kv_heads, G, L, 1) in float32. One pass over the scores: MLX's take_along_axis, two
    wheres, a concatenate, a softmax and a slice are six arrays of the scores' size (most of 163 ms of a 256-token
    chunk's 960; 0.64 ms a token)."""
    B, Hkv, G, L, S = scores.shape
    pos = mx.stack([mx.broadcast_to(mx.array(q0), (B,)), mx.broadcast_to(mx.array(k0), (B,))], axis=-1).astype(mx.int32)
    shares, sums = _kernel("soft", ["scores", "bias", "sinks", "pos", "prm"], ["shares", "sums"], _SOFT_SRC)(
        inputs=[scores, bias, sinks, pos, _f(scale)],
        template=[("T", bias.dtype), ("O", dtype), ("HQ", Hkv * G), ("R", bias.shape[-1]), ("WINDOW", window_size)],
        grid=(32, B * Hkv * G * L, 1), threadgroup=(32, 1, 1),
        output_shapes=[scores.shape, (B, Hkv, G, L, 1)], output_dtypes=[dtype, mx.float32])
    return shares, sums


def route(x, weight, bias, k: int, route_scale: float):
    """The router: of each row (M, D) the ``k`` experts with the highest sigmoid score plus bias, uint32, and
    their weights (the scores, normalized to sum to ``route_scale``)."""
    E, D = weight.shape
    M = x.size // D
    parts = max(1, min(8, 1024 // E))
    return _kernel("route", ["x", "w", "bias", "prm"], ["inds", "weights"], _ROUTE_SRC)(
        inputs=[x, weight, bias, _f(route_scale)], template=[("T", x.dtype), ("D", D), ("E", E), ("K", k), ("PARTS", parts)],
        grid=(M * E * parts, 1, 1), threadgroup=(E * parts, 1, 1), output_shapes=[(M, k), (M, k)],
        output_dtypes=[mx.uint32, _F32])


def act(gate_up, gate_cap: float, up_cap: float):
    """The softcapped SwiGLU of interleaved gate and up rows (..., 2 I): (..., I)."""
    shape = (*gate_up.shape[:-1], gate_up.shape[-1] // 2)
    n = gate_up.size // 2
    return _kernel("act", ["gu", "prm"], ["out"], _ACT_SRC)(
        inputs=[gate_up, _f(gate_cap, up_cap)], template=[("T", gate_up.dtype)], grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1), output_shapes=[shape], output_dtypes=[gate_up.dtype])[0]


def combine(ys, weights, state, taps, resid, seq: bool = False, direction=None):
    """The experts' outputs (M, k, ..., D) summed by their weights, the short convolution of that (state (M, 3, D), taps
    (D, 4)) and the residual (M, D) float32: the visit's output (M, D) and the convolution's new state.
    ``direction`` (D,) float32, a build with an adapter: the sum without its part along it."""
    M, D = resid.shape
    K = ys.size // (M * D)
    T = ys.dtype
    if direction is not None:
        return _kernel("combine_without", ["ys", "weights", "state", "taps", "resid", "dir"], ["out", "new_state"], _COMBINE_WITHOUT_SRC)(
            inputs=[ys, weights, state, taps, resid, direction], template=[("T", T), ("K", K), ("D", D), ("SEQ", seq), ("TG", 1024)],
            grid=(1024, M, 1), threadgroup=(1024, 1, 1), output_shapes=[(M, D), (M, 3, D)], output_dtypes=[T, _F32])
    return _kernel("combine", ["ys", "weights", "state", "taps", "resid"], ["out", "new_state"], _COMBINE_SRC)(
        inputs=[ys, weights, state, taps, resid], template=[("T", T), ("K", K), ("D", D), ("SEQ", seq)], grid=(D, M, 1),
        threadgroup=(min(256, D), 1, 1), output_shapes=[(M, D), (M, 3, D)], output_dtypes=[T, _F32])


ONCE_MAX = 32  # rows whose experts are read once (inkling_experts.xmm: 16 rows 77 ms against MLX's 115; 64 rows 280 against 269)


def _read_once(E) -> bool:
    """Whether the experts are of the kind ``inkling_experts.xmm`` reads: NVFP4 with a scale an expert, the release's
    (and a build's for every layer; MXFP8 ones, and experts not quantized, stay MLX's). ``LOCALS_INKLING_XMM=0``:
    MLX's gather for all."""
    ok = getattr(E, "_ink_read_once", None)
    if ok is None:
        ok = os.environ.get("LOCALS_INKLING_XMM", "1") != "0"
        for lin in (E.gate_up_proj, E.down_proj):
            ok = ok and (getattr(lin, "mode", None) == "nvfp4" and lin.bits == 4 and lin.group_size == 16 and "global_scale" in lin
                         and lin.weight.shape[1] % 8 == 0 and lin.weight.shape[2] % 2 == 0 and "bias" not in lin)
        object.__setattr__(E, "_ink_read_once", ok)
    return ok


class _Consts:
    """A layer's arrays in the layouts the kernels read, made once."""

    def __init__(self, layer: ink.DecoderLayer, args: ink.TextArgs):
        A, D = layer.self_attn, args.hidden_size
        f32 = lambda a: a.astype(_F32)
        self.attn_query = f32(layer.attention_input.query).reshape(-1, D) if layer.attention_input is not None else None
        norms = (layer.input_layernorm,) if layer.sliding else (layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm)
        self.attn_norm = mx.stack([n.weight for n in norms])
        self.mlp_query = f32(layer.mlp_input.query).reshape(1, D)
        self.mlp_norm = layer.post_attention_layernorm.weight.reshape(1, D)
        self.attn_taps = f32(layer.post_attention_convolution.conv1d.weight).reshape(D, -1)
        self.mlp_taps = f32(layer.post_mlp_convolution.conv1d.weight).reshape(D, -1)
        if layer.sliding:
            self.k_taps = f32(A.k_sconv.conv1d.weight).reshape(-1, args.sconv_kernel_size)
            self.v_taps = f32(A.v_sconv.conv1d.weight).reshape(-1, args.sconv_kernel_size)
        else:
            self.k_taps, self.v_taps = A.k_sconv.static_weight, A.v_sconv.static_weight
        self.attn_gates = self.mlp_gates = None
        if hasattr(layer, "q_message_gate"):
            def stack(gates):
                return (mx.stack([g.down_proj.weight for g in gates]), mx.stack([f32(g.forget_gate_proj.weight) for g in gates]),
                        mx.stack([f32(g.message_gate_proj.weight) for g in gates]))
            self.attn_gates = stack((layer.q_message_gate, layer.k_message_gate, layer.v_message_gate))
            self.mlp_gates = stack((layer.mlp_message_gate,))
        self.router_w, self.router_b = f32(layer.mlp.gate.weight), f32(layer.mlp.gate.e_score_correction_bias)
        self.sink_w = A.sink_proj.weight
        self.kdyn_w = self.vdyn_w = None
        if not layer.sliding:
            self.kdyn_w, self.vdyn_w = A.k_sconv.weight_proj.weight, A.v_sconv.weight_proj.weight
        # the projections of the queries' input (queries, relative, gate), then keys and values, as one quantized
        # matrix: one read where there were five (a sliding layer's keys and values have the queries' input, but
        # for the loop's later passes). Each layer's own arrays become rows of it, so nothing is held twice
        self.stack = self.head = None
        mods = [A.q_proj, A.r_proj, A.gate_proj, A.k_proj, A.v_proj]
        m0 = mods[0]
        if all(type(m).__name__.endswith("QuantizedLinear") and getattr(m, "mode", "affine") == "affine" and "bias" not in m
               and (m.bits, m.group_size) == (m0.bits, m0.group_size) for m in mods):
            parts = [mx.concatenate([m[n] for m in mods]) for n in ("weight", "scales", "biases")]
            mx.eval(parts)
            start = 0
            self.at = []
            for m in mods:
                rows = m.weight.shape[0]
                self.at.append(start)
                m.weight, m.scales, m.biases = (p[start : start + rows] for p in parts)
                start += rows
            n_head = self.at[3]
            self.stack = (*parts, m0.group_size, m0.bits)
            self.head = (*(p[:n_head] for p in parts), m0.group_size, m0.bits)
            mx.eval(self.head[:3], [m.parameters() for m in mods])
        mx.eval([v for v in vars(self).values() if isinstance(v, mx.array)])


class FusedText(ink.TextModel):
    """Inkling's decoder with one-token steps in kernels; anything else as the reference."""

    def _consts(self) -> List[_Consts]:
        if getattr(self, "_ink_consts", None) is None:
            object.__setattr__(self, "_ink_consts", [_Consts(layer, self.args) for layer in self.layers])
        return self._ink_consts

    def _steps(self, h: mx.array, cache) -> int:
        """How a pass goes: 0 the reference, 1 a token a row in kernels, 2 one text's few tokens in kernels."""
        B, L = h.shape[:2]
        if cache is None or not mx.metal.is_available() or h.dtype not in (mx.bfloat16, mx.float16, mx.float32):
            return 0
        if L != 1 and (B != 1 or L > SEQ_MAX):
            return 0
        own = 2 * len(self.visits)  # (the draft model's caches come after)
        for kv, conv in zip(cache[0:own:2], cache[1:own:2]):
            if not isinstance(kv, _KV) or not isinstance(conv, ArraysCache) or conv.lengths is not None:
                return 0
        first = cache[1]
        if first.left_padding is not None and not getattr(first, "_ink_live", False):
            # a batch's rows are padded on the left while their prompts are read: those passes are the reference's.
            # Padding only shrinks, so this is looked at once a cache
            if int(mx.max(first.left_padding)) > 0:
                return 0
            first._ink_live = True
        return 1 if L == 1 else 2

    def __call__(self, inputs: mx.array, cache: Optional[List[Any]] = None,
                 input_embeddings: Optional[mx.array] = None) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed(inputs)
        mode = self._steps(h, cache)
        if not mode:
            return super().__call__(inputs, cache, h)
        consts, core, args = self._consts(), self.args.loop_core, self.args
        caches = iter(cache)
        seq = mode == 2
        x = h[0] if seq else h[:, 0]  # (rows: a text's tokens, or a token of each text)

        taps = getattr(self, "taps", None)  # (the draft model reads what enters some visits)
        got, visit = {}, [0]

        def run(first, layers, x, blocks, bank=None):
            for i, layer in enumerate(layers):
                messages = None if bank is None else mix(bank[1], bank[0], self.message_input.query[i])[0]
                if taps is not None and visit[0] in taps.visits:
                    got[visit[0]] = x
                visit[0] += 1
                x, blocks = self._visit(layer, consts[first + i], x, blocks, next(caches), next(caches), messages, seq)
            return x, blocks

        start = run(0, self.layers[: core.start], x, None)
        x, blocks = start
        for loop_pass in range(args.loop_passes if len(core) else 0):
            bank = None
            if loop_pass > 0:
                bank = (blocks[(0 if start[1] is None else start[1].shape[0]) + 1 :], x)
                x, blocks = start
            x, blocks = run(core.start, self.layers[core.start : core.stop], x, blocks, bank)
        x, blocks = run(core.stop, self.layers[core.stop :], x, blocks)
        out = mix(x, blocks, self.final_input.query.reshape(1, -1), norm_w=self.norm.weight.reshape(1, -1), eps=self.norm.eps,
                  shape=h.shape)[0]
        if taps is not None:
            taps.leave([got[v].reshape(h.shape).astype(h.dtype) for v in taps.visits[:-1]] + [out], cache, 2 * visit[0])
        return out

    def _visit(self, layer, c: _Consts, x, blocks, kv, conv, messages, seq: bool = False):
        """One visit of the rows ``x`` (M, D), a token of each text or (``seq``) one text's consecutive tokens: the
        output (M, D) and the block states after."""
        args, A = self.args, layer.self_attn
        M, D = x.shape
        C = A.k_proj.weight.shape[0]
        st = [s if s is not None else mx.zeros((1 if seq else M, 3, n), _F32) for s, n in zip(conv.cache, (C, C, D, D))]
        if layer.attention_input is None:  # layer 0 reads the embeddings as they are
            normed = ink.rms(x.astype(_F32))
            xq = xk = xv = layer.input_layernorm(x)
        else:
            gated = messages is not None
            out, normed = mix(x, blocks, c.attn_query, norm_w=c.attn_norm, eps=args.rms_norm_eps, messages=messages,
                              gates=c.attn_gates if gated else None, outputs=3 if gated else 0,
                              shape=(M, D) if layer.sliding and not gated else None)
            xq, xk, xv = (out,) * 3 if out.ndim == 2 else (out[:, 0], out[:, 1], out[:, 2])
        tau = None
        batch = isinstance(kv.offset, mx.array)
        if not layer.sliding:  # queries sharpen with the log of the position
            if batch:  # each row at its own position; a text's tokens one after another
                pos = (kv.offset + 1).astype(_F32) if not seq else kv.offset[0].astype(_F32) + mx.arange(1, M + 1)
                tau = 1.0 + A.log_alpha * mx.log(mx.maximum(pos / A.log_floor, 1.0))
            else:
                at = [kv.offset + 1 + i for i in range(M)] if seq else [kv.offset + 1] * M
                tau = _small(_F32, tuple(1.0 + A.log_alpha * max(0.0, _log(p / A.log_floor)) for p in at))
        if c.stack is not None and xk is xq and xv is xq:  # one input: every projection in one read
            w, sc, bi, gs, bits = c.stack
            q = r = k = v = gate = mx.quantized_matmul(xq, w, scales=sc, biases=bi, transpose=True, group_size=gs, bits=bits)
            at, gate_at = (c.at[0], c.at[1], c.at[3], c.at[4]), c.at[2]
        elif c.stack is not None:
            w, sc, bi, gs, bits = c.head
            q = r = gate = mx.quantized_matmul(xq, w, scales=sc, biases=bi, transpose=True, group_size=gs, bits=bits)
            k, v, at, gate_at = A.k_proj(xk), A.v_proj(xv), (c.at[0], c.at[1], 0, 0), c.at[2]
        else:
            q, r, k, v, gate, at, gate_at = A.q_proj(xq), A.r_proj(xq), A.k_proj(xk), A.v_proj(xv), A.gate_proj(xq), (0,) * 4, 0
        qo, bias, ko, vo, sinks, ks, vs = kv_prep(
            q, r, k, v, at, xq, xk, xv, c.sink_w, c.kdyn_w, c.vdyn_w, st[0], st[1], c.k_taps, c.v_taps, A.q_l2_clip.scale,
            A.rel_l2_clip.scale, A.rel_logits_proj.proj, tau, heads=A.n_heads, kv_heads=A.n_kv_heads, hd=ko_hd(A),
            qk_threshold=A.qk_threshold, rel_threshold=A.rel_l2_clip.threshold, eps=A.eps, seq=seq)
        if seq:  # (M, kv heads, 1, hd) to the cache's (1, kv heads, M, hd)
            ko, vo = ko.transpose(2, 1, 0, 3), vo.transpose(2, 1, 0, 3)
        keys, values = kv.update_and_fetch(ko, vo)
        S = keys.shape[2]
        if S <= window(qo.shape[-1]) or (seq and layer.sliding):
            # (a pass over several tokens leaves a window in order, with the keys the pass pushes out still there)
            idx = kv._idx if not seq and isinstance(kv, (RotatingKVCache, BatchRotatingKVCache)) else 0
            counts = mx.minimum(kv.offset, S) if batch else _small(mx.int32, (S,) * (1 if seq else M))
            a = attend(qo, keys, values, bias, sinks, gate, idx, counts, A.scale, gate_at, A.window, seq)
        else:  # a global layer past the kernel's window: MLX's attention
            g = gate[:, gate_at : gate_at + qo.shape[1] * qo.shape[2]]
            if seq:
                a = A._attend(qo.transpose(1, 0, 2)[None], keys, values, bias[None], sinks[None], kv)
            else:
                a = A._attend(qo[:, :, None, :], keys, values, bias[:, None], sinks[:, None], kv)
            a = (a.reshape(M, -1).astype(_F32) * mx.sigmoid(g.astype(_F32))).astype(g.dtype)
        a = A.o_proj(a)

        boundary = layer.is_block_boundary
        if boundary:
            blocks = normed[None] if blocks is None else mx.concatenate([blocks, normed[None]])
        gated = messages is not None
        mlp_in, _, resid, cs = mix(a, blocks, c.mlp_query, norm_w=c.mlp_norm, eps=args.rms_norm_eps,
                                   conv=(st[2], c.attn_taps, None if boundary else x), messages=messages,
                                   gates=c.mlp_gates if gated else None, moff=3, shape=(M, 1, 1, D), seq=seq)
        E = layer.mlp.experts
        inds, weights = route(mlp_in, c.router_w, c.router_b, layer.mlp.gate.top_k, layer.mlp.gate.route_scale)
        if _read_once(E) and M <= ONCE_MAX:
            # each expert read once for the rows routed to it (locals/inkling_experts.py): MLX's gather reads it
            # again for every row, 8 ms a row over the 48 visits whatever the rows share
            table = once.slots(inds, E.gate_up_proj.weight.shape[0], 1 if M == 1 else 2)
            hidden = once.xmm(E.gate_up_proj, mlp_in.reshape(M, D), table, inds.size, inds.shape[-1], False, (E.gate_softcap, E.up_softcap))
            ys = once.xmm(E.down_proj, hidden, table, inds.size, inds.shape[-1], True)
        elif inds.size >= 64:  # by expert, as mlx-lm: MLX then reads an expert once for its tokens
            xs, by, back = _gather_sort(mlp_in, inds)
            hidden = act(E.gate_up_proj(xs, by, sorted_indices=True), E.gate_softcap, E.up_softcap)
            ys = _scatter_unsort(E.down_proj(hidden, by, sorted_indices=True), back, inds.shape)
        else:
            ys = E.down_proj(act(E.gate_up_proj(mlp_in, inds), E.gate_softcap, E.up_softcap), inds)
        # (a build with an adapter: the experts' sum without its layer's direction, ``mlp.ablate``)
        x, ms = combine(ys, weights, st[3], c.mlp_taps, resid, seq, getattr(layer.mlp, "ablate", None))
        states = (ks, vs, cs, ms)
        for i, a in enumerate(states):  # (of a text's tokens: the state after each; the last is the cache's)
            conv[i] = a[M - 1 :] if seq else a
        conv.advance(M if seq else 1)
        # a pass whose last tokens may be taken back (a draft: rewind) keeps the state after each
        conv.ink_states = states if seq and getattr(conv, "want_states", False) else None
        conv.ink_windows = None
        return x, blocks


def ko_hd(A) -> int:
    return A.q_l2_clip.scale.shape[-1]


def _log(v: float) -> float:
    import math

    return math.log(v) if v > 0 else float("-inf")


PREFILL_CHUNK = 512  # tokens of a prompt read in one pass (see fuse)


def fuse(model) -> int:
    """Give an Inkling model its one-token steps. Returns the number of layers it covers."""
    text = getattr(getattr(model, "model", None), "language_model", None)
    if not isinstance(text, ink.TextModel) or isinstance(text, FusedText):
        return 0
    text.__class__ = FusedText
    for layer in text.layers:  # (the reference's passes over a prompt's chunks: attention's weights in one kernel)
        layer.self_attn.soft = soft
    # A prompt in chunks of 512: 322 tokens/s on an M4 Pro once the window is full, against 246 in chunks of 128 and
    # 294 at 256 (a chunk reads every expert of a layer once, 180 ms of a chunk whatever its length, and MLX's expert
    # kernel fills its tiles from about 5 tokens an expert), 333 at 768 and 338 at 1,024, whose passes hold 1.5 and
    # 1.8 GB where 512 hold 1.0. Attention keeps up with the chunk: a sliding layer scores a block of queries
    # against the keys it sees (ink.QUERY_BLOCK) and the weights come from one kernel (soft); before, chunks of 256
    # read 272 tokens/s and longer ones no more. The chunk's attention is bounded by the model (MAX_SCORES), so it
    # does not halve as the context grows.
    object.__setattr__(model, "prefill_chunk_tokens", PREFILL_CHUNK)
    object.__setattr__(model, "prefill_scores", None)
    # A pass over a prompt is evaluated a visit at a time (ink.EVAL_TOKENS), so what it holds goes by its tokens:
    # 0.24 GB for 128, 0.50 for 256, 0.99 for 512. The engine keeps room for a read of its size, not for the
    # largest read it has seen (hybrid_batch._read_bytes): an agent's step reads a tool's result, a few dozen tokens.
    object.__setattr__(model, "read_room_by_size", True)
    # Where the weights leave the engine under 5 GB (the 15.6 GB build on a 24 GB Mac), a prompt cache of 1.25 GB:
    # a conversation's state is 0.25 GB at 1k tokens and 0.48 GB at 7k (40 windows and the rest), so that is the
    # prefix a request went on from, its last reply and the system prompt (service.BatchService).
    object.__setattr__(model, "prompt_cache_bytes", int(1.25 * (1 << 30)))
    # What does not fit there goes to the SSD (the engine's disk cache, which kept system prompts alone): a
    # conversation's prefix every 2,048 tokens read, 7 s of reading for about half a second of writing.
    object.__setattr__(model, "disk_checkpoint_tokens", 2048)
    # A pass over n tokens of one text takes about 25 + 6 n ms where a step takes 28 (attention's weights and the
    # output layer are read once, and so is an expert, for the tokens that share it: 12 n with MLX's gather): a
    # drafted token costs a quarter of a step, and a draft may be as long as a pass in kernels.
    object.__setattr__(model, "draft_token_cost", 0.25)
    object.__setattr__(model, "draft_max", SEQ_MAX - 1)
    # The server keeps a prompt's prefixes to start later requests from (the system part, the conversation before
    # the last message, where the reply starts) only from 1,024 tokens: shorter ones are cheaper to read again on
    # the models it was set for, at 750 tokens/s and more. At 270 a 1,000-token prompt is 4 s: `locals run`'s agents
    # read theirs (900 to 1,100 tokens) again on every step. From 192 tokens here.
    object.__setattr__(model, "checkpoint_tokens", 192)
    # A grammar's token tables are 7 bytes a state a token: 1.3 GB on the host and 0.4 on the GPU for a set of tools
    # over these 201,024 tokens, 3 s to make, next to 15.6 GB of weights on a 24 GB Mac. The walk kernel takes the
    # mask from the automaton each step instead (locals/walk.py; forced spans still go in one pass).
    object.__setattr__(model, "walk_constraints", True)
    if "mtp" in model:  # the build's draft model (locals/inkling_draft.py); LOCALS_INKLING_DRAFT=0: as if it had none
        import os

        if os.environ.get("LOCALS_INKLING_DRAFT", "1") == "0":
            text.taps = None
            object.__setattr__(model, "draft_off", True)
        else:
            model.mtp.bind(model)
            # (a draft costs a quarter of a step; its tokens are tried from this probability: hybrid_batch._model_draft_seen)
            object.__setattr__(model, "draft_model_cost", float(os.environ.get("LOCALS_INKLING_DRAFT_COST", "0.25")))
            object.__setattr__(model, "draft_min_p", float(os.environ.get("LOCALS_INKLING_DRAFT_P", "0.8")))
    return len(text.layers)


def drafts_ok(cache) -> bool:
    """Whether a pass may feed tokens that are taken back after it (rewind): a batch of one row, nothing padded."""
    for c in cache:
        if isinstance(c, ArraysCache):
            if c.lengths is not None:
                return False
        elif not isinstance(c, (BatchKVCache, BatchRotatingKVCache)):
            return False
    return True


def rewind(cache, drop: int) -> None:
    """Take back the last ``drop`` tokens of the pass just made over several tokens (a draft whose end the model
    did not confirm). A full layer's keys are cut off; a window's too, which that pass left in order with the keys
    it would otherwise have pushed out (mlx-lm keeps a window plus the pass); each convolution goes back to the
    inputs before the first token dropped, from the pass's own (``ink_windows``, which the reference keeps for a
    pass it is told may be rewound)."""
    for c in cache:
        if isinstance(c, ArraysCache):
            states, windows = getattr(c, "ink_states", None), getattr(c, "ink_windows", None)
            if states is not None:  # (a pass in kernels: the state after each of its tokens)
                for i, a in enumerate(states):
                    keep = a.shape[0] - drop
                    c[i] = a[keep - 1 : keep] + 0
            elif windows is not None:
                for i, w in enumerate(windows):
                    keep = w.shape[1] - 3 - drop  # (tokens of the pass that stay)
                    c[i] = w[:, keep : keep + 3] + 0
            else:
                raise RuntimeError("a pass kept no convolution inputs to rewind to")
            c.ink_states = c.ink_windows = None
        feats = getattr(c, "ink_feats", None)
        if feats is not None:  # (the draft model's first cache: the pass's states, to the last token kept)
            c.ink_feats = feats[:, : feats.shape[1] - drop]
        if isinstance(c, ArraysCache):
            pass
        elif isinstance(c, BatchRotatingKVCache):
            n = c.keys.shape[2] - drop
            c.keys, c.values = c.keys[..., :n, :], c.values[..., :n, :]
            c._idx, c._offset, c.offset = n, c._offset - drop, c.offset - drop
        else:
            c.trim(drop)
