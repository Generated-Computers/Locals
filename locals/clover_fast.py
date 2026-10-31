"""Locals's fast paths for DiffuSpace's Clover (locals/models/clover_moe.py), put in place by fuse.py.

- The MoE of a block (a preview, or a commit of up to 32 rows): the routing in one kernel (the top 8, the
  block coreset, the renormalized weights, the slots in expert order and their tiles) and the routed experts
  in moe_tiles.py's kernels, gate and up as one projection (the shared expert's too). Longer passes (a prompt)
  keep the module's routing and SwitchGLU's sorted gather.
- A preview's GDN convolution (causal over the history and the block, right over the block's next
  positions, SiLU) in one kernel.

The same tokens chosen and weights as the module's computation, up to the order of float sums (tested).
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .moe import expert_order
from .moe_tiles import TILE, TILED_MAX_ROWS, _kernel, experts_down, experts_up, tile_lengths


# The routing for a block of up to 32 rows in one threadgroup, a simdgroup a row (lane i holds experts i, i + 32,
# ...): the top K logits by K rounds of a simdgroup argmax (ties to the lower index, as a stable sort); with CORESET,
# the block's mass per expert (each row's top-K softmax scores, summed in slot order), the experts by mass (ties to the
# lower index) and their inclusive running sum, the experts before which less than top_p was summed allowed, and the
# top K again among them; the weights the softmax of the K chosen logits (the top K of the softmax, renormalized). Then
# the slots in expert order (within an expert, by slot) and each position's tile length, as tile_lengths gives them.
_ROUTE_SRC = """
    constexpr int PER = E / 32, S = NR * K;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    threadgroup uint pick_e[S];
    threadgroup float pick_p[S];
    threadgroup float mass[E], sorted_p[E], incl[E];
    threadgroup uint rank_of[E], allowed[E], cnt[E], start[E];
    threadgroup float total[1];
    float lg[PER];
    uint taken = 0;
    if (sg < NR) {
        for (int i = 0; i < PER; i++) lg[i] = float(logits[sg * E + lane + 32 * i]);
        float vmax = 0.0f, denom = 0.0f;
        for (int k = 0; k < K; k++) {
            float bv = -INFINITY;
            uint bi = E;
            for (int i = 0; i < PER; i++) {
                const uint e = lane + 32 * i;
                if (!(taken & (1u << i)) && (lg[i] > bv || (lg[i] == bv && e < bi))) { bv = lg[i]; bi = e; }
            }
            const float mv = simd_max(bv);
            const uint mi = simd_min(bv == mv ? bi : uint(E));
            if (mi % 32 == lane) taken |= 1u << (mi / 32);
            if (k == 0) {
                vmax = mv;
                float s = 0.0f;
                for (int i = 0; i < PER; i++) s += metal::exp(lg[i] - vmax);
                denom = simd_sum(s);
            }
            if (lane == 0) {
                pick_e[sg * K + k] = mi;
                pick_p[sg * K + k] = CORESET ? metal::exp(mv - vmax) / denom : mv;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (CORESET) {
        if (tid < E) {
            float m = 0.0f;
            for (int j = 0; j < S; j++) m += pick_e[j] == tid ? pick_p[j] : 0.0f;
            mass[tid] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float t = 0.0f;
            for (int e = 0; e < E; e++) t += mass[e];
            total[0] = max(t, 1e-9f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < E) {
            const float pe = mass[tid] / total[0];
            uint r = 0;
            for (int e = 0; e < E; e++) {
                const float q = mass[e] / total[0];
                r += (q > pe || (q == pe && uint(e) < tid));
            }
            rank_of[tid] = r;
            sorted_p[r] = pe;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float c = 0.0f;
            for (int r = 0; r < E; r++) { c += sorted_p[r]; incl[r] = c; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < E) {
            const uint r = rank_of[tid];
            allowed[tid] = r == 0 || (incl[r] - sorted_p[r]) < top_p[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < NR) {
            for (int i = 0; i < PER; i++) if (!allowed[lane + 32 * i]) lg[i] = -INFINITY;
            taken = 0;
            for (int k = 0; k < K; k++) {
                float bv = -INFINITY;
                uint bi = E;
                for (int i = 0; i < PER; i++) {
                    const uint e = lane + 32 * i;
                    if (!(taken & (1u << i)) && (lg[i] > bv || (lg[i] == bv && e < bi))) { bv = lg[i]; bi = e; }
                }
                const float mv = simd_max(bv);
                const uint mi = simd_min(bv == mv ? bi : uint(E));
                if (mi % 32 == lane) taken |= 1u << (mi / 32);
                if (lane == 0) { pick_e[sg * K + k] = mi; pick_p[sg * K + k] = mv; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg < NR && lane == 0) {  // the softmax of the row's K chosen logits
        const float v0 = pick_p[sg * K];
        float z = 0.0f;
        for (int k = 0; k < K; k++) z += metal::exp(pick_p[sg * K + k] - v0);
        for (int k = 0; k < K; k++) {
            inds[sg * K + k] = pick_e[sg * K + k];
            weights[sg * K + k] = metal::exp(pick_p[sg * K + k] - v0) / z;
        }
    }
    if (tid < E) {
        uint c = 0;
        for (int j = 0; j < S; j++) c += pick_e[j] == tid;
        cnt[tid] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint c = 0;
        for (int e = 0; e < E; e++) { start[e] = c; c += cnt[e]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < S) {
        const uint e = pick_e[tid];
        uint before = 0;
        for (uint j = 0; j < tid; j++) before += pick_e[j] == e;
        const uint pos = start[e] + before;
        order[pos] = tid;
        tiles[pos] = before % TILE == 0 ? min(uint(TILE), cnt[e] - before) : 0u;
    }
"""

ROUTE_MAX_ROWS = 32  # one threadgroup, a simdgroup a row


def clover_route(logits: mx.array, k: int, top_p: Optional[float], tile: int):
    """Clover's routing of a block of up to ROUTE_MAX_ROWS rows of router ``logits`` (rows, E), in one kernel: (inds
    (rows, k), weights (rows, k) float32, order, tiles) for experts_up and experts_down. ``top_p``: the block coreset's
    (None: each row's own top k)."""
    R, E = logits.shape
    S = R * k
    kern = _kernel("locals_clover_route", _ROUTE_SRC, ["logits", "top_p"], ["inds", "weights", "order", "tiles"])
    inds, weights, order, tiles = kern(
        inputs=[logits, mx.array([top_p if top_p is not None else 1.0], dtype=mx.float32)],
        template=[("E", E), ("K", k), ("NR", R), ("CORESET", int(top_p is not None)), ("TILE", tile)],
        grid=(1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(S,), (S,), (S,), (S,)],
        output_dtypes=[mx.uint32, mx.float32, mx.uint32, mx.uint32])
    return inds.reshape(R, k), weights.reshape(R, k), order, tiles


def route_fits(rows: int, E: int, k: int) -> bool:
    return rows <= ROUTE_MAX_ROWS and E % 32 == 0 and E // 32 <= 32 and E <= 1024 and rows * k <= 1024 and k <= E


# a thread an output: the K causal taps over [history, block] then the K - 1 right taps within the block, in float32
_CONV_SRC = """
    const uint i = thread_position_in_grid.x;
    const uint c = i % C, l = (i / C) % L, b = i / (C * L);
    if (b >= B) return;
    float acc = 0.0f;
    for (int t = 0; t < KW; t++) {
        const int j = int(l) + t - (KW - 1);  // the block position the tap reads; < 0: the history
        const float v = j < 0 ? float(hist[(size_t(b) * (KW - 1) + (KW - 1 + j)) * C + c]) : float(qkv[(size_t(b) * L + j) * C + c]);
        acc += v * float(w[c * KW + t]);
    }
    for (int d = 1; d < KW; d++) {
        if (l + d < L) acc += float(qkv[(size_t(b) * L + l + d) * C + c]) * float(right[c * (KW - 1) + d - 1]);
    }
    out[i] = static_cast<T>(acc / (1.0f + metal::exp(-acc)));
"""
_conv_kernel = None


def _preview_conv_kernel(history, qkv, w, right):
    global _conv_kernel
    if _conv_kernel is None:
        _conv_kernel = mx.fast.metal_kernel(name="clover_preview_conv", input_names=["hist", "qkv", "w", "right"],
                                            output_names=["out"], source=_CONV_SRC)
    B, L, C = qkv.shape
    KW = w.shape[1]
    (out,) = _conv_kernel(inputs=[history, qkv, w, right], template=[("T", qkv.dtype), ("B", B), ("L", L), ("C", C), ("KW", KW)],
                          grid=(B * L * C, 1, 1), threadgroup=(256, 1, 1), output_shapes=[qkv.shape], output_dtypes=[qkv.dtype])
    return out


class FusedCloverMoe:
    """Clover's SparseMoe (locals/models/clover_moe.py) with gate and up as one projection and, for a block of up to
    TILED_MAX_ROWS rows, the routed experts in the tile kernels; the router and the coreset as the module's."""

    def __call__(self, x: mx.array, preview: bool = False) -> mx.array:
        from .models.clover_moe import block_coreset

        B, L, D = x.shape
        k = self.top_k
        raw = self.gate(x)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        sw = self.switch_mlp
        top_p = self.coreset_top_p if preview else None
        if B == 1 and route_fits(L, raw.shape[-1], k):  # a block: the routing in one kernel
            inds2, w2, order, tiles = clover_route(raw.reshape(L, -1), k, top_p, TILE)
            h = experts_up(x.reshape(-1, D), inds2, sw.gate_up_proj, order, tiles, tile=TILE)
            return experts_down(h, inds2, w2, sw.down_proj, shared.reshape(-1, D), order, tiles, tile=TILE).reshape(B, L, D)
        logits = raw.astype(mx.float32)
        if top_p is not None:
            logits = block_coreset(logits, k, top_p)
        inds = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        scores = mx.softmax(mx.take_along_axis(logits, inds, axis=-1), axis=-1)
        if B * L <= TILED_MAX_ROWS:
            inds2 = inds.reshape(-1, k)
            order = expert_order(inds2)
            tiles = tile_lengths(inds2, order, TILE)
            h = experts_up(x.reshape(-1, D), inds2, sw.gate_up_proj, order, tiles, tile=TILE)
            return experts_down(h, inds2, scores.reshape(-1, k), sw.down_proj, shared.reshape(-1, D), order, tiles,
                                tile=TILE).reshape(B, L, D)
        return (sw(x, inds) * scores[..., None]).sum(axis=-2).astype(x.dtype) + shared


def fuse_clover_moe(block) -> bool:
    """Clover's SparseMoe in place (gate and up fused, the tile kernels); False if it is another kind of block or its
    experts do not fit the kernels."""
    from .moe import experts_fit, fuse_switch_glu

    if type(block).__name__ != "SparseMoe" or type(block).__module__.split(".")[-1] != "clover_moe":
        return False
    sw = block.switch_mlp
    if type(sw).__name__ != "SwitchGLU" or not hasattr(sw.gate_proj, "bits"):
        return False
    probe = mx.zeros((1, sw.gate_proj.input_dims), dtype=sw.gate_proj["scales"].dtype)  # (a row of the block's input)
    if not experts_fit(probe, sw.gate_proj, sw.down_proj) or sw.gate_proj.bits != sw.up_proj.bits or not fuse_switch_glu(sw):
        return False
    from .fuse import _fusable, _fuse_mlp, _swiglu_mlp

    se = block.shared_expert
    if isinstance(getattr(se, "gate_proj", None), nn.QuantizedLinear) and _fusable([se.gate_proj, se.up_proj]) and _swiglu_mlp(se):
        _fuse_mlp(se)  # the shared expert's gate and up one matmul (FusedMLP)
    mx.eval(sw.gate_up_proj.parameters())  # a layer at a time: the separate gate and up freed before the next's copy
    mx.clear_cache()
    block.__class__ = type("Fused" + type(block).__name__, (FusedCloverMoe, type(block)), {})
    return True


@mx.compile
def _preview_memory(q, k, v, a, state, A_log, dt_bias, logit_scale, history_bias, history_c, bucket_log):
    """CloverGatedDeltaNet._preview_memory as one compiled graph (its elementwise chains fused)."""
    B, L, Hk, Dk = q.shape
    r = v.shape[2] // Hk
    qf, kf = mx.repeat(q, r, axis=2).astype(mx.float32), mx.repeat(k, r, axis=2).astype(mx.float32)
    qf = qf * mx.rsqrt(mx.sum(qf * qf, axis=-1, keepdims=True) + 1e-6)
    kf = kf * mx.rsqrt(mx.sum(kf * kf, axis=-1, keepdims=True) + 1e-6)
    qh, kh = qf.transpose(0, 2, 1, 3), kf.transpose(0, 2, 1, 3)
    vh = v.astype(mx.float32).transpose(0, 2, 1, 3)
    g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a.astype(mx.float32) + dt_bias.astype(mx.float32))
    history = ((qh * Dk**-0.5) @ state.astype(mx.float32).swapaxes(-1, -2)) * mx.exp(g).transpose(0, 2, 1)[..., None]
    logits = (qh @ kh.swapaxes(-1, -2)) * logit_scale.astype(mx.float32)[:, None, None]
    bucket = history_bias.astype(mx.float32) + history_c.astype(mx.float32) * bucket_log
    bucket = mx.broadcast_to(bucket[:, None, None], logits.shape[:3] + (1,))
    w = mx.softmax(mx.concatenate([logits, bucket], axis=-1), axis=-1)
    out = w[..., L:] * history + w[..., :L] @ vh
    return out.transpose(0, 2, 1, 3).astype(q.dtype)


class FusedCloverGatedDeltaNet:
    """Clover's GDN layer with a preview's convolution in one kernel and its softmax memory compiled; in_proj_qkv
    and in_proj_z one matmul where they are quantized alike."""

    def _project(self, x: mx.array):
        if not hasattr(self, "in_proj_qkvz"):
            return super()._project(x)
        p = self.in_proj_qkvz(x)
        return p[..., : self.conv_dim], p[..., self.conv_dim :], self.in_proj_b(x), self.in_proj_a(x)

    def _preview_conv(self, history: mx.array, qkv: mx.array) -> mx.array:
        if qkv.dtype not in (mx.bfloat16, mx.float16):
            return super()._preview_conv(history, qkv)
        return _preview_conv_kernel(history, qkv, self.conv1d.weight, self.conv_right_weight)

    def _preview_memory(self, q, k, v, a, state) -> mx.array:
        return _preview_memory(q, k, v, a, state, self.A_log, self.dt_bias, self.bd_softmax_logit_scale,
                               self.bd_softmax_history_bias, self.bd_softmax_history_c,
                               mx.array(math.log(q.shape[1] / self.block_size), dtype=mx.float32))


def fuse_clover_gdn(layer) -> bool:
    from .fuse import _concat_quantized, _fusable

    if type(layer).__name__ != "CloverGatedDeltaNet" or type(layer).__module__.split(".")[-1] != "clover_moe":
        return False
    parts = [layer.in_proj_qkv, layer.in_proj_z]
    if _fusable(parts) and not any("bias" in m for m in parts):
        layer.in_proj_qkvz = _concat_quantized(parts)
        del layer.in_proj_qkv, layer.in_proj_z
    layer.__class__ = type("Fused" + type(layer).__name__, (FusedCloverGatedDeltaNet, type(layer)), {})
    return True


class FusedCloverAttention:
    """Clover's attention with q (and its gate), k and v as one matmul."""

    def _project(self, x: mx.array):
        p = self.qkv_proj(x)
        nq, nk = 2 * self.num_attention_heads * self.head_dim, self.num_key_value_heads * self.head_dim
        return p[..., :nq], p[..., nq : nq + nk], p[..., nq + nk :]


def fuse_clover_attention(attn) -> bool:
    from .fuse import _concat_quantized, _fusable

    if type(attn).__name__ != "CloverAttention" or type(attn).__module__.split(".")[-1] != "clover_moe":
        return False
    parts = [attn.q_proj, attn.k_proj, attn.v_proj]
    if not _fusable(parts) or any("bias" in m for m in parts):
        return False
    attn.qkv_proj = _concat_quantized(parts)
    del attn.q_proj, attn.k_proj, attn.v_proj
    attn.__class__ = type("Fused" + type(attn).__name__, (FusedCloverAttention, type(attn)), {})
    return True
