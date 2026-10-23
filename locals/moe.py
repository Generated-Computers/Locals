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

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu

# One simdgroup per token: lane i holds experts i, i+32, ...; k rounds of a simdgroup argmax pick the experts in
# order of score (ties to the lower index, as a stable sort), then the chosen scores are renormalized to sum to 1.
_ROUTE_SRC = """
    const uint t = thread_position_in_grid.x / 32;
    const uint lane = thread_index_in_simdgroup;
    if (t >= nrows[0]) return;
    constexpr int PER = (E + 31) / 32;
    float s[PER];
    for (int j = 0; j < PER; j++) {
        const int e = lane + 32 * j;
        s[j] = e < E ? 1.0f / (1.0f + metal::precise::exp(-float(logits[t * E + e]))) : -1.0f;
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
        const uint win = simd_min(best == top ? best_e : uint(E));
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


def route_topk(logits: mx.array, k: int):
    """(experts, weights) of each row of router ``logits`` (..., E): the ``k`` largest sigmoid scores, as uint32
    expert indices and float32 weights that sum to 1. ``flint.route`` in one kernel."""
    global _route_kernel
    if _route_kernel is None:
        _route_kernel = mx.fast.metal_kernel(name="locals_route_topk", input_names=["logits", "nrows"],
                                             output_names=["inds", "weights"], source=_ROUTE_SRC)
    E = logits.shape[-1]
    T = logits.size // E
    inds, weights = _route_kernel(
        inputs=[logits, mx.array([T], dtype=mx.uint32)], template=[("E", E), ("K", k)], grid=(T * 32, 1, 1),
        threadgroup=(min(256, T * 32), 1, 1),
        output_shapes=[(*logits.shape[:-1], k)] * 2, output_dtypes=[mx.uint32, mx.float32])
    return inds, weights


# --- the routed experts of a few tokens, in two kernels ------------------------------------------------------------
#
# gather_qmm, the gate/up slices, swiglu, gather_qmm, the weighted sum over the k experts, its cast and the shared
# expert's add were eight dependent kernels a layer, 70 us at one token (the weights they read take 39 us): every
# dependent launch costs about 5 us on an M4 Pro. ``experts_up`` writes swiglu(gate, up) of each (token, expert)
# pair, ``experts_down`` the down projections summed with the routing weights, plus the shared expert. Rounding is
# where the stock graph rounds: each projection to the activation dtype, swiglu's steps, the float32 weighted sum.
# Weights are MLX's affine format with groups of 64: a group is a little-endian bit stream of 2 * BITS 32-bit words.

_QDOT_HEADER = """
// Values 8c..8c+7 of a group as one integer: value i at bits BITS*i (3 bits: four 24-bit chunks in three words).
template <int BITS>
inline uint chunk8(const device uint32_t* wp, int c) {
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
    return uint(v);
}

// The 8 inputs of a chunk divided by 2^(BITS*i), so a value is masked in place, not shifted (as MLX's qmv does);
// a power of two scales a float exactly. BITS * 8 > 32 (5 to 8 bits) does not fit a uint: shifted there.
template <int BITS>
inline float qdot8(uint v, thread const float* xs) {
    constexpr uint mask = (1u << BITS) - 1;
    float s = 0.0f;
    if (BITS * 8 <= 32) {
        for (int i = 0; i < 8; i++) s += float(v & (mask << (BITS * i))) * xs[i];
    } else {
        for (int i = 0; i < 8; i++) s += float((v >> (BITS * i)) & mask) * xs[i];
    }
    return s;
}

template <int BITS>
inline void prescale8(thread float* xs) {
    if (BITS * 8 <= 32) {
        for (int i = 1; i < 8; i++) xs[i] *= 1.0f / float(1u << (BITS * i));
    }
}
"""

# one simdgroup per (token-expert slot, RJ hidden units); lanes take the input's groups of 64
_UP_SRC = """
    constexpr int NG = D / 64, WPG = 2 * BITS;
    constexpr uint ROWW = NG * WPG;
    const uint sgid = thread_position_in_grid.x / 32;  // (a partial last threadgroup is smaller: not its size times its index)
    const uint lane = thread_index_in_simdgroup;
    const uint jb = sgid % (I / RJ), slot = sgid / (I / RJ);
    if (slot >= nslots[0]) return;
    const uint t = slot / K, e = inds[slot];
    float ga[RJ], ua[RJ];
    for (int r = 0; r < RJ; r++) { ga[r] = 0.0f; ua[r] = 0.0f; }
    for (uint g = lane; g < NG; g += 32) {
        const device T* xr = x + size_t(t) * D + g * 64;
        float qg[RJ], qu[RJ], xs = 0.0f;
        for (int r = 0; r < RJ; r++) { qg[r] = 0.0f; qu[r] = 0.0f; }
        for (int c = 0; c < 8; c++) {
            float xv[8];
            for (int i = 0; i < 8; i++) { xv[i] = float(xr[c * 8 + i]); xs += xv[i]; }
            prescale8<BITS>(xv);
            for (int r = 0; r < RJ; r++) {
                const size_t gate_row = size_t(e) * 2 * I + jb * RJ + r;
                qg[r] += qdot8<BITS>(chunk8<BITS>(w + gate_row * ROWW + g * WPG, c), xv);
                qu[r] += qdot8<BITS>(chunk8<BITS>(w + (gate_row + I) * ROWW + g * WPG, c), xv);
            }
        }
        for (int r = 0; r < RJ; r++) {
            const size_t gate_row = size_t(e) * 2 * I + jb * RJ + r;
            ga[r] += float(scales[gate_row * NG + g]) * qg[r] + float(biases[gate_row * NG + g]) * xs;
            ua[r] += float(scales[(gate_row + I) * NG + g]) * qu[r] + float(biases[(gate_row + I) * NG + g]) * xs;
        }
    }
    for (int r = 0; r < RJ; r++) {
        const T gt = T(simd_sum(ga[r])), ut = T(simd_sum(ua[r]));
        if (lane == 0) {
            const T sig = T(1.0f / (1.0f + metal::exp(-float(gt))));
            const T act = T(float(gt) * float(sig));
            h[size_t(slot) * I + jb * RJ + r] = T(float(act) * float(ut));
        }
    }
"""

# one simdgroup per (token, 32 / (I / 64) output rows); a row's I / 64 groups go to as many lanes
_DOWN_SRC = """
    constexpr int NG = I / 64, WPG = 2 * BITS, RPS = 32 / NG;
    constexpr uint ROWW = NG * WPG;
    const uint sgid = thread_position_in_grid.x / 32;  // (a partial last threadgroup is smaller: not its size times its index)
    const uint lane = thread_index_in_simdgroup;
    const uint t = sgid / (D / RPS), row = (sgid % (D / RPS)) * RPS + lane / NG, g = lane % NG;
    if (t >= ntok[0]) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        const uint slot = t * K + k;
        const size_t wrow = size_t(inds[slot]) * D + row;
        const device T* hr = h + size_t(slot) * I + g * 64;
        const device uint32_t* wp = w + wrow * ROWW + g * WPG;
        float q = 0.0f, xs = 0.0f;
        for (int c = 0; c < 8; c++) {
            float xv[8];
            for (int i = 0; i < 8; i++) { xv[i] = float(hr[c * 8 + i]); xs += xv[i]; }
            prescale8<BITS>(xv);
            q += qdot8<BITS>(chunk8<BITS>(wp, c), xv);
        }
        float p = float(scales[wrow * NG + g]) * q + float(biases[wrow * NG + g]) * xs;
        for (uint off = NG / 2; off > 0; off /= 2) p += simd_shuffle_xor(p, off);
        acc += float(T(p)) * weights[slot];
    }
    if (g == 0) {
        T yv = T(acc);
        if (HAS_SHARED) yv = T(float(yv) + float(shared[size_t(t) * D + row]));
        y[size_t(t) * D + row] = yv;
    }
"""

_up_kernel = _down_kernel = None
USE_EXPERT_KERNELS = os.environ.get("LOCALS_EXPERT_KERNELS", "1") != "0"
EXPERT_MAX_ROWS = 16  # more tokens (a prompt): gather_qmm, sorted, reads each expert once for all its tokens


def experts_fit(x: mx.array, gate_up, down) -> bool:
    """Whether experts_up/experts_down take these: few rows of a 16-bit dtype, affine groups of 64, input sizes the
    kernels split evenly (the down projection's input in at most 32 groups, a power of 2)."""
    D, I = x.shape[-1], down.input_dims
    ng = I // 64
    return (x.size // D <= EXPERT_MAX_ROWS and x.dtype in (mx.bfloat16, mx.float16)
            and all(m.group_size == 64 and m.mode == "affine" and m.get("biases") is not None and 2 <= m.bits <= 8
                    for m in (gate_up, down))
            and D % 64 == 0 and I % 64 == 0 and 0 < ng <= 32 and ng & (ng - 1) == 0 and I % 2 == 0
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


def experts_up(x: mx.array, inds: mx.array, gate_up, rj: int = 1, tg: int = 128) -> mx.array:
    """swiglu(gate, up) of each token for each of its experts: ``x`` (T, D), ``inds`` (T, K) -> (T * K, I).
    ``rj`` hidden units per simdgroup and ``tg`` threads per threadgroup: the fastest on an M4 Pro at 1 and 8 tokens
    (43.7 and 239 us a layer of Flint against 49 and 265 with two units; four, eight were slower still)."""
    global _up_kernel
    if _up_kernel is None:
        _up_kernel = mx.fast.metal_kernel(name="locals_experts_up", input_names=["x", "inds", "w", "scales", "biases", "nslots"],
                                          output_names=["h"], source=_UP_SRC, header=_QDOT_HEADER)
    T_, D = x.shape
    K = inds.shape[-1]
    I = gate_up["weight"].shape[1] // 2
    S, RJ = T_ * K, rj
    n_sg = S * (I // RJ)
    w, sc, bi = _experts_buffers(gate_up)
    (h,) = _up_kernel(inputs=[x, inds, w, sc, bi, mx.array([S], dtype=mx.uint32)],
                      template=[("T", x.dtype), ("BITS", gate_up.bits), ("D", D), ("I", I), ("K", K), ("RJ", RJ)],
                      grid=(n_sg * 32, 1, 1), threadgroup=(tg, 1, 1), output_shapes=[(S, I)], output_dtypes=[x.dtype])
    return h


def experts_down(h: mx.array, inds: mx.array, weights: mx.array, down, shared=None, tg: int = 256) -> mx.array:
    """sum_k weights[t, k] * down_{inds[t, k]}(h[t, k]) (+ ``shared``): ``h`` (T * K, I) -> (T, D)."""
    global _down_kernel
    if _down_kernel is None:
        _down_kernel = mx.fast.metal_kernel(name="locals_experts_down",
                                            input_names=["h", "inds", "weights", "w", "scales", "biases", "shared", "ntok"],
                                            output_names=["y"], source=_DOWN_SRC, header=_QDOT_HEADER)
    NT, K = inds.shape
    D, I = down["weight"].shape[1], h.shape[-1]
    rps = 32 // (I // 64)
    n_sg = NT * (D // rps)
    has = shared is not None
    w, sc, bi = _experts_buffers(down)
    (y,) = _down_kernel(inputs=[h, inds, weights, w, sc, bi,
                                shared if has else mx.zeros((1,), dtype=h.dtype), mx.array([NT], dtype=mx.uint32)],
                        template=[("T", h.dtype), ("BITS", down.bits), ("D", D), ("I", I), ("K", K), ("HAS_SHARED", int(has))],
                        grid=(n_sg * 32, 1, 1), threadgroup=(tg, 1, 1), output_shapes=[(NT, D)], output_dtypes=[h.dtype])
    return y


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
    """Flint's SparseMoeBlock: the router's float32 logits from its bf16 weight in one kernel, the choice in another."""

    def __call__(self, x: mx.array) -> mx.array:
        inds, weights = route_topk(router(x, self.gate.weight), self.top_k)
        sw, D, k = self.switch_mlp, x.shape[-1], self.top_k
        if USE_EXPERT_KERNELS and hasattr(sw, "gate_up_proj") and experts_fit(x, sw.gate_up_proj, sw.down_proj):
            h = experts_up(x.reshape(-1, D), inds.reshape(-1, k), sw.gate_up_proj)
            y = experts_down(h, inds.reshape(-1, k), weights.reshape(-1, k), sw.down_proj, self.shared_expert(x).reshape(-1, D))
            return y.reshape(x.shape)
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        return y + self.shared_expert(x)


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
