"""Routed experts for a block of many rows that share few experts (Clover's preview blocks).

A Clover preview block is 32 tokens routed within the block's coreset: its 256 (token, expert) slots
fall on 30 to 70 experts, 4 to 8 slots each (10 or so experts when every position is still a mask
token). moe.py's kernels give a run of slots that share an expert to one simdgroup, which reads the
expert once and then works through the run one slot after another; at 32 rows a run is long, the
GPU has few simdgroups to spread, and each slot unpacks the expert's weights again. Here a run is cut
in tiles of ``TILE`` slots, each tile its own simdgroup (the expert is read once a tile, mostly from
cache), and a simdgroup unpacks each chunk of its weights once for all the slots of its tile.

The slots are taken in expert order (moe.expert_order), and ``tile_lengths`` gives each position of
that order the number of slots its simdgroup serves (0 for a position inside a tile). The sums run as
moe.py's: the same bits as experts_up and experts_down, tested.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .moe import _COMBINE_SRC, _QDOT_HEADER, _experts_buffers, expert_order

_HEADER = _QDOT_HEADER + """
#define UNROLL _Pragma("clang loop unroll(full)")
// The 8 values of a chunk as floats where qdot8 has them (each at its place for the masked widths; the input
// prescaled by prescale8 to match).
// 8 consecutive 16-bit inputs (16-byte aligned) as floats, in one load.
template <typename T, int BF16>
inline void load8(const device T* p, thread float* out) {
    const uint4 u = *((const device uint4*)p);
    const uint v[4] = {u.x, u.y, u.z, u.w};
    for (int i = 0; i < 4; i++) {
        if (BF16) {
            out[2 * i] = as_type<float>(v[i] << 16);
            out[2 * i + 1] = as_type<float>(v[i] & 0xffff0000u);
        } else {
            const float2 f = float2(as_type<half2>(v[i]));
            out[2 * i] = f.x;
            out[2 * i + 1] = f.y;
        }
    }
}

template <int BITS>
inline void unpack8(ulong v, thread float* out) {
    constexpr uint mask = (1u << BITS) - 1;
    if (BITS == 3) {
        const uint w = uint(v);
        for (int i = 0; i < 7; i++) out[i] = as_type<float>((w & (7u << (3 * i))) | 0x4B000000u) - 8388608.0f;
        out[7] = as_type<float>(((w >> 3) & (7u << 18)) | 0x4B000000u) - 8388608.0f;
    } else if (BITS * 8 <= 32) {
        const uint w = uint(v);
        for (int i = 0; i < 8; i++) out[i] = float(w & (mask << (BITS * i)));
    } else {
        for (int i = 0; i < 8; i++) out[i] = float(uint(v >> (BITS * i)) & mask);
    }
}
"""

# One simdgroup per (tile, R hidden units), a lane per group of 64 inputs (D <= 2048).
_UP_SRC = """
    constexpr int NG = D / 64, WPG = 2 * BITS;
    constexpr uint ROWW = NG * WPG, NB = I / R;
    const uint sgid = thread_position_in_grid.x / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint S = nslots[0];
    const uint j0 = (sgid % NB) * R, p = sgid / NB;
    if (p >= S || lane >= NG) return;
    const uint n = tiles[p];
    if (n == 0) return;
    const uint e = inds[order[p]];
    const uint g = lane;
    const size_t row0 = size_t(e) * 2 * I + j0;  // the units' gate rows; their up rows are I further
    float acc[2 * R][TILE], xs[TILE];
    const device T* xr[TILE];
    UNROLL for (int t = 0; t < TILE; t++) {
        xs[t] = 0.0f;
        UNROLL for (int r = 0; r < 2 * R; r++) acc[r][t] = 0.0f;
        xr[t] = x + size_t(order[p + min(uint(t), n - 1)] / K) * D + g * 64;
    }
    for (int c = 0; c < 8; c++) {
        float wv[2 * R][8];
        UNROLL for (int r = 0; r < R; r++) {
            unpack8<BITS>(chunk8<BITS>(w + (row0 + r) * ROWW + g * WPG, c), wv[r]);
            unpack8<BITS>(chunk8<BITS>(w + (row0 + r + I) * ROWW + g * WPG, c), wv[R + r]);
        }
        UNROLL for (int t = 0; t < TILE; t++) {
            if (uint(t) < n) {
                float xv[8];
                load8<T, BF16>(xr[t] + c * 8, xv);
                UNROLL for (int i = 0; i < 8; i++) xs[t] += xv[i];
                prescale8<BITS>(xv);
                UNROLL for (int r = 0; r < 2 * R; r++) {
                    float s = 0.0f;
                    UNROLL for (int i = 0; i < 8; i++) s += wv[r][i] * xv[i];
                    acc[r][t] += s;
                }
            }
        }
    }
    UNROLL for (int r = 0; r < R; r++) {
        const float sg = float(scales[(row0 + r) * NG + g]), bg = float(biases[(row0 + r) * NG + g]);
        const float su = float(scales[(row0 + r + I) * NG + g]), bu = float(biases[(row0 + r + I) * NG + g]);
        UNROLL for (int t = 0; t < TILE; t++) {
            if (uint(t) >= n) continue;
            const T gt = T(simd_sum(sg * acc[r][t] + bg * xs[t])), ut = T(simd_sum(su * acc[R + r][t] + bu * xs[t]));
            if (lane == 0) {
                const T sig = mlx_sigmoid<T>(gt);
                const T act = T(float(gt) * float(sig));
                h[size_t(order[p + t]) * I + j0 + r] = T(float(act) * float(ut));
            }
        }
    }
"""

# One simdgroup per (tile, 32 / (I / 64) * R output rows): a lane takes one of a row's I / 64 groups in R rows.
_DOWN_SRC = """
    constexpr int NG = I / 64, WPG = 2 * BITS, RPS = 32 / NG;
    constexpr uint ROWW = NG * WPG, ROWS = RPS * R;
    const uint sgid = thread_position_in_grid.x / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint S = nslots[0];
    const uint rb = sgid % (D / ROWS), p = sgid / (D / ROWS);
    const uint g = lane % NG, row0 = rb * ROWS + lane / NG;  // this lane's rows: row0 + r * RPS
    if (p >= S) return;
    const uint n = tiles[p];
    if (n == 0) return;
    const uint e = inds[order[p]];
    float acc[R][TILE], xs[TILE];
    const device T* hr[TILE];
    UNROLL for (int t = 0; t < TILE; t++) {
        xs[t] = 0.0f;
        UNROLL for (int r = 0; r < R; r++) acc[r][t] = 0.0f;
        hr[t] = h + size_t(order[p + min(uint(t), n - 1)]) * I + g * 64;
    }
    for (int c = 0; c < 8; c++) {
        float wv[R][8];
        UNROLL for (int r = 0; r < R; r++)
            unpack8<BITS>(chunk8<BITS>(w + (size_t(e) * D + row0 + r * RPS) * ROWW + g * WPG, c), wv[r]);
        UNROLL for (int t = 0; t < TILE; t++) {
            if (uint(t) < n) {
                float xv[8];
                load8<T, BF16>(hr[t] + c * 8, xv);
                UNROLL for (int i = 0; i < 8; i++) xs[t] += xv[i];
                prescale8<BITS>(xv);
                UNROLL for (int r = 0; r < R; r++) {
                    float s = 0.0f;
                    UNROLL for (int i = 0; i < 8; i++) s += wv[r][i] * xv[i];
                    acc[r][t] += s;
                }
            }
        }
    }
    UNROLL for (int r = 0; r < R; r++) {
        const size_t wrow = size_t(e) * D + row0 + r * RPS;
        const float sc = float(scales[wrow * NG + g]), bi = float(biases[wrow * NG + g]);
        UNROLL for (int t = 0; t < TILE; t++) {
            if (uint(t) >= n) continue;
            float pv = sc * acc[r][t] + bi * xs[t];
            for (uint off = NG / 2; off > 0; off /= 2) pv += simd_shuffle_xor(pv, off);
            if (g == 0) ys[size_t(order[p + t]) * D + row0 + r * RPS] = T(pv);
        }
    }
"""

_kernels: dict = {}


def _kernel(name, src, inputs, outputs):
    k = _kernels.get(name)
    if k is None:
        k = _kernels[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=src,
                                                  header=_HEADER)
    return k


def tile_lengths(inds: mx.array, order: mx.array, tile: int) -> mx.array:
    """For each position of ``order`` (the slots of ``inds`` in expert order): how many slots the simdgroup at that
    position serves (up to ``tile``, within its expert's run), or 0 inside a tile."""
    se = mx.take(inds.reshape(-1), order)
    S = se.size
    pos = mx.arange(S, dtype=mx.int32)
    new = mx.concatenate([mx.array([True]), se[1:] != se[:-1]])
    last = mx.concatenate([se[1:] != se[:-1], mx.array([True])])
    start = mx.cummax(mx.where(new, pos, 0))
    end = mx.cummin(mx.where(last, pos, S - 1), reverse=True)
    off = pos - start
    return mx.where(off % tile == 0, mx.minimum(tile, end - pos + 1), 0).astype(mx.uint32)


def experts_up(x: mx.array, inds: mx.array, gate_up, order: Optional[mx.array] = None, tiles: Optional[mx.array] = None,
               tile: int = 8, r: int = 2) -> mx.array:
    """moe.experts_up (swiglu(gate, up) of each token for each of its experts: ``x`` (T, D), ``inds`` (T, K) ->
    (T * K, I)), a tile of slots a simdgroup."""
    T_, D = x.shape
    K = inds.shape[-1]
    I = gate_up["weight"].shape[1] // 2
    S = T_ * K
    order = expert_order(inds) if order is None else order
    tiles = tile_lengths(inds, order, tile) if tiles is None else tiles
    w, sc, bi = _experts_buffers(gate_up)
    kern = _kernel("locals_experts_up_tiles", _UP_SRC, ["x", "inds", "order", "tiles", "w", "scales", "biases", "nslots"], ["h"])
    (h,) = kern(inputs=[x, inds.reshape(-1), order, tiles, w, sc, bi, mx.array([S], dtype=mx.uint32)],
                template=[("T", x.dtype), ("BITS", gate_up.bits), ("D", D), ("I", I), ("K", K), ("R", r), ("TILE", tile),
                          ("BF16", int(x.dtype == mx.bfloat16))],
                grid=(S * (I // r) * 32, 1, 1), threadgroup=(64, 1, 1), output_shapes=[(S, I)], output_dtypes=[x.dtype])
    return h


def experts_down(h: mx.array, inds: mx.array, weights: mx.array, down, shared=None, order: Optional[mx.array] = None,
                 tiles: Optional[mx.array] = None, tile: int = 8, r: int = 4) -> mx.array:
    """moe.experts_down (sum_k weights[t, k] * down_{inds[t, k]}(h[t, k]) (+ ``shared``): ``h`` (T * K, I) -> (T, D)),
    a tile of slots a simdgroup."""
    NT, K = inds.shape
    D, I = down["weight"].shape[1], h.shape[-1]
    S = NT * K
    rps = 32 // (I // 64)
    order = expert_order(inds) if order is None else order
    tiles = tile_lengths(inds, order, tile) if tiles is None else tiles
    w, sc, bi = _experts_buffers(down)
    kern = _kernel("locals_experts_down_tiles", _DOWN_SRC, ["h", "inds", "order", "tiles", "w", "scales", "biases", "nslots"], ["ys"])
    (ys,) = kern(inputs=[h, inds.reshape(-1), order, tiles, w, sc, bi, mx.array([S], dtype=mx.uint32)],
                 template=[("T", h.dtype), ("BITS", down.bits), ("D", D), ("I", I), ("K", K), ("R", r), ("TILE", tile),
                           ("BF16", int(h.dtype == mx.bfloat16))],
                 grid=(S * (D // (rps * r)) * 32, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(S, D)],
                 output_dtypes=[h.dtype])
    has = shared is not None
    none = mx.zeros((1,), dtype=h.dtype)
    combine = _kernel("locals_experts_combine_tiles", _COMBINE_SRC, ["ys", "weights", "shared", "res", "ntok"], ["y"])
    (y,) = combine(inputs=[ys, weights.astype(mx.float32), shared if has else none, none, mx.array([NT], dtype=mx.uint32)],
                   template=[("T", h.dtype), ("D", D), ("K", K), ("HAS_SHARED", int(has)), ("HAS_RES", 0)],
                   grid=(NT * D, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(NT, D)], output_dtypes=[h.dtype])
    return y



TILE = 4  # slots a simdgroup: at 32 rows on 48 experts, 1.4 ms a layer for both kernels (2 for SwitchGLU)
TILED_MAX_ROWS = 64  # more rows (a prompt): SwitchGLU's sorted gather, which reads each expert once for many rows

