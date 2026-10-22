"""Metal kernels for Flint's decode step, from three facts about its MoE layers.

1. Every router a layer evaluates reads the same input. Decoding evaluates the layer's own router and, to read
   experts ahead, the routers of the next MoE layers on that input: five 256 x 2048 matrices, five products and
   five top-k selections, about 25 small kernels that took 0.27 ms of a layer's 0.54 on the M2 Pro (more than
   its six experts). ``routers`` computes all their logits in one pass and ``route`` selects in one more: the
   layer's own top 6 with their renormalized sigmoid weights, and each guess. Sigmoid is increasing, so the
   top k of the scores are the top k of the logits.
2. The routed experts and the shared expert read the same input too, and their outputs are summed. ``experts``
   computes every expert's gate and up rows together, SwiGLU in registers, then the down projections of all of
   them, weighted, summed and added to the residual, in two kernels instead of about fourteen.
3. A sum over experts splits at any partition of them. The experts already in slots are computed while the
   others are read (``mask``), and the second part adds to the first; the GPU then works while it would wait,
   and an Apple GPU left idle for a millisecond runs its next work at about half speed.

The 4-bit products follow locals/qmv_multi.py's v3 kernel (MLX's affine format, group size 64, nibbles used in
place against x pre-divided by powers of 16). Everything accumulates in float32."""
from typing import List, Optional

import mlx.core as mx

_HEADER = '#define UNROLL _Pragma("clang loop unroll(full)")\n'

# --- 1. routers -------------------------------------------------------------------------------------------------

_ROUTERS = r"""
    // logits[b, j] for j < NR * E: row j % E of router j / E, on x[b]. A simdgroup computes R rows.
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int row0 = (int(threadgroup_position_in_grid.x) * NSG + int(sg)) * R;
    const int b = int(threadgroup_position_in_grid.y);
    if (row0 >= NR * E) {
        return;
    }
    const int ri = row0 / E;
    const device T* W = ri == 0 ? w0 : (ri == 1 ? w1 : (ri == 2 ? w2 : (ri == 3 ? w3 : w4)));
    W += (row0 % E) * K;
    const device T* xb = x + b * K;
    float acc[R];
    UNROLL
    for (int r = 0; r < R; r++) {
        acc[r] = 0.0f;
    }
    for (int k = int(lane) * 8; k < K; k += 256) {
        float xv[8];
        UNROLL
        for (int i = 0; i < 8; i++) {
            xv[i] = float(xb[k + i]);
        }
        UNROLL
        for (int r = 0; r < R; r++) {
            const device T* wr = W + r * K + k;
            UNROLL
            for (int i = 0; i < 8; i++) {
                acc[r] = fma(xv[i], float(wr[i]), acc[r]);
            }
        }
    }
    UNROLL
    for (int r = 0; r < R; r++) {
        const float v = simd_sum(acc[r]);
        if (lane == 0) {
            logits[b * NR * E + row0 + r] = v;
        }
    }
"""

_ROUTE = r"""
    // One threadgroup of E threads per (router, row): the top k of the router's logits, largest first, ties to
    // the lower expert. Router 0 is the layer's own: its experts and their sigmoid weights, renormalized.
    const int ri = int(threadgroup_position_in_grid.x);
    const int b = int(threadgroup_position_in_grid.y);
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int k = ri == 0 ? K0 : (ri == 1 ? K1 : (ri == 2 ? K2 : (ri == 3 ? K3 : K4)));
    const int off = ri == 0 ? 0 : (ri == 1 ? 0 : (ri == 2 ? K1 : (ri == 3 ? K1 + K2 : K1 + K2 + K3)));
    threadgroup float tv[E / 32];
    threadgroup uint ti[E / 32];
    threadgroup uint pick[8];
    threadgroup float pickv[8];
    float v = logits[b * NR * E + ri * E + int(t)];
    v = isnan(v) ? -INFINITY : v;  // (a NaN logit is never chosen over a number)
    for (int j = 0; j < k; j++) {
        const float m = simd_max(v);
        const uint c = simd_min(v == m ? t : 0xffffffffu);
        if (lane == 0) {
            tv[sg] = m;
            ti[sg] = c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            float bv = tv[0];
            uint bi = ti[0];
            for (int s = 1; s < E / 32; s++) {
                if (tv[s] > bv || (tv[s] == bv && ti[s] < bi)) {
                    bv = tv[s];
                    bi = ti[s];
                }
            }
            pick[j] = bi;
            pickv[j] = bv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == pick[j]) {
            v = -INFINITY;
        }
    }
    if (t == 0) {
        if (ri == 0) {
            // each expert's slot (table: -1 for an expert not in one) and this row's flag: clear when one of its
            // experts has no slot or a flag it came with (okin: the layer before's, every row's) was clear
            bool good = true;
            for (int r = 0; r < NOK; r++) {
                good = good && okin[r] != 0;
            }
            for (int j = 0; j < K0; j++) {
                const int sl = table[pick[j]];
                slots[b * K0 + j] = sl < 0 ? 0u : uint(sl);
                good = good && sl >= 0;
            }
            ok[b] = good ? 1u : 0u;
            float s[8];
            float total = 0.0f;
            for (int j = 0; j < K0; j++) {
                s[j] = 1.0f / (1.0f + metal::exp(-pickv[j]));
                total += s[j];
            }
            for (int j = 0; j < K0; j++) {
                inds[b * K0 + j] = pick[j];
                weights[b * K0 + j] = s[j] / (total + 1e-20f);
            }
        } else {
            for (int j = 0; j < k; j++) {
                guesses[b * G + off + j] = pick[j];
            }
        }
    }
"""

_KERNELS = {}


def _kernel(name: str, inputs: List[str], outputs: List[str], source: str):
    k = _KERNELS.get(name)
    if k is None:
        k = _KERNELS[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=source,
                                                   header=_HEADER)
    return k


def routers(x: mx.array, gates: List[mx.array], R: int = 4, NSG: int = 4) -> mx.array:
    """Logits (rows, len(gates) * E), float32, of every router in ``gates`` (each (E, K)) on ``x`` (rows, K)."""
    rows, K = x.shape
    E = gates[0].shape[0]
    NR = len(gates)
    assert 1 <= NR <= 5 and K % 256 == 0 and E % R == 0 and all(g.shape == (E, K) and g.dtype == x.dtype for g in gates)
    ws = list(gates) + [gates[0]] * (5 - NR)
    n_tg = (NR * E + NSG * R - 1) // (NSG * R)
    (out,) = _kernel("locals_flint_routers", ["x", "w0", "w1", "w2", "w3", "w4"], ["logits"], _ROUTERS)(
        inputs=[x] + ws, template=[("T", x.dtype), ("K", K), ("E", E), ("NR", NR), ("R", R), ("NSG", NSG)],
        grid=(n_tg * 32 * NSG, rows, 1), threadgroup=(32 * NSG, 1, 1), output_shapes=[(rows, NR * E)],
        output_dtypes=[mx.float32])
    return out


_CONST = {}


def one() -> mx.array:
    """A set flag (uint32 (1,)): ``ok`` when there is nothing to check."""
    a = _CONST.get("one")
    if a is None:
        a = _CONST["one"] = mx.ones((1,), dtype=mx.uint32)
        mx.eval(a)
    return a


def _placeholder(dtype) -> mx.array:
    a = _CONST.get(("placeholder", dtype))
    if a is None:
        a = _CONST[("placeholder", dtype)] = mx.zeros((1024,), dtype=dtype)  # (one element would be a Metal constant)
        mx.eval(a)
    return a


def _no_table(E: int) -> mx.array:
    a = _CONST.get(E)
    if a is None:
        a = _CONST[E] = mx.zeros((E,), dtype=mx.int32)
        mx.eval(a)
    return a


def route(logits: mx.array, k: int, ranks: List[int], table: Optional[mx.array] = None, okin: Optional[mx.array] = None):
    """From ``routers``' logits: (experts (rows, k) uint32, their weights (rows, k) float32, guesses (rows,
    sum(ranks)) uint32, slots (rows, k) uint32, ok (1,) uint32): router 0's top k with renormalized sigmoid weights,
    router i's top ranks[i - 1]; with ``table`` (int32 (E,): each expert's slot, -1 if it has none), each expert's
    slot and, per row, ok (rows,) uint32: whether every expert of the row has a slot and every flag in ``okin``
    is set (the layer before's; otherwise slots and ok are not meaningful)."""
    rows, n = logits.shape
    NR = 1 + len(ranks)
    E = n // NR
    assert k <= 8 and all(0 < r <= 8 for r in ranks) and len(ranks) <= 4 and E % 32 == 0 and E <= 1024
    rk = list(ranks) + [0] * (4 - len(ranks))
    G = max(1, sum(ranks))
    okin = okin if okin is not None else one()
    inds, weights, guesses, slots, ok = _kernel("locals_flint_route", ["logits", "table", "okin"],
                                                ["inds", "weights", "guesses", "slots", "ok"], _ROUTE)(
        inputs=[logits, table if table is not None else _no_table(E), okin],
        template=[("E", E), ("NR", NR), ("K0", k), ("K1", rk[0]), ("K2", rk[1]), ("K3", rk[2]), ("K4", rk[3]), ("G", G),
                  ("NOK", okin.size)],
        grid=(NR * E, rows, 1), threadgroup=(E, 1, 1), output_shapes=[(rows, k), (rows, k), (rows, G), (rows, k), (rows,)],
        output_dtypes=[mx.uint32, mx.float32, mx.uint32, mx.uint32, mx.uint32])
    return inds, weights, guesses[:, :sum(ranks)], slots, ok


# --- 2. experts --------------------------------------------------------------------------------------------------

# Shared by both kernels: one lane's VPT = 16 consecutive inputs of a 512-wide block, pre-divided by the powers of
# 16 its nibbles carry in place (qmv_multi v3), and their sum (for the biases).
_X_BLOCK = r"""
            float xt[16];
            float xs = 0.0f;
            UNROLL
            for (int p = 0; p < 2; p++) {
                UNROLL
                for (int i = 0; i < 8; i++) {
                    const float a = float(XSRC[p * 8 + i]);
                    xs += a;
                    xt[p * 8 + i] = a * (i < 5 ? 1.0f / float(1u << (4 * i)) : 1.0f / float(1u << (4 * i - 12)));
                }
            }
"""

# d = the dot product of a lane's 16 inputs with WORDS (two uint32 of 4-bit weights)
_DOT = r"""
                float d = 0.0f;
                UNROLL
                for (int p = 0; p < 2; p++) {
                    const uint32_t word = WORDS[p];
                    const uint32_t hi = word >> 12;
                    UNROLL
                    for (int i = 0; i < 8; i++) {
                        const uint32_t bits = i < 5 ? (word & (0xfu << (4 * i))) : (hi & (0xfu << (4 * i - 12)));
                        d = fma(xt[p * 8 + i], as_type<float>(0x4B000000u | bits) - 8388608.0f, d);
                    }
                }
"""

_UP = r"""
    // h[b, j * EI + r] = silu(gate_j[r] . x_b) * (up_j[r] . x_b) for routed expert j (in slot slots[b, j], if
    // mask[b, j]); rows KE * EI + r: the shared expert's (if SHARED). A simdgroup computes R rows of one expert.
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int row0 = (int(threadgroup_position_in_grid.x) * NSG + int(sg)) * R;
    const int b = int(threadgroup_position_in_grid.y);
    constexpr int NH = KE * EI + SI;
    bool go = true;
    for (int r = 0; r < NOK; r++) {
        go = go && ok[r] != 0;
    }
    if (row0 >= KE * EI + (SHARED ? SI : 0) || !go) {
        return;
    }
    constexpr int Kw = K / 8;
    constexpr int Kg = K / 64;
    // routed experts' scales and biases have type ST (the files' bf16), the shared expert's T: both read as float
    const device uint32_t* gwp;
    const device uint32_t* uwp;
    const device ST* gsp = nullptr;
    const device ST* gbp = nullptr;
    const device ST* usp = nullptr;
    const device ST* ubp = nullptr;
    const device T* hgsp = nullptr;
    const device T* hgbp = nullptr;
    const device T* husp = nullptr;
    const device T* hubp = nullptr;
    const bool routed = row0 < KE * EI;
    if (routed) {
        const int j = row0 / EI;
        if (mask[b * KE + j] == 0) {
            return;
        }
        const int r = int(slots[b * KE + j]) * EI + row0 % EI;
        gwp = gw + r * Kw; gsp = gs + r * Kg + (lane * 16) / 64; gbp = gb + r * Kg + (lane * 16) / 64;
        uwp = uw + r * Kw; usp = us + r * Kg + (lane * 16) / 64; ubp = ub + r * Kg + (lane * 16) / 64;
    } else {
        const int r = row0 - KE * EI;
        gwp = sgw + r * Kw; hgsp = sgs + r * Kg + (lane * 16) / 64; hgbp = sgb + r * Kg + (lane * 16) / 64;
        uwp = suw + r * Kw; husp = sus + r * Kg + (lane * 16) / 64; hubp = sub + r * Kg + (lane * 16) / 64;
    }
    gwp += lane * 2; uwp += lane * 2;
    const device T* xp = x + b * K + lane * 16;
    float ag[R], au[R];
    UNROLL
    for (int r = 0; r < R; r++) {
        ag[r] = 0.0f;
        au[r] = 0.0f;
    }
    for (int k = 0; k < K; k += 512) {
#define XSRC (xp + k)
""" + _X_BLOCK + r"""
#undef XSRC
        UNROLL
        for (int r = 0; r < R; r++) {
            {
#define WORDS (gwp + r * Kw + k / 8)
""" + _DOT + r"""
#undef WORDS
                const int q = r * Kg + k / 64;
                ag[r] += (routed ? float(gsp[q]) : float(hgsp[q])) * d + (routed ? float(gbp[q]) : float(hgbp[q])) * xs;
            }
            {
#define WORDS (uwp + r * Kw + k / 8)
""" + _DOT + r"""
#undef WORDS
                const int q = r * Kg + k / 64;
                au[r] += (routed ? float(usp[q]) : float(husp[q])) * d + (routed ? float(ubp[q]) : float(hubp[q])) * xs;
            }
        }
    }
    UNROLL
    for (int r = 0; r < R; r++) {
        const float g = simd_sum(ag[r]);
        const float u = simd_sum(au[r]);
        if (lane == 0) {
            h[b * NH + row0 + r] = g / (1.0f + metal::exp(-g)) * u;
        }
    }
"""

_DOWN = r"""
    // y[b, n] = residual[b, n] + sum_j mask[b, j] weights[b, j] (down_j[n] . h[b, j]) (+ the shared expert's
    // down row on its h, if SHARED). A simdgroup computes R rows of y.
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int n0 = (int(threadgroup_position_in_grid.x) * NSG + int(sg)) * R;
    const int b = int(threadgroup_position_in_grid.y);
    constexpr int NH = KE * EI + SI;
    if (n0 >= D) {
        return;
    }
    float acc[R];
    UNROLL
    for (int r = 0; r < R; r++) {
        acc[r] = 0.0f;
    }
    bool go = true;
    for (int r = 0; r < NOK; r++) {
        go = go && ok[r] != 0;
    }
    const int nj = go ? KE + (SHARED ? 1 : 0) : 0;  // (a pass with a flag clear: the residual alone)
    for (int j = 0; j < nj; j++) {
        const bool shared = j == KE;
        if (!shared && mask[b * KE + j] == 0) {
            continue;
        }
        const int KI = shared ? SI : EI;
        const float wj = shared ? 1.0f : weights[b * KE + j];
        const device float* hp = h + b * NH + j * EI + lane * 16;
        const int row = shared ? n0 : int(slots[b * KE + j]) * D + n0;
        const device uint32_t* wp = (shared ? sdw : dw) + row * (KI / 8) + lane * 2;
        const device ST* sc = ds + row * (KI / 64) + (lane * 16) / 64;  // (routed: ST; shared: T, below)
        const device ST* bi = db + row * (KI / 64) + (lane * 16) / 64;
        const device T* hsc = sds + row * (KI / 64) + (lane * 16) / 64;
        const device T* hbi = sdb + row * (KI / 64) + (lane * 16) / 64;
        for (int k = 0; k < KI; k += 512) {
#define XSRC (hp + k)
""" + _X_BLOCK + r"""
#undef XSRC
            UNROLL
            for (int r = 0; r < R; r++) {
#define WORDS (wp + r * (KI / 8) + k / 8)
""" + _DOT + r"""
#undef WORDS
                const int q = r * (KI / 64) + k / 64;
                acc[r] += wj * ((shared ? float(hsc[q]) : float(sc[q])) * d + (shared ? float(hbi[q]) : float(bi[q])) * xs);
            }
        }
    }
    UNROLL
    for (int r = 0; r < R; r++) {
        const float v = simd_sum(acc[r]);
        if (lane == 0) {
            y[b * D + n0 + r] = static_cast<O>(float(residual[b * D + n0 + r]) + v);
        }
    }
"""


def experts(x: mx.array, residual: mx.array, routed: List[mx.array], shared: Optional[List[mx.array]], slots: mx.array,
            mask: mx.array, weights: mx.array, out_dtype=None, ok: Optional[mx.array] = None, R: int = 4,
            NSG: int = 2) -> mx.array:
    """residual + the masked routed experts of each row (weighted) + the shared expert (if ``shared``).

    x, residual: (rows, D); routed: the nine slot arrays (gate, up, down x weight, scales, biases), slot first, the
    scales and biases in any float type (the files' bfloat16); shared: the shared expert's nine arrays, scales and
    biases in x's type, or None; slots, mask: (rows, k) uint32; weights: (rows, k) float32.
    Returns (rows, D) in ``out_dtype`` (default x's). With a flag of ``ok`` (uint32, route's per-row flags) clear,
    nothing is computed and the result is the residual."""
    rows, D = x.shape
    gw, gs, gb, uw, us, ub, dw, ds, db = routed
    EI = gw.shape[1]
    KE = slots.shape[1]
    SI = shared[0].shape[0] if shared is not None else 0
    if shared is not None:
        sh = shared
    else:  # (not read: SHARED is 0)
        z, zw = _placeholder(x.dtype), _placeholder(mx.uint32)
        sh = [zw, z, z, zw, z, z, zw, z, z]
    if shared is None:
        SI = EI  # (h keeps the width; the shared rows are not computed)
    assert D % 512 == 0 and EI % 512 == 0 and SI % 512 == 0 and EI % R == 0 and D % R == 0
    T = x.dtype
    O = out_dtype or T
    okf = ok if ok is not None else one()
    tmpl = [("T", T), ("ST", gs.dtype), ("K", D), ("D", D), ("EI", EI), ("SI", SI), ("KE", KE),
            ("SHARED", int(shared is not None)), ("R", R), ("NSG", NSG), ("NOK", okf.size)]
    n_up = KE * EI + (SI if shared is not None else 0)
    (h,) = _kernel("locals_flint_experts_up", ["x", "gw", "gs", "gb", "uw", "us", "ub", "sgw", "sgs", "sgb", "suw", "sus",
                                               "sub", "slots", "mask", "ok"], ["h"], _UP)(
        inputs=[x, gw, gs, gb, uw, us, ub, sh[0], sh[1], sh[2], sh[3], sh[4], sh[5], slots, mask, okf],
        template=tmpl, grid=(((n_up + NSG * R - 1) // (NSG * R)) * 32 * NSG, rows, 1), threadgroup=(32 * NSG, 1, 1),
        output_shapes=[(rows, KE * EI + SI)], output_dtypes=[mx.float32])
    (y,) = _kernel("locals_flint_experts_down", ["h", "dw", "ds", "db", "sdw", "sds", "sdb", "slots", "mask", "weights",
                                                 "residual", "ok"], ["y"], _DOWN)(
        inputs=[h, dw, ds, db, sh[6], sh[7], sh[8], slots, mask, weights, residual, okf],
        template=tmpl + [("O", O)], grid=((D // (NSG * R)) * 32 * NSG, rows, 1), threadgroup=(32 * NSG, 1, 1),
        output_shapes=[(rows, D)], output_dtypes=[O])
    return y
