"""Grouped expert kernels for decoding many requests at once through a mixture-of-experts model.

A decode step of B requests, each token routed to k of E experts, has B * k (token, expert) pairs, but only
E * (1 - (1 - k / E)^B) distinct experts: for 64 experts and 8 a token, 26 for 4 requests, 54 for 16, 59 for 32.
MLX's gather_qmm (mlx-lm's SwitchGLU) reads an expert's weights for each pair: on the M2 Pro a Mellum layer's
experts took 0.22 ms for one request and 3.0 ms for 16, and the 54 experts of 16 requests are 173 MB, about
0.9 ms of reads at the GPU's bandwidth. Here the pairs are first grouped by expert (one small kernel: a counting
sort), then each threadgroup takes one used expert and a block of its output rows, decodes the expert's weights
once and applies them to every pair routed to it: an expert is read once a step whatever the number of requests.

Formats: MLX affine 4-bit with group size 64 (scales and biases) and MLX mxfp4 (4-bit e2m1 values, a power-of-two
uint8 scale per 32, no biases); both decoded with bit operations only (no table, no int -> float conversion). Accumulation in float32; the intermediate activations stay float32."""
from typing import Optional

import mlx.core as mx

_HEADER = """
constant float FP4[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                          -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};  // e2m1 (mxfp4's values)
// nibble i of w as a float, without an int -> float conversion (slow on Apple GPUs): OR-ed into the mantissa of
// 2^23, minus 2^23
#define Q4(w, i) (as_type<float>(0x4B000000u | (((w) >> (4 * (i))) & 0xfu)) - 8388608.0f)
// nibble i of w read as e2m1 (mxfp4's sign, 2 exponent bits with bias 1, 1 mantissa bit), as half bits: sign to
// bit 15, exponent and mantissa to bits 9-11; the value is then 2^-14 times e2m1's (its subnormal 0.5 too)
#define E2M1(w, i) float(as_type<half>(ushort(((((w) >> (4 * (i))) & 0x8u) << 12) | ((((w) >> (4 * (i))) & 0x7u) << 9))))
"""

_GROUP = r"""
    // The pairs grouped by expert (one threadgroup): order = pair ids by expert, starts[s] = where segment s
    // starts (starts[nseg] = P), experts[s] = its expert, nseg[0] = the number of segments.
    const uint tid = thread_position_in_threadgroup.x;
    threadgroup atomic_int cnt[E];
    threadgroup int base[E];
    for (int e = int(tid); e < E; e += TPG) {
        atomic_store_explicit(&cnt[e], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = int(tid); p < P; p += TPG) {
        atomic_fetch_add_explicit(&cnt[int(inds[p])], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int at = 0, s = 0;
        for (int e = 0; e < E; e++) {
            const int c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
            base[e] = at;
            if (c > 0) {
                starts[s] = at;
                experts[s] = e;
                s++;
            }
            at += c;
        }
        starts[s] = at;
        nseg[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = int(tid); e < E; e += TPG) {
        atomic_store_explicit(&cnt[e], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = int(tid); p < P; p += TPG) {
        const int e = int(inds[p]);
        order[base[e] + atomic_fetch_add_explicit(&cnt[e], 1, memory_order_relaxed)] = p;
    }
"""

_EXPERTS = r"""
    // One threadgroup: segment s (one used expert) and output rows [blk * NSG * R, +NSG * R) of it. GLU:
    // h[p, n] = silu(gate_e[n] . x) * (up_e[n] . x) with x the token row of pair p; else out[p, n] = w_e[n] . x
    // with x = row p of the input. Each lane decodes its 8 weights of a row once and applies them to every pair.
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int s = int(threadgroup_position_in_grid.x) / NB;
    const int blk = int(threadgroup_position_in_grid.x) % NB;
    if (s >= nseg[0]) {
        return;
    }
    const int e = experts[s];
    const int p0 = starts[s];
    const int m = starts[s + 1] - p0;
    const int n0 = (blk * NSG + int(sg)) * R;
    if (n0 >= N) {
        return;
    }
    constexpr int Kw = K / 8;
    constexpr int Kg = K / G;
    for (int c0 = 0; c0 < m; c0 += C) {
        const int cn = min(C, m - c0);
        float a0[C][R];
        float a1[C][R];
        for (int c = 0; c < C; c++) {
            for (int r = 0; r < R; r++) {
                a0[c][r] = 0.0f;
                a1[c][r] = 0.0f;
            }
        }
        for (int kb = 0; kb < K; kb += 256) {
            const int kk = kb + int(lane) * 8;
            if (kk >= K) {
                break;
            }
            float f0[R][8];
            float f1[R][8];
            for (int r = 0; r < R; r++) {
                const int row = e * N + n0 + r;
                const uint32_t w0 = W0[row * Kw + kk / 8];
                const float s0 = FMT == 0 ? float(S0[row * Kg + kk / G]) : metal::exp2(float(S0[row * Kg + kk / G]) - 127.0f);
                const float b0 = FMT == 0 ? float(B0[row * Kg + kk / G]) : 0.0f;
                for (int i = 0; i < 8; i++) {
                    f0[r][i] = FMT == 0 ? fma(s0, DEC ? Q4(w0, i) : float((w0 >> (4 * i)) & 0xfu), b0)
                                        : s0 * (DEC ? E2M1(w0, i) * 16384.0f : FP4[(w0 >> (4 * i)) & 0xfu]);
                }
                if (GLU) {
                    const uint32_t w1 = W1[row * Kw + kk / 8];
                    const float s1 = FMT == 0 ? float(S1[row * Kg + kk / G]) : metal::exp2(float(S1[row * Kg + kk / G]) - 127.0f);
                    const float b1 = FMT == 0 ? float(B1[row * Kg + kk / G]) : 0.0f;
                    for (int i = 0; i < 8; i++) {
                        f1[r][i] = FMT == 0 ? fma(s1, DEC ? Q4(w1, i) : float((w1 >> (4 * i)) & 0xfu), b1)
                                            : s1 * (DEC ? E2M1(w1, i) * 16384.0f : FP4[(w1 >> (4 * i)) & 0xfu]);
                    }
                }
            }
            for (int c = 0; c < cn; c++) {
                const int p = order[p0 + c0 + c];
                const int xr = GLU ? p / KE : p;
                float xv[8];
                for (int i = 0; i < 8; i++) {
                    xv[i] = float(x[xr * K + kk + i]);
                }
                for (int r = 0; r < R; r++) {
                    float d0 = 0.0f, d1 = 0.0f;
                    for (int i = 0; i < 8; i++) {
                        d0 = fma(xv[i], f0[r][i], d0);
                        if (GLU) {
                            d1 = fma(xv[i], f1[r][i], d1);
                        }
                    }
                    a0[c][r] += d0;
                    a1[c][r] += d1;
                }
            }
        }
        for (int c = 0; c < cn; c++) {
            const int p = order[p0 + c0 + c];
            for (int r = 0; r < R; r++) {
                const float g = simd_sum(a0[c][r]);
                const float u = GLU ? simd_sum(a1[c][r]) : 0.0f;
                if (lane == 0) {
                    out[p * N + n0 + r] = GLU ? g / (1.0f + metal::exp(-g)) * u : g;
                }
            }
        }
    }
"""

_KERNELS = {}


def _kernel(name, inputs, outputs, source):
    k = _KERNELS.get(name)
    if k is None:
        k = _KERNELS[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=source,
                                                   header=_HEADER)
    return k


def _fmt(mode: str, bits: int, group: int) -> Optional[int]:
    if bits == 4 and mode == "affine" and group == 64:
        return 0
    if bits == 4 and mode == "mxfp4" and group == 32:
        return 1
    return None


def supported(mode: str, bits: int, group: int, K: int) -> bool:
    return _fmt(mode, bits, group) is not None and K % 8 == 0 and K % group == 0


_DUMMY = {}


def _dummy(dtype) -> mx.array:
    a = _DUMMY.get(dtype)
    if a is None:
        a = _DUMMY[dtype] = mx.zeros((1024,), dtype=dtype)  # (more than one element: a device buffer, not a constant)
        mx.eval(a)
    return a


def group(inds: mx.array, E: int):
    """(order, starts, experts, nseg) of the flat expert indices ``inds`` (P,) uint32 over E experts."""
    P = inds.size
    S = min(E, P)
    return _kernel("locals_group_pairs", ["inds"], ["order", "starts", "experts", "nseg"], _GROUP)(
        inputs=[inds], template=[("E", E), ("P", P), ("TPG", 256)], grid=(256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(P,), (S + 1,), (S,), (1,)], output_dtypes=[mx.int32, mx.int32, mx.int32, mx.int32])


# weights decoded with bit operations (1) or a table and int -> float conversions (0), per format: on the M2 Pro the
# bit operations won for affine (1.31x mlx-lm at 16 requests against 1.23x) and the table for mxfp4 (1.36 against
# 1.11: e2m1 through half bits costs a conversion too)
DEC = {0: 1, 1: 0}


def _launch(x, k, grouping, mats, fmt, gsize, glu, R=2, NSG=4, C=None, dec=None):
    """mats: [(weight, scales, biases or None)] - two for GLU (gate, up), one otherwise."""
    order, starts, experts, nseg = grouping
    K = x.shape[1]
    P = order.size
    W0, S0, B0 = mats[0]
    E, N = W0.shape[0], W0.shape[1]
    W1, S1, B1 = mats[1] if glu else (W0, S0, B0)
    zb = _dummy(S0.dtype if fmt == 0 else x.dtype)
    NB = (N + NSG * R - 1) // (NSG * R)
    S = experts.size
    if C is None:  # pairs a lane takes at once: about the pairs an expert has (2 for 16 requests of 8 of 64 experts;
        # more pairs at once than that cost registers, fewer decode the weights again for the next ones)
        per = P / (E * (1 - (1 - min(1.0, k / E)) ** max(1, P // max(1, k))))
        C = 2 if per <= 2.5 else (4 if per <= 5 else 8)
    (out,) = _kernel("locals_grouped_experts", ["x", "order", "starts", "experts", "nseg", "W0", "S0", "B0", "W1", "S1", "B1"],
                     ["out"], _EXPERTS)(
        inputs=[x, order, starts, experts, nseg, W0, S0, B0 if B0 is not None else zb, W1, S1, B1 if B1 is not None else zb],
        template=[("K", K), ("N", N), ("KE", k), ("G", gsize), ("FMT", fmt), ("GLU", int(glu)), ("R", R), ("NSG", NSG),
                  ("NB", NB), ("C", C), ("DEC", DEC[fmt] if dec is None else dec)],
        grid=(S * NB * 32 * NSG, 1, 1), threadgroup=(32 * NSG, 1, 1), output_shapes=[(P, N)], output_dtypes=[mx.float32])
    return out


def switch_glu(x: mx.array, inds: mx.array, gate, up, down, mode: str, gsize: int, out_dtype=None) -> mx.array:
    """mlx-lm's SwitchGLU (SwiGLU activation) for rows x (M, D), experts inds (M, k): (M, k, D), each expert's
    weights read once. gate, up, down: (weight, scales, biases or None) stacked over experts."""
    fmt = _fmt(mode, 4, gsize)
    M, k = inds.shape
    E = gate[0].shape[0]
    grouping = group(inds.reshape(-1).astype(mx.uint32), E)
    h = _launch(x, k, grouping, [gate, up], fmt, gsize, True)
    y = _launch(h, k, grouping, [down], fmt, gsize, False)
    return y.reshape(M, k, -1).astype(out_dtype or x.dtype)


ENABLED = __import__("os").environ.get("LOCALS_MOE_KERNEL", "1") != "0"
MAX_ROWS = 64  # tokens of a step the grouped kernels take (decoding several requests); longer passes: MLX's
# Pairs an expert has, on average, up to which the grouped kernels win: MLX's gather_qmm sorts a step's pairs by
# expert from 64 of them and runs its tiled kernel, which fills its tiles from about 5 tokens an expert. On the M2
# Pro the grouped kernels were 1.2-1.45x faster up to 4 an expert (64 mxfp4 experts, 8 a token: 32 requests; 128
# affine experts: 64) and 0.71x at 6.
MAX_PER_EXPERT = 4.5


def _mats(glu) -> Optional[tuple]:
    """The SwitchGLU's (gate, up, down) arrays and (mode, group size) if the grouped kernels compute it, else None."""
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwiGLU

    layers = [getattr(glu, n, None) for n in ("gate_proj", "up_proj", "down_proj")]
    if not all(isinstance(l, QuantizedSwitchLinear) and "bias" not in l for l in layers) or not isinstance(glu.activation, SwiGLU):
        return None
    modes = {(l.mode, l.bits, l.group_size) for l in layers}
    if len(modes) != 1:
        return None
    mode, bits, gsize = modes.pop()
    if not all(supported(mode, bits, gsize, l.input_dims) for l in layers):
        return None
    return tuple((l["weight"], l["scales"], l.get("biases")) for l in layers), (mode, gsize)


class GroupedSwitchGLU:
    """A SwitchGLU whose decode steps (up to MAX_ROWS tokens) run on the grouped expert kernels."""

    def __call__(self, x, indices):
        rows = x.size // x.shape[-1]
        if rows <= MAX_ROWS and indices.size <= 512 and indices.size / self.gate_proj["weight"].shape[0] <= MAX_PER_EXPERT:
            got = _mats(self)
            if got is not None:
                (gate, up, down), (mode, gsize) = got
                k = indices.shape[-1]
                y = switch_glu(x.reshape(rows, x.shape[-1]), indices.reshape(rows, k), gate, up, down, mode, gsize)
                return y.reshape(*indices.shape, -1)
        return type(self).__mro__[2].__call__(self, x, indices)


def enable(model) -> int:
    """Give every SwitchGLU of ``model`` the grouped kernels for decode steps; returns how many."""
    if not ENABLED:
        return 0
    from mlx_lm.models.switch_layers import SwitchGLU

    n = 0
    for _, m in model.named_modules():
        if isinstance(m, SwitchGLU) and not isinstance(m, GroupedSwitchGLU) and _mats(m) is not None:
            m.__class__ = type("Grouped" + type(m).__name__, (GroupedSwitchGLU, type(m)), {})
            n += 1
    return n
