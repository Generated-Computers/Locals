"""Inkling 2's experts read once: the NVFP4 matrices of a visit's experts, each applied to every token routed to it.

MLX's ``gather_qmm`` runs one matrix-vector product a (token, expert) pair (``fp_gather_qmv``: the expert's 4.5 MB
are read again for every token that uses it), and at 4 bits a product is as fast as the memory: 8 ms a token over the
48 visits, whatever the tokens share. They share a lot: consecutive tokens of a text go to the same experts (4 tokens
use 19 of their 32, 8 use 29 of 64), and so do a prompt's (all 128 experts, 16 tokens each in a chunk of 256).

``slots`` groups a pass's (token, expert) pairs by expert, ``xmm`` reads each group's expert once: a thread decodes 8
weights of 4 output rows and multiplies them with those 8 inputs of each of the group's tokens (MLX's layout of
threads: 32 across the inputs, 4 rows a simdgroup, 2 simdgroups a threadgroup). The gate/up product applies the
softcapped SwiGLU as it writes (rows alternate gate, up: a simdgroup has two pairs of them).
"""

from typing import Optional, Tuple

import mlx.core as mx

_HEADER = """
#define UNROLL _Pragma("clang loop unroll(full)")
inline float e4m3_(uint8_t b) {  // the scale of 16 weights (MLX's fp8_e4m3; scales are positive)
    const ushort v = b & 127;
    return float(as_type<half>(ushort((v << 7) | (((v + 1) >> 7) << 14)))) * 256.0f;
}
// Two weights at once, 16 bits apart in a word (nibbles i and i + 4): each 0, 0.5, 1, 1.5, 2, 3, 4, 6 or its negative,
// over 16384 (MLX's fp4_e2m1: the three low bits as a half's lowest exponent bits and highest mantissa bit; here the
// sign bit goes to the half's too, and the caller's scale carries the 16384).
inline float2 fp4x2_(uint v) {
    return float2(as_type<half2>(((v & 0x00070007u) << 9) | ((v & 0x00080008u) << 12)));
}
template <typename T> inline void load8(const device T* p, thread float4& a, thread float4& b);
template <> inline void load8<float>(const device float* p, thread float4& a, thread float4& b) {
    a = *((const device packed_float4*)p); b = *((const device packed_float4*)(p + 4));
}
template <> inline void load8<half>(const device half* p, thread float4& a, thread float4& b) {
    a = float4(*((const device packed_half4*)p)); b = float4(*((const device packed_half4*)(p + 4)));
}
template <> inline void load8<bfloat16_t>(const device bfloat16_t* p, thread float4& a, thread float4& b) {
    const uint4 u = *((const device packed_uint4*)p);  // (a bfloat16 is the top half of a float)
    a = float4(as_type<float>(u.x << 16), as_type<float>(u.x & 0xffff0000u), as_type<float>(u.y << 16), as_type<float>(u.y & 0xffff0000u));
    b = float4(as_type<float>(u.z << 16), as_type<float>(u.z & 0xffff0000u), as_type<float>(u.w << 16), as_type<float>(u.w & 0xffff0000u));
}
inline float sigmoid_(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }
"""

# One thread, the pairs in order: how many an expert has, where its slots start (a slot: an expert and at most M of
# its pairs), then each pair into its slot. (2048 pairs of a prompt's chunk: 70 us; a step's 8: under 10.)
_SLOTS_SRC = """
    if (thread_position_in_grid.x != 0) return;
    int count[E], start[E];
    for (int e = 0; e < E; e++) count[e] = 0;
    for (int p = 0; p < P; p++) count[inds[p]]++;
    int s = 0;
    for (int e = 0; e < E; e++) {
        start[e] = s;
        for (int left = count[e]; left > 0; left -= M) { slot_e[s] = e; slot_n[s] = min(left, M); s++; }
        count[e] = 0;
    }
    for (; s < S; s++) { slot_e[s] = -1; slot_n[s] = 0; }
    for (int p = 0; p < P; p++) {
        const int e = int(inds[p]), i = count[e]++;
        slot_p[size_t(start[e] + i / M) * M + i % M] = p;
    }
"""

# A thread has 8 inputs of every 256 (one word of each row's weights) and 4 output rows; the 32 threads of a simdgroup
# cover the 256, two simdgroups a threadgroup. A block: the 8 inputs of each of the slot's tokens into registers once,
# then of each row its word unpacked once (the arithmetic these products are bound by: 8 operations a weight in MLX's
# kernel, of which 6 unpack it) and a dot product a token. A slot's tokens are compiled in (M): with a slot short of
# them the last one is computed twice, which costs less than asking (11.2 against 9.7 ms for a token's 48 visits),
# and from 4 a slot the registers run out (8 tokens, 29 experts: 41 ms at 2 a slot, 46 at 4, 152 at 8; MLX's 59).
_XMM_SRC = """
    const uint s = thread_position_in_grid.y;
    const int e = slot_e[s];
    if (e < 0) return;
    const int m = slot_n[s];
    const uint lid = thread_index_in_simdgroup;
    const int row0 = int(threadgroup_position_in_grid.x * 2 + simdgroup_index_in_threadgroup) * 4;
    const size_t WROW = IN / 8, GROW = IN / 16;
    const device uint* wp = w + (size_t(e) * OUT + row0) * WROW + lid;
    const device uint8_t* sp = sc + (size_t(e) * OUT + row0) * GROW + lid / 2;
    const device T* xp[M];
    float acc[4][M];
    UNROLL for (int j = 0; j < M; j++) {
        const int p = slot_p[size_t(s) * M + min(j, m - 1)];
        xp[j] = x + size_t(XPAIR ? p : p / K) * IN + lid * 8;
        UNROLL for (int r = 0; r < 4; r++) acc[r][j] = 0.0f;
    }
    for (int k = 0; k + int(lid) * 8 < IN; k += 256) {
        float4 xa[M], xb[M];
        UNROLL for (int j = 0; j < M; j++) load8<T>(xp[j] + k, xa[j], xb[j]);
        UNROLL for (int r = 0; r < 4; r++) {
            const uint pack = wp[size_t(r) * WROW];
            const float scale = e4m3_(sp[size_t(r) * GROW]);
            const float2 w04 = fp4x2_(pack), w15 = fp4x2_(pack >> 4), w26 = fp4x2_(pack >> 8), w37 = fp4x2_(pack >> 12);
            const float4 wa = float4(w04.x, w15.x, w26.x, w37.x), wb = float4(w04.y, w15.y, w26.y, w37.y);
            UNROLL for (int j = 0; j < M; j++) acc[r][j] += scale * (dot(wa, xa[j]) + dot(wb, xb[j]));
        }
        wp += 32;
        sp += 16;
    }
    const float g = gs[e] * (16384.0f / 2688.0f);  // (the expert's own scale over the two formats' largest values)
    UNROLL for (int j = 0; j < M; j++) {
        float v[4];
        UNROLL for (int r = 0; r < 4; r++) v[r] = simd_sum(acc[r][j]) * g;
        if (lid != 0 || j >= m) continue;
        const size_t p = size_t(slot_p[size_t(s) * M + j]);
        if (ACT) {
            UNROLL for (int h = 0; h < 2; h++) {
                const float gt = float(T(v[2 * h])), up = float(T(v[2 * h + 1]));
                y[p * (OUT / 2) + row0 / 2 + h] =
                    T(metal::precise::tanh(gt / prm[0]) * prm[0] * sigmoid_(gt) * metal::precise::tanh(up / prm[1]) * prm[1]);
            }
        } else {
            UNROLL for (int r = 0; r < 4; r++) y[p * OUT + row0 + r] = T(v[r]);
        }
    }
"""

_kernels: dict = {}


def _kernel(name, inputs, outputs, source):
    if name not in _kernels:
        _kernels[name] = mx.fast.metal_kernel(name="locals_inkling_" + name, input_names=inputs, output_names=outputs,
                                              source=source, header=_HEADER)
    return _kernels[name]


def slots(inds: mx.array, experts: int, most: int):
    """The pairs of ``inds`` (tokens, k), a token's experts, grouped by expert: of each slot its expert (-1: unused),
    how many pairs it has (at most ``most``) and which. An expert with more has several slots."""
    P = inds.size
    S = min(P, min(experts, P) + P // most)
    return _kernel("slots", ["inds"], ["slot_e", "slot_n", "slot_p"], _SLOTS_SRC)(
        inputs=[inds], template=[("E", experts), ("P", P), ("S", S), ("M", most)], grid=(1, 1, 1), threadgroup=(1, 1, 1),
        output_shapes=[(S,), (S,), (S, most)], output_dtypes=[mx.int32, mx.int32, mx.int32])


def xmm(lin, x: mx.array, table, pairs: int, k: int, per_pair: bool, caps: Optional[Tuple[float, float]] = None) -> mx.array:
    """``lin`` (a quantized SwitchLinear in NVFP4 with a scale an expert) of the slots' pairs: (pairs, out), or with
    ``caps`` the softcapped SwiGLU of its interleaved gate and up rows, (pairs, out / 2). ``x``: a row a token
    (tokens, in), or a row a pair (``per_pair``)."""
    slot_e, slot_n, slot_p = table
    E, OUT, wrow = lin.weight.shape
    IN, M = wrow * 8, slot_p.shape[1]
    act = caps is not None
    return _kernel("xmm", ["w", "sc", "gs", "x", "slot_e", "slot_n", "slot_p", "prm"], ["y"], _XMM_SRC)(
        inputs=[lin.weight, lin.scales, lin.global_scale, x, slot_e, slot_n, slot_p, mx.array(caps or (1.0, 1.0), dtype=mx.float32)],
        template=[("T", x.dtype), ("IN", IN), ("OUT", OUT), ("M", M), ("K", k), ("XPAIR", per_pair), ("ACT", act)],
        grid=(OUT // 8 * 64, slot_e.shape[0], 1), threadgroup=(64, 1, 1),
        output_shapes=[(pairs, OUT // 2 if act else OUT)], output_dtypes=[x.dtype])[0]
