"""Keys and values in 8 bits for full-attention layers (Flint), attended to without unpacking them.

A full-attention layer's cache grows with the context: Flint's 14 such layers keep 28 KB a token in bf16, so a
24 GB Mac holds about 35,000 tokens next to the model, and at 32,768 tokens a one-token step reads 0.9 GB of keys
and values, more than half its weights. Flint's own config asks for 8-bit keys and values (``kv_key_bits``,
``kv_value_bits``): stored as MLX's affine 8-bit groups of 64 they take 15 KB a token.

mlx-lm stores such a cache (QuantizedKVCache) but attends to it with three kernels and no attention sinks, which
Flint has on every head. Here a step reads the 8-bit groups directly: ``attend`` splits the keys into blocks, one
threadgroup a (row, key/value head, block), each simdgroup a slice of the block's keys for every query head that
shares the key/value head, so a key is read and unpacked once for all of them, keeping a running softmax for each
(flash decoding); a second kernel joins the blocks with the head's
sink. A prompt (several queries) dequantizes what it reads and goes through MLX's own attention.

``BatchQuantizedKVCache`` is mlx-lm's BatchKVCache with every buffer an (8-bit words, scales, biases) triple, for
the batch engine; ``QuantizedKVCache8`` is mlx-lm's QuantizedKVCache that merges into it.
"""

from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import QuantizedKVCache, _BaseCache, dynamic_roll

BITS, GROUP = 8, 64
BLOCK = 0  # keys a threadgroup reads in the first pass; 0: 1,024 up to 16k keys, 2,048 past it

_PASS1 = """
    // threadgroup: (block j, key/value head h, row b); its NSG simdgroups take a slice of the block's keys each, for
    // all G query heads that share h: a key and a value are read and unpacked once for the G heads. A lane holds 4 of
    // the 128 dims. Each (simdgroup, head) keeps a running softmax; the simdgroups' are joined in threadgroup memory
    // into one part a (head, block) for the second kernel (a part each would be more traffic than the keys).
    constexpr uint WPR = HD / 4, GPR = HD / 64, NSG = 8, SUB = BLOCK / NSG;
    threadgroup float tm[NSG], tl[NSG], to[NSG][HD];
    const uint j = threadgroup_position_in_grid.x, h = threadgroup_position_in_grid.y, b = threadgroup_position_in_grid.z;
    const uint s = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint H = KVH * G, n = meta[0], cap = meta[1], np = meta[2];
    const uint lo = max(j * BLOCK + s * SUB, uint(max(start[b], 0))), hi = min(j * BLOCK + (s + 1) * SUB, n);
    float q4[G][4], m[G], l[G], o[G][4];
    for (int g = 0; g < G; g++) {
        const device T* qr = q + (size_t(b) * H + h * G + g) * HD + lane * 4;
        for (int i = 0; i < 4; i++) { q4[g][i] = float(qr[i]) * scale[0]; o[g][i] = 0.0f; }
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }
    const uint grp = lane * 4 / 64;
    const size_t row0 = (size_t(b) * KVH + h) * cap;
    for (uint t = lo; t < hi; t++) {
        const size_t r = row0 + t;
        const uint kw = kq[r * WPR + lane], vw = vq[r * WPR + lane];
        const float ksc = float(ks[r * GPR + grp]), kbi = float(kb[r * GPR + grp]);
        const float vsc = float(vs[r * GPR + grp]), vbi = float(vb[r * GPR + grp]);
        float k4[4], v4[4];
        for (int i = 0; i < 4; i++) {
            k4[i] = ksc * float((kw >> (8 * i)) & 0xffu) + kbi;
            v4[i] = vsc * float((vw >> (8 * i)) & 0xffu) + vbi;
        }
        for (int g = 0; g < G; g++) {
            const float sc = simd_sum(q4[g][0] * k4[0] + q4[g][1] * k4[1] + q4[g][2] * k4[2] + q4[g][3] * k4[3]);
            const float m2 = max(m[g], sc), corr = fast::exp(m[g] - m2), p = fast::exp(sc - m2);
            l[g] = l[g] * corr + p;
            for (int i = 0; i < 4; i++) o[g][i] = o[g][i] * corr + p * v4[i];
            m[g] = m2;
        }
    }
    for (int g = 0; g < G; g++) {
        if (lane == 0) { tm[s] = m[g]; tl[s] = l[g]; }
        for (int i = 0; i < 4; i++) to[s][lane * 4 + i] = o[g][i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s == 0) {
            float M = tm[0];
            for (uint u = 1; u < NSG; u++) M = max(M, tm[u]);
            float Lsum = 0.0f, O[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint u = 0; u < NSG; u++) {
                const float w = tm[u] == -INFINITY ? 0.0f : fast::exp(tm[u] - M);
                Lsum += tl[u] * w;
                for (int i = 0; i < 4; i++) O[i] += to[u][lane * 4 + i] * w;
            }
            const size_t part = (size_t(b) * H + h * G + g) * np + j;
            if (lane == 0) { pm[part] = M; pl[part] = Lsum; }
            for (int i = 0; i < 4; i++) po[part * HD + lane * 4 + i] = O[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

# The same on the GPU's 8x8 matrix units, for 8 query heads a key/value head (Flint): the heads' queries are an 8 x HD
# matrix, so 8 keys at a time give an 8 x 8 tile of scores in HD / 8 matrix products, with no sums across lanes. A lane
# owns two elements of each 8 x 8 fragment, (row, col) and (row, col + 1) with row = (lane / 4 & 4) + lane / 2 % 4 and
# col = (lane / 4 & 2) * 2 + lane % 2 * 2 (measured on this GPU), so it unpacks its own elements of the keys and
# values straight from their 8-bit words into fragments: no threadgroup memory, every core full of simdgroups. A
# head's row of scores lives in four lanes (lane bits 0 and 3 vary), its softmax takes two shuffles, and rescaling
# its running output is a multiply in each lane. A simdgroup's (row of heads, slice of keys) is a part of its own.
_PASS1_MMA = """
    constexpr uint WPR = HD / 4, GPR = HD / 64, NT = HD / 8, NSG = 4, SUB = BLOCK / NSG;
    const uint j = threadgroup_position_in_grid.x, h = threadgroup_position_in_grid.y, b = threadgroup_position_in_grid.z;
    const uint s = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint H = KVH * 8, n = meta[0], cap = meta[1], np = meta[2];
    const uint qid = lane / 4, fr = (qid & 4) + (lane / 2) % 4, fc = (qid & 2) * 2 + (lane % 2) * 2;
    const uint lo = max(j * BLOCK + s * SUB, uint(max(start[b], 0))), hi = min(j * BLOCK + (s + 1) * SUB, n);
    // Dims are permuted (a dot product does not mind): row r of tile t of K^T is dim 16 r + t, so a lane's 16 tiles
    // read 4 whole words of a key (dims 16 fr .. 16 fr + 15, one group of 64); likewise column c of tile t of V and of
    // the output is dim 16 c + t, so a lane's values are 8 whole words of a key (dims 16 fc .. 16 fc + 31).
    simdgroup_float8x8 Qt[NT], O[NT];
    const device T* qr = q + (size_t(b) * H + h * 8 + fr) * HD;
    for (uint t = 0; t < NT; t++) {
        Qt[t].thread_elements()[0] = float(qr[16 * fc + t]) * scale[0];
        Qt[t].thread_elements()[1] = float(qr[16 * (fc + 1) + t]) * scale[0];
        O[t] = simdgroup_float8x8(0.0f);
    }
    float m = -INFINITY, l = 0.0f;  // head fr's running softmax, the same in its four lanes
    const size_t row0 = (size_t(b) * KVH + h) * cap;
    const uint kg = (16 * fr) / 64, vg = (16 * fc) / 64;
    const device uint4* k4 = reinterpret_cast<const device uint4*>(kq);  // a key row: WPR / 4 uint4s
    const device uint4* v4 = reinterpret_cast<const device uint4*>(vq);
    // the next chunk's words are loaded while this chunk is computed (software pipelining)
    uint4 na, nb, nv0, nv1;
    float nsca, nbia, nscb, nbib, nvsc, nvbi;
    // (a macro: Metal before 3.2, macOS 14's, has no lambdas)
#define LOCALS_FETCH(c0) { \
        const uint ka = (c0) + fc, kb2 = (c0) + fc + 1, kv = (c0) + fr; \
        const size_t ra = row0 + (ka < hi ? ka : lo), rb = row0 + (kb2 < hi ? kb2 : lo), rv = row0 + (kv < hi ? kv : lo); \
        na = k4[ra * (WPR / 4) + fr]; nb = k4[rb * (WPR / 4) + fr]; \
        nv0 = v4[rv * (WPR / 4) + fc]; nv1 = v4[rv * (WPR / 4) + fc + 1]; \
        nsca = float(ks[ra * GPR + kg]); nbia = float(kb[ra * GPR + kg]); \
        nscb = float(ks[rb * GPR + kg]); nbib = float(kb[rb * GPR + kg]); \
        nvsc = float(vs[rv * GPR + vg]); nvbi = float(vb[rv * GPR + vg]); \
    }
    if (lo < hi) LOCALS_FETCH(lo);
    for (uint t0 = lo; t0 < hi; t0 += 8) {
        const uint ka = t0 + fc, kb2 = t0 + fc + 1, kv = t0 + fr;
        const bool va = ka < hi, vb2 = kb2 < hi, vv = kv < hi;
        const uint wa[4] = {na.x, na.y, na.z, na.w}, wb[4] = {nb.x, nb.y, nb.z, nb.w};
        const uint wv[8] = {nv0.x, nv0.y, nv0.z, nv0.w, nv1.x, nv1.y, nv1.z, nv1.w};
        const float sca = va ? nsca : 0.0f, bia = va ? nbia : 0.0f, scb = vb2 ? nscb : 0.0f, bib = vb2 ? nbib : 0.0f;
        const float vsc = vv ? nvsc : 0.0f, vbi = vv ? nvbi : 0.0f;
        if (t0 + 8 < hi) LOCALS_FETCH(t0 + 8);
        simdgroup_float8x8 S(0.0f);
        for (uint t = 0; t < NT; t++) {  // dim 16 fr + t: word t / 4, byte t % 4
            simdgroup_float8x8 Kt;
            Kt.thread_elements()[0] = sca * float((wa[t / 4] >> (8 * (t % 4))) & 0xffu) + bia;
            Kt.thread_elements()[1] = scb * float((wb[t / 4] >> (8 * (t % 4))) & 0xffu) + bib;
            simdgroup_multiply_accumulate(S, Qt[t], Kt, S);
        }
        const float s0 = va ? S.thread_elements()[0] : -INFINITY, s1 = vb2 ? S.thread_elements()[1] : -INFINITY;
        float mx = max(s0, s1);
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        const float m2 = max(m, mx), corr = m == -INFINITY ? 0.0f : fast::exp(m - m2);
        const float p0 = va ? fast::exp(s0 - m2) : 0.0f, p1 = vb2 ? fast::exp(s1 - m2) : 0.0f;
        float sum = p0 + p1;
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 8);
        l = l * corr + sum;
        m = m2;
        simdgroup_float8x8 P;
        P.thread_elements()[0] = p0;
        P.thread_elements()[1] = p1;
        for (uint t = 0; t < NT; t++) {  // dims 16 fc + t (word t / 4) and 16 (fc + 1) + t (word 4 + t / 4), byte t % 4
            simdgroup_float8x8 Vt;
            Vt.thread_elements()[0] = vsc * float((wv[t / 4] >> (8 * (t % 4))) & 0xffu) + vbi;
            Vt.thread_elements()[1] = vsc * float((wv[4 + t / 4] >> (8 * (t % 4))) & 0xffu) + vbi;
            O[t].thread_elements()[0] *= corr;
            O[t].thread_elements()[1] *= corr;
            simdgroup_multiply_accumulate(O[t], P, Vt, O[t]);
        }
    }
    const size_t part = (size_t(b) * H + h * 8 + fr) * np + j * NSG + s;
    if ((lane & 9) == 0) { pm[part] = m; pl[part] = l; }
    for (uint t = 0; t < NT; t++) {
        po[part * HD + 16 * fc + t] = O[t].thread_elements()[0];
        po[part * HD + 16 * (fc + 1) + t] = O[t].thread_elements()[1];
    }
#undef LOCALS_FETCH
"""

_PASS2 = """
    // one threadgroup a (row, query head), its 8 simdgroups each joining every 8th part, then the 8 joined in
    // threadgroup memory with the head's sink (one simdgroup walking hundreds of parts was the slower pass)
    threadgroup float tm[8], tl[8], to[8][HD];
    const uint sg = threadgroup_position_in_grid.x, u = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint np = meta[2];
    float m = -INFINITY;
    for (uint j = u; j < np; j += 8) m = max(m, pm[size_t(sg) * np + j]);
    float l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint j = u; j < np; j += 8) {
        const size_t part = size_t(sg) * np + j;
        const float w = pm[part] == -INFINITY ? 0.0f : fast::exp(pm[part] - m);
        l += pl[part] * w;
        for (int i = 0; i < 4; i++) o[i] += po[part * HD + lane * 4 + i] * w;
    }
    if (lane == 0) { tm[u] = m; tl[u] = l; }
    for (int i = 0; i < 4; i++) to[u][lane * 4 + i] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (u != 0) return;
    const float sink = float(sinks[sg % (KVH * G)]);
    float M = sink;
    for (uint v = 0; v < 8; v++) M = max(M, tm[v]);
    float L = fast::exp(sink - M), O[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint v = 0; v < 8; v++) {
        const float w = tm[v] == -INFINITY ? 0.0f : fast::exp(tm[v] - M);
        L += tl[v] * w;
        for (int i = 0; i < 4; i++) O[i] += to[v][lane * 4 + i] * w;
    }
    for (int i = 0; i < 4; i++) out[size_t(sg) * HD + lane * 4 + i] = T(O[i] / L);
"""

_k1 = _k2 = _k1m = None


def attend(q: mx.array, keys, values, n: int, sinks: mx.array, scale: float, start: mx.array) -> mx.array:
    """One query per row against 8-bit keys and values: ``q`` (B, H, 1, HD) with HD = 128; ``keys``/``values`` a
    cache's whole buffers, (words, scales, biases) of shape (B, KVH, capacity, ...), read in place (a copy of the
    valid part would be a step's worth of traffic), of which positions ``start[b]`` to ``n`` are row b's. Softmax over
    those keys and the head's sink."""
    global _k1, _k2, _k1m
    if _k1m is None:
        _k1m = mx.fast.metal_kernel(name="locals_qkv_attend_mma", source=_PASS1_MMA, header="#include <metal_simdgroup_matrix>\n",
                                    input_names=["q", "kq", "ks", "kb", "vq", "vs", "vb", "start", "scale", "meta"],
                                    output_names=["pm", "pl", "po"])
    if _k1 is None:
        _k1 = mx.fast.metal_kernel(name="locals_qkv_attend", source=_PASS1,
                                   input_names=["q", "kq", "ks", "kb", "vq", "vs", "vb", "start", "scale", "meta"],
                                   output_names=["pm", "pl", "po"])
        _k2 = mx.fast.metal_kernel(name="locals_qkv_join", source=_PASS2, input_names=["pm", "pl", "po", "sinks", "meta"],
                                   output_names=["out"])
    B, H, _, HD = q.shape
    KVH, cap = keys[0].shape[1], keys[0].shape[2]
    G = H // KVH
    block = BLOCK if BLOCK else (1024 if n <= 16384 else 2048)  # the fastest on an M4 Pro at 8k and 32k keys
    nb = max(1, -(-n // block))
    meta = mx.array([n, cap, nb, B * H], dtype=mx.uint32)
    kq, ks, kb = keys
    vq, vs, vb = values
    mma = G == 8 and HD % 64 == 0
    kern, nsg = (_k1m, 4) if mma else (_k1, 8)
    if mma:  # one part a simdgroup
        meta = mx.array([n, cap, nb * 4, B * H], dtype=mx.uint32)
    pm, pl, po = kern(inputs=[q, kq, ks, kb, vq, vs, vb, start.astype(mx.int32), mx.array([scale], dtype=mx.float32), meta],
                     template=[("T", q.dtype), ("HD", HD), ("G", G), ("KVH", KVH), ("BLOCK", block)],
                     grid=(nb * nsg * 32, KVH, B), threadgroup=(nsg * 32, 1, 1),
                     output_shapes=[(B * H * nb * (4 if mma else 1),)] * 2 + [(B * H * nb * (4 if mma else 1) * HD,)],
                     output_dtypes=[mx.float32, mx.float32, mx.float32])
    (out,) = _k2(inputs=[pm, pl, po, sinks, meta], template=[("T", q.dtype), ("HD", HD), ("G", G), ("KVH", KVH)],
                 grid=(B * H * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(B, H, 1, HD)],
                 output_dtypes=[q.dtype])
    return out


def dequantized(t):
    return mx.dequantize(*t, group_size=GROUP, bits=BITS)


def quantized(cache) -> bool:
    return getattr(cache, "bits", None) == BITS and getattr(cache, "group_size", None) == GROUP


def attention(q, k, v, cache, scale: float, mask, sinks) -> mx.array:
    """A layer's attention through a quantized cache: stores ``k``/``v`` (B, KVH, L, HD), then one query a row
    through ``attend``, a prompt (several) through MLX's attention over the keys and values unpacked."""
    keys, values = cache.update_and_fetch(k, v)
    if q.shape[2] == 1 and q.shape[3] == 128:
        start = getattr(cache, "left_padding", None)
        if start is None:
            start = mx.zeros((q.shape[0],), dtype=mx.int32)
        n = cache._idx if hasattr(cache, "_idx") else cache.offset
        return attend(q, cache.keys, cache.values, n, sinks, scale, start)
    from .fuse import attend_few

    return attend_few(q, dequantized(keys), dequantized(values), scale, mask, sinks.astype(q.dtype))


def _zeros(B, H, n, D, dtype):
    return (mx.zeros((B, H, n, D * BITS // 32), dtype=mx.uint32), mx.zeros((B, H, n, D // GROUP), dtype=dtype),
            mx.zeros((B, H, n, D // GROUP), dtype=dtype))


def _head_dim(t):
    return t[0].shape[-1] * 32 // BITS


class QuantizedKVCache8(QuantizedKVCache):
    """mlx-lm's QuantizedKVCache in 8 bits, which a batch takes as a BatchQuantizedKVCache."""

    def __init__(self):
        super().__init__(group_size=GROUP, bits=BITS)

    @classmethod
    def merge(cls, caches):
        return BatchQuantizedKVCache.merge(caches)


class BatchQuantizedKVCache(_BaseCache):
    """mlx-lm's BatchKVCache (left-padded rows, one buffer index ``_idx``, ``offset`` each row's position) with the
    keys and values as 8-bit (words, scales, biases) triples."""

    step = 256
    bits, group_size = BITS, GROUP

    def __init__(self, left_padding: List[int]):
        self.keys = self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-p for p in left_padding])
        self._idx = 0
        self._right_padding = None

    def update_and_fetch(self, keys, values):
        prev, S = self._idx, keys.shape[2]
        if self.keys is None or prev + S > self.keys[0].shape[2]:
            B, H, _, D = keys.shape
            n = (self.step + S - 1) // self.step * self.step
            nk, nv = _zeros(B, H, n, D, keys.dtype), _zeros(B, H, n, values.shape[3], values.dtype)
            if self.keys is None:
                self.keys, self.values = nk, nv
            else:
                self.keys = tuple(mx.concatenate([a[..., :prev, :], z], axis=2) for a, z in zip(self.keys, nk))
                self.values = tuple(mx.concatenate([a[..., :prev, :], z], axis=2) for a, z in zip(self.values, nv))
        self.offset = self.offset + S
        self._idx += S
        for buf, new in ((self.keys, mx.quantize(keys, group_size=GROUP, bits=BITS)),
                         (self.values, mx.quantize(values, group_size=GROUP, bits=BITS))):
            for a, x in zip(buf, new):
                a[..., prev:self._idx, :] = x
        return self.keys_and_values()

    def keys_and_values(self):
        return (tuple(a[..., : self._idx, :] for a in self.keys), tuple(a[..., : self._idx, :] for a in self.values))

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError("left padding can only be added to an empty cache")
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset = self.offset - left_padding
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            p = self._right_padding
            self.keys = tuple(dynamic_roll(a, p[:, None], axis=2) for a in self.keys)
            self.values = tuple(dynamic_roll(a, p[:, None], axis=2) for a in self.values)
            self.offset = self.offset - p
            self.left_padding += p
            self._right_padding = None

    @property
    def state(self):
        return self.keys, self.values, self.offset, self.left_padding, self._idx

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding, self._idx = v

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset = self.offset - n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(N, offset=self._idx, left_padding=self.left_padding, **kwargs)

    def filter(self, batch_indices):
        if self.keys is not None:
            self.keys = tuple(a[batch_indices] for a in self.keys)
            self.values = tuple(a[batch_indices] for a in self.values)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        drop = min(self.left_padding.tolist())
        if drop > 0:
            if self.keys is not None:
                self.keys = tuple(a[..., drop:, :] for a in self.keys)
                self.values = tuple(a[..., drop:, :] for a in self.values)
            self._idx -= drop
            self.left_padding -= drop

    def extend(self, other):
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return
        ref = self if self.keys is not None else other
        H, D, M, dtype = ref.keys[0].shape[1], _head_dim(ref.keys), _head_dim(ref.values), ref.keys[1].dtype
        max_idx = max(self._idx, other._idx)
        size = max(c.keys[0].shape[2] for c in (self, other) if c.keys is not None)

        def pad(c):
            k = c.keys if c.keys is not None else _zeros(c.offset.shape[0], H, 0, D, dtype)
            v = c.values if c.values is not None else _zeros(c.offset.shape[0], H, 0, M, dtype)
            left = max_idx - c._idx
            right = size - k[0].shape[2] - left
            if right < 0:
                k, v = tuple(a[..., :right, :] for a in k), tuple(a[..., :right, :] for a in v)
                right = 0
            if left or right:
                k = tuple(mx.pad(a, [(0, 0), (0, 0), (left, right), (0, 0)]) for a in k)
                v = tuple(mx.pad(a, [(0, 0), (0, 0), (left, right), (0, 0)]) for a in v)
            return k, v, c.offset, c.left_padding + left

        (k1, v1, o1, p1), (k2, v2, o2, p2) = pad(self), pad(other)
        self.keys = tuple(mx.concatenate([a, b]) for a, b in zip(k1, k2))
        self.values = tuple(mx.concatenate([a, b]) for a, b in zip(v1, v2))
        self.offset, self.left_padding = mx.concatenate([o1, o2]), mx.concatenate([p1, p2])
        self._idx = max_idx

    def extract(self, idx):
        c = QuantizedKVCache8()
        mx.eval(self.left_padding)
        p = self.left_padding.tolist()[idx]
        c.keys = tuple(mx.contiguous(a[idx: idx + 1, :, p: self._idx]) for a in self.keys)
        c.values = tuple(mx.contiguous(a[idx: idx + 1, :, p: self._idx]) for a in self.values)
        c.offset = c.keys[0].shape[2]
        return c

    @classmethod
    def merge(cls, caches):
        lengths = [c.offset if c.keys is not None else 0 for c in caches]
        L = max(lengths)
        out = cls([L - n for n in lengths])
        if L == 0:
            return out
        ref = next(c for c in caches if c.keys is not None)
        H, D, M, dtype = ref.keys[0].shape[1], _head_dim(ref.keys), _head_dim(ref.values), ref.keys[1].dtype
        keys, values = _zeros(len(caches), H, L, D, dtype), _zeros(len(caches), H, L, M, dtype)
        for i, (n, c) in enumerate(zip(lengths, caches)):
            if n:
                for a, x in zip(keys, c.keys):
                    a[i: i + 1, :, L - n:] = x[..., :n, :]
                for a, x in zip(values, c.values):
                    a[i: i + 1, :, L - n:] = x[..., :n, :]
        out.keys, out.values, out._idx = keys, values, L
        out.offset = out.offset + L
        return out

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return sum(a.nbytes for a in self.keys + self.values)


def wants_8bit(model) -> bool:
    """Whether a Flint model keeps its full-attention keys and values in 8 bits: by default, as its config asks
    (``kv_key_bits``/``kv_value_bits``, which mlx-lm's loader does not keep); LOCALS_KV_BITS=16 keeps bf16."""
    import os

    return os.environ.get("LOCALS_KV_BITS", "16") == "8"


def opt_in(model) -> Optional[bool]:
    """Full-attention layers of a Flint model in QuantizedKVCache8 (its make_cache replaced); True if done."""
    if not wants_8bit(model):
        return False
    from mlx_lm.models.cache import RotatingKVCache

    window = model.args.sliding_window
    layers = model.layers
    model.make_cache = lambda: [RotatingKVCache(max_size=window) if l.use_sliding else QuantizedKVCache8() for l in layers]
    return True
