"""Per-head q/k RMS norms and RoPE in one kernel (Flint's attention).

After the fused q/k/v projection, a Flint layer normed q and k (two kernels)
and rotated them (two more, on the sliding-window layers): four dependent
launches of about 5 us each for a few kilobytes of data, 0.6 ms of a 10 ms
one-token step. ``qk_norm_rope`` does all four. It rounds where MLX does: the
normalized value to the activation dtype, then the weight's product
(mx.fast.rms_norm, matched exactly), then the rotation (mx.fast.rope with
traditional=False: the same in 99% of values, a last-bit difference in the
rest from the angle's cosine). Negative positions, a batch's left padding,
which the mask hides, are rotated differently from MLX.
"""

from __future__ import annotations

import os

import mlx.core as mx

USE_QK_KERNEL = os.environ.get("LOCALS_QK_KERNEL", "1") != "0"

# one simdgroup per (row, position, head) of q and k; lane j holds elements 2j, 2j+1 (and 64 + 2j, ... for a larger
# head) of the first half and their RoPE partners in the second (with HD = 128: four elements a lane)
_SRC = """
    constexpr int HALF = HD / 2, PER = HALF / 64;  // pairs of elements a lane holds in each half
    const uint sgid = thread_position_in_grid.x / 32;  // (a partial last threadgroup is smaller: not its size times its index)
    const uint lane = thread_index_in_simdgroup;
    const uint B = dims[0], L = dims[1];  // runtime: a kernel for each prompt chunk's length was a compile each
    const uint head = sgid % (H + 2 * KV), bl = sgid / (H + 2 * KV);
    if (bl >= B * L) return;
    const uint b = bl / L, l = bl % L;
    const device T* src = qkv + size_t(bl) * (H + 2 * KV) * HD + head * HD;
    if (head >= H + KV) {  // v: copied to its own (B, KV, L, HD) array
        device T* vd = v + ((size_t(b) * KV + (head - H - KV)) * L + l) * HD;
        for (uint i = lane; i < HD; i += 32) vd[i] = src[i];
        return;
    }
    const bool is_q = head < H;
    const device T* wn = is_q ? qw : kw;
    float lo[PER * 2], hi[PER * 2];
    float ss = 0.0f;
    for (int p = 0; p < PER; p++) {
        for (int u = 0; u < 2; u++) {
            const uint i = (p * 32 + lane) * 2 + u;
            lo[p * 2 + u] = float(src[i]);
            hi[p * 2 + u] = float(src[HALF + i]);
            ss += lo[p * 2 + u] * lo[p * 2 + u] + hi[p * 2 + u] * hi[p * 2 + u];
        }
    }
    const float r = metal::rsqrt(simd_sum(ss) / float(HD) + eps[0]);
    const float pos = float(offsets[OFFSETS_PER_ROW ? b : 0] + int(l));
    device T* dst = is_q ? q + ((size_t(b) * H + head) * L + l) * HD : k + ((size_t(b) * KV + (head - H)) * L + l) * HD;
    for (int p = 0; p < PER; p++) {
        for (int u = 0; u < 2; u++) {
            const uint i = (p * 32 + lane) * 2 + u;
            const float a = float(T(float(wn[i]) * float(T(lo[p * 2 + u] * r))));
            const float c = float(T(float(wn[HALF + i]) * float(T(hi[p * 2 + u] * r))));
            if (ROPE) {
                const float theta = pos * metal::exp2(-float(2 * i) / float(HD) * log2_base[0]);
                const float cs = metal::precise::cos(theta), sn = metal::precise::sin(theta);
                dst[i] = T(a * cs - c * sn);
                dst[HALF + i] = T(a * sn + c * cs);
            } else {
                dst[i] = T(a);
                dst[HALF + i] = T(c);
            }
        }
    }
"""

_kernel = None


def fits(attn, qkv: mx.array) -> bool:
    """Whether qk_norm_rope computes this layer: a 16-bit activation, head size a multiple of 128, the same norm eps
    for q and k, and RoPE (where the layer has it) rotating whole heads by halves at scale 1."""
    hd = attn.head_dim
    rope = attn.rope
    return (USE_QK_KERNEL and qkv.dtype in (mx.bfloat16, mx.float16) and hd % 128 == 0
            and attn.q_norm.weight.dtype == qkv.dtype and attn.k_norm.weight.dtype == qkv.dtype
            and abs(attn.q_norm.eps - attn.k_norm.eps) < 1e-12
            and (rope is None or (not rope.traditional and rope.dims == hd and getattr(rope, "scale", 1.0) == 1.0)))


def qk_norm_rope(qkv: mx.array, attn, offset):
    """(q, k, v) of ``qkv`` (B, L, (H + 2 KV) * HD): q and k normed and, where ``attn.rope``, rotated to positions
    ``offset`` + 0..L-1 (an int, or one per row as cache.offset of a batch), q (B, H, L, HD), k and v (B, KV, L, HD).
    v is its own array, not a view of ``qkv``: mlx-lm's ring keeps a prompt's first chunk as it is handed it, and
    a view kept the whole q/k/v projection of every sliding layer alive (0.7 GB of Flint's for a 2k chunk)."""
    import math

    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(name="locals_qk_norm_rope",
                                       input_names=["qkv", "qw", "kw", "offsets", "eps", "log2_base", "dims"],
                                       output_names=["q", "k", "v"], source=_SRC)
    B, L, _ = qkv.shape
    H, KV, HD = attn.n_heads, attn.n_kv_heads, attn.head_dim
    per_row = isinstance(offset, mx.array) and offset.size > 1
    offsets = offset.astype(mx.int32).reshape(-1) if isinstance(offset, mx.array) else mx.array([int(offset)], dtype=mx.int32)
    base = attn.rope.base if attn.rope is not None else 10000.0
    n_sg = B * L * (H + 2 * KV)
    q, k, v = _kernel(inputs=[qkv, attn.q_norm.weight, attn.k_norm.weight, offsets, mx.array([attn.q_norm.eps], dtype=mx.float32),
                           mx.array([math.log2(base)], dtype=mx.float32), mx.array([B, L], dtype=mx.uint32)],
                   template=[("T", qkv.dtype), ("H", H), ("KV", KV), ("HD", HD),
                             ("ROPE", int(attn.rope is not None)), ("OFFSETS_PER_ROW", int(per_row))],
                   grid=(n_sg * 32, 1, 1), threadgroup=(128, 1, 1),
                   output_shapes=[(B, H, L, HD), (B, KV, L, HD), (B, KV, L, HD)], output_dtypes=[qkv.dtype] * 3)
    return q, k, v
