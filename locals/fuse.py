"""Fuse projections in Llama/Qwen2-style mlx-lm models (and Flint's, with locals/moe.py).

q/k/v become one quantized matmul and gate/up another. Quantization groups
run along the input dimension, so concatenating along the output dimension
keeps the quantized format exact. Fewer, larger matmuls mean fewer kernel
launches per step (less CPU encode time) and better GPU occupancy for the
small k/v projections.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import scaled_dot_product_attention


def _fusable(mods) -> bool:
    """Projections that concatenate into one quantized matmul: the same bits, group size and mode
    (mlx-lm's mixed recipes give v_proj 6 bits next to 4-bit q/k), and a bias on all or none."""
    return _same_quant(mods) and len({"bias" in m for m in mods}) == 1


def _concat_quantized(mods) -> nn.QuantizedLinear:
    """One QuantizedLinear for ``mods`` stacked along the output (see _fusable). Built without
    QuantizedLinear's constructor, which quantizes random weights: that took from the global
    random state, and seeded sampling then differed from mlx-lm's."""
    if not _fusable(mods):
        raise ValueError("projections with different quantization or biases do not fuse")
    first = mods[0]
    q = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(q)
    q.group_size, q.bits, q.mode = first.group_size, first.bits, getattr(first, "mode", "affine")
    q.weight = mx.concatenate([m["weight"] for m in mods], axis=0)
    q.scales = mx.concatenate([m["scales"] for m in mods], axis=0)
    if "biases" in first:  # affine quantization; mxfp4 and the like have none
        q.biases = mx.concatenate([m["biases"] for m in mods], axis=0)
    if "bias" in first:
        q.bias = mx.concatenate([m["bias"] for m in mods], axis=0)
    q.freeze()
    return q


class FusedAttention(nn.Module):
    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        B, L, _ = x.shape
        qkv = self.qkv_proj(x).reshape(B, L, self.n_heads + 2 * self.n_kv_heads, self.head_dim)
        qkv = qkv.transpose(0, 2, 1, 3)
        h, kv = self.n_heads, self.n_kv_heads
        queries, keys, values = qkv[:, :h], qkv[:, h : h + kv], qkv[:, h + kv :]
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        out = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class FusedFlintAttention(nn.Module):
    """Flint's attention (locals.models.flint or mlx-lm's) with q/k/v as one matmul: per-head q/k norms, RoPE on the
    sliding-window layers only, a learned sink per head. ``_mix`` is everything between the projections, so that
    locals/mixed.py can run them once over decode rows and a prompt chunk together."""

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        return self.o_proj(self._mix(self.qkv_proj(x), mask, cache))

    def _mix(self, qkv: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        from . import qkv as qkv_cache
        from .qk_kernel import fits, qk_norm_rope

        B, L, _ = qkv.shape
        nq, nk = self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
        if fits(self, qkv):  # norms and RoPE in one kernel
            q, k, v = qk_norm_rope(qkv, self, cache.offset if cache is not None else 0)
            if qkv_cache.quantized(cache):  # 8-bit keys and values (locals/qkv.py)
                out = qkv_cache.attention(q, k, v, cache, self.scale, mask, self.softmax_offset)
                return out.transpose(0, 2, 1, 3).reshape(B, L, -1)
            if cache is not None:
                k, v = cache.update_and_fetch(k, v)
            if L == 1 and isinstance(mask, mx.array) and _unpadded(cache):
                mask = None  # one query a row and no row padded: the mask is all true, and costs MLX's attention 40%
            W = getattr(cache, "max_size", None)  # a sliding window's ring (mlx-lm's RotatingKVCache)
            if FLINT_BAND and L >= 2 * FLINT_BAND and isinstance(mask, mx.array) and isinstance(W, int):
                out = self._banded(q, k, v, mask, cache, W)
            elif getattr(cache, "bits", None) is None:  # (mlx-lm's quantized caches take its own path)
                out = attend_few(q, k, v, self.scale, mask, self.softmax_offset)
            else:
                out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask, sinks=self.softmax_offset)
            return out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        q = self.q_norm(qkv[..., :nq].reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(qkv[..., nq : nq + nk].reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = mx.contiguous(qkv[..., nq + nk :].reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3))  # (see qk_kernel)
        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
        if qkv_cache.quantized(cache):
            out = qkv_cache.attention(q, k, v, cache, self.scale, mask, self.softmax_offset)
            return out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask, sinks=self.softmax_offset)
        return out.transpose(0, 2, 1, 3).reshape(B, L, -1)


    def _banded(self, q, k, v, mask, cache, W: int) -> mx.array:
        """A prompt chunk's attention in a sliding-window layer, FLINT_BAND queries at a time against the keys their
        window reaches (slices of the keys and of the mask): the same scores, without the 2,048 x 2,560 of a chunk
        under a 513-token window, most of them masked. 8 to 10% of a Flint prompt's reading."""
        L = q.shape[2]
        P = k.shape[2] - L  # the ring's keys before the chunk's
        outs = []
        for a in range(0, L, FLINT_BAND):
            b = min(L, a + FLINT_BAND)
            lo, hi = max(0, P + a - (W - 1)), P + b
            outs.append(scaled_dot_product_attention(q[:, :, a:b], k[:, :, lo:hi], v[:, :, lo:hi], cache=cache, scale=self.scale,
                                                     mask=mask[..., a:b, lo:hi], sinks=self.softmax_offset))
        return mx.concatenate(outs, axis=2)


FEW_PAD = os.environ.get("LOCALS_FEW_QUERY_PAD", "1") != "0"


def attend_few(q: mx.array, k: mx.array, v: mx.array, scale: float, mask, sinks) -> mx.array:
    """MLX's attention, a pass of 5 to 8 queries a row padded in front to 9. MLX runs its one-query kernel up to 4
    queries and its matrix kernel from 9; from 5 to 8 it takes a path twice as slow as 9 (Flint's full layer at 8,192
    keys: 2.1 ms against 1.2; at 1,024: 306 us against 159), which a draft or a forced span of 5 to 8 tokens hit in
    all 49 layers. The causal mask aligns the last query with the last key, so the real queries keep their keys; an
    array mask gets copies of its first row in front; the padding's outputs are dropped."""
    L = q.shape[2]
    if not FEW_PAD or not 5 <= L <= 8 or k.shape[2] < 9:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask, sinks=sinks)
    pad = 9 - L
    q = mx.concatenate([mx.repeat(q[:, :, :1], pad, axis=2), q], axis=2)
    if isinstance(mask, mx.array):
        mask = mx.concatenate([mx.repeat(mask[..., :1, :], pad, axis=-2), mask], axis=-2)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask, sinks=sinks)[:, :, pad:]


def _unpadded(cache) -> bool:
    """Whether ``cache`` is a batch of full-attention rows none of which is left-padded (mlx-lm's BatchKVCache, whose
    mask then only says every key is in view: at 30k keys MLX's attention took 510 us with it, 358 without, a 15%
    slower step for one request through the engine). The padding is read on the host, where it is already
    evaluated but for the step after rows join or leave."""
    from mlx_lm.models.cache import BatchKVCache

    return type(cache) is BatchKVCache and not any(cache.left_padding.tolist())


FLINT_BAND = int(os.environ.get("LOCALS_FLINT_BAND", "512"))  # queries per band; 0: one attention over the chunk


class FusedMLP(nn.Module):
    def __call__(self, x) -> mx.array:
        gu = self.gate_up_proj(x)
        g, u = gu[..., : self.hidden], gu[..., self.hidden :]
        return self.down_proj(swiglu(g, u))


_MLP_KINDS: dict = {}  # MLP class -> whether it computes down(silu(gate(x)) * up(x)) (verified once)


def _swiglu_mlp(mlp) -> bool:
    """Whether ``mlp`` is the SwiGLU block FusedMLP computes. Its source must use
    swiglu/silu and no other activation (Gemma's MLPs use GELU), and the first
    fused instance of each class must reproduce the original numerically."""
    import inspect

    cls = type(mlp)
    if cls not in _MLP_KINDS:
        try:
            src = inspect.getsource(cls.__call__)
        except (OSError, TypeError):
            src = ""
        ok = ("swiglu(" in src or "silu(" in src) and not any(a in src for a in ("gelu", "relu", "tanh", "sigmoid("))
        if ok:  # numerical check on a copy of this layer's own weights
            dims = mlp.gate_proj["weight"].shape[1] * 32 // mlp.gate_proj.bits
            x = mx.random.normal((1, 3, dims), key=mx.random.key(0)).astype(mlp.gate_proj["scales"].dtype)  # not the global state
            ref = mlp(x).astype(mx.float32)
            hidden = mlp.gate_proj["weight"].shape[0]
            gu = _concat_quantized([mlp.gate_proj, mlp.up_proj])(x)  # what FusedMLP computes, mlp left as it is
            out = mlp.down_proj(swiglu(gu[..., :hidden], gu[..., hidden:])).astype(mx.float32)
            ok = bool(mx.max(mx.abs(out - ref)) <= 0.02 * (1 + float(mx.max(mx.abs(ref)))))
        _MLP_KINDS[cls] = ok
    return _MLP_KINDS[cls]


def _fuse_mlp(mlp) -> None:
    mlp.hidden = mlp.gate_proj["weight"].shape[0]
    mlp.gate_up_proj = _concat_quantized([mlp.gate_proj, mlp.up_proj])
    del mlp.gate_proj, mlp.up_proj
    mlp.__class__ = type("Fused" + type(mlp).__name__, (FusedMLP, type(mlp)), {})


USE_GDN_KERNEL = os.environ.get("LOCALS_GDN_KERNEL", "1") != "0"  # one fused Metal kernel per decode step (see gdn_kernel.py)
_GDN_MAX_S = 4  # longer passes (drafts) take the kernel too when per-token states are wanted; see below


def _fast_gdn(layer, dtype, S: int) -> bool:
    from .kernels import gdn_fast

    try:
        return gdn_fast(layer.num_k_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim, dtype, S)
    except Exception:  # noqa: BLE001  # (an MLX without the op, or a shape it refuses: mlx-lm's kernel)
        return False


class FusedGatedDeltaNet(nn.Module):
    """Qwen3.5 GatedDeltaNet with in_proj_{qkv,z,b,a} as one matmul, and the
    whole single-token step after it as one kernel."""

    def __call__(self, inputs, mask=None, cache=None):
        return self.out_proj(self._mix(self.in_proj_all(inputs), mask, cache))

    def _mix(self, proj, mask=None, cache=None):
        """Everything between the input projection (``proj``: in_proj_all of the inputs) and the
        output projection; locals/mixed.py runs the projections of two groups of rows together."""
        from mlx_lm.models.gated_delta import gated_delta_update

        B, S, _ = proj.shape
        if (
            USE_GDN_KERNEL
            and (S <= _GDN_MAX_S or (getattr(cache, "want_states", False) and S <= 16))
            and mask is None
            and cache is not None
            and cache[0] is not None
            and cache[1] is not None
            and getattr(cache, "lengths", None) is None
            and cache[1].dtype == mx.float32
            and cache[0].dtype == proj.dtype
            and self._gdn_ok
        ):
            from .gdn_kernel import gdn_decode

            want = bool(getattr(cache, "want_states", False)) and S > 1
            outs = gdn_decode(
                proj, cache[0], self._conv_w, self.A_log, self.dt_bias, cache[1], self.norm.weight,
                self.num_k_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim, states=want,
            )
            y = outs[0]
            if want:
                # for rewinding a partially accepted draft (engine.rewind): the
                # recurrent state after every token, and what rebuilds the conv state
                cache.token_states = outs[3]
                cache.token_conv = (cache[0], proj[..., : self.conv_dim])
            cache[0], cache[1] = outs[1], outs[2]
            cache.advance(S)
            return y
        c, v = self.conv_dim, self.value_dim
        qkv = proj[..., :c]
        z = proj[..., c : c + v].reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = proj[..., c + v : c + v + self.num_v_heads]
        a = proj[..., c + v + self.num_v_heads :]

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=proj.dtype)
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, vv = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        # Reference l2norm: x / sqrt(sum(x^2) + 1e-6). rms_norm adds its eps to
        # mean(x^2), hence 1e-6 / Dk (mlx-lm >= 0.32 does the same; 0.31 used 1e-6).
        eps = 1e-6 * inv_scale**2
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, eps)
        k = inv_scale * mx.fast.rms_norm(k, None, eps)
        if (mask is None and not self.training and S > _GDN_MAX_S and getattr(cache, "lengths", None) is None
                and _fast_gdn(self, q.dtype, S)):
            # an unpadded prompt: MLX's chunked op where it is faster on this GPU (kernels.gdn_fast)
            from mlx_lm.models.gated_delta import compute_g

            if state is None:
                state = mx.zeros((B, self.num_v_heads, self.head_v_dim, self.head_k_dim), dtype=mx.float32)
            out, state = mx.fast.gated_delta_update(q, k, vv, compute_g(self.A_log, a, self.dt_bias), mx.sigmoid(b), state, None)
        else:
            out, state = gated_delta_update(q, k, vv, a, b, self.A_log, self.dt_bias, state, mask, use_kernel=not self.training)
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        out = self.norm(out, z)
        return out.reshape(B, S, -1)


class FusedGatedAttention(nn.Module):
    """Qwen3-Next/3.5 gated attention with q/k/v as one matmul."""

    def __call__(self, x, mask=None, cache=None):
        return self.o_proj(self._mix(self.qkv_proj(x), mask, cache))

    def _mix(self, proj, mask=None, cache=None):
        """Between the q/k/v projection (``proj``) and the output projection (see FusedGatedDeltaNet._mix)."""
        from mlx_lm.models.base import scaled_dot_product_attention as sdpa

        B, L, _ = proj.shape
        h, kv, hd = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        q_out = proj[..., : 2 * h * hd]
        keys = proj[..., 2 * h * hd : 2 * h * hd + kv * hd]
        values = proj[..., 2 * h * hd + kv * hd :]
        queries, gate = mx.split(q_out.reshape(B, L, h, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, kv, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, kv, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        output = sdpa(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return output * mx.sigmoid(gate)


USE_ADD_RMS = False  # fused residual add + RMS norm (norm_kernel.py): measured no faster than MLX's two kernels, off


class FusedHybridBody(nn.Module):
    """Qwen3.5 text model body with each residual add fused into the RMS norm
    that follows it (the post-mixer norm, and the next layer's input norm)."""

    def __call__(self, inputs, cache=None, input_embeddings=None):
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        from .norm_kernel import add_rms

        if not USE_ADD_RMS:
            return self._stock_call(inputs, cache, input_embeddings=input_embeddings)
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        layers = self.layers
        n = layers[0].input_layernorm(h)
        for i, (layer, c) in enumerate(zip(layers, cache)):
            if layer.is_linear:
                r = layer.linear_attn(n, ssm_mask, c)
            else:
                r = layer.self_attn(n, fa_mask, c)
            pn = layer.post_attention_layernorm
            h, n2 = add_rms(h, r, pn.weight, pn.eps)
            nxt = layers[i + 1].input_layernorm if i + 1 < len(layers) else self.norm
            h, n = add_rms(h, layer.mlp(n2), nxt.weight, nxt.eps)
        return n


def _same_quant(mods) -> bool:
    return all(isinstance(m, nn.QuantizedLinear) for m in mods) and len({(m.bits, m.group_size, getattr(m, "mode", "affine")) for m in mods}) == 1


_PLAIN_ATTN = {"Attention"}  # Llama/Qwen2-style attention modules (no q/k norms or gates)


def fuse_model(model) -> int:
    """Fuse in place. Returns the number of layers changed."""
    model = getattr(model, "language_model", model)
    body = getattr(model, "model", None)
    if body is None or not hasattr(body, "layers"):
        return 0
    from .norm_kernel import supported as add_rms_supported

    if type(body).__name__ == "Qwen3_5TextModel" and add_rms_supported(model.args.hidden_size):
        stock = type(body)
        body.__class__ = type("FusedQwen3_5TextModel", (FusedHybridBody, stock), {"_stock_call": stock.__call__})
    n = 0
    for layer in body.layers:
        attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)  # LFM2: feed_forward
        plain = (
            attn is not None
            and type(attn).__name__ in _PLAIN_ATTN
            and type(attn).__module__.split(".")[-1] in ("qwen2", "llama")
            and all(isinstance(getattr(attn, k, None), nn.QuantizedLinear) for k in ("q_proj", "k_proj", "v_proj"))
            and _fusable([attn.q_proj, attn.k_proj, attn.v_proj])
        )
        if plain:
            attn.qkv_proj = _concat_quantized([attn.q_proj, attn.k_proj, attn.v_proj])
            attn.head_dim = attn.q_proj["weight"].shape[0] // attn.n_heads
            del attn.q_proj, attn.k_proj, attn.v_proj
            attn.__class__ = type("Fused" + type(attn).__name__, (FusedAttention, type(attn)), {})
        lin = getattr(layer, "linear_attn", None)
        if lin is not None and type(lin).__name__ == "GatedDeltaNet" and hasattr(lin, "in_proj_qkv"):
            parts = [lin.in_proj_qkv, lin.in_proj_z, lin.in_proj_b, lin.in_proj_a]
            if _fusable(parts):
                lin.in_proj_all = _concat_quantized(parts)
                del lin.in_proj_qkv, lin.in_proj_z, lin.in_proj_b, lin.in_proj_a
                from .gdn_kernel import supported as gdn_supported

                w = lin.conv1d.weight
                eps = getattr(getattr(lin, "norm", None), "eps", 1e-6)
                lin._gdn_ok = abs(float(eps) - 1e-6) < 1e-12 and w.ndim == 3 and w.shape[-1] == 1 and gdn_supported(  # the kernel's norm eps
                    lin.num_k_heads, lin.num_v_heads, lin.head_k_dim, lin.head_v_dim, w.shape[1])
                lin._conv_w = mx.contiguous(w.reshape(w.shape[0], w.shape[1]))
                lin.__class__ = type("Fused" + type(lin).__name__, (FusedGatedDeltaNet, type(lin)), {})
        if attn is not None and type(attn).__name__ == "Qwen3NextAttention":
            parts = [attn.q_proj, attn.k_proj, attn.v_proj]
            if _fusable(parts) and not any("bias" in m for m in parts):
                attn.qkv_proj = _concat_quantized(parts)
                del attn.q_proj, attn.k_proj, attn.v_proj
                attn.__class__ = type("Fused" + type(attn).__name__, (FusedGatedAttention, type(attn)), {})
        if attn is not None and type(attn).__name__ == "Attention" and type(attn).__module__.split(".")[-1] == "flint":
            parts = [attn.q_proj, attn.k_proj, attn.v_proj]
            if all(isinstance(m, nn.QuantizedLinear) for m in parts) and _fusable(parts) and not any("bias" in m for m in parts):
                attn.head_dim = attn.q_norm.weight.shape[0]
                attn.qkv_proj = _concat_quantized(parts)
                del attn.q_proj, attn.k_proj, attn.v_proj
                attn.__class__ = type("Fused" + type(attn).__name__, (FusedFlintAttention, type(attn)), {})
        if mlp is not None and type(mlp).__name__ == "SparseMoeBlock":
            from .moe import fuse_flint_moe

            fuse_flint_moe(mlp)  # Flint's: router, experts, shared expert (another model's block stays as it is)
        if (isinstance(getattr(mlp, "gate_proj", None), nn.QuantizedLinear) and isinstance(getattr(mlp, "up_proj", None), nn.QuantizedLinear)
                and _fusable([mlp.gate_proj, mlp.up_proj]) and _swiglu_mlp(mlp)):
            _fuse_mlp(mlp)
        n += 1
    if type(model).__module__.split(".")[-1] == "flint" and hasattr(model, "make_cache"):
        from .qkv import opt_in

        opt_in(model)  # full-attention keys and values in 8 bits, as Flint's config asks (locals/qkv.py)
        # no chunk's scores held (hybrid_batch.PREFILL_SCORES), and 2,048-token chunks read 5% faster than 1,024: a
        # prompt goes in 2,048-token chunks to about 30k tokens, 1,024 to about 63k
        model.prefill_scores, model.prefill_chunk_tokens = 1 << 26, 2048
    mx.eval(model.parameters())
    # the projections fused away stay in MLX's buffer cache, which keeps freed buffers for reuse: 2.5 GB of Flint's
    # (its experts' gate and up), which a 24 GB Mac then swapped for
    mx.clear_cache()
    return n

