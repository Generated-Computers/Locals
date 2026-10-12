"""Fuse projections in Llama/Qwen2-style mlx-lm models.

q/k/v become one quantized matmul and gate/up another. Quantization groups
run along the input dimension, so concatenating along the output dimension
keeps the quantized format exact. Fewer, larger matmuls mean fewer kernel
launches per step (less CPU encode time) and better GPU occupancy for the
small k/v projections.
"""

from __future__ import annotations

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


USE_GDN_KERNEL = True  # one fused Metal kernel per decode step (see gdn_kernel.py)
_GDN_MAX_S = 4  # longer passes (drafts) take the kernel too when per-token states are wanted; see below


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
        if (isinstance(getattr(mlp, "gate_proj", None), nn.QuantizedLinear) and isinstance(getattr(mlp, "up_proj", None), nn.QuantizedLinear)
                and _fusable([mlp.gate_proj, mlp.up_proj]) and _swiglu_mlp(mlp)):
            _fuse_mlp(mlp)
        n += 1
    mx.eval(model.parameters())
    return n
