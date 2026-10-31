"""Decode rows and a prompt chunk in one forward pass (Qwen3.5, Flint).

A batch engine that prefills a new request's prompt on its own stops every running row for as long
as the prefill takes (1.5 s for 1k tokens on a 4B model on an M2 Pro, 12 s for 8k), and the GPU
does not run a second stream's work alongside (measured: two streams take as long as one after the
other). Here the running rows' next tokens are computed in the same pass as a chunk of the new
prompt: every layer's projections and MLP run once over both groups' tokens, so their weights are
read once, and only what is per sequence (the attention over each group's cache, the Gated DeltaNet
recurrence) runs per group. The rows keep decoding while the prompt is prefilled, for little more
than the prefill's own cost.

The model's layers must be locals's fused Qwen3.5 or Flint layers (locals/fuse.py: ``_mix`` between the
input and output projections); ``supported`` says whether a model's are. In Flint the MLP is a mixture of
experts: each token still reads its own experts, but the attention projections, the router and the shared
expert are read once for both groups, and the rows keep decoding while a prompt is read.
"""

from __future__ import annotations

import mlx.core as mx


def _flint(body) -> bool:
    from .fuse import FusedFlintAttention

    layers = getattr(body, "layers", None)
    stock = next((c for c in type(body).__mro__ if c.__name__ == "FlintModel"), None)  # (or locals's FusedFlintModel)
    return (stock is not None and stock.__module__.rsplit(".", 1)[-1] == "flint" and bool(layers)
            and all(isinstance(getattr(l, "self_attn", None), FusedFlintAttention) and hasattr(l, "use_sliding")
                    for l in layers))


def supported(body) -> bool:
    """Whether ``body`` (the text model's layer stack) can run mixed passes: Qwen3.5-style decoder layers
    (residual mixer then residual MLP) whose mixers are locals's fused ones, or Flint's."""
    from .fuse import FusedGatedAttention, FusedGatedDeltaNet

    if _flint(body):
        return True
    layers = getattr(body, "layers", None)
    if not layers or not hasattr(body, "embed_tokens") or not hasattr(body, "norm"):
        return False
    if getattr(body, "fa_idx", None) is None or getattr(body, "ssm_idx", None) is None:
        return False
    if type(layers[0]).__module__.rsplit(".", 1)[-1] not in ("qwen3_5", "qwen3_next"):
        return False  # the layer's own __call__ must be the plain pre-norm residual one written out below
    for layer in layers:
        if not all(hasattr(layer, a) for a in ("input_layernorm", "post_attention_layernorm", "mlp", "is_linear")):
            return False
        mixer = getattr(layer, "linear_attn" if layer.is_linear else "self_attn", None)
        if not isinstance(mixer, FusedGatedDeltaNet if layer.is_linear else FusedGatedAttention):
            return False
    return True


def forward(body, xd: mx.array, cache_d, xp: mx.array, cache_p) -> mx.array:
    """One pass over decode tokens ``xd`` (B, 1) with the batch cache ``cache_d`` and prompt tokens
    ``xp`` (1, S) with the single-sequence cache ``cache_p``. Both caches advance; returns the final
    normed hidden states of the decode rows, (B, 1, hidden), as ``body(xd, cache_d)`` would."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    if _flint(body):
        return _forward_flint(body, xd, cache_d, xp, cache_p)
    hd = body.embed_tokens(xd)
    hp = body.embed_tokens(xp)
    B = hd.shape[0]
    fa_d = create_attention_mask(hd, cache_d[body.fa_idx])
    ssm_d = create_ssm_mask(hd, cache_d[body.ssm_idx])
    fa_p = create_attention_mask(hp, cache_p[body.fa_idx])
    ssm_p = create_ssm_mask(hp, cache_p[body.ssm_idx])

    def split(t):  # (1, B + S, X) -> (B, 1, X), (1, S, X)
        return t[:, :B].reshape(B, 1, -1), t[:, B:]

    h = mx.concatenate([hd.reshape(1, B, -1), hp], axis=1)
    for layer, cd, cp in zip(body.layers, cache_d, cache_p):
        if layer.is_linear:
            mixer, proj, out, md, mp = layer.linear_attn, "in_proj_all", "out_proj", ssm_d, ssm_p
        else:
            mixer, proj, out, md, mp = layer.self_attn, "qkv_proj", "o_proj", fa_d, fa_p
        pd, pp = split(getattr(mixer, proj)(layer.input_layernorm(h)))
        yd, yp = mixer._mix(pd, md, cd), mixer._mix(pp, mp, cp)
        h = h + getattr(mixer, out)(mx.concatenate([yd.reshape(1, B, -1), yp], axis=1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return body.norm(split(h)[0])


def _forward_flint(body, xd, cache_d, xp, cache_p):
    """``forward`` for Flint: full-attention and sliding-window masks for each group, every layer's attention
    projections and MLP once over both groups' tokens."""
    from mlx_lm.models.base import create_attention_mask

    hd = body.embed_tokens(xd)
    hp = body.embed_tokens(xp)
    B, W = hd.shape[0], body.args.sliding_window
    fa_d = create_attention_mask(hd, cache_d[body.fa_idx])
    swa_d = create_attention_mask(hd, cache_d[body.swa_idx], window_size=W)
    fa_p = create_attention_mask(hp, cache_p[body.fa_idx])
    swa_p = create_attention_mask(hp, cache_p[body.swa_idx], window_size=W)

    def split(t):  # (1, B + S, X) -> (B, 1, X), (1, S, X)
        return t[:, :B].reshape(B, 1, -1), t[:, B:]

    h = mx.concatenate([hd.reshape(1, B, -1), hp], axis=1)
    for layer, cd, cp in zip(body.layers, cache_d, cache_p):
        attn = layer.self_attn
        md, mp = (swa_d, swa_p) if layer.use_sliding else (fa_d, fa_p)
        pd, pp = split(attn.qkv_proj(layer.input_layernorm(h)))
        yd, yp = attn._mix(pd, md, cd), attn._mix(pp, mp, cp)
        h = h + attn.o_proj(mx.concatenate([yd.reshape(1, B, -1), yp], axis=1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return body.norm(split(h)[0])
