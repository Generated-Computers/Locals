"""Liquid's LFM3 (``model_type: lfm3``) for mlx-lm 0.32.

LFM2 with a third layer type. A layer mixes its tokens with either a gated
short convolution (as LFM2's, 7 wide instead of 3) or grouped-query attention
(16 query heads, 4 key/value heads, head size 128) with q/k RMS norms and a
sigmoid gate on each head's output. LFM3-1.7B has 24 layers: 13 convolutions,
5 attention layers over a sliding window of 256 tokens (RoPE on the whole
head) and 6 over the whole context (RoPE on the first half of each head's
frequencies, the rest left unrotated). Each mixer's and each MLP's output is
normed again before it joins the residual ("sandwich" norms).

The computation of Liquid's ``modeling_lfm3.py`` (transformers 5.12), with MLX
operations only, until an mlx-lm release has LFM3. Liquid's MLX builds keep
the weights under ``language_model.`` (their VL models share the layout);
``sanitize`` takes either that or the original checkpoint's names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, create_ssm_mask, scaled_dot_product_attention
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    layer_types: List[str]
    conv_L_cache: int
    sliding_window: int
    head_dim: Optional[int] = None
    norm_eps: float = 1e-5
    conv_bias: bool = False
    rope_parameters: Optional[Dict[str, Any]] = None
    use_sandwich_norm: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        for kind, p in (self.rope_parameters or {}).items():
            if p.get("rope_type", "default") != "default":
                raise ValueError(f"LFM3: RoPE type {p['rope_type']!r} on {kind} layers is not supported")


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim, hd = args.hidden_size, args.head_dim
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.scale = hd**-0.5
        self.q_proj = nn.Linear(dim, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * hd, dim, bias=False)
        self.gate_proj = nn.Linear(dim, self.n_heads, bias=False)  # one output gate per head
        self.q_norm = nn.RMSNorm(hd, eps=args.norm_eps)
        self.k_norm = nn.RMSNorm(hd, eps=args.norm_eps)
        rope = (args.rope_parameters or {}).get(args.layer_types[layer_idx], {})
        theta = float(rope.get("rope_theta", 10000.0))
        # Partial RoPE keeps the whole head's frequency progression and zeroes its tail: the first
        # factor * hd / 2 pairs (i, i + hd / 2) turn at theta ** (-2i / hd), the others not at all
        # (an infinite wavelength). mx.fast.rope's ``dims`` would instead rotate a smaller head.
        turned = int(float(rope.get("partial_rotary_factor", 1.0)) * hd) // 2
        self.head_dim = hd
        self._freqs = None
        if turned < hd // 2:
            self._freqs = mx.concatenate([theta ** (mx.arange(0, 2 * turned, 2, dtype=mx.float32) / hd),
                                          mx.full((hd // 2 - turned,), mx.inf, dtype=mx.float32)])
        self._theta = theta

    def rope(self, x: mx.array, offset=0) -> mx.array:
        if self._freqs is None:
            return mx.fast.rope(x, self.head_dim, traditional=False, base=self._theta, scale=1.0, offset=offset)
        return mx.fast.rope(x, self.head_dim, traditional=False, base=None, scale=1.0, offset=offset, freqs=self._freqs)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset)
        k = self.rope(k, offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3) * mx.sigmoid(self.gate_proj(x))[..., None]
        return self.o_proj(out.reshape(B, L, -1))


class ShortConv(nn.Module):
    """LFM2's gated short convolution: in_proj gives B, C and x; the depthwise convolution runs over
    B * x and C gates its output. The cache keeps the last width - 1 inputs of the convolution."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.width = args.conv_L_cache
        self.hidden_size = args.hidden_size
        self.conv = nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=self.width, groups=args.hidden_size,
                              bias=args.conv_bias)
        self.in_proj = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=args.conv_bias)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=args.conv_bias)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        B, C, x = mx.split(self.in_proj(x), 3, axis=-1)
        Bx = B * x
        if mask is not None:
            Bx = mx.where(mask[..., None], Bx, 0)
        keep = self.width - 1
        if cache is not None:
            state = cache[0] if cache[0] is not None else mx.zeros((Bx.shape[0], keep, self.hidden_size), dtype=Bx.dtype)
            Bx = mx.concatenate([state, Bx], axis=1)
            t = x.shape[1]
            if cache.lengths is not None:  # rows of a batch whose prompts end before the pass does
                ends = mx.clip(cache.lengths, 0, t)
                cache[0] = mx.take_along_axis(Bx, (ends[:, None] + mx.arange(keep))[..., None], axis=1)
            else:
                cache[0] = Bx[:, -keep:, :]
            cache.advance(t)
        else:
            Bx = mx.pad(Bx, [(0, 0), (keep, 0), (0, 0)])
        return self.out_proj(C * self.conv(Bx))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        kind = args.layer_types[layer_idx]
        self.is_attention_layer = kind != "conv"
        self.use_sliding = kind == "sliding_attention"
        if self.is_attention_layer:
            self.self_attn = Attention(args, layer_idx)
        else:
            self.conv = ShortConv(args)
        self.feed_forward = MLP(args.hidden_size, args.intermediate_size)
        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.sandwich = args.use_sandwich_norm
        if self.sandwich:
            self.post_operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
            self.post_ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        mixer = self.self_attn if self.is_attention_layer else self.conv
        r = mixer(self.operator_norm(x), mask, cache)
        h = x + (self.post_operator_norm(r) if self.sandwich else r)
        r = self.feed_forward(self.ffn_norm(h))
        return h + (self.post_ffn_norm(r) if self.sandwich else r)


class Lfm3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)  # (LFM2's name for the final norm)
        kinds = args.layer_types
        self.fa_idx = kinds.index("full_attention") if "full_attention" in kinds else None
        self.swa_idx = kinds.index("sliding_attention") if "sliding_attention" in kinds else None
        self.conv_idx = kinds.index("conv") if "conv" in kinds else None

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        pick = lambda i: cache[i] if i is not None else None  # noqa: E731
        fa_mask = create_attention_mask(h, pick(self.fa_idx))
        swa_mask = create_attention_mask(h, pick(self.swa_idx), window_size=self.args.sliding_window)
        conv_mask = create_ssm_mask(h, pick(self.conv_idx))
        for layer, c in zip(self.layers, cache):
            mask = conv_mask if not layer.is_attention_layer else swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, c)
        return self.embedding_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Lfm3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        out = {}
        for name, w in weights.items():
            if name.startswith("language_model."):
                name = name[len("language_model."):]
            if name.startswith(("vision_tower.", "multi_modal_projector.", "audio_")):
                continue
            if name == "lm_head.weight" and self.args.tie_word_embeddings:
                continue
            if name.endswith("conv.conv.weight") and w.ndim == 3 and w.shape[-1] > w.shape[1]:
                w = w.transpose(0, 2, 1)  # transformers' (channels, 1, width) -> MLX's (channels, width, 1)
            out[name] = w
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() if l.is_attention_layer and not l.use_sliding
                else RotatingKVCache(max_size=self.args.sliding_window) if l.is_attention_layer
                else ArraysCache(size=1)
                for l in self.layers]
