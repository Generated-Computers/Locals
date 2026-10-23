"""Recursive's Flint (``model_type: flint``) for mlx-lm 0.32.

A 40B mixture-of-experts decoder with about 2.5B parameters active per token:
49 layers, the first two with a dense SwiGLU MLP, the other 47 with 256 routed
experts (6 per token, chosen by sigmoid scores and renormalized) plus one
shared expert. Attention is grouped-query (32 query heads, 4 key/value heads,
head size 128) with q/k RMS norms and a learned sink per head; three of every
four layers see a sliding window of 513 tokens and use RoPE, the rest see the
whole context without position encoding.

This is the reference computation, written with MLX operations only, so that
it runs on the mlx-lm Locals pins. It computes what ml-explore/mlx-lm#1998
computes; Locals's fused paths for it are in ``fuse.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    mlp_only_layers: List[int]
    layer_types: List[str]
    sliding_window: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float = 10000.0
    rope_parameters: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.rope_parameters and self.rope_parameters.get("rope_theta") is not None:
            self.rope_theta = self.rope_parameters["rope_theta"]


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
        self.q_norm = nn.RMSNorm(hd, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(hd, eps=args.rms_norm_eps)
        self.softmax_offset = mx.zeros((self.n_heads,))  # the learned sink logit of each head
        sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.rope = nn.RoPE(hd, traditional=False, base=args.rope_theta) if sliding else None

    def _attend(self, q, k, v, mask, cache):
        """Attention from projected (B, L, heads * hd) q, k, v to the (B, L, heads * hd) output, before o_proj."""
        B, L, _ = q.shape
        q = self.q_norm(q.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask, sinks=self.softmax_offset)
        return out.transpose(0, 2, 1, 3).reshape(B, L, -1)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        return self.o_proj(self._attend(self.q_proj(x), self.k_proj(x), self.v_proj(x), mask, cache))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def route(logits: mx.array, k: int):
    """The ``k`` experts of each token and their weights: top-k sigmoid scores, renormalized to sum to 1.
    Scores are computed in float32, as in training."""
    scores = mx.sigmoid(logits.astype(mx.float32))
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    return inds, weights / weights.sum(axis=-1, keepdims=True)


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(dim, args.num_experts, bias=False)  # the router, kept unquantized
        self.switch_mlp = SwitchGLU(dim, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = MLP(dim, args.shared_expert_intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        inds, weights = route(self.gate(x.astype(mx.float32)), self.top_k)
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        return y + self.shared_expert(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.use_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args.hidden_size, args.intermediate_size) if layer_idx in args.mlp_only_layers else SparseMoeBlock(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class FlintModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx = args.layer_types.index("full_attention")
        self.swa_idx = args.layer_types.index("sliding_attention")

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        swa_mask = create_attention_mask(h, cache[self.swa_idx], window_size=self.args.sliding_window)
        for layer, c in zip(self.layers, cache):
            h = layer(h, swa_mask if layer.use_sliding else fa_mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FlintModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        """Unquantized checkpoints store each expert apart; stack them for SwitchGLU."""
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        for i in range(self.args.num_hidden_layers):
            if i in self.args.mlp_only_layers:
                continue
            prefix = f"model.layers.{i}.mlp"
            for n in ("up_proj", "down_proj", "gate_proj"):
                if f"{prefix}.experts.0.{n}.weight" in weights:
                    weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(
                        [weights.pop(f"{prefix}.experts.{e}.{n}.weight") for e in range(self.args.num_experts)])
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [RotatingKVCache(max_size=self.args.sliding_window) if layer.use_sliding else KVCache()
                for layer in self.layers]

    @property
    def quant_predicate(self):
        return lambda path, _: not path.endswith("mlp.gate")
