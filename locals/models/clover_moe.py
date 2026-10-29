"""DiffuSpace's Clover (``model_type: clover_moe``), a block-diffusion language model, for mlx-lm 0.32.

Clover is Qwen3.6-35B-A3B's network (40 layers, three Gated DeltaNet layers to each full-attention
layer; 256 routed experts, 8 a token chosen by softmax, and one shared expert) trained to fill a block
of 32 masked tokens a few positions at a time. It runs two ways:

- commit (the default): causal, Qwen3.5-MoE's forward; it extends the caches. The prompt goes in this
  way, and each block once every position of it is resolved.
- preview (``preview=True``): a block, the mask token where a position is not resolved yet, reads the
  cached prefix and sees itself both ways; the caches are left as they were. Its GDN layers add a right
  convolution (the next three positions of the block) to the causal one and replace the recurrence with
  a softmax over the block's keys plus one bucket for the prefix (the clean state's readout), attention
  has no causal mask, and each MoE layer routes among the experts that carry 90% of the block's routing
  mass (the block coreset).

The logits at a block position are for the token at that position. ``BlockModel`` runs both ways and
``locals/blockdiff.py`` decodes with it (``locals serve --blocks``), written with MLX operations only:
the computation of DiffuSpace's Hugging Face code (modeling_clover_moe.py), checked against a NumPy
port of it in tests/test_clover.py.

``Model``, what ``mlx_lm.load`` builds, is the commit pass alone: an autoregressive model, a token a
step, and on a Mac the faster way (a pass over one token reads the weights once; a preview of a block
is compute-bound and resolves one to three tokens). It is mlx-lm's Qwen3.5-MoE text model, so every
engine serves it as one, with Locals's Qwen3.5 paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, create_ssm_mask, scaled_dot_product_attention
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class BlockArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    layer_types: List[str]
    rms_norm_eps: float
    vocab_size: int
    mask_token_id: int
    block_size: int = 32
    max_position_embeddings: int = 131072
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10000000.0
    rope_parameters: Optional[Dict[str, Any]] = None
    num_shared_experts: int = 1
    moe_block_coreset_enabled: bool = True
    moe_block_coreset_top_p: float = 0.9
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    denoising_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.rope_parameters:
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
            self.partial_rotary_factor = self.rope_parameters.get("partial_rotary_factor", self.partial_rotary_factor)


class GatedRMSNorm(nn.Module):
    """RMS norm (weight as stored, not 1 + w) times SiLU of the gate, in float32, rounded once."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(dims)

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        y = mx.fast.rms_norm(x.astype(mx.float32), self.weight.astype(mx.float32), self.eps)
        return (y * nn.silu(gate.astype(mx.float32))).astype(x.dtype)


class CloverGatedDeltaNet(nn.Module):
    def __init__(self, args: BlockArgs):
        super().__init__()
        self.num_k_heads, self.num_v_heads = args.linear_num_key_heads, args.linear_num_value_heads
        self.head_k_dim, self.head_v_dim = args.linear_key_head_dim, args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.block_size = args.block_size
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.conv_kernel_size, groups=self.conv_dim, bias=False)
        # (channels, 1, kernel - 1): weight d - 1 multiplies the projection d positions to the right
        self.conv_right_weight = mx.zeros((self.conv_dim, 1, self.conv_kernel_size - 1))
        self.in_proj_qkv = nn.Linear(args.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(args.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.A_log = mx.zeros(self.num_v_heads)
        self.dt_bias = mx.ones(self.num_v_heads)
        self.bd_softmax_logit_scale = mx.full((self.num_v_heads,), 10.0)
        self.bd_softmax_history_bias = mx.full((self.num_v_heads,), 5.0)
        self.bd_softmax_history_c = mx.ones(self.num_v_heads)
        self.norm = GatedRMSNorm(self.head_v_dim, args.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, args.hidden_size, bias=False)

    def _project(self, x: mx.array):
        return self.in_proj_qkv(x), self.in_proj_z(x), self.in_proj_b(x), self.in_proj_a(x)

    def __call__(self, x: mx.array, mask=None, cache=None, preview: bool = False) -> mx.array:
        B, S, _ = x.shape
        qkv, z, b, a = self._project(x)
        z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
        conv_state = cache[0] if cache is not None and cache[0] is not None else None
        if conv_state is None:
            conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
        if preview:
            mixed = self._preview_conv(conv_state, qkv)
        else:
            if mask is not None:
                qkv = mx.where(mask[..., None], qkv, 0)
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            if cache is not None:
                cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1):, :])
            mixed = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(mixed, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        state = cache[1] if cache is not None else None
        if preview:
            if state is None:
                state = mx.zeros((B, self.num_v_heads, self.head_v_dim, self.head_k_dim), dtype=mx.float32)
            out = self._preview_memory(q, k, v, a, state)
        else:
            q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)
            before = state
            out, state = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, state, mask, use_kernel=not self.training)
            if cache is not None:
                if getattr(cache, "save_replay", False):  # what rewind needs to drop this pass's last tokens
                    cache.replay = (conv_state, before, qkv, q, k, v, a, b)
                cache[1] = state
                cache.advance(S)
        return self.out_proj(self.norm(out, z).reshape(B, S, -1))

    def rewind(self, cache, drop: int) -> None:
        """Drop the last ``drop`` tokens of the causal pass made with ``cache.save_replay``: the conv state from the
        projections kept, the recurrent state by the recurrence over the tokens kept, from the state before the pass."""
        conv, before, qkv, q, k, v, a, b = cache.replay
        keep = qkv.shape[1] - drop
        cache[0] = mx.contiguous(mx.concatenate([conv, qkv[:, :keep]], axis=1)[:, -(self.conv_kernel_size - 1):])
        if keep > 0:
            _, cache[1] = gated_delta_update(q[:, :keep], k[:, :keep], v[:, :keep], a[:, :keep], b[:, :keep], self.A_log,
                                             self.dt_bias, before, None, use_kernel=not self.training)
        else:
            cache[1] = before

    def _preview_conv(self, history: mx.array, qkv: mx.array) -> mx.array:
        """The causal convolution over the committed history and the block, plus the right convolution
        over the block's next positions, then SiLU; float32, rounded once."""
        L, K = qkv.shape[1], self.conv_kernel_size
        joined = mx.concatenate([history, qkv], axis=1).astype(mx.float32)
        w = self.conv1d.weight[..., 0].astype(mx.float32)  # (C, K): tap K - 1 is the position itself
        acc = joined[:, :L, :] * w[:, 0]
        for t in range(1, K):
            acc = acc + joined[:, t : t + L, :] * w[:, t]
        right = self.conv_right_weight[:, 0, :].astype(mx.float32)
        cur = qkv.astype(mx.float32)
        for d in range(1, min(K, L)):
            acc = acc + mx.pad(cur[:, d:, :] * right[:, d - 1], [(0, 0), (0, d), (0, 0)])
        return nn.silu(acc).astype(qkv.dtype)

    def _preview_memory(self, q, k, v, a, state) -> mx.array:
        """The block's softmax over its own (l2-normalized) keys and one bucket for the prefix, whose value
        is the clean state's readout of the (scaled) query decayed by this position's gate. float32."""
        B, L = q.shape[:2]
        r = self.num_v_heads // self.num_k_heads
        qf, kf = mx.repeat(q, r, axis=2).astype(mx.float32), mx.repeat(k, r, axis=2).astype(mx.float32)
        qf = qf * mx.rsqrt(mx.sum(qf * qf, axis=-1, keepdims=True) + 1e-6)
        kf = kf * mx.rsqrt(mx.sum(kf * kf, axis=-1, keepdims=True) + 1e-6)
        qh, kh = qf.transpose(0, 2, 1, 3), kf.transpose(0, 2, 1, 3)  # (B, Hv, L, Dk)
        vh = v.astype(mx.float32).transpose(0, 2, 1, 3)  # (B, Hv, L, Dv)
        g = -mx.exp(self.A_log.astype(mx.float32)) * nn.softplus(a.astype(mx.float32) + self.dt_bias.astype(mx.float32))
        history = ((qh * self.head_k_dim**-0.5) @ state.astype(mx.float32).swapaxes(-1, -2)) * mx.exp(g).transpose(0, 2, 1)[..., None]
        logits = (qh @ kh.swapaxes(-1, -2)) * self.bd_softmax_logit_scale.astype(mx.float32)[:, None, None]
        bucket = self.bd_softmax_history_bias.astype(mx.float32) + self.bd_softmax_history_c.astype(mx.float32) * math.log(L / self.block_size)
        bucket = mx.broadcast_to(bucket[:, None, None], (B, self.num_v_heads, L, 1))
        w = mx.softmax(mx.concatenate([logits, bucket], axis=-1), axis=-1)
        out = w[..., L:] * history + w[..., :L] @ vh
        return out.transpose(0, 2, 1, 3).astype(q.dtype)


class CloverAttention(nn.Module):
    def __init__(self, args: BlockArgs):
        super().__init__()
        self.num_attention_heads, self.num_key_value_heads = args.num_attention_heads, args.num_key_value_heads
        self.head_dim = hd = args.head_dim
        self.scale = hd**-0.5
        self.q_proj = nn.Linear(args.hidden_size, self.num_attention_heads * hd * 2, bias=args.attention_bias)  # q and its gate
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * hd, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * hd, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * hd, args.hidden_size, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(hd, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(hd, eps=args.rms_norm_eps)
        self.rope = nn.RoPE(int(hd * args.partial_rotary_factor), traditional=False, base=args.rope_theta)

    def _project(self, x: mx.array):
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def __call__(self, x: mx.array, mask=None, cache=None, preview: bool = False) -> mx.array:
        B, L, _ = x.shape
        qg, k, v = self._project(x)
        q, gate = mx.split(qg.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
        k = k.reshape(B, L, self.num_key_value_heads, -1)
        v = v.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
            if preview:
                cache.offset = offset  # written where the block's commit will write; not part of the prefix
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=None if preview else mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out * mx.sigmoid(gate.reshape(B, L, -1)))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def block_coreset(logits: mx.array, k: int, top_p: float) -> mx.array:
    """Clover's block coreset: ``logits`` (B, L, E, float32) with -inf for every expert outside the fewest
    that carry ``top_p`` of the block's routing mass (each row's top-``k`` softmax scores, summed)."""
    B, L, E = logits.shape
    ids = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(mx.softmax(logits, axis=-1), ids, axis=-1)
    totals = mx.zeros((B, E), dtype=mx.float32).at[mx.arange(B)[:, None, None], ids].add(scores)
    p = totals / mx.maximum(totals.sum(axis=-1, keepdims=True), 1e-9)
    order = mx.argsort(-p, axis=-1)
    sp = mx.take_along_axis(p, order, axis=-1)
    keep = (mx.cumsum(sp, axis=-1) - sp) < top_p
    keep = mx.concatenate([mx.ones((B, 1), dtype=mx.bool_), keep[:, 1:]], axis=-1)
    allowed = mx.put_along_axis(mx.zeros((B, E), dtype=mx.bool_), order, keep, axis=-1)
    return mx.where(allowed[:, None, :], logits, -mx.inf)


class SparseMoe(nn.Module):
    def __init__(self, args: BlockArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.coreset_top_p = args.moe_block_coreset_top_p if args.moe_block_coreset_enabled else None
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(self, x: mx.array, preview: bool = False) -> mx.array:
        logits = self.gate(x).astype(mx.float32)
        if preview and self.coreset_top_p is not None:
            logits = block_coreset(logits, self.top_k, self.coreset_top_p)
        inds = mx.argpartition(-logits, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        scores = mx.softmax(mx.take_along_axis(logits, inds, axis=-1), axis=-1)  # the top k of the softmax, renormalized
        y = (self.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2).astype(x.dtype)
        return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: BlockArgs, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        if self.is_linear:
            self.linear_attn = CloverGatedDeltaNet(args)
        else:
            self.self_attn = CloverAttention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = SparseMoe(args)

    def __call__(self, x: mx.array, mask=None, cache=None, preview: bool = False) -> mx.array:
        mixer = self.linear_attn if self.is_linear else self.self_attn
        h = x + mixer(self.input_layernorm(x), mask, cache, preview)
        return h + self.mlp(self.post_attention_layernorm(h), preview)


class CloverModel(nn.Module):
    def __init__(self, args: BlockArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx = args.layer_types.index("full_attention")
        self.ssm_idx = args.layer_types.index("linear_attention")

    def __call__(self, inputs: mx.array, cache=None, preview: bool = False) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        fa_mask = None if preview else create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = None if preview else create_ssm_mask(h, cache[self.ssm_idx])
        for layer, c in zip(self.layers, cache):
            h = layer(h, ssm_mask if layer.is_linear else fa_mask, c, preview)
        return self.norm(h)


class BlockModel(nn.Module):
    def __init__(self, args: BlockArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = CloverModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, preview: bool = False, rows: Optional[mx.array] = None) -> mx.array:
        """Logits at each position of ``inputs`` (or at ``rows`` of them), for the token at that position."""
        h = self.model(inputs, cache, preview)
        if rows is not None:
            h = h[:, rows]
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers

    def rewind(self, cache, drop: int) -> None:
        """Undo the last ``drop`` tokens of the last causal pass (made with ``save_replay`` set on the GDN caches)."""
        if drop <= 0:
            return
        for layer, c in zip(self.layers, cache):
            if layer.is_linear:
                layer.linear_attn.rewind(c, drop)
            else:
                c.trim(drop)

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights):
        return _sanitize(weights, self.args.tie_word_embeddings)

    @property
    def quant_predicate(self):
        return _quant_predicate

    @property
    def cast_predicate(self):
        return lambda path: not path.endswith("A_log")


def _sanitize(weights, tied: bool):
    """The Hugging Face checkpoint's names and layouts as this module's (a converted one passes as it is)."""
    if tied:
        weights.pop("lm_head.weight", None)
    raw = any(k.endswith("conv1d.weight") and v.shape[-1] != 1 for k, v in weights.items())
    out = {}
    for k, v in weights.items():
        if k.endswith(".mlp.experts.gate_up_proj"):
            base = k[: -len("experts.gate_up_proj")] + "switch_mlp."
            mid = v.shape[-2] // 2
            out[base + "gate_proj.weight"], out[base + "up_proj.weight"] = v[..., :mid, :], v[..., mid:, :]
            continue
        if k.endswith(".mlp.experts.down_proj"):
            out[k[: -len("experts.down_proj")] + "switch_mlp.down_proj.weight"] = v
            continue
        if k.endswith("conv1d.weight") and v.shape[-1] != 1:
            v = v.moveaxis(2, 1)
        elif raw and v.ndim == 1 and k.endswith((".input_layernorm.weight", ".post_attention_layernorm.weight",
                                                  "model.norm.weight", ".q_norm.weight", ".k_norm.weight")):
            v = v + 1.0  # zero-centred: the checkpoint stores w of (1 + w)
        out[k] = v
    return out


def _quant_predicate(path, _):
    if path.endswith(("mlp.gate", "shared_expert_gate", "in_proj_a", "in_proj_b")):
        return {"group_size": 64, "bits": 8}
    return True


# --- the causal pass alone: what mlx-lm and Locals's engines load (model_type clover_moe) ----------------------------

BLOCK_ONLY = ("conv_right_weight", "bd_softmax_logit_scale", "bd_softmax_history_bias", "bd_softmax_history_c")


@dataclass
class ModelArgs(qwen3_5.TextModelArgs):
    """Clover's config read as Qwen3.5-MoE's text model's: one full-attention layer every
    ``full_attention_interval``, as its ``layer_types`` say."""

    layer_types: Optional[List[str]] = None
    mask_token_id: Optional[int] = None
    block_size: int = 32

    def __post_init__(self):
        super().__post_init__()
        n = self.full_attention_interval
        if self.layer_types is not None and list(self.layer_types) != [
                "full_attention" if (i + 1) % n == 0 else "linear_attention" for i in range(self.num_hidden_layers)]:
            raise ValueError(f"clover_moe: layer_types are not a full-attention layer every {n}, as Qwen3.5's")


class Model(qwen3_5.TextModel):
    """Clover's causal pass as an autoregressive model: mlx-lm's Qwen3.5-MoE text model, the same parameters
    under the same names without the block-diffusion ones (BLOCK_ONLY). Locals's Qwen3.5 paths apply to it
    (fuse.py, mixed.py, the engine's drafts), and the stock engine runs stock mlx-lm. ``BlockModel`` is the
    whole model, for decoding in blocks (locals/blockdiff.py)."""

    def sanitize(self, weights):
        return {k: v for k, v in _sanitize(weights, self.args.tie_word_embeddings).items() if not k.endswith(BLOCK_ONLY)}

    @property
    def quant_predicate(self):
        return _quant_predicate
