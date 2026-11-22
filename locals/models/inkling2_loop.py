"""Thinking Machines' Inkling 2 Mini (``model_type: inkling2_loop``) for mlx-lm 0.32.

A 26B mixture-of-experts decoder whose middle layers loop: 24 layers of weights, of which layers 6 to 17 run three
times, so a token passes 48 layers ("visits"), each with its own attention cache. Every layer routes a token to 8 of
128 experts (sigmoid scores, a softcapped SwiGLU); the experts are stored as the release stores them, NVFP4 (4-bit
floats, an 8-bit scale per 16 and one scale per expert) or MXFP8.

What differs from a plain transformer:

* Attention is grouped-query (24 query heads, 8 key/value heads) with L2-clipped queries and keys, RMS-normalized
  values, a learned bias by distance up to 1024 tokens, a sink logit per head and token, and a sigmoid gate on the
  output. 20 layers see the last 1024 tokens; layers 5, 11, 17 and 23 see everything, with queries sharpened past
  8192 tokens. Keys and values pass a short causal convolution (4 tokens), as do the outputs of attention and experts.
* There is no plain residual stream: every 6th layer starts a block, keeping its normalized input as a "block state".
  The inputs of attention and of the experts are attention over depth, of a learned query to the block states and
  the current state.
* The passes of the loop after the first restart from the loop's input, and read the pass before them through
  "messages" (attention over depth again), mixed into the inputs of attention and experts by low-rank gates.

This is the reference computation, written with MLX operations only, so that it runs on the mlx-lm Locals pins. It
computes what ml-explore/mlx-lm#2086 computes, under the same parameter names, so a build made with either loads in
both. Text only: the vision tower's weights are left out.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import ArraysCache, BatchRotatingKVCache, KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchLinear, _gather_sort, _scatter_unsort

ROUTER_EPS = 1e-20
# query-key pairs attention holds at once in a pass over several tokens: each is about 500 bytes here (24 heads,
# float32 scores, their bias, mask and weights), so 2^21 of them is 1 GB whatever the context's length
MAX_SCORES = 1 << 21
QUERY_BLOCK = 256  # queries of a sliding layer's long chunk scored at once, over the keys they see (Attention._attend)
EVAL_TOKENS = 128  # tokens of a pass from which each visit is evaluated on its own (TextModel.__call__)
SHUT_OUT = -1e30  # the logit of a vocabulary id that is no token
NVFP4_SCALE = 6.0 * 448.0  # the largest 4-bit value times the largest 8-bit scale


@dataclass
class TextArgs(BaseModelArgs):
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    unpadded_vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    swa_head_dim: int
    sliding_window_size: int
    local_layer_ids: List[int]
    d_rel: int
    rel_extent: int
    log_scaling_n_floor: int
    log_scaling_alpha: float
    rms_norm_eps: float
    qk_l2_clip_threshold: float
    rel_l2_clip_threshold: float
    sconv_kernel_size: int
    n_layers_per_attn_res_block: int
    n_routed_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    mlp_gate_softcap: float
    mlp_up_softcap: float
    route_scale: float
    final_logit_softcapping: float
    loop_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        loop = self.loop_config or {}
        self.loop_core = range(loop.get("core_start", 0), loop.get("core_end", 0))
        self.loop_passes = loop.get("num_passes", 1) if self.loop_config else 1


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: Dict[str, Any]
    vision_config: Optional[Dict[str, Any]] = None
    image_token_id: int = 200054
    draft_config: Optional[Dict[str, Any]] = None  # (a build with the draft model: locals.convert_inkling)
    adapter: Optional[Dict[str, Any]] = None  # (a build with a direction taken out of each layer's output: --adapter)

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config)
        self.tie_word_embeddings = False


def rms(x: mx.array, eps: float = 1e-6) -> mx.array:
    """RMS norm with no learned weight."""
    return mx.fast.rms_norm(x, None, eps)


def l2_clip(x: mx.array, scale, threshold: float) -> mx.array:
    """``x * scale`` with its L2 norm over the last axis clipped to ``threshold``; computed in float32."""
    y = x.astype(mx.float32) * scale
    norm = mx.sqrt((y * y).sum(axis=-1, keepdims=True))
    return (y * mx.minimum(threshold / (norm + 1e-6), 1.0)).astype(x.dtype)


def softcapped_swiglu(gate: mx.array, up: mx.array, gate_cap: float, up_cap: float) -> mx.array:
    """SiLU(gate) * up with both factors softcapped by tanh (the sigmoid sees the uncapped gate); in float32."""
    dtype = gate.dtype
    gate, up = gate.astype(mx.float32), up.astype(mx.float32)
    return (mx.tanh(gate / gate_cap) * gate_cap * mx.sigmoid(gate) * mx.tanh(up / up_cap) * up_cap).astype(dtype)


def depth_attention(blocks: mx.array, partial: mx.array, query: mx.array) -> mx.array:
    """Attention of each learned query over depth: the (K, B, L, D) block states and the (B, L, D) current state
    are both the keys and the values. ``query``: (Q, D). (B, L, Q, D), float32."""
    states = mx.moveaxis(mx.concatenate([blocks, partial[None]]), 0, -2)
    weights = mx.softmax((states @ query.astype(mx.float32).T) * partial.shape[-1] ** -0.5, axis=-2, precise=True)
    return weights.swapaxes(-1, -2) @ states


def conv_window(x: mx.array, state: Optional[mx.array], mask: Optional[mx.array], kernel_size: int) -> mx.array:
    """A short convolution's input, float32: the ``kernel_size - 1`` inputs before ``x`` (zeros at the start of a
    text), then ``x`` with its padding zeroed."""
    x = x.astype(mx.float32)
    if mask is not None:
        x = mx.where(mask[..., None], x, 0)
    if state is None:
        state = mx.zeros((x.shape[0], kernel_size - 1, x.shape[-1]), mx.float32)
    return mx.concatenate([state, x], axis=1)


def conv_tail(window: mx.array, lengths: Optional[mx.array], n: int) -> mx.array:
    """The last ``n`` inputs of a window, each row's own last when rows are padded on the right. A gather, which
    copies: a slice would keep the whole window alive."""
    if lengths is None:
        pos = mx.arange(window.shape[1] - n, window.shape[1])[None, :, None]
    else:
        pos = (mx.clip(lengths, 0, window.shape[1] - n)[:, None] + mx.arange(n))[..., None]
    return mx.take_along_axis(window, pos, axis=1)


class ShortConvolution(nn.Module):
    """A causal depthwise convolution over the token and the ``kernel_size - 1`` before it, plus the token itself."""

    def __init__(self, dims: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(dims, dims, kernel_size, groups=dims, bias=False)

    def __call__(self, x: mx.array, state: Optional[mx.array], mask: Optional[mx.array]):
        window = conv_window(x, state, mask, self.kernel_size)
        y = self.conv1d(window)  # in float32, as the reference: sanitize() makes the taps float32
        y = (y.astype(mx.float32) + x.astype(mx.float32)) / math.sqrt(self.kernel_size + 1)
        return y.astype(x.dtype), window


class DynamicShortConvolution(nn.Module):
    """A short convolution whose taps are learned ones plus a projection of ``source``, per head and token."""

    def __init__(self, dims: int, source_dims: int, num_heads: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.static_weight = mx.zeros((kernel_size, dims))
        self.weight_proj = nn.Linear(source_dims, num_heads * kernel_size, bias=False)

    def __call__(self, x: mx.array, source: mx.array, state: Optional[mx.array], mask: Optional[mx.array]):
        B, L, C = x.shape
        K, H = self.kernel_size, self.num_heads
        taps = self.static_weight.astype(mx.float32).T.reshape(H, -1, K)
        taps = taps + self.weight_proj(source).astype(mx.float32).reshape(B, L, H, 1, K)
        window = conv_window(x, state, mask, K)
        lags = mx.stack([window[:, K - 1 - lag : K - 1 - lag + L] for lag in range(K)], axis=-1)  # tap 0: the token
        y = (lags.reshape(B, L, H, -1, K) * taps).sum(axis=-1).reshape(B, L, C)
        y = (y + x.astype(mx.float32)) / math.sqrt(2 * K + 1)
        return y.astype(x.dtype), window


class ScaledL2Clip(nn.Module):
    def __init__(self, num_heads: int, dims: int, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.scale = mx.ones((num_heads, dims))

    def __call__(self, x: mx.array) -> mx.array:
        return l2_clip(x, self.scale, self.threshold)


class RelativeLogits(nn.Module):
    """The attention bias of each distance below ``extent``, from a head's ``d_rel`` relative state."""

    def __init__(self, d_rel: int, extent: int):
        super().__init__()
        self.proj = mx.zeros((d_rel, extent))

    def __call__(self, r: mx.array) -> mx.array:
        return r @ self.proj.astype(r.dtype)


def _positions(offset, length: int) -> mx.array:
    pos = mx.arange(length)
    return offset[:, None] + pos if isinstance(offset, mx.array) else pos + offset


def _key_positions(cache: Any, num_keys: int, num_queries: int, end) -> mx.array:
    k = mx.arange(num_keys)
    if num_queries == 1 and isinstance(cache, (RotatingKVCache, BatchRotatingKVCache)):
        k = (k - cache._idx) % num_keys  # a one-token step writes in place: the window's keys are rotated
    return (end - num_keys)[:, None] + k if isinstance(end, mx.array) else k + (end - num_keys)


def attend(q, k, v, bias, sinks, *, scale: float, window: int, q_pos, k_pos, soft=None) -> mx.array:
    """Causal attention of (B, Hq, L, D) queries to (B, Hkv, S, D) keys and values at the given positions (a key
    at a negative position is padding). ``bias``: (B, L, Hq, R), added by distance where it is below R.
    ``sinks``: (B, L, Hq), a logit of each row that takes weight and has no value. (B, L, Hq, D).

    ``soft``, where the keys' positions are consecutive: a function from the scores to each key's share before it is
    divided by the row's sum, and the sums (locals/inkling.py: the bias, the mask and the softmax in one pass over
    the scores, where they are six arrays of the scores' size here)."""
    B, Hq, L, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    R = bias.shape[-1]
    if soft is not None:
        scores = q.astype(mx.float32).reshape(B, Hkv, -1, L, D) @ k.astype(mx.float32)[:, :, None].swapaxes(-1, -2)
        shares, sums = soft(scores, bias, sinks, q_pos[..., 0], k_pos[..., 0], scale, window, v.dtype)
        out = ((shares @ v[:, :, None]).astype(mx.float32) / sums).astype(v.dtype)
        return out.reshape(B, Hq, L, D).transpose(0, 2, 1, 3)
    dist = q_pos[..., :, None] - k_pos[..., None, :]
    visible = (dist >= 0) & (k_pos[..., None, :] >= 0)
    if window:
        visible = visible & (dist < window)
    if dist.ndim == 3:
        dist, visible = dist[:, None], visible[:, None]
    by_dist = mx.take_along_axis(bias.transpose(0, 2, 1, 3), mx.broadcast_to(mx.clip(dist, 0, R - 1), (B, Hq, L, S)), axis=-1)
    by_dist = mx.where(dist < R, by_dist.astype(mx.float32), 0.0)
    mask = mx.where(visible, by_dist, -mx.inf).reshape(B, Hkv, -1, L, S)
    scores = (q.astype(mx.float32).reshape(B, Hkv, -1, L, D) @ k.astype(mx.float32)[:, :, None].swapaxes(-1, -2)) * scale
    sinks = sinks.astype(mx.float32).transpose(0, 2, 1).reshape(B, Hkv, -1, L, 1)
    weights = mx.softmax(mx.concatenate([scores + mask, sinks], axis=-1), axis=-1)[..., :S]
    out = weights.astype(v.dtype) @ v[:, :, None]
    return out.reshape(B, Hq, L, D).transpose(0, 2, 1, 3)


class Attention(nn.Module):
    """``sliding``: over the last ``sliding_window_size`` tokens, with plain short convolutions on keys and values.
    Otherwise over everything, with dynamic ones and queries scaled by the log of the position."""

    soft = None  # attention weights from scores in one kernel (set by locals/inkling.py: see attend)

    def __init__(self, args: TextArgs, sliding: bool):
        super().__init__()
        dim = args.hidden_size
        if sliding:
            self.n_heads, self.n_kv_heads, hd = args.swa_num_attention_heads, args.swa_num_key_value_heads, args.swa_head_dim
            self.window = extent = args.sliding_window_size
        else:
            self.n_heads, self.n_kv_heads, hd = args.num_attention_heads, args.num_key_value_heads, args.head_dim
            self.window, extent = 0, args.rel_extent
        self.sliding = sliding
        self.scale = hd**-0.5
        self.eps = args.rms_norm_eps
        self.qk_threshold = args.qk_l2_clip_threshold
        self.log_floor, self.log_alpha = args.log_scaling_n_floor, args.log_scaling_alpha
        self.q_proj = nn.Linear(dim, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * hd, bias=False)
        self.r_proj = nn.Linear(dim, self.n_heads * args.d_rel, bias=False)
        self.o_proj = nn.Linear(self.n_heads * hd, dim, bias=False)
        self.gate_proj = nn.Linear(dim, self.n_heads * hd, bias=False)  # the sigmoid gate on the output
        self.sink_proj = nn.Linear(dim, self.n_heads, bias=False)  # the sink logit of each head
        self.rel_logits_proj = RelativeLogits(args.d_rel, extent)
        self.q_l2_clip = ScaledL2Clip(self.n_heads, hd, args.qk_l2_clip_threshold)
        self.rel_l2_clip = ScaledL2Clip(self.n_heads, args.d_rel, args.rel_l2_clip_threshold)
        kv_dims, K = self.n_kv_heads * hd, args.sconv_kernel_size
        if sliding:
            self.k_sconv, self.v_sconv = ShortConvolution(kv_dims, K), ShortConvolution(kv_dims, K)
        else:
            self.k_sconv = DynamicShortConvolution(kv_dims, dim, self.n_kv_heads, K)
            self.v_sconv = DynamicShortConvolution(kv_dims, dim, self.n_kv_heads, K)

    def __call__(self, xq: mx.array, xk: mx.array, xv: mx.array, kv_cache: Optional[Any], convs: list,
                 conv_mask: Optional[mx.array]) -> mx.array:
        """From the inputs of queries, keys and values (B, L, dim) to the layer's attention output. ``kv_cache``:
        the visit's keys and values, or None; ``convs``: the visit's convolution windows, of which this fills
        the first two."""
        B, L, _ = xq.shape
        if self.sliding:
            k, convs[0] = self.k_sconv(self.k_proj(xk), convs[0], conv_mask)
            v, convs[1] = self.v_sconv(self.v_proj(xv), convs[1], conv_mask)
        else:
            k, convs[0] = self.k_sconv(self.k_proj(xk), xk, convs[0], conv_mask)
            v, convs[1] = self.v_sconv(self.v_proj(xv), xv, convs[1], conv_mask)
        q = self.q_l2_clip(self.q_proj(xq).reshape(B, L, self.n_heads, -1))
        k = l2_clip(k.reshape(B, L, self.n_kv_heads, -1), 1.0, self.qk_threshold)
        v = v.reshape(B, L, self.n_kv_heads, -1)
        v = rms(v.astype(mx.float32), self.eps).astype(v.dtype)
        r = self.rel_l2_clip(self.r_proj(xq).reshape(B, L, self.n_heads, -1))
        if not self.sliding:
            start = kv_cache.offset if kv_cache is not None else 0
            pos = _positions(start, L).astype(mx.float32)
            tau = 1.0 + self.log_alpha * mx.log(mx.maximum((pos + 1) / self.log_floor, 1.0))[..., None, None]
            q = (q.astype(mx.float32) * tau).astype(q.dtype)
            r = (r.astype(mx.float32) * tau).astype(r.dtype)
        bias = self.rel_logits_proj(r)
        gate, sinks = self.gate_proj(xq), self.sink_proj(xq)

        q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(k, v)
        out = self._attend(q, k, v, bias, sinks, kv_cache)
        out = out.reshape(B, L, -1).astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))
        return self.o_proj(out.astype(gate.dtype))

    def _attend(self, q, k, v, bias, sinks, cache) -> mx.array:
        B, _, L, _ = q.shape
        S = k.shape[2]
        end = cache.offset if cache is not None else L
        q_pos, k_pos = _positions(end - L, L), _key_positions(cache, S, L, end)
        if L == 1 and B == 1:  # one token of one text: MLX's fused attention, the bias as its mask
            dist = q_pos.reshape(1) - k_pos.reshape(S)
            R = bias.shape[-1]
            by_dist = mx.take(bias[0, 0], mx.clip(dist, 0, R - 1), axis=-1)
            by_dist = mx.where(dist < R, by_dist, mx.zeros_like(by_dist))
            visible = (dist >= 0) & (k_pos.reshape(S) >= 0)
            if self.window:
                visible = visible & (dist < self.window)
            mask = mx.where(visible, by_dist, -mx.inf).astype(q.dtype)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask[None, :, None, :],
                                                       sinks=sinks[0, 0].astype(q.dtype))
            return out.transpose(0, 2, 1, 3)
        step = max(1, MAX_SCORES // S)  # (a chunk of a prompt over a long context: a few queries at a time)
        soft = self.soft if L > 1 else None  # (a one-token step's window is written round and round: not in order)
        if self.window and L > QUERY_BLOCK and not isinstance(end, mx.array):
            # A sliding layer's long chunk, a block of queries at a time over the keys that block sees: the window
            # before its first query, to its last. All at once, each query was scored against every key the chunk
            # holds, the window and the chunk (1.21 ms a token in chunks of 1,024 where 256 took 0.64).
            outs = []
            for a in range(0, L, QUERY_BLOCK):
                n = min(QUERY_BLOCK, L - a)
                lo, hi = max(0, S - L + a - (self.window - 1)), S - L + a + n
                outs.append(attend(q[:, :, a : a + n], k[:, :, lo:hi], v[:, :, lo:hi], bias[:, a : a + n], sinks[:, a : a + n],
                                   scale=self.scale, window=self.window, q_pos=q_pos[a : a + n], k_pos=k_pos[lo:hi], soft=soft))
            return mx.concatenate(outs, axis=1)
        if L <= step:
            return attend(q, k, v, bias, sinks, scale=self.scale, window=self.window, q_pos=q_pos, k_pos=k_pos, soft=soft)
        return mx.concatenate([attend(q[:, :, a : a + step], k, v, bias[:, a : a + step], sinks[:, a : a + step], scale=self.scale,
                                      window=self.window, q_pos=q_pos[..., a : a + step], k_pos=k_pos, soft=soft)
                               for a in range(0, L, step)], axis=1)


class Experts(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        dims, hidden, n = args.hidden_size, args.intermediate_size, args.n_routed_experts
        self.gate_up_proj = SwitchLinear(dims, 2 * hidden, n, bias=False)  # rows alternate gate, up, as released
        self.down_proj = SwitchLinear(hidden, dims, n, bias=False)
        self.gate_softcap, self.up_softcap = args.mlp_gate_softcap, args.mlp_up_softcap

    def __call__(self, x: mx.array, inds: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        inds = mx.stop_gradient(inds)
        sort = inds.size >= 64
        idx, inv_order = inds, None
        if sort:
            x, idx, inv_order = _gather_sort(x, inds)
        gate_up = self.gate_up_proj(x, idx, sorted_indices=sort)
        h = softcapped_swiglu(gate_up[..., 0::2], gate_up[..., 1::2], self.gate_softcap, self.up_softcap)
        y = self.down_proj(h, idx, sorted_indices=sort)
        if sort:
            y = _scatter_unsort(y, inv_order, inds.shape)
        return y.squeeze(-2)


class Router(nn.Module):
    """The ``k`` experts of each token: the highest sigmoid scores plus a learned bias; the weights are the scores
    without it, normalized to sum to ``route_scale``. In float32."""

    def __init__(self, args: TextArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.route_scale = args.route_scale
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((args.n_routed_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        scores = mx.sigmoid(x.astype(mx.float32) @ self.weight.T)
        k = self.top_k
        inds = mx.argpartition(-(scores + self.e_score_correction_bias), kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(scores, inds, axis=-1)
        return inds, weights / (weights.sum(axis=-1, keepdims=True) + ROUTER_EPS) * self.route_scale


def without(y: mx.array, direction: Optional[mx.array]) -> mx.array:
    """``y`` (float32) with its part along a unit ``direction`` taken out: what a rank-1 adapter whose B is the
    direction and whose A is minus the direction times the layer's weights does to the layer's output."""
    if direction is None:
        return y
    d = direction.astype(mx.float32)
    return y - (y @ d)[..., None] * d


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.gate = Router(args)
        self.experts = Experts(args)

    def __call__(self, x: mx.array) -> mx.array:
        inds, weights = self.gate(x)
        y = (self.experts(x, inds).astype(mx.float32) * weights[..., None]).sum(axis=-2)
        # (a build with an adapter has ``ablate``: the direction the experts' output is cleared of, Model.__init__)
        return without(y, getattr(self, "ablate", None)).astype(x.dtype)


class DepthInput(nn.Module):
    """One or more learned queries for attention over depth."""

    def __init__(self, dims: int, num_queries: int = 1):
        super().__init__()
        self.query = mx.zeros((dims,) if num_queries == 1 else (num_queries, dims), dtype=mx.float32)

    def __call__(self, blocks: mx.array, partial: mx.array) -> mx.array:
        return depth_attention(blocks, partial, self.query.reshape(-1, partial.shape[-1]))


class MessageGate(nn.Module):
    """Mixes a message of the pass before into a state: forgets some of the state and adds some of the message, both
    decided, per dimension, by a low-rank projection of the two (normalized)."""

    def __init__(self, dims: int, rank: int):
        super().__init__()
        self.down_proj = nn.Linear(2 * dims, rank, bias=False)
        self.forget_gate_proj = nn.Linear(rank, dims, bias=False)
        self.message_gate_proj = nn.Linear(rank, dims, bias=False)

    def __call__(self, x: mx.array, message: mx.array) -> mx.array:
        both = mx.concatenate([rms(x), rms(message)], axis=-1)
        low = self.down_proj(both.astype(self.down_proj.weight.dtype)).astype(mx.float32)
        return (1 - mx.tanh(self.forget_gate_proj(low))) * x + mx.tanh(self.message_gate_proj(low)) * message


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        dims, K = args.hidden_size, args.sconv_kernel_size
        self.sliding = layer_idx in args.local_layer_ids
        self.is_block_boundary = layer_idx % args.n_layers_per_attn_res_block == 0
        self.self_attn = Attention(args, self.sliding)
        # the inputs of attention: one for a sliding layer; of queries, keys and values otherwise. Layer 0 has
        # only the embeddings to read
        self.attention_input = DepthInput(dims, 1 if self.sliding else 3) if layer_idx > 0 else None
        self.mlp_input = DepthInput(dims)
        self.mlp = SparseMoeBlock(args)
        if self.sliding:
            self.input_layernorm = nn.RMSNorm(dims, eps=args.rms_norm_eps)
        else:
            self.q_input_layernorm = nn.RMSNorm(dims, eps=args.rms_norm_eps)
            self.k_input_layernorm = nn.RMSNorm(dims, eps=args.rms_norm_eps)
            self.v_input_layernorm = nn.RMSNorm(dims, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(dims, eps=args.rms_norm_eps)
        self.post_attention_convolution = ShortConvolution(dims, K)
        self.post_mlp_convolution = ShortConvolution(dims, K)
        if layer_idx in args.loop_core:
            rank = args.loop_config["gate_rank"]
            self.q_message_gate = MessageGate(dims, rank)
            self.k_message_gate = MessageGate(dims, rank)
            self.v_message_gate = MessageGate(dims, rank)
            self.mlp_message_gate = MessageGate(dims, rank)

    def _attention_inputs(self, x: mx.array, inputs: Optional[mx.array], messages: Optional[mx.array]):
        """The normalized inputs of queries, keys and values: read over depth (the layer's input in layer 0),
        with the pass before mixed in."""
        dtype = x.dtype
        if self.sliding:
            xs = (x if inputs is None else inputs[..., 0, :],) * 3
            norms = (self.input_layernorm,) * 3
        else:
            xs = (x,) * 3 if inputs is None else tuple(inputs[..., i, :] for i in range(3))
            norms = (self.q_input_layernorm, self.k_input_layernorm, self.v_input_layernorm)
        if messages is None:
            if self.sliding:
                return (norms[0](xs[0]).astype(dtype),) * 3
        else:
            gates = (self.q_message_gate, self.k_message_gate, self.v_message_gate)
            xs = tuple(g(s, messages[..., i, :]) for i, (g, s) in enumerate(zip(gates, xs)))
        return tuple(n(s).astype(dtype) for n, s in zip(norms, xs))

    def __call__(self, x: mx.array, blocks: Optional[mx.array], kv: Optional[Any] = None, conv: Optional[Any] = None,
                 conv_mask: Optional[mx.array] = None, messages: Optional[mx.array] = None):
        """One visit: ``x`` (B, L, dim) and the block states so far (K, B, L, dim; None before layer 0) to the
        output and the block states after. ``kv``, ``conv``: the visit's two cache entries. ``messages``:
        (B, L, 4, dim) from the pass before, for the inputs of queries, keys, values and experts."""
        dtype = x.dtype
        convs = list(conv.cache) if conv is not None else [None] * 4
        normed = rms(x.astype(mx.float32))
        inputs = self.attention_input(blocks, normed) if self.attention_input is not None else None
        h = self.self_attn(*self._attention_inputs(x, inputs, messages), kv, convs, conv_mask)
        h, convs[2] = self.post_attention_convolution(h, convs[2], conv_mask)

        residual = h = h.astype(mx.float32)
        if self.is_block_boundary:  # a block starts: its input is kept as a block state, and not added
            blocks = normed[None] if blocks is None else mx.concatenate([blocks, normed[None]])
        else:
            residual = h = x.astype(mx.float32) + h
        h = self.mlp_input(blocks, rms(h))[..., 0, :]
        if messages is not None:
            h = self.mlp_message_gate(h, messages[..., 3, :])
        h = self.mlp(self.post_attention_layernorm(h).astype(dtype))
        h, convs[3] = self.post_mlp_convolution(h, convs[3], conv_mask)

        if conv is not None:
            n = self.post_mlp_convolution.kernel_size - 1
            for i, window in enumerate(convs):
                conv[i] = conv_tail(window, conv.lengths, n)
            conv.advance(x.shape[1])
            # a pass whose last tokens may be taken back (a draft: locals/inkling.py, rewind) keeps its inputs
            conv.ink_windows = convs if getattr(conv, "want_states", False) else None
            conv.ink_states = None
        return (residual + h.astype(mx.float32)).astype(dtype), blocks


class TextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.final_input = DepthInput(args.hidden_size)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if args.loop_config is not None:
            # of each looped layer, the queries that read the pass before: for queries, keys, values and experts
            self.message_input = DepthInput(args.hidden_size, 1)
            self.message_input.query = mx.zeros((len(args.loop_core), 4, args.hidden_size), dtype=mx.float32)

    def embed(self, inputs: mx.array) -> mx.array:
        x = self.embed_tokens(inputs)  # normalized in float32 and rounded once, as the reference
        return (rms(x.astype(mx.float32), self.embed_norm.eps) * self.embed_norm.weight.astype(mx.float32)).astype(x.dtype)

    @property
    def visits(self) -> List[DecoderLayer]:
        """The layers in the order a token passes them, the looped ones once per pass."""
        core = self.args.loop_core
        return self.layers[: core.start] + self.layers[core.start : core.stop] * self.args.loop_passes + self.layers[core.stop :]

    def __call__(self, inputs: mx.array, cache: Optional[List[Any]] = None,
                 input_embeddings: Optional[mx.array] = None) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed(inputs)
        dtype = h.dtype
        core = self.args.loop_core
        caches = iter(cache if cache is not None else [None] * (2 * len(self.visits)))
        conv_mask = cache[1].make_mask(h.shape[1]) if cache is not None else None  # (padding, in a batch of texts)

        taps = getattr(self, "taps", None)  # (the draft model reads what enters some visits: locals/inkling_draft.py)
        got, visit = {}, [0]

        def run(layers, h, blocks, bank=None):
            for i, layer in enumerate(layers):
                messages = None if bank is None else depth_attention(*bank, self.message_input.query[i])
                if taps is not None and visit[0] in taps.visits:
                    got[visit[0]] = h
                visit[0] += 1
                kv, conv = next(caches), next(caches)
                h, blocks = layer(h, blocks, kv, conv, conv_mask, messages)
                if long:  # (with what the visit left in its caches: a convolution's kept inputs hold its whole window)
                    mx.eval(h, blocks, kv.state, conv.state)
            return h, blocks

        # A prompt's chunk, a visit at a time: evaluated whole, the arrays of several visits are held at once, 1.14 GB
        # for 256 tokens and 1.97 GB for 512 next to 16 GB of weights on a 24 GB Mac; 0.50 and 0.99 GB so, for 1.5%
        # of the chunk's time
        long = cache is not None and h.shape[0] * h.shape[1] >= EVAL_TOKENS
        start = run(self.layers[: core.start], h, None)
        h, blocks = start
        for loop_pass in range(self.args.loop_passes if len(core) else 0):
            bank = None
            if loop_pass > 0:
                # what the pass before left: its block states inside the loop, and its output. Then start over
                first = (0 if start[1] is None else start[1].shape[0]) + 1
                bank = (blocks[first:], rms(h.astype(mx.float32)))
                h, blocks = start
            h, blocks = run(self.layers[core.start : core.stop], h, blocks, bank)
        h, blocks = run(self.layers[core.stop :], h, blocks)
        h = self.final_input(blocks, rms(h.astype(mx.float32)))[..., 0, :]
        out = self.norm(h).astype(dtype)
        if taps is not None:
            got[visit[0]] = out
            taps.leave([got[v] for v in taps.visits], cache, 2 * visit[0])
        return out


class Taps:
    """What a pass leaves for the draft model: the stream entering ``visits`` (the count of visits: the output)."""

    def __init__(self, visits, drafter=None):
        self.visits, self.states, self.drafter = list(visits), None, drafter

    def leave(self, states, cache, own: int) -> None:
        """A pass's states (each (rows, tokens, hidden)) into the draft model's caches, the entries after ``own``."""
        self.states = states
        if self.drafter is not None and cache is not None and len(cache) > own:
            self.drafter.absorb(mx.stack(states, axis=2), cache[own:])


class _Embedding:
    """The embedding as callers of ``model.embed_tokens`` mean it: a token's row, normalized as the model reads it."""

    def __init__(self, text: TextModel):
        self._text = text

    @property
    def weight(self) -> mx.array:
        return self._text.embed_tokens.weight

    def __call__(self, inputs: mx.array) -> mx.array:
        return self._text.embed(inputs)


class Inkling2Model(nn.Module):
    """The decoder under the name the release gives it (``model.language_model``), callable as mlx-lm's bodies are."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args.text
        self.language_model = TextModel(args.text)

    def __call__(self, inputs: mx.array, cache: Optional[List[Any]] = None,
                 input_embeddings: Optional[mx.array] = None) -> mx.array:
        return self.language_model(inputs, cache, input_embeddings)

    @property
    def layers(self) -> List[DecoderLayer]:
        return self.language_model.layers

    @property
    def embed_tokens(self) -> _Embedding:
        return _Embedding(self.language_model)

    @property
    def norm(self) -> nn.RMSNorm:
        return self.language_model.norm


_RENAMES = [  # from the release's names to these modules', applied in order
    (r"^model\.llm\.unembed\.", "lm_head."),
    (r"^model\.llm\.embed\.", "model.language_model.embed_tokens."),
    (r"^model\.llm\.", "model.language_model."),
    (r"\.attn_res\.mlp_queries\.(\d+)$", r".layers.\1.mlp_input.query"),
    (r"\.attn_res\.output_query$", ".final_input.query"),
    (r"\.attn\.wq_du\.", ".self_attn.q_proj."),
    (r"\.attn\.wk_dv\.", ".self_attn.k_proj."),
    (r"\.attn\.wv_dv\.", ".self_attn.v_proj."),
    (r"\.attn\.wr_du\.", ".self_attn.r_proj."),
    (r"\.attn\.wo_ud\.", ".self_attn.o_proj."),
    (r"\.attn\.gate_attn\.", ".self_attn.gate_proj."),
    (r"\.attn\.attn_sink_proj\.", ".self_attn.sink_proj."),
    (r"\.controllers\.attn\.", ".q_message_gate."),
    (r"\.controllers\.(k|v|mlp)\.", r".\1_message_gate."),
    (r"_message_gate\.down_weight$", "_message_gate.down_proj.weight"),
    (r"_message_gate\.forget_weight$", "_message_gate.forget_gate_proj.weight"),
    (r"_message_gate\.gain_weight$", "_message_gate.message_gate_proj.weight"),
    (r"\.attn\.", ".self_attn."),
    (r"\.attn_sconv\.", ".post_attention_convolution."),
    (r"\.mlp_sconv\.", ".post_mlp_convolution."),
    (r"sconv\.weight$", "sconv.conv1d.weight"),
    (r"convolution\.weight$", "convolution.conv1d.weight"),
    (r"sconv\.static_w$", "sconv.static_weight"),
    (r"sconv\.weight_proj$", "sconv.weight_proj.weight"),
    (r"\.mlp_norm\.", ".post_attention_layernorm."),
    (r"\.attn_norm\.", ".input_layernorm."),
    (r"\.(q|k|v)_input_norm\.", r".\1_input_layernorm."),
    (r"\.mlp\.gate\.bias$", ".mlp.gate.e_score_correction_bias"),
]
# what the reference computes in float32: the routers, the loop's gates and the short convolutions
_FLOAT32 = (".mlp.gate.weight", ".forget_gate_proj.weight", ".message_gate_proj.weight", "conv1d.weight")
_EXPERT_MODES = {(4, 16): "nvfp4", (4, 32): "mxfp4", (8, 32): "mxfp8"}


def _rename(key: str) -> str:
    if m := re.match(r"model\.llm\.attn_res\.attn_queries\.(\d+)$", key):
        return f"model.language_model.layers.{int(m.group(1)) + 1}.attention_input.query"  # layer 0 has none
    for pattern, replacement in _RENAMES:
        key = re.sub(pattern, replacement, key)
    return key


def post_logits(args: ModelArgs, logits: mx.array) -> mx.array:
    """What the model does to its output layer's logits: a tanh softcap, in float32, and the padded end of the
    vocabulary shut out (the reference cuts it off; here the width stays the embedding's, for callers that
    index by it, and the value is finite, for callers that subtract logits)."""
    cap, n = args.text.final_logit_softcapping, args.text.unpadded_vocab_size
    logits = logits.astype(mx.float32)
    logits = mx.tanh(logits / cap) * cap
    if logits.shape[-1] > n:
        logits = mx.where(mx.arange(logits.shape[-1]) < n, logits, SHUT_OUT)
    return logits


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Inkling2Model(args)
        self.lm_head = nn.Linear(args.text.hidden_size, args.text.vocab_size, bias=False)
        if args.draft_config:
            from ..inkling_draft import Drafter

            t = args.text
            self.mtp = Drafter(args.draft_config, t.vocab_size, t.unpadded_vocab_size or t.vocab_size, t.final_logit_softcapping or 0.0)
            self.model.language_model.taps = Taps(self.mtp.visits, self.mtp)
        if args.adapter:
            # A direction taken out of what each layer writes (locals.convert_inkling --adapter): attention's output
            # weights and the embedding carry it already, at their 8 bits; the experts, 4-bit as released, are
            # cleared of it as they are summed.
            for layer in self.model.language_model.layers:
                layer.mlp.ablate = mx.zeros((args.text.hidden_size,), dtype=mx.float32)

    def __call__(self, inputs: mx.array, cache: Optional[List[Any]] = None,
                 input_embeddings: Optional[mx.array] = None) -> mx.array:
        return post_logits(self.args, self.lm_head(self.model(inputs, cache, input_embeddings)))

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """The release's weights under these modules' names (a converted build passes as it is), and the experts'
        layers made the quantized kind their weights are: the release's config does not say which, as mlx-lm reads it."""
        expert = re.compile(r"^(.+\.mlp\.experts)\.(w13|w2)_weight(\w*)$")
        amax, denominator, queries, out = {}, {}, {}, {}
        for k, v in weights.items():
            if k.startswith(("model.visual.", "model.vision_tower.")):
                continue  # text only
            if k.startswith("model.llm."):
                k = _rename(k)
                if k.endswith("conv1d.weight"):
                    v = v.swapaxes(1, 2)
            if k.endswith(("cross_pass_queries", "cross_pass_kv_queries")):
                queries["kv" if "_kv_" in k else "q"] = v
            elif m := expert.match(k):
                prefix, which, suffix = m.groups()
                name = f"{prefix}.{'gate_up_proj' if which == 'w13' else 'down_proj'}"
                if suffix == "_amax":
                    amax[name] = v
                elif suffix == "_outer_scale_denominator":
                    denominator[name] = v
                elif suffix == "_scale":
                    out[f"{name}.scales"] = v
                else:
                    out[f"{name}.weight"] = v.view(mx.uint32)
            else:
                out[k] = v.astype(mx.float32) if k.endswith(_FLOAT32) else v
        for k in [k for k in out if k.startswith("adapted.")]:  # (an adapter's own weights in place of the build's)
            out[k[len("adapted."):]] = out.pop(k)
        if queries:  # the release has (query, experts) and (key, value) rows of each looped layer
            dims = queries["q"].shape[-1]
            q, kv = queries["q"].reshape(-1, 2, dims), queries["kv"].reshape(-1, 2, dims)
            out["model.language_model.message_input.query"] = mx.concatenate([q[:, :1], kv, q[:, 1:]], axis=1)
        for name in amax.keys() & denominator.keys():
            out[f"{name}.global_scale"] = amax[name].astype(mx.float32) * (NVFP4_SCALE / denominator[name].astype(mx.float32))

        quantized = []
        for path, module in self.named_modules():
            scales = out.get(f"{path}.scales")
            if not isinstance(module, SwitchLinear) or scales is None or scales.dtype != mx.uint8:
                continue
            in_dims = module.weight.shape[-1]
            group_size, bits = in_dims // scales.shape[-1], out[f"{path}.weight"].shape[-1] * 32 // in_dims
            if (bits, group_size) in _EXPERT_MODES:
                quantized.append((path, module.to_quantized(group_size, bits, mode=_EXPERT_MODES[bits, group_size],
                                                            global_scale=f"{path}.global_scale" in out)))
        if quantized:
            self.update_modules(tree_unflatten(quantized))
        return out

    @property
    def layers(self) -> List[DecoderLayer]:
        return self.model.language_model.layers

    @property
    def quant_predicate(self):
        return lambda path, _: "_message_gate." not in path and not path.endswith("mlp.gate")

    @property
    def cast_predicate(self):
        keep = (".query", ".mlp.gate.weight", ".mlp.gate.e_score_correction_bias", ".forget_gate_proj.weight",
                ".message_gate_proj.weight", "conv1d.weight", ".global_scale")
        return lambda k: not k.endswith(keep)

    def make_cache(self) -> List[Any]:
        """Two entries a visit: its keys and values (the last ``sliding_window_size`` of a sliding layer), then
        the last inputs of its four short convolutions. (ml-explore/mlx-lm#2086 pairs them in a CacheList; flat,
        every entry is of a kind Locals's batches already merge, pad and trim.)"""
        window = self.args.text.sliding_window_size
        own = [c for layer in self.model.language_model.visits
               for c in (RotatingKVCache(max_size=window) if layer.sliding else KVCache(), ArraysCache(4))]
        # (then the draft model's three, where the build has it: what its attention reads of the text so far)
        return own + (self.mtp.make_cache() if "mtp" in self and not getattr(self, "draft_off", False) else [])
