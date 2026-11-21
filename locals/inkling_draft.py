"""Inkling 2's draft model (the release's ``mtp.safetensors``, "DSpark"): up to 7 tokens ahead in one small pass.

Three dense layers over a block of 8 positions, the token to feed next and 7 masks, each position attending to the
whole block and to a context: what the target's stream held entering visits 5, 23 and 41 and at its output, for every
token the target has read, mixed per draft layer (``fuse_weight``) and normalized. The block's states go through the
target's own output layer, and a small recurrent head adds to each position's logits what the tokens drafted before it
make likely (``markov_head``), so the 7 are drafted in order.

The computation is mlx-vlm's ``speculative/drafters/inkling2_dspark`` (Apache-2.0, adapted there from vLLM's), for one
text, with the confidence of each drafted token returned too, and with the context kept apart from a draft: the
target's passes put it into three caches of the kinds the engine's are (``absorb``), which are entries of the model's
cache after its own, so prompt caches, checkpoints, batches and rewinds carry them (the draft model keeps 6.2 of 7
tokens of a tool call that writes a file with the whole prompt in its context, 2.8 with the reply alone).
"""

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, RotatingKVCache


def _stored(cache):
    """The keys and values a cache holds (any of mlx-lm's four kinds; a window's in whatever order: the positions
    are in the keys)."""
    if hasattr(cache, "max_size"):
        return cache.keys, cache.values
    n = cache._idx if hasattr(cache, "_idx") else cache.offset
    return cache.keys[..., :n, :], cache.values[..., :n, :]


class DraftAttention(nn.Module):
    def __init__(self, cfg: dict, local: bool):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.window = cfg["sliding_window_size"] if local else None
        self.base = cfg["rope_theta"]
        h = cfg["hidden_size"]
        self.q_proj = nn.Linear(h, self.heads * self.head_dim, bias=False)
        for name in ("k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx"):
            setattr(self, name, nn.Linear(h, self.kv_heads * self.head_dim, bias=False))
        self.o_proj = nn.Linear(self.heads * self.head_dim, h, bias=False)
        for name in ("q_norm", "k_norm", "k_norm_ctx"):
            setattr(self, name, nn.RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"]))

    def _split(self, x, heads):
        return x.reshape(*x.shape[:2], heads, self.head_dim).transpose(0, 2, 1, 3)

    def rope(self, x, offset):
        return mx.fast.rope(x, self.head_dim, traditional=False, base=self.base, scale=1.0, offset=offset)

    def absorb(self, context: mx.array, cache) -> None:
        """The context's keys and values into the cache, at the positions of its tokens (a row's own count: the
        cache's offset, one per row in a batch)."""
        k = self.k_norm_ctx(self._split(self.k_proj_ctx(context), self.kv_heads))
        cache.update_and_fetch(self.rope(k, cache.offset), self._split(self.v_proj_ctx(context), self.kv_heads))

    def __call__(self, x: mx.array, cache) -> mx.array:
        """The block ``x`` over the cache's context and itself; the block's keys and values are not kept."""
        k, v = _stored(cache)
        offset = cache.offset
        q = self.rope(self.q_norm(self._split(self.q_proj(x), self.heads)), offset)
        k = mx.concatenate([k, self.rope(self.k_norm(self._split(self.k_proj(x), self.kv_heads)), offset)], axis=2)
        v = mx.concatenate([v, self._split(self.v_proj(x), self.kv_heads)], axis=2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=None)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(*x.shape[:2], -1))


class DraftMLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        h, m = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(h, m, bias=False), nn.Linear(h, m, bias=False), nn.Linear(m, h, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = self.gate_proj(x).astype(mx.float32), self.up_proj(x).astype(mx.float32)
        return self.down_proj((nn.silu(gate) * up).astype(x.dtype))


class DraftLayer(nn.Module):
    def __init__(self, cfg: dict, index: int):
        super().__init__()
        for name in ("input_layernorm", "post_attention_layernorm", "dflare_norm"):
            setattr(self, name, nn.RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"]))
        self.self_attn = DraftAttention(cfg, index in cfg["local_layer_ids"])
        self.mlp = DraftMLP(cfg)

    def __call__(self, x, residual, cache):
        # (the sums in float32, the residual stream kept in the model's type)
        summed = x if residual is None else x.astype(mx.float32) + residual.astype(mx.float32)
        residual = summed.astype(x.dtype)
        x = self.self_attn(self.input_layernorm(summed).astype(x.dtype), cache)
        summed = x.astype(mx.float32) + residual.astype(mx.float32)
        return self.mlp(self.post_attention_layernorm(summed).astype(x.dtype)), summed.astype(x.dtype)


class MarkovHead(nn.Module):
    def __init__(self, cfg: dict, vocab: int):
        super().__init__()
        rank, h = cfg["markov_rank"], cfg["hidden_size"]
        self.markov_w1 = nn.Embedding(vocab, rank)
        self.rnn_proj = nn.Linear(2 * rank + 2 * h, 3 * rank)
        self.hidden_out = nn.Linear(rank, h)
        self.dflare_norm = nn.RMSNorm(h, eps=cfg["rms_norm_eps"])

    def context(self, hidden: mx.array, f_rnn: mx.array) -> mx.array:
        rank = self.markov_w1.weight.shape[1]
        inputs = mx.concatenate([hidden, mx.broadcast_to(f_rnn, hidden.shape)], axis=-1)
        return inputs.astype(mx.float32) @ self.rnn_proj.weight[:, 2 * rank:].astype(mx.float32).T + self.rnn_proj.bias.astype(mx.float32)

    def step(self, state: mx.array, previous: mx.array, context: mx.array):
        """The state after ``previous``, and what the head adds to the hidden state of the position (rank-256 latent
        through ``hidden_out``; the caller takes it through the output layer, or through the two folded: Drafter)."""
        rank = state.shape[-1]
        inputs = mx.concatenate([state, self.markov_w1(previous)], axis=-1)
        raw = (inputs.astype(mx.float32) @ self.rnn_proj.weight[:, : 2 * rank].astype(mx.float32).T + context).astype(state.dtype)
        gate, candidate, output = mx.split(raw, 3, axis=-1)
        gate = mx.sigmoid(gate.astype(mx.float32)).astype(state.dtype)
        candidate = mx.tanh(candidate.astype(mx.float32)).astype(state.dtype)
        return gate * state + (1 - gate) * candidate, mx.tanh(output.astype(mx.float32)).astype(state.dtype)


class Drafter(nn.Module):
    def __init__(self, cfg: dict, vocab: int, unpadded: int, softcap: float = 30.0):
        super().__init__()
        self.cfg, self.unpadded, self.softcap = dict(cfg), unpadded, softcap
        self.block = cfg["block_size"]
        self.layers = [DraftLayer(cfg, i) for i in range(cfg["num_hidden_layers"])]
        self.norm = nn.RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.markov_head = MarkovHead(cfg, vocab)
        self.fuse_weight = mx.zeros((cfg["num_hidden_layers"] + 1, len(cfg["target_layer_ids"])))
        self.mask_embed = mx.zeros((cfg["hidden_size"],))
        self._fold = None

    @property
    def visits(self) -> List[int]:
        return list(self.cfg["target_layer_ids"])

    def bind(self, target) -> "Drafter":
        """The target whose embedding and output layer the draft model uses. The recurrent head's projection is
        folded into the output layer once (a tile of the vocabulary at a time): 256 numbers a token."""
        object.__setattr__(self, "_target", target)
        head = target.lm_head
        proj = self.markov_head.hidden_out.weight.astype(mx.float32)
        bias = self.markov_head.hidden_out.bias.astype(mx.float32)
        dtype = self.markov_head.hidden_out.weight.dtype
        ws, bs = [], []
        for a in range(0, head.weight.shape[0], 8192):
            rows = slice(a, a + 8192)
            w = head.weight[rows]
            if isinstance(head, nn.QuantizedLinear):
                w = mx.dequantize(w, head.scales[rows], head.biases[rows] if "biases" in head else None,
                                  group_size=head.group_size, bits=head.bits, mode=head.mode)
            w = w.astype(mx.float32)
            ws.append((w @ proj).astype(dtype)), bs.append((w @ bias).astype(dtype))
            mx.eval(ws[-1], bs[-1])
        fold = (mx.concatenate(ws), mx.concatenate(bs))
        mx.eval(fold)
        object.__setattr__(self, "_fold", fold)
        return self

    def make_cache(self) -> list:
        return [KVCache() if l.self_attn.window is None else RotatingKVCache(max_size=l.self_attn.window, keep=0) for l in self.layers]

    def mix(self, feats: mx.array, row: int, norm) -> mx.array:
        """The context a draft layer (or the head, row -1) reads: the tapped states weighted and normalized.
        ``feats``: (1, n, taps, hidden)."""
        alpha = mx.softmax(self.fuse_weight.astype(mx.float32), axis=-1).astype(feats.dtype)[row]
        f = feats.astype(mx.float32)
        mixed = mx.zeros_like(f[:, :, 0])
        for i in range(f.shape[2]):
            mixed = mixed + f[:, :, i] * alpha[i].astype(mx.float32)
        return norm(mixed).astype(feats.dtype)

    def absorb(self, feats: mx.array, cache: list) -> None:
        """What the target's pass left for its tokens, (rows, tokens, taps, hidden), into the draft caches: after
        every pass of the target, a prompt's chunk or a step, so the caches hold what the target's do."""
        for i, (layer, entry) in enumerate(zip(self.layers, cache)):
            layer.self_attn.absorb(self.mix(feats, i, layer.dflare_norm), entry)
        cache[0].ink_feats = feats  # (the last token's, for the recurrent head: locals.inkling.rewind cuts it too)

    def draft(self, anchor: int, cache: list, count: Optional[int] = None):
        """``count`` tokens after ``anchor`` (the token the target reads next; at most 7), greedy, and each one's
        probability under the draft model; None where the caches hold no pass to draft from. One text."""
        count = self.block - 1 if count is None else count
        feats = getattr(cache[0], "ink_feats", None)
        if feats is None or feats.shape[0] != 1 or feats.shape[1] < 1 or not 0 < count < self.block:
            return None
        target = self._target
        embed = target.model.language_model.embed(mx.array([[anchor]], dtype=mx.int32))
        masks = mx.broadcast_to(self.mask_embed.astype(embed.dtype), (1, self.block - 1, embed.shape[-1]))
        x, residual = mx.concatenate([embed, masks], axis=1), None  # (the whole trained block, whatever the count)
        for layer, entry in zip(self.layers, cache):
            x, residual = layer(x, residual, entry)
        x = self.norm(x.astype(mx.float32) + residual.astype(mx.float32)).astype(embed.dtype)[:, 1: count + 1]
        base = target.lm_head(x).astype(x.dtype)
        ctx = self.markov_head.context(x, self.mix(feats[:, -1:], -1, self.markov_head.dflare_norm))
        state = mx.zeros((1, self.cfg["markov_rank"]), x.dtype)
        fold_w, fold_b = self._fold
        # (the recurrent head moves a position's logits by what was drafted before it, and far: applied to the 64
        # tokens the block favours alone, the target kept 3.5 of 7 tokens where it keeps 6.2)
        prev, tokens, probs = mx.array([anchor], dtype=mx.int32), [], []
        for i in range(count):
            state, latent = self.markov_head.step(state, prev, ctx[:, i])
            logits = base[:, i] + mx.addmm(fold_b, latent, fold_w.T)
            if self.softcap:
                logits = self.softcap * mx.tanh(logits.astype(mx.float32) / self.softcap)
            logits = logits[:, : self.unpadded].astype(mx.float32)
            prev = mx.argmax(logits, axis=-1).astype(mx.int32)
            probs.append(mx.exp(mx.max(logits, axis=-1) - mx.logsumexp(logits, axis=-1)))
            tokens.append(prev)
        return mx.concatenate(tokens), mx.concatenate(probs)
