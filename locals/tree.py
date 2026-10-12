"""Shared-prefix (tree) KV cache and batched branch decoding.

The generation state of n samples from one prompt is a tree of KV segments: a
shared root segment (the prompt) and one leaf segment per branch. Every node
has exactly one root path (a mangrove, k = 1), so a branch's context is its
root-to-leaf path, and the root segment is shared by all leaves below it
(its leaf count is the batching factor, Thm A18.1).

Attention for a branch decomposes over the segments on its path. The root
segment is attended once for all branches together (queries of every branch
batched against one copy of the prefix K/V), the leaf segment per branch, and
the two partial softmaxes are merged exactly with log-sum-exp. The prompt K/V
is stored once instead of n times.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from .kernels import forward_last


class TreeLayerCache:
    """One layer of a two-level KV tree in a single buffer:

        [ prefix (P) | slot 0 (cap) | slot 1 (cap) | ... ]

    Branch b's context is the prefix plus the first S entries of its slot.
    All branches' queries are attended in one fused SDPA call with a mask,
    so the prefix K/V is stored once and read about once per step.
    """

    def __init__(self, prefix_keys: mx.array, prefix_values: mx.array, n_branches: int, cap: int):
        _, H, P, D = prefix_keys.shape
        self.P, self.cap, self.n = P, cap, n_branches
        pad_k = mx.zeros((1, H, n_branches * cap, D), prefix_keys.dtype)
        pad_v = mx.zeros((1, H, n_branches * cap, prefix_values.shape[3]), prefix_values.dtype)
        self.k = mx.concatenate([prefix_keys, pad_k], axis=2)
        self.v = mx.concatenate([prefix_values, pad_v], axis=2)
        self.slots = mx.arange(n_branches)
        self.S = 0
        self._mask = None

    @property
    def offset(self) -> int:
        return self.P + self.S

    def update_and_fetch(self, keys, values):
        B, H, L, D = keys.shape
        if L != 1:
            raise NotImplementedError("tree branches decode one token per step")
        if self.S >= self.cap:
            raise RuntimeError("branch capacity exceeded")
        pos = self.P + self.slots * self.cap + self.S  # (B,)
        self.k[:, :, pos, :] = keys.transpose(2, 1, 0, 3)
        self.v[:, :, pos, :] = values.transpose(2, 1, 0, 3)
        self.S += 1
        self._mask = None
        return self.k, self.v

    def make_mask(self, N, **kwargs):
        return None

    def keep(self, idx: mx.array) -> None:
        self.slots = self.slots[idx]
        self._mask = None

    def mask(self, rep: int, Hkv: int, shared: dict) -> mx.array:
        key = (self.S, id(self.slots), rep, Hkv)
        m = shared.get(key)
        if m is None:
            j = mx.arange(self.P + self.n * self.cap)[None]
            lo = (self.P + self.slots * self.cap)[:, None]
            m = (j < self.P) | ((j >= lo) & (j < lo + self.S))  # (B, Lk)
            m = mx.repeat(m, rep, axis=0)  # (B*rep, Lk), rows ordered (b, r)
            m = mx.tile(m, (Hkv, 1))[None, :, None, :]  # (1, Hkv*B*rep, 1, Lk)
            shared.clear()
            shared[key] = m
        return m

    def attend(self, q, k, v, scale):
        # Pack branches into the head axis: query head (kv, b, r) uses kv head
        # kv, so one single-query SDPA call covers every branch (GQA layout)
        # and the prefix is read once per kv head.
        B, H, L, D = q.shape
        Hkv = k.shape[1]
        rep = H // Hkv
        qq = q.reshape(B, Hkv, rep, D).transpose(1, 0, 2, 3).reshape(1, Hkv * B * rep, 1, D)
        o = mx.fast.scaled_dot_product_attention(qq, k, v, scale=scale, mask=self.mask(rep, Hkv, _MASKS))
        return o.reshape(Hkv, B, rep, D).transpose(1, 0, 2, 3).reshape(B, H, 1, D)


_MASKS: dict = {}


_patched = set()


def install_tree_attention(model) -> None:
    """Make the model's attention dispatch to ``TreeLayerCache.attend``."""
    for layer in model.model.layers:
        mod = sys.modules[type(layer.self_attn).__module__]
        for base in type(layer.self_attn).__mro__:
            m = sys.modules.get(base.__module__)
            if m is not None and hasattr(m, "scaled_dot_product_attention"):
                _patch_module(m)
        _patch_module(mod)


def _patch_module(mod) -> None:
    if mod in _patched or not hasattr(mod, "scaled_dot_product_attention"):
        return
    orig = mod.scaled_dot_product_attention

    def sdpa(queries, keys, values, cache, scale, mask, **kw):
        if isinstance(cache, TreeLayerCache):
            return cache.attend(queries, keys, values, scale)
        return orig(queries, keys, values, cache=cache, scale=scale, mask=mask, **kw)

    mod.scaled_dot_product_attention = sdpa
    _patched.add(mod)


@dataclass
class TreeResult:
    branches: List[List[int]]
    texts: List[str]
    prefill_s: float
    decode_s: float
    steps: int
    peak_gb: float
    tokens: int = 0
    finish: List[str] = field(default_factory=list)


def sample_branches(
    model,
    tokenizer,
    prompt: List[int],
    n: int,
    max_tokens: int = 128,
    temp: float = 0.8,
    seed: int = 0,
    eos_ids=None,
) -> TreeResult:
    """n samples from one prompt with the prompt KV stored once."""
    install_tree_attention(model)
    eos = set(eos_ids or tokenizer.eos_token_ids)
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    base = make_prompt_cache(model)
    if len(prompt) > 1:
        model(mx.array([prompt[:-1]]), cache=base)
    mx.eval([c.state for c in base])
    caches = [TreeLayerCache(c.keys[..., : c.offset, :], c.values[..., : c.offset, :], n, max_tokens + 1) for c in base]
    del base
    mx.eval([c.k for c in caches])
    t1 = time.perf_counter()

    key = mx.random.key(seed)
    x = mx.array([[prompt[-1]]] * n)
    alive = list(range(n))
    outs: List[List[int]] = [[] for _ in range(n)]
    finish = ["length"] * n

    def step(x, key):
        logits = forward_last(model, x, caches)
        if temp == 0:
            return mx.argmax(logits, axis=-1), key
        key, sub = mx.random.split(key)
        return mx.random.categorical(logits / temp, key=sub), key

    y, key = step(x, key)
    mx.async_eval(y)
    steps = 1
    while alive:
        nxt, key2 = step(y[:, None], key)
        mx.async_eval(nxt)
        ys = y.tolist()
        done = []
        for j, b in enumerate(alive):
            t = ys[j]
            if t in eos:
                finish[b] = "stop"
                done.append(j)
            else:
                outs[b].append(t)
                if len(outs[b]) >= max_tokens:
                    done.append(j)
        if done:
            keep = [j for j in range(len(alive)) if j not in done]
            alive = [alive[j] for j in keep]
            if not alive:
                break
            idx = mx.array(keep)
            for c in caches:
                c.keep(idx)
            # The in-flight step used the old batch; recompute for survivors.
            nxt = nxt[idx]
        y, key = nxt, key2
        steps += 1
    mx.synchronize()
    t2 = time.perf_counter()
    texts = [tokenizer.decode(o) for o in outs]
    return TreeResult(outs, texts, t1 - t0, t2 - t1, steps, mx.get_peak_memory() / 1e9, sum(map(len, outs)), finish)
