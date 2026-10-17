"""Exact decisions in few-path regions.

A region of a constraint is a set of automaton states from which the
language to an exit is *finite*: an enum, a boolean, a small integer
range, a choice among tool names. Token-level constrained decoding decides
such a region one token at a time and is biased: the first token's
probability aggregates every completion that starts with it, so a family of
similar values can outweigh the single most likely value. The theory says
how to decide it exactly: enumerate the completions (the walks to the exit,
at most ``budget`` of them, which the truncation lemma certifies: a capped
search either lists them all or reports that the region is too large),
score every completion's canonical token path in one batched pass from the
same prefix, and take the sequence-level argmax. One weight stream decides
the whole region, however many tokens its value has.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache

from .kernels import packed_rows, text_model
from .regex import DEAD


class RegionTooLarge(RuntimeError):
    pass


def region_completions(dfa, q_start: int, exits: Sequence[int], budget: int = 64, max_len: int = 128) -> List[bytes]:
    """Every byte string that leads from ``q_start`` to an exit state (first
    arrival), by a depth-first search over the automaton. Raises
    ``RegionTooLarge`` past ``budget`` completions or ``max_len`` bytes (a
    cycle inside the region): the search is capped, so it is either complete
    or reports a violation."""
    table = dfa.table
    exits = set(int(e) for e in exits)
    out: List[bytes] = []
    stack: List[Tuple[int, bytes]] = [(q_start, b"")]
    while stack:
        q, w = stack.pop()
        if len(w) > max_len:
            raise RegionTooLarge("a path longer than max_len: the region is not finite")
        row = table[q, :256]
        nxt = [(b, int(row[b])) for b in range(256) if row[b] != DEAD]
        for b, nq in reversed(nxt):
            w2 = w + bytes([b])
            if nq in exits:
                out.append(w2)
                if len(out) > budget:
                    raise RegionTooLarge(f"more than {budget} completions")
            else:
                stack.append((nq, w2))
    return out


def canonical_tokens(hf_tokenizer, w: bytes) -> List[int]:
    return list(hf_tokenizer.encode(w.decode("utf-8"), add_special_tokens=False))


class RegionScorer:
    """Scores candidate token paths after a prefix in one batched pass."""

    def __init__(self, model):
        self.model = model
        self.lm = text_model(model)

    def prefill(self, prompt: List[int]):
        cache = make_prompt_cache(self.model)
        i = 0
        while i < len(prompt):
            n = min(2048, len(prompt) - i)
            self.lm.model(mx.array([prompt[i : i + n]]), cache=cache)
            mx.eval([c.state for c in cache])
            i += n
        return cache

    def _head(self, h):
        from .kernels import head_logits

        return head_logits(self.lm, h)

    def logprobs(self, prefix_cache, last_prefix_token: int, candidates: List[List[int]]) -> List[float]:
        """Sum of log p(token | prefix, earlier tokens of the candidate) over
        each candidate's tokens, from the same prefix, in one right-padded
        batched pass (every row copies the prefix cache; arrays are shared)."""
        K = len(candidates)
        feeds = [[last_prefix_token] + list(c) for c in candidates]
        L = max(len(f) for f in feeds)
        lens = [len(f) for f in feeds]
        rows = [copy.deepcopy(prefix_cache) for _ in range(K)]
        cache = [rows[0][i].merge([r[i] for r in rows]) for i in range(len(rows[0]))]
        for c in cache:
            if isinstance(c, ArraysCache):
                c.left_padding = None
        if min(lens) < L:
            from .hybrid_batch import _prepare_ragged  # (sliding-window layers need the lengths too)

            _prepare_ragged(cache, lens)
        x = mx.array([f + [0] * (L - len(f)) for f in feeds])
        with packed_rows(K, L, lens):
            h = self.lm.model(x, cache)  # (K, L, D)
        logits = self._head(h)  # (K, L, V)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        out = []
        for i, c in enumerate(candidates):
            tot = 0.0
            for j, t in enumerate(c):
                tot += float(lp[i, j, t].item())  # row j predicts feeds[i][j + 1] = c[j]
            out.append(tot)
        return out


def decide_region(scorer: RegionScorer, hf_tokenizer, prefix_cache, last_prefix_token: int, completions: List[bytes],
                  value_only: Optional[int] = None):
    """The completion with the highest sequence log-probability. With
    ``value_only`` the first that many bytes of each completion are scored
    (the rest is a forced separator the sequential decoder never samples)."""
    cands = [canonical_tokens(hf_tokenizer, w[:value_only] if value_only else w) for w in completions]
    lps = scorer.logprobs(prefix_cache, last_prefix_token, cands)
    best = int(np.argmax(lps))
    return best, lps, cands
