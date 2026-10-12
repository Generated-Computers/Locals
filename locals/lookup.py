"""Greedy decoding with prompt-lookup drafts verified in one pass.

A draft is proposed whenever the last n tokens of the stream (prompt and
output) occurred earlier in it: the k tokens that followed that occurrence.
Under a constraint the draft is first walked through the automaton on the
host (a draft is one path in the constraint's step graph; the transition
table composed along it, A14.2) and cut at the first token the constraint
would not allow. The sampled token, its forced span and the draft are fed in
one pass; the logits at every draft position, masked with the constraint's
state there, give the greedy choice the sequential decoder would have made,
so the accepted prefix is exact: greedy lookup decoding equals greedy
decoding up to floating-point ties. Rejected draft tokens are undone by
rewinding the KV offsets and selecting the recurrent state the fused GDN
kernel recorded after the last accepted token (engine.rewind), so a
rejection costs nothing beyond the pass.
"""

from __future__ import annotations

import time
from typing import List, Optional

import mlx.core as mx

from .engine import NEG_INF, ConstraintExhausted, Controller, GenResult, cache_offset, make_spec_cache, rewind, set_want_states
from .kernels import text_model


def _head_all(model, h):
    """Logits at every position of h (1, S, D) -> (S, V)."""
    from .kernels import head_logits

    return head_logits(text_model(model), h)[0]


def propose(stream: List[int], n: int, k: int, eos: set) -> List[int]:
    """The k tokens that followed the most recent earlier occurrence of the
    stream's last n tokens (none if there is none); cut before any EOS."""
    if len(stream) <= n:
        return []
    key = stream[-n:]
    for j in range(len(stream) - n - 1, -1, -1):
        if stream[j : j + n] == key:
            draft = stream[j + n : j + n + k]
            for i, t in enumerate(draft):
                if t in eos:
                    return draft[:i]
            return draft
    return []


def generate_lookup(engine, prompt: List[int], ctl: Optional[Controller], max_tokens: int = 256, n: int = 3, k: int = 8,
                    cache=None) -> GenResult:
    model = engine.model
    ctl = ctl or Controller()
    eos = set(ctl.eos_ids) or set(engine.tokenizer.eos_token_ids)
    constrained = hasattr(ctl, "step_config")
    t0 = time.perf_counter()
    if cache is None:
        cache = make_spec_cache(model)
    engine._prefill(cache, prompt)
    mx.synchronize()
    t1 = time.perf_counter()
    fed = list(prompt[:-1])
    stream = list(prompt)
    out: List[int] = []
    passes = forced_n = drafted = 0
    sizes: List[int] = []
    finish = "length"
    y = prompt[-1]  # the next token to feed; before the first pass, the prompt's last token
    first = True
    while True:
        # the tokens after y that are known: its forced span (once y is an output token), then a draft
        extra: List[int] = []
        replaces = False
        if not first:
            extra, replaces = ctl.forced(y) if hasattr(ctl, "forced") else ([], False)
            if replaces:
                out.pop()
                stream.pop()
                forced_n += len(extra) - 1
            else:
                forced_n += len(extra)
            out.extend(extra)
            stream.extend(extra)
            if ctl.finished():
                finish = "stop"
                break
            if len(out) >= max_tokens:
                break
        head = (list(extra) if replaces else [y] + list(extra))
        draft = propose(stream, n, k, eos) if len(out) < max_tokens - 1 else []
        cfgs = []  # constraint configuration after each draft token
        if constrained and draft:
            cfg = ctl.config()
            kept = []
            for t in draft:
                cfg = ctl.step_config(cfg, t)
                if cfg is None:
                    break
                kept.append(t)
                cfgs.append(cfg)
                if ctl.finished_at(cfg):
                    break
            draft = kept
        draft = draft[: max(0, max_tokens - len(out) - 1)]
        feed = head + draft
        S = len(feed)
        set_want_states(cache, S > 1)
        h = text_model(model).model(mx.array([feed]), cache)  # (1, S, D), every position
        passes += 1
        sizes.append(S)
        p0 = len(head) - 1  # position whose logits sample the next token after head
        logits = _head_all(model, h[:, p0:, :])  # (1 + len(draft), V)
        # masks: the constraint state after head, then after each draft token
        if constrained:
            states = [ctl.device_state(ctl.config())] + [ctl.device_state(c) for c in cfgs]
            masks = []
            for i, st in enumerate(states):
                m = ctl.gpu_mask(st)
                bm = ctl.budget_mask(st, max_tokens - len(out) - i)
                if bm is not None:
                    m = bm if m is None else m & bm
                masks.append(m)
            logits = mx.where(mx.stack(masks), logits, NEG_INF)
        pred = mx.argmax(logits, axis=-1).tolist()
        a = 0
        while a < len(draft) and pred[a] == draft[a]:
            a += 1
        fed.extend(feed[: len(head) + a])
        rewind(cache, len(draft) - a)
        set_want_states(cache, False)
        # commit the accepted draft tokens, then the newly sampled token
        for i in range(a):
            ctl.accept(draft[i])
            out.append(draft[i])
            stream.append(draft[i])
        drafted += a
        y = pred[a]
        first = False
        if a > 0 and ctl.finished():  # the accepted draft completed the output
            finish = "stop"
            break
        if len(out) >= max_tokens:
            break
        if y in eos:
            finish = "stop"
            break
        try:
            ctl.accept(y)
        except ConstraintExhausted:
            finish = "exhausted"
            break
        out.append(y)
        stream.append(y)
        if len(out) >= max_tokens:
            break
        if ctl.finished():
            finish = "stop"
            break
    mx.synchronize()
    t2 = time.perf_counter()
    assert cache_offset(cache) == len(fed), (cache_offset(cache), len(fed))
    text = engine.tokenizer.decode(out)
    res = GenResult(out, text, len(prompt), t1 - t0, t2 - t1, passes, forced_n, sizes, finish, fed, cache, 0)
    res.drafted = drafted
    return res
