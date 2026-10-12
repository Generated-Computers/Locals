"""Exact finite-region tree decoding.

At a decision whose candidate set C (constraint mask plus the canonical
boundary rule) is small, every candidate c is appended to the same forward
pass as a branch: c followed by the forced span the constraint implies after
it. The branches form a tree rooted at the current position (a mangrove: one
root path per node), laid out in preorder, so node j is an ancestor of node i
exactly when i lies in j's preorder interval (Thm A27.1). Attention uses that
ancestor mask and each token gets its own RoPE position (its depth).

One pass then gives the logits for this decision (at the end of the main
input) and, for every candidate, the logits of the decision that follows it
(at the end of its branch). We pick the candidate exactly as sequential
decoding would, keep that branch's KV, and immediately sample the next
decision from its branch logits. No draft model and nothing is rejected:
every alternative is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .packed import _qkv


@dataclass
class Branch:
    token: int  # the candidate
    tokens: List[int]  # tokens fed for this branch (candidate + forced span)
    replaces: bool  # tokens re-encode the candidate's bytes (healing)
    q_end: int  # constraint state after the branch


def tree_forward(model, cache, main: List[int], branches: List[Branch]):
    """One forward pass over ``main`` followed by the branch tree.

    ``cache`` is a list of SpecKVCache. KV for all fed tokens is appended in
    flattened order. Returns (logits at main end, logits at each branch end,
    branch start offsets relative to the end of main).
    """
    off = cache[0].offset
    m = len(main)
    toks = list(main)
    pos = list(range(off, off + m))
    starts = []
    for b in branches:
        starts.append(len(toks) - m)
        toks.extend(b.tokens)
        pos.extend(range(off + m, off + m + len(b.tokens)))
    L = len(toks)
    # Ancestor mask over keys [0, off + L): prefix and main are visible to all;
    # within the tree a branch token sees earlier tokens of its own branch.
    br_id = np.full(L, -1, dtype=np.int32)
    for i, b in enumerate(branches):
        s = m + starts[i]
        br_id[s : s + len(b.tokens)] = i
    qi = np.arange(L)[:, None]
    kj = np.arange(L)[None, :]
    causal = kj <= qi
    same = (br_id[:, None] == br_id[None, :]) | (br_id[None, :] < 0)
    local = causal & same
    mask = np.concatenate([np.ones((L, off), dtype=bool), local], axis=1)
    mask = mx.array(mask)[None, None]
    pos_a = mx.array(pos, dtype=mx.int32)

    body = model.model
    h = body.embed_tokens(mx.array(toks)[None])
    for li, layer in enumerate(body.layers):
        attn = layer.self_attn
        x = layer.input_layernorm(h)
        q, k, v = _qkv(attn, x, L)  # (L, heads, 1, D)
        q = attn.rope(q, offset=pos_a)
        k = attn.rope(k, offset=pos_a)
        keys, values = cache[li].update_and_fetch(k.transpose(2, 1, 0, 3), v.transpose(2, 1, 0, 3))
        o = mx.fast.scaled_dot_product_attention(q.transpose(2, 1, 0, 3), keys, values, scale=attn.scale, mask=mask)
        h = h + attn.o_proj(o.transpose(0, 2, 1, 3).reshape(1, L, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    h = body.norm(h)
    rows = [m - 1] + [m + s + len(b.tokens) - 1 for s, b in zip(starts, branches)]
    hl = h[:, mx.array(rows, dtype=mx.int32), :]
    from .kernels import head_logits

    logits = head_logits(model, hl)[0]
    return logits[0], logits[1:], starts


def keep_branch(cache, base: int, m: int, start: int, length: int) -> None:
    """Keep main + one branch: move the branch's KV right after main."""
    for c in cache:
        if start:
            src_k = c.keys[..., base + m + start : base + m + start + length, :]
            src_v = c.values[..., base + m + start : base + m + start + length, :]
            st = mx.array([0, 0, base + m, 0], dtype=mx.int32)
            c.keys = mx.slice_update(c.keys, src_k, st, (0, 1, 2, 3))
            c.values = mx.slice_update(c.values, src_v, st, (0, 1, 2, 3))
        c.offset = base + m + length


def candidate_branches(ctl, max_candidates: int, budget: int) -> Optional[List[Branch]]:
    """Branches for the decision at the controller's current state, or None
    when the decision is open (too many candidates) or the tree is too big."""
    c = ctl.c
    q = ctl.q
    row = c.allowed[q]
    cand = np.nonzero(row)[0]
    cover = ctl.planner.cover_rule(q) if ctl.canonical else 0
    if cover:
        lens = ctl._tok_len_np()
        cand = cand[lens[cand] >= cover]
    if len(cand) == 0 or len(cand) > max_candidates:
        return None
    out: List[Branch] = []
    total = 0
    for t in cand.tolist():
        if t in c.eos_ids:
            continue
        q1 = int(c.next_state[t, q])
        plan = ctl.planner.plan(t, q1) if ctl.use_forced else None
        if plan is None:
            b = Branch(t, [t], False, q1)
        elif plan.replaces_prev:
            b = Branch(t, list(plan.tokens), True, plan.q_end)
        else:
            b = Branch(t, [t] + list(plan.tokens), False, plan.q_end)
        out.append(b)
        total += len(b.tokens)
        if total > budget:
            return None
    return out or None
