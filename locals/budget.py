"""Budget-aware masking: never sample a token after which no valid completion
fits in the remaining ``max_tokens``.

The engines stop right after a *sampled* token once ``len(out) >= max_tokens``
(before that token's forced span is fed), and forced tokens count toward
``len(out)``. So with r tokens left at a sampling point in DFA state q, a
token leading to q' leaves a valid output iff

    cost(q') <= r,   cost(q') = 1                          if q' accepts
                     cost(q') = 1 + F(q') + need(E(q'))    otherwise

where F(q') is the length of the forced span planned after q' (no healing),
E(q') the state after it, and need(q) the smallest budget from which q can
still finish: 1 at an accepting state (sample EOS), 0 after a span that ends
in a state with no successors (the engine stops there), else the minimum cost
over the tokens allowed at q (canonical boundary rule included). This is a
shortest-path problem on the DFA's token-step graph, the budget reading of
Thm 4.2 / A15 (shortest completions are exact functions of the product
graph). Masking by it keeps need(q) <= r at every sampling point by
induction, so an output cut by max_tokens is always in an accepting state,
and forced spans never push an output past max_tokens.

Healing re-encodes the sampled token with its span and can change the span's
length, so engines plan spans without healing once r <= heal_threshold =
max cost + longest token + longest forced chain (in bytes). Above that, a
healed span consumes at most |token| + |chain| tokens, which leaves more than
the max cost, so every state is still completable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .constraint import CompiledConstraint
from .regex import DEAD

INF = np.int32(1 << 30)


@dataclass
class BudgetTables:
    need: np.ndarray  # (Q,) smallest remaining budget at a sampling point in state q
    cost: np.ndarray  # (Q,) budget a sampled token must leave when it leads to q'
    max_cost: int  # largest finite cost; with r >= max_cost the mask is a no-op
    has_inf: bool  # some allowed token leads to a state that cannot finish
    heal_threshold: int
    span: np.ndarray  # (Q,) F(q'): forced tokens planned after q' (0 if none)


def budget_tables(c: CompiledConstraint, planner=None, cover=None, tok_len: Optional[np.ndarray] = None) -> BudgetTables:
    """``planner``: the ForcedPlanner whose spans the engine feeds (None when
    forced spans are off). ``cover``: planner providing the canonical boundary
    rule (None when off). ``tok_len``: byte length per token id (needed with
    ``cover``, and for the healing threshold when spans are on)."""
    Q = c.n_states
    acc = np.asarray(c.dfa.accepting, dtype=bool)
    only_eos = np.asarray(c.only_eos, dtype=bool)
    eos = np.zeros(c.next_state.shape[0], dtype=bool)
    eos[list(c.eos_ids)] = True

    # Edges q -> q' of the token-step graph (distinct successors under the mask).
    src, dst = [], []
    for q in range(Q):
        ok = c.allowed[q] & ~eos
        if cover is not None:
            cv = cover.cover_rule(q)
            if cv:
                ok &= tok_len >= cv
        nxt = np.unique(c.next_state[ok, q])
        nxt = nxt[nxt != DEAD]
        src.append(np.full(len(nxt), q, np.int64))
        dst.append(nxt.astype(np.int64))
    src = np.concatenate(src) if src else np.zeros(0, np.int64)
    dst = np.concatenate(dst) if dst else np.zeros(0, np.int64)

    # Forced span after entering q' (planned without healing, as the engines do
    # near the end of the budget).
    span = np.zeros(Q, np.int32)
    end = np.arange(Q)
    if planner is not None:
        for q in range(Q):
            if c.forced_bytes[q]:
                p = planner.plan(None, q)
                if p is not None and p.tokens:
                    span[q] = len(p.tokens)
                    end[q] = p.q_end

    need = np.where(acc, 1, INF).astype(np.int64)
    for _ in range(Q + 1):  # Bellman-Ford on positive weights; converges in <= Q rounds
        after = np.where(only_eos[end] & (span > 0), 0, need[end])
        cost = np.where(acc, 1, np.minimum(INF, 1 + span + after))
        new = need.copy()
        if len(src):
            np.minimum.at(new, src, cost[dst])
        new = np.where(acc, 1, new)
        if np.array_equal(new, need):
            break
        need = new
    after = np.where(only_eos[end] & (span > 0), 0, need[end])
    cost = np.where(acc, 1, np.minimum(INF, 1 + span + after)).astype(np.int32)
    finite = cost[cost < INF]
    max_cost = int(finite.max()) if len(finite) else 1
    has_inf = bool(len(dst) and (cost[dst] >= INF).any())
    if planner is not None and tok_len is None:
        raise ValueError("tok_len is needed to bound healed spans")
    max_tok = int(tok_len.max()) if tok_len is not None else 0
    max_chain = max((len(b) for b in c.forced_bytes), default=0)
    return BudgetTables(need.astype(np.int32), cost, max_cost, has_inf, max_cost + max_tok + max_chain, span)
