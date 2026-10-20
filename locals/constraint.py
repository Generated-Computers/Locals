"""Compile a regex constraint against a vocabulary.

For a byte DFA with states Q and the vocabulary trie T_V, the step graph at
state q is the product T_V x DFA rooted at (eps, q). It is acyclic (depth
grows) and, for a DFA, its reachable part is an out-tree, so every count is
0 or 1. All |Q| roots are handled at once by the universal-source lift
(Construction 4.6): level d of the trie carries a (nodes_d, Q) array of DFA
states, and dead rows are pruned before descending. The work is bounded by a
budget and the pass fails closed when the budget is exceeded (Thm 4.2/4.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .regex import DEAD, DFA, TooManyStates, compile_dfa
from .vocab import ByteTrie, TokenVocab


class WorkBudgetExceeded(ValueError):
    """The token table would take too much work or memory (a request error: the grammar asks too much)."""


@dataclass
class CompiledConstraint:
    dfa: DFA
    next_state: np.ndarray  # (n_model, Q) next DFA state per token, DEAD if not allowed
    allowed: np.ndarray  # (Q, n_model) bool; EOS allowed exactly at accepting states
    forced_bytes: List[bytes]  # maximal forced byte chain from each state
    only_eos: np.ndarray  # (Q,) accepting states with no byte successors
    eos_ids: List[int]
    stats: dict = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return self.dfa.n

    def advance(self, q: int, token: int) -> int:
        return int(self.next_state[token, q])


def token_table(trie: ByteTrie, dfa: DFA, n_model: int, work_budget: Optional[int] = None):
    Q = dfa.n
    sink = Q
    dtype = np.int16 if Q + 1 < 2**15 else np.int32
    tab = np.where(dfa.table == DEAD, sink, dfa.table).astype(dtype)
    tab = np.concatenate([tab, np.full((1, 256), sink, dtype)], axis=0)
    T = np.full((n_model, Q), sink, dtype)
    prev = np.arange(Q, dtype=dtype)[None, :]  # level 0 row: every state is a root
    prev_map = np.zeros(1, dtype=np.int64)  # trie index at level d-1 -> row in prev
    work = Q
    levels = trie.tok_level
    for d in range(1, trie.depth + 1):
        par = trie.parent[d]
        rows = prev_map[par]
        keep = rows >= 0
        idx = np.nonzero(keep)[0]
        cur = tab[prev[rows[idx]], trie.byte[d][idx][:, None]]
        work += cur.size
        if work_budget is not None and work > work_budget:
            raise WorkBudgetExceeded(f"product work {work} exceeds budget {work_budget}")
        sel = np.nonzero(levels == d)[0]
        if len(sel):
            node_rows = np.full(len(par), -1, dtype=np.int64)
            node_rows[idx] = np.arange(len(idx))
            r = node_rows[trie.tok_index[sel]]
            ok = r >= 0
            T[sel[ok]] = cur[r[ok]]
        alive = (cur != sink).any(axis=1)
        prev = cur[alive]
        m = np.full(len(par), -1, dtype=np.int64)
        m[idx[alive]] = np.arange(int(alive.sum()))
        prev_map = m
        if len(prev) == 0:
            break
    T = T.astype(np.int32)
    T[T == sink] = DEAD
    return T, work


def forced_chains(dfa: DFA) -> List[bytes]:
    """Maximal forced byte chain per state: follow states that are not
    accepting and have exactly one live byte (the unary vertices that the
    branch kernel of Thm A10.1 suppresses). The DFA is trimmed, so a forced
    cycle would never reach acceptance; chains are finite."""
    Q = dfa.n
    live = dfa.table != DEAD
    nsucc = live.sum(axis=1)
    out: List[Optional[bytes]] = [None] * Q
    for q0 in range(Q):
        path = []
        q = q0
        while out[q] is None and not dfa.accepting[q] and nsucc[q] == 1:
            b = int(np.nonzero(live[q])[0][0])
            path.append((q, b))
            q = int(dfa.table[q, b])
            if len(path) > Q:
                raise RuntimeError("forced cycle in a trimmed DFA")
        tail = out[q] if out[q] is not None else b""
        if out[q] is None:
            out[q] = b""
        for s, b in reversed(path):
            tail = bytes([b]) + tail
            out[s] = tail
    return [x or b"" for x in out]


def compile_constraint(
    pattern: str,
    vocab: TokenVocab,
    trie: ByteTrie,
    max_dfa_states: int = 200_000,
    work_budget: Optional[int] = None,
) -> CompiledConstraint:
    import time

    import os

    t0 = time.perf_counter()
    # the token table and its copies take about 7 bytes per (state, token): a string with maxLength 1000
    # (10k states) on a 248k-token vocabulary would need 17 GB, and end the server with a Metal error.
    # The DFA's minimization stops once it is past what fits (a request with maxLength 6000 took 214 s
    # to be refused, with every other request stopped meanwhile; 7 s now).
    limit = float(os.environ.get("LOCALS_MAX_TABLE_GB", "3"))
    cap = int(limit * 1e9 / (7 * vocab.n_model))

    def too_big(size: str) -> WorkBudgetExceeded:
        return WorkBudgetExceeded(f"the schema or regex is too large: its token table would take {size} (the limit is "
                                  f"{limit:g} GB, LOCALS_MAX_TABLE_GB). Make it smaller (a shorter maxLength or maxItems, "
                                  "fewer alternatives), or serve with --walk, which needs no table")

    try:
        dfa = compile_dfa(pattern, max_dfa_states, max_minimal=cap)
    except TooManyStates:
        raise too_big(f"more than {limit:g} GB (over {cap} states x {vocab.n_model} tokens)") from None
    t1 = time.perf_counter()
    gb = 7 * dfa.n * vocab.n_model / 1e9
    if gb > limit:  # (cannot happen: cap is the most states that fit)
        raise too_big(f"{gb:.1f} GB ({dfa.n} states x {vocab.n_model} tokens)")
    T, work = token_table(trie, dfa, vocab.n_model, work_budget)
    t2 = time.perf_counter()
    allowed = (T != DEAD).T.copy()
    for e in vocab.eos_ids:
        allowed[:, e] = dfa.accepting
    live_bytes = (dfa.table != DEAD).sum(axis=1)
    only_eos = dfa.accepting & (live_bytes == 0)
    fb = forced_chains(dfa)
    t3 = time.perf_counter()
    per_state = allowed.sum(axis=1)
    stats = {
        "dfa_states": dfa.n,
        "product_work": int(work),
        "trie_nodes": trie.n_nodes,
        "allowed_min": int(per_state.min()),
        "allowed_max": int(per_state.max()),
        "forced_states": int(sum(1 for b in fb if b)),
        "t_dfa": t1 - t0,
        "t_table": t2 - t1,
        "t_forced": t3 - t2,
    }
    return CompiledConstraint(dfa, T, allowed, fb, only_eos, list(vocab.eos_ids), stats)
