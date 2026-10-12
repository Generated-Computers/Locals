"""Constraint tables in space proportional to what is visited.

The all-state token table materializes ``T[t, q]`` for every state: a
$V \times |Q|$ array (3 bytes per entry with the mask), which is the
breadth-first side of the time--space trade-off. The unambiguous
traversal behind the ReachUL membership is the other side: the row of one
state is one rooted walk of the vocabulary trie (Construction 4.6 with a
single root), computable when the state is first reached, and a decode of
$n$ tokens reaches at most $n$ states. So the rows are computed on demand
and kept in a bounded cache; memory is O(V · visited),
independent of the grammar's size, and there is no compile pass. The
successors of the current state are known from its own row, so their rows
are computed in a background thread while the model runs the step
(the SC2 discipline: recompute along the unique path at no wall-time cost).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from .constraint import forced_chains
from .engine import NEG_INF, Controller, _token_lengths
from .forced import ForcedPlanner
from .regex import DEAD, DFA, compile_dfa
from .vocab import ByteTrie, TokenVocab


def token_rows(trie: ByteTrie, dfa: DFA, n_model: int, roots: Sequence[int]) -> np.ndarray:
    """``T[t, i]`` = state reached from ``roots[i]`` by token ``t`` (DEAD if
    not allowed): the level pass of the all-state table with the given
    roots only."""
    Q = dfa.n
    sink = Q
    dtype = np.int32
    tab = np.where(dfa.table == DEAD, sink, dfa.table).astype(dtype)
    tab = np.concatenate([tab, np.full((1, 256), sink, dtype)], axis=0)
    R = len(roots)
    T = np.full((n_model, R), sink, dtype)
    prev = np.array(roots, dtype=dtype)[None, :]
    prev_map = np.zeros(1, dtype=np.int64)
    levels = trie.tok_level
    for d in range(1, trie.depth + 1):
        par = trie.parent[d]
        rows = prev_map[par]
        keep = rows >= 0
        idx = np.nonzero(keep)[0]
        cur = tab[prev[rows[idx]], trie.byte[d][idx][:, None]]
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
    T[T == sink] = DEAD
    return T


class LazyConstraint:
    """A byte DFA with token rows computed per visited state."""

    def __init__(self, dfa: DFA, vocab: TokenVocab, trie: ByteTrie, max_rows: int = 512):
        self.dfa = dfa
        self.vocab = vocab
        self.trie = trie
        self.n_model = vocab.n_model
        self.eos_ids = list(vocab.eos_ids)
        self.forced_bytes = forced_chains(dfa)
        live = (dfa.table != DEAD).sum(axis=1)
        self.only_eos = dfa.accepting & (live == 0)
        self.max_rows = max_rows
        self._rows: "OrderedDict[int, Tuple[np.ndarray, mx.array]]" = OrderedDict()
        self._lock = threading.Lock()
        self.computed = 0
        self.compute_s = 0.0

    @property
    def n_states(self) -> int:
        return self.dfa.n

    @property
    def bytes_resident(self) -> int:
        return len(self._rows) * self.n_model * (3 if self.n_states < 2**15 else 5)  # next state + mask, as the full table stores them

    def ensure(self, qs: Sequence[int]) -> None:
        with self._lock:
            need = [q for q in dict.fromkeys(qs) if q not in self._rows]
        if not need:
            return
        t = time.perf_counter()
        T = token_rows(self.trie, self.dfa, self.n_model, need)
        acc = self.dfa.accepting
        with self._lock:
            for i, q in enumerate(need):
                nxt = T[:, i].astype(np.int16 if self.n_states < 2**15 else np.int32)
                allowed = nxt != DEAD
                if acc[q]:
                    allowed[self.eos_ids] = True
                self._rows[q] = (nxt, mx.array(allowed))
                self._rows.move_to_end(q)
            while len(self._rows) > self.max_rows:
                self._rows.popitem(last=False)
            self.computed += len(need)
            self.compute_s += time.perf_counter() - t

    def row(self, q: int) -> Tuple[np.ndarray, mx.array]:
        with self._lock:
            r = self._rows.get(q)
            if r is not None:
                self._rows.move_to_end(q)
                return r
        self.ensure([q])
        with self._lock:
            return self._rows[q]

    def successors(self, q: int) -> List[int]:
        nxt, _ = self.row(q)
        s = np.unique(nxt[nxt != DEAD])
        return [int(x) for x in s]


class LazyController(Controller):
    """The synchronous engine's controller on lazy rows. After a step's mask
    is built, the rows of the state's successors are computed in a thread,
    so they are ready when the sampled token is known."""

    supports_gpu = False  # its masks are built on the host: the pipelined engine would decode unconstrained

    def __init__(self, lz: LazyConstraint, planner: Optional[ForcedPlanner], use_forced: bool = True, canonical: bool = True,
                 prefetch: bool = False, heal: bool = False):
        self.lz = lz
        self.heal = heal  # the full controller heals only far from the token budget; off matches its usual behaviour
        self.planner = planner
        self.use_forced = use_forced and planner is not None
        self.canonical = canonical and planner is not None
        self.eos_ids = lz.eos_ids
        self.q = lz.dfa.start
        self.prefetch = prefetch
        self._thread: Optional[threading.Thread] = None
        if self.canonical:
            self.tok_len = _token_lengths(planner.vocab)
        self.waits = 0

    def _join(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _sampling_successors(self, q: int) -> List[int]:
        """The states at which the next sample can happen: each successor of
        q, or the end of its forced chain (chain states never need a row)."""
        out = []
        for s in self.lz.successors(q):
            if self.use_forced and self.lz.forced_bytes[s]:
                s = self.planner._end_state(s)
            out.append(s)
        return list(dict.fromkeys(out))

    def _start_prefetch(self, q: int):
        if not self.prefetch:
            return
        self._join()
        lz = self.lz
        ctl = self

        def work():
            lz.ensure(ctl._sampling_successors(q))

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def apply(self, logits):
        _, allowed = self.lz.row(self.q)
        m = allowed
        if self.canonical:
            cover = self.planner.cover_rule(self.q)
            if cover:
                m = m & (self.tok_len >= cover)
        self._start_prefetch(self.q)
        return mx.where(m, logits, NEG_INF)

    def accept(self, token):
        nxt, _ = self.lz.row(self.q)
        nq = int(nxt[token])
        if nq == DEAD:
            raise RuntimeError(f"token {token} not allowed at state {self.q}")
        self.q = nq

    def forced(self, token, heal: bool = True):
        if not self.use_forced:
            return [], False
        plan = self.planner.plan(token if (heal and self.heal) else None, self.q)
        if plan is None:
            return [], False
        self.q = plan.q_end
        return plan.tokens, plan.replaces_prev

    def finished(self):
        return bool(self.lz.only_eos[self.q])

    def may_force(self):
        return self.use_forced


def compile_lazy(rt, regex: str, max_rows: int = 512, max_dfa_states: int = 400_000) -> Tuple[LazyConstraint, ForcedPlanner]:
    """A lazy constraint for the runtime's vocabulary: the automaton is
    compiled, no token table is built."""
    dfa = compile_dfa(regex, max_dfa_states)
    lz = LazyConstraint(dfa, rt.vocab, rt.trie, max_rows=max_rows)
    planner = ForcedPlanner(rt.tokenizer._tokenizer, rt.vocab, lz)
    return lz, planner
