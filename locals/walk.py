"""Constraints without tables: every step walks every token through the
automaton on the device.

The all-state table stores, for each state, where every token leads: a
$V \times |Q|$ array. The ReachUL proof suggests the opposite: a
configuration is reached by one path, so nothing about it needs storing;
recompute it. Here the recomputation is one Metal kernel per decode step:
a thread per vocabulary token walks the token's bytes through the byte
automaton from the current state (a few table lookups, since a token has a
handful of bytes) and writes the state it reaches. That row is the step's
mask and its transition; the state stays on the device, so the engine is as
pipelined as with a table, and the memory is the automaton
(256 · |Q| entries) plus the vocabulary's bytes, whatever the
grammar's size. There is no compile pass beyond the automaton.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .constraint import forced_chains
from .engine import NEG_INF, Controller, _token_lengths
from .forced import ForcedPlanner
from .regex import DEAD, DFA, compile_dfa
from .vocab import TokenVocab

_SOURCE = r"""
    uint t = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    if (t >= (uint)V) return;
    int s = (int)q[b];
    int start = offs[t];
    int end = offs[t + 1];
    if (start == end) { nxt[b * V + t] = (short)(-1); return; }
    for (int i = start; i < end && s >= 0; ++i) {
        s = (int)table[s * 256 + (int)tbytes[i]];
    }
    nxt[b * V + t] = (short)s;
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(name="locals_dfa_walk", input_names=["q", "table", "tbytes", "offs"], output_names=["nxt"],
                                       source=_SOURCE)
    return _KERNEL


class PackedVocab:
    """Every token's bytes concatenated, with offsets: the kernel's view of the vocabulary."""

    def __init__(self, vocab: TokenVocab):
        offs = [0]
        chunks = []
        for b in vocab.token_bytes:
            if b:
                chunks.append(np.frombuffer(b, dtype=np.uint8))
                offs.append(offs[-1] + len(b))
            else:
                offs.append(offs[-1])
        self.n_model = vocab.n_model
        self.tbytes = mx.array(np.concatenate(chunks) if chunks else np.zeros(1, np.uint8))
        self.offs = mx.array(np.array(offs, dtype=np.int32))
        self.nbytes = int(self.tbytes.size) + int(self.offs.size) * 4


_PACKED = {}


def packed_vocab(vocab: TokenVocab) -> PackedVocab:
    k = id(vocab)
    if k not in _PACKED:
        _PACKED[k] = (vocab, PackedVocab(vocab))
    return _PACKED[k][1]


class WalkConstraint:
    """A byte automaton on the device; rows are computed per step by the walk kernel."""

    def __init__(self, dfa: DFA, vocab: TokenVocab, trie=None):
        self.dfa = dfa
        self.vocab = vocab
        self.eos_ids = list(vocab.eos_ids)
        self.forced_bytes = forced_chains(dfa)
        live = (dfa.table != DEAD).sum(axis=1)
        self.only_eos = dfa.accepting & (live == 0)
        if dfa.n >= 2**15:
            raise ValueError("the walk kernel stores states as int16")
        self.table = mx.array(dfa.table[:, :256].astype(np.int16).reshape(-1))
        self.accepting = mx.array(dfa.accepting.astype(np.bool_))
        eos = np.zeros(vocab.n_model, dtype=np.bool_)
        eos[self.eos_ids] = True
        self.eos_vec = mx.array(eos)
        self.packed = packed_vocab(vocab)
        self.trie = trie
        self.sparse = {}  # host state -> allowed ids (device) or False when too many
        self.calls = 0

    @property
    def n_states(self) -> int:
        return self.dfa.n

    @property
    def nbytes(self) -> int:
        return int(self.table.size) * 2 + self.packed.nbytes

    def rows(self, q: mx.array) -> mx.array:
        """(B, V) int16: the state each token leads to from each of the B states (DEAD = -1)."""
        q = q.reshape(-1).astype(mx.int32)
        B = int(q.shape[0])
        V = self.vocab.n_model
        self.calls += 1
        (out,) = _kernel()(inputs=[q, self.table, self.packed.tbytes, self.packed.offs], template=[("V", V)],
                           grid=(V, B, 1), threadgroup=(256, 1, 1), output_shapes=[(B, V)], output_dtypes=[mx.int16])
        return out


class WalkController(Controller):
    """The pipelined engine's controller with no table: the mask and the
    transition of the device state come from the walk kernel each step."""

    def __init__(self, wc: WalkConstraint, planner: Optional[ForcedPlanner], use_forced: bool = True, canonical: bool = True,
                 heal: bool = False):
        self.wc = wc
        self.planner = planner
        self.use_forced = use_forced and planner is not None
        self.canonical = canonical and planner is not None
        self.heal = heal
        self.eos_ids = wc.eos_ids
        self.q = wc.dfa.start
        self._last: Optional[Tuple[mx.array, mx.array]] = None
        if self.canonical:
            self.tok_len = _token_lengths(planner.vocab)
            self.cover = mx.array(np.array([planner.cover_rule(s) for s in range(wc.dfa.n)], dtype=np.int32))

    # host side
    def apply(self, logits):
        return mx.where(self.gpu_mask(self.gpu_state()), logits, NEG_INF)

    def accept(self, token):
        b = self.wc.vocab.token_bytes[token]
        nq = self.wc.dfa.run(b, self.q) if b else DEAD
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
        return bool(self.wc.only_eos[self.q])

    def may_force(self):
        return self.use_forced

    SPARSE_MAX = 4096

    def sparse_ids(self, q: int):
        """Allowed token ids at host state q as a device array when at most
        SPARSE_MAX are allowed (the engine then computes the head for those
        rows only), else None. Computed once per state by a CPU walk
        (token_rows with one root) and kept as ids only, at most 16 KB each."""
        ids = self.wc.sparse.get(q)
        if ids is None:
            from .lazy import token_rows

            nxt = token_rows(self.wc.trie, self.wc.dfa, self.wc.vocab.n_model, [q])[:, 0]
            allowed = nxt != DEAD
            if self.wc.dfa.accepting[q]:
                allowed[self.eos_ids] = True
            if self.canonical:
                cov = self.planner.cover_rule(q)
                if cov:
                    lens = np.array(self.tok_len)
                    allowed &= lens >= cov
            a = np.nonzero(allowed)[0]
            ids = mx.array(a.astype(np.int32)) if 0 < len(a) <= self.SPARSE_MAX else False
            self.wc.sparse[q] = ids
        return ids if ids is not False else None

    # device side
    def gpu_state(self):
        return mx.array(self.q, dtype=mx.int32)

    def _row(self, qg):
        if self._last is not None and self._last[0] is qg:
            return self._last[1]
        row = self.wc.rows(qg)[0]
        self._last = (qg, row)
        return row

    def gpu_mask(self, qg):
        row = self._row(qg)
        m = (row != -1) | (self.wc.accepting[qg] & self.wc.eos_vec)
        if self.canonical:
            m = m & (self.tok_len >= self.cover[qg])
        return m

    def gpu_next(self, y, qg):
        return self._row(qg)[y].astype(mx.int32)


def compile_walk(rt, regex: str, max_dfa_states: int = 400_000) -> Tuple[WalkConstraint, ForcedPlanner]:
    dfa = compile_dfa(regex, max_dfa_states)
    wc = WalkConstraint(dfa, rt.vocab, rt.trie)
    planner = ForcedPlanner(rt.tokenizer._tokenizer, rt.vocab, wc)
    return wc, planner
