"""Controller for recursive (pushdown) constraints: the automaton state and
its stack live on the device, so masking and state advance need no host
round trip and decoding stays pipelined (see locals/pda.py)."""

from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
import numpy as np

from .engine import NEG_INF, ConstraintExhausted, Controller, _TABLES, _token_lengths
from .forced import ForcedPlanner
from .pda import MAX_POP, MAX_PUSH, RET, STACK_DEPTH, PDAConstraint
from .regex import DEAD


class PDAController(Controller):
    """Configuration = (state, stack). ``q`` is the hashable configuration
    the engines use as a key (sparse-head ids, force statistics)."""

    def __init__(self, constraint: PDAConstraint, planner: Optional[ForcedPlanner], use_forced: bool = True, canonical: bool = False):
        self.c = constraint
        self.planner = planner
        self.use_forced = use_forced and planner is not None
        self.canonical = canonical and planner is not None
        self.eos_ids = constraint.eos_ids
        self.qs = constraint.pda.root
        self.stack: List[int] = []
        self.dev = _device_tables(constraint)
        if self.canonical:
            self.tok_len = _token_lengths(planner.vocab)
            prepare_cover(self)

    @property
    def q(self):
        return (self.qs, tuple(self.stack))

    # -- host ------------------------------------------------------------------------
    def accept(self, token):
        before = list(self.stack)
        nq = self.c.advance(token, self.qs, self.stack)
        if nq == DEAD:
            if len(self.c.allowed_ids(self.qs, before)) == 0:
                raise ConstraintExhausted(f"no token allowed at depth {len(before)}")
            raise RuntimeError(f"token {token} not allowed at {(self.qs, tuple(before))}")
        self.qs = nq

    def finished(self):
        return self.qs == self.c.pda.n

    def forced(self, token, heal: bool = True):
        if not self.use_forced or self.qs == self.c.pda.n:
            return [], False
        plan = self.planner.plan(token if heal else None, self.q)
        if plan is None:
            return [], False
        self.qs, st = plan.q_end
        self.stack = list(st)
        return plan.tokens, plan.replaces_prev

    def may_force(self):
        if not self.use_forced:
            return False
        return bool(_may_force(self.c, self.planner)[self.qs])

    def apply(self, logits):
        m = self.gpu_mask(self.gpu_state())
        return mx.where(m, logits, NEG_INF)

    SPARSE_MAX = 4096

    def sparse_ids(self, cfg):
        q, st = cfg
        # the allowed set depends on the top MAX_POP entries, whether the stack is
        # empty, and (near the depth limit) the exact depth
        dk = len(st) if len(st) > STACK_DEPTH - MAX_PUSH else min(len(st), 1)
        k = ("pda_sparse", id(self.c), id(self.planner), self.canonical, q, tuple(st[-MAX_POP:]), dk)
        if k not in _TABLES:
            ids = self.c.allowed_ids(q, list(st))
            if self.canonical:
                cov = self.planner.cover_rule(cfg)
                if cov:
                    ids = ids[_tok_len_np(self.planner.vocab)[ids] >= cov]
            _TABLES[k] = ((self.c, self.planner), mx.array(ids.astype(np.int32)) if 0 < len(ids) <= self.SPARSE_MAX else None)
        return _TABLES[k][1]

    # -- device ----------------------------------------------------------------------
    def gpu_state(self):
        st = self.stack + [0] * (STACK_DEPTH - len(self.stack))
        need = int(sum(int(self.dev["bn_np"][r]) for r in self.stack))
        return mx.array([self.qs, len(self.stack)] + st + [need], dtype=mx.int32)

    def _columns(self, qg):
        """(next, aux) over the vocabulary at device configuration qg."""
        d = self.dev
        q = qg[0]
        nx = d["next"][:, q]
        ax = d["aux"][:, q]
        if d["n_ctx"]:
            qi = d["qc_index"][q]
            j = self._ctx(qg)
            cn = d["ctx_next"][:, mx.maximum(qi, 0), j]
            ca = d["ctx_aux"][:, mx.maximum(qi, 0), j]
            nx[d["ctx_ids"]] = mx.where(qi >= 0, cn, nx[d["ctx_ids"]])
            ax[d["ctx_ids"]] = mx.where(qi >= 0, ca, ax[d["ctx_ids"]])
        return nx.astype(mx.int32), ax.astype(mx.int32)

    def _ctx(self, qg):
        """Context index from the top MAX_POP stack entries (C = none)."""
        Q = self.c.pda.n
        C1 = len(self.c.pda.returns) + 1
        ptr = qg[1]
        idx = mx.array(0, dtype=mx.int32)
        for k in range(MAX_POP):
            top = mx.where(ptr > k, qg[2 + mx.maximum(ptr - 1 - k, 0)], Q)
            idx = idx * C1 + self.dev["ridx"][top]
        return idx

    def gpu_mask(self, qg):
        d = self.dev
        q, ptr = qg[0], qg[1]
        nx, ax = self._columns(qg)
        pops = (ax >> 8) & 3
        m = (nx != DEAD) & ((pops <= ptr) | ((nx == RET) & (pops == ptr + 1)))
        plen = d["pat_len"][ax & 0xFF]
        m = m & (plen + ptr - pops <= STACK_DEPTH)
        m[d["eos"]] = d["accepting"][q] & (ptr == 0)
        if self.canonical:
            m = m & (self.tok_len >= d["cover"][q])
        return m

    def gpu_next(self, y, qg):
        d = self.dev
        Q = self.c.pda.n
        q, ptr, st, S = qg[0], qg[1], qg[2 : 2 + STACK_DEPTH], qg[2 + STACK_DEPTH]
        nx = d["next"][y, q].astype(mx.int32)
        ax = d["aux"][y, q].astype(mx.int32)
        if d["n_ctx"]:
            ci, qi = d["ctx_index"][y], d["qc_index"][q]
            use = (ci >= 0) & (qi >= 0)
            j = self._ctx(qg)
            nx = mx.where(use, d["ctx_next"][mx.maximum(ci, 0), mx.maximum(qi, 0), j].astype(mx.int32), nx)
            ax = mx.where(use, d["ctx_aux"][mx.maximum(ci, 0), mx.maximum(qi, 0), j].astype(mx.int32), ax)
        pat = ax & 0xFF
        pops = mx.minimum((ax >> 8) & 3, ptr)
        # RET: continue at the last popped return state; past the bottom it is the root's end (FIN)
        popped = mx.where(ptr - pops >= 0, st[mx.maximum(ptr - pops, 0)], Q)
        nq = mx.where(nx == RET, mx.where(((ax >> 8) & 3) <= ptr, popped, Q), nx)
        ptr2 = ptr - pops
        n = d["pat_len"][pat]
        pushes = d["patterns"][pat]
        pos = mx.arange(STACK_DEPTH, dtype=mx.int32)
        st2 = mx.where((pos >= ptr2) & (pos < ptr2 + n), pushes[mx.clip(pos - ptr2, 0, MAX_PUSH - 1)], st)
        top1 = mx.where(ptr > 0, st[mx.maximum(ptr - 1, 0)], Q)
        top2 = mx.where(ptr > 1, st[mx.maximum(ptr - 2, 0)], Q)
        S2 = S - mx.where(pops >= 1, d["bn"][top1], 0) - mx.where(pops >= 2, d["bn"][top2], 0) + d["pat_need"][pat]
        return mx.concatenate([mx.stack([nq, ptr2 + n]), st2, mx.stack([S2])]).astype(mx.int32)

    # -- budget-aware masking (see locals/budget.py for the regex path) -------------
    # The shortest byte completion bounds the tokens needed (every token and
    # forced span covers >= 1 byte), and it decomposes over the stack:
    # need(q, stack) = bytes(q) + sum of bytes(r) over the return states. So a
    # token costs 1 + need(after it), read off the device columns and the
    # running stack sum S. Where need(q, stack) > r the mask is dropped.
    budget = True
    MARGIN = 16

    def budget_active(self, r: Optional[int]) -> bool:
        if not self.budget or r is None:
            return False
        d = self.dev
        need = int(d["bn_np"][self.qs]) + int(sum(int(d["bn_np"][x]) for x in self.stack))
        # the device may be one token ahead of the host state: allow for its pushes
        return r < need + (MAX_PUSH + 1) * int(d["bn_max"]) + self.MARGIN

    def _cost_after(self, qg, nx, ax):
        d = self.dev
        Q = self.c.pda.n
        ptr, st, S = qg[1], qg[2 : 2 + STACK_DEPTH], qg[2 + STACK_DEPTH]
        pops = (ax >> 8) & 3
        pat = ax & 0xFF
        top1 = mx.where(ptr > 0, st[mx.maximum(ptr - 1, 0)], Q)
        top2 = mx.where(ptr > 1, st[mx.maximum(ptr - 2, 0)], Q)
        popneed = mx.where(pops >= 1, d["bn"][top1], 0) + mx.where(pops >= 2, d["bn"][top2], 0)
        popped = mx.where(pops == 1, top1, mx.where(pops == 2, top2, Q))
        after = mx.where(nx == RET, mx.where(pops <= ptr, popped, Q), mx.where(nx >= 0, nx, Q))
        return 1 + d["bn"][after] + S - popneed + d["pat_need"][pat]

    def budget_mask(self, qg, r: Optional[int]):
        if not self.budget_active(r):
            return None
        d = self.dev
        nx, ax = self._columns(qg)
        cost = self._cost_after(qg, nx, ax)
        need_now = d["bn"][qg[0]] + qg[2 + STACK_DEPTH]
        m = (cost <= r) | (need_now > r)
        m[d["eos"]] = True
        return m

    def budget_ids(self, cfg, r: Optional[int], ids):
        if ids is None or not self.budget_active(r):
            return ids
        q, st = cfg
        st = list(st)
        d = self.dev
        need_now = int(d["bn_np"][q]) + sum(int(d["bn_np"][x]) for x in st)
        if need_now > r:
            return ids
        a = np.array(ids)
        keep = []
        for t in a.tolist():
            if t in self.eos_ids:
                keep.append(True)
                continue
            s2 = list(st)
            nq = self.c.advance(int(t), q, s2)
            need = int(d["bn_np"][nq]) + sum(int(d["bn_np"][x]) for x in s2) if nq != DEAD else 1 << 30
            keep.append(1 + need <= r)
        keep = np.array(keep, dtype=bool)
        return mx.array(a[keep].astype(np.int32)) if keep.any() else ids

    def heal_ok(self, r: int) -> bool:
        return True  # re-encoded spans still cover >= 1 byte per token

    # -- configurations as values (for verifying drafts, locals/lookup.py) ----------
    def config(self):
        return (self.qs, tuple(self.stack))

    def set_config(self, cfg):
        self.qs, st = cfg
        self.stack = list(st)

    def step_config(self, cfg, token: int):
        if token in self.eos_ids:
            return None
        q, st = cfg
        st = list(st)
        nq = self.c.advance(token, q, st)
        if nq == DEAD:
            return None
        if self.canonical:
            cov = self.planner.cover_rule(cfg)
            if cov and len(self.planner.vocab.token_bytes[token] or b"") < cov:
                return None
        return (nq, tuple(st))

    def device_state(self, cfg):
        q, st = cfg
        st = list(st)
        pad = st + [0] * (STACK_DEPTH - len(st))
        need = int(sum(int(self.dev["bn_np"][r]) for r in st))
        return mx.array([q, len(st)] + pad + [need], dtype=mx.int32)

    def finished_at(self, cfg) -> bool:
        return cfg[0] == self.c.pda.n


def _tok_len_np(vocab):
    k = ("toklen_np", id(vocab))
    if k not in _TABLES:
        _TABLES[k] = (vocab, np.array([len(b) if b else 0 for b in vocab.token_bytes], dtype=np.int32))
    return _TABLES[k][1]


def _may_force(c: PDAConstraint, planner) -> np.ndarray:
    """Per state (any stack): can an allowed token lead to a forced span?
    Conservative: a token that returns (RET) or depends on the stack counts."""
    k = ("pda_may_force", id(c), id(planner))
    if k not in _TABLES:
        Q = c.pda.n
        forced = np.array([bool(c.pda.forced_bytes(q, [])) for q in range(Q)] + [False])
        nx = c.next_state.astype(np.int64)
        tgt = np.where(nx >= 0, nx, Q)
        hit = (forced[tgt] | (nx == RET)).any(axis=0)  # (Q,)
        # tokens whose walk ends at a popped return state: forced chains there depend on the stack
        hit[c.ctx_states] = True
        _TABLES[k] = (c, hit)
    return _TABLES[k][1]


def _device_tables(c: PDAConstraint) -> dict:
    k = ("pda_dev", id(c))
    if k not in _TABLES:
        Q, V = c.pda.n, c.n_model
        allowed = np.concatenate([(c.next_state != DEAD).T, np.zeros((1, V), bool)])  # FIN row: EOS set on device
        eos = np.zeros(V, dtype=bool)
        eos[c.eos_ids] = True
        d = {
            "allowed": mx.array(allowed),
            "next": mx.array(np.concatenate([c.next_state, np.full((V, 1), DEAD, np.int16)], axis=1)),
            "aux": mx.array(np.concatenate([c.aux, np.zeros((V, 1), np.int16)], axis=1)),
            "patterns": mx.array(c.patterns),
            "pat_len": mx.array(c.pat_len.astype(np.int32)),
            "ctx_ids": mx.array(c.ctx_ids.astype(np.int32)),
            "ctx_index": mx.array(c.ctx_index),
            "qc_index": mx.array(np.append(c.qc_index, -1).astype(np.int32)),
            "ctx_next": mx.array(c.ctx_next) if len(c.ctx_ids) else None,
            "ctx_aux": mx.array(c.ctx_aux) if len(c.ctx_ids) else None,
            "n_ctx": int(len(c.ctx_ids)),
            "ridx": mx.array(c.pda.ridx),
            "accepting": mx.array(np.append(c.pda.accepting, True)),  # FIN accepts
            "eos": mx.array(np.nonzero(eos)[0].astype(np.int32)),
        }
        bn = _byte_need(c.pda)  # (Q+1,), FIN = 0
        d["bn_np"] = bn
        d["bn"] = mx.array(bn)
        d["bn_max"] = int(bn[bn < (1 << 28)].max()) if (bn < (1 << 28)).any() else 0
        pn = np.array([sum(int(bn[x]) for x in row if x >= 0) for row in c.patterns], dtype=np.int32)
        d["pat_need"] = mx.array(pn)
        _TABLES[k] = (c, d)
    return _TABLES[k][1]


def _byte_need(pda) -> np.ndarray:
    """Shortest byte completion of each state's nonterminal from that state:
    0 at an accepting state (it can return), else the cheapest byte edge, or a
    call (the callee's shortest completion plus the return state's). A
    Bellman-Ford fixpoint; recursion converges through the non-recursive
    alternative. FIN gets 0."""
    Q, INF = pda.n, 1 << 28
    bn = np.where(pda.accepting, 0, INF).astype(np.int64)
    table, starts = pda.table, pda.starts
    for _ in range(2 * Q + 2):
        nxt = table[:, :256]
        new = np.minimum(bn, np.where(nxt >= 0, 1 + bn[np.maximum(nxt, 0)], INF).min(axis=1))
        for i in range(pda.n_calls):
            col = table[:, 256 + i]
            cand = np.where(col >= 0, bn[starts[i]] + bn[np.maximum(col, 0)], INF)
            new = np.minimum(new, cand)
        new = np.where(pda.accepting, 0, new)
        if np.array_equal(new, bn):
            break
        bn = new
    return np.append(np.minimum(bn, INF), 0).astype(np.int32)


def prepare_cover(ctl: PDAController) -> None:
    """Per-state boundary-cover table for the device mask (chain inside the
    current nonterminal, an empty stack)."""
    if not ctl.canonical or "cover" in ctl.dev:
        return
    Q = ctl.c.pda.n
    cov = np.array([ctl.planner.cover_rule((q, ())) for q in range(Q)] + [0], dtype=np.int32)
    ctl.dev["cover"] = mx.array(cov)
