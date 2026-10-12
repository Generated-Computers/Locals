"""Recursive constraints as a deterministic pushdown automaton, with token
tables that stay GPU-resident.

A grammar (locals/grammar.py) gives one regex per nonterminal, with calls as
extra alphabet symbols. Each rule becomes a DFA; taking a call edge pushes the
caller's *return state* r and continues at the callee's start, and a callee
that reaches a terminal accepting state (no successors: the closing ``}`` or
``]``) pops and continues at r. An accepting state that still has successors
(a number alternative) pops lazily, when the next byte is not its own: the
compile checks that no byte is both a continuation of the callee and a
follower of a return state (LL(1)), so the machine is deterministic. This is
the capped deterministic AuxPDA of Thm 3.12 restricted to an explicit stack.

The step graph of a token from a configuration is still a single walk, so the
token table is again one universal-source pass over the vocabulary trie
(Construction 4.6), with rows that carry a symbolic stack: pushes made inside
the token are matched by pops inside it. At most MAX_POP *real* pops per token are
resolved (tokens popping deeper are masked, which loses nothing: ``}``
exists as a token of its own). A token ending exactly at a real pop is
context-free (RET: the GPU continues at the popped return state); a token
that continues past a pop depends on the popped return states, and those
entries live in a small context table indexed by the top MAX_POP stack
entries. Everything the engine needs per step is then a gather on
(state, stack top), and the stack itself is a fixed-depth device array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .grammar import Grammar, GrammarError, compile_grammar
from .regex import DEAD, compile_dfa
from .vocab import ByteTrie, TokenVocab

MAX_PUSH = 3  # pushes a single token may make (fail-closed beyond)
MAX_POP = 2  # real pops a single token may make (fail-closed beyond)
MAX_PATTERNS = 255
RET = -2  # next-state marker: continue at the stack top
STACK_DEPTH = 64  # nesting the device state holds; deeper pushes are masked (fail-closed)


@dataclass
class PDA:
    table: np.ndarray  # (Q, 256 + C) int32; column 256+i = call of nonterminal i, entry = return state
    accepting: np.ndarray  # (Q,) bool
    terminal: np.ndarray  # (Q,) accepting with no successors: pop eagerly
    nt_of: np.ndarray  # (Q,) nonterminal of each state
    starts: List[int]
    root: int
    returns: List[int]  # stack alphabet: return states, in a fixed order
    ridx: np.ndarray  # (Q+1,) index of a return state in ``returns``, C otherwise
    returns_of: List[List[int]]  # per nonterminal: the return states of its call sites
    eff_next: np.ndarray  # (Q, 256) next state with call edges resolved
    eff_push: np.ndarray  # (Q, 256) return state pushed by that byte, or -1

    @property
    def n(self) -> int:
        return self.table.shape[0]

    @property
    def n_calls(self) -> int:
        return self.table.shape[1] - 256

    @property
    def FIN(self) -> int:
        return self.n

    # -- host stepping on the real stack ---------------------------------------------
    def step(self, q: int, stack: List[int], b: int) -> int:
        """Consume byte b from (q, stack); mutates stack. Returns q' or DEAD."""
        if q == self.FIN:
            return DEAD
        while True:
            nq = int(self.eff_next[q, b])
            if nq != DEAD:
                pr = int(self.eff_push[q, b])
                if pr >= 0:
                    stack.append(pr)
                break
            if self.accepting[q] and stack:
                q = stack.pop()  # lazy pop, then retry the byte at the return state
                continue
            return DEAD
        while self.terminal[nq]:  # eager pops
            if stack:
                nq = stack.pop()
            else:
                return self.FIN
        return nq

    def run(self, data: bytes, q: int, stack: List[int]) -> int:
        """Consume ``data``; DEAD if any byte fails or the stack would exceed
        STACK_DEPTH (the device state's capacity; such pushes are masked)."""
        for b in data:
            q = self.step(q, stack, b)
            if q == DEAD or len(stack) > STACK_DEPTH:
                return DEAD
        return q

    def live_bytes(self, q: int, stack: List[int]) -> np.ndarray:
        """Bytes consumable from (q, stack): own edges, plus the followers of
        return states popped lazily."""
        if q == self.FIN:
            return np.zeros(256, dtype=bool)
        live = self.eff_next[q] != DEAD
        i = len(stack)
        while self.accepting[q] and i > 0:
            q = stack[i - 1]
            i -= 1
            live = live | (self.eff_next[q] != DEAD)
        return live

    def forced_bytes(self, q: int, stack: List[int]) -> bytes:
        """Maximal forced byte chain from (q, stack), across calls and pops:
        the unary vertices of the branch kernel (Thm A10.1)."""
        st = list(stack)
        out = bytearray()
        while q != self.FIN and len(out) < 4096:
            if self.accepting[q] and not st:
                break  # EOS is an option
            live = self.live_bytes(q, st)
            nz = np.nonzero(live)[0]
            if len(nz) != 1:
                break
            b = int(nz[0])
            q = self.step(q, st, b)
            if q == DEAD or len(st) > STACK_DEPTH:
                break
            out.append(b)
        return bytes(out)


def build_pda(g: Grammar, max_states: int = 200_000) -> PDA:
    C = len(g.rules)
    dfas = [compile_dfa(r, max_states, n_calls=C) for r in g.rules]
    starts, tables, acc, nt_of = [], [], [], []
    off = 0
    for i, d in enumerate(dfas):
        starts.append(off + d.start)
        t = d.table.copy()
        byte_cols = t[:, :256]
        t[:, :256] = np.where(byte_cols == DEAD, DEAD, byte_cols + off)
        call_cols = t[:, 256:]
        t[:, 256:] = np.where(call_cols == DEAD, DEAD, call_cols + off)
        tables.append(t)
        acc.append(d.accepting)
        nt_of.append(np.full(d.n, i, dtype=np.int32))
        off += d.n
    table = np.concatenate(tables)
    accepting = np.concatenate(acc)
    nt_of = np.concatenate(nt_of)
    Q = table.shape[0]
    has_succ = (table != DEAD).any(axis=1)
    terminal = accepting & ~has_succ
    # Call sites and return states.
    returns_of: List[List[int]] = [[] for _ in range(C)]
    call_sites = np.argwhere(table[:, 256:] != DEAD)
    returns = []
    for q, i in call_sites:
        r = int(table[q, 256 + i])
        returns.append(r)
        returns_of[int(i)].append(r)
    returns = sorted(set(returns))
    returns_of = [sorted(set(v)) for v in returns_of]
    ridx = np.full(Q + 1, len(returns), dtype=np.int32)
    for k, r in enumerate(returns):
        ridx[r] = k
    called = [bool(v) for v in returns_of]
    for i in range(C):
        if called[i]:
            s = starts[i]
            if (table[s, 256:] != DEAD).any():
                raise GrammarError(f"nonterminal {g.names[i]} starts with a call")
            if accepting[s]:
                raise GrammarError(f"nonterminal {g.names[i]} matches the empty string")
    for r in returns:
        if accepting[r]:
            raise GrammarError("a return state accepts: end-of-output would depend on the stack")
    # Effective byte transitions: a call edge on nonterminal i is taken on the
    # first bytes of i. Determinism: those bytes are disjoint from the state's
    # own byte edges and from other callees' first bytes.
    eff_next = table[:, :256].copy()
    eff_push = np.full((Q, 256), -1, dtype=np.int32)
    first = [table[starts[i], :256] != DEAD for i in range(C)]
    for q, i in call_sites:
        i = int(i)
        f = first[i]
        if (f & (eff_next[q] != DEAD)).any():
            raise GrammarError(f"ambiguous at state {q}: a byte both continues and calls {g.names[i]}")
        eff_next[q, f] = table[starts[i], :256][f]
        eff_push[q, f] = int(table[q, 256 + i])
    # Lazy pops: an accepting non-terminal state of a called nonterminal must
    # not share a live byte with any return state's followers.
    for i in range(C):
        if not called[i]:
            continue
        follow = np.zeros(256, dtype=bool)
        for r in returns_of[i]:
            follow |= eff_next[r] != DEAD
        for q in np.nonzero((nt_of == i) & accepting & ~terminal)[0]:
            if (follow & (eff_next[q] != DEAD)).any():
                raise GrammarError(f"ambiguous return at state {q} of {g.names[i]}")
    return PDA(table, accepting, terminal, nt_of, starts, starts[g.root], returns, ridx, returns_of, eff_next, eff_push)


# ---------------------------------------------------------------------------- token tables


@dataclass
class PDAConstraint:
    pda: PDA
    next_state: np.ndarray  # (V, Q) int16: q', RET, or DEAD
    aux: np.ndarray  # (V, Q) int16: pattern id (8 bits) | pops << 8
    patterns: np.ndarray  # (n_pat, MAX_PUSH) int32 pushed return states, -1 padded
    pat_len: np.ndarray  # (n_pat,)
    ctx_ids: np.ndarray  # tokens with context-dependent entries
    ctx_index: np.ndarray  # (V,) index into ctx_ids or -1
    ctx_states: np.ndarray  # states that have context-dependent tokens
    qc_index: np.ndarray  # (Q,) index into ctx_states or -1
    ctx_next: np.ndarray  # (n_ctx, n_qc, (C+1)**MAX_POP) int16; context = top1 * (C+1) + top2
    ctx_aux: np.ndarray  # same shape, int16
    eos_ids: List[int]
    n_model: int
    stats: dict = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return self.pda.n

    # -- host mirror of the device step ---------------------------------------------
    def ctx_of(self, stack: List[int]) -> int:
        C1 = len(self.pda.returns) + 1
        idx = 0
        for k in range(MAX_POP):
            top = int(self.pda.ridx[stack[-1 - k]]) if len(stack) > k else C1 - 1
            idx = idx * C1 + top
        return idx

    def entry(self, t: int, q: int, stack: List[int]) -> Tuple[int, int]:
        """(next, aux) for token t at (q, stack) from the tables."""
        if q == self.pda.n:
            return DEAD, 0
        ci, qi = int(self.ctx_index[t]), int(self.qc_index[q])
        if ci >= 0 and qi >= 0:
            j = self.ctx_of(stack)
            return int(self.ctx_next[ci, qi, j]), int(self.ctx_aux[ci, qi, j])
        return int(self.next_state[t, q]), int(self.aux[t, q])

    def allowed_ids(self, q: int, stack: List[int]) -> np.ndarray:
        """Allowed token ids at (q, stack) (host)."""
        if q == self.pda.n:
            return np.array(self.eos_ids, dtype=np.int64)
        nx, ax = self.next_state[:, q].copy(), self.aux[:, q].copy()
        qi = int(self.qc_index[q])
        if qi >= 0:
            j = self.ctx_of(stack)
            nx[self.ctx_ids] = self.ctx_next[:, qi, j]
            ax[self.ctx_ids] = self.ctx_aux[:, qi, j]
        pops = (ax >> 8) & 3
        h = len(stack)
        ok = (nx != DEAD) & ((pops <= h) | ((nx == RET) & (pops == h + 1)))
        plen = self.pat_len[ax & 0xFF]
        ok &= plen + h - pops <= STACK_DEPTH
        for e in self.eos_ids:
            ok[e] = bool(self.pda.accepting[q]) and not stack
        return np.nonzero(ok)[0]

    def advance(self, t: int, q: int, stack: List[int]) -> int:
        """Apply token t to (q, stack); mutates stack. Returns q' or DEAD."""
        if t in self.eos_ids:
            return q if (q == self.pda.n or (self.pda.accepting[q] and not stack)) else DEAD
        nx, ax = self.entry(t, q, stack)
        if nx == DEAD:
            return DEAD
        pat, pops = ax & 0xFF, (ax >> 8) & 3
        if nx == RET:
            if pops <= len(stack):
                for _ in range(pops - 1):
                    stack.pop()
                nx = stack.pop()
            elif pops == len(stack) + 1:  # the last pop is the root's end
                del stack[:]
                nx = self.pda.n
            else:
                return DEAD
        else:
            if pops > len(stack):
                return DEAD
            for _ in range(pops):
                stack.pop()
        for k in range(int(self.pat_len[pat])):
            stack.append(int(self.patterns[pat, k]))
        return nx if len(stack) <= STACK_DEPTH else DEAD


def _pattern_ids(P: np.ndarray, npush: np.ndarray, Q: int):
    key = np.zeros(len(P), dtype=np.int64)
    for k in range(MAX_PUSH):
        key = key * (Q + 2) + np.where(k < npush, P[:, k] + 1, 0)
    return key


def token_tables(pda: PDA, trie: ByteTrie, vocab: TokenVocab, work_budget: Optional[int] = None) -> PDAConstraint:
    Q, C = pda.n, len(pda.returns)
    C1 = C + 1
    V = vocab.n_model
    # FIN (index Q) gets a dead row, so rows that reached the root's end die on the next byte
    effn = np.concatenate([pda.eff_next, np.full((1, 256), DEAD, np.int32)])
    effp = np.concatenate([pda.eff_push, np.full((1, 256), -1, np.int32)])
    termx = np.append(pda.terminal, False)
    accx = np.append(pda.accepting, False)
    ntx = np.append(pda.nt_of, -1)
    # Candidate return states per nonterminal. The root's stack may also be
    # empty, so its pops may end the output (FIN).
    root_nt = int(pda.nt_of[pda.root])
    cand = [np.array(list(v) + ([pda.FIN] if i == root_nt or not v else []), dtype=np.int32) for i, v in enumerate(pda.returns_of)]
    DEADQ = np.int32(DEAD)
    levels = trie.tok_level

    class Rows:
        """Column arrays of the live rows: state, symbolic stack, and the
        real pops made so far (ctx[k] = index of the k-th popped return
        state, -1 = none yet)."""

        def __init__(self, n):
            self.node = np.zeros(n, np.int64)
            self.q0 = np.arange(n, dtype=np.int32)
            self.q = np.arange(n, dtype=np.int32)
            self.ctx = np.full((n, MAX_POP), -1, np.int32)
            self.pop = np.zeros(n, np.int8)
            self.post = np.zeros(n, np.int8)  # a byte consumed after the last real pop
            self.npush = np.zeros(n, np.int8)
            self.P = np.full((n, MAX_PUSH), -1, np.int32)
            self.b = np.zeros(n, np.int64)

        names = ("node", "q0", "q", "ctx", "pop", "post", "npush", "P", "b")

        def take(self, idx):
            for k in self.names:
                setattr(self, k, getattr(self, k)[idx])

        def append(self, other):
            for k in self.names:
                setattr(self, k, np.concatenate([getattr(self, k), getattr(other, k)]))

        def __len__(self):
            return len(self.node)

    def real_pop(R, idx, popped_q):
        """Rows idx pop a real stack entry while at state popped_q (an
        accepting state of its nonterminal). Returns the replacement rows,
        one per candidate return state, with R filtered to the rest; rows
        that already made MAX_POP real pops are dropped (fail-closed)."""
        can = R.pop[idx] < MAX_POP
        idx, popped_q = idx[can], popped_q[can]
        nts = ntx[popped_q]
        counts = np.array([len(cand[n]) if n >= 0 else 1 for n in nts], dtype=np.int64)
        rep = np.repeat(idx, counts)
        r_vals = np.concatenate([cand[n] if n >= 0 else np.array([pda.FIN], np.int32) for n in nts]) if len(nts) else np.zeros(0, np.int32)
        A = Rows(0)
        for k in A.names:
            setattr(A, k, getattr(R, k)[rep])
        k = R.pop[rep].astype(np.int64)
        A.ctx[np.arange(len(rep)), k] = pda.ridx[r_vals]
        A.pop = (R.pop[rep] + 1).astype(np.int8)
        A.post = np.zeros(len(rep), np.int8)
        A.npush = np.zeros(len(rep), np.int8)
        A.P = np.full((len(rep), MAX_PUSH), -1, np.int32)
        return A, r_vals, rep

    R = Rows(Q)
    n_prev_nodes = 1
    work = Q
    res = {k: [] for k in ("t", "q0", "ctx", "q", "pop", "post", "key")}
    for d in range(1, trie.depth + 1):
        par = trie.parent[d]
        order = np.argsort(R.node, kind="stable")
        R.take(order)
        counts = np.bincount(R.node, minlength=n_prev_nodes)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        cnt = counts[par]
        total = int(cnt.sum())
        if total == 0:
            break
        work += total
        if work_budget is not None and work > work_budget:
            raise RuntimeError(f"product work {work} exceeds budget {work_budget}")
        base = np.repeat(np.cumsum(cnt) - cnt, cnt)
        src = np.repeat(starts[par], cnt) + (np.arange(total) - base)
        R.take(src)
        R.node = np.repeat(np.arange(len(par)), cnt)
        R.b = trie.byte[d][R.node].astype(np.int64)
        # --- consume one byte
        qn = effn[R.q, R.b]
        pr = effp[R.q, R.b]
        cur = R.q.copy()  # the state that consumed the byte, after lazy pops
        for _ in range(MAX_PUSH + 1):
            sym = (qn == DEADQ) & accx[cur] & (R.npush > 0)
            if not sym.any():
                break
            i = np.nonzero(sym)[0]
            rq = R.P[i, R.npush[i] - 1]
            R.npush[i] -= 1
            cur[i] = rq
            qn[i] = effn[rq, R.b[i]]
            pr[i] = effp[rq, R.b[i]]
        lazy = (qn == DEADQ) & accx[cur] & (R.npush == 0)
        if lazy.any():
            i = np.nonzero(lazy)[0]
            A, r_vals, rep = real_pop(R, i, cur[i])
            A.post[:] = 1  # the byte is consumed after the pop
            keep = ~lazy
            R.take(keep)
            qn = np.concatenate([qn[keep], effn[r_vals, A.b]])
            pr = np.concatenate([pr[keep], effp[r_vals, A.b]])
            R.append(A)
        alive = qn != DEADQ
        # pushes
        pushing = alive & (pr >= 0)
        over = pushing & (R.npush >= MAX_PUSH)
        alive &= ~over
        i = np.nonzero(pushing & ~over)[0]
        R.P[i, R.npush[i]] = pr[i]
        R.npush[i] += 1
        R.post = np.where(R.pop > 0, 1, R.post).astype(np.int8)
        # eager pops at terminal states
        for _ in range(MAX_PUSH + MAX_POP + 1):
            t_ = alive & (qn >= 0) & termx[np.maximum(qn, 0)]
            if not t_.any():
                break
            sym = t_ & (R.npush > 0)
            i = np.nonzero(sym)[0]
            qn[i] = R.P[i, R.npush[i] - 1]
            R.npush[i] -= 1
            real = t_ & ~sym
            if real.any():
                i = np.nonzero(real)[0]
                A, r_vals, rep = real_pop(R, i, qn[i])
                keep = ~real
                R.take(keep)
                qn = np.concatenate([qn[keep], r_vals])
                alive = np.concatenate([alive[keep], np.ones(len(r_vals), bool)])
                R.append(A)
        R.q = qn
        # --- record tokens ending at this level (several tokens may share a node)
        sel = np.nonzero(levels == d)[0]
        if len(sel):
            sn = trie.tok_index[sel]
            o = np.argsort(sn, kind="stable")
            sel, sn = sel[o], sn[o]
            live = np.nonzero(alive)[0]
            lo = np.searchsorted(sn, R.node[live], "left")
            hi = np.searchsorted(sn, R.node[live], "right")
            cnt_t = hi - lo
            has = cnt_t > 0
            if has.any():
                live, lo, cnt_t = live[has], lo[has], cnt_t[has]
                rows = np.repeat(live, cnt_t)
                tk = sel[np.repeat(lo, cnt_t) + (np.arange(int(cnt_t.sum())) - np.repeat(np.cumsum(cnt_t) - cnt_t, cnt_t))]
                res["t"].append(tk)
                res["q0"].append(R.q0[rows])
                res["ctx"].append(R.ctx[rows])
                res["q"].append(R.q[rows])
                res["pop"].append(R.pop[rows])
                res["post"].append(R.post[rows])
                res["key"].append(_pattern_ids(R.P[rows], R.npush[rows], Q))
        R.take(alive)
        n_prev_nodes = len(par)
        if len(R) == 0:
            break
    # --- assemble
    t_all = np.concatenate(res["t"]) if res["t"] else np.zeros(0, np.int64)
    q0_all, ctx_all, q_all = np.concatenate(res["q0"]), np.concatenate(res["ctx"]), np.concatenate(res["q"])
    pop_all, post_all, key_all = np.concatenate(res["pop"]), np.concatenate(res["post"]), np.concatenate(res["key"])
    ukeys, pat_all = np.unique(key_all, return_inverse=True)
    pat_all = pat_all.reshape(-1)
    if len(ukeys) > MAX_PATTERNS:
        raise GrammarError(f"{len(ukeys)} distinct push patterns; at most {MAX_PATTERNS} are supported")
    patterns = np.full((len(ukeys), MAX_PUSH), -1, np.int32)
    pat_len = np.zeros(len(ukeys), np.int32)
    for j, k in enumerate(ukeys):
        vals = []
        for _ in range(MAX_PUSH):
            vals.append(int(k % (Q + 2)) - 1)
            k //= Q + 2
        vals = [v for v in reversed(vals) if v >= 0]
        patterns[j, : len(vals)] = vals
        pat_len[j] = len(vals)
    if Q + 2 >= 2**15:  # the tables are int16, and RET / DEAD are negative: more states would wrap into them
        raise GrammarError(f"the grammar has {Q} states; at most {2**15 - 3} are supported")
    NC0 = (C + 1) ** MAX_POP
    if V * Q * 4 / 1e9 > 3 or len(set(t_all[(pop_all > 0)].tolist())) * NC0 * Q * 4 / 1e9 > 3:
        raise GrammarError("the grammar's token tables would take more than 3 GB; make the schema smaller")
    aux_all = (pat_all | (pop_all.astype(np.int64) << 8)).astype(np.int16)
    next_state = np.full((V, Q), DEAD, np.int16)
    aux = np.zeros((V, Q), np.int16)
    # case A: no real pop
    a = pop_all == 0
    next_state[t_all[a], q0_all[a]] = q_all[a]
    aux[t_all[a], q0_all[a]] = aux_all[a]
    # case B: token ends exactly at its single real pop (continue at the popped return state)
    bm = (pop_all == 1) & (post_all == 0)
    next_state[t_all[bm], q0_all[bm]] = RET
    aux[t_all[bm], q0_all[bm]] = aux_all[bm]
    # case C: continues past a real pop, or pops twice (the bytes between the
    # pops ran at the first popped state): depends on the popped return states
    cm = (pop_all > 0) & ~bm
    ctx_ids = np.unique(t_all[cm])
    ctx_index = np.full(V, -1, np.int32)
    ctx_index[ctx_ids] = np.arange(len(ctx_ids))
    ctx_states = np.unique(q0_all[cm])
    qc_index = np.full(Q, -1, np.int32)
    qc_index[ctx_states] = np.arange(len(ctx_states))
    NC = C1 ** MAX_POP
    ctx_next = np.full((len(ctx_ids), len(ctx_states), NC), DEAD, np.int16)
    ctx_aux = np.zeros((len(ctx_ids), len(ctx_states), NC), np.int16)
    if len(ctx_ids):
        # context-free verdicts first, so the override is consistent for every (t, q)
        ctx_next[:, :, :] = next_state[ctx_ids][:, ctx_states][:, :, None]
        ctx_aux[:, :, :] = aux[ctx_ids][:, ctx_states][:, :, None]
        # a token that continues past a pop is never allowed context-free
        next_state[t_all[cm], q0_all[cm]] = DEAD
        aux[t_all[cm], q0_all[cm]] = 0
        ci, qi = ctx_index[t_all[cm]], qc_index[q0_all[cm]]
        ctx1 = ctx_all[cm, 0]
        # the entries below the popped ones are unconstrained: broadcast over them
        for j in range(NC):
            top1, top2 = j // C1, j % C1
            m = ci >= 0
            m &= ctx1 == top1
            m &= np.where(pop_all[cm] >= 2, ctx_all[cm, 1] == top2, True)
            ends = post_all[cm][m] == 0  # ends exactly at the last pop: RET
            ctx_next[ci[m], qi[m], j] = np.where(ends, RET, q_all[cm][m])
            ctx_aux[ci[m], qi[m], j] = aux_all[cm][m]
        # rows also void the broadcast base for (t, q0) pairs handled here
    for e in vocab.eos_ids:
        next_state[e, :] = np.where(pda.accepting, np.arange(Q), DEAD).astype(np.int16)  # EOS: stack must be empty (device check)
        aux[e, :] = 0
    stats = {"states": Q, "returns": C, "patterns": len(ukeys), "ctx_tokens": len(ctx_ids), "ctx_states": len(ctx_states),
             "product_work": int(work), "entries": int(len(t_all))}
    return PDAConstraint(pda, next_state, aux, patterns, pat_len, ctx_ids, ctx_index, ctx_states, qc_index,
                         ctx_next, ctx_aux, list(vocab.eos_ids), V, stats)


def compile_pda_constraint(schema, vocab: TokenVocab, trie: ByteTrie, max_states: int = 200_000,
                           work_budget: Optional[int] = None) -> PDAConstraint:
    import time

    t0 = time.perf_counter()
    g = compile_grammar(schema)
    pda = build_pda(g, max_states)
    t1 = time.perf_counter()
    c = token_tables(pda, trie, vocab, work_budget)
    c.stats.update({"t_pda": t1 - t0, "t_table": time.perf_counter() - t1, "grammar": g.stats})
    return c
