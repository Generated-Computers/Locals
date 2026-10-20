"""Regex -> byte-level NFA -> DFA.

Supports the syntax JSON-schema compilers emit: literals and escapes, classes
with ranges and negation, ``.``, groups, alternation, and the quantifiers
``* + ? {m} {m,} {m,n}``. Code points are compiled to UTF-8 byte sequences, so
the automaton reads the same bytes as the tokenizer's vocabulary trie.
Shorthand classes (``\\d \\w \\s``) are ASCII.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

MAX_CP = 0x10FFFF
MAX_REPEAT = 10_000  # a larger count would build the automaton for minutes before any state budget applies
Ranges = List[Tuple[int, int]]


# ----------------------------------------------------------------------------
# Character sets as sorted disjoint code-point ranges


def normalize(r: Ranges) -> Ranges:
    r = sorted(r)
    out: Ranges = []
    for lo, hi in r:
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def complement(r: Ranges) -> Ranges:
    out: Ranges = []
    prev = 0
    for lo, hi in normalize(r):
        if lo > prev:
            out.append((prev, lo - 1))
        prev = hi + 1
    if prev <= MAX_CP:
        out.append((prev, MAX_CP))
    return out


DIGIT: Ranges = [(0x30, 0x39)]
WORD: Ranges = [(0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A)]
SPACE: Ranges = [(0x09, 0x0D), (0x20, 0x20)]
DOT: Ranges = complement([(0x0A, 0x0A)])


# ----------------------------------------------------------------------------
# AST


@dataclass
class Chars:
    ranges: Ranges


@dataclass
class Cat:
    items: List


@dataclass
class Alt:
    items: List


@dataclass
class Rep:
    item: object
    lo: int
    hi: Optional[int]  # None = unbounded


class RegexError(ValueError):
    pass


class _Parser:
    def __init__(self, s: str):
        self.s = s
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.s[self.i] if self.i < len(self.s) else None

    def take(self) -> str:
        if self.i >= len(self.s):
            raise RegexError("unexpected end of pattern")
        c = self.s[self.i]
        self.i += 1
        return c

    def parse(self):
        node = self.alt()
        if self.i != len(self.s):
            raise RegexError(f"unexpected {self.s[self.i]!r} at {self.i}")
        return node

    def alt(self):
        items = [self.cat()]
        while self.peek() == "|":
            self.take()
            items.append(self.cat())
        return items[0] if len(items) == 1 else Alt(items)

    def cat(self):
        items = []
        while self.peek() not in (None, "|", ")"):
            items.append(self.quantified())
        return Cat(items)

    def quantified(self):
        atom = self.atom()
        while True:
            c = self.peek()
            if c == "*":
                self.take()
                atom = Rep(atom, 0, None)
            elif c == "+":
                self.take()
                atom = Rep(atom, 1, None)
            elif c == "?":
                self.take()
                atom = Rep(atom, 0, 1)
            elif c == "{" and self._looks_like_count():
                self.take()
                lo = self._int() if self.peek() != "," else 0  # {,n} is {0,n}, as Python reads it
                hi: Optional[int] = lo
                if self.peek() == ",":
                    self.take()
                    hi = None if self.peek() == "}" else self._int()
                if self.take() != "}":
                    raise RegexError("bad repetition")
                if hi is not None and hi < lo:
                    raise RegexError(f"bad repetition {{{lo},{hi}}}: the maximum is below the minimum")
                if lo > MAX_REPEAT or (hi or 0) > MAX_REPEAT:
                    raise RegexError(f"repetition counts above {MAX_REPEAT} are not supported")
                atom = Rep(atom, lo, hi)
            else:
                return atom
            if self.peek() == "?":  # lazy modifier: same language
                self.take()

    def _looks_like_count(self) -> bool:
        j = self.i + 1
        s = self.s
        k = j
        while k < len(s) and s[k].isdigit():
            k += 1
        if k == j:  # {,n}
            if k < len(s) and s[k] == ",":
                k += 1
                m = k
                while k < len(s) and s[k].isdigit():
                    k += 1
                return k > m and k < len(s) and s[k] == "}"
            return False
        if k < len(s) and s[k] == "}":
            return True
        if k < len(s) and s[k] == ",":
            k += 1
            while k < len(s) and s[k].isdigit():
                k += 1
            return k < len(s) and s[k] == "}"
        return False

    def _int(self) -> int:
        j = self.i
        while self.peek() is not None and self.peek().isdigit():
            self.i += 1
        return int(self.s[j : self.i])

    def atom(self):
        c = self.take()
        if c == "(":
            if self.s.startswith("?:", self.i):
                self.i += 2
            elif self.peek() == "?":
                raise RegexError("lookaround and named groups are not supported")
            node = self.alt()
            if self.take() != ")":
                raise RegexError("unbalanced parenthesis")
            return node
        if c == "[":
            return Chars(self.cls())
        if c == ".":
            return Chars(DOT)
        if c == "\\":
            return Chars(self.escape(in_class=False))
        if c in "^$":
            return Cat([])  # anchors: matching is always whole-string
        return Chars([(ord(c), ord(c))])

    def escape(self, in_class: bool) -> Ranges:
        c = self.take()
        simple = {"n": 10, "t": 9, "r": 13, "f": 12, "v": 11, "0": 0}
        if c in simple:
            return [(simple[c], simple[c])]
        if c in "xuU":
            n = {"x": 2, "u": 4, "U": 8}[c]
            digits = self.s[self.i : self.i + n]
            try:
                v = int(digits, 16)
            except ValueError:
                raise RegexError(f"bad escape \\{c}{digits}") from None
            if len(digits) != n or v > MAX_CP and not (c == "U" and v >= CALL_BASE):
                raise RegexError(f"bad escape \\{c}{digits}")
            self.i += n
            return [(v, v)]
        if c == "d":
            return DIGIT
        if c == "D":
            return complement(DIGIT)
        if c == "w":
            return WORD
        if c == "W":
            return complement(WORD)
        if c == "s":
            return SPACE
        if c == "S":
            return complement(SPACE)
        if c == "b" and in_class:
            return [(8, 8)]
        if c.isalnum():
            raise RegexError(f"unsupported escape \\{c}")
        return [(ord(c), ord(c))]

    def cls(self) -> Ranges:
        neg = False
        if self.peek() == "^":
            self.take()
            neg = True
        ranges: Ranges = []
        first = True
        while True:
            c = self.take()
            if c == "]" and not first:
                break
            first = False
            if c == "\\":
                r = self.escape(in_class=True)
            else:
                r = [(ord(c), ord(c))]
            if len(r) == 1 and r[0][0] == r[0][1] and self.peek() == "-" and self.s[self.i + 1 : self.i + 2] not in ("]", ""):
                self.take()
                d = self.take()
                hi = self.escape(in_class=True) if d == "\\" else [(ord(d), ord(d))]
                if len(hi) != 1 or hi[0][0] != hi[0][1] or hi[0][0] < r[0][0]:
                    raise RegexError("bad class range")
                ranges.append((r[0][0], hi[0][0]))
            else:
                ranges.extend(r)
        ranges = normalize(ranges)
        return complement(ranges) if neg else ranges


def parse(pattern: str):
    return _Parser(pattern).parse()


# ----------------------------------------------------------------------------
# UTF-8 byte sequences for a code-point range


def _utf8(cp: int) -> bytes:
    return chr(cp).encode("utf-8", "surrogatepass")


def utf8_sequences(lo: int, hi: int) -> List[List[Tuple[int, int]]]:
    """Split [lo, hi] into byte-range sequences whose product is exactly the
    UTF-8 encodings of the range (surrogates excluded)."""
    out = []
    stack = [(lo, hi)]
    while stack:
        s, e = stack.pop()
        if s > e:
            continue
        if s <= 0xDFFF and e >= 0xD800:
            stack.append((s, 0xD7FF))
            stack.append((0xE000, e))
            continue
        split = False
        for mx in (0x7F, 0x7FF, 0xFFFF):
            if s <= mx < e:
                stack.append((s, mx))
                stack.append((mx + 1, e))
                split = True
                break
        if split:
            continue
        if e <= 0x7F:
            out.append([(s, e)])
            continue
        for i in (1, 2, 3):
            m = (1 << (6 * i)) - 1
            if (s & ~m) != (e & ~m):
                if s & m:
                    stack.append((s, s | m))
                    stack.append(((s | m) + 1, e))
                    split = True
                    break
                if (e & m) != m:
                    stack.append((s, (e & ~m) - 1))
                    stack.append((e & ~m, e))
                    split = True
                    break
        if split:
            continue
        bs, be = _utf8(s), _utf8(e)
        out.append(list(zip(bs, be)))
    return out


# ----------------------------------------------------------------------------
# Thompson NFA over bytes


class NFA:
    def __init__(self):
        self.eps: List[List[int]] = []
        self.byte: List[List[Tuple[int, int, int]]] = []  # (lo, hi, target)
        self.start = 0
        self.accept = 0

    def new(self) -> int:
        self.eps.append([])
        self.byte.append([])
        return len(self.eps) - 1

    @property
    def n(self) -> int:
        return len(self.eps)


# Code points from CALL_BASE up denote calls of nonterminals (locals/pda.py):
# CALL_BASE + i is nonterminal i, written in a pattern as the escape
# \U00200000 + i. It becomes a single edge on symbol 256 + i, a column of the
# DFA table beyond the 256 bytes.
CALL_BASE = 0x200000


def _build(nfa: NFA, node, s: int, t: int) -> None:
    if isinstance(node, Chars):
        for lo, hi in node.ranges:
            if lo >= CALL_BASE:
                for sym in range(lo, hi + 1):
                    nfa.byte[s].append((256 + sym - CALL_BASE, 256 + sym - CALL_BASE, t))
        for seq in [q for lo, hi in node.ranges if lo < CALL_BASE for q in utf8_sequences(lo, min(hi, CALL_BASE - 1))]:
            cur = s
            for j, (blo, bhi) in enumerate(seq):
                nxt = t if j == len(seq) - 1 else nfa.new()
                nfa.byte[cur].append((blo, bhi, nxt))
                cur = nxt
        return
    if isinstance(node, Cat):
        if not node.items:
            nfa.eps[s].append(t)
            return
        cur = s
        for j, item in enumerate(node.items):
            nxt = t if j == len(node.items) - 1 else nfa.new()
            _build(nfa, item, cur, nxt)
            cur = nxt
        return
    if isinstance(node, Alt):
        for item in node.items:
            a, b = nfa.new(), nfa.new()
            nfa.eps[s].append(a)
            _build(nfa, item, a, b)
            nfa.eps[b].append(t)
        return
    if isinstance(node, Rep):
        cur = s
        for _ in range(node.lo):
            nxt = nfa.new()
            _build(nfa, node.item, cur, nxt)
            cur = nxt
        if node.hi is None:
            a, b = nfa.new(), nfa.new()
            nfa.eps[cur].append(a)
            _build(nfa, node.item, a, b)
            nfa.eps[b].append(a)
            nfa.eps[a].append(t)
        else:
            for _ in range(node.hi - node.lo):
                nxt = nfa.new()
                nfa.eps[cur].append(t)
                _build(nfa, node.item, cur, nxt)
                cur = nxt
            nfa.eps[cur].append(t)
        return
    raise TypeError(node)


def compile_nfa(pattern: str) -> NFA:
    nfa = NFA()
    s, t = nfa.new(), nfa.new()
    nfa.start, nfa.accept = s, t
    _build(nfa, parse(pattern), s, t)
    return nfa


# ----------------------------------------------------------------------------
# DFA by subset construction, with a fail-closed state budget


DEAD = -1


class BudgetExceeded(ValueError):
    """The automaton is too large to compile (a request error: the grammar asks too much)."""


@dataclass
class DFA:
    table: np.ndarray  # (Q, 256) int32, DEAD for no transition
    accepting: np.ndarray  # (Q,) bool
    start: int

    @property
    def n(self) -> int:
        return self.table.shape[0]

    def run(self, data: bytes, q: Optional[int] = None) -> int:
        q = self.start if q is None else q
        for b in data:
            if q == DEAD:
                return DEAD
            q = int(self.table[q, b])
        return q

    def matches(self, data: bytes) -> bool:
        q = self.run(data)
        return q != DEAD and bool(self.accepting[q])


def _closure(nfa: NFA, states) -> FrozenSet[int]:
    seen = set(states)
    stack = list(states)
    while stack:
        u = stack.pop()
        for v in nfa.eps[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return frozenset(seen)


def determinize(nfa: NFA, max_states: int = 200_000, width: int = 256) -> DFA:
    """Subset construction. Stops fail-closed at ``max_states`` (the same cap
    discipline as Thm 4.2: a bounded construction either completes or reports
    that the budget is violated). ``width`` = 256 + number of call symbols."""
    start = _closure(nfa, [nfa.start])
    ids: Dict[FrozenSet[int], int] = {start: 0}
    order = [start]
    rows: List[np.ndarray] = []
    i = 0
    while i < len(order):
        S = order[i]
        i += 1
        edges = [e for u in S for e in nfa.byte[u]]
        row = np.full(width, DEAD, dtype=np.int32)
        if edges:
            cuts = sorted({lo for lo, _, _ in edges} | {hi + 1 for _, hi, _ in edges})
            for a, b in zip(cuts, cuts[1:]):
                targets = [t for lo, hi, t in edges if lo <= a and b - 1 <= hi]
                if not targets:
                    continue
                T = _closure(nfa, targets)
                j = ids.get(T)
                if j is None:
                    if len(order) >= max_states:
                        raise BudgetExceeded(f"DFA exceeds {max_states} states")
                    j = len(order)
                    ids[T] = j
                    order.append(T)
                row[a:b] = j
        rows.append(row)
    table = np.stack(rows)
    accepting = np.array([nfa.accept in S for S in order], dtype=bool)
    return DFA(table, accepting, 0)


def coreachable(dfa: DFA) -> np.ndarray:
    """Backward cone of the accepting states (the backward half of Cor. 4.12)."""
    Q = dfa.n
    rev: List[List[int]] = [[] for _ in range(Q)]
    for q in range(Q):
        for t in np.unique(dfa.table[q]):
            if t != DEAD:
                rev[int(t)].append(q)
    live = dfa.accepting.copy()
    stack = list(np.nonzero(live)[0])
    while stack:
        u = stack.pop()
        for v in rev[u]:
            if not live[v]:
                live[v] = True
                stack.append(v)
    return live


_ROW_MIX = np.random.default_rng(0).integers(1, 1 << 62, size=4096, dtype=np.int64) | 1


def _row_classes(sig: np.ndarray) -> np.ndarray:
    """A class id per row of ``sig``, equal exactly for equal rows: rows are
    hashed to one integer (a sort of integers, not of whole rows), and the
    classes are checked against the rows; a collision falls back to sorting rows."""
    mix = _ROW_MIX[:sig.shape[1]] if sig.shape[1] <= len(_ROW_MIX) else None
    if mix is not None:
        with np.errstate(over="ignore"):
            h = (sig * mix).sum(axis=1)  # wraps around: a hash
        _, first, inv = np.unique(h, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        if np.array_equal(sig, sig[first[inv]]):
            return inv
    _, inv = np.unique(sig, axis=0, return_inverse=True)
    return inv.reshape(-1)


class TooManyStates(BudgetExceeded):
    """The minimal DFA has more states than asked for."""


def trim_and_minimize(dfa: DFA, max_states: Optional[int] = None) -> DFA:
    """Drop states that cannot reach acceptance, then Moore minimization. ``max_states``: fail as soon as the
    refinement has more classes (they only grow, and end as the minimal DFA's states): minimizing a long chain
    takes a round per link (a string with maxLength 3000, 66k states: 55 s) for a table that was refused after."""
    live = coreachable(dfa)
    if not live[dfa.start]:
        raise ValueError("the pattern matches nothing")
    keep = np.nonzero(live)[0]
    remap = np.full(dfa.n + 1, DEAD, dtype=np.int32)
    remap[keep] = np.arange(len(keep), dtype=np.int32)
    t = dfa.table[keep]
    t = np.where(t == DEAD, dfa.n, t)  # index n maps to DEAD
    table = remap[t]
    acc = dfa.accepting[keep]
    start = int(remap[dfa.start])
    # Moore refinement: class id per state, with DEAD as its own class. Bytes
    # that every state sends to the same place (most of them) are one column.
    Q = len(keep)
    seen, first = set(), []  # (np.unique(table, axis=1) held the GIL 1.3 s on 132k states: every other thread stopped)
    for b in range(table.shape[1]):
        col = table[:, b].tobytes()
        if col not in seen:
            seen.add(col)
            first.append(b)
    narrow = table[:, first]
    cls = acc.astype(np.int64)
    while True:
        succ = np.where(narrow == DEAD, -1, cls[np.maximum(narrow, 0)])
        sig = np.concatenate([cls[:, None], succ], axis=1)
        new = _row_classes(sig)
        if max_states is not None and new.max() + 1 > max_states:
            raise TooManyStates(f"the minimal DFA has more than {max_states} states")
        if new.max() + 1 == len(np.unique(cls)):
            break
        cls = new
    _, labels = np.unique(cls, return_inverse=True)
    labels = labels.reshape(-1)
    ncls = int(labels.max()) + 1
    swap = np.arange(ncls)
    s0 = labels[start]
    swap[[0, s0]] = swap[[s0, 0]]
    labels = swap[labels]
    new_table = np.full((ncls, table.shape[1]), DEAD, dtype=np.int32)
    new_acc = np.zeros(ncls, dtype=bool)
    for q in range(Q):
        row = table[q]
        new_table[labels[q]] = np.where(row == DEAD, DEAD, labels[np.maximum(row, 0)])
        new_acc[labels[q]] = acc[q]
    return DFA(new_table, new_acc, 0)


def compile_dfa(pattern: str, max_states: int = 200_000, n_calls: int = 0, max_minimal: Optional[int] = None) -> DFA:
    return trim_and_minimize(determinize(compile_nfa(pattern), max_states, 256 + n_calls), max_minimal)
