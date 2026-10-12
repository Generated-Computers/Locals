"""Turn forced byte chains into forced token spans.

A forced byte string w is certain under the constraint, but many token
sequences spell it. We emit the canonical one. For GPT-style BPE, tokens never
cross pre-token boundaries, so those boundaries are cut vertices of the
tokenization lattice and the canonical count factors over blocks (Thm A17.1)
with exactly one path per block.

Only the last block can be unstable: bytes after w may merge into it. We test
every byte the constraint allows next; if none can merge, the last block is
committed too. Otherwise it is held back, and the next step requires a token
that covers the whole block (the canonical boundary rule), which rules out
non-canonical prefix tokens such as ``":`` followed by ``"To``.

Healing: the most recent sampled token has not been fed to the model yet, so
if the canonical split merges it with the start of w we re-encode its bytes
together with w instead of feeding it alone. No KV rollback is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constraint import CompiledConstraint
from .regex import DEAD
from .vocab import TokenVocab


class DFAAutomaton:
    """What the planner needs from a constraint, for a DFA: configurations
    are state ids."""

    def __init__(self, c: CompiledConstraint):
        self.c = c

    def forced_bytes(self, q: int) -> bytes:
        return self.c.forced_bytes[q]

    cover_bytes = forced_bytes

    def run(self, w: bytes, q: int) -> int:
        return self.c.dfa.run(w, q)

    def live_bytes(self, q: int) -> np.ndarray:
        return self.c.dfa.table[q, :256] != DEAD


class PDAAutomaton:
    """Configurations are (state, stack tuple). Forced chains and stability
    use the real stack; the boundary-cover rule uses the chain inside the
    current nonterminal only (an empty stack), so it matches the per-state
    device table."""

    def __init__(self, c):
        self.c = c
        self.pda = c.pda

    def forced_bytes(self, cfg) -> bytes:
        q, st = cfg
        return self.pda.forced_bytes(q, list(st))

    def cover_bytes(self, cfg) -> bytes:
        return self.pda.forced_bytes(cfg[0], [])

    def run(self, w: bytes, cfg):
        q, st = cfg
        st = list(st)
        q2 = self.pda.run(w, q, st)
        return DEAD if q2 == DEAD else (q2, tuple(st))

    def live_bytes(self, cfg) -> np.ndarray:
        q, st = cfg
        return self.pda.live_bytes(q, list(st))

# Representatives for non-ASCII next characters: letter, punctuation, CJK,
# symbol, non-ASCII digit, non-ASCII space.
_NON_ASCII_REPS = ["é", "—", "中", "€", "٣", " "]


@dataclass
class ForcedPlan:
    tokens: List[int]  # tokens to feed after (or instead of) the sampled token
    replaces_prev: bool  # True: tokens re-encode the sampled token's bytes
    q_end: int  # constraint state after the committed bytes
    committed_forced_bytes: int
    cover: int  # the next token must be at least this many bytes (0 = no rule)
    # q_end is a configuration: a DFA state, or (state, stack) for a PDA


def _valid_utf8_prefix(b: bytes) -> int:
    for cut in range(len(b), max(-1, len(b) - 4), -1):
        try:
            b[:cut].decode("utf-8")
            return cut
        except UnicodeDecodeError:
            continue
    return 0


class ForcedPlanner:
    def __init__(
        self,
        hf_tokenizer,
        vocab: TokenVocab,
        constraint,
        heal: bool = True,
        commit_stable_tail: bool = True,
        boundary_rule: bool = True,
    ):
        self.tok = hf_tokenizer
        self.pre = hf_tokenizer.backend_tokenizer.pre_tokenizer
        self.vocab = vocab
        self.c = constraint
        self.auto = PDAAutomaton(constraint) if hasattr(constraint, "pda") else DFAAutomaton(constraint)
        self.heal = heal
        self.commit_stable_tail = commit_stable_tail
        self.boundary_rule = boundary_rule
        self._cache: Dict[Tuple[Optional[int], object], Optional[ForcedPlan]] = {}
        self._cover_cache: Dict[object, int] = {}
        self._chain_end: Dict[object, object] = {}
        # Added tokens that are not special (Qwen's <tool_call>, </tool_call>,
        # </think>, ...) are matched by the tokenizer *before* pre-tokenization,
        # as atoms. They are therefore blocks of their own in a forced string,
        # whatever the pre-tokenizer would do with their characters, and a
        # block that is an added token is always stable.
        try:
            special = set(hf_tokenizer.all_special_ids)
            self._added = sorted((t for t, i in hf_tokenizer.get_added_vocab().items() if i not in special and t), key=len, reverse=True)
        except Exception:
            self._added = []

    def _end_state(self, cfg):
        e = self._chain_end.get(cfg)
        if e is None:
            w = self.auto.forced_bytes(cfg)
            e = self._chain_end[cfg] = self.auto.run(w, cfg) if w else cfg
        return e

    def _added_spans(self, text: str) -> List[Tuple[int, int]]:
        """Occurrences of added tokens in ``text`` (character offsets), longest first, non-overlapping."""
        spans: List[Tuple[int, int]] = []
        for t in self._added:
            start = 0
            while True:
                i = text.find(t, start)
                if i < 0:
                    break
                if all(i >= b or i + len(t) <= a for a, b in spans):
                    spans.append((i, i + len(t)))
                start = i + 1
        return sorted(spans)

    def _piece_starts(self, s: bytes) -> List[int]:
        text = s.decode("utf-8")
        starts = {a for _, (a, _) in self.pre.pre_tokenize_str(text)}
        for a, b in self._added_spans(text):
            starts = {x for x in starts if x <= a or x >= b} | {a} | ({b} if b < len(text) else set())
        return [len(text[:a].encode("utf-8")) for a in sorted(starts)]

    def _ends_with_added(self, text: str) -> bool:
        return any(text.endswith(t) for t in self._added)

    def _tail_is_stable(self, w: bytes, end_cfg) -> bool:
        """True when no byte the constraint allows after w can merge into w's
        last pre-token block."""
        text = w.decode("utf-8")
        if self._ends_with_added(text):
            return True  # added tokens are split off before anything else
        if text[-1:].isspace():
            return False  # whitespace blocks use lookahead in the pre-tokenizer
        nxt = np.nonzero(self.auto.live_bytes(end_cfg))[0]
        chars = [chr(b) for b in nxt if b < 0x80]
        if any(b >= 0x80 for b in nxt):
            chars += _NON_ASCII_REPS
        n = len(text)
        for ch in chars:
            pieces = self.pre.pre_tokenize_str(text + ch)
            if not any(a == n for _, (a, _) in pieces):
                return False
        return True

    def plan(self, prev_token: Optional[int], q) -> Optional[ForcedPlan]:
        key = (prev_token if self.heal else None, q)
        if key not in self._cache:
            self._cache[key] = self._plan(prev_token, q)
        return self._cache[key]

    def cover_rule(self, q) -> int:
        """Minimum byte length of the next token at configuration q (0 = no rule).

        Applies when the remaining forced bytes at q form one pre-token block
        that is a single vocabulary token and does not end in whitespace.
        """
        if not self.boundary_rule:
            return 0
        if q in self._cover_cache:
            return self._cover_cache[q]
        w = self.auto.cover_bytes(q)
        r = 0
        if w:
            cut = _valid_utf8_prefix(w)
            if cut == len(w):
                text = w.decode("utf-8")
                if not text[-1].isspace() and len(self._piece_starts(w)) == 1:
                    ids = self.tok.encode(text, add_special_tokens=False)
                    # one token, and that token's bytes are these (a SentencePiece tokenizer encodes "rue" as
                    # "▁rue": a cover of 3 bytes would then allow no token at all, and empty the mask)
                    if len(ids) == 1 and self.vocab.token_bytes[ids[0]] == w:
                        r = len(w)
        self._cover_cache[q] = r
        return r

    def _plan(self, prev_token: Optional[int], q) -> Optional[ForcedPlan]:
        w = self.auto.forced_bytes(q)
        if not w:
            return None
        ctx = b""
        if self.heal and prev_token is not None:
            ctx = self.vocab.token_bytes[prev_token] or b""
        s = ctx + w
        cut = _valid_utf8_prefix(s)
        if cut <= len(ctx):
            return None
        s = s[:cut]
        starts = self._piece_starts(s)
        end = starts[-1]
        if (
            self.commit_stable_tail
            and cut == len(ctx) + len(w)
            and self._tail_is_stable(s, self._end_state(q))
        ):
            end = len(s)
        if end <= len(ctx):
            return None
        boundaries = set(starts)
        replaces_prev = bool(ctx) and len(ctx) not in boundaries
        begin = 0 if replaces_prev else len(ctx)
        seg = s[begin:end]
        ids = self.tok.encode(seg.decode("utf-8"), add_special_tokens=False)
        if b"".join(self.vocab.token_bytes[i] or b"\x00" for i in ids) != seg:
            return None
        committed = end - len(ctx)
        q_end = self.auto.run(w[:committed], q)
        if q_end == DEAD:
            return None
        return ForcedPlan(ids, replaces_prev, q_end, committed, self.cover_rule(q_end))
