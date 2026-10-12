"""The library API: compile a grammar once for a tokenizer, then use it in any
decode loop, or in ours.

    import locals

    vocab = locals.Vocabulary.from_tokenizer(tokenizer)      # mlx-lm TokenizerWrapper or HF tokenizer
    compiler = locals.GrammarCompiler(vocab)
    grammar = compiler.compile_json_schema(schema)          # or compile_regex / compile_tool_calls

    matcher = grammar.matcher()
    logits = matcher.apply_token_mask(logits)               # forbid every token the grammar rejects
    matcher.accept_token(token)                             # advance after sampling
    forced = matcher.jump_forward()                         # tokens that are now certain: feed them, do not sample

With mlx-lm's own generator the grammar is a logits processor::

    from mlx_lm import generate
    generate(model, tokenizer, prompt, logits_processors=[locals.logits_processor(grammar)])

With our engine (forced spans fed in the same forward pass, GPU-resident
masks, batching) use :class:`locals.Runtime` or the server.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Union

import mlx.core as mx

from .constraint import compile_constraint
from .engine import NEG_INF, ConstraintExhausted
from .forced import ForcedPlanner
from .runtime import Compiled, Runtime
from .vocab import ByteTrie, TokenVocab

__all__ = ["Vocabulary", "GrammarCompiler", "CompiledGrammar", "Matcher", "logits_processor", "load"]


def _unwrap(tokenizer):
    """The Hugging Face (fast) tokenizer behind an mlx-lm TokenizerWrapper."""
    return getattr(tokenizer, "_tokenizer", tokenizer)


@dataclass
class Vocabulary:
    """What a grammar needs to know about a tokenizer: the byte string of every
    token, arranged as a trie, and the end-of-sequence ids. Build it once per
    tokenizer and share it between grammars."""

    tokens: TokenVocab
    trie: ByteTrie
    hf: Any  # the Hugging Face tokenizer (its pre-tokenizer defines the canonical token split)

    @property
    def n_tokens(self) -> int:
        """Width of the masks: the logits width this vocabulary was built for."""
        return self.tokens.n_model

    @property
    def eos_ids(self) -> List[int]:
        return list(self.tokens.eos_ids)

    @classmethod
    def from_tokenizer(cls, tokenizer, vocab_size: Optional[int] = None, eos_ids: Optional[Sequence[int]] = None) -> "Vocabulary":
        """``tokenizer``: an mlx-lm ``TokenizerWrapper`` or a Hugging Face tokenizer.
        ``vocab_size``: the model's logits width (defaults to the tokenizer's
        size; masks are padded to the logits they are applied to anyway)."""
        hf = _unwrap(tokenizer)
        if eos_ids is None:
            eos = set(getattr(tokenizer, "eos_token_ids", None) or [])
            if getattr(hf, "eos_token_id", None) is not None:
                eos.add(hf.eos_token_id)
            eos_ids = sorted(eos)
        n = int(vocab_size or len(hf))
        tokens = TokenVocab.from_hf(hf, n, list(eos_ids))
        return cls(tokens, ByteTrie.build(tokens), hf)

    @classmethod
    def from_model(cls, model, tokenizer) -> "Vocabulary":
        """Like :meth:`from_tokenizer`, with the logits width read off the model."""
        from .kernels import text_model

        n = text_model(model).model.embed_tokens.weight.shape[0]
        return cls.from_tokenizer(tokenizer, vocab_size=n)


@dataclass
class CompiledGrammar(Compiled):
    """A grammar compiled against a vocabulary. ``constraint`` holds the token
    tables (regular grammars: a deterministic automaton; recursive JSON
    schemas: a deterministic pushdown), ``planner`` the forced-span planner."""

    kind: str = ""  # "regex", "json_schema", "tool_calls"
    source: Any = None  # the regex, schema or tool list it was compiled from
    vocab: Optional[Vocabulary] = None

    @property
    def n_states(self) -> int:
        return self.constraint.pda.n if self.recursive else self.constraint.n_states

    @property
    def vocab_size(self) -> int:
        return self.vocab.n_tokens if self.vocab is not None else self.constraint.next_state.shape[0]

    @property
    def table_bytes(self) -> int:
        """Memory of the token tables (next state and mask) for this grammar."""
        return int(self.constraint.next_state.nbytes + self.constraint.next_state.size // 8) if not self.recursive else \
            int(self.constraint.next_state.nbytes + self.constraint.aux.nbytes + self.constraint.ctx_next.nbytes + self.constraint.ctx_aux.nbytes)

    def matcher(self, forced: bool = True, canonical: bool = True) -> "Matcher":
        """A fresh :class:`Matcher` at the start of the grammar."""
        return Matcher(self, forced=forced, canonical=canonical)


class GrammarCompiler:
    """Compiles JSON schemas, regular expressions and tool-call formats into
    grammars for one vocabulary. Compiled grammars are cached by their source."""

    def __init__(self, vocab: Union[Vocabulary, Any], max_dfa_states: int = 200_000, work_budget: Optional[int] = None):
        self.vocab = vocab if isinstance(vocab, Vocabulary) else Vocabulary.from_tokenizer(vocab)
        self.max_dfa_states = max_dfa_states
        self.work_budget = work_budget
        self._cache: dict = {}

    # -- entry points -------------------------------------------------------------------------
    def compile_regex(self, regex: str) -> CompiledGrammar:
        """A regular expression over the output text. Matching is byte-level:
        non-ASCII characters and classes are expanded to UTF-8; ``\\d \\w \\s``
        are ASCII; lookarounds and named groups are not supported."""
        return self._regex(regex, "regex", regex)

    def compile_json_schema(self, schema: Union[dict, str], whitespace: str = "") -> CompiledGrammar:
        """A JSON schema. ``whitespace`` is the pattern allowed between tokens of
        the JSON text ("" = compact output). Schemas whose definitions refer to
        themselves (a ``$ref`` on a cycle) compile to a pushdown automaton."""
        from outlines_core.json_schema import build_regex_from_schema

        from .grammar import is_recursive, json_safe_schema

        s = json_safe_schema(schema if isinstance(schema, str) else json.dumps(schema))
        if is_recursive(s):
            key = ("pda", s)
            if key not in self._cache:
                from .pda import compile_pda_constraint

                cc = compile_pda_constraint(s, self.vocab.tokens, self.vocab.trie, self.max_dfa_states, self.work_budget)
                self._cache[key] = CompiledGrammar(cc, self._planner(cc), recursive=True, kind="json_schema", source=s, vocab=self.vocab)
            return self._cache[key]
        return self._regex(build_regex_from_schema(s, whitespace), "json_schema", s)

    def compile_tool_calls(self, tools: List[dict], tool_choice: Any = "auto", typed: bool = False, thinking: bool = False) -> CompiledGrammar:
        """Replies that may call ``tools`` (OpenAI ``{"type": "function", "function": {...}}``
        specs) in Qwen's native ``<tool_call>`` format: free text, then calls
        as ``tool_choice`` demands ("auto", "required", one named function, or
        "none": text that never opens a call).
        ``typed`` fixes every parameter's name and type (large automata: use
        with the walk kernel). ``thinking`` starts inside an open ``<think>`` block."""
        from . import oai

        regex = oai.reply_regex(tools, tool_choice=tool_choice, thinking=thinking, typed=typed)
        if regex is None:
            raise ValueError("no tools to call")
        return self._regex(regex, "tool_calls", tools)

    def clear_cache(self) -> None:
        """Forget the compiled grammars (they are cached by their source)."""
        self._cache.clear()

    # -- internals ---------------------------------------------------------------------------
    def _planner(self, constraint) -> Optional[ForcedPlanner]:
        if getattr(self.vocab.hf, "backend_tokenizer", None) is None:
            return None  # a slow tokenizer has no pre-tokenizer to define the canonical split
        return ForcedPlanner(self.vocab.hf, self.vocab.tokens, constraint)

    def _regex(self, regex: str, kind: str, source) -> CompiledGrammar:
        key = ("regex", regex)
        if key not in self._cache:
            cc = compile_constraint(regex, self.vocab.tokens, self.vocab.trie, self.max_dfa_states, self.work_budget)
            self._cache[key] = CompiledGrammar(cc, self._planner(cc), kind=kind, source=source, vocab=self.vocab)
        return self._cache[key]


class Matcher:
    """Tracks one output under a grammar: which tokens may come next, which
    tokens are already certain, and whether the output is complete.

    The protocol per step: ``apply_token_mask`` the logits, sample, then
    ``accept_token``; then feed the tokens of ``jump_forward`` (if any) to the
    model without sampling them. ``forced=False`` turns jump-forward off;
    ``canonical`` keeps forced text in the tokenizer's own split and requires
    the next sampled token to cover a held-back block.
    """

    def __init__(self, grammar: CompiledGrammar, forced: bool = True, canonical: bool = True):
        self.grammar = grammar
        self._forced, self._canonical = forced, canonical
        self.reset()

    def reset(self) -> None:
        """Back to the start of the grammar."""
        self.ctl = self.grammar.controller(use_forced=self._forced, canonical=self._canonical)
        self._terminated = False
        self._exhausted = False
        self._last: Optional[int] = None

    # -- inspection -----------------------------------------------------------------------
    @property
    def eos_ids(self) -> List[int]:
        return list(self.ctl.eos_ids)

    @property
    def state(self):
        """The grammar's configuration: a state, or (state, stack) for a pushdown."""
        return self.ctl.q

    def is_terminated(self) -> bool:
        """An end-of-sequence token was accepted, or no token is allowed any more."""
        return self._terminated or self._exhausted

    def is_completed(self) -> bool:
        """The output so far is a complete member of the language (EOS is allowed now)."""
        if self._terminated:
            return True
        if self._exhausted:
            return False
        if self.ctl.finished():
            return True
        m = self.next_token_mask()
        return any(bool(m[e].item()) for e in self.eos_ids)

    # -- masks ----------------------------------------------------------------------------
    def next_token_mask(self, tokens_left: Optional[int] = None) -> mx.array:
        """Boolean array over the vocabulary: True where a token keeps the output
        valid. ``tokens_left``: with a token budget, also require that a valid
        ending still fits after the token (the output is then never cut mid-value)."""
        qg = self.ctl.gpu_state()
        m = self.ctl.gpu_mask(qg)
        if tokens_left is not None:
            bm = self.ctl.budget_mask(qg, tokens_left)
            if bm is not None:
                m = m & bm
        return m

    def apply_token_mask(self, logits: mx.array, tokens_left: Optional[int] = None) -> mx.array:
        """``logits`` with every forbidden token set to -inf (shape (V,) or (..., V))."""
        if self.is_terminated():
            return logits
        m = self.next_token_mask(tokens_left)
        V = logits.shape[-1]
        if m.shape[0] < V:
            m = mx.concatenate([m, mx.zeros((V - m.shape[0],), dtype=mx.bool_)])
        elif m.shape[0] > V:
            m = m[:V]
        return mx.where(m, logits, NEG_INF)

    # -- advancing ------------------------------------------------------------------------
    def accept_token(self, token: int, check: bool = True) -> bool:
        """Advance past ``token``. Returns False, without changing the state, if the
        grammar does not allow it (``check=False`` skips the mask lookup for
        tokens that were sampled from masked logits)."""
        token = int(token)
        if self.is_terminated():
            return False
        if check or token in self.ctl.eos_ids:
            if not bool(self.next_token_mask()[token].item()):
                return False
        if token in self.ctl.eos_ids:
            self._terminated = True
            return True
        try:
            self.ctl.accept(token)
        except ConstraintExhausted:
            self._exhausted = True
            return False
        except RuntimeError:
            return False
        self._last = token
        return True

    def jump_forward(self) -> List[int]:
        """The tokens that must follow the last accepted token, in the
        tokenizer's own split (or [] if the model has a choice). They are
        accepted as a side effect: feed them to the model, do not sample them."""
        if self._last is None or self.is_terminated():
            return []
        extra, _ = self.ctl.forced(self._last, heal=False)
        self._last = None
        return list(extra)

    def fork(self) -> "Matcher":
        """An independent copy at the same state (for beams or drafts)."""
        other = copy.copy(self)
        other.ctl = copy.copy(self.ctl)
        if hasattr(other.ctl, "stack"):
            other.ctl.stack = list(self.ctl.stack)
        return other


def logits_processor(grammar: CompiledGrammar, canonical: bool = False):
    """A logits processor for mlx-lm's ``generate`` / ``stream_generate`` /
    ``BatchGenerator``: ``processor(tokens, logits) -> logits``. Masks only
    (mlx-lm samples one token per step, so forced text is not skipped there);
    ``canonical`` additionally requires sampled tokens to cover held-back
    pre-token blocks. The matcher is ``processor.matcher``. One processor serves one sequence at
    a time: a call whose context is not longer than the last one's starts a new sequence (a second
    ``generate`` with the same processor). A token the grammar does not allow raises ValueError (it
    used to switch the constraint off without a word). mlx-lm's ``max_tokens`` can still cut the
    output before the grammar's end (no token budget here; the server and ``Runtime`` have one). Not for
    speculative decoding (mlx-lm's draft_model): the drafts' rewinds look like new sequences."""
    state = {"n": None}

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        n = int(tokens.shape[-1])
        if state["n"] is None or n != state["n"] + 1:  # a new sequence: its context ends with the last prompt token
            processor.matcher = Matcher(grammar, forced=False, canonical=canonical)
        elif not processor.matcher.is_terminated():
            tok = int(tokens[-1].item())
            if not processor.matcher.accept_token(tok, check=False):
                raise ValueError(f"token {tok} is not allowed by the grammar here (was the processor shared between "
                                 "sequences, or the logits changed after masking?)")
        state["n"] = n
        return processor.matcher.apply_token_mask(logits)

    processor.matcher = Matcher(grammar, forced=False, canonical=canonical)
    return processor


def load(model: str, **kw) -> Runtime:
    """Load a model with our engine: ``locals.load("mlx-community/Qwen3.5-4B-MLX-4bit")``.
    Keyword arguments are those of :class:`locals.Runtime`."""
    return Runtime.load(model, **kw)
