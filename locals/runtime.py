"""High-level runtime: load a model once, generate with or without a schema.

    rt = Runtime.load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    out = rt.generate("Extract ...", schema={...})
    samples = rt.sample("Summarize ...", n=4)
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import LRUPromptCache

from .constraint import compile_constraint
from .engine import Engine, GenResult, LocalsController
from .forced import ForcedPlanner
from .fuse import fuse_model
from .kernels import enable_small_m, text_model
from .tree import TreeResult, sample_branches
from .vocab import ByteTrie, TokenVocab


@dataclass
class Compiled:
    constraint: object  # CompiledConstraint, or pda.PDAConstraint for a recursive schema
    planner: ForcedPlanner
    recursive: bool = False

    def controller(self, use_forced: bool = True, canonical: bool = True):
        if self.recursive:
            from .pdactl import PDAController

            return PDAController(self.constraint, self.planner, use_forced=use_forced, canonical=canonical)
        return LocalsController(self.constraint, self.planner, use_forced=use_forced, canonical=canonical)


def _prepare_device_tables(c) -> None:
    """Build the per-constraint tables decoding needs (mask, transitions,
    may-force, boundary cover, token budgets) at compile time; built lazily they cost ~0.25 s
    per schema inside the first request."""
    from .engine import LocalsController, _may_force

    if c.recursive:
        from .pdactl import PDAController, _may_force as _pda_may_force

        ctl = PDAController(c.constraint, c.planner, True, True)
        _pda_may_force(c.constraint, c.planner)
        mx.eval(ctl.dev["allowed"], ctl.dev["next"], ctl.dev["aux"], ctl.dev["cover"])
        return
    ctl = LocalsController(c.constraint, c.planner, True, True)
    _may_force(c.constraint)
    mx.eval(ctl._gpu_T(), ctl._gpu_cover(), ctl.allowed, *ctl.budget_tables()[1])


def native_bf16() -> bool:
    """Apple GPUs before the M3 generation (g13 = M1, g14 = M2) emulate bf16
    arithmetic; there MLX's fp16 kernels run up to 1.7x faster than bf16."""
    arch = mx.device_info().get("architecture", "")
    return not (arch.startswith("applegpu_g13") or arch.startswith("applegpu_g14"))


def _fp16_special(model) -> bool:
    """Whether the model's own code treats fp16 specially (Gemma 3 clips every
    residual add through fp32 in fp16): such a model stays in bf16, where it
    runs its plain path (and faster: 68 against 53 tok/s for Gemma 3 4B on an M2)."""
    import inspect
    import sys

    from .kernels import text_model

    try:
        src = inspect.getsource(sys.modules[type(text_model(model)).__module__])
    except (OSError, TypeError, KeyError):
        return False
    return "float16" in src


def set_compute_dtype(model, dtype: Optional[str] = "auto") -> str:
    """Cast the model's bf16 arrays (activation-dtype weights, quantization
    scales and biases) to ``dtype``; float32 arrays such as A_log are kept.
    "auto" picks float16 on GPUs without native bf16. On Qwen3.5 checkpoints
    fp16 is also closer to a float32 run than bf16 is.
    Returns the dtype in use."""
    from mlx.utils import tree_flatten

    params = [v for _, v in tree_flatten(model.parameters())]
    has_bf16 = any(v.dtype == mx.bfloat16 for v in params)
    auto = dtype == "auto"
    if auto:
        dtype = "float16" if has_bf16 and not native_bf16() and not _fp16_special(model) else None
    if dtype in (None, "bfloat16") or not has_bf16:
        return "bfloat16" if has_bf16 else "float16"
    ref = None
    if auto:
        # fp16 holds every bf16 value up to 65504 exactly, so the cast is undone exactly if the check fails
        if any(float(mx.max(mx.abs(v))) > 6.0e4 for v in params if v.dtype == mx.bfloat16):
            return "bfloat16"
        probe = mx.array([[(i * 7919) % 2000 + 100 for i in range(12)]])
        ref = model(probe)[0, -1].astype(mx.float32)
    saved = [(k, v) for k, v in tree_flatten(model.parameters()) if v.dtype == mx.bfloat16] if ref is not None else []
    model.set_dtype(getattr(mx, dtype), predicate=lambda t: t == mx.bfloat16)
    if ref is not None:
        out = model(probe)[0, -1].astype(mx.float32)
        ok = bool(mx.all(mx.isfinite(out))) and int(mx.argmax(out)) == int(mx.argmax(ref)) \
            and float(mx.max(mx.abs(out - ref))) <= 0.05 * float(mx.max(mx.abs(ref))) + 0.5
        if not ok:  # the model's activations do not fit fp16 (or it computes differently in it): its own arrays back
            from mlx.utils import tree_unflatten

            model.update(tree_unflatten(saved))  # exactly the bf16 arrays it had (a cast back loses small values)
            return "bfloat16"
    return dtype


class Runtime:
    """A loaded model with our decode loop.

    ``tune`` controls the one-time autotuning of the small-batch matrix kernels
    (results are cached on disk per GPU, so it runs once per model shape):
    ``"auto"`` uses the cache or a shipped preset when they cover the model and
    otherwise tunes the common batch sizes (a few minutes); ``"full"`` also
    times whole decode steps for every batch size up to 32 (10--20 minutes,
    the best throughput); ``"quick"`` forces the short procedure; ``"off"``
    tunes nothing ahead of time. The environment variable ``LOCALS_TUNE``
    sets the default for the server and the command line.
    """

    def __init__(self, model, tokenizer, fuse: bool = True, small_m_kernel: bool = True, prompt_cache_bytes: int = 2 << 30,
                 dtype: Optional[str] = "auto", tune: Optional[str] = None, log=None):
        self.model = model
        self.tokenizer = tokenizer
        self.dtype = set_compute_dtype(model, dtype)
        if fuse:
            fuse_model(model)
        enable_small_m(small_m_kernel, model)
        if small_m_kernel:
            from .kernels import tune_model

            tune_model(model, mode=tune or os.environ.get("LOCALS_TUNE", "auto"), log=log)
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        hf = tokenizer._tokenizer
        body = getattr(text_model(model), "model", None)
        if getattr(body, "embed_tokens", None) is None:
            raise ValueError(f"{type(text_model(model)).__module__.split('.')[-1]} models are not supported: locals runs the "
                             "decoder families with model.embed_tokens (Llama, Qwen, Gemma, Mistral, Phi-3 and the like)")
        n_model = body.embed_tokens.weight.shape[0]
        eos = sorted(set(tokenizer.eos_token_ids) | {i for i in (hf.eos_token_id,) if i is not None})
        self.vocab = TokenVocab.from_hf(hf, n_model, eos)
        self.trie = ByteTrie.build(self.vocab)
        self.engine = Engine(model, tokenizer)
        # Compiled constraints, least recently used first. A token-table constraint holds a
        # (states x vocabulary) table on the GPU: 60-100 MB for a 40-state regex on Qwen3.5's 248k
        # vocabulary. Kept forever, a few dozen distinct regexes (locals solve --plan put the file
        # path in each part's) filled 4-5 GB, and the batch's memory check then admitted one
        # request at a time. A running request keeps its own constraint alive.
        self._compiled: "OrderedDict[str, Compiled]" = OrderedDict()
        self.max_compiled = int(os.environ.get("LOCALS_MAX_COMPILED", "16"))
        # Radix trie of previously fed token sequences -> KV (mlx-lm's LRU
        # prompt cache). Our KV caches never mutate arrays in place, so a hit
        # shares arrays instead of copying them.
        self.prompt_cache = LRUPromptCache(max_size=16, max_bytes=prompt_cache_bytes)

    @classmethod
    def load(cls, path: str, **kw) -> "Runtime":
        from .defaults import model_id

        model, tok = mlx_load(model_id(path))  # short names too: "qwen", "gemma"
        return cls(model, tok, **kw)

    def compile(self, schema: Optional[Union[dict, str]] = None, regex: Optional[str] = None, whitespace: str = "") -> Compiled:
        if regex is None:
            from outlines_core.json_schema import build_regex_from_schema

            from .grammar import is_recursive, json_safe_schema

            s = json_safe_schema(schema if isinstance(schema, str) else json.dumps(schema))
            if is_recursive(s):
                # Recursive schemas (a $ref on a cycle) get a pushdown automaton
                # (locals/pda.py) instead of a depth-unrolled regex.
                key = ("pda", s)  # typed keys: a regex text never looks like a schema's
                c = self._cget(key)
                if c is None:
                    from .pda import compile_pda_constraint

                    cc = compile_pda_constraint(s, self.vocab, self.trie)
                    c = Compiled(cc, ForcedPlanner(self.tokenizer._tokenizer, self.vocab, cc), recursive=True)
                    self._cput(key, c)
                    _prepare_device_tables(c)
                return c
            regex = build_regex_from_schema(s, whitespace)
        c = self._cget(("regex", regex))
        if c is None:
            cc = compile_constraint(regex, self.vocab, self.trie)
            c = Compiled(cc, ForcedPlanner(self.tokenizer._tokenizer, self.vocab, cc))
            self._cput(("regex", regex), c)
            _prepare_device_tables(c)
        return c

    def compile_walk(self, schema: Optional[Union[dict, str]] = None, regex: Optional[str] = None, whitespace: str = ""):
        """A constraint for the walk kernel (no token table): the automaton and
        a planner, cached by regex. Regular constraints only."""
        if regex is None:
            from outlines_core.json_schema import build_regex_from_schema

            from .grammar import json_safe_schema

            s = json_safe_schema(schema if isinstance(schema, str) else json.dumps(schema))
            regex = build_regex_from_schema(s, whitespace)
        key = ("walk", regex)
        c = self._cget(key)
        if c is None:
            from .walk import compile_walk

            c = compile_walk(self, regex)
            self._cput(key, c)
        return c

    def _cget(self, key: tuple):
        c = self._compiled.get(key)
        if c is not None:
            self._compiled.move_to_end(key)
        return c

    @staticmethod
    def _cbytes(c) -> int:
        """About what a compiled constraint holds: its token tables on the host and the GPU (the
        allowed-token mask and the transition table, about 7 bytes a state a token), or its automaton."""
        cc = c.constraint if hasattr(c, "constraint") else getattr(c, "wc", c)
        n = getattr(cc, "n_states", None) or getattr(getattr(cc, "dfa", None), "n", 0) or 0
        nt = getattr(getattr(cc, "next_state", None), "shape", (0,))[0] or 0
        return int(n * nt * 7) if nt else int(n * 256 * 4)

    def _cput(self, key: tuple, c) -> None:
        """Compiled constraints kept least recently used: at most ``max_compiled`` and ``LOCALS_COMPILED_GB``
        (1.5) of tables. A long-running server saw many grammars (a client's tool sets, response
        formats): sixteen big ones held gigabytes the prompt cache then had to give up. A request in
        flight keeps its own constraint alive."""
        self._compiled[key] = c
        self._compiled.move_to_end(key)
        limit = float(os.environ.get("LOCALS_COMPILED_GB", "1.5")) * 1e9
        while len(self._compiled) > max(1, self.max_compiled) or (
                len(self._compiled) > 1 and sum(self._cbytes(v) for v in self._compiled.values()) > limit):
            self._compiled.popitem(last=False)

    def encode_chat(self, prompt: str, system: Optional[str] = None) -> List[int]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)

    def generate(
        self,
        prompt: Union[str, List[int]],
        schema: Optional[Union[dict, str]] = None,
        regex: Optional[str] = None,
        max_tokens: int = 512,
        temp: float = 0.0,
        seed: int = 0,
        forced: bool = True,
        canonical: bool = True,
        chat: bool = True,
        reuse_prompt: bool = True,
        grammar: Optional[Compiled] = None,
    ) -> GenResult:
        """Decode ``prompt`` (a string, rendered through the chat template unless
        ``chat=False``, or token ids). The output follows ``schema`` (JSON Schema),
        ``regex``, or a compiled ``grammar`` (``locals.GrammarCompiler``) if given.
        ``forced``: feed text the grammar forces in the same pass; ``canonical``:
        keep it in the tokenizer's own split."""
        ids = prompt if isinstance(prompt, list) else (self.encode_chat(prompt) if chat else self.tokenizer.encode(prompt))
        ctl = None
        if grammar is not None:
            ctl = grammar.controller(use_forced=forced, canonical=canonical)
        elif schema is not None or regex is not None:
            c = self.compile(schema, regex)
            ctl = c.controller(use_forced=forced, canonical=canonical)
        cache = None
        if reuse_prompt:
            cache, _ = self.prompt_cache.fetch_nearest_cache("model", ids)
        res = self.engine.generate(ids, ctl, max_tokens=max_tokens, temp=temp, seed=seed, cache=cache)
        if reuse_prompt and res.cache is not None:
            self.prompt_cache.insert_cache("model", list(res.fed), res.cache)
        return res

    def sample(self, prompt: Union[str, List[int]], n: int, max_tokens: int = 256, temp: float = 0.8, seed: int = 0, chat: bool = True) -> TreeResult:
        ids = prompt if isinstance(prompt, list) else (self.encode_chat(prompt) if chat else self.tokenizer.encode(prompt))
        layers = getattr(text_model(self.model).model, "layers", [])
        if any(getattr(l, "self_attn", None) is None for l in layers):  # recurrent layers (Qwen3.5): no tree attention
            raise ValueError("sample() shares the prompt through tree attention, which needs attention in every layer; "
                             f"{type(text_model(self.model)).__module__.split('.')[-1]} has recurrent layers: use "
                             "locals.BatchService (or the server) with n requests of the same prompt instead")
        if len(ids) < 2:
            raise ValueError("sample() needs a prompt of at least two tokens")
        return sample_branches(self.model, self.tokenizer, ids, n, max_tokens=max_tokens, temp=temp, seed=seed)
