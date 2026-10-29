"""The server's generation service for a block-diffusion model (locals/blockdiff.py), with BatchService's interface.

Requests run one at a time: a block-diffusion step is already 32 rows of matrix products, which keep this GPU's
arithmetic busy, so rows of other requests beside them would not come cheaper. Each resolved block's tokens go to the
request's ``on_token`` as they come. The state after a reply (its prompt and every reply token) is kept, and a next
prompt that starts with those tokens (an agent's next turn) only reads what it adds. Grammars (JSON schemas, regexes)
are refused: the model writes a block of tokens at once, not one token after another a grammar can steer. The stock
engine (``mlx/``) runs the model's own computation; ours adds Locals's kernels (locals/clover_fast.py).
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import replace
from typing import Callable, List, Optional, Tuple

from .service import Job

REFUSED = ("invalid: this model writes a block of tokens at a time (block diffusion) and cannot follow a JSON schema "
           "or a regular expression")


class BlockDiffusionService:
    constrains = False  # (the server adds no tool-call grammar for it)

    def __init__(self, engine: str, model: str, slots: int = 16, temp: float = 0.0, seed: int = 0,
                 cache_gb: Optional[float] = None, walk: bool = False, top_p: float = 0.0, top_k: int = 0,
                 block_size: Optional[int] = None):
        self.engine, self.model_id, self.seed = engine, model, seed
        self.block_size = block_size  # (None: blockdiff.load's, MAC_BLOCK)
        self._q: "queue.Queue[Job]" = queue.Queue()
        self._closed = False
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._job: Optional[Job] = None
        self.tokenizer = None
        self.active = self.tokens_out = self.prompt_tokens = self.cached_tokens = 0
        self.steps = 0
        self._kept: Tuple[List[int], object] = ([], None)  # (tokens, the cache holding them)
        self._thread = threading.Thread(target=self._main, name=f"blockdiff-{engine}", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            raise self._error

    # -- client API (BatchService's) ---------------------------------------------------------------
    def submit(self, prompt: List[int], schema: Optional[dict] = None, max_tokens: int = 512, regex: Optional[str] = None,
               on_token: Optional[Callable[[List[int]], None]] = None, checkpoints=None, temp: Optional[float] = None,
               ignore_eos: bool = False, top_p: Optional[float] = None, top_k: Optional[int] = None,
               seed: Optional[int] = None) -> Job:
        if not prompt:
            raise ValueError("empty prompt")
        job = Job(list(prompt), schema, max_tokens, regex, t_submit=time.perf_counter(), on_token=on_token, temp=temp,
                  ignore_eos=ignore_eos, top_p=top_p, top_k=top_k, seed=seed)
        if schema is not None or regex is not None:
            self._finish(job, [], REFUSED)
        elif self._closed:
            self._finish(job, [], "error: the server is shutting down")
        else:
            self._q.put(job)
        return job

    def prime(self, tokens: List[int]) -> None:
        pass

    def close(self) -> None:
        self._closed = True
        self._thread.join(timeout=30)

    def context_fit(self) -> Optional[int]:
        """Prompt + reply tokens that fit under the GPU limit next to the weights: the attention layers' keys and
        values a token, with 2.5 GB for a block's and a prompt chunk's activations."""
        import mlx.core as mx

        a = getattr(self, "model", None)
        if a is None:
            return None
        args = a.args
        n_attn = sum(t == "full_attention" for t in args.layer_types)
        per_token = n_attn * 2 * args.num_key_value_heads * args.head_dim * 2
        limit = mx.device_info()["max_recommended_working_set_size"]
        free = limit - self.weights_bytes - (5 << 29)
        return int(free // per_token) // 1024 * 1024 if free > 0 else None

    def stats(self) -> dict:
        return {"engine": self.engine, "active": self.active, "tokens_out": self.tokens_out,
                "prompt_tokens": self.prompt_tokens, "cached_tokens": self.cached_tokens, "denoising_steps": self.steps}

    # -- worker ------------------------------------------------------------------------------------
    def _finish(self, job: Job, tokens: List[int], finish: str) -> None:
        job.tokens = list(tokens)
        job.text = self.tokenizer.decode(job.tokens) if self.tokenizer is not None else ""
        job.finish = finish
        job.t_done = time.perf_counter()
        job._event.set()

    def _main(self) -> None:
        try:
            import mlx.core as mx

            from . import blockdiff
            from .fuse import fuse_model

            mx.random.seed(self.seed)
            self.model, self.tokenizer, self.settings = blockdiff.load(
                self.model_id, **({"block_size": self.block_size} if self.block_size else {}))
            if self.engine == "locals":
                fuse_model(self.model)
            mx.clear_cache()
            self.weights_bytes = mx.get_active_memory()
            mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        except BaseException as e:  # noqa: BLE001  # (the constructor raises it)
            self._error = e
            self._ready.set()
            return
        self._ready.set()
        while not self._closed:
            try:
                job = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._run(job)
            except BaseException as e:  # noqa: BLE001
                import traceback

                traceback.print_exc(file=sys.stderr)
                self._kept = ([], None)
                self._finish(job, job.tokens, f"error: {e!r}")
        while True:  # closed: what is queued gets an answer
            try:
                self._finish(self._q.get_nowait(), [], "error: the server is shutting down")
            except queue.Empty:
                return

    def _run(self, job: Job) -> None:
        from . import blockdiff

        if job.cancelled:
            return self._finish(job, [], "cancelled")
        s = self.settings
        over = {"max_tokens": max(1, job.max_tokens)}
        if job.temp is not None:
            over["temperature"] = float(job.temp)
        if job.top_p is not None and job.top_p > 0:
            over["top_p"] = float(job.top_p)
        if job.top_k is not None:
            over["top_k"] = int(job.top_k)
        if job.seed is not None:
            over["seed"] = int(job.seed)
        s = replace(s, **over)
        ids = job.prompt
        kept, cache = self._kept
        cached = len(kept) if cache is not None and len(kept) < len(ids) and ids[: len(kept)] == kept else 0
        if not cached:
            cache = self.model.make_cache()
        self._kept = ([], None)  # (in use: a failure midway leaves no half-written cache to reuse)
        job.cached = cached
        eos = set(s.eos_token_ids) or set(self.tokenizer.eos_token_ids)
        out: List[int] = []  # the reply's tokens, without end tokens
        every: List[int] = []  # and with them: what the cache holds after the prompt
        finish = "length"
        self.active = 1
        self.prompt_tokens += len(ids)
        self.cached_tokens += cached
        try:
            for r in blockdiff.stream_generate(self.model, self.tokenizer, ids, s, cache=cache, cached=cached,
                                               ignore_eos=job.ignore_eos, commit_last=True):
                self.steps += r.steps
                every += r.tokens
                toks = [t for t in r.tokens if job.ignore_eos or t not in eos]
                out += toks
                self.tokens_out += len(toks)
                if toks and job.on_token is not None:
                    job.on_token(toks)
                if r.finish_reason:
                    finish = r.finish_reason
                if job.cancelled:
                    finish = "cancelled"
                    break
        finally:
            self.active = 0
        if finish != "cancelled":  # (a reply cut short leaves its last block out of the cache)
            self._kept = (list(ids) + every, cache)
        self._finish(job, out, finish)
