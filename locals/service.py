"""A shared generation service: one model, one batch, many concurrent clients.

Clients (agents, HTTP handlers, ...) call ``submit`` from any thread. One
background thread owns the model and runs continuous batching, so requests that
arrive while others decode join the running batch. Two backends:

  locals : our runtime (fused layers, small-M kernels) and HybridBatchEngine's
          pipelined drip loop; JSON-schema requests use our constraint engine
          (forced spans, canonical boundaries).
  stock : plain mlx-lm, i.e. the classical stack: BatchGenerator continuous
          batching, and llguidance logits processors for JSON-schema requests.

Both keep mlx-lm's LRUPromptCache: a multi-turn client that resubmits its
growing token stream only prefills what it appended since its last call.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

ENGINES = ("locals", "stock")


@dataclass
class Job:
    prompt: List[int]
    schema: Optional[dict] = None
    max_tokens: int = 512
    regex: Optional[str] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    finish: str = ""
    cached: int = 0  # prompt tokens served from the prompt cache
    t_submit: float = 0.0
    t_done: float = 0.0
    on_token: Optional[Callable[[List[int]], None]] = None  # called from the worker with each committed token group
    cancelled: bool = False
    checkpoints: List[Tuple[int, str]] = field(default_factory=list)  # (prompt position, LRU cache type) also stored
    temp: Optional[float] = None  # the request's sampling temperature (None: the service's)
    top_p: Optional[float] = None  # its own top_p, top_k (None: the service's)
    top_k: Optional[int] = None
    seed: Optional[int] = None  # its own seed: the same reply for the same request alone (None: unseeded)
    ignore_eos: bool = False  # run to max_tokens: end-of-sequence tokens do not end the reply (a benchmark option)
    _event: threading.Event = field(default_factory=threading.Event)
    _req: object = None

    def wait(self, timeout: Optional[float] = None) -> "Job":
        self._event.wait(timeout)
        return self

    def cancel(self) -> None:
        """Stop generating (the client went away); the job finishes with
        ``finish == "cancelled"`` at the engine's next commit."""
        self.cancelled = True
        r = self._req
        if r is not None:
            r.cancel = True

    @property
    def done(self) -> bool:
        return self._event.is_set()


def _llg_grammar(llguidance, regex: str, specials: dict):
    """llguidance grammar for a regex whose outer literals may be special tokens.

    llguidance never matches a special token against regex text (it has no
    bytes), so a native tool-call format such as ``<tool_call>...</tool_call>``
    must reference those tokens by id in lark: <[id]> MIDDLE <[id]>."""
    import re

    head = tail = None
    for text, tid in sorted(specials.items(), key=lambda kv: -len(kv[0])):
        for lit in (text, re.escape(text), text.replace("/", "\\/")):
            if head is None and regex.startswith(lit):
                head, regex = tid, regex[len(lit):]
            if tail is None and regex.endswith(lit):
                tail, regex = tid, regex[: -len(lit)]
    if head is None and tail is None:
        return llguidance.LLMatcher.grammar_from_regex(regex)
    mid = regex.replace("/", "\\/")
    parts = ([f"<[{head}]>"] if head is not None else []) + (["MID"] if regex else []) + ([f"<[{tail}]>"] if tail is not None else [])
    lark = "start: " + " ".join(parts) + ("\nMID: /" + mid + "/" if regex else "")
    return llguidance.LLMatcher.grammar_from_lark(lark)


def _weights_fingerprint(model_id: str) -> str:
    """Which weights a model id names now, and the mlx-lm that wrote their caches: an updated
    repository, another fine-tune in the same folder or another mlx-lm gets other saved prompts
    (a saved cache of other weights would give wrong output, or fail the batch at its merge)."""
    import hashlib
    from pathlib import Path

    import mlx_lm

    parts = [getattr(mlx_lm, "__version__", "")]
    path = Path(model_id).expanduser()
    if not path.exists():
        try:
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(model_id, local_files_only=True, allow_patterns=["config.json"]))
        except Exception:  # noqa: BLE001
            path = None
    if path is not None:
        parts.append(str(path.resolve()))
        for f in sorted(path.glob("*.json")) + sorted(path.glob("*.safetensors")):
            try:
                st = f.stat()
                parts.append(f"{f.name}:{st.st_size}:{int(st.st_mtime)}")
            except OSError:
                pass
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def memory_plan(limit: int, weights: int) -> tuple:
    """(MLX's pool of freed buffers, the memory guard's margin below the GPU limit) for a model whose weights take
    ``weights`` of ``limit``. The margin is the pool and 0.5 GB for what the admission estimate misses. It was 1.5
    GB more while the estimate left out each request's fixed state and kept 1.5 GB for reading a prompt that takes
    2.1 GB; with them counted, a 2.5 GB margin made room by emptying the prompt cache (four agents at once: 71-88%
    of their prompts from it, first token 5.1 s, against 87-95% and 1.2 s with 1.5); with 1.5 for a 4B on 16 GB, 16 x
    2k, 8 x 8k and 4 x 30k-token prompts at once all finished, at most 9.06 GB of 11.45 in use. A model that leaves
    less than 8 GB (a 27B on 24 GB: 3.7 GB) gets a pool of an eighth of what is left, 256 MB at least, so the
    requests have the rest. LOCALS_MEMORY_MARGIN_GB sets the margin itself."""
    room = max(0, limit - weights)
    pool = 1 << 30 if room >= (8 << 30) else max(256 << 20, min(1 << 30, room // 8))
    env = os.environ.get("LOCALS_MEMORY_MARGIN_GB")
    return pool, (int(float(env) * (1 << 30)) if env else pool + (512 << 20))


_SERVICES: "weakref.WeakSet" = weakref.WeakSet()  # the engines of this process (locals bench serves two at once)
_POOL_PLANNED = False  # our engine sized MLX's pool of freed buffers (memory_plan): the stock engine keeps it


_REQUEST_ERRORS = (ValueError, TypeError, KeyError, AttributeError, RecursionError)  # (a schema or regex that cannot be read)


_HEARTBEAT = threading.Event()


def _start_heartbeat() -> None:
    """Once per process: a thread that marks the GPU busy (kernels.mark_busy) each second while an engine of this
    process has a request in flight or queued, or times a kernel shape: another locals server's idle tuner waits."""
    if _HEARTBEAT.is_set():
        return
    _HEARTBEAT.set()

    def beat():
        from .kernels import mark_busy

        while True:
            time.sleep(1.0)
            try:
                if any(s._inflight or not s._q.empty() or getattr(s, "_tuning", False) for s in list(_SERVICES)):
                    mark_busy()
            except Exception:  # noqa: BLE001  # (a service going away meanwhile)
                pass

    threading.Thread(target=beat, name="locals-busy", daemon=True).start()


def _job_sampler(temp: float, top_p: float, top_k: int, seed: Optional[int]):
    """mlx-lm's sampler for a request with its own settings; with a seed, its draws come from that seed."""
    from mlx_lm.sample_utils import make_sampler

    if seed is None or not temp:
        return make_sampler(temp, top_p=top_p, top_k=top_k)
    from .engine import row_key, sample_logits

    n = [0]

    def sample(logprobs):
        key = row_key(seed, n[0])
        n[0] += 1
        return sample_logits(logprobs, temp, top_p, top_k, key=key)

    return sample


def _reason(e: BaseException) -> str:
    """A request's error as the client reads it: a ValueError's message says what is wrong (the grammar
    compilers' own say which part of the schema); anything else is named."""
    return str(e) if isinstance(e, ValueError) and str(e) else f"{type(e).__name__}: {e}"


class BatchService:
    def __init__(self, engine: str, model: str, slots: int = 16, temp: float = 0.0, seed: int = 0, cache_gb: Optional[float] = None,
                 walk: bool = False, top_p: float = 0.0, top_k: int = 0):
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}")
        self.engine, self.model_id, self.slots, self.temp, self.seed = engine, model, slots, temp, seed
        self.top_p, self.top_k = top_p, top_k  # sampling filters at temp > 0, the same rule on both engines
        self.walk = walk  # locals engine: constraints by the walk kernel (no token tables)
        # None: our engine sizes its prompt cache from the free GPU memory after loading
        # (the stock engine keeps 1.5 GB)
        self.cache_bytes = int(cache_gb * (1 << 30)) if cache_gb else None
        self._q: "queue.Queue[Job]" = queue.Queue()
        self._compile_lock = threading.Lock()
        self._inflight = {}  # id(job) -> job, taken by the worker and not yet finished
        self._closed = False
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self.tokenizer = None
        self.active = 0
        self.tokens_out = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        _SERVICES.add(self)
        _start_heartbeat()
        self._thread = threading.Thread(target=self._main, name=f"batch-{engine}", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            raise self._error

    # -- client API -----------------------------------------------------------------
    def submit(self, prompt: List[int], schema: Optional[dict] = None, max_tokens: int = 512, regex: Optional[str] = None,
               on_token: Optional[Callable[[List[int]], None]] = None, checkpoints: Optional[List[Tuple[int, str]]] = None,
               temp: Optional[float] = None, ignore_eos: bool = False, top_p: Optional[float] = None,
               top_k: Optional[int] = None, seed: Optional[int] = None) -> Job:
        """Queue a generation; the output matches ``schema`` (JSON Schema) or ``regex`` if given.
        ``on_token`` streams committed tokens (called from the worker thread). For
        each ``(n, kind)`` in ``checkpoints`` the cache of ``prompt[:n]`` is also kept
        in the prompt cache (our engine) as an entry of type ``kind`` ("system"
        entries are evicted last, then "user", then finished replies), for a next
        request that shares only that much. ``temp``: this request's sampling
        temperature (0: greedy), else the service's; ``top_p``, ``top_k`` likewise; ``seed``: its draws come
        from that seed (our engine: the same reply for the same request alone). ``ignore_eos``: generate ``max_tokens``
        tokens whatever the model ends with (benchmarks' fixed output lengths, as vLLM's option)."""
        if not prompt:
            raise ValueError("empty prompt")
        job = Job(list(prompt), schema, max_tokens, regex, t_submit=time.perf_counter(), on_token=on_token,
                  checkpoints=list(checkpoints or []), temp=temp, ignore_eos=ignore_eos, top_p=top_p, top_k=top_k,
                  seed=seed)
        rt = getattr(self, "_rt", None)
        if rt is not None and not self.walk and (schema is not None or regex is not None):
            # its grammar is made in a thread of its own, and the job queued once it is: made in the batch loop, a
            # JSON schema with maxItems 50 (1302 states) stopped every running reply for 8 s (31 s on a 16 GB Mac
            # next to the model)
            threading.Thread(target=self._compile_then_queue, args=(rt, job), name="locals-compile", daemon=True).start()
        else:
            self._q.put(job)
        return job

    def _compile_then_queue(self, rt, job: "Job") -> None:
        """The job's grammar made (or found made, or refused) for it to carry into the batch loop, then the job
        queued. Only a grammar not made yet waits for the compile lock (one at a time: each holds its token table
        while it is made); a cancel meanwhile skips it."""
        try:
            ready = rt.ready(job.schema, job.regex, recursive_ok=False)
            if ready is None and not job.cancelled:
                with self._compile_lock:
                    if not job.cancelled:
                        ready = rt.precompile(job.schema, job.regex, recursive_ok=False)
            job._ready = ready
        except Exception:  # noqa: BLE001  # (compile in the batch loop does it again: this request's error)
            pass
        finally:
            if self._closed:  # (the batch loop has stopped: the job would wait forever)
                self._finish(job, [], "error: the server is shutting down")
            else:
                self._q.put(job)

    def prime(self, tokens: List[int]) -> None:
        """Prefill ``tokens`` once and store them in the prompt cache (e.g. the
        system prompt and tool definitions every client starts with)."""
        job = Job(list(tokens), max_tokens=0, t_submit=time.perf_counter())
        job.prime = True
        self._q.put(job)
        job.wait()

    def close(self) -> None:
        self._closed = True
        self._thread.join(timeout=30)

    def context_fit(self) -> Optional[int]:
        """Prompt + reply tokens one request can have on this Mac (our engine): the KV cache that fits under
        the engine's memory limit next to the weights and a prefill's activations. None if unknown."""
        eng = getattr(self, "_eng", None)
        weights = getattr(self, "weights_bytes", None)
        if eng is None or not weights or not getattr(eng, "memory_limit", None):
            return None
        kv = eng._kv_bytes_per_token()
        free = eng.memory_limit - weights - (3 << 29)  # the admission estimate's 1.5 GB for a prefill's activations
        return int(free // kv) // 1024 * 1024 if kv and free > 0 else None

    def stats(self) -> dict:
        eng = getattr(self, "_eng", None)
        # the engine's own counters, read now (the copies made at each pipelined step stand still while one
        # request runs alone, several tokens a pass)
        out = {"engine": self.engine, "active": getattr(eng, "active", self.active), "tokens_out": getattr(eng, "tokens_out", self.tokens_out),
               "prompt_tokens": self.prompt_tokens, "cached_tokens": self.cached_tokens}
        out.update(getattr(eng, "counters", {}) if eng is not None else {})
        lru = getattr(self, "_lru", None)
        if lru is not None:
            out["prompt_cache_mb"] = lru.nbytes / 1e6
            out["prompt_cache_entries"] = len(lru)
            try:  # entries and megabytes per kind (system prompts, request checkpoints, finished replies)
                out["prompt_cache_by_kind"] = {k: {"entries": v["n_sequences"], "mb": round(v["n_bytes"] / 1e6)}
                                               for k, v in lru.stats_by_type().items()}
            except Exception:  # noqa: BLE001
                pass
        return out

    # -- worker -----------------------------------------------------------------------
    def _take(self, block: bool) -> List[Job]:
        out = []
        while True:
            try:
                job = self._q.get(block=block and not out, timeout=0.05)
                self._inflight[id(job)] = job
                out.append(job)
            except queue.Empty:
                if block and not out and not self._closed:
                    self._tune_while_idle()
                    continue
                return out

    def _tune_while_idle(self) -> None:
        """One pending kernel shape, tuned while there is nothing to do (in the engine's own thread,
        so the GPU is not shared); a request that arrives waits for this one shape at most."""
        work = getattr(self, "_tune_work", None)
        if not work or time.time() < getattr(self, "_tune_after", 0.0):
            return
        if any(s is not self and (s._inflight or not s._q.empty()) for s in list(_SERVICES)):
            return  # the other engine works (locals bench): timings on a shared GPU would be kept for good
        import fcntl

        from . import kernels

        if kernels.others_busy():  # another locals server works on this GPU (or times a shape): looked at again soon
            self._tune_after = time.time() + 1.0
            return
        os.makedirs(kernels._BUSY_DIR, exist_ok=True)
        with open(os.path.join(kernels._BUSY_DIR, "tuning.lock"), "a") as lock:
            try:  # one tuner at a time across processes, and never while another marks itself busy
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self._tune_after = time.time() + 1.0
                return
            mod, M = work.pop(0)
            self._tuning = True
            kernels.mark_busy(now=True)
            try:
                kernels.tune_one(mod, M)
            except Exception:  # noqa: BLE001  # an untunable shape keeps MLX's kernel
                pass
            finally:
                self._tuning = False
        if not work:
            print("locals: kernel tuning finished; batches now run on the tuned kernels", file=sys.stderr, flush=True)

    def _finish(self, job: Job, tokens, finish: str) -> None:
        job.tokens = list(tokens)
        job.text = self.tokenizer.decode(job.tokens)
        job.finish = finish
        job.t_done = time.perf_counter()
        self._inflight.pop(id(job), None)
        job._event.set()

    def _main(self):
        try:
            import mlx.core as mx

            mx.random.seed(self.seed)
            if not _POOL_PLANNED:  # (loaded after ours, the stock engine set it back to 1 GB under ours's margin)
                mx.set_cache_limit(1 << 30)  # MLX keeps freed buffers up to the whole memory limit by default
            (self._run_locals if self.engine == "locals" else self._run_stock)()
        except BaseException as e:  # surface load errors to the constructor, others to clients
            import sys
            import traceback

            loaded = self._ready.is_set()
            if loaded:  # a load error is raised again by the constructor
                traceback.print_exc(file=sys.stderr)
            self._error = e
            self._ready.set()
            if not loaded:
                return
            for job in list(self._inflight.values()):
                job.finish = f"error: {e!r}"
                job._event.set()
            while True:
                for job in self._take(False):
                    job.finish = f"error: {e!r}"
                    job._event.set()
                if self._closed:
                    return
                time.sleep(0.05)

    def _run_locals(self):
        import mlx.core as mx

        from .engine import Controller, LocalsController
        from .hybrid_batch import HybridBatchEngine, chain_cache
        from .packed import Request
        from .runtime import Runtime

        rt = Runtime.load(self.model_id, tune=os.environ.get("LOCALS_TUNE", "background"))
        self._rt = rt
        self.weights_bytes = mx.get_active_memory()
        self.tokenizer = rt.tokenizer
        eos = list(rt.vocab.eos_ids)
        from . import kernels

        if kernels._DEFER:  # tuned while idle, smallest batches first
            self._tune_work = kernels.pending_work(rt.model)

        class Free(Controller):
            eos_ids = eos

        class Endless(Controller):  # ignore_eos
            eos_ids = ()

        eng = HybridBatchEngine(rt.model, rt.tokenizer, n_slots=self.slots)
        try:
            disk_gb = float(os.environ.get("LOCALS_DISK_CACHE_GB") or 4)
        except ValueError:
            print(f"locals: LOCALS_DISK_CACHE_GB={os.environ['LOCALS_DISK_CACHE_GB']!r} is not a number; 4 GB", file=sys.stderr)
            disk_gb = 4.0
        if os.environ.get("LOCALS_DISK_CACHE", "1") != "0" and disk_gb > 0:  # system prompts kept across restarts (0 GB: off)
            from pathlib import Path

            from .hybrid_batch import DiskPromptCache

            tag = (self.model_id.strip("/").replace("/", "--")[-80:] + "--" + str(getattr(rt, "dtype", "")).split(".")[-1]
                   + "--" + _weights_fingerprint(self.model_id))
            root = Path(os.environ.get("LOCALS_DISK_CACHE_DIR") or Path.home() / ".cache" / "locals" / "prompts")
            try:
                eng.disk = DiskPromptCache(root / tag, disk_gb)
                import shutil

                stem = tag.rsplit("--", 1)[0]  # the same model and dtype, other weights or mlx-lm (or no fingerprint)
                for old in root.iterdir():
                    if old.is_dir() and old.name != tag and (old.name == stem or old.name.startswith(stem + "--")):
                        shutil.rmtree(old, ignore_errors=True)
            except OSError:
                eng.disk = None
        eng.top_p, eng.top_k = self.top_p, self.top_k
        self._eng = eng
        limit = int(mx.device_info()["max_recommended_working_set_size"])
        pool, margin = memory_plan(limit, self.weights_bytes)
        global _POOL_PLANNED
        _POOL_PLANNED = True
        mx.set_cache_limit(pool)
        eng.memory_limit = limit - margin
        if self.cache_bytes is None:  # what is left after the weights and 5 GB for running rows and prefill,
            free = limit - mx.get_active_memory() - (5 << 30)  # and at least one conversation as long as fits
            # (its follow-up question then continues from it: 1.5 s instead of 137 s at 60k tokens). The memory
            # guard empties the prompt cache for running rows as they need it.
            longest = (self.context_fit() or 0) * eng._kv_bytes_per_token()
            self.cache_bytes = int(min(max(free, longest, 1.5 * (1 << 30)), 16 << 30))
        lru = chain_cache(max_size=256, max_bytes=self.cache_bytes)
        self._lru = lru
        owner = {}
        self._ready.set()

        def source(block):
            jobs = self._take(block)
            reqs = []
            for job in jobs:
                # One request's fault (a regex that does not compile, a schema too large or recursive)
                # is that request's error: it used to escape serve(), fail every request in flight and
                # leave this one waiting forever.
                try:
                    if getattr(job, "prime", False):
                        if lru.fetch_nearest_cache("locals", job.prompt)[1]:
                            c, rest = eng._prefill_start(job.prompt, [])
                            eng._run(c, list(rest))
                            mx.eval([x.state for x in c])
                            lru.insert_cache("locals", list(job.prompt), c, cache_type="system")
                        self._finish(job, [], "primed")
                        continue
                    if job.cancelled and (job.schema is not None or job.regex is not None):
                        self._finish(job, [], "cancelled")  # (gone while its grammar was made: nothing to compile)
                        continue
                    if job.schema is not None or job.regex is not None:
                        if self.walk:
                            from .walk import WalkController

                            wc, planner = rt.compile_walk(job.schema) if job.regex is None else rt.compile_walk(regex=job.regex)
                            ctl = WalkController(wc, planner, True, True)
                        else:
                            c = rt.compile(job.schema, regex=job.regex, ready=getattr(job, "_ready", None), recursive_ok=False)
                            job._ready = None
                            ctl = LocalsController(c.constraint, c.planner, True, True)
                    else:
                        ctl = Endless() if job.ignore_eos else Free()
                except _REQUEST_ERRORS as e:  # the request's grammar: a 400
                    self._finish(job, [], f"invalid: {_reason(e)}")
                    continue
                except Exception as e:  # noqa: BLE001  # the engine's (a Metal error making the tables): a 500, retried once
                    self._finish(job, [], f"error: {e!r}")
                    continue
                r = Request(job.prompt, ctl, job.max_tokens)
                r.on_token = job.on_token
                r.checkpoints = job.checkpoints
                r.temp, r.top_p, r.top_k, r.seed = job.temp, job.top_p, job.top_k, job.seed
                job._req = r  # before reading job.cancelled: a cancel in between sets r.cancel itself
                r.cancel = job.cancelled
                owner[id(r)] = (job, r)
                reqs.append(r)
                self.prompt_tokens += len(job.prompt)
            return reqs

        def done(r):
            job, _ = owner.pop(id(r))
            job.cached = getattr(r, "cached_tokens", 0)
            self.cached_tokens += job.cached
            self._finish(job, r.out, r.finish)

        orig_launch = eng._launch

        def launch(*a, **k):  # expose live counters to clients (every step, mixed passes too)
            self.active = eng.active if hasattr(eng, "active") else 0
            self.tokens_out = getattr(eng, "tokens_out", 0)
            return orig_launch(*a, **k)

        eng._launch = launch
        while not self._closed:
            try:
                eng.serve(source, temp=self.temp, lru=lru, on_done=done)
            except Exception as e:  # noqa: BLE001  # e.g. Metal out of memory: fail what is in flight, keep serving
                import gc
                import sys
                import traceback

                traceback.print_exc(file=sys.stderr)
                for job in list(self._inflight.values()):  # every job taken, whether or not it reached the batch
                    job.finish = f"error: {e!r}"
                    self._inflight.pop(id(job), None)
                    job._event.set()
                owner.clear()
                eng._fill_eval = []  # a prompt that was being prefilled in mixed passes: its cache goes too
                lru.trim_to(n_bytes=lru.nbytes // 2)
                gc.collect()
                mx.clear_cache()
                print(f"[{self.engine}] engine error, {len(lru)} cache entries kept; serving again", file=sys.stderr, flush=True)
            self.active = 0

    def _run_stock(self):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.models.cache import ArraysCache, LRUPromptCache
        from mlx_lm.sample_utils import make_sampler

        from .models import register

        register()  # architectures this mlx-lm lacks (Flint)
        model, tok = load(self.model_id)
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        self.tokenizer = tok
        lm = getattr(model, "language_model", model)
        n_vocab = lm.model.embed_tokens.weight.shape[0]
        eos = sorted(set(tok.eos_token_ids))
        llg_tok, grammars = None, {}
        specials = {t: i for t, i in tok._tokenizer.get_added_vocab().items()}

        def processors(schema, regex=None):
            nonlocal llg_tok
            if schema is None and regex is None:
                return []
            import llguidance
            import llguidance.hf

            from .llg import LLGProcessor

            if llg_tok is None:
                llg_tok = llguidance.hf.from_tokenizer(tok._tokenizer, n_vocab=n_vocab, eos_token=eos[0])
            from .grammar import json_safe_schema, schema_regex

            rx = regex if regex is not None else schema_regex(json_safe_schema(schema))
            if rx not in grammars:
                grammars[rx] = _llg_grammar(llguidance, rx, specials)
            key = rx
            return [LLGProcessor(llg_tok, grammars[key])]

        # mlx-lm's prompt processing computes logits at every prompt position; with
        # a 248k vocabulary a 16 x 2048 prefill chunk needs ~16 GB for them alone,
        # so prefill runs in batches of <= 4 prompts and 512-token chunks.
        def new_gen():
            return BatchGenerator(model, stop_tokens=[[t] for t in eos], completion_batch_size=self.slots,
                                  prefill_batch_size=min(self.slots, 4), prefill_step_size=512)

        gen = new_gen()
        lru = LRUPromptCache(max_size=256, max_bytes=self.cache_bytes or int(1.5 * (1 << 30)))
        self._lru = lru
        sampler = make_sampler(self.temp, top_p=self.top_p, top_k=self.top_k)
        live = {}
        self._ready.set()
        from mlx_lm.models.cache import make_prompt_cache

        while not self._closed:
            try:
                for job in self._take(block=not live):
                    if getattr(job, "prime", False):
                        if lru.fetch_nearest_cache("stock", job.prompt)[1]:
                            c = make_prompt_cache(model)
                            for i in range(0, len(job.prompt), 2048):
                                model(mx.array([job.prompt[i : i + 2048]]), cache=c)
                            mx.eval([x.state for x in c])
                            lru.insert_cache("stock", list(job.prompt), c)
                        self._finish(job, [], "primed")
                        continue
                    cache, rest = lru.fetch_nearest_cache("stock", job.prompt[:-1])
                    rest = list(rest) + job.prompt[-1:]
                    job.cached = len(job.prompt) - len(rest)
                    self.cached_tokens += job.cached
                    self.prompt_tokens += len(job.prompt)
                    s = sampler if all(v is None for v in (job.temp, job.top_p, job.top_k, job.seed)) else _job_sampler(
                        self.temp if job.temp is None else job.temp, self.top_p if job.top_p is None else job.top_p,
                        self.top_k if job.top_k is None else job.top_k, job.seed)
                    try:
                        procs = processors(job.schema, job.regex)
                    except Exception as e:  # noqa: BLE001  # a regex or schema that does not compile: this request's error
                        self._finish(job, [], f"invalid: {_reason(e)}")
                        continue
                    more = {}
                    if job.ignore_eos and not procs:  # no stop tokens for this row (a grammar's end still ends it)
                        from mlx_lm.generate import StopSequences

                        more["stop_sequences"] = [StopSequences()]
                    (uid,) = gen.insert([rest], [job.max_tokens], caches=[cache], all_tokens=[job.prompt[: job.cached]],
                                        samplers=[s], logits_processors=[procs], **more)
                    live[uid] = (job, [])
                if not live:
                    continue
                drop = [uid for uid, (job, _) in live.items() if job.cancelled]
                for uid in drop:
                    gen.remove([uid])
                    job, toks = live.pop(uid)
                    self._finish(job, toks, "cancelled")
                if not live:
                    continue
                for r in gen.next_generated():
                    job, toks = live[r.uid]
                    if r.finish_reason != "stop":
                        toks.append(r.token)
                        self.tokens_out += 1
                        if job.on_token is not None:
                            job.on_token([r.token])
                    if r.finish_reason is not None:
                        if r.prompt_cache is not None:
                            # mlx-lm extracts recurrent-layer rows as views of the whole
                            # batch buffer; copy them, or each cached conversation pins
                            # every row's state (~0.3 GB at 16 rows on Qwen3.5-2B).
                            for c in r.prompt_cache:
                                if isinstance(c, ArraysCache):
                                    c.cache = [None if a is None else a + 0 for a in c.cache]
                            mx.eval([c.state for c in r.prompt_cache])
                            lru.insert_cache("stock", list(r.all_tokens), r.prompt_cache)
                        del live[r.uid]
                        self._finish(job, toks, r.finish_reason)
                self.active = len(live)
            except Exception as e:  # noqa: BLE001  # e.g. Metal out of memory: fail what is in flight, keep serving
                import gc
                import traceback

                traceback.print_exc(file=sys.stderr)  # (one error left the stock engine failing every later request)
                for job in [j for j, _ in live.values()] + list(self._inflight.values()):
                    if not job._event.is_set():
                        self._finish(job, [], f"error: {e!r}")
                live.clear()
                self._inflight.clear()
                try:
                    gen.close()
                except Exception:  # noqa: BLE001
                    pass
                gen = new_gen()
                lru.trim_to(n_bytes=lru.nbytes // 2)
                gc.collect()
                mx.clear_cache()
                self.active = 0
        gen.close()
