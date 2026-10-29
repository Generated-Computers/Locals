"""OpenAI-compatible HTTP server over the batch service.

    locals serve                      # http://127.0.0.1:8321/v1
    locals serve --engines locals,mlx  # which engines may be loaded (lazily, on first use)

Endpoints: ``POST /v1/chat/completions`` (streaming and not, tools, JSON
schema response formats), ``GET /v1/models``, ``GET /health``,
``GET /stats`` (live counters), ``GET /`` (a small status page).

Two engines serve the same model, chosen by the model id of a request:

  locals/...  our engine: fused layers and small-batch kernels, pipelined
             continuous batching, prompt cache, and constrained decoding of
             tool calls (forced tags, valid tool names, budget-closed replies)
             and of JSON-schema response formats;
  mlx/...    plain mlx-lm (BatchGenerator, llguidance for JSON schemas): the
             default MLX stack, for comparison.

A model id containing ``thinking`` opens a <think> block in the reply, which
is streamed as ``reasoning_content``. Every request is one row of the running
batch: concurrent agents decode together, and a conversation's growing
prefix is served from the prompt cache.
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from . import apis, formats, oai

from . import __version__
from .defaults import DEFAULT_MODEL, MODEL_HELP, build_id, model_id, short_name

BUILD_ID = build_id()  # the code this server runs (locals code / solve restart a background server on older code)
ENGINE_NAMES = {"locals": "locals", "mlx": "stock"}  # api name -> BatchService engine


def _now() -> int:
    return int(time.time())


class ContextTooLong(ValueError):
    """A prompt longer than the server's context; answered as OpenAI's
    ``context_length_exceeded`` (or Anthropic's "prompt is too long") so agents compact their history and retry."""

    def __init__(self, msg: str, n: int = 0, limit: int = 0):
        super().__init__(msg)
        self.n, self.limit = n, limit


class BadRequest(ValueError):
    """A request the engine could not serve as asked (a regex or schema that does not compile): a 400."""


MAX_BODY = 64 << 20  # bytes of a request body
BODY_TIMEOUT = 60  # seconds a request body may take to arrive, in all (Handler.do_POST)
SEND_TIMEOUT = 120  # seconds one write to a client may block (a client that stopped reading: its job is cancelled),
# and an idle keep-alive connection waits for its next request
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}
_APP_SCHEMES = ("app", "file", "tauri", "vscode-webview", "vscode-file",  # desktop apps and VS Code
                "chrome-extension", "moz-extension", "safari-web-extension")  # browser extensions (as Ollama allows)
# chat_template_kwargs keys a client may not set: a template of its own (it would run in the server:
# a loop in it holds a thread and the GIL, as in vLLM's CVE for the same switch) and what the server sets
_TEMPLATE_DENY = {"chat_template", "tokenizer_kwargs", "documents", "continue_final_message", "return_assistant_tokens_mask",
                  "tools", "messages", "add_generation_prompt", "tokenize", "return_dict", "padding", "truncation", "max_length",
                  "return_tensors", "enable_thinking"}


def _template_switches(kw) -> dict:
    """The chat_template_kwargs a template may see: plain switches (a bool, a number, a short string)."""
    if not isinstance(kw, dict):
        return {}
    return {k: v for k, v in kw.items() if isinstance(k, str) and k not in _TEMPLATE_DENY
            and (isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 200))}


def _origin_ok(origin: str) -> bool:
    """Whether a web page from ``origin`` may call the server: one served from this machine
    (localhost, any port), a desktop app's (app://, file://, tauri://, VS Code webviews), or one in
    LOCALS_ORIGINS (comma-separated; "*" for any). Without this, any site the user visits could use
    the model and read the replies (a cross-site request needs no preflight)."""
    extra = {o.strip() for o in os.environ.get("LOCALS_ORIGINS", "").split(",") if o.strip()}
    if "*" in extra or origin in extra:
        return True
    try:
        u = urlparse(origin)
        host = (u.hostname or "").lower()
    except ValueError:
        return False
    if u.scheme in ("http", "https"):
        return host in _LOOPBACK or host.endswith(".localhost")
    return u.scheme in _APP_SCHEMES


def _host_ok(host_header: Optional[str], bind_host: str) -> bool:
    """The Host header of a server that listens on this machine only names this machine: a page whose
    domain was pointed at 127.0.0.1 (DNS rebinding) names its own. This machine's names: localhost and
    *.localhost, *.local, *.internal (Docker's host.docker.internal: Open WebUI in a container), its
    host name, loopback and private addresses, 0.0.0.0; LOCALS_HOSTS adds more (comma-separated, "*")."""
    if bind_host not in _LOOPBACK or not host_header:
        return True
    try:
        name = (urlparse("//" + host_header.strip()).hostname or "").lower().rstrip(".")
    except ValueError:  # a malformed header ("[::1")
        return False
    extra = {h.strip().lower() for h in os.environ.get("LOCALS_HOSTS", "").split(",") if h.strip()}
    if "*" in extra or name in extra or name in _LOOPBACK or name == "0.0.0.0":
        return True
    if name.endswith((".localhost", ".local", ".internal")):
        return True
    import ipaddress
    import socket

    if name in (socket.gethostname().lower(), socket.gethostname().lower().split(".")[0]):
        return True
    try:
        ip = ipaddress.ip_address(name)
        return ip.is_loopback or ip.is_private
    except ValueError:
        return False


def _dump_request(req: dict, kind: str = "") -> None:
    """LOCALS_DEBUG: the request as received, readable by this user only (prompts can hold secrets)."""
    d = Path(os.environ.get("LOCALS_DEBUG_DIR") or Path.home() / ".cache" / "locals" / "requests")
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    f = d / f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}{'_' + kind if kind else ''}.json"
    fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(req, fh, indent=1, ensure_ascii=False)


class Server:
    def __init__(self, model: str = DEFAULT_MODEL, engines=("locals", "mlx"), preload=("locals",), slots: int = 16,
                 temp: float = 0.2, max_tokens: int = 4096, thinking: bool = False, cache_gb: Optional[float] = None, seed: int = 0,
                 max_calls: int = 6, walk: bool = False, typed_tools: bool = False, log=None, top_p: float = 0.0, top_k: int = 0,
                 max_context: Optional[int] = None, blocks: Optional[int] = None):
        self.model_id = model_id(model)  # a short name (qwen, gemma) or any model id
        # a block-diffusion model (Clover) writing blocks of this many tokens (locals/blockdiff_service.py); None: a
        # token a step, its causal pass, on the batch engine as any model
        self.blocks = blocks
        if blocks:
            from .blockdiff import is_block_diffusion

            if not is_block_diffusion(self.model_id):
                raise ValueError(f"--blocks: {self.model_id} is not a block-diffusion model (Clover); it writes a token a step")
        self._recent_systems: List[str] = []  # the system messages of recent sessions (see _session_start)
        # Prompt + reply tokens per request; longer prompts get a context_length_exceeded error (past
        # what fits, a Metal out-of-memory error would end every request in flight). None: what fits
        # this Mac's GPU memory next to the model (service.context_fit: 178k tokens for Qwen3.5-4B on
        # 16 GB), at least 64k and at most the model's own context, set once our engine is loaded.
        self._auto_context = max_context is None
        self.max_context = max_context or 65536
        # the stock engine (mlx/...) has no memory guard: never sized from our engine's memory model
        self._stock_context = max_context or 65536
        self.short = short_name(self.model_id)
        self.engines = tuple(engines)
        self.slots, self.temp, self.max_tokens, self.thinking, self.cache_gb, self.seed = slots, temp, max_tokens, thinking, cache_gb, seed
        self.max_calls = max_calls
        self.top_p, self.top_k = top_p, top_k  # sampling filters at temp > 0 (Qwen3.5 thinking: temp 0.6, top_p 0.95, top_k 20)
        self.walk = walk  # constraints by the walk kernel: no token tables, any grammar size
        self.typed_tools = typed_tools  # tool calls with parameter names and types fixed (needs walk for large tool sets)
        self.log = log or (lambda s: print(s, file=sys.stderr, flush=True))
        self._svc: Dict[str, object] = {}
        self._lock = threading.Lock()  # the stream and raw-reply caches
        self._loading = {e: threading.Lock() for e in self.engines}
        self._load_error: Dict[str, tuple] = {}  # engine -> (when, error) of its last failed load
        self.bind_host = "127.0.0.1"  # set by serve()
        self.t_start = time.time()
        self.requests = 0
        self.errors = 0
        self.per_engine = {e: {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "forced_tokens": 0,
                               "passes": 0, "decode_s": 0.0, "tool_calls": 0} for e in self.engines}
        self._active: Dict[str, dict] = {}
        # Exact token streams of recent replies, so a conversation that comes
        # back re-rendered continues on the tokens the model actually saw and
        # hits the prompt cache: (rendered prompt text + reply text) -> ids.
        self._streams: Dict[str, List[dict]] = {e: [] for e in self.engines}
        self._raw: Dict[str, str] = {}  # tool-call id -> the raw reply text it came from
        self._fmt: Dict[str, tuple] = {}  # engine -> (tool-call format, thinking style), from the chat template
        self._ckpt: Dict[str, bool] = {}  # engine -> whether the model's cache needs prompt snapshots
        for e in preload or ():
            if e in self.engines:
                self.service(e)
                self.format_of(e)
        self._cap_context()

    # -- the names other APIs' clients use ------------------------------------------------------
    def check_model(self, name) -> None:
        """A request for another model by this server's own names (locals/<name>, mlx/<name>) is refused: the model
        was switched under a connected agent by another locals command (`locals chat --model gemma` while
        OpenCode ran on Qwen), and the agent's requests were answered by the new model without a word. Any
        other name (gpt-4o in an app, Ollama's names) is the model loaded, as before."""
        if not isinstance(name, str):
            return
        parts = name.strip().lower().split("/")
        if len(parts) < 2 or parts[0] not in ("locals", "mlx", "stock"):
            return
        raw = parts[-1].removesuffix(":latest")
        want = raw.removesuffix("-thinking")  # (the thinking variant /v1/models lists; a model may be named so too)
        if raw in ("", self.short) or want == self.short:
            return
        from .defaults import model_id, short_name

        try:
            if short_name(model_id(want)) == self.short:  # (an alias: locals/qwen)
                return
        except Exception:  # noqa: BLE001
            pass
        raise ValueError(f"this server runs {self.short}, not {want}. Ask for locals/{self.short}, or just \"locals\" "
                         f"(the model the server runs); an agent set up for {want} before another locals command switched "
                         f"the model is set up again by its locals command (locals code, locals launch ...); or serve "
                         f"{want} on another port: locals start --model {want} --port <another port>")

    def chat_model(self, name: str) -> str:
        """The model id for a name an Ollama client sends: the server has one model, so any name is it
        (apps ask for what /api/tags lists); mlx/... picks the stock engine, and a name with "think"
        in it turns thinking on. A request for another model by our own names is refused (check_model)."""
        if name and not isinstance(name, str):
            raise ValueError("model must be a string")
        self.check_model(name)
        engine = oai.engine_of(name) if name else "locals"
        if engine not in self.engines:
            engine = self.engines[0]
        return f"{engine}/{self.short}" + ("-thinking" if oai.thinking_of(name) else "")

    def ollama_name(self, engine: str = "locals") -> str:
        return f"{self.short}:latest" if engine == "locals" else f"{engine}/{self.short}:latest"

    def ollama_models(self, running: bool = False) -> List[dict]:
        family = self.short.split("-")[0].split(".")[0]
        out = []
        for e in self.engines:
            if running and e not in self._svc:
                continue
            m = apis.ollama_model_entry(self.ollama_name(e), self.short, family, self.context_for(e))
            if running:
                m["expires_at"] = "2099-01-01T00:00:00Z"
                # the weights, all in GPU memory: `ollama ps` showed "100% CPU" and 0 B for size_vram 0
                m["size"] = m["size_vram"] = int(getattr(self._svc.get(e), "weights_bytes", 0) or 0)
            out.append(m)
        return out

    def ollama_show(self, name: str) -> dict:
        from . import formats

        family = self.short.split("-")[0].split(".")[0]
        caps = ["completion", "tools"]
        loaded = list(self._svc.values())
        if loaded and formats.offers_thinking(loaded[0].tokenizer):
            caps.append("thinking")
        ctx = self.context_for(oai.engine_of(name) if name else "locals")
        entry = apis.ollama_model_entry(name or self.ollama_name(), self.short, family, ctx)
        return {"license": "", "modelfile": f"# served by locals: {self.model_id}", "parameters": "", "template": "",
                "details": entry["details"], "capabilities": caps,
                "model_info": {"general.architecture": family, "general.basename": self.short,
                               f"{family}.context_length": ctx}}

    # -- engines ------------------------------------------------------------------------------
    def service(self, engine: str):
        if engine not in self.engines:
            raise ValueError(f"engine {engine!r} is not enabled on this server (enabled: {', '.join(self.engines)})")
        svc = self._svc.get(engine)
        if svc is not None:
            return svc
        with self._loading[engine]:  # one load per engine; requests for the other engine go on meanwhile
            svc = self._svc.get(engine)
            if svc is not None:
                return svc
            failed = self._load_error.get(engine)
            if failed and time.time() - failed[0] < 60:
                raise RuntimeError(f"engine {engine} failed to load {time.time() - failed[0]:.0f}s ago: {failed[1]}")
            from .service import BatchService

            cls, extra = BatchService, {}
            if self.blocks:  # (Clover in blocks: its own decoding on both engines, locals/blockdiff_service.py)
                from .blockdiff_service import BlockDiffusionService as cls

                extra = {"block_size": self.blocks}
            t = time.perf_counter()
            self.log(f"loading {self.model_id} on engine {engine} ...")
            try:
                svc = cls(ENGINE_NAMES[engine], self.model_id, slots=self.slots, temp=self.temp, seed=self.seed,
                          cache_gb=self.cache_gb, walk=self.walk and engine == "locals", top_p=self.top_p, top_k=self.top_k,
                          **extra)
            except Exception as e:
                self._load_error[engine] = (time.time(), f"{type(e).__name__}: {e}")
                raise
            self._svc[engine] = svc
            self.log(f"engine {engine} ready in {time.perf_counter() - t:.1f}s")
            return svc

    def _can_think(self, engine: str, tok) -> bool:
        if engine not in getattr(self, "_thinks", {}):
            from . import formats

            self._thinks = {**getattr(self, "_thinks", {}), engine: formats.offers_thinking(tok)}
        return self._thinks[engine]

    def models(self) -> List[dict]:
        out = []
        from . import formats

        loaded = [svc for svc in self._svc.values()]
        thinks = not loaded or formats.offers_thinking(loaded[0].tokenizer)  # a "-thinking" id only where the model can think
        for e in self.engines:
            for suffix in ("", "-thinking") if thinks else ("",):
                info = {"id": f"{e}/{self.short}{suffix}", "object": "model", "created": int(self.t_start), "owned_by": e,
                        "context_length": self.context_for(e), "max_model_len": self.context_for(e)}
                if e in self._fmt:  # known once the engine is loaded
                    info["tool_call_format"] = self._fmt[e][0].name
                    info["parallel_tool_calls"] = self._fmt[e][0].multi
                out.append(info)
        return out

    def stats(self) -> dict:
        import resource

        import mlx.core as mx

        from .hybrid_batch import peak_memory

        mem = {"gpu_active_mb": mx.get_active_memory() / 1e6, "gpu_peak_mb": peak_memory() / 1e6,
               "gpu_cache_mb": mx.get_cache_memory() / 1e6, "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
        out = {"model": self.model_id, "uptime_s": time.time() - self.t_start, "requests": self.requests, "errors": self.errors,
               "active": list(self._active.values()), "memory": mem, "engines": {}}
        for e in self.engines:
            d = dict(self.per_engine[e])
            svc = self._svc.get(e)
            d["loaded"] = svc is not None
            if svc is not None:
                d.update({k: v for k, v in svc.stats().items() if k != "engine"})
            out["engines"][e] = d
        return out

    # -- one chat completion ------------------------------------------------------------------
    def prepare(self, req: dict) -> dict:
        if os.environ.get("LOCALS_DEBUG"):
            _dump_request(req)
        _unsupported(req)
        _check_shape(req)
        model = req.get("model") or f"locals/{self.short}"
        self.check_model(model)
        engine = oai.engine_of(model)
        thinking = oai.thinking_of(model, self.thinking)
        kw = req.get("chat_template_kwargs") or {}
        if isinstance(kw, dict) and isinstance(kw.get("enable_thinking"), bool):  # vLLM / SGLang style switch
            thinking = kw["enable_thinking"]
        effort = req.get("reasoning_effort")
        if effort is None and isinstance(req.get("reasoning"), dict):  # the Responses API's shape
            effort = req["reasoning"].get("effort")
        if effort == "none":
            thinking = False
        svc = self.service(engine)
        tok = svc.tokenizer
        self._cap_context()
        fmt, style = self.format_of(engine)
        if style in ("none", "switch"):
            thinking = False  # no switch, or thoughts in a form the parser does not know
        rf0 =req.get("response_format") if isinstance(req.get("response_format"), dict) else {}
        if rf0.get("type") in ("json_object", "json_schema", "regex") or req.get("guided_regex") or oai.response_schema(req):
            thinking = False  # the grammar writes the JSON from the first token: a think block the prompt opened would hold it
        if thinking and not self._can_think(engine, tok):
            thinking = False  # a model without a thinking mode (Qwen3-Instruct-2507) never closes a think block
        prefill = req.get("_prefill") if isinstance(req.get("_prefill"), str) else ""  # (an Anthropic prefill)
        if prefill:
            thinking = False
        msgs = oai.convert_messages(req.get("messages") or [], raw=self._raw if fmt.replay_raw else None)
        tools = oai.tool_specs(req.get("tools"))
        _volatile_to_end(msgs, tools)
        if tools:
            _cwd_hint(msgs)
        msgs, tpl_tools = fmt.prepare(msgs, tools)
        think_kw = {} if style == "none" else {"enable_thinking": thinking}
        think_kw.update(_template_switches(req.get("chat_template_kwargs")))  # vLLM-style template switches
        if thinking and isinstance(effort, str) and effort and "reasoning_effort" not in think_kw:
            think_kw["reasoning_effort"] = effort
        if isinstance(think_kw.get("reasoning_effort"), str):
            # the template's own values (Qwen3.8 raises on any but xhigh, medium, low): the nearest one
            allowed = formats.reasoning_efforts(formats.chat_template(tok))
            if allowed:
                think_kw["reasoning_effort"] = formats.effort_for(think_kw["reasoning_effort"], allowed)
        text = _render(tok, msgs, tpl_tools, add_generation_prompt=True, **think_kw)
        if style == "always" and not thinking:
            # the template opens a think block whatever the switch says (LFM2.5): closed empty here, the model
            # answers at once (a short request answered nothing, every token spent thinking; JSON started in it)
            text += fmt.think_close
        text += prefill  # the reply goes on from it
        prompt = self._tokens_for(engine, tok, text)
        limit = self.limit_for(engine)
        if len(prompt) >= limit:
            raise ContextTooLong(f"This model's maximum context length is {limit} tokens. However, your messages "
                                 f"resulted in {len(prompt)} tokens. Please reduce the length of the messages.", len(prompt), limit)
        asked = _whole(req.get("max_completion_tokens") or req.get("max_tokens"), "max_tokens")
        max_tokens = asked if asked is not None and asked > 0 else self.max_tokens  # -1: "no limit"
        room = limit - len(prompt)
        if room < min(max_tokens, 256):  # a reply of a few tokens, cut: the client is told to make room instead
            raise ContextTooLong(f"This model's maximum context length is {limit} tokens. However, your messages "
                                 f"resulted in {len(prompt)} tokens, which leaves {room} for the reply. Please reduce the "
                                 f"length of the messages.", len(prompt) + min(max_tokens, 256), limit)
        max_tokens = max(1, min(max_tokens, self.max_tokens, room))
        schema = oai.response_schema(req)
        regex = output_regex = None
        rf = req.get("response_format") if isinstance(req.get("response_format"), dict) else {}
        if rf.get("type") not in (None, "text", "json_object", "json_schema", "regex"):  # never unconstrained by mistake
            raise ValueError(f"response_format type {rf.get('type')!r} is not one of text, json_object, json_schema, regex")
        if rf.get("type") == "json_object":
            schema, regex = None, oai.json_object_regex()
        elif rf.get("type") == "regex" or req.get("guided_regex"):  # a regular expression (vLLM: guided_regex)
            schema, regex = None, rf.get("regex") or rf.get("pattern") or req.get("guided_regex")
            if not isinstance(regex, str) or not regex:
                raise ValueError("response_format regex needs a 'regex' string")
        elif engine == "locals" and schema is None and tools and getattr(svc, "constrains", True):
            tc = req.get("tool_choice", "auto")
            _check_tool_choice(tc, [t["function"]["name"] for t in tools])
            # ignore_eos (a benchmark's fixed reply length): free text past the end of turn, unless a call is required
            if not req.get("ignore_eos") or tc == "required" or isinstance(tc, dict):
                regex = oai.reply_regex(tools, tc, thinking, typed=self.typed_tools, fmt=fmt, preamble=False,
                                        parallel=req.get("parallel_tool_calls") is not False)
        if engine == "locals" and tools and getattr(svc, "constrains", True) and (
                schema is not None or rf.get("type") == "json_object" or rf.get("type") == "regex" or req.get("guided_regex")):
            tc = req.get("tool_choice", "auto")
            _check_tool_choice(tc, [t["function"]["name"] for t in tools])
            if tc != "none":
                # tools and a response format at once (the OpenAI Agents SDK's agent with an output type): the reply
                # is the calls or the output, as OpenAI's is. With the format's grammar alone the model could not
                # call, and made the tool's answer up (asked for the weather in Paris, it wrote a forecast as JSON).
                from .grammar import json_safe_schema, schema_regex

                out = regex if regex is not None else schema_regex(json_safe_schema(schema))
                calls = oai.reply_regex(tools, "required" if tc == "auto" else tc, thinking, typed=self.typed_tools,
                                        fmt=fmt, preamble=False, parallel=req.get("parallel_tool_calls") is not False)
                schema, regex = None, ("(" + calls + "|" + out + ")" if tc == "auto" else calls)
                output_regex = out if tc == "auto" else None
        cps = self._checkpoints(tok, msgs, tpl_tools, think_kw, text, prompt) if engine == "locals" and self._needs_checkpoints(engine) else []
        return {"engine": engine, "thinking": thinking, "svc": svc, "prompt": prompt, "checkpoints": cps, "text": text,
                "tools": tools, "schema": schema, "regex": regex, "output_regex": output_regex, "max_tokens": max_tokens,
                "model": model, "fmt": fmt,
                "tool_choice": req.get("tool_choice", "auto"), "stop": _stops(req), "temp": _temperature(req),
                "sampling": _sampling(req), "ignore_eos": bool(req.get("ignore_eos")), "include_usage": _include_usage(req)}

    def prepare_text(self, req: dict) -> dict:
        """A /v1/completions request: a raw prompt (no chat template), as mlx-lm's server encodes it."""
        if os.environ.get("LOCALS_DEBUG"):
            _dump_request(req, "text")
        _unsupported(req)
        _check_shape(req)
        if req.get("echo") or req.get("suffix"):  # the prompt echoed back, or text to fill in before a suffix
            raise ValueError("echo and suffix are not supported")
        model = req.get("model") or f"locals/{self.short}"
        self.check_model(model)
        engine = oai.engine_of(model)
        svc = self.service(engine)
        tok = svc.tokenizer
        self._cap_context()
        prompt = req.get("prompt", "")
        if isinstance(prompt, list) and prompt and all(type(t) is int for t in prompt):
            ids = list(prompt)  # token ids, which must be the tokenizer's (an id past the vocabulary reads past the embedding)
            n_vocab = len(tok._tokenizer)
            if any(t < 0 or t >= n_vocab for t in ids):
                raise ValueError(f"token ids must be between 0 and {n_vocab - 1}")
        else:
            if isinstance(prompt, list):
                if len(prompt) > 1:
                    raise ValueError("only one prompt per request is supported")
                prompt = prompt[0] if prompt else ""
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string, a list with one string, or a list of token ids")
            from . import formats

            ids = list(tok.encode(prompt))
            bos = getattr(tok, "bos_token_id", None)
            if ids and bos is not None and ids[0] != bos and "bos_token" in formats.chat_template(tok):
                ids.insert(0, bos)  # the model saw every sequence start with it (Gemma 4, Ministral: encode leaves it out)
        if not ids:
            raise ValueError("empty prompt")
        limit = self.limit_for(engine)
        if len(ids) >= limit:
            raise ContextTooLong(f"This model's maximum context length is {limit} tokens. However, your prompt "
                                 f"resulted in {len(ids)} tokens. Please reduce the length of the prompt.", len(ids), limit)
        asked = _whole(req.get("max_tokens"), "max_tokens")
        max_tokens = asked if asked is not None and asked > 0 else 256
        max_tokens = max(1, min(max_tokens, self.max_tokens, limit - len(ids)))
        return {"engine": engine, "thinking": False, "svc": svc, "prompt": ids, "checkpoints": [], "text": None, "tools": [],
                "schema": None, "regex": None, "max_tokens": max_tokens, "model": model, "fmt": None, "stop": _stops(req),
                "temp": _temperature(req), "sampling": _sampling(req), "ignore_eos": bool(req.get("ignore_eos")),
                "include_usage": _include_usage(req)}

    def text_completion(self, req: dict, p: dict, write=None) -> Optional[dict]:
        """One /v1/completions request, streamed when ``write`` is given."""
        t0 = time.perf_counter()
        cid = "cmpl-" + uuid.uuid4().hex[:24]
        tok = p["svc"].tokenizer
        stops = p["stop"]
        hold = max((len(s) for s in stops), default=1) - 1  # text that may be the start of a stop sequence

        def chunk(text: str, finish=None, usage=None):
            if write is None:
                return
            d = {"id": cid, "object": "text_completion", "created": _now(), "model": p["model"],
                 "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": None}]}
            if usage is not None:
                d["usage"] = usage
            write(("data: " + json.dumps(d, ensure_ascii=False) + "\n\n").encode())

        key = uuid.uuid4().hex[:8]
        self._active[key] = {"engine": p["engine"], "prompt_tokens": len(p["prompt"]), "tokens": 0, "started": time.time()}
        q: "queue.Queue[List[int]]" = queue.Queue()
        job = p["svc"].submit(p["prompt"], max_tokens=p["max_tokens"], on_token=q.put, temp=p.get("temp"),
                              ignore_eos=p.get("ignore_eos", False), **p.get("sampling", {}))
        toks: List[int] = []
        sent, stopped, last_emit = 0, None, time.perf_counter()
        last_probe = last_emit
        try:
            while True:
                got = False
                try:
                    toks.extend(q.get(timeout=0.25))
                    got = True
                    while True:
                        toks.extend(q.get_nowait())
                except queue.Empty:
                    pass
                if got:
                    self._active[key]["tokens"] = len(toks)
                    text = tok.decode(toks)
                    cut = _stop_cut(text, stops)
                    if cut is not None:  # every pass: tokens committed after the cancel come after the stop too
                        text = text[:cut]
                        if stopped is None:
                            stopped = "stop"
                            job.cancel()
                    upto = len(text) if stopped else len(text) - hold
                    if not text.endswith("\ufffd") and upto > sent:
                        chunk(text[sent:upto])
                        sent, last_emit = upto, time.perf_counter()
                elif write is not None and time.perf_counter() - last_emit > 15:
                    write(KEEPALIVE)
                    last_emit = time.perf_counter()
                if p.get("alive") is not None and time.perf_counter() - last_probe > 1:
                    last_probe = time.perf_counter()  # while tokens come too: a client that left frees its row
                    if not p["alive"]():
                        job.cancel()
                        raise ConnectionResetError("the client went away")
                if job.done and q.empty():
                    break
            if job.finish.startswith("invalid:"):
                raise BadRequest(job.finish[len("invalid:"):].strip())
            if job.finish.startswith("error"):
                raise RuntimeError(job.finish)
            final = job.text
            cut = _stop_cut(final, stops)
            if cut is not None:
                final, stopped = final[:cut], "stop"
            if len(final) > sent:
                chunk(final[sent:])
            usage = self._account(p, job, len(job.tokens), 0, t0, stopped)
            usage.pop("locals", None)
            finish = "length" if job.finish in ("length", "exhausted") and not stopped else "stop"
            if write is not None:
                chunk("", finish, None if p.get("include_usage") else usage)  # (asked for: only in a chunk of its own)
                if p.get("include_usage"):
                    write(_usage_chunk({"id": cid, "object": "text_completion", "created": _now(), "model": p["model"]}, usage))
                write(b"data: [DONE]\n\n")
                return None
            return {"id": cid, "object": "text_completion", "created": _now(), "model": p["model"],
                    "choices": [{"index": 0, "text": final, "finish_reason": finish, "logprobs": None}], "usage": usage}
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):  # the client went away
            job.cancel()
            return None
        finally:
            self._active.pop(key, None)

    def _cap_context(self) -> None:
        """Set max_context once: what fits in memory (unless --max-context was given; once our engine is
        loaded), then at most the model's own context length (its config)."""
        if getattr(self, "_context_capped", False):
            return
        if not hasattr(self, "_native_ctx"):
            self._native_ctx = native_context(self.model_id)  # once (it may ask Hugging Face)
        n = self._native_ctx
        if n:  # the model's own limit applies at once, whatever the rest waits for
            self._stock_context = min(getattr(self, "_stock_context", self.max_context), n)
        auto = getattr(self, "_auto_context", False) and "locals" in self.engines
        if auto:
            svc = self._svc.get("locals")
            if svc is None:
                if n and n < self.max_context:
                    self.max_context = n
                return  # the fit: again when our engine is loaded
            fit = svc.context_fit() if hasattr(svc, "context_fit") else None
            if fit and fit > self.max_context:
                self.max_context = fit
            elif fit:
                # less fits than the 64k declared to tools (Hermes refuses a model with less): a request past the
                # fit is refused, as one past the context, instead of admitted (alone, nothing checks its memory)
                # into a Metal out-of-memory error that fails every request in flight
                self._fit_limit = fit
                self.log(f"this Mac's GPU memory fits about {fit} tokens next to {self.model_id}: longer requests get "
                         f"a context_length_exceeded error (tools are told {self.max_context}, as some need that much)")
        self._context_capped = True
        if n and n < self.max_context:
            if not auto:
                self.log(f"{self.model_id} has a {n}-token context: requests are limited to that (was {self.max_context})")
            self.max_context = n
        if auto:
            self.log(f"requests up to {self.max_context} tokens (what fits in GPU memory next to the model, "
                     f"up to its own context; --max-context sets it)")

    def context_for(self, engine: str) -> int:
        """Prompt + reply tokens a request may have on ``engine``, as declared to clients."""
        return self.max_context if engine == "locals" else min(self.max_context, getattr(self, "_stock_context", self.max_context))

    def limit_for(self, engine: str) -> int:
        """Prompt + reply tokens a request is held to: ``context_for``, or what fits in memory where that is less."""
        fit = getattr(self, "_fit_limit", None)
        return min(self.context_for(engine), fit) if fit and engine == "locals" else self.context_for(engine)

    def format_of(self, engine: str):
        """The tool-call format and thinking style of the model's chat template."""
        if engine not in self._fmt:
            from . import formats

            tok = self.service(engine).tokenizer
            self._fmt[engine] = (formats.detect(tok), formats.thinking_style(tok))
            self.log(f"[{engine}] tool calls: {self._fmt[engine][0].name}; thinking: {self._fmt[engine][1]}")
        return self._fmt[engine]

    def _needs_checkpoints(self, engine: str) -> bool:
        """Whether prompt snapshots pay: only where a cached state cannot be cut
        back to a shorter prefix (Qwen3.5's recurrent layers, Gemma's sliding
        windows). A plain KV cache is trimmed to any shared prefix by the prompt
        cache itself, so snapshots of it would only hold the prompt's KV two more
        times (5 GB for a 20k-token prompt on an 8B model)."""
        if engine not in self._ckpt:
            from mlx_lm.models.cache import KVCache, make_prompt_cache

            model = getattr(getattr(self.service(engine), "_eng", None), "model", None)
            self._ckpt[engine] = model is None or any(type(c) is not KVCache for c in make_prompt_cache(model))
        return self._ckpt[engine]

    _SESSION_MARKERS = ("<env>", "Working directory:", "Current Working Directory", "Current working directory",
                        "Workspace root folder:", "Today's date:", "Current date:", "<CURRENT_DATETIME>",
                        "Your current working directory is:")

    def _session_start(self, sys_text: str) -> int:
        """Where the part of an agent's system message that differs from session to session starts (the
        beginning of its line): at a marker agents write there (the working directory, the date), else where it
        first differs from the system message of a recent session (other agents); -1 if not found."""
        hits = [i for i in (sys_text.find(m) for m in self._SESSION_MARKERS) if i >= 0]
        if hits:
            i = min(hits)
        else:
            i = -1
            for other in self._recent_systems:
                if other == sys_text:
                    continue
                n = len(os.path.commonprefix([other, sys_text]))
                if n < len(sys_text):
                    i = max(i, n)
            if sys_text not in self._recent_systems:
                self._recent_systems = (self._recent_systems + [sys_text])[-16:]
        if i < 0:
            return -1
        return sys_text.rfind("\n", 0, i) + 1 if i > 0 else 0

    def _checkpoints(self, tok, msgs: List[dict], tools, think_kw: dict, text: str, prompt: List[int]) -> List[tuple]:
        """Where to snapshot the prompt's cache on the way (a recurrent GDN
        cache cannot be cut back afterwards, so only stored prefixes are reused):

        * the end of the system message (tools included): every session of an
          agent starts with it, so a new session skips its 5-15k tokens; kept
          longest by the prompt cache ("system").
        * the start of the last message: the previous turn plus its reply is a
          prefix of the next request only if the client resends history
          unchanged, and some (OpenClaw) move a context message to the end on
          every turn; the history before the last message stays put ("user").

        Both are found by rendering the conversation up to that point with the
        model's own template; a position is used only where that text is a
        prefix of the prompt and tokenizes to the prompt's own first tokens."""
        out = []
        if not msgs:
            return out
        hf = tok._tokenizer
        cands = []
        if len(msgs) >= 2 and msgs[0]["role"] == "system":
            # A template may refuse a conversation without a user message (Qwen3.5 does), so the end
            # of the system part is where two renderings with different user messages part: the
            # system message, tools included, and the user turn's header, the same in every session.
            try:
                a = _render(tok, msgs[:1] + [{"role": "user", "content": "\x01A"}], tools, add_generation_prompt=False, **think_kw)
                b = _render(tok, msgs[:1] + [{"role": "user", "content": "\x02B"}], tools, add_generation_prompt=False, **think_kw)
                n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                end = a[:n]
                # * the start of the session's own part of the system message: agents end it with the project's
                #   folder and the date (OpenCode's <env>), so the end above is reached only by sessions in the
                #   same folder on the same day; everything before is the same in every project (7,157 of
                #   OpenCode's 7,222 tokens), and a new project's first request read it all again (7 s)
                sys_text = msgs[0].get("content").strip() if isinstance(msgs[0].get("content"), str) else ""
                at = text.find(sys_text) if sys_text else -1  # (templates may trim it)
                j = self._session_start(sys_text) if at >= 0 else -1
                if j > 0:
                    cands.append((text[:at + j], "system"))
                    if len(end) - (at + j) > 4096:  # far from the end (a long folder-specific part): keep both
                        cands.append((end, "system"))
                else:
                    cands.append((end, "system"))
            except Exception:  # noqa: BLE001
                pass
        if len(msgs) > 2 or (len(msgs) == 2 and msgs[0]["role"] != "system"):
            try:
                cands.append((_render(tok, msgs[:-1], tools, add_generation_prompt=False, **think_kw), "user"))
            except Exception:  # noqa: BLE001
                pass
        # * where this reply starts as the next request renders it ("reply", the recurrent state only: the
        #   engine keeps the reply's KV prefix from the finished row). The prompt itself is no prefix of the
        #   next request: templates write an earlier reply differently from the open turn (Qwen3.5 drops the
        #   empty think block), so the conversation was read again from the last checkpoint on every turn
        #   (a follow-up question on a 60k-token document: 137 s).
        try:
            if os.environ.get("LOCALS_REPLY_CHECKPOINT", "1") == "0":
                raise LookupError("off")
            follow = [{"role": "user", "content": "\x03"}]
            # thinking on, the reply has thoughts, which the template drops from an earlier turn: the stand-in reply
            # has some too (one without them keeps its empty think block, _stable_template, and the position fell
            # past the prompt's end: Claude Code, thinking on, read every earlier step of the conversation again)
            thought = {"reasoning_content": "\x04R"} if think_kw.get("enable_thinking") else {}
            a = _render(tok, msgs + [{"role": "assistant", "content": "\x01A", **thought}] + follow, tools, add_generation_prompt=False, **think_kw)
            b = _render(tok, msgs + [{"role": "assistant", "content": "\x02B", **thought}] + follow, tools, add_generation_prompt=False, **think_kw)
            n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            cands.append((a[:n], "reply"))
        except Exception:  # noqa: BLE001
            pass
        for t, kind in cands:
            if len(t) < 2048 or not text.startswith(t):  # a short prefix is cheaper to prefill again
                continue
            ids = hf.encode(t, add_special_tokens=False)
            # where the prompt's tokens and these part (the last token may merge with what follows): one pass
            # (backing off a token at a time cost seconds on a 60k-token prompt)
            n = next((k for k, (a, b) in enumerate(zip(prompt, ids)) if a != b), min(len(prompt), len(ids)))
            if n >= 1024 and n < len(prompt) and all(n != m for m, _ in out):
                out.append((n, kind))
        return out

    def _tokens_for(self, engine: str, tok, text: str) -> List[int]:
        """Token ids of a rendered prompt: a remembered stream's ids where the
        text continues one, plus the tokens of the remainder."""
        best = None
        with self._lock:
            for st in self._streams[engine]:
                if text.startswith(st["text"]) and (best is None or len(st["text"]) > len(best["text"])):
                    best = st
        hf = tok._tokenizer
        if best is None:
            return list(hf.encode(text, add_special_tokens=False))
        return list(best["ids"]) + list(hf.encode(text[len(best["text"]):], add_special_tokens=False))

    def _remember(self, engine: str, p: dict, job, calls: List[dict], prefix: str = "", raw: Optional[str] = None,
                  exact: bool = True) -> None:
        raw = prefix + job.text if raw is None else raw
        exact = exact and raw == prefix + job.text
        ids = list(p["prompt"])
        if prefix:
            ids += list(p["svc"].tokenizer._tokenizer.encode(prefix, add_special_tokens=False))
        with self._lock:
            for c in calls:
                self._raw[oai.id_key(c["id"])] = raw
            if len(self._raw) > 4096:
                for k in list(self._raw)[:1024]:
                    del self._raw[k]
            if exact:
                st = self._streams[engine]
                st.append({"text": p["text"] + raw, "ids": ids + list(job.tokens)})
                del st[:-64]

    def _submit(self, p: dict, on_token=None):
        return p["svc"].submit(p["prompt"], schema=p["schema"], max_tokens=p["max_tokens"], regex=p["regex"], on_token=on_token,
                               temp=p.get("temp"), ignore_eos=p.get("ignore_eos", False),
                               checkpoints=p.get("checkpoints"), **p.get("sampling", {}))

    def _account(self, p: dict, job, out_tokens: int, calls: int, t0: float, stopped: Optional[str] = None) -> dict:
        r = getattr(job, "_req", None)
        d = self.per_engine[p["engine"]]
        d["requests"] += 1
        d["prompt_tokens"] += len(p["prompt"])
        d["cached_tokens"] += job.cached
        d["completion_tokens"] += out_tokens
        d["forced_tokens"] += getattr(r, "forced", 0) or 0
        d["passes"] += getattr(r, "passes", 0) or 0
        d["decode_s"] += max(0.0, job.t_done - job.t_submit)
        d["tool_calls"] += calls
        dt = max(1e-9, job.t_done - job.t_submit)
        usage = {"prompt_tokens": len(p["prompt"]), "completion_tokens": out_tokens, "total_tokens": len(p["prompt"]) + out_tokens,
                 "prompt_tokens_details": {"cached_tokens": job.cached},
                 "locals": {"engine": p["engine"], "forced_tokens": getattr(r, "forced", 0) or 0, "passes": getattr(r, "passes", 0) or 0,
                           "constrained": p["regex"] is not None or p["schema"] is not None, "wall_s": time.perf_counter() - t0}}
        self.log(f"[{p['engine']}] prompt {len(p['prompt'])} ({job.cached} cached)  out {out_tokens} tok"
                 f"{'  forced ' + str(getattr(r, 'forced', 0)) if getattr(r, 'forced', 0) else ''}  {out_tokens / dt:.0f} tok/s"
                 f"  {calls} call(s)  finish={job.finish}" + (f"  stopped={stopped}" if stopped else ""))
        return usage

    def _guard(self, toks: List[int], text: str, fmt=None, tools: bool = True, state=None) -> Optional[str]:
        """Why to stop a reply early: enough tool calls, a repeated call, or a
        token loop (a small model can repeat forever; the grammar cannot see it).
        A reply may also repeat on purpose: without tools ("write it ten times")
        and inside a call's arguments (a file of identical lines), so there it
        takes many more repeats to count as a loop, and then LOOP_GRACE tokens
        more of it (``state`` keeps where it began): a loop goes on to the token
        limit, a repetition asked for ends. Cut at once, "the line 50 times" had
        13 lines, and a write call of 40 identical lines streamed arguments that
        never closed."""
        in_call = False
        if text:
            if fmt is None or fmt.name == "qwen_xml":
                i = text.find(oai.TOOL_OPEN)
                found = oai.qwen_calls(text[i:]) if i >= 0 else []  # a "</tool_call>" inside a value is not a call's end
                n_calls = len(found)
                blocks = [json.dumps([n, a], sort_keys=True) for n, a, _ in found]
                in_call = i >= 0
            else:
                part = fmt.split(text)[1]
                found = fmt.parse_calls(part)
                n_calls = fmt.count_complete(text)
                blocks = [json.dumps(c, sort_keys=True) for c in found[:n_calls]]
                in_call = bool(part) or fmt.opener in text
            if n_calls >= self.max_calls:
                return "max_calls"
            if len(blocks) != len(set(blocks)):
                return "duplicate_call"
        if in_call:
            # a file written through a call repeats lines on purpose (a grid, a table, test data): 16 repeats over
            # 256 tokens or more (8 over 128 cut an 8 x 8 board out of a write call, and the call was dropped)
            looping = _looping(toks, reps=16, min_span=256, short_span=256)
            return "loop_in_call" if _past_grace(looping, len(toks), state) else None
        elif tools:
            looping = _looping(toks) or _line_looping(text)
        else:
            looping = _past_grace(_looping(toks, reps=12, min_span=96, short_span=256), len(toks), state)
        return "loop" if looping else None

    def _run(self, req: dict, write=None, p: Optional[dict] = None) -> Optional[dict]:
        """One chat completion. With ``write`` (bytes -> None, raising when the
        client is gone) the reply is streamed as server-sent events and None is
        returned; otherwise the completion object is returned."""
        t0 = time.perf_counter()
        p = p or self.prepare(req)
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        head = {"id": cid, "object": "chat.completion.chunk", "created": _now(), "model": p["model"]}

        role = [True]

        def chunk(delta: dict, finish=None, usage=None, stop_seq=None):
            if write is None:
                return
            if role[0]:  # with the first delta, after the prefill (as vLLM and SGLang): sent before it, clients
                role[0] = False  # that time the first chunk measured no time to first token at all
                delta = {"role": "assistant", **delta}
            d = dict(head)
            d["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
            if stop_seq is not None:  # the stop sequence matched, as vLLM's stop_reason
                d["choices"][0]["stop_reason"] = stop_seq
            if usage is not None:
                d["usage"] = usage
            write(("data: " + json.dumps(d, ensure_ascii=False) + "\n\n").encode())

        key = uuid.uuid4().hex[:8]
        self._active[key] = {"engine": p["engine"], "prompt_tokens": len(p["prompt"]), "tokens": 0, "started": time.time()}
        prefix = ""
        job = None
        try:
            prev = self._previous_calls(req) if p["tools"] else {}
            p["prev_calls"] = list(prev)
            p["tool_choice"] = req.get("tool_choice")
            banned: set = set()  # tools a steered reply may not call: its call to them repeated the previous turn's
            shown = False  # the steering text has gone to the client
            for attempt in range(4):
                if prefix and not shown:  # inside the think block the prompt opened (Qwen3.5 thinking), it is reasoning
                    chunk({"reasoning_content": prefix} if _in_think(p) else {"content": prefix})
                    shown = True
                job, text, stopped = self._decode(p, prefix, chunk, write, key)
                if stopped == "max_calls":
                    text = p["fmt"].cut_after(text, self.max_calls)
                reasoning, content, calls = oai.parse_reply(text, p["tools"], fmt=p["fmt"], thinking=_in_think(p))
                if not p["tools"]:  # no tools offered: whatever the reply looks like, it is not a call
                    reasoning, content = oai.split_reasoning(text, p["fmt"], _in_think(p))
                    content, calls = content.strip(), []
                if calls and req.get("tool_choice") == "none":  # only the stock engine can write one here
                    reasoning, content = oai.split_reasoning(text, p["fmt"], _in_think(p))
                    content, calls = content.strip(), []
                if stopped == "duplicate_call":
                    uniq, keys = [], set()
                    for c in calls:
                        k = (c["function"]["name"], c["function"]["arguments"])
                        if k not in keys:
                            keys.add(k)
                            uniq.append(c)
                    if len(uniq) < len(calls):
                        text = p["fmt"].cut_after(text, len(uniq))  # what is remembered and replayed: the calls sent
                    calls = uniq
                if content:  # held back as a possible call (a code block, a "{"), and text after all: send it now
                    # (the steering text was sent; content is stripped, so a call right after it leaves less of it)
                    rest = job.stream.unsent(content[len(prefix):] if content.startswith(prefix.rstrip()) else content)
                    if rest:
                        chunk({"content": rest})
                cs = getattr(job, "call_stream", None)
                if calls and self._repeats(calls, prev) and cs is not None and cs.streamed():
                    # calls already sent (none of them a repeat: a call that might be one waits) cannot be steered
                    # away from: the reply keeps them, up to the first repeat
                    n = next(i for i, c in enumerate(calls) if self._repeats([c], prev))
                    text, calls = p["fmt"].cut_after(text, n), calls[:n]
                if calls and self._repeats(calls, prev):
                    # steering writes text before the calls, which formats whose calls
                    # are the whole reply (Llama's JSON, Mistral 7B's) cannot hold
                    steerable = p["fmt"].name not in ("llama3_json", "mistral", "mistral_legacy")
                    named = isinstance(req.get("tool_choice"), dict)
                    if p.get("output_regex") and attempt == 0:
                        # with a response format, the answer once a call's result is in is the output itself (steering
                        # text would not be the format's; Qwen3.5 called again, and the turn ended with "I am stuck")
                        self.log(f"[{p['engine']}] reply repeats the previous call; writing the response format instead")
                        p = dict(p, regex=p["output_regex"])
                        continue
                    if attempt == 0 and steerable:
                        # The model wants to repeat the previous turn's call. Decode
                        # the turn again with its text started for it (a grammar that
                        # calls at once, without a thought block that would open the reply).
                        self.log(f"[{p['engine']}] reply repeats the previous call; steering")
                        prefix = self._steer_text(calls, prev)
                        # the steered reply should be a different call, not an essay
                        # (Gemma 4 wrote 4,096 tokens of prose after the steering text)
                        p = dict(p, max_tokens=min(p["max_tokens"], STEERED_MAX_TOKENS))
                        p["regex"] = self._steered_regex(p, req, banned)
                        continue
                    if steerable and p["regex"] and not named and attempt < 3:
                        # The steered call repeats as well. Ending the turn with a question for the user ended a
                        # headless agent's task (Gemma 4 E4B's, in OpenCode): once more, without the tools whose
                        # call repeats, for this reply only (the next turn offers them all again)
                        banned |= {c["function"]["name"] for c in calls if self._repeats([c], prev)}
                        if any(t["function"]["name"] not in banned for t in oai.tool_specs(p["tools"])):
                            self.log(f"[{p['engine']}] steered reply repeats the call too; steering to another tool")
                            p["regex"] = self._steered_regex(p, req, banned)
                            continue
                    # Still repeating, and no other tool to call: end the turn with a message instead of looping
                    # (unless a call is required: then the repeated call it is, as asked)
                    if req.get("tool_choice") == "required" or named:
                        break
                    self.log(f"[{p['engine']}] reply repeats the previous call again; stopping the turn")
                    stop = self._stop_text(calls, prev)
                    chunk({"content": "\n\n" + stop})
                    content, calls, reasoning = (content + "\n\n" + stop).strip(), [], reasoning
                break
            # the reply as the client got it (cut after max_calls, without a duplicate call): the replay of a
            # tool-call turn is this text, and only an uncut reply is a token stream to continue on
            cs = getattr(job, "call_stream", None)
            ends = cs.finish(calls) if cs is not None else [  # (the ids of calls sent while written become the calls')
                {"index": i, "id": c["id"], "type": "function",
                 "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}} for i, c in enumerate(calls)]
            self._remember(p["engine"], p, job, calls, prefix, raw=text, exact=stopped is None)
            for d in ends:
                chunk({"tool_calls": [d]})
            usage = self._account(p, job, len(job.tokens), len(calls), t0, stopped)
            # a reply cut by the loop guard inside a call (the call is dropped), or without tools (a JSON answer, text
            # that went on repeating past LOOP_GRACE), is cut short: "length" says so (with "stop" an agent took the
            # reply without its call as the end of its task, and a client the cut text as the whole answer)
            cut = stopped == "loop_in_call" or (stopped == "loop" and not p["tools"])
            finish = "tool_calls" if calls else ("length" if job.finish in ("length", "exhausted") or cut else "stop")
            stop_seq = getattr(job, "stop_sequence", None) if finish == "stop" else None
            chunk({}, finish=finish, usage=None if p.get("include_usage") else usage, stop_seq=stop_seq)
            if write is not None:
                if p.get("include_usage"):
                    write(_usage_chunk(head, usage))
                write(b"data: [DONE]\n\n")
                return None
            msg = {"role": "assistant", "content": content}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = calls
            choice = {"index": 0, "message": msg, "finish_reason": finish}
            if stop_seq is not None:
                choice["stop_reason"] = stop_seq
            return {"id": cid, "object": "chat.completion", "created": _now(), "model": p["model"],
                    "choices": [choice], "usage": usage}
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):  # the client went away
            if job is not None:
                job.cancel()
            self.log(f"[{p['engine']}] client disconnected; cancelled")
            return None
        finally:
            self._active.pop(key, None)

    def _decode_loop(self, p, job, q, toks, tok, rs, chunk, write, key, stops, hold, last_emit, last_probe):
        """_decode's streaming loop: tokens to deltas, the guards, keepalives and the hang-up probe, until the
        job is done; returns why the reply was stopped early (or None)."""
        stopped = None
        while True:
            got = False
            try:
                toks.extend(q.get(timeout=0.25))
                got = True
                while True:
                    toks.extend(q.get_nowait())
            except queue.Empty:
                pass
            if got:
                self._active[key]["tokens"] = len(toks)
                text = tok.decode(toks)
                if stopped is None:
                    if not p.get("ignore_eos"):  # a benchmark's reply runs to its length, loops too
                        seen = oai.split_reasoning(text, p.get("fmt"), _in_think(p))[1] if p["tools"] else ""  # drafts in thoughts are not calls
                        stopped = self._guard(toks, seen, p.get("fmt"), bool(p["tools"]), state=job)
                    cut = _stop_cut(text, stops)
                    if cut is not None:
                        stopped, text = "stop", text[:cut]
                    if stopped is not None and not job.done:
                        job.cancel()
                elif stopped == "stop":
                    text = text[:_stop_cut(text, stops)]
                if stops and stopped is None:
                    text = text[: max(0, len(text) - hold)]  # what might begin a stop sequence waits
                if not text.endswith("\ufffd"):
                    for kind, delta in rs.feed(text):
                        chunk({"reasoning_content": delta} if kind == "reasoning" else {"content": delta})
                        last_emit = time.perf_counter()
                    cs = getattr(job, "call_stream", None)
                    if cs is not None and rs.mode == "tool" and stopped is None:
                        for d in cs.feed(text[rs.sent:]):
                            chunk({"tool_calls": [d]})
                            last_emit = time.perf_counter()
            now = time.perf_counter()
            if write is not None and now - last_emit > 15:
                # keeps the connection open through a long prefill, and through a tool call (sent when
                # it is complete, so nothing else is written meanwhile); a client that left fails the write
                write(KEEPALIVE)
                last_emit = now
            if p.get("alive") is not None and now - last_probe > 1:
                # a client that left, found within a second (a write fails only once the socket is known to be
                # closed: a stream waiting for a long prompt or a held-back tool call writes nothing for 15 s)
                last_probe = now
                if not p["alive"]():
                    job.cancel()
                    raise ConnectionResetError("the client went away")
            if job.done and q.empty():
                break
        return stopped

    def _decode(self, p: dict, prefix: str, chunk, write, key: str):
        """Generate one reply after ``prefix`` (assistant text already
        written), streaming deltas through ``chunk``; returns (job, reply
        text including the prefix, why it was stopped early or None)."""
        tok = p["svc"].tokenizer
        prompt = list(p["prompt"])
        if prefix:
            prompt += list(tok._tokenizer.encode(prefix, add_special_tokens=False))
        q: "queue.Queue[List[int]]" = queue.Queue()
        job = p["svc"].submit(prompt, schema=p["schema"], max_tokens=p["max_tokens"], regex=p["regex"], on_token=q.put,
                              temp=p.get("temp"), ignore_eos=p.get("ignore_eos", False),
                              checkpoints=p.get("checkpoints"), **p.get("sampling", {}))
        job.prefix_tokens = len(prompt) - len(p["prompt"])
        toks: List[int] = []
        rs = oai.ReplyStream(p["thinking"] and (not prefix or _in_think(p)), p.get("fmt"),
                             calls=bool(p["tools"]) and p.get("tool_choice") != "none")
        fmt = p.get("fmt")
        # a streamed reply's calls go out while they are written (Qwen3.5's format; others when they are complete)
        job.call_stream = (oai.CallStream(p["tools"], p.get("prev_calls") or (), self.max_calls)
                           if write is not None and p["tools"] and (fmt is None or fmt.name == "qwen_xml")
                           and p.get("tool_choice") != "none" else None)
        stopped = None
        stops = [] if p["tools"] else (p.get("stop") or [])  # stop sequences apply to plain replies
        hold = max((len(s) for s in stops), default=1) - 1
        last_emit = last_probe = time.perf_counter()
        try:
            stopped = self._decode_loop(p, job, q, toks, tok, rs, chunk, write, key, stops, hold, last_emit, last_probe)
        except BaseException:
            # the client hung up (a write failed, or the probe found the socket closed) or this thread failed:
            # the reply is unread, so its row stops now instead of decoding to its end (up to max_tokens)
            if not job.done:
                job.cancel()
            raise
        if job.finish.startswith("invalid:"):
            raise BadRequest(job.finish[len("invalid:"):].strip())
        if job.finish.startswith("error") and not toks and not p.get("_retried"):
            # an engine error (a Metal out-of-memory error ends every request in flight) before
            # this reply wrote anything: the engine has dropped its cache and serves again, so
            # decode once more instead of failing the client's turn
            self.log(f"[{p['engine']}] engine error before the reply began; trying once more")
            p["_retried"] = True
            return self._decode(p, prefix, chunk, write, key)
        if job.finish.startswith("error"):
            raise RuntimeError(job.finish)
        text = job.text
        if stopped == "stop" or (stops and _stop_cut(text, stops) is not None):
            job.stop_sequence = _stop_which(text, stops)  # (Anthropic's stop_sequence; vLLM's stop_reason)
            text, stopped = text[:_stop_cut(text, stops)], "stop"
        if stopped == "max_calls":
            text = p["fmt"].cut_after(text, self.max_calls)
        for kind, delta in rs.feed(text):
            chunk({"reasoning_content": delta} if kind == "reasoning" else {"content": delta})
        job.stream = rs
        return job, prefix + text, stopped

    def _previous_calls(self, req: dict) -> dict:
        """(name, arguments) of the tool calls in the last assistant turn -> the
        text of their results."""
        msgs = req.get("messages") or []
        results = {}
        out = {}
        for m in reversed(msgs):
            if m.get("role") == "tool":
                results[m.get("tool_call_id")] = oai.text_of(m.get("content"))
            elif m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    key = (fn.get("name"), json.dumps(oai._args_dict(fn.get("arguments")), sort_keys=True))
                    out[key] = results.get(tc.get("id"), "")
                return out
            elif m.get("role") == "user":
                return out
        return out

    @staticmethod
    def _steer_text(calls: List[dict], prev: dict) -> str:
        for c in calls:
            key = (c["function"]["name"], json.dumps(json.loads(c["function"]["arguments"]), sort_keys=True))
            if key in prev:
                res = " ".join(prev[key].split())[:160]
                return (f"The {c['function']['name']} call above was already made and returned: {res!r}. "
                        "Repeating it would give the same result, so I will do something different now:\n\n")
        return ""

    def _steered_regex(self, p: dict, req: dict, banned=frozenset()) -> Optional[str]:
        """The grammar of a reply decoded again because it repeated the previous turn's call: a call right after the
        steering text (given room for text, Qwen3.5 wrote a plan and no call, and Kilo took that as the end of its
        task; Gemma 4 wrote prose to the token limit), to none of the tools in ``banned``, and without the thought
        block that may open a reply (Gemma 4 went on as if thinking, and wrote <channel|> into its text). A named
        tool_choice keeps its own grammar (no other call can be made)."""
        rx = p["regex"]
        if rx and not isinstance(req.get("tool_choice"), dict):
            tools = [t for t in oai.tool_specs(p["tools"]) if t["function"]["name"] not in banned]
            rx = oai.reply_regex(tools, "required", p["thinking"], typed=self.typed_tools, fmt=p["fmt"], preamble=False,
                                 parallel=req.get("parallel_tool_calls") is not False)
        lead = p["fmt"].thought_regex() if hasattr(p["fmt"], "thought_regex") else ""
        if rx and lead and rx.startswith(lead):
            rx = rx[len(lead):]
        return rx

    @staticmethod
    def _stop_text(calls: List[dict], prev: dict) -> str:
        c = calls[0]["function"]
        return (f"I am stuck: I keep wanting to repeat the same {c['name']} call ({c['arguments'][:200]}), whose result is already "
                "above. Please tell me how to proceed (for example, which command or file to use).")

    def _repeats(self, calls: List[dict], prev: dict) -> bool:
        return any((c["function"]["name"], json.dumps(json.loads(c["function"]["arguments"]), sort_keys=True)) in prev for c in calls)

    def complete(self, req: dict, p: Optional[dict] = None) -> dict:
        return self._run(req, p=p)

    def stream(self, req: dict, write, p: Optional[dict] = None) -> None:
        self._run(req, write, p)


def _alive_socket(conn) -> bool:
    """Whether the client on ``conn`` is still connected (a closed socket reads as end of file). poll, not
    select: select refuses file descriptors above 1023 (many agents at once, many open files)."""
    import select
    import socket

    try:
        pl = select.poll()
        pl.register(conn, select.POLLIN | select.POLLHUP | select.POLLERR)
        if not pl.poll(0):
            return True
        return conn.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) != b""
    except BlockingIOError:  # readable a moment ago, nothing to read now: still there
        return True
    except (OSError, ValueError):
        return False


_CWD = re.compile(r"Working directory:\s*(\S+)")


# Parts of an agent's system message that change with every session, written in its middle: everything after
# them was read again by every new session (OpenHands: the date and time to the microsecond, 36% into a
# 39,000-character system message). Moved to the end of the system message, where the model still reads them.
# The same for a sentence in a tool's description: tools come before the system message in Qwen3.5's template,
# and OpenHands names the project's folder in its file editor's (two folders parted after 1,555 tokens).
_VOLATILE = (re.compile(r"<CURRENT_DATETIME>.{0,400}?</CURRENT_DATETIME>\n*", re.S),)
_TOOL_VOLATILE = re.compile(r"\n*(Your current working directory is: \S+)\n?")


def _volatile_to_end(msgs: List[dict], tools=None) -> None:
    if not msgs or msgs[0].get("role") != "system" or not isinstance(msgs[0].get("content"), str):
        return
    text = msgs[0]["content"]
    moved = []
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        desc = fn.get("description") if isinstance(fn, dict) else None
        if isinstance(desc, str) and _TOOL_VOLATILE.search(desc):
            for m in _TOOL_VOLATILE.finditer(desc):
                if m.group(1) not in moved and m.group(1) not in text:
                    moved.append(m.group(1))
            fn["description"] = _TOOL_VOLATILE.sub("\n", desc).strip()
    for rx in _VOLATILE:
        for m in list(rx.finditer(text)):
            if m.end() < len(text.rstrip()):  # (already at the end: nothing to move)
                moved.append(m.group(0).strip())
        if moved:
            text = rx.sub("", text)
    if moved:
        msgs[0]["content"] = text.rstrip() + "\n\n" + "\n".join(moved)


def _cwd_hint(msgs: List[dict]) -> None:
    """Small models invent absolute paths (/workspace/x.py). The coding agent's
    system prompt names the working directory in an <env> block far from the
    end; repeat it as the last line of the system prompt, where it is attended."""
    if not msgs or msgs[0]["role"] != "system":
        return
    m = _CWD.search(msgs[0]["content"])
    if m and "Every file path you pass to a tool" not in msgs[0]["content"]:
        cwd = m.group(1).rstrip("/")
        msgs[0]["content"] += (f"\n\nThe working directory is {cwd}. Every file path you pass to a tool must start with {cwd}/ "
                               f"(for example {cwd}/main.py); run commands there with: cd {cwd} && <command>.")


# --- HTTP -----------------------------------------------------------------------------------

STATUS_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>locals serve</title>
<style>body{font:14px -apple-system,Helvetica,sans-serif;margin:2em;color:#222}table{border-collapse:collapse}
td,th{padding:4px 12px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}
code{background:#f3f3f3;padding:1px 4px}</style></head><body><h2>locals serve</h2><div id=s>loading…</div>
<script>async function r(){const s=await (await fetch('/stats')).json();let h='<p>model <code>'+s.model+'</code> · uptime '+Math.round(s.uptime_s)+' s · requests '+s.requests+' · active rows '+s.active.length+'</p>';
h+='<table><tr><th>engine</th><th>loaded</th><th>requests</th><th>prompt tok</th><th>cached</th><th>output tok</th><th>forced</th><th>passes</th><th>tool calls</th><th>tok/s (decode)</th></tr>';
for(const [e,d] of Object.entries(s.engines)){h+='<tr><td>'+e+'</td><td>'+(d.loaded?'yes':'no')+'</td><td>'+d.requests+'</td><td>'+d.prompt_tokens+'</td><td>'+(d.prompt_tokens?Math.round(100*d.cached_tokens/d.prompt_tokens)+'%':'-')+'</td><td>'+d.completion_tokens+'</td><td>'+d.forced_tokens+'</td><td>'+d.passes+'</td><td>'+d.tool_calls+'</td><td>'+(d.decode_s?(d.completion_tokens/d.decode_s).toFixed(0):'-')+'</td></tr>'}
h+='</table>';if(s.active.length){h+='<h3>active</h3><table><tr><th>engine</th><th>prompt</th><th>tokens</th><th>seconds</th></tr>';for(const a of s.active){h+='<tr><td>'+a.engine+'</td><td>'+a.prompt_tokens+'</td><td>'+a.tokens+'</td><td>'+Math.round(Date.now()/1000-a.started)+'</td></tr>'}h+='</table>'}
document.getElementById('s').innerHTML=h}r();setInterval(r,1000)</script></body></html>"""


OLLAMA_VERSION = "0.12.6"  # the Ollama API version the endpoints follow (apps check it)
OLLAMA_POSTS = ("/api/chat", "/api/generate", "/api/show", "/api/embed", "/api/embeddings", "/api/pull")
OTHER_POSTS = OLLAMA_POSTS


def make_handler(server: Server, ollama_root: bool = False):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 600  # a client that sends nothing for this long is dropped (its thread is freed)

        def log_message(self, fmt, *args):  # quiet; the server logs per request
            pass

        def _cors(self):
            origin = self.headers.get("Origin")
            if origin and _origin_ok(origin):  # a page allowed to call (see _origin_ok); others get no CORS headers
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                # the headers the page asks for, by name: "*" never covers Authorization (the Fetch standard), so a
                # page with the OpenAI SDK (dangerouslyAllowBrowser) failed its preflight
                asked = self.headers.get("Access-Control-Request-Headers")
                self.send_header("Access-Control-Allow-Headers", asked if asked else "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _chunked_body(self, deadline: float) -> Optional[bytes]:
            """A body in HTTP/1.1 chunked transfer encoding; None if malformed or past MAX_BODY. A chunk size is hex
            digits only (int() also took "-1", which read to the end of the connection past MAX_BODY, and "0x10")."""
            parts, total = [], 0
            while True:
                size = self.rfile.readline(1024).split(b";")[0].strip()
                if not re.fullmatch(rb"[0-9A-Fa-f]{1,8}", size):
                    return None
                n = int(size, 16)
                if n == 0:
                    for _ in range(100):  # trailers, up to the blank line
                        if not self.rfile.readline(1024).strip():
                            return b"".join(parts)
                    return None
                total += n
                if total > MAX_BODY:
                    return None
                part = self._read_by(n, deadline)
                if len(part) < n or self.rfile.readline(4) not in (b"\r\n", b"\n"):
                    return None
                parts.append(part)

        def _read_by(self, n: int, deadline: float) -> bytes:
            """``n`` bytes of the body, or fewer if the client stops; TimeoutError once ``deadline`` passes (the
            socket's timeout bounded each read, and a body trickled in a byte at a time kept the thread for good)."""
            import time as _t

            out = bytearray()
            while len(out) < n:
                left = deadline - _t.monotonic()
                if left <= 0:
                    raise TimeoutError
                self.connection.settimeout(left)
                piece = self.rfile.read1(min(n - len(out), 1 << 16))
                if not piece:
                    break
                out += piece
            return bytes(out)

        def _allowed(self) -> bool:
            """False (with a 403 sent) for a request a web page may not make: from another site's page,
            or under another name than this machine's (DNS rebinding)."""
            if not _host_ok(self.headers.get("Host"), getattr(server, "bind_host", "127.0.0.1")):
                self._error("this server answers to localhost only", 403)
                return False
            origin = self.headers.get("Origin")
            if origin and not _origin_ok(origin):
                self._error(f"pages from {origin} may not call this server (LOCALS_ORIGINS lists the ones that may)", 403)
                return False
            return True

        def _alive(self) -> bool:
            return _alive_socket(self.connection)

        def _path(self) -> str:
            try:
                return urlparse(self.path).path.rstrip("/")
            except ValueError:  # ("GET http://[x/health")
                return "/?"

        def send_response(self, *a, **k):
            self._answered = True
            super().send_response(*a, **k)

        def _guarded(self, handle) -> None:
            """``handle`` with a net under it: an error nobody caught is a 500 when no answer has begun (the client
            got a closed connection, and the log a traceback), and the connection is closed."""
            self._answered = False
            try:
                handle()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                self.close_connection = True
                if not self._answered:
                    try:
                        self._error(f"internal error: {type(e).__name__}: {e}", 500)
                    except Exception:  # noqa: BLE001
                        pass

        def _json(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if self.close_connection:  # (the client then knows not to send its next request on it)
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, msg: str, status=400, code=None):
            server.errors += 1
            server.log(f"{status} {self.command} {_printable(self.path)[:300]}: {msg[:200]}")
            path = self._path()
            if path.endswith(("/v1/messages", "/v1/messages/count_tokens")):  # in each API's own shape (its SDK reads it)
                kind = {403: "permission_error", 404: "not_found_error", 413: "request_too_large"}.get(
                    status, "invalid_request_error" if status < 500 else "api_error")
                return self._json(apis.anthropic_error(msg, kind), status)
            if path.startswith("/api/"):  # Ollama's
                return self._json({"error": msg}, status)
            self._json({"error": {"message": msg, "type": "invalid_request_error" if status < 500 else "server_error",
                                  "param": "messages" if code else None, "code": code}}, status)

        def _no_method(self):
            path = self._path()
            if path == "/api/delete":  # `ollama rm`: an answer the ollama command can show
                return self._error("locals does not delete models: they are in ~/.cache/huggingface/hub (delete a "
                                   "model's folder there to free the space)", 400)
            return self._error(f"{self.command} is not supported here: use GET or POST", 405)

        def do_PUT(self):
            if self._allowed():
                self._no_method()

        def do_DELETE(self):
            if self._allowed():
                self._no_method()

        def do_PATCH(self):
            if self._allowed():
                self._no_method()

        def do_OPTIONS(self):
            if not self._allowed():
                return
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _text(self, text: str, status=200):
            body = text.encode()
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _chunked(self) -> bool:
            """Whether a stream goes in chunks: an HTTP/1.0 client has no chunked encoding (it saw the chunk sizes in
            the body), and its stream ends when the connection closes."""
            return self.request_version != "HTTP/1.0"

        def _stream_start(self, content_type: str) -> None:
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            if self._chunked():
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.close_connection = True
                self.send_header("Connection", "close")
            self.end_headers()

        def _part(self, b: bytes) -> None:
            """One piece of a stream (an empty one is not sent: as a chunk, it ended the stream)."""
            if not b:
                return
            self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n" if self._chunked() else b)
            self.wfile.flush()

        def _stream_end(self) -> None:
            if self._chunked():
                self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def do_HEAD(self):
            self.send_response(200)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            self._guarded(self._get)

        def _get(self):
            if not self._allowed():
                return
            path = self._path() or "/"
            if path in ("/v1/models", "/models"):
                # OpenAI's list, with the fields of Anthropic's (type, display_name, created_at, has_more) beside them
                data = [dict(m, type="model", display_name=m.get("id"), created_at=apis._now_iso()) for m in server.models()]
                return self._json({"object": "list", "data": data, "has_more": False,
                                   "first_id": data[0]["id"] if data else None, "last_id": data[-1]["id"] if data else None})
            if path.startswith(("/v1/models/", "/models/")):  # the OpenAI SDK's models.retrieve(id)
                # the model served, under the name asked for, as a chat request with that name is answered by it
                asked = path.split("/models/", 1)[1]
                ms = server.models()
                m = next((x for x in ms if x.get("id") == asked), None) or (dict(ms[0], id=asked) if ms else None)
                return self._json(m) if m else self._error("no model is loaded", 404)
            if path == "/" and ollama_root:  # what Ollama answers, and some apps check
                return self._text("Ollama is running")
            if path == "/api/version":
                return self._json({"version": OLLAMA_VERSION})
            if path in ("/api/tags", "/api/ps"):
                return self._json({"models": server.ollama_models(running=path == "/api/ps")})
            if path == "/health":
                return self._json({"status": "ok", "engines": {e: e in server._svc for e in server.engines},
                                   "version": __version__, "build": BUILD_ID})
            if path in ("/stats", "/v1/stats"):
                return self._json(server.stats())
            if path == "/":
                body = STATUS_HTML.encode()
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._error("not found", 404)

        def _ollama_raw(self, req: dict, shown: str, model: str, fail):
            """/api/generate with raw: the prompt untemplated, through /v1/completions."""
            t0 = time.time()
            try:
                text_req = apis.ollama_to_text(req, model)
                p = server.prepare_text(text_req)
            except ContextTooLong as e:
                return fail(str(e), 400)
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                return fail(_bad(e), 400)
            if not text_req["stream"]:
                p["alive"] = self._alive
                try:
                    resp = server.text_completion(text_req, p)
                except BadRequest as e:
                    return fail(str(e), 400)
                except Exception as e:  # noqa: BLE001  # an engine error: an answer, not a dropped connection
                    import traceback

                    traceback.print_exc()
                    return fail(f"{type(e).__name__}: {e}", 500)
                return None if resp is None else self._json(apis.text_to_ollama(resp, shown, t0))

            write = self._part

            out = apis.OllamaStream(write, shown, t0, generate=True)
            self._stream_start("application/x-ndjson")
            p["alive"] = self._alive
            try:
                server.text_completion(text_req, p, out.feed)
                out.close()
                self._stream_end()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001  # mid-stream: an error line and a clean end
                import traceback

                bad = isinstance(e, BadRequest)  # (the request's own fault: its message, no traceback)
                if not bad:
                    traceback.print_exc()
                server.errors += 1
                try:
                    write((json.dumps({"error": str(e) if bad else f"{type(e).__name__}: {e}"}) + "\n").encode())
                    self._stream_end()
                except Exception:  # noqa: BLE001  # the client is gone
                    pass

        def _messages_api(self, path: str, req: dict):
            """Anthropic's Messages API (Claude Code, the Anthropic SDKs), through the Chat Completions path."""
            def fail(msg: str, status: int, kind: str = "invalid_request_error"):
                server.errors += 1
                server.log(f"{status} POST {path}: {msg[:200]}")
                return self._json(apis.anthropic_error(msg, kind), status)

            if os.environ.get("LOCALS_DEBUG"):
                _dump_request(req, "anthropic")
            name = req.get("model") or ""
            if not isinstance(req.get("messages"), list) or not req["messages"]:
                return fail("messages: at least one message is required", 400)
            try:
                chat = apis.anthropic_to_chat(req, server.chat_model(name))
                p = server.prepare(chat)
            except ContextTooLong as e:  # Anthropic's wording: Claude Code compacts the conversation on it
                server.log(f"rejected: {e}")
                return fail(f"prompt is too long: {e.n} tokens > {e.limit} maximum", 400)
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                return fail(_bad(e), 400)
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                return fail(f"{type(e).__name__}: {e}", 500, "api_error")
            if path.endswith("/count_tokens"):
                return self._json({"input_tokens": len(p["prompt"])})
            server.requests += 1
            shown = name or p["model"]
            p["alive"] = self._alive
            if not chat["stream"]:
                try:
                    resp = server.complete(chat, p)
                except BadRequest as e:
                    return fail(str(e), 400)
                except Exception as e:  # noqa: BLE001
                    import traceback

                    traceback.print_exc()
                    return fail(f"{type(e).__name__}: {e}", 500, "api_error")
                return None if resp is None else self._json(apis.chat_to_anthropic(resp, shown))

            write = self._part

            out = apis.AnthropicStream(write, shown, len(p["prompt"]))
            self._stream_start("text/event-stream; charset=utf-8")
            try:
                out.start()  # at once: the client knows the request is in, before a long prompt is read
                server.stream(chat, out.feed, p)
                out.close()
                self._stream_end()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001  # mid-stream: an error event and a clean end
                import traceback

                bad = isinstance(e, BadRequest)
                if not bad:
                    traceback.print_exc()
                server.errors += 1
                try:
                    out.error(str(e) if bad else f"{type(e).__name__}: {e}", "invalid_request_error" if bad else "api_error")
                    self._stream_end()
                except Exception:  # noqa: BLE001  # the client is gone
                    pass

        def _other_api(self, path: str, req: dict):
            """The Ollama API, through the Chat Completions path."""
            kind = "ollama"

            def fail(msg: str, status: int):
                server.errors += 1
                server.log(f"{status} POST {path}: {msg[:200]}")
                return self._json({"error": msg}, status)

            if os.environ.get("LOCALS_DEBUG"):  # the request as the client sent it (prepare saves the translated one)
                _dump_request(req, kind)
            name = req.get("model") or ""
            if not isinstance(name, str) or not isinstance(req.get("name") or "", str):
                return fail("model must be a string", 400)
            if path == "/api/show":
                return self._json(server.ollama_show(name or req.get("name") or ""))
            if path == "/api/pull":
                return self._json({"status": "success"})  # the model is the server's; nothing to pull
            if path in ("/api/embed", "/api/embeddings"):
                return fail("embeddings are not served yet", 501)
            try:
                model = server.chat_model(name)
            except ValueError as e:
                return fail(str(e), 404)
            if (path == "/api/chat" and not req.get("messages")) or (path == "/api/generate" and not req.get("prompt")):
                # Ollama's way to load a model (or, with keep_alive 0, unload it): an answer at once, nothing generated
                reason = "unload" if str(req.get("keep_alive")) in ("0", "0s", "0m") else "load"
                body = {"model": name or model, "created_at": apis._now_iso(), "done": True, "done_reason": reason}
                body.update({"message": {"role": "assistant", "content": ""}} if path == "/api/chat" else {"response": ""})
                return self._json(body)
            server.requests += 1
            if path == "/api/generate" and req.get("suffix"):  # fill-in-the-middle: not what this server does
                return fail(f"{name or model} does not support insert", 400)
            if path == "/api/generate" and req.get("raw"):  # the prompt as it is (a code completion's prefix)
                return self._ollama_raw(req, name or model, model, fail)
            try:
                chat = apis.ollama_to_chat(req, model, generate=path == "/api/generate")
                p = server.prepare(chat)
            except ContextTooLong as e:
                server.log(f"rejected: {e}")
                return fail(str(e), 400)
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                return fail(_bad(e), 400)
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                return fail(f"{type(e).__name__}: {e}", 500)
            shown = name or model
            t0 = time.time()
            try:
                if not chat.get("stream"):
                    p["alive"] = self._alive
                    resp = server.complete(chat, p)
                    if resp is None:  # the client went away
                        return
                    return self._json(apis.chat_to_ollama(resp, shown, t0, generate=path == "/api/generate"))
            except BadRequest as e:
                return fail(str(e), 400)
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                return fail(f"{type(e).__name__}: {e}", 500)

            write = self._part

            out = apis.OllamaStream(write, shown, t0, generate=path == "/api/generate")
            p["alive"] = self._alive  # a client that left is found by the probe, within a second
            self._stream_start("application/x-ndjson")
            try:
                server.stream(chat, out.feed, p)
                out.close()
                self._stream_end()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001
                import traceback

                bad = isinstance(e, BadRequest)  # (the request's own fault: its message, no traceback)
                if not bad:
                    traceback.print_exc()
                server.errors += 1
                msg = str(e) if bad else f"{type(e).__name__}: {e}"
                try:
                    out.feed(f"data: {json.dumps({'error': {'message': msg}})}\n\n".encode())
                    out.close()
                    self._stream_end()
                except Exception:  # noqa: BLE001
                    pass

        def do_POST(self):
            self._guarded(self._post)

        def _post(self):
            # until its body is read, an answer closes the connection (a 403 left the body on it, read as the next
            # request): the request's own keep-alive comes back once the body is in
            keep_alive, self.close_connection = not self.close_connection, True
            if not self._allowed():
                return
            path = self._path()
            # a client that stops sending its body (it crashed mid-upload, or announced more than it sent) held this
            # thread waiting forever: the body is read with a time limit, then the reply has none (long streams)
            self.connection.settimeout(BODY_TIMEOUT)
            import time as _t

            deadline = _t.monotonic() + BODY_TIMEOUT
            te = (self.headers.get("Transfer-Encoding") or "").lower()
            if te and "chunked" not in te:  # (a body that is not chunked nor counted: read as empty, it stayed on the connection)
                self.close_connection = True
                return self._error(f"Transfer-Encoding {te} is not supported (send chunked, or a Content-Length)", 501)
            try:
                if "chunked" in te:
                    # a body sent in chunks (Go's io.Pipe, streamed bodies of requests/httpx/Node): read as empty, it
                    # got a misleading 400 and its bytes stayed on the connection, read as the next request
                    body = self._chunked_body(deadline)
                    if body is None:
                        self.close_connection = True
                        return self._error(f"the request body must be 0 to {MAX_BODY} bytes, in valid chunks", 413)
                else:
                    try:
                        n = int(self.headers.get("Content-Length") or 0)
                    except ValueError:
                        self.close_connection = True  # (its body stays unread on the connection)
                        return self._error("invalid Content-Length")
                    if n < 0 or n > MAX_BODY:
                        self.close_connection = True
                        return self._error(f"the request body must be 0 to {MAX_BODY} bytes", 413)
                    body = self._read_by(n, deadline)
                    if len(body) < n:
                        self.close_connection = True
                        return self._error(f"the request body ended after {len(body)} of {n} bytes", 400)
            except (TimeoutError, OSError):
                self.close_connection = True
                return self._error(f"the request body did not arrive within {BODY_TIMEOUT} s", 408)
            finally:
                # (None here kept a client that stopped reading, or an idle pooled connection, on its thread forever;
                # a long stream is unaffected: each write has the limit, not the whole reply)
                self.connection.settimeout(SEND_TIMEOUT)
            self.close_connection = not keep_alive or bool(te and self.headers.get("Content-Length"))
            enc = (self.headers.get("Content-Encoding") or "").strip().lower()
            if enc in ("gzip", "deflate"):  # Claude Code can compress its request bodies (CLAUDE_CODE_GZIP_REQUEST_BODIES)
                import zlib

                try:
                    d = zlib.decompressobj(wbits=47)  # gzip or zlib, found from the header
                    body = d.decompress(body, MAX_BODY)
                except zlib.error:
                    return self._error(f"the body is not valid {enc}")
                if d.unconsumed_tail:
                    self.close_connection = True
                    return self._error(f"the request body must be at most {MAX_BODY} bytes", 413)
            elif enc not in ("", "identity"):
                return self._error(f"Content-Encoding {enc} is not supported (gzip, deflate)", 415)
            try:
                req = json.loads(body or b"{}")
            except RecursionError:  # (Python 3.11's parser stops there; 3.12's goes on, and _clean_text stops it)
                return self._error("the request is nested too deeply")
            except ValueError:
                return self._error("invalid JSON body")
            if not isinstance(req, dict):
                return self._error("the request body must be a JSON object")
            try:
                req = _clean_text(req)
            except RecursionError:
                return self._error("the request is nested too deeply")
            text_api = path in ("/v1/completions", "/completions")
            if path in OTHER_POSTS:
                return self._other_api(path, req)
            if path in apis.ANTHROPIC_POSTS:
                return self._messages_api(path, req)
            if path not in ("/v1/chat/completions", "/chat/completions") and not text_api:
                if path.endswith("/embeddings"):
                    return self._error("embeddings are not served: locals answers chat and text completions (and the "
                                       "Anthropic and Ollama APIs)", 404)
                return self._error(f"no such endpoint: POST {path}", 404)
            if not text_api and (not isinstance(req.get("messages"), list) or not req["messages"]):
                return self._error("messages must be a non-empty list")
            server.requests += 1
            try:  # errors in the request answer before any streaming starts
                p = server.prepare_text(req) if text_api else server.prepare(req)
            except ContextTooLong as e:
                server.log(f"rejected: {e}")
                return self._error(str(e), 400, "context_length_exceeded")
            except (ValueError, TypeError, KeyError, AttributeError) as e:  # a malformed request (messages, tools, schema)
                return self._error(_bad(e), 400)
            except Exception as e:  # noqa: BLE001  # our bug, not the client's
                import traceback

                traceback.print_exc()
                return self._error(f"{type(e).__name__}: {e}", 500)
            started = False

            write = self._part

            try:
                if not req.get("stream"):
                    p["alive"] = self._alive
                    resp = server.text_completion(req, p) if text_api else server.complete(req, p)
                    if resp is None:  # the client went away
                        return
                    return self._json(resp)
                p["alive"] = self._alive  # a client that hangs up (Esc in the agent) stops its reply within a second
                self._stream_start("text/event-stream; charset=utf-8")
                started = True
                if text_api:
                    server.text_completion(req, p, write)
                else:
                    server.stream(req, write, p)
                self._stream_end()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001
                import traceback

                bad = isinstance(e, BadRequest)  # the request's own fault (a grammar that does not compile)
                if not bad:
                    traceback.print_exc()
                msg = str(e) if bad else f"{type(e).__name__}: {e}"
                try:
                    if not started:
                        self._error(msg, 400 if bad else 500)
                    else:  # mid-stream: an error event and a clean end, not a second HTTP response
                        server.errors += 1
                        server.log(f"{400 if bad else 500} POST {path} (after the stream began): {msg[:200]}")
                        err = json.dumps({"error": {"message": msg, "type": "invalid_request_error" if bad else "server_error",
                                                    "param": None, "code": None}})
                        write(f"data: {err}\n\n".encode())
                        write(b"data: [DONE]\n\n")
                        self._stream_end()
                except Exception:  # noqa: BLE001  # the client is gone
                    pass

    return Handler


def _check_tool_choice(tc, names: List[str]) -> None:
    """A 400 for a tool_choice the server would not honour: an unknown name forced a call to any tool."""
    if isinstance(tc, dict):
        name = (tc.get("function") or {}).get("name") or tc.get("name")
        if tc.get("type") not in (None, "function") or not name:
            raise ValueError("tool_choice must be auto, required, none, or {\"type\": \"function\", \"function\": {\"name\": ...}}")
        if name not in names:
            raise ValueError(f"tool_choice names {name!r}, which is not one of the tools")
    elif tc not in ("auto", "required", "none", None):
        raise ValueError("tool_choice must be auto, required, none, or a named function")


def _whole(v, name: str) -> Optional[int]:
    """A count from a request (None if not given): a finite number, taken as a whole number."""
    import math

    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
        raise ValueError(f"{name} must be a whole number")
    return int(v)


def _printable(text: str) -> str:
    """``text`` for the log: control characters (a path's escape codes reached the terminal) as \\x.."""
    return "".join(c if c.isprintable() else f"\\x{ord(c):02x}" for c in text)


def _clean_text(x):
    """The request with every lone surrogate replaced by U+FFFD: JSON allows "\\ud83d" alone (a JavaScript client
    that cut a string inside an emoji sends one), the tokenizer does not, and the whole request failed with
    "TextEncodeInput must be Union[...]"."""
    if isinstance(x, str):
        try:
            x.encode("utf-8")
            return x
        except UnicodeEncodeError:
            return x.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    if isinstance(x, dict):
        return {k: _clean_text(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean_text(v) for v in x]
    return x


def _check_shape(req: dict) -> None:
    """A clear 400 for a request of the wrong shape, which otherwise failed inside the server with Python's words
    ("AttributeError: 'str' object has no attribute 'get'")."""
    if "model" in req and req["model"] is not None and not isinstance(req["model"], str):
        raise ValueError("model must be a string")
    msgs = req.get("messages")
    if isinstance(msgs, list):
        for i, m in enumerate(msgs):
            if not isinstance(m, dict):
                raise ValueError(f"messages[{i}] must be an object with a role and a content")
            c = m.get("content")
            if c is not None and not isinstance(c, (str, list)):
                raise ValueError(f"messages[{i}].content must be a string or a list of parts")
            if isinstance(c, list) and not all(isinstance(part, (str, dict)) for part in c):
                raise ValueError(f"messages[{i}].content: every part must be an object such as {{\"type\": \"text\", \"text\": ...}}")
            calls = m.get("tool_calls")
            if calls is not None and (not isinstance(calls, list) or not all(isinstance(t, dict) for t in calls)):
                raise ValueError(f"messages[{i}].tool_calls must be a list of objects")
    tools = req.get("tools")
    if tools is not None and (not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools)):
        raise ValueError("tools must be a list of objects such as {\"type\": \"function\", \"function\": {...}}")
    stop = req.get("stop")
    if stop is not None and not (isinstance(stop, str) or (isinstance(stop, list) and all(isinstance(x, str) for x in stop))):
        raise ValueError("stop must be a string or a list of strings")


def _bad(e: Exception) -> str:
    """The message of a 400: ours as it is, Python's (a shape _check_shape does not catch) with what it was."""
    return str(e) if isinstance(e, ValueError) and str(e) else f"invalid request ({type(e).__name__}: {e})"


def _unsupported(req: dict) -> None:
    """A 400 for what the server would otherwise ignore and the client would rely on."""
    n = req.get("n")
    if n is not None and n != 1:
        raise ValueError("n must be 1: send the request n times instead (they run together in one batch)")
    if req.get("logprobs") or req.get("top_logprobs"):
        raise ValueError("logprobs are not supported")


# What a stream writes after 15 s without a token: an empty line, which SSE readers skip (an event ends
# with one; a second one dispatches nothing). It was a ": keepalive" comment, which benchmark clients
# that json-parse every non-empty line (InferenceX's, vLLM's benchmark_serving) failed on: every
# request of 16 arriving together failed, as their prefill took over 15 s.
KEEPALIVE = b"\n"


def _include_usage(req: dict) -> bool:
    """stream_options.include_usage: the usage also in a last chunk of its own, with no choices (OpenAI's
    stream; clients such as vLLM's and InferenceX's benchmarks read it only there)."""
    so = req.get("stream_options")
    return isinstance(so, dict) and bool(so.get("include_usage"))


def _usage_chunk(head: dict, usage: dict) -> bytes:
    return ("data: " + json.dumps(dict(head, choices=[], usage=usage), ensure_ascii=False) + "\n\n").encode()


def _stops(req: dict) -> List[str]:
    s = req.get("stop")
    if isinstance(s, str):
        s = [s]
    return [x for x in (s or []) if isinstance(x, str) and x][:16]


def _stop_cut(text: str, stops: List[str]) -> Optional[int]:
    """Where the first stop sequence begins in ``text`` (None if none does)."""
    found = [i for i in (text.find(s) for s in stops) if i >= 0]
    return min(found) if found else None


def _stop_which(text: str, stops: List[str]) -> Optional[str]:
    """The stop sequence that ends ``text`` first (None if none does)."""
    found = [(i, s) for s, i in ((s, text.find(s)) for s in stops) if i >= 0]
    return min(found)[1] if found else None


def native_context(model_id: str) -> Optional[int]:
    """The model's context length from its config (max_position_embeddings), if known."""
    try:
        path = Path(model_id)
        if not path.exists():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(model_id, allow_patterns=["config.json"]))
        cfg = json.loads((path / "config.json").read_text())
    except Exception:
        return None
    for c in (cfg.get("text_config") or {}, cfg):
        n = c.get("max_position_embeddings")
        if isinstance(n, int) and n > 0:
            return n
    return None


def _in_think(p: dict) -> bool:
    """Whether text written at the start of the reply is inside a think block the
    prompt opened (Qwen3.5 with thinking on), not content."""
    return bool(p.get("thinking")) and not getattr(p.get("fmt"), "opens_thinking", False)


STEERED_MAX_TOKENS = 2048  # a reply decoded again after it repeated the previous call


def _temperature(req: dict) -> Optional[float]:
    """The request's sampling temperature (0 to 2, as OpenAI's), or None for the server's."""
    t = req.get("temperature")
    if t is None or isinstance(t, bool):
        return None
    try:
        return min(2.0, max(0.0, float(t)))
    except (TypeError, ValueError):
        raise ValueError("temperature must be a number")


def _sampling(req: dict) -> dict:
    """The request's own top_p, top_k and seed, checked; those it does not give are the server's (an empty dict)."""
    import math

    def number(name):
        v = req.get(name)
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name} must be a number")
        return v

    out = {}
    p = number("top_p")
    if p is not None:
        if not 0 <= p <= 1:
            raise ValueError("top_p must be from 0 to 1 (0 and 1: no filter)")
        out["top_p"] = float(p)
    k = number("top_k")
    if k is not None:
        if k < 0 or k != int(k):
            raise ValueError("top_k must be a whole number, 0 or more (0: no filter)")
        out["top_k"] = int(k)
    seed = number("seed")
    if seed is not None:
        if seed != int(seed):
            raise ValueError("seed must be a whole number")
        out["seed"] = int(seed)
    return out


LOOP_GRACE = 1024  # tokens a repetition that may be wanted goes on after it looks like a loop (Server._guard)


def _past_grace(looping: bool, n: int, state) -> bool:
    """Whether a reply that looks like a loop has gone on looking like one for LOOP_GRACE tokens (``n`` tokens so
    far; ``state.loop_at`` where it began, cleared when the repetition stops)."""
    if state is None:
        return looping
    if not looping:
        state.loop_at = None
        return False
    if getattr(state, "loop_at", None) is None:
        state.loop_at = n
    return n - state.loop_at >= LOOP_GRACE


def _looping(toks: List[int], reps: int = 3, max_period: int = 256, min_span: int = 24, short_span: int = 128) -> bool:
    """True if the reply ends in one block of tokens repeated ``reps`` times (any
    block length up to ``max_period``; the repeats spanning at least ``min_span``
    tokens): a small model saying the same sentences over and over, which no
    grammar can see. Gemma 4 E4B looped on a 45-token cycle of "I will execute
    the edit." lines until the token limit. A block shorter than 8 tokens must repeat over at least
    ``short_span`` tokens: data repeats a few tokens on purpose ("[0, 0, 0, ...]", a grid of dots),
    a loop goes on until the limit."""
    n = len(toks)
    if n < min_span:
        return False
    # a short block repeated is also periodic at its multiples: decide by the smallest period
    for p in range(1, 8):
        if n >= 32 and toks[n - 32 + p:] == toks[n - 32:n - p]:
            span = max(reps * p, min_span, short_span)
            return span <= n and toks[n - span + p:] == toks[n - span:n - p]
    last = toks[-1]
    for p in range(8, min(max_period, n // reps) + 1):
        if toks[n - 1 - p] != last:  # a period must bring the last token back
            continue
        span = max(reps * p, min_span, short_span if p < 8 else 0)
        if span <= n and toks[n - span + p:] == toks[n - span:n - p]:  # the last ``span`` tokens have period p
            return True
    return False


def _line_looping(text: str, window: int = 30, distinct: int = 6, min_len: int = 12) -> bool:
    """True if the last ``window`` sentence-length lines of an agent's reply say at
    most ``distinct`` different things: a model rephrasing the same intention
    without acting (Gemma 4, thinking: "I'll use the read tool. / Let's read the
    file. / I'll proceed with the read call." for 4,000 tokens), which an exact
    repeat check cannot see because the order of the lines changes."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= min_len]
    return len(lines) >= window and len(set(lines[-window:])) <= distinct


# Qwen3.5's template writes an assistant turn's think block only while the turn is the current one: once the
# user writes again, every earlier reply loses it, empty or not. With thinking off, each reply was generated
# after "<think>\n\n</think>\n\n", so at every new user message the conversation stopped being a prefix of what
# was cached at its first reply, and the whole previous turn (its tool calls and results) was read again: in
# an OpenCode session, 1-4k tokens and several seconds per new message, growing with the session. The empty
# block is kept on replies without reasoning (as they were generated); replies with reasoning are left as the
# template writes them (it drops old thoughts on purpose). LOCALS_KEEP_THINK=0: the template as it is.
# Gemma 4's template does the same with its thought channel, which it writes only for a call-making reply of the
# current turn: Gemma writes real thoughts with thinking off (it opens the channel itself), and at each new user
# message the previous turn read again from its first reply (in a measured OpenCode session, from 3,584 of 17,297
# tokens). Its variant keeps the thoughts a reply carries (raw replay gives them back as they were written).
_THINK_TURN = "{%- if loop.index0 > ns.last_query_index %}"
_THOUGHT_TURN = "{%- if thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls') -%}"
_KEEP_THINK = os.environ.get("LOCALS_KEEP_THINK", "1") != "0"
_STABLE: Dict[int, Optional[str]] = {}


def _stable_template(tok) -> Optional[str]:
    hf = getattr(tok, "_tokenizer", tok)
    key = id(hf)
    if key not in _STABLE:
        src = getattr(hf, "chat_template", None)
        out = None
        if isinstance(src, str) and src.count(_THINK_TURN) == 1:
            out = src.replace(_THINK_TURN, "{%- if loop.index0 > ns.last_query_index or (locals_keep_empty_think "
                                           "is defined and locals_keep_empty_think and not reasoning_content) %}")
        elif isinstance(src, str) and src.count(_THOUGHT_TURN) == 1:
            out = src.replace(_THOUGHT_TURN, "{%- if thinking_text and ((loop.index0 > ns_turn.last_user_idx and "
                                             "message.get('tool_calls')) or (locals_keep_empty_think is defined and "
                                             "locals_keep_empty_think)) -%}")
        _STABLE[key] = out
    return _STABLE[key]


def _render(tok, msgs: List[dict], tools, add_generation_prompt: bool, **kw) -> str:
    """The chat template's text; a template's own error (role order, a field it
    insists on) is the request's problem, so it surfaces as a 400."""
    tpl = _stable_template(tok) if _KEEP_THINK and "chat_template" not in kw else None
    if tpl is not None:
        kw = dict(kw, chat_template=tpl, locals_keep_empty_think=True)
    try:
        return tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=add_generation_prompt, tokenize=False, **kw)
    except Exception as e:  # jinja2's TemplateError and the template's own raise_exception
        if type(e).__name__ in ("TemplateError", "TemplateSyntaxError", "UndefinedError", "TemplateRuntimeError"):
            raise ValueError(f"the model's chat template rejected the conversation: {e}") from e
        raise


def _more_files() -> None:
    """Open files for more connections than launchd's default of 256 (each client a socket; agents hold several):
    past it, new clients were reset."""
    import resource

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = 10240 if hard == resource.RLIM_INFINITY else min(10240, hard)
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except (ValueError, OSError):
        pass


class _HTTPServer(ThreadingHTTPServer):
    # Connections waiting to be accepted: socketserver's default of 5 reset connections past it on macOS 27
    # (an M5 Pro: 1-8 of every batch of 8 or 16 requests at once failed with "Connection reset by peer")
    request_queue_size = 128

    def __init__(self, addr, handler):
        import socket

        if ":" in addr[0]:  # an IPv6 address (::1)
            self.address_family = socket.AF_INET6
        super().__init__(addr, handler)

    def handle_error(self, request, client_address):
        import sys

        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return  # a client closed its keep-alive connection
        super().handle_error(request, client_address)


def serve(server: Server, host: str = "127.0.0.1", port: int = 8321, ollama_port: int = 0) -> None:
    server.bind_host = host
    _more_files()
    httpd = _HTTPServer((host, port), make_handler(server))
    httpd.daemon_threads = True
    server.log(f"locals serve: http://{host}:{port}/v1  (models: {', '.join(m['id'] for m in server.models())})")
    server.log(f"  also: the Ollama API at http://{host}:{port}/api")
    if ollama_port:  # where Ollama listens, for apps that only know Ollama's address
        side = _HTTPServer((host, ollama_port), make_handler(server, ollama_root=True))
        side.daemon_threads = True
        threading.Thread(target=side.serve_forever, daemon=True).start()
        server.log(f"  and as Ollama at http://{host}:{ollama_port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        for svc in server._svc.values():
            svc.close()


def _on_sigterm() -> None:
    """SIGTERM (locals stop, a model switch, a restart) still ends the server at once, as it did, but first takes
    back the kernel-timing marker: a stop while a shape was being timed counted as a crash of its kernels."""
    import signal

    def term(sig, frame):
        from . import kernels

        kernels.stop_timing()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, term)


def main(argv=None):
    import argparse

    from .defaults import MAC_BLOCK as BLOCKS

    _on_sigterm()

    ap = argparse.ArgumentParser(prog="locals serve", description="OpenAI-compatible server: our engine and stock mlx-lm side by side.")
    ap.add_argument("--model", default=DEFAULT_MODEL, type=model_id, help=MODEL_HELP)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--ollama-port", type=int, default=0, help="also listen where Ollama does (11434) for apps that only "
                    "know Ollama's address (default: off; Ollama itself must not be running)")
    ap.add_argument("--engines", default="locals", help="engines that may be loaded (comma separated): locals, and mlx (stock "
                    "mlx-lm) for comparisons")
    ap.add_argument("--preload", default="locals", help="engines loaded at start (comma separated, or 'none')")
    ap.add_argument("--slots", type=int, default=16, help="maximum batch size per engine")
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.0, help="nucleus sampling at temp > 0 (0 = off; Qwen3.5 thinking: 0.95)")
    ap.add_argument("--top-k", type=int, default=0, help="top-k sampling at temp > 0 (0 = off; Qwen3.5 thinking: 20)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="cap on completion tokens per request")
    ap.add_argument("--cache-gb", type=float, default=None, help="prompt cache of our engine (default: the GPU memory left "
                    "after the weights and 4 GB, about 4 GB for a 4B model on 16 GB; the stock engine keeps 1.5)")
    ap.add_argument("--max-context", type=int, default=None, help="prompt + reply tokens per request; longer prompts get "
                    "a context_length_exceeded error (default: what fits in GPU memory next to the model, at least 65536, "
                    "at most the model's own context)")
    ap.add_argument("--thinking", action="store_true", help="think before every reply (also: model ids ending in -thinking)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=6, help="tool calls kept per reply; the reply is cut after that many")
    ap.add_argument("--walk", action="store_true", help="constraints by the walk kernel (no token tables; any grammar size)")
    ap.add_argument("--typed-tools", action="store_true", help="tool calls with parameter names and types fixed (use with --walk)")
    ap.add_argument("--blocks", type=int, nargs="?", const=BLOCKS, default=None, help="a block-diffusion model (Clover): "
                    f"write blocks of this many tokens by diffusion (default {BLOCKS}, the fastest on a Mac), not a token a "
                    "step by its causal pass; requests one at a time, no grammars")
    args = ap.parse_args(argv)
    engines = tuple(e.strip() for e in args.engines.split(",") if e.strip())
    preload = () if args.preload == "none" else tuple(e.strip() for e in args.preload.split(",") if e.strip())
    busy = _port_taken(args.host, args.port)  # before loading: a second copy of the model may not fit next to the first
    if busy:
        sys.exit(f"locals serve: port {args.port} is already in use by {busy}; "
                 + ("stop it with `locals stop`, or " if busy.startswith("a locals") else "") + "choose another port with --port")
    if args.ollama_port:  # checked now too, not after loading the model (minutes on a first start)
        if args.ollama_port == args.port:
            sys.exit("locals serve: the Ollama port (--ollama) must differ from --port (the Ollama API is served at "
                     "--port's /api too)")
        busy = _port_taken(args.host, args.ollama_port)
        if busy:
            sys.exit(f"locals serve: Ollama's port {args.ollama_port} is in use by {busy} (Ollama itself?); quit it, "
                     "or leave out --ollama (locals answers Ollama's API on its own port too)")
    from .defaults import gguf_hint, too_big

    if args.blocks:
        from .blockdiff import is_block_diffusion

        if not is_block_diffusion(args.model):
            sys.exit(f"locals serve: --blocks is for a block-diffusion model (Clover) on this Mac; {args.model} writes a "
                     "token a step")
    why = gguf_hint(args.model) or too_big(args.model, remote=True)  # `locals serve` downloads the model itself: checked before that
    if why:
        sys.exit(f"locals serve: {why}")
    try:
        srv = Server(args.model, engines=engines, preload=preload, slots=args.slots, temp=args.temp, max_tokens=args.max_tokens,
                     thinking=args.thinking, cache_gb=args.cache_gb, seed=args.seed, max_calls=args.max_calls, walk=args.walk,
                     typed_tools=args.typed_tools, top_p=args.top_p, top_k=args.top_k, max_context=args.max_context,
                     blocks=args.blocks)
    except Exception as e:  # noqa: BLE001
        why = _load_error(args.model, e)
        if why is None:
            raise
        sys.exit(f"locals serve: {why}")
    serve(srv, args.host, args.port, args.ollama_port)


def _port_taken(host: str, port: int) -> Optional[str]:
    """None if the server can listen on (host, port); else what holds it."""
    import socket

    try:
        family = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][0]  # ::1 is IPv6
    except OSError as e:
        return f"nothing it can listen on ({host}: {e})"
    s = socket.socket(family, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as the HTTP server binds
    try:
        s.bind((host, port))
        return None
    except OSError:
        pass
    finally:
        s.close()
    try:
        import urllib.request

        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as r:
            if json.loads(r.read()).get("status") == "ok":
                return "a locals server"
    except Exception:  # noqa: BLE001
        pass
    return "another program"


def _load_error(model: str, e: Exception) -> Optional[str]:
    """A one-line explanation of why ``model`` could not be loaded, or None for an unexpected error."""
    name = type(e).__name__
    if name == "GatedRepoError":
        return (f"model {model!r} is gated on Hugging Face: accept its license at https://huggingface.co/{model}, sign "
                f"in on this Mac (hf auth login, or set HF_TOKEN), then start it again.")
    if name == "RepositoryNotFoundError" or (name == "HfHubHTTPError" and "404" in str(e)):
        return (f"model {model!r} was not found on Hugging Face (or it is private or gated). Use a model id such as "
                f"mlx-community/Qwen3.5-4B-MLX-4bit, or the path of a local MLX model folder.")
    if name in ("LocalEntryNotFoundError", "OfflineModeIsEnabled") or isinstance(e, (ConnectionError, TimeoutError)):
        return f"model {model!r} is not downloaded and Hugging Face cannot be reached ({e})."
    if isinstance(e, FileNotFoundError):
        return f"cannot load {model!r}: {e}"
    if isinstance(e, ValueError) and "not supported" in str(e):
        return f"mlx-lm cannot load {model!r}: {e}"
    return None


if __name__ == "__main__":
    main()
