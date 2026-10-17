"""The Ollama API, on the same engine.

The server's own API is OpenAI Chat Completions. Many local tools default to Ollama
(Open WebUI, Continue, Cline, Roo, Kilo Code, Zed, ...), so the server also answers
Ollama's ``/api/chat``, ``/api/generate``, ``/api/tags``, ``/api/show``, ``/api/ps``
and ``/api/version``: a request becomes a Chat Completions request, the reply is
translated back, and a stream goes through a translator that reads the server's own
server-sent events and writes Ollama's newline-delimited JSON. Every client gets the
same batch, prompt cache, tool-call grammar and loop guards.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Callable, Dict, List, Optional

from .oai import IMAGE_NOTE  # noqa: E402,F401  # one wording for both APIs


def _uid(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:24]


def _openai_events(buf: bytearray, data: bytes):
    """Complete ``data: {...}`` events in the stream so far (the rest stays in ``buf``)."""
    buf.extend(data)
    while b"\n\n" in buf:
        raw, _, rest = bytes(buf).partition(b"\n\n")
        buf[:] = rest
        for line in raw.split(b"\n"):
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                try:
                    yield json.loads(line[6:])
                except ValueError:
                    pass


def _args_text(args) -> str:
    return args if isinstance(args, str) else json.dumps(args if args is not None else {}, ensure_ascii=False)


def _args_obj(args) -> dict:
    if isinstance(args, dict):
        return args
    try:
        v = json.loads(args or "{}")
        return v if isinstance(v, dict) else {"value": v}
    except ValueError:
        return {"value": args}


# --- Ollama ---------------------------------------------------------------------------------

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + ".000000Z"


def ollama_to_chat(req: dict, model: str, generate: bool = False) -> dict:
    """An Ollama /api/chat (or /api/generate) request as a Chat Completions request."""
    msgs: List[dict] = []
    if generate:
        if req.get("system"):
            msgs.append({"role": "system", "content": req["system"]})
        msgs.append({"role": "user", "content": (req.get("prompt") or "") + (" " + IMAGE_NOTE if req.get("images") else "")})
    else:
        pending: List[str] = []  # the ids of the last assistant turn's calls, for results that come without one
        for mi, m in enumerate(req.get("messages") or []):
            content = m.get("content") or ""
            if m.get("images"):
                content = (content + "\n" if content else "") + IMAGE_NOTE
            msg = {"role": m.get("role") or "user", "content": content}
            if m.get("role") == "assistant" and m.get("tool_calls"):
                # Ollama's calls have no ids: the same ones on every request (a random one changed the history the
                # template writes, so the prompt cache missed, and results could not be paired with their calls)
                msg["tool_calls"] = [{"id": tc.get("id") or f"call_{mi}_{j}", "type": "function",
                                      "function": {"name": (tc.get("function") or {}).get("name") or "",
                                                   "arguments": _args_text((tc.get("function") or {}).get("arguments") or {})}}
                                     for j, tc in enumerate(m["tool_calls"])]
                pending = [tc["id"] for tc in msg["tool_calls"]]
            if m.get("thinking") and m.get("role") == "assistant":
                msg["reasoning_content"] = m["thinking"]
            if m.get("role") == "tool":  # paired by position, as Ollama does, when it names no call
                msg["tool_call_id"] = m.get("tool_call_id") or (pending.pop(0) if pending else m.get("tool_name") or "")
            msgs.append(msg)
    opts = req.get("options") or {}
    out = {"model": model, "messages": msgs, "stream": req.get("stream", True) is not False}
    if opts.get("num_predict") and opts["num_predict"] > 0:
        out["max_tokens"] = opts["num_predict"]
    for a, b in (("temperature", "temperature"), ("top_p", "top_p"), ("stop", "stop"), ("seed", "seed")):
        if opts.get(a) is not None:
            out[b] = opts[a]
    fmt = req.get("format")
    if fmt == "json":
        out["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        out["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": fmt}}
    if req.get("tools"):
        out["tools"] = req["tools"]
    if isinstance(req.get("think"), bool):
        out["chat_template_kwargs"] = {"enable_thinking": req["think"]}
    elif isinstance(req.get("think"), str) and req["think"]:  # Ollama's levels ("low", "medium", "high")
        out["chat_template_kwargs"] = {"enable_thinking": True}
        out["reasoning_effort"] = req["think"]
    return out


def ollama_to_text(req: dict, model: str) -> dict:
    """An Ollama /api/generate request with ``raw`` (the prompt as it is, no chat template: a code
    completion's prefix) as a /v1/completions request."""
    opts = req.get("options") or {}
    out = {"model": model, "prompt": req.get("prompt") or "", "stream": req.get("stream", True) is not False}
    if opts.get("num_predict") and opts["num_predict"] > 0:
        out["max_tokens"] = opts["num_predict"]
    for a in ("temperature", "stop"):
        if opts.get(a) is not None:
            out[a] = opts[a]
    return out


def text_to_ollama(resp: dict, model: str, t0: float) -> dict:
    ch = (resp.get("choices") or [{}])[0]
    out = _ollama_done(model, resp.get("usage") or {}, ch.get("finish_reason"), t0)
    out["response"] = ch.get("text") or ""
    return out


def _ollama_done(model: str, usage: dict, finish: Optional[str], t0: float) -> dict:
    total = int((time.time() - t0) * 1e9)
    return {"model": model, "created_at": _now_iso(), "done": True, "done_reason": "length" if finish == "length" else "stop",
            "total_duration": total, "load_duration": 0, "prompt_eval_count": usage.get("prompt_tokens") or 0,
            "prompt_eval_duration": 0, "eval_count": usage.get("completion_tokens") or 0, "eval_duration": total}


def _ollama_calls(calls) -> List[dict]:
    return [{"function": {"name": (tc.get("function") or {}).get("name") or "", "arguments": _args_obj((tc.get("function") or {}).get("arguments"))}}
            for tc in calls or []]


def chat_to_ollama(resp: dict, model: str, t0: float, generate: bool = False) -> dict:
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    out = _ollama_done(model, resp.get("usage") or {}, ch.get("finish_reason"), t0)
    if generate:
        out["response"] = msg.get("content") or ""
        if msg.get("reasoning_content"):
            out["thinking"] = msg["reasoning_content"]
        return out
    m = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("reasoning_content"):
        m["thinking"] = msg["reasoning_content"]
    if msg.get("tool_calls"):
        m["tool_calls"] = _ollama_calls(msg["tool_calls"])
    out["message"] = m
    return out


class OllamaStream:
    """The server's Chat Completions stream as Ollama's newline-delimited JSON."""

    def __init__(self, write: Callable[[bytes], None], model: str, t0: float, generate: bool = False):
        self.write, self.model, self.t0, self.generate = write, model, t0, generate
        self.buf, self.calls, self.usage, self.finish = bytearray(), {}, {}, None

    def _line(self, d: dict) -> None:
        self.write((json.dumps(d, ensure_ascii=False) + "\n").encode())

    def _chunk(self, content: str = "", thinking: str = "") -> None:
        d = {"model": self.model, "created_at": _now_iso(), "done": False}
        if self.generate:
            d["response"] = content
            if thinking:
                d["thinking"] = thinking
        else:
            d["message"] = {"role": "assistant", "content": content, **({"thinking": thinking} if thinking else {})}
        self._line(d)

    def feed(self, data: bytes) -> None:
        for ev in _openai_events(self.buf, data):
            if ev.get("error"):
                self._line({"error": str((ev["error"] or {}).get("message"))})
                continue
            if ev.get("usage"):
                self.usage = ev["usage"]
            for ch in ev.get("choices") or []:
                if ch.get("text"):  # a /v1/completions event (a raw generate)
                    self._chunk(ch["text"])
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    self._chunk(d.get("content") or "", d.get("reasoning_content") or "")
                for tc in d.get("tool_calls") or []:
                    i = tc.get("index", 0)
                    c = self.calls.setdefault(i, {"function": {"name": "", "arguments": ""}})
                    fn = tc.get("function") or {}
                    c["function"]["name"] += fn.get("name") or ""
                    c["function"]["arguments"] += _args_text(fn.get("arguments") or "")
                if ch.get("finish_reason"):
                    self.finish = ch["finish_reason"]

    def close(self) -> None:
        last = _ollama_done(self.model, self.usage, self.finish, self.t0)
        if self.generate:
            last["response"] = ""
        else:
            last["message"] = {"role": "assistant", "content": ""}
            if self.calls:
                last["message"]["tool_calls"] = _ollama_calls([self.calls[i] for i in sorted(self.calls)])
        self._line(last)


def ollama_model_entry(name: str, short: str, family: str, context: int) -> dict:
    return {"name": name, "model": name, "modified_at": _now_iso(), "size": 0, "digest": uuid.uuid5(uuid.NAMESPACE_URL, name).hex,
            "details": {"parent_model": "", "format": "mlx", "family": family, "families": [family],
                        "parameter_size": _param_size(short), "quantization_level": "Q4"},
            "context_length": context}


def _param_size(short: str) -> str:
    import re

    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", short)
    return f"{m.group(1)}B" if m else ""
