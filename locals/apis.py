"""The Ollama API and Anthropic's Messages API, on the same engine.

The server's own API is OpenAI Chat Completions. Many local tools default to Ollama
(Open WebUI, Continue, Cline, Roo, Kilo Code, Zed, ...), so the server also answers
Ollama's ``/api/chat``, ``/api/generate``, ``/api/tags``, ``/api/show``, ``/api/ps``
and ``/api/version``; Claude Code and the Anthropic SDKs speak the Messages API
(``/v1/messages``, ``/v1/messages/count_tokens``). A request becomes a Chat Completions
request, the reply is translated back, and a stream goes through a translator that
reads the server's own server-sent events and writes the other API's events. Every
client gets the same batch, prompt cache, tool-call grammar and loop guards.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Callable, Dict, List, Optional

from .oai import FILE_NOTE, IMAGE_NOTE  # noqa: E402,F401  # one wording for every API


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


def _ollama_options(req: dict) -> dict:
    """Ollama's options, checked: a number where one is due (num_predict "x" failed inside the server with
    "TypeError: '>' not supported between instances of 'str' and 'int'")."""
    got = req.get("options") or {}
    if not isinstance(got, dict):
        raise ValueError("options must be an object")
    for k in ("num_predict", "temperature", "top_p", "top_k", "seed", "num_ctx"):
        v = got.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), -float("inf"))):
            raise ValueError(f"options.{k} must be a number")  # (1e999 was an OverflowError, and no answer)
    return got


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
    opts = _ollama_options(req)
    out = {"model": model, "messages": msgs, "stream": req.get("stream", True) is not False}
    if opts.get("num_predict") and opts["num_predict"] > 0:
        out["max_tokens"] = opts["num_predict"]
    for a in ("temperature", "top_p", "top_k", "stop", "seed"):
        if opts.get(a) is not None:
            out[a] = opts[a]
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
    opts = _ollama_options(req)
    out = {"model": model, "prompt": req.get("prompt") or "", "stream": req.get("stream", True) is not False}
    if opts.get("num_predict") and opts["num_predict"] > 0:
        out["max_tokens"] = opts["num_predict"]
    for a in ("temperature", "top_p", "top_k", "stop", "seed"):  # (top_p, top_k and seed were dropped here)
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


# --- Anthropic's Messages API -------------------------------------------------------------------

ANTHROPIC_POSTS = ("/v1/messages", "/v1/messages/count_tokens")


def _blocks(content) -> List[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content or [] if isinstance(b, dict)]


def _block_text(content) -> str:
    """The text of a tool result's content (a string, or text and image blocks)."""
    if isinstance(content, str):
        return content
    out = []
    for b in _blocks(content):
        if b.get("type") == "text":
            out.append(b.get("text") or "")
        elif b.get("type") == "image":
            out.append(IMAGE_NOTE)
        elif b.get("type") == "document":
            out.append(_document_text(b))
    return "\n".join(out)


def _document_text(b: dict) -> str:
    src = b.get("source") if isinstance(b.get("source"), dict) else {}
    if src.get("type") == "text" and isinstance(src.get("data"), str):
        return src["data"]
    if src.get("type") == "content":
        return _block_text(src.get("content"))
    return FILE_NOTE


def anthropic_to_chat(req: dict, model: str) -> dict:
    """A Messages API request as a Chat Completions request for ``model``: the system blocks as one system
    message, tool_use blocks as the assistant's tool_calls, tool_result blocks as tool messages (in their
    order, before the user's text), Anthropic's own server tools (web search, code execution) left out."""
    msgs: List[dict] = []
    system = req.get("system")
    text = system if isinstance(system, str) else "\n\n".join(
        b.get("text") or "" for b in _blocks(system) if b.get("type") == "text")
    if text.strip():
        msgs.append({"role": "system", "content": text})
    for m in req.get("messages") or []:
        if not isinstance(m, dict):
            raise ValueError("messages: every message must be an object")
        role, blocks = m.get("role"), _blocks(m.get("content"))
        if role == "assistant":
            texts, thoughts, calls = [], [], []
            for b in blocks:
                kind = b.get("type")
                if kind == "text":
                    texts.append(b.get("text") or "")
                elif kind == "thinking":
                    thoughts.append(b.get("thinking") or "")
                elif kind in ("tool_use", "server_tool_use"):
                    calls.append({"id": b.get("id") or _uid("toolu_"), "type": "function",
                                  "function": {"name": b.get("name") or "", "arguments": _args_text(b.get("input") or {})}})
            out = {"role": "assistant", "content": "".join(texts)}
            if thoughts:
                out["reasoning_content"] = "\n".join(thoughts)
            if calls:
                out["tool_calls"] = calls
            msgs.append(out)
        elif role in ("user", "system"):  # a mid-conversation system message: an instruction in the user's turn
            parts: List[str] = []
            for b in blocks:
                kind = b.get("type")
                if kind == "tool_result":
                    if parts:  # (results answer the previous turn's calls: they come first)
                        msgs.append({"role": "user", "content": "\n".join(parts)})
                        parts = []
                    content = _block_text(b.get("content"))
                    msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id") or "",
                                 "content": ("Error: " + content) if b.get("is_error") else content})
                elif kind == "text":
                    parts.append(b.get("text") or "")
                elif kind == "image":
                    parts.append(IMAGE_NOTE)
                elif kind == "document":
                    parts.append(_document_text(b))
            if parts:
                msgs.append({"role": "user", "content": "\n".join(parts)})
        else:
            raise ValueError(f"messages: unknown role {role!r}")
    chat = {"model": model, "messages": msgs, "stream": bool(req.get("stream")), "stream_options": {"include_usage": True}}
    tools = []
    for t in req.get("tools") or []:
        if not isinstance(t, dict) or not t.get("name") or (t.get("type") not in (None, "custom") and "input_schema" not in t):
            continue  # an Anthropic server tool (web_search_..., code_execution_...): run by Anthropic, not here
        tools.append({"type": "function", "function": {"name": t["name"], "description": t.get("description") or "",
                                                       "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}})
    if tools:
        chat["tools"] = tools
    tc = req.get("tool_choice") if isinstance(req.get("tool_choice"), dict) else {}
    if tc.get("type") == "any":
        chat["tool_choice"] = "required"
    elif tc.get("type") == "tool" and tc.get("name"):
        chat["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
    elif tc.get("type") == "none":
        chat["tool_choice"] = "none"
    if tc.get("disable_parallel_tool_use"):
        chat["parallel_tool_calls"] = False
    for k in ("max_tokens", "temperature", "top_p", "top_k"):
        if req.get(k) is not None:
            chat[k] = req[k]
    if req.get("stop_sequences"):
        chat["stop"] = list(req["stop_sequences"])
    thinking = req.get("thinking") if isinstance(req.get("thinking"), dict) else {}
    if thinking.get("type") in ("enabled", "adaptive"):
        chat["chat_template_kwargs"] = {"enable_thinking": True}
    elif thinking.get("type") == "disabled":
        chat["chat_template_kwargs"] = {"enable_thinking": False}
    last = chat["messages"][-1] if chat.get("messages") else None
    if last and last["role"] == "assistant" and not last.get("tool_calls") and last.get("content"):
        # a prefill: the reply continues this text (the answer holds only the continuation); rendered as a finished
        # turn, the model began a new reply (prefill {"name": then {"name": "Ann", ...})
        chat["messages"].pop()
        chat["_prefill"] = last["content"]
        chat["chat_template_kwargs"] = {"enable_thinking": False}  # (Anthropic allows no prefill with thinking)
    return chat


_STOP_REASONS = {"tool_calls": "tool_use", "length": "max_tokens", "stop": "end_turn"}


def _anthropic_usage(usage: dict) -> dict:
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    return {"input_tokens": max(0, prompt - cached), "output_tokens": int(usage.get("completion_tokens") or 0),
            "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}


def chat_to_anthropic(resp: dict, model: str) -> dict:
    """A Chat Completions reply as a Messages API message."""
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content: List[dict] = []
    if msg.get("reasoning_content"):
        content.append({"type": "thinking", "thinking": msg["reasoning_content"], "signature": ""})
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        content.append({"type": "tool_use", "id": c.get("id") or _uid("toolu_"), "name": fn.get("name") or "",
                        "input": _args_obj(fn.get("arguments"))})
    return {"id": _uid("msg_"), "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": "stop_sequence" if isinstance(ch.get("stop_reason"), str) else _STOP_REASONS.get(ch.get("finish_reason"), "end_turn"),
            "stop_sequence": ch.get("stop_reason") if isinstance(ch.get("stop_reason"), str) else None,
            "usage": _anthropic_usage(resp.get("usage") or {})}


def anthropic_error(message: str, kind: str = "invalid_request_error") -> dict:
    return {"type": "error", "error": {"type": kind, "message": message}}


class AnthropicStream:
    """The server's Chat Completions stream as the Messages API's events: message_start at once, then a content
    block per kind (thinking, text, each tool call) with its deltas, ping for the server's keepalives, and
    message_delta (stop reason, usage) and message_stop at the end."""

    def __init__(self, write: Callable[[bytes], None], model: str, input_tokens: int):
        self.write, self.model, self.input_tokens = write, model, input_tokens
        self.buf, self.index, self.open, self.usage, self.finish = bytearray(), -1, None, {}, None
        self.failed = False

    def _event(self, name: str, d: dict) -> None:
        self.write(f"event: {name}\ndata: {json.dumps(d, ensure_ascii=False)}\n\n".encode())

    def start(self) -> None:
        self._event("message_start", {"type": "message_start", "message": {
            "id": _uid("msg_"), "type": "message", "role": "assistant", "model": self.model, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": 0, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}}})

    def _begin(self, kind, block: dict) -> None:
        self._end()
        self.index += 1
        self.open = kind
        self._event("content_block_start", {"type": "content_block_start", "index": self.index, "content_block": block})

    def _end(self) -> None:
        if self.open is None:
            return
        if self.open == "thinking":
            self._delta({"type": "signature_delta", "signature": ""})
        self._event("content_block_stop", {"type": "content_block_stop", "index": self.index})
        self.open = None

    def _delta(self, delta: dict) -> None:
        self._event("content_block_delta", {"type": "content_block_delta", "index": self.index, "delta": delta})

    def feed(self, data: bytes) -> None:
        if not data.strip():  # the server's keepalive (a long prompt being read, a call being written)
            self._event("ping", {"type": "ping"})
            return
        for ev in _openai_events(self.buf, data):
            if ev.get("error"):
                self.error(str((ev["error"] or {}).get("message") or ev["error"]))
                continue
            if ev.get("usage"):
                self.usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("reasoning_content"):
                    if self.open != "thinking":
                        self._begin("thinking", {"type": "thinking", "thinking": "", "signature": ""})
                    self._delta({"type": "thinking_delta", "thinking": d["reasoning_content"]})
                if d.get("content"):
                    if self.open != "text":
                        self._begin("text", {"type": "text", "text": ""})
                    self._delta({"type": "text_delta", "text": d["content"]})
                for tc in d.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    key = ("tool", tc.get("index", 0))
                    if self.open != key:
                        self._begin(key, {"type": "tool_use", "id": tc.get("id") or _uid("toolu_"), "name": fn.get("name") or "",
                                          "input": {}})
                    if fn.get("arguments"):
                        self._delta({"type": "input_json_delta", "partial_json": _args_text(fn["arguments"])})
                if ch.get("finish_reason"):
                    self.finish = ch["finish_reason"]
                    if isinstance(ch.get("stop_reason"), str):  # a stop sequence matched
                        self.stop_seq = ch["stop_reason"]

    def error(self, message: str, kind: str = "api_error") -> None:
        self._end()
        self.failed = True
        self._event("error", anthropic_error(message, kind))

    def close(self) -> None:
        if self.failed:
            return
        self._end()
        u = _anthropic_usage(self.usage)
        seq = getattr(self, "stop_seq", None)
        self._event("message_delta", {"type": "message_delta", "delta": {
            "stop_reason": "stop_sequence" if seq else _STOP_REASONS.get(self.finish, "end_turn"), "stop_sequence": seq},
            "usage": u})
        self._event("message_stop", {"type": "message_stop"})
