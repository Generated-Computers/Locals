"""OpenAI chat-completions <-> Qwen3.5 native tool calling.

Qwen3.5's chat template renders tools and expects a call as

    <tool_call>
    <function=NAME>
    <parameter=P>
    raw value
    </parameter>
    </function>
    </tool_call>

This module converts OpenAI-style messages to what the template wants,
builds the reply grammar our engine decodes under, parses replies back into
OpenAI ``tool_calls`` and streams them.

The grammar (``reply_regex``) is a regular language: optional free text that
does not contain ``<tool_call>``, then zero or more calls whose tags are
literal. Every tag is a forced chain for the engine, the tool name is drawn
from the tools offered, a reply cut by the token budget still closes its call
(budget masking), and a reply always parses. Parameter names and values are
free (their types are coerced from the tool's schema after parsing), which
keeps the automaton at a few hundred states: a table over the 248k-token
vocabulary costs |Q| x V x 3 bytes, so the state count is what the engine
pays for.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
THINK_CLOSE = "</think>"


# --- messages ------------------------------------------------------------------------------


IMAGE_NOTE = "[an image was attached here; this model reads text only]"
FILE_NOTE = "[a file was attached here; this model reads only the text of the conversation]"

def text_of(content) -> str:
    """The text of an OpenAI message content (a string or a list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for part in content:
        if isinstance(part, str):
            out.append(part)
        elif part.get("type") == "text":
            out.append(part.get("text") or "")
        elif part.get("type") in ("image_url", "image", "input_image"):
            out.append(IMAGE_NOTE)  # the model can say it cannot see it (a bare "[image omitted]" it took for text)
        elif part.get("type") in ("file", "input_file"):
            out.append(FILE_NOTE)
    return "\n".join(out)  # parts are separate pieces of text (vLLM joins them the same way)


def _args_dict(args) -> dict:
    if isinstance(args, dict):
        return args
    if args is None or args == "":
        return {}
    if isinstance(args, str):
        try:
            v = json.loads(args)
            if isinstance(v, dict):
                return v
            return {"value": v}
        except ValueError:
            return {"value": args}
    return {"value": args}


def id_key(call_id: Optional[str]) -> str:
    """A tool-call id as clients may send it back: some (OpenClaw) strip
    everything but letters and digits from the ids they were given."""
    return re.sub(r"[^A-Za-z0-9]", "", call_id or "")


def convert_messages(messages: List[dict], raw: Optional[Dict[str, str]] = None) -> List[dict]:
    """OpenAI messages -> messages for Qwen's chat template: system messages
    merged into one at the front, content flattened to text, tool-call
    arguments as dicts, tool results as ``tool`` messages. ``raw`` maps
    tool-call ids to the reply text they were parsed from: such an assistant
    turn is rendered verbatim, so the prompt re-tokenizes to the tokens the
    model produced and the prompt cache is hit (ids compared by ``id_key``)."""
    system: List[str] = []
    out: List[dict] = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            t = text_of(m.get("content"))
            if t.strip():
                system.append(t)
        elif role == "user":
            out.append({"role": "user", "content": text_of(m.get("content"))})
        elif role == "assistant":
            ids = [tc.get("id") for tc in m.get("tool_calls") or []]
            replay = raw.get(id_key(ids[0])) if raw and ids else None  # one lookup: the server may drop old entries meanwhile
            if replay is not None:
                if "</think>" in replay:
                    # a thinking reply: its thoughts as reasoning_content, split as Qwen3.5's template did
                    # itself; Qwen3.8's template no longer splits, and wrote the whole reply after an
                    # empty think block
                    out.append({"role": "assistant",
                                "reasoning_content": replay.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n"),
                                "content": replay.split("</think>")[-1].lstrip("\n")})
                else:
                    out.append({"role": "assistant", "content": replay})
                continue
            msg: dict = {"role": "assistant", "content": text_of(m.get("content"))}
            rc = m.get("reasoning_content") or m.get("reasoning")
            if isinstance(rc, str) and rc.strip():
                msg["reasoning_content"] = rc
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                calls.append({"id": tc.get("id") or _call_id(), "type": "function",
                              "function": {"name": fn.get("name") or "", "arguments": _args_dict(fn.get("arguments"))}})
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "content": text_of(m.get("content")), "tool_call_id": m.get("tool_call_id") or ""})
    if system:
        out.insert(0, {"role": "system", "content": "\n\n".join(system)})
    if not any(m["role"] == "user" and not m["content"].startswith("<tool_response>") for m in out):
        # the template requires a user query; a tool-only tail still has one
        # somewhere in real conversations, so this only covers odd clients
        out.append({"role": "user", "content": ""})
    return out


def tool_specs(tools: Optional[List[dict]]) -> List[dict]:
    """The tools in the ``{"type": "function", "function": {...}}`` form the template renders."""
    out = []
    for t in tools or []:
        if t.get("type", "function") == "function" and "function" in t:
            fn = t["function"]
        else:
            fn = t
        if not fn.get("name"):
            continue
        spec = {"name": fn["name"], "description": fn.get("description") or ""}
        spec["parameters"] = fn.get("parameters") or {"type": "object", "properties": {}}
        out.append({"type": "function", "function": spec})
    return out


# --- the reply grammar -------------------------------------------------------------------------

def _cls(ch: str) -> str:
    return ch if ch.isalnum() or ch == "_" else "\\" + ch


def not_containing(*lits: str) -> str:
    """A regex for exactly the strings that contain none of ``lits``, where
    every literal starts with the same character (``<`` for most formats'
    tags, ``[`` for Mistral's) and has no other one. A string is split at
    every such character into segments without it; it avoids the literals iff
    no segment after one starts with the rest of a literal. In the trie of
    those rests, such a segment either follows a path to a node that is not a
    complete rest and ends there (the lead character or the end of the string
    follows), or leaves the trie at some node with a character that is not one
    of its children."""
    lead = lits[0][0]
    rests = []
    for lit in lits:
        assert lit.startswith(lead) and lead not in lit[1:], lit
        rests.append(lit[1:])
    L = _cls(lead)
    alts = []
    seen = set()
    for rest in rests:
        for i in range(len(rest)):
            p = rest[:i]
            if p in seen or any(p.startswith(r) for r in rests):
                continue
            seen.add(p)
            kids = {r[i] for r in rests if r.startswith(p) and len(r) > i}
            alts.append(re.escape(p) + "[^" + L + "".join(_cls(c) for c in sorted(kids)) + "][^" + L + "]*")
            alts.append(re.escape(p))
    return "[^" + L + "]*(" + L + "(" + "|".join(alts) + "))*"


FREE_TEXT = ("<tool_call>", "<function=")  # free text may not start a call, with or without its wrapper

NAME_RE = r"[A-Za-z0-9_.\-]+"
PARAM_RE_ = r"[A-Za-z0-9_.\-]+"


def _value_regex(schema: Optional[dict]) -> str:
    """A parameter value's language from its JSON-schema type (raw text, as the
    template writes values): typed where the type is simple, free otherwise."""
    free = not_containing("</parameter>")
    if not schema:
        return free
    if "enum" in schema:
        return "(" + "|".join(re.escape(str(v)) for v in schema["enum"]) + ")"
    t = schema.get("type")
    if isinstance(t, list):
        t = [x for x in t if x != "null"]
        t = t[0] if len(t) == 1 else None
    if t == "integer":
        return r"-?[0-9]+"
    if t == "number":
        return r"-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?"
    if t == "boolean":
        return "(true|false)"
    return free


def typed_call_regex(tools: List[dict]) -> str:
    """One call of one of the tools with every parameter name fixed, in the
    schema's order, required ones mandatory, values typed (Section: the
    walk kernel makes the resulting automaton, thousands of states, free)."""
    calls = []
    for t in tool_specs(tools):
        fn = t["function"]
        params = (fn.get("parameters") or {}).get("properties") or {}
        required = set((fn.get("parameters") or {}).get("required") or [])
        body = ""
        for p, sc in params.items():
            piece = "<parameter=" + re.escape(p) + ">\\n" + _value_regex(sc) + "\\n</parameter>\\n"
            body += piece if p in required else "(" + piece + ")?"
        calls.append("<tool_call>\\n<function=" + re.escape(fn["name"]) + ">\\n" + body + "</function>\\n</tool_call>")
    return "(" + "|".join(calls) + ")"


def call_regex(names: Optional[Iterable[str]] = None) -> str:
    """One tool call. ``names``: the tool names allowed (None: any name)."""
    names = [n for n in (names or []) if n]
    name = "(" + "|".join(re.escape(n) for n in names) + ")" if names else NAME_RE
    value = not_containing("</parameter>")
    param = "<parameter=" + PARAM_RE_ + ">\\n" + value + "\\n</parameter>\\n"
    return "<tool_call>\\n<function=" + name + ">\\n(" + param + ")*</function>\\n</tool_call>"


def reply_regex(tools: List[dict], tool_choice: Any = "auto", thinking: bool = False, typed: bool = False,
                fmt=None, preamble: bool = True, parallel: bool = True) -> Optional[str]:
    """The grammar of a reply to a request that offers ``tools``: free text
    (without ``<tool_call>``), then calls as ``tool_choice`` demands:
    ``auto`` zero or more, ``required`` one or more, a named function exactly
    that one, ``none`` free text only (the tools stay in the prompt, so its
    cache still matches the turns that offered them). ``preamble=False``: a
    reply that must call starts with the call (the server's choice: a small
    model given room to talk first can talk until the token limit).
    ``parallel=False`` (a request's ``parallel_tool_calls: false``): at most
    one call. With ``thinking`` the reply starts inside an open ``<think>``
    block. ``fmt`` is the model's tool-call format (locals.formats; default
    Qwen3.5's XML calls, the only one with ``typed`` calls)."""
    names = [t["function"]["name"] for t in tool_specs(tools)]
    if fmt is not None and fmt.name != "qwen_xml":
        return fmt.reply_regex(names, tool_choice, thinking, preamble=preamble, parallel=parallel)
    free = not_containing(*FREE_TEXT)
    if not names:
        return None
    specs = tool_specs(tools)
    lead = free if preamble else ""
    if tool_choice == "none":
        body = free
    elif isinstance(tool_choice, dict):
        fn = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
        chosen = [t for t in specs if t["function"]["name"] == fn] or specs
        body = lead + (typed_call_regex(chosen) if typed else call_regex([fn] if fn in names else names))
    elif tool_choice == "required":
        call = typed_call_regex(specs) if typed else call_regex(names)
        body = lead + call + ("(\\n" + call + ")*" if parallel else "")
    else:
        call = typed_call_regex(specs) if typed else call_regex(names)
        body = free + "(" + call + ("(\\n" + call + ")*" if parallel else "") + ")?"
    if thinking:
        body = not_containing(THINK_CLOSE) + "</think>\\n\\n" + body
    return body


# --- parsing replies ---------------------------------------------------------------------------

_FN_OPEN = re.compile(r"<tool_call>\s*<function=([^>\n]+)>\n?")
_P_OPEN = re.compile(r"\s*<parameter=([^>\n]+)>\n?")
_FN_CLOSE = re.compile(r"\s*</function>\s*(?:</tool_call>|\Z)")


def qwen_calls(text: str) -> List[Tuple[str, Dict[str, str], int]]:
    """The complete calls in a reply's calls part: (name, raw parameter values, end), read the way the
    grammar writes them: each value runs to its "</parameter>" (the grammar keeps that out of values),
    so a value may hold "</function>" or a whole example call (a test fixture, documentation). A call
    cut before its "</function>" is not a call; one without its "</tool_call>" at the very end is."""
    out, pos = [], 0
    while True:
        m = _FN_OPEN.search(text, pos)
        if not m:
            return out
        name, j, args = m.group(1).strip(), m.end(), {}
        while j >= 0:
            pm = _P_OPEN.match(text, j)
            if not pm:
                break
            k = text.find("</parameter>", pm.end())
            if k < 0:
                j = -1
                break
            v = text[pm.end():k]
            args[pm.group(1).strip()] = v[:-1] if v.endswith("\n") else v
            j = k + len("</parameter>")
        cm = _FN_CLOSE.match(text, j) if j >= 0 else None
        if cm is None:  # cut, or not a call the grammar writes: skip this opener
            if j < 0:
                return out
            pos = m.end()
            continue
        out.append((name, args, cm.end()))
        pos = cm.end()


def _call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


def _types(schema: dict) -> set:
    """The JSON types a schema allows: its "type" (one or a list) and its anyOf/oneOf branches' types."""
    out = set()
    t = schema.get("type")
    out |= set(t) if isinstance(t, list) else ({t} if t else set())
    for br in (schema.get("anyOf") or schema.get("oneOf") or []):
        if isinstance(br, dict):
            out |= _types(br)
    return out


def coerce(value: str, schema: Optional[dict]):
    """A raw parameter value (Qwen's XML calls write every value as text) as the JSON type its schema
    declares. Where the schema allows a string (Optional[str]: anyOf string, null), or says nothing,
    the text stays text: "1.0" or a file that happens to be JSON is not turned into a number or an
    object."""
    import math

    if not schema:
        return value
    types = _types(schema)
    s = value.strip()
    if not types or "string" in types:
        return value
    if "null" in types and s == "null":
        return None
    for t in [x for x in ("integer", "number", "boolean", "array", "object") if x in types]:
        if t in ("integer", "number"):
            try:
                v = int(s) if t == "integer" or not any(c in s for c in ".eE") else float(s)
            except ValueError:
                try:
                    v = float(s)
                except ValueError:
                    continue
                if t == "integer" and v.is_integer():
                    v = int(v)
            if isinstance(v, float) and not math.isfinite(v):  # NaN and inf are not JSON
                continue
            return v
        if t == "boolean":
            if s.lower() in ("true", "yes", "1"):
                return True
            if s.lower() in ("false", "no", "0"):
                return False
            continue
        try:
            v = json.loads(s)
        except ValueError:
            continue
        if (t == "array") == isinstance(v, list) and (t == "object") == isinstance(v, dict):
            return v
    return value


def split_reasoning(text: str, fmt=None, thinking: Optional[bool] = None) -> Tuple[str, str]:
    """(reasoning, the rest) of a reply: Qwen's <think> block, or the format's own thought block,
    which the model opens itself (Gemma 4: <|channel>thought ... <channel|>; unclosed when the token
    limit cut it). ``thinking``: the prompt opened a think block (the reply starts inside it; one the
    token limit cut is all reasoning). Otherwise only a reply that opens a block itself has one: a
    "</think>" further on (in a tool call's code, in an answer about it) is text."""
    if fmt is not None and getattr(fmt, "opens_thinking", False):
        s = text.lstrip()
        if s.startswith(fmt.think_open):
            body, sep, rest = s[len(fmt.think_open):].partition(fmt.think_close)
            return body.strip(), (rest.lstrip("\n") if sep else "")
        return "", text
    s = text.lstrip()
    opened = s.startswith("<think>")
    if thinking or opened or (thinking is None and THINK_CLOSE in text and "<think>" not in text):
        body = s[len("<think>"):] if opened else text
        reasoning, sep, rest = body.partition(THINK_CLOSE)
        if not sep:
            return (body.strip(), "") if (thinking or opened) else ("", text)
        return reasoning.strip(), rest.lstrip("\n")
    return "", text


def parse_reply(text: str, tools: Optional[List[dict]] = None, fmt=None,
                thinking: Optional[bool] = None) -> Tuple[str, str, List[dict]]:
    """A finished reply -> (reasoning, content, tool_calls) in OpenAI form.
    ``fmt``: the model's tool-call format (default Qwen3.5's XML calls); ``thinking``: see split_reasoning.
    Calls to tools the request did not offer are not calls."""
    reasoning, text = split_reasoning(text, fmt, thinking)
    schemas: Dict[str, dict] = {}
    for t in tool_specs(tools):
        props = (t["function"].get("parameters") or {}).get("properties") or {}
        schemas[t["function"]["name"]] = props
    if fmt is not None and fmt.name != "qwen_xml":
        content, part = fmt.split(text)
        calls = []
        found = fmt.parse_calls(part)
        if getattr(fmt, "loose_calls", False) and not part.startswith(fmt.opener):
            found = [(n, a) for n, a in found if n in schemas]  # a JSON block is a call only if it names an offered tool
        if schemas:
            found = [(n, a) for n, a in found if n in schemas]
        for name, args in found:
            props = schemas.get(name, {})
            args = {k: (coerce(v, props.get(k)) if isinstance(v, str) and (props.get(k) or {}).get("type") not in (None, "string")
                        else v) for k, v in args.items()}
            calls.append({"id": fmt.new_id(), "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
        if part and not calls:  # the calls part did not parse (only without our grammar): keep it as text
            content = text
        return reasoning, content.rstrip(), calls
    calls = []
    i = text.find(TOOL_OPEN)
    content = text if i < 0 else text[:i]
    if i >= 0:
        for name, raw, _ in qwen_calls(text[i:]):
            if schemas and name not in schemas:
                continue
            props = schemas.get(name, {})
            args = {k: coerce(v, props.get(k)) for k, v in raw.items()}
            calls.append({"id": _call_id(), "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return reasoning, content.rstrip(), calls


class ReplyStream:
    """Turns a growing reply text into (kind, delta) events: ``reasoning``
    while inside the think block, ``content`` up to the first ``<tool_call>``.
    A possible partial tag at the end is held back until it is decided."""

    def __init__(self, thinking: bool = False, fmt=None):
        self.think_open = getattr(fmt, "think_open", "<think>")
        self.think_close = getattr(fmt, "think_close", THINK_CLOSE)
        # a model that opens its reasoning itself (Gemma 4) may or may not start with it
        self.maybe = bool(getattr(fmt, "opens_thinking", False))
        self.mode = "reasoning" if thinking or self.maybe else "content"
        self.sent = 0  # index into the full text up to which everything was emitted
        self.base = 0  # start of the content section
        self.opener = fmt.opener if fmt is not None else TOOL_OPEN
        # formats whose calls are the whole reply (Llama's JSON, Mistral's [TOOL_CALLS]):
        # the first character decides between text and calls
        self.at_start = fmt is not None and fmt.name in ("llama3_json", "mistral", "mistral_legacy")
        # Llama may write a sentence before its call: from the call's start on, text is held back
        self.after_text = '{"name"' if fmt is not None and fmt.name == "llama3_json" else "\x00"
        # a call the model began without its opening token (Gemma 4 does, now and then)
        self.bare = getattr(fmt, "bare_start", None)
        # models without tool training write calls as JSON blocks: from a fence (or a JSON
        # reply's first brace) on, text is held back until the reply shows what it is
        self.loose = bool(getattr(fmt, "loose_calls", False))
        self.content_sent = ""  # content emitted so far

    @staticmethod
    def _safe(text: str, lit: str, start: int) -> int:
        """Length of text that cannot be the start of ``lit``."""
        for k in range(len(lit) - 1, 0, -1):
            if text.endswith(lit[:k]) and len(text) - k >= start:
                return len(text) - k
        return len(text)

    def feed(self, text: str) -> List[Tuple[str, str]]:
        out = self._feed(text)
        self.content_sent += "".join(d for k, d in out if k == "content")
        return out

    def unsent(self, content: str) -> str:
        """The part of a reply's final ``content`` this stream held back (it looked like a
        call, and the reply turned out to be text): to be sent before the reply ends."""
        return content[len(self.content_sent):] if content.startswith(self.content_sent) else ""

    def _feed(self, text: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        if self.mode == "reasoning":
            head = text.lstrip()
            if self.sent == 0 and head.startswith(self.think_open):  # models that open the block themselves (Qwen3, Gemma 4)
                self.sent = text.index(self.think_open) + len(self.think_open)
                while self.sent < len(text) and text[self.sent] == "\n":
                    self.sent += 1
                self.maybe = False
            elif self.sent == 0 and self.think_open.startswith(head):
                return out
            elif self.sent == 0 and self.maybe:  # no reasoning block: the reply is content
                self.mode = "content"
            if self.mode == "reasoning":
                j = text.find(self.think_close, self.sent)
                if j >= 0:
                    if j > self.sent:
                        out.append(("reasoning", text[self.sent:j]))
                    k = j + len(self.think_close)
                    while k < len(text) and text[k] == "\n":
                        k += 1
                    self.sent = self.base = k
                    self.mode = "content"
                else:
                    n = self._safe(text, self.think_close, self.sent)
                    if n > self.sent:
                        out.append(("reasoning", text[self.sent:n]))
                        self.sent = n
                    return out
        if self.mode == "content":
            while self.sent == self.base and self.sent < len(text) and text[self.sent] == "\n":
                self.sent = self.base = self.sent + 1  # the blank lines after </think> are not content
            if self.at_start:
                head = text[self.base:].lstrip()
                if not head:
                    return out
                if head.startswith(self.opener[0]) or head.startswith("<|python_tag|>"):
                    self.mode = "tool"
                    return out
                self.at_start = False  # text first: content, up to a call that may follow it
                self.opener = self.after_text
            if self.bare and self.sent == self.base:
                head = text[self.base:].lstrip()
                if self.bare.startswith(head):
                    return out  # might still become the bare call
                if head.startswith(self.bare):
                    self.mode = "tool"
                    return out
                self.bare = None
            if self.loose and self.sent == self.base and text[self.base:].lstrip().startswith("{"):
                self.mode = "tool"  # a reply that is a JSON object may be a call
                return out
            j = text.find(self.opener, self.base)
            if self.loose:
                k = text.find("```", self.base)
                j = k if j < 0 or 0 <= k < j else j
            if j >= 0:
                if j > self.sent:
                    out.append(("content", text[self.sent:j]))
                self.sent = j
                self.mode = "tool"
            else:
                n = self._safe(text, self.opener, self.sent)
                if self.loose:
                    n = min(n, self._safe(text, "```", self.sent))
                if n > self.sent:
                    out.append(("content", text[self.sent:n]))
                    self.sent = n
        return out


# --- request helpers --------------------------------------------------------------------------

# Whitespace a ``json_object`` reply may use between JSON tokens: a space, or a
# newline and an indent. Its keys are free, and in compact form the model,
# which writes ``{`` and a newline by habit, is pushed into keys such as
# ``"/a"`` (both engines did this); a schema's keys are forced, so schemas stay compact.
JSON_OBJECT_WS = r"( |\n *)?"


def json_object_regex() -> str:
    from outlines_core.json_schema import build_regex_from_schema

    return build_regex_from_schema('{"type": "object"}', JSON_OBJECT_WS)


def response_schema(req: dict) -> Optional[dict]:
    """The JSON schema a request's ``response_format`` asks for, if any."""
    rf = req.get("response_format")
    if not isinstance(rf, dict):
        return None
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema") or {}
        return js.get("schema") or {}
    if rf.get("type") == "json_object":
        return {"type": "object"}
    return None


def engine_of(model: str, default: str = "locals") -> str:
    """``mlx`` for the stock mlx-lm engine, ``locals`` for ours. The stock engine
    is chosen only by an explicit prefix (``mlx/...`` as /v1/models lists it,
    or ``stock/...``): any other id, including a Hugging Face id such as
    ``mlx-community/Qwen3.5-4B-MLX-4bit``, is served by ours."""
    m = (model or "").lower()
    if m.startswith(("mlx/", "stock/", "stock-")):
        return "mlx"
    return default


def thinking_of(model: str, default: bool = False) -> bool:
    return "think" in (model or "").lower() or default
