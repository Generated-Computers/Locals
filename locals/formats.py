"""Tool-call formats of chat models, for the server's reply grammar and parser.

Every model family writes a tool call its own way, as its chat template
shows it (and the template renders earlier calls the same way):

  qwen_xml     Qwen3.5, Qwen3-Coder   <tool_call>\\n<function=NAME>\\n<parameter=P>\\nV\\n</parameter>\\n</function>\\n</tool_call>
  hermes       Qwen2.5, Qwen3, Hermes <tool_call>\\n{"name": "NAME", "arguments": {...}}\\n</tool_call>
  llama3_json  Llama 3.1-3.3          {"name": "NAME", "parameters": {...}}   (the whole reply; one call)
  mistral      Mistral 7B v0.3        [TOOL_CALLS] [{"name": "NAME", "arguments": {...}}, ...]
  mistral_v11  Ministral 3, Devstral  [TOOL_CALLS]NAME[ARGS]{...}
  minicpm_xml  MiniCPM5               <function name="NAME"><param name="P">V</param></function>   (V with <, & or a
               newline as <![CDATA[...]]>)
  lfm2_python  LFM2.5                 <|tool_call_start|>[NAME(P='text', N=3, O={"k": 1}), ...]<|tool_call_end|>
  fallback     templates without tools (Gemma 3, ...): the tools are described in
               the system prompt and calls use the hermes form

``detect(tokenizer)`` picks the format from the chat template. A format gives
the reply grammar (free text that cannot start a call, then calls whose
fixed parts are literal, so the engine writes them as forced spans and a reply
always parses), the parser back to OpenAI ``tool_calls``, where a call starts
(for streaming), and the message adjustments its template needs.
"""

from __future__ import annotations

import json
import re
import uuid
from functools import lru_cache
from typing import Any, List, Optional, Tuple

THINK_CLOSE = "</think>"


def _alts(names: List[str]) -> str:
    return "(" + "|".join(re.escape(n) for n in names) + ")" if names else r"[A-Za-z0-9_.\-]+"


_JSTR = r'"([^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'
_JNUM = r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?"


def _json_value(depth: int, sp: str) -> str:
    scalar = f"({_JSTR}|{_JNUM}|true|false|null)"
    if depth <= 0:
        return scalar
    inner = _json_value(depth - 1, sp)
    obj = r"\{" + sp + "(" + _JSTR + sp + ":" + sp + inner + "(" + sp + "," + sp + _JSTR + sp + ":" + sp + inner + ")*)?" + sp + r"\}"
    arr = r"\[" + sp + "(" + inner + "(" + sp + "," + sp + inner + ")*)?" + sp + r"\]"
    return "(" + scalar + "|" + obj + "|" + arr + ")"


@lru_cache(maxsize=4)
def json_args_regex(space: str = r"[ ]?") -> str:
    """Any JSON object four levels deep (an object of lists of objects of lists: Hermes Agent's
    clarify tool), with JSON's string escapes (\\uXXXX too) and numbers, and ``space`` allowed between
    JSON tokens (the templates write ``", "`` and ``": "``). About 700 automaton states."""
    inner = _json_value(3, space)
    return (r"\{" + space + "(" + _JSTR + space + ":" + space + inner + "(" + space + "," + space + _JSTR + space + ":"
            + space + inner + ")*)?" + space + r"\}")


def _json_object_at(text: str, i: int) -> Tuple[Optional[Any], int]:
    """The JSON value starting at ``text[i]`` and the index after it (None, i if none)."""
    try:
        v, end = json.JSONDecoder().raw_decode(text, i)
        return v, end
    except ValueError:
        return None, i


def _args_of(obj: dict, key: str) -> dict:
    a = obj.get(key)
    if a is None:
        a = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            a = {"value": a}
    return a if isinstance(a, dict) else {"value": a}


class ToolFormat:
    """Base: the hermes form (JSON inside <tool_call> tags)."""

    name = "hermes"
    opener = "<tool_call>"  # text that starts the calls of a reply; free text never contains it
    closer = "</tool_call>"  # text that ends one call ("" if calls end with the reply)
    multi = True  # several calls in one reply
    args_key = "arguments"
    native_tools = True  # the template renders the tools itself
    replay_raw = True  # a turn with calls may be replayed as the reply text it was parsed from
    think_open, think_close = "<think>", "</think>"  # how the model marks its reasoning
    think_sep = "\\n\\n"  # between the end of the reasoning and the reply (a regex)
    opens_thinking = False  # the model opens its reasoning block itself, thinking on or not (Gemma 4)

    # -- grammar -----------------------------------------------------------------------
    def free_text(self) -> str:
        from .oai import not_containing

        return not_containing(self.opener)

    def call(self, names: List[str]) -> str:
        return ("<tool_call>\\n\\{\"name\": \"" + _alts(names) + "\", \"" + self.args_key + "\": " + json_args_regex()
                + "\\}\\n</tool_call>")

    def calls(self, call: str, many: bool, at_least_one: bool) -> str:
        if not (self.multi and many):
            return call
        return call + "(\\n" + call + ")*"

    def reply_regex(self, names: List[str], tool_choice: Any = "auto", thinking: bool = False, preamble: bool = True,
                    parallel: bool = True) -> Optional[str]:
        """The grammar of a reply to a request offering tools ``names``
        (``preamble=False``: a reply that must call starts with the call;
        ``parallel=False``: at most one call)."""
        if not names:
            return None
        free = self.free_text()
        if tool_choice == "none":
            body = free
        else:
            chosen = names
            if isinstance(tool_choice, dict):
                fn = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
                chosen = [fn] if fn in names else names
            calls = self.calls(self.call(chosen), many=parallel and not isinstance(tool_choice, dict), at_least_one=True)
            if tool_choice == "required" or isinstance(tool_choice, dict):
                body = (free if preamble else "") + calls
            else:
                body = free + "(" + calls + ")?"
        if thinking:
            from .oai import not_containing

            body = not_containing(THINK_CLOSE) + "</think>" + self.think_sep + body
        return body

    # -- parsing -----------------------------------------------------------------------
    def split(self, text: str) -> Tuple[str, str]:
        """(free text, the calls part) of a reply after its thinking."""
        i = text.find(self.opener)
        return (text, "") if i < 0 else (text[:i], text[i:])

    def parse_calls(self, part: str) -> List[Tuple[str, dict]]:
        out = []
        pos = 0
        while True:
            i = part.find(self.opener, pos)
            if i < 0:
                break
            j = i + len(self.opener)
            while j < len(part) and part[j] in " \n":
                j += 1
            obj, end = _json_object_at(part, j)
            if isinstance(obj, dict) and obj.get("name"):
                out.append((str(obj["name"]), _args_of(obj, self.args_key)))
            pos = max(end, j)
        return out

    def count_complete(self, text: str) -> int:
        part = self.split(text)[1]  # a closing tag in the free text before the calls is not a call
        return part.count(self.closer) if self.closer else len(self.parse_calls(part))

    def cut_after(self, text: str, n: int) -> str:
        """``text`` up to the end of its ``n``-th complete call (all of it if fewer)."""
        if not self.closer:
            return text
        free, part = self.split(text)
        i = -1
        for _ in range(n):
            i = part.find(self.closer, i + 1)
            if i < 0:
                return text
        return free + part[: i + len(self.closer)]

    # -- messages ----------------------------------------------------------------------
    def new_id(self) -> str:
        return "call_" + uuid.uuid4().hex[:24]

    def prepare(self, msgs: List[dict], tools: List[dict]) -> Tuple[List[dict], Optional[List[dict]]]:
        """Messages and tools as this model's template takes them."""
        return msgs, (tools or None)


class QwenXML(ToolFormat):
    name = "qwen_xml"
    opener = "<tool_call>"
    closer = "</tool_call>"

    def free_text(self) -> str:
        from .oai import FREE_TEXT, not_containing

        return not_containing(*FREE_TEXT)

    def call(self, names: List[str]) -> str:
        from .oai import call_regex

        return call_regex(names)

    def parse_calls(self, part: str) -> List[Tuple[str, dict]]:
        from .oai import qwen_calls

        return [(name, args) for name, args, _ in qwen_calls(part)]

    def count_complete(self, text: str) -> int:
        from .oai import qwen_calls

        return len(qwen_calls(self.split(text)[1]))

    def cut_after(self, text: str, n: int) -> str:
        """``text`` up to the end of its ``n``-th complete call, by the parser (a "</tool_call>" inside a
        value is not the end of a call)."""
        from .oai import qwen_calls

        free, part = self.split(text)
        calls = qwen_calls(part)
        return text if len(calls) <= n else free + part[: calls[n - 1][2]]


class Hermes(ToolFormat):
    pass


def _py_value(node):
    """A literal of a Python-style call's argument (strings, numbers, True/False/None, and the JSON the
    template writes for lists and objects: true/false/null too)."""
    import ast

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in ("true", "false", "null", "True", "False", "None"):
        return {"true": True, "false": False, "null": None, "True": True, "False": False, "None": None}[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant):
        v = node.operand.value
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_py_value(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_py_value(k): _py_value(v) for k, v in zip(node.keys, node.values)}
    raise ValueError(f"not a literal: {ast.dump(node)[:80]}")


def _py_calls(inner: str) -> List[Tuple[int, int]]:
    """Spans of the complete calls in a Python-style call list's text after its "[" (the list may be
    unfinished: a reply still streaming), found by brackets outside of quoted strings."""
    spans, depth, start, i, quote = [], 0, None, 0, ""
    while i < len(inner):
        c = inner[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = ""
        elif c in "'\"":
            quote = c
        elif c in "([{":
            if depth == 0 and start is None:
                start = inner.rfind(",", 0, i) + 1  # the call's name starts after the last separator
            depth += 1
        elif c in ")]}":
            if depth == 0:  # the list's own "]"
                break
            depth -= 1
            if depth == 0 and c == ")" and start is not None:
                spans.append((len(inner[:start]) + len(inner[start:i]) - len(inner[start:i].lstrip()), i + 1))
                start = None
        i += 1
    return spans


class LFM2Python(ToolFormat):
    """LFM2.5 (Liquid): the calls of a reply as one Python-style list,
    ``<|tool_call_start|>[get_weather(city='Paris'), run(cmd='ls')]<|tool_call_end|>``; a string argument in
    single quotes with backslash escapes, a number or True/False/None as Python writes it, a list or an
    object as JSON. The prompt always opens a think block (its template has no switch): the model always
    thinks, and its reasoning ends with "</think>" right before the reply."""

    name = "lfm2_python"
    opener = "<|tool_call_start|>"
    closer = "<|tool_call_end|>"
    think_sep = ""

    def call(self, names: List[str]) -> str:
        from .oai import PARAM_RE_

        text = "'([^'\\\\\\n\\r]|\\\\[^\\n])*'"
        num = "-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][-+]?[0-9]+)?"
        arr = "\\[[ ]?(" + _json_value(3, "[ ]?") + "(,[ ]?" + _json_value(3, "[ ]?") + ")*)?[ ]?\\]"
        value = "(" + text + "|" + num + "|True|False|None|" + json_args_regex() + "|" + arr + ")"
        arg = PARAM_RE_ + "=" + value
        return _alts(names) + "\\((" + arg + "(, " + arg + ")*)?\\)"

    def calls(self, call: str, many: bool, at_least_one: bool) -> str:
        inner = call + ("(, " + call + ")*" if self.multi and many else "")
        return "<\\|tool_call_start\\|>\\[" + inner + "\\]<\\|tool_call_end\\|>"

    def _inner(self, part: str) -> str:
        i = part.find(self.opener)
        s = part[i + len(self.opener):] if i >= 0 else ""
        return s[1:] if s.startswith("[") else ""

    def parse_calls(self, part: str) -> List[Tuple[str, dict]]:
        import ast

        inner, out = self._inner(part), []
        for a, b in _py_calls(inner):
            try:
                node = ast.parse(inner[a:b].strip(), mode="eval").body
                if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
                    name = ast.unparse(node.func)
                    out.append((name, {k.arg: _py_value(k.value) for k in node.keywords if k.arg}))
            except (SyntaxError, ValueError):
                continue
        return out

    def count_complete(self, text: str) -> int:
        return len(_py_calls(self._inner(self.split(text)[1])))

    def cut_after(self, text: str, n: int) -> str:
        free, part = self.split(text)
        inner = self._inner(part)
        spans = _py_calls(inner)
        if len(spans) <= n:
            return text
        return free + self.opener + "[" + ", ".join(inner[a:b].strip() for a, b in spans[:n]) + "]" + self.closer


_MC_CALL = re.compile(r'<function name="([^"]+)">(.*?)</function>', re.S)
_MC_PARAM = re.compile(r'<param name="([^"]+)">(?:<!\[CDATA\[(.*?)\]\]>|(.*?))</param>', re.S)


class MiniCPMXML(ToolFormat):
    """MiniCPM5: ``<function name="NAME"><param name="P">V</param>...</function>``, a call per line; a value
    holding <, & or a newline in a CDATA block, as the template writes it. Values are text, typed by the
    tool's schema like Qwen's XML values. Without it the model got the fallback (hermes) form, never
    wrote a call in OpenCode, and described the commands it would run instead (0 of 13 tasks)."""

    name = "minicpm_xml"
    opener = '<function name="'
    closer = "</function>"

    def call(self, names: List[str]) -> str:
        from .oai import PARAM_RE_

        # CDATA holds any text without "]]>": a run of "]" never followed by ">" (not_containing wants a
        # literal whose first character does not come back)
        cdata = "([^\\]]|\\]([^\\]]|\\]+[^\\]>]))*\\]*"
        value = "(<!\\[CDATA\\[" + cdata + "\\]\\]>|[^<&\\n]*)"
        return '<function name="' + _alts(names) + '">(<param name="' + PARAM_RE_ + '">' + value + "</param>)*</function>"

    def parse_calls(self, part: str) -> List[Tuple[str, dict]]:
        out = []
        for m in _MC_CALL.finditer(part):
            out.append((m.group(1), {p.group(1): p.group(2) if p.group(2) is not None else p.group(3)
                                     for p in _MC_PARAM.finditer(m.group(2))}))
        return out


class Llama3JSON(ToolFormat):
    """Llama 3.x: a call is the whole reply, ``{"name": ..., "parameters": {...}}``;
    the template accepts one call per assistant turn."""

    name = "llama3_json"
    opener = "{"
    closer = ""
    multi = False
    args_key = "parameters"

    def free_text(self) -> str:
        # anything that does not start with "{" or with <|python_tag|> (which opens a call)
        return "([^{<](.|\\n)*|<([^|]|\\|[^p])(.|\\n)*|<|<\\|)?"

    def call(self, names: List[str]) -> str:
        return "(<\\|python_tag\\|>)?\\{\"name\": \"" + _alts(names) + "\", \"parameters\": " + json_args_regex() + "\\}"

    def reply_regex(self, names: List[str], tool_choice: Any = "auto", thinking: bool = False, preamble: bool = True,
                    parallel: bool = True) -> Optional[str]:
        if not names:
            return None
        if tool_choice == "none":
            return self.free_text()
        chosen = names
        if isinstance(tool_choice, dict):
            fn = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
            chosen = [fn] if fn in names else names
        call = self.call(chosen)
        if tool_choice == "required" or isinstance(tool_choice, dict):
            return call
        return "(" + call + "|" + self.free_text() + ")"

    def split(self, text: str) -> Tuple[str, str]:
        s = text.lstrip()
        if s.startswith("<|python_tag|>"):
            s = s[len("<|python_tag|>"):].lstrip()
        if s.startswith("{"):
            return "", s
        # Small Llamas sometimes announce the call first ("Here is the JSON for a function
        # call: {...}"): a call that parses after such a preamble is still a call.
        i = text.rfind('{"name"')
        if i > 0:
            obj, end = _json_object_at(text, i)
            if isinstance(obj, dict) and obj.get("name") and isinstance(obj.get("parameters", obj.get("arguments")), dict) \
                    and not text[end:].strip().strip("`").strip():
                return re.sub(r"`{3}\w*\s*$", "", text[:i].rstrip()).rstrip(), text[i:]
        return text, ""

    def parse_calls(self, part: str) -> List[Tuple[str, dict]]:
        obj, _ = _json_object_at(part, 0)
        if isinstance(obj, dict) and obj.get("name"):
            return [(str(obj["name"]), _args_of(obj, "parameters"))]
        return []

    def prepare(self, msgs, tools):
        out = []
        for m in msgs:
            if m["role"] == "tool":
                # the template writes a string result as a JSON-quoted string; a JSON result as JSON
                try:
                    v = json.loads(m.get("content") or "")
                    if isinstance(v, (dict, list)):
                        m = dict(m, content=v)
                except ValueError:
                    pass
            if m["role"] == "assistant" and len(m.get("tool_calls") or []) > 1:
                # the template raises on several calls in one turn: one turn per call
                for tc in m["tool_calls"]:
                    out.append({"role": "assistant", "content": "", "tool_calls": [tc]})
                continue
            if m["role"] == "assistant" and "tool_calls" in m and not m["tool_calls"]:
                m = {k: v for k, v in m.items() if k != "tool_calls"}  # any "tool_calls" key selects the call branch
            out.append(m)
        return out, (tools or None)


class MistralV3(ToolFormat):
    """Mistral 7B v0.3: ``[TOOL_CALLS] [{"name": ..., "arguments": {...}}, ...]``;
    tool-call ids are nine letters and digits (the template checks)."""

    name = "mistral"
    opener = "[TOOL_CALLS]"
    closer = ""
    # the templates count an assistant turn without tool_calls in their role
    # alternation: calls replayed as text would break a chain of calls
    replay_raw = False

    def call(self, names):
        return "\\{\"name\": \"" + _alts(names) + "\", \"arguments\": " + json_args_regex() + "\\}"

    def calls(self, call, many, at_least_one):
        inner = call + ("(, " + call + ")*" if self.multi and many else "")
        return "\\[TOOL_CALLS\\] \\[" + inner + "\\]"

    def free_text(self):
        # free text neither starts with "[" nor holds a call: the model's calls start the reply
        from .oai import not_containing

        return "([^\\[]" + not_containing(self.opener) + ")?"

    def reply_regex(self, names, tool_choice="auto", thinking=False, preamble=True, parallel=True):
        if not names:
            return None
        if tool_choice == "none":
            return self.free_text()
        chosen = names
        if isinstance(tool_choice, dict):
            fn = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
            chosen = [fn] if fn in names else names
        calls = self.calls(self.call(chosen), many=parallel and not isinstance(tool_choice, dict), at_least_one=True)
        if tool_choice == "required" or isinstance(tool_choice, dict):
            return calls
        return "(" + calls + "|" + self.free_text() + ")"

    def split(self, text):
        s = text.lstrip()
        return ("", s) if s.startswith(self.opener) else (text, "")

    def parse_calls(self, part):
        i = part.find("[", len(self.opener))
        obj, _ = _json_object_at(part, i) if i >= 0 else (None, 0)
        out = []
        for c in obj if isinstance(obj, list) else []:
            if isinstance(c, dict) and c.get("name"):
                out.append((str(c["name"]), _args_of(c, "arguments")))
        return out

    def new_id(self):
        return uuid.uuid4().hex[:9]

    legacy = False  # the older template: roles "tool_calls" / "tool_results", no system role

    def prepare(self, msgs, tools):
        msgs = _merge_users(msgs)
        if not self.legacy:
            return _mistral_ids(msgs), (tools or None)
        out: List[dict] = []
        system = ""
        for m in msgs:
            role, content = m["role"], m.get("content") or ""
            if role == "system":
                system += (("\n\n" if system else "") + content)
                continue
            if role == "user" and system and not any(x["role"] == "user" for x in out):
                content = system + "\n\n" + content  # the template drops system turns; the first user turn keeps the prefix stable
            if role == "assistant" and m.get("tool_calls"):
                if content.strip():
                    out.append({"role": "assistant", "content": content})
                calls = [{"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]} for tc in m["tool_calls"]]
                out.append({"role": "tool_calls", "content": " " + json.dumps(calls, ensure_ascii=False)})
                continue
            if role == "assistant" and content.lstrip().startswith(self.opener):  # the model's own reply, replayed verbatim
                out.append({"role": "tool_calls", "content": content.lstrip()[len(self.opener):]})
                continue
            if role == "tool":
                try:
                    body = json.loads(content)
                except ValueError:
                    body = content
                out.append({"role": "tool_results", "content": " " + json.dumps({"content": body}, ensure_ascii=False)})
                continue
            out.append({"role": role, "content": content})
        return out, (tools or None)


class MistralV11(MistralV3):
    """Ministral 3, Devstral, Mistral Small 3.2+: ``[TOOL_CALLS]NAME[ARGS]{...}`` per call,
    after the turn's text (the template writes an assistant turn as its content, then its calls)."""

    name = "mistral_v11"

    def free_text(self):
        from .oai import not_containing

        return not_containing(self.opener)

    def reply_regex(self, names, tool_choice="auto", thinking=False, preamble=True, parallel=True):
        return ToolFormat.reply_regex(self, names, tool_choice, thinking, preamble, parallel)

    def split(self, text):
        return ToolFormat.split(self, text)

    def call(self, names):
        return "\\[TOOL_CALLS\\]" + _alts(names) + "\\[ARGS\\]" + json_args_regex()

    def calls(self, call, many, at_least_one):
        return call + ("(" + call + ")*" if self.multi and many else "")

    def parse_calls(self, part):
        out = []
        for chunk in part.split(self.opener)[1:]:
            name, sep, rest = chunk.partition("[ARGS]")
            obj, _ = _json_object_at(rest, 0) if sep else (None, 0)
            if name.strip() and isinstance(obj, dict):  # arguments that are not a whole object: a cut call, not {}
                out.append((name.strip(), obj))
        return out

    def count_complete(self, text):
        return len(self.parse_calls(self.split(text)[1]))


def _merge_users(msgs: List[dict]) -> List[dict]:
    """Turns as Mistral templates take them: they raise on two user turns in a
    row (OpenCode's title request sends two), on two assistant turns in a row
    and on an assistant turn with neither text nor calls (smolagents sends one
    before its "Calling tools:" turn). Such turns are merged or dropped."""
    out: List[dict] = []
    for m in msgs:
        text = m.get("content")
        if m.get("role") == "assistant" and not m.get("tool_calls") and not (text or "").strip():
            continue
        prev = out[-1] if out else None
        if prev is not None and m.get("role") in ("user", "assistant") and prev.get("role") == m["role"] \
                and not m.get("tool_calls") and not prev.get("tool_calls") \
                and isinstance(text, str) and isinstance(prev.get("content"), str):
            out[-1] = dict(prev, content=prev["content"] + "\n\n" + text)
        else:
            out.append(m)
    return out


def _mistral_ids(msgs: List[dict]) -> List[dict]:
    """Tool-call ids as Mistral templates require (nine letters and digits), and an
    assistant turn between tool results and a following user turn where the
    templates' alternation check needs one (they count user turns and assistant
    turns without tool_calls, and raise "roles must alternate"; agents such as
    OpenCode send user turns after tool results). A call replayed as the model's
    own text is an assistant turn without tool_calls, so it already counts."""
    def nine(i: str) -> str:
        s = re.sub(r"[^A-Za-z0-9]", "", i or "")
        return (s[-9:] if len(s) >= 9 else (s + "0" * 9)[:9]) or "000000000"

    out = []
    counted = None  # the role of the last turn the alternation check counts
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [dict(tc, id=nine(tc.get("id"))) for tc in m["tool_calls"]]
        if m.get("role") == "tool":
            m["tool_call_id"] = nine(m.get("tool_call_id"))
        if m.get("role") == "user" and out and out[-1].get("role") == "tool" and counted == "user":
            out.append({"role": "assistant", "content": "Done."})
        if m.get("role") == "user" or (m.get("role") == "assistant" and not m.get("tool_calls")):
            counted = m["role"]
        out.append(m)
    return out


FALLBACK_INSTRUCTIONS = """# Tools

You may call functions to assist with the user query. You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

To call a function, end your reply with a json object with the function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
Call one function per reply. Its result comes back to you inside <tool_response></tool_response> tags; read it before you decide the next step."""


class Fallback(Hermes):
    """For templates that do not render tools: the tools go into the system
    prompt in the hermes form, earlier calls are written as the model wrote
    them and results come back as user turns; consecutive turns of one role
    are merged (templates such as Gemma's require alternation)."""

    name = "fallback"
    native_tools = False
    multi = False  # a model without tool training chains dependent calls blindly when it may write several
    # A model without tool training often writes its call as a JSON block at the end
    # of the reply instead (Gemma 3: ```tool_call {...}```, ```json {"tool_call": {...}}```):
    # such a call counts when it names an offered tool (oai.parse_reply checks the name).
    loose_calls = True
    _FENCE = re.compile(r"```[\w-]*[ \t]*\n?(\{.*\})\s*```\s*$", re.S)

    @staticmethod
    def _loose(obj) -> Optional[Tuple[str, dict]]:
        """(name, arguments) of a call-like JSON object, or None."""
        if not isinstance(obj, dict):
            return None
        for wrap in ("tool_call", "function_call", "call"):
            if isinstance(obj.get(wrap), dict) and len(obj) == 1:
                obj = obj[wrap]
        name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
        if not isinstance(name, str) and isinstance(obj.get("function"), dict):
            obj = obj["function"]
            name = obj.get("name")
        if not isinstance(name, str) or not name:
            return None
        args = next((obj[k] for k in ("arguments", "parameters", "args", "input") if k in obj), {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return None
        return (name, args) if isinstance(args, dict) else None

    def split(self, text: str) -> Tuple[str, str]:
        free, part = super().split(text)
        if part:
            return free, part
        m = self._FENCE.search(text)
        if m is not None:  # a fenced JSON call ends the reply
            obj, end = _json_object_at(text, m.start(1))
            if end == m.end(1) and self._loose(obj) is not None:
                return text[:m.start()].rstrip(), text[m.start():]
        s = text.lstrip()
        if s.startswith("{"):  # the whole reply is a JSON call
            obj, end = _json_object_at(s, 0)
            if not s[end:].strip() and self._loose(obj) is not None:
                return "", s
        return text, ""

    def parse_calls(self, part: str) -> List[Tuple[str, dict]]:
        if part.startswith(self.opener):
            return super().parse_calls(part)
        i = part.find("{")
        obj, _ = _json_object_at(part, i) if i >= 0 else (None, 0)
        call = self._loose(obj)
        return [call] if call else []

    def prepare(self, msgs, tools):
        out: List[dict] = []
        if tools:
            spec = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
            text = FALLBACK_INSTRUCTIONS.format(tools=spec)
            if msgs and msgs[0]["role"] == "system":
                msgs = [{"role": "system", "content": msgs[0]["content"] + "\n\n" + text}] + msgs[1:]
            else:
                msgs = [{"role": "system", "content": text}] + list(msgs)
        for m in msgs:
            role, content = m["role"], m.get("content") or ""
            if role == "assistant" and m.get("tool_calls"):
                calls = "\n".join("<tool_call>\n" + json.dumps({"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
                                                              ensure_ascii=False) + "\n</tool_call>" for tc in m["tool_calls"])
                content = (content + "\n" if content else "") + calls
                m = {"role": "assistant", "content": content}
            elif role == "tool":
                m = {"role": "user", "content": "<tool_response>\n" + content + "\n</tool_response>"}
            else:
                m = {"role": role, "content": content}
            if out and out[-1]["role"] == m["role"] and m["role"] != "system":
                out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n" + m["content"]}
            else:
                out.append(m)
        return out, None


Q = '<|"|>'  # Gemma 4's string quote token


def _gemma_value(depth: int = 2) -> str:
    """A Gemma 4 argument value: a <|"|>-quoted string, a number, a boolean, or
    (``depth`` levels) a {key:value,...} object or a [value,...] list."""
    from .oai import not_containing

    s = re.escape(Q) + not_containing(Q) + re.escape(Q)
    scalar = "(" + s + r"|-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?|true|false|null)"  # 007 is not a number
    if depth <= 0:
        return scalar
    inner = _gemma_value(depth - 1)
    obj = r"\{(" + _GKEY + ":" + inner + "(," + _GKEY + ":" + inner + r")*)?\}"
    arr = r"\[(" + inner + "(," + inner + r")*)?\]"
    return "(" + scalar + "|" + obj + "|" + arr + ")"


_GKEY = r"[A-Za-z_][A-Za-z0-9_.\-]*"


def _gemma_to_json(text: str):
    """Gemma 4's argument syntax -> a Python value (strings first, then bare keys quoted)."""
    out, i = [], 0
    while i < len(text):
        j = text.find(Q, i)
        if j < 0:
            out.append(re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_.\-]*)\s*:", r'\1"\2":', text[i:]))
            break
        out.append(re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_.\-]*)\s*:", r'\1"\2":', text[i:j]))
        k = text.find(Q, j + len(Q))
        if k < 0:
            k = len(text)
        out.append(json.dumps(text[j + len(Q):k], ensure_ascii=False))
        i = k + len(Q)
    return json.loads("".join(out))


class Gemma4(ToolFormat):
    """Gemma 4: <|tool_call>call:NAME{key:value,...}<tool_call|>, strings between <|"|> tokens."""

    name = "gemma4"
    opener = "<|tool_call>"
    closer = "<tool_call|>"
    replay_raw = False  # the template writes tool results only after an assistant turn's tool_calls
    bare_start = "call:"  # the model sometimes leaves out <|tool_call> before a call
    # Gemma 4 reasons in a thought channel, <|channel>thought ... <channel|>, which it may open
    # even with thinking off: the reply may start with one, and it is reasoning, not content
    think_open, think_close = "<|channel>thought", "<channel|>"
    opens_thinking = True

    def thought_regex(self) -> str:
        """The thought block a reply may open with (the server leaves it out of a reply it has begun)."""
        from .oai import not_containing

        return "(<\\|channel>thought" + not_containing(self.think_close) + "<channel\\|>\\n*)?"

    def reply_regex(self, names, tool_choice="auto", thinking=False, preamble=True, parallel=True):
        body = super().reply_regex(names, tool_choice, False, preamble, parallel)
        if body is None:
            return None
        return self.thought_regex() + body

    def split(self, text: str) -> Tuple[str, str]:
        free, part = super().split(text)
        s = text.lstrip()
        if not part and s.startswith(self.bare_start) and self.closer in s:  # a whole reply that is a call without its opener
            return "", s
        return free, part

    def free_text(self) -> str:
        from .oai import not_containing

        # a thought block only opens the reply (see reply_regex); its markers are never content
        return not_containing(self.opener, "<|channel>", self.think_close)

    def prepare(self, msgs, tools):
        # The template writes an assistant turn as its calls, their results, then its text,
        # and closes the model's turn after that text without opening another: text written
        # before calls (a preamble, or the server's steering) ended the turn mid-task, and
        # the next reply was empty. Such text goes in the thought channel, which the
        # template writes before the calls in the current exchange (and drops for earlier
        # exchanges, as it does the model's thoughts).
        out = []
        for m in msgs:
            if m["role"] == "assistant" and m.get("tool_calls") and (m.get("content") or "").strip():
                thought = "\n\n".join(t for t in ((m.get("reasoning_content") or "").strip(), m["content"].strip()) if t)
                m = dict(m, content="", reasoning_content=thought)
            out.append(m)
        return out, (tools or None)

    def call(self, names: List[str]) -> str:
        v = _gemma_value(3)  # objects of lists of objects (Hermes Agent's clarify tool)
        return ("<\\|tool_call>call:" + _alts(names) + "\\{(" + _GKEY + ":" + v + "(," + _GKEY + ":" + v + ")*)?\\}<tool_call\\|>")

    def calls(self, call, many, at_least_one):
        return call + ("(" + call + ")*" if self.multi and many else "")

    def parse_calls(self, part: str) -> List[Tuple[str, dict]]:
        out = []
        if part.startswith(self.bare_start):
            part = self.opener + part
        for chunk in part.split(self.opener)[1:]:
            if self.closer not in chunk:  # cut before <tool_call|>: not a call
                continue
            body = chunk.split(self.closer)[0]
            if not body.startswith("call:") or "{" not in body:
                continue
            name, _, args = body[len("call:"):].partition("{")
            try:
                val = _gemma_to_json("{" + args)
            except ValueError:  # arguments that do not parse: not a call (they were sent as {})
                continue
            out.append((name.strip(), val if isinstance(val, dict) else {"value": val}))
        return out


class MistralLegacy(MistralV3):
    name = "mistral_legacy"
    legacy = True
    replay_raw = True  # prepare turns a replayed call into the template's own tool_calls turn


FORMATS = {f.name: f for f in (QwenXML(), Hermes(), Llama3JSON(), MistralV3(), MistralLegacy(), MistralV11(), Gemma4(), MiniCPMXML(),
                                LFM2Python(), Fallback())}


def detect_from_template(template: Optional[str]) -> ToolFormat:
    t = template or ""
    if "<function=" in t and "<parameter=" in t:
        return FORMATS["qwen_xml"]
    if '<function name="' in t and '<param name="' in t:
        return FORMATS["minicpm_xml"]
    if "<|tool_call_start|>" in t:
        return FORMATS["lfm2_python"]
    if "[TOOL_CALLS]" in t:
        if "[ARGS]" in t:
            return FORMATS["mistral_v11"]
        return FORMATS["mistral_legacy"] if "tool_results" in t else FORMATS["mistral"]
    if "<|tool_call>call:" in t:
        return FORMATS["gemma4"]
    if "<tool_call>" in t:
        return FORMATS["hermes"]
    if "\"parameters\"" in t and ("ipython" in t or "<|python_tag|>" in t):
        return FORMATS["llama3_json"]
    return FORMATS["fallback"]


def chat_template(tokenizer) -> str:
    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    t = getattr(hf, "chat_template", None)
    if isinstance(t, dict):  # named templates: the default one, or the tool-use one when present
        t = t.get("tool_use") or t.get("default") or next(iter(t.values()), "")
    return t or ""


_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def reasoning_efforts(template: str) -> Optional[List[str]]:
    """The ``reasoning_effort`` values a chat template accepts, in its own words (Qwen3.8: xhigh,
    medium, low; it raises on any other), [] if it has the switch but no list, None without one."""
    if "reasoning_effort" not in (template or ""):
        return None
    m = re.search(r"reasoning_effort\w*\s+not\s+in\s+[(\[]([^)\]]*)[)\]]", template)
    return [a or b for a, b in re.findall(r"'([^']*)'|\"([^\"]*)\"", m.group(1))] if m else []


def effort_for(asked: str, allowed: List[str]) -> str:
    """The value of ``allowed`` nearest to ``asked`` (OpenAI's none ... xhigh scale; a tie goes up)."""
    a = asked.strip().lower()
    if not allowed or a in allowed:
        return a if a in allowed or not allowed else allowed[0]
    known = [x for x in allowed if x in _EFFORTS]
    if a not in _EFFORTS or not known:
        return allowed[0]
    i = _EFFORTS.index(a)
    return min(known, key=lambda x: (abs(_EFFORTS.index(x) - i), -_EFFORTS.index(x)))


def detect(tokenizer) -> ToolFormat:
    return detect_from_template(chat_template(tokenizer))


def offers_thinking(tokenizer) -> bool:
    """Whether requests can turn thinking on: the generation prompt opens a think
    block (Qwen3.5), or the template has a switch and the model writes the block
    (Qwen3). Qwen3-4B-Instruct-2507's template mentions <think> but has no switch."""
    style = thinking_style(tokenizer)
    return style in ("open", "channel", "always") or (style == "model" and "enable_thinking" in chat_template(tokenizer))


def thinking_style(tokenizer) -> str:
    """'open' if the generation prompt opens a <think> block when thinking is
    on (Qwen3.5), 'model' if the model writes <think> itself (Qwen3),
    'channel' for Gemma 4 (a switch, and thoughts in <|channel>thought ...
    <channel|>), 'switch' if the template has an enable_thinking switch but
    marks thoughts some other way (the server keeps it off), 'none' if the
    template has no thinking switch, 'always' if the generation prompt opens
    a <think> block with thinking off too (LFM2.5: the model always thinks)."""
    t = chat_template(tokenizer)
    if "<|channel>" in t and "enable_thinking" in t:
        return "channel"  # Gemma 4: the switch adds <|think|> to the system turn; thoughts go in a channel
    if "enable_thinking" not in t and "<think>" not in t:
        return "none"
    if "<think>" not in t:
        return "switch"
    try:
        s = tokenizer.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True, enable_thinking=True,
                                          tokenize=False)
    except Exception:
        return "none"
    if not s.rstrip().endswith("<think>"):
        return "model"
    try:
        off = tokenizer.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True, enable_thinking=False,
                                            tokenize=False)
    except Exception:
        return "open"
    return "always" if off.rstrip().endswith("<think>") else "open"
