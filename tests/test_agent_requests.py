"""Agents' real requests through the server's prompt path, for every cached model.

tests/data/opencode_requests.json holds requests OpenCode 1.18 sent (its title
request, and agent conversations: a call, text and a call, two calls in one
reply, a failed call, a chain of three, a follow-up turn, a file write);
tests/data/agent_requests.json those of OpenClaw (two user turns to start, a
context message after every tool result) and Hermes Agent (24 tools, a
JSON-schema title request); tests/data/framework_requests.json those of
pydantic-ai, smolagents (calls replayed as text, stop sequences), the OpenAI
Agents SDK, LlamaIndex, DSPy, LangChain, LiteLLM and Instructor. Each goes through Server.prepare as a request
would (the model's own call text replayed where the server would have it),
with the model's chat template: the template must accept it, and every user
message and tool result must be in the prompt, in order. No GPU: the service
is a stand-in holding the tokenizer.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("mlx_lm")

from locals import formats, oai  # noqa: E402

DATA = json.loads((Path(__file__).parent / "data" / "opencode_requests.json").read_text())
OTHERS = json.loads((Path(__file__).parent / "data" / "agent_requests.json").read_text())
FRAMEWORKS = json.loads((Path(__file__).parent / "data" / "framework_requests.json").read_text())
MODELS = [
    "mlx-community/Qwen3.5-2B-MLX-4bit", "mlx-community/Qwen3.5-4B-MLX-4bit", "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen2.5-7B-Instruct-4bit", "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit", "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Ministral-3-3B-Instruct-2512-4bit", "mlx-community/gemma-3-4b-it-qat-4bit", "mlx-community/gemma-4-e4b-it-4bit",
]


# what ends an assistant turn in the templates above
TURN_ENDS = ("<turn|>", "<end_of_turn>", "<|im_end|>", "<|eot_id|>", "</s>")


def _gemma(v):
    if isinstance(v, str):
        return formats.Q + v + formats.Q
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{_gemma(x)}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        return "[" + ",".join(_gemma(x) for x in v) + "]"
    return json.dumps(v)


def model_reply(fmt, content, calls):
    """The reply text a model of this format writes for these calls (what the server would have generated)."""
    pairs = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]
    lead = content or ""
    if fmt.name == "qwen_xml":
        body = "\n".join("<tool_call>\n<function=" + n + ">\n" + "".join(
            f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n" for k, v in a.items())
            + "</function>\n</tool_call>" for n, a in pairs)
        return (lead + "\n\n" if lead else "") + body
    if fmt.name in ("hermes", "fallback"):
        return (lead + "\n" if lead else "") + "\n".join(
            "<tool_call>\n" + json.dumps({"name": n, "arguments": a}) + "\n</tool_call>" for n, a in pairs)
    if fmt.name == "llama3_json":
        return json.dumps({"name": pairs[0][0], "parameters": pairs[0][1]})
    if fmt.name in ("mistral", "mistral_legacy"):
        return "[TOOL_CALLS] " + json.dumps([{"name": n, "arguments": a} for n, a in pairs])
    if fmt.name == "mistral_v11":
        return lead + "".join(f"[TOOL_CALLS]{n}[ARGS]{json.dumps(a)}" for n, a in pairs)
    if fmt.name == "gemma4":
        return lead + "".join("<|tool_call>call:" + n + _gemma(a) + "<tool_call|>" for n, a in pairs)
    raise AssertionError(fmt.name)


class _Svc:
    def __init__(self, tok):
        self.tokenizer = tok


@pytest.fixture(scope="module", params=MODELS)
def server(request):
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(request.param, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    from mlx_lm.utils import load_tokenizer

    from locals.server import Server

    srv = Server(request.param, engines=("locals",), preload=(), log=lambda s: None)
    srv._svc["locals"] = _Svc(load_tokenizer(request.param))
    return srv


def _appears(s, text, after=0):
    """Where ``s`` (or its JSON-escaped or re-serialized form, as templates write tool results) is in ``text`` after ``after``; -1 if not."""
    forms = [s.strip()]
    try:  # a JSON result, as templates that take it as an object write it again
        obj = json.loads(s)
        forms += [json.dumps(obj, ensure_ascii=False), json.dumps(obj, ensure_ascii=False, separators=(",", ":"))]
    except ValueError:
        pass
    for form in [f[:60] for f in forms] + [json.dumps(f[:60])[1:-1] for f in forms]:
        i = text.find(form, after)
        if i >= 0:
            return i
    return -1


def _conversations():
    """name -> (messages, tools, request options) for every recorded agent conversation."""
    system = {"role": "system", "content": DATA["system"]}
    out = {f"opencode/{name}": ([system] + msgs, DATA["tools"], DATA["request_options"]) for name, msgs in DATA["conversations"].items()}
    out["opencode/user_after_tool"] = ([system] + DATA["conversations"]["basic"] + [{"role": "user", "content": "Thanks. Now count the files too."}],
                                       DATA["tools"], DATA["request_options"])
    for agent in ("openclaw", "hermes"):
        a = OTHERS[agent]
        for name, msgs in a["conversations"].items():
            out[f"{agent}/{name}"] = (msgs, a["tools"], a["request_options"])
    for name, c in FRAMEWORKS["conversations"].items():  # pydantic-ai, smolagents, LlamaIndex, DSPy, ...
        out[f"framework/{name}"] = (c["messages"], c["tools"], c["options"])
    return out


@pytest.mark.parametrize("name", sorted(_conversations()))
def test_agent_conversation(server, name):
    msgs, tools, options = _conversations()[name]
    fmt, _ = server.format_of("locals")
    server._raw.clear()
    if fmt.replay_raw:  # the server remembers each reply with calls it generated
        for m in msgs:
            if m["role"] == "assistant" and m.get("tool_calls") and (fmt.multi or len(m["tool_calls"]) == 1):
                text = model_reply(fmt, m.get("content"), m["tool_calls"])
                server._raw[oai.id_key(m["tool_calls"][0]["id"])] = text
    req = dict(options, model=f"locals/{server.short}", messages=msgs)
    if tools:
        req["tools"] = tools
    p = server.prepare(req)
    text = p["text"]
    at = 0
    for m in msgs[1:]:
        content = oai.text_of(m.get("content"))
        if m["role"] in ("user", "tool") and content.strip():
            i = _appears(content, text, at)
            assert i >= 0, (name, m["role"], content[:80], text[-1500:])
            at = i
    assert len(p["prompt"]) < server.max_context
    # the model's turn is open: a prompt that ends its turn gets an empty reply (Gemma 4's
    # template closed the turn after an assistant message with text and calls)
    if msgs[-1]["role"] != "assistant":
        assert not text.rstrip().endswith(TURN_ENDS), (name, text[-300:])


def test_hermes_title_request(server):
    """Hermes names a session with a JSON-schema response and no tools."""
    t = OTHERS["hermes"]["title_request"]
    p = server.prepare(dict(t, model=f"locals/{server.short}", max_tokens=64))
    assert p["schema"] is not None and _appears(oai.text_of(t["messages"][-1]["content"]), p["text"]) >= 0


def test_title_request(server):
    """OpenCode's title request: two user messages in a row, no tools."""
    msgs = DATA["title_request"]
    p = server.prepare({"model": f"locals/{server.short}", "messages": msgs, "max_tokens": 64, "stream": True})
    for m in msgs[1:]:
        assert _appears(m["content"], p["text"]) >= 0


def test_regex_response_format(server):
    """response_format {"type": "regex"} (and vLLM's guided_regex) constrain the reply to a regular expression."""
    for extra in ({"response_format": {"type": "regex", "regex": "FILE a\\.py\\n```[a-z]*\\n[^`]*```"}},
                  {"guided_regex": "(yes|no)"}):
        p = server.prepare(dict({"model": f"locals/{server.short}", "max_tokens": 64,
                                 "messages": [{"role": "user", "content": "Write a.py"}]}, **extra))
        assert p["regex"] == (extra.get("guided_regex") or extra["response_format"]["regex"]) and p["schema"] is None


def test_unknown_response_format_is_an_error(server):
    """A response_format the server does not know is a 400, not an unconstrained reply (a server that
    predated the regex type answered such requests freely, and the client found no files in the text)."""
    import pytest

    with pytest.raises(ValueError, match="response_format type 'grammar'"):
        server.prepare({"model": f"locals/{server.short}", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}],
                        "response_format": {"type": "grammar", "grammar": "x"}})


def test_tools_and_a_response_format_together(server):
    """An agent with tools and an output type (the OpenAI Agents SDK): the reply is the calls or the output, as
    OpenAI's is. With the format's grammar alone the model could not call: asked for the weather in Paris, it wrote
    a forecast as JSON."""
    from locals.regex import compile_dfa

    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}
    base = {"model": f"locals/{server.short}", "max_tokens": 64, "tools": tools,
            "messages": [{"role": "user", "content": "The weather in Paris?"}]}
    for rf in ({"type": "json_schema", "json_schema": {"name": "R", "schema": schema}}, {"type": "json_object"}):
        p = server.prepare(dict(base, response_format=rf))
        assert p["schema"] is None
        d = compile_dfa(p["regex"])
        assert d.matches(b'{"summary":"rain"}') and not d.matches(b"It rains.")
        if p["fmt"].name == "qwen_xml":
            call = model_reply(p["fmt"], "", [{"function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}])
            assert d.matches(call.encode())
        calls_only = server.prepare(dict(base, response_format=rf, tool_choice="required"))
        assert not compile_dfa(calls_only["regex"]).matches(b'{"summary":"rain"}')
    none = server.prepare(dict(base, response_format={"type": "json_schema", "json_schema": {"name": "R", "schema": schema}},
                               tool_choice="none"))
    assert none["schema"] == schema and none["regex"] is None  # (the output only)


def test_a_request_for_another_model_by_our_names_is_refused(server):
    """Another locals command switched the server's model under a connected agent (`locals chat --model gemma`
    while OpenCode ran on Qwen), and the agent's requests were answered by the new model without a word. Our own
    names for another model are refused with a sentence; any other name is the model loaded, as before."""
    import pytest

    base = {"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    for ok in (f"locals/{server.short}", f"locals/{server.short}-thinking", "gpt-4o", "locals", "",
               f"locals/locals/{server.short}"):
        server.prepare(dict(base, model=ok))
    other = "gemma-4-e4b" if "gemma" not in server.short else "qwen3.5-4b"
    for bad in (f"locals/{other}", f"locals/locals/{other}"):
        with pytest.raises(ValueError, match=f"runs {server.short}, not {other}"):
            server.prepare(dict(base, model=bad))
        with pytest.raises(ValueError, match="another locals command switched"):
            server.chat_model(bad)
    assert server.chat_model("llama3.2:latest").endswith(server.short)  # (Ollama's names: the model loaded)

