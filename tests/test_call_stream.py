"""Tool calls streamed while the model writes them (oai.CallStream, Qwen3.5's XML calls): the pieces sent for a
call add up to exactly the arguments the finished reply's parse gives, a call that may be a repeat waits, and
through the server a long write call reaches the client in many pieces before it is complete."""

import json

import pytest

from locals import oai

TOOLS = [{"type": "function", "function": {"name": "write", "description": "Write a file.", "parameters": {
    "type": "object", "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a command.", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"},
                                        "background": {"type": "boolean"}, "env": {"type": "object"}}}}}]


def _call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def _run(reply, prev=(), step=1, tools=TOOLS):
    """Feed ``reply`` ``step`` characters at a time: (deltas, final calls, the arguments each index added up to)."""
    cs = oai.CallStream(tools, prev)
    deltas = []
    start = reply.find("<tool_call>")
    for n in range(1, len(reply) + 1, step):
        if start >= 0 and n > start:
            deltas += cs.feed(reply[start:n])
    deltas += cs.feed(reply[start:]) if start >= 0 else []
    _, _, calls = oai.parse_reply(reply, tools)
    deltas += cs.finish(calls)
    got, names, ids = {}, {}, {}
    for d in deltas:
        got[d["index"]] = got.get(d["index"], "") + d["function"].get("arguments", "")
        if d.get("id"):
            ids.setdefault(d["index"], d["id"])
            names[d["index"]] = d["function"]["name"]
    return deltas, calls, got, names, ids


CONTENT = 'def f(x):\n    """Say "hi" \\ there."""\n    return x  # über 🎉\n\n# </param ends here?\n'


@pytest.mark.parametrize("step", [1, 3, 17])
def test_pieces_add_up_to_the_final_arguments(step):
    reply = "I will write it.\n\n" + _call("write", filePath="src/a.py", content=CONTENT) + "\n" + \
        _call("bash", command="python3 -m pytest -q", timeout="120", background="false", env='{"A": "1"}')
    deltas, calls, got, names, ids = _run(reply, step=step)
    assert len(calls) == 2
    for i, c in enumerate(calls):
        assert got[i] == c["function"]["arguments"]  # the client's sum of pieces is the call
        assert names[i] == c["function"]["name"] and ids[i] == c["id"]
    assert json.loads(got[1]) == {"command": "python3 -m pytest -q", "timeout": 120, "background": False, "env": {"A": "1"}}
    first = [d for d in deltas if d["index"] == 0 and d["function"].get("arguments")]
    assert len(first) > 10  # the file went out while it was written


def test_a_call_that_may_repeat_the_previous_turns_waits():
    prev = [("bash", json.dumps({"command": "ls -la"}, sort_keys=True))]
    deltas, calls, got, _, _ = _run(_call("bash", command="ls -la"), prev=prev)
    assert [d for d in deltas if "id" in d][0]["function"]["arguments"] == "{\"command\": \"ls -la\"}"  # sent whole, at the end
    assert len(deltas) == 1
    deltas, calls, got, _, _ = _run(_call("bash", command="ls -l src"), prev=prev)
    assert got[0] == calls[0]["function"]["arguments"] and len(deltas) > 2  # differs: sent as it was written


def test_a_second_identical_call_in_the_reply_is_never_sent():
    one = _call("bash", command="git status")
    deltas, calls, got, _, _ = _run(one + "\n" + one)
    assert len(calls) == 2  # (the guard drops the duplicate; the parse alone keeps both)
    assert {d["index"] for d in deltas if d.get("id")} >= {0}
    sent_before_end = oai.CallStream(TOOLS)
    for n in range(1, len(one + "\n" + one) + 1):
        sent_before_end.feed((one + "\n" + one)[:n])
    assert sent_before_end.streamed() == 1  # the second one waited: it might be (and was) the same call


def test_calls_to_tools_not_offered_and_repeated_parameters():
    reply = _call("delete_everything", path="/") + "\n" + \
        "<tool_call>\n<function=write>\n<parameter=filePath>\na.txt\n</parameter>\n<parameter=filePath>\nb.txt\n</parameter>\n" \
        "<parameter=content>\nx\n</parameter>\n</function>\n</tool_call>"
    deltas, calls, got, names, _ = _run(reply)
    assert [c["function"]["name"] for c in calls] == ["write"] and names == {0: "write"}
    assert json.loads(got[0]) == {"filePath": "a.txt", "content": "x"} and got[0] == calls[0]["function"]["arguments"]


def test_a_value_ending_in_a_newline_or_a_partial_close():
    for content in ("line\n\n", "a</parame", "x\n</par", ""):
        reply = _call("write", filePath="f", content=content)
        _, calls, got, _, _ = _run(reply)
        assert got[0] == calls[0]["function"]["arguments"], content


# --- through the server ------------------------------------------------------------------------

def test_a_streamed_reply_sends_its_call_in_pieces(monkeypatch):
    """The OpenAI SDK accumulates the pieces into the call; they arrive before the call is complete."""
    openai = pytest.importorskip("openai")
    import test_messages_api as m

    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(m.MODEL, local_files_only=True, allow_patterns=["*.json", "*.jinja", "tokenizer*"])
    except Exception:  # noqa: BLE001
        pytest.skip("tokenizer not cached")
    from mlx_lm.utils import load_tokenizer

    httpd, svc = m._serve(load_tokenizer(path), "Writing it.\n\n" + _call("write", filePath="notes.md", content=CONTENT * 4))
    try:
        client = openai.OpenAI(base_url=f"http://127.0.0.1:{httpd.server_address[1]}/v1", api_key="x", max_retries=0)
        chunks = list(client.chat.completions.create(model="locals/qwen3.5-4b", stream=True, tools=TOOLS,
                                                     messages=[{"role": "user", "content": "Write notes.md."}]))
        pieces = [tc for c in chunks if c.choices for tc in (c.choices[0].delta.tool_calls or [])]
        args = "".join(tc.function.arguments or "" for tc in pieces)
        assert json.loads(args) == {"filePath": "notes.md", "content": CONTENT * 4}
        assert len(pieces) > 20 and pieces[0].id and pieces[0].function.name == "write"
        assert sum(1 for tc in pieces if tc.id) == 1  # one call: its id once
        finish = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
        assert finish == ["tool_calls"]
    finally:
        httpd.shutdown()


def test_the_anthropic_sdk_gets_the_call_in_pieces():
    """The same through the Messages API (Claude Code): many input_json_delta events, the SDK's input is the call."""
    anthropic = pytest.importorskip("anthropic")
    import test_messages_api as m

    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(m.MODEL, local_files_only=True, allow_patterns=["*.json", "*.jinja", "tokenizer*"])
    except Exception:  # noqa: BLE001
        pytest.skip("tokenizer not cached")
    from mlx_lm.utils import load_tokenizer

    httpd, svc = m._serve(load_tokenizer(path), "Writing it.\n\n" + _call("write", filePath="notes.md", content=CONTENT * 4))
    try:
        client = anthropic.Anthropic(base_url=f"http://127.0.0.1:{httpd.server_address[1]}", api_key="x", max_retries=0)
        tools = [{"name": t["function"]["name"], "description": t["function"]["description"],
                  "input_schema": t["function"]["parameters"]} for t in TOOLS]
        with client.messages.stream(model="claude-anything", max_tokens=2000, tools=tools,
                                    messages=[{"role": "user", "content": "Write notes.md."}]) as stream:
            pieces = [ev.partial_json for ev in stream if ev.type == "input_json"]
            final = stream.get_final_message()
        use = [b for b in final.content if b.type == "tool_use"]
        assert len(use) == 1 and use[0].name == "write"
        assert use[0].input == {"filePath": "notes.md", "content": CONTENT * 4}
        assert len(pieces) > 20 and final.stop_reason == "tool_use"
    finally:
        httpd.shutdown()


# FastMCP and Pydantic write a nested model as a $ref and Optional[...] as an anyOf; some clients send a property
# schema that is no object (true, or the shorthand "string")
NESTED = [{"type": "function", "function": {"name": "search", "parameters": {
    "type": "object", "$defs": {"P": {"type": "object", "properties": {"q": {"type": "string"}, "n": {"type": "integer"}}}},
    "properties": {"params": {"$ref": "#/$defs/P"}, "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                   "note": {"anyOf": [{"type": "string"}, {"type": "null"}]}, "flag": True, "kind": "string",
                   "mode": {"enum": ["fast", "slow"]}, "size": {"const": 3}}}}}]


@pytest.mark.parametrize("step", [1, 5])
def test_nested_and_optional_parameters_arrive_typed(step):
    reply = _call("search", params='{"q": "x", "n": 3}', limit="5", note="null", flag="yes", kind="file", mode="fast",
                  size="3")
    _, calls, got, _, _ = _run(reply, step=step, tools=NESTED)
    want = {"params": {"q": "x", "n": 3}, "limit": 5, "note": "null", "flag": "yes", "kind": "file", "mode": "fast",
            "size": 3}
    assert json.loads(calls[0]["function"]["arguments"]) == want  # (params came as the text '{"q": ...}'; flag crashed)
    assert got[0] == calls[0]["function"]["arguments"]


def test_other_formats_read_nested_parameters_too():
    from locals import formats

    fmt = formats.FORMATS["hermes"]
    reply = '<tool_call>\n{"name": "search", "arguments": {"params": "{\\"q\\": \\"x\\"}", "limit": "5", "note": "null"}}\n</tool_call>'
    _, _, calls = oai.parse_reply(reply, NESTED, fmt=fmt)
    # a string the model wrote stays one where the schema allows strings
    assert json.loads(calls[0]["function"]["arguments"]) == {"params": {"q": "x"}, "limit": 5, "note": "null"}
    assert oai.param_schemas('{"properties": {"a": {"type": "integer"}}}') == {"a": {"type": "integer"}}
    assert oai.param_schemas("not json") == oai.param_schemas(None) == oai.param_schemas({"properties": [1]}) == {}
    assert oai.typed_call_regex(NESTED)  # (the typed grammar read the same schemas: true ended it with a TypeError)


def test_parameter_names_outside_the_usual_letters_can_be_written():
    """MCP tools with OData-style ($select), non-English (größe) or spaced parameter names: the grammar allowed only
    [A-Za-z0-9_.-], so the model could not write them."""
    from locals.regex import compile_dfa

    tools = [{"type": "function", "function": {"name": "query", "parameters": {"type": "object", "properties": {
        "$select": {"type": "string"}, "größe": {"type": "integer"}, "user id": {"type": "string"}, "plain": {"type": "string"}}}}}]
    d = compile_dfa(oai.reply_regex(tools, "required", preamble=False), 100_000)
    reply = _call("query", **{"$select": "name", "größe": "42", "user id": "u1", "plain": "x"})
    assert d.matches(reply.encode())
    assert not d.matches(_call("query", **{"a>b": "1"}).encode())
    _, calls, got, _, _ = _run(reply, tools=tools)
    assert json.loads(got[0]) == {"$select": "name", "größe": 42, "user id": "u1", "plain": "x"}


@pytest.mark.parametrize("step", [1, 4])
def test_typed_enum_and_nullable_values(step):
    """--typed-tools wrote an enum's null as None and true as True (back as the strings "None" and "True"), and an
    Optional[int] could not be null; the stream sends what the final parse gives."""
    from locals.regex import compile_dfa

    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "mode": {"enum": ["a", None]}, "flag": {"enum": [True, False]}, "n": {"type": ["integer", "null"]},
        "s": {"enum": ["null", "x"]}}, "required": ["mode", "flag", "n", "s"]}}}]
    reply = _call("f", mode="null", flag="true", n="null", s="null")
    assert compile_dfa(oai.reply_regex(tools, "required", typed=True, preamble=False), 200_000).matches(reply.encode())
    _, calls, got, _, _ = _run(reply, step=step, tools=tools)
    assert json.loads(calls[0]["function"]["arguments"]) == {"mode": None, "flag": True, "n": None, "s": "null"}
    assert got[0] == calls[0]["function"]["arguments"]

