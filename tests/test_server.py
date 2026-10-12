"""The OpenAI-compatible server (locals/server.py) on Qwen3.5-2B: tool calls
decode under the reply grammar and parse, JSON-schema response formats
conform, thinking streams as reasoning, repeated calls are steered, and the
re-rendered conversation hits the prompt cache. Skipped if the model is not
cached."""

import os
import json

import pytest

pytest.importorskip("mlx_lm")

MODEL = os.environ.get("LOCALS_TEST_MODEL", "mlx-community/Qwen3.5-2B-MLX-4bit")
SHORT = MODEL.split("/")[-1].replace("-MLX-4bit", "").replace("-4bit", "").lower()
TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file.", "parameters": {"type": "object", "properties": {
        "filePath": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["filePath"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}, "description": {"type": "string"}}, "required": ["command", "description"]}}},
]


@pytest.fixture(scope="module")
def server():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    from locals.server import Server

    srv = Server(MODEL, engines=("locals",), preload=("locals",), slots=4, temp=0.0, log=lambda s: None)
    yield srv
    for svc in srv._svc.values():
        svc.close()


def test_tool_call_grammar(server):
    r = server.complete({"model": f"locals/{SHORT}", "tools": TOOLS, "max_tokens": 200,
                         "messages": [{"role": "system", "content": "You are a coding agent. Use the tools."},
                                      {"role": "user", "content": "Show me the first 20 lines of /tmp/app/main.py"}]})
    msg = r["choices"][0]["message"]
    assert r["choices"][0]["finish_reason"] == "tool_calls" and msg["tool_calls"]
    names = {c["function"]["name"] for c in msg["tool_calls"]}
    assert names <= {"read", "bash"}
    args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
    assert isinstance(args, dict) and args
    assert r["usage"]["locals"]["constrained"] and r["usage"]["locals"]["forced_tokens"] > 0
    # the round trip re-renders the reply verbatim, so the prompt cache serves the whole previous turn
    msgs = [{"role": "system", "content": "You are a coding agent. Use the tools."},
            {"role": "user", "content": "Show me the first 20 lines of /tmp/app/main.py"}, msg,
            {"role": "tool", "tool_call_id": msg["tool_calls"][0]["id"], "content": "1: import os\n2: print('hi')\n"}]
    r2 = server.complete({"model": f"locals/{SHORT}", "tools": TOOLS, "max_tokens": 100, "messages": msgs})
    assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] >= r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"] - 2


def test_json_schema(server):
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}
    r = server.complete({"model": f"locals/{SHORT}", "max_tokens": 60, "response_format": {"type": "json_schema", "json_schema": {"name": "p", "schema": schema}},
                         "messages": [{"role": "user", "content": "Maria Gonzalez is 34."}]})
    obj = json.loads(r["choices"][0]["message"]["content"])
    assert obj["name"] and isinstance(obj["age"], int)


def test_thinking_stream(server):
    chunks = []
    server.stream({"model": f"locals/{SHORT}-thinking", "max_tokens": 200, "stream": True,
                   "messages": [{"role": "user", "content": "What is 17*23? Answer briefly."}]},
                  lambda b: chunks.append(b.decode()))
    deltas = [json.loads(c[6:]) for c in chunks if c.startswith("data: {")]
    reasoning = "".join(d["choices"][0]["delta"].get("reasoning_content") or "" for d in deltas)
    content = "".join(d["choices"][0]["delta"].get("content") or "" for d in deltas)
    assert reasoning and "</think>" not in reasoning and "<think>" not in content
    assert chunks[-1].startswith("data: [DONE]")


def test_repeat_is_steered(server):
    call = {"id": "call_x1", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "python hello.py", "description": "run"})}}
    msgs = [{"role": "system", "content": "You are a coding agent. Use the tools."},
            {"role": "user", "content": "Run hello.py with the bash tool."},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_x1", "content": "zsh: command not found: python"}]
    r = server.complete({"model": f"locals/{SHORT}", "tools": TOOLS, "max_tokens": 200, "messages": msgs})
    msg = r["choices"][0]["message"]
    for c in msg.get("tool_calls") or []:
        assert json.loads(c["function"]["arguments"]).get("command") != "python hello.py"


def test_moving_context_message_hits_checkpoint(server):
    """OpenClaw re-appends a context message at the end of every request and
    rewrites tool-call ids to letters and digits: the history before the last
    message is still served from the cache (the checkpoint at its start)."""
    system = "You are a coding agent. Use the tools.\n" + "\n".join(  # past the 1024-token checkpoint threshold
        f"Rule {i}: read a file before you edit it, run what you change, and keep replies short ({i} of 90)." for i in range(90))
    ctx = {"role": "user", "content": "<context>Active exec sessions: none</context>"}
    task = {"role": "user", "content": "Show me the first 20 lines of /tmp/app/main.py"}
    r = server.complete({"model": f"locals/{SHORT}", "tools": TOOLS, "tool_choice": "required", "max_tokens": 120,
                         "messages": [{"role": "system", "content": system}, task, ctx]})
    msg = r["choices"][0]["message"]
    assert msg.get("tool_calls")
    call = dict(msg["tool_calls"][0], id=msg["tool_calls"][0]["id"].replace("_", ""))
    msgs = [{"role": "system", "content": system}, task, {"role": "assistant", "content": msg["content"], "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": "1: import os\n"}, ctx]
    r2 = server.complete({"model": f"locals/{SHORT}", "tools": TOOLS, "max_tokens": 60, "messages": msgs})
    cached, prompt = r2["usage"]["prompt_tokens_details"]["cached_tokens"], r2["usage"]["prompt_tokens"]
    # everything the first request had before the moved message comes from the cache
    assert cached >= r["usage"]["prompt_tokens"] - 40, (cached, prompt, r["usage"]["prompt_tokens"])


def test_tool_choice_none_and_context_limit(server):
    from locals.server import ContextTooLong

    r = server.complete({"model": f"locals/{SHORT}", "tools": TOOLS, "tool_choice": "none", "max_tokens": 60,
                         "messages": [{"role": "user", "content": "Show me /tmp/app/main.py"}]})
    assert not r["choices"][0]["message"].get("tool_calls") and r["choices"][0]["finish_reason"] in ("stop", "length")
    limit, server.max_context = server.max_context, 64
    try:
        with pytest.raises(ContextTooLong, match="maximum context length is 64 tokens"):
            server.prepare({"model": f"locals/{SHORT}", "messages": [{"role": "user", "content": "word " * 200}]})
    finally:
        server.max_context = limit
