"""Anthropic's Messages API (/v1/messages): the translation to and from Chat Completions, the event stream,
and a round trip through the real HTTP handler with the Anthropic SDK, the engine a stand-in that writes a
fixed reply in Qwen3.5's tool-call format (Qwen3.5-4B's tokenizer, from the cache)."""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from locals import apis

WEATHER = {"name": "get_weather", "description": "The weather in a city.",
           "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}


def test_a_request_becomes_chat_messages_in_order():
    req = {"model": "claude-x", "max_tokens": 500, "stream": True, "system": [{"type": "text", "text": "You are a coder."},
                                                                               {"type": "text", "text": "Be brief."}],
           "tools": [WEATHER, {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
           "tool_choice": {"type": "any", "disable_parallel_tool_use": True}, "stop_sequences": ["END"],
           "thinking": {"type": "enabled", "budget_tokens": 2000},
           "messages": [
               {"role": "user", "content": [{"type": "text", "text": "Weather in Oslo?"}, {"type": "image", "source": {}}]},
               {"role": "assistant", "content": [{"type": "thinking", "thinking": "a tool", "signature": "s"},
                                                 {"type": "text", "text": "Checking."},
                                                 {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Oslo"}}]},
               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "4 C"}]},
                                            {"type": "text", "text": "And in Rome?"}]},
               {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_2", "name": "get_weather", "input": {"city": "Rome"}}]},
               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_2", "content": "no data", "is_error": True}]}]}
    chat = apis.anthropic_to_chat(req, "locals/qwen3.5-4b")
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "user", "assistant", "tool"]
    assert chat["messages"][0]["content"] == "You are a coder.\n\nBe brief."
    assert chat["messages"][1]["content"] == "Weather in Oslo?\n" + apis.IMAGE_NOTE
    a = chat["messages"][2]
    assert a["content"] == "Checking." and a["reasoning_content"] == "a tool"
    assert a["tool_calls"][0]["id"] == "toolu_1" and json.loads(a["tool_calls"][0]["function"]["arguments"]) == {"city": "Oslo"}
    assert chat["messages"][3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "4 C"}
    assert chat["messages"][6]["content"] == "Error: no data"
    assert [t["function"]["name"] for t in chat["tools"]] == ["get_weather"]  # the web search is Anthropic's own
    assert chat["tool_choice"] == "required" and chat["parallel_tool_calls"] is False
    assert chat["stop"] == ["END"] and chat["max_tokens"] == 500 and chat["stream"] is True
    assert chat["chat_template_kwargs"] == {"enable_thinking": True}


def test_a_reply_becomes_a_message():
    resp = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": "Let me look.",
            "reasoning_content": "hm", "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "get_weather", "arguments": "{\"city\": \"Lima\"}"}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 60}}}
    m = apis.chat_to_anthropic(resp, "claude-x")
    assert [b["type"] for b in m["content"]] == ["thinking", "text", "tool_use"]
    assert m["content"][2]["input"] == {"city": "Lima"} and m["stop_reason"] == "tool_use" and m["model"] == "claude-x"
    assert m["usage"] == {"input_tokens": 40, "output_tokens": 7, "cache_read_input_tokens": 60, "cache_creation_input_tokens": 0}


def _events(raw: bytes):
    out = []
    for block in raw.decode().strip().split("\n\n"):
        lines = dict(ln.split(": ", 1) for ln in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_the_stream_becomes_message_events():
    got = []
    st = apis.AnthropicStream(got.append, "claude-x", 120)
    st.start()

    def chunk(delta=None, finish=None, usage=None):
        d = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]} if usage is None else {"choices": [], "usage": usage}
        st.feed(("data: " + json.dumps(d) + "\n\n").encode())

    chunk({"role": "assistant", "reasoning_content": "think"})
    st.feed(b"\n")  # a keepalive
    chunk({"content": "Hello"})
    chunk({"content": " there"})
    chunk({"tool_calls": [{"index": 0, "id": "call_9", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]})
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\": "}}]})
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": "\"Rome\"}"}}]})
    chunk({}, finish="tool_calls")
    chunk(usage={"prompt_tokens": 120, "completion_tokens": 30, "prompt_tokens_details": {"cached_tokens": 100}})
    st.feed(b"data: [DONE]\n\n")
    st.close()
    ev = _events(b"".join(got))
    names = [n for n, _ in ev]
    assert names == ["message_start", "content_block_start", "content_block_delta", "ping", "content_block_delta",
                     "content_block_stop", "content_block_start", "content_block_delta", "content_block_delta",
                     "content_block_stop", "content_block_start", "content_block_delta", "content_block_delta",
                     "content_block_stop", "message_delta", "message_stop"]
    assert ev[0][1]["message"]["usage"]["input_tokens"] == 120
    tool_start = ev[10][1]
    assert tool_start["index"] == 2 and tool_start["content_block"]["name"] == "get_weather"
    assert "".join(d["delta"]["partial_json"] for n, d in ev if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta") \
        == "{\"city\": \"Rome\"}"
    assert ev[-2][1]["delta"]["stop_reason"] == "tool_use" and ev[-2][1]["usage"]["cache_read_input_tokens"] == 100


def test_an_error_in_the_stream_is_an_error_event():
    got = []
    st = apis.AnthropicStream(got.append, "claude-x", 5)
    st.start()
    st.feed(b'data: {"error": {"message": "Metal ran out of memory"}}\n\n')
    st.close()
    ev = _events(b"".join(got))
    assert [n for n, _ in ev] == ["message_start", "error"] and ev[1][1]["error"]["type"] == "api_error"


# --- the real handler, the Anthropic SDK -------------------------------------------------------

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"


class _Job(SimpleNamespace):
    def cancel(self):
        self.cancelled = True


class _Svc:
    """Writes a fixed reply (as the model's tokens), a few tokens every 10 ms."""

    def __init__(self, tok, reply):
        self.tokenizer, self.reply = tok, reply
        self.prompts = []

    def context_fit(self):
        return None

    def submit(self, prompt, max_tokens=16, on_token=None, **kw):
        self.prompts.append(list(prompt))
        ids = self.tokenizer._tokenizer.encode(self.reply, add_special_tokens=False)
        job = _Job(done=False, finish="stop", tokens=[], text="", cached=0, t_submit=time.perf_counter(), t_done=0.0, cancelled=False)

        def run():
            for i in range(0, len(ids), 3):
                if job.cancelled:
                    break
                on_token(ids[i:i + 3])
                job.tokens.extend(ids[i:i + 3])
                time.sleep(0.01)
            job.text = self.tokenizer.decode(job.tokens)
            job.t_done = time.perf_counter()
            job.done = True

        threading.Thread(target=run, daemon=True).start()
        return job


@pytest.fixture(scope="module")
def tokenizer():
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(MODEL, local_files_only=True, allow_patterns=["*.json", "*.jinja", "tokenizer*"])
    except Exception:  # noqa: BLE001
        pytest.skip("tokenizer not cached")
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(path)


def _serve(tokenizer, reply):
    from locals.server import Server, _HTTPServer, make_handler

    srv = Server(MODEL, engines=("locals",), preload=(), log=lambda s: None)
    srv._svc["locals"] = svc = _Svc(tokenizer, reply)
    httpd = _HTTPServer(("127.0.0.1", 0), make_handler(srv))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, svc


CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>"


def test_the_anthropic_sdk_gets_a_tool_call_and_a_text_reply(tokenizer):
    anthropic = pytest.importorskip("anthropic")
    httpd, svc = _serve(tokenizer, CALL)
    try:
        client = anthropic.Anthropic(base_url=f"http://127.0.0.1:{httpd.server_address[1]}", api_key="locals", max_retries=0)
        msg = client.messages.create(model="claude-anything", max_tokens=200, tools=[WEATHER],
                                     messages=[{"role": "user", "content": "Weather in Rome?"}])
        assert msg.stop_reason == "tool_use" and msg.model == "claude-anything"
        use = [b for b in msg.content if b.type == "tool_use"][0]
        assert use.name == "get_weather" and use.input == {"city": "Rome"}
        with client.messages.stream(model="claude-anything", max_tokens=200, tools=[WEATHER],
                                    messages=[{"role": "user", "content": "Weather in Rome?"}]) as stream:
            final = stream.get_final_message()
        assert [b.type for b in final.content] == ["tool_use"] and final.content[0].input == {"city": "Rome"}
        count = client.messages.count_tokens(model="claude-anything", tools=[WEATHER],
                                             messages=[{"role": "user", "content": "Weather in Rome?"}])
        assert count.input_tokens == len(svc.prompts[-1])
        svc.reply = "It is sunny in Rome."
        with client.messages.stream(model="claude-anything", max_tokens=200, messages=[
                {"role": "user", "content": "Weather in Rome?"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": use.id, "name": "get_weather", "input": use.input}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": "sunny, 21 C"}]}],
                tools=[WEATHER]) as stream:
            text = "".join(stream.text_stream)
            final = stream.get_final_message()
        assert text == "It is sunny in Rome." and final.stop_reason == "end_turn"
        assert "sunny, 21 C" in tokenizer.decode(svc.prompts[-1])  # the tool result reached the prompt
    finally:
        httpd.shutdown()


def test_errors_have_anthropics_shape(tokenizer):
    anthropic = pytest.importorskip("anthropic")
    httpd, _ = _serve(tokenizer, "hi")
    try:
        client = anthropic.Anthropic(base_url=f"http://127.0.0.1:{httpd.server_address[1]}", api_key="locals", max_retries=0)
        with pytest.raises(anthropic.BadRequestError) as e:
            client.messages.create(model="claude-anything", max_tokens=10, messages=[{"role": "user", "content": "word " * 300_000}])
        assert e.value.body["error"]["type"] == "invalid_request_error"
        assert e.value.body["error"]["message"].startswith("prompt is too long: ")  # Claude Code compacts on these words
        with pytest.raises(anthropic.BadRequestError):
            client.messages.create(model="claude-anything", max_tokens=10, messages=[])
        models = client.models.list()
        assert models.data and models.data[0].id
    finally:
        httpd.shutdown()


def test_a_compressed_body_is_read(tokenizer):
    """Claude Code can gzip its request bodies (CLAUDE_CODE_GZIP_REQUEST_BODIES)."""
    import gzip
    import urllib.request

    httpd, _ = _serve(tokenizer, "Hello.")
    try:
        body = gzip.compress(json.dumps({"model": "claude-anything", "max_tokens": 20,
                                         "messages": [{"role": "user", "content": "Hi"}]}).encode())
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/messages", data=body,
                                     headers={"Content-Type": "application/json", "Content-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=30) as r:
            msg = json.loads(r.read())
        assert msg["content"][0]["text"] == "Hello." and msg["stop_reason"] == "end_turn"
    finally:
        httpd.shutdown()


def test_launch_claude_points_claude_code_at_locals(monkeypatch):
    """locals launch claude: Claude Code on the local server for that launch only (its own settings untouched),
    with the server's context window, and without the user's key for Anthropic's API."""
    import argparse

    from locals import app

    seen = {}
    monkeypatch.setattr(app, "_installed", lambda name, env: "/opt/claude")
    monkeypatch.setattr(app, "fetch_model", lambda model: None)
    monkeypatch.setattr(app, "ensure_server", lambda args: None)
    monkeypatch.setattr(app, "get_json", lambda url, timeout=5: {"data": [{"context_length": 145408}]})
    def execve(exe, argv, env):  # (the real one never returns)
        seen.update(exe=exe, argv=argv, env=env)
        raise SystemExit(0)

    monkeypatch.setattr(app.os, "execve", execve)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real-key")
    with pytest.raises(SystemExit):
        app.cmd_launch(argparse.Namespace(tool="claude", rest=["-p", "hi"], port=8321, model="mlx-community/Qwen3.5-4B-MLX-4bit",
                                          ollama_port=None))
    env = seen["env"]
    assert seen["argv"] == ["/opt/claude", "-p", "hi"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8321" and env["ANTHROPIC_MODEL"] == "locals/qwen3.5-4b"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "145408" and "ANTHROPIC_API_KEY" not in env
