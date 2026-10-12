"""The Ollama API: its translation into Chat Completions and back, streaming included (no
model: the server's own output is scripted)."""

import json

from locals import apis


def _sse(*chunks):
    """The server's own stream: Chat Completions chunks as server-sent events."""
    return b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"


def _delta(d, finish=None, usage=None):
    c = {"choices": [{"index": 0, "delta": d, "finish_reason": finish}]}
    if usage:
        c["usage"] = usage
    return c


CALL = {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}}]}
USAGE = {"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 60}}


def test_ollama_request_reply_and_stream():
    chat = apis.ollama_to_chat({"model": "qwen3.5-4b:latest", "messages": [
        {"role": "user", "content": "hi", "images": ["AAAA"]},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]},
        {"role": "tool", "content": "ok", "tool_name": "f"}],
        "options": {"num_predict": 20, "temperature": 0.1}, "format": "json", "think": False}, "locals/q")
    assert chat["stream"] and chat["max_tokens"] == 20 and chat["temperature"] == 0.1 and chat["response_format"] == {"type": "json_object"}
    assert apis.IMAGE_NOTE in chat["messages"][0]["content"] and chat["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert chat["chat_template_kwargs"] == {"enable_thinking": False}
    g = apis.ollama_to_chat({"prompt": "2+2=", "system": "math", "stream": False}, "locals/q", generate=True)
    assert [m["role"] for m in g["messages"]] == ["system", "user"] and not g["stream"]
    resp = {"choices": [{"message": {"role": "assistant", "content": "", **CALL}, "finish_reason": "tool_calls"}], "usage": USAGE}
    r = apis.chat_to_ollama(resp, "q:latest", 0)
    assert r["done"] and r["message"]["tool_calls"] == [{"function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}]
    assert r["prompt_eval_count"] == 100 and r["eval_count"] == 7
    out = []
    st = apis.OllamaStream(out.append, "q:latest", 0)
    st.feed(_sse(_delta({"content": "Hel"}), _delta({"content": "lo"}), _delta(CALL), _delta({}, "stop", USAGE)))
    st.close()
    lines = [json.loads(x) for x in b"".join(out).decode().splitlines()]
    assert "".join(x["message"]["content"] for x in lines) == "Hello" and [x["done"] for x in lines] == [False, False, True]
    assert lines[-1]["message"]["tool_calls"][0]["function"]["arguments"] == {"city": "Oslo"}


def test_ollama_think_levels():
    chat = apis.ollama_to_chat({"model": "m", "messages": [{"role": "user", "content": "hi"}], "think": "high"}, "locals/q")
    assert chat["chat_template_kwargs"] == {"enable_thinking": True} and chat["reasoning_effort"] == "high"
    chat = apis.ollama_to_chat({"model": "m", "messages": [{"role": "user", "content": "hi"}], "think": False}, "locals/q")
    assert chat["chat_template_kwargs"] == {"enable_thinking": False} and "reasoning_effort" not in chat
