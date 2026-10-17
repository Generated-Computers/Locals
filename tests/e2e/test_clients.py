"""The libraries apps are built with, against the running server: the OpenAI Python SDK (sync and async,
streams, tools, JSON schemas, errors) and Ollama's HTTP API as Ollama clients call it."""

import asyncio
import json
import urllib.request

import pytest

openai = pytest.importorskip("openai")

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "The weather in a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


@pytest.fixture(scope="module")
def client(stack):
    return openai.OpenAI(base_url=f"http://127.0.0.1:{stack.port}/v1", api_key="locals", max_retries=0, timeout=300)


@pytest.fixture(scope="module")
def model(client):
    models = client.models.list().data
    assert models, "no model listed"
    return models[0].id


def test_chat_and_stream(client, model):
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Say hi in three words."}], max_tokens=20)
    assert r.choices[0].message.content and r.usage.completion_tokens > 0
    chunks = list(client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Count to five."}],
                                                 max_tokens=40, stream=True, stream_options={"include_usage": True}))
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert "3" in text or "three" in text.lower()
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens > 0
    assert any(c.choices and c.choices[0].finish_reason for c in chunks)


def test_tool_calls(client, model):
    r = client.chat.completions.create(model=model, tools=[WEATHER], max_tokens=200,
                                       messages=[{"role": "user", "content": "What is the weather in Paris and in Rome?"}])
    calls = r.choices[0].message.tool_calls or []
    assert calls and r.choices[0].finish_reason == "tool_calls"
    for c in calls:
        assert c.function.name == "get_weather" and json.loads(c.function.arguments)["city"]
    # the tool results go back, the model answers in text
    msgs = [{"role": "user", "content": "What is the weather in Paris and in Rome?"}, r.choices[0].message.model_dump(exclude_none=True)]
    for c in calls:
        msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps({"city": json.loads(c.function.arguments)["city"], "sky": "sunny", "c": 21})})
    r2 = client.chat.completions.create(model=model, tools=[WEATHER], messages=msgs, max_tokens=120)
    assert r2.choices[0].message.content and "sun" in r2.choices[0].message.content.lower()


def test_streamed_tool_call(client, model):
    chunks = list(client.chat.completions.create(model=model, tools=[WEATHER], tool_choice="required", stream=True, max_tokens=120,
                                                 messages=[{"role": "user", "content": "Weather in Oslo?"}]))
    args, names, ids = "", set(), set()
    for c in chunks:
        for tc in (c.choices[0].delta.tool_calls or []) if c.choices else []:
            args += tc.function.arguments or "" if tc.function else ""
            names.add(tc.function.name) if tc.function and tc.function.name else None
            ids.add(tc.id) if tc.id else None
    assert names == {"get_weather"} and len(ids) == 1
    assert json.loads(args)["city"].lower() == "oslo"


def test_json_schema(client, model):
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"],
              "additionalProperties": False}
    r = client.chat.completions.create(model=model, max_tokens=80, response_format={"type": "json_schema", "json_schema": {
        "name": "person", "schema": schema, "strict": True}}, messages=[{"role": "user", "content": "Ana is 31. As JSON."}])
    assert json.loads(r.choices[0].message.content) == {"name": "Ana", "age": 31}


def test_errors_are_clean(client, model):
    with pytest.raises(openai.BadRequestError) as e:
        client.chat.completions.create(model=model, max_tokens=10, messages=[{"role": "user", "content": "word " * 400_000}])
    assert "context" in str(e.value).lower() or "token" in str(e.value).lower()
    # a model name the server does not know is served by its model (agents' configs carry all sorts of names);
    # the reply says which model answered
    r = client.chat.completions.create(model="no-such-model-at-all", max_tokens=5, messages=[{"role": "user", "content": "hi"}])
    assert r.choices and r.model
    # the server still answers
    assert client.chat.completions.create(model=model, max_tokens=5, messages=[{"role": "user", "content": "hi"}]).choices


def test_eight_async_streams_at_once(stack, model, report):
    aclient = openai.AsyncOpenAI(base_url=f"http://127.0.0.1:{stack.port}/v1", api_key="locals", max_retries=0, timeout=300)

    async def one(i):
        s = await aclient.chat.completions.create(model=model, stream=True, max_tokens=150,
                                                  messages=[{"role": "user", "content": f"Write a haiku about the number {i}."}])
        return "".join([c.choices[0].delta.content or "" async for c in s if c.choices])

    async def main():
        return await asyncio.gather(*[one(i) for i in range(8)])

    texts = asyncio.run(main())
    assert all(t.strip() for t in texts)


def _post(stack, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{stack.port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return [json.loads(ln) for ln in r.read().decode().splitlines() if ln.strip()]


def test_ollama_api(stack):
    tags = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{stack.port}/api/tags", timeout=10).read())
    name = tags["models"][0]["name"]
    lines = _post(stack, "/api/chat", {"model": name, "messages": [{"role": "user", "content": "Say hi."}], "options": {"num_predict": 20}})
    assert lines[-1]["done"] and "".join(ln["message"]["content"] for ln in lines).strip()
    lines = _post(stack, "/api/chat", {"model": name, "stream": False, "tools": [WEATHER],
                                       "messages": [{"role": "user", "content": "Weather in Lima?"}]})
    calls = lines[-1]["message"].get("tool_calls") or []
    assert calls and calls[0]["function"]["name"] == "get_weather" and isinstance(calls[0]["function"]["arguments"], dict)
    lines = _post(stack, "/api/generate", {"model": name, "prompt": "2+2=", "stream": False, "options": {"num_predict": 5}})
    assert lines[-1]["done"] and lines[-1]["response"]
    for path in ("/api/version", "/api/ps"):
        assert urllib.request.urlopen(f"http://127.0.0.1:{stack.port}{path}", timeout=10).status == 200
    show = _post(stack, "/api/show", {"model": name})
    assert show and show[0].get("model_info") is not None or show[0].get("details") is not None


def test_anthropic_messages_api(stack):
    """Claude Code and the Anthropic SDKs: /v1/messages, streamed and not, a tool-use round trip, token counts."""
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic(base_url=f"http://127.0.0.1:{stack.port}", api_key="locals", max_retries=0, timeout=300)
    tool = {"name": "get_weather", "description": "The weather in a city.",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
    ask = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
    msg = client.messages.create(model="claude-sonnet-anything", max_tokens=200, tools=[tool], messages=ask)
    use = next(b for b in msg.content if b.type == "tool_use")
    assert msg.stop_reason == "tool_use" and use.name == "get_weather" and use.input["city"]
    turn = ask + [{"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in msg.content]},
                  {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": "sunny, 21 C"}]}]
    with client.messages.stream(model="claude-sonnet-anything", max_tokens=150, tools=[tool], messages=turn) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
    assert "sun" in text.lower() and final.stop_reason == "end_turn"
    assert final.usage.cache_read_input_tokens > 0  # the conversation so far came from the prompt cache
    assert client.messages.count_tokens(model="x", messages=ask).input_tokens > 5
