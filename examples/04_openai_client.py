"""Use `locals serve` from any OpenAI-compatible client.

    locals serve                          # in another terminal: http://127.0.0.1:8321/v1
    python examples/04_openai_client.py

Plain HTTP here so the example has no dependencies; the `openai` package works
the same way with base_url="http://127.0.0.1:8321/v1" and any api_key.
Requests whose model id starts with "locals" use our engine; "mlx/..." uses
stock mlx-lm with llguidance, for comparison.
"""

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8321/v1"


def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return json.load(r)


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def post_stream(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        for line in r:  # server-sent events
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                yield json.loads(line[6:])


try:
    models = [m["id"] for m in get("/models")["data"]]
except Exception as e:  # noqa: BLE001
    sys.exit(f"is `locals serve` running? ({e})")
model = next(m for m in models if m.startswith("locals") and "thinking" not in m)
print("models:", ", ".join(models), "\nusing:", model)

# 1. a reply, streamed
print("\n1. streaming:", end=" ", flush=True)
for chunk in post_stream("/chat/completions", {"model": model, "stream": True, "max_tokens": 60,
                                               "messages": [{"role": "user", "content": "Name three uses of a hash table, briefly."}]}):
    print(chunk["choices"][0]["delta"].get("content") or "", end="", flush=True)
print()

# 2. a JSON schema as response_format: the reply is in the schema's language
schema = {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"},
                                           "founded": {"type": "integer"}}, "required": ["city", "country", "founded"]}
r = post("/chat/completions", {"model": model, "max_tokens": 60,
                               "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": schema}},
                               "messages": [{"role": "user", "content": "The oldest capital city in Europe, as JSON."}]})
print("\n2. response_format:", json.loads(r["choices"][0]["message"]["content"]))

# 3. tools: the reply is decoded under a grammar, so every call parses and names an offered tool
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city.",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"]}},
                                                          "required": ["city"]}}},
         {"type": "function", "function": {"name": "convert", "description": "Convert an amount between currencies.",
                                           "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}},
                                                          "required": ["amount", "from", "to"]}}}]
r = post("/chat/completions", {"model": model, "max_tokens": 200, "tools": tools,
                               "messages": [{"role": "user", "content": "What is the weather in Kyoto, and how much is 100 euros in yen?"}]})
msg = r["choices"][0]["message"]
print("\n3. tool calls:")
for call in msg.get("tool_calls") or []:
    print("   ", call["function"]["name"], json.loads(call["function"]["arguments"]))
print("   usage:", r["usage"])
