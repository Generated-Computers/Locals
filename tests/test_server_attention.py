"""The server on a model whose cache is plain KV (Qwen2.5, Llama, Qwen3, Mistral):
no prompt snapshots are taken (the prompt cache trims a cached conversation to
any shared prefix), and a turn whose history comes back with a context message
moved to the end (as OpenClaw sends it) still continues from the cache.
Skipped if the model is not cached."""

import os

import pytest

pytest.importorskip("mlx_lm")

MODEL = os.environ.get("LOCALS_TEST_ATTENTION_MODEL", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
SHORT = MODEL.split("/")[-1].replace("-MLX-4bit", "").replace("-4bit", "").lower()
TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file.", "parameters": {
    "type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}}}]


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


def test_moved_context_message_hits_the_cache_without_snapshots(server):
    assert not server._needs_checkpoints("locals")
    system = "You are a coding agent. Use the tools.\n" + "\n".join(
        f"Rule {i}: read a file before you edit it, and keep replies short ({i} of 120)." for i in range(120))
    ctx = {"role": "user", "content": "<context>Active exec sessions: none</context>"}
    task = {"role": "user", "content": "Show me the first lines of /tmp/app/main.py"}
    r = server.complete({"model": f"locals/{SHORT}", "tools": TOOLS, "tool_choice": "required", "max_tokens": 80,
                         "messages": [{"role": "system", "content": system}, task, ctx]})
    msg = r["choices"][0]["message"]
    assert msg.get("tool_calls")
    p1 = r["usage"]["prompt_tokens"]
    msgs = [{"role": "system", "content": system}, task, {"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg["tool_calls"]},
            {"role": "tool", "tool_call_id": msg["tool_calls"][0]["id"], "content": "1: import os\n"}, ctx]
    r2 = server.complete({"model": f"locals/{SHORT}", "tools": TOOLS, "max_tokens": 40, "messages": msgs})
    assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] >= p1 - 40, (r2["usage"], p1)
