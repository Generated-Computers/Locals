"""The walk kernel in the batch engine and the server: several constrained
rows decode together with the combined automaton table (no token tables),
token-identical to the table engine, and typed tool calls come out typed.
Skipped if the model is not cached."""

import os
import json

import pytest

pytest.importorskip("mlx_lm")

MODEL = os.environ.get("LOCALS_TEST_MODEL", "mlx-community/Qwen3.5-2B-MLX-4bit")
SHORT = MODEL.split("/")[-1].replace("-MLX-4bit", "").replace("-4bit", "").lower()


@pytest.fixture(scope="module")
def rt():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    from locals.runtime import Runtime

    return Runtime.load(MODEL)


def test_batch_walk_matches_tables(rt):
    from workloads import prompt_text
    from workloads import WORKLOADS

    from locals.blocks import NoHealController
    from locals.hybrid_batch import HybridBatchEngine
    from locals.packed import Request
    from locals.walk import WalkController

    jobs = [(w, rt.encode_chat(prompt_text(w, True))) for w in WORKLOADS[:4]]
    eng = HybridBatchEngine(rt.model, rt.tokenizer, n_slots=8)

    def run(make):
        reqs = [Request(list(ids), make(w["schema"]), 200) for w, ids in jobs]
        queue = list(reqs)

        def source(block):
            got, queue[:] = list(queue), []
            return got

        eng.snapshots.clear()
        eng.serve(source)
        return [r.out for r in reqs]

    table = run(lambda sc: (lambda c: NoHealController(c.constraint, c.planner, True, True))(rt.compile(sc)))
    walk = run(lambda sc: (lambda cw: WalkController(cw[0], cw[1], True, True))(rt.compile_walk(sc)))
    for (w, _), a, b in zip(jobs, table, walk):
        assert a == b, (w["name"], rt.tokenizer.decode(a), rt.tokenizer.decode(b))
        json.loads(rt.tokenizer.decode(b))


def test_server_typed_tools(rt):
    from locals.server import Server

    tools = [{"type": "function", "function": {"name": "read", "description": "Read a file.", "parameters": {"type": "object", "properties": {
        "filePath": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["filePath"]}}},
             {"type": "function", "function": {"name": "bash", "description": "Run a command.", "parameters": {"type": "object", "properties": {
                 "command": {"type": "string"}, "timeout": {"type": "integer"}, "background": {"type": "boolean"}}, "required": ["command"]}}}]
    srv = Server(MODEL, engines=("locals",), preload=(), slots=4, temp=0.0, walk=True, typed_tools=True, log=lambda s: None)
    srv._svc["locals"] = __import__("locals.service", fromlist=["BatchService"]).BatchService("locals", MODEL, slots=4, temp=0.0, walk=True)
    try:
        for ask in ("Show the first 20 lines of /tmp/app/main.py", "List the files in /tmp/app with a shell command, 5 second timeout, in the background"):
            r = srv.complete({"model": f"locals/{SHORT}", "tools": tools, "max_tokens": 120,
                              "messages": [{"role": "system", "content": "You are a coding agent. Use the tools."}, {"role": "user", "content": ask}]})
            msg = r["choices"][0]["message"]
            assert msg.get("tool_calls"), msg
            for c in msg["tool_calls"]:
                fn = c["function"]
                args = json.loads(fn["arguments"])
                spec = next(t for t in tools if t["function"]["name"] == fn["name"])["function"]["parameters"]
                assert set(args) <= set(spec["properties"]) and set(spec["required"]) <= set(args), (fn["name"], args)
                for k, v in args.items():
                    t = spec["properties"][k]["type"]
                    assert (t == "string" and isinstance(v, str)) or (t == "integer" and isinstance(v, int)) or (t == "boolean" and isinstance(v, bool)), (k, v)
    finally:
        for svc in srv._svc.values():
            svc.close()
