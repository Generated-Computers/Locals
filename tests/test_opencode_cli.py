"""Opt-in real CLI contract: LOCALS_TEST_OPENCODE=/path/to/opencode pytest tests/test_opencode_cli.py.

Uses a deterministic loopback provider, no model weights, cloud requests, or real user configuration.
Run once per supported OpenCode binary (v1.18.32 and v2.1.2).
"""

import http.server
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import pytest

from locals import app


@pytest.mark.parametrize("engine", ["locals", "mlx"])
def test_real_cli_delegates_executes_tools_and_streams_answer(tmp_path, monkeypatch, engine):
    binary = os.environ.get("LOCALS_TEST_OPENCODE")
    if not binary:
        pytest.skip("set LOCALS_TEST_OPENCODE to test a real OpenCode binary")
    binary = str(Path(binary).resolve())
    project = tmp_path / "project"
    project.mkdir()
    for key, directory in (("HOME", "home"), ("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
                           ("XDG_STATE_HOME", "state"), ("XDG_CACHE_HOME", "cache")):
        monkeypatch.setenv(key, str(tmp_path / directory))
    for key in ("OPENCODE_CONFIG_CONTENT", "OPENCODE_CONFIG_DIR", "OPENCODE_PASSWORD", "OPENCODE_SERVER_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENCODE_DISABLE_MODELS_FETCH", "1")
    monkeypatch.setenv("OPENCODE_DISABLE_CLAUDE_CODE", "1")
    monkeypatch.setenv("OPENCODE_DISABLE_AUTOUPDATE", "1")
    monkeypatch.setenv("PWD", str(project))
    monkeypatch.chdir(project)
    v2 = app._opencode_v2(binary)
    delegate, shell, agent_key = ("subagent", "shell", "agent") if v2 else ("task", "bash", "subagent_type")
    requests = []

    class Provider(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            tools = {t["function"]["name"] for t in body.get("tools", [])}
            messages = body["messages"]
            child = any(m.get("role") == "user" and "COMPAT_CHILD" in str(m.get("content")) for m in messages)
            complete = any(m.get("role") == "tool" for m in messages)
            if tools and not complete:
                name = shell if child else delegate
                arguments = ({"command": "printf verified > smoke.txt", "description": "Write smoke test file"} if child
                             else {agent_key: "coder", "description": "Verify child execution", "prompt": "COMPAT_CHILD"})
                message = {"role": "assistant", "content": None, "tool_calls": [
                    {"index": 0, "id": "call_compat", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(arguments)}}]}
                reason = "tool_calls"
            else:
                message, reason = {"role": "assistant", "content": "LOCALS_COMPAT_OK"}, "stop"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            common = {"id": "chatcmpl-compat", "object": "chat.completion.chunk", "created": int(time.time()),
                      "model": body["model"]}
            for delta, finish in ((message, None), ({}, reason)):
                chunk = {**common, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        cfg = app.opencode_config("mlx-community/Qwen3.5-4B-MLX-4bit", server.server_port, "mlx")
        config = tmp_path / "opencode.json"
        config.write_text(json.dumps(cfg))
        monkeypatch.setenv("OPENCODE_CONFIG", str(config))
        # The explicit launch selection must beat a project's own model and agent defaults.
        (project / "opencode.jsonc").write_text(json.dumps({"model": "cloud/wrong", "default_agent": "build"}))
        ref = app.oc_model(engine, "qwen3.5-4b")
        env = dict(os.environ)
        if v2:
            env["OPENCODE_CONFIG_CONTENT"] = json.dumps(app._opencode_v2_config(cfg, ref))
        result = subprocess.run([binary, "run", *(["--standalone"] if v2 else []), "--format", "json", "-m", ref,
                                 *([] if v2 else ["--agent", "orchestrator"]), "COMPAT_PARENT"],
                                env=env, text=True, capture_output=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        assert app._opencode_answer(result.stdout) == "LOCALS_COMPAT_OK"
        assert any(delegate in app._opencode_step(line) for line in result.stdout.splitlines())
        assert (project / "smoke.txt").read_text() == "verified"
        inference = [r for r in requests if r.get("tools")]
        assert len(inference) >= 4  # parent delegates, child runs shell, both finish
        assert all(r["model"] == engine + "/qwen3.5-4b" for r in inference)
        if v2:
            # Saved sessions carry their own model. Continuation must apply a newly requested engine too.
            other = "mlx" if engine == "locals" else "locals"
            flags = app._opencode_v2_continue(binary, project, app.oc_model(other, "qwen3.5-4b"), env)
            assert flags[0] == "--session"
            saved = subprocess.run([binary, "api", "--standalone", "session.get", "--param", f"sessionID={flags[1]}"],
                                   env=env, text=True, capture_output=True, timeout=60)
            assert saved.returncode == 0, saved.stdout + saved.stderr
            selected = json.loads(saved.stdout)["data"]["model"]
            assert selected["providerID"] == other and selected["id"] == other + "/qwen3.5-4b"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
