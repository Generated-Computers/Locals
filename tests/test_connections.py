"""`locals launch` (which tools are connected), `locals disconnect`, `locals models` and `locals bench`:
no model, no server, stand-ins for the tools' own commands."""

import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from locals import app, bench, defaults

QWEN = "mlx-community/Qwen3.5-4B-MLX-4bit"


def _opencode(cfg):
    app.OPENCODE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    app.OPENCODE_CONFIG.write_text(cfg if isinstance(cfg, str) else json.dumps(cfg))


def test_opencode_connection_states(monkeypatch):
    monkeypatch.setattr(app, "opencode_bin", lambda: "/x/opencode")
    state = lambda: app.connection("opencode", 8321)[0]  # noqa: E731
    assert state() == "not connected"  # no config yet
    app.write_config(app.opencode_config(QWEN, 8321))
    assert app.connection("opencode", 8321) == ("connected", "default model locals/locals/qwen3.5-4b")
    st, why = app.connection("opencode", 8341)  # the server runs on another port than the config says
    assert st == "needs repair" and "8321" in why and "locals code" in why
    _opencode({"provider": {"stcon": {"options": {"baseURL": "http://127.0.0.1:8321/v1"}}}, "model": "stcon/qwen3.5-4b"})
    assert app.connection("opencode", 8321) == ("needs repair", "it uses Locals under its old name, stcon: locals code updates it")
    _opencode('{"model": "x", // a comment\n}')
    assert state() == "needs repair"
    monkeypatch.setattr(app, "opencode_bin", lambda: None)
    assert state() == "not installed"


def test_a_setup_drops_the_old_stcon_provider():
    _opencode({"provider": {"stcon": {"options": {}}, "anthropic": {}}, "model": "stcon/qwen3.5-4b"})
    app.write_config(app.opencode_config(QWEN, 8321))
    cfg = json.loads(app.OPENCODE_CONFIG.read_text())
    assert set(cfg["provider"]) == {"anthropic", "locals"} and cfg["model"] == "locals/locals/qwen3.5-4b"


def test_disconnect_puts_opencode_back_as_it_was():
    mine = {"model": "anthropic/claude-x", "provider": {"anthropic": {"options": {"apiKey": "k"}}}, "theme": "dark",
            "agent": {"build": {"temperature": 0.1}}}
    _opencode(mine)
    app.write_config(app.opencode_config(QWEN, 8321))
    changed = app.disconnect_opencode()
    assert "the locals provider" in changed and "model" in changed
    assert json.loads(app.OPENCODE_CONFIG.read_text()) == mine
    assert app.disconnect_opencode() == []  # nothing left to take out


def test_disconnect_opencode_without_a_config_of_the_users():
    app.write_config(app.opencode_config(QWEN, 8321))  # OpenCode had no config: locals wrote the first one
    app.disconnect_opencode()
    assert json.loads(app.OPENCODE_CONFIG.read_text()) == {}


def _record_runs(monkeypatch):
    seen = []
    monkeypatch.setattr(app.subprocess, "run",
                        lambda cmd, **kw: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    return seen


def test_disconnect_openclaw_with_its_own_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(tmp_path / "oc"))
    (tmp_path / "oc").mkdir()
    patch = app.openclaw_patch(QWEN, 8321)
    (tmp_path / "oc" / "openclaw.json").write_text(json.dumps({**patch, "plugins": {}}))
    keep = app.LOCALS_DIR / "backups" / "openclaw-openclaw.json.before-locals"
    keep.parent.mkdir(parents=True)
    keep.write_text(json.dumps({"agents": {"defaults": {"model": {"primary": "anthropic/claude-x"}}}}))
    monkeypatch.setattr(app, "_tool_bin", lambda name, env: "/opt/openclaw")
    seen = _record_runs(monkeypatch)
    assert app.connection("openclaw", 8321)[0] == "connected"
    app.cmd_disconnect(argparse.Namespace(tool="openclaw"))
    assert ["/opt/openclaw", "config", "unset", "models.providers.locals"] in seen
    assert ["/opt/openclaw", "config", "set", "agents.defaults.model.primary", "anthropic/claude-x"] in seen


def test_disconnect_openclaw_set_up_under_the_old_name(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(tmp_path))
    (tmp_path / "openclaw.json").write_text(json.dumps({"models": {"providers": {"stcon": {"baseUrl": "x"}}},
                                                        "agents": {"defaults": {"model": {"primary": "stcon/stcon/qwen3.5-4b"}}}}))
    monkeypatch.setattr(app, "_tool_bin", lambda name, env: "/opt/openclaw")
    monkeypatch.setattr(app, "_installed", lambda name, env: "/opt/openclaw")
    assert app.connection("openclaw", 8321)[0] == "needs repair"
    seen = _record_runs(monkeypatch)
    app.disconnect_openclaw()
    assert ["/opt/openclaw", "config", "unset", "models.providers.stcon"] in seen
    assert ["/opt/openclaw", "config", "unset", "agents.defaults.model.primary"] in seen  # no backup: OpenClaw picks again


def test_disconnect_hermes_puts_back_its_model_settings(tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ours = {k.split(".", 1)[1]: v for k, v in app.hermes_settings(QWEN, 8321).items()}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"model": ours, "display": {"theme": "x"}}))
    keep = app.LOCALS_DIR / "backups" / "hermes-config.yaml.before-locals"
    keep.parent.mkdir(parents=True)
    keep.write_text(yaml.safe_dump({"model": {"provider": "openrouter", "default": "some/model"}}))
    monkeypatch.setattr(app, "_tool_bin", lambda name, env: "/opt/hermes")
    monkeypatch.setattr(app, "_installed", lambda name, env: "/opt/hermes")
    assert app.connection("hermes", 8321)[0] == "connected"
    seen = _record_runs(monkeypatch)
    app.disconnect_hermes()
    assert ["/opt/hermes", "config", "set", "model.provider", "openrouter"] in seen
    assert ["/opt/hermes", "config", "set", "model.default", "some/model"] in seen
    assert ["/opt/hermes", "config", "unset", "model.base_url"] in seen
    assert ["/opt/hermes", "config", "unset", "model.context_length"] in seen


def test_models_shows_fit_and_the_speed_measured_here(monkeypatch, capsys):
    monkeypatch.setattr(defaults, "gpu_limit_gb", lambda: 6.0)
    monkeypatch.setattr(defaults, "cached_snapshot", lambda model, allow_patterns: None)
    monkeypatch.setattr(defaults, "chip", lambda: "Apple M9")
    monkeypatch.setattr(defaults, "memory_gb", lambda: 8.0)
    monkeypatch.setattr(app, "load_bench", lambda: {QWEN: {"locals": {"8": 300.0}, "mlx": {"8": 90.0}}})
    app.cmd_models(argparse.Namespace())
    out = capsys.readouterr().out
    lines = {ln.split()[0]: ln for ln in out.splitlines()[3:8]}
    assert "Apple M9 with 8 GB" in out
    assert "300 tps, measured here (mlx-lm 90)" in lines["qwen"]
    assert "too big" in lines["gemma"] and "fits" in lines["lfm"]  # 4.8 GB of weights + 2 GB > 6 GB
    assert "on an M2 Pro" in lines["lfm"]


_ONE_AT_A_TIME = threading.Lock()


class _Stream(BaseHTTPRequestHandler):
    """A chat server that streams 12 tokens per request, 10 ms apart, then its usage; the model "serial"
    answers one request at a time, as Ollama does for Qwen3.5."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["model"] == "broken":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"no")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        lock = _ONE_AT_A_TIME if body["model"] == "serial" else None
        if lock:
            lock.acquire()
        try:
            for i in range(12):
                time.sleep(0.01)
                self.wfile.write(f'data: {json.dumps({"choices": [{"delta": {"content": "t"}}]})}\n\n'.encode())
                self.wfile.flush()
        finally:
            if lock:
                lock.release()
        self.wfile.write(f'data: {json.dumps({"choices": [], "usage": {"completion_tokens": 12}})}\n\ndata: [DONE]\n\n'.encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def stream_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Stream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_bench_measures_requests_at_once(stream_server):
    one = bench.measure(stream_server, "m", 1, 12)
    four = bench.measure(stream_server, "m", 4, 12)
    assert one["tokens"] == [12] and four["tokens"] == [12] * 4
    # each request writes about 100 tokens a second; four at once write about four times that
    assert 40 < one["tokens_per_s"] < 110 and 2.5 < four["tokens_per_s"] / one["tokens_per_s"] < 4.5
    with pytest.raises(RuntimeError, match="HTTP 500"):
        bench.measure(stream_server, "broken", 2, 12)


def test_bench_does_not_credit_a_server_that_answers_one_at_a_time(stream_server):
    """Four requests a server answers one after another write no faster than one: adding up each
    request's own rate (the first version) counted them as four times as fast."""
    one = bench.measure(stream_server, "serial", 1, 12)
    four = bench.measure(stream_server, "serial", 4, 12)
    assert four["tokens_per_s"] < 1.3 * one["tokens_per_s"]
    assert four["ttft_max_s"] > 0.25  # the last one waited for the three before it


def test_cmd_bench_prints_the_comparison_and_saves_it(monkeypatch, capsys):
    monkeypatch.setattr(app, "fetch_model", lambda model: None)
    monkeypatch.setattr(app, "ensure_server", lambda args: None)
    speeds = {"locals": {1: 64.0, 8: 270.0}, "mlx": {1: 61.0, 8: 87.0}}
    monkeypatch.setattr(bench, "sweep", lambda base, model, sizes, max_tokens, extra: {
        n: {"tokens_per_s": speeds[model.split("/")[0]][n]} for n in sizes})
    app.cmd_bench(argparse.Namespace(agents=False, model=QWEN, port=8321, sizes="1,8", bench="locals,mlx",
                                     bench_max_tokens=256))
    out = capsys.readouterr().out
    assert "3.1x" in out and "1.0x" in out
    saved = json.loads((app.LOCALS_DIR / "bench.json").read_text())[QWEN]
    assert saved["locals"] == {"1": 64.0, "8": 270.0} and saved["mlx"]["8"] == 87.0
