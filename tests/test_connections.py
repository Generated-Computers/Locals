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
    bench.measure(stream_server, "m", 1, 12)  # (the first request on a cold CI runner was slow: four read as 5.4x one)
    one = bench.measure(stream_server, "m", 1, 12)
    four = bench.measure(stream_server, "m", 4, 12)
    assert one["tokens"] == [12] and four["tokens"] == [12] * 4
    # each request writes at most about 100 tokens a second (10 ms apart; a busy CI Mac sleeps longer: 22-30 there);
    # four at once write about four times what one does, where a server answering one at a time writes no more (below)
    assert 0 < one["tokens_per_s"] < 110 and 2.5 < four["tokens_per_s"] / one["tokens_per_s"] < 8
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
    monkeypatch.setattr(app, "server_alive", lambda port: None)  # no server before: the bench starts one ...
    stopped = []
    monkeypatch.setattr(app, "cmd_stop", lambda args: stopped.append(args.port))  # ... and stops it (never a real one)
    monkeypatch.setattr(defaults, "weights_gb", lambda model, remote=False: 2.8)
    monkeypatch.setattr(defaults, "gpu_limit_gb", lambda: 10.7)
    speeds = {"locals": {1: 64.0, 8: 270.0}, "mlx": {1: 61.0, 8: 87.0}}
    monkeypatch.setattr(bench, "sweep", lambda base, model, sizes, max_tokens, extra: {
        n: {"tokens_per_s": speeds[model.split("/")[0]][n]} for n in sizes})
    app.cmd_bench(argparse.Namespace(agents=False, model=QWEN, port=8321, sizes="1,8", bench="locals,mlx",
                                     bench_max_tokens=256))
    out = capsys.readouterr().out
    assert "3.1x" in out and "1.0x" in out
    saved = json.loads((app.LOCALS_DIR / "bench.json").read_text())[QWEN]
    assert saved["locals"] == {"1": 64.0, "8": 270.0} and saved["mlx"]["8"] == 87.0
    assert stopped == [8321]  # the stock engine's second copy of the weights is not left loaded


def test_bench_measures_locals_only_when_two_copies_do_not_fit(monkeypatch, capsys):
    monkeypatch.setattr(app, "fetch_model", lambda model: None)
    started = []
    monkeypatch.setattr(app, "ensure_server", lambda args: started.append(args.engines))
    monkeypatch.setattr(app, "server_alive", lambda port: {"status": "ok", "engines": {}})  # the everyday server runs
    monkeypatch.setattr(app, "cmd_stop", lambda args: pytest.fail("the user's own server must not be stopped"))
    monkeypatch.setattr(defaults, "weights_gb", lambda model, remote=False: 4.8)  # Gemma 4 on a 16 GB Mac
    monkeypatch.setattr(defaults, "gpu_limit_gb", lambda: 10.7)
    seen = []
    monkeypatch.setattr(bench, "sweep", lambda base, model, sizes, max_tokens, extra: seen.append(model) or {
        n: {"tokens_per_s": 100.0} for n in sizes})
    app.cmd_bench(argparse.Namespace(agents=False, model=QWEN, port=8321, sizes="1", bench="locals,mlx", bench_max_tokens=8))
    assert started == ["locals"] and seen == ["locals/qwen3.5-4b"]
    assert "do not fit" in capsys.readouterr().err


GEMMA = "mlx-community/gemma-4-e4b-it-4bit"


def test_disconnect_after_two_setups_without_a_config_of_the_users(monkeypatch):
    """OpenCode had no config; locals wrote one, then rewrote it (another model): the second write must not
    keep locals's own config as the user's, or the disconnect puts locals back."""
    monkeypatch.setattr(app, "opencode_bin", lambda: "/x/opencode")
    app.write_config(app.opencode_config(QWEN, 8321))
    app.write_config(app.opencode_config(GEMMA, 8321, default_agent="coder"))
    app.disconnect_opencode()
    cfg = json.loads(app.OPENCODE_CONFIG.read_text())
    assert "locals" not in (cfg.get("provider") or {}) and "model" not in cfg
    assert app.connection("opencode", 8321)[0] == "not connected"


def test_the_users_own_agents_of_the_same_names_are_kept(monkeypatch):
    mine = {"model": "anthropic/claude-x", "agent": {"reviewer": {"prompt": "MY OWN REVIEWER", "mode": "subagent"}}}
    _opencode(mine)
    app.write_config(app.opencode_config(QWEN, 8321))
    assert json.loads(app.OPENCODE_CONFIG.read_text())["agent"]["reviewer"]["prompt"] == "MY OWN REVIEWER"
    app.write_config(app.opencode_config(GEMMA, 8321))  # again: still the user's
    assert json.loads(app.OPENCODE_CONFIG.read_text())["agent"]["reviewer"]["prompt"] == "MY OWN REVIEWER"
    app.disconnect_opencode()
    assert json.loads(app.OPENCODE_CONFIG.read_text()) == mine


def test_the_users_reviewer_overwritten_by_an_older_locals_comes_back(monkeypatch):
    mine = {"agent": {"reviewer": {"prompt": "MY OWN REVIEWER"}, "build": {"steps": 12}}}
    _opencode(mine)
    app._back_up_once(app.OPENCODE_CONFIG, "opencode", ours=False)  # what locals 0.2.0 kept
    _opencode({"agent": {"reviewer": app.AGENTS["reviewer"], "build": {"steps": app.AGENT_STEPS}},
               "provider": {"locals": {}}, "model": "locals/locals/qwen3.5-4b"})  # what it wrote
    app.disconnect_opencode()
    assert json.loads(app.OPENCODE_CONFIG.read_text()) == mine


def test_the_stock_engine_once_does_not_become_opencodes_default(monkeypatch):
    monkeypatch.setattr(app, "opencode_bin", lambda: "/x/opencode")
    app.write_config(app.opencode_config(QWEN, 8321, "mlx"))  # locals code --engine mlx
    cfg = json.loads(app.OPENCODE_CONFIG.read_text())
    assert cfg["model"] == cfg["small_model"] == "locals/locals/qwen3.5-4b" and "mlx" in cfg["provider"]
    _opencode({**cfg, "model": "mlx/mlx/qwen3.5-4b", "small_model": "mlx/mlx/qwen3.5-4b"})  # what 0.2.0 wrote
    assert not app._opencode_ready(QWEN, 8321)  # the next plain locals code rewrites it


def test_a_backup_of_stcons_settings_puts_no_stcon_model_back(monkeypatch):
    _opencode({"provider": {"stcon": {"options": {"baseURL": "http://127.0.0.1:8321/v1"}}, "anthropic": {}},
               "model": "stcon/stcon/qwen3.5-4b", "small_model": "stcon/stcon/qwen3.5-4b", "theme": "dark"})
    app._back_up_once(app.OPENCODE_CONFIG, "opencode", ours=False)  # kept by an earlier locals
    app.write_config(app.opencode_config(QWEN, 8321))
    app.disconnect_opencode()
    cfg = json.loads(app.OPENCODE_CONFIG.read_text())
    assert cfg == {"provider": {"anthropic": {}}, "theme": "dark"}


def test_a_reconnect_after_a_disconnect_keeps_the_config_of_then(monkeypatch):
    _opencode({"model": "openai/gpt-old"})
    app.write_config(app.opencode_config(QWEN, 8321))
    app.disconnect_opencode()
    _opencode({"model": "anthropic/claude-new"})  # the user's choice since
    app.write_config(app.opencode_config(QWEN, 8321))
    app.disconnect_opencode()
    assert json.loads(app.OPENCODE_CONFIG.read_text()) == {"model": "anthropic/claude-new"}


def test_hermes_without_a_config_at_first_gets_no_settings_of_locals_back(tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ours = {k.split(".", 1)[1]: v for k, v in app.hermes_settings(QWEN, 8321).items()}
    app._back_up_once(app._hermes_config(), "hermes")  # the first launch: Hermes had no config.yaml
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"model": ours}))  # hermes config set wrote it
    app._back_up_once(app._hermes_config(), "hermes")  # the second launch
    monkeypatch.setattr(app, "_tool_bin", lambda name, env: "/opt/hermes")
    seen = _record_runs(monkeypatch)
    app.disconnect_hermes()
    assert all(c[2] == "unset" for c in seen), seen  # nothing of locals's set back


def test_openclaw_json5_config_is_read(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(tmp_path))
    (tmp_path / "openclaw.json").write_text("""{
  // OpenClaw writes JSON5
  models: {providers: {locals: {baseUrl: 'http://127.0.0.1:8321/v1',},},},
  agents: {defaults: {model: {primary: "locals/locals/qwen3.5-4b"}}},
}""")
    monkeypatch.setattr(app, "_installed", lambda name, env: "/opt/openclaw")
    assert app.connection("openclaw", 8321) == ("connected", "default model locals/locals/qwen3.5-4b")


def test_a_foreign_health_endpoint_is_not_a_locals_server(monkeypatch):
    monkeypatch.setattr(app, "get_json", lambda url, timeout=5.0: {"status": "ok"})  # a dev server's /health
    assert app.server_alive(8321) is None and not app._answers_as_locals(8321)
    monkeypatch.setattr(app, "get_json", lambda url, timeout=5.0: {"status": "ok", "engines": {"locals": True}, "build": "x"})
    assert app.server_alive(8321)


def test_locals_code_refuses_the_home_folder(monkeypatch):
    monkeypatch.setattr(app, "fetch_model", lambda model: pytest.fail("nothing starts"))
    for d in (str(app.HOME), "/"):
        with pytest.raises(SystemExit) as e:
            app.cmd_code(argparse.Namespace(dir=d, model=QWEN, port=8321, engine=None, agent=None, run=None, cont=False, logs=False))
        assert "project folder" in str(e.value)


def test_a_launch_that_finds_the_tool_set_up_runs_none_of_its_commands(tmp_path, monkeypatch):
    """locals launch hermes / openclaw ran five `hermes config set` (17 s) or `openclaw config patch` (4 s) on
    every launch: none when the config already holds these settings."""
    yaml = pytest.importorskip("yaml")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(tmp_path / "openclaw"))
    (tmp_path / "hermes").mkdir()
    (tmp_path / "openclaw").mkdir()
    monkeypatch.setattr(app, "_tool_bin", lambda name, env: f"/opt/{name}")
    args = argparse.Namespace(model=QWEN, port=8321, context=None, output=None)
    monkeypatch.setattr(app, "agent_context", lambda model, args=None: 65536)
    ours = {k.split(".", 1)[1]: v for k, v in app.hermes_settings(QWEN, 8321).items()}
    (tmp_path / "hermes" / "config.yaml").write_text(yaml.safe_dump({"model": {**ours, "context_length": 65536}}))
    patch = app.openclaw_patch(QWEN, 8321)
    (tmp_path / "openclaw" / "openclaw.json").write_text(json.dumps({**patch, "gateway": {"mode": "local"}}))
    seen = _record_runs(monkeypatch)
    assert app.setup_hermes(args) and app.setup_openclaw(args)
    assert seen == []
    args.port = 8399  # another port: written again
    app.setup_hermes(args)
    app.setup_openclaw(args)
    assert any(c[:3] == ["/opt/hermes", "config", "set"] for c in seen) and any(c[1:3] == ["config", "patch"] for c in seen)


def test_a_weak_agent_model_gets_a_note():
    gemma, qwen = defaults.MODELS["gemma-4-e4b"], defaults.MODELS["qwen3.5-4b"]
    assert "--model qwen" in defaults.agent_note(gemma) and defaults.agent_note(qwen) == ""
    assert defaults.agent_note("someone/unknown-model") == ""


def test_launch_aider_uses_search_replace_edits(monkeypatch):
    """locals launch aider: diff edits (in the whole-file format a sentence before quoted code became a file
    name), and the user's own --edit-format, given after, still wins."""
    import argparse

    import pytest

    from locals import app

    seen = {}
    monkeypatch.setattr(app, "_installed", lambda name, env: "/opt/aider")
    monkeypatch.setattr(app, "fetch_model", lambda model: None)
    monkeypatch.setattr(app, "ensure_server", lambda args: None)

    def execve(exe, argv, env):  # (the real one never returns)
        seen.update(argv=argv, env=env)
        raise SystemExit(0)

    monkeypatch.setattr(app.os, "execve", execve)
    with pytest.raises(SystemExit):
        app.cmd_launch(argparse.Namespace(tool="aider", rest=["--edit-format", "whole", "a.py"], port=8321,
                                          model="mlx-community/Qwen3.5-4B-MLX-4bit", ollama_port=None))
    argv = seen["argv"]
    assert argv[:5] == ["/opt/aider", "--model", "openai/locals/qwen3.5-4b", "--no-show-model-warnings", "--edit-format"]
    assert argv[5] == "diff" and argv[-3:] == ["--edit-format", "whole", "a.py"]  # (the user's own come last, and win)
    assert seen["env"]["OPENAI_API_BASE"] == "http://127.0.0.1:8321/v1"


def test_a_users_own_mlx_provider_is_theirs():
    """An OpenCode provider of the user's called "mlx" (another MLX server) was taken for locals's: no backup, and a
    disconnect deleted it. It is kept now, and set back where `--engine mlx` replaced it."""
    mine = {"provider": {"mlx": {"npm": "@ai-sdk/openai-compatible", "name": "my mlx_lm.server",
                                 "options": {"baseURL": "http://localhost:8080/v1"}, "models": {"qwen-coder": {}}}},
            "model": "mlx/qwen-coder"}
    _opencode(mine)
    assert not app._opencode_has_locals(mine)
    app.write_config(app.opencode_config(QWEN, 8321))
    cfg = json.loads(app.OPENCODE_CONFIG.read_text())
    assert cfg["provider"]["mlx"] == mine["provider"]["mlx"] and cfg["model"] == "locals/locals/qwen3.5-4b"
    app.disconnect_opencode()
    assert json.loads(app.OPENCODE_CONFIG.read_text()) == mine
    app.write_config(app.opencode_config(QWEN, 8321, default_engine="mlx"))  # locals's mlx provider replaces it
    assert json.loads(app.OPENCODE_CONFIG.read_text())["provider"]["mlx"]["name"] == app.PROVIDERS["mlx"][0]
    changed = app.disconnect_opencode()
    assert "your mlx provider, back as it was" in changed
    assert json.loads(app.OPENCODE_CONFIG.read_text()) == mine


def test_hermes_on_another_local_server_is_set_back(tmp_path, monkeypatch):
    """Hermes set up for LM Studio (127.0.0.1 too) before locals: a disconnect sets those settings back instead of
    taking them for locals's and unsetting them (and the backup is not lost before that)."""
    yaml = pytest.importorskip("yaml")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    lmstudio = {"provider": "custom", "base_url": "http://127.0.0.1:1234/v1", "default": "qwen2.5-coder-7b"}
    ours = {k.split(".", 1)[1]: v for k, v in app.hermes_settings(QWEN, 8321).items()}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"model": lmstudio}))
    app._back_up_once(app._hermes_config(), "hermes")  # what a launch does before `hermes config set`
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"model": ours}))
    monkeypatch.setattr(app, "_tool_bin", lambda name, env: "/opt/hermes")
    seen = _record_runs(monkeypatch)
    app.disconnect_hermes()
    assert ["/opt/hermes", "config", "set", "model.base_url", "http://127.0.0.1:1234/v1"] in seen
    assert ["/opt/hermes", "config", "set", "model.default", "qwen2.5-coder-7b"] in seen


@pytest.mark.parametrize("text,want", [
    ('{"note": "Hello, world: hi", a: 1,}', {"note": "Hello, world: hi", "a": 1}),
    ("{a: 1, b: '{c: 2}'}", {"a": 1, "b": "{c: 2}"}),
    ('["a,]", 2,]', ["a,]", 2]),
    ("{x: 'it\\'s', y: 'say \"hi\"'}", {"x": "it's", "y": 'say "hi"'}),
    ('{\n // c\n u: "http://x/y", /* b */ v: [1,2,],\n}', {"u": "http://x/y", "v": [1, 2]}),
])
def test_json5_strings_are_left_as_they_are(text, want):
    """OpenClaw's JSON5: commas, colons and braces inside strings are text (a regular expression over the whole file
    once broke "Hello, world: hi"; then the config could not be read, and a disconnect refused)."""
    assert app._loads_jsonish(text) == want


def test_openclaw_with_a_model_written_as_a_string(tmp_path, monkeypatch):
    """OpenClaw's agents.defaults.model may be the reference itself, a string: `locals launch`, setup and
    disconnect ended with an AttributeError on it."""
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(tmp_path))
    (tmp_path / "openclaw.json").write_text(json.dumps({"agents": {"defaults": {"model": "anthropic/claude-x"}}}))
    monkeypatch.setattr(app, "_tool_bin", lambda name, env: "/opt/openclaw")
    monkeypatch.setattr(app, "_installed", lambda name, env: "/opt/openclaw")
    assert app.connection("openclaw", 8321)[0] == "not connected"
    assert app._openclaw_primary({"agents": {"defaults": {"model": "a/b"}}}) == "a/b"
    assert app._openclaw_primary({"agents": {"defaults": {"model": {"primary": "c/d"}}}}) == "c/d"
    assert app._openclaw_primary({}) == "" and app._openclaw_primary({"agents": {"defaults": {"model": 3}}}) == ""
    assert "mode" not in app.openclaw_patch(QWEN, 8321)["models"]  # (a user's "replace" stays theirs)


def test_disconnect_hermes_puts_back_a_model_written_as_a_string(tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ours = {k.split(".", 1)[1]: v for k, v in app.hermes_settings(QWEN, 8321).items()}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"model": ours}))
    keep = app.LOCALS_DIR / "backups" / "hermes-config.yaml.before-locals"
    keep.parent.mkdir(parents=True)
    keep.write_text(yaml.safe_dump({"model": "anthropic/claude-sonnet-4"}))  # (Hermes's shorthand)
    monkeypatch.setattr(app, "_tool_bin", lambda name, env: "/opt/hermes")
    monkeypatch.setattr(app, "_installed", lambda name, env: "/opt/hermes")
    seen = _record_runs(monkeypatch)
    app.disconnect_hermes()
    assert ["/opt/hermes", "config", "set", "model.default", "anthropic/claude-sonnet-4"] in seen  # (it was unset: no model)


def test_a_tool_set_up_for_another_model_needs_repair(monkeypatch):
    """Since the server refuses our names for another model, a tool still set up for the old one would get a 400
    on every request: it is shown as needing repair, not as connected."""
    monkeypatch.setattr(app, "opencode_bin", lambda: "/x/opencode")
    app.write_config(app.opencode_config(QWEN, 8321))
    monkeypatch.setattr(app, "_serving_short", lambda port: "gemma-4-e4b")
    st, why = app.connection("opencode", 8321)
    assert st == "needs repair" and "gemma-4-e4b" in why and "locals code" in why
    monkeypatch.setattr(app, "_serving_short", lambda port: "qwen3.5-4b")
    assert app.connection("opencode", 8321)[0] == "connected"


def test_opencode_config_keeps_the_users_own_text(monkeypatch):
    _opencode({"agent": {"ревьюер": {"description": "Мой ревьюер 🔍", "mode": "subagent"}}})
    app.write_config(app.opencode_config(QWEN, 8321))
    text = app.OPENCODE_CONFIG.read_text()
    assert "Мой ревьюер 🔍" in text and "\\u" not in text


def test_a_config_link_into_a_read_only_folder_is_a_sentence(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "opencode.json").write_text("{}")
    store.chmod(0o555)
    link = tmp_path / "opencode.json"
    link.symlink_to(store / "opencode.json")
    try:
        with pytest.raises(SystemExit, match="cannot write .*store.*where .* points"):
            app._write_atomic(link, "{}")
    finally:
        store.chmod(0o755)


def test_the_context_told_to_tools_is_the_servers_limit_at_most(monkeypatch):
    monkeypatch.setattr(app, "get_json", lambda url, timeout=5: (_ for _ in ()).throw(OSError("no server")))
    assert app.agent_context(QWEN) == 65536
    assert app.agent_context(QWEN, argparse.Namespace(max_context=16384, port=8321)) == 16384
    monkeypatch.setattr(app, "get_json", lambda url, timeout=5: {"data": [{"id": "locals/qwen3.5-4b", "context_length": 20480}]})
    assert app.agent_context(QWEN, argparse.Namespace(max_context=None, port=8321)) == 20480
    # a server running another model (about to be switched) says nothing about this one
    monkeypatch.setattr(app, "get_json", lambda url, timeout=5: {"data": [{"id": "locals/gemma-4-e4b", "context_length": 8192}]})
    assert app.agent_context(QWEN, argparse.Namespace(max_context=None, port=8321)) == 65536


def test_aider_gets_the_base_url_over_the_users_own(monkeypatch, tmp_path):
    """Aider reads ~/.env and the project's .env over the environment, and litellm reads OPENAI_BASE_URL first: a
    base URL of the user's won, and the request for locals/<short> went there."""
    calls = []
    monkeypatch.setattr(app.os, "execve", lambda exe, argv, env: calls.append((argv, env)) or (_ for _ in ()).throw(SystemExit(0)))
    monkeypatch.setattr(app, "fetch_model", lambda m: None)
    monkeypatch.setattr(app, "ensure_server", lambda a: None)
    monkeypatch.setattr(app, "_installed", lambda name, env: "/opt/aider")
    monkeypatch.setattr(app, "_tool_bin", lambda name, env: "/opt/aider")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    with pytest.raises(SystemExit):
        app.main(["launch", "--model", QWEN, "aider"])
    argv, env = calls[0]
    assert env["OPENAI_BASE_URL"] == env["OPENAI_API_BASE"] == "http://127.0.0.1:8321/v1"
    sets = [argv[i + 1] for i, a in enumerate(argv) if a == "--set-env"]
    assert "OPENAI_BASE_URL=http://127.0.0.1:8321/v1" in sets and "OPENAI_API_BASE=http://127.0.0.1:8321/v1" in sets



def test_openclaws_config_is_found_as_openclaw_finds_it(tmp_path, monkeypatch):
    for k in ("OPENCLAW_CONFIG_PATH", "OPENCLAW_STATE_DIR", "OPENCLAW_HOME"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(app, "HOME", tmp_path)
    assert app._openclaw_config() == tmp_path / ".openclaw" / "openclaw.json"
    monkeypatch.setenv("OPENCLAW_HOME", str(tmp_path / "oh"))
    assert app._openclaw_config() == tmp_path / "oh" / ".openclaw" / "openclaw.json"
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(tmp_path / "state"))
    assert app._openclaw_config() == tmp_path / "state" / "openclaw.json"
    monkeypatch.setenv("OPENCLAW_CONFIG_PATH", str(tmp_path / "elsewhere.json"))
    assert app._openclaw_config() == tmp_path / "elsewhere.json"


def test_a_jsonc_opencode_config_says_why_it_cannot_be_read(monkeypatch):
    monkeypatch.setattr(app, "opencode_bin", lambda: "/x/opencode")
    _opencode('{"model": "x", // a comment\n}')
    st, why = app.connection("opencode", 8321)
    assert st == "needs repair" and "comments" in why
