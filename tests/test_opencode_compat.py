"""OpenCode launch contracts, including v2's full TUI without a model flag."""

import json
import subprocess

import pytest

from locals import app


QWEN = "mlx-community/Qwen3.5-4B-MLX-4bit"
V1_HELP = "FLAGS\n  --model, -m string Model to use\n  --continue, -c\n"
V2_HELP = "FLAGS\n  --standalone Run with a private server\n  --continue, -c\n"


@pytest.mark.parametrize("help_text, expected", [(V1_HELP, False), (V2_HELP, True), (V1_HELP + V2_HELP, True)])
@pytest.mark.parametrize("stderr", [False, True])
def test_detects_tui_capabilities(monkeypatch, help_text, expected, stderr):
    def run(cmd, **kw):
        assert cmd == ["/opt/opencode", "--help"]
        assert kw["timeout"] <= 10
        return subprocess.CompletedProcess(cmd, 0, "" if stderr else help_text, help_text if stderr else "")

    monkeypatch.setattr(app.subprocess, "run", run)
    assert app._opencode_v2("/opt/opencode") is expected


@pytest.mark.parametrize("failure", ["unknown", "exit", "timeout", "missing"])
def test_unknown_or_broken_cli_fails_before_starting_a_model(monkeypatch, failure):
    def run(cmd, **kw):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, 10)
        if failure == "missing":
            raise FileNotFoundError(cmd[0])
        return subprocess.CompletedProcess(cmd, int(failure == "exit"), "unknown CLI", "")

    monkeypatch.setattr(app.subprocess, "run", run)
    with pytest.raises(SystemExit, match="OpenCode"):
        app._opencode_v2("/opt/opencode")


def test_v2_overlay_preserves_local_models_agents_and_permissions():
    cfg = app.opencode_config(QWEN, 8399, "mlx")
    overlay = app._opencode_v2_config(cfg, "mlx/mlx/qwen3.5-4b")
    assert overlay["model"] == {"providerID": "mlx", "model": "mlx/qwen3.5-4b"}
    assert overlay["default_agent"] == "orchestrator"
    assert overlay["experimental"]["subagent_depth"] == 1
    assert overlay["agents"]["orchestrator"]["model"] == overlay["model"]
    for pid in ("locals", "mlx"):
        provider = overlay["providers"][pid]
        assert provider["settings"]["baseURL"] == "http://127.0.0.1:8399/v1"
        assert provider["settings"]["apiKey"] == "local"
        for name, model in provider["models"].items():
            assert name.startswith(pid + "/")
            assert model["capabilities"]["tools"] is True
            assert model["capabilities"]["reasoning"]["supported"] is name.endswith("-thinking")
            assert model["limit"] == {"context": 32768, "output": 4096}
    orchestrator = overlay["agents"]["orchestrator"]
    assert orchestrator["steps"] == app.AGENT_STEPS
    assert "subagent tool" in orchestrator["system"] and "agent: coder" in orchestrator["system"]
    permissions = {p["action"]: p["effect"] for p in orchestrator["permissions"]}
    assert permissions["shell"] == permissions["edit"] == permissions["subagent"] == "allow"
    assert permissions["external_directory"] == permissions["question"] == "deny"
    reviewer = overlay["agents"]["reviewer"]["permissions"]
    assert {"action": "edit", "resource": "*", "effect": "deny"} in reviewer
    assert {"action": "shell", "resource": "*", "effect": "deny"} in reviewer
    assert {"action": "shell", "resource": "git diff*", "effect": "allow"} in reviewer
    assert cfg["model"] == "locals/locals/qwen3.5-4b"  # overlay never persists the comparison engine


def test_v2_overlay_keeps_user_agents_and_inherited_environment(monkeypatch):
    cfg = app.opencode_config(QWEN, 8321)
    cfg["agent"]["coder"] = {"prompt": "My coder", "steps": 12}
    inherited = {"providers": {"other": {"name": "mine"}}, "agents": {"other": {"system": "mine"}},
                 "experimental": {"todo": True}, "model": "cloud/model"}
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps(inherited))
    overlay = app._opencode_v2_config(cfg, "locals/locals/qwen3.5-4b")
    assert overlay["providers"]["other"] == inherited["providers"]["other"]
    assert overlay["agents"]["other"] == inherited["agents"]["other"]
    assert overlay["experimental"] == {"todo": True, "subagent_depth": 1}
    assert "coder" not in overlay["agents"]  # OpenCode migrates the user's definition from disk
    assert overlay["model"] == {"providerID": "locals", "model": "locals/qwen3.5-4b"}


def test_v2_native_user_agent_definitions_are_kept(monkeypatch):
    cfg = app.opencode_config(QWEN, 8321)
    cfg["agents"] = {"coder": {"system": "My native coder"}}
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps({
        "agents": {"orchestrator": {"system": "My native orchestrator", "steps": 9}},
        "agent": {"researcher": {"prompt": "My inherited v1 researcher"}}}))
    overlay = app._opencode_v2_config(cfg, "locals/locals/qwen3.5-4b")
    assert "coder" not in overlay["agents"]  # existing native definition comes from disk
    assert "researcher" not in overlay["agents"]
    assert overlay["agents"]["orchestrator"]["system"] == "My native orchestrator"
    assert overlay["agents"]["orchestrator"]["steps"] == 9


def test_v2_inherited_jsonc_preserves_strings(monkeypatch):
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", '''{
        // These are user settings, in the same JSONC syntax OpenCode reads.
        "instructions": ["https://example.com/*literal*/", ",}", "quoted \\\"text\\\"",],
        "experimental": {"todo": true, /* comment before trailing comma */},
    }''')
    overlay = app._opencode_v2_config(app.opencode_config(QWEN, 8321), "locals/locals/qwen3.5-4b")
    assert overlay["instructions"] == ["https://example.com/*literal*/", ",}", 'quoted "text"']
    assert overlay["experimental"]["todo"] is True


@pytest.mark.parametrize("inherited", ["not json", "[]", "null"])
def test_invalid_environment_config_is_reported_not_discarded(monkeypatch, inherited):
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", inherited)
    with pytest.raises(SystemExit, match="OPENCODE_CONFIG_CONTENT"):
        app._opencode_v2_config(app.opencode_config(QWEN, 8321), "locals/locals/qwen3.5-4b")


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("headless", [False, True])
def test_code_launch_preserves_model_engine_continue_and_logs(tmp_path, monkeypatch, v2, headless):
    project = tmp_path / "project"
    project.mkdir()
    app.write_config(app.opencode_config(QWEN, 8321, "mlx"))
    before = app.OPENCODE_CONFIG.read_bytes()
    monkeypatch.setattr(app, "opencode_bin", lambda: "/opt/opencode")
    monkeypatch.setattr(app, "_opencode_v2", lambda oc: v2)
    monkeypatch.setattr(app, "fetch_model", lambda model: None)
    monkeypatch.setattr(app, "ensure_server", lambda args: None)
    monkeypatch.setattr(app, "_space_opencode_starts", lambda: None)
    monkeypatch.setattr(app, "_opencode_v2_continue", lambda oc, d, ref, env: ["--session", "ses_existing"])
    monkeypatch.chdir(tmp_path)
    seen = []

    def launch(cmd, env):
        seen.append((cmd, env))
        return 0

    monkeypatch.setattr(app, "_opencode_run", launch)
    monkeypatch.setattr(app.os, "execve", lambda oc, cmd, env: launch(cmd, env))
    argv = ["code", str(project), "--model", QWEN, "--engine", "mlx", "--continue", "--logs"]
    if headless:
        argv += ["--run", "write a test"]
        with pytest.raises(SystemExit) as e:
            app.main(argv)
        assert e.value.code == 0
    else:
        app.main(argv)
    cmd, env = seen[0]
    assert env["PWD"] == str(project) and env["OPENCODE_DISABLE_AUTOUPDATE"] == "1"
    assert ("--session" in cmd) if v2 and not headless else ("-c" in cmd)
    assert ("--standalone" in cmd) is v2
    assert ("-m" in cmd) is (headless or not v2)
    if "-m" in cmd:
        assert cmd[cmd.index("-m") + 1] == "mlx/mlx/qwen3.5-4b"
    if headless:
        assert cmd[1] == "run" and cmd[-1] == "write a test" and "json" in cmd
    else:
        assert str(project) in cmd and "--print-logs" in cmd
    if v2:
        assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["model"] == {"providerID": "mlx", "model": "mlx/qwen3.5-4b"}
    assert app.OPENCODE_CONFIG.read_bytes() == before


@pytest.mark.parametrize("exists", [False, True])
def test_v2_tui_continuation_updates_the_saved_session_model(monkeypatch, tmp_path, exists):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        assert kw["env"] == {"test": "env"} and kw["cwd"] == str(tmp_path)
        assert kw["timeout"] == 60
        result = {"data": [{"id": "ses_existing"}] if exists else []} if len(calls) == 1 else None
        return subprocess.CompletedProcess(cmd, 0, json.dumps(result), "")

    monkeypatch.setattr(app.subprocess, "run", run)
    flags = app._opencode_v2_continue("/opt/opencode", tmp_path, "mlx/mlx/qwen3.5-4b", {"test": "env"})
    assert flags == (["--session", "ses_existing"] if exists else [])
    assert "session.list" in calls[0] and "parentID=null" in calls[0]
    if exists:
        assert "session.switchModel" in calls[1] and "sessionID=ses_existing" in calls[1]
        assert json.loads(calls[1][-1]) == {"model": {"providerID": "mlx", "id": "mlx/qwen3.5-4b"}}
