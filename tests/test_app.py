"""The `locals` command's agent configs: the model ids they send must route to the intended engine."""

from locals import app, oai


def test_opencode_models_route_to_their_engine():
    cfg = app.opencode_config("mlx-community/Qwen3.5-4B-MLX-4bit", 8321)
    for pid, prov in cfg["provider"].items():
        for model_id in prov["models"]:
            # OpenCode sends only the model id (not the provider id) to the server
            assert oai.engine_of(model_id) == pid, (pid, model_id)
    provider, model_id = cfg["model"].split("/", 1)
    assert provider == "locals" and model_id in cfg["provider"]["locals"]["models"]
    assert app.oc_model("mlx", "qwen3.5-4b") == "mlx/mlx/qwen3.5-4b"


def test_openclaw_and_hermes_configs_route_to_locals():
    patch = app.openclaw_patch("mlx-community/Qwen3.5-4B-MLX-4bit", 8321)
    prov = patch["models"]["providers"]["locals"]
    model_id = prov["models"][0]["id"]
    assert oai.engine_of(model_id) == "locals" and prov["baseUrl"] == "http://127.0.0.1:8321/v1"
    assert patch["agents"]["defaults"]["model"]["primary"] == "locals/" + model_id
    assert prov["models"][0]["contextWindow"] >= 64000  # what Hermes demands; the server's default limit
    kv = app.hermes_settings("mlx-community/Qwen3.5-4B-MLX-4bit", 8321)
    assert oai.engine_of(kv["model.default"]) == "locals" and int(kv["model.context_length"]) >= 64000


def test_one_command_helpers(tmp_path, monkeypatch):
    """`locals code` sets OpenCode up when its config does not serve the model, and
    `--run` prints the agent's last answer from OpenCode's JSON events."""
    import json

    cfg_path = tmp_path / "opencode.json"
    monkeypatch.setattr(app, "OPENCODE_CONFIG", cfg_path)
    monkeypatch.setattr(app, "opencode_bin", lambda: "/usr/bin/true")
    assert not app._opencode_ready("mlx-community/Qwen3.5-4B-MLX-4bit")  # no config yet
    cfg_path.write_text(json.dumps(app.opencode_config("mlx-community/Qwen3.5-4B-MLX-4bit", 8321)))
    assert app._opencode_ready("mlx-community/Qwen3.5-4B-MLX-4bit")
    assert not app._opencode_ready("mlx-community/gemma-4-e4b-it-4bit")  # another model: set up again
    events = "\n".join([json.dumps({"type": "step_start"}), json.dumps({"type": "text", "part": {"type": "text", "text": "first"}}),
                        "not json", json.dumps({"type": "text", "part": {"type": "text", "text": "The output is 2 3 5."}})])
    assert app._opencode_answer(events) == "The output is 2 3 5."
    assert "error" in app._opencode_answer(json.dumps({"type": "error", "error": {"name": "UnknownError"}}))


def test_stop_without_a_server_touches_nothing(tmp_path, monkeypatch, capsys):
    import argparse

    monkeypatch.setattr(app, "LOCALS_DIR", tmp_path)
    app.cmd_stop(argparse.Namespace(port=8399))
    assert "no locals server on port 8399" in capsys.readouterr().out


def test_model_short_names(tmp_path, monkeypatch):
    """`--model gemma` (or qwen), before or after the sub-command, means the tested checkpoint."""
    from locals.defaults import model_id

    assert model_id("gemma") == model_id("Gemma4") == model_id("gemma-4-e4b") == "mlx-community/gemma-4-e4b-it-4bit"
    assert model_id("qwen") == model_id("qwen3.5-4b") == "mlx-community/Qwen3.5-4B-MLX-4bit"
    assert model_id("mlx-community/Llama-3.2-3B-Instruct-4bit") == "mlx-community/Llama-3.2-3B-Instruct-4bit"
    seen = []
    monkeypatch.setattr(app, "LOCALS_DIR", tmp_path / "locals")
    monkeypatch.setattr(app, "cmd_status", seen.append)
    app.main(["status", "--model", "gemma"])
    app.main(["--model", "qwen", "status"])
    assert [a.model for a in seen] == ["mlx-community/gemma-4-e4b-it-4bit", "mlx-community/Qwen3.5-4B-MLX-4bit"]
    (tmp_path / "gemma").mkdir()
    monkeypatch.chdir(tmp_path)
    assert model_id("gemma") == "mlx-community/gemma-4-e4b-it-4bit"  # a folder named gemma here does not hide the name
    assert model_id("./gemma") == str((tmp_path / "gemma").absolute())  # a path is a path, made absolute


def test_setup_keeps_the_users_opencode_config(tmp_path):
    """locals adds its providers to an existing OpenCode config and keeps the user's own file once."""
    import json

    p = tmp_path / "opencode.json"
    mine = {"model": "anthropic/claude-x", "provider": {"anthropic": {"options": {"apiKey": "k"}}}, "theme": "dark"}
    p.write_text(json.dumps(mine))
    app.write_config(app.opencode_config("mlx-community/gemma-4-e4b-it-4bit", 8321), p)
    merged = json.loads(p.read_text())
    assert merged["theme"] == "dark" and {"anthropic", "locals"} <= set(merged["provider"])  # the stock engine only on request
    assert merged["model"] == "locals/locals/gemma-4-e4b-it"
    app.write_config(app.opencode_config("mlx-community/Qwen3.5-4B-MLX-4bit", 8321), p)  # a later setup keeps the first copy
    assert json.loads((tmp_path / "opencode.json.before-locals").read_text()) == mine
    # the user's own settings of OpenCode's build agent stay; every agent gets the step limit
    p.write_text(json.dumps(dict(mine, agent={"build": {"model": "anthropic/claude-x", "temperature": 0.1}})))
    app.write_config(app.opencode_config("mlx-community/Qwen3.5-4B-MLX-4bit", 8321), p)
    agents = json.loads(p.read_text())["agent"]
    assert agents["build"] == {"model": "anthropic/claude-x", "temperature": 0.1, "steps": app.AGENT_STEPS}
    assert all(a["steps"] == app.AGENT_STEPS for a in agents.values())


def test_switching_models_restarts_only_our_idle_background_server(tmp_path, monkeypatch):
    """`locals code --model gemma` with a Qwen server up: the background server locals
    started is restarted with Gemma when idle; a busy one, or one started by hand, is left alone."""
    import argparse

    import pytest

    monkeypatch.setattr(app, "LOCALS_DIR", tmp_path)
    monkeypatch.setattr(app, "server_alive", lambda port: {"ok": True})
    stats = {"model": "mlx-community/Qwen3.5-4B-MLX-4bit", "active": []}
    monkeypatch.setattr(app, "get_json", lambda url, timeout=5.0: stats)
    monkeypatch.setattr(app, "_listener_pid", lambda port: 4242)
    args = argparse.Namespace(port=8399, model="mlx-community/gemma-4-e4b-it-4bit")
    with pytest.raises(SystemExit, match="already running"):  # not started by locals
        app.ensure_server(args)
    (tmp_path / "server-8399.pid").write_text("4242")
    stats["active"] = [{"engine": "locals"}]
    with pytest.raises(SystemExit, match="busy"):
        app.ensure_server(args)
    stats["active"] = []
    killed = []

    def kill(pid, sig):
        killed.append(pid)
        monkeypatch.setattr(app, "_listener_pid", lambda port: None)

    class Started(Exception):
        pass

    def popen(*a, **k):
        raise Started

    monkeypatch.setattr(app.os, "kill", kill)
    monkeypatch.setattr(app.subprocess, "Popen", popen)
    with pytest.raises(Started):  # stopped, then a new server is started for Gemma
        app.ensure_server(args)
    assert killed == [4242]
    app.ensure_server(argparse.Namespace(port=8399, model="mlx-community/Qwen3.5-4B-MLX-4bit"))  # same model: nothing to do
    killed.clear()
    stats["model"] = "mlx-community/Qwen3.5-4B-MLX-4bit"
    monkeypatch.setattr(app, "_listener_pid", lambda port: 4242)
    with pytest.raises(Started):  # a fine-tune with the same short name is another model: it is loaded
        app.ensure_server(argparse.Namespace(port=8399, model="/Users/x/ft/Qwen3.5-4B-MLX-4bit"))
    assert killed == [4242]


def test_agent_setup_without_the_agent_installed(tmp_path, monkeypatch, capsys):
    """`locals setup openclaw` / `hermes` without the tool: say how to install it and
    stop there (no download, no server, no "Next: openclaw")."""
    import argparse

    monkeypatch.setattr(app, "LOCALS_DIR", tmp_path)
    monkeypatch.setattr(app, "_tool_bin", lambda name, env: None)
    monkeypatch.setattr(app, "agent_context", lambda model, args=None: 65536)

    def boom(*a, **k):
        raise AssertionError("must not start anything")

    monkeypatch.setattr(app, "fetch_model", boom)
    monkeypatch.setattr(app, "ensure_server", boom)
    monkeypatch.setattr(app, "check_model", lambda model: None)  # (checks the name on Hugging Face; downloads nothing)
    import pytest

    for target in ("openclaw", "hermes"):
        with pytest.raises(SystemExit) as e:  # status 1: a script took 0 for a finished setup
            app.cmd_setup(argparse.Namespace(target=target, model="mlx-community/gemma-4-e4b-it-4bit", port=8321,
                                             context=None, output=4096, no_start=False))
        assert e.value.code == 1
        out = capsys.readouterr().out
        assert "is not installed" in out and f"locals setup {target}" in out and "Next:" not in out


def test_code_run_end_to_end_with_stand_ins(tmp_path, monkeypatch, capsys):
    """`locals code DIR --run TASK` on a first run: OpenCode set up for the model, the model
    fetched, the server ensured, the task run headless and its answer printed."""
    import argparse
    import json
    import subprocess

    import pytest

    cfg_path = tmp_path / "oc" / "opencode.json"
    monkeypatch.setattr(app, "OPENCODE_CONFIG", cfg_path)
    monkeypatch.setattr(app, "LOCALS_DIR", tmp_path / "locals")
    monkeypatch.setattr(app, "opencode_bin", lambda: "/bin/echo")
    monkeypatch.setattr(app, "default_agent", lambda model: "coder")
    monkeypatch.setattr(app, "model_thinks", lambda model: True)
    monkeypatch.setattr(app, "agent_context", lambda model, args=None: 65536)
    calls = []
    monkeypatch.setattr(app, "fetch_model", lambda model: calls.append(("fetch", model)))
    monkeypatch.setattr(app, "ensure_server", lambda args: calls.append(("server", args.model)))
    real_run = subprocess.run

    def run(cmd, **kw):
        if cmd[0] == "/bin/echo" and "--version" in cmd:
            return real_run(["echo", "1.18.32"], **kw)
        calls.append(("opencode", cmd[1:]))
        ev = json.dumps({"type": "text", "part": {"type": "text", "text": "Done: 2 3 5 7"}})
        return subprocess.CompletedProcess(cmd, 0, ev + "\n", "")

    monkeypatch.setattr(app.subprocess, "run", run)

    class Proc:  # `opencode run`: its events streamed (a tool call shown as it happens), then the answer
        def __init__(self, cmd, **kw):
            calls.append(("opencode", cmd[1:], kw.get("stdin")))
            import io

            tool = {"type": "tool_use", "part": {"type": "tool", "tool": "bash",
                                                 "state": {"status": "completed", "input": {"command": "python3 primes.py"}}}}
            self.stdout = io.StringIO(json.dumps(tool) + "\n" + json.dumps(
                {"type": "text", "part": {"type": "text", "text": "Done: 2 3 5 7"}}) + "\n")
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self):
            return 0

    real_popen = subprocess.Popen
    monkeypatch.setattr(app.subprocess, "Popen", lambda cmd, **kw: Proc(cmd, **kw) if "--format" in cmd else real_popen(cmd, **kw))
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(dir=str(project), model="mlx-community/gemma-4-e4b-it-4bit", port=8321, engine=None, agent=None,
                              logs=False, run="write primes.py", cont=False)
    with pytest.raises(SystemExit) as e:
        app.cmd_code(args)
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "Done: 2 3 5 7" in out and "gemma-4-e4b-it on the locals engine, agent coder" in out
    assert "bash: python3 primes.py" in out
    assert calls[0] == ("fetch", args.model) and calls[1] == ("server", args.model) and calls[2][0] == "opencode"
    # the model on OpenCode's command line (no project setting can swap it), and no stdin to wait on
    assert calls[2][1][:4] == ["run", "--format", "json", "-m"] and calls[2][1][4] == "locals/locals/gemma-4-e4b-it"
    assert calls[2][2] == subprocess.DEVNULL
    cfg = json.loads(cfg_path.read_text())
    assert cfg["model"] == "locals/locals/gemma-4-e4b-it" and cfg["default_agent"] == "coder"
    assert json.loads((tmp_path / "locals" / "config.json").read_text())["model"] == args.model


def test_openclaw_setup_replaces_localss_model_list(tmp_path, monkeypatch, capsys):
    """A setup for another model replaces the locals provider's model list: OpenClaw refuses
    a patch that would drop entries (switching from Qwen3.5 to Gemma 4 failed)."""
    import argparse
    import subprocess

    monkeypatch.setattr(app, "_tool_bin", lambda name, env: "/opt/openclaw")
    monkeypatch.setattr(app, "agent_context", lambda model, args=None: 65536)
    seen = []
    monkeypatch.setattr(app.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    assert app.setup_openclaw(argparse.Namespace(model="mlx-community/gemma-4-e4b-it-4bit", port=8321, context=None, output=4096))
    cmd = seen[0]
    assert cmd[:3] == ["/opt/openclaw", "config", "patch"]
    assert cmd[cmd.index("--replace-path") + 1] == "models.providers.locals.models"
    assert "locals/locals/gemma-4-e4b-it" in capsys.readouterr().out


def test_a_background_server_on_older_code_is_restarted(tmp_path, monkeypatch):
    """The same model, but the server started before locals changed (an upgrade): ours and idle,
    it is restarted; started by hand, it is used with a note."""
    import argparse

    import pytest

    from locals import defaults

    monkeypatch.setattr(app, "LOCALS_DIR", tmp_path)
    monkeypatch.setattr(app, "server_alive", lambda port: {"status": "ok", "build": "old-build"})
    monkeypatch.setattr(app, "get_json", lambda url, timeout=5.0: {"model": "mlx-community/Qwen3.5-4B-MLX-4bit", "active": []})
    monkeypatch.setattr(defaults, "build_id", lambda: "new-build")
    monkeypatch.setattr(app, "_listener_pid", lambda port: 4242)
    args = argparse.Namespace(port=8399, model="mlx-community/Qwen3.5-4B-MLX-4bit")
    app.ensure_server(args)  # not ours (no pid file): used as it is
    (tmp_path / "server-8399.pid").write_text("4242")
    killed = []

    def kill(pid, sig):
        killed.append(pid)
        monkeypatch.setattr(app, "_listener_pid", lambda port: None)

    class Started(Exception):
        pass

    monkeypatch.setattr(app.os, "kill", kill)
    monkeypatch.setattr(app.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(Started()))
    with pytest.raises(Started):
        app.ensure_server(args)
    assert killed == [4242]


def test_stop_never_kills_what_a_stale_pid_file_names(tmp_path, monkeypatch, capsys):
    """A pid file can outlive its server; its number may then be anyone's process (this test's own here)."""
    import os

    from locals import app

    monkeypatch.setattr(app, "_listener_pid", lambda port: None)
    app.LOCALS_DIR.mkdir(parents=True, exist_ok=True)
    app._pid_file(8321).write_text(str(os.getpid()))
    killed = []
    monkeypatch.setattr(os, "kill", lambda *a: killed.append(a))
    app.cmd_stop(type("A", (), {"port": 8321})())
    assert not killed and "no locals server" in capsys.readouterr().out


def test_a_commented_opencode_config_is_not_rewritten(tmp_path):
    import pytest

    from locals import app

    path = tmp_path / "opencode.json"
    text = '{\n  // my providers\n  "provider": {"mine": {}},\n}\n'
    path.write_text(text)
    with pytest.raises(SystemExit):
        app.write_config({"model": "locals/locals/x"}, path)
    assert path.read_text() == text


def test_model_before_a_command_with_its_own_parser(monkeypatch):
    from locals import app

    seen = []
    monkeypatch.setattr(app, "_delegate", lambda name: (lambda argv: seen.append((name, argv))) if name == "chat" else None)
    app.main(["--model", "gemma", "chat", "--temp", "0"])
    assert seen == [("chat", ["--model", "gemma", "--temp", "0"])]


def test_a_loading_server_is_waited_for_not_started_twice(tmp_path, monkeypatch):
    """A command stopped with Ctrl-C while the server loaded left it loading; the next command saw nothing
    listening and started a second server, which loaded the model a second time (on a 16 GB Mac, the
    double load that once took the machine down). It waits for the loading one instead."""
    from types import SimpleNamespace

    monkeypatch.setattr(app, "LOCALS_DIR", tmp_path)
    up = {"now": False}
    monkeypatch.setattr(app, "_saved_pid", lambda port: 4242)
    monkeypatch.setattr(app, "_is_locals_server", lambda pid, port=None: True)
    monkeypatch.setattr(app, "_listener_pid", lambda port: 4242 if up["now"] else None)
    monkeypatch.setattr(app, "server_alive", lambda port: {"status": "ok", "engines": {"locals": True}} if up["now"] else None)
    monkeypatch.setattr(app, "get_json", lambda url, timeout=5: {"model": "m/x", "active": []})
    monkeypatch.setattr(app, "_background_pid", lambda port: 4242)

    def popen(*a, **k):
        raise AssertionError("a second server was started")

    monkeypatch.setattr(app.subprocess, "Popen", popen)

    def wait(args, running, log_path, start, wait_s):
        assert running()
        up["now"] = True

    monkeypatch.setattr(app, "_wait_for_server", wait)
    args = SimpleNamespace(port=8399, model="m/x", engine=None, engines=None)
    app._ensure_server(args, 10)
    assert up["now"]
    up["now"] = False
    import pytest

    with pytest.raises(SystemExit, match="starting"):
        app.cmd_serve(args)


def test_a_foreground_server_loading_is_not_started_twice(tmp_path, monkeypatch):
    """`locals serve` loading the model (it writes no background pid file) was invisible to a second command, which
    started another server and loaded the model a second time. Its pid is written before it becomes the server."""
    from types import SimpleNamespace

    import pytest

    monkeypatch.setattr(app, "LOCALS_DIR", tmp_path)
    monkeypatch.setattr(app, "fetch_model", lambda model: None)
    monkeypatch.setattr(app, "remember", lambda **kw: None)

    class Exec(Exception):
        pass

    def execv(*a):
        raise Exec()

    monkeypatch.setattr(app.os, "execv", execv)
    args = SimpleNamespace(port=8399, model="m/x", engine=None, engines=None)
    with pytest.raises(Exec):
        app.cmd_serve(args)
    assert app._serve_pid_file(8399).read_text() == str(app.os.getpid())
    monkeypatch.setattr(app, "_is_locals_server", lambda pid, port=None: pid == app.os.getpid())  # (it is the server now)
    monkeypatch.setattr(app, "_listener_pid", lambda port: None)  # loading: not listening yet
    assert app._starting_pid(8399) == app.os.getpid()
    with pytest.raises(SystemExit, match="starting"):
        app.cmd_serve(args)
    assert app._background_pid(8399) is None  # never taken for a background server (not locals's to restart)


def test_serve_holds_the_start_lock_until_it_is_the_server(tmp_path, monkeypatch):
    """`locals serve` checked for a server starting, then fetched the model (minutes for a download) before it
    wrote its pid: `locals start` meanwhile started a second server. It now holds the port's start lock, which
    ensure_server waits on, and refuses while another command holds it."""
    import fcntl
    from types import SimpleNamespace

    import pytest

    monkeypatch.setattr(app, "LOCALS_DIR", tmp_path)
    monkeypatch.setattr(app, "remember", lambda **kw: None)
    args = SimpleNamespace(port=8399, model="m/x", engine=None, engines=None)
    held = []

    def fetch(model):  # while serve fetches the model, the lock is taken
        with open(tmp_path / "server-8399.lock", "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held.append(True)

    class Exec(Exception):
        pass

    def execv(*a):
        raise Exec()

    monkeypatch.setattr(app, "fetch_model", fetch)
    monkeypatch.setattr(app.os, "execv", execv)
    with pytest.raises(Exec):
        app.cmd_serve(args)
    assert held == [True]
    (tmp_path / "serve-8399.pid").unlink()
    with open(tmp_path / "server-8399.lock", "w") as f:  # another command is starting the server
        fcntl.flock(f, fcntl.LOCK_EX)
        with pytest.raises(SystemExit, match="another locals command is starting"):
            app.cmd_serve(args)


def test_the_old_single_pid_file_counts_for_the_default_port_only(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "LOCALS_DIR", tmp_path)
    (tmp_path / "server.pid").write_text("4242")
    assert app._saved_pid(app.DEFAULT_PORT) == 4242
    assert app._saved_pid(9000) is None  # (it made `locals stop --port 9000` stop the 8321 server)


def test_a_restart_keeps_the_options_the_server_was_started_with(tmp_path, monkeypatch):
    """Restarted after an upgrade by a command without them (locals chat), the server lost --ollama-port (Ollama apps
    stopped working), --max-context and --slots."""
    import argparse

    import pytest

    from locals import defaults

    monkeypatch.setattr(app, "LOCALS_DIR", tmp_path)
    monkeypatch.setattr(app, "server_alive", lambda port: {"status": "ok", "build": "old-build"})
    monkeypatch.setattr(app, "get_json", lambda url, timeout=5.0: {"model": "mlx-community/Qwen3.5-4B-MLX-4bit", "active": []})
    monkeypatch.setattr(defaults, "build_id", lambda: "new-build")
    monkeypatch.setattr(app, "_listener_pid", lambda port: 4242)
    (tmp_path / "server-8399.pid").write_text("4242")
    (tmp_path / "server-8399.json").write_text('{"ollama_port": 11434, "max_context": 32000, "slots": 8}')

    def kill(pid, sig):
        monkeypatch.setattr(app, "_listener_pid", lambda port: None)
        monkeypatch.setattr(app, "_is_locals_server", lambda pid, port=None: False)  # (gone)

    class Started(Exception):
        pass

    seen = {}

    def popen(cmd, **k):
        seen["cmd"] = cmd
        raise Started()

    monkeypatch.setattr(app.os, "kill", kill)
    monkeypatch.setattr(app.subprocess, "Popen", popen)
    with pytest.raises(Started):
        app.ensure_server(argparse.Namespace(port=8399, model="mlx-community/Qwen3.5-4B-MLX-4bit", slots=None))
    cmd = " ".join(seen["cmd"])
    assert "--ollama-port 11434" in cmd and "--max-context 32000" in cmd and "--slots 8" in cmd


def test_a_port_before_chat_is_the_one_it_uses(monkeypatch):
    """`locals --port 9000 chat` talked to the remembered port: the leading --port was not passed on."""
    from locals import client

    monkeypatch.delenv("LOCALS_PORT", raising=False)
    seen = {}
    monkeypatch.setattr(app, "_delegate", lambda name: (lambda argv: seen.update(argv=argv)) if name == "chat" else None)
    app.main(["--port", "9000", "chat"])
    assert app.os.environ.get("LOCALS_PORT") == "9000" and seen["argv"] == []
    monkeypatch.setattr(app, "_load_saved", lambda: {"port": 8321})
    monkeypatch.setattr(app, "fetch_model", lambda m: None)
    monkeypatch.setattr(app, "ensure_server", lambda args: seen.update(port=args.port))
    monkeypatch.setattr(app, "remember", lambda **kw: None)
    assert client.connect("mlx-community/Qwen3.5-4B-MLX-4bit")[0].endswith(":9000") and seen["port"] == 9000
    monkeypatch.delenv("LOCALS_PORT")


def test_a_project_folder_is_never_the_home_folder_and_a_missing_one_is_named(tmp_path, monkeypatch):
    """locals run / solve / swarm in a new terminal (the home folder) copied and hashed all of it; the README's first
    command names ~/my-project, which a new user does not have."""
    import pytest

    monkeypatch.setattr(app, "HOME", tmp_path)
    for command in ("run", "solve", "swarm", "code"):
        with pytest.raises(SystemExit, match="is your home folder"):
            app.project_folder(str(tmp_path), command)
    with pytest.raises(SystemExit, match="mkdir -p"):
        app.project_folder(str(tmp_path / "new-project"), "run")
    monkeypatch.setattr(app.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(app.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "")  # Enter: yes
    assert app.project_folder(str(tmp_path / "new-project"), "code", create_ok=True).is_dir()


def test_ctrl_c_ends_a_command_with_a_word(monkeypatch, capsys):
    import pytest

    def interrupted(argv):
        raise KeyboardInterrupt

    monkeypatch.setattr(app, "_main", interrupted)
    with pytest.raises(SystemExit) as e:
        app.main(["status"])
    assert e.value.code == 130 and "stopped" in capsys.readouterr().err


def test_chat_says_when_the_server_is_not_answering():
    import socket

    import pytest

    from locals import client

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # a port nothing listens on
    with pytest.raises(ValueError, match="not answering"):
        client._post(f"http://127.0.0.1:{port}/v1/chat/completions", {})


def test_swarm_task_lists_are_checked_before_the_server_starts(tmp_path, monkeypatch):
    import argparse

    import pytest

    monkeypatch.setattr(app, "opencode_bin", lambda: "/x/opencode")
    monkeypatch.setattr(app, "ensure_server", lambda args: (_ for _ in ()).throw(AssertionError("server started")))
    args = argparse.Namespace(tasks=str(tmp_path / "missing.txt"), task=None, n=1, dir=str(tmp_path / "p"), port=8399, model="m/x")
    with pytest.raises(SystemExit, match="cannot read the task list"):
        app.cmd_swarm(args)
    (tmp_path / "t.json").write_text('[{"what": "x"}]')
    args.tasks = str(tmp_path / "t.json")
    with pytest.raises(SystemExit, match='"task"'):
        app.cmd_swarm(args)
