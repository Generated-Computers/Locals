"""The kernel tuning cache when two processes write it (`locals tune` next to a server tuning while idle): each
save merges with the file as it is, so neither erases the other's results."""

import json

from locals import app, kernels


def test_a_save_keeps_what_another_process_wrote(tmp_path, monkeypatch):
    path = tmp_path / "tune.json"
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(path))
    monkeypatch.setattr(kernels, "_TUNE", {"2560x4096x8": ("frag", 2, 2, 1), "__times__": {"2560x4096x8": {"frag": 10.0}},
                                           "__model_tuned_pipe__x": [2, 3]})
    path.write_text(json.dumps({"9216x2560x8": ["mlx"], "__times__": {"9216x2560x8": {"mlx": 20.0}},
                                "__model_tuned_pipe__x": [3, 4]}))  # another process saved meanwhile
    kernels._save_tune()
    data = json.loads(path.read_text())
    assert data["2560x4096x8"] == ["frag", 2, 2, 1] and data["9216x2560x8"] == ["mlx"]
    assert set(data["__times__"]) == {"2560x4096x8", "9216x2560x8"}
    assert data["__model_tuned_pipe__x"] == [3, 4, 2]
    assert kernels._TUNE["9216x2560x8"] == ("mlx",)  # and this process uses the other's results from now on


def test_tune_refuses_while_a_server_runs(monkeypatch):
    monkeypatch.setattr(app, "_locals_server_pids", lambda: [4242])
    try:
        app.cmd_tune(["--model", "mlx-community/Qwen3.5-4B-MLX-4bit"])
    except SystemExit as e:
        assert "4242" in str(e) and "locals stop" in str(e)
    else:
        raise AssertionError("tuned next to a running server")


def test_a_shape_that_stops_the_process_twice_keeps_mlx(tmp_path, monkeypatch):
    """A Metal error MLX raises in a command buffer's completion handler aborts the process (no except catches
    it): the shape being timed would be tuned again, and abort the server again, at every start. A marker names
    it while it is timed; two deaths there pin it to MLX's kernel, one (a kill, a power cut) only tunes it again."""
    import os
    import subprocess

    path = tmp_path / "tune.json"
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(path))
    monkeypatch.setattr(kernels, "_preset_path", lambda: None)
    seen = []
    monkeypatch.setattr(kernels, "_time_shape", lambda *a: seen.append(os.path.exists(kernels._marker(os.getpid()))) or ("mlx",))
    kernels._tune_shape(None, None, 8, 0, 0, "float16", "k", {})
    assert seen == [True] and not os.path.exists(kernels._marker(os.getpid()))
    dead = subprocess.Popen(["true"])
    dead.wait()
    for deaths in (1, 2):
        monkeypatch.setattr(kernels, "_TUNE", None)
        (tmp_path / f"tune.json.tuning-{dead.pid}").write_text("2560x4096x8")
        tune = kernels._load_tune()
        assert tune["__crashed__"] == {"2560x4096x8": deaths} and not list(tmp_path.glob("*.tuning-*"))
        assert ("2560x4096x8" in tune) == (deaths == 2)
    assert tune["2560x4096x8"] == ("mlx",) and json.loads(path.read_text())["2560x4096x8"] == ["mlx"]
    monkeypatch.setattr(kernels, "_TUNE", None)
    live = tmp_path / f"tune.json.tuning-{os.getppid()}"  # a process still tuning: left alone
    live.write_text("9216x2560x8")
    assert "9216x2560x8" not in kernels._load_tune() and live.exists()


def test_a_stop_while_a_shape_is_timed_is_not_a_crash(tmp_path):
    """locals stop (SIGTERM) during idle tuning left the marker: two stops there pinned the shape to MLX's kernel
    for good, with a message about crashes. The server still dies by SIGTERM at once, with its marker gone."""
    import os
    import signal
    import subprocess
    import sys

    child = (
        "import threading, time\n"
        "from locals import kernels, server\n"
        "server._on_sigterm()\n"
        "kernels._time_shape = lambda *a: time.sleep(60)\n"
        "threading.Thread(target=kernels._tune_shape, args=(None, None, 8, 0, 0, 'float16', 'k', {}), daemon=True).start()\n"
        "time.sleep(0.5)\n"
        "print('timing', flush=True)\n"
        "time.sleep(60)\n")
    env = dict(os.environ, LOCALS_TUNE_CACHE=str(tmp_path / "tune.json"))
    p = subprocess.Popen([sys.executable, "-c", child], env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "timing"
        assert (tmp_path / f"tune.json.tuning-{p.pid}").exists()
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=20) == -signal.SIGTERM
        assert not list(tmp_path.glob("*.tuning-*"))
    finally:
        p.kill()


def test_a_save_a_killed_process_left_is_removed(tmp_path, monkeypatch):
    import os
    import time

    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_preset_path", lambda: None)
    monkeypatch.setattr(kernels, "_TUNE", None)
    old, new = tmp_path / ".tune-old.json", tmp_path / ".tune-new.json"
    old.write_text("{}")
    new.write_text("{}")  # (another process saving right now)
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    kernels._load_tune()
    assert not old.exists() and new.exists()
