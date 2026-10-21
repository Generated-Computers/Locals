"""What Locals keeps on disk survives crashes, full disks and odd states: the disk prompt cache, ~/.locals's
config, pid files, and the workspaces an interrupted locals solve left."""

import json
import os
import subprocess

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from locals import app, solve
from locals.hybrid_batch import DiskPromptCache


def _cache(n=8):
    c = KVCache()
    k = mx.random.normal((1, 2, n, 4), key=mx.random.key(n))
    c.update_and_fetch(k, k)
    return [c]


def test_a_failed_snapshot_save_leaves_nothing_behind(tmp_path, monkeypatch):
    import mlx_lm.models.cache as mc

    d = DiskPromptCache(tmp_path / "m", 1.0)

    def half(path, cache):
        open(path, "wb").write(b"x" * 1000)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(mc, "save_prompt_cache", half)
    d.save([1, 2, 3], _cache())
    assert list((tmp_path / "m").iterdir()) == []  # (each failed save kept its partial file, for good)
    (tmp_path / "m" / ".abc.999999.tmp.safetensors").write_bytes(b"x")  # one a dead process left
    DiskPromptCache(tmp_path / "m", 1.0)
    assert not (tmp_path / "m" / ".abc.999999.tmp.safetensors").exists()


def test_the_snapshot_cap_covers_every_models_folder(tmp_path):
    old = tmp_path / "old-model"
    old.mkdir()
    (old / "a.safetensors").write_bytes(b"x" * 10000)
    (old / "a.tokens.json").write_text("[1]")
    os.utime(old / "a.safetensors", (1, 1))  # long ago
    d = DiskPromptCache(tmp_path / "new-model", 20000 / (1 << 30))  # (a tiny snapshot is 16.6 kB)
    d.save([1, 2, 3], _cache())
    assert not old.exists()  # the old model's snapshot went first, and its folder with it
    assert any((tmp_path / "new-model").glob("*.safetensors"))


def test_a_broken_config_is_set_aside_and_rewritten(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(app, "LOCALS_DIR", tmp_path)
    (tmp_path / "config.json").write_text("")  # renamed after a power cut, the write came back empty
    assert app._load_saved() == {} and "did not parse" in capsys.readouterr().err
    assert (tmp_path / "config.json.bad").exists()
    monkeypatch.setattr(app, "server_alive", lambda port: None)
    app.remember(model="m/x", port=8399)
    assert json.loads((tmp_path / "config.json").read_text())["model"] == "m/x"


def test_a_pid_file_counts_only_for_its_own_port(monkeypatch):
    class P:
        stdout = "/usr/bin/python3 -m locals.server --port 8400 --model m/x"

    monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: P())
    assert app._is_locals_server(42) and app._is_locals_server(42, 8400)
    assert not app._is_locals_server(42, 8321)  # (a stale server-8321.pid naming it: locals stop --port 8321 stopped it)


def test_workspaces_of_an_interrupted_solve_run_are_removed(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    (project / "f.py").write_text("x = 1\n")
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qam", "x", "--allow-empty"], cwd=project)
    subprocess.run(["git", "add", "."], cwd=project, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "f"], cwd=project, check=True)
    root = tmp_path / "solve"
    dead, kept = root / "proj-20261021-010101", root / "proj-20261021-020202"  # (as solve names them)
    for run in (dead, kept):
        run.mkdir(parents=True)
        solve.make_workspace(project, run / "a0")
    (dead / "pid").write_text("999999")  # a run whose terminal was closed
    assert solve.cleanup_stale_runs(project, root) == 1
    assert not (dead / "a0").exists() and (kept / "a0").exists()  # a finished run (no pid file) keeps its own (--keep)
    listed = subprocess.run(["git", "worktree", "list"], cwd=project, capture_output=True, text=True).stdout
    assert str(dead / "a0") not in listed


def test_a_snapshot_folder_deleted_under_a_running_server_is_made_again(tmp_path):
    """The user cleared ~/.cache/locals, or another model's server pruned the folder away: the next save raised out of
    the engine, which failed every request in flight and threw out half the prompt cache, at each new system prompt."""
    import shutil

    d = DiskPromptCache(tmp_path / "m", 1.0)
    d.save([1, 2, 3], _cache())
    shutil.rmtree(tmp_path / "m")
    d.save([4, 5, 6], _cache())  # (raised FileNotFoundError)
    d.save([1, 2, 3], _cache())  # the index named it, the file was gone: saved again
    assert len(list((tmp_path / "m").glob("*.safetensors"))) == 2


def test_a_cap_below_one_snapshot_saves_nothing_and_deletes_nothing(tmp_path):
    """LOCALS_DISK_CACHE_GB=0 (or a tiny cap): each save wrote a snapshot, then deleted every model's, its own too."""
    other = tmp_path / "other-model"
    other.mkdir()
    (other / "a.safetensors").write_bytes(b"x" * 10)
    (other / "a.tokens.json").write_text("[1]")
    d = DiskPromptCache(tmp_path / "m", 1000 / (1 << 30))
    d.save([1, 2, 3], _cache())
    assert not list((tmp_path / "m").glob("*.safetensors")) and (other / "a.safetensors").exists()


def test_each_port_has_its_own_server_log(tmp_path, monkeypatch):
    """One log for every port: a start showed another server's lines (and could take its failure hint from them),
    and a rotation left the other server writing into a deleted file."""
    monkeypatch.setattr(app, "LOCALS_DIR", tmp_path)
    assert app._log_file(8321) == tmp_path / "server.log" and app._log_file(8342) == tmp_path / "server-8342.log"
