"""A new user: an empty HOME, install.sh with a wheel built from this checkout, then the first
`locals code PROJECT --run TASK` in a terminal (a pseudo-terminal here), answering yes when locals offers to
install OpenCode. Nothing is tuned for this Mac yet. The models come from the shared Hugging Face cache
(not downloaded again), and uv's package cache is shared too; everything else is new."""

import os
import pty
import select
import shutil
import signal
import subprocess
import time

import pytest
from conftest import OUT, REAL_HOME, REPO, new_home, user_env
from tasks import TASKS, tree_digest

PORT = 8343


def _pty_run(argv, env, cwd, timeout, answers):
    """Run in a pseudo-terminal; ``answers``: (prompt text, reply) pairs typed when the prompt shows."""
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(cwd)
        os.execve(argv[0], argv, env)
    out, t0, todo = b"", time.time(), list(answers)
    marks = {}
    while True:
        if time.time() - t0 > timeout:
            os.kill(pid, signal.SIGTERM)
            break
        r, _, _ = select.select([fd], [], [], 1)
        if not r:
            continue
        try:
            data = os.read(fd, 65536)
        except OSError:
            break
        if not data:
            break
        out += data
        for key in (b"server ready", b"opencode in "):
            if key in out and key not in marks:
                marks[key] = time.time() - t0
        if todo and todo[0][0].encode() in out:
            os.write(fd, todo.pop(0)[1].encode())
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), out.decode(errors="replace"), marks, time.time() - t0


def test_new_user_install_and_first_task(report):
    uv = shutil.which("uv") or str(REAL_HOME / ".local" / "bin" / "uv")
    if not os.path.exists(uv):
        pytest.skip("uv is needed to build the wheel")
    dist = OUT / "dist"
    shutil.rmtree(dist, ignore_errors=True)
    b = subprocess.run([uv, "build", "--wheel", "--out-dir", str(dist), str(REPO)], capture_output=True, text=True, timeout=600)
    assert b.returncode == 0, b.stderr[-2000:]
    wheel = next(dist.glob("locals-*.whl"))

    home = new_home("first-run", tuned=False)
    (home / ".local" / "bin").mkdir(parents=True)
    shutil.copy2(uv, home / ".local" / "bin" / "uv")  # a user who has uv (install.sh would fetch it otherwise)
    env = user_env(home, path=[])
    env["UV_CACHE_DIR"] = os.environ.get("UV_CACHE_DIR") or str(REAL_HOME / ".cache" / "uv")
    env["SHELL"] = "/bin/zsh"
    runs = OUT / "runs" / "first_run"
    runs.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    p = subprocess.run(["sh", str(REPO / "install.sh")], env={**env, "LOCALS_SOURCE": str(wheel)}, cwd=home,
                       capture_output=True, text=True, timeout=900)
    install_s = time.time() - t0
    (runs / "install.out").write_text(p.stdout + "\n--- stderr\n" + p.stderr)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "is ready" in p.stdout, p.stdout
    locals_bin = home / ".local" / "bin" / "locals"
    assert locals_bin.exists(), "install.sh did not put locals in ~/.local/bin"
    env["PATH"] = f"{home / '.local' / 'bin'}:{env['PATH']}"  # a new terminal, as install.sh says

    for args in (["--version"], ["models"], ["launch"]):
        q = subprocess.run([str(locals_bin), *args], env=env, cwd=home, capture_output=True, text=True, timeout=120)
        (runs / f"locals-{'-'.join(args).strip('-')}.out").write_text(q.stdout + q.stderr)
        assert q.returncode == 0, (args, q.stdout[-1500:], q.stderr[-1500:])

    task = TASKS["fix_bug"]
    d = task.copy_to(runs)
    before = tree_digest(d)
    try:
        rc, out, marks, took = _pty_run([str(locals_bin), "code", str(d), "--run", task.prompt, "--port", str(PORT)],
                                        {**env, "PWD": str(d)}, d, 3600, [("[Y/n]", "y\r")])
    finally:
        subprocess.run([str(locals_bin), "stop", "--port", str(PORT)], env=env, cwd=home, capture_output=True, timeout=120)
    (runs / "first-task.out").write_text(out)
    ok, why = task.check(d, out, before)
    report.add({"scenario": "first run", "tool": "opencode", "task": task.name, "rc": rc, "ok": ok, "why": why, "wall": took})
    report.note(f"new user: install.sh {install_s:.0f} s; first `locals code --run`: server ready after "
                f"{marks.get(b'server ready', float('nan')):.0f} s, task done in {took:.0f} s ({'done' if ok else why})")
    assert (home / ".opencode" / "bin" / "opencode").exists(), "OpenCode was not installed:\n" + out[-2000:]
    assert rc == 0, out[-3000:]
    assert b"server ready" in marks, out[-3000:]
