"""`locals solve`'s parts that need no model: workspaces that hold the project exactly as
the user has it (uncommitted changes included), the changes an attempt made, the check,
and the prompts."""

import subprocess

import pytest

from locals import solve


def _git(d, *args):
    subprocess.run(["git", "-C", str(d), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(d)})


@pytest.fixture
def repo(tmp_path):
    d = tmp_path / "proj"
    (d / "pkg").mkdir(parents=True)
    (d / "pkg" / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (d / "keep.txt").write_text("keep\n")
    (d / "gone.txt").write_text("to be deleted\n")
    (d / "test_calc.py").write_text("from pkg.calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    _git(d, "init", "-q")
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "init")
    # the user's uncommitted work: a change, a deletion and a new file
    (d / "keep.txt").write_text("keep, edited\n")
    (d / "gone.txt").unlink()
    (d / "notes.md").write_text("untracked\n")
    return d


def test_worktree_holds_the_working_tree(repo, tmp_path):
    ws = tmp_path / "ws"
    assert solve.make_workspace(repo, ws) == "worktree"
    assert (ws / "keep.txt").read_text() == "keep, edited\n"
    assert not (ws / "gone.txt").exists() and (ws / "notes.md").read_text() == "untracked\n"
    assert solve.manifest(ws) == solve.manifest(repo)  # the same files, .git left out
    # an attempt's work
    (ws / "pkg" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (ws / "pkg" / "extra.py").write_text("X = 1\n")
    (ws / "notes.md").unlink()
    changed, added, deleted = solve.changes(solve.manifest(repo), solve.manifest(ws))
    assert (changed, added, deleted) == (["pkg/calc.py"], ["pkg/extra.py"], ["notes.md"])
    diff = solve.unified_diff(repo, ws, changed + added, deleted)
    assert "-    return a - b" in diff and "+    return a + b" in diff and "+X = 1" in diff and "-untracked" in diff
    solve.remove_workspace(repo, ws, "worktree")
    assert not ws.exists()
    assert "ws" not in subprocess.run(["git", "-C", str(repo), "worktree", "list"], capture_output=True, text=True).stdout
    assert (repo / "pkg" / "calc.py").read_text().endswith("a - b\n")  # the project itself is untouched


def test_copy_workspace_without_git(tmp_path):
    d = tmp_path / "plain"
    (d / "node_modules" / "x").mkdir(parents=True)
    (d / "node_modules" / "x" / "big.js").write_text("//")
    (d / "app.py").write_text("print(1)\n")
    ws = tmp_path / "ws"
    assert solve.make_workspace(d, ws) == "copy"
    assert (ws / "app.py").exists() and not (ws / "node_modules").exists()


def test_check_detection_and_run(repo, tmp_path):
    assert solve.detect_check(repo) in ("python3 -m pytest -q -x", "python3 -m unittest discover -q")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert solve.detect_check(empty) is None
    (empty / "package.json").write_text('{"scripts": {"test": "node t.js"}}')
    assert solve.detect_check(empty) == "npm test --silent"
    ok, out, secs = solve.run_check("echo hi; exit 3", tmp_path)
    assert not ok and out == "hi" and secs >= 0
    ok, out, _ = solve.run_check("sleep 5", tmp_path, timeout=0.5)
    assert not ok and "ran over" in out


def test_prompts():
    p = solve.attempt_prompt("Fix add.", "python3 -m pytest -q", solve.HINTS[1], "")
    assert p.startswith("Fix add.") and "python3 -m pytest -q" in p and solve.HINTS[1] in p
    a = solve.Attempt(0, None, "copy", "")
    a.answer, a.check_out = "Changed calc.py to add.", "E   assert -1 == 5\n1 failed"
    notes = solve.failure_notes([a])
    assert "Changed calc.py to add." in notes and "1 failed" in notes
    assert solve.attempt_prompt("Fix add.", "c", "", notes).startswith("Earlier attempts")
    assert "assert -1 == 5" in solve.fix_prompt("c", a.check_out)


class FakeOpenCode:
    """OpenCode's session API with scripted agents: agent i (in order of creation) runs
    ``behaviors[i](workspace, turn, aborted)`` for each message it is sent."""

    def __init__(self, behaviors):
        import threading

        self.behaviors, self.sessions, self.aborted, self.prompts = behaviors, {}, set(), {}
        self.lock = threading.Lock()

    def call(self, method, path, body=None, directory=None, timeout=3600):
        from pathlib import Path

        if method == "POST" and path == "/session":
            with self.lock:
                sid = f"s{len(self.sessions)}"
                self.sessions[sid] = {"i": len(self.sessions), "dir": Path(directory), "turn": 0}
            return {"id": sid}
        sid = path.split("/")[2]
        if path.endswith("/abort"):
            self.aborted.add(sid)
            return True
        s = self.sessions[sid]
        if method == "POST":
            self.prompts.setdefault(s["i"], []).append(body["parts"][0]["text"])
            self.behaviors[s["i"]](s["dir"], s["turn"], lambda: sid in self.aborted)
            s["turn"] += 1
            return {}
        return [{"info": {"role": "assistant"}, "parts": [{"type": "text", "text": f"agent {s['i']} done"}]}]


def _agent(code, delay):
    import time

    def run(ws, turn, aborted):
        t = time.time()
        while time.time() - t < delay:
            if aborted():
                return
            time.sleep(0.05)
        (ws / "calc.py").write_text(code)
    return run


GOOD, BAD = "def add(a, b):\n    return a + b\n", "def add(a, b):\n    return a * b\n"
CHECK = "python3 -c 'import calc; assert calc.add(2, 3) == 5'"


def _race(tmp_path, behaviors, **kw):
    proj = tmp_path / "p"
    proj.mkdir()
    (proj / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    oc = FakeOpenCode(behaviors)
    race = solve.Race(oc, proj, "Fix add.", CHECK, "coder", "locals", "locals/x", run_dir, timeout=30, fixes=kw.pop("fixes", 1))
    race.run(status=lambda attempts: None, **kw)
    return race, oc


def test_hedge_starts_a_second_agent_that_outruns_the_first(tmp_path):
    race, oc = _race(tmp_path, [_agent(BAD, 3.0), _agent(GOOD, 0.3), _agent(GOOD, 0.1)], parallel=3, max_attempts=4, hedge=1.0)
    assert race.winner.name == "a1" and race.attempts[0].state == "outrun" and len(race.attempts) == 2
    assert "s0" in oc.aborted  # the slow one was stopped when the second passed


def test_a_failed_attempt_is_replaced_by_one_told_what_failed(tmp_path):
    race, oc = _race(tmp_path, [_agent(BAD, 0.1), _agent(BAD, 0.2), _agent(GOOD, 0.1)], parallel=2, max_attempts=3, hedge=0)
    assert [a.state for a in race.attempts] == ["failed", "failed", "passed"]
    assert race.attempts[0].turns == 2  # the failed check was shown to it once (--fixes 1)
    assert "The check failed" in oc.prompts[0][1] and "assert" in oc.prompts[0][1]
    first = oc.prompts[2][0]
    assert first.startswith("Earlier attempts") and "agent 0 done" in first and race.attempts[2].after >= 1


def test_no_attempt_passes_within_the_budget(tmp_path):
    race, _ = _race(tmp_path, [_agent(BAD, 0.05)] * 2, parallel=1, max_attempts=2, hedge=0, fixes=0)
    assert race.winner is None and [a.state for a in race.attempts] == ["failed", "failed"]


def _writer(script):
    def run(ws, turn, aborted):
        if script is not None:
            (ws / "locals_check.py").write_text(script)
    return run


def test_check_writer(tmp_path):
    """Without tests, an agent writes locals_check.py; a check that compiles and fails on the
    untouched project is kept (outside the workspaces), one that passes there is rejected."""
    proj = tmp_path / "p"
    proj.mkdir()
    (proj / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    good = "import calc\nassert calc.add(2, 3) == 5, 'add is wrong'\n"
    useless = "print('ok')\n"
    for behaviors, expect in (([_writer(good)], True), ([_writer(useless), _writer(None)], False),
                              ([_writer("def (:\n"), _writer(good)], True)):
        run_dir = tmp_path / f"run{len(list(tmp_path.iterdir()))}"
        run_dir.mkdir()
        oc = FakeOpenCode(behaviors)
        cmd = solve.write_check(oc, proj, "Fix add.", run_dir, "coder", "locals", "locals/x", timeout=30, log=lambda s: None)
        assert (cmd is not None) == expect
        if cmd:
            assert cmd == solve.check_command(run_dir / "locals_check.py") and "add is wrong" in solve.run_check(cmd, proj)[1]
            (proj / "calc.py").write_text("def add(a, b):\n    return a + b\n")  # a right solution passes the moved check
            assert solve.run_check(cmd, proj)[0]
            (proj / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        assert not any(p.name.startswith(("check", "probe")) for p in run_dir.iterdir())  # its workspaces are gone


def test_escalation_starts_two_fresh_agents_after_a_failure(tmp_path):
    """--start 1 --burst 2: one agent alone; when it fails, two fresh ones at once."""
    race, oc = _race(tmp_path, [_agent(BAD, 0.1), _agent(BAD, 1.5), _agent(GOOD, 0.3)], parallel=3, max_attempts=5,
                     hedge=0, start=1, burst=2, fixes=0)
    assert [a.state for a in race.attempts] == ["failed", "outrun", "passed"]
    a1, a2 = race.attempts[1], race.attempts[2]
    assert abs(a1.t0 - a2.t0) < 0.3 and a1.t0 >= race.attempts[0].t1 - 0.05  # together, after the first failed


def test_notes_leave_out_the_servers_steering_text():
    """A failed agent's answer may hold what the server wrote into it when it repeated a
    call; the notes for the next agent keep only the agent's own words."""
    a = solve.Attempt(0, None, "copy", "")
    a.answer = ("The edit call above was already made and returned: 'Could not find oldString'. Repeating it would give "
                "the same result, so I will do something different now: The mean function was replaced by mistake.\n\n"
                "I am stuck: I keep wanting to repeat the same bash call ({\"command\": \"ls\"}), whose result is already "
                "above. Please tell me how to proceed (for example, which command or file to use).")
    a.check_out = "ImportError: cannot import name 'mean'"
    notes = solve.failure_notes([a])
    assert "The mean function was replaced by mistake." in notes and "cannot import name" in notes
    assert "already made" not in notes and "I am stuck" not in notes
