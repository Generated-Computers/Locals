"""`locals solve --plan` without a model: the plan made safe to run, the prompts, and the
team (parts at once in one workspace, then the check and a fix) with scripted agents."""

import threading
import time

from locals import plan, solve


def test_normalize_makes_the_plan_safe():
    raw = {"parts": [
        {"title": "store", "files": ["./inventory.py"], "instructions": "class Inventory", "needs": [2]},  # needs a later part
        {"title": "cli", "files": ["cli.py", "/etc/passwd", "../up.py"], "instructions": "commands", "needs": [0]},
        {"title": "tests", "files": ["test_inventory.py", "inventory.py"], "instructions": "tests", "needs": []},
        {"title": "nothing", "files": [], "instructions": "", "needs": []}]}
    parts = plan.normalize(raw)
    assert [p["title"] for p in parts] == ["store", "cli", "tests"]
    assert parts[0]["files"] == ["inventory.py"] and parts[0]["needs"] == []
    assert parts[1]["files"] == ["cli.py"] and parts[1]["needs"] == [0]
    assert parts[2]["needs"] == [0]  # it shares inventory.py with part 0, so it waits for it


def test_part_prompt_and_digest(tmp_path):
    parts = plan.normalize({"parts": [{"title": "store", "files": ["inventory.py"], "instructions": "Write Inventory.", "needs": []},
                                      {"title": "cli", "files": ["cli.py"], "instructions": "Write the CLI.", "needs": [0]}]})
    p = plan.part_prompt("Build a tool.", parts, 1)
    assert "Build a tool." in p and "Write the CLI." in p and "Write only these files: cli.py" in p
    assert "part 0, store: inventory.py" in p and "are done: read their files" in p
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.pyc").write_bytes(b"\0")
    d = plan.project_digest(tmp_path)
    assert d.startswith("Files: a.py") and "x = 1" in d and "pyc" not in d


class _Race:
    def __init__(self, ocs, check, run_dir):
        self.ocs, self.check, self.run_dir = ocs, check, run_dir
        self.agent, self.provider, self.model, self.timeout = "coder", "locals", "locals/x", 30
        self.attempts, self.winner, self.base = [], None, None


def test_team_runs_parts_at_once_then_checks_and_fixes(tmp_path):
    from test_solve import FakeOpenCode

    starts = {}

    def writer(name, text, delay):
        def run(ws, turn, aborted):
            starts[name] = time.time()
            time.sleep(delay)
            (ws / name).write_text(text)
        return run

    def fixer(ws, turn, aborted):
        (ws / "b.py").write_text("from a import A\nB = A + 1\n")

    ws = tmp_path / "ws"
    ws.mkdir()
    oc = FakeOpenCode([writer("a.py", "A = 1\n", 0.5), writer("c.py", "C = 3\n", 0.5), writer("b.py", "B = 0\n", 0.1), fixer])
    parts = plan.normalize({"parts": [{"title": "a", "files": ["a.py"], "instructions": "", "needs": []},
                                      {"title": "c", "files": ["c.py"], "instructions": "", "needs": []},
                                      {"title": "b", "files": ["b.py"], "instructions": "", "needs": [0]}]})
    race = _Race(oc, "python3 -c 'import b; assert b.B == 2'", tmp_path)
    report = plan.Team(race, ws).run("Make B two.", parts, fixes=1)
    assert report["passed"] and [c["ok"] for c in report["checks"]] == [False, True]
    assert abs(starts["a.py"] - starts["c.py"]) < 0.3  # independent parts started together
    assert starts["b.py"] >= starts["a.py"] + 0.45  # the part that needs a waited for it
    assert "The check fails" in oc.prompts[3][0]


def test_a_part_that_only_talks_is_told_to_write_its_files(tmp_path):
    """Gemma 4 answered its part with a description (no tool calls) and stopped; the part
    is told once which of its files are missing, in the same session."""
    from test_solve import FakeOpenCode

    def talker(ws, turn, aborted):
        if turn == 1:  # only after the nudge
            (ws / "a.py").write_text("A = 2\n")

    ws = tmp_path / "ws"
    ws.mkdir()
    oc = FakeOpenCode([talker])
    parts = plan.normalize({"parts": [{"title": "a", "files": ["a.py"], "instructions": "", "needs": []}]})
    race = _Race(oc, "python3 -c 'import a; assert a.A == 2'", tmp_path)
    report = plan.Team(race, ws).run("Make A two.", parts, fixes=0)
    assert report["passed"] and report["part_runs"][0]["nudged"] == ["a.py"] and report["part_runs"][0]["missing"] == []
    assert "You have not written a.py yet" in oc.prompts[0][1]


def test_samples_write_their_parts_in_waves_at_once(tmp_path, monkeypatch):
    """--plan's default: every part one generation; the parts that need nothing are written at
    once, in every sample at once, and a part that needs another after it, with that sample's code;
    a complete sample is checked at once."""
    calls = []
    code = {"a.py": "A = 1", "c.py": "C = 3", "b.py": "from a import A\nB = A + 1"}

    def fake_chat_files(base_url, model, messages, paths, max_tokens=4096, timeout=900, temperature=None, stop=None, note=False):
        calls.append((time.time(), paths, messages[1]["content"], temperature))
        time.sleep(0.3)
        return {"files": {p: code[p] for p in paths}, "usage": {"completion_tokens": 5}}

    monkeypatch.setattr(plan, "chat_files", fake_chat_files)
    parts = plan.normalize({"parts": [{"title": "a", "files": ["a.py"], "instructions": "A is one.", "needs": []},
                                      {"title": "c", "files": ["c.py"], "instructions": "C is three.", "needs": []},
                                      {"title": "b", "files": ["b.py"], "instructions": "B is A + 1.", "needs": [0]}]})
    ws = tmp_path / "ws"
    ws.mkdir()
    team = plan.Team(_Race(None, "python3 -c 'import b, c; assert b.B == 2 and c.C == 3'", tmp_path), ws)
    w = plan._write_samples(team, "http://x", "Make B two.", parts, "Files: (none)", 2, plan._Checks(team), status=lambda s: None)
    assert [(r["sample"], r["part"], r["wave"]) for r in w["runs"]] == [(0, 0, 1), (0, 1, 1), (0, 2, 2), (1, 0, 1), (1, 1, 1), (1, 2, 2)]
    first = [c[0] for c in calls if c[1] != ["b.py"]]
    assert len(first) == 4 and max(first) - min(first) < 0.2  # both samples' independent parts in one batch
    assert min(c[0] for c in calls if c[1] == ["b.py"]) >= min(first) + 0.25
    assert "--- a.py\nA = 1" in next(c[2] for c in calls if c[1] == ["b.py"])
    assert {c[3] for c in calls} == {None, 0.6}  # the samples differ
    assert any(ok for _, ok, _ in w["results"])


def test_search_mixes_parts_of_different_samples(tmp_path):
    """Sample 0 has a good a.py and a bad b.py, sample 1 the other way round: neither passes whole,
    a combination does."""
    ws = tmp_path / "ws"
    ws.mkdir()
    team = plan.Team(_Race(None, "python3 -c 'import a, b; assert a.A == 1 and b.B == 2'", tmp_path), ws)
    options = [[{"a.py": "A = 1"}, {"a.py": "A = 0"}], [{"b.py": "B = 0"}, {"b.py": "B = 2"}]]
    checks = plan._Checks(team)
    tried = {(0, 0), (1, 1)}
    first = [((0, 0), False, "AssertionError: B is 0"), ((1, 1), False, "AssertionError: A is 0")]
    results = plan._search(options, checks, tried, first)
    assert [c for c, ok, _ in results if ok] == [(0, 1)]
    assert not (ws / "a.py").exists()  # combinations are checked in copies


def test_search_climbs_part_by_part(tmp_path):
    """Six parts, two versions each, every version 1 right and every version 0 wrong but one:
    the swaps that fail fewer tests are put together, and that passes; far fewer checks than
    the 64 combinations."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "t.py").write_text(
        "import unittest\n" + "".join(f"class T{i}(unittest.TestCase):\n    def test(self):\n        import m{i}\n"
                                     f"        self.assertEqual(m{i}.V, 1)\n" for i in range(6)))
    team = plan.Team(_Race(None, "python3 -m unittest t", tmp_path), ws)
    options = [[{f"m{i}.py": "V = 1" if i in (2, 4) else "V = 0"}, {f"m{i}.py": "V = 0" if i == 4 else "V = 1"}] for i in range(6)]
    checks = plan._Checks(team)
    first = plan._check_all(options, checks, [(0,) * 6, (1,) * 6])
    assert not any(ok for _, ok, _ in first) and plan._failed(first[0][2]) == 4 and plan._failed(first[1][2]) == 1
    results = plan._search(options, checks, {(0,) * 6, (1,) * 6}, first)
    good = [c for c, ok, _ in results if ok]
    assert good == [(1, 1, 1, 1, 0, 1)] and len(results) <= 6


def test_search_puts_the_better_swaps_together(tmp_path):
    """Two parts wrong in the best version, each right in another: the two swaps that each fail
    fewer tests are put together."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "t.py").write_text(
        "import unittest\n" + "".join(f"class T{i}(unittest.TestCase):\n    def test(self):\n        import m{i}\n"
                                     f"        self.assertEqual(m{i}.V, 1)\n" for i in range(4)))
    team = plan.Team(_Race(None, "python3 -m unittest t", tmp_path), ws)
    options = [[{f"m{i}.py": "V = 1" if i in (0, 1) else "V = 0"}, {f"m{i}.py": "V = 0" if i in (0, 1) else "V = 1"}] for i in range(4)]
    checks = plan._Checks(team)
    first = plan._check_all(options, checks, [(0,) * 4, (1,) * 4])
    results = plan._search(options, checks, {(0,) * 4, (1,) * 4}, first)
    assert [c for c, ok, _ in results if ok] == [(0, 0, 1, 1)]


def test_a_repair_checks_each_new_version_as_it_comes(tmp_path, monkeypatch):
    """A repair round: new versions of the part the check points at (b.py), each checked with the
    others' current files as it comes; the one that passes wins, the others are stopped."""
    seen = []

    def fake_chat_files(base_url, model, messages, paths, max_tokens=4096, timeout=900, temperature=None, stop=None, note=False):
        seen.append((paths, temperature, messages[1]["content"]))
        if paths == ["b.py"] and temperature == 0.9:
            time.sleep(0.2)
            return {"files": {"b.py": "B = 2"}, "usage": {"completion_tokens": 3}}
        for _ in range(40):  # the others write slowly, and stop when told
            if stop is not None and stop.is_set():
                return {"files": {}, "usage": {}, "stopped": True}
            time.sleep(0.05)
        return {"files": {paths[0]: "A = 1" if paths == ["a.py"] else "B = 1"}, "usage": {"completion_tokens": 3}}

    monkeypatch.setattr(plan, "chat_files", fake_chat_files)
    ws = tmp_path / "ws"
    ws.mkdir()
    parts = plan.normalize({"parts": [{"title": "a", "files": ["a.py"], "instructions": "", "needs": []},
                                      {"title": "b", "files": ["b.py"], "instructions": "", "needs": []}]})
    team = plan.Team(_Race(None, "python3 -c 'import a, b; assert a.A == 1 and b.B == 2, b.B'", tmp_path), ws)
    cur = [{"a.py": "A = 1"}, {"b.py": "B = 0"}]
    t0 = time.time()
    rr = plan._repair(team, "http://x", "Make B two.", parts, cur, "FAIL: b.py: AssertionError: 0", plan._Checks(team), lambda s: None)
    assert time.time() - t0 < 1.5  # the slow ones were stopped
    assert rr["todo"] == [1] and len(seen) == 3 and all(p == ["b.py"] for p, _, _ in seen)
    good = [r for r in rr["results"] if r[1]]
    assert good and plan._merge(rr["options"], good[0][0]) == {"a.py": "A = 1", "b.py": "B = 2"}
    assert "--- a.py\nA = 1" in seen[0][2] and "AssertionError: 0" in seen[0][2]


def test_repair_candidates_differ_and_copies_are_not_checked(tmp_path, monkeypatch):
    """Shown its own file and asked to fix it, the model copied it back: one candidate says what is
    wrong first and sees the file, two start afresh without it; a copy of the current file is not
    checked again."""
    seen = []

    def fake_chat_files(base_url, model, messages, paths, max_tokens=4096, timeout=900, temperature=None, stop=None, note=False):
        seen.append((temperature, note, messages[0]["content"], messages[1]["content"]))
        return {"files": {"b.py": "B = 0" if note and temperature is None else "B = 1"}, "usage": {"completion_tokens": 3}}

    monkeypatch.setattr(plan, "chat_files", fake_chat_files)
    ws = tmp_path / "ws"
    ws.mkdir()
    parts = plan.normalize({"parts": [{"title": "a", "files": ["a.py"], "instructions": "", "needs": []},
                                      {"title": "b", "files": ["b.py"], "instructions": "", "needs": []}]})
    team = plan.Team(_Race(None, "python3 -c 'import a, b; assert b.B == 2, b.B'", tmp_path), ws)
    team.original = {"b.py": '"""B is two."""'}
    cur = [{"a.py": "A = 1"}, {"b.py": "B = 0"}]
    rr = plan._repair(team, "http://x", "Make B two.", parts, cur, "FAIL: b.py: AssertionError: 0", plan._Checks(team), lambda s: None)
    by = sorted(seen, key=lambda x: (x[0] is not None, x[0] or 0))
    assert [(t, n) for t, n, _, _ in by] == [(None, True), (0.6, False), (0.9, True)]
    assert "--- b.py\nB = 0" in by[0][3] and all("--- b.py\nB = 0" not in u for _, _, _, u in by[1:])  # the fresh ones do not see it
    assert all("--- a.py\nA = 1" in u for _, _, _, u in by) and "The problem:" in by[0][3]
    assert len({sy for _, _, sy, _ in by}) == 1  # one system prompt: the candidates share their prefix
    assert all('before the task:\n--- b.py\n"""B is two."""' in u for _, _, _, u in by[1:])  # the stub: the spec
    assert [g["same"] for g in rr["gens"]][0] and sorted(g["same"] for g in rr["gens"]) == [False, True, True]
    assert len(rr["results"]) == 1  # the copy of b.py, and the second "B = 1", were not checked
    assert plan.files_regex(["b.py"], note=True).startswith("The problem: ")


def test_a_runaway_version_is_stopped_when_the_others_are_done(monkeypatch):
    """When all but a few generations are done, the rest get half the time so far (15 s at least)."""
    monkeypatch.setattr(plan.time, "time", lambda t=[0.0]: t.__setitem__(0, t[0] + 1.0) or t[0])  # 1 s per call
    late = threading.Event()
    stopped = []

    def quick():
        pass

    def runaway():
        while not late.is_set():
            time.sleep(0.01)
        stopped.append(True)

    threads = [threading.Thread(target=f) for f in (quick, quick, quick, quick, runaway)]
    for th in threads:
        th.start()
    plan._wait(threads, late, lambda s: None, {})
    assert late.is_set() and stopped == [True]


def test_agent_repair_round_runs_agents_in_copies_and_combines(tmp_path):
    """Later repair rounds: an agent for each part the check points at (only its files are
    taken) and one that may change anything, at once, each in its own copy; the part agents'
    files go into the search."""
    from test_solve import FakeOpenCode

    def agent(ws, turn, aborted):
        if ws.name.endswith("-0"):  # the agent of part b: fixes b.py, and also (wrongly) touches a.py
            (ws / "b.py").write_text("B = 2\n")
            (ws / "a.py").write_text("A = 99\n")

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.py").write_text("A = 1\n")
    (ws / "b.py").write_text("B = 0\n")
    parts = plan.normalize({"parts": [{"title": "a", "files": ["a.py"], "instructions": "", "needs": []},
                                      {"title": "b", "files": ["b.py"], "instructions": "", "needs": []}]})
    oc = FakeOpenCode([agent] * 2)
    race = _Race(oc, "python3 -c 'import a, b; assert a.A == 1 and b.B == 2, b.B'", tmp_path)
    team = plan.Team(race, ws)
    cur = [{"a.py": "A = 1\n"}, {"b.py": "B = 0\n"}]
    rr = plan._agent_repair(team, "Make B two.", parts, cur, "FAIL: b.py: AssertionError: 0", plan._Checks(team), lambda s: None, 1)
    assert rr["todo"] == [1] and len(rr["agents"]) == 2
    good = [c for c, ok, _ in rr["results"] if ok]
    assert good and plan._merge(rr["options"], good[0]) == {"a.py": "A = 1\n", "b.py": "B = 2\n"}  # its a.py was not taken
    assert all("change only b.py" in t[0] or "change only a.py, b.py" in t[0] for t in oc.prompts.values())


def test_blame_points_at_the_parts_a_failure_is_about():
    parts = plan.normalize({"parts": [{"title": "s", "files": ["toolkit/strings.py"], "instructions": "", "needs": []},
                                      {"title": "n", "files": ["toolkit/numbers.py"], "instructions": "", "needs": []},
                                      {"title": "t", "files": ["tests/test_more.py"], "instructions": "", "needs": []}]})
    cur = [{"toolkit/strings.py": "import re\n\ndef slugify(t):\n    pass\n"}, {"toolkit/numbers.py": "def clamp(x, lo, hi):\n    pass\n"},
           {"tests/test_more.py": "def test_x():\n    pass\n"}]
    failed_test = 'File "tests/test_toolkit.py", line 10, in test_slugify\n    self.assertEqual(strings.slugify("Hi!"), "hi")'
    assert plan._blame(failed_test, parts, cur) == [0]  # the call in the failed line, not the test file (not a part)
    assert plan._blame("FAIL more cases: toolkit/numbers.py: clamp(1, 2, 3) is None", parts, cur) == [1]
    assert plan._blame("FAIL tests/test_more.py: 1 failed", parts, cur) == [2]
    assert plan._failed("FAILED (failures=1)\nFAIL more\n3 failed, 22 passed\ncurl: (22) error") == 3  # the last summary
    logs = plan.normalize({"parts": [{"title": "r", "files": ["report.py"], "instructions": "", "needs": []},
                                     {"title": "t", "files": ["test_report.py"], "instructions": "", "needs": []}]})
    out = "FAIL python3 -m unittest test_report: FAIL: test_numbers (test_report.TestReport)\nAssertionError: 3 != 4"
    assert plan._blame(out, logs, [{}, {}]) == [1]  # the test module by name; "report" alone is no file


def test_a_failure_further_on_scores_better():
    base = "FAIL: counting.py: AssertionError: sentence_count('a. b.') is 1, expected 2"
    key = plan._key(base)
    same = "FAIL: counting.py: AssertionError: sentence_count('a. b.') is 3, expected 2"
    further = "FAIL: cli sentences: AttributeError: 'X' object has no attribute 'y'"
    broken = "FAIL: counting.py:   File \"counting.py\", line 3\nSyntaxError: invalid syntax"
    tests = "Ran 4 tests\n\nFAILED (failures=1, errors=1)"
    assert plan._score(further, key) < plan._score(same, key) == plan._score(base, key) < plan._score(broken, key)
    assert plan._failed(tests) == 2 and plan._failed("3 failed, 9 passed in 0.2s") == 3 and plan._failed(base) is None
    assert plan._score(tests, key) < plan._score(further, key)  # tests that ran beat no count at all


def test_chat_files_streams_and_stops(tmp_path):
    """The reply comes streamed: the files are parsed from the deltas, and a stop closes the
    connection (the server cancels the generation) and returns no files."""
    import http.server
    import json as _json

    text = "FILE a.py\n```python\nA = 1\n```\n"

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["stream"] and body["response_format"]["type"] == "regex"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if getattr(H, "fail", False):
                self.wfile.write(b'data: {"error": {"message": "RuntimeError: Insufficient Memory"}}\n\ndata: [DONE]\n\n')
                return
            try:
                for i in range(0, len(text), 4):
                    d = {"choices": [{"delta": {"content": text[i:i + 4]}}]}
                    self.wfile.write(("data: " + _json.dumps(d) + "\n\n").encode())
                    self.wfile.flush()
                    time.sleep(0.02)
                self.wfile.write(b'data: {"choices": [], "usage": {"completion_tokens": 9}}\n\ndata: [DONE]\n\n')
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        out = plan.chat_files(url, "m", [{"role": "user", "content": "x"}], ["a.py"])
        assert out["files"] == {"a.py": "A = 1\n"} and out["usage"]["completion_tokens"] == 9 and not out["stopped"]
        stop = threading.Event()
        threading.Timer(0.1, stop.set).start()
        out = plan.chat_files(url, "m", [{"role": "user", "content": "x"}], ["a.py"], stop=stop)
        assert out["stopped"] and out["files"] == {}
        H.fail = True  # an error mid-stream (out of GPU memory) is an error, not an empty reply
        try:
            plan.chat_files(url, "m", [{"role": "user", "content": "x"}], ["a.py"])
            raise AssertionError("no error")
        except RuntimeError as e:
            assert "Insufficient Memory" in str(e)
    finally:
        srv.shutdown()


def test_solve_with_plan_direct_end_to_end(tmp_path, monkeypatch):
    """Plan, two samples of two parts, the whole samples fail, a combination passes, it is the
    workspace's; no repair and no agent is needed."""
    (tmp_path / "proj").mkdir()
    proj = tmp_path / "proj"
    monkeypatch.setattr(plan, "make_plan", lambda *a, **k: plan.normalize({"parts": [
        {"title": "a", "files": ["a.py"], "instructions": "A is one.", "needs": []},
        {"title": "b", "files": ["b.py"], "instructions": "B is two.", "needs": []}]}))
    versions = {("a.py", None): "A = 1", ("a.py", 0.6): "A = 0", ("b.py", None): "B = 0", ("b.py", 0.6): "B = 2"}

    def fake_chat_files(base_url, model, messages, paths, max_tokens=4096, timeout=900, temperature=None, stop=None, note=False):
        return {"files": {paths[0]: versions[(paths[0], temperature)]}, "usage": {"completion_tokens": 3}}

    monkeypatch.setattr(plan, "chat_files", fake_chat_files)
    race = _Race(None, "python3 -c 'import a, b; assert a.A == 1 and b.B == 2'", tmp_path / "run")
    race.run_dir.mkdir()
    report = plan.solve_with_plan(race, proj, "Write a.py and b.py.", "http://x", 6, 2, lambda s: None, log=lambda *a: None,
                                  samples=2)
    assert report["passed"] and report["samples"] == 2 and report["searched"] >= 1 and report["search_passed"]
    assert report["repairs"] == [] and report["fix_runs"] == []
    ws = race.attempts[0].ws
    assert (ws / "a.py").read_text() == "A = 1\n" and (ws / "b.py").read_text() == "B = 2\n"


def test_existing_files_the_task_does_not_name_are_not_written(tmp_path):
    (tmp_path / "sample.txt").write_text("The cat sat.\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "cli.py").write_text("")
    parts = plan.normalize({"parts": [
        {"title": "cli", "files": ["pkg/cli.py"], "instructions": "", "needs": []},
        {"title": "init", "files": ["pkg/__init__.py"], "instructions": "", "needs": [0]},
        {"title": "data", "files": ["sample.txt"], "instructions": "", "needs": [0, 1]},
        {"title": "tests", "files": ["tests/test_cli.py"], "instructions": "", "needs": [0, 2]}]})
    kept = plan.writable(parts, "Add a command to pkg/cli.py, with tests.", tmp_path)
    assert [p["files"] for p in kept] == [["pkg/cli.py"], ["tests/test_cli.py"]] and kept[1]["needs"] == [0]



def test_files_regex_and_parsing():
    """Parts write plain fenced code after a FILE line, under a regular expression: nothing to escape,
    no text before the first file; one expression for every one-file part (the path is checked when
    the reply is read)."""
    from locals.regex import compile_dfa

    rx = plan.files_regex(["a.py", "t/test_a.py"])
    assert rx == plan.files_regex(["b.py", "c.py"])  # compiled once for every part of two files
    dfa = compile_dfa(rx, 400_000)
    good = "FILE a.py\n```python\nd = {'k': 1}  # `x` and ``y``\n```\nFILE t/test_a.py\n```python\nimport a\n```\n"
    assert dfa.matches(good.encode())
    assert not dfa.matches(("Here it is:\n" + good).encode()) and not dfa.matches(good.replace("```\nFILE", "```\nfile", 1).encode())
    assert {m[1]: m[2] for m in plan._FILE_BLOCK.finditer(good)} == {"a.py": "d = {'k': 1}  # `x` and ``y``\n", "t/test_a.py": "import a\n"}


def test_a_plan_that_does_not_parse_is_asked_for_once_more(monkeypatch):
    """The server's loop guard can cut a plan whose instructions go round in circles: the plan is
    asked for once more, then there is none."""
    import io
    import json as _json

    asked = []

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=None):
        asked.append(_json.loads(req.data)["temperature"])
        return Reply(_json.dumps({"choices": [{"message": {"content": '{"parts": [{"title": "x", "instr'}}]}).encode())

    monkeypatch.setattr(plan.urllib.request, "urlopen", urlopen)
    parts = plan.make_plan("http://x", "locals/m", "Build an inventory tool with a command line and tests.", "Files: README.md")
    assert asked == [0.2, 0.7] and parts == []


def test_a_task_that_names_its_files_needs_no_planner(tmp_path, monkeypatch):
    """A part for each code file the task names, without a planning call, new or existing (a task
    that asks for tests in an existing test file names it)."""
    monkeypatch.setattr(plan.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_old.py").write_text("")
    task = "Write inventory.py and cli.py; add tests to tests/test_old.py."
    parts = plan.make_plan("http://x", "locals/m", task, "Files: README.md", project=tmp_path)
    assert [p["files"] for p in parts] == [["inventory.py"], ["cli.py"], ["tests/test_old.py"]]


def test_a_task_that_names_its_files_gets_no_invented_ones(tmp_path):
    """log_report names report.py and test_report.py: a part for run_tests.sh and a sample log
    (the planner wrote 4,096 tokens of it) is dropped; a new test file is allowed."""
    parts = plan.normalize({"parts": [
        {"title": "report", "files": ["report.py"], "instructions": "", "needs": []},
        {"title": "tests", "files": ["test_report.py"], "instructions": "", "needs": [0]},
        {"title": "extras", "files": ["run_tests.sh", "sample_logs.log"], "instructions": "", "needs": [0, 1]}]})
    kept = plan.writable(parts, "Write report.py that writes report.json. Also write test_report.py.", tmp_path)
    assert [p["files"] for p in kept] == [["report.py"], ["test_report.py"]]


def test_tests_are_written_after_the_code():
    """Code parts are written at once (the planner chained independent modules); tests written
    blind expected another error than the code raised, so they come after the code."""
    parts = plan.normalize({"parts": [{"title": "core", "files": ["inventory.py"], "instructions": "", "needs": []},
                                      {"title": "cli", "files": ["cli.py"], "instructions": "", "needs": []},
                                      {"title": "tests", "files": ["test_inventory.py"], "instructions": "", "needs": []}]})
    assert [p["needs"] for p in plan.tests_after_code(parts)] == [[], [], [0, 1]]
    parts = plan.normalize({"parts": [{"title": "core", "files": ["pkg/counting.py"], "instructions": "", "needs": []},
                                      {"title": "cli", "files": ["pkg/cli.py"], "instructions": "", "needs": [0]},
                                      {"title": "tests", "files": ["tests/test_counting.py"], "instructions": "", "needs": []}]})
    assert [p["needs"] for p in plan.tests_after_code(parts)] == [[], [], [0, 1]]  # code at once, even when it uses code


def test_a_bare_file_name_is_the_existing_file_in_a_subfolder(tmp_path):
    """"fix parser.py" for src/mylib/parser.py: the part owned a new top-level parser.py, and every edit of the
    real file was refused as not the agent's. The existing file of that name is the part's; a name several
    files have is the plan's call (no part from the name), and virtualenvs do not count."""
    from locals.plan import cover

    (tmp_path / "src/mylib").mkdir(parents=True)
    (tmp_path / "src/mylib/parser.py").write_text("x = 1\n")
    (tmp_path / ".venv/lib").mkdir(parents=True)
    (tmp_path / ".venv/lib/parser.py").write_text("")
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "util.py").write_text("")
    assert [p["files"] for p in cover([], "fix the bug in parser.py", tmp_path)] == [["src/mylib/parser.py"]]
    assert [p["files"] for p in cover([], "add a helper to util.py and write report.py", tmp_path)] == [["report.py"]]
    assert [p["files"] for p in cover([], "write new.py", tmp_path)] == [["new.py"]]


def test_files_the_task_names_past_the_part_cap_are_still_planned(tmp_path):
    """make_plan gives a part per named file, and the cap cut the last ones off: g.py and main.py were never
    written, no agent could touch them, and the check could never pass."""
    task = "Write a.py, b.py, c.py, d.py, e.py, f.py, g.py and main.py, which uses them all."
    parts = plan.make_plan("http://x", "m", task, "", project=tmp_path)
    assert len(parts) == 8
    capped = plan.cap_parts(parts, 6)
    assert len(capped) == 6 and [f for p in capped for f in p["files"]] == [f for p in parts for f in p["files"]]
    assert capped[-1]["files"] == ["f.py", "g.py", "main.py"]
    deps = [{"title": f"p{i}", "files": [f"{i}.py"], "instructions": "", "needs": n}
            for i, n in enumerate([[], [3], [0], [1], [2]])]
    capped = plan.cap_parts(deps, 3)  # parts 2-4 become one: who needed 3 needs it; its own needs inside it go
    assert [p["needs"] for p in capped] == [[], [2], [0, 1]]
    assert plan.cap_parts(parts, 8) is parts


def test_a_candidates_copy_is_removed_when_the_run_is_stopped(tmp_path):
    """Ctrl-C while a candidate's check ran: the copy is made in a daemon thread, whose finally never ran, and each
    stop left a whole copy of the project in the temporary folder."""
    import os
    import subprocess
    import sys

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.py").write_text("x = 0\n")
    child = (
        "import sys, threading, time\n"
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "from locals import plan\n"
        "team = SimpleNamespace(ws=Path(sys.argv[1]), race=SimpleNamespace(check='sleep 30'))\n"
        "threading.Thread(target=plan._Checks(team), args=({'a.py': 'x = 1'},), daemon=True).start()\n"
        "time.sleep(1)\n"
        "print(*plan._LIVE_DIRS, flush=True)\n"
        "raise KeyboardInterrupt\n")
    p = subprocess.run([sys.executable, "-c", child, str(ws)], capture_output=True, text=True, timeout=60,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    copy = p.stdout.strip()
    assert "locals-cand-" in copy and not os.path.exists(copy)


def test_the_last_version_is_not_cut_while_the_others_gave_nothing(monkeypatch):
    """The cut that stops a version running away came once most versions were done, whatever they gave: when the
    done ones had all failed, it stopped the only one that could still give a version."""
    monkeypatch.setattr(plan, "_TAIL_MIN", 0.2)
    for useful, cut in ((False, False), (True, True)):
        late = threading.Event()
        fast = [threading.Thread(target=lambda: None) for _ in range(2)]
        slow = threading.Thread(target=lambda: late.wait(3.0))  # ends at the cut, or after 3 s
        for th in fast + [slow]:
            th.start()
        t0 = time.time()
        plan._wait(fast + [slow], late, lambda s: None, {}, useful=lambda: useful)
        assert late.is_set() == cut and (time.time() - t0 < 2.0) == cut
