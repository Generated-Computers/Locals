"""locals run's harness without a model: the agents' replies are scripted."""

import json
from pathlib import Path
import threading
import time

from locals import harness


def _reply(name, **args):
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{name}{len(json.dumps(args))}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}],
        "usage": {"completion_tokens": 5}}


def _script(monkeypatch, by_agent):
    """by_agent: a function (the agent's last user prompt, step) -> reply."""
    lock = threading.Lock()
    steps = {}

    def chat(base_url, body, timeout=900):
        key = body["messages"][1]["content"] + str(body.get("temperature"))  # one agent: its prompt and its temperature
        with lock:
            steps[key] = steps.get(key, 0) + 1
            k = steps[key]
        return by_agent(body, k)

    monkeypatch.setattr(harness, "_chat", chat)


def test_an_agent_writes_only_its_files_and_finishes(tmp_path, monkeypatch):
    def agent(body, k):
        return [_reply("write_file", path="b.py", content="B = 1"), _reply("write_file", path="a.py", content="A = 2"),
                _reply("finish", summary="wrote a")][k - 1]

    _script(monkeypatch, agent)
    a = harness.Agent("w0", tmp_path, "http://x", "m", ["a.py"], "write a.py").run()
    assert (tmp_path / "a.py").read_text() == "A = 2\n" and not (tmp_path / "b.py").exists()
    assert a.summary == "wrote a" and a.wrote == ["a.py"] and "not one of your files" in a.messages[3]["content"]


def test_finish_is_refused_while_the_check_blames_the_part(tmp_path, monkeypatch):
    def agent(body, k):
        return [_reply("write_file", path="a.py", content="A = 1"), _reply("finish", summary="done"),
                _reply("write_file", path="a.py", content="A = 2"), _reply("finish", summary="fixed")][k - 1]

    _script(monkeypatch, agent)
    parts = [{"title": "a", "files": ["a.py"], "instructions": "", "needs": []}]
    verify = harness.part_verifier(tmp_path, "python3 -c 'import a; assert a.A == 2, \"a.py: A is wrong\"'", parts, 0)
    a = harness.Agent("w0", tmp_path, "http://x", "m", ["a.py"], "write a.py", verify=verify).run()
    assert a.refused == 1 and a.summary == "fixed" and (tmp_path / "a.py").read_text() == "A = 2\n"
    assert any("finish refused" in t for t in a.trace)


def test_a_stuck_agent_ends(tmp_path, monkeypatch):
    def agent(body, k):  # the server's loop guard answered without a call, twice
        return {"choices": [{"message": {"role": "assistant", "content": "The run call above was already made"}}], "usage": {}}

    _script(monkeypatch, agent)
    a = harness.Agent("w0", tmp_path, "http://x", "m", ["a.py"], "x", max_steps=10).run()
    assert a.steps == 2 and a.trace[-1] == "no call: stuck, ended"


def test_versions_in_copies_and_the_check_picks(tmp_path, monkeypatch):
    """Two agents per part, each in its own copy; the project gets the combination that passes."""
    (tmp_path / "p").mkdir()
    root = tmp_path / "p"

    def agent(body, k):
        prompt = body["messages"][1]["content"]
        mine = "a.py" if "Your files: a.py" in prompt else "b.py"
        good = body.get("temperature") is not None  # the second version of each part is the right one
        value = {"a.py": "A = 1" if good else "A = 0", "b.py": "B = 2" if good else "B = 0"}[mine]
        return [_reply("write_file", path=mine, content=value), _reply("finish", summary="ok")][min(k, 2) - 1]

    _script(monkeypatch, agent)
    run = harness.Run("Write a.py (A = 1) and b.py (B = 2).", root, "http://x", "m",
                      "python3 -c 'import a, b; assert a.A == 1 and b.B == 2'", log=lambda m: None)
    run.k, run.solos = 2, 0
    report = run.go()
    assert report["passed"] and (root / "a.py").read_text() == "A = 1\n" and (root / "b.py").read_text() == "B = 2\n"
    assert report["most_at_once"] == 4  # two parts x two versions, all at once
    assert not run._copies or not any(d.exists() for d in run._copies)  # the copies are gone


def test_a_write_gets_the_checks_word_and_a_pass_stops_the_others(tmp_path, monkeypatch):
    """After each write the check runs in the agent's copy; the first copy that passes stops every agent."""
    (tmp_path / "p").mkdir()
    root = tmp_path / "p"

    def agent(body, k):
        good = body.get("temperature") is not None
        if not good:  # the first version reads forever (a step takes a while); it is stopped when the other passes
            time.sleep(0.05)
            return _reply("list_files", path=".")
        return [_reply("write_file", path="a.py", content="A = 0"), _reply("write_file", path="a.py", content="A = 1"),
                _reply("finish", summary="done")][min(k, 3) - 1]

    _script(monkeypatch, agent)
    run = harness.Run("Write a.py (A = 1).", root, "http://x", "m",
                      "python3 -c 'import a; assert a.A == 1, \"a.py: A is wrong\"'", log=lambda m: None, max_steps=40)
    run.k = 2
    report = run.go()
    assert report["passed"] and (root / "a.py").read_text() == "A = 1\n"
    good = next(a for a in run.agents if a.temperature is not None)
    assert good.verified and any("still fails in your part" in m.get("content", "") for m in good.messages if m["role"] == "tool")
    assert all(a.steps < 40 for a in run.agents)  # stopped, not run out


def test_edit_file_changes_one_exact_piece(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("A = 1\nB = 1\n")

    def agent(body, k):
        return [_reply("edit_file", path="a.py", old="= 1", new="= 2"), _reply("edit_file", path="a.py", old="B = 1", new="B = 2"),
                _reply("finish", summary="ok")][k - 1]

    _script(monkeypatch, agent)
    a = harness.Agent("w0", tmp_path, "http://x", "m", ["a.py"], "fix b").run()
    assert (tmp_path / "a.py").read_text() == "A = 1\nB = 2\n" and "2 times" in a.messages[3]["content"]


def test_a_whole_task_agent_races_the_parts_and_wins(tmp_path, monkeypatch):
    """The whole-task agent (both files its own) passes first; the parts stop and its files are taken."""
    (tmp_path / "p").mkdir()
    root = tmp_path / "p"

    def agent(body, k):
        prompt = body["messages"][1]["content"]
        if "Your files: a.py, b.py" in prompt:
            return [_reply("write_file", path="a.py", content="A = 1"), _reply("write_file", path="b.py", content="B = 2"),
                    _reply("finish", summary="both")][min(k, 3) - 1]
        time.sleep(0.05)  # the parts' agents are slow and never right
        return _reply("list_files", path=".")

    _script(monkeypatch, agent)
    run = harness.Run("Write a.py (A = 1) and b.py (B = 2).", root, "http://x", "m",
                      "python3 -c 'import a, b; assert a.A == 1 and b.B == 2'", log=lambda m: None, max_steps=40)
    report = run.go()
    assert report["passed"] and report["won_by"].startswith("s")
    assert (root / "a.py").read_text() == "A = 1\n" and (root / "b.py").read_text() == "B = 2\n"
    assert all(a.steps < 40 for a in run.agents)


def test_edit_file_forgives_whitespace_once():
    text = "def f():\n    return 1\n\n\ndef g():\n    return 2\n"
    assert harness._loose_replace(text, "def g():\n  return 2  ", "def g():\n    return 3") == \
        "def f():\n    return 1\n\n\ndef g():\n    return 3\n"
    assert harness._loose_replace(text, "return", "x") is None  # whole lines only
    assert harness._loose_replace(text, "def h():", "x") is None


def test_edit_file_forgives_blank_lines_and_keeps_indentation():
    text = "def f(x):\n    if x:\n\n        return 1\n    return 2\n"
    # the blank line left out and the indentation lost, in old and new alike
    assert harness._loose_replace(text, "if x:\nreturn 1", "if x > 0:\n    return 10") == \
        "def f(x):\n    if x > 0:\n        return 10\n    return 2\n"
    # indented too far, by the same amount in old and new
    assert harness._loose_replace(text, "            return 1\n        return 2", "            return 11\n        return 22") == \
        "def f(x):\n    if x:\n\n        return 11\n    return 22\n"
    assert harness._loose_replace("a = 1\n\nb = 2\nx\na = 1\n\nb = 2\n", "a = 1\nb = 2", "c") is None  # twice: ambiguous


def test_a_missed_edit_shows_the_closest_lines(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("import re\n\n\ndef count(text):\n    parts = re.split(r'[.!?]', text)\n    return len(parts)\n")

    def agent(body, k):
        return [_reply("edit_file", path="a.py", old="parts = text.split('.')\nreturn len(parts)", new="x"),
                _reply("finish", summary="ok")][k - 1]

    _script(monkeypatch, agent)
    a = harness.Agent("w0", tmp_path, "http://x", "m", ["a.py"], "fix").run()
    said = a.messages[3]["content"]
    assert "not in a.py" in said and "parts = re.split(r'[.!?]', text)" in said and "return len(parts)" in said
    assert "import re" not in said  # the region, not the whole file


def test_a_lone_files_path_may_be_left_out(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("A = 1\n")

    def agent(body, k):
        return [_reply("edit_file", old="A = 1", new="A = 2"), _reply("finish", summary="ok")][k - 1]

    _script(monkeypatch, agent)
    harness.Agent("w0", tmp_path, "http://x", "m", ["a.py"], "fix a").run()
    assert (tmp_path / "a.py").read_text() == "A = 2\n"


def test_a_failing_run_leaves_the_project_as_it_was(tmp_path, monkeypatch):
    """The agents' work goes to the project only when the check passes; else it is kept aside."""
    (tmp_path / "p").mkdir()
    root = tmp_path / "p"
    (root / "a.py").write_text("A = 0  # the user's own\n")

    def agent(body, k):
        return [_reply("write_file", path="a.py", content="A = 5"), _reply("finish", summary="done")][min(k, 2) - 1]

    _script(monkeypatch, agent)
    run = harness.Run("Fix a.py (A = 1).", root, "http://x", "m", "python3 -c 'import a; assert a.A == 1, \"a.py: A\"'",
                      log=lambda m: None, keep_dir=tmp_path / "kept", rounds=0)
    run.solos = 0
    report = run.go()
    assert report["passed"] is False and report["changed"] == []
    assert (root / "a.py").read_text() == "A = 0  # the user's own\n"
    assert (Path(report["kept"]) / "a.py").read_text() == "A = 5\n"
    assert not any(d.exists() for d in run._copies)


def test_the_tests_that_exist_are_the_check_not_a_part(tmp_path):
    (tmp_path / "test_inv.py").write_text("x = 1\n")
    run = harness.Run("fix inv.py so that test_inv.py passes", tmp_path, "http://x", "m", "python3 -m pytest -q", log=lambda m: None)
    parts = run._owned([{"title": "inv", "files": ["inv.py"], "instructions": "", "needs": []},
                        {"title": "tests", "files": ["test_inv.py"], "instructions": "", "needs": []},
                        {"title": "again", "files": ["inv.py", "new.py"], "instructions": "", "needs": []}])
    assert [p["files"] for p in parts] == [["inv.py"], ["new.py"]] and run.report["tests_kept"] == ["test_inv.py"]
    run = harness.Run("update the tests in test_inv.py", tmp_path, "http://x", "m", "python3 -m pytest -q", log=lambda m: None)
    assert run._owned([{"title": "t", "files": ["test_inv.py"], "instructions": "", "needs": []}])[0]["files"] == ["test_inv.py"]


def test_paths_an_agent_gives():
    assert harness._rel("./.env") == ".env" and harness._rel(".github/x.yml") == ".github/x.yml"
    assert harness._rel("./") == "" and harness._rel("a/b/") == "a/b"
    assert harness._rel("/etc/passwd") is None and harness._rel("../x") is None and harness._rel("a/../../x") is None


def test_a_task_without_files_is_one_part(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "make_plan", lambda *a, **k: [])
    run = harness.Run("do something", tmp_path, "http://x", "m", None, log=lambda m: None)
    parts = run.plan()
    assert len(parts) == 1 and parts[0]["files"] == []


def test_a_check_that_ran_no_tests_fails(tmp_path):
    from locals.solve import run_check

    ok, out, _ = run_check("echo 'Ran 0 tests in 0.000s'; echo OK", tmp_path)
    assert not ok and "ran no tests" in out
    ok, _, _ = run_check("sleep 5", tmp_path, timeout=0.5)
    assert not ok


def test_a_file_edited_during_the_run_is_not_overwritten(tmp_path, monkeypatch):
    (tmp_path / "p").mkdir()
    root = tmp_path / "p"
    (root / "a.py").write_text("A = 0\n")

    def agent(body, k):
        if k == 1:
            (root / "a.py").write_text("A = 0  # the user's edit, meanwhile\n")
        return [_reply("write_file", path="a.py", content="A = 1"), _reply("finish", summary="done")][min(k, 2) - 1]

    _script(monkeypatch, agent)
    run = harness.Run("Fix a.py (A = 1).", root, "http://x", "m", None, log=lambda m: None, keep_dir=tmp_path / "kept")
    run.solos = 0
    report = run.go()
    assert (root / "a.py").read_text() == "A = 0  # the user's edit, meanwhile\n"
    assert report["skipped"] == ["a.py"] and (Path(report["kept"]) / "a.py").read_text() == "A = 1\n"


def test_the_tests_that_are_the_check_are_out_of_reach_of_every_agent(tmp_path, monkeypatch):
    (tmp_path / "test_a.py").write_text("x = 1\n")
    a = harness.Agent("s0", tmp_path, "http://x", "m", harness._AnyFile(), "x", readonly={"test_a.py"})
    assert "tests the check runs" in a._tool("write_file", {"path": "test_a.py", "content": "pass"})
    assert (tmp_path / "test_a.py").read_text() == "x = 1\n"


def test_which_tasks_ask_for_tests():
    from locals.plan import asks_test_changes as asks

    assert asks("3. Tests for both functions in tests/test_counting.py.") and asks("Update test_api.py")
    assert asks("write inventory.py, cli.py and test_inventory.py") and asks("add tests for parse() and make them pass")
    assert not asks("fix the failing tests") and not asks("Make the tests for parse() pass")
    assert not asks("fix inventory.py so that test_inventory.py passes")
    assert asks("Rename the function calc_area to area_of everywhere in this project: its definition, the imports, "
                "the calls and the tests. Then run the tests")


def test_a_fixer_is_shown_its_own_failures(tmp_path):
    (tmp_path / "toolkit").mkdir()
    (tmp_path / "toolkit" / "dates.py").write_text("def days_between(a, b):\n    return 1\n")
    (tmp_path / "toolkit" / "files.py").write_text("def count_lines(path):\n    return 3\n")
    out = ('Traceback (most recent call last):\n  File "./tests/test_toolkit.py", line 52, in test_count_lines\n'
           "    self.assertEqual(files.count_lines(p), 4)\nAssertionError: 3 != 4\n\nFAILED (failures=1)\n"
           "FAIL more cases: toolkit/dates.py: dates.days_between('2024-02-28', '2024-03-01') is 1, expected 2\n"
           "FAIL more cases: toolkit/files.py: files.count_lines(CFG) is 2, expected 3\n2 failed, 22 passed\n")
    parts = [{"title": "dates", "files": ["toolkit/dates.py"], "needs": [], "instructions": ""},
             {"title": "files", "files": ["toolkit/files.py"], "needs": [], "instructions": ""}]
    dates = harness.fixer_prompt("t", parts, 0, "", "check", out, tmp_path, [0, 1])
    mine = dates.split("What fails in your files (from the output above):\n")[1]
    assert "days_between('2024-02-28'" in mine and "count_lines" not in mine.split("\n\n")[0]
    files = harness.fixer_prompt("t", parts, 1, "", "check", out, tmp_path, [0, 1])
    mine = files.split("What fails in your files (from the output above):\n")[1].split("\n\n")[0]
    assert 'File "./tests/test_toolkit.py", line 52' in mine and "AssertionError: 3 != 4" in mine
    assert "count_lines(CFG) is 2" in mine and "days_between" not in mine


def test_what_the_code_writes_when_run_comes_with_it(tmp_path, monkeypatch):
    """report.py writes report.json, as the task says, when the agent runs it: the file comes with the
    agent's version into the project (the check passed in the agent's copy, and failed without it)."""
    root = tmp_path / "p"
    root.mkdir()
    (root / "sales.csv").write_text("region,units\nNorth,3\n")
    code = "import json\njson.dump({'units': 3}, open('report.json', 'w'))\n"

    def agent(body, k):
        return [_reply("write_file", path="report.py", content=code), _reply("run", command="python3 report.py"),
                _reply("finish", summary="done")][min(k, 3) - 1]

    _script(monkeypatch, agent)
    run = harness.Run("Write report.py that reads sales.csv and writes report.json with the units. Run it.", root, "http://x",
                      "m", "python3 -c 'import json; assert json.load(open(\"report.json\")) == {\"units\": 3}, \"report.json\"'",
                      log=lambda m: None, race=0)
    report = run.go()
    assert report["passed"] and json.loads((root / "report.json").read_text()) == {"units": 3}
    assert (root / "sales.csv").read_text() == "region,units\nNorth,3\n"


def test_a_round_that_changes_the_failure_is_progress():
    from locals.harness import _progress

    ties = "FAIL: test_top_words (tests.test_counting.TestCounting)\nAssertionError: [('banana', 2)] != [('the', 2)]\n1 failed, 7 passed"
    code = "FAIL: test_sentence_count (tests.test_counting.TestCounting)\nAssertionError: 1 != 2\n1 failed, 7 passed"
    assert _progress(code, ties)  # the tests were fixed and found the code wrong: one more round
    assert not _progress(ties.replace("2)]", "3)]"), ties)  # the same failure, other values: stuck
    assert _progress("0 failed, 8 passed", code) and not _progress("2 failed, 6 passed", code)


def test_a_stuck_repair_reaches_the_code_the_failing_part_uses(tmp_path):
    from locals.harness import _uses

    parts = [{"title": "inventory", "files": ["inventory.py"], "needs": [], "instructions": ""},
             {"title": "cli", "files": ["cli.py"], "needs": [], "instructions": ""},
             {"title": "tests", "files": ["test_inventory.py"], "needs": [0, 1], "instructions": ""}]
    cur = [{"inventory.py": "import json\n"}, {"cli.py": "import sys\nfrom inventory import Inventory\n"},
           {"test_inventory.py": "from inventory import Inventory\n"}]
    assert _uses(parts, cur, 1) == [0] and _uses(parts, cur, 0) == []
    pkg = [{"files": ["textstats/counting.py"]}, {"files": ["textstats/cli.py"]}]
    for imp in ("from . import counting", "from .counting import top_words", "from textstats.counting import top_words",
                "import textstats.counting"):
        assert _uses(pkg, [{"textstats/counting.py": ""}, {"textstats/cli.py": imp + "\n"}], 1) == [0], imp
    (tmp_path / "inventory.py").write_text("class Inventory:\n    pass\n")
    said = harness.fixer_prompt("t", parts, 0, "", "check", "FAIL cli.py: count apple printed 0", tmp_path, [1, 0], ["cli.py"])
    assert "What fails shows in cli.py, which uses your files" in said
    assert "which uses your files" not in harness.fixer_prompt("t", parts, 1, "", "check", "FAIL cli.py", tmp_path, [1])


def test_agent_commands_write_only_in_their_copy(tmp_path):
    import os
    import shutil

    import pytest

    from locals.solve import run_shell

    if not shutil.which("sandbox-exec"):
        pytest.skip("no sandbox-exec (not macOS)")
    outside = Path.home() / f".locals-sandbox-test-{os.getpid()}"
    try:
        code, out = run_shell(f"echo a > inside.txt; echo b > {outside}; python3 -c \"import tempfile; "
                              f"open(tempfile.mkdtemp() + '/x', 'w').write('1'); print('tmp ok')\"", tmp_path, 30, confine=tmp_path)
        assert (tmp_path / "inside.txt").read_text() == "a\n" and "tmp ok" in out
        assert not outside.exists() and "Operation not permitted" in out
    finally:
        outside.unlink(missing_ok=True)
    code, out = run_shell(f"echo b > {outside}", tmp_path, 30)  # the check, the user's own command: not confined
    assert outside.exists()
    outside.unlink()
