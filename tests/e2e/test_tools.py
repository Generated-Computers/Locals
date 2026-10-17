"""Each agent tool on real tasks, one at a time, the way a user runs them. A tool must finish without a
crash or a timeout, and every request it sends must be served without a server fault or a broken stream.
Whether the model then does the task is reported (report.md); the share of tasks done is checked in
test_tasks_done, since a small model fails some tasks on its own."""

import pytest
from agents import find_tool, run_task
from tasks import TASKS

TOOLS = ["opencode", "hermes", "openclaw", "kilo", "cline", "openhands", "aider"]
RESULTS = []


@pytest.mark.parametrize("task", ["fix_bug", "rename", "feature", "question"])
@pytest.mark.parametrize("tool", TOOLS)
def test_tool_on_task(stack, report, tool, task):
    if find_tool(tool, stack.env) is None:
        pytest.skip(f"{tool} is not installed (PATH or LOCALS_E2E_{tool.upper()}_BIN)")
    if tool != "opencode" and task in ("feature", "question"):
        pytest.skip("the longer tasks run on OpenCode only (time)")
    row = report.add(run_task(stack, tool, TASKS[task]))
    RESULTS.append(row)
    assert row["n_requests"] > 0, f"{tool} sent no request to the server; see {stack.runs}/{tool}-{task}.err"
    assert row["rc"] != "timeout", f"{tool} did not finish in time"
    assert not row["faults"], "\n".join(row["faults"][:10])
    assert row["rc"] == 0, f"{tool} exited with {row['rc']}; see {stack.runs}/{tool}-{task}.err"


def test_tasks_done(report):
    """At least half of the tasks done, over every tool that ran (Qwen3.5-4B does about two in three)."""
    ran = [r for r in RESULTS if r["rc"] != "timeout"]
    if not ran:
        pytest.skip("no tool ran")
    done = sum(r["ok"] for r in ran)
    report.note(f"agent tasks done: {done} of {len(ran)}")
    assert done >= len(ran) / 2, [(r["tool"], r["task"], r["why"]) for r in ran if not r["ok"]]
