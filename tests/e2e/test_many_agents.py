"""Several agents at once, as a user runs them: a terminal per project, each with its own `locals code --run`
(the same HOME, so the same OpenCode settings and data), and other tools beside them. Every agent must
finish, every request be served without a fault, and the batch must beat running them one after another."""

import time

import pytest
from agents import escaped, find_tool, popen_task, run_task
from conftest import kill_tree, summarize
from tasks import TASKS


def _wait_all(stack, jobs, timeout):
    t0, rows = time.time(), []
    for p, d, before, name, tool, task in jobs:
        try:
            rc = p.wait(timeout=max(1, timeout - (time.time() - t0)))
        except Exception:  # noqa: BLE001
            kill_tree(p)
            rc = "timeout"
        ok, why = task.check(d, (stack.runs / f"{name}.out").read_text(errors="replace"), before)
        rows.append({"tool": tool, "task": task.name, "name": name, "rc": rc, "ok": ok, "why": why, "end": time.time() - t0})
    return rows


def _run_at_once(stack, report, plan, scenario, timeout=1500):
    mark, off = stack.proxy.mark(scenario), stack.log_offset()
    t0 = time.time()
    jobs = []
    for i, (tool, task) in enumerate(plan):
        p, d, before, name = popen_task(stack, tool, TASKS[task], name=f"{scenario}-{i}-{tool}-{task}")
        jobs.append((p, d, before, name, tool, TASKS[task]))
    rows = _wait_all(stack, jobs, timeout)
    wall = time.time() - t0
    out_of_bounds = escaped()
    assert not out_of_bounds, f"an agent changed files outside its project (put back): {out_of_bounds}"
    records = stack.proxy.since(mark)
    problems = stack.problems(off)
    for r in rows:
        mine = [x for x in records if x.get("project") == r["name"]]
        report.add({"scenario": scenario, "wall": r["end"], **summarize(mine), **{k: r[k] for k in ("tool", "task", "rc", "ok", "why")}})
    total = summarize(records)
    peak = stack.stats()["memory"]["gpu_peak_mb"] / 1e3
    report.note(f"{scenario}: {len(plan)} agents at once in {wall:.0f} s; {sum(r['ok'] for r in rows)} done; "
                f"{total['n_requests']} requests, first token median {total['ttft_median']:.1f} s, max {total['ttft_max']:.1f} s; "
                f"GPU peak {peak:.1f} GB")
    return rows, wall, total, problems


def test_four_tools_at_once(stack, report):
    """Two OpenCode terminals and two other tools, all at once on different projects."""
    plan = [("opencode", "fix_bug"), ("opencode", "rename")]
    plan += [(t, k) for t, k in (("hermes", "fix_bug"), ("openclaw", "rename"), ("kilo", "fix_bug")) if find_tool(t, stack.env)][:2]
    rows, wall, total, problems = _run_at_once(stack, report, plan, "four-at-once")
    assert not problems, problems[:10]
    assert not total["faults"], total["faults"][:10]
    assert all(r["rc"] == 0 for r in rows), [(r["name"], r["rc"]) for r in rows]


def test_six_opencode_terminals_at_once(stack, report):
    """Six `locals code --run` at once (six terminals, one HOME), against the same task alone first."""
    alone = run_task(stack, "opencode", TASKS["fix_bug"], label="alone-opencode-fix_bug")
    report.add({**alone, "scenario": "alone"})
    rows, wall, total, problems = _run_at_once(stack, report, [("opencode", "fix_bug")] * 6, "six-opencode")
    report.note(f"six OpenCode agents at once: {wall:.0f} s, one alone {alone['wall']:.0f} s "
                f"({6 * alone['wall'] / wall:.1f}x the work per minute)")
    assert not problems, problems[:10]
    assert not total["faults"], total["faults"][:10]
    assert all(r["rc"] == 0 for r in rows), [(r["name"], r["rc"]) for r in rows]
    if alone["ok"]:
        assert wall < 6 * alone["wall"] * 0.75, f"six at once took {wall:.0f} s, one alone {alone['wall']:.0f} s"
