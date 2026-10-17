"""A working session: one project, the user asking for one thing after another in the same OpenCode session
(`locals code --run ... --continue`). Each follow-up resends the whole conversation, so it is fast only if
the server finds the conversation in its prompt cache: the first request of every follow-up turn must come
mostly from the cache, and its first token soon."""

import time

from conftest import summarize
from tasks import TASKS

TURNS = [
    "Read todo.py and tell me in two sentences what it does.",
    "Add a `clear` command to todo.py that removes every to-do and saves the empty list.",
    "Add a unit test for `clear` to test_todo.py and run python3 -m unittest -q.",
    "Add a `count` command that prints how many to-dos are not done yet, then run the tests again.",
    "Show me the final usage line of every command, as a short list.",
]


def test_follow_up_turns_come_from_the_cache(stack, report):
    d = TASKS["feature"].copy_to(stack.runs)
    env = dict(stack.env, PWD=str(d))
    firsts = []
    for i, turn in enumerate(TURNS):
        mark, off = stack.proxy.mark(f"turn-{i}"), stack.log_offset()
        t0 = time.time()
        args = ["code", str(d), "--run", turn, "--port", stack.port] + (["--continue"] if i else [])
        p = stack.locals(*args, cwd=d, env=env, timeout=900)
        (stack.runs / f"turn-{i}.out").write_text(p.stdout + "\n--- stderr\n" + p.stderr)
        recs = [r for r in stack.proxy.since(mark) if (r.get("tools") or 0) > 0]  # the agent's own requests (not titles)
        s = summarize(recs)
        report.add({"scenario": "session", "tool": "opencode", "task": f"turn {i + 1}", "rc": p.returncode, "ok": p.returncode == 0,
                    "wall": time.time() - t0, **s, "faults": s["faults"] + stack.problems(off)})
        assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
        assert not s["faults"], s["faults"]
        if recs:
            u = recs[0].get("usage") or {}
            firsts.append((i, u.get("prompt_tokens") or 0, ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                           recs[0].get("first_token") or 0.0))
    report.note("session, first request of each turn (prompt, cached, first token s): " +
                ", ".join(f"turn {i + 1}: {n}/{c}/{t:.1f}" for i, n, c, t in firsts))
    for i, n, c, t in firsts[1:]:
        assert c >= 0.5 * n, f"turn {i + 1}: only {c} of {n} prompt tokens came from the cache"
