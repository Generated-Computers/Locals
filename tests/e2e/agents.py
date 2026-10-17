"""Agent tools run the way a user runs them, one task to the end without their interface, through the
`locals` command (locals code --run for OpenCode, locals launch TOOL for the others)."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

from conftest import LOCALS, Stack, kill_tree, summarize
from tasks import Task, tree_digest

# the tool's command (looked up on the PATH, or in <NAME>_BIN) and how locals starts it on a task
COMMANDS = {"opencode": "opencode", "openclaw": "openclaw", "hermes": "hermes", "kilo": "kilo", "cline": "cline",
            "openhands": "openhands", "aider": "aider"}


def find_tool(tool: str, env: dict) -> Optional[str]:
    name = COMMANDS[tool]
    given = os.environ.get(f"LOCALS_E2E_{tool.upper()}_BIN")
    if given:
        return given
    found = shutil.which(name, path=env["PATH"])
    if not found and tool == "opencode":
        p = Path(env["HOME"]) / ".opencode" / "bin" / "opencode"
        found = str(p) if p.exists() else None
    return found


def argv(tool: str, task: str, d: Path, port: int) -> Tuple[List[str], Path]:
    """The command line and working directory for one task."""
    launch = [*LOCALS, "launch", "--port", str(port)]
    if tool == "opencode":
        return [*LOCALS, "code", str(d), "--run", task, "--port", str(port)], d
    if tool == "openclaw":
        return [*launch, "openclaw", "agent", "exec", task, "--cwd", str(d), "--json", "--timeout", "900"], d
    if tool == "hermes":
        return [*launch, "hermes", "chat", "--in", str(d), "--oneshot", "-Q", "-q", task, "--yolo", "--max-turns", "25"], d
    if tool == "kilo":
        return [*launch, "kilo", "run", "--auto", task], d
    if tool == "cline":
        return [*launch, "cline", "--yolo", "-c", str(d), task], d
    if tool == "openhands":
        return [*launch, "openhands", "--headless", "--json", "--always-approve", "-t", task], d
    if tool == "aider":
        files = [str(p.relative_to(d)) for p in sorted(d.rglob("*.py")) if "__pycache__" not in p.parts]
        return [*launch, "aider", "--message", task, "--yes-always", "--no-auto-commits", "--no-check-update",
                "--no-analytics", *files], d
    raise ValueError(tool)


def run_task(stack: Stack, tool: str, task: Task, timeout: float = 900, env: Optional[dict] = None, label: str = "",
             root: Optional[Path] = None) -> dict:
    """One tool on one task, to the end: its exit code and time, whether the check passes, and what the proxy
    saw of its requests (the wait for the first token, pauses, errors) and the server log said meanwhile."""
    env = dict(env or stack.env)
    d = task.copy_to(root or stack.runs)
    before = tree_digest(d)
    cmd, cwd = argv(tool, task.prompt, d, stack.port)
    if tool == "opencode":
        env["PWD"] = str(d)
    name = label or f"{tool}-{task.name}"
    out_path, err_path = stack.runs / f"{name}.out", stack.runs / f"{name}.err"
    mark, off = stack.proxy.mark(name), stack.log_offset()
    t0 = time.time()
    with open(out_path, "w") as out, open(err_path, "w") as err:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_tree(p)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                kill_tree(p, 9)
            rc = "timeout"
    wall = time.time() - t0
    output = out_path.read_text(errors="replace")
    ok, why = task.check(d, output, before)
    records = stack.proxy.since(mark)
    row = {"scenario": "agent", "tool": tool, "task": task.name, "rc": rc, "wall": wall, "ok": ok, "why": why, "dir": str(d),
           "server_problems": stack.problems(off), **summarize(records)}
    row["faults"] = row["faults"] + [f"server: {ln[:300]}" for ln in row["server_problems"]]
    return row


def popen_task(stack: Stack, tool: str, task: Task, env: Optional[dict] = None, name: str = "") -> Tuple[subprocess.Popen, Path, dict, str]:
    """Start one tool on one task without waiting (several agents at once): the process, the project, its
    digests before, and the run's name."""
    env = dict(env or stack.env)
    d = task.copy_to(stack.runs)
    before = tree_digest(d)
    cmd, cwd = argv(tool, task.prompt, d, stack.port)
    if tool == "opencode":
        env["PWD"] = str(d)
    name = name or f"{tool}-{task.name}"
    stack.proxy.projects[str(d)] = name
    out, err = open(stack.runs / f"{name}.out", "w"), open(stack.runs / f"{name}.err", "w")
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL, start_new_session=True)
    return p, d, before, name
