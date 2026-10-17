"""Coding tasks the way a user hands them to an agent, each on a small project with a check that says whether
the agent did it. The projects are copied fresh for every run (tests/e2e/projects)."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

PROJECTS = Path(__file__).parent / "projects"
REPO = Path(__file__).resolve().parents[2]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(d: Path) -> dict:
    """Every source file of a project and its digest (caches and the agents' own folders left out)."""
    return {str(p.relative_to(d)): _digest(p) for p in sorted(d.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and not any(s.startswith(".") for s in p.relative_to(d).parts)}


def _unittest(d: Path) -> Tuple[bool, str]:
    p = subprocess.run(["python3", "-m", "unittest", "-q"], cwd=d, capture_output=True, text=True, timeout=120)
    last = (p.stderr.strip().splitlines() or ["(no output)"])[-1]
    return p.returncode == 0, last


@dataclass
class Task:
    name: str
    project: str  # a folder of tests/e2e/projects, or "locals" (a copy of this package: a bigger code base)
    prompt: str
    check: Callable[[Path, str, dict], Tuple[bool, str]]  # (project, the agent's output, digests before) -> ok, why

    def copy_to(self, root: Path) -> Path:
        d = Path(tempfile.mkdtemp(prefix=f"{self.name}-", dir=root))
        if self.project == "locals":
            shutil.copytree(REPO / "locals", d / "locals", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copytree(PROJECTS / self.project, d, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        subprocess.run(["git", "init", "-q"], cwd=d)  # agents look for a repository root (OpenCode, Aider)
        subprocess.run(["git", "add", "-A"], cwd=d)
        subprocess.run(["git", "-c", "user.name=e2e", "-c", "user.email=e2e@example.com", "commit", "-qm", "start"], cwd=d)
        return d


def _fix_bug(d: Path, out: str, before: dict) -> Tuple[bool, str]:
    if tree_digest(d).get("test_textstats.py") != before.get("test_textstats.py"):
        return False, "the agent changed the tests"
    return _unittest(d)


def _feature(d: Path, out: str, before: dict) -> Tuple[bool, str]:
    ok, why = _unittest(d)
    if not ok:
        return False, f"tests: {why}"
    with tempfile.TemporaryDirectory() as t:
        for cmd in (["add", "a"], ["add", "b"], ["done", "2"]):
            p = subprocess.run(["python3", str(d / "todo.py"), *cmd], cwd=t, capture_output=True, text=True, timeout=60)
            if p.returncode != 0:
                return False, f"todo.py {' '.join(cmd)}: {(p.stderr.strip().splitlines() or [''])[-1][:200]}"
        shown = subprocess.run(["python3", str(d / "todo.py"), "list"], cwd=t, capture_output=True, text=True, timeout=60).stdout
    if shown.split() != "1. [ ] a 2. [x] b".split():
        return False, f"list printed {shown.strip()!r}"
    if "done" not in (d / "test_todo.py").read_text():
        return False, "no test for done"
    return True, "done works and is tested"


def _rename(d: Path, out: str, before: dict) -> Tuple[bool, str]:
    left = [str(p.relative_to(d)) for p in d.rglob("*.py") if "find_by_tag" in p.read_text()]
    if left:
        return False, f"find_by_tag still in {left}"
    if "def with_tag" not in (d / "inventory" / "store.py").read_text():
        return False, "no Store.with_tag"
    ok, why = _unittest(d)
    run = subprocess.run(["python3", "main.py"], cwd=d, capture_output=True, text=True, timeout=60)
    if run.returncode != 0:
        return False, "main.py fails: " + (run.stderr.strip().splitlines() or [""])[-1][:200]
    return ok, why


def _question(d: Path, out: str, before: dict) -> Tuple[bool, str]:
    if tree_digest(d) != before:
        return False, "the agent changed files in a read-only task"
    found = "context_fit" in out
    return found, "named context_fit" if found else f"answer without context_fit: {out[-300:]!r}"


TASKS = {
    "fix_bug": Task("fix_bug", "textstats",
                    "The unit tests fail (run: python3 -m unittest -q). Find the bug in textstats.py and fix it so that all "
                    "tests pass. Do not change test_textstats.py.", _fix_bug),
    "feature": Task("feature", "todo",
                    "Add a `done N` command to todo.py that marks to-do number N (1-based, the number `list` shows) as done "
                    "and saves the list. Add a unit test for it to test_todo.py, then run python3 -m unittest -q.", _feature),
    "rename": Task("rename", "inventory",
                   "Rename the method Store.find_by_tag to Store.with_tag everywhere in this project: its definition and "
                   "every call, the tests included. Then run python3 -m unittest -q and python3 main.py.", _rename),
    "question": Task("question", "locals",
                     "Which function in this code base computes how many tokens of context fit into the GPU memory next to "
                     "the model? Answer with the file name and the function name. Do not change any files.", _question),
}
