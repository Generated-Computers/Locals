"""locals solve: agents work on one task in copies of the project, the first to pass the check wins.

    locals solve "make the failing tests pass" --check "python3 -m pytest -q"

Every attempt is an OpenCode session (one ``opencode serve``, many sessions) in a
workspace of its own: a repository of its own (git clone --shared) with the project's uncommitted changes, or a
copy. The sessions decode together in one batch of the locals server, and their
common prompt (the agent's instructions and the task) is computed once. When an
agent stops, the check runs in its workspace; if it fails, the agent sees the
output and gets ``--fixes`` more turns. The first attempt whose check passes wins:
the others are aborted and its changes are applied to the project.

One agent starts; each attempt that fails is replaced by ``--burst`` fresh agents
told what failed (the Ralph loop: a fresh context each time, the check as
back-pressure), never more than ``--n`` at once and ``--attempts`` in all;
``--hedge S`` adds an agent after S seconds without a result, ``--start N`` starts
N at once. The defaults run them one at a time, which was fastest on one Mac.

Measured on an M2 Pro: with Qwen3.5-4B two agents each run
at 0.9x the speed of one alone, four at 0.6x; on ten real tasks fresh agents one
after another solved all ten with either model, fastest, and agents side by side
took 1.2-1.9x as long for the same result.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Not part of an attempt's work: tool and cache directories (copied as they are into a
# copy-based workspace only when they hold the project's own files, i.e. never).
SKIP_DIRS = {".git", ".locals", "node_modules", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
             ".ruff_cache", ".tox", ".idea", ".vscode", ".DS_Store", ".opencode"}

# One per attempt, cycled: the same task approached a little differently, so that the
# attempts do not all fail the same way.
HINTS = [
    "",
    "Read every file the task involves before you change anything.",
    "Work in small steps: one change, then run the check, then the next change.",
    "If your first idea does not work, step back and try a different approach.",
    "Keep the change as small as possible; do not rewrite what already works.",
    "Write down for yourself what the check expects before you start.",
]


# --- workspaces --------------------------------------------------------------------------------

# Left out wherever they are; the others in SKIP_DIRS only where they are what their name says (a
# virtualenv has pyvenv.cfg; .idea and .vscode at the top): a source package may be called env.
_ALWAYS_SKIP = {".git", ".locals", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
                ".DS_Store", ".opencode"}


def _skip_dir(parent: Path, name: str, root: Path) -> bool:
    if name in _ALWAYS_SKIP:
        return True
    if name in (".venv", "venv", "env"):
        return (parent / name / "pyvenv.cfg").exists()
    return name in SKIP_DIRS and parent == root


def manifest(root: Path) -> Dict[str, str]:
    """relative path -> sha1 of every file of the project (tool and cache directories left out)."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _skip_dir(Path(dirpath), d, root)]
        for f in filenames:
            if f in SKIP_DIRS:
                continue
            p = Path(dirpath) / f
            h = manifest_one(root, str(p.relative_to(root)))
            if h is not None:
                out[str(p.relative_to(root))] = h
    return out


def read_source(p: Path) -> str:
    """A project file's text for an agent: its lines with \n (as the model writes them), and bytes that are not
    UTF-8 kept as they are (surrogates), so writing it back changes nothing the agent did not change."""
    return p.read_text(encoding="utf-8", errors="surrogateescape")


def write_source(p: Path, text: str) -> None:
    """``text`` into ``p`` with the file's own line endings (CRLF stays CRLF) and its bytes that were not UTF-8 as
    they were. (One edit of a Windows file made every line \n, and a Latin-1 é became U+FFFD.)"""
    crlf = False
    try:
        with open(p, "rb") as f:
            head = f.read(65536)
        crlf = head.count(b"\r\n") * 2 > head.count(b"\n")
    except OSError:
        pass
    with open(p, "w", encoding="utf-8", errors="surrogateescape", newline="\r\n" if crlf else "\n") as f:
        f.write(text)


def _plain(entry: Optional[str]) -> bool:
    """A manifest entry of a plain readable file (or none: a file that is not there)."""
    return entry is None or not (entry.startswith("link:") or entry == "unreadable")


def changes(base: Dict[str, str], now: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    """(changed, added, deleted) relative paths between two manifests."""
    changed = sorted(k for k in now if k in base and now[k] != base[k])
    added = sorted(k for k in now if k not in base)
    deleted = sorted(k for k in base if k not in now)
    return changed, added, deleted


def _git(project: Path, *args, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(project), *args], capture_output=True, text=True, check=check)


def git_root(project: Path) -> Optional[Path]:
    try:
        p = _git(project, "rev-parse", "--show-toplevel")
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(p.stdout.strip()).resolve()


def make_workspace(project: Path, dest: Path) -> str:
    """A copy of the project as it is now (uncommitted changes included) at ``dest``: a repository of its own when
    the project is a git repository's root (``git clone --shared``: the project's objects, read-only, but its own
    refs, stash, config and hooks), else a copy. (A worktree shared the user's stash, config and branches: an
    agent's `git stash pop` took the user's newest stash into an attempt that was then deleted.)"""
    if git_root(project) == project and _git(project, "rev-parse", "--verify", "HEAD", check=False).returncode == 0:
        head = _git(project, "rev-parse", "HEAD").stdout.strip()
        quiet = ["-c", "core.hooksPath=/dev/null", "-c", "advice.detachedHead=false"]  # (no hook of the user's runs)
        subprocess.run(["git", *quiet, "clone", "--shared", "--no-checkout", "--quiet", str(project), str(dest)],
                       capture_output=True, text=True, check=True)
        subprocess.run(["git", "-C", str(dest), *quiet, "checkout", "--quiet", "--detach", head],
                       capture_output=True, text=True, check=True)
        # the working tree as the user has it: modified, deleted and untracked files
        st = _git(project, "status", "--porcelain", "-z", "--untracked-files=all").stdout.split("\0")
        i = 0
        while i < len(st):
            entry = st[i]
            i += 1
            if len(entry) < 4:
                continue
            code, path = entry[:2], entry[3:]
            if code.startswith("R"):  # a rename: the next entry is the old path
                old = st[i]
                i += 1
                (dest / old).unlink(missing_ok=True)
            src, dst = project / path, dest / path
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            elif not src.exists():
                dst.unlink(missing_ok=True)
        return "clone"
    shutil.copytree(project, dest, symlinks=True,
                    ignore=lambda d, names: {n for n in names if (Path(d) / n).is_dir() and _skip_dir(Path(d), n, project)}
                    | {n for n in names if n == ".DS_Store"})
    return "copy"


def cleanup_stale_runs(project: Path, root: Path) -> int:
    """Workspaces that solve runs of ``project`` left when their terminal was closed or they were killed (no cleanup
    runs then): git worktrees still registered in the project (git worktree list), and copies of it. Their
    transcripts and diffs stay. A run counts as dead when the pid it wrote is no running process. How many."""
    removed = 0
    mine = re.compile(re.escape(project.name) + r"-\d{8}-\d{6}")  # (not app-v2's runs for project app)
    for run in root.glob(f"{project.name}-*"):
        owner = run / "project"  # (two projects of the same name in different folders)
        try:
            if not mine.fullmatch(run.name) or (owner.exists() and owner.read_text() != str(project)):
                continue
        except OSError:
            continue
        try:
            os.kill(int((run / "pid").read_text()), 0)  # (only asks whether it still runs)
            continue
        except (OSError, ValueError):  # no pid file (an older run, or done) or a dead one
            if not (run / "pid").exists():
                continue
        for ws in [d for d in run.iterdir() if d.is_dir() and d.name[:1] in ("a", "c", "p")]:  # aN, checkN, probeN
            remove_workspace(project, ws, "worktree" if (ws / ".git").is_file() else "copy")  # (a worktree: older runs)
            removed += 1
        (run / "pid").unlink(missing_ok=True)
    return removed


def remove_workspace(project: Path, ws: Path, kind: str) -> None:
    if kind == "worktree":
        _git(project, "worktree", "remove", "--force", str(ws), check=False)
    shutil.rmtree(ws, ignore_errors=True)
    if kind == "worktree":
        _git(project, "worktree", "prune", check=False)  # one whose folder went another way


def unified_diff(base_root: Path, ws: Path, files: List[str], deleted: List[str]) -> str:
    out = []
    for rel in files + deleted:
        a, b = base_root / rel, ws / rel
        try:
            old = a.read_text().splitlines(keepends=True) if a.exists() else []
            new = b.read_text().splitlines(keepends=True) if b.exists() else []
        except (OSError, UnicodeDecodeError):
            out.append(f"Binary file {rel} differs\n")
            continue
        out.extend(difflib.unified_diff(old, new, f"a/{rel}", f"b/{rel}"))
    return "".join(out)


# --- the check ----------------------------------------------------------------------------------

def detect_check(project: Path) -> Optional[str]:
    """A command that tells whether the project works, from what the project has."""
    has_py_tests = any(p.name.startswith("test") and p.suffix == ".py" for p in project.rglob("test*.py")
                       if not (set(p.relative_to(project).parts) & SKIP_DIRS))
    if has_py_tests:
        try:
            subprocess.run(["python3", "-c", "import pytest"], check=True, capture_output=True, cwd=project, timeout=60)
            return "python3 -m pytest -q -x"
        except (OSError, subprocess.SubprocessError):
            top = any(project.glob("test*.py"))
            tests = project / "tests"
            if not top and tests.is_dir() and any(tests.rglob("test*.py")):
                # -t . only for a package (without __init__.py: "Start directory is not importable");
                # python3 -m puts the project on the path either way
                return "python3 -m unittest discover -q -s tests" + (" -t ." if (tests / "__init__.py").exists() else "")
            return "python3 -m unittest discover -q"
    pkg = project / "package.json"
    if pkg.exists():
        try:
            if (json.loads(pkg.read_text()).get("scripts") or {}).get("test"):
                return "npm test --silent"
        except ValueError:
            pass
    if (project / "Cargo.toml").exists():
        return "cargo test -q"
    if (project / "go.mod").exists():
        return "go test ./..."
    return None


_LIVE: set = set()  # process groups of commands running now (killed at exit: a Ctrl-C reaches the main thread only)


def _kill_live() -> None:
    import signal

    for pid in list(_LIVE):
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass


import atexit  # noqa: E402

atexit.register(_kill_live)


# where OpenCode keeps its own state (sessions, logs, config): written by an `opencode serve` run under the sandbox
OPENCODE_DIRS = (".local/share/opencode", ".local/state/opencode", ".config/opencode", ".opencode")


def write_sandbox(root: Path, *more: Path) -> str:
    """A macOS sandbox profile (sandbox-exec) for an agent's commands: everything allowed but writing
    outside ``root`` (the agent's copy; ``more``: other folders), the temporary folders and the tools' caches. A
    command that an agent ran with the absolute path the check named wrote into the folder of the check's script."""
    import tempfile

    def q(path) -> str:
        return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'

    home = Path.home()
    places = {os.path.realpath(r) for r in (root, *more)}
    places |= {os.path.realpath(tempfile.gettempdir()), "/private/tmp", "/private/var/tmp",
              os.path.realpath(home / "Library" / "Caches"), os.path.realpath(home / ".cache")}
    # package managers' own stores (a check's `cargo test` or `npm test` fetches into them)
    places |= {os.path.realpath(home / d) for d in (".cargo", ".npm", ".pnpm-store", ".yarn", ".bun", ".gradle", ".m2",
                                                    "go/pkg", "Library/pnpm")}
    allow = " ".join(f"(subpath {q(x)})" for x in sorted(places))
    return ('(version 1)(allow default)(deny file-write* (subpath "/"))'
            f'(allow file-write* {allow} (literal "/dev/null") (literal "/dev/tty") (literal "/dev/zero") (regex #"^/dev/fd/"))')


def run_shell(cmd: str, cwd: Path, timeout: float, confine: Optional[Path] = None) -> Tuple[Optional[int], str]:
    """``cmd`` in bash: no terminal input (a test calling input() would wait on the user's keyboard),
    in a process group of its own, and the whole group killed when it runs over ``timeout`` (killing
    only bash left ``cd x && python3 loop.py`` running). With ``confine``, it writes files only there,
    in the temporary folders and in caches (macOS's sandbox-exec; LOCALS_NO_SANDBOX=1: not confined).
    (exit code or None on timeout, output)."""
    import shutil
    import signal

    argv = ["bash", "-c", cmd]
    if confine is not None and not os.environ.get("LOCALS_NO_SANDBOX") and shutil.which("sandbox-exec"):
        argv = ["sandbox-exec", "-p", write_sandbox(confine)] + argv
    p = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True,
                         env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    _LIVE.add(p.pid)

    def kill():
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        _LIVE.discard(p.pid)

    try:
        out, _ = p.communicate(timeout=timeout)
        kill()  # what it left running in the background (a server for its curl) ends with it
        return p.returncode, out or ""
    except subprocess.TimeoutExpired:
        kill()
        try:  # a child that left the group (setsid, a daemon) may still hold the pipe: do not wait for it
            out, _ = p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            p.stdout.close()
            out = ""
        return None, out or ""
    except BaseException:  # Ctrl-C (the group gets no terminal signal) or an error: not left running
        kill()
        raise


_NO_TESTS = re.compile(r"^Ran 0 tests\b|^NO TESTS RAN\b|\bno tests ran\b", re.M)


# test runners and linters a check runs as python -m: never the project's own module, which the task may create
_TOOLS = {"pytest", "nose2", "tox", "nox", "mypy", "ruff", "flake8", "pylint", "black", "isort", "coverage", "hypothesis",
          "behave", "green"}
_BUILTINS = {"cd", "export", "set", "unset", "source", ".", "true", "false", "test", "[", "echo", "exit", "eval", "exec",
             "env", "time", "command", "builtin", "pushd", "popd", "ulimit", "umask", "printf", "read", "trap", "wait"}


def check_cannot_run(cmd: str, cwd: Path) -> Optional[str]:
    """Why the check command cannot run at all here (its test runner or a program missing), or None.
    Its failing is what the agents work against; a check that fails before any code exists only because
    pytest is not installed kept 17 agents busy for three minutes, and the run then said only that the
    check fails. Commands are split at ;, &&, || and |, assignments (VAR=value) and shell builtins skipped.
    Programs given as paths (./run_tests.sh) and the project's own modules are not judged: the task may
    create them."""
    for part in re.split(r"&&|\|\||;|\|", cmd):
        try:
            words = shlex.split(part)
        except ValueError:
            return None  # quoting we do not follow: the shell will tell
        while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
            words = words[1:]
        if not words or words[0] in _BUILTINS:
            continue
        exe = words[0]
        found = shutil.which(exe, path=os.environ.get("PATH"))
        if not found:
            if "/" in exe:
                continue
            return f"`{exe}` is not installed or not on PATH, so the check `{cmd}` cannot run."
        if re.match(r"^python(\d+(\.\d+)?)?$", Path(exe).name) and "-m" in words[1:-1]:
            module = words[words.index("-m", 1) + 1]
            if module.split(".")[0] not in _TOOLS:
                continue
            probe = subprocess.run([found, "-c", "import importlib.util, sys; "
                                    f"sys.exit(0 if importlib.util.find_spec({module.split('.')[0]!r}) else 3)"],
                                   cwd=cwd, capture_output=True, text=True, timeout=60)
            if probe.returncode == 3:
                return (f"`{exe}` has no module named {module.split('.')[0]}, so the check `{cmd}` cannot run. Install it "
                        f"({exe} -m pip install {module.split('.')[0]}), or give the command your project's tests run with "
                        f"(--check).")
    return None


def run_check(cmd: str, cwd: Path, timeout: float = 300, keep: int = 2000) -> Tuple[bool, str, float]:
    """(passed, the end of the output, seconds). A run that ran no tests is a failure: unittest's
    discovery that finds nothing exits 0 ("Ran 0 tests ... OK"), and would pass every attempt. The check writes
    only in ``cwd`` (an attempt's copy), the temporary folders and the tools' caches, as the agents' commands do:
    through a link out of the project, or the project's own .venv, a failed run's check had changed files
    outside the copy, while it said the project was not changed."""
    t0 = time.time()
    code, out = run_shell(cmd, cwd, timeout, confine=cwd)
    if code is None:
        ok, out = False, f"(the check ran over {timeout:.0f} s)\n" + out
    else:
        ok = code == 0
        if ok and _NO_TESTS.search(out):
            ok, out = False, out + "\n(the check ran no tests)"
    return ok, out.strip()[-keep:], time.time() - t0


# --- prompts ------------------------------------------------------------------------------------

def attempt_prompt(task: str, check: str, hint: str, notes: str) -> str:
    parts = []
    if notes:
        parts.append(notes)
    parts.append(task.strip())
    parts.append(f"When you are done, run this check from the project directory and make sure it passes:\n    {check}\n"
                 "Do not change the check itself or the tests it runs, unless the task asks for that.")
    if hint:
        parts.append(hint)
    return "\n\n".join(parts)


def fix_prompt(check: str, output: str) -> str:
    return (f"The check failed:\n    {check}\n\nIts output (the end of it):\n{output[-1500:]}\n\n"
            "Fix the code so that the check passes, run it again, and stop when it passes.")


# What the server writes into a reply that repeats a call (locals.server): not the agent's own words.
_STEERING = re.compile(r"The \w+ call above was already made and returned: .*?so I will do something different now:\s*"
                       r"|I am stuck: I keep wanting to repeat the same .*?(which command or file to use\)\.)", re.S)


def failure_notes(failed: List["Attempt"]) -> str:
    lines = ["Earlier attempts at this task were discarded because the check did not pass. What they did, and what "
             "the check said:"]
    for a in failed[-6:]:
        did = " ".join(_STEERING.sub(" ", a.answer or "").split())[:300] or "(no answer)"
        out = " ".join((a.check_out or "(not run)").split())[-300:]
        lines.append(f"- {did}\n  check: {out}")
    lines.append("The project is back as it was before them. Avoid what failed above.")
    return "\n".join(lines)


CHECK_WRITER = (
    "Do not do the task below. Write a check for it instead: a Python program named locals_check.py in the project "
    "directory that exits with code 0 when the task has been done correctly, and fails otherwise. It should check the "
    "result the way a user would: run the project's programs with subprocess, read the files the task produces, and "
    "assert on what they print or contain; before failing, print what is wrong. Read the files the task involves "
    "first, so that the check uses their real names, paths and formats. Change no other file. Then run it with  "
    "python3 locals_check.py : it must fail now, because the task is not done yet.\n\nThe task:\n{task}")


def check_command(script: Path) -> str:
    """The command that runs a check kept outside the project as if it were in it: from the
    attempt's directory, with that directory on Python's import path (the check imports the
    project's modules, and Python would look for them next to the script)."""
    return f'PYTHONPATH="$PWD${{PYTHONPATH:+:$PYTHONPATH}}" python3 {shlex.quote(str(script))}'


def write_check(ocs, project: Path, task: str, run_dir: Path, agent: str, provider: str, model: str, timeout: float,
                tries: int = 2, log=print) -> Optional[str]:
    """No tests to judge the attempts by: an agent writes ``locals_check.py`` from the task
    (CodeT's idea: tests from the model, solutions judged by them). The check must compile
    and must fail on the project as it is; it is kept outside every workspace, so the
    attempts can read it but not change it. Returns the check command, or None."""
    dst = run_dir / "locals_check.py"
    for k in range(tries):
        ws = run_dir / f"check{k}"
        kind = "copy"  # removed in the finally even if making it fails
        try:
            kind = make_workspace(project, ws)
            sid = ocs.call("POST", "/session", {"title": "locals solve: the check", "agent": agent,
                                                "model": {"providerID": provider, "id": model}}, str(ws), timeout=60)["id"]
            timer = threading.Timer(timeout, lambda: ocs.call("POST", f"/session/{sid}/abort", None, str(ws), timeout=10))
            timer.daemon = True
            timer.start()
            try:
                ocs.call("POST", f"/session/{sid}/message", {"model": {"providerID": provider, "modelID": model}, "agent": agent,
                                                            "parts": [{"type": "text", "text": CHECK_WRITER.format(task=task.strip())}]},
                         str(ws), timeout=timeout + 60)
            except Exception:  # noqa: BLE001
                pass
            finally:
                timer.cancel()
            script = ws / "locals_check.py"
            if not script.is_file():
                log(f"  the check writer ({k + 1}/{tries}) wrote no locals_check.py")
                continue
            shutil.copy2(script, dst)
            ok, out, _ = run_check(f"python3 -m py_compile {shlex.quote(str(dst))}", run_dir)
            if not ok:
                log(f"  the check ({k + 1}/{tries}) does not compile: {out[-200:]}")
                continue
            probe = run_dir / f"probe{k}"
            pkind = "copy"
            try:
                pkind = make_workspace(project, probe)
                cmd = check_command(dst)
                ok, out, _ = run_check(cmd, probe)
            finally:
                remove_workspace(project, probe, pkind)
            if ok:
                log(f"  the check ({k + 1}/{tries}) passes on the untouched project, so it tells nothing; another one")
                continue
            return cmd
        except Exception as e:  # noqa: BLE001
            log(f"  the check writer failed: {type(e).__name__}: {e}")
        finally:
            remove_workspace(project, ws, kind)
    return None


# --- one attempt --------------------------------------------------------------------------------

class Attempt:
    def __init__(self, i: int, ws: Path, kind: str, hint: str, after: int = 0):
        self.i, self.ws, self.kind, self.hint = i, ws, kind, hint
        self.after = after  # failed attempts it was told about
        self.session: Optional[str] = None
        self.state = "starting"
        self.t0 = self.t1 = 0.0
        self.turns = 0
        self.passed = False
        self.check_out = ""
        self.check_s = 0.0
        self.answer = ""
        self.error: Optional[str] = None
        self.tools: List[str] = []
        self.t_run = 0.0  # when the run began

    @property
    def name(self) -> str:
        return f"a{self.i}"

    def summary(self) -> dict:
        return {"name": self.name, "after_failures": self.after, "hint": self.hint, "session": self.session, "state": self.state,
                "start_s": round(self.t0 - self.t_run, 1) if self.t0 else None,
                "passed": self.passed, "wall_s": round((self.t1 or time.time()) - self.t0, 1), "turns": self.turns,
                "check_s": round(self.check_s, 1), "check_tail": self.check_out[-600:], "answer": self.answer[-600:],
                "error": self.error, "tools": self.tools}


def _final_text(msgs: List[dict]) -> Tuple[str, List[str]]:
    texts, tools = [], []
    for m in msgs or []:
        if (m.get("info") or {}).get("role") != "assistant":
            continue
        for p in m.get("parts") or []:
            if p.get("type") == "text" and p.get("text", "").strip():
                texts.append(p["text"])
            elif p.get("type") == "tool":
                tools.append(p.get("tool") or "?")
    return (texts[-1] if texts else ""), tools


class Race:
    """N attempts on one task at once; the first whose check passes wins."""

    def __init__(self, ocs, project: Path, task: str, check: str, agent: str, provider: str, model: str, run_dir: Path,
                 timeout: float, fixes: int, log=print):
        self.ocs, self.project, self.task, self.check = ocs, project, task, check
        self.agent, self.provider, self.model, self.run_dir = agent, provider, model, run_dir
        self.timeout, self.fixes, self.log = timeout, fixes, log
        self.lock = threading.Lock()
        self.winner: Optional[Attempt] = None
        self.attempts: List[Attempt] = []
        self.check_slots = threading.Semaphore(2)  # back-pressure: few checks at a time
        self.base: Optional[Dict[str, str]] = None  # the project's files as every attempt started from them
        self._replaced: set = set()  # failed attempts a fresh one has been started for

    def _abort(self, a: Attempt) -> None:
        if a.session:
            try:
                self.ocs.call("POST", f"/session/{a.session}/abort", None, str(a.ws), timeout=10)
            except Exception:  # noqa: BLE001
                pass

    def _send(self, a: Attempt, text: str) -> None:
        self.ocs.call("POST", f"/session/{a.session}/message",
                      {"model": {"providerID": self.provider, "modelID": self.model}, "agent": self.agent,
                       "parts": [{"type": "text", "text": text}]}, str(a.ws), timeout=self.timeout + 60)

    def _run(self, a: Attempt, prompt: str) -> None:
        a.t0 = time.time()
        try:
            s = self.ocs.call("POST", "/session", {"title": f"locals solve {a.name}", "agent": self.agent,
                                                   "model": {"providerID": self.provider, "id": self.model}}, str(a.ws), timeout=60)
            a.session = s["id"]
            timer = threading.Timer(self.timeout, lambda: self._abort(a))
            timer.daemon = True
            timer.start()
            text = prompt
            try:
                for turn in range(1 + self.fixes):
                    if self.winner is not None:
                        a.state = "outrun"
                        return
                    if not timer.is_alive():  # its time ran out during the check: no further turn
                        a.state, a.error = "timed out", f"over {self.timeout:.0f} s"
                        return
                    a.state = "working" if turn == 0 else f"fixing ({turn})"
                    a.turns += 1
                    try:
                        self._send(a, text)
                    except Exception as e:  # noqa: BLE001  # aborted (timeout or another attempt won), or an HTTP error
                        a.error = f"{type(e).__name__}: {e}"
                        self._abort(a)  # a turn the client gave up on does not go on generating on the server
                    if not timer.is_alive():
                        a.state, a.error = "timed out", f"over {self.timeout:.0f} s"
                        return
                    if self.winner is not None:
                        a.state = "outrun"
                        return
                    a.state = "checking"
                    a.files = manifest(a.ws)  # (before the check: what it writes, target/ or htmlcov/, is not the attempt's work)
                    with self.check_slots:
                        ok, out, secs = run_check(self.check, a.ws)
                    a.check_out, a.check_s = out, a.check_s + secs
                    if ok:
                        with self.lock:
                            if self.winner is None:
                                self.winner = a
                                a.passed, a.state = True, "passed"
                        if a.passed:
                            for other in self.attempts:
                                if other is not a and other.state not in ("passed", "failed", "timed out"):
                                    self._abort(other)
                        else:
                            a.state = "outrun"
                        return
                    text = fix_prompt(self.check, out)
                a.state = "failed"
            finally:
                timer.cancel()
        except Exception as e:  # noqa: BLE001
            a.state, a.error = "error", f"{type(e).__name__}: {e}"
        finally:
            a.t1 = time.time()
            try:
                msgs = self.ocs.call("GET", f"/session/{a.session}/message", directory=str(a.ws), timeout=60) if a.session else []
            except Exception:  # noqa: BLE001
                msgs = []
            a.answer, a.tools = _final_text(msgs)
            (self.run_dir / f"{a.name}.json").write_text(json.dumps({"attempt": a.summary(), "messages": msgs}, indent=1))

    def run(self, parallel: int, max_attempts: int, hedge: float, status, start: int = 0, burst: int = 1) -> None:
        """Attempts until one passes or ``max_attempts`` have run: ``start`` to begin with
        (default: all ``parallel`` when ``hedge`` is 0, else one), another whenever
        ``hedge`` (> 0) seconds pass without a winner, and ``burst`` fresh ones, told what
        failed, for each attempt that fails, never more than ``parallel`` at a time."""
        running: List[Tuple[Attempt, threading.Thread]] = []
        failed: List[Attempt] = []
        t_run, last = time.time(), 0.0

        def launch():
            nonlocal last
            i = len(self.attempts)
            ws = self.run_dir / f"a{i}"
            kind = make_workspace(self.project, ws)
            if self.base is None:
                self.base = manifest(ws)
            a = Attempt(i, ws, kind, HINTS[i % len(HINTS)] if parallel > 1 else "", after=len(failed))
            a.t_run = t_run
            notes = failure_notes(failed) if failed else ""
            th = threading.Thread(target=self._run, args=(a, attempt_prompt(self.task, self.check, a.hint, notes)), daemon=True)
            self.attempts.append(a)
            running.append((a, th))
            th.start()
            last = time.time()

        start = start or (parallel if hedge <= 0 else 1)
        for _ in range(min(start, parallel, max_attempts)):
            launch()
        while running:
            for a, th in list(running):
                if not th.is_alive():
                    running.remove((a, th))
                    if not a.passed and self.winner is None:
                        failed.append(a)
            if self.winner is None and len(self.attempts) < max_attempts and len(running) < parallel:
                fresh = [a for a in failed if a.i not in self._replaced]  # failed attempts not yet followed up
                if fresh or (hedge > 0 and time.time() - last >= hedge):
                    for a in fresh:
                        self._replaced.add(a.i)
                    for _ in range(max(1, burst * len(fresh))):
                        if len(self.attempts) >= max_attempts or len(running) >= parallel:
                            break
                        launch()
            status(self.attempts)
            time.sleep(0.5)
        status(self.attempts)


def cmd_solve(args) -> None:
    from . import app

    project = app.project_folder(args.dir, "solve")
    if args.check:  # before the model loads and the attempts start: a check that cannot run fails every one of them
        why = check_cannot_run(args.check, project)
        if why:
            sys.exit(f"locals solve: {why}")
    check = args.check or detect_check(project)
    if not check and args.no_write_check:
        sys.exit("locals solve needs a way to tell a good result from a bad one: give the command, e.g.\n"
                 "  --check 'python3 -m pytest -q'   or   --check 'python3 app.py --selftest'\n"
                 "(no tests were found in the project; without --no-write-check an agent writes a check first)")
    app.fetch_model(args.model)  # a mistyped or too large model is refused here, before OpenCode is set up for it
    if not app._opencode_ready(args.model, args.port):
        print(f"setting up OpenCode for {args.model} ...", flush=True)
        app.setup_opencode(args)
    oc = app.opencode_bin()
    app.ensure_server(args)
    app.remember(model=args.model, port=args.port)
    short = app.model_short(args.model)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if cleanup_stale_runs(project, app.LOCALS_DIR / "solve"):
        print("removed the workspaces an interrupted earlier run left behind", flush=True)
    run_dir = app.LOCALS_DIR / "solve" / f"{project.name}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pid").write_text(str(os.getpid()))
    (run_dir / "project").write_text(str(project))
    import signal

    def _stop(sig, frame):  # a closed terminal (SIGHUP) or kill (SIGTERM): the cleanup below runs, as for Ctrl-C
        raise SystemExit(128 + sig)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _stop)
    base_project = manifest(project)
    # OpenCode under the same sandbox as the checks: its agents' shell wrote through a link out of the project
    # (data -> ~/datasets) into the user's files, from an attempt that was then thrown away
    ocs = app.OpenCodeServer(oc, project, run_dir / "opencode-serve.log",
                             confine=[run_dir, *(Path.home() / d for d in OPENCODE_DIRS)])
    ocs.start_approver()
    if not check:
        print("no tests in the project: an agent writes a check for the task first ...", flush=True)
        engine = getattr(args, "engine", None) or "locals"
        check = write_check(ocs, project, args.task, run_dir, args.agent, engine, f"{engine}/{short}", args.timeout)
        if not check:
            ocs.close()
            sys.exit("no usable check came out (it must compile and fail on the untouched project); give one with --check")
        print(f"  the check: {run_dir / 'locals_check.py'}", flush=True)
    pre = run_dir / "precheck"  # the check on the project as it is, in a copy: a check an agent wrote runs its programs
    pkind = "copy"
    try:
        pkind = make_workspace(project, pre)
        ok0, out0, _ = run_check(check, pre)
    finally:
        remove_workspace(project, pre, pkind)
    if ok0:  # the first attempt that kept it passing would "win", done or not
        ocs.close()
        sys.exit(f"the check passes on the project as it is, so it cannot tell when the task is done:\n  {check}\n"
                 "give a --check that fails now and passes when the task is done (or add a test for the task first)")
    if getattr(args, "plan", False):
        if getattr(args, "plan_agents", False):
            plan = f"the task split into at most {args.parts} parts that agents do at once, then the check and {args.fixes} fix(es)"
        else:
            plan = (f"the task split into at most {args.parts} parts, {getattr(args, 'samples', 0) or 'two or three'} version(s) "
                    f"of each written at once in one batch, the check picks a combination that passes; then up to "
                    f"{args.fixes} repair round(s) and an agent")
    else:
        plan = (f"{args.start} agent(s) at the start" + (f", another every {args.hedge:.0f}s without a result" if args.hedge > 0 else "")
                + f", {args.burst} fresh one(s) told what failed for each that fails, {max(args.n, args.start)} at once and "
                f"{args.attempts} in all at most")
    print(f"locals solve in {project}\n  check: {check}  ({'passes already' if ok0 else 'fails now'})\n"
          f"  {plan} ({args.agent}, {short}); attempts in {run_dir}", flush=True)
    engine = getattr(args, "engine", None) or "locals"  # mlx: the same, on stock mlx-lm (for comparison)
    race = Race(ocs, project, args.task, check, args.agent, engine, f"{engine}/{short}", run_dir, args.timeout, args.fixes)
    t0 = time.time()
    last = [""]

    def status(attempts):
        try:
            s = app.get_json(app.base_url(args.port) + "/stats", timeout=2)
            rows = len(s["active"])
        except Exception:  # noqa: BLE001
            rows = "?"
        line = (f"\r  {time.time() - t0:5.0f}s  " + "  ".join(f"{a.name}:{a.state}" for a in attempts) + f"  | batch rows {rows}   ")
        if line != last[0]:
            print(line, end="", flush=True)
            last[0] = line

    try:
        if getattr(args, "plan", False):
            from .plan import solve_with_plan

            def plan_status(states):
                line = f"\r  {time.time() - t0:5.0f}s  " + "  ".join(f"{k}:{v}" for k, v in list(states.items())) + "   "
                if line != last[0]:
                    print(line, end="", flush=True)
                    last[0] = line

            solve_with_plan(race, project, args.task, app.base_url(args.port), args.parts, args.fixes, plan_status,
                            direct=not getattr(args, "plan_agents", False), samples=getattr(args, "samples", 0) or 0)
        else:
            race.run(max(args.n, args.start), max(args.attempts, 1), args.hedge, status, start=max(args.start, 1),
                     burst=max(args.burst, 1))
        print()
    except BaseException:  # Ctrl-C or an error: no worktree is left registered in the user's repository
        if not args.keep:
            for a in race.attempts:
                remove_workspace(project, a.ws, a.kind)
        raise
    finally:
        ocs.close()
    wall = time.time() - t0
    w = race.winner
    report = {"project": str(project), "task": args.task, "check": check, "model": args.model, "agent": args.agent, "n": args.n,
              "attempts_max": args.attempts, "hedge_s": args.hedge, "fixes": args.fixes, "wall_s": round(wall, 1),
              "winner": w.name if w else None,
              "attempts": [a.summary() for a in race.attempts]}
    for a in race.attempts:  # every attempt's changes as a diff against the project as it was (before anything is applied)
        if a.ws.exists():
            ch, ad, de = changes(race.base, getattr(a, "files", None) or manifest(a.ws))
            if ch or ad or de:
                (run_dir / f"{a.name}.diff").write_text(unified_diff(project, a.ws, ch + ad, de))
    if w is not None:
        files = getattr(w, "files", None) or manifest(w.ws)
        changed, added, deleted = changes(race.base, files)
        report["changes"] = {"changed": changed, "added": added, "deleted": deleted}
        # a link or an unreadable file, in the winner or where it writes: its bytes are not what was checked, and
        # a file that became a link was deleted from the project (the link never made)
        odd = [f for f in changed + added + deleted if not _plain(files.get(f)) or not _plain(race.base.get(f))]
        diff_path = run_dir / f"{w.name}.diff"
        # changed in the project during the run, or "added" over a file the project has (a gitignored .env or
        # database the worktree did not have): nothing is applied over those
        drift = [f for f in changed + deleted if manifest_one(project, f) != base_project.get(f)]
        drift += [f for f in added if (project / f).exists() or (project / f).is_symlink()]
        from .plan import _is_test, asks_test_changes

        tests = [] if asks_test_changes(args.task) else [f for f in changed + deleted if _is_test(f)]
        top = project.resolve()
        outside = [f for f in changed + added + deleted if top not in (project / f).resolve().parents]
        if args.no_apply:
            print(f"{w.name} passed the check after {w.t1 - w.t0:.0f}s; not applied (--no-apply): {diff_path}")
        elif drift or tests or outside or odd:
            why = (f"these files changed in the project during the run (or exist there already): {', '.join(drift[:5])}" if drift else
                   f"it changed the tests the check runs ({', '.join(tests[:5])}), which the task does not ask for" if tests else
                   f"these paths lead outside the project: {', '.join(outside[:5])}" if outside else
                   f"these are links or unreadable files, here or in the winner: {', '.join(odd[:5])}")
            print(f"{w.name} passed, but nothing was applied: {why}. The winning changes: {diff_path}")
        else:
            _apply(project, w.ws, changed + added, deleted)
            report["applied"] = True
            print(f"{w.name} passed the check after {w.t1 - w.t0:.0f}s ({wall:.0f}s in all); applied to the project: "
                  f"{len(changed)} changed, {len(added)} added, {len(deleted)} deleted ({diff_path})")
        if w.answer:
            print("\n" + w.answer.strip()[:1500])
    else:
        print(f"no attempt passed the check ({len(race.attempts)} attempts, {wall:.0f}s). The last check output:\n"
              f"{(race.attempts[-1].check_out if race.attempts else out0)[-800:]}\nTranscripts and diffs: {run_dir}")
    if not args.keep:
        for a in race.attempts:
            remove_workspace(project, a.ws, a.kind)
    (run_dir / "summary.json").write_text(json.dumps(report, indent=1))
    (run_dir / "pid").unlink(missing_ok=True)  # finished: a later run leaves its workspaces (--keep) alone
    # 0 only when the project has the result (or --no-apply asked for none): a winner left unapplied (drift, tests
    # changed, paths outside) exited 0 too, and a script took the unchanged project for the solved one
    sys.exit(0 if w is not None and (args.no_apply or report.get("applied")) else (2 if w is not None else 1))


def _apply(project: Path, ws: Path, write: List[str], delete: List[str]) -> None:
    """The winner's files into the project: all written under temporary names first, then renamed in
    (an error half-way leaves the project as it was, but for a crash between two renames)."""
    tmps = []
    try:
        for rel in write:
            dst = project / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.locals-{os.getpid()}")
            shutil.copy2(ws / rel, tmp)
            if dst.exists():  # (the user's own mode: copy2 gave the workspace's checkout's, 0644 for a 0600 file)
                os.chmod(tmp, dst.stat().st_mode & 0o7777)
            tmps.append((tmp, dst))
        for tmp, dst in tmps:
            os.replace(tmp, dst)
        tmps = []
        for rel in delete:
            (project / rel).unlink(missing_ok=True)
    finally:
        for tmp, _ in tmps:
            tmp.unlink(missing_ok=True)


def manifest_one(root: Path, rel: str) -> Optional[str]:
    """A file's entry in a manifest: the sha1 of its bytes; "link:<target>" for a symbolic link (its target is not
    read: it may be outside the project); "unreadable" for a file that cannot be read; None for anything else."""
    p = root / rel
    if p.is_symlink():
        try:
            return "link:" + os.readlink(p)
        except OSError:
            return "unreadable"
    if not p.is_file():
        return None
    try:
        return hashlib.sha1(p.read_bytes()).hexdigest()
    except OSError:
        return "unreadable"
