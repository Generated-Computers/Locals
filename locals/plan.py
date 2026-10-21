"""locals solve --plan: one task split into parts written at the same time.

The engine writes more tokens a second in all with more streams (Qwen3.5-4B on an M2 Pro:
64 alone, 271 in all with eight), so a task split into parts that are written at once takes
about as long as its longest part. The parts: a part for each code file the task names, or
the model's plan under a JSON schema. Every part is one generation (plain fenced code after
a FILE line, under a regular expression); two or three versions of every part are written
at once in one batch, and the check picks: a whole version that passes, else a combination
of the parts' versions (a part-by-part search), else repair rounds that write new versions
of the parts the check's output points at. An OpenCode agent is the last resort.
``--plan-agents`` runs an OpenCode agent per part instead (the first plan mode).

The control flow is in code, not in the model: a 4B model writes one well-described part
well, and does not plan its own delegation (Qwen3.5-4B never called a subagent in our
OpenCode runs, where an orchestrator prompt asked it to).
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .solve import SKIP_DIRS, make_workspace, read_source, run_check, write_source

VERSION = "2026-10-09.12"  # recorded in plan.json: which way the parts and repairs were written

# No maxLength on the strings: counted repetitions blow the automaton up (98 s to build with
# them, 0.08 s without); the reply's token limit bounds the plan instead.
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "parts": {
            "type": "array", "minItems": 1, "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "files": {"type": "array", "minItems": 1, "maxItems": 6, "items": {"type": "string"}},
                    "instructions": {"type": "string"},
                    "needs": {"type": "array", "maxItems": 7, "items": {"type": "integer", "minimum": 0, "maximum": 7}},
                },
                "required": ["title", "files", "instructions", "needs"],
            },
        },
    },
    "required": ["parts"],
}

PLANNER = (
    "You split software tasks into parts that different developers do at the same time. Rules:\n"
    "1. Cover the whole task: every file the task asks to create or change belongs to exactly one part, tests included.\n"
    "2. Split by files: one part per file, or per small group of files that only make sense together. A task that "
    "names three files has two or three parts, not one. Tests belong to the part that writes the code they test (its "
    "developer knows what the code does); other code that uses that code is a part of its own.\n"
    "3. Parts are written at the same time. A part lists another part's number (counting from 0) in \"needs\" only "
    "when it cannot be written without seeing that part's code; when the task or your instructions give the names, "
    "signatures and formats it uses, it needs nothing.\n"
    "4. Give every part complete instructions, at most six sentences: the names, signatures, formats and behaviour the "
    "task asks for, so that its developer needs nothing else.\n"
    "Use a single part only when the task touches a single file.")


def project_digest(root: Path, limit: int = 20000) -> str:
    """The project for the planner: every file's path, and the text of the small ones."""
    import os

    files = []
    for dirpath, dirnames, filenames in os.walk(root):  # tool folders (.venv, node_modules) not walked into
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.is_file() and name not in SKIP_DIRS and not name.endswith((".pyc", ".lock")):
                files.append((p.relative_to(root), p.stat().st_size))
    listed = [str(r) for r, _ in files[:400]]
    out = ["Files: " + ", ".join(listed) + (f", ... and {len(files) - 400} more" if len(files) > 400 else "")]
    used = len(out[0])
    for rel, size in sorted(files, key=lambda x: x[1]):
        if size > 6000 or used > limit:
            continue
        try:
            text = (root / rel).read_text()
        except (OSError, UnicodeDecodeError):
            continue
        out.append(f"\n--- {rel}\n{text}")
        used += len(text) + 10
    return "\n".join(out)


def _clean(path: str) -> Optional[str]:
    """A file name relative to the project, or None for one outside it (absolute, ~, ..)."""
    p = path.strip().strip("`'\"")
    if not p or p.startswith(("/", "~")) or ".." in Path(p).parts:
        return None
    while p.startswith("./"):
        p = p[2:]
    return p or None


def normalize(plan: dict) -> List[dict]:
    """Parts with clean relative file names, needs only on earlier parts (so the plan is
    acyclic), and a part that shares a file with an earlier one made to need it."""
    parts = []
    for i, part in enumerate(plan.get("parts") or []):
        files = [f for f in (_clean(x) for x in part.get("files") or []) if f]
        if not files:
            continue
        parts.append({"title": (part.get("title") or f"part {i}").strip(), "files": files,
                      "instructions": (part.get("instructions") or "").strip(),
                      "needs": sorted({n for n in part.get("needs") or [] if isinstance(n, int) and 0 <= n < len(parts)})})
    for i, part in enumerate(parts):
        for j in range(i):
            if set(part["files"]) & set(parts[j]["files"]) and j not in part["needs"]:
                part["needs"] = sorted(part["needs"] + [j])
    return parts


# Code files a task names (data files it names are often what the code reads or writes when it runs).
_CODE_FILE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w-]+\.(?:py|js|ts|sh))\b")


def _file_name(f: str, project: Optional[Path]) -> bool:
    """Whether a name the task mentions is a file to write: not a product ("Node.js", "Chart.js": a
    capitalized name that is not in the project) or a web address ("github.com/x/y.py")."""
    parts = Path(f).parts
    if any("." in d for d in parts[:-1]):  # a domain or a dotted folder: a link, not a path here
        return project is not None and (project / f).exists()
    stem = Path(f).stem
    if stem[:1].isupper() and not (project is not None and (project / f).exists()):
        return False
    return True


def _existing(project: Path, name: str) -> List[str]:
    """The project's files with this name, as paths relative to it (virtualenvs, node_modules and the like skipped)."""
    from .solve import SKIP_DIRS

    out = []
    for q in project.rglob(name):
        rel = q.relative_to(project)
        if q.is_file() and not (set(rel.parts[:-1]) & SKIP_DIRS):
            out.append(str(rel))
    return sorted(out)


def cover(parts: List[dict], task: str, project: Optional[Path] = None) -> List[dict]:
    """Every code file the task names belongs to a part: one the plan left out gets a part of its own
    (Qwen3.5 once planned one part for a task naming three files)."""
    have = {f for p in parts for f in p["files"]}
    for raw in dict.fromkeys(_CODE_FILE.findall(task)):
        f = _clean(raw)
        if f and project is not None and "/" not in f and not (project / f).exists():
            # "fix parser.py" for src/mylib/parser.py: the existing file of that name. A part for a new parser.py at
            # the top owned the only file the agents could edit, and the real one was out of their reach. A name
            # several files have is left to the plan.
            found = _existing(project, f)
            if len(found) > 1:
                continue
            f = found[0] if found else f
        if f and f not in have and _file_name(f, project):
            parts.append({"title": f"write {f}", "files": [f], "needs": [],
                          "instructions": f"Write {f} as the task asks: everything the task says about {f}."})
            have.add(f)
    return parts


def tests_after_code(parts: List[dict]) -> List[dict]:
    """Parts that write code are written at once: a part that uses another's code works from the
    names and signatures the task and the plan give (the 4B planner chained six independent
    modules into five waves, and nothing ran side by side). Tests come after the code: tests
    written blind expected another error than the code raised, and the repairs bent the code
    to the test."""
    code = [i for i, p in enumerate(parts) if not all(_is_test(f) for f in p["files"])]
    for i, p in enumerate(parts):
        p["needs"] = [] if i in code else sorted(set(code))
    return parts


def _is_test(f: str) -> bool:
    """A Python test file: test_*.py, or a .py file under tests/."""
    return f.endswith(".py") and (Path(f).name.startswith("test") or "tests" in Path(f).parts)


def writable(parts: List[dict], task: str, project: Optional[Path]) -> List[dict]:
    """A part writes only the code files the task names, when it names any; else new files and
    existing ones the task names (an existing file the task does not name is context, not work:
    the planner gave a part to textstats' sample.txt, the input whose output the check compares).
    Parts left with nothing to write are dropped, needs renumbered."""
    named = set(_CODE_FILE.findall(task))
    keep, index = [], {}
    for i, p in enumerate(parts):
        if named:  # the task names its files: those, and new test files (the planner added run_tests.sh and a sample log)
            files = [f for f in p["files"] if f in named or Path(f).name in named
                     or (_is_test(f) and not (project is not None and (project / f).exists()))]
        elif project is not None:  # new files, or existing ones the task names
            files = [f for f in p["files"] if not (project / f).exists() or f in task or Path(f).name in task]
        else:
            files = list(p["files"])
        if files:
            index[i] = len(keep)
            keep.append(dict(p, files=files))
    for p in keep:
        p["needs"] = sorted({index[n] for n in p["needs"] if n in index})
    return keep


# A task that asks for its tests to change (else the tests that exist are the check, and no part writes
# them): "update test_api.py", "add tests for ...", "rewrite the tests"; and "fix test_x.py", where the
# test file follows the verb. Not "fix the failing tests" or "fix x.py so that test_x.py passes": those
# ask for the code to change.
_TEST_WORD = r"[\w./-]*test[\w./-]*"
_TEST_CHANGE = re.compile(r"\b(?:write|add|update|change|edit|rewrite|extend|replace|modify|adjust)\w*\s+(?:[\w./,-]+\s+){0,4}"
                          + _TEST_WORD + r"|\b(?:fix|correct|repair)\w*\s+(?:the\s+)?[\w./-]*(?:test_\w+|\w+_test)\.py", re.I)


_TESTS_ASKED = re.compile(r"\btests?\s+(?:for|in|into|to)\b", re.I)  # "Tests for both functions in tests/test_x.py"
_MAKE_PASS = re.compile(r"\b(?:make|so\s+that|until|get)\b[^.\n]{0,60}\bpass", re.I)  # "make the tests for parse() pass"


_RENAME = re.compile(r"\brenam(?:e|es|ed|ing)\b", re.I)  # a rename changes every use of the name, the tests' too


def asks_test_changes(task: str) -> bool:
    """Whether the task asks for tests to be written or changed (else the tests that exist are the
    check): a verb on tests ("update test_api.py", "add tests"), tests asked for by name ("Tests for
    both functions in tests/test_counting.py") in a task that is not about making tests pass, or a
    rename (its tests call the old name: kept as they were, no version could pass)."""
    return bool(_TEST_CHANGE.search(task)) or bool(_RENAME.search(task)) or \
        (bool(_TESTS_ASKED.search(task)) and not _MAKE_PASS.search(task))


def protect_tests(parts: List[dict], task: str, project: Optional[Path]) -> List[dict]:
    """The parts without the test files that exist already, unless the task asks to change tests:
    those tests are the check, and a part that rewrote them could make it pass ("implement slugify so
    that tests/test_strings.py passes" got a part that wrote that test anew, with one easy test).
    Parts left without files are dropped, needs renumbered."""
    if project is None or asks_test_changes(task):
        return parts
    keep, index = [], {}
    for i, p in enumerate(parts):
        files = [f for f in p["files"] if not (_is_test(f) and (project / f).exists())]
        if files:
            index[i] = len(keep)
            keep.append(dict(p, files=files))
    for p in keep:
        p["needs"] = sorted({index[n] for n in p["needs"] if n in index})
    return keep


def make_plan(base_url: str, model: str, task: str, digest: str, max_tokens: int = 1200,
              project: Optional[Path] = None, log=lambda m: None) -> List[dict]:
    """A part for each code file the task names; else the model's plan under the JSON schema, made
    safe (normalize, writable, cover). A plan that does not parse (the server's loop guard cut one
    whose instructions went round in circles) is asked for once more; then there is no plan."""
    named = protect_tests(cover([], task, project), task, project)  # files the task names, new or existing
    ambiguous = project is not None and any(
        "/" not in f and not (project / f).exists() and len(_existing(project, f)) > 1
        for f in (_clean(raw) for raw in _CODE_FILE.findall(task)) if f)
    if named and not ambiguous:  # (a name several files have: which one is the plan's call)  # the task names its files: a part each, no planning call (10-15 s, and the 4B planner chained
        log(f"  the task names its files: a part for each ({', '.join(q['files'][0] for q in named)})")  # parts, dropped files)
        return named
    body = {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": PLANNER},
                         {"role": "user", "content": f"The project:\n{digest}\n\nThe task:\n{task.strip()}"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "plan", "schema": PLAN_SCHEMA}}}
    for attempt in range(2):
        req = urllib.request.Request(base_url + "/v1/chat/completions", data=json.dumps(dict(body, temperature=0.2 + 0.5 * attempt)).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                reply = json.loads(r.read())
            plan = json.loads(reply["choices"][0]["message"]["content"])
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:  # no reply, or not a plan
            log(f"  the plan did not come ({type(e).__name__}){'; once more' if attempt == 0 else ''}")
            continue
        if not isinstance(plan, dict):
            continue
        parts = protect_tests(cover(writable(normalize(plan), task, project), task, project), task, project)
        if parts:
            return parts
    parts = protect_tests(cover([], task, project), task, project)  # a part per code file the task names
    log(f"  plan from the task's file names: {', '.join(f for p in parts for f in p['files']) or 'none'}")
    return parts


def chat_json(base_url: str, model: str, messages: List[dict], schema: dict, max_tokens: int = 4096, timeout: float = 900,
              temperature: Optional[float] = None) -> dict:
    """One chat completion under a JSON schema (the reply always parses); returns (object, usage)."""
    body = {"model": model, "max_tokens": max_tokens, "messages": messages,
            **({"temperature": temperature} if temperature is not None else {}),
            "response_format": {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}}
    req = urllib.request.Request(base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        reply = json.loads(r.read())
    return {"obj": json.loads(reply["choices"][0]["message"]["content"]), "usage": reply.get("usage") or {}}


def files_schema(paths: List[str]) -> dict:
    """A reply holding the complete text of exactly these files."""
    return {"type": "object", "required": ["files"], "properties": {"files": {
        "type": "array", "minItems": len(paths), "maxItems": len(paths),
        "items": {"type": "object", "required": ["path", "content"],
                  "properties": {"path": {"enum": list(paths)}, "content": {"type": "string"}}}}}}


def _lit(text: str) -> str:
    """A literal for the engine's regular expressions."""
    return "".join("\\" + c if c in ".^$*+?()[]{}|\\" else c for c in text)


def files_regex(paths: List[str], note: bool = False) -> str:
    """Each file as  FILE <path>, then the file's text in a fenced block: the code as the model
    writes it anywhere else, nothing escaped (Python inside JSON strings came out broken: an
    unmatched brace, a string left open). ``note``: one line saying what is wrong comes first.
    The path is not in the expression: one expression for every part of one file, compiled once
    (one per path filled the GPU with constraint tables, and the server then ran the parts one
    at a time); the prompt names the path, and the reply's paths are checked when it is read."""
    body = "([^`]|`[^`]|``[^`])*"  # no three backticks in a row (a file does not end with one)
    one = "FILE [^\\n`]+\\n```[a-z0-9]*\\n" + body + "```"
    files = "\\n*".join(one for _ in paths) + "\\n?"
    return (_lit("The problem: ") + "[^\\n]+\\n\\n?" + files) if note else files


_FILE_BLOCK = re.compile(r"FILE (\S+)\n```[a-z0-9]*\n(.*?)```", re.S)


def chat_files(base_url: str, model: str, messages: List[dict], paths: List[str], max_tokens: int = 4096,
               timeout: float = 900, temperature: Optional[float] = None, stop: Optional[threading.Event] = None,
               note: bool = False) -> dict:
    """One chat completion that writes ``paths`` in the FILE format; returns the files, the usage,
    and whether ``stop`` cut it short (the reply is streamed: closing the connection when ``stop``
    is set cancels the generation on the server, and its batch row goes to the others)."""
    body = {"model": model, "max_tokens": max_tokens, "messages": messages, "stream": True,
            "response_format": {"type": "regex", "regex": files_regex(paths, note)},
            **({"temperature": temperature} if temperature is not None else {})}
    req = urllib.request.Request(base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    pieces, usage, stopped = [], {}, False
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if stop is not None and stop.is_set():
                stopped = True
                break
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("error"):  # the server failed mid-stream (it cannot change the status code by then)
                raise RuntimeError(f"server error: {(ev['error'] or {}).get('message', ev['error'])}")
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                pieces.append((ch.get("delta") or {}).get("content") or "")
    text = "".join(pieces)
    found = [(m[1], m[2]) for m in _FILE_BLOCK.finditer(text)]
    files = {f: c for f, c in found if f in paths}
    if not files and len(found) == len(paths):  # the files in the order asked, under other names (./a.py, a typo)
        files = dict(zip(paths, (c for _, c in found)))
    if not files:  # no FILE lines (a server that ignores the regex): fenced blocks in order, when they match up
        blocks = re.findall(r"```[a-z0-9]*\n(.*?)```", text, re.S)
        if len(blocks) == len(paths):
            files = dict(zip(paths, blocks))
    return {"files": {} if stopped else files, "usage": usage, "stopped": stopped}


_PY: Dict[str, str] = {}


def python_note() -> str:
    """Which python3 runs the code, for the prompts: a part written for 3.10 (an "X | None"
    annotation) failed at import under macOS's python3, 3.9."""
    if "note" not in _PY:
        import subprocess

        try:
            v = subprocess.run(["python3", "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], capture_output=True,
                               text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            v = ""
        _PY["note"] = f"Python is run as python3, version {v}: use nothing newer." if v else "Python is run as python3."
    return _PY["note"]


FILES_FORMAT = "Reply with every file of your part in this form, one after another, and nothing else:\nFILE <path>\n```python\n<the complete text of the file>\n```"


WRITER = (
    "You are one of several developers who write the parts of a task at the same time, each part in its own files. "
    "Write the complete text of every file of your part (not a diff, not a sketch): working code, with the names, "
    "signatures and formats the task and the descriptions give, so that it fits the other parts without seeing them. "
    "Python is run as python3. Tests are few and short (one or two for each function or command the task names), on "
    "small inputs (a few lines, a few items) whose expected results you work out exactly from the task. " + FILES_FORMAT)


def direct_prompt(task: str, parts: List[dict], i: int, digest: str, written: Dict[str, str],
                  own: Optional[Dict[str, str]] = None) -> List[dict]:
    me = parts[i]
    system = WRITER.replace("Python is run as python3.", python_note())
    others = [f"- part {j}, {p['title']} ({', '.join(p['files'])}): {p['instructions']}" for j, p in enumerate(parts) if j != i]
    text = [f"The project as it is:\n{digest}", f"The whole task:\n{task.strip()}",
            f"Your part: {me['title']}.\n{me['instructions']}", f"Your files: {', '.join(me['files'])}."]
    if others:
        text.append("The other parts, written at the same time (use them as described):\n" + "\n".join(others))
    if written:
        text.append("Code the other parts have written already:\n" + "\n".join(f"--- {f}\n{c}" for f, c in written.items()))
    if own:  # the digest leaves out large files: a part rewriting one must see all of it
        text.append("Your files as they are now (keep what the task does not ask to change):\n"
                    + "\n".join(f"--- {f}\n{c}" for f, c in own.items()))
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(text)}]


def part_prompt(task: str, parts: List[dict], i: int) -> str:
    me = parts[i]
    others = [f"- part {j}, {p['title']}: {', '.join(p['files'])}" for j, p in enumerate(parts) if j != i]
    lines = [f"The whole task, which several developers share:\n{task.strip()}",
             f"Your part: {me['title']}.\n{me['instructions']}",
             f"Write only these files: {', '.join(me['files'])}."]
    if others:
        lines.append("The other parts, written by others at the same time (do not write their files; use them as the task "
                     "and their descriptions say):\n" + "\n".join(others))
    if me["needs"]:
        lines.append("The parts you build on are done: read their files before you start.")
    lines.append("Write your files with the write tool (code pasted into your answer is lost). When they are written, check "
                 "them as far as you can on their own (run them, or a quick test), and stop. Do not run the whole project's "
                 "tests: the other parts may not be finished.")
    return "\n\n".join(lines)


class Team:
    """The parts of a plan as concurrent agents in one workspace, then the check and fixes."""

    def __init__(self, race, ws: Path):
        self.race, self.ws = race, ws  # the Race supplies the OpenCode server, model, agent and timeout
        self.states: Dict[str, str] = {}

    def _session(self, title: str, text: str, files=()) -> dict:
        """One agent's turn in the workspace; when it ends without writing ``files`` (a small model
        may describe its code, or paste it into the answer, and stop), it is told so once."""
        out = self._turn(title, text)
        missing = [f for f in files if not (self.ws / f).is_file()]
        if missing and out["error"] is None:
            out["nudged"] = missing
            more = self._turn(None, f"You have not written {', '.join(missing)} yet. Write {'it' if len(missing) == 1 else 'them'} "
                                    f"now with the write tool, then check {'it' if len(missing) == 1 else 'them'} and stop.",
                              out["session"])
            out["wall_s"] = round(out["wall_s"] + more["wall_s"], 1)
            out["error"] = more["error"]
        out["missing"] = [f for f in files if not (self.ws / f).is_file()]
        return out

    def _turn(self, title, text, sid=None) -> dict:
        r = self.race
        if sid is None:
            try:
                sid = r.ocs.call("POST", "/session", {"title": title, "agent": r.agent, "model": {"providerID": r.provider, "id": r.model}},
                                 str(self.ws), timeout=60)["id"]
            except Exception as e:  # noqa: BLE001
                return {"title": title, "session": None, "wall_s": 0.0, "error": f"no session: {type(e).__name__}: {e}"}
        timer = threading.Timer(r.timeout, lambda: r.ocs.call("POST", f"/session/{sid}/abort", None, str(self.ws), timeout=10))
        timer.daemon = True
        timer.start()
        t0 = time.time()
        try:
            r.ocs.call("POST", f"/session/{sid}/message", {"model": {"providerID": r.provider, "modelID": r.model}, "agent": r.agent,
                                                          "parts": [{"type": "text", "text": text}]}, str(self.ws), timeout=r.timeout + 60)
            error = None
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
        finally:
            timer.cancel()
        return {"title": title, "session": sid, "wall_s": round(time.time() - t0, 1), "error": error}

    def run(self, task: str, parts: List[dict], fixes: int, status=lambda s: None, part_runs=None) -> dict:
        report = {"parts": parts, "part_runs": part_runs or [], "fix_runs": [], "checks": []}
        if part_runs is not None:  # the parts are written: the check and the fixes
            return self._check_and_fix(task, parts, fixes, report, status)
        done: set = set()
        threads: Dict[int, threading.Thread] = {}
        results: Dict[int, dict] = {}

        def work(i):
            self.states[f"p{i}"] = "working"
            results[i] = self._session(f"locals plan part {i}: {parts[i]['title'][:40]}", part_prompt(task, parts, i), parts[i]["files"])
            self.states[f"p{i}"] = "done"

        while len(done) < len(parts):
            for i in range(len(parts)):
                if i not in threads and set(parts[i]["needs"]) <= done:
                    threads[i] = threading.Thread(target=work, args=(i,), daemon=True)
                    self.states[f"p{i}"] = "starting"
                    threads[i].start()
            for i, th in threads.items():
                if i not in done and not th.is_alive():
                    done.add(i)
            for i in range(len(parts)):
                if i not in threads:
                    self.states[f"p{i}"] = "waiting"
            status(self.states)
            time.sleep(0.5)
        report["part_runs"] = [results[i] for i in range(len(parts))]
        return self._check_and_fix(task, parts, fixes, report, status)

    def _check_and_fix(self, task: str, parts: List[dict], fixes: int, report: dict, status) -> dict:
        r = self.race
        for k in range(fixes + 1):
            ok, out, secs = run_check(r.check, self.ws)
            report["checks"].append({"ok": ok, "tail": out[-600:], "s": round(secs, 1)})
            if ok:
                report["passed"] = True
                return report
            if k == fixes:
                break
            self.states["fix"] = f"fixing ({k + 1})"
            status(self.states)
            text = (f"Several developers wrote the parts of this task at the same time:\n{task.strip()}\n\n"
                    f"The parts: " + "; ".join(f"{p['title']} ({', '.join(p['files'])})" for p in parts) + ".\n\n"
                    f"The check fails:\n    {r.check}\n\nIts output (the end of it):\n{out[-1500:]}\n\n"
                    "Read the files involved, make the parts work together and the check pass, run it again, and stop when it passes.")
            report["fix_runs"].append(self._session(f"locals plan fix {k + 1}", text))
        report["passed"] = False
        return report


_TEMPS = (None, 0.6, 0.9, 0.75, 0.45)  # sample s's temperature (None: the server's, 0.2): the samples differ


def _put(dst: Path, files: Dict[str, str]) -> None:
    """Write ``files`` (relative path -> text) under ``dst``; a path that leads outside it (``..``, or
    through a symlinked folder) is refused."""
    top = dst.resolve()
    for f, c in files.items():
        p = (dst / f).resolve()
        if top not in p.parents:
            raise ValueError(f"refusing to write {f}: it is outside {dst}")
        p.parent.mkdir(parents=True, exist_ok=True)
        write_source(p, c if c.endswith("\n") else c + "\n")


class _Checks:
    """The check on a set of files, each in a throwaway copy of the workspace, a few at once
    (a check costs a second or two; generating a part costs ten or more)."""

    def __init__(self, team, at_once: int = 4):
        self.team, self.sem, self.lock = team, threading.Semaphore(at_once), threading.Lock()
        self.secs: List[float] = []

    def __call__(self, files: Dict[str, str], stop: Optional[threading.Event] = None) -> Tuple[bool, str]:
        import shutil
        import statistics
        import tempfile

        with self.sem:
            if stop is not None and stop.is_set():
                return False, "(not checked: another one passed)"
            tmp = Path(tempfile.mkdtemp(prefix="locals-cand-")) / "w"
            names = {str(tmp.parent.resolve() / "w"), str(tmp)}
            try:
                shutil.copytree(self.team.ws, tmp, symlinks=True, ignore=shutil.ignore_patterns(*_COPY_SKIP))
                _put(tmp, files)
                with self.lock:  # a candidate that hangs (a loop, a read of stdin) costs a few checks' time, not 300 s
                    limit = max(30.0, 4 * statistics.median(self.secs)) if self.secs else 120.0
                ok, out, secs = run_check(self.team.race.check, tmp, timeout=limit, keep=8000)
            finally:
                shutil.rmtree(tmp.parent, ignore_errors=True)
        with self.lock:
            self.secs.append(secs)
        for n in sorted(names, key=len, reverse=True):
            out = out.replace(n, ".")
        return ok, out


# Left out of a copy of the project: the repository and caches.
_COPY_SKIP = (".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".DS_Store")


_LOAD = re.compile(r"\b(SyntaxError|IndentationError|TabError|ImportError|ModuleNotFoundError|NameError)\b")
_MARK = re.compile(r"error|fail|expected|assert|exited|no such|not found|traceback", re.I)


def _says(out: str) -> str:
    """The last lines of a check's output that say what failed, for the run's trace."""
    lines = [line.strip() for line in out.splitlines() if _MARK.search(line) and "curl:" not in line]
    return " | ".join(lines[-2:])[-240:]


def _trace(results: List[tuple], phase: str) -> List[dict]:
    return [{"phase": phase, "combo": list(c), "ok": ok, "says": "" if ok else _says(o)} for c, ok, o in results]


def _key(out: str) -> str:
    """What a failing check says, without its numbers (line numbers, values)."""
    return "\n".join(sorted({re.sub(r"\d+", "#", line.strip()) for line in out.splitlines() if _MARK.search(line)}))


def _failed(out: str) -> Optional[int]:
    """The failures in the last summary of the output (unittest's "FAILED (failures=1, errors=2)",
    pytest's "3 failed, 9 passed"), when it has one."""
    for line in reversed(out.splitlines()):
        m = re.search(r"FAILED \(([^)]*)\)", line)
        if m:  # failures, errors, unexpected successes; not skipped or expected failures
            kv = [x.split("=", 1) for x in m[1].split(",") if "=" in x]
            return sum(int(v) for k, v in kv if k.strip() in ("failures", "errors", "unexpected successes") and v.strip().isdigit())
        nums = re.findall(r"(\d+) (?:failed|errors?)\b", line)
        if nums:
            return sum(map(int, nums))
        if re.search(r"\b\d+ passed\b", line):
            return 0
    return None


def _score(out: str, key: Optional[str]) -> tuple:
    """How bad a failing check's output is (lower is better): code that does not load, then the
    failed tests (none counted is worst), then the same failure as ``key`` (no further than before)."""
    n = _failed(out)
    return (bool(_LOAD.search(out)), 1 << 20 if n is None else n, key is not None and _key(out) == key)


class _Any:
    """Set when any of these events is."""

    def __init__(self, *events):
        self.events = events

    def is_set(self) -> bool:
        return any(e.is_set() for e in self.events)


def _wait(threads: List[threading.Thread], late: threading.Event, status, states) -> None:
    """Wait for the threads; when all but a few are done, the rest get a little longer (half the
    time so far, 15 s at least), then ``late`` stops them: a version that runs away (4,096 tokens
    of a loop the guard did not see) held a repair round for two minutes."""
    t0, cut = time.time(), None
    while any(th.is_alive() for th in threads):
        alive = sum(th.is_alive() for th in threads)
        if cut is None and alive < len(threads) and alive <= max(1, len(threads) // 5):  # not before one is done
            cut = time.time() + max(15.0, 0.5 * (time.time() - t0))
        if cut is not None and time.time() > cut:
            late.set()
        status(states)
        time.sleep(0.5)


def _gen(team, base_url: str, messages: List[dict], paths: List[str], temperature, stop, note: bool = False,
         max_tokens: int = 4096) -> dict:
    t0 = time.time()
    try:
        out = chat_files(base_url, team.race.model, messages, paths, temperature=temperature, stop=stop, note=note,
                         max_tokens=max_tokens)
        r = {"files": out["files"], "tokens": out["usage"].get("completion_tokens"), "error": None,
             "stopped": out.get("stopped", False)}
    except Exception as e:  # noqa: BLE001
        r = {"files": {}, "tokens": None, "error": f"{type(e).__name__}: {e}", "stopped": False}
    r["wall_s"] = round(time.time() - t0, 1)
    return r


def _merge(options: List[List[Dict[str, str]]], combo: tuple) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for i, j in enumerate(combo):
        files.update(options[i][j])
    return files


def _write_samples(team, base_url: str, task: str, parts: List[dict], digest: str, samples: int, checks, status) -> dict:
    """``samples`` versions of the whole solution at once: each sample writes its parts in waves
    (the parts that need nothing, then those whose parts are written, with that code in their
    prompt), all samples' parts in one batch on the server; a sample that is complete is checked
    at once, and the first that passes stops the others."""
    n = len(parts)
    options: List[List[Dict[str, str]]] = [[{} for _ in range(samples)] for _ in range(n)]
    runs: List[dict] = []
    results: List[tuple] = []
    lock, found, late = threading.Lock(), threading.Event(), threading.Event()
    stop = _Any(found, late)

    def sample(s):
        done: set = set()
        written: Dict[str, str] = {}
        wave = 0
        while len(done) < n and not stop.is_set():
            wave += 1
            now = [i for i in range(n) if i not in done and set(parts[i]["needs"]) <= done] or \
                  [i for i in range(n) if i not in done]  # needs that will never be met: write the rest anyway
            res: Dict[int, dict] = {}

            def one(i):
                team.states[f"p{i}" + (f".{s}" if samples > 1 else "")] = "writing"
                seen = {f: c for f, c in written.items() if any(f in parts[j]["files"] for j in parts[i]["needs"])}
                own = {f: read_source((team.ws / f)) for f in parts[i]["files"] if (team.ws / f).is_file()}
                res[i] = _gen(team, base_url, direct_prompt(task, parts, i, digest, seen, own), parts[i]["files"], _TEMPS[s % len(_TEMPS)],
                              stop)
                team.states[f"p{i}" + (f".{s}" if samples > 1 else "")] = "written" if res[i]["files"] else "failed"

            threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in now]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            for i in now:
                options[i][s] = res[i]["files"]
                written.update(res[i]["files"])
                done.add(i)
                with lock:
                    runs.append({"sample": s, "part": i, "title": parts[i]["title"], "wave": wave, "wall_s": res[i]["wall_s"],
                                 "tokens": res[i]["tokens"], "files": sorted(res[i]["files"]), "error": res[i]["error"],
                                 "stopped": res[i]["stopped"]})
        if stop.is_set() or not all(options[i][s] for i in range(n)):
            return
        ok, out = checks(_merge(options, (s,) * n), stop=found)
        with lock:
            results.append(((s,) * n, ok, out))
        if ok:
            found.set()

    threads = [threading.Thread(target=sample, args=(s,), daemon=True) for s in range(samples)]
    for th in threads:
        th.start()
    _wait(threads, late, status, team.states)
    return {"options": options, "runs": sorted(runs, key=lambda r: (r["sample"], r["part"])), "results": results}


def _check_all(options: List[List[Dict[str, str]]], checks, combos: List[tuple]) -> List[tuple]:
    """Check these combinations a few at once, until one passes; [(combo, passed, output)] in their order."""
    results: Dict[tuple, tuple] = {}
    lock, found = threading.Lock(), threading.Event()

    def one(c):
        ok, out = checks(_merge(options, c), stop=found)
        if out.startswith("(not checked"):
            return
        with lock:
            results[c] = (c, ok, out)
        if ok:
            found.set()

    threads = [threading.Thread(target=one, args=(c,), daemon=True) for c in combos]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return [results[c] for c in combos if c in results]


def _search(options: List[List[Dict[str, str]]], checks, tried: set, results: List[tuple], cap: int = 24) -> List[tuple]:
    """The parts are independent by design, so a part of one version goes with the parts of
    another. From the best combination checked so far (``results``), every combination that
    swaps one part for another version of it is checked, a few at once; the swaps that did better
    are put together (fewer failed tests, or a failure further on), and that is the next start,
    until one passes, nothing does better, or ``cap`` checks. Returns the new results."""
    n = len(options)
    out: List[tuple] = []
    known = {r[0]: r for r in results}
    if not known or any(r[1] for r in results):
        return out
    base = min(results, key=lambda r: _score(r[2], None))
    visited = {base[0]}
    budget = cap
    while budget > 0:
        b, key = base[0], _key(base[2])
        singles = [tuple(j if x == i else b[x] for x in range(n)) for i in range(n) for j in range(len(options[i]))
                   if j != b[i] and options[i][j]]
        todo = [c for c in singles if c not in tried][:budget]
        tried.update(todo)
        budget -= len(todo)
        res = _check_all(options, checks, todo)
        out += res
        known.update({r[0]: r for r in res})
        if any(r[1] for r in res):
            break
        better = sorted((known[c] for c in singles if c in known and c not in visited and _score(known[c][2], key) < _score(base[2], key)),
                        key=lambda r: _score(r[2], key))
        if not better:
            break
        merged = list(b)
        for r in better:  # the best better version of each part, together
            i = next(x for x in range(n) if r[0][x] != b[x])
            if merged[i] == b[i]:
                merged[i] = r[0][i]
        nxt = better[0]
        if sum(m != v for m, v in zip(merged, b)) > 1 and tuple(merged) not in tried and budget > 0:
            tried.add(tuple(merged))
            budget -= 1
            res = _check_all(options, checks, [tuple(merged)])
            out += res
            if res and res[0][1]:
                break
            if res and _score(res[0][2], key) <= _score(nxt[2], key):
                nxt = res[0]
                known[nxt[0]] = nxt
        base = nxt
        visited.add(base[0])
    return out


REPAIRER = (
    "You are one of several developers who write the parts of a task at the same time, each part in its own files. "
    "The project's check fails; its output is below, then the code. Write the complete text of every file of your part "
    "(not a diff, not a sketch), with the names, signatures and formats the task gives, so that it fits the others' "
    "code and the check passes. Python is run as python3. Reply with every file of your part in this form, one after "
    "another:\nFILE <path>\n```python\n<the complete text of the file>\n```")

_NOTE = "First write one line that starts with \"The problem:\" and says what makes the check fail, then the files."

# What repair candidate c is told, and whether it sees its own files and writes a line on the problem
# first. One prompt for all, the candidate's own lines last (the batch computes the common prefix once).
# Shown its own file and asked to fix it, the 4B model copied it back unchanged at any temperature
# (three candidates, 318 tokens each, the same failures): two of three start afresh, told what failed,
# as a fresh Ralph attempt is.
_REPAIRS = (
    (True, True, "Fix your files: where your code is wrong, fix the code; where your test expects what the task does not "
                 "say, fix the test; keep what already works. " + _NOTE),
    (False, False, "An earlier version of your files failed the check: write them again from scratch."),
    (False, True, "An earlier version of your files failed the check: write them again from scratch, simply, with "
                  "Python's standard library where it fits. " + _NOTE))


def _names(f: str, out: str) -> bool:
    """The output names the file's path or name as a whole word ("inventory.py" is not in "test_inventory.py"),
    or, for a test file, its module ("test_report", as unittest names it; a common stem like "report" is not)."""
    forms = {f, Path(f).name}
    if _is_test(f):
        forms |= {Path(f).stem, f[:-3].replace("/", ".")}
    return any(re.search(r"(?<![\w.-])" + re.escape(x) + r"(?![\w-])", out) for x in forms)


_DEF = re.compile(r"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+([A-Za-z]\w+)", re.M)


def _blame(out: str, parts: List[dict], cur: List[Dict[str, str]]) -> List[int]:
    """The parts a failing check points at: it names one of their files, or calls a function or
    class one of them defines ("slugify(" in a failed test's line points at the part that writes
    strings.py)."""
    blamed = []
    for i, p in enumerate(parts):
        defs = {n for f, c in cur[i].items() if f.endswith(".py") for n in _DEF.findall(c) if len(n) > 2}
        if any(_names(f, out) for f in p["files"]) or any(re.search(r"\b" + re.escape(n) + r"\(", out) for n in defs):
            blamed.append(i)
    return blamed


def _excerpt(out: str, limit: int = 3000) -> str:
    """The start and the end of a long check output (failures first, the summary last)."""
    return out if len(out) <= limit else out[: limit // 2] + "\n...\n" + out[-limit // 2:]


def _repair(team, base_url: str, task: str, parts: List[dict], cur: List[Dict[str, str]], out: str, checks, status,
            k: int = 3) -> dict:
    """One repair round: ``k`` new versions of the parts the check's output points at (else of
    the code), all at once, each seeing the current code and the output; each new version is
    checked with the others' current files as soon as it is written, then combinations of new
    versions, until one passes."""
    n = len(parts)
    team.states.clear()
    code = [i for i, p in enumerate(parts) if not all(_is_test(f) for f in p["files"])]
    todo = _blame(out, parts, cur) or code or list(range(n))
    current = {f: c for files in cur for f, c in files.items() if len(c) <= 12000}  # not big data files
    options: List[List[Dict[str, str]]] = [[cur[i]] for i in range(n)]
    tried = {(0,) * n}
    gens: List[dict] = []
    results: List[tuple] = []
    lock, found, late = threading.Lock(), threading.Event(), threading.Event()
    stop = _Any(found, late)

    def one(i, c):
        team.states[f"p{i}" + (f".{c}" if k > 1 else "")] = "repairing"
        cap = min(4096, 1024 + sum(len(x) for x in cur[i].values()) // 2)  # about twice the part's tokens, at least 1,024
        own, note, what = _REPAIRS[c % len(_REPAIRS)]
        mine = set(parts[i]["files"])
        # the same for every candidate first (a common prefix): the task, the parts, the output, the
        # others' code; then the candidate's own files (when it sees them) and what it is to do
        text = [f"The whole task:\n{task.strip()}",
                "The parts:\n" + "\n".join(f"- part {j}, {p['title']} ({', '.join(p['files'])})" for j, p in enumerate(parts)),
                f"The check fails:\n{_excerpt(out)}",
                "The other parts' code:\n" + ("\n".join(f"--- {f}\n{c_}" for f, c_ in current.items() if f not in mine) or "(none)")]
        if own:
            text.append("Your files as they are now:\n" + "\n".join(f"--- {f}\n{c_}" for f, c_ in current.items() if f in mine))
        else:  # a fresh start still sees what the project had in its files (a stub's docstrings are the spec)
            first = {f: c_ for f, c_ in getattr(team, "original", {}).items() if f in mine}
            if first:
                text.append("Your files as the project had them before the task:\n"
                            + "\n".join(f"--- {f}\n{c_}" for f, c_ in first.items()))
        text.append(f"Your part: part {i}, {parts[i]['title']}.\n{parts[i]['instructions']}\nYour files: "
                    f"{', '.join(parts[i]['files'])}.\n{what}")
        r_ = _gen(team, base_url, [{"role": "system", "content": REPAIRER.replace("Python is run as python3.", python_note())},
                                   {"role": "user", "content": "\n\n".join(text)}],
                  parts[i]["files"], _TEMPS[c % len(_TEMPS)], stop, note=note, max_tokens=cap)
        team.states[f"p{i}" + (f".{c}" if k > 1 else "")] = "written" if r_["files"] else "failed"
        def norm(files):
            return {f: files.get(f, "").rstrip("\n") for f in parts[i]["files"]}

        with lock:  # a copy of the current files, or of another candidate, is not checked again
            same = bool(r_["files"]) and any(norm(r_["files"]) == norm(o) for o in options[i])
            gens.append({"part": i, "candidate": c, "wall_s": r_["wall_s"], "tokens": r_["tokens"], "error": r_["error"],
                         "stopped": r_["stopped"], "same": same})
            if not r_["files"] or same or found.is_set():
                return
            options[i].append(r_["files"])
            combo = tuple(len(options[i]) - 1 if x == i else 0 for x in range(n))
            tried.add(combo)
        ok, o = checks(_merge(options, combo), stop=found)
        if o.startswith("(not checked"):
            return
        with lock:
            results.append((combo, ok, o))
        if ok:
            found.set()

    threads = [threading.Thread(target=one, args=(i, c), daemon=True) for i in todo for c in range(k)]
    for th in threads:
        th.start()
    _wait(threads, late, status, team.states)
    if not found.is_set() and len(todo) > 1:  # new versions of two parts may only pass together
        results += _search(options, checks, tried, [((0,) * n, False, out)] + results)
    return {"todo": todo, "gens": sorted(gens, key=lambda g: (g["part"], g["candidate"])), "options": options, "results": results}


def _agent_repair(team, task: str, parts: List[dict], cur: List[Dict[str, str]], out: str, checks, status, k: int) -> dict:
    """A repair round of agents, at once, each in its own copy of the current files: one for
    each part the check points at (it may change only that part's files) and one that may
    change anything. An agent runs the code and the check, sees the error and fixes it, where
    a rewrite only guesses (the 4B model's wrong test survived every rewrite). A copy that
    passes wins; else the part agents' files go into the part-by-part search."""
    import shutil

    n = len(parts)
    r = team.race
    code = [i for i, p in enumerate(parts) if not all(_is_test(f) for f in p["files"])]
    todo = (_blame(out, parts, cur) or code or list(range(n)))[:3]
    jobs = [(i, parts[i]["files"]) for i in todo] + [(None, [f for p in parts for f in p["files"]])]
    copies, runs = {}, {}
    team.states.clear()

    def one(j, i, files):
        ws = r.run_dir / f"fix{k}-{j}"
        shutil.rmtree(ws, ignore_errors=True)
        shutil.copytree(team.ws, ws, symlinks=True, ignore=shutil.ignore_patterns(*_COPY_SKIP), ignore_dangling_symlinks=True)
        copies[j] = ws
        mate = Team(r, ws)
        who = "the whole task" if i is None else f"part {i}, {parts[i]['title']}"
        text = (f"Several developers wrote the parts of this task at the same time:\n{task.strip()}\n\n"
                "The parts: " + "; ".join(f"{p['title']} ({', '.join(p['files'])})" for p in parts) + ".\n\n"
                f"The check fails:\n    {r.check}\n\nIts output:\n{_excerpt(out, 2500)}\n\n"
                f"You fix {who}: change only {', '.join(files)}. Read the files involved, run the code and the check to see "
                "what fails and why, fix it, run the check again, and stop when it passes or when what is left is "
                "outside your files.")
        team.states[f"a{j}"] = "fixing"
        runs[j] = dict(mate._session(f"locals plan fix {k}.{j}", text), part=i)
        team.states[f"a{j}"] = "done"

    limit = r.timeout
    r.timeout = min(limit, 240)
    try:
        threads = [threading.Thread(target=one, args=(j, i, files), daemon=True) for j, (i, files) in enumerate(jobs)]
        for th in threads:
            th.start()
        while any(th.is_alive() for th in threads):
            status(team.states)
            time.sleep(0.5)
    finally:
        r.timeout = limit

    def read(ws, files):
        return {f: read_source((ws / f)) for f in files if (ws / f).is_file()}

    options: List[List[Dict[str, str]]] = [[cur[i]] for i in range(n)]
    results: List[tuple] = []
    whole = len(jobs) - 1
    if whole in copies:  # the agent that may change anything: its copy as it is
        mine = [read(copies[whole], p["files"]) for p in parts]
        for i in range(n):
            if mine[i] and mine[i] != cur[i]:
                options[i].append(mine[i])
        combo = tuple(len(options[i]) - 1 if mine[i] and mine[i] != cur[i] else 0 for i in range(n))
        if any(combo):
            ok, o = checks(_merge(options, combo))
            results.append((combo, ok, o))
    if not any(ok for _, ok, _ in results):
        for j, (i, files) in enumerate(jobs[:-1]):
            got = read(copies[j], files) if j in copies else {}
            if got and got != cur[i] and got not in options[i]:
                options[i].append(got)
        tried = {c for c, _, _ in results} | {(0,) * n}
        results += _search(options, checks, tried, [((0,) * n, False, out)] + results)
    for ws in copies.values():  # their files are in options now; the copies were left in the run's folder
        shutil.rmtree(ws, ignore_errors=True)
    return {"todo": todo, "agents": [runs[j] for j in sorted(runs)], "options": options, "results": results}


def _best(results: List[tuple], key: Optional[str]) -> Optional[tuple]:
    """The combination that passed, else the one whose failure is least bad."""
    for r in results:
        if r[1]:
            return r
    return min(results, key=lambda r: _score(r[2], key), default=None)


def auto_samples(n_parts: int) -> int:
    """Versions of the solution written at once: three while that is at most a dozen generations
    in one batch, two up to sixteen (the batch's speed per row falls off above that)."""
    return 3 if n_parts <= 4 else 2 if n_parts <= 8 else 1


def solve_with_plan(race, project: Path, task: str, base_url: str, max_parts: int, fixes: int, status, log=print,
                    direct: bool = True, samples: int = 0) -> dict:
    """Plan, run the parts, check and fix in one workspace; returns the report (with the
    workspace in race.attempts for the caller to apply and clean up)."""
    from .solve import Attempt, manifest

    t0 = time.time()
    parts = tests_after_code(make_plan(base_url, race.model, task, project_digest(project), project=project, log=log)[:max_parts])
    if not parts:
        raise RuntimeError("no plan: the model's plan did not parse and the task names no code files")
    log(f"  plan ({time.time() - t0:.0f}s): " + "; ".join(f"[{i}] {p['title']} -> {', '.join(p['files'])}"
                                                       + (f" (after {', '.join(map(str, p['needs']))})" if p["needs"] else "")
                                                       for i, p in enumerate(parts)))
    ws = race.run_dir / "plan"
    kind = make_workspace(project, ws)
    race.base = manifest(ws)
    a = Attempt(0, ws, kind, "plan")
    a.t_run = a.t0 = t0
    race.attempts.append(a)
    team = Team(race, ws)
    team.original = {f: read_source((project / f)) for p in parts for f in p["files"]
                     if (project / f).is_file() and (project / f).stat().st_size <= 12000}
    if direct:
        t1 = time.time()
        n = len(parts)
        samples = samples or auto_samples(n)
        checks = _Checks(team)
        w = _write_samples(team, base_url, task, parts, project_digest(project), samples, checks, status)
        tried = {c for c, _, _ in w["results"]}
        results = list(w["results"])
        log(f"\n  {samples} version(s) of the {n} part(s) written in {time.time() - t1:.0f}s; "
            + ("one passes" if any(ok for _, ok, _ in results) else f"{len(results)} checked whole, none passes"))
        searched = []
        if not any(ok for _, ok, _ in results) and samples > 1:
            t2 = time.time()
            searched = _search(w["options"], checks, tried, results)
            results += searched
            log(f"  {len(searched)} combination(s) of their parts checked in {time.time() - t2:.0f}s"
                + (": one passes" if any(ok for _, ok, _ in searched) else ""))
        trace = _trace(w["results"], "whole") + _trace(searched, "search")
        best = _best(results, None)
        if best is None:  # no version of some part was written: what there is goes into the workspace
            cur = [next((o for o in w["options"][i] if o), {}) for i in range(n)]
            for files in cur:
                _put(ws, files)
            ok, out, _ = run_check(race.check, ws, keep=8000)
        else:
            cur = [w["options"][i][best[0][i]] for i in range(n)]
            ok, out = best[1], best[2]
            for files in cur:
                _put(ws, files)
        repairs = []
        for k in range(fixes):  # repair rounds: new versions of the parts at fault, checked as they come
            if ok:
                break
            t2 = time.time()
            if k == 0:  # rewrites first: a round costs 20-40 s
                rr = _repair(team, base_url, task, parts, cur, out, checks, status)
            else:  # then agents that run the code, at once
                rr = _agent_repair(team, task, parts, cur, out, checks, status, k)
            trace += _trace(rr["results"], f"repair {k + 1}")
            pick = _best(rr["results"], _key(out))
            took = pick is not None and (pick[1] or _score(pick[2], _key(out)) < _score(out, _key(out)))
            if took:
                cur = [rr["options"][i][pick[0][i]] for i in range(n)]
                for files in cur:
                    _put(ws, files)
                ok, out = pick[1], pick[2]
            repairs.append({"kind": "rewrite" if k == 0 else "agents", "todo": rr["todo"], "gens": rr.get("gens", []),
                            "agents": rr.get("agents", []), "checked": len(rr["results"]), "passed": bool(pick and pick[1]),
                            "moved": took, "wall_s": round(time.time() - t2, 1)})
            log(f"  repair {k + 1} in {time.time() - t2:.0f}s: parts {rr['todo']}, {len(rr['results'])} checked, "
                + ("passes" if ok else "a further failure kept" if took else "no progress"))
        report = team.run(task, parts, 0, status, part_runs=w["runs"])  # the final check (the agents ran in the rounds)
        report.update(samples=samples, whole_checked=len(w["results"]), searched=len(searched),
                      search_passed=any(r[1] for r in searched), repairs=repairs, trace=trace)
    else:
        report = team.run(task, parts, fixes, status)
    a.t1 = time.time()
    a.passed = bool(report.get("passed"))
    a.state = "passed" if a.passed else "failed"
    a.check_out = report["checks"][-1]["tail"] if report["checks"] else ""
    if a.passed:
        race.winner = a
    report["plan_s"] = round(time.time() - t0, 1)
    report["version"] = VERSION
    (race.run_dir / "plan.json").write_text(json.dumps(report, indent=1))
    return report

