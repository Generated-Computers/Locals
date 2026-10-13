"""locals run: a multi-agent harness of locals's own.

    locals run "write inventory.py, cli.py and test_inventory.py: ..." [--dir PROJECT] [--check CMD]

One task, split into parts that agents do at the same time:

1. **Plan.** A part for each code file the task names; otherwise the model's plan
   (JSON, under a schema). Code parts start at once; test parts when the code is in.
2. **Versions in copies.** Every part gets several agents at once (tool loops on the
   server: list, read, write, edit, run, finish), each in its own copy of the
   project, at different temperatures; two whole-task agents race them. They are
   rows of one batch: N agents cost far less than N times one (Qwen3.5-4B on an M2
   Pro: 64 tok/s alone, 271 in all with eight), and their common prompt is
   computed once.
3. **The check after every change.** Each write is followed by the check in that
   agent's copy, whose word goes back with the write; the first copy that passes
   ends the stage (a whole-task agent's, the run).
4. **Pick and repair.** The check combines the parts' versions; the parts its
   output blames get fresh agents, at once, until it passes or stops improving.

The control flow is code: the model writes the plan (when the task names no files)
and does the parts; scheduling, file ownership and the check are the harness's.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from .plan import (_COPY_SKIP, _DEF, asks_test_changes, _best, _blame, _check_all, _Checks, _is_test, _key, _merge, _names, _put, _score,
                   _search, make_plan, project_digest, tests_after_code)
from .solve import SKIP_DIRS, run_check, run_shell

_TEMPS = (None, 0.7, 0.9, 0.5, 1.0, 0.6, 0.8, 0.4)  # version v's temperature (None: the server's, 0.2): the versions differ

SYSTEM = (
    "You are one developer in a team working on one task at the same time; each developer has their own part "
    "and their own files. You work only through the tools. Write a new file with write_file (its whole text); fix a "
    "few lines of a file with edit_file (old text, new text), which is much faster than writing it again. Read any file you need. Run commands (python3, the tests) with run "
    "to check your part. Never write a file that is not yours: another developer is writing it. Python is run as "
    "python3. When your part is written and checked, call finish with one line on what you did.")

TOOLS = [
    {"type": "function", "function": {"name": "list_files", "description": "The files of the project (or of one folder).",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "read_file", "description": "The text of a file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write the complete text of one of your files.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                                                     "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace one exact piece of one of your files "
                                      "(old must appear exactly once) with new: a small fix without writing the whole file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                                                     "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {"name": "run", "description": "Run a shell command in the project folder (60 s at most).",
                                      "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "finish", "description": "Your part is done: say what you did in one line.",
                                      "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]


def _rel(path: str) -> Optional[str]:
    """A path as the agent gives it, relative to the project: "./" prefixes removed, "" for the project
    itself; None for an absolute path or one with "..". (``str.lstrip("./")`` strips characters, and
    turned ".env" into "env".)"""
    p = (path or "").strip()
    while p.startswith("./"):
        p = p[2:]
    if p in ("", "."):
        return ""
    if p.startswith(("/", "~")) or ".." in Path(p).parts:
        return None
    return p.rstrip("/")


def _chat(base_url: str, body: dict, timeout: float = 900, stop: Optional[threading.Event] = None) -> dict:
    """One chat completion. With ``stop``, setting it closes the connection, and the server stops decoding
    the reply (it cancels a request whose client left): a stopped agent used to finish its reply (a
    3,000-token write_file is over a minute at batch speed), and the next stage waited for it."""
    if stop is None:
        req = urllib.request.Request(base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    import http.client
    import socket
    import urllib.parse

    u = urllib.parse.urlsplit(base_url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    done = threading.Event()

    def watch():
        while not done.wait(0.25):
            if stop.is_set():
                try:
                    if conn.sock is not None:
                        conn.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return

    threading.Thread(target=watch, daemon=True).start()
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = conn.getresponse()
        data = r.read()
        if r.status >= 400:
            raise RuntimeError(f"HTTP {r.status}: {data[:300].decode(errors='replace')}")
        return json.loads(data)
    finally:
        done.set()
        conn.close()


class Agent:
    """One agent: a tool loop on the locals server, writing only ``owns``."""

    NUDGE = ("Use a tool. If you were about to run the same command again, its result would be the same: change your "
             "files first (write_file), then run it. If your part is done, call finish.")

    def __init__(self, name: str, root: Path, base_url: str, model: str, owns: List[str], prompt: str,
                 max_steps: int = 16, log=None, verify=None, temperature: Optional[float] = None, after_write=None,
                 readonly=()):
        self.name, self.root, self.base_url, self.model = name, Path(root).resolve(), base_url, model  # /var is /private/var
        self.owns, self.max_steps = (owns if isinstance(owns, _AnyFile) else list(owns)), max_steps
        self.messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        self.state, self.steps, self.tokens, self.summary, self.error = "waiting", 0, 0, "", None
        self.wrote: List[str] = []
        self.trace: List[str] = []  # each step, short: the tool, its argument, the start of its result
        self.t0 = self.t1 = None
        self.log = log or (lambda m: None)
        self.verify, self.refused, self.temperature = verify, 0, temperature
        self.after_write = after_write  # after each write: the check's word on the part (back-pressure without a step)
        self.readonly = set(readonly)  # never written, even by an agent that owns every file (the tests that are the check)
        self.stop = threading.Event()  # set by the harness: a sibling finished the part, or time is up
        self.verified = False  # the check agreed that the part (or, for a whole-task agent, the task) is done
        self.won_files: Optional[Dict[str, str]] = None  # a whole-task agent's files as they passed the check

    def _relpath(self, p: str) -> Optional[str]:
        """_rel, and an absolute path inside the agent's own copy as relative: tracebacks print
        /private/var/folders/.../w/inventory.py, small models pass that on, and "no such file" cost them steps."""
        p = (p or "").strip()
        if p.startswith("/"):
            for root in {str(self.root), str(self.root.resolve())}:
                for r in (root, "/private" + root if root.startswith("/var/") else root):
                    if p == r or p.startswith(r.rstrip("/") + "/"):
                        return _rel(p[len(r.rstrip("/")) + 1:])
        return _rel(p)

    def _clean(self, text: str) -> str:
        """Output shown to the agent, with its copy's folder written as the project's (a relative path)."""
        for root in sorted({str(self.root), str(self.root.resolve())}, key=len, reverse=True):
            text = text.replace(root.rstrip("/") + "/", "")
        return text

    def _path(self, p: str) -> Optional[Path]:
        """The path inside the agent's copy, or None if it is not (absolute elsewhere, "..", or through a link out)."""
        rel = self._relpath(p)
        if rel is None:
            return None
        q = (self.root / rel).resolve()
        return q if q == self.root or self.root in q.parents else None

    def _tool(self, name: str, args: dict) -> str:
        if name in ("write_file", "edit_file") and not (args.get("path") or "").strip() and len(self.owns) == 1:
            args = {**args, "path": self.owns[0]}  # an agent with one file that left out the path (Gemma 4 does) means it
        if name == "list_files":
            base = self._path(args.get("path") or ".")
            if base is None or not base.exists():
                return "no such folder"
            out = [str(p.relative_to(self.root)) for p in sorted(base.rglob("*"))
                   if p.is_file() and not (set(p.relative_to(self.root).parts) & SKIP_DIRS)]
            return "\n".join(out[:300]) or "(no files)"
        if name == "read_file":
            p = self._path(args.get("path", ""))
            if p is None or not p.is_file():
                return f"no such file: {args.get('path')}"
            text = p.read_text(errors="replace")
            return text if len(text) <= 6000 else text[:6000] + f"\n... ({len(text) - 6000} more characters)"
        if name == "write_file":
            rel = self._relpath(args.get("path") or "")
            if not rel:
                return "not written: give the file's path, relative to the project"
            if rel in self.readonly:
                return f"not written: {rel} is one of the tests the check runs; change the code, not the tests"
            if rel not in self.owns:
                return f"not written: {rel} is not one of your files ({', '.join(self.owns)}); another developer owns it"
            p = self._path(rel)
            if p is None:
                return "not written: the path is outside the project"
            if p.is_dir():
                return f"not written: {rel} is a folder"
            p.parent.mkdir(parents=True, exist_ok=True)
            content = args.get("content") or ""
            p.write_text(content if content.endswith("\n") else content + "\n")
            if rel not in self.wrote:
                self.wrote.append(rel)
            return f"wrote {rel} ({len(content)} characters)"
        if name == "edit_file":
            rel = self._relpath(args.get("path") or "")
            if not rel:
                return "not changed: give the file's path, relative to the project"
            if rel in self.readonly:
                return f"not changed: {rel} is one of the tests the check runs; change the code, not the tests"
            if rel not in self.owns:
                return f"not changed: {rel} is not one of your files ({', '.join(self.owns)}); another developer owns it"
            p = self._path(rel)
            if p is None or not p.is_file():
                return f"not changed: no such file {rel} (write it with write_file)"
            text, old = p.read_text(errors="replace"), args.get("old") or ""
            n = text.count(old) if old else 0
            new_text = text.replace(old, args.get("new") or "", 1) if n == 1 else (_loose_replace(text, old, args.get("new") or "") if n == 0 else None)
            if new_text is None and n > 1:
                return (f"not changed: the old text is in {rel} {n} times, it must be exactly once; give a line or two "
                        "around it too")
            if new_text is None:
                return (f"not changed: the old text is not in {rel}. The lines of {rel} most like it are:\n"
                        f"{_closest(text, old)}\nCopy the old text exactly from these lines and edit again.")
            p.write_text(new_text)
            if rel not in self.wrote:
                self.wrote.append(rel)
            return f"changed {rel}"
        if name == "run":
            code, out = run_shell(args.get("command") or "true", self.root, 60, confine=self.root)
            out = out.strip()
            if "Operation not permitted" in out:
                out += "\n(commands write files only in the project's copy you work in, temporary folders and caches)"
            if code is None:
                return "the command ran over 60 s and was stopped" + (f"; its output:\n{out[-1000:]}" if out else "")
            return f"exit {code}\n" + (out[-1500:] if out else "(no output)")
        return f"unknown tool {name}"

    def run(self) -> "Agent":
        self.state, self.t0 = "working", time.time()
        last: List[str] = []
        idle = nudged = 0
        try:
            while self.steps < self.max_steps and not self.stop.is_set():
                if sum(len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or "")) for m in self.messages) > 90_000:
                    self.trace.append("context too long, ended")  # a model writing more and more of the same: a fresh agent does better
                    break
                self.steps += 1
                body = {"model": self.model, "messages": self.messages, "tools": TOOLS, "tool_choice": "required",
                        "parallel_tool_calls": False, "max_tokens": 3000}
                if self.temperature is not None:
                    body["temperature"] = self.temperature
                try:
                    resp = _chat(self.base_url, body, stop=self.stop)
                except Exception:  # noqa: BLE001
                    if self.stop.is_set():  # its connection closed by the stop
                        self.trace.append("stopped during a reply")
                        break
                    raise
                if self.stop.is_set():  # stopped while the reply was written: its call is not carried out
                    self.trace.append("stopped during a reply")
                    break
                self.tokens += (resp.get("usage") or {}).get("completion_tokens") or 0
                msg = (resp.get("choices") or [{}])[0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if not calls:
                    # The server ends a reply that repeats the previous call (its loop guard). Once, the agent is
                    # told to change something first; twice, it is stuck, and a fresh agent does better.
                    nudged += 1
                    self.trace.append("no call" + (": stuck, ended" if nudged > 1 else ": nudged"))
                    if nudged > 1:
                        break
                    self.messages.append({"role": "assistant", "content": msg.get("content") or ""})
                    self.messages.append({"role": "user", "content": self.NUDGE})
                    continue
                call = calls[0]
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                self.messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": [call]})
                if fn.get("name") == "finish":
                    checked = bool(self.verify) and self.refused < 2  # after two refusals the agent may stop, unverified
                    why = self.verify() if checked else None
                    self.verified = self.verified or (checked and why is None)
                    if why:  # back-pressure: the check, not the agent, says when the part is done
                        self.refused += 1
                        self.trace.append(f"finish refused: {why[-120:]!r}")
                        self.messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                              "content": self._clean(f"Not finished: the check still fails in your part.\n{why}\nFix it, then finish.")})
                        continue
                    self.summary = str(args.get("summary") or "")[:2000]  # a question's answer, too
                    self.messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "done"})
                    break
                idle = 0 if fn.get("name") in ("write_file", "edit_file") else idle + 1
                key = json.dumps([fn.get("name"), args], sort_keys=True)
                last = (last + [key])[-6:]
                try:
                    result = self._tool(fn.get("name") or "", args)
                except OSError as e:  # a folder for a file, a file for a folder, ...
                    result = f"error: {e.strerror or e}"
                self.trace.append(f"{fn.get('name')}({str(args.get('path') or args.get('command') or '')[:50]}) -> {result[:80]!r}")
                if fn.get("name") in ("write_file", "edit_file") and result.startswith(("wrote", "changed")) and self.after_write:
                    word = self.after_write(self)
                    if word:
                        result += "\n" + word
                result = self._clean(result)
                if len(last) >= 3 and len(set(last[-3:])) == 1:  # the same call three times: a loop
                    result += "\n(the same call three times: change your approach, or finish)"
                elif len(last) >= 4 and last[-1] == last[-3] and last[-2] == last[-4]:  # two calls in turn, again and again
                    if len(last) == 6 and last[-5] == last[-1] and last[-6] == last[-2]:
                        self.trace.append("two calls in turn, three times: ended")
                        self.messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": result})
                        break
                    result += "\n(the same two calls again: they change nothing; do something else, or finish)"
                if idle >= 4 and self.owns:  # reading and running without writing
                    result += ("\n(You have read and run four times without changing a file. Decide what is wrong, write "
                               "the fix now (edit_file for a few lines, write_file for the whole file): the check runs after every change.)")
                    idle = 0
                self.messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": result})
                self.log(f"{self.name}: {fn.get('name')} {str(args.get('path') or args.get('command') or '')[:60]}")
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        self.state, self.t1 = ("failed" if self.error else "done"), time.time()
        return self


_OUTPUT = re.compile(r"(?<![\w/.-])([A-Za-z0-9_][\w./-]*\.(?:json|csv|tsv|txt|md|html|xml|ya?ml|log|ini|toml))(?![\w-])", re.I)


def _outputs(task: str, copy: Path, base: Path) -> Dict[str, str]:
    """Data files the task names (report.json, big_logs.txt) that an agent's commands made or changed
    in its copy, to go with its version: a part that wrote report.py and ran it, as the task said,
    passed in its copy and then failed in the project, where report.json never came."""
    out = {}
    for name in dict.fromkeys(_OUTPUT.findall(task)):
        rel = _rel(name)
        p = copy / rel if rel else None
        if p is None or not p.is_file() or p.stat().st_size > 1 << 20:
            continue
        text = p.read_text(errors="replace")
        if (base / rel).is_file() and (base / rel).read_text(errors="replace") == text:
            continue
        out[rel] = text
    return out


def _loose_replace(text: str, old: str, new: str) -> Optional[str]:
    """``old`` found line by line with each line's surrounding whitespace ignored, then also with blank
    lines ignored (small models copy indentation, trailing spaces and blank lines wrong: Gemma 4's
    edits missed that way again and again, and a third of all repair edits did); the lines it
    matches, once, replaced by ``new``, shifted by the indentation ``old`` was off by (so a loose
    match does not break Python's indentation). None if it matches nowhere or in more than one place."""
    want = [l.strip() for l in old.strip("\n").splitlines()]
    if not want or not any(want):
        return None
    lines = text.splitlines(keepends=True)
    hits = [(i, i + len(want)) for i in range(len(lines) - len(want) + 1) if [l.strip() for l in lines[i:i + len(want)]] == want]
    if not hits:  # blank lines ignored on both sides
        want = [w for w in want if w]
        idx = [i for i, l in enumerate(lines) if l.strip()]
        seen = [lines[i].strip() for i in idx]
        hits = [(idx[k], idx[k + len(want) - 1] + 1) for k in range(len(idx) - len(want) + 1) if seen[k:k + len(want)] == want]
    if len(hits) != 1:
        return None
    a, b = hits[0]
    body = new.strip("\n")
    first_old = next(l for l in old.strip("\n").splitlines() if l.strip())
    have, gave = lines[a][:len(lines[a]) - len(lines[a].lstrip())], first_old[:len(first_old) - len(first_old.lstrip())]
    if body and have != gave:
        out = []
        for l in body.splitlines():
            if not l.strip():
                out.append(l)
            elif have.startswith(gave):  # old was indented less than the file: so is new
                out.append(have[len(gave):] + l)
            elif gave.startswith(have) and l.startswith(gave[len(have):]):  # indented more
                out.append(l[len(gave) - len(have):])
            else:
                out.append(l)
        body = "\n".join(out)
    return "".join(lines[:a]) + (body + "\n" if body else "") + "".join(lines[b:])


def _closest(text: str, old: str, room: int = 40) -> str:
    """The lines of ``text`` most like ``old`` (a line around them), for an edit that missed: the agent
    copies its next old text from them instead of reading the whole file again or rewriting it."""
    import difflib

    lines = text.splitlines()
    want = [l.strip() for l in old.strip("\n").splitlines()] or [""]
    n = min(len(want), max(1, len(lines)))
    target = "\n".join(want)
    best = max(range(max(1, len(lines) - n + 1)),
               key=lambda i: difflib.SequenceMatcher(None, "\n".join(l.strip() for l in lines[i:i + n]), target, autojunk=False).ratio())
    a, b = max(0, best - 1), min(len(lines), best + n + 1)
    return "\n".join(lines[a:min(b, a + room)])


def _shared_prompt(task: str, parts: List[dict], digest: str) -> str:
    """What every worker reads first, in the same words (the server computes this prefix once)."""
    plan = "\n".join(f"- part {i}, {p['title']}: {', '.join(p['files'])}" for i, p in enumerate(parts))
    return f"The project as it is:\n{digest}\n\nThe task:\n{task.strip()}\n\nThe team's parts:\n{plan}"


TEST_HINT = (
    "Test the code as it is written: read it in the prompt above and call it the way it works (its functions, their "
    "arguments, what they return). If it reads or writes files relative to the current directory, make a temporary "
    "directory (tempfile.mkdtemp), write a small input there (3 to 5 lines) and run the code with that directory as "
    "the current one (os.chdir in setUp and back in tearDown, or subprocess.run([...], cwd=tmp)). Work the expected "
    "values out by hand from that small input. Run your test before you finish.")


def _texts(root: Path, files: List[str], limit: int = 9000) -> str:
    """The text of ``files`` as they are in ``root``, for a prompt (one read step less for every agent)."""
    out, used = [], 0
    for f in files:
        p = root / f
        if not p.is_file():
            continue
        t = p.read_text(errors="replace")
        if used + len(t) > limit:
            out.append(f"--- {f}: {len(t)} characters, read it with read_file")
            continue
        used += len(t)
        out.append(f"--- {f}\n{t.rstrip()}")
    return "\n".join(out)


def worker_prompt(task: str, parts: List[dict], i: int, digest: str, check: Optional[str] = None,
                  root: Optional[Path] = None) -> str:
    me = parts[i]
    done = [f for j in me["needs"] for f in parts[j]["files"]]
    text = [_shared_prompt(task, parts, digest)]
    if done:  # the same for every part that builds on them, so it stays in the shared prefix
        shown = _texts(root, done) if root else ""
        text.append("Written already, by the parts you build on:\n" + shown if shown else
                    f"Written already, by the parts you build on (read them first): {', '.join(done)}.")
    text.append(f"Your part: part {i}, {me['title']}.\n{me['instructions']}\nYour files: {', '.join(me['files'])}.")
    if me["files"] and all(_is_test(f) for f in me["files"]):
        text.append(TEST_HINT)
    if check:
        text.append(f"The project's check is:\n    {check}\nIt runs after each of your writes, and what it says comes back "
                    "with the write: failures in your own files or functions are yours to fix; failures in the other parts' "
                    "are not (they may not be finished yet).")
    text.append("The other parts are being written at the same time; use them as the task describes. Write your "
                "files, check them, and finish." if not done else "Write your files, check them, and finish.")
    return "\n\n".join(text)


_GOES_ON = re.compile(r"[ \t]|\w*(Error|Exception)\b")  # a traceback's next line: indented, or the exception


def _mine(out: str, part: dict, files: Dict[str, str], limit: int = 1500) -> str:
    """The lines of a failing check's output that point at ``part`` (its file names, or a call of a
    function or class it defines), with the traceback line before and what goes on after (the
    AssertionError): a repair agent given the whole output fixed another part's failure, the first
    one listed, and spent its steps on a file it may not change."""
    defs = {n for f, c in files.items() if f.endswith(".py") for n in _DEF.findall(c) if len(n) > 2}
    lines = out.splitlines()
    keep: List[int] = []
    for i, line in enumerate(lines):
        if any(_names(f, line) for f in part["files"]) or any(re.search(r"\b" + re.escape(n) + r"\(", line) for n in defs):
            if i and lines[i - 1].strip().startswith('File "'):
                keep.append(i - 1)
            keep.append(i)
            j = i + 1
            while j < len(lines) and j < i + 4 and _GOES_ON.match(lines[j]):
                keep.append(j)
                j += 1
    return "\n".join(lines[j] for j in sorted(set(keep)))[-limit:]


_IMPORT = re.compile(r"^[ \t]*(?:from[ \t]+([\w.]+)[ \t]+import[ \t]+([\w, \t]+)|import[ \t]+([\w.]+))", re.M)


def _uses(parts: List[dict], cur: List[Dict[str, str]], i: int) -> List[int]:
    """The other code parts whose Python files part ``i``'s files import (cli.py's "from inventory import
    Inventory" -> the part that writes inventory.py; "from . import counting" -> textstats/counting.py)."""
    mods = set()
    for f, c in cur[i].items():
        if f.endswith(".py"):
            for frm, names, imp in _IMPORT.findall(c):
                if imp:
                    mods.add(imp)
                elif frm.strip("."):
                    mods.add(frm.lstrip("."))
                else:  # from . import a, b
                    mods.update(n.strip() for n in names.split(",") if n.strip())
    out = []
    for j, p in enumerate(parts):
        mod = [f[:-3].replace("/", ".") for f in p["files"] if f.endswith(".py") and not _is_test(f)]
        if j != i and any(m == x or x.endswith("." + m) or m.endswith("." + x.rsplit(".", 1)[-1]) for m in mods for x in mod):
            out.append(j)
    return out


def fixer_prompt(task: str, parts: List[dict], i: int, digest: str, check: str, out: str,
                 root: Optional[Path] = None, blamed: Optional[List[int]] = None, users: Optional[List[str]] = None) -> str:
    me = parts[i]
    files = [f for p in parts for f in p["files"]]
    text = [_shared_prompt(task, parts, digest),
            f"All parts are written, and the project's check fails:\n    {check}\nIts output (the end of it):\n{out[-2500:]}"]
    shown = _texts(root, files, limit=5000) if root else ""
    if shown:  # every fixer of this round reads the same project: a shared prefix
        text.append("The project's files now:\n" + shown)
    text.append(f"Your part: part {i}, {me['title']}. Your files: {', '.join(me['files'])}.")
    mine = _mine(out, me, {f: (root / f).read_text(errors="replace") for f in me["files"] if (root / f).is_file()} if root else {})
    if mine:  # the failures other parts' fixers work on are not this one's
        text.append(f"What fails in your files (from the output above):\n{mine}")
    if users:  # a part repaired because the failing part, repaired alone, did not get better
        text.append(f"What fails shows in {', '.join(users)}, which use{'s' if len(users) == 1 else ''} your files, and fixing "
                    f"{'it' if len(users) == 1 else 'them'} alone did not help: the cause may be in your code. Read how your code "
                    "is called there, and fix your side (how it loads, stores, computes or returns things).")
    tests_only = blamed is not None and all(parts[j]["files"] and all(_is_test(f) for f in parts[j]["files"]) for j in blamed)
    if tests_only:
        text.append("The check blames the tests only, so take the code as right: make the tests check what the task "
                    "asks, calling the code the way it works. " + TEST_HINT)
    text.append("Find in the output what fails in your files and why. Fix it first (edit_file for a few lines, write_file "
                "for the whole file): the check runs after every change, and what it says comes back with it. Repeat "
                "until what fails is no longer in your files; then finish.")
    return "\n\n".join(text)


def _progress(out: str, before: str) -> bool:
    """Whether a repair round got somewhere: fewer failed tests (code that loads, where it did not), or as
    many but another failure (the tests fixed, and the code they then found wrong)."""
    now, was = _score(out, None)[:2], _score(before, None)[:2]
    return now < was or (now == was and _key(out) != _key(before))


def part_verifier(root: Path, check: Optional[str], parts: List[dict], i: int):
    """None when the check passes or blames other parts only; else the end of its output."""
    if not check:
        return None

    def verify() -> Optional[str]:
        ok, out, _ = run_check(check, root, keep=6000)
        if ok:
            return None
        cur = [{f: (root / f).read_text(errors="replace") for f in p["files"] if (root / f).is_file()} for p in parts]
        return out[-2000:] if i in _blame(out, parts, cur) else None

    return verify


# Linked into every copy of the project, not copied (big, and the check may run from them); top level only.
_LINK_DIRS = (".venv", "venv", "env", "node_modules")
class Run:
    """One locals run: plan, workers in parallel, check, fixers in parallel.

    Every agent works in a copy of a staging copy of the project; the project itself is written
    once, at the end, and only when the check passes (or there is no check): a run that fails, is
    stopped or crashes leaves it as it was, and keeps its best attempt in ``keep_dir``."""

    def __init__(self, task: str, root: Path, base_url: str, model: str, check: Optional[str], max_parts: int = 8,
                 workers: int = 0, rounds: int = 2, max_steps: int = 10, log=print, status=None, versions: int = 3,
                 race: int = 2, head: float = 0, keep_dir: Optional[Path] = None):
        self.project = Path(root).resolve()
        self.root = self.project  # the staging copy, once go() has made it
        self.task, self.base_url, self.model, self.check = task, base_url, model, check
        self.given_check = check  # the run's own check (None: the tests the parts write become one)
        self.k = max(1, versions if workers != 1 else 1)
        self.at_once = 6  # agents in flight at most, the whole-task ones included: the prompt cache must hold their
        # conversations (Qwen3.5-4B, 16 GB: with eight agents of 10k tokens a third of the steps found nothing cached)
        self.max_parts, self.workers, self.rounds, self.max_steps = max_parts, workers, rounds, max_steps
        self.log, self.status = log, status or (lambda agents: None)
        self.keep_dir = keep_dir
        self.agents: List[Agent] = []
        self.report: dict = {}
        self.solved = threading.Event()  # some agent's copy passes the whole check: every agent stops
        # whether the check passing means the task is done: not when it passed before any change (a feature added
        # to a project whose tests pass), where it only guards that nothing breaks
        self.check_ends = True
        self.won = threading.Event()  # a whole-task agent's copy passes the check: the run is done
        self.solo: List[Agent] = []
        self.solos = race  # whole-task agents racing the parts (a task one agent does fast is not slowed by the split)
        self.head = head  # seconds the whole-task agents run alone first
        self._copies: List[Path] = []
        self._parts: List[dict] = []
        self.touched: set = set()  # files the run wrote into the staging copy
        self.readonly: set = set()  # files no agent writes (see _owned)
        self._links: List[str] = []  # top-level folders linked into the copies, not copied (see go)
        self._start: dict = {}  # the project's files at the start: relative path -> sha1 (see _finish)

    # -- the plan -----------------------------------------------------------------------------------
    def plan(self) -> List[dict]:
        t0 = time.time()
        parts = make_plan(self.base_url, self.model, self.task, project_digest(self.project), project=self.project, log=self.log)
        parts = self._owned(parts[: self.max_parts] if parts else [])
        parts = tests_after_code(parts) if parts else []
        if not parts:  # nothing to split: one part, every file the agent decides (normalize drops parts without files)
            parts = [{"title": "the whole task", "files": [], "instructions": "Do the whole task.", "needs": []}]
        if self.workers == 1:  # the single-agent baseline: one agent owns every file, with the parts' steps
            self.max_steps *= max(1, len(parts))
            parts = [{"title": "the whole task", "files": sorted({f for p in parts for f in p["files"]}),
                      "instructions": "Do the whole task.", "needs": []}]
        self.report["plan_s"] = round(time.time() - t0, 1)
        self.report["parts"] = [{"title": p["title"], "files": p["files"], "needs": p["needs"]} for p in parts]
        return parts

    def _owned(self, parts: List[dict]) -> List[dict]:
        """Each file owned by one part (the first that names it: two parts writing one file at once
        would leave the last one's); and with a check, the test files that exist are not anyone's,
        unless the task asks to change tests: they are the check, and an agent that rewrote them could
        make it pass ("fix inventory.py so that test_inventory.py passes")."""
        free = asks_test_changes(self.task) or not self.check
        if not free:  # also out of reach of an agent that owns every file (the whole task, a plan without files)
            self.readonly = {str(f.relative_to(self.project)) for f in self.project.rglob("*.py")
                             if _is_test(str(f.relative_to(self.project))) and not (set(f.relative_to(self.project).parts) & SKIP_DIRS)}
        seen, kept, out = set(), [], []
        for p in parts:
            files = []
            for f in p["files"]:
                if f in seen:
                    continue
                if not free and _is_test(f) and (self.project / f).exists():
                    kept.append(f)
                    continue
                seen.add(f)
                files.append(f)
            if files:
                out.append({**p, "files": files})
        if kept:
            self.report["tests_kept"] = kept
            self.log("the tests that exist stay as they are (they are the check): " + ", ".join(kept))
        return out

    # -- agents ------------------------------------------------------------------------------------
    def _run_agents(self, agents: List[Agent], groups: Optional[Dict[int, List[Agent]]] = None) -> None:
        """All at once: every agent is a request in flight, one row of the server's batch. In a group
        (the versions of one part), the first version the check confirms stops the others; when all but
        a few agents are done, the rest get half the time so far (20 s at least), then stop."""
        threads = [threading.Thread(target=a.run, daemon=True) for a in agents]
        t0, cut = time.time(), None
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            if self.solved.is_set() or self.won.is_set():
                for a in agents:
                    a.stop.set()
            for g in (groups or {}).values():
                if any(a.verified for a in g):
                    for a in g:
                        a.stop.set()
            alive = sum(t.is_alive() for t in threads)
            if cut is None and len(threads) > 1 and alive <= max(1, len(threads) // 4):  # not a lone agent
                cut = time.time() + max(20.0, 0.5 * (time.time() - t0))
            if cut is not None and time.time() > cut:
                for a in agents:
                    a.stop.set()
            self.status(self.agents)
            time.sleep(0.5)

    def _after_write(self, parts: List[dict], i: int):
        """The check, run in the agent's copy after each of its writes; its word goes back with the
        write's result, so the agent sees what still fails without spending a step to run it (small
        models spent half their steps running the check and reading its output). A copy that passes
        the whole check ends the stage for every agent."""

        def after(a: Agent) -> Optional[str]:
            t = time.time()
            ok, out, _ = run_check(self.check, a.root, keep=6000)
            if time.time() - t > 30:  # a slow check: once is enough
                a.after_write = None
            if ok and not self.check_ends:
                return "The project's check still passes."
            if ok:
                a.verified = True
                self.solved.set()
                return "The project's check passes now. Call finish."
            cur = [{f: (a.root / f).read_text(errors="replace") for f in p["files"] if (a.root / f).is_file()} for p in parts]
            if i in _blame(out, parts, cur):
                return "The project's check after this write still fails in your part:\n" + out[-1500:]
            return "The project's check after this write: nothing it reports is in your files."

        return after

    def _start_solo(self, parts: List[dict], digest: str) -> List[threading.Thread]:
        """Whole-task agents, each in its own copy, racing the parts in the same batch: the first whose
        copy passes the check wins the run. One agent alone did the easy tasks in a fifth of the time
        the parts took (code, then tests, then the pick); the parts did the hard ones it failed."""
        files = sorted({f for p in parts for f in p["files"]})
        whole = [{"title": "the whole task", "files": files, "instructions": "Do the whole task: every file of the plan.",
                  "needs": []}]
        threads = []
        for v in range(self.solos):
            d = self._copy(f"s{v}")
            a = Agent(f"s{v}", d, self.base_url, self.model, files or _AnyFile(),
                      worker_prompt(self.task, whole, 0, digest, self.check), self.max_steps * len(parts),
                      verify=part_verifier(d, self.check, whole, 0), temperature=_TEMPS[v % len(_TEMPS)],
                      after_write=self._solo_word, readonly=self.readonly)
            self.solo.append(a)
            self.agents.append(a)
            t = threading.Thread(target=a.run, daemon=True)
            t.start()
            threads.append(t)
        return threads

    def _solo_word(self, a: Agent) -> Optional[str]:
        t = time.time()
        ok, out, _ = run_check(self.check, a.root, keep=6000)
        if time.time() - t > 30:
            a.after_write = None
        if ok and not self.check_ends:
            return "The project's check still passes."
        if ok:
            if a.won_files is None:  # its files as they passed (the agent may go on and change them)
                names = sorted({f for p in self._parts for f in p["files"]} | set(a.wrote))
                a.won_files = {f: (a.root / f).read_text(errors="replace") for f in names if (a.root / f).is_file()}
                a.won_files.update(_outputs(self.task, a.root, self.root))
            a.verified = True
            self.won.set()
            a.stop.set()
            return "The project's check passes now. Call finish."
        return "The project's check after this change still fails:\n" + out[-1500:]

    def _take_solo(self) -> Optional[str]:
        """A winning whole-task agent's files, as they passed, into the staging copy if the check passes
        there too; its output, or None (and the staging copy as it was)."""
        for a in self.solo:
            if a.won_files is None:
                continue
            before = {f: ((self.root / f).read_text(errors="replace") if (self.root / f).is_file() else None) for f in a.won_files}
            self._stage(a.won_files)
            ok, out, _ = run_check(self.check, self.root, keep=8000)
            if ok:
                self.report["won_by"] = a.name
                return out
            for f, c in before.items():  # put back what was there
                if c is None:
                    (self.root / f).unlink(missing_ok=True)
                else:
                    _put(self.root, {f: c})
        return None

    def _more(self, n: int) -> int:
        """Versions for the tests and the repairs: ``n`` more than the code's (they start with fewer parts,
        so the batch has the rows); one for the single-agent baseline."""
        return 1 if self.workers == 1 else self.k + n

    # -- copies ------------------------------------------------------------------------------------
    def _copy(self, tag: str, src: Optional[Path] = None) -> Path:
        """A copy of the staging copy (or ``src``) as it is now, for one agent: its own files to write, its own
        check. Caches and .git are left out; .venv and node_modules are links to the project's own."""
        import os
        import shutil
        import tempfile

        src = (src or self.root).resolve()
        base = Path(tempfile.mkdtemp(prefix=f"locals-run-{tag}-"))
        self._copies.append(base)  # before the copy: a copy that fails half-way is removed too

        def ignore(d, names):
            skip = {n for n in names if n in _COPY_SKIP}
            if Path(d).resolve() == src:
                skip |= {n for n in names if n in self._links}
            return skip

        d = base / "w"
        shutil.copytree(src, d, symlinks=True, ignore=ignore, ignore_dangling_symlinks=True)
        for dirpath, dirnames, filenames in os.walk(d):  # a relative link out of the project would dangle here
            for name in dirnames + filenames:
                link = Path(dirpath) / name
                if link.is_symlink() and not os.path.isabs(os.readlink(link)):
                    orig = src / link.relative_to(d)
                    target = Path(os.path.realpath(orig))
                    if target != src and src not in target.parents:
                        link.unlink()
                        os.symlink(target, link)
        for name in self._links:
            if not (d / name).exists():
                os.symlink(self.project / name, d / name)
        return d

    def _stage(self, files: Dict[str, str]) -> None:
        """Files into the staging copy (the project is written at the end, from there); a file whose
        text is what the staging copy has is left as it is (not rewritten with a final newline)."""
        changed = {}
        for f, c in files.items():
            p = self.root / f
            try:
                same = p.is_file() and p.read_text(errors="replace") == c
            except OSError:
                same = False
            if not same:
                changed[f] = c
        _put(self.root, changed)
        self.touched.update(changed)

    def _current(self, parts: List[dict]) -> List[Dict[str, str]]:
        return [{f: (self.root / f).read_text(errors="replace") for f in p["files"] if (self.root / f).is_file()} for p in parts]

    def _versions(self, parts: List[dict], todo: List[int], k: int, prompt, tag: str) -> Dict[int, List[Dict[str, str]]]:
        """``k`` agents for each part in ``todo`` (fewer when that is more than ``self.at_once`` agents),
        all at once, each in its own copy of the project; the versions of the part's files they leave
        (an agent that wrote nothing leaves none). The cap is memory: each agent's conversation stays
        in the server's prompt cache only while they all fit (Qwen3.5-4B, 16 GB: about eight agents
        of 5k tokens; with twelve, every step prefilled its whole prompt again)."""
        if self.won.is_set():  # a whole-task agent passed: nothing more to do
            return {i: [] for i in todo}
        room = self.at_once - sum(1 for a in self.solo if a.state != "done" and a.state != "failed")
        k = max(1, min(k if self.check else 1, max(1, room) // max(1, len(todo))))  # no check: nothing to pick by
        self.solved.clear()
        jobs = []
        for i in todo:
            for v in range(k):
                d = self._copy(f"{tag}{i}.{v}")
                a = Agent(f"{tag}{i}.{v}", d, self.base_url, self.model, parts[i]["files"] or _AnyFile(), prompt(i, v),
                          self.max_steps, verify=part_verifier(d, self.check, parts, i), temperature=_TEMPS[v % len(_TEMPS)],
                          after_write=self._after_write(parts, i) if self.check else None, readonly=self.readonly)
                jobs.append((i, a))
        self.agents += [a for _, a in jobs]
        groups: Dict[int, List[Agent]] = {}
        for i, a in jobs:
            groups.setdefault(i, []).append(a)
        self._run_agents([a for _, a in jobs], groups)
        out: Dict[int, List[Dict[str, str]]] = {i: [] for i in todo}
        for i, a in jobs:
            files = {f: (a.root / f).read_text(errors="replace") for f in (parts[i]["files"] or a.wrote) if (a.root / f).is_file()}
            if files:
                files.update(_outputs(self.task, a.root, self.root))
            if files and a.wrote and files not in out[i]:
                out[i].append(files)
        return out

    def _pick(self, parts: List[dict], options: List[List[Dict[str, str]]], current_out: Optional[str]) -> tuple:
        """The best combination of the parts' versions by the check (whole versions first, then a
        part-by-part search), written into the staging copy. Every part has at least one option (an
        empty one: the part as it is). Returns (passed, output)."""
        from types import SimpleNamespace

        n = len(options)
        if n == 0 or any(not o for o in options):
            ok, out, _ = run_check(self.check, self.root, keep=8000)
            return ok, out
        checks = _Checks(SimpleNamespace(ws=self.root, race=SimpleNamespace(check=self.check)))
        longest = max(len(o) for o in options)
        diag = list(dict.fromkeys(tuple(min(v, len(options[i]) - 1) for i in range(n)) for v in range(longest)))
        results = _check_all(options, checks, diag)
        tried = set(diag)
        if not any(ok for _, ok, _ in results):
            results += _search(options, checks, tried, results + ([((0,) * n, False, current_out)] if current_out else []))
        best = _best(results, _key(current_out) if current_out else None)
        if best is None:  # nothing was checked: what the staging copy says
            ok, out, _ = run_check(self.check, self.root, keep=8000)
            return ok, out
        if current_out is not None and not best[1] and _score(best[2], _key(current_out)) >= _score(current_out, _key(current_out)):
            return False, current_out  # nothing better than what the project has
        self._stage(_merge(options, best[0]))
        self.report.setdefault("picks", []).append({"checked": len(results), "passed": best[1]})
        return best[1], best[2]

    # -- the run -----------------------------------------------------------------------------------
    def go(self) -> dict:
        """The run; the project is written only at the end (see the class). Exceptions and Ctrl-C
        propagate, with the copies removed and the project untouched."""
        import shutil

        from .solve import manifest

        t0 = time.time()
        ok, out, checks, fixes, solo = False, None, [], [], []
        # .venv / venv / env are linked only where they are environments (a source package may be called env)
        self._links = [n for n in _LINK_DIRS if (self.project / n).is_dir()
                       and (n == "node_modules" or (self.project / n / "pyvenv.cfg").exists())]
        self._start = manifest(self.project)
        finished = False
        try:
            self.root = self._copy("stage", self.project)
            parts = self._parts = self.plan()
            self.log(f"plan ({self.report['plan_s']:.0f}s): " + "; ".join(
                f"[{i}] {', '.join(p['files']) or p['title']}" + (f" after {p['needs']}" if p["needs"] else "")
                for i, p in enumerate(parts)))
            digest = project_digest(self.root, limit=6000)
            n = len(parts)
            code = [i for i, p in enumerate(parts) if not p["needs"]]
            later = [i for i in range(n) if i not in code]
            if self.check:
                self.report["passed_before"] = run_check(self.check, self.root, keep=2000)[0]
                if self.report["passed_before"]:
                    # every first write would "pass": the agents stopped at once and a half-done task was written
                    self.check_ends = False
                    self.log("the check passes before any change: it keeps the project from breaking, but cannot tell "
                             "when the task is done, so the agents finish their parts")
            if self.workers != 1 and self.check and n > 1 and self.solos and self.check_ends:
                solo = self._start_solo(parts, digest)
            # A head start: the whole-task agents alone, at the batch's full speed per stream (off by default:
            # it helped neither Qwen3.5 nor Gemma 4)
            until = time.time() + self.head
            while solo and time.time() < until and not self.won.is_set() and any(t.is_alive() for t in solo):
                self.status(self.agents)
                time.sleep(0.5)
            # 1. the parts that need nothing, k versions each, at once (a part that got none: as it is)
            t1 = time.time()
            cur = self._current(parts)
            vers = self._versions(parts, code, self.k, lambda i, v: worker_prompt(self.task, parts, i, digest, self.check), "w")
            options = [vers.get(i) or [cur[i]] for i in code]
            if self.check and not self.won.is_set():
                ok, out = self._pick([parts[i] for i in code], options, None)
            elif not self.check:
                for o in options:
                    self._stage(o[0])
            # 2. the parts that build on them (tests), in copies with that code
            if later and not self.won.is_set():
                vers = self._versions(parts, later, self._more(2),
                                      lambda i, v: worker_prompt(self.task, parts, i, digest, self.check, self.root), "t")
                cur = self._current(parts)
                # the part as it is first, unless that passes anyway: then the new tests the task asked for came last
                options = [([cur[i]] + vers.get(i, []) if self.check_ends else vers.get(i, []) + [cur[i]])
                           if i in later else [cur[i]] for i in range(n)]
                if self.check:
                    ok, out = self._pick(parts, options, None)
                else:
                    for i in later:
                        if vers.get(i):
                            self._stage(vers[i][0])
            self.report["workers_s"] = round(time.time() - t1, 1)
            if not self.check:  # no check given: the tests the parts wrote are one now, for the repairs
                from .solve import detect_check

                found = detect_check(self.root)
                if found:
                    self.check = re.sub(r"\s-x\b", "", found)
                    self.report["check"] = self.check
                    self.log(f"check: {self.check} (the tests the parts wrote)")
            if self.check and out is None and not self.won.is_set():
                ok, out = run_check(self.check, self.root, keep=8000)[:2]
            if self.check:
                checks.append({"ok": ok, "tail": (out or "")[-600:]})
            # 3. repairs: fresh agents for the parts at fault, at once; the check picks. ``rounds``
            # rounds, and up to two more while each round makes progress: fewer failed tests, or as many
            # but another failure (the tests fixed, then the code they found wrong: a run that stopped
            # there had just blamed the code for the first time)
            better, last = False, None
            for r in range(self.rounds + 2):
                if ok or not self.check or self.won.is_set() or (r >= self.rounds and not better):
                    break
                t2 = time.time()
                cur = self._current(parts)
                code_parts = [i for i, p in enumerate(parts) if not all(_is_test(f) for f in p["files"])]
                todo = _blame(out, parts, cur) or code_parts or list(range(n))
                users: Dict[int, List[str]] = {}
                if last is not None and set(todo) == set(last[0]) and _key(out) == last[1]:
                    # the same parts at fault, failing the same way after a round on them: the cause may be in
                    # the code they use (an Inventory that never loaded its file failed only through cli.py)
                    for i in todo:
                        for j in _uses(parts, cur, i):
                            if j not in todo:
                                users.setdefault(j, []).extend(parts[i]["files"])
                last = (list(todo), _key(out))
                todo = todo + list(users)
                fail_out = out
                vers = self._versions(parts, todo, self._more(1),
                                      lambda i, v: fixer_prompt(self.task, parts, i, digest, self.check, fail_out, self.root, todo,
                                                                users.get(i)),
                                      f"f{r + 1}.")
                options = [[cur[i]] + vers.get(i, []) for i in range(n)]
                ok, out = self._pick(parts, options, out)
                better = _progress(out, fail_out)
                checks.append({"ok": ok, "tail": out[-600:]})
                fixes.append({"parts": todo, "wall_s": round(time.time() - t2, 1)})
                self.log(f"repair {r + 1}: parts {todo}, {time.time() - t2:.0f}s, check {'passes' if ok else 'fails'}")
            if solo:
                if not ok:  # the parts failed: the race goes on until a whole-task agent passes or stops
                    while not self.won.is_set() and any(t.is_alive() for t in solo):
                        self.status(self.agents)
                        time.sleep(0.5)
                for a in self.solo:
                    a.stop.set()
                if not ok and self.won.is_set():
                    won = self._take_solo()
                    if won is not None:
                        ok, out = True, won
                        checks.append({"ok": True, "tail": won[-600:], "by": self.report.get("won_by")})
                for t in solo:  # daemons, stopped: a request in flight is not waited for
                    t.join(timeout=2)
            self._finish(ok)
            finished = True
        finally:
            if not finished and self.touched and self.root != self.project:  # stopped or crashed: the work so far kept
                try:
                    self._keep()
                except Exception:  # noqa: BLE001
                    pass
            for d in self._copies:
                shutil.rmtree(d, ignore_errors=True)
        self.report.update(passed=ok if self.check else None, wall_s=round(time.time() - t0, 1), checks=checks, repairs=fixes,
                           agents=[{"name": a.name, "files": list(a.owns) if not isinstance(a.owns, _AnyFile) else ["*"],
                                    "steps": a.steps, "tokens": a.tokens, "wrote": a.wrote, "summary": a.summary, "error": a.error,
                                    "start_s": round(a.t0 - t0, 1) if a.t0 else None, "wall_s": round(a.t1 - a.t0, 1) if a.t1 and a.t0 else None,
                                    "verified": a.verified, "trace": a.trace}
                                   for a in self.agents],
                           most_at_once=_most_at_once(self.agents))
        return self.report

    def _finish(self, ok: bool) -> None:
        """The run's files into the project when its check passes, or when it had none; else the
        staging copy's files are kept (``keep_dir``) and the project is left as it was. A file that
        changed in the project while the run worked is not overwritten (it is kept instead). The
        files are written under temporary names first and then renamed, so the project gets all of
        them or, but for a crash between two renames, none."""
        import hashlib
        import os

        if not (ok or not self.given_check):
            self.report["changed"] = []
            self._keep()
            return
        todo, skipped = [], []
        for f in sorted(self.touched):
            src, dst = self.root / f, self.project / f
            if not src.is_file():
                continue
            data = src.read_bytes()
            now = hashlib.sha1(dst.read_bytes()).hexdigest() if dst.is_file() else None
            if now == hashlib.sha1(data).hexdigest():
                continue
            if now != self._start.get(f):  # edited (or created) in the project during the run
                skipped.append(f)
                continue
            todo.append((f, data))
        top = self.project.resolve()
        tmps = []
        try:
            for f, data in todo:
                dst = (self.project / f).resolve()
                if top not in dst.parents:
                    raise ValueError(f"refusing to write {f}: it is outside the project")
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(f".{dst.name}.locals-{os.getpid()}")
                tmp.write_bytes(data)
                if dst.exists():
                    os.chmod(tmp, dst.stat().st_mode & 0o777)
                tmps.append((tmp, dst))
            for tmp, dst in tmps:
                os.replace(tmp, dst)
            tmps = []
        finally:
            for tmp, _ in tmps:
                tmp.unlink(missing_ok=True)
        self.report["changed"] = [f for f, _ in todo]
        if skipped:
            self.report["skipped"] = skipped
            self._keep(skipped)

    def _keep(self, only=None) -> None:
        """The staging copy's files (those the run wrote, or ``only``) into ``keep_dir``."""
        import shutil

        names = sorted(only if only is not None else self.touched)
        if self.keep_dir is None or not names:
            return
        dst = Path(self.keep_dir) / time.strftime(f"{self.project.name}-%Y%m%d-%H%M%S")
        for f in names:
            if (self.root / f).is_file():
                (dst / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.root / f, dst / f)
        self.report["kept"] = str(dst)


class _AnyFile(list):
    """Owns every file (a task without named files, or the single-agent baseline without names)."""

    def __contains__(self, item) -> bool:  # noqa: D105
        return True


def _most_at_once(agents: List[Agent]) -> int:
    ev = sorted([(a.t0, 1) for a in agents if a.t0] + [(a.t1, -1) for a in agents if a.t1], key=lambda e: (e[0], e[1]))
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


def cmd_run(args) -> None:
    """locals run: the harness from the command line, with a live line of the agents' states."""
    import sys

    from . import app
    from .solve import detect_check

    root = Path(args.dir).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"no such directory: {root}")
    if args.check:  # before the model loads and the agents start: a check that cannot run would fail them all
        from .solve import check_cannot_run

        why = check_cannot_run(args.check, root)
        if why:
            sys.exit(f"locals run: {why}")
    app.fetch_model(args.model)
    app.ensure_server(args)
    app.remember(model=args.model, port=args.port)
    check = args.check or detect_check(root)
    if check and not args.check:  # every failure, not the first: the fixers of all the parts at fault work at once
        check = re.sub(r"\s-x\b", "", check)
    model = f"{getattr(args, 'engine', None) or 'locals'}/{app.model_short(args.model)}"
    t0 = time.time()
    last = [""]
    rows = {"n": "?", "at": 0.0}
    tty = sys.stdout.isatty()

    def status(agents):
        """One line, redrawn: the time, how many agents work, are done or failed, the working ones with their
        steps, and the server's batch. It listed every agent of the run (30 and more), wrapped, and filled the
        terminal with half lines; /stats was asked twice a second."""
        if time.time() - rows["at"] > 2:
            try:
                rows["n"] = len(app.get_json(app.base_url(args.port) + "/stats", timeout=2)["active"])
            except Exception:  # noqa: BLE001
                rows["n"] = "?"
            rows["at"] = time.time()
        working = [a for a in agents if a.state == "working"]
        done = sum(a.state == "done" for a in agents)
        failed = sum(a.state == "failed" for a in agents)
        head = f"  {time.time() - t0:4.0f}s  {len(working)} working, {done} done" + (f", {failed} failed" if failed else "")
        tail = f"  | batch {rows['n']} rows"
        names = "  " + " ".join(f"{a.name}:{a.steps}" for a in working)
        width = shutil.get_terminal_size((100, 20)).columns - 1
        if len(head) + len(names) + len(tail) > width:
            names = names[: max(0, width - len(head) - len(tail) - 3)] + "..."
        line = head + names + tail
        if line != last[0] and tty:
            print("\r\x1b[2K" + line, end="", flush=True)
            last[0] = line

    print(f"locals run in {root}\n  model {model}; check: {check or 'none (the workers check their own parts)'}", flush=True)
    if check and str(root) in check:
        print(f"  note: the check names the project's folder; the agents work in copies of it, so it should use paths "
              f"relative to the project (it runs from the project's folder)", flush=True)
    run = Run(args.task, root, app.base_url(args.port), model, check, max_parts=args.parts, workers=args.workers,
              rounds=args.rounds, max_steps=args.steps, log=lambda m: print("\n  " + m, flush=True), status=status,
              versions=args.versions, race=getattr(args, "race", 2), head=getattr(args, "head", 0),
              keep_dir=app.LOCALS_DIR / "runs")
    try:
        report = run.go()
    except KeyboardInterrupt:
        kept = run.report.get("kept")
        sys.exit("\n  stopped; the project was not changed" + (f"; the work so far is in {kept}" if kept else ""))
    except Exception as e:  # noqa: BLE001
        if args.report:
            Path(args.report).write_text(json.dumps({**run.report, "error": f"{type(e).__name__}: {e}"}, indent=1))
        sys.exit(f"\n  locals run failed: {type(e).__name__}: {e}\n  the project was not changed")
    print()
    for a in report["agents"]:
        print(f"  {a['name']:6s} {', '.join(a['files'])[:50]:50s} {a['steps']:2d} steps {a['wall_s'] or 0:6.1f}s  {a['summary'][:70]}"
              + (f"  ERROR {a['error'][:60]}" if a["error"] else ""))
    verdict = {True: "the check passes", False: "the check FAILS", None: "no check"}[report["passed"]]
    print(f"\n  {verdict}; {report['wall_s']:.0f}s in all, {len(report['agents'])} agents, {report['most_at_once']} at once at most")
    if report.get("won_by"):
        print(f"  the whole-task agent {report['won_by']} passed the check before the parts did")
    if report.get("changed"):
        print(f"  written into the project: {', '.join(report['changed'])}")
    if report.get("skipped"):
        print(f"  not written, changed in the project during the run: {', '.join(report['skipped'])} (the run's versions are in "
              f"{report.get('kept')})")
    elif report.get("kept") and not report.get("changed"):
        print(f"  the project was not changed; the best attempt's files are in {report['kept']}")
    said = next((a for a in report["agents"] if a["summary"].strip() and (a["files"] == ["*"] or not a["files"])), None)
    if said and not report.get("changed"):  # a question ("which function parses dates?"): the answer is what the agents said
        print(f"\n  {said['name']} said:\n    " + said["summary"].strip()[:1500].replace("\n", "\n    "))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1))
    sys.exit(0 if report["passed"] in (True, None) else 1)
