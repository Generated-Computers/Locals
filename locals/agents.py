"""Run many tool-using agents in parallel on one local model.

    python -m locals.agents --engine locals --agents 8            # built-in task suite
    python -m locals.agents --engine stock --agents 8            # classical stack
    python -m locals.agents --engine both  --agents 8            # both, then compare
    python -m locals.agents --task "Write hello.py and run it" --agents 4

Every agent is a thread with its own sandbox directory and its own conversation.
Each turn the model must answer with one JSON action (think / tool / path /
content), enforced by constrained decoding, so actions always parse. All agents'
model calls go to one BatchService: whichever agents are waiting for the model
at a given moment decode together in one batch, and each agent's conversation
prefix is served from the prompt cache, so a turn only prefills the new tool
output.

Engines: ``locals`` (our kernels, fused layers, pipelined drip batching, our
constraint engine) or ``stock`` (plain mlx-lm BatchGenerator + llguidance).

Warning: ``run_python`` executes model-written code on this machine (in the
agent's sandbox directory, isolated mode, with a timeout). It is not a security
sandbox. Use --no-exec to disable it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional
from .defaults import DEFAULT_MODEL, MODEL_HELP, model_id

# Tools in the model's native (Qwen3.5) function-calling format. Parameter
# values are raw text between XML tags, so code needs no escaping.
TOOL_SPECS = {
    "list_files": ("List the files in your sandbox directory.", []),
    "read_file": ("Show the contents of a file.", [("path", "file path relative to your sandbox, e.g. main.py")]),
    "write_file": ("Save text as a file (creates or overwrites it).", [("path", "file path relative to your sandbox, e.g. main.py"),
                                                                        ("content", "the complete file text")]),
    "run_file": ("Run a Python file from your sandbox and show its output.", [("path", "the Python file to run, e.g. main.py")]),
    "run_python": ("Run a snippet of Python code in your sandbox and show its output.", [("code", "Python code")]),
    "finish": ("Call this when the task is done.", [("answer", "the final answer or a short summary")]),
}
TOOLS = list(TOOL_SPECS)


def tool_schemas() -> list:
    return [{"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": {p: {"type": "string", "description": d} for p, d in params},
                       "required": [p for p, _ in params]}}} for name, (desc, params) in TOOL_SPECS.items()]


# One reply = exactly one call of a known tool (no free-text preamble: a 2B model
# otherwise writes the whole solution as prose) with its required parameters.
# Enforced by constrained decoding, so every reply parses; a value is any text without "</".
_VAL = r"([^<]|<[^/])*"
ACTION_REGEX = (r"<tool_call>\n(" + "|".join(
    rf"<function={name}>\n" + "".join(rf"<parameter={p}>\n{_VAL}\n</parameter>\n" for p, _ in params)
    for name, (_, params) in TOOL_SPECS.items()) + r")</function>\n</tool_call>")


def parse_action(text: str) -> Optional[dict]:
    import re

    m = re.search(r"<tool_call>\n<function=(\w+)>\n(.*)</function>\n</tool_call>$", text, re.S)
    if not m:
        return None
    act = {"tool": m.group(1), "thought": text[: m.start()].strip()}
    for k, v in re.findall(r"<parameter=(\w+)>\n(.*?)\n</parameter>\n", m.group(2), re.S):
        act[k] = v
    return act


SYSTEM = """You are agent {i}, a careful software agent. You work alone in your own sandbox directory and solve the task with the tools, one tool call per reply.
Use file paths relative to your sandbox, like main.py. Write complete, correct Python. Check the output of your code before you call finish."""


# --- built-in task suite ---------------------------------------------------------------

def _run_file(d: Path, name: str) -> str:
    if not (d / name).exists():
        return ""
    try:
        return subprocess.run([sys.executable, "-I", name], cwd=d, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def _fizz(n):
    return "FizzBuzz" if n % 15 == 0 else "Fizz" if n % 3 == 0 else "Buzz" if n % 5 == 0 else str(n)


def _fibs(n):
    a, b, out = 0, 1, []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out


@dataclass
class Task:
    name: str
    prompt: str
    check: Optional[Callable[[Path, str], bool]] = None  # (sandbox, final answer) -> passed


SUITE = [
    Task("fizzbuzz", "Write fizzbuzz.py that prints FizzBuzz for the numbers 1 to 15, one per line (Fizz for multiples of 3, "
         "Buzz for multiples of 5, FizzBuzz for both). Run it to check.",
         lambda d, a: _run_file(d, "fizzbuzz.py").split() == [_fizz(i) for i in range(1, 16)]),
    Task("primes", "Create primes.py with a function is_prime(n) and make it print the list of primes below 50 as a Python list. Run it.",
         lambda d, a: str([p for p in range(2, 50) if all(p % k for k in range(2, p))]) in _run_file(d, "primes.py")),
    Task("reverse", "Write reverse.py that prints the string 'locals agents' reversed. Run it.",
         lambda d, a: _run_file(d, "reverse.py").strip() == "stnega nocts"),
    Task("squares", "Compute the sum of the squares of the integers 1 to 100 with Python, then finish with the number as the answer.",
         lambda d, a: "338350" in a),
    Task("fib", "Write fib.py that prints the first 15 Fibonacci numbers separated by spaces, starting with 0 1. Run it.",
         lambda d, a: _run_file(d, "fib.py").split() == [str(x) for x in _fibs(15)]),
    Task("vowels", "Count the vowels (a, e, i, o, u) in the sentence 'Parallel agents share one batch' using Python, then finish "
         "with the count as the answer.",
         lambda d, a: a.strip().rstrip(".").split()[-1:] == ["10"] or a.strip() == "10"),
    Task("csv", "Create data.csv with a header line name,age and 4 rows of made-up people. Then compute the average age with "
         "Python and finish with it as the answer.",
         lambda d, a: (d / "data.csv").exists() and len((d / "data.csv").read_text().strip().splitlines()) == 5),
    Task("wordcount", "Write word_count.py that finds the most common word in the text 'the quick brown fox jumps over the lazy "
         "dog the end' and prints only that word. Run it.",
         lambda d, a: _run_file(d, "word_count.py").strip() == "the"),
]


# --- one agent ---------------------------------------------------------------------------

@dataclass
class AgentState:
    i: int
    task: Task
    dir: Path
    step: int = 0
    status: str = "waiting"
    last_tool: str = ""
    tokens_out: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    answer: str = ""
    finished: bool = False
    passed: Optional[bool] = None
    t_model: float = 0.0
    t_tools: float = 0.0
    transcript: List[dict] = field(default_factory=list)


def _run(base: Path, argv: list) -> str:
    """The model's program, in its sandbox folder, with its whole process group ended after it (or after 10 s)."""
    from .solve import run_shell
    import shlex

    code, out = run_shell(" ".join(shlex.quote(a) for a in [sys.executable, "-I"] + argv), base, 10)
    if code is None:
        return "error: timed out after 10 s"
    return out.strip()[:1500] or "(no output)"


def _tool(st: AgentState, act: dict, allow_exec: bool) -> str:
    tool, path = act.get("tool"), (act.get("path") or "").strip()
    base = st.dir.resolve()
    if path.startswith(("/", "~")) or ":" in path[:3]:
        path = Path(path.replace("\\", "/")).name  # absolute paths: keep the file name, inside the sandbox
    target = (base / path).resolve() if path else base
    if path and base not in target.parents and target != base:
        return "error: path must stay inside the sandbox"
    if tool == "list_files":
        files = sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file())
        return "\n".join(files) or "(no files)"
    if tool == "read_file":
        if not target.is_file():
            return f"error: no such file {path!r}"
        return target.read_text(errors="replace")[:2000]
    if tool == "write_file":
        if not path or target == base or target.is_dir():
            return "error: write_file needs a file's path"
        content = act.get("content", "")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content + ("\n" if content and not content.endswith("\n") else ""))
        return f"wrote {len(content)} characters to {path}"
    if tool in ("run_file", "run_python"):
        if not allow_exec:
            return "error: code execution is disabled"
        if tool == "run_file":
            if not target.is_file():
                return f"error: no such file {path!r}"
            return _run(base, [target.name if target.parent == base else str(target)])
        return _run(base, ["-c", act.get("code", "")])
    if tool == "finish":
        return "done"
    return f"error: unknown tool {tool!r}"


def first_prompt(tok, st: AgentState) -> List[int]:
    msgs = [{"role": "system", "content": SYSTEM.format(i=st.i)}, {"role": "user", "content": "Task: " + st.task.prompt}]
    return tok.apply_chat_template(msgs, tools=tool_schemas(), add_generation_prompt=True, enable_thinking=False, tokenize=True)


class ModelFormat:
    """The tool-call format of the model's chat template (locals.formats). Qwen3.5's
    XML calls keep the typed one-call grammar and the incremental prompt above;
    any other model re-renders the conversation with its own template each step
    (the prompt cache reuses the unchanged prefix) and decodes exactly one call in
    its own format."""

    def __init__(self, tok):
        from . import formats

        self.tok = tok
        self.fmt = formats.detect(tok)
        self.native = self.fmt.name == "qwen_xml"
        self.kw = {} if formats.thinking_style(tok) == "none" else {"enable_thinking": False}
        self.regex = ACTION_REGEX if self.native else self.fmt.calls(self.fmt.call(TOOLS), many=False, at_least_one=True)

    def render(self, msgs: List[dict]) -> List[int]:
        m, tools = self.fmt.prepare([dict(x) for x in msgs], tool_schemas())
        return self.tok.apply_chat_template(m, tools=tools, add_generation_prompt=True, tokenize=True, **self.kw)

    def parse(self, text: str) -> Optional[dict]:
        if self.native:
            return parse_action(text)
        calls = self.fmt.parse_calls(self.fmt.split(text)[1])
        if not calls:
            return None
        name, args = calls[0]
        act = {"tool": name, "thought": ""}
        act.update({k: v if isinstance(v, str) else json.dumps(v) for k, v in args.items()})
        return act


def run_agent(svc, st: AgentState, max_steps: int, max_tokens: int, allow_exec: bool, mf: Optional["ModelFormat"] = None) -> None:
    tok = svc.tokenizer
    mf = mf or ModelFormat(tok)
    msgs = [{"role": "system", "content": SYSTEM.format(i=st.i)}, {"role": "user", "content": "Task: " + st.task.prompt}]
    history = first_prompt(tok, st) if mf.native else mf.render(msgs)
    st.dir.mkdir(parents=True, exist_ok=True)
    while st.step < max_steps:
        st.step += 1
        st.status = "model"
        t = time.perf_counter()
        job = svc.submit(history, regex=mf.regex, max_tokens=max_tokens).wait()
        st.t_model += time.perf_counter() - t
        st.tokens_out += len(job.tokens)
        st.prompt_tokens += len(history)
        st.cached_tokens += job.cached
        if job.finish.startswith(("error", "invalid")):
            st.status = job.finish
            return
        act = mf.parse(job.text)
        if act is None:
            act = {"tool": "?", "thought": job.text[-200:]}
            result = f"error: your reply was cut off after {max_tokens} tokens; keep it shorter"
        else:
            st.status = "tool"
            st.last_tool = act.get("tool", "?")
            t = time.perf_counter()
            try:
                result = _tool(st, act, allow_exec)
            except (OSError, ValueError) as e:  # a folder for a file, a NUL in a path: the model's to see, not the agent's end
                result = f"error: {e}"
            st.t_tools += time.perf_counter() - t
            prev = st.transcript[-1] if st.transcript else None
            if prev is not None and prev["action"] == act and prev["result"] == result:
                result += "\n(note: this repeats your previous step exactly; change your approach)"
        st.transcript.append({"step": st.step, "action": act, "result": result})
        if act.get("tool") == "finish":
            st.answer = act.get("answer") or ""
            st.finished = True
            break
        if mf.native:
            delta = ("<|im_end|>\n<|im_start|>user\n<tool_response>\n" + result
                     + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
            history = history + job.tokens + tok.encode(delta, add_special_tokens=False)
        else:
            cid = mf.fmt.new_id()
            args = {k: v for k, v in act.items() if k not in ("tool", "thought")}
            msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function",
                                                                         "function": {"name": act["tool"], "arguments": args}}]},
                     {"role": "tool", "tool_call_id": cid, "content": result}]
            history = mf.render(msgs)
    st.status = "finished" if st.finished else "out of steps"
    if st.task.check is not None:
        try:
            st.passed = bool(st.task.check(st.dir, st.answer))
        except Exception:
            st.passed = False


# --- dashboard ---------------------------------------------------------------------------

def _render(svc, states, t0, engine):
    from rich.table import Table

    el = time.perf_counter() - t0
    s = svc.stats()
    tab = Table(title=f"locals agents  ·  engine: {engine}  ·  {el:6.1f}s", title_justify="left", expand=False)
    for col in ["agent", "task", "step", "tool", "status", "tokens", "check"]:
        tab.add_column(col, justify="right" if col in ("step", "tokens") else "left")
    for st in states:
        chk = "" if st.passed is None else ("pass" if st.passed else "fail")
        tab.add_row(str(st.i), st.task.name, str(st.step), st.last_tool, st.status, str(st.tokens_out), chk)
    gen = sum(st.tokens_out for st in states)
    cached = s["cached_tokens"] / max(1, s["prompt_tokens"])
    tab.caption = (f"batch {s['active']:2d} rows  ·  {gen} tokens generated  ·  {gen / max(el, 1e-9):6.1f} tok/s  ·  "
                   f"prompt cache {cached:4.0%}")
    return tab


def run(args) -> dict:
    from rich.console import Console
    from rich.live import Live

    from .service import BatchService

    console = Console(stderr=True)
    tasks = [Task("custom", args.task)] * args.agents if args.task else [SUITE[i % len(SUITE)] for i in range(args.agents)]
    out_dir = Path(args.out) / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.engine}"
    from .app import check_local_load

    check_local_load(args.model, "agents")
    console.print(f"loading {args.model} on engine [bold]{args.engine}[/] ...")
    t = time.perf_counter()
    svc = BatchService(args.engine, args.model, slots=args.slots, temp=args.temp, seed=args.seed)
    load_s = time.perf_counter() - t
    # Warm the batch (kernel compilation, first-token paths) so timings are steady-state.
    warm = svc.tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=True,
                                             enable_thinking=False, tokenize=True)
    mf = ModelFormat(svc.tokenizer)
    for j in [svc.submit(warm, regex=mf.regex, max_tokens=24) for _ in range(min(4, args.agents))]:
        j.wait()
    states = [AgentState(i, task, out_dir / f"agent_{i:02d}") for i, task in enumerate(tasks)]
    # Every agent's first prompt starts with the same system prompt and tool
    # definitions: prefill that prefix once, for both engines alike.
    firsts = [first_prompt(svc.tokenizer, st) if mf.native else
              mf.render([{"role": "system", "content": SYSTEM.format(i=st.i)}, {"role": "user", "content": "Task: " + st.task.prompt}])
              for st in states]
    lcp = firsts[0]
    for f in firsts[1:]:
        n = next((k for k, (a, b) in enumerate(zip(lcp, f)) if a != b), min(len(lcp), len(f)))
        lcp = lcp[:n]
    if len(lcp) > 32:
        svc.prime(lcp)
    base = svc.stats()
    threads = [threading.Thread(target=run_agent, args=(svc, st, args.max_steps, args.max_tokens, not args.no_exec, mf), daemon=True)
               for st in states]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    if args.quiet:
        for th in threads:
            th.join()
    else:
        with Live(_render(svc, states, t0, args.engine), console=console, refresh_per_second=4) as live:
            while any(th.is_alive() for th in threads):
                time.sleep(0.25)
                live.update(_render(svc, states, t0, args.engine))
            live.update(_render(svc, states, t0, args.engine))
    wall = time.perf_counter() - t0
    s = svc.stats()
    svc.close()
    gen = sum(st.tokens_out for st in states)
    engine_counters = {k: s[k] - base.get(k, 0) for k in ("steps", "rows", "prefill_s", "prefill_tokens", "admissions", "idle_s") if k in s}
    summary = {
        "engine_counters": engine_counters,
        "engine": args.engine, "model": args.model, "agents": args.agents, "wall_s": wall, "load_s": load_s,
        "tokens_out": gen, "tok_per_s": gen / wall, "steps": sum(st.step for st in states),
        "finished": sum(st.finished for st in states), "passed": sum(bool(st.passed) for st in states),
        "errors": sum(st.status.startswith("error") for st in states),
        "checked": sum(st.passed is not None for st in states),
        "prompt_tokens": s["prompt_tokens"] - base["prompt_tokens"], "cached_tokens": s["cached_tokens"] - base["cached_tokens"],
        "agents_detail": [{"agent": st.i, "task": st.task.name, "steps": st.step, "status": st.status, "tokens": st.tokens_out,
                           "passed": st.passed, "answer": st.answer[:200], "model_s": st.t_model, "tools_s": st.t_tools}
                          for st in states],
        "out_dir": str(out_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "transcripts.json").write_text(json.dumps([{"agent": st.i, "task": st.task.prompt, "steps": st.transcript}
                                                          for st in states], indent=1))
    console.print(f"[bold]{args.engine}[/]: {args.agents} agents, {summary['steps']} steps, {gen} tokens in {wall:.1f}s "
                  f"({gen / wall:.1f} tok/s)  ·  finished {summary['finished']}/{args.agents}"
                  + (f"  ·  checks passed {summary['passed']}/{summary['checked']}" if summary["checked"] else "")
                  + f"  ·  prompt cache {summary['cached_tokens'] / max(1, summary['prompt_tokens']):.0%}"
                  + f"\nrun saved to {out_dir}")
    return summary


def _without(argv: list, opt: str) -> list:
    """``argv`` without ``opt VALUE`` and ``opt=VALUE``."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a == opt:
            skip = True
        elif not a.startswith(opt + "="):
            out.append(a)
    return out


def compare(args, argv=None) -> None:
    """Run both engines in fresh processes (one model in memory at a time) and compare."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    results = {}
    for engine in ("stock", "locals"):
        given = list(sys.argv[1:] if argv is None else argv)  # this command's own arguments (python -m locals.agents ...)
        # (the user's --summary-json came after ours and won: the runs printed nothing, and the comparison crashed)
        given = _without(_without(given, "--engine"), "--summary-json")
        cmd = [sys.executable, "-m", "locals.agents", "--engine", engine, "--summary-json", "-"] + given
        p = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        if p.returncode != 0:
            console.print(f"[red]{engine} run failed[/]")
            sys.exit(p.returncode or 1)
        results[engine] = json.loads(p.stdout.strip().splitlines()[-1])
        if results[engine].get("errors"):
            console.print(f"[red]{engine}: {results[engine]['errors']} agents hit engine errors; not comparing[/]")
            return
    a, b = results["stock"], results["locals"]
    tab = Table(title=f"{args.agents} agents on {args.model}")
    tab.add_column("")
    tab.add_column("stock (mlx-lm + llguidance)", justify="right")
    tab.add_column("locals", justify="right")
    rows = [("wall time", f"{a['wall_s']:.1f} s", f"{b['wall_s']:.1f} s"),
            ("tokens generated", str(a["tokens_out"]), str(b["tokens_out"])),
            ("throughput", f"{a['tok_per_s']:.1f} tok/s", f"{b['tok_per_s']:.1f} tok/s"),
            ("agent steps", str(a["steps"]), str(b["steps"])),
            ("finished", f"{a['finished']}/{a['agents']}", f"{b['finished']}/{b['agents']}")]
    if a["checked"]:
        rows.append(("checks passed", f"{a['passed']}/{a['checked']}", f"{b['passed']}/{b['checked']}"))
    rows.append(("prompt cache hit", f"{a['cached_tokens'] / max(1, a['prompt_tokens']):.0%}",
                 f"{b['cached_tokens'] / max(1, b['prompt_tokens']):.0%}"))
    for r in rows:
        tab.add_row(*r)
    tab.caption = f"locals: {a['wall_s'] / b['wall_s']:.2f}x less wall time, {b['tok_per_s'] / a['tok_per_s']:.2f}x the throughput"
    console.print(tab)
    if args.summary_json == "-":
        print(json.dumps(results))
    elif args.summary_json:
        Path(args.summary_json).write_text(json.dumps(results, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="locals agents", description="Run many tool-using agents in parallel on one local model.")
    ap.add_argument("--engine", choices=["locals", "stock", "both"], default="locals")
    ap.add_argument("--model", default=DEFAULT_MODEL, type=model_id, help=MODEL_HELP)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--task", default=None, help="one task for every agent (default: the built-in suite)")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=700, help="per model call")
    ap.add_argument("--slots", type=int, default=16, help="maximum batch size")
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="agent_runs")
    ap.add_argument("--no-exec", action="store_true", help="disable run_python")
    ap.add_argument("--quiet", action="store_true", help="no live dashboard")
    ap.add_argument("--summary-json", default=None, help="write the summary as JSON ('-' for stdout)")
    args = ap.parse_args(argv)
    if not args.no_exec:
        print("locals agents: the agents' run_python and run_file tools run the code the model writes, as you, in "
              f"{args.out}/ (--no-exec turns them off)", file=sys.stderr, flush=True)
    if args.engine == "both":
        return compare(args, argv)
    summary = run(args)
    if args.summary_json == "-":
        print(json.dumps({k: v for k, v in summary.items() if k != "agents_detail"}))
    elif args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
