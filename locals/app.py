"""locals: your MLX model behind an OpenAI-compatible server, faster.

    locals code [DIR]        OpenCode on a project; the first time it installs OpenCode, downloads
                            the model and starts the server (--run TASK: one task, no interface)
    locals solve "TASK"      several agents race on one task in copies of the project; the first
                            whose result passes the check (--check, or the project's tests) wins
    locals setup openclaw    connect OpenClaw and start the server in the background
    locals setup hermes      the same for Hermes Agent
    locals serve             the OpenAI-compatible server in this terminal (http://127.0.0.1:8321/v1)
    locals start | stop      the server in the background | stop it
    locals status            the server's requests, prompt cache and memory
    locals setup             only install OpenCode (if needed) and write its config: providers
                            locals (our engine) and mlx (stock mlx-lm), and the multi-agent team
    locals generate ...      one structured output from the command line (--prompt, --schema | --regex)
    locals chat              an interactive chat with /json <schema> and /n N
    locals agents ...        run many tool-using agents in parallel on one model (built-in suite)
    locals swarm ...         run many OpenCode agents at once on a list of tasks (one batch)
    locals bench             our engine against stock mlx-lm through the same API, concurrently
    locals tune --model M    the full one-time kernel autotuning for a model (optional, 10-20 minutes)

Models (--model): qwen (Qwen3.5-4B, the default), gemma (Gemma 4 E4B), or any mlx-lm
chat model id or local path; the last one used is remembered.
In OpenCode, /models switches between "Locals" and "MLX default";
Tab switches between the primary agents.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional
from .defaults import DEFAULT_MODEL, MODEL_HELP, model_id

HOME = Path.home()
LOCALS_DIR = HOME / ".locals"
OPENCODE_CONFIG = HOME / ".config" / "opencode" / "opencode.json"
OPENCODE_BIN = HOME / ".opencode" / "bin" / "opencode"
OPENCODE_VERSION = "1.18.32"  # the version this integration was built and tested against
DEFAULT_PORT = 8321

PROVIDERS = {
    "locals": ("Locals", ""),
    "mlx": ("MLX default (mlx-lm)", "plain mlx-lm, for comparison"),
}

# Permissions double as the tool list: a denied tool is not offered to the
# model. A 2B model does best with few tools and a short, procedural prompt.
_NO = {"skill": "deny", "todowrite": "deny", "webfetch": "deny", "doom_loop": "deny", "question": "deny", "external_directory": "deny"}

AGENTS = {
    "orchestrator": {
        "description": "Coordinates a team: splits a task into independent parts, runs coder/researcher/tester subagents in parallel, then integrates and verifies.",
        "mode": "primary",
        "color": "#7c3aed",
        "prompt": (
            "You are the orchestrator of a small software team working in this repository.\n"
            "Rules:\n"
            "1. Small task (one file, one change): do it yourself: write the file with write, run it with bash, then answer.\n"
            "2. Bigger task with independent parts: delegate each part with the task tool "
            "(subagent_type: coder to implement, researcher to find things out, tester to run checks). "
            "Put ALL the task calls for independent parts in the SAME reply so they run at the same time.\n"
            "3. When the subagents report back, run the tests or the program once with bash, fix what is broken, "
            "and finish with a short summary of what changed.\n"
            "Use paths relative to the working directory (shapes/circle.py, not an absolute path). Python is run as python3 (there is no python). "
            "Never run the same command twice in a row: after a tool result, do the NEXT step. "
            "Be concrete: file paths, function names, commands. Do not ask the user questions; make reasonable choices."
        ),
        "permission": {**_NO, "edit": "allow", "bash": "allow", "task": "allow"},
    },
    "coder": {
        "description": "Implements one well-described change: reads the relevant code, edits or creates files, runs a quick check, reports what changed.",
        "mode": "all",
        "color": "#2563eb",
        "prompt": (
            "You are a coder working in the current directory. Implement exactly the task.\n"
            "Procedure:\n"
            "1. If the task is about existing code, read the files involved (read, grep, glob). For a new file, skip this.\n"
            "2. Create or change the files with write (whole file) or edit (one exact replacement).\n"
            "3. Run the program or the tests once with bash to check. If it fails, fix the file and run once more.\n"
            "4. Reply with the files you changed and one line per change. Then stop.\n"
            "Rules: complete, working code; relative paths (e.g. hello.py); Python is run as python3 (there is no python); "
            "if a command fails, change something before running again; never list the directory more than once; "
            "no questions; do not touch unrelated files."
        ),
        "permission": {**_NO, "edit": "allow", "bash": "allow", "task": "deny"},
    },
    "researcher": {
        "description": "Read-only: finds where things are in the codebase and reports facts, file paths and line numbers.",
        "mode": "all",
        "color": "#059669",
        "prompt": (
            "You are a researcher. Answer the question about this repository using read, grep, glob and list only. "
            "Reply with the facts found: file paths with line numbers, function names, short quotes. "
            "Never repeat a search you already did. Do not modify anything."
        ),
        "permission": {**_NO, "edit": "deny", "bash": "deny", "task": "deny"},
    },
    "tester": {
        "description": "Runs tests, programs or commands and reports exactly what passed, what failed and the relevant error output.",
        "mode": "all",
        "color": "#d97706",
        "prompt": (
            "You are a tester. Run the tests or the commands the task names with bash (for Python: python3 -m pytest -q, or python3 file.py). "
            "Run each command once. Reply with: the command you ran, whether it passed, and for failures the error message "
            "and the file and line it points to. Do not edit files."
        ),
        "permission": {**_NO, "edit": "deny", "bash": "allow", "task": "deny"},
    },
    "reviewer": {
        "description": "Read-only code review of a change: correctness, missing cases, style; reports findings with file and line.",
        "mode": "all",
        "color": "#dc2626",
        "prompt": (
            "You are a code reviewer. Read the files or the diff named in the task (read, grep, bash git diff) and report problems: "
            "bugs, unhandled cases, unclear names, missing tests. One finding per line with file:line. Do not edit files."
        ),
        "permission": {**_NO, "edit": "deny", "task": "deny",
                       "bash": {"*": "deny", "git diff*": "allow", "git log*": "allow", "git status*": "allow"}},
    },
}

# A small model that loops (calls of one kind over and over, never done) runs until it is
# stopped: in our OpenCode runs every task where one agent turn took 30 or more steps
# failed (up to 218 steps, 10+ minutes), and no passing task took that many. After this
# many steps OpenCode asks the agent for a text answer; telling it to continue goes on.
AGENT_STEPS = 40
for _agent in AGENTS.values():
    _agent["steps"] = AGENT_STEPS


# --- helpers ------------------------------------------------------------------------------------

def python() -> str:
    return sys.executable


def opencode_bin() -> Optional[str]:
    p = shutil.which("opencode")
    if p:
        return p
    if OPENCODE_BIN.exists():
        return str(OPENCODE_BIN)
    return None


OLLAMA_PORT = 11434


def base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def get_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def server_alive(port: int) -> Optional[dict]:
    try:
        return get_json(base_url(port) + "/health", timeout=2)
    except Exception:
        return None


def _write_atomic(path: Path, text: str) -> None:
    """``text`` into ``path`` whole: a crash or a full disk never leaves half a config file."""
    import tempfile

    path = path.resolve()  # a config kept as a link (dotfile managers): the file it points at is written
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        if path.exists():
            os.chmod(tmp, path.stat().st_mode & 0o777)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _back_up_once(path: Path, tool: str) -> Optional[Path]:
    """A copy of a tool's own config, the first time locals changes it (kept: later runs do not
    overwrite it); where it is, or None if the tool had no config yet."""
    if not path.is_file():
        return None
    keep = LOCALS_DIR / "backups" / f"{tool}-{path.name}.before-locals"
    if not keep.exists():
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, keep)
        print(f"your {tool} config is saved as {keep}; to undo locals's changes: cp '{keep}' '{path}'")
    return keep


def _pid_file(port: int) -> Path:
    return LOCALS_DIR / f"server-{port}.pid"


def _is_locals_server(pid: int) -> bool:
    """Whether the process is a locals server (a pid file can outlive its process, and its number be reused)."""
    p = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True)
    return "locals.server" in p.stdout


def model_short(model: str) -> str:
    from .defaults import short_name

    return short_name(model)


def oc_model(engine: str, short: str) -> str:
    """OpenCode's model reference: provider id, then the model id the server
    routes on (``locals/<short>`` or ``mlx/<short>``, as /v1/models lists them).
    The provider prefix is not sent to the server, so the model id must carry
    the engine itself."""
    return f"{engine}/{engine}/{short}"


# --- setup: OpenCode install and config --------------------------------------------------------

def install_opencode(version: str = OPENCODE_VERSION) -> str:
    """Install OpenCode with its official installer (a single binary in ~/.opencode/bin), if the user agrees."""
    command = f"curl -fsSL https://opencode.ai/install | bash -s -- --no-modify-path --version {version}"
    if not sys.stdin.isatty():
        sys.exit(f"OpenCode is not installed. Install it with:\n  {command}\nthen run the locals command again.")
    answer = input(f"OpenCode is not installed. Install OpenCode {version} with its official installer (opencode.ai) "
                   f"into ~/.opencode/bin? [Y/n] ").strip().lower()
    if answer not in ("", "y", "yes"):
        sys.exit(f"Install it with:\n  {command}\nthen run the locals command again.")
    print(f"installing opencode {version} from opencode.ai ...")
    try:
        script = urllib.request.urlopen("https://opencode.ai/install", timeout=60).read()
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"could not download OpenCode's installer ({e}); check the network, or install it with:\n  {command}")
    p = subprocess.run(["bash", "-s", "--", "--no-modify-path", "--version", version], input=script, capture_output=True)
    if p.returncode != 0 or not OPENCODE_BIN.exists():
        sys.exit("opencode install failed:\n" + p.stdout.decode(errors="replace")[-2000:] + p.stderr.decode(errors="replace")[-2000:])
    return str(OPENCODE_BIN)


def opencode_config(model: str, port: int, default_engine: str = "locals", default_agent: str = "orchestrator",
                    context: int = 32768, output: int = 4096, thinking: bool = True) -> dict:
    short = model_short(model)
    prov = {}
    # the stock engine is for comparisons (locals bench, --engine mlx); in everyday use it only confuses
    for pid, (name, note) in [(k, v) for k, v in PROVIDERS.items() if k == "locals" or k == default_engine]:
        models = {}
        for suffix, label in (("", ""), ("-thinking", " · thinking")) if thinking else (("", ""),):
            models[f"{pid}/{short}{suffix}"] = {"name": f"{short}{label}" + (f"  ({note})" if note else ""), "tool_call": True,
                                                "reasoning": suffix != "",
                                      "temperature": True, "limit": {"context": context, "output": output}}
        prov[pid] = {"npm": "@ai-sdk/openai-compatible", "name": name, "options": {"baseURL": base_url(port) + "/v1", "apiKey": "local"},
                     "models": models}
    return {"$schema": "https://opencode.ai/config.json", "provider": prov, "model": oc_model(default_engine, short),
            "small_model": oc_model(default_engine, short), "default_agent": default_agent, "subagent_depth": 1,
            "agent": {**AGENTS, "build": {"steps": AGENT_STEPS}, "plan": {"steps": AGENT_STEPS}}}


def write_config(cfg: dict, path: Optional[Path] = None) -> None:
    path = path or OPENCODE_CONFIG  # looked up now, not when the module was loaded
    path.parent.mkdir(parents=True, exist_ok=True)
    old = {}
    if path.exists():
        keep = path.with_name(path.name + ".before-locals")
        if not keep.exists():  # the user's own config, once: locals makes itself OpenCode's default model
            shutil.copy(path, keep)
            print(f"your OpenCode config is kept as {keep}; locals adds its providers and becomes the default model")
        try:
            old = json.loads(path.read_text())
        except ValueError:  # comments or trailing commas (OpenCode reads JSONC): rewritten, they would be lost
            sys.exit(f"{path} is not plain JSON (comments or trailing commas?), so locals does not rewrite it. Remove "
                     f"them, or move the file away and run this again (locals then writes a new one).")
        if not isinstance(old, dict):
            sys.exit(f"{path} does not hold a JSON object; move it away and run this again")
    merged = dict(old)
    for k, v in cfg.items():
        if k in ("provider", "agent") and isinstance(old.get(k), dict):
            merged[k] = {**old[k], **v}
            if k == "agent":  # the user's own settings of OpenCode's agents (build, plan) stay
                for name in ("build", "plan"):
                    if isinstance(old[k].get(name), dict):
                        merged[k][name] = {**old[k][name], **v.get(name, {})}
        else:
            merged[k] = v
    _write_atomic(path, json.dumps(merged, indent=2) + "\n")


# --- setup: other agent tools (OpenClaw, Hermes Agent) -----------------------------------------
#
# Both speak OpenAI chat completions to a custom base URL. The config is written
# through the tool's own CLI when it is installed (so its format and validation
# stay the tool's business); otherwise the snippet is saved and printed.

AGENT_CONTEXT = 65536  # Hermes refuses models under 64,000 tokens; the server's default --max-context


def openclaw_patch(model: str, port: int, context: int = AGENT_CONTEXT, output: int = 4096) -> dict:
    """OpenClaw config fragment: a ``locals`` provider and the default agent model.
    The model id carries the engine (``locals/<short>``) because only the id is
    sent to the server; OpenClaw's reference is then ``locals/locals/<short>``."""
    mid = f"locals/{model_short(model)}"
    return {"models": {"mode": "merge", "providers": {"locals": {
                "baseUrl": base_url(port) + "/v1", "apiKey": "locals-local", "api": "openai-completions", "timeoutSeconds": 900,
                "models": [{"id": mid, "name": f"{model_short(model)} (locals, local)", "reasoning": False, "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": context, "maxTokens": output}]}}},
            "agents": {"defaults": {"model": {"primary": f"locals/{mid}"}}}}


def hermes_settings(model: str, port: int, context: int = AGENT_CONTEXT) -> dict:
    """Hermes Agent ``config.yaml`` keys (dotted) for a custom OpenAI-compatible endpoint."""
    return {"model.provider": "custom", "model.base_url": base_url(port) + "/v1", "model.default": f"locals/{model_short(model)}",
            "model.api_key": "locals-local", "model.context_length": str(context)}


def _tool_bin(name: str, env: str) -> Optional[str]:
    return os.environ.get(env) or shutil.which(name)


def agent_context(model: str) -> int:
    """The context declared to agent tools: 64k, or the model's own if smaller."""
    from .server import native_context

    n = native_context(model)
    return min(AGENT_CONTEXT, n) if n else AGENT_CONTEXT


def setup_openclaw(args) -> bool:
    """Write the locals provider into OpenClaw's config; False if OpenClaw is not installed."""
    patch = openclaw_patch(args.model, args.port, args.context or agent_context(args.model), args.output or 4096)
    LOCALS_DIR.mkdir(exist_ok=True)
    path = LOCALS_DIR / "openclaw-locals.json5"
    path.write_text(json.dumps(patch, indent=2) + "\n")
    oc = _tool_bin("openclaw", "OPENCLAW_BIN")
    if oc is None:
        print(f"openclaw is not installed (https://docs.openclaw.ai/install). Saved the config to {path}; after installing "
              f"it, run  locals setup openclaw  again (or: openclaw config patch --file {path} "
              f"--replace-path models.providers.locals.models)")
        return False
    state = Path(os.environ.get("OPENCLAW_STATE_DIR") or HOME / ".openclaw")
    _back_up_once(state / "openclaw.json", "openclaw")
    print(f"writing the locals provider into OpenClaw's config ({oc} config patch; its first start can take a few minutes) ...",
          flush=True)
    # locals's provider lists the model the server serves: a setup for another model replaces the
    # list (OpenClaw refuses a patch that would drop entries, "locals/qwen3.5-4b" for Gemma)
    p = subprocess.run([oc, "config", "patch", "--file", str(path), "--replace-path", "models.providers.locals.models"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"openclaw config patch failed:\n{p.stdout[-1500:]}{p.stderr[-1500:]}\nthe fragment is in {path}")
    print(f"OpenClaw now uses {patch['agents']['defaults']['model']['primary']} at {base_url(args.port)}/v1 "
          f"(context {patch['models']['providers']['locals']['models'][0]['contextWindow']}).")
    return True


def setup_hermes(args) -> bool:
    """Point Hermes Agent at the server; False if Hermes is not installed."""
    ctx = args.context or agent_context(args.model)
    if ctx < 64000:
        print(f"warning: {args.model} has a {ctx}-token context; Hermes Agent requires at least 64,000 and will refuse it. "
              "Pick a model with a longer context (Qwen3.5, Llama 3.1+, Gemma 3+).")
    kv = hermes_settings(args.model, args.port, ctx)
    hm = _tool_bin("hermes", "HERMES_BIN")
    if hm is None:
        print("hermes is not installed (https://hermes-agent.nousresearch.com). After installing it, run  "
              "locals setup hermes  again (or these):")
        for k, v in kv.items():
            print(f"  hermes config set {k} {v}")
        return False
    keep = _back_up_once(Path(os.environ.get("HERMES_HOME") or HOME / ".hermes") / "config.yaml", "hermes")
    for k, v in kv.items():
        p = subprocess.run([hm, "config", "set", k, v], capture_output=True, text=True)
        if p.returncode != 0:
            sys.exit(f"hermes config set {k} failed:\n{p.stdout[-1500:]}{p.stderr[-1500:]}"
                     + (f"\nyour config before locals: {keep}" if keep else ""))
    print(f"Hermes Agent now uses {kv['model.default']} at {kv['model.base_url']} (context {kv['model.context_length']}).")
    return True


def model_thinks(model: str) -> bool:
    """Whether the model's chat template has a thinking mode the server uses (Qwen3, Qwen3.5, Gemma 4)."""
    try:
        from mlx_lm.utils import load_tokenizer

        from . import formats

        return formats.offers_thinking(load_tokenizer(model))
    except Exception:
        return True


def default_agent(model: str) -> str:
    """OpenCode's default agent for this model. Qwen3.5 delegates sensibly to subagents
    (orchestrator: 11 of 13 real tasks); other models tested (Llama 3.x, Qwen3, Gemma)
    call the subagent tool over and over with junk arguments, and do the work
    themselves far better as the coder agent."""
    try:
        from mlx_lm.utils import load_tokenizer

        from . import formats

        return "orchestrator" if formats.detect(load_tokenizer(model)).name == "qwen_xml" else "coder"
    except Exception:
        return "orchestrator"


def setup_opencode(args) -> str:
    """Install OpenCode if needed and write its config for the model; returns the binary."""
    oc = opencode_bin()
    if oc is None or getattr(args, "reinstall", False):
        oc = install_opencode(getattr(args, "opencode_version", None) or OPENCODE_VERSION)
    ver = subprocess.run([oc, "--version"], capture_output=True, text=True).stdout.strip()
    print(f"opencode {ver} at {oc}")
    agent = getattr(args, "agent", None) or default_agent(args.model)
    context = getattr(args, "context", None) or min(32768, agent_context(args.model))
    engine = getattr(args, "engine", None) or "locals"
    write_config(opencode_config(args.model, args.port, engine, agent, context,
                                 getattr(args, "output", None) or 4096, thinking=model_thinks(args.model)))
    print(f"wrote {OPENCODE_CONFIG}: provider{'s locals + mlx' if engine == 'mlx' else ' locals'} at {base_url(args.port)}/v1, "
          f"agents {', '.join(AGENTS)} (default: {agent}); locals is now OpenCode's default model")
    LOCALS_DIR.mkdir(exist_ok=True)
    remember(model=args.model, port=args.port, opencode=oc)
    return oc


def cmd_setup(args) -> None:
    target = getattr(args, "target", "opencode")
    if target in ("openclaw", "hermes"):
        if not (setup_openclaw if target == "openclaw" else setup_hermes)(args):
            return
        remember(model=args.model, port=args.port)  # a later `locals start` serves the model the agent asks for
        if getattr(args, "no_start", False):
            print(f"Next:  locals start  (or locals serve in another terminal), then:  {target}")
            return
        fetch_model(args.model)  # the agent works right away: the server runs in the background
        ensure_server(args)
        print(f"the server keeps running in the background (locals status, locals stop). Next:  {target}")
        return
    setup_opencode(args)
    me = Path(sys.argv[0]).resolve() if sys.argv[0].endswith("locals") else None
    launcher = HOME / ".local" / "bin" / "locals"
    if me is not None and shutil.which("locals") is None and not (launcher.exists() or launcher.is_symlink()):
        # a development checkout: put locals on the PATH (an install by uv or pip already did)
        launcher.parent.mkdir(parents=True, exist_ok=True)
        launcher.symlink_to(me)
        print(f"launcher: {launcher} -> {me}")
    if shutil.which("locals") is None:
        print(f"add {launcher.parent} to your PATH, e.g.  export PATH=$HOME/.local/bin:$PATH")
    print("done. Next:  locals code <project dir>")


def fetch_model(model: str) -> None:
    """Download the model (first run only), with progress, before the server needs it."""
    from .defaults import gguf_hint, too_big

    why = gguf_hint(model) or too_big(model, remote=True)  # before a download of gigabytes that cannot run
    if why:
        sys.exit(f"locals: {why}")
    if Path(model).expanduser().exists():
        return
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "model*.safetensors", "*.py", "tokenizer.model", "*.tiktoken", "tiktoken.model", "*.txt", "*.jsonl", "*.jinja"]
    try:
        snapshot_download(model, allow_patterns=patterns, local_files_only=True)
        return
    except Exception:  # noqa: BLE001  # not downloaded yet
        pass
    print(f"downloading {model} (first run only) ...", flush=True)
    try:
        snapshot_download(model, allow_patterns=patterns)
    except Exception as e:  # noqa: BLE001
        from .server import _load_error

        sys.exit(f"locals: {_load_error(model, e) or f'could not download {model}: {e}'}")


# --- the server ---------------------------------------------------------------------------------

def server_cmd(args, extra: List[str] = ()) -> List[str]:
    cmd = [python(), "-m", "locals.server", "--port", str(args.port), "--model", args.model]
    for k in ("engines", "preload", "slots", "temp", "top_p", "top_k", "max_tokens", "cache_gb", "max_calls", "max_context",
              "ollama_port"):
        v = getattr(args, k, None)
        if v is not None:
            cmd += ["--" + k.replace("_", "-"), str(v)]
    for flag in ("thinking", "walk", "typed_tools"):
        if getattr(args, flag, False):
            cmd.append("--" + flag.replace("_", "-"))
    return cmd + list(extra)


def remember(**kw) -> None:
    """Save choices (the model, the port) as the defaults of later locals commands."""
    try:
        LOCALS_DIR.mkdir(exist_ok=True)
        path = LOCALS_DIR / "config.json"
        cfg = json.loads(path.read_text()) if path.exists() else {}
        cfg.update({k: v for k, v in kw.items() if v is not None})
        path.write_text(json.dumps(cfg, indent=1))
    except (OSError, ValueError):
        pass


def cmd_serve(args) -> None:
    remember(model=args.model, port=args.port)  # `locals setup` / `locals code` then use the same model
    os.execv(python(), server_cmd(args))


def ensure_server(args, wait_s: float = 3600) -> None:
    """Start the server in the background unless one answers on the port, and wait for it (see
    _ensure_server). One locals command at a time starts the server on a port: another one (a second
    terminal, the menu bar) waits for it instead of starting a second copy of the model."""
    import fcntl

    LOCALS_DIR.mkdir(exist_ok=True)
    with open(LOCALS_DIR / f"server-{args.port}.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"another locals command is starting the server on port {args.port}; waiting for it ...", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        _ensure_server(args, wait_s)


_SERVER_OPTS = ("engines", "slots", "temp", "top_p", "top_k", "max_tokens", "cache_gb", "max_calls", "max_context")


def _ensure_server(args, wait_s: float) -> None:
    """The first load of a model tunes its kernels (minutes), so the wait has
    no short limit while the process is alive; what the server says on the way
    (loading, tuning) is shown, and its last lines if it stops."""
    if getattr(args, "engine", None) == "mlx" and not getattr(args, "engines", None):
        args.engines = "locals,mlx"  # a comparison on the stock engine needs it loaded beside ours
    if server_alive(args.port):
        try:
            st = get_json(base_url(args.port) + "/stats", timeout=5)
        except Exception:  # noqa: BLE001
            st = {}
        running = st.get("model")
        from .defaults import build_id

        health = server_alive(args.port) or {}
        stale = bool(health.get("build")) and health["build"] != build_id()  # started before locals was upgraded
        same = not running or running == args.model or model_short(running) == model_short(args.model)
        if same and not stale:
            given = [k for k in _SERVER_OPTS if k != "engines" and getattr(args, k, None) is not None]
            if given:
                print(f"note: the server on port {args.port} is running already; --{', --'.join(k.replace('_', '-') for k in given)} "
                      f"apply when it starts (locals stop, then this command again)", flush=True)
            if "mlx" in (getattr(args, "engines", None) or "").split(",") and "mlx" not in (health.get("engines") or {}):
                sys.exit(f"the server on port {args.port} runs without the stock engine; stop it (locals stop) and run this "
                         f"command again to start it with both")
            return
        pid = _background_pid(args.port)
        if pid is None or st.get("active"):  # started by hand, or answering someone: not ours to restart
            if same:  # older code, but not ours to restart: use it
                print(f"note: the locals server on port {args.port} runs older locals code; restart it to use this version")
                return
            sys.exit(f"a locals server with {running} is {'busy' if pid else 'already running'} on port {args.port}; "
                     f"stop it with  locals stop  (or use another --port) to use {args.model}")
        print(f"restarting the background server: it runs older locals code ..." if same else
              f"switching the background server from {running} to {args.model} ...", flush=True)
        os.kill(pid, signal.SIGTERM)
        for _ in range(120):
            if _listener_pid(args.port) is None:
                break
            time.sleep(0.5)
        else:
            sys.exit(f"the server on port {args.port} did not stop; stop it with  locals stop")
    holder = _listener_pid(args.port)
    if holder is not None and not _is_locals_server(holder):  # found before loading the model, not after
        sys.exit(f"port {args.port} is in use by another program (pid {holder}); choose another port with --port")
    LOCALS_DIR.mkdir(exist_ok=True)
    log_path = LOCALS_DIR / "server.log"
    if log_path.exists() and log_path.stat().st_size > 50 << 20:  # the log of earlier runs, kept once
        os.replace(log_path, log_path.with_suffix(".log.1"))
    start = log_path.stat().st_size if log_path.exists() else 0
    log = open(log_path, "ab")
    p = subprocess.Popen(server_cmd(args), stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                         cwd=str(Path(__file__).resolve().parent.parent))
    _pid_file(args.port).write_text(str(p.pid))
    print(f"starting the locals server for {args.model} on port {args.port} (log {log_path}) ...", flush=True)
    t0, seen = time.time(), start
    while time.time() - t0 < wait_s:
        try:
            with open(log_path, "rb") as f:
                f.seek(seen)
                new = f.read()
            seen += len(new)
            for line in new.decode(errors="replace").splitlines():
                if line.startswith(("locals:", "loading ", "engine ")) or "tuning" in line:
                    print("  " + line.strip()[:200], flush=True)
        except OSError:
            pass
        if p.poll() is not None:
            # ``start`` counts bytes (the log has multi-byte progress bars), so slice before decoding
            tail = log_path.read_bytes()[start:].decode(errors="replace").strip().splitlines()[-15:] if log_path.exists() else []
            sys.exit("the server stopped:\n  " + "\n  ".join(tail))
        if server_alive(args.port):
            print(f"server ready in {time.time() - t0:.0f}s")
            return
        time.sleep(1)
    sys.exit(f"the server did not come up in {wait_s:.0f}s; see {log_path}")


def _saved_pid(port: int) -> Optional[int]:
    for f in (_pid_file(port), LOCALS_DIR / "server.pid"):  # server.pid: locals 0.2.0 before per-port files
        try:
            pid = int(f.read_text() or 0)
        except (OSError, ValueError):
            continue
        if pid:
            return pid
    return None


def _background_pid(port: int) -> Optional[int]:
    """The server locals started in the background on this port, if it is the one listening."""
    pid = _saved_pid(port)
    return pid if pid and _listener_pid(port) == pid else None


def _listener_pid(port: int) -> Optional[int]:
    """The process listening on the port (lsof), if any."""
    p = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True)
    pids = [int(x) for x in p.stdout.split() if x.strip().isdigit()]
    return pids[0] if pids else None


def cmd_start(args) -> None:
    """The server in the background, for agents other than OpenCode (or several at once)."""
    remember(model=args.model, port=args.port)
    fetch_model(args.model)
    ensure_server(args)
    print(f"locals serves {args.model} at {base_url(args.port)}/v1 in the background (locals status, locals stop)")


def cmd_stop(args) -> None:
    """Stop the locals server on this port (and only that one): the process listening there if it is
    a locals server, or one locals started that is still loading. Never another program: a pid file
    can outlive its server, and its number then belong to anything."""
    listener = _listener_pid(args.port)
    saved = _saved_pid(args.port)
    target = None
    if listener is not None:
        if not (_answers_as_locals(args.port) or _is_locals_server(listener)):
            sys.exit(f"port {args.port} is held by another program (pid {listener}), not a locals server; nothing stopped")
        target = listener
    elif saved is not None and _is_locals_server(saved):  # started, not listening yet (loading the model)
        target = saved
    for f in (_pid_file(args.port), LOCALS_DIR / "server.pid"):
        if saved is not None and f.exists() and f.read_text().strip() == str(saved):
            f.unlink(missing_ok=True)
    if target is None:
        print(f"no locals server on port {args.port}")
        return
    try:
        os.kill(target, signal.SIGTERM)
    except ProcessLookupError:
        print(f"no locals server on port {args.port}")
        return
    for _ in range(40):  # the port free again before the next command uses it
        if _listener_pid(args.port) is None:
            break
        time.sleep(0.25)
    print(f"stopped the locals server on port {args.port} (pid {target})")


def cmd_status(args) -> None:
    st = server_alive(args.port)
    if not st:
        print(f"no server on port {args.port}")
        return
    s = get_json(base_url(args.port) + "/stats")
    print(f"model {s['model']}  uptime {s['uptime_s']:.0f}s  requests {s['requests']}  active rows {len(s['active'])}")
    print(f"{'engine':8s} {'loaded':>6s} {'reqs':>5s} {'prompt':>8s} {'cached':>7s} {'output':>7s} {'forced':>7s} {'calls':>5s} {'tok/s':>6s}  (tok/s = output tokens over request wall time, prefill and queueing included)")
    for e, d in s["engines"].items():
        cached = f"{d['cached_tokens'] / d['prompt_tokens']:.0%}" if d.get("prompt_tokens") else "-"
        tps = f"{d['completion_tokens'] / d['decode_s']:.0f}" if d.get("decode_s") else "-"
        print(f"{e:8s} {str(d['loaded']):>6s} {d['requests']:5d} {d['prompt_tokens']:8d} {cached:>7s} {d['completion_tokens']:7d} "
              f"{d['forced_tokens']:7d} {d['tool_calls']:5d} {tps:>6s}")
    m = s.get("memory", {})
    for e, d in s["engines"].items():
        if d.get("prompt_cache_mb") is not None:
            print(f"{e}: prompt cache {d['prompt_cache_mb'] / 1e3:.1f} GB in {d.get('prompt_cache_entries', 0)} entries"
                  f" (trimmed for memory {d.get('cache_trims', 0) or 0}x, requests held back {d.get('deferred', 0) or 0}x)")
    if m:
        print(f"GPU memory: {m['gpu_active_mb'] / 1e3:.1f} GB in use, peak {m['gpu_peak_mb'] / 1e3:.1f} GB")


# --- opencode ---------------------------------------------------------------------------------------

def _opencode_ready(model: str, port: int = DEFAULT_PORT) -> bool:
    """OpenCode is installed and its config serves this model, at this port."""
    if opencode_bin() is None or not OPENCODE_CONFIG.exists():
        return False
    try:
        cfg = json.loads(OPENCODE_CONFIG.read_text())
    except ValueError:
        return False
    prov = (cfg.get("provider") or {}).get("locals") or {}
    return (str(cfg.get("model", "")).endswith("/" + model_short(model)) and isinstance(prov, dict)
            and (prov.get("options") or {}).get("baseURL") == base_url(port) + "/v1")


def _has_provider(name: str) -> bool:
    try:
        return name in (json.loads(OPENCODE_CONFIG.read_text()).get("provider") or {})
    except (OSError, ValueError, AttributeError):
        return False


def _answers_as_locals(port: int) -> bool:
    try:
        return get_json(base_url(port) + "/health", timeout=2).get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


LAUNCH_TOOLS = {
    "opencode": "OpenCode (the same as locals code)",
    "openhands": "OpenHands",
    "cline": "Cline, in the terminal",
    "kilo": "Kilo Code, in the terminal",
    "openclaw": "OpenClaw",
    "hermes": "Hermes Agent",
    "aider": "Aider",
    "ollama": "apps with an Ollama setting: locals answers as Ollama",
    "editor": "VS Code (Roo Code, Cline, Kilo Code, Continue), Zed, JetBrains: the settings to enter",
}
TOOL_SITES = {"openclaw": "https://openclaw.ai", "hermes": "https://hermes-agent.nousresearch.com"}


INSTALLERS = {  # (package manager, its arguments) for tools a launch can install itself
    "openhands": ("uv", ["tool", "install", "--python", "3.12", "openhands"]),
    "aider": ("uv", ["tool", "install", "aider-chat"]),
    "cline": ("npm", ["install", "-g", "cline"]),
    "kilo": ("npm", ["install", "-g", "@kilocode/cli"]),
}


def _uv_bin() -> Optional[str]:
    return shutil.which("uv") or next((str(p) for p in (Path.home() / ".local/bin/uv", Path("/opt/homebrew/bin/uv")) if p.exists()), None)


def _installed(name: str, env: str) -> Optional[str]:
    """The tool's command: on the PATH, or where uv put it (uv's tool folder may not be on the PATH yet)."""
    found = _tool_bin(name, env)
    if found:
        return found
    uv = _uv_bin()
    if uv:
        d = subprocess.run([uv, "tool", "dir", "--bin"], capture_output=True, text=True).stdout.strip()
        if d and (Path(d) / name).exists():
            return str(Path(d) / name)
    return None


def _offer_install(tool: str, name: str, env: str) -> str:
    """The tool's command, installed first if the user agrees; exits with what to do otherwise."""
    manager, argv = INSTALLERS[tool]
    exe = _uv_bin() if manager == "uv" else shutil.which("npm")
    if exe is None:
        why = ("uv is not installed: curl -LsSf https://astral.sh/uv/install.sh | sh" if manager == "uv" else
               "it needs Node.js (npm): brew install node, or the installer from https://nodejs.org")
        sys.exit(f"{tool} is not installed, and {why}; then run locals launch {tool} again.")
    command = " ".join([manager, *argv])
    if not sys.stdin.isatty():
        sys.exit(f"{tool} is not installed. Install it with:\n  {command}\nthen run locals launch {tool} again.")
    answer = input(f"{tool} is not installed. Install it now with `{command}`? [Y/n] ").strip().lower()
    if answer not in ("", "y", "yes"):
        sys.exit(f"Install it with:\n  {command}\nthen run locals launch {tool} again.")
    if subprocess.run([exe, *argv]).returncode != 0:
        sys.exit(f"installing {tool} failed; try: {command}")
    found = _installed(name, env)
    if not found:
        sys.exit(f"{tool} was installed but its command {name!r} is not on the PATH; open a new terminal and run locals launch {tool}")
    return found


def cmd_launch(args) -> None:
    """Start the server (if it is not running) and open a tool on it, pointed at the local model:
    for this launch only, through its environment variables or a config file of locals's (Aider,
    OpenHands, Cline, Kilo Code); OpenCode, OpenClaw and Hermes get locals in their own config (a
    copy of it is kept the first time, see _back_up_once)."""
    tool, rest = args.tool, list(args.rest or [])
    if tool == "list":
        print("locals launch <tool> starts the server (if needed) and opens the tool on the local model:\n")
        for name, what in LAUNCH_TOOLS.items():
            if name in ("ollama", "editor", "opencode") or _installed(name, name.upper() + "_BIN"):
                note = ""
            else:
                note = "   (not installed: the launch offers to install it)" if name in INSTALLERS else f"   (not installed: {TOOL_SITES[name]})"
            print(f"  locals launch {name:10s} {what}{note}")
        return
    if tool == "opencode":
        args.dir = rest[0] if rest and not rest[0].startswith("-") else "."
        args.engine = args.agent = args.run = None
        args.logs = args.cont = False
        return cmd_code(args)
    if tool == "ollama":
        args.ollama_port = args.ollama_port or OLLAMA_PORT
        holder = _listener_pid(args.ollama_port)
        if holder is not None and not _answers_as_locals(args.ollama_port):
            sys.exit(f"port {args.ollama_port} is taken (Ollama itself?): quit it first, or point the app at "
                     f"{base_url(args.port)} (locals answers Ollama's API there too)")
        if holder is None and server_alive(args.port):  # running without the Ollama port: start it again with it
            try:
                busy = len(get_json(base_url(args.port) + "/stats", timeout=5).get("active") or [])
            except Exception:  # noqa: BLE001
                busy = 0
            if busy:
                sys.exit(f"the locals server on port {args.port} is answering {busy} request(s); run this again when it is idle "
                         f"(it restarts to answer on Ollama's port too)")
            print("restarting the locals server to answer on Ollama's port as well ...", flush=True)
            cmd_stop(args)
    exe = None
    if tool in INSTALLERS:  # the tool first (installed if the user agrees), then the server
        env_name = tool.upper() + "_BIN"
        exe = _installed(tool, env_name) or _offer_install(tool, tool, env_name)
    elif tool in TOOL_SITES:
        exe = _tool_bin(tool, tool.upper() + "_BIN")
        if not exe:
            sys.exit(f"{tool} is not installed; see {TOOL_SITES[tool]}")
    import contextlib

    with contextlib.redirect_stdout(sys.stderr):  # the tool's own output (aider --message, ...) stays clean
        fetch_model(args.model)
        ensure_server(args)
    remember(model=args.model, port=args.port)
    base, short = base_url(args.port), model_short(args.model)
    if tool == "aider":
        env = dict(os.environ, OPENAI_API_BASE=f"{base}/v1", OPENAI_API_KEY="locals")
        os.execve(exe, [exe, "--model", f"openai/locals/{short}", "--no-show-model-warnings", *rest], env)
    if tool == "cline":
        cfg = LOCALS_DIR / "cline"  # a config folder of locals's: the user's own ~/.cline is left alone
        subprocess.run([exe, "auth", "--config", str(cfg), "-p", "openai", "-k", "locals", "-m", f"locals/{short}",
                        "-b", f"{base}/v1"], capture_output=True)
        os.execv(exe, [exe, "--config", str(cfg), *rest])
    if tool == "kilo":
        cfg = LOCALS_DIR / "kilo.json"  # read through KILO_CONFIG: the user's own config is left alone
        LOCALS_DIR.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({
            "$schema": "https://app.kilo.ai/config.json",
            "provider": {"locals": {"npm": "@ai-sdk/openai-compatible", "name": "locals (local)",
                                   "options": {"baseURL": f"{base}/v1", "apiKey": "locals"},
                                   "models": {f"locals/{short}": {"name": f"{short} (locals)", "tool_call": True,
                                                                 "limit": {"context": agent_context(args.model), "output": 4096}}}}},
            "model": f"locals/locals/{short}"}, indent=2) + "\n")
        os.execve(exe, [exe, *rest], dict(os.environ, KILO_CONFIG=str(cfg)))
    if tool == "openhands":
        env = dict(os.environ, LLM_MODEL=f"openai/locals/{short}", LLM_BASE_URL=f"{base}/v1", LLM_API_KEY="locals",
                   OPENHANDS_SUPPRESS_BANNER="1")
        os.execve(exe, [exe, "--override-with-envs", *rest], env)
    if tool in ("openclaw", "hermes"):
        args.context = None
        args.output = None
        if not (setup_openclaw if tool == "openclaw" else setup_hermes)(args):
            sys.exit(f"{tool} is not installed; see {TOOL_SITES[tool]}")
        exe = _tool_bin(tool, "OPENCLAW_BIN" if tool == "openclaw" else "HERMES_BIN")
        os.execv(exe, [exe, *rest])
    if tool == "ollama":
        print(f"locals answers as Ollama at http://127.0.0.1:{args.ollama_port} (and at {base}/api), model {short}:latest.\n"
              "In the app, choose Ollama and keep its default address; the model is listed as "
              f"{short}:latest.\nlocals stop stops it.")
        return
    if tool == "editor":
        try:  # the running server's limit (--max-context)
            ctx = int(get_json(base + "/v1/models", timeout=5)["data"][0]["context_length"])
        except Exception:  # noqa: BLE001
            ctx = 65536
        print(f"""Point the editor's OpenAI-compatible (or Ollama) provider at Locals:

  base URL        {base}/v1          (Ollama-style settings: {base})
  model           locals/{short}
  API key         any text, e.g. locals
  context window  {ctx} tokens  (Cline, Roo and Kilo ask for it: a larger one gets requests the server refuses)

  Continue  (~/.continue/config.yaml)   models: - provider: openai, model: locals/{short}, apiBase: {base}/v1
  Cline / Roo / Kilo Code               API provider "OpenAI Compatible", base URL and model as above
  Zed       (settings.json)             "language_models": {{"openai_compatible": {{"locals": {{"api_url": "{base}/v1",
                                          "available_models": [{{"name": "locals/{short}", "max_tokens": {ctx}}}]}}}}}}
  JetBrains AI Assistant                Third-party AI providers -> OpenAI-compatible, URL {base}/v1

The server stays up in the background (locals stop stops it).""")
        return
    sys.exit(f"unknown tool {tool!r}; locals launch list shows the tools")


def cmd_code(args) -> None:
    """One command for a project: OpenCode set up for the model (first run), the
    model downloaded, the server started, then OpenCode on the project, or one
    task run to the end with --run."""
    d = Path(args.dir).expanduser().resolve()
    if not d.is_dir():
        sys.exit(f"no such directory: {d}")
    if not _opencode_ready(args.model, args.port) or (args.engine == "mlx" and not _has_provider("mlx")):
        print(f"setting up OpenCode for {args.model} ...", flush=True)
        setup_opencode(args)
    oc = opencode_bin()
    fetch_model(args.model)
    ensure_server(args)
    remember(model=args.model, port=args.port)
    cfg = json.loads(OPENCODE_CONFIG.read_text())
    short = model_short(args.model)
    changed = False
    if args.engine and cfg.get("model") != oc_model(args.engine, short):
        cfg["model"] = cfg["small_model"] = oc_model(args.engine, short)
        changed = True
    if args.agent and cfg.get("default_agent") != args.agent:
        cfg["default_agent"] = args.agent
        changed = True
    if changed:  # OpenCode's defaults, kept for later launches too
        _write_atomic(OPENCODE_CONFIG, json.dumps(cfg, indent=2) + "\n")
    env = dict(os.environ, OPENCODE_DISABLE_AUTOUPDATE="1", PWD=str(d))  # OpenCode reads its project directory from $PWD
    os.chdir(d)
    shown = f"{short} on the {cfg['model'].split('/')[0]} engine, agent {cfg.get('default_agent')}"
    if args.run:
        print(f"opencode in {d}: running the task ({shown}) ...", flush=True)
        p = subprocess.run([oc, "run", "--format", "json", *(["-c"] if args.cont else []), args.run], env=env,
                           capture_output=True, text=True)
        answer = _opencode_answer(p.stdout)
        print(answer or (p.stdout[-2000:] + p.stderr[-2000:]))
        sys.exit(p.returncode)
    print(f"opencode in {d}  ({shown}; /models switches the engine, Tab the agent)")
    os.execve(oc, [oc, str(d)] + (["--print-logs"] if args.logs else []), env)


def _opencode_answer(stdout: str) -> str:
    """The agent's text from `opencode run --format json` events (the last answer)."""
    texts = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        part = ev.get("part") or {}
        if ev.get("type") == "text" or part.get("type") == "text":
            t = part.get("text") or ev.get("text") or ""
            if t.strip():
                texts.append(t)
        if ev.get("type") == "error":
            texts.append("error: " + json.dumps(ev.get("error"))[:500])
    return texts[-1] if texts else ""


# --- swarm: many agents at once ------------------------------------------------------------------

def _free_port() -> int:
    import socket

    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        return sk.getsockname()[1]


class OpenCodeServer:
    """One ``opencode serve`` process driven through its HTTP API. Every
    session is a separate agent; they run concurrently inside the one process
    (separate ``opencode run`` processes contend for OpenCode's SQLite
    database)."""

    def __init__(self, oc: str, cwd: Path, log_path: Path):
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        env = dict(os.environ, OPENCODE_DISABLE_AUTOUPDATE="1", PWD=str(cwd))  # OpenCode reads its directory from $PWD
        self.proc = subprocess.Popen([oc, "serve", "--port", str(self.port)], stdout=open(log_path, "wb"), stderr=subprocess.STDOUT,
                                     env=env, cwd=str(cwd))
        t0 = time.time()
        while time.time() - t0 < 90:
            if self.proc.poll() is not None:
                sys.exit(f"opencode serve exited with {self.proc.returncode}; see {log_path}")
            try:  # a short timeout: a request sent while OpenCode starts can go unanswered, and is sent again
                if self.call("GET", "/global/health", timeout=5).get("healthy"):
                    return
            except Exception:
                time.sleep(0.5)
        self.close()
        sys.exit(f"opencode serve did not come up in 90 s; see {log_path}")

    def call(self, method: str, path: str, body=None, directory: Optional[str] = None, timeout: float = 3600):
        import urllib.parse

        url = self.base + path + (("?" + urllib.parse.urlencode({"directory": directory})) if directory else "")
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw else None

    def start_approver(self) -> None:
        """Headless mode: answer permission requests with "once" and reject
        questions, from the server's event stream (rules set to deny in the
        agent config never reach here)."""

        def listen():
            while self.proc.poll() is None:
                try:
                    req = urllib.request.Request(self.base + "/global/event", headers={"Accept": "text/event-stream"})
                    with urllib.request.urlopen(req, timeout=None) as r:
                        for line in r:
                            line = line.decode(errors="replace").strip()
                            if not line.startswith("data:"):
                                continue
                            try:
                                ev = json.loads(line[5:])
                            except ValueError:
                                continue
                            ev = ev.get("payload", ev)
                            t, pr = ev.get("type"), ev.get("properties") or {}
                            sid = pr.get("sessionID")
                            # outside the session's own folder, or a loop OpenCode itself flagged: refused, not approved
                            refuse = any(v in ("external_directory", "doom_loop") for v in pr.values() if isinstance(v, str))
                            try:
                                if t == "permission.asked":
                                    self.call("POST", f"/session/{sid}/permissions/{pr['id']}",
                                              {"response": "reject" if refuse else "once"}, timeout=10)
                                elif t == "permission.v2.asked":
                                    self.call("POST", f"/api/session/{sid}/permission/{pr['id']}/reply",
                                              {"reply": "reject" if refuse else "once"}, timeout=10)
                                elif t in ("question.asked", "question.v2.asked"):
                                    self.call("POST", f"/api/session/{sid}/question/{pr['id']}/reject", {}, timeout=10)
                            except Exception:
                                pass
                except Exception:
                    time.sleep(1)

        threading.Thread(target=listen, daemon=True).start()

    def run_task(self, task: str, directory: str, agent: str, provider: str, model: str, title: str, timeout: float = 900) -> dict:
        """Create a session in ``directory``, send the task, wait for the agent
        to finish (aborting it after ``timeout`` seconds); returns the transcript."""
        s = self.call("POST", "/session", {"title": title, "agent": agent, "model": {"providerID": provider, "id": model}}, directory,
                      timeout=60)
        sid = s["id"]
        t0 = time.time()
        error = None
        timer = threading.Timer(timeout, lambda: self.call("POST", f"/session/{sid}/abort", None, directory, timeout=10))
        timer.daemon = True
        timer.start()
        try:
            self.call("POST", f"/session/{sid}/message", {"model": {"providerID": provider, "modelID": model}, "agent": agent,
                                                          "parts": [{"type": "text", "text": task}]}, directory)
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
        finally:
            timed_out = not timer.is_alive()
            timer.cancel()
        if timed_out:
            error = (error or "") + f" aborted after {timeout:.0f}s"
        msgs = self.call("GET", f"/session/{sid}/message", directory=directory, timeout=60) or []
        return {"session": sid, "wall_s": time.time() - t0, "error": error or None, "messages": msgs}

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _transcript_summary(msgs: List[dict]) -> dict:
    steps, tools, texts = 0, [], []
    for m in msgs:
        if m["info"].get("role") != "assistant":
            continue
        steps += 1
        for p in m["parts"]:
            if p["type"] == "tool":
                st = p.get("state", {})
                tools.append({"tool": p.get("tool"), "input": st.get("input"), "status": st.get("status"),
                              "output": (st.get("output") or st.get("error") or "")[:400]})
            elif p["type"] == "text" and p.get("text"):
                texts.append(p["text"])
    return {"steps": steps, "tool_calls": tools, "final": texts[-1] if texts else ""}


def cmd_swarm(args) -> None:
    oc = opencode_bin()
    if oc is None:
        sys.exit("run  locals setup  first")
    if args.tasks:
        raw = Path(args.tasks).read_text()
        try:
            tasks = json.loads(raw)
            tasks = [t if isinstance(t, str) else t["task"] for t in tasks]
        except ValueError:
            tasks = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.startswith("#")]
    elif args.task:
        tasks = [args.task] * args.n
    else:
        sys.exit("give --tasks FILE (one task per line) or --task TEXT [--n N]")
    ensure_server(args)
    short = model_short(args.model)
    run_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.dir).resolve()
    dirs = []
    for i in range(len(tasks)):
        d = base
        if args.workspaces:
            d = base / "swarm" / f"task_{i:02d}"
            d.mkdir(parents=True, exist_ok=True)
        dirs.append(d)
    if not args.workspaces and len(tasks) > 1:
        print(f"note: {len(tasks)} agents work in the same folder ({base}) at once and may edit the same files; "
              "--workspaces gives each its own subfolder", flush=True)
    ocs = OpenCodeServer(oc, base, run_dir / "opencode-serve.log")
    if args.auto:
        ocs.start_approver()
    print(f"{len(tasks)} agents ({args.agent} on {args.engine}/{short}) on one OpenCode server ({ocs.base}); "
          f"model server {base_url(args.port)} (open it in a browser for live batch rows)")
    results: List[Optional[dict]] = [None] * len(tasks)

    def work(i):
        results[i] = ocs.run_task(tasks[i], str(dirs[i]), args.agent, args.engine, f"{args.engine}/{short}", f"swarm {i}: {tasks[i][:40]}", args.timeout)

    t0 = time.time()
    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(tasks))]
    for th in threads:
        th.start()
    last = ""
    try:
        while any(th.is_alive() for th in threads):
            time.sleep(1)
            try:
                s = get_json(base_url(args.port) + "/stats", timeout=2)
                e = s["engines"][args.engine]
                done = sum(r is not None for r in results)
                line = (f"\r{time.time() - t0:5.0f}s  agents done {done}/{len(tasks)}  batch rows {len(s['active'])}  "
                        f"output tokens {e['completion_tokens']}  forced {e['forced_tokens']}  tool calls {e['tool_calls']}   ")
                if line != last:
                    print(line, end="", flush=True)
                    last = line
            except Exception:
                pass
    finally:
        print()
        wall = time.time() - t0
        ocs.close()
    summary = []
    for i, (task, d) in enumerate(zip(tasks, dirs)):
        r = results[i] or {"error": "no result", "messages": [], "wall_s": 0}
        sm = _transcript_summary(r["messages"])
        (run_dir / f"task_{i:02d}.json").write_text(json.dumps(r, indent=1))
        summary.append({"task": task, "dir": str(d), "session": r.get("session"), "wall_s": r["wall_s"], "error": r.get("error"),
                        "steps": sm["steps"], "tools": [t["tool"] for t in sm["tool_calls"]], "answer": sm["final"]})
        print(f"\n=== task {i}: {task}\n[{r['wall_s']:.0f}s, {sm['steps']} steps, tools: {', '.join(t['tool'] for t in sm['tool_calls']) or '-'}"
              f"{', error: ' + r['error'] if r.get('error') else ''}]  {d}\n{sm['final'][:800]}")
    (run_dir / "summary.json").write_text(json.dumps({"model": f"{args.engine}/{short}", "agent": args.agent, "wall_s": wall, "tasks": summary}, indent=1))
    try:
        s = get_json(base_url(args.port) + "/stats", timeout=2)["engines"][args.engine]
        print(f"\n{len(tasks)} agents in {wall:.0f}s wall; engine {args.engine} totals: {s['completion_tokens']} output tokens, "
              f"{s['forced_tokens']} forced, {s['tool_calls']} tool calls, prompt cache "
              f"{s['cached_tokens'] / max(1, s['prompt_tokens']):.0%}. Saved to {run_dir}")
    except Exception:
        print(f"\n{len(tasks)} agents in {wall:.0f}s. Saved to {run_dir}")


# --- bench: both engines through the API ---------------------------------------------------------

BENCH_TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file.", "parameters": {"type": "object", "properties": {
        "filePath": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["filePath"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write a file.", "parameters": {"type": "object", "properties": {
        "filePath": {"type": "string"}, "content": {"type": "string"}}, "required": ["filePath", "content"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}, "description": {"type": "string"}}, "required": ["command", "description"]}}},
]
BENCH_TASKS = [
    "Create hello.py that prints Hello and run it.",
    "Write fizzbuzz.py for 1..15 and run it.",
    "Add a function is_prime(n) to primes.py and print the primes below 30.",
    "Write a CSV file people.csv with 3 rows and compute the average age with Python.",
    "Reverse the string 'locals agents' with a Python one-liner.",
    "Count vowels in 'parallel agents share one batch' with Python.",
    "Write fib.py printing the first 12 Fibonacci numbers and run it.",
    "Create word_count.py that prints the most common word in a sentence and run it.",
]


def cmd_bench(args) -> None:
    import threading

    ensure_server(args)
    short = model_short(args.model)
    url = base_url(args.port) + "/v1/chat/completions"

    def one(engine, task, out):
        body = {"model": f"{engine}/{short}", "max_tokens": args.bench_max_tokens, "tools": BENCH_TOOLS,
                "messages": [{"role": "system", "content": "You are a coding agent working in an empty directory. Use the tools; one step at a time."},
                             {"role": "user", "content": task}]}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t = time.perf_counter()
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        out.append((time.perf_counter() - t, r["usage"]))

    rows = []
    for engine in args.bench.split(","):  # warm-up: loads the engine, compiles kernels and the tool grammar
        one(engine.strip(), BENCH_TASKS[0], [])
    for rnd in range(args.rounds):
        for engine in args.bench.split(","):
            engine = engine.strip()
            outs: list = []
            tasks = [BENCH_TASKS[i % len(BENCH_TASKS)] for i in range(args.n)]
            t0 = time.perf_counter()
            ths = [threading.Thread(target=one, args=(engine, t, outs)) for t in tasks]
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            wall = time.perf_counter() - t0
            toks = sum(u["completion_tokens"] for _, u in outs)
            forced = sum(u["locals"]["forced_tokens"] for _, u in outs)
            rows.append((engine, args.n, wall, toks, forced))
            print(f"round {rnd}  {engine:6s}  {args.n} concurrent requests: {toks} output tokens in {wall:.1f}s = {toks / wall:.0f} tok/s"
                  f"{'  (' + str(forced) + ' forced)' if forced else ''}")
    by = {}
    for e, n, w, t, f in rows:
        by.setdefault(e, []).append(t / w)
    if len(by) > 1:
        import statistics

        s = {e: statistics.median(v) for e, v in by.items()}
        print("\nmedian tokens/s: " + "  ".join(f"{e} {v:.0f}" for e, v in s.items())
              + (f"   locals/mlx = {s['locals'] / s['mlx']:.2f}x" if "locals" in s and "mlx" in s else ""))


# --- main -------------------------------------------------------------------------------------------

def cmd_tune(argv: List[str]) -> None:
    ap = argparse.ArgumentParser(prog="locals tune", description="Autotune the small-batch matrix kernels for a model once; "
                                 "results are cached in ~/.cache/locals and used by every later load.")
    ap.add_argument("--model", default=DEFAULT_MODEL, type=model_id, help=MODEL_HELP)
    ap.add_argument("--mode", choices=["full", "quick"], default="full", help="full: every batch size up to 32 plus whole-step "
                    "timing (10-20 minutes, best throughput); quick: shapes for batch sizes up to 16 (a few minutes)")
    a = ap.parse_args(argv)
    from .runtime import Runtime

    t0 = time.time()
    print(f"loading {a.model} and tuning ({a.mode}) ...", flush=True)
    Runtime.load(a.model, tune=a.mode)
    print(f"done in {time.time() - t0:.0f}s; results cached under ~/.cache/locals")


def _solve(args) -> None:
    from .solve import cmd_solve

    cmd_solve(args)


def _delegate(name: str):
    """Sub-commands implemented by other modules, with their own options."""
    if name == "generate":
        from .cli import main as fn
    elif name == "chat":
        from .chat import main as fn
    elif name == "agents":
        from .agents import main as fn
    elif name == "tune":
        fn = cmd_tune
    elif name == "menubar":
        from .menubar import main as fn
    else:
        return None
    return fn


def _load_saved() -> dict:
    """The choices saved by earlier commands (the model, the port)."""
    try:
        return json.loads((LOCALS_DIR / "config.json").read_text())
    except (OSError, ValueError):
        return {}


def _platform_ok() -> Optional[str]:
    import platform

    if sys.platform != "darwin" or platform.machine() != "arm64":
        return f"locals runs on Apple-silicon Macs (MLX on Metal); this is {sys.platform} on {platform.machine()}"
    return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a in ("--version", "-h", "--help") for a in argv[:1]):
        why = _platform_ok()
        if why:
            sys.exit(why)
    # a command with its own parser (chat, generate, agents, tune, menubar), after --model / --port or not
    lead, i = [], 0
    while i < len(argv) and (argv[i] in ("--model", "--port") or argv[i].startswith(("--model=", "--port="))):
        step = 1 if "=" in argv[i] else 2
        if argv[i].startswith("--model"):  # their parsers take --model; the port is not theirs
            lead += argv[i:i + step]
        i += step
    if i < len(argv) and _delegate(argv[i]) is not None:
        return _delegate(argv[i])((lead if argv[i] != "menubar" else []) + argv[i + 1:])  # its own options last: they win
    saved = _load_saved()
    ap = argparse.ArgumentParser(prog="locals", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from . import __version__

    ap.add_argument("--version", action="version", version=f"locals {__version__}")
    ap.add_argument("--model", default=saved.get("model", DEFAULT_MODEL), type=model_id, help=MODEL_HELP)
    ap.add_argument("--port", type=int, default=saved.get("port", DEFAULT_PORT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("generate", "one structured output from the command line (locals generate --help)"),
                        ("chat", "interactive chat (locals chat --help)"),
                        ("agents", "many tool-using agents in parallel (locals agents --help)"),
                        ("tune", "full one-time kernel autotuning for a model (locals tune --help)"),
                        ("menubar", "locals in the macOS menu bar: start, stop, open OpenCode")):
        sub.add_parser(name, help=help_, add_help=False)

    def add_common(p):
        # --model and --port may come before or after the sub-command; SUPPRESS keeps the
        # top-level value when they are not repeated after it.
        p.add_argument("--model", default=argparse.SUPPRESS, type=model_id, help=MODEL_HELP + " (default: the last one used)")
        p.add_argument("--port", type=int, default=argparse.SUPPRESS, help=f"server port (default: the last one used, else {DEFAULT_PORT})")

    def add_server_opts(p):
        add_common(p)
        p.add_argument("--engines", default=None, help="engines the server may load (default locals; mlx adds stock mlx-lm, "
                       "for comparisons)")
        p.add_argument("--preload", default=None, help="engines loaded at start (default locals)")
        p.add_argument("--slots", type=int, default=None, help="max batch size per engine (default 16)")
        p.add_argument("--temp", type=float, default=None, help="sampling temperature (default 0.2)")
        p.add_argument("--top-p", dest="top_p", type=float, default=None, help="nucleus sampling at temp > 0 (Qwen3.5 thinking: 0.95)")
        p.add_argument("--top-k", dest="top_k", type=int, default=None, help="top-k sampling at temp > 0 (Qwen3.5 thinking: 20)")
        p.add_argument("--max-tokens", dest="max_tokens", type=int, default=None, help="cap on completion tokens (default 4096)")
        p.add_argument("--cache-gb", dest="cache_gb", type=float, default=None, help="prompt cache of our engine (default: sized from free GPU memory)")
        p.add_argument("--max-context", dest="max_context", type=int, default=None,
                       help="prompt + reply tokens per request (default 65536)")
        p.add_argument("--thinking", action="store_true", help="think before every reply")
        p.add_argument("--max-calls", dest="max_calls", type=int, default=None, help="tool calls kept per reply (default 6)")
        p.add_argument("--walk", action="store_true", help="constraints by the walk kernel: no token tables, any grammar size")
        p.add_argument("--typed-tools", dest="typed_tools", action="store_true",
                       help="tool calls with parameter names and types fixed by the grammar (use with --walk)")
        p.add_argument("--ollama", dest="ollama_port", action="store_const", const=OLLAMA_PORT, default=None,
                       help=f"also answer as Ollama on its port ({OLLAMA_PORT}), for apps that only know Ollama's address")

    p = sub.add_parser("setup", help="connect an agent tool to the server: opencode (default), openclaw, hermes")
    p.add_argument("target", nargs="?", choices=["opencode", "openclaw", "hermes"], default="opencode",
                   help="opencode: install it if needed and write its config; openclaw / hermes: write their config")
    add_common(p)
    p.add_argument("--engine", choices=list(PROVIDERS), default="locals", help="default provider in OpenCode")
    p.add_argument("--agent", default=None, help="default agent in OpenCode (orchestrator, coder, build, plan; "
                   "default: orchestrator for Qwen3.5, coder for other models)")
    p.add_argument("--context", type=int, default=None, help="context window declared to the tool "
                   "(default 32768 for OpenCode, 65536 for OpenClaw and Hermes)")
    p.add_argument("--output", type=int, default=4096)
    p.add_argument("--reinstall", action="store_true")
    p.add_argument("--opencode-version", default=OPENCODE_VERSION)
    p.add_argument("--no-start", dest="no_start", action="store_true",
                   help="openclaw / hermes: only write the config (default: also start the server in the background)")
    p.set_defaults(fn=cmd_setup)

    p = sub.add_parser("serve", help="run the server in the foreground")
    add_server_opts(p)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("run", help="locals's own multi-agent harness: the task split into parts that agents do at once, "
                       "then the check and parallel fixes")
    p.add_argument("task", help="the task")
    p.add_argument("--dir", default=".", help="the project (default: this folder): the agents work in copies of it, and it is written only when the check passes")
    p.add_argument("--check", default=None, help="the command that says the task is done (default: the project's tests)")
    p.add_argument("--parts", type=int, default=8, help="parts at most (default 8)")
    p.add_argument("--workers", type=int, default=0, help="1: one agent does the whole task (a baseline); default: one per part")
    p.add_argument("--rounds", type=int, default=2, help="repair rounds after the check (default 2)")
    p.add_argument("--steps", type=int, default=10, help="tool steps per agent (default 10: a fresh agent does better than a long session)")
    p.add_argument("--race", type=int, default=2, help="whole-task agents racing the parts, each in its own copy (default 2; 0: none)")
    p.add_argument("--head", type=float, default=0, help="seconds the whole-task agents run alone before the parts start (default 0: at once)")
    p.add_argument("--versions", type=int, default=3, help="agents per code part, each in its own copy (tests get two more, repairs one more; six agents at once at most); the check picks (default 3)")
    p.add_argument("--engine", choices=list(PROVIDERS), default="locals", help="locals (ours) or mlx (stock mlx-lm, to compare)")
    p.add_argument("--report", default=None, help="write the run's report (JSON) here")
    add_server_opts(p)

    def _run(a):
        from .harness import cmd_run

        cmd_run(a)

    p.set_defaults(fn=_run)

    p = sub.add_parser("launch", help="open a tool on the local model: opencode, openhands, cline, kilo, openclaw, hermes, "
                       "aider, ollama, editor (locals launch list)")
    p.add_argument("tool", nargs="?", default="list", choices=["list", *LAUNCH_TOOLS], help="the tool to open (none: the list)")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="arguments for the tool (locals's own options, such as --model, "
                   "go before the tool's name: locals launch --model gemma aider)")
    add_server_opts(p)
    p.set_defaults(fn=cmd_launch)

    p = sub.add_parser("code", help="open OpenCode on a project (starts the server if needed)")
    p.add_argument("dir", nargs="?", default=".", help="the project directory (default: this one)")
    p.add_argument("--engine", choices=list(PROVIDERS), default=None, help="OpenCode's default provider (kept for later launches)")
    p.add_argument("--agent", default=None, help="OpenCode's default agent (kept for later launches)")
    p.add_argument("--logs", action="store_true", help="print OpenCode logs")
    p.add_argument("--run", default=None, metavar="TASK", help="run this task to the end without the interface, print the answer")
    p.add_argument("--continue", dest="cont", action="store_true", help="with --run: continue the project's last session")
    add_server_opts(p)
    p.set_defaults(fn=cmd_code)

    p = sub.add_parser("solve", help="several agents race on one task in copies of the project; the first that passes the check wins")
    p.add_argument("task", help="what to do, as you would tell the agent")
    p.add_argument("--dir", default=".", help="the project directory (default: this one)")
    p.add_argument("--check", default=None, help="the command that decides success (exit code 0), e.g. 'python3 -m pytest -q' "
                   "(default: the project's tests; without tests, an agent writes a check first)")
    p.add_argument("--start", type=int, default=1, help="agents at the start (default 1; --start 4 --n 4 races four)")
    p.add_argument("--n", type=int, default=1, help="agents at once, at most (default 1: on one Mac the agents share the "
                   "GPU, and fresh agents one after another were fastest)")
    p.add_argument("--burst", type=int, default=1, help="fresh agents started for each that fails, told what failed "
                   "(default 1; --n 3 --burst 2: two at once after a failure)")
    p.add_argument("--attempts", type=int, default=5, help="attempts in all (default 5)")
    p.add_argument("--hedge", type=float, default=0, help="seconds without a result before one more agent starts "
                   "(default 0: none; on one Mac a second agent slows the first)")
    p.add_argument("--fixes", type=int, default=1, help="times an agent sees its failed check and fixes it (default 1)")
    p.add_argument("--agent", default="coder", help="OpenCode agent for the attempts (default coder)")
    p.add_argument("--timeout", type=float, default=600, help="seconds per attempt (default 600)")
    p.add_argument("--keep", action="store_true", help="keep the attempts' workspaces (default: their diffs only)")
    p.add_argument("--no-apply", dest="no_apply", action="store_true", help="leave the project alone; save the winner as a diff")
    p.add_argument("--plan", action="store_true", help="split the task into parts written at the same time, a few "
                   "versions of each, all in one batch (for tasks over several files); the check picks a combination "
                   "of versions that passes, then repair rounds")
    p.add_argument("--plan-agents", dest="plan_agents", action="store_true",
                   help="with --plan: an OpenCode agent per part instead of one generation")
    p.add_argument("--parts", type=int, default=6, help="with --plan: parts at most (default 6)")
    p.add_argument("--samples", type=int, default=0, help="with --plan: versions of each part written at once "
                   "(default: 3 for up to 4 parts, 2 for up to 8)")
    p.add_argument("--engine", choices=list(PROVIDERS), default="locals",
                   help="locals (ours, default) or mlx (stock mlx-lm, for comparison)")
    p.add_argument("--no-write-check", dest="no_write_check", action="store_true",
                   help="without tests or --check, stop instead of having an agent write a check first")
    add_server_opts(p)
    p.set_defaults(fn=_solve)

    p = sub.add_parser("swarm", help="run many OpenCode agents concurrently on a task list")
    p.add_argument("--tasks", default=None, help="file with one task per line (or a JSON list)")
    p.add_argument("--task", default=None, help="one task, run --n times")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--dir", default=".", help="project directory")
    p.add_argument("--workspaces", action="store_true", help="give every task its own subdirectory under DIR/swarm/")
    p.add_argument("--agent", default="coder")
    p.add_argument("--engine", choices=list(PROVIDERS), default="locals")
    p.add_argument("--auto", action=argparse.BooleanOptionalAction, default=True,
                   help="approve the agents' permission requests (headless; default on, --no-auto to refuse them)")
    p.add_argument("--timeout", type=float, default=900, help="seconds per task before the agent is aborted")
    p.add_argument("--out", default="swarm_runs")
    add_server_opts(p)
    p.set_defaults(fn=cmd_swarm)

    p = sub.add_parser("bench", help="both engines through the API with concurrent requests")
    add_common(p)
    p.add_argument("--n", type=int, default=8, help="concurrent requests")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--engines", default="locals,mlx")
    p.add_argument("--max-tokens", dest="bench_max_tokens", type=int, default=300, help="per request (not the server's cap)")
    p.add_argument("--preload", default=None)
    p.add_argument("--slots", type=int, default=None)
    p.add_argument("--temp", type=float, default=None)
    p.add_argument("--cache-gb", dest="cache_gb", type=float, default=None)
    p.add_argument("--thinking", action="store_true")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("start", help="start the server in the background (locals stop stops it)")
    add_server_opts(p)
    p.set_defaults(fn=cmd_start)
    p = sub.add_parser("status", help="server counters")
    add_common(p)
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("stop", help="stop the background server")
    add_common(p)
    p.set_defaults(fn=cmd_stop)

    args = ap.parse_args(argv)
    if args.cmd == "bench":
        args.bench, args.engines = args.engines, "locals,mlx"  # the server must be able to load both
    args.fn(args)


if __name__ == "__main__":
    main()
