"""locals: run many AI agents at once on your Mac, about three times faster.

    locals code [DIR]          a coding assistant (OpenCode) on a project; the first time it installs
                               OpenCode, downloads the model and starts the server (--run TASK: one task)
    locals run "TASK" --dir DIR --check "CMD"
                               a team of agents on one task, each in its own copy of the project; the
                               project is changed only when the check passes
    locals launch TOOL         another coding tool or app on the local model (locals launch alone: the
                               tools on this Mac and whether they are connected; locals disconnect TOOL)
    locals models              the tested models: size, memory, whether they fit this Mac, speed
    locals chat                an interactive chat (/json <schema>, /n N, /reset)
    locals generate ...        one answer from the command line (--prompt, --schema | --regex, --n)
    locals start | stop        the server in the background | stop it
    locals status              what the server is doing: requests, prompt cache, memory
    locals serve               the server in this terminal (http://127.0.0.1:8321/v1)
    locals setup [TOOL]        only configure OpenCode (or openclaw, hermes) for the model

  More: locals solve (agents race on a task), locals swarm (OpenCode agents on a list of tasks),
  locals agents, locals bench (the speed test on this Mac), locals tune (full kernel tuning),
  locals menubar. Each has --help.

Models (--model): qwen (Qwen3.5-4B, the default from 16 GB of memory; Qwen3.5-2B below), qwen3.5-2b,
gemma (Gemma 4 E4B), lfm (LFM2.5-2.6B), minicpm (MiniCPM5-2B), or any mlx-lm model id or local path;
the last one used is remembered.
"""

from __future__ import annotations

import argparse
import contextlib
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
    """The /health answer of a locals server on the port, or None (nothing there, or another program: a
    dev server's own /health is not ours, and agents must not be pointed at it)."""
    try:
        d = get_json(base_url(port) + "/health", timeout=2)
    except Exception:
        return None
    return d if isinstance(d, dict) and d.get("status") == "ok" and "engines" in d else None


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


def _backup_path(tool: str, path: Optional[Path] = None) -> Path:
    """Where the tool's config from before locals is kept (OpenCode's next to its config, as 0.2.0 did)."""
    if tool == "opencode":
        path = path or OPENCODE_CONFIG
        return path.with_name(path.name + ".before-locals")
    return LOCALS_DIR / "backups" / f"{tool}-{(path or _config_path(tool)).name}.before-locals"


def _back_up_once(path: Path, tool: str, ours: Optional[bool] = None) -> Optional[Path]:
    """The tool's own config, kept the first time locals changes it (later runs do not overwrite it); where
    it is kept. A config that already holds locals's settings (``ours``; an earlier setup) is not the user's
    own and is not kept; a tool without a config gets an empty one as its "before", so that
    `locals disconnect` puts back nothing of locals's."""
    keep = _backup_path(tool, path)
    if keep.exists():
        return keep
    if ours is None:
        s = _tool_settings(tool) if path == _config_path(tool) else None
        ours = bool(s and (s.get("url") or s.get("ours") or s.get("legacy")))
    if ours:
        return None
    keep.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        shutil.copy2(path, keep)
        print(f"your {tool} config is saved as {keep}; to undo locals's changes: cp '{keep}' '{path}' "
              f"(or: locals disconnect {tool})")
    else:
        keep.write_text("{}\n")
    return keep


def _pid_file(port: int) -> Path:
    return LOCALS_DIR / f"server-{port}.pid"


def _serve_pid_file(port: int) -> Path:
    """The pid of a server started in the foreground (locals serve): read only to see that one is loading, never
    to stop or restart it (a server started by hand is not locals's to replace)."""
    return LOCALS_DIR / f"serve-{port}.pid"


def _legacy_pid_files(port: int) -> list:
    return [LOCALS_DIR / "server.pid"] if port == DEFAULT_PORT else []  # locals 0.2.0 before per-port files: 8321

def _is_locals_server(pid: int) -> bool:
    """Whether the process is a locals server (a pid file can outlive its process, and its number be reused)."""
    p = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True)
    return "locals.server" in p.stdout


def _locals_server_pids() -> List[int]:
    """Every locals server running on this Mac (any port, started by any command)."""
    p = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True)
    out = []
    for line in p.stdout.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if "locals.server" in cmd and pid.isdigit() and int(pid) != os.getpid():
            out.append(int(pid))
    return out


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
    try:
        answer = input(f"OpenCode is not installed. Install OpenCode {version} with its official installer (opencode.ai) "
                       f"into ~/.opencode/bin? [Y/n] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        sys.exit("\nnot installed")
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
    # OpenCode's default model is always ours: the stock engine (--engine mlx) is chosen for one launch with -m,
    # and the server loads it only when asked (a default on it failed every launch after the next restart)
    return {"$schema": "https://opencode.ai/config.json", "provider": prov, "model": oc_model("locals", short),
            "small_model": oc_model("locals", short), "default_agent": default_agent, "subagent_depth": 1,
            "agent": {**AGENTS, "build": {"steps": AGENT_STEPS}, "plan": {"steps": AGENT_STEPS}}}


def write_config(cfg: dict, path: Optional[Path] = None) -> None:
    path = path or OPENCODE_CONFIG  # looked up now, not when the module was loaded
    path.parent.mkdir(parents=True, exist_ok=True)
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except ValueError:  # comments or trailing commas (OpenCode reads JSONC): rewritten, they would be lost
            sys.exit(f"{path} is not plain JSON (comments or trailing commas?), so locals does not rewrite it. Remove "
                     f"them, or move the file away and run this again (locals then writes a new one).")
        if not isinstance(old, dict):
            sys.exit(f"{path} does not hold a JSON object; move it away and run this again")
    # the user's own config, once (locals makes itself OpenCode's default model); none to keep when it already
    # holds locals's settings, or an empty one when OpenCode had no config
    _back_up_once(path, "opencode", ours=_opencode_has_locals(old))
    users = _users_agents(old, path)
    merged = dict(old)
    for k, v in cfg.items():
        if k in ("provider", "agent") and isinstance(old.get(k), dict):
            merged[k] = {**old[k], **v}
            if k == "agent":
                for name in ("build", "plan"):  # the user's own settings of OpenCode's agents stay
                    if isinstance(old[k].get(name), dict):
                        merged[k][name] = {**old[k][name], **v.get(name, {})}
                for name in users:  # an agent of the user's that has the name of one of locals's: the user's stays
                    merged[k][name] = old[k][name]
        else:
            merged[k] = v
    if users:
        print(f"kept your own OpenCode agent{'s' if len(users) > 1 else ''} {', '.join(users)} (locals has agents of "
              f"the same name{'s' if len(users) > 1 else ''}; yours {'are' if len(users) > 1 else 'is'} used)")
    for name in LEGACY:  # locals's provider under its old name: the server no longer answers to it
        (merged.get("provider") or {}).pop(name, None)
    _write_atomic(path, json.dumps(merged, indent=2) + "\n")


def _locals_providers(provs) -> set:
    """The providers of an OpenCode config that are locals's: "locals" (and its old names), and "mlx" where locals
    wrote it (by the name it gives it). A provider of the user's own called "mlx" (another MLX server) is theirs:
    taken for locals's, it was neither backed up nor kept by a disconnect."""
    if not isinstance(provs, dict):
        return set()
    out = {p for p in ("locals", *LEGACY) if p in provs}
    if isinstance(provs.get("mlx"), dict) and provs["mlx"].get("name") == PROVIDERS["mlx"][0]:
        out.add("mlx")
    return out


def _locals_model(cfg: dict, value) -> bool:
    """Whether a model setting of an OpenCode config (``model``, ``small_model``) names one of locals's providers."""
    head = str(value or "").split("/")[0]
    provs = cfg.get("provider") if isinstance(cfg, dict) else None
    return head in ("locals", *LEGACY) or (head in PROVIDERS and head in _locals_providers(provs))


def _opencode_has_locals(cfg: dict) -> bool:
    """Whether an OpenCode config already holds locals's settings (a provider or the default model)."""
    if not isinstance(cfg, dict):
        return False
    return bool(_locals_providers(cfg.get("provider"))) or _locals_model(cfg, cfg.get("model"))


def _users_agents(old: dict, path: Path) -> List[str]:
    """The user's own agents with the names of locals's (coder, reviewer, ...): there before locals (in the
    backup) and not locals's own definition."""
    agents = old.get("agent") if isinstance(old.get("agent"), dict) else {}
    keep = _backup_path("opencode", path)
    try:
        before = json.loads(keep.read_text()) if keep.is_file() else {}
    except ValueError:
        before = {}
    had = (before.get("agent") or {}) if isinstance(before, dict) else {}
    return [n for n in AGENTS if isinstance(agents.get(n), dict) and isinstance(had.get(n), dict)
            and agents[n].get("prompt") != AGENTS[n]["prompt"]]


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
    cur = _loads_jsonish((state / "openclaw.json").read_text()) if (state / "openclaw.json").is_file() else None
    want = patch["models"]["providers"]["locals"]
    provs = ((cur.get("models") or {}).get("providers") or {}) if isinstance(cur, dict) else {}
    if isinstance(cur, dict) and (provs.get("locals") or {}) == want and not any(n in provs for n in LEGACY) \
            and (((cur.get("agents") or {}).get("defaults") or {}).get("model") or {}).get("primary") == \
            patch["agents"]["defaults"]["model"]["primary"]:
        print(f"OpenClaw uses {patch['agents']['defaults']['model']['primary']} at {base_url(args.port)}/v1 already.")
        return True  # set up like this already: `openclaw config patch` takes seconds (minutes on its first start)
    _back_up_once(state / "openclaw.json", "openclaw")
    print(f"writing the locals provider into OpenClaw's config ({oc} config patch; its first start can take a few minutes) ...",
          flush=True)
    # locals's provider lists the model the server serves: a setup for another model replaces the
    # list (OpenClaw refuses a patch that would drop entries, "locals/qwen3.5-4b" for Gemma)
    p = subprocess.run([oc, "config", "patch", "--file", str(path), "--replace-path", "models.providers.locals.models"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"openclaw config patch failed:\n{p.stdout[-1500:]}{p.stderr[-1500:]}\nthe fragment is in {path}")
    for name in LEGACY:  # locals's provider under its old name (stcon): the server no longer answers to it
        if name in provs:
            subprocess.run([oc, "config", "unset", f"models.providers.{name}"], capture_output=True, text=True)
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
    cur = _read_yaml(_hermes_config()) or {}
    have = cur.get("model") if isinstance(cur.get("model"), dict) else {}
    if all(str(have.get(k.split(".", 1)[1])) == v for k, v in kv.items()):
        print(f"Hermes Agent uses {kv['model.default']} at {kv['model.base_url']} already.")
        return True  # set up like this already: each `hermes config set` takes 3-4 s
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
    check_model(args.model)  # a model that cannot run (a typo, too big) is refused before any config names it
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


_MODEL_FILES = ["*.json", "model*.safetensors", "*.py", "tokenizer.model", "*.tiktoken", "tiktoken.model", "*.txt", "*.jsonl", "*.jinja"]


def check_model(model: str) -> None:
    """Exit with what to do if ``model`` cannot run here: a GGUF model, too big for this Mac, or a name that
    is neither a folder nor a model on Hugging Face (a typo). Nothing is downloaded; offline, a model not
    downloaded yet passes (its download says what is wrong)."""
    from .defaults import cached_snapshot, gguf_hint, too_big

    remembered = _load_saved().get("model") == model
    hint = (f" ({model} is the model locals remembered from an earlier command; choose one with --model, e.g. "
            f"--model qwen)") if remembered else ""
    why = gguf_hint(model) or too_big(model, remote=True)  # before a download of gigabytes that cannot run
    if why:
        sys.exit(f"locals: {why}{hint}")
    if Path(model).expanduser().exists() or cached_snapshot(model, _MODEL_FILES) is not None:
        return
    if model.startswith(("/", "~", ".")):
        sys.exit(f"locals: no such folder: {model}{hint}")
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        HfApi().model_info(model, timeout=10)
    except (RepositoryNotFoundError, GatedRepoError):
        from .defaults import ALIASES

        import difflib

        near = difflib.get_close_matches(model.split("/")[-1].lower(), list(ALIASES), n=1)
        sys.exit(f"locals: {model} is not a model on Hugging Face (or it needs a sign-in){hint}"
                 + (f"; did you mean --model {near[0]}?" if near else "; locals models lists the tested ones"))
    except Exception:  # noqa: BLE001  # offline or Hugging Face slow: the download tells
        pass


def fetch_model(model: str) -> None:
    """Download the model (first run only), with progress, before the server needs it."""
    check_model(model)
    if Path(model).expanduser().exists():
        return
    from huggingface_hub import snapshot_download

    from .defaults import cached_snapshot

    patterns = _MODEL_FILES
    if cached_snapshot(model, patterns) is not None:
        return
    print(f"downloading {model} (first run only) ...", flush=True)
    try:
        snapshot_download(model, allow_patterns=patterns)
    except KeyboardInterrupt:
        sys.exit("\nstopped the download; run the command again to go on where it stopped")
    except Exception as e:  # noqa: BLE001
        from .server import _load_error

        sys.exit(f"locals: {_load_error(model, e) or f'could not download {model}: {e}'}")


# --- the server ---------------------------------------------------------------------------------

_CMD_OPTS = ("engines", "preload", "slots", "temp", "top_p", "top_k", "max_tokens", "cache_gb", "max_calls", "max_context",
             "ollama_port")
_CMD_FLAGS = ("thinking", "walk", "typed_tools")
_KEPT = tuple(k for k in _CMD_OPTS + _CMD_FLAGS if k not in ("engines", "preload"))  # (a restart keeps these)


def _opts_file(port: int) -> Path:
    return LOCALS_DIR / f"server-{port}.json"


def server_cmd(args, extra: List[str] = ()) -> List[str]:
    cmd = [python(), "-m", "locals.server", "--port", str(args.port), "--model", args.model]
    for k in _CMD_OPTS:
        v = getattr(args, k, None)
        if v is not None:
            cmd += ["--" + k.replace("_", "-"), str(v)]
    for flag in _CMD_FLAGS:
        if getattr(args, flag, False):
            cmd.append("--" + flag.replace("_", "-"))
    return cmd + list(extra)


def remember(**kw) -> None:
    """Save choices (the model, the port) as the defaults of later locals commands."""
    try:
        LOCALS_DIR.mkdir(exist_ok=True)
        path = LOCALS_DIR / "config.json"
        cfg = json.loads(path.read_text()) if path.exists() else {}
        was = cfg.get("port")
        if kw.get("port") is not None and was not in (None, kw["port"]) and server_alive(int(was)) \
                and not _same_server(int(was), kw["port"]):
            print(f"note: a locals server still runs on port {was} (the port used before), with its own copy of the "
                  f"model; locals stop --port {was} stops it", flush=True)
        cfg.update({k: v for k, v in kw.items() if v is not None})
        _write_atomic(path, json.dumps(cfg, indent=1))  # (two commands at once, or a crash, never leave half a file)
    except (OSError, ValueError):
        pass


def _same_server(a: int, b: int) -> bool:
    """Whether ports a and b answer from one server (a port forward, a tunnel, a proxy): the same model, started
    at the same time."""
    try:
        sa, sb = get_json(base_url(a) + "/stats", timeout=3), get_json(base_url(b) + "/stats", timeout=3)
    except Exception:  # noqa: BLE001
        return False
    return sa.get("model") == sb.get("model") and abs((sa.get("uptime_s") or 0) - (sb.get("uptime_s") or 0)) < 3


def cmd_serve(args) -> None:
    loading = _starting_pid(args.port)
    if loading is not None:  # a server loading the model: a second one would load it twice
        sys.exit(f"a locals server is starting on port {args.port} (pid {loading}); wait for it (locals status), "
                 f"or stop it (locals stop) first")
    fetch_model(args.model)  # a model that cannot run is refused here, before it becomes the default
    remember(model=args.model, port=args.port)  # `locals setup` / `locals code` then use the same model
    # this process becomes the server (same pid): while it loads, another command waits for it instead of
    # starting a second copy of the model next to it (16 GB Macs swap, or the GPU panics)
    LOCALS_DIR.mkdir(parents=True, exist_ok=True)
    _serve_pid_file(args.port).write_text(str(os.getpid()))
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
        # (args.model is a full id or an absolute path by now; short names matched a fine-tune folder, or another
        # publisher's conversion, to the model running, which then kept serving in its place)
        same = not running or running == args.model
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
        try:  # the options it was started with (--ollama-port, --max-context, --slots, ...) that this command leaves unset
            kept = json.loads(_opts_file(args.port).read_text())
        except (OSError, ValueError):
            kept = {}
        for k in _KEPT:
            if kept.get(k) not in (None, False) and getattr(args, k, None) in (None, False):
                setattr(args, k, kept[k])
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
    loading = _starting_pid(args.port)
    if loading is not None:
        # started by an earlier command (one stopped with Ctrl-C while it waited) and still loading the model:
        # a second server here would load the model a second time next to it
        print(f"the locals server on port {args.port} is starting already (pid {loading}); waiting for it ...", flush=True)
        _wait_for_server(args, lambda: _is_locals_server(loading), log_path, log_path.stat().st_size if log_path.exists() else 0,
                         wait_s)
        return _ensure_server(args, wait_s)  # up now: the same checks as for any running server (its model, its code)
    if log_path.exists() and log_path.stat().st_size > 50 << 20:  # the log of earlier runs, kept once
        os.replace(log_path, log_path.with_suffix(".log.1"))
    start = log_path.stat().st_size if log_path.exists() else 0
    log = open(log_path, "ab")
    p = subprocess.Popen(server_cmd(args), stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                         cwd=str(Path(__file__).resolve().parent.parent))
    _pid_file(args.port).write_text(str(p.pid))
    _opts_file(args.port).write_text(json.dumps({k: getattr(args, k, None) for k in _KEPT}))
    print(f"starting the locals server for {args.model} on port {args.port} (log {log_path}) ...", flush=True)
    _wait_for_server(args, lambda: p.poll() is None, log_path, start, wait_s)


def _starting_pid(port: int) -> Optional[int]:
    """A locals server started on this port (in the background, or by locals serve) that is still loading (running,
    not listening yet)."""
    try:
        serving = int(_serve_pid_file(port).read_text() or 0)
    except (OSError, ValueError):
        serving = 0
    for pid in (_saved_pid(port), serving):
        if pid and _is_locals_server(pid) and _listener_pid(port) is None:
            return pid
    return None


def _wait_for_server(args, running, log_path: Path, start: int, wait_s: float) -> None:
    """Wait for the server on args.port to answer, showing what its log says on the way (loading, tuning);
    ``running()``: whether its process is still alive. Ctrl-C stops the waiting, not the server."""
    t0, seen = time.time(), start
    try:
        _follow(args, running, log_path, start, t0, seen, wait_s)
    except KeyboardInterrupt:
        sys.exit(f"\nstopped waiting; the server keeps starting in the background (locals status shows it, "
                 f"locals stop stops it)")


def _follow(args, running, log_path: Path, start: int, t0: float, seen: int, wait_s: float) -> None:
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
        if not running():
            # ``start`` counts bytes (the log has multi-byte progress bars), so slice before decoding
            tail = log_path.read_bytes()[start:].decode(errors="replace").strip().splitlines()[-15:] if log_path.exists() else []
            sys.exit("the server stopped:\n  " + "\n  ".join(tail))
        if server_alive(args.port):
            print(f"server ready in {time.time() - t0:.0f}s")
            return
        time.sleep(1)
    sys.exit(f"the server did not come up in {wait_s:.0f}s; see {log_path}")


def _saved_pid(port: int) -> Optional[int]:
    for f in (_pid_file(port), *_legacy_pid_files(port)):
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
    fetch_model(args.model)  # checked (and downloaded) before it becomes the default of later commands
    ensure_server(args)
    remember(model=args.model, port=args.port)
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
    for f in (_pid_file(args.port), *_legacy_pid_files(args.port)):
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
        loading = _starting_pid(args.port)
        print(f"the locals server on port {args.port} is starting (pid {loading}); log: {LOCALS_DIR / 'server.log'}"
              if loading else f"no server on port {args.port}")
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
    return (str(cfg.get("model", "")) == oc_model("locals", model_short(model)) and isinstance(prov, dict)
            and (prov.get("options") or {}).get("baseURL") == base_url(port) + "/v1")


def _has_provider(name: str) -> bool:
    try:
        return name in (json.loads(OPENCODE_CONFIG.read_text()).get("provider") or {})
    except (OSError, ValueError, AttributeError):
        return False


def _answers_as_locals(port: int) -> bool:
    return server_alive(port) is not None


LAUNCH_TOOLS = {
    "opencode": "OpenCode (the same as locals code)",
    "openhands": "OpenHands",
    "cline": "Cline, in the terminal",
    "kilo": "Kilo Code, in the terminal",
    "openclaw": "OpenClaw",
    "hermes": "Hermes Agent",
    "aider": "Aider",
    "claude": "Claude Code, through Anthropic's Messages API (for this launch only)",
    "ollama": "apps with an Ollama setting: locals answers as Ollama",
    "editor": "VS Code (Roo Code, Cline, Kilo Code, Continue), Zed, JetBrains: the settings to enter",
}
TOOL_SITES = {"openclaw": "https://openclaw.ai", "hermes": "https://hermes-agent.nousresearch.com"}


INSTALLERS = {  # (package manager, its arguments) for tools a launch can install itself
    "openhands": ("uv", ["tool", "install", "--python", "3.12", "openhands"]),
    "aider": ("uv", ["tool", "install", "aider-chat"]),
    "cline": ("npm", ["install", "-g", "cline"]),
    "kilo": ("npm", ["install", "-g", "@kilocode/cli"]),
    "claude": ("npm", ["install", "-g", "@anthropic-ai/claude-code"]),
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


# --- connections: which tools use locals, and taking locals out of a tool's config --------------
#
# OpenCode, OpenClaw and Hermes keep locals in their own config (written by locals setup / launch);
# Aider, OpenHands, Cline and Kilo Code get locals's settings each time locals launch opens them.

CONFIGURED = ("opencode", "openclaw", "hermes")
LEGACY = ("stcon",)  # locals's name before 0.2: early setups still have a provider called stcon
TOOL_NAMES = {"opencode": "OpenCode", "openclaw": "OpenClaw", "hermes": "Hermes Agent", "aider": "Aider",
              "openhands": "OpenHands", "cline": "Cline", "kilo": "Kilo Code", "claude": "Claude Code"}


def _openclaw_config() -> Path:
    return Path(os.environ.get("OPENCLAW_STATE_DIR") or HOME / ".openclaw") / "openclaw.json"


def _hermes_config() -> Path:
    return Path(os.environ.get("HERMES_HOME") or HOME / ".hermes") / "config.yaml"


def _read_yaml(path: Path) -> Optional[dict]:
    try:
        import yaml

        d = yaml.safe_load(path.read_text())
        return d if isinstance(d, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:  # noqa: BLE001
        return None


def _tool_settings(tool: str) -> Optional[dict]:
    """What a tool's own config says about locals: {"url": base URL of its locals provider (None if it has
    none), "model": its default model, "ours": whether that default is a locals model}; {} without a
    config, None when the config cannot be read."""
    if tool == "opencode":
        if not OPENCODE_CONFIG.exists():
            return {}
        try:
            cfg = json.loads(OPENCODE_CONFIG.read_text())
        except ValueError:
            return None
        provs = (cfg.get("provider") or {}) if isinstance(cfg, dict) else {}
        prov = provs.get("locals")
        model = str(cfg.get("model") or "") if isinstance(cfg, dict) else ""
        return {"url": ((prov or {}).get("options") or {}).get("baseURL") if isinstance(prov, dict) else None,
                "model": model, "ours": _locals_model(cfg, model), "legacy": any(n in provs for n in LEGACY)}
    if tool == "openclaw":
        path = _openclaw_config()
        if not path.exists():
            return {}
        cfg = _loads_jsonish(path.read_text())  # OpenClaw's config is JSON5
        if not isinstance(cfg, dict):
            return None
        provs = (cfg.get("models") or {}).get("providers") or {}
        prov = provs.get("locals")
        model = str((((cfg.get("agents") or {}).get("defaults") or {}).get("model") or {}).get("primary") or "")
        return {"url": prov.get("baseUrl") if isinstance(prov, dict) else None, "model": model,
                "ours": model.startswith("locals/"), "legacy": any(n in provs for n in LEGACY)}
    if tool == "hermes":
        cfg = _read_yaml(_hermes_config())
        if cfg is None:
            return None
        m = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        model = str(m.get("default") or "")
        ours = model.startswith("locals/")
        return {"url": m.get("base_url") if ours else None, "model": model, "ours": ours,
                "legacy": model.split("/")[0] in LEGACY}
    return {}


def connection(tool: str, port: int):
    """(state, detail) of a tool's connection to locals: "connected", "needs repair", "not connected",
    "not installed", or "ready" (a tool that gets locals's settings each time it is launched)."""
    if tool == "opencode":
        installed = opencode_bin() is not None
    else:
        installed = _installed(tool, tool.upper() + "_BIN") is not None
    if not installed:
        how = (f"locals launch {tool} offers to install it" if tool in INSTALLERS else
               "locals code installs it" if tool == "opencode" else TOOL_SITES.get(tool, ""))
        return "not installed", how
    if tool not in CONFIGURED:
        return "ready", f"gets the settings each time: locals launch {tool}"
    s = _tool_settings(tool)
    fix = "locals code" if tool == "opencode" else f"locals launch {tool}"
    if s is None:
        return "needs repair", f"its config cannot be read ({_config_path(tool)})"
    if s.get("legacy") and not s.get("url"):
        return "needs repair", f"it uses Locals under its old name, stcon: {fix} updates it"
    if not s.get("url"):
        return "not connected", f"{fix} connects it"
    want = base_url(port) + "/v1"
    if s["url"] != want:
        return "needs repair", f"it points at {s['url']}, the server is at {want}: {fix} fixes it"
    if not s["ours"]:
        return "connected", f"but its default model is {s['model'] or 'not set'}; choose locals/... in it"
    return "connected", f"default model {s['model']}"


def _config_path(tool: str) -> Path:
    return {"opencode": OPENCODE_CONFIG, "openclaw": _openclaw_config(), "hermes": _hermes_config()}[tool]


def print_connections(port: int) -> None:
    st = server_alive(port)
    server = (f"running at {base_url(port)}/v1" if st else f"not running (a launch starts it at {base_url(port)}/v1)")
    print(f"Tools that can use Locals. The server: {server}.\n")
    for tool in ("opencode", "openclaw", "hermes", "aider", "openhands", "cline", "kilo", "claude"):
        state, detail = connection(tool, port)
        print(f"  {tool:10s} {TOOL_NAMES[tool]:13s} {state:14s} {detail}")
    print(f"  {'ollama':10s} {'Ollama apps':13s} {'':14s} locals launch ollama: Locals answers as Ollama")
    print(f"  {'editor':10s} {'editors':13s} {'':14s} locals launch editor: the settings for VS Code, Zed, JetBrains")
    print("\nlocals launch <tool> opens one on the local model; locals disconnect <tool> takes Locals out of "
          "OpenCode's, OpenClaw's or Hermes's config.")


def _before_locals(tool: str) -> dict:
    """The tool's config from before locals first changed it (its backup), parsed; {} without one. A backup
    taken of locals's own settings (an earlier locals, or stcon) does not count: their values are not the user's."""
    keep = _backup_path(tool)
    if not keep.is_file():
        return {}
    if tool == "hermes":
        d = _read_yaml(keep) or {}
        m = d.get("model") if isinstance(d.get("model"), dict) else {}
        # locals's own settings (backed up by a second launch): its model names, or its server's address. Another
        # server on this Mac (LM Studio, llama.cpp, vLLM: 127.0.0.1 too) is the user's, set back by a disconnect
        if (str(m.get("default") or "").split("/")[0] in ("locals", *LEGACY)
                or str(m.get("base_url") or "").startswith(base_url(DEFAULT_PORT) + "/")):
            d = {k: v for k, v in d.items() if k != "model"}
        return d
    d = _loads_jsonish(keep.read_text())
    if not isinstance(d, dict):
        return {}
    if tool == "opencode":
        mine = _locals_providers(d.get("provider"))
        for key in ("model", "small_model"):
            if _locals_model(d, d.get(key)):
                d.pop(key)
        if isinstance(d.get("provider"), dict):
            d["provider"] = {k: v for k, v in d["provider"].items() if k not in mine}
        if isinstance(d.get("agent"), dict):  # locals's own agents written by an earlier locals
            d["agent"] = {k: v for k, v in d["agent"].items() if not (k in AGENTS and isinstance(v, dict)
                                                                     and v.get("description") == AGENTS[k]["description"])}
    return d


def _loads_jsonish(text: str):
    """JSON, or JSON5 / JSONC as OpenClaw and OpenCode allow it (comments, trailing commas, unquoted keys,
    single quotes); None if it still does not parse. Read as tokens, so that nothing inside a string changes
    (regular expressions over the whole text turned "Hello, world: hi" into a broken key)."""
    try:
        return json.loads(text)
    except ValueError:
        pass
    import re

    toks, i, n = [], 0, len(text)  # (kind, text): s a string (now JSON), p punctuation, w space, a anything else
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j, buf = i + 1, []
            while j < n and text[j] != ch:
                if text[j] == "\\" and j + 1 < n:
                    buf.append("'" if text[j + 1] == "'" else text[j:j + 2])  # JSON has no \'
                    j += 2
                    continue
                buf.append('\\"' if text[j] == '"' else text[j])  # a double quote inside single quotes
                j += 1
            toks.append(("s", '"' + "".join(buf) + '"'))
            i = j + 1
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
        elif ch.isspace():
            toks.append(("w", ch))
            i += 1
        elif ch in "{}[],:":
            toks.append(("p", ch))
            i += 1
        else:
            j = i + 1
            while j < n and not text[j].isspace() and text[j] not in "{}[],:\"'" and not text.startswith(("//", "/*"), j):
                j += 1
            toks.append(("a", text[i:j]))
            i = j
    sig = [k for k, t in enumerate(toks) if t[0] != "w"]
    after = {a: toks[b][1] for a, b in zip(sig, sig[1:])}  # the next token that is not space
    out = []
    for k, (kind, t) in enumerate(toks):
        if kind == "p" and t == "," and after.get(k) in ("}", "]"):  # a trailing comma
            continue
        if kind == "a" and after.get(k) == ":" and re.fullmatch(r"[A-Za-z_$][\w$]*", t):  # an unquoted key
            t = '"' + t + '"'
        out.append(t)
    try:
        return json.loads("".join(out))
    except ValueError:
        return None


def disconnect_opencode() -> List[str]:
    """Take locals out of OpenCode's config: its providers, and the default model, agents and settings
    locals wrote, put back as they were before locals (or removed when OpenCode had none). What changed."""
    path = OPENCODE_CONFIG
    if not path.exists():
        return []
    try:
        cfg = json.loads(path.read_text())
    except ValueError:
        sys.exit(f"{path} is not plain JSON, so locals does not rewrite it; remove the \"locals\" provider by hand")
    before, changed = _before_locals("opencode"), []
    prov, prov_before = cfg.get("provider") or {}, before.get("provider") or {}
    mine = _locals_providers(prov)
    model_ours = {key: _locals_model(cfg, cfg.get(key)) for key in ("model", "small_model")}  # (before the providers go)
    for pid in sorted(mine):
        if pid in prov_before:  # the user's own provider of that name, which locals's replaced (--engine mlx)
            prov[pid] = prov_before[pid]
            changed.append(f"your {pid} provider, back as it was")
        else:
            del prov[pid]
            changed.append(f"the {pid} provider")
    if "provider" in cfg and not prov and "provider" not in before:
        del cfg["provider"]
    ours = {"model": lambda v: model_ours["model"], "small_model": lambda v: model_ours["small_model"],
            "default_agent": lambda v: v in AGENTS, "subagent_depth": lambda v: v == 1,
            "$schema": lambda v: v == "https://opencode.ai/config.json"}
    for key, is_ours in ours.items():
        if key in cfg and is_ours(cfg[key]) and cfg[key] != before.get(key):
            if key in before:
                cfg[key] = before[key]
            else:
                del cfg[key]
            changed.append(key)
    agents, agents_before = cfg.get("agent") or {}, before.get("agent") or {}
    for name in list(agents):
        if name in AGENTS and name not in agents_before:
            del agents[name]
            changed.append(f"the {name} agent")
        elif name in AGENTS and agents[name] != agents_before[name]:  # the user's own agent of that name, back
            agents[name] = agents_before[name]
            changed.append(f"your {name} agent, back as it was")
        elif name in ("build", "plan") and isinstance(agents[name], dict):
            was = agents_before.get(name) if isinstance(agents_before.get(name), dict) else {}
            if "steps" in was:
                if agents[name].get("steps") != was["steps"]:
                    agents[name]["steps"] = was["steps"]
                    changed.append(f"the {name} agent's step limit, back to {was['steps']}")
            elif agents[name].pop("steps", None) is not None:
                changed.append(f"the {name} agent's step limit")
            if not agents[name] and name not in agents_before:
                del agents[name]
    if "agent" in cfg and not agents and "agent" not in before:
        del cfg["agent"]
    if changed:
        _write_atomic(path, json.dumps(cfg, indent=2) + "\n")
    _backup_path("opencode").unlink(missing_ok=True)  # a later connect keeps the config as it is then
    return changed


def disconnect_openclaw() -> List[str]:
    """Take locals out of OpenClaw's config with its own CLI (openclaw config unset / set)."""
    s = _tool_settings("openclaw")
    if not s or not (s.get("url") or s.get("ours") or s.get("legacy")):
        return []
    oc = _tool_bin("openclaw", "OPENCLAW_BIN")
    if oc is None:
        sys.exit(f"openclaw is not on the PATH; remove models.providers.locals from {_openclaw_config()} by hand")
    changed = []
    if s.get("url"):
        _run_tool([oc, "config", "unset", "models.providers.locals"], "openclaw")
        changed.append("the locals provider")
    if s.get("legacy"):
        for name in LEGACY:
            _run_tool([oc, "config", "unset", f"models.providers.{name}"], "openclaw")
            changed.append(f"the {name} provider")
    if s.get("ours") or s["model"].split("/")[0] in LEGACY:
        old = str((((_before_locals("openclaw").get("agents") or {}).get("defaults") or {}).get("model") or {}).get("primary") or "")
        if old and old.split("/")[0] not in ("locals", *LEGACY):
            _run_tool([oc, "config", "set", "agents.defaults.model.primary", old], "openclaw")
            changed.append(f"the default model, back to {old}")
        else:
            _run_tool([oc, "config", "unset", "agents.defaults.model.primary"], "openclaw")
            changed.append("the default model (choose one in OpenClaw)")
    _backup_path("openclaw").unlink(missing_ok=True)  # a later connect keeps the config as it is then
    return changed


def disconnect_hermes() -> List[str]:
    """Put back the model settings locals gave Hermes (hermes config set / unset)."""
    s = _tool_settings("hermes")
    if not s or not (s.get("ours") or s.get("legacy")):
        return []
    hm = _tool_bin("hermes", "HERMES_BIN")
    if hm is None:
        sys.exit(f"hermes is not on the PATH; edit the model section of {_hermes_config()} by hand")
    old = _before_locals("hermes").get("model")
    old = old if isinstance(old, dict) else {}
    changed = []
    for key in hermes_settings("x", DEFAULT_PORT):  # the keys locals sets
        field = key.split(".", 1)[1]
        if old.get(field) not in (None, ""):
            _run_tool([hm, "config", "set", key, str(old[field])], "hermes")
        else:
            _run_tool([hm, "config", "unset", key], "hermes")
        changed.append(key)
    _backup_path("hermes").unlink(missing_ok=True)  # a later connect keeps the config as it is then
    return changed


def _run_tool(cmd: List[str], tool: str) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"{' '.join(cmd[1:])} failed:\n{(p.stdout + p.stderr)[-1500:]}")


def cmd_disconnect(args) -> None:
    tool = args.tool
    if tool in ("openclaw", "hermes") and _tool_settings(tool) is None:
        how = ("openclaw config unset models.providers.locals" if tool == "openclaw" else
               "hermes config set model.provider ... (the provider you used before)")
        sys.exit(f"locals cannot read {_config_path(tool)}, so it changed nothing; take Locals out with: {how}")
    changed = {"opencode": disconnect_opencode, "openclaw": disconnect_openclaw, "hermes": disconnect_hermes}[tool]()
    if not changed:
        print(f"{TOOL_NAMES[tool]} does not use Locals; nothing to change.")
        return
    print(f"took Locals out of {TOOL_NAMES[tool]}'s config ({_config_path(tool)}): " + ", ".join(changed) + ".")
    print(f"Restart {TOOL_NAMES[tool]} if it is open. {'locals code' if tool == 'opencode' else 'locals launch ' + tool} "
          "connects it again.")


def cmd_launch(args) -> None:
    """Start the server (if it is not running) and open a tool on it, pointed at the local model:
    for this launch only, through its environment variables or a config file of locals's (Aider,
    OpenHands, Cline, Kilo Code, Claude Code); OpenCode, OpenClaw and Hermes get locals in their own config (a
    copy of it is kept the first time, see _back_up_once)."""
    tool, rest = args.tool, list(args.rest or [])
    # the tool's project folder is this one: OpenCode and Kilo read it from $PWD, which a shell keeps right but a
    # script that started locals in another folder does not (Kilo then edited the files of the folder in $PWD)
    os.environ["PWD"] = os.getcwd()
    if tool == "list":
        print_connections(args.port)
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
    from .defaults import agent_note

    if tool not in ("ollama", "editor") and agent_note(args.model):
        print(agent_note(args.model), file=sys.stderr, flush=True)
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
        # search/replace edits: in the whole-file format (Aider's choice for a model it does not know) code quoted
        # after a sentence became a new file named after the sentence (5 of 7 bug fixes with Qwen3.5-4B, none of 7
        # with diff); the user's own --edit-format comes later and wins
        os.execve(exe, [exe, "--model", f"openai/locals/{short}", "--no-show-model-warnings", "--edit-format", "diff",
                        *rest], env)
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
    if tool == "claude":
        try:  # the server's context: Claude Code assumes 200k tokens for a model it does not know, and compacts too late
            ctx = int(get_json(base + "/v1/models", timeout=5)["data"][0]["context_length"])
        except Exception:  # noqa: BLE001
            ctx = AGENT_CONTEXT
        ref = f"locals/{short}"
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}  # a key for
        # Anthropic's own API stays out of a session that talks to this Mac
        env.update(ANTHROPIC_BASE_URL=base, ANTHROPIC_AUTH_TOKEN="locals", ANTHROPIC_MODEL=ref, ANTHROPIC_SMALL_FAST_MODEL=ref,
                   ANTHROPIC_DEFAULT_HAIKU_MODEL=ref, ANTHROPIC_DEFAULT_SONNET_MODEL=ref, ANTHROPIC_DEFAULT_OPUS_MODEL=ref,
                   CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(ctx), CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
                   API_TIMEOUT_MS="3600000")  # a long prompt is read for minutes before the first token
        os.execve(exe, [exe, *rest], env)
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
    if d in (HOME.resolve(), Path("/")) or d in HOME.resolve().parents:
        # the agent edits files and runs commands in its folder without asking: a new terminal opens in the home
        # folder, and `locals code` there would hand it every file the user has
        sys.exit(f"locals code works on a project folder, and {d} is your {'home folder' if d == HOME.resolve() else 'disk'}: "
                 f"the agent edits files and runs commands there. Run it on a project:  locals code ~/my-project  "
                 f"(mkdir it first for a new one)")
    from .defaults import agent_note

    if agent_note(args.model):
        print(agent_note(args.model), flush=True)
    if opencode_bin() is None:  # asked before the model's gigabytes download, not after
        install_opencode(OPENCODE_VERSION)
    fetch_model(args.model)  # checked (and downloaded) before OpenCode's settings name it
    if not _opencode_ready(args.model, args.port) or (args.engine == "mlx" and not _has_provider("mlx")):
        print(f"setting up OpenCode for {args.model} ...", flush=True)
        setup_opencode(args)
    oc = opencode_bin()
    ensure_server(args)
    remember(model=args.model, port=args.port)
    cfg = json.loads(OPENCODE_CONFIG.read_text())
    short = model_short(args.model)
    if args.agent and cfg.get("default_agent") != args.agent:  # OpenCode's default agent, kept for later launches
        cfg["default_agent"] = args.agent
        _write_atomic(OPENCODE_CONFIG, json.dumps(cfg, indent=2) + "\n")
    # The model is given on OpenCode's command line on every launch: a model set by a project's own opencode.json
    # (or a global .jsonc) would otherwise win over ours, and could send the project to a cloud model. An engine
    # asked for here (--engine mlx) is for this launch only: later servers start without the stock engine.
    ref = oc_model(args.engine or "locals", short)
    env = dict(os.environ, OPENCODE_DISABLE_AUTOUPDATE="1", PWD=str(d))  # OpenCode reads its project directory from $PWD
    os.chdir(d)
    shown = f"{short} on the {args.engine or 'locals'} engine, agent {args.agent or cfg.get('default_agent')}"
    if args.run:
        print(f"opencode in {d}: running the task ({shown}) ...", flush=True)
        sys.exit(_opencode_run([oc, "run", "--format", "json", "-m", ref, *(["-c"] if args.cont else []), args.run], env))
    print(f"opencode in {d}  ({shown}; /models switches the engine, Tab the agent)")
    _space_opencode_starts()
    os.execve(oc, [oc, str(d), "-m", ref] + (["--print-logs"] if args.logs else []), env)


OPENCODE_START_GAP = 2.0  # seconds between OpenCode starts (see _space_opencode_starts)


def _space_opencode_starts(gap: float = OPENCODE_START_GAP) -> None:
    """Wait until ``gap`` seconds have passed since the last OpenCode a locals command started. OpenCode switches
    its database to WAL before it sets its busy timeout, so two that start in the same second (two terminals,
    a script starting several agents) fail with "database is locked"; once started they wait for each other."""
    import fcntl

    LOCALS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOCALS_DIR / "opencode-start"
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            last = float(f.read().strip() or 0)
        except ValueError:
            last = 0.0
        wait = last + gap - time.time()
        if 0 < wait <= gap:
            time.sleep(wait)
        f.seek(0)
        f.truncate()
        f.write(str(time.time()))


def _opencode_run(cmd: List[str], env: dict) -> int:
    """`opencode run` with a line per action as it happens, then the agent's answer; its exit code. Its stdin
    is closed: `opencode run` reads a piped stdin into the task and waited forever when locals ran from a script."""
    for attempt in range(3):
        _space_opencode_starts()
        t0 = time.time()
        rc, out, err = _opencode_run_once(cmd, env)
        if rc == 130:  # Ctrl-C
            return rc
        locked = "database is locked" in (out + err)
        if not (locked and time.time() - t0 < 30 and rc not in (0, 130)):
            break
        print("  (OpenCode's database was locked by another OpenCode starting; trying again)", flush=True)
    answer = _opencode_answer(out)
    print("\n" + (answer or (out[-2000:] + err[-2000:])))
    return rc


def _opencode_run_once(cmd: List[str], env: dict):
    """One `opencode run`: (exit code, stdout, stderr), a line per tool call printed as it happens."""
    import threading

    p = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    err: List[str] = []
    t = threading.Thread(target=lambda: err.append(p.stderr.read()), daemon=True)
    t.start()
    out: List[str] = []
    try:
        for line in p.stdout:
            out.append(line)
            step = _opencode_step(line)
            if step:
                print("  " + step, flush=True)
        p.wait()
    except KeyboardInterrupt:
        p.terminate()
        print("\nstopped")
        return 130, "".join(out), ""
    t.join(timeout=5)
    return p.returncode, "".join(out), "".join(err)


def _opencode_step(line: str) -> str:
    """A short line for a tool call in `opencode run --format json` output ("bash: pytest -q"), else ""."""
    try:
        ev = json.loads(line)
    except ValueError:
        return ""
    part = ev.get("part") or {}
    state = part.get("state") or {}
    if ev.get("type") != "tool_use" or state.get("status") not in ("completed", "error"):
        return ""
    args = state.get("input") or {}
    what = next((str(args[k]) for k in ("command", "filePath", "path", "pattern", "description", "url", "query") if args.get(k)), "")
    what = what.replace("\n", " ")
    return f"{part.get('tool')}: {what[:100]}" + (" (failed)" if state.get("status") == "error" else "")


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
        _space_opencode_starts()
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


BENCH_FILE_NAME = "bench.json"


def _bench_file() -> Path:
    return LOCALS_DIR / BENCH_FILE_NAME


def load_bench() -> dict:
    """Speeds locals bench measured on this Mac, by model id."""
    try:
        d = json.loads(_bench_file().read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def cmd_bench(args) -> None:
    """The README's speed test on this Mac: the same model through locals and through stock mlx-lm, with
    1, 4 and 8 requests at once (--sizes), 256 tokens each; or, with --agents, tool-calling agent
    requests (the older test). The speeds are saved for locals models."""
    if args.agents:
        return _bench_agents(args)
    from . import bench
    from .defaults import HEADROOM_GB, chip, gpu_limit_gb, memory_gb, weights_gb

    engines = [e.strip() for e in args.bench.split(",") if e.strip()]
    w, limit = weights_gb(args.model), gpu_limit_gb()
    if "mlx" in engines and w and limit and 2 * w + HEADROOM_GB > limit and not os.environ.get("LOCALS_IGNORE_MEMORY"):
        print(f"note: two copies of {model_short(args.model)} ({2 * w:.1f} GB) do not fit next to each other in this Mac's "
              f"{limit:.1f} GB of GPU memory; measuring Locals only", file=sys.stderr)
        engines = [e for e in engines if e != "mlx"]
    args.engines = ",".join(dict.fromkeys(["locals", *engines]))
    before = _background_pid(args.port) if server_alive(args.port) else None
    started = server_alive(args.port) is None
    with contextlib.redirect_stdout(sys.stderr):
        fetch_model(args.model)
        ensure_server(args)
    started = started or _background_pid(args.port) != before  # (an idle server of another model, switched to this one)
    try:
        _bench_run(args, bench, engines, chip, memory_gb)
    finally:
        if started and "mlx" in engines:  # the stock engine's copy of the weights is not left loaded
            with contextlib.redirect_stdout(sys.stderr):
                cmd_stop(args)


def _bench_run(args, bench, engines, chip, memory_gb) -> None:
    sizes = [int(s) for s in str(args.sizes).split(",") if s.strip()]
    short, base = model_short(args.model), base_url(args.port) + "/v1"
    print(f"{short} on {chip() or 'this Mac'} ({memory_gb():.0f} GB), {args.bench_max_tokens} tokens per request; "
          "speed is the tokens per second all requests write together.", flush=True)
    res = {}
    for engine in engines:
        name = "Locals" if engine == "locals" else "mlx-lm" if engine == "mlx" else engine
        print(f"\n{name} ({engine}/{short}):", flush=True)
        try:
            r = bench.sweep(base, f"{engine}/{short}", sizes, args.bench_max_tokens, {"ignore_eos": True})
        except Exception as e:  # noqa: BLE001  # what was measured before is still saved
            print(f"locals bench: {engine} failed: {e}", file=sys.stderr)
            continue
        res[engine] = {str(n): round(d["tokens_per_s"], 1) for n, d in r.items()}
    if not res:
        sys.exit("locals bench: nothing was measured")
    print()
    if "locals" in res and "mlx" in res:
        print(f"  {'requests at once':18s} {'mlx-lm':>10s} {'Locals':>10s}  faster by")
        for n in sizes:
            a, b = res["mlx"][str(n)], res["locals"][str(n)]
            print(f"  {n:<18d} {a:>6.0f} tps {b:>6.0f} tps  {b / a:.1f}x")
    saved = load_bench()
    saved[args.model] = {"chip": chip(), "memory_gb": round(memory_gb()), "date": time.strftime("%Y-%m-%d"),
                         "max_tokens": args.bench_max_tokens, **res}
    LOCALS_DIR.mkdir(parents=True, exist_ok=True)
    _write_atomic(_bench_file(), json.dumps(saved, indent=2) + "\n")
    print(f"\nsaved to {_bench_file()}; locals models shows it.")


def _bench_agents(args) -> None:
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


def cmd_models(args) -> None:
    """The tested models: their size, the memory they need, whether they fit this Mac, and their speed with
    8 requests at once (measured here by locals bench, else on an M2 Pro)."""
    from .defaults import (DEFAULT_MODEL, HEADROOM_GB, MEASURED_M2_PRO, MODELS, SHORT, cached_snapshot, chip,
                           gpu_limit_gb, memory_gb)

    limit, here = gpu_limit_gb(), load_bench()
    print(f"Models for {chip() or 'this Mac'} with {memory_gb():.0f} GB of memory"
          + (f" (the GPU can use {limit:.1f} GB)" if limit else "") + ":\n")
    print(f"  {'name':12s} {'download':>8s} {'needs':>7s}  {'on this Mac':30s} speed with 8 requests at once")
    for key, mid in MODELS.items():
        m = MEASURED_M2_PRO[key]
        local = cached_snapshot(mid, ["*.json", "*.safetensors"])
        w = sum(f.stat().st_size for f in local.glob("*.safetensors")) / 2**30 if local is not None and any(local.glob("*.safetensors")) else m["weights_gb"]
        need = w + HEADROOM_GB
        fit = "fits" if not limit or need <= limit else "too big"
        notes = [fit] + (["the default"] if mid == DEFAULT_MODEL else []) + (["downloaded"] if local is not None else [])
        mine = here.get(mid) or {}
        if "8" in (mine.get("locals") or {}):
            speed = f"{mine['locals']['8']:.0f} tps, measured here" + (f" (mlx-lm {mine['mlx']['8']:.0f})" if "8" in (mine.get("mlx") or {}) else "")
        else:
            speed = f"{m['locals']} tps on an M2 Pro (mlx-lm {m['mlx']})"
        print(f"  {SHORT[key]:12s} {w:>5.1f} GB {need:>4.1f} GB  {', '.join(notes):30s} {speed}")
    from .defaults import AGENT_TASKS

    q, g = AGENT_TASKS["qwen3.5-4b"], AGENT_TASKS["gemma-4-e4b"]
    print(f"\nFor coding agents use qwen: in our end-to-end tests (agent tools on real coding tasks, an M2 Pro) "
          f"Qwen3.5-4B finished {q[0]} of {q[1]} tasks and Gemma 4 E4B {g[0]} of {g[1]}.")
    print("Start one with  locals start --model NAME  (or --model with any command). Any MLX model from "
          "Hugging Face works too, for example --model mlx-community/Qwen3.5-9B-MLX-4bit.")
    if not any(here.get(mid) for mid in MODELS.values()):
        print("locals bench measures the speeds on this Mac.")


# --- main -------------------------------------------------------------------------------------------

def cmd_tune(argv: List[str]) -> None:
    ap = argparse.ArgumentParser(prog="locals tune", description="Autotune the small-batch matrix kernels for a model once; "
                                 "results are cached in ~/.cache/locals and used by every later load.")
    ap.add_argument("--model", default=DEFAULT_MODEL, type=model_id, help=MODEL_HELP)
    ap.add_argument("--mode", choices=["full", "quick"], default="full", help="full: every batch size up to 32 plus whole-step "
                    "timing (10-20 minutes, best throughput); quick: shapes for batch sizes up to 16 (a few minutes)")
    ap.add_argument("--report", action="store_true", help="then print what was measured: MLX's matrix kernels against the "
                    "chosen ones per batch size, and how prompts are read (for a new GPU: tells where Locals gains)")
    ap.add_argument("--force", action="store_true", help="tune even while a locals server runs (its work skews the timings)")
    a = ap.parse_args(argv)
    running = _locals_server_pids()
    if running and not a.force:  # tuning times kernels: a server on the same GPU (serving, or tuning while idle) skews
        sys.exit(f"a locals server is running (pid {', '.join(map(str, running))}): stop it first (locals stop), "
                 f"or the timings, kept for good, are taken on a shared GPU (--force tunes anyway)")
    from .runtime import Runtime

    t0 = time.time()
    print(f"loading {a.model} and tuning ({a.mode}) ...", flush=True)
    rt = Runtime.load(a.model, tune=a.mode)
    from . import kernels

    print(f"done in {time.time() - t0:.0f}s; results cached in {kernels._TUNE_PATH}")
    if a.report:
        print(kernels.tune_report(rt.model))


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
        else:  # the port: for the server chat and generate use (client.connect), which ignored it
            value = argv[i].split("=", 1)[1] if step == 1 else (argv[i + 1] if i + 1 < len(argv) else "")
            if value.isdigit():
                os.environ["LOCALS_PORT"] = value
        i += step
    if i < len(argv) and _delegate(argv[i]) is not None:
        return _delegate(argv[i])((lead if argv[i] != "menubar" else []) + argv[i + 1:])  # its own options last: they win
    if not argv:  # `locals` alone: what it can do, not a usage error
        print(__doc__.strip())
        return
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
                       help="prompt + reply tokens per request (default: what fits in memory, at least 65536)")
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

    p = sub.add_parser("bench", help="the speed test on this Mac: locals against stock mlx-lm with 1, 4 and 8 requests at once")
    add_common(p)
    p.add_argument("--sizes", default="1,4,8", help="requests at once to measure (default 1,4,8)")
    p.add_argument("--engines", default="locals,mlx", help="engines to measure (default locals,mlx)")
    p.add_argument("--max-tokens", dest="bench_max_tokens", type=int, default=256, help="tokens per request (default 256)")
    p.add_argument("--agents", action="store_true", help="tool-calling agent requests instead (with --n and --rounds)")
    p.add_argument("--n", type=int, default=8, help="with --agents: concurrent requests")
    p.add_argument("--rounds", type=int, default=2, help="with --agents: rounds")
    p.add_argument("--preload", default=None)
    p.add_argument("--slots", type=int, default=None)
    p.add_argument("--temp", type=float, default=None)
    p.add_argument("--cache-gb", dest="cache_gb", type=float, default=None)
    p.add_argument("--thinking", action="store_true")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("models", help="the tested models: size, memory, whether they fit this Mac, speed")
    add_common(p)
    p.set_defaults(fn=cmd_models)

    p = sub.add_parser("disconnect", help="take Locals out of a tool's config: opencode, openclaw, hermes")
    p.add_argument("tool", choices=list(CONFIGURED))
    add_common(p)
    p.set_defaults(fn=cmd_disconnect)

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
        args.bench = args.engines  # the engines measured; the server's (args.engines) are set by cmd_bench
    args.fn(args)


if __name__ == "__main__":
    main()
