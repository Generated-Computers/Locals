"""The end-to-end suite: the `locals` command, the server and real agent tools, run as a user runs them.

Opt-in (minutes per test, a real model on the GPU):  LOCALS_E2E=1 python -m pytest tests/e2e -v

Each module starts its own server through `locals start` in a separate HOME (one server, one model on the
GPU at a time), with a recording proxy in front of it (proxy.py): the tools talk to the proxy's port, so
the suite sees each request as the tool sees it. Nothing in the real home directory is read or written,
except the Hugging Face model cache (models are not downloaded twice) and, read only, the kernel tuning file.

    LOCALS_E2E_MODEL   the model (default qwen: Qwen3.5-4B)
    LOCALS_E2E_HOME    a HOME where the agent tools are installed and set up (default: a new, empty one;
                       tools that need their own onboarding then skip)
    LOCALS_E2E_PATH    folders with the tools' commands, put first on the PATH (colon separated)
    LOCALS_E2E_DIR     where runs, logs and the report go (default: a new temporary folder)
    LOCALS_E2E_PORT    the proxy's port (default 8341; the server listens on the next one)
    LOCALS_E2E_SERVER_ENV  extra settings for the server, e.g. "LOCALS_SOLO=0,LOCALS_MIX_CHUNK=512"
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from proxy import Proxy  # noqa: E402

E2E = os.environ.get("LOCALS_E2E") == "1"
REAL_HOME = Path.home()
MODEL = os.environ.get("LOCALS_E2E_MODEL", "qwen")
PROXY_PORT = int(os.environ.get("LOCALS_E2E_PORT", "8341"))
SERVER_PORT = PROXY_PORT + 1
EXTRA_PATH = [p for p in os.environ.get("LOCALS_E2E_PATH", "").split(":") if p]
REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
LOCALS = [sys.executable, "-m", "locals.app"]
collect_ignore = ["projects"]  # the agents' sample projects: their tests fail on purpose until an agent fixes them


def pytest_collection_modifyitems(config, items):
    for item in items:
        if HERE in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.model)
            if not E2E:
                item.add_marker(pytest.mark.skip(reason="end-to-end suite: set LOCALS_E2E=1"))


def _out_dir() -> Path:
    d = os.environ.get("LOCALS_E2E_DIR")
    path = Path(d).expanduser() if d else Path(tempfile.mkdtemp(prefix="locals-e2e-"))
    path.mkdir(parents=True, exist_ok=True)
    return path


OUT = _out_dir() if E2E else Path(tempfile.gettempdir())


def hf_hub() -> str:
    return os.environ.get("HF_HUB_CACHE") or str(REAL_HOME / ".cache" / "huggingface" / "hub")


def new_home(name: str, tuned: bool = True) -> Path:
    """A HOME of its own; ``tuned``: with this Mac's kernel tuning (a returning user), else none (a new one)."""
    home = OUT / "homes" / name
    if home.exists():
        shutil.rmtree(home)
    (home / ".config").mkdir(parents=True)
    if tuned:
        copy_tuning(home)
    return home


def copy_tuning(home: Path) -> None:
    src = REAL_HOME / ".cache" / "locals"
    dst = home / ".cache" / "locals"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("tune_*.json"):
        if not (dst / f.name).exists():
            shutil.copy2(f, dst / f.name)


def user_env(home: Path, path: Optional[List[str]] = None) -> dict:
    """The environment of a user whose home is ``home``: none of this shell's LOCALS_ (or API key) settings."""
    skip = ("LOCALS_", "STCON_", "OPENAI_", "ANTHROPIC_", "PYTEST_", "OPENCODE_", "HERMES_", "OPENCLAW_", "VIRTUAL_ENV")
    env = {k: v for k, v in os.environ.items() if not k.startswith(skip)}
    env.update(HOME=str(home), HF_HUB_CACHE=hf_hub(), OPENCODE_DISABLE_AUTOUPDATE="1", NO_COLOR="1", TERM="dumb",
               PATH=":".join([*(EXTRA_PATH if path is None else path), "/usr/local/bin", "/opt/homebrew/bin", "/usr/bin",
                              "/bin", "/usr/sbin", "/sbin"]))
    for kv in os.environ.get("LOCALS_E2E_SERVER_ENV", "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def get_json(url: str, timeout: float = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


class Stack:
    """The server (started by `locals start`, as a user starts it) and the proxy in front of it."""

    def __init__(self, home: Path, name: str):
        self.home, self.name = home, name
        self.env = user_env(home)
        self.port, self.server_port = PROXY_PORT, SERVER_PORT
        self.proxy = Proxy(SERVER_PORT, PROXY_PORT)
        self.runs = OUT / "runs" / name
        self.runs.mkdir(parents=True, exist_ok=True)
        self.load_s = None

    @property
    def log_path(self) -> Path:
        return self.home / ".locals" / "server.log"

    def locals(self, *args, cwd=None, timeout: float = 900, env=None, check=False) -> subprocess.CompletedProcess:
        return subprocess.run([*LOCALS, *map(str, args)], cwd=cwd or self.home, env=env or self.env, capture_output=True,
                              text=True, timeout=timeout, stdin=subprocess.DEVNULL, check=check)

    def start(self) -> "Stack":
        _port_free_or_die(self.port)
        _port_free_or_die(self.server_port)
        t = time.time()
        p = self.locals("start", "--model", MODEL, "--port", self.server_port, timeout=3600)
        (self.runs / "start.out").write_text(p.stdout + p.stderr)
        if p.returncode != 0:
            raise RuntimeError(f"locals start failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
        self.load_s = time.time() - t
        self.proxy.start()
        return self

    def stop(self) -> None:
        self.proxy.stop()
        self.locals("stop", "--port", self.server_port, timeout=120)
        for _ in range(120):
            if not _listening(self.server_port):
                return
            time.sleep(0.5)

    def stats(self) -> dict:
        return get_json(f"http://127.0.0.1:{self.server_port}/stats")

    def log_offset(self) -> int:
        return self.log_path.stat().st_size if self.log_path.exists() else 0

    def log_since(self, offset: int) -> str:
        if not self.log_path.exists():
            return ""
        with open(self.log_path, "rb") as f:
            f.seek(offset)
            return f.read().decode(errors="replace")

    def problems(self, offset: int) -> List[str]:
        """Server log lines since ``offset`` that a user would call a fault (tracebacks, engine errors, memory)."""
        bad = ("Traceback", "Error", "error:", "Insufficient Memory", "out of memory", "Exception", "engine error",
               "Segmentation", "Abort")
        return [ln for ln in self.log_since(offset).splitlines() if any(b in ln for b in bad)]


def _listening(port: int) -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _port_free_or_die(port: int) -> None:
    if _listening(port):
        pytest.exit(f"port {port} is in use: the e2e suite needs it (one server on the GPU at a time); stop what holds it",
                    returncode=3)


@pytest.fixture(scope="module")
def stack(request):
    """A server and proxy for this module, in the tools' HOME (LOCALS_E2E_HOME) or a new one."""
    name = Path(str(request.fspath)).stem
    given = os.environ.get("LOCALS_E2E_HOME")
    home = Path(given).expanduser() if given else new_home(name)
    if given:
        copy_tuning(home)
    s = Stack(home, name).start()
    yield s
    s.stop()


class Report:
    """Every run of the suite, written to OUT/report.json and OUT/report.md at the end."""

    def __init__(self):
        self.rows: List[dict] = []
        self.notes: List[str] = []

    def add(self, row: dict) -> dict:
        self.rows.append(row)
        self.write()
        return row

    def note(self, text: str) -> None:
        self.notes.append(text)
        self.write()

    def write(self) -> None:
        (OUT / "report.json").write_text(json.dumps({"rows": self.rows, "notes": self.notes}, indent=1, default=str))
        lines = ["| scenario | tool | task | done | wall s | requests | first token median / max s | longest pause s | "
                 "cached | errors |", "|" + "---|" * 10]
        for r in self.rows:
            lines.append(f"| {r.get('scenario', '')} | {r.get('tool', '')} | {r.get('task', '')} | "
                         f"{'yes' if r.get('ok') else 'no'} | {r.get('wall', 0):.0f} | {r.get('n_requests', 0)} | "
                         f"{r.get('ttft_median', 0):.1f} / {r.get('ttft_max', 0):.1f} | {r.get('max_gap', 0):.1f} | "
                         f"{r.get('cached', 0):.0%} | {len(r.get('faults') or [])} |")
        (OUT / "report.md").write_text("\n".join(lines + [""] + [f"- {n}" for n in self.notes]) + "\n")


_REPORT = Report()


@pytest.fixture(scope="session")
def report():
    yield _REPORT
    print(f"\ne2e report: {OUT / 'report.md'}")


def summarize(records) -> dict:
    """What a list of proxy records says about the tool's experience."""
    import statistics

    ttft = [r["first_token"] for r in records if r.get("first_token") is not None]
    prompt = sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in records)
    cached = sum(((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0 for r in records)
    faults = []
    for r in records:
        if (r.get("status") or 0) >= 500 or r.get("upstream_error") or r.get("upstream_cut"):
            faults.append(f"{r['path']}: status {r.get('status')} {r.get('upstream_error') or r.get('upstream_cut') or ''}")
        for e in r.get("errors") or []:
            faults.append(f"{r['path']}: stream error {e}")
        if r.get("odd_lines"):
            faults.append(f"{r['path']}: lines that are not JSON: {r['odd_lines'][:2]}")
    return {"n_requests": len(records), "ttft_median": statistics.median(ttft) if ttft else 0.0,
            "ttft_max": max(ttft) if ttft else 0.0, "max_gap": max((r.get("max_gap") or 0 for r in records), default=0.0),
            "prompt_tokens": prompt, "cached": cached / prompt if prompt else 0.0,
            "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in records),
            "client_errors": [f"{r['path']}: {r.get('status')}" for r in records if 400 <= (r.get("status") or 0) < 500],
            "faults": faults}


def kill_tree(p: subprocess.Popen, sig=signal.SIGTERM) -> None:
    try:
        os.killpg(p.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass
