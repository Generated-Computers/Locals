"""The server's life as a user sees it: `locals status` while agents work, `locals stop` and `locals start`
again (the prompt cache on disk then answers an agent's first request), a port held by another program,
and `locals launch` listing the tools' connections."""

import json
import socket
import time
from pathlib import Path

import pytest
from conftest import get_json

DATA = Path(__file__).resolve().parents[1] / "data" / "opencode_requests.json"


def _agent_request(stack, user: str) -> dict:
    """OpenCode's real first request (its system prompt and tools), with this user message."""
    rec = json.loads(DATA.read_text())
    model = get_json(f"http://127.0.0.1:{stack.port}/v1/models")["data"][0]["id"]
    return {"model": model, "stream": True, "max_tokens": 30, "stream_options": {"include_usage": True}, "tools": rec["tools"],
            "messages": [{"role": "system", "content": rec["system"]}, {"role": "user", "content": user}]}


def _send(stack, body) -> dict:
    import http.client

    c = http.client.HTTPConnection("127.0.0.1", stack.port, timeout=600)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
    first, usage = None, None
    for raw in c.getresponse():
        line = raw.decode().strip()
        if line.startswith("data: {"):
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            if first is None and any((ch.get("delta") or {}).get("content") or (ch.get("delta") or {}).get("tool_calls")
                                     for ch in ev.get("choices") or []):
                first = time.perf_counter() - t0
    return {"first": first, "usage": usage or {}}


def test_status_while_serving(stack):
    p = stack.locals("status", "--port", stack.server_port, timeout=30)
    assert p.returncode == 0 and "model" in p.stdout and "GPU memory" in p.stdout, p.stdout + p.stderr


def test_restart_keeps_the_agent_prompt_on_disk(stack, report):
    cold = _send(stack, _agent_request(stack, "List the files here."))
    stack.stop()
    stack.start()
    warm = _send(stack, _agent_request(stack, "What does main.py do?"))
    cached = (warm["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    report.note(f"restart: OpenCode's first request {cold['first']:.1f} s cold, {warm['first']:.1f} s after a restart "
                f"({cached} of {warm['usage'].get('prompt_tokens')} prompt tokens from the disk cache); server load {stack.load_s:.0f} s")
    assert cached > 0.5 * (warm["usage"].get("prompt_tokens") or 1), warm["usage"]
    assert warm["first"] < cold["first"], (cold, warm)


def test_port_held_by_another_program(stack):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        t0 = time.time()
        p = stack.locals("start", "--port", port, timeout=120)
        took = time.time() - t0
    assert p.returncode != 0 and "in use" in (p.stdout + p.stderr), p.stdout + p.stderr
    assert took < 20, f"locals start took {took:.0f} s to say the port is taken"


def test_stop_twice_and_launch_list(stack):
    p = stack.locals("launch", "--port", stack.port, "list", timeout=60)
    assert p.returncode == 0, p.stderr
    assert "OpenCode" in p.stdout, p.stdout
    other = 8399
    p = stack.locals("stop", "--port", other, timeout=30)
    assert p.returncode == 0 and "no locals server" in p.stdout, p.stdout + p.stderr


@pytest.mark.parametrize("bad", ["{", "[]", '{"messages": "x"}'])
def test_bad_requests_get_400(stack, bad):
    import urllib.error
    import urllib.request

    req = urllib.request.Request(f"http://127.0.0.1:{stack.port}/v1/chat/completions", data=bad.encode(),
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=30)
    assert e.value.code == 400


OFFLINE = {"HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9",
           "http_proxy": "http://127.0.0.1:9", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}


def test_offline_start_and_agent_turn(stack, report):
    """No network (a plane, a captive Wi-Fi): every connection out goes to a closed port. A downloaded model
    must start and serve, and OpenCode must finish a turn."""
    from agents import run_task
    from tasks import TASKS

    stack.stop()
    env = {**stack.env, **OFFLINE}
    t0 = time.time()
    p = stack.locals("start", "--port", stack.server_port, env=env, timeout=600)
    took = time.time() - t0
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    stack.proxy.start()
    try:
        row = run_task(stack, "opencode", TASKS["fix_bug"], env=env, timeout=600, label="offline-opencode-fix_bug")
        report.add({**row, "scenario": "offline"})
        report.note(f"offline: locals start {took:.0f} s; OpenCode's task {'done' if row['ok'] else 'not done'} in {row['wall']:.0f} s")
        assert took < 120, f"locals start took {took:.0f} s offline"
        assert row["rc"] == 0 and row["n_requests"] > 0, row
    finally:
        stack.stop()
        stack.start()  # online again, for the tests after this one


def test_switching_models_keeps_one_in_memory(stack, report):
    """`locals start --model gemma` while the server runs Qwen: the server is switched (one model in memory),
    and the new model answers."""
    import os

    other = os.environ.get("LOCALS_E2E_SWITCH", "gemma")
    if not other:
        pytest.skip("LOCALS_E2E_SWITCH is empty")
    before = get_json(f"http://127.0.0.1:{stack.server_port}/stats")["model"]
    t0 = time.time()
    p = stack.locals("start", "--model", other, "--port", stack.server_port, timeout=1800)
    took = time.time() - t0
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    import subprocess

    servers = subprocess.run(["pgrep", "-f", f"locals.server --port {stack.server_port}"], capture_output=True,
                             text=True).stdout.split()  # (this suite's servers only, not the user's own)
    now = get_json(f"http://127.0.0.1:{stack.server_port}/stats")["model"]
    report.note(f"switch {before} -> {now}: {took:.0f} s; locals servers running after: {len(servers)}")
    assert now != before, (before, now)
    assert len(servers) == 1, f"{len(servers)} locals servers run after a switch: {servers}"
    r = _send(stack, {"model": get_json(f"http://127.0.0.1:{stack.port}/v1/models")["data"][0]["id"], "stream": True,
                      "max_tokens": 10, "messages": [{"role": "user", "content": "Say hi."}]})
    assert r["first"] is not None
