"""A user stops an agent: Esc or Ctrl-C while it writes, or while it still waits for the first token of a
long prompt. The tool hangs up; the server must notice, stop that request and give its place (and the GPU)
to the requests still running and the next one."""

import http.client
import json
import os
import signal
import socket
import subprocess
import threading
import time

from conftest import LOCALS, get_json
from tasks import TASKS



def _model(stack) -> str:
    return get_json(f"http://127.0.0.1:{stack.server_port}/v1/models")["data"][0]["id"]


def _rows(stack) -> int:
    """Rows in the engine's batch (its count is refreshed at every step, so it is current while rows run; the
    HTTP handlers' own list empties when a handler returns, even if the generation goes on unread)."""
    return max(e.get("active") or 0 for e in stack.stats()["engines"].values())


def _out_tokens(stack) -> int:
    return sum(e.get("tokens_out") or 0 for e in stack.stats()["engines"].values())


def _idle(stack) -> bool:
    """No handler answering and the engine writing no more tokens (over half a second)."""
    a = _out_tokens(stack)
    time.sleep(0.5)
    return not stack.stats()["active"] and _out_tokens(stack) == a


def _wait(cond, timeout, step=0.2):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return time.time() - t0
        time.sleep(step)
    return None


class Stream(threading.Thread):
    """One streamed chat request on its own connection, to the server itself (through the proxy a hang-up
    would reach the server later); ``hang_up()`` closes the socket as a tool does."""

    def __init__(self, port, body):
        super().__init__(daemon=True)
        self.port, self.body = port, body
        self.tokens, self.finish, self.error, self.first = 0, None, None, None
        self.conn = None
        self.t0 = None

    def run(self):
        self.t0 = time.perf_counter()
        try:
            self.conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
            self.conn.request("POST", "/v1/chat/completions", body=json.dumps(self.body), headers={"Content-Type": "application/json"})
            r = self.conn.getresponse()
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                for c in ev.get("choices") or []:
                    if (c.get("delta") or {}).get("content"):
                        self.tokens += 1
                        self.first = self.first or time.perf_counter() - self.t0
                    self.finish = c.get("finish_reason") or self.finish
        except (OSError, ValueError, http.client.HTTPException) as e:
            self.error = e

    def hang_up(self):
        try:
            self.conn.sock.shutdown(socket.SHUT_RDWR)
        except (OSError, AttributeError):
            pass
        self.conn.close()


def _story(stack, n=1200):
    return {"model": _model(stack), "stream": True, "max_tokens": n, "ignore_eos": True, "temperature": 0.7,
            "messages": [{"role": "user", "content": "Write a very long story about a lighthouse keeper."}]}


def test_streams_cut_off_free_their_places(stack, report):
    streams = [Stream(stack.server_port, _story(stack)) for _ in range(6)]
    for s in streams:
        s.start()
    assert _wait(lambda: all(s.tokens > 5 for s in streams), 120), "the streams did not start"
    for s in streams[:3]:
        s.hang_up()
    freed = _wait(lambda: _rows(stack) <= 3, 10)
    report.note(f"three of six streams cut off: their places freed in {freed if freed is None else round(freed, 2)} s")
    assert freed is not None, f"{_rows(stack)} rows still decoding 10 s after three of six streams hung up"
    for s in streams[3:]:
        s.join(timeout=300)
        assert s.finish == "length" and s.error is None, (s.finish, s.error)
    nxt = Stream(stack.server_port, {**_story(stack, 20), "ignore_eos": False})
    nxt.start()
    nxt.join(timeout=60)
    assert nxt.first is not None and nxt.first < 3, f"the next request's first token after {nxt.first} s"


def test_hang_up_while_a_long_prompt_is_read(stack, report):
    """Esc while the agent waits for the first token of a 20k-token prompt: the reading must stop soon,
    not after the whole prompt, so that the next request does not wait for it."""
    filler = " ".join(f"line {i}: the quick brown fox jumps over the lazy dog." for i in range(1700))
    long = Stream(stack.server_port, {"model": _model(stack), "stream": True, "max_tokens": 50,
                               "messages": [{"role": "user", "content": filler + "\n\nSummarize this in one line."}]})
    long.start()
    assert _wait(lambda: stack.stats()["active"], 10), "the long request never became active"
    time.sleep(1.5)
    long.hang_up()
    t0 = time.perf_counter()
    nxt = Stream(stack.server_port, {"model": _model(stack), "stream": True, "max_tokens": 10,
                              "messages": [{"role": "user", "content": "Say hello."}]})
    nxt.start()
    nxt.join(timeout=300)
    waited = time.perf_counter() - t0
    gone = _wait(lambda: _idle(stack), 60)
    report.note(f"hang-up during a 20k-token prompt: the next request answered after {waited:.1f} s "
                f"(first token {nxt.first and round(nxt.first, 1)} s); the long one gone after {gone and round(gone, 1)} s more")
    assert nxt.first is not None
    assert nxt.first < 8, f"the next request waited {nxt.first:.1f} s for its first token: the abandoned prompt was read to the end"


def test_ctrl_c_on_a_writing_agent(stack, report):
    d = TASKS["feature"].copy_to(stack.runs)
    env = dict(stack.env, PWD=str(d))
    mark, off = stack.proxy.mark("ctrl-c"), stack.log_offset()
    task = "Write a file essay.md with a 3000-word essay on the history of computing, in one go."
    p = subprocess.Popen([*LOCALS, "code", str(d), "--run", task, "--port", str(stack.port)], cwd=d, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, start_new_session=True)
    writing = _wait(lambda: any((r.get("token_events") or 0) > 20 and r.get("total") is None for r in stack.proxy.since(mark)), 300)
    assert writing is not None, "the agent never started writing"
    os.killpg(p.pid, signal.SIGINT)  # Ctrl-C in the terminal: the whole foreground group
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        raise AssertionError("locals code did not stop on Ctrl-C within 30 s")
    gone = _wait(lambda: _idle(stack), 15)
    log = stack.log_since(off)
    report.note(f"Ctrl-C on a writing agent: requests gone after {gone and round(gone, 1)} s; "
                f"server saw the hang-up: {'client disconnected' in log}")
    assert gone is not None, "the engine still writes tokens 15 s after Ctrl-C"
    assert not stack.problems(off), stack.problems(off)
