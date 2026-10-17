"""A client that hangs up (Esc or Ctrl-C in the agent) stops its reply: the job the server submitted is
cancelled whether the hang-up shows as a failed write inside the stream or is found by the probe of the
socket, so the engine's row does not decode on to max_tokens for nobody. No GPU: the engine is a stand-in."""

import threading
import time
from types import SimpleNamespace

import pytest

from locals import server as srv_mod
from locals.server import Server


class Job(SimpleNamespace):
    def cancel(self):
        self.cancelled = True


class Svc:
    """Writes a token every 10 ms until its job is cancelled (or 2000 tokens)."""

    tokenizer = SimpleNamespace(decode=lambda toks: "".join(chr(97 + t % 26) for t in toks))

    def __init__(self):
        self.jobs = []

    def submit(self, prompt, max_tokens=16, on_token=None, **kw):
        job = Job(done=False, finish="length", tokens=[], text="", cached=0, t_submit=0.0, t_done=0.0, cancelled=False)
        self.jobs.append(job)

        def run():
            n = 0
            while not job.cancelled and n < 2000:
                on_token([n % 26])
                job.tokens.append(n % 26)
                n += 1
                time.sleep(0.01)
            job.finish = "cancelled" if job.cancelled else "length"
            job.text = self.tokenizer.decode(job.tokens)
            job.done = True

        threading.Thread(target=run, daemon=True).start()
        return job


def _server():
    srv = Server.__new__(Server)
    srv._active, srv.log, srv.max_calls = {}, (lambda s: None), 6
    srv.per_engine = {"locals": {k: 0 for k in ("requests", "prompt_tokens", "cached_tokens", "completion_tokens",
                                                 "forced_tokens", "passes", "decode_s", "tool_calls")}}
    return srv


def _p(svc, **kw):
    return {"engine": "locals", "svc": svc, "prompt": [1, 2, 3], "max_tokens": 2000, "model": "locals/m", "stop": [],
            "temp": 0.0, "ignore_eos": False, "include_usage": False, "regex": None, "schema": None, "tools": None,
            "fmt": None, "thinking": False, "checkpoints": None, **kw}


def _until(cond, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_a_failed_write_cancels_the_reply():
    svc, srv = Svc(), _server()
    sent = []

    def write(b):
        if len(sent) >= 3:
            raise BrokenPipeError("the agent hung up")
        sent.append(b)

    assert srv.stream({"messages": []}, write, _p(svc)) is None
    assert _until(lambda: svc.jobs and svc.jobs[0].done), "the job was never cancelled"
    assert svc.jobs[0].cancelled and len(svc.jobs[0].tokens) < 500
    assert not srv._active


def test_the_probe_finds_a_quiet_stream_that_hung_up(monkeypatch):
    """Nothing is written while a long prompt is read (or a tool call is held back): the socket probe, not
    the next keepalive 15 s later, stops the reply."""
    svc, srv = Svc(), _server()
    gone_at = time.time() + 0.3
    t0 = time.time()
    srv.stream({"messages": []}, lambda b: None, _p(svc, alive=lambda: time.time() < gone_at))
    assert svc.jobs[0].cancelled
    assert time.time() - t0 < 3, "the hang-up was found late"


def test_a_failed_text_completion_write_cancels_too():
    svc, srv = Svc(), _server()

    def write(b):
        raise ConnectionResetError

    srv.text_completion({}, _p(svc), write=write)
    assert _until(lambda: svc.jobs[0].done) and svc.jobs[0].cancelled


@pytest.mark.parametrize("ready", [True, False])
def test_alive_reads_a_closed_socket_as_gone(ready):
    import socket

    a, b = socket.socketpair()
    if ready:
        b.sendall(b"x")  # a pipelined byte: still connected
        assert srv_mod._alive_socket(a)
    else:
        assert srv_mod._alive_socket(a)
        b.close()
        assert not srv_mod._alive_socket(a)
    a.close()
    b.close()
