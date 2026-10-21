"""`locals chat` and `locals generate` on the server (locals/client.py), against stub servers: what Ctrl-C, a reply cut
at the token limit and an error mid-thought do."""

import argparse
import http.server
import io
import json
import os
import signal
import threading
import time
from contextlib import redirect_stdout

import pytest

from locals import client


def test_ctrl_c_during_alternatives_closes_their_connections():
    """Ctrl-C during /n alternatives (or generate --n) left the command waiting until every reply was done, and
    the server decoded them all: the connections stayed open."""
    closed = []

    class Slow(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.connection.settimeout(10)
            try:
                gone = self.connection.recv(1) == b""  # the client closed its side: a server would stop decoding
            except OSError:
                gone = True
            closed.append(gone)

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGINT))
    timer.start()
    t0 = time.time()
    try:
        with pytest.raises(KeyboardInterrupt):
            client.several(base, {"messages": []}, 3)
        assert time.time() - t0 < 3
        deadline = time.time() + 3
        while len(closed) < 3 and time.time() < deadline:
            time.sleep(0.05)
        assert closed == [True, True, True]
    finally:
        timer.cancel()
        httpd.shutdown()


def _args(**kw):
    return argparse.Namespace(**{"model": None, "max_tokens": 5, "temp": 0.0, "n": 1, "regex": None, "stats": False,
                                 "seed": None, **kw})


def test_a_constrained_reply_cut_at_the_limit_is_not_a_success(monkeypatch, capsys):
    """A JSON object cut at --max-tokens printed as if it were whole, and the command exited 0."""
    cut = {"choices": [{"message": {"content": '{"name": "a'}, "finish_reason": "length"}], "usage": {"completion_tokens": 5}}
    monkeypatch.setattr(client, "connect", lambda m: ("http://x", "locals/m"))
    monkeypatch.setattr(client, "complete", lambda base, body: cut)
    with pytest.raises(SystemExit, match="cut at --max-tokens 5"):
        client.generate(_args(), "p", {"type": "object"})
    assert '{"name": "a' in capsys.readouterr().out
    client.generate(_args(), "p", None)  # plain text: printed, with a note
    assert "cut at --max-tokens 5" in capsys.readouterr().err


def test_a_seed_goes_to_the_server_and_each_sample_gets_its_own(monkeypatch):
    """--seed was not sent (the server ignored seeds then); with --n, the same seed would give n copies of one reply."""
    import json as _json

    whole = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}
    sent = []
    monkeypatch.setattr(client, "connect", lambda m: ("http://x", "locals/m"))
    monkeypatch.setattr(client, "complete", lambda base, body: sent.append(body) or whole)
    client.generate(_args(seed=5), "p", None)
    assert sent[-1]["seed"] == 5
    client.generate(_args(), "p", None)
    assert "seed" not in sent[-1]

    class Conn:
        def __init__(self, *a, **k):
            self.sock = None

        def request(self, method, path, data, headers):
            sent.append(_json.loads(data))

        def getresponse(self):
            class R:
                status = 200

                def read(self):
                    return _json.dumps(whole).encode()

            return R()

        def close(self):
            pass

    monkeypatch.setattr(client.http.client, "HTTPConnection", Conn)
    del sent[:]
    client.several("http://127.0.0.1:1", {"messages": [], "seed": 10}, 3)
    assert sorted(b["seed"] for b in sent) == [10, 11, 12]


class _Reply:
    def __init__(self, events):
        self.lines = [f"data: {json.dumps(e)}\n".encode() for e in events] + [b"data: [DONE]\n"]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.lines)


def _delta(kind, text, finish=None):
    return {"choices": [{"delta": {kind: text}, "finish_reason": finish}]}


def test_an_error_mid_thought_ends_the_dim_text_and_is_not_called_a_stop(monkeypatch):
    """The error and every later line stayed dim, and chat then said "(stopped; ...)" as if Ctrl-C had been pressed."""
    monkeypatch.setattr(client, "_post", lambda url, body: _Reply([_delta("reasoning_content", "hmm"),
                                                                   {"error": {"message": "out of memory"}}]))
    out = io.StringIO()
    with redirect_stdout(out):
        text, _, _, end = client._stream("http://x", {})
    assert text is None and end == "failed"
    shown = out.getvalue()
    assert shown.index(client.END) < shown.index("(error: out of memory)")


def test_thinking_cut_at_the_limit_is_said(monkeypatch):
    monkeypatch.setattr(client, "_post", lambda url, body: _Reply([_delta("reasoning_content", "hmm"),
                                                                   _delta("reasoning_content", " more", "length")]))
    with redirect_stdout(io.StringIO()):
        text, _, _, end = client._stream("http://x", {})
    assert text == "" and end == "length"


def test_a_prompt_file_that_is_not_utf8_is_named(tmp_path):
    """It was reported as the schema's fault: "None is not valid JSON"."""
    from locals import cli

    (tmp_path / "p.txt").write_bytes(b"caf\xe9")
    with pytest.raises(SystemExit, match="p.txt is not UTF-8 text"):
        cli.main(["--prompt", f"@{tmp_path / 'p.txt'}"])
