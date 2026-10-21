"""What a request may do: web pages from other sites are refused (any site the user visits could
otherwise use the model and read the replies), the body is a bounded JSON object, a template
cannot be swapped in, and what the server would silently ignore is refused."""

import http.client
import json
import re
import threading

import pytest

from locals import server as srv


def test_origins():
    assert srv._origin_ok("http://localhost:3000") and srv._origin_ok("http://127.0.0.1") and srv._origin_ok("vscode-webview://abc")
    assert not srv._origin_ok("https://example.com") and not srv._origin_ok("null") and not srv._origin_ok("http://localhost.evil.com")
    assert srv._origin_ok("chrome-extension://abcdef") and srv._origin_ok("http://app.localhost:3000")


def test_origins_from_the_environment(monkeypatch):
    monkeypatch.setenv("LOCALS_ORIGINS", "https://my.app")
    assert srv._origin_ok("https://my.app") and not srv._origin_ok("https://other.app")
    monkeypatch.setenv("LOCALS_ORIGINS", "*")
    assert srv._origin_ok("https://other.app")


def test_hosts():
    assert srv._host_ok("127.0.0.1:8321", "127.0.0.1") and srv._host_ok("localhost", "127.0.0.1") and srv._host_ok(None, "127.0.0.1")
    assert srv._host_ok("[::1]:8321", "127.0.0.1")
    assert not srv._host_ok("evil.com:8321", "127.0.0.1")  # a rebound domain
    assert srv._host_ok("host.docker.internal:11434", "127.0.0.1")  # Open WebUI in Docker
    assert srv._host_ok("app.localhost:8321", "127.0.0.1") and srv._host_ok("192.168.1.5:8321", "127.0.0.1")
    assert srv._host_ok("0.0.0.0:8321", "127.0.0.1") and not srv._host_ok("[::1", "127.0.0.1")
    assert srv._host_ok("my-mac.local:8321", "0.0.0.0")  # served to the network on purpose


def test_template_switches():
    kw = {"enable_thinking": True, "reasoning_effort": "low", "chat_template": "{% for a in range(9) %}{% endfor %}",
          "documents": [{"x": 1}], "tokenizer_kwargs": {}, "x": "y" * 500}
    assert srv._template_switches(kw) == {"reasoning_effort": "low"}
    assert srv._template_switches(["not", "a", "dict"]) == {}


def test_unsupported_parameters():
    srv._unsupported({"n": 1, "logprobs": False})
    for bad in ({"n": 2}, {"logprobs": True}, {"top_logprobs": 3}):
        with pytest.raises(ValueError):
            srv._unsupported(bad)


@pytest.fixture()
def http_server():
    class Stub:
        requests = errors = 0

        def log(self, s):
            pass

        def prepare(self, req):
            return {}

        def complete(self, req, p):
            return {"ok": True}

    httpd = srv._HTTPServer(("127.0.0.1", 0), srv.make_handler(Stub()))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()


def _post(port, body, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json", **(headers or {})})
    r = c.getresponse()
    return r.status, r.read(), dict(r.getheaders())


def test_requests_from_pages(http_server):
    ok = json.dumps({"messages": [{"role": "user", "content": "hi"}]})
    status, _, h = _post(http_server, ok, {"Origin": "https://example.com"})
    assert status == 403 and "Access-Control-Allow-Origin" not in h
    status, _, h = _post(http_server, ok, {"Origin": "http://localhost:5173"})
    assert status == 200 and h["Access-Control-Allow-Origin"] == "http://localhost:5173"
    status, _, h = _post(http_server, ok)  # an agent, an SDK, curl: no Origin
    assert status == 200 and "Access-Control-Allow-Origin" not in h
    status, _, _ = _post(http_server, ok, {"Host": "rebound.example.com:8321"})
    assert status == 403


def test_bodies_that_are_not_requests(http_server):
    for body in ("[]", "null", '"x"', "[" * 5000 + "]" * 5000, "{"):
        status, _, _ = _post(http_server, body)
        assert status == 400, body[:20]


def test_a_preflight_names_the_headers_asked_for(http_server):
    """"*" never covers Authorization (the Fetch standard): a page using the OpenAI SDK failed its preflight."""
    c = http.client.HTTPConnection("127.0.0.1", http_server, timeout=10)
    c.request("OPTIONS", "/v1/chat/completions", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST",
                                                          "Access-Control-Request-Headers": "authorization,content-type"})
    r = c.getresponse()
    r.read()
    assert r.getheader("Access-Control-Allow-Headers") == "authorization,content-type"


def test_a_chunked_body_is_read(http_server):
    """A body sent in chunks (no Content-Length) was read as empty, got a 400, and its bytes stayed on the connection."""
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    c = http.client.HTTPConnection("127.0.0.1", http_server, timeout=10)
    c.request("POST", "/v1/chat/completions", iter([body[:10], body[10:]]), {"Content-Type": "application/json"},
              encode_chunked=True)
    r = c.getresponse()
    assert r.status == 200 and json.loads(r.read()) == {"ok": True}
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})  # the same connection after it
    r = c.getresponse()
    assert r.status == 200 and json.loads(r.read()) == {"ok": True}


def test_errors_come_in_each_apis_own_shape(http_server):
    """An invalid body on /v1/messages or /api/chat was answered in OpenAI's error shape, which Anthropic's SDKs
    and Ollama clients do not read."""
    for path, check in (("/v1/messages", lambda d: d["type"] == "error" and d["error"]["type"] == "invalid_request_error"),
                        ("/api/chat", lambda d: isinstance(d["error"], str)),
                        ("/v1/chat/completions", lambda d: d["error"]["type"] == "invalid_request_error")):
        c = http.client.HTTPConnection("127.0.0.1", http_server, timeout=10)
        c.request("POST", path, "{", {"Content-Type": "application/json"})
        r = c.getresponse()
        assert r.status == 400 and check(json.loads(r.read())), path


def test_requests_of_the_wrong_shape_are_told_what_is_wrong():
    """A malformed request failed inside the server with Python's words ("AttributeError: 'str' object has no
    attribute 'get'"); each shape gets its own sentence now, and a lone surrogate (a JavaScript client that cut a
    string inside an emoji) is replaced instead of failing the request."""
    for req, said in (({"messages": ["hi"]}, "messages[0] must be an object"),
                      ({"messages": [{"role": "user", "content": 5}]}, "messages[0].content must be a string"),
                      ({"messages": [{"role": "assistant", "tool_calls": "x"}]}, "tool_calls must be a list"),
                      ({"messages": [], "tools": "x"}, "tools must be a list"),
                      ({"messages": [], "stop": 5}, "stop must be a string"),
                      ({"messages": [], "model": 5}, "model must be a string")):
        with pytest.raises(ValueError, match=re.escape(said)):
            srv._check_shape(req)
    assert srv._clean_text({"a": ["x\ud800y", 3]}) == {"a": ["x\ufffdy", 3]}
    assert srv._bad(ValueError("n must be 1")) == "n must be 1"
    assert srv._bad(KeyError("x")).startswith("invalid request (KeyError")


def test_wrong_methods_and_short_bodies_get_answers(http_server):
    import socket

    c = http.client.HTTPConnection("127.0.0.1", http_server, timeout=10)
    c.request("PUT", "/v1/chat/completions", "{}")
    r = c.getresponse()
    assert r.status == 405 and "error" in json.loads(r.read())
    s = socket.create_connection(("127.0.0.1", http_server), timeout=10)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 1000\r\n\r\n{}")
    s.shutdown(socket.SHUT_WR)  # the client stops sending: the body ends short
    assert b" 400 " in s.recv(200)


def _raw(port, data: bytes, wait: float = 3.0) -> bytes:
    import socket

    s = socket.create_connection(("127.0.0.1", port), timeout=wait)
    s.sendall(data)
    out = b""
    try:
        while True:
            d = s.recv(65536)
            if not d:
                break
            out += d
    except (socket.timeout, ConnectionResetError):
        pass
    s.close()
    return out


def test_chunk_sizes_are_hex_digits_only(http_server):
    """int() also took "-1", which read to the end of the connection past MAX_BODY (300 MB in memory), and "-2"
    ended the request without an answer."""
    head = b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\n"
    for size in (b"-1", b"-2", b"0x10", b"+10", b"1_0"):
        out = _raw(http_server, head + size + b"\r\n" + b"x" * 16 + b"\r\n0\r\n\r\n")
        assert b" 413 " in out.split(b"\r\n")[0] and b"Connection: close" in out, size


def test_a_body_has_one_deadline_not_one_per_read(http_server, monkeypatch):
    import socket
    import time

    monkeypatch.setattr(srv, "BODY_TIMEOUT", 1.5)
    s = socket.create_connection(("127.0.0.1", http_server), timeout=10)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 40\r\n\r\n")
    t0, out = time.time(), b""
    try:
        for _ in range(40):  # a byte every 0.2 s: each read is quick, the whole body is not
            s.sendall(b"x")
            time.sleep(0.2)
    except OSError:
        pass
    try:
        out = s.recv(200)
    except OSError:
        pass
    assert b" 408 " in out and time.time() - t0 < 6


def test_an_answer_before_the_body_closes_the_connection(http_server):
    """A 403 left the request's body on the connection, read as the next request (a GET after it got a 501)."""
    body = b'{"messages": []}'
    out = _raw(http_server, b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://evil.example\r\n"
                            b"Content-Length: %d\r\n\r\n" % len(body) + body + b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
    assert out.startswith(b"HTTP/1.1 403") and b"Connection: close" in out and b" 501 " not in out


def test_bad_inputs_get_an_answer(http_server):
    """Inputs that ended the request with no answer and a traceback: a URL that does not parse, JSON nested past
    Python's recursion, an infinite max_tokens (a 500 that SDKs retry)."""
    out = _raw(http_server, b"GET http://[x/health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
    assert out.split(b"\r\n")[0].startswith(b"HTTP/1.1 4")
    status, body, _ = _post(http_server, '{"messages": [{"role": "user", "content": "hi"}], "x": ' + "[" * 3000 + "]" * 3000 + "}")
    assert status == 400 and b"nested too deeply" in body
    assert srv._whole(5.0, "n") == 5 and srv._whole(None, "n") is None
    for bad in (float("inf"), float("nan"), True, "5"):
        with pytest.raises(ValueError, match="whole number"):
            srv._whole(bad, "max_tokens")
    assert srv._printable("/x\x1b[31m") == "/x\\x1b[31m"
