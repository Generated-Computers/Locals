"""What a request may do: web pages from other sites are refused (any site the user visits could
otherwise use the model and read the replies), the body is a bounded JSON object, a template
cannot be swapped in, and what the server would silently ignore is refused."""

import http.client
import json
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
