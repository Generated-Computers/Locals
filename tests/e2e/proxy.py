"""A recording proxy between an agent tool and the Locals server: every request is forwarded as it is and
every response streamed back as it arrives, while the proxy notes what the tool experiences: the wait for
the first token, the longest pause between tokens, the HTTP status, error events inside a stream, the
usage the server reported, and whether the tool hung up before the reply ended."""

from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding",
       "upgrade", "content-length"}
CHAT = ("/chat/completions", "/api/chat", "/completions", "/api/generate", "/v1/messages")


class Record(dict):
    """One request as the tool saw it (times in seconds from the request's arrival)."""

    @property
    def ttft(self) -> Optional[float]:
        return self.get("first_token")


def _summary(body: bytes) -> dict:
    try:
        req = json.loads(body or b"{}")
    except ValueError:
        return {"bad_json": True}
    if not isinstance(req, dict):
        return {}
    msgs = req.get("messages") or []
    return {"model": req.get("model"), "stream": req.get("stream"), "messages": len(msgs), "tools": len(req.get("tools") or []), "request_bytes": len(body),
            "max_tokens": req.get("max_tokens") or req.get("max_completion_tokens"),
            "last_role": (msgs[-1] or {}).get("role") if msgs and isinstance(msgs[-1], dict) else None}


class _Scan:
    """Reads SSE (OpenAI) or NDJSON (Ollama) lines as they stream and notes token-bearing events."""

    def __init__(self, rec: Record, t0: float):
        self.rec, self.t0, self.buf, self.last_token = rec, t0, b"", None

    def feed(self, data: bytes, now: float) -> None:
        self.buf += data
        *lines, self.buf = self.buf.split(b"\n")
        for line in lines:
            self.line(line.strip(), now)

    def line(self, line: bytes, now: float) -> None:
        if not line:
            return
        if line.startswith(b":"):
            self.rec["keepalives"] = self.rec.get("keepalives", 0) + 1
            return
        if line.startswith(b"event:"):
            if b"error" in line:
                self.rec.setdefault("errors", []).append(line.decode(errors="replace")[:300])
            return
        if line.startswith(b"data:"):
            line = line[5:].strip()
            if line == b"[DONE]":
                self.rec["done"] = True
                return
        try:
            ev = json.loads(line)
        except ValueError:
            self.rec.setdefault("odd_lines", []).append(line.decode(errors="replace")[:200])
            return
        if not isinstance(ev, dict):
            return
        if ev.get("type") in ("message_start", "content_block_start", "content_block_delta", "content_block_stop",
                              "message_delta", "message_stop", "ping", "error"):
            return self._anthropic(ev, now)
        if ev.get("error"):
            self.rec.setdefault("errors", []).append(json.dumps(ev["error"])[:300])
        if ev.get("usage"):
            self.rec["usage"] = ev["usage"]
        token = False
        for c in ev.get("choices") or []:
            d = c.get("delta") or c.get("message") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls") or c.get("text"):
                token = True
            if d.get("tool_calls"):
                self.rec["tool_call_chunks"] = self.rec.get("tool_call_chunks", 0) + 1
            if c.get("finish_reason"):
                self.rec["finish"] = c["finish_reason"]
        msg = ev.get("message") or {}  # Ollama /api/chat
        if msg.get("content") or msg.get("thinking") or msg.get("tool_calls") or ev.get("response"):
            token = True
        if ev.get("done") is True:
            self.rec["done"] = True
            self.rec["finish"] = ev.get("done_reason") or self.rec.get("finish")
            if "prompt_eval_count" in ev:
                self.rec["usage"] = {"prompt_tokens": ev.get("prompt_eval_count"), "completion_tokens": ev.get("eval_count")}
        if token:
            t = now - self.t0
            self.rec["token_events"] = self.rec.get("token_events", 0) + 1
            if self.last_token is None:
                self.rec["first_token"] = t
            else:
                self.rec["max_gap"] = max(self.rec.get("max_gap", 0.0), t - self.last_token)
            self.last_token = t


    def _anthropic(self, ev: dict, now: float) -> None:
        """Anthropic's Messages API events (Claude Code): its usage in OpenAI's terms, for summarize()."""
        kind = ev["type"]
        if kind == "error":
            self.rec.setdefault("errors", []).append(json.dumps(ev.get("error"))[:300])
        elif kind == "ping":
            self.rec["keepalives"] = self.rec.get("keepalives", 0) + 1
        elif kind == "message_delta":
            u = ev.get("usage") or {}
            cached = u.get("cache_read_input_tokens") or 0
            self.rec["usage"] = {"prompt_tokens": (u.get("input_tokens") or 0) + cached, "completion_tokens": u.get("output_tokens"),
                                 "prompt_tokens_details": {"cached_tokens": cached}}
            self.rec["finish"] = (ev.get("delta") or {}).get("stop_reason")
        elif kind == "message_stop":
            self.rec["done"] = True
        elif kind in ("content_block_delta", "content_block_start") and (kind == "content_block_delta" or
                                                                         (ev.get("content_block") or {}).get("type") == "tool_use"):
            if (ev.get("content_block") or {}).get("type") == "tool_use" or (ev.get("delta") or {}).get("type") == "input_json_delta":
                self.rec["tool_call_chunks"] = self.rec.get("tool_call_chunks", 0) + 1
            t = now - self.t0
            self.rec["token_events"] = self.rec.get("token_events", 0) + 1
            if self.last_token is None:
                self.rec["first_token"] = t
            else:
                self.rec["max_gap"] = max(self.rec.get("max_gap", 0.0), t - self.last_token)
            self.last_token = t


class Proxy:
    """``Proxy(upstream_port, port).start()``; ``records`` holds one Record per request, newest last;
    ``label`` names the scenario that the next requests belong to."""

    def __init__(self, upstream_port: int, port: int, host: str = "127.0.0.1"):
        self.upstream_port, self.port, self.host = upstream_port, port, host
        self.records: List[Record] = []
        self.label = ""
        self.projects: dict = {}  # a project folder -> the run it belongs to (requests name their folder)
        self._lock = threading.Lock()
        self._srv: Optional[ThreadingHTTPServer] = None

    # -- the records -------------------------------------------------------------------------
    def mark(self, label: str) -> int:
        """Start a scenario: later records carry ``label``; returns the index to pass to ``since``."""
        with self._lock:
            self.label = label
            return len(self.records)

    def since(self, i: int, chat_only: bool = True) -> List[Record]:
        with self._lock:
            out = list(self.records[i:])
        return [r for r in out if not chat_only or r.get("chat")]

    # -- the server --------------------------------------------------------------------------
    def start(self) -> "Proxy":
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # quiet
                pass

            def do_GET(self):
                proxy._forward(self, "GET")

            def do_POST(self):
                proxy._forward(self, "POST")

            def do_OPTIONS(self):
                proxy._forward(self, "OPTIONS")

            def do_DELETE(self):
                proxy._forward(self, "DELETE")

            def do_HEAD(self):
                proxy._forward(self, "HEAD")

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):  # a client resetting an idle keep-alive connection
                import sys

                if not isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, TimeoutError)):
                    super().handle_error(request, client_address)

        self._srv = Server((self.host, self.port), Handler)
        threading.Thread(target=self._srv.serve_forever, daemon=True, name="e2e-proxy").start()
        return self

    def stop(self) -> None:
        if self._srv is not None:
            self._srv.shutdown()
            self._srv.server_close()
            self._srv = None

    def _forward(self, h: BaseHTTPRequestHandler, method: str) -> None:
        t0 = time.perf_counter()
        n = int(h.headers.get("Content-Length") or 0)
        body = h.rfile.read(n) if n else b""
        path = h.path
        rec = Record(path=path, method=method, label=self.label, t_wall=time.time(), chat=path.split("?")[0].endswith(CHAT))
        if rec["chat"] and method == "POST":
            rec.update(_summary(body))
            for folder, run in list(self.projects.items()):
                if folder.encode() in body:
                    rec["project"] = run
                    break
        with self._lock:
            self.records.append(rec)
        headers = {k: v for k, v in h.headers.items() if k.lower() not in HOP}
        up = http.client.HTTPConnection("127.0.0.1", self.upstream_port, timeout=3600)
        try:
            up.request(method, path, body=body if method in ("POST", "PUT") else None, headers=headers)
            resp = up.getresponse()
        except OSError as e:
            rec.update(status=502, upstream_error=str(e), total=time.perf_counter() - t0)
            try:
                msg = json.dumps({"error": {"message": f"e2e proxy: upstream unreachable: {e}"}}).encode()
                h.send_response(502)
                h.send_header("Content-Type", "application/json")
                h.send_header("Content-Length", str(len(msg)))
                h.end_headers()
                h.wfile.write(msg)
            except OSError:
                pass
            up.close()
            return
        rec["status"] = resp.status
        rec["headers_at"] = time.perf_counter() - t0
        length = resp.getheader("Content-Length")
        h.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in HOP:
                h.send_header(k, v)
        if length is not None:
            h.send_header("Content-Length", length)
        else:
            h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()
        scan = _Scan(rec, t0)
        whole = []
        try:
            while True:
                data = resp.read1(65536) if method != "HEAD" else b""
                now = time.perf_counter()
                if not data:
                    break
                if rec["chat"]:
                    scan.feed(data, now)
                    if length is not None:
                        whole.append(data)
                if length is not None:
                    h.wfile.write(data)
                else:
                    h.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                h.wfile.flush()
            if length is None:
                h.wfile.write(b"0\r\n\r\n")
                h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            rec["client_hung_up"] = True
            h.close_connection = True
        except (http.client.IncompleteRead, OSError) as e:
            rec["upstream_cut"] = str(e)[:200]
            h.close_connection = True
        finally:
            up.close()  # the server sees the tool's hang-up as its own socket closing
        if rec["chat"] and scan.buf.strip():
            scan.line(scan.buf.strip(), time.perf_counter())
        if whole and rec["chat"]:  # a reply in one piece (not streamed)
            try:
                d = json.loads(b"".join(whole))
                if isinstance(d, dict) and d.get("type") == "message":  # Anthropic's Messages API
                    u = d.get("usage") or {}
                    cached = u.get("cache_read_input_tokens") or 0
                    d = {"usage": {"prompt_tokens": (u.get("input_tokens") or 0) + cached, "completion_tokens": u.get("output_tokens"),
                                   "prompt_tokens_details": {"cached_tokens": cached}}, "choices": [{"finish_reason": d.get("stop_reason")}]}
                if isinstance(d, dict):
                    rec["usage"] = d.get("usage") or rec.get("usage")
                    if d.get("error"):
                        rec.setdefault("errors", []).append(json.dumps(d["error"])[:300])
                    ch = (d.get("choices") or [{}])[0]
                    rec["finish"] = ch.get("finish_reason") or d.get("done_reason")
                    rec.setdefault("first_token", time.perf_counter() - t0)
            except ValueError:
                pass
        rec["total"] = time.perf_counter() - t0
