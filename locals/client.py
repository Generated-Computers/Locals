"""`locals chat` and `locals generate` on the background server.

Both used to load the model in their own process: after `locals code` (which leaves the server
running) that put a second copy of the model in memory, and tuned the kernels a second time.
Here they start the background server if it is not running (as `locals code` does) and talk to
it: one model in memory, and its prompt cache keeps a conversation between turns.
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from argparse import Namespace
from typing import List, Optional


def connect(model: Optional[str]) -> tuple:
    """(base URL, model id for requests): the background server for ``model`` (else the model used
    last, else this Mac's default), started or switched as `locals code` does."""
    from . import app
    from .defaults import DEFAULT_MODEL, model_id

    saved = app._load_saved()
    port = int(os.environ.get("LOCALS_PORT") or saved.get("port") or 8321)  # (locals --port N chat)
    chosen = model_id(model) if model else (saved.get("model") or DEFAULT_MODEL)
    args = Namespace(port=port, model=chosen, engine=None, engines=None)
    app.fetch_model(chosen)
    app.ensure_server(args)
    app.remember(model=chosen, port=port)
    return app.base_url(port), f"locals/{app.model_short(chosen)}"


NOT_ANSWERING = ("the server at {base} is not answering (another command may have stopped or restarted it): "
                 "locals status shows it, locals start starts it again")


def _post(url: str, body: dict, timeout: float = 3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read())["error"]["message"]
        except Exception:  # noqa: BLE001
            msg = str(e)
        raise ValueError(msg) from None
    except (urllib.error.URLError, OSError, http.client.HTTPException):  # (a traceback ended locals chat here)
        raise ValueError(NOT_ANSWERING.format(base=url.split("/v1/")[0])) from None


def complete(base: str, body: dict) -> dict:
    with _post(base + "/v1/chat/completions", dict(body, stream=False)) as r:
        return json.load(r)


def _format(schema: Optional[dict], regex: Optional[str]) -> dict:
    if schema is not None:
        return {"response_format": {"type": "json_schema", "json_schema": {"name": "output", "schema": schema}}}
    if regex is not None:
        return {"response_format": {"type": "regex", "regex": regex}}
    return {}


def several(base: str, body: dict, n: int) -> List[dict]:
    """n requests at once: they decode together in the server's batch. Ctrl-C closes their connections (the server
    stops decoding them) and is raised again: it left the command waiting for every reply to finish."""
    import socket
    from urllib.parse import urlparse

    u = urlparse(base)
    data = json.dumps(dict(body, stream=False)).encode()
    out: List[Optional[dict]] = [None] * n
    errors: List[str] = []
    stopped = threading.Event()
    conns = [http.client.HTTPConnection(u.hostname, u.port, timeout=3600) for _ in range(n)]

    def one(i):
        c = conns[i]
        try:
            if stopped.is_set():
                return
            c.request("POST", "/v1/chat/completions", data, {"Content-Type": "application/json"})
            r = c.getresponse()
            raw = r.read()
            if r.status != 200:
                try:
                    errors.append(json.loads(raw)["error"]["message"])
                except Exception:  # noqa: BLE001
                    errors.append(f"HTTP {r.status}")
                return
            out[i] = json.loads(raw)
        except (OSError, http.client.HTTPException, ValueError):
            if not stopped.is_set():
                errors.append(NOT_ANSWERING.format(base=base))
        finally:
            c.close()

    threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        stopped.set()
        for c in conns:
            try:
                if c.sock is not None:
                    c.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        raise
    if errors:
        raise ValueError(errors[0])
    return [o for o in out if o is not None]


# --- locals generate ----------------------------------------------------------------------------

def generate(args, prompt: str, schema: Optional[dict]) -> None:
    base, model = connect(args.model)
    if args.seed is not None:  # (the server samples every request's rows from one generator: no seed of its own)
        print("locals generate: --seed applies with --local only; the server's replies are not seeded", file=sys.stderr)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens,
            "temperature": args.temp if args.n == 1 else (args.temp or 0.8), **_format(schema, args.regex)}
    t0 = time.time()
    try:
        replies = several(base, body, args.n) if args.n > 1 else [complete(base, body)]
    except ValueError as e:
        sys.exit(f"locals generate: {e}")
    cut = []
    for i, r in enumerate(replies):
        if args.n > 1:
            print(f"--- sample {i}")
        print(r["choices"][0]["message"].get("content") or "")
        if r["choices"][0].get("finish_reason") == "length":
            cut.append(i)
    if args.stats:
        out = sum(r["usage"]["completion_tokens"] for r in replies)
        print(f"[{out} tokens in {time.time() - t0:.1f}s on {model} at {base}]", file=sys.stderr)
    if cut:  # (a JSON object cut at the limit printed as if whole, and the command said it went well)
        what = "the reply was" if args.n == 1 else f"sample(s) {', '.join(map(str, cut))} were"
        note = f"locals generate: {what} cut at --max-tokens {args.max_tokens}"
        if schema is not None or args.regex:
            sys.exit(note + ", before the end the schema or regex asks for: raise --max-tokens")
        print(note, file=sys.stderr)


# --- locals chat --------------------------------------------------------------------------------

DIM, END = "\x1b[2m", "\x1b[0m"


def chat(args) -> None:
    import os

    base, model = connect(args.model)
    if args.think:
        model += "-thinking"
    messages: List[dict] = [{"role": "system", "content": args.system}] if args.system else []
    schema, n_alt, show_stats = None, 1, not args.no_stats
    print(f"chat with {model} on {base}. /reset, /n <N>, /json <schema>, /stats, /quit", file=sys.stderr)
    while True:
        try:
            msg = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/reset":
            messages = messages[:1] if args.system else []
            print("(new conversation)")
            continue
        if msg == "/stats":
            show_stats = not show_stats
            continue
        if msg.startswith("/n"):
            try:
                n_alt = max(1, int(msg[2:].strip() or 1))
            except ValueError:
                print("(/n takes a number: /n 3)")
                continue
            print(f"(next reply: {n_alt} alternatives at once)")
            continue
        if msg.startswith("/json"):
            arg = msg[5:].strip()
            try:
                schema = json.load(open(arg)) if os.path.exists(arg) else json.loads(arg)
            except Exception as e:  # noqa: BLE001
                schema = None
                print(f"(not a usable schema: {type(e).__name__}: {e})")
                continue
            print("(next reply is constrained to the schema)")
            continue

        turn = messages + [{"role": "user", "content": msg}]
        body = {"model": model, "messages": turn, "max_tokens": args.max_tokens, "temperature": args.temp,
                **_format(schema, None)}
        t0 = time.time()
        if n_alt > 1:
            try:
                replies = several(base, dict(body, temperature=args.temp or 0.8), n_alt)
            except ValueError as e:
                print(f"(error: {e})")
                continue
            except KeyboardInterrupt:
                print("\n(stopped; this turn is not kept)")
                n_alt, schema = 1, None
                continue
            texts = [r["choices"][0]["message"].get("content") or "" for r in replies]
            for i, t in enumerate(texts):
                print(f"\n--- reply {i + 1}\n{t}")
            if show_stats:
                out = sum(r["usage"]["completion_tokens"] for r in replies)
                print(f"[{n_alt} replies, {out} tokens in {time.time() - t0:.1f}s]", file=sys.stderr, flush=True)
            messages = turn + [{"role": "assistant", "content": texts[0]}]  # the conversation goes on from reply 1
            n_alt, schema = 1, None
            continue
        text, usage, first, end = _stream(base, dict(body, stream=True, stream_options={"include_usage": True}))
        schema = None
        if end == "length":  # (thinking that took the whole budget ended with no answer and no word why)
            print(f"\n(cut at --max-tokens {args.max_tokens}" + ("; the thinking took it all, and there is no answer"
                                                                if not text else "") + ")", file=sys.stderr)
        if text is None or (end == "length" and not text):
            print("\n(stopped; this turn is not kept)" if end == "stopped" else "(this turn is not kept)")
            continue
        messages = turn + [{"role": "assistant", "content": text}]
        if show_stats and usage:
            n = usage.get("completion_tokens", 0)
            took = time.time() - (first or t0)
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            print(f"\n[{n} tokens, {n / took if took > 0 else 0:.1f} tok/s, first token after {(first or t0) - t0:.2f}s, "
                  f"context {usage.get('prompt_tokens', 0) + n} tokens ({cached} read from memory)]", file=sys.stderr, flush=True)


def _stream(base: str, body: dict):
    """Print a streamed reply as it comes: (its text, usage, when the first token came, how it ended). The end is
    "stop", "length" (cut at max_tokens), "stopped" (Ctrl-C closed the connection: the server stops decoding it;
    no text) or "failed" (an error, printed; no text)."""
    text, usage, first, thinking, end = [], None, None, False, "stop"
    try:
        r = _post(base + "/v1/chat/completions", body)
    except ValueError as e:
        print(f"(error: {e})")
        return None, None, None, "failed"
    try:
        with r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                if d.get("error"):
                    print((END if thinking else "") + f"\n(error: {d['error'].get('message')})")  # (it stayed dim)
                    return None, None, None, "failed"
                if d.get("usage") and not d.get("choices"):
                    usage = d["usage"]
                for c in d.get("choices") or []:
                    if c.get("finish_reason") == "length":
                        end = "length"
                    delta = c.get("delta") or {}
                    for kind in ("reasoning_content", "content"):
                        piece = delta.get(kind)
                        if not piece:
                            continue
                        first = first or time.time()
                        if kind == "reasoning_content" and not thinking:
                            print(DIM, end="")
                            thinking = True
                        elif kind == "content" and thinking:
                            print(END + "\n", end="")
                            thinking = False
                        print(piece, end="", flush=True)
                        if kind == "content":
                            text.append(piece)
    except KeyboardInterrupt:
        if thinking:
            print(END, end="")
        return None, None, None, "stopped"
    except (OSError, http.client.HTTPException, ValueError):  # the server went away mid-reply
        if thinking:
            print(END, end="")
        print(f"\n(error: {NOT_ANSWERING.format(base=base)})")
        return None, None, None, "failed"
    if thinking:
        print(END, end="")
    return "".join(text), usage, first, end
