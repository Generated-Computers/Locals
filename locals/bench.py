"""Speed of a chat server with several requests at once, measured the way the README's numbers are.

Each request is one of sixteen short questions, answered with a fixed number of tokens (the end of the
text is ignored where the server allows it, so every request does the same work), streamed. The result
for ``n`` requests at once is the number of tokens written per second by all of them together while
they write: all their tokens over the time from the first token of any of them to the last token of any,
so reading the prompts at the start is left out (as in the README's numbers), and a server that
answers the requests one after another, rather than together, is measured as such. It works with any
OpenAI-compatible server (Locals, stock mlx-lm behind Locals' server, Ollama, others): only the
request body differs.
"""

from __future__ import annotations

import http.client
import json
import statistics
import threading
import time
import urllib.parse
from typing import Dict, List, Optional

QUESTIONS = [
    "Explain how a refrigerator works.", "Write a short story about a lighthouse keeper.",
    "What are the main causes of inflation?", "Describe the water cycle for a ten-year-old.",
    "Give me tips for learning a new language.", "How do vaccines train the immune system?",
    "Summarize the history of the printing press.", "What makes a good software design?",
    "Explain black holes simply.", "How does compound interest work?",
    "Describe a day in the life of a honeybee.", "What is the difference between weather and climate?",
    "How do airplanes stay in the air?", "Write a poem about autumn rain.",
    "Why is sleep important?", "Explain recursion with an example.",
]


def _one(base: str, body: dict, out: List[Optional[dict]], i: int, start: threading.Event, timeout: float) -> None:
    """Send one streamed chat request; record when its first and last tokens came and how many it wrote."""
    u = urllib.parse.urlsplit(base)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    path = (u.path.rstrip("/") or "") + "/chat/completions"
    start.wait()
    t0 = time.perf_counter()
    first = last = None
    pieces = 0
    usage = None
    try:
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        r = conn.getresponse()
        if r.status != 200:
            out[i] = {"error": f"HTTP {r.status}: {r.read()[:300].decode(errors='replace')}"}
            return
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                if any(delta.get(k) for k in ("content", "reasoning_content", "reasoning")):
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    pieces += 1
    except Exception as e:  # noqa: BLE001
        out[i] = {"error": f"{type(e).__name__}: {e}"}
        return
    finally:
        conn.close()
    tokens = (usage or {}).get("completion_tokens") or pieces
    out[i] = {"t0": t0, "first": first, "last": last, "tokens": tokens, "pieces": pieces}


def measure(base: str, model: str, n: int, max_tokens: int = 256, extra: Optional[dict] = None,
            timeout: float = 900) -> dict:
    """``n`` requests at once against the OpenAI-compatible ``base`` (http://host:port/v1); the tokens per
    second they write together, the time to the first token, and the raw per-request records."""
    out: List[Optional[dict]] = [None] * n
    start = threading.Event()
    bodies = [{"model": model, "max_tokens": max_tokens, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True},
               "messages": [{"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}], **(extra or {})}
              for i in range(n)]
    threads = [threading.Thread(target=_one, args=(base, bodies[i], out, i, start, timeout), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()
    errors = [o["error"] for o in out if o and "error" in o]
    if errors:
        raise RuntimeError(errors[0])
    rec = [o for o in out if o and o.get("first")]
    if len(rec) < n:
        raise RuntimeError(f"{n - len(rec)} of {n} requests wrote nothing")
    span = max(o["last"] for o in rec) - min(o["first"] for o in rec)
    tps = sum(o["tokens"] - 1 for o in rec) / max(1e-9, span)
    ttft = [o["first"] - o["t0"] for o in rec]
    return {"n": n, "tokens_per_s": tps, "ttft_median_s": statistics.median(ttft), "ttft_max_s": max(ttft),
            "tokens": [o["tokens"] for o in rec]}


def sweep(base: str, model: str, sizes=(1, 4, 8), max_tokens: int = 256, extra: Optional[dict] = None,
          warmup: bool = True, log=print) -> Dict[int, dict]:
    """``measure`` at each number of requests at once, after one short warm-up request."""
    if warmup:
        measure(base, model, 1, 16, extra)
    res = {}
    for n in sizes:
        res[n] = measure(base, model, n, max_tokens, extra)
        log(f"  {n:2d} at once: {res[n]['tokens_per_s']:6.1f} tokens/s   first token after {res[n]['ttft_median_s']:.2f} s")
    return res
