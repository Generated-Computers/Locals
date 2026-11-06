"""The numbers in README.md: arithmetic questions with exact answers sent to a running server, ``--conc`` at a time
(temperature 0, a fixed seed for the questions). A reply is right when the last number of its content equals the exact
value. Prints how many are right, how many used calc, and the replies' tokens per second in total.

    locals start --model qwenputer          # (or --model qwenputer-cpu, or qwen, to compare)
    python docs/qwenputer/arithmetic.py 8321 qwenputer --conc 8
"""
import argparse
import json
import random
import re
import threading
import time
import urllib.request


def questions(n, seed=0):
    rnd = random.Random(seed)
    big = lambda lo, hi: rnd.randint(10 ** lo, 10 ** hi)  # noqa: E731
    out = []
    kinds = ["mul", "add", "sub", "floordiv", "mod", "pow", "mixed"]
    for i in range(n):
        k = kinds[i % len(kinds)]
        if k == "mul":
            a, b = big(6, 12), big(6, 12)
            out.append((f"What is {a} * {b}?", a * b))
        elif k == "add":
            a, b = big(12, 20), big(12, 20)
            out.append((f"What is {a} + {b}?", a + b))
        elif k == "sub":
            a, b = big(12, 20), big(12, 20)
            out.append((f"What is {a} - {b}?", a - b))
        elif k == "floordiv":
            a, b = big(14, 20), big(3, 8)
            out.append((f"What is {a} // {b} (integer division)?", a // b))
        elif k == "mod":
            a, b = big(14, 20), big(3, 8)
            out.append((f"What is {a} mod {b}?", a % b))
        elif k == "pow":
            a, e = rnd.randint(3, 19), rnd.randint(15, 40)
            out.append((f"What is {a} to the power of {e}?", a ** e))
        else:
            a, b, c = big(5, 9), big(5, 9), big(3, 7)
            out.append((f"What is ({a} + {b}) * {c} - {a}?", (a + b) * c - a))
    return out


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*", text.replace("−", "-"))
    nums = [x.replace(",", "") for x in nums if x.replace(",", "").lstrip("-")]
    return int(nums[-1]) if nums else None


def ask(url, model, q, out, i, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": q + " Use calc if you have it. End with the exact number."}],
            "temperature": 0, "max_tokens": max_tokens}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read())
        m = d["choices"][0]["message"]
        out[i] = {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
                  "tokens": d["usage"]["completion_tokens"], "seconds": time.perf_counter() - t0}
    except Exception as e:  # noqa: BLE001
        out[i] = {"error": repr(e), "tokens": 0, "content": "", "reasoning": "", "seconds": time.perf_counter() - t0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--save")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    qs = questions(a.n)
    res = [None] * len(qs)
    t0 = time.perf_counter()
    i = 0
    live = []
    while i < len(qs) or live:
        live = [t for t in live if t.is_alive()]
        while i < len(qs) and len(live) < a.conc:
            t = threading.Thread(target=ask, args=(url, a.model, qs[i][0], res, i, a.max_tokens))
            t.start()
            live.append(t)
            i += 1
        time.sleep(0.05)
    wall = time.perf_counter() - t0
    right = sum(1 for (q, v), r in zip(qs, res) if last_number(r["content"]) == v)
    toks = sum(r["tokens"] for r in res)
    errs = sum(1 for r in res if "error" in r)
    used = sum(1 for r in res if "<tool_response>" in r["reasoning"] or "<tool_response>" in r["content"])
    print(f"{a.model}: {right}/{len(qs)} right, {errs} errors, calc used in {used}; {toks} reply tokens in {wall:.1f} s: "
          f"{toks / wall:.1f} tokens/s total ({a.conc} at a time)")
    if a.save:
        json.dump([{"q": q, "want": str(v), **r} for (q, v), r in zip(qs, res)], open(a.save, "w"), indent=1)
