"""Answers that can be checked: word problems with computed answers (two or three steps of arithmetic), asked of a
server eight at a time, temperature 0. The share answered right, the replies kept, and the time it took.

    python quality_audit.py --base http://127.0.0.1:8080/v1 --model mlx-community/Qwen3.5-4B-MLX-4bit --out q_mlxlm.json
"""
import argparse
import json
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def problems(n=48, seed=7):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = i % 4
        if k == 0:
            a, p, b = rng.randint(3, 40), rng.randint(2, 15), rng.randint(2, 9)
            q, ans = (f"A shop sells {a} notebooks at {p} dollars each and {b} pens at 3 dollars each. How many dollars "
                      f"does the shop take in?"), a * p + b * 3
        elif k == 1:
            t, s, r = rng.randint(2, 9), rng.randint(30, 90), rng.randint(1, 5)
            q, ans = (f"A train travels {t} hours at {s} km per hour, then {r} more hours at 20 km per hour. How many "
                      f"kilometers does it travel in total?"), t * s + r * 20
        elif k == 2:
            c, e, g = rng.randint(50, 400), rng.randint(2, 12), rng.randint(5, 30)
            q, ans = (f"A library has {c} books. It buys {e} boxes of {g} books each, then gives away 17 books. How many "
                      f"books does it have now?"), c + e * g - 17
        else:
            w, h, d = rng.randint(3, 25), rng.randint(3, 25), rng.randint(2, 6)
            q, ans = (f"A garden is {w} meters long and {h} meters wide. A path {d} meters wide is laid along one "
                      f"{w}-meter side, inside the garden. What area, in square meters, is left for plants?"), w * h - w * d
        out.append((q + " Think step by step, then write the final number on the last line as 'Answer: N'.", ans))
    return out


def ask(base, model, q, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": q}], "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())["choices"][0]["message"]["content"] or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--at-once", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args()
    ps = problems()
    t0 = time.time()
    with ThreadPoolExecutor(a.at_once) as ex:
        replies = list(ex.map(lambda p: ask(a.base, a.model, p[0], a.max_tokens), ps))
    took = time.time() - t0
    right = []
    for (q, ans), r in zip(ps, replies):
        m = re.findall(r"Answer:\s*\**\s*\$?(-?[\d,]+)", r)
        right.append(bool(m) and int(m[-1].replace(",", "")) == ans)
    print(f"{a.model}: {sum(right)} of {len(ps)} right in {took:.0f} s")
    json.dump({"model": a.model, "right": right, "replies": replies, "s": took}, open(a.out, "w"))


if __name__ == "__main__":
    main()
