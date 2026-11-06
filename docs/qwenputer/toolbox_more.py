"""The C tools' numbers in README.md: 45 questions with exact answers (3 each of JSON, tables, versions, lists, logic,
regular expressions, statistics, money, primes, units, roman numerals, text, times, check digits and hashes, and a
small computation), sent to a running server ``--conc`` at a time (temperature 0, a fixed seed). Prints how many are
right, by kind, and how many used a tool of the computer's.

    locals start --model qwenputer-cpu      # (or --model qwenputer, or qwen, to compare)
    python docs/qwenputer/toolbox_more.py 8321 qwenputer-cpu --conc 8
"""
import argparse
import datetime
import hashlib
import json
import random
import re
import threading
import time
import urllib.request
from decimal import ROUND_HALF_UP, Decimal

NAMES = ["Ann", "Bo", "Cy", "Dee", "Eli", "Fay", "Gus", "Hal", "Ivy", "Jo"]
WORDS = ["river", "stone", "lamp", "orbit", "maple", "quartz", "violet", "harbor", "pixel", "canyon", "meadow", "falcon"]


def roman(n):
    out = ""
    for v, s in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                 (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out, n = out + s, n - v
    return out


def luhn_digit(body):
    s = sum((d * 2 - 9 if d * 2 > 9 else d * 2) if i % 2 == 0 else d for i, d in enumerate(reversed(body)))
    return (10 - s % 10) % 10


def questions(seed=0):
    """(kind, question, answer, how it is checked: "num" the last number, "has" a text in the reply)"""
    rnd = random.Random(seed)
    q = []
    for _ in range(3):
        people = [{"name": n, "age": rnd.randint(18, 80), "city": rnd.choice(["Oslo", "Lima", "Pune", "Kyiv"])}
                  for n in rnd.sample(NAMES, 4)]
        i = rnd.randrange(4)
        q.append(("json", f"In this JSON, what is the age of the person named {people[i]['name']}? {json.dumps({'people': people})}",
                  people[i]["age"], "num"))
    for _ in range(3):
        rows = [(n, Decimal(rnd.randint(100, 99999)) / 100) for n in rnd.sample(NAMES, 6)]
        table = "| Name | Amount |\n|---|---:|\n" + "\n".join(f"| {n} | {a:,} |" for n, a in rows)
        q.append(("table", f"What is the total of the Amount column?\n{table}", sum(a for _, a in rows), "num"))
    for _ in range(3):
        vs = list({f"{rnd.randint(1, 3)}.{rnd.randint(0, 12)}.{rnd.randint(0, 20)}" for _ in range(6)})
        key = lambda v: tuple(map(int, v.split(".")))  # noqa: E731
        q.append(("versions", "Sort these version numbers from oldest to newest: " + ", ".join(vs), sorted(vs, key=key), "order"))
    for _ in range(3):
        a, b = rnd.sample(WORDS, 6), rnd.sample(WORDS, 6)
        common = [w for w in a if w in b]
        q.append(("lists", f"How many items are on both lists? List A: {', '.join(a)}. List B: {', '.join(b)}.", len(common), "num"))
    for _ in range(3):
        ops = ["and", "or", "->", "xor"]
        e = f"(p {rnd.choice(ops)} q) {rnd.choice(ops)} (not r {rnd.choice(ops)} p)"
        py = e.replace("->", "<=").replace("xor", "!=").replace("not r", "(not r)")
        n = sum(bool(eval(py, {}, {"p": x, "q": y, "r": z})) for x in (False, True) for y in (False, True) for z in (False, True))
        q.append(("logic", f"In the truth table of {e}, how many of the 8 rows are true?", n, "num"))
    for _ in range(3):
        text = " ".join(f"{rnd.choice(WORDS)}-{rnd.randint(1, 999)}" for _ in range(8))
        k = rnd.randint(2, 3)
        n = len(re.findall(rf"\b\d{{{k}}}\b", text))
        q.append(("regex", f"Using the regular expression \\b\\d{{{k}}}\\b, how many matches are there in: {text}", n, "num"))
    for _ in range(3):
        xs = [rnd.randint(1, 200) for _ in range(rnd.choice([7, 9, 11]))]
        q.append(("stats", "What is the median of these numbers: " + ", ".join(map(str, xs)) + "?", sorted(xs)[len(xs) // 2], "num"))
    for _ in range(3):
        p, r, y = rnd.choice([1000, 2500, 5000]), rnd.choice([3, 4, 5, 6]), rnd.choice([5, 10, 15])
        a = (Decimal(p) * (1 + Decimal(r) / 1200) ** (12 * y)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        q.append(("money", f"How much is ${p} after {y} years at {r}% interest a year, compounded monthly? Give it to the cent.",
                  a, "num"))
    for _ in range(3):
        fs = sorted(rnd.sample([101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157], 3))
        q.append(("primes", f"What is the largest prime factor of {fs[0] * fs[1] * fs[2]}?", fs[2], "num"))
    for _ in range(3):
        km = rnd.randint(5, 500)
        q.append(("units", f"How many miles are {km} km? Round to 2 decimals.",
                  (Decimal(km) * 1000 / Decimal("1609.344")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "num"))
    for _ in range(3):
        n = rnd.randint(1000, 3999)
        q.append(("roman", f"Write {n} in Roman numerals.", roman(n), "has"))
    for _ in range(3):
        s = " ".join(rnd.choice(WORDS) for _ in range(9))
        k = rnd.randint(3, 8)
        q.append(("text", f"What is word number {k} of this text: {s}", s.split()[k - 1], "last_word"))
    for _ in range(3):
        h, m, dh, dm = rnd.randint(0, 23), rnd.choice([0, 15, 30, 45]), rnd.randint(1, 9), rnd.choice([10, 25, 40, 55])
        t = (h * 60 + m + dh * 60 + dm) % 1440
        q.append(("time", f"What time is it {dh} hours and {dm} minutes after {h:02d}:{m:02d}? (24-hour clock)",
                  f"{t // 60:02d}:{t % 60:02d}", "has"))
    for _ in range(3):
        body = [rnd.randint(0, 9) for _ in range(15)]
        q.append(("codes", "What is the Luhn check digit for the card number " + "".join(map(str, body)) + " (the 16th digit)?",
                  luhn_digit(body), "num"))
    for _ in range(3):
        n = rnd.choice([100, 200, 300])
        a, b = rnd.sample([3, 5, 7, 11], 2)
        q.append(("compute", f"What is the sum of all the numbers below {n} that are multiples of {a} or {b}?",
                  sum(x for x in range(n) if x % a == 0 or x % b == 0), "num"))
    return q


def right(how, want, text):
    """The answer in a reply's content: a number the last one in bold (**...**) if there is one, else the last one;
    a text in it; the last of the words named; the versions in order."""
    if how == "num":
        bold = re.findall(r"\*\*([^*]*)\*\*", text)
        nums = [n for b in bold for n in re.findall(r"-?\d[\d,]*(?:\.\d+)?", b)] or re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        return bool(nums) and Decimal(nums[-1].replace(",", "")) == Decimal(str(want))
    if how == "has":
        return str(want) in text
    if how == "last_word":
        found = [w for w in WORDS if re.search(rf"\b{w}\b", text)]
        return bool(found) and max(found, key=lambda w: text.rfind(w)) == want
    if how == "order":
        at = [text.rfind(v) for v in want]
        return all(x >= 0 for x in at) and at == sorted(at)
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("model")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--save")
    a = ap.parse_args()
    qs = questions()
    res = [None] * len(qs)
    lock = threading.Lock()
    nxt = [0]

    def worker():
        while True:
            with lock:
                i = nxt[0]
                nxt[0] += 1
            if i >= len(qs):
                return
            body = {"model": a.model, "messages": [{"role": "user", "content": qs[i][1]}], "temperature": 0, "max_tokens": 1500}
            req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            try:
                d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
                m = d["choices"][0]["message"]
                res[i] = {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
                          "tokens": d["usage"]["completion_tokens"]}
            except Exception as e:  # noqa: BLE001
                res[i] = {"error": repr(e), "content": "", "reasoning": "", "tokens": 0}

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(a.conc)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    by = {}
    for (kind, _, want, how), r in zip(qs, res):
        b = by.setdefault(kind, [0, 0, 0])
        b[0] += right(how, want, r["content"])
        b[1] += 1
        b[2] += "<tool_response>" in r["reasoning"] or "<tool_response>" in r["content"]
    print(f"{a.model}: {sum(b[0] for b in by.values())}/{len(qs)} right in {dt:.0f} s ({sum(r['tokens'] for r in res)} reply "
          "tokens); " + ", ".join(f"{k} {b[0]}/{b[1]} (a tool in {b[2]})" for k, b in by.items()))
    if a.save:
        json.dump([{"kind": k, "q": q_, "want": w, **r} for (k, q_, w, _), r in zip(qs, res)], open(a.save, "w"), indent=1, default=str)
