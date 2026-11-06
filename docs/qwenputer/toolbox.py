"""The toolbox's numbers in README.md: questions a model miscounts or misjudges (the letters of a word, a text's length,
a word backwards, sorting, weekdays, days between dates, a date some days on, a sentence's words, a number in another
base, a word's vowels) with exact answers, sent to a running server ``--conc`` at a
time (temperature 0, a fixed seed for the questions). Prints how many are right, by kind, and how many used a tool.

    locals start --model qwenputer-cpu      # (or --model qwenputer, or qwen, to compare)
    python docs/qwenputer/toolbox.py 8321 qwenputer-cpu --conc 8
"""
import argparse
import datetime
import json
import random
import re
import threading
import time
import urllib.request

WORDS = ["strawberry", "bookkeeper", "Mississippi", "assessment", "committee", "banana", "parallel", "possession",
         "occurrence", "embarrassment", "accommodate", "millennium", "successful", "Tennessee", "referee", "raspberry",
         "necessary", "giraffe", "balloon", "coffee", "address", "worcestershire", "onomatopoeia", "rhythm",
         "pepperoni", "difficulty", "unnecessary", "programming", "subbookkeeper", "hippopotamus"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def questions(n_each=8, seed=0):
    """(kind, question, exact answer), n_each of each kind."""
    rnd = random.Random(seed)
    out = []
    for _ in range(n_each):
        w = rnd.choice(WORDS)
        c = rnd.choice(sorted(set(w.lower())))
        out.append(("count", f"How many times does the letter {c} appear in the word {w}?", w.lower().count(c)))
    for _ in range(n_each):
        w = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(1, 3)))
        out.append(("length", f'How many characters (letters and spaces) are in "{w}"?', len(w)))
    for _ in range(n_each):
        w = rnd.choice(WORDS) + rnd.choice(["", "ing", "s", "ly"])
        out.append(("reverse", f"Write the word {w} backwards.", w[::-1]))
    for _ in range(n_each):
        xs = [rnd.randint(-999, 999) for _ in range(rnd.randint(6, 10))]
        out.append(("sort", "Sort these numbers in ascending order: " + ", ".join(map(str, xs)), sorted(xs)))
    for _ in range(n_each):
        d = datetime.date(1900, 1, 1) + datetime.timedelta(days=rnd.randint(0, 73000))
        out.append(("weekday", f"What day of the week was {d.isoformat()}?", DAYS[d.weekday()]))
    for _ in range(n_each):
        a = datetime.date(1950, 1, 1) + datetime.timedelta(days=rnd.randint(0, 36000))
        b = a + datetime.timedelta(days=rnd.randint(1, 3000))
        out.append(("days", f"How many days are there from {a.isoformat()} to {b.isoformat()}?", (b - a).days))
    for _ in range(n_each):
        a = datetime.date(1950, 1, 1) + datetime.timedelta(days=rnd.randint(0, 36000))
        k = rnd.choice([1, -1]) * rnd.randint(10, 2000)
        out.append(("add_days", f"What date is {abs(k)} days {'after' if k > 0 else 'before'} {a.isoformat()}?",
                    a + datetime.timedelta(days=k)))
    for _ in range(n_each):
        w = " ".join(rnd.choice(WORDS + ["the", "a", "of", "and", "quick", "brown", "fox"]) for _ in range(rnd.randint(5, 14)))
        out.append(("words", f'How many words are in this sentence: "{w}"?', len(w.split())))
    for _ in range(n_each):
        n, b = rnd.randint(100, 2**31), rnd.choice([2, 8, 16])
        name = {2: "binary", 8: "octal", 16: "hexadecimal"}[b]
        out.append(("base", f"What is {n} in {name}?", format(n, {2: "b", 8: "o", 16: "X"}[b])))
    for _ in range(n_each):
        w = rnd.choice(WORDS)
        out.append(("vowels", f"How many vowels (a, e, i, o, u) are in the word {w}?", sum(c in "aeiou" for c in w.lower())))
    return out


def right(kind, want, text):
    """Whether a reply's content gives the answer: the number before "days" (the reply restates the dates), the last
    number (a count or a length), the word backwards in it, the sorted numbers in a row, the last day named."""
    if kind == "add_days":  # (as YYYY-MM-DD, or as Month D, YYYY)
        return want.isoformat() in text or f"{want:%B} {want.day}, {want.year}" in text
    if kind == "words":
        nums = re.findall(r"(-?\d[\d,]*)\s+words?\b", text) or re.findall(r"-?\d[\d,]*", text)
        return bool(nums) and int(nums[-1].replace(",", "")) == want
    if kind == "base":
        return want.lower() in re.sub(r"[\s_]", "", text).lower()
    if kind == "days":
        nums = re.findall(r"(-?\d[\d,]*)\s+(?:calendar\s+)?days?\b", text) or re.findall(r"-?\d[\d,]*", text)
        return bool(nums) and int(nums[-1].replace(",", "")) == want
    if kind in ("count", "length", "vowels"):
        nums = re.findall(r"-?\d[\d,]*", text)
        return bool(nums) and int(nums[-1].replace(",", "")) == want
    if kind == "reverse":
        return want in text
    if kind == "weekday":
        found = [d for d in DAYS if d in text]
        return bool(found) and text.rfind(want) == max(text.rfind(d) for d in found)
    if kind == "sort":
        nums = [int(x) for x in re.findall(r"-?\d+", text)]
        return any(nums[i:i + len(want)] == want for i in range(len(nums)))
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
            body = {"model": a.model, "messages": [{"role": "user", "content": qs[i][1]}], "temperature": 0, "max_tokens": 600}
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
    for (kind, _, want), r in zip(qs, res):
        b = by.setdefault(kind, [0, 0, 0])
        b[0] += right(kind, want, r["content"])
        b[1] += 1
        b[2] += "<tool_response>" in r["reasoning"] or "<tool_response>" in r["content"]
    print(f"{a.model}: {sum(b[0] for b in by.values())}/{len(qs)} right in {dt:.0f} s ({sum(r['tokens'] for r in res)} reply "
          "tokens); " + ", ".join(f"{k} {b[0]}/{b[1]} (a tool in {b[2]})" for k, b in by.items()))
    if a.save:
        json.dump([{"kind": k, "q": q, "want": w, **r} for (k, q, w), r in zip(qs, res)], open(a.save, "w"), indent=1, default=str)
