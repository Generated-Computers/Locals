"""Fixed-length decode throughput through the running locals server: n concurrent /v1/completions
requests, 256 tokens each, end-of-text ignored, 2 rounds per point.
usage: decode_sweep.py <short model name> [engines, default locals,mlx; the ratio is first/second] [port, default 8321]
       [requests at once, comma separated, default 1,2,4,8,16]
The appendix script of docs/measurements/m2-max.md, with the port and the request counts as parameters (issue #18)."""
import json, statistics, sys, threading, time, urllib.request

PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8321
URL = f"http://127.0.0.1:{PORT}/v1/completions"
SHORT = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5-4b"
ENGINES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["locals", "mlx"]
MAX_TOKENS, ROUNDS = 256, 2
NS = [int(n) for n in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1, 2, 4, 8, 16]
PROMPTS = ["Explain how a refrigerator works.", "Why is the sky blue?", "What does a compiler do?", "How do vaccines work?",
           "What is a prime number?", "Why do we have leap years?", "What is photosynthesis?", "How does a hash table work?",
           "What causes tides?", "How does GPS find a position?", "What is inflation?", "How do birds fly?",
           "What is a black hole?", "How does a battery store energy?", "What is machine learning?", "Why do leaves change color?"]

def one(engine, prompt, out):
    body = {"model": f"{engine}/{SHORT}", "prompt": prompt, "max_tokens": MAX_TOKENS, "ignore_eos": True, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        out.append((time.perf_counter() - t, r["usage"]["completion_tokens"], None))
    except Exception as e:  # noqa: BLE001
        out.append((time.perf_counter() - t, 0, f"{type(e).__name__}: {str(e)[:120]}"))

def batch(engine, n):
    outs = []
    ths = [threading.Thread(target=one, args=(engine, PROMPTS[i % len(PROMPTS)], outs)) for i in range(n)]
    t0 = time.perf_counter()
    for th in ths: th.start()
    for th in ths: th.join()
    wall = time.perf_counter() - t0
    errs = [e for _, _, e in outs if e]
    if errs:
        print(f"  {len(errs)} failed request(s) on {engine}: {errs[0]}", flush=True)
    return wall, sum(t for _, t, _ in outs)

for engine in ENGINES:
    batch(engine, 1)  # warm-up
results = {}
for n in NS:
    for rnd in range(ROUNDS):
        for engine in ENGINES:
            wall, toks = batch(engine, n)
            results.setdefault((engine, n), []).append(toks / wall)
            print(f"n={n:2d} round {rnd} {engine:6s}: {toks:5d} tokens in {wall:6.1f}s = {toks / wall:6.1f} tok/s", flush=True)
two = len(ENGINES) == 2
print("\n| Requests at once | " + " | ".join(f"{e} tok/s" for e in ENGINES) + (" | Ratio |" if two else " |"))
print("|" + "---|" * (len(ENGINES) + 1 + two))
for n in NS:
    med = [statistics.median(results[(e, n)]) for e in ENGINES]
    print(f"| {n} | " + " | ".join(f"{m:.0f}" for m in med) + (f" | {med[0] / med[1]:.2f}x |" if two else " |"))
