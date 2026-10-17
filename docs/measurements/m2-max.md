# Locals vs mlx-lm on an Apple M2 Max (96 GB)

Locals at commit `3f9db32` (2026-10-13) against the stock mlx-lm engine that ships inside the same server, on one
machine, 14–15 October 2026. Part of the [measurements](README.md).

## Machine and software

| | |
|---|---|
| Chip | Apple M2 Max, 38-core GPU (`applegpu_g14s`) |
| Memory | 96 GB unified; `max_recommended_working_set_size` = 83.5 GB |
| macOS | 26.6.2 (25G83) |
| Locals | 0.2.0, commit `3f9db32`, installed with `uv sync --extra dev` from a clone |
| Python | 3.11.11 |
| mlx / mlx-lm / transformers | 0.32.3 / 0.32.0 / 5.19.0 (versions pinned by `uv.lock`) |
| Kernel tuning | Qwen3.5-4B: shipped preset `tune_v6_applegpu_g14s.json`, no tuning. 9B / 27B: `locals tune --mode quick` run once per model before any timing (354 s, 419 s) |
| Disk prompt cache | off (`LOCALS_DISK_CACHE=0`) for every run |

Nothing else was running on the machine during the benchmark except the terminal.

## Models

All 4-bit, from `mlx-community`:

| Model | HF repo | model_type | Weights on disk |
|---|---|---|---|
| Qwen3.5-4B | `mlx-community/Qwen3.5-4B-MLX-4bit` | qwen3_5 | 2.8 GB |
| Qwen3.5-9B | `mlx-community/Qwen3.5-9B-MLX-4bit` | qwen3_5 | 5.5 GB |
| Qwen3.8-27B | `mlx-community/Qwen3.8-27B-4bit` | qwen3_5 | 15 GB |

## Method

Two measurements per model, both through the HTTP server (`locals serve`, port 8321, both engines loaded in one
process):

1. **`locals bench`**: the repo's own tool: `--n` concurrent tool-calling chat requests, 2 rounds, median tokens/s per
   engine. Output lengths are whatever the model produces, so they differ a little between engines and rounds.
2. **Fixed-length decode sweep**: the method described in [Measurements](README.md): `n` concurrent `/v1/completions`
   requests, 256 tokens each, `ignore_eos: true`, `temperature: 0`, one warm-up request per engine, 2 rounds per
   point, median of the two rounds. Aggregate tokens/s = total completion tokens / wall time of the batch. The script
   is in the appendix. Ratio = Locals / mlx-lm.

## Results

### Summary: fixed-length decode, 8 concurrent requests

| Model | mlx-lm tok/s | Locals tok/s | Ratio |
|---|---|---|---|
| Qwen3.5-4B | 152 | 403 | 2.66x |
| Qwen3.5-9B | 72 | 220 | 3.03x |
| Qwen3.8-27B | 25 | 78 | 3.13x |

### Qwen3.5-4B

Decode sweep (256 tokens per request, eos ignored):

| Requests at once | mlx-lm tok/s | Locals tok/s | Ratio |
|---|---|---|---|
| 1 | 91 | 104 | 1.15x |
| 2 | 121 | 177 | 1.46x |
| 4 | 142 | 244 | 1.71x |
| 8 | 152 | 403 | 2.66x |
| 16 | 267 | 530 | 1.99x |

`locals bench` (tool-calling chat, median of 2 rounds):

| `--n` | mlx-lm tok/s | Locals tok/s | Ratio |
|---|---|---|---|
| 1 | 34 | 62 | 1.81x |
| 2 | 61 | 113 | 1.84x |
| 4 | 69 | 140 | 2.03x |
| 8 | 80 | 131 | 1.63x |
| 16 | 95 | 142 | 1.49x |

GPU memory with both engines loaded: peak 11.6 GB.

For comparison, the main README reports for an M2 Pro 16 GB with the same model: 1 → 61/64, 2 → 82/116,
4 → 87/160, 8 → 87/271 (3.1x), 16 → 157/341 tok/s (mlx-lm/Locals). Those were measured with the engines in process,
these through the HTTP server.

### Qwen3.5-9B

Decode sweep:

| Requests at once | mlx-lm tok/s | Locals tok/s | Ratio |
|---|---|---|---|
| 1 | 49 | 56 | 1.15x |
| 2 | 60 | 86 | 1.43x |
| 4 | 69 | 125 | 1.81x |
| 8 | 72 | 220 | 3.03x |
| 16 | 138 | 300 | 2.18x |

`locals bench`:

| `--n` | mlx-lm tok/s | Locals tok/s | Ratio |
|---|---|---|---|
| 8 | 41 | 108 | 2.60x |
| 16 | 49 | 77 | 1.56x |

GPU memory with both engines loaded: 14.4 GB in use, peak 16.1 GB. Locals engine load time after tuning: 4.4 s.

### Qwen3.8-27B

Decode sweep:

| Requests at once | mlx-lm tok/s | Locals tok/s | Ratio |
|---|---|---|---|
| 1 | 18 | 20 | 1.14x |
| 2 | 21 | 30 | 1.43x |
| 4 | 24 | 42 | 1.80x |
| 8 | 25 | 78 | 3.13x |
| 16 | 47 | 106 | 2.25x |

`locals bench`:

| `--n` | mlx-lm tok/s | Locals tok/s | Ratio |
|---|---|---|---|
| 8 | 11 | 24 | 2.09x |
| 16 | 13 | 23 | 1.73x |

GPU memory with both engines loaded: 38.6 GB in use, peak 43.9 GB. Locals engine load time after tuning: 5.2 s.

## Observations

- **The shape of the curve is the same on all three models.** Near parity with one request (1.14–1.15x), rising
  steadily to a peak at 8 (2.66–3.13x), then a smaller ratio at 16 because mlx-lm's throughput roughly doubles between
  8 and 16 while Locals gains less.
- **The peak ratio grows with model size**: 2.66x (4B) → 3.03x (9B) → 3.13x (27B) at n=8.
- **Absolute single-stream numbers** with Locals: 104 / 56 / 20 tok/s for 4B / 9B / 27B.
- **`locals bench` ratios are lower than the fixed-length sweep at n=8 and n=16** on every model (e.g. 4B: 1.63x vs
  2.66x at n=8). The bench runs short tool-calling replies (roughly 40–100 tokens per request) where prefill and
  per-request overhead weigh more than sustained decode. At n=1 and n=2 the bench ratio is actually higher than the
  sweep (1.8x vs 1.15–1.46x on the 4B), presumably from the prefill side.
- **On the 4B the measured speedups are lower than the README's M2 Pro table** (2.66x vs 3.1x at n=8), mostly because
  mlx-lm itself is faster here (152 vs 87 tok/s at n=8), while Locals is 403 vs 271. The ratio compresses as the
  baseline gets more bandwidth.
- No errors, failed requests, memory trimming or held-back requests were seen in any run (`locals status` showed
  "trimmed for memory 0x, requests held back 0x" throughout).

## Caveats

- Two rounds per point; medians of two. Round-to-round spread was small (typically under 3%).
- `locals bench` output lengths vary between engines and rounds, so its tok/s is less controlled than the sweep.
- The 9B and 27B were tuned with `--mode quick`, not `full`.

## Appendix: decode sweep script

Run against an already-started server (`locals serve --model <repo> --engines locals,mlx --preload locals,mlx`, or
the server started by `locals bench`), with `LOCALS_DISK_CACHE=0`.

```python
"""Fixed-length decode throughput through the running locals server: n concurrent /v1/completions
requests, 256 tokens each, end-of-text ignored, 2 rounds per point.
usage: decode_sweep.py <short model name> [engines, default locals,mlx; the ratio is first/second]"""
import json, statistics, sys, threading, time, urllib.request

URL = "http://127.0.0.1:8321/v1/completions"
SHORT = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5-4b"
ENGINES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["locals", "mlx"]
MAX_TOKENS, ROUNDS, NS = 256, 2, [1, 2, 4, 8, 16]
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
```
