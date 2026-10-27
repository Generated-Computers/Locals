# Recursive's Flint on an Apple M5 Pro (24 GB)

Locals at commit `01127a8` against the stock mlx-lm engine that ships inside the same server, on one MacBook Pro,
27 October 2026. Part of the [measurements](README.md). The same comparison as the M4 Pro's in
[docs/benchmarks, section 11](../benchmarks/README.md#11-recursives-flint-m4-pro-24-gb), but through the HTTP server
(see "Method").

## Machine and software

| | |
|---|---|
| Chip | Apple M5 Pro, 16-core GPU (`applegpu_g17s`) |
| Memory | 24 GB unified; `max_recommended_working_set_size` 19.07 GB, GPU limit raised to 20 GB (`sudo sysctl iogpu.wired_limit_mb=20480`) |
| macOS | 27.0.1 |
| Locals | 0.2.0, commit `01127a8` (checkout, editable install) |
| Python | 3.11.17 |
| mlx / mlx-lm / transformers | 0.32.4 / 0.32.0 / 5.19.0 (the main tables); 0.32.3 / 0.32.0 / 5.19.0 (what `uv.lock` pins) |
| Model | `mlx-community/flint-mixed-3-4bit` (16.6 GB of weights, experts at 3 bits) |
| Kernel tuning | `locals tune --model flint --mode quick`, once per MLX version, with no Flint entries before (523 s on 0.32.3, 413 s on 0.32.4; the 0.32.4 run into its own cache file). `full` does not fit in 24 GB |
| Prompt cache | `--cache-gb 0.1` for both engines, `LOCALS_DISK_CACHE=0` (see "Memory") |

Nothing else was running during the timings: Chrome, ChatGPT, Claude and the other large apps were quit first (Cursor
stayed open). Each engine ran alone on the GPU, one server at a time (two copies of Flint do not fit).

## Results

Tokens per second that all requests write together, the median of 3 rounds after a warm-up at that number of
requests. Every reply was 256 tokens.

### MLX 0.32.4 (the main table)

| Requests at once | mlx-lm | Locals | Locals / mlx-lm | Rounds, mlx-lm | Rounds, Locals |
|---|---|---|---|---|---|
| 1 | 101 | 89 | 0.89x | 101.0, 100.5, 100.7 | 89.5, 89.2, 89.3 |
| 2 | 152 | 211 | 1.39x | 151.5, 151.2, 154.1 | 210.7, 209.9, 210.6 |
| 4 | 216 | 286 | 1.32x | 212.8, 216.1, 215.9 | 285.6, 285.5, 285.4 |
| 8 | 257 | 356 | 1.39x | 256.8, 256.4, 256.6 | 356.3, 355.7, 355.8 |
| 16 | 313 | 449 | 1.43x | 314.9, 313.3, 301.5 | 449.3, 449.3, 450.1 |

One request with `LOCALS_SOLO=0` (below): Locals 138 tokens/s (138.3, 138.4, 138.6), **1.37x** mlx-lm's 101.

### MLX 0.32.3 (what `uv.lock` pins; kernels tuned on 0.32.3)

| Requests at once | mlx-lm | Locals | Locals / mlx-lm |
|---|---|---|---|
| 1 | 101 | 89 | 0.88x |
| 2 | 153 | 211 | 1.38x |
| 4 | 215 | 280 | 1.30x |
| 8 | 256 | 343 | 1.34x |
| 16 | 310 | 448 | 1.45x |

The two MLX versions agree within 4% for Locals (the largest gap is 8 requests, 356 against 343) and within 1.5% for
mlx-lm.

For reference, the M4 Pro (24 GB), measured in the process with MLX 0.32.4 (section 11 of the benchmarks), mlx-lm /
Locals: 85 / 117 at 1, 177 / 227 at 4, 202 / 319 at 8, 245 / 354 at 16. The two methods differ, so only the ratios are
comparable: on the M4 Pro through the server, the same Mac wrote 305 tokens/s at 8 requests against 319 in the
process, and 338 against 354 at 16.

## One request: slower by default here, a bug in the server's loop (fixed since)

With one request through the server, Locals writes 89 tokens/s against mlx-lm's 101. With `LOCALS_SOLO=0` it writes
138 (1.37x, the M4 Pro's ratio). One request, MLX 0.32.4, each component switched off on its own:

| Variant | Tokens/s (3 rounds) |
|---|---|
| default | 89.3 |
| `LOCALS_SOLO=0` | **138.4** |
| `LOCALS_EXPERT_KERNELS=0` | 85.9 |
| `LOCALS_QK_KERNEL=0` | 87.7 |
| `LOCALS_KERNELS=0` | 89.7 |
| `LOCALS_FUSE=0` | 73.2 |

Only `LOCALS_SOLO=0` moves it. The `solo_steps` counter is never incremented in the default run, so the one-request
path itself never runs; the cost is in the check before it. `HybridBatchEngine.serve` calls `_solo_ok(rows[0], cache)` after every token while one
row runs (`hybrid_batch.py`, the line `if _SOLO and len(rows) == 1 and ... and self._solo_ok(...)`). For Flint,
`_solo_ok` reaches `int(mx.max(c.left_padding).item()) > 0` for the sliding-window layers (added in `c0f9baa`), and
`.item()` waits for the GPU, so the step launched ahead is no longer overlapped with the host's work. Timed with a
wrapper around `_solo_ok`: 778 calls in the three rounds, 5.29 ms each on average, 4.11 s in
all, and it returned True every time (so it never led to a solo step). A step is 11.2 ms at 89 tokens/s and 7.2 ms at
138.

It does not affect 2 requests and more (`len(rows) == 1` is false), nor the in-process `run_drip` loop, where
`_solo_ok` is not called. No code was changed for these measurements.

**Fixed since.** The padding is now read by `_solo_unpadded`, once for all the layers, and only where a span is
about to go in one pass (where the pipeline drains anyway), never after every token: the check that runs after
every token reads nothing from the GPU (`tests/test_flint.py` fails if it does). The one-request rows above are from
before the fix; `LOCALS_SOLO=0`, which skips the check, wrote 138 tokens/s, what the default is expected to write
now. Not measured again on this Mac yet.

## Memory: 16 requests at once sit at the GPU limit

With the GPU limit at 20 GB, the weights take 17 GB; the server's prompt cache (finished replies are kept as entries,
about 27 MB each) and MLX's buffer pool take the rest. Locals' peak GPU memory is 20.5 GB already at load
(`gpu_peak_mb` 20,518 against the limit of 20,480). The rows below are the runs before the tables above: first with
the apps open, then with them quit, both with the default prompt cache, then with `--cache-gb 0.1`:

| Setup | 1 | 2 | 4 | 8 | 16 (3 rounds) |
|---|---|---|---|---|---|
| Locals, apps open, default cache | 88 | 209 | 276 | 337 | 212, 80, 78 |
| mlx-lm, apps open, default cache | 94 | 150 | 211 | 252 | 23 (stopped) |
| Locals, apps quit, default cache | 88 | 211 | 279 | 340 | 437, 328, 359 |
| mlx-lm, apps quit, default cache | 101 | 151 | 213 | 251 | 60, 180, 39 |
| Locals, apps quit, `--cache-gb 0.1` | 89 | 211 | 280 | 343 | 446, 448, 448 |
| mlx-lm, apps quit, `--cache-gb 0.1` | 101 | 153 | 215 | 256 | 310, 313, 296 |

(MLX 0.32.3.) Both engines hold 19.9 to 20.3 GB of active GPU memory with the default cache, and the system pages
(swap in use 9 to 14.6 GB before the apps were quit, 2 to 5 GB after). With a 0.1 GB prompt cache the active memory is
18.4 GB (Locals) and 18.6 GB (mlx-lm) and the rounds agree. These timings reuse no prompt, so the tables above use
the small cache; a 24 GB Mac running Flint should have little else open.

## What the tuner chose on this GPU

The shipped preset for `applegpu_g17s` has 21 entries and none for Flint, so the kernels were tuned on this Mac.
One decode step's 4-bit matrix products, MLX's kernels against the chosen ones (MLX 0.32.4; 0.32.3 in brackets):

| Rows | MLX's kernels | Chosen | Speed-up | Chosen kernels (layers) |
|---|---|---|---|---|
| 1 | 1.88 ms | 1.88 ms | 1.00x (1.00x) | mlx 197 |
| 2 | 1.99 ms | 1.99 ms | 1.00x (1.00x) | mlx 197 |
| 4 | 3.66 ms | 2.99 ms | 1.22x (1.06x) | mlx 143, nax 54 |
| 6 | 5.65 ms | 2.96 ms | 1.91x (2.25x) | nax 150, mlx 47 |
| 8 | 8.93 ms | 3.71 ms | 2.41x (3.10x) | nax 150, fragq 47 |
| 12 | 11.42 ms | 3.23 ms | 3.53x (3.61x) | nax 197 |
| 16 | 15.17 ms | 3.29 ms | 4.61x (4.63x) | nax 197 |

From 6 rows on the tuner took Locals' own `nax` kernel (the M5's neural accelerators, experimental; `LOCALS_NAX=0`
turns it off) for most layers. (`locals tune --report` prints this table.)

## Method

- `locals.bench.measure` (the `locals bench` method) against `locals.server` with a single engine
  (`--engines E --preload E --temp 0 --slots 16 --cache-gb 0.1`, `LOCALS_DISK_CACHE=0`): n streamed
  `/v1/chat/completions` requests at once, each one of sixteen short questions, `max_tokens` 256, `ignore_eos`,
  temperature 0. The speed is all the tokens over the time from the first token of any request to the last token of
  any, so reading the prompts (8 to 12 tokens) is left out. Every reply was 256 tokens, in every round.
- A first request per engine, then for each n a warm-up of n requests of 32 tokens, then 3 timed rounds; the median.
- Not the M4 Pro's in-process method (256-token prompts, 48 greedy steps after 4 warm-up steps): its script is not in
  the repository.
- The stock engine is mlx-lm's `BatchGenerator` behind Locals' server (`--engines mlx`), running Locals' copy of
  Flint's model code registered at `mlx_lm.models.flint`. The M4 Pro's mlx-lm ran the pull request's own code
  (ml-explore/mlx-lm#1998); on MLX 0.32.4 the two give the same logits bit for bit (docs/runtime, "Flint").
- MLX 0.32.4 ran in a separate virtual environment built from `uv pip freeze` of the checkout's with `mlx` and
  `mlx-metal` changed to 0.32.4, `PYTHONPATH` at the checkout, and `LOCALS_TUNE_CACHE` at its own file, so the
  user's cache and environment were left alone.

## Repeating it

```bash
sudo sysctl iogpu.wired_limit_mb=20480        # until the next restart; quit large apps first
locals tune --model flint --mode quick         # once; with LOCALS_TUNE_CACHE=... for a separate MLX version
export FLINT_SERVER_ARGS="--cache-gb 0.1"
python flint_m5_sweep.py locals               # writes /tmp/flint_m5_locals.json
python flint_m5_sweep.py mlx                  # one engine at a time: they do not fit together
FLINT_SIZES=1 LOCALS_SOLO=0 python flint_m5_sweep.py locals solo0
```

`FLINT_PYTHON` picks another interpreter for the server, `FLINT_ROUNDS` and `FLINT_SIZES` change the rounds and the
request counts.

## Caveats

- Through the server, not in the process: expect the in-process numbers a few percent higher at 2 requests and more.
- The one-request rows are the default's before the fix (above); expect about 138 (1.37x) with it.
- The tuning is `quick` (up to 16 rows), as the background tuner does; `full` was not run.
- 24 GB with the GPU limit at 20 GB leaves little room: results at 16 requests depend on what else runs and on the
  prompt cache's size.
- A transformers warning at load, "tokenizer ... with an incorrect regex pattern (Mistral)", appears for this build; it
  does not change the speed.

## Appendix: the sweep script

`flint_m5_sweep.py`; `REPO` is the checkout.

```python
"""Flint decode sweep on this Mac, one engine alone on the GPU (two copies of Flint do not fit in 24 GB).

usage: flint_m5_sweep.py <locals|mlx> [tag]

Starts `locals.server` with only that engine loaded, measures n requests at once for n in SIZES with the repo's own
`locals.bench.measure` (256 tokens each, end of text ignored, temperature 0, tokens/s all requests write together
from the first token of any to the last token of any), ROUNDS rounds per size after a warm-up at that size, then stops
the server. Writes /tmp/flint_m5_<tag or engine>.json.
"""
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

REPO = "/path/to/Locals"  # the checkout
sys.path.insert(0, REPO)
from locals import bench  # noqa: E402

MODEL = "mlx-community/flint-mixed-3-4bit"
SHORT = "flint"
PORT = 8341
SIZES = [int(x) for x in os.environ.get("FLINT_SIZES", "1,2,4,8,16").split(",")]
ROUNDS = int(os.environ.get("FLINT_ROUNDS", "3"))
EXTRA_ARGS = os.environ.get("FLINT_SERVER_ARGS", "").split()
MAX_TOKENS = 256

engine = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else engine
log_path = f"/tmp/flint_m5_server_{tag}.log"
base = f"http://127.0.0.1:{PORT}/v1"


def get(path, timeout=5):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return json.loads(r.read())


PY = os.environ.get("FLINT_PYTHON", f"{REPO}/.venv/bin/python")
env = dict(os.environ, LOCALS_DISK_CACHE="0", PYTHONPATH=REPO)
ENTRY = os.environ.get("FLINT_ENTRY")
cmd = [PY] + ([ENTRY] if ENTRY else ["-m", "locals.server"]) + ["--model", MODEL, "--engines", engine, "--preload", engine,
       "--port", str(PORT), "--temp", "0", "--slots", "16"] + EXTRA_ARGS
log = open(log_path, "wb")
t_start = time.time()
srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)
print(f"[{engine}] server pid {srv.pid}, log {log_path}", flush=True)


def _swap_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used =")[1].split("M")[0])


def _pageouts():
    for line in subprocess.run(["vm_stat"], capture_output=True, text=True).stdout.splitlines():
        if line.startswith("Pageouts"):
            return int(line.split(":")[1].strip().rstrip("."))
    return -1


MON = []
_stop = threading.Event()


def _monitor():
    while not _stop.wait(2.0):
        try:
            m = get("/stats", 3).get("memory", {})
        except Exception:  # noqa: BLE001
            m = {}
        MON.append({"t": round(time.time() - t_start, 1), "gpu_active_mb": round(m.get("gpu_active_mb", 0)),
                    "gpu_peak_mb": round(m.get("gpu_peak_mb", 0)), "rss_mb": round(m.get("rss_mb", 0)),
                    "swap_used_mb": _swap_mb(), "pageouts": _pageouts()})


result = {"engine": engine, "model": MODEL, "max_tokens": MAX_TOKENS, "rounds": ROUNDS, "points": {}}
try:
    while True:  # wait for the model to load
        if srv.poll() is not None:
            sys.exit(f"[{engine}] server stopped (exit {srv.returncode}); see {log_path}")
        try:
            if get("/health", 2).get("status") == "ok":
                break
        except Exception:  # noqa: BLE001
            pass
        if time.time() - t_start > 1800:
            sys.exit(f"[{engine}] server did not come up in 30 minutes; see {log_path}")
        time.sleep(2)
    result["load_s"] = round(time.time() - t_start, 1)
    print(f"[{engine}] server up after {result['load_s']} s", flush=True)

    threading.Thread(target=_monitor, daemon=True).start()
    name = f"{engine}/{SHORT}"
    extra = {"ignore_eos": True}
    bench.measure(base, name, 1, 16, extra)  # the engine's first request
    for n in SIZES:
        bench.measure(base, name, n, 32, extra)  # warm-up at this size (kernels compile, shapes settle)
        rounds = []
        for r in range(ROUNDS):
            m = bench.measure(base, name, n, MAX_TOKENS, extra)
            short = [t for t in m["tokens"] if t != MAX_TOKENS]
            rounds.append({"tps": round(m["tokens_per_s"], 1), "ttft_median_s": round(m["ttft_median_s"], 2),
                           "tokens": m["tokens"], "not_256": short})
            print(f"[{engine}] n={n:2d} round {r}: {m['tokens_per_s']:6.1f} tokens/s  first token {m['ttft_median_s']:.2f} s"
                  + (f"  (replies not 256 tokens: {short})" if short else ""), flush=True)
        tps = [x["tps"] for x in rounds]
        result["points"][str(n)] = {"median": statistics.median(tps), "min": min(tps), "max": max(tps), "rounds": rounds}
        print(f"[{engine}] n={n:2d} median {statistics.median(tps):.1f}  (min {min(tps):.1f}, max {max(tps):.1f})", flush=True)
    try:
        result["stats"] = get("/stats", 10)
    except Exception as e:  # noqa: BLE001
        result["stats_error"] = str(e)
finally:
    if srv.poll() is None:
        os.killpg(srv.pid, signal.SIGTERM)
        try:
            srv.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(srv.pid, signal.SIGKILL)
    _stop.set()
    result["monitor"] = MON
    out = f"/tmp/flint_m5_{tag}.json"
    json.dump(result, open(out, "w"), indent=1, default=str)
    print(f"[{engine}] wrote {out}", flush=True)
```
