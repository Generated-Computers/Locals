"""Issue #22: the cost of a decode pass of 1, 2 and 3 tokens, and where it goes.
The model is loaded with Locals' Runtime (float16, fused layers, small-batch kernels), as mtp_test.py loads it. The
cache holds a prompt and its greedy reply (at most 192 tokens), as in mtp_test.py's timing. Then, in each of
--rounds rounds:
  sync      S = 1, 2, 3, 1: mtp_test.py's timing (a fresh cache, 3 warm steps, 30 timed steps, one mx.eval a step)
  split     the same steps, each cut into building the graph (Python), encoding it (mx.async_eval returns) and
            waiting for the GPU (mx.eval returns); medians
  pipelined step s+1 built and queued while step s runs, as the engines decode: the GPU's time, with the CPU's
            time hidden behind it
  probe     before each sync measurement, 30 synchronous calls of MLX's own 4-bit matmul (mx.quantized_matmul, no
            Locals code) on the output head's weights, 1 row: how fast this GPU is at that moment
First it prints the kernel each 4-bit matmul shape gets at 1, 2 and 3 rows.
--force M=FAMILY gives every shape at M rows that kernel family's tuned winner (mlx = MLX's own kernel);
--force M@NxK=FAMILY only the shape N x K. Repeatable. The choices are set in this process only.
Usage: step_timing.py MODEL OUT.json [--rounds 5] [--force SPEC ...]"""
import argparse
import json
import os
import platform
import statistics as stt
import subprocess
import time

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

import locals as L
from locals import kernels as K
from locals.runtime import Runtime

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--force", action="append", default=[])
a = ap.parse_args()

repo = os.path.dirname(os.path.dirname(L.__file__))
commit = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--", "locals"], capture_output=True, text=True,
                       check=False).stdout.strip()
dev = mx.device_info()
setup = {"mac": f"{dev['device_name']} ({dev['architecture']}, {dev['memory_size'] >> 30} GB)", "macos": platform.mac_ver()[0],
         "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__, "commit": commit + (" with changes in locals/" if dirty else ""),
         "model": a.model, "tune_cache": K._TUNE_PATH, "preset": K._preset_path(), "force": a.force}
for k, v in setup.items():
    print(f"{k}: {v}", flush=True)

rt = Runtime.load(a.model)
model, tok = rt.model, rt.tokenizer
tune = K._load_tune()
print(f"tune cache says mlx {tune.get('__mlx__')}, {len(tune.get('__stale__') or [])} stale shapes", flush=True)
setup["tune_cache_mlx"] = tune.get("__mlx__")

shapes = {}
for name, mod in model.named_modules():
    if K._tunable(mod):
        shapes.setdefault((mod["weight"].shape[0], mod["weight"].shape[1] * 8, K._mod_dt(mod)), []).append(name)


def key(N, Kd, M, dt):
    return f"{N}x{Kd}x{M}" + ("" if dt == "float16" else f"x{dt}")


for spec in a.force:
    rows, fam = spec.split("=")
    M, _, only = rows.partition("@")
    M = int(M)
    for N, Kd, dt in shapes:
        if only and only != f"{N}x{Kd}":
            continue
        c = ("mlx",) if fam == "mlx" else tune.get(key(N, Kd, M, dt) + "|" + fam)
        if c is None:
            raise SystemExit(f"--force {spec}: no tuned {fam} kernel for {key(N, Kd, M, dt)}")
        K._CHOICE[(N, Kd, M, dt)] = tuple(c)

# the kernels the model's matmuls get (a forward pass at each row count fills K._CHOICE; None: not called)
x0 = mx.array([[100, 200, 300]])
for S in (1, 2, 3):
    mx.eval(model(x0[:, :S], cache=make_prompt_cache(model)))
choices = {}
print("\nkernel per matmul shape (layers using it): 1 row | 2 rows | 3 rows", flush=True)
for (N, Kd, dt), names in sorted(shapes.items()):
    row = [K._CHOICE.get((N, Kd, M, dt)) for M in (1, 2, 3)]
    choices[f"{N}x{Kd}"] = {"layers": len(names), "example": names[0], **{str(M): c for M, c in zip((1, 2, 3), row)}}
    print(f"  {N}x{Kd} ({len(names)}, e.g. {names[0].removeprefix('language_model.model.')}): "
          + " | ".join(str(c) for c in row), flush=True)

eos = set(tok.eos_token_ids) | {tok.convert_tokens_to_ids("<|im_end|>")}
prompt = tok.encode(tok.apply_chat_template([{"role": "user", "content": "Explain how a refrigerator works, in about 150 words."}],
                                            add_generation_prompt=True, enable_thinking=False, tokenize=False), add_special_tokens=False)
cache = make_prompt_cache(model)
logits, reply = model(mx.array(prompt)[None], cache=cache), []
for _ in range(192):
    y = int(mx.argmax(logits[0, -1]).item())
    reply.append(y)
    if y in eos:
        break
    logits = model(mx.array([[y]]), cache=cache)
ids = mx.array(prompt + reply)[None]
print(f"\ncache: {ids.shape[1]} tokens (prompt {len(prompt)}, reply {len(reply)})", flush=True)
tokx = int(ids[0, 5].item())  # the token mtp_test.py's timing repeats (feats[0][1][0, 5]), so the two time the same steps


def filled(S):
    c = make_prompt_cache(model)
    mx.eval(model(ids, cache=c))
    x = mx.array([[tokx] * S])
    for _ in range(3):
        mx.eval(model(x, cache=c))
    return c, x


lm = K.text_model(model)
head = lm.lm_head if hasattr(lm, "lm_head") else lm.model.embed_tokens
xp = mx.random.normal((1, head["weight"].shape[1] * 8), key=mx.random.key(0)).astype(mx.float16)


def probe(n=30):
    def f():
        return mx.quantized_matmul(xp, head["weight"], head["scales"], head["biases"], transpose=True, group_size=64, bits=4)
    for _ in range(3):
        mx.eval(f())
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(f())
    return (time.perf_counter() - t) / n * 1e3


def sync(S, n=30):
    c, x = filled(S)
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(model(x, cache=c))
    return (time.perf_counter() - t) / n * 1e3


def split(S, n=30):
    c, x = filled(S)
    parts = [[], [], []]
    for _ in range(n):
        t0 = time.perf_counter()
        y = model(x, cache=c)
        t1 = time.perf_counter()
        mx.async_eval(y)
        t2 = time.perf_counter()
        mx.eval(y)
        t3 = time.perf_counter()
        for p, v in zip(parts, (t1 - t0, t2 - t1, t3 - t2)):
            p.append(v * 1e3)
    return [stt.median(p) for p in parts]


def pipelined(S, n=30):
    c, x = filled(S)
    y = model(x, cache=c)
    mx.eval(y)
    t = time.perf_counter()
    for _ in range(n):
        y2 = model(x, cache=c)
        mx.async_eval(y2)
        mx.eval(y)
        y = y2
    mx.eval(y)
    return (time.perf_counter() - t) / n * 1e3


sync(1)  # (warm)
rounds = []
t_start = time.perf_counter()
for r in range(a.rounds):
    t_round = time.perf_counter() - t_start
    s, pr = [], []
    for S in (1, 2, 3, 1):
        pr.append(probe())
        s.append(sync(S))
    sp = {S: split(S) for S in (1, 2, 3)}
    pp = {S: pipelined(S) for S in (1, 2, 3)}
    rounds.append({"t_s": t_round, "sync": {"1_first": s[0], "2": s[1], "3": s[2], "1_last": s[3]},
                   "probe_before_sync": {"1_first": pr[0], "2": pr[1], "3": pr[2], "1_last": pr[3]},
                   "split": {str(S): v for S, v in sp.items()}, "pipelined": {str(S): v for S, v in pp.items()}})
    print(f"round {r} (t={t_round:.0f} s): sync S=1 {s[0]:.2f}, S=2 {s[1]:.2f}, S=3 {s[2]:.2f}, S=1 {s[3]:.2f} ms"
          f" | probe before each {pr[0]:.3f}, {pr[1]:.3f}, {pr[2]:.3f}, {pr[3]:.3f} ms | split (build + encode + wait) "
          + ", ".join(f"S={S} {b:.2f}+{e:.2f}+{w:.2f}" for S, (b, e, w) in sp.items())
          + " | pipelined " + ", ".join(f"S={S} {v:.2f}" for S, v in pp.items()), flush=True)


def summary(vals):
    return {"mean": stt.mean(vals), "min": min(vals), "max": max(vals)}


out = {"setup": setup, "cache_tokens": int(ids.shape[1]), "choices": choices, "rounds": rounds, "summary": {}}
one_sync = [min(r["sync"]["1_first"], r["sync"]["1_last"]) for r in rounds]
print(f"\nover {a.rounds} rounds: mean (min to max), ms; ratio = mean / mean of S=1", flush=True)
for mode in ("sync", "pipelined"):
    base = stt.mean(one_sync) if mode == "sync" else stt.mean(r["pipelined"]["1"] for r in rounds)
    for S in ("1", "2", "3"):
        vals = one_sync if (mode == "sync" and S == "1") else [r[mode][S] for r in rounds]
        sm = summary(vals)
        sm["ratio"] = sm["mean"] / base
        out["summary"][f"{mode}_S{S}"] = sm
        print(f"  {mode:9s} S={S}: {sm['mean']:.2f} ({sm['min']:.2f} to {sm['max']:.2f}), {sm['ratio']:.2f}x"
              + (" (S=1: the lower of first and last each round)" if mode == "sync" and S == "1" else ""), flush=True)
pv = [v for r in rounds for v in r["probe_before_sync"].values()]
out["summary"]["probe"] = summary(pv)
print(f"  probe: {stt.mean(pv):.3f} ({min(pv):.3f} to {max(pv):.3f}) ms", flush=True)
with open(a.out, "w") as f:
    json.dump(out, f, indent=1)
