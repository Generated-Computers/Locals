"""The tables of README.md from the result files of step_timing.py: each run's means, and how the slow sync
measurements line up with the probe (MLX's own matmul, timed just before each one).
Usage: summarize.py results/*.json"""
import json
import os
import statistics as stt
import sys
from collections import namedtuple

SLOW_PROBE = 1.25  # a probe this many times its median marks a slow moment (a line chosen here, not given)
SLOW_STEP = 1.10  # a step this many times its run's median counts as slow (the same)
Row = namedtuple("Row", "run step probe ratio")  # ratio: the step's time / that step's median in its run

runs = {}
for path in sys.argv[1:]:
    with open(path) as f:
        d = json.load(f)
    if "rounds" in d and "probe_before_sync" in d["rounds"][0]:
        runs[os.path.basename(path)[:-5]] = d
    else:
        print(f"(skipped {os.path.basename(path)}: not a step_timing.py result with the probe)")

print("run | rounds | force | sync S=1, S=2, S=3: mean (min to max) ms, S=2 / S=1 | pipelined S=1, S=2, S=3, S=2 / S=1 | probe")
for name, d in runs.items():
    s = d["summary"]
    sync = ", ".join(f"{s[f'sync_S{S}']['mean']:.2f} ({s[f'sync_S{S}']['min']:.2f} to {s[f'sync_S{S}']['max']:.2f})" for S in (1, 2, 3))
    pipe = ", ".join(f"{s[f'pipelined_S{S}']['mean']:.2f}" for S in (1, 2, 3))
    print(f"{name} | {len(d['rounds'])} | {' '.join(d['setup']['force']) or '-'} | {sync}, {s['sync_S2']['ratio']:.2f}x | "
          f"{pipe}, {s['pipelined_S2']['ratio']:.2f}x | {s['probe']['mean']:.3f} ({s['probe']['min']:.3f} to {s['probe']['max']:.3f})")

print("\nrun | split, the mean over rounds of each round's medians: build + encode + wait (ms) for S=1, S=2, S=3")
for name, d in runs.items():
    parts = {S: [stt.mean(r["split"][S][i] for r in d["rounds"]) for i in range(3)] for S in ("1", "2", "3")}
    print(f"{name} | " + ", ".join(f"S={S} {b:.2f} + {e:.2f} + {w:.2f}" for S, (b, e, w) in parts.items()))

rows = []
for name, d in runs.items():
    for k in ("1_first", "2", "3", "1_last"):
        med = stt.median(r["sync"][k] for r in d["rounds"])
        rows += [Row(name, k, r["probe_before_sync"][k], r["sync"][k] / med) for r in d["rounds"]]
pm = stt.median(x.probe for x in rows)
print(f"\n{len(rows)} sync measurements in {len(runs)} runs; the probe's median {pm:.3f} ms")
for label, group in ((f"probe more than {SLOW_PROBE}x its median", [x for x in rows if x.probe > SLOW_PROBE * pm]),
                     (f"probe at most {SLOW_PROBE}x its median", [x for x in rows if x.probe <= SLOW_PROBE * pm])):
    rel = [x.ratio for x in group]
    print(f"  {label}: {len(group)} measurements, step / its run's median {stt.mean(rel):.3f} on average, "
          f"{sum(r > SLOW_STEP for r in rel)} of them more than {SLOW_STEP}x")
print(f"measurements more than {SLOW_STEP}x their run's median (run, step, probe ms, ratio; * the probe was slow too):")
for x in sorted(rows, key=lambda x: -x.ratio):
    if x.ratio > SLOW_STEP:
        print(f"  {x.run}, S={x.step}, probe {x.probe:.3f}{' *' if x.probe > SLOW_PROBE * pm else ''}, {x.ratio:.2f}x")
