# A 2-token decode pass: what it costs, and how steady it is

The measurements behind [issue #22](https://github.com/kaziko/Locals-Qwen-Vision/issues/22). In the three runs of
`mtp_test.py` in [`../mtp`](../mtp/README.md), a main-model pass over 2 tokens took 12.57 and 13.53 ms on
`mlx-community/Qwen3.5-4B-MLX-4bit` and 14.05 ms on `Kaziko/Qwen3.5-4B-Zwyrol` (1.23x to 1.39x of a 1-token pass),
while the 1- and 3-token passes stayed put. Here both models are measured again, with more rounds, and the pass is
taken apart.

Setup: M2 Max (`applegpu_g14s`, 32 GB; 30 GPU cores, from `system_profiler SPDisplaysDataType`), macOS 27.0, the
repo's `.venv` (mlx 0.32.3, mlx-lm 0.32.0), commit `cdc605c` with `locals/` unchanged, 2026-11-06 and 07. One GPU job
at a time: before each run, `lsof` found nothing listening on port 8321 (checked by hand; the result files do not
record it). Every run read a copy of this Mac's kernel-tuning file (`LOCALS_TUNE_CACHE`), so a run could not change the
real one. That file has no entry for Qwen3.5-4B's matmul shapes, so every kernel choice came from the preset shipped
for this GPU architecture (`locals/presets/tune_v6_applegpu_g14s.json`), the same as without the copy.

## What the scripts do

`step_timing.py` loads the model with `Runtime.load` (as `mtp_test.py` does), prints the kernel each 4-bit matmul shape
gets at 1, 2 and 3 rows, fills a cache with the same prompt and greedy reply as `mtp_test.py`'s timing (167 tokens for
Zwyrol, 191 for the original model), and then, in each round:

- **sync**: 1, 2, 3 and 1 tokens, timed as `mtp_test.py` times them: a fresh cache, 3 warm steps, 30 timed steps,
  one `mx.eval` a step. This is how a lone request's draft pass runs: the next step waits for this one's result.
- **probe**: before each sync measurement (ahead of its fresh cache and warm steps), 30 synchronous calls of MLX's
  own 4-bit matmul (`mx.quantized_matmul`, no Locals code) on the output head's weights, 1 row. It tells how fast the
  GPU is at that moment (for 35 ms of the sync measurement's 300 to 500).
- **split**: the same steps, each cut into building the graph (Python), encoding it (`mx.async_eval` returns) and
  waiting for the GPU (`mx.eval` returns). The median of 30 steps per round.
- **pipelined**: step s+1 built and queued while step s runs, as the engines decode a batch. The CPU's time hides
  behind the GPU's, so this is the GPU's own time for the pass.

`--force M=FAMILY` gives every shape at M rows that kernel family's tuned winner (`mlx`: MLX's own kernel), and
`--force M@NxK=FAMILY` only the shape N x K. The choice holds in that process only.

`summarize.py` prints the tables below from the result files. `hold_memory.py` is a neighbour process that holds GPU
memory and runs no GPU work.

## Commands

Run from this folder. `<repo>` is the clone's folder, `<scratch>` a folder outside it; in the result files they stand
for the paths the runs printed.

```
cp ~/.cache/locals/tune_v6_applegpu_g14s.json <scratch>/tune/      # the copy the runs read and write
export LOCALS_DISK_CACHE=0 LOCALS_TUNE_CACHE=<scratch>/tune/tune_v6_applegpu_g14s.json
<repo>/.venv/bin/python step_timing.py Kaziko/Qwen3.5-4B-Zwyrol <scratch>/runs/base_p1.json > <scratch>/runs/base_p1.txt 2>&1
<repo>/.venv/bin/python step_timing.py mlx-community/Qwen3.5-4B-MLX-4bit <scratch>/runs/orig_1.json > <scratch>/runs/orig_1.txt 2>&1
<repo>/.venv/bin/python step_timing.py Kaziko/Qwen3.5-4B-Zwyrol <scratch>/runs/head_fragh_p1.json --force 2@248320x2560=fragh > <scratch>/runs/head_fragh_p1.txt 2>&1
<repo>/.venv/bin/python step_timing.py Kaziko/Qwen3.5-4B-Zwyrol <scratch>/runs/all2_mlx.json --rounds 3 --force 2=mlx > <scratch>/runs/all2_mlx.txt 2>&1
<repo>/.venv/bin/python summarize.py results/*.json
```

The runs wrote to `<scratch>/runs/` and were copied into `results/` with the paths replaced and these names:

| In `results/` | Run as | How |
|---|---|---|
| `unchanged_1` to `_3` | `base_p1` to `_3` | the first command, three times |
| `original_1` to `_3` | `orig_1` to `_3` | the second command, three times |
| `head_fragh`; `head_mlx`, `head_frag`, `head_mma` | `head_fragh_p1`; `head_mlx`, ... | the third; the others with that family and `--rounds 3` |
| `rows2_mlx`, `rows2_fragh`, `rows2_frag`, `rows2_mma` | `all2_mlx`, ... | the fourth, with that family |
| `neighbour_12gb` | `hold12_1` | `<repo>/.venv/bin/python hold_memory.py 12 300 &` (its PID from `$!`; its output in `neighbour_12gb_holder.txt`), then the first command once it printed `holding 12.0 GB`, then that PID stopped |
| `mtp_test_rerun` | `mtp_repro_1` | `mtp_test.py` unchanged, from `../mtp`, with the environment above: `<repo>/.venv/bin/python mtp_test.py <scratch>/mtp_bf16.safetensors <scratch>/runs/mtp_repro_1.json Kaziko/Qwen3.5-4B-Zwyrol > <scratch>/runs/mtp_repro_1.txt 2>&1` |
| `early_unchanged_1` to `_3`, `early_head_fragh` | `before_1` to `_3`, `head_fragh_1` | the first and third commands, with `step_timing.py` before the probe was added |

Each `.txt` starts with its setup lines and its `force:` line. After the runs, `step_timing.py` changed only by a lint
pass (`check=False` on its two `git` calls, `with open` for the JSON), two comments, and a cleanup that keeps every
call and its order: how a round's and the summary's numbers are stored and printed (the same JSON and lines), and the
output head taken with `K.head_module` (the same module for both models). A review then added two checks of
`--force`. A spec the script cannot read, or one that forces no shape, now stops the run. And a shape's key takes the
activation dtype (`rt.dtype`, the one `K._choose` keys a choice on), not the scales' dtype. On Zwyrol, the model of
every forced run, the two are the same: Runtime printed `compute dtype float16` (`mtp_test_rerun.txt`), and
`set_compute_dtype` casts every bf16 array to that dtype, the scales included.

## Results

### The unchanged code: the noise floor

Mean over 5 rounds (lowest to highest round), ms. For 1 token, each round counts the lower of its first and last
measurement. From `summarize.py`:

| Run | Sync, 1 token | Sync, 2 tokens | Sync, 3 tokens | 2 / 1 | Pipelined, 1 / 2 / 3 tokens | 2 / 1 | Probe |
|---|---|---|---|---|---|---|---|
| `unchanged_1` (Zwyrol) | 10.16 (10.02 to 10.44) | 12.09 (11.96 to 12.22) | 15.82 (15.52 to 16.07) | 1.19x | 7.82 / 8.86 / 11.93 | 1.13x | 1.148 |
| `unchanged_2` (Zwyrol) | 10.28 (10.06 to 10.85) | 12.25 (11.78 to 12.72) | 15.98 (15.53 to 16.65) | 1.19x | 7.78 / 8.82 / 12.10 | 1.13x | 1.158 |
| `unchanged_3` (Zwyrol) | 10.34 (10.07 to 10.79) | 12.20 (11.96 to 12.43) | 15.94 (15.50 to 16.16) | 1.18x | 7.88 / 8.93 / 11.84 | 1.13x | 1.170 |
| `original_1` | 10.46 (10.05 to 11.33) | 12.20 (12.04 to 12.38) | 16.04 (15.52 to 16.71) | 1.17x | 7.89 / 8.88 / 11.99 | 1.13x | 1.165 |
| `original_2` | 10.88 (10.68 to 11.38) | 12.91 (12.58 to 13.15) | 16.86 (16.34 to 17.57) | 1.19x | 8.12 / 9.12 / 12.14 | 1.12x | 1.200 |
| `original_3` | 10.52 (10.13 to 10.98) | 12.39 (12.19 to 12.74) | 16.56 (15.84 to 18.14) | 1.18x | 7.87 / 9.01 / 12.27 | 1.15x | 1.160 |

On Zwyrol the 2-token pass is steady: its run means differ by 0.16 ms in sync (12.09 to 12.25) and 0.11 ms pipelined
(8.82 to 8.93). On the original model they differ by 0.71 ms in sync (12.20 to 12.91), but `original_2` is slower at
1 and 3 tokens too, and its 2 / 1 ratio, 1.19x, is the others'. On both models a 2-token pass costs 1.17x to 1.19x of
a 1-token pass in sync and 1.12x to 1.15x on the GPU (pipelined); a 3-token pass 1.53x to 1.57x in sync.

The issue's runs had 2 / 1 ratios of 1.23x (12.57 ms), 1.34x (13.53) and 1.39x (14.05). None of these 6 runs came near
that. The early runs agree: `early_unchanged_1` to `_3` (Zwyrol) gave 12.20, 12.14 and 12.03 ms (1.18x to 1.19x).
`mtp_test.py` itself, unchanged, run again on Zwyrol (`mtp_test_rerun.txt`): 1 token 10.34 and 10.36 ms, 2 tokens
12.26 ms, 3 tokens 15.89 ms, against 14.05 ms in `../mtp/results/zwyrol.txt`. (For issue #18: with 12.26 ms, that
run's estimate for one MTP draft is 1.36x with the 4-bit layer, against 1.19x in `zwyrol.txt`.)

Where a sync pass's time goes (`split`, `unchanged_1`; the mean over 5 rounds of each round's median): 1 token 1.36 ms
building the graph + 1.34 encoding + 7.57 waiting; 2 tokens 1.60 + 1.31 + 9.10; 3 tokens 1.56 + 1.55 + 12.39. A
2-token pass costs 1.5 ms more GPU wait and 0.2 ms more Python than a 1-token pass.

### The kernel each matmul gets

The same in every run of a model (each `.txt` prints it):

| Shape (layers) | 1 row | 2 rows | 3 rows |
|---|---|---|---|
| 2560x4096 (32, `o_proj`) | `mlx` | `multi` (8, 4, 2) | `v3` (8, 4, 2) |
| 2560x9216 (32, `down_proj`) | `mlx` | `multi` (8, 4, 2) | `v3` (16, 4, 2) |
| 10240x2560 (8, `qkv_proj`) | `mlx` | `multi` (8, 4, 2) | `v3` (8, 4, 2) |
| 12352x2560 (24, `in_proj_all`) | `mlx` | `multi` (8, 4, 2) | `v3` (8, 8, 2) |
| 18432x2560 (32, `gate_up_proj`) | `mlx` | `multi` (8, 4, 2) | `v3` (8, 8, 2) |
| 248320x2560 (the output head) | `mlx` | `multi` (8, 4, 2) | `v3` (8, 8, 2) |

The output head is Zwyrol's own `lm_head` (its `embed_tokens` has the same shape, so its `.txt` counts 2), and the
original model's `embed_tokens`, used as the head. At 2 rows every shape takes `multi`. The preset lists 2 rows among
the row counts whose whole decode step it timed for this model (`__model_tuned_pipe__2a38aeaf5f`: 2 to 32). Whatever
Mac those timings came from, here each other family was tried in the model on Zwyrol (pipelined 2-token pass, ms;
`multi` 8.82 to 8.93 above):

| Forced at 2 rows | Only the output head | Every shape |
|---|---|---|
| `mlx` | 9.35 (`head_mlx`) | 11.49 (`rows2_mlx`) |
| `fragh` | 9.55 (`head_fragh`) | 15.90 (`rows2_fragh`) |
| `frag` | 10.70 (`head_frag`) | 14.81 (`rows2_frag`) |
| `mma` | 10.12 (`head_mma`) | 17.59 (`rows2_mma`) |

Each is slower than `multi`: none of the kernel families tuned for 2 rows beats it on this Mac.

### Slow moments hit every pass

In 15 runs `summarize.py` counts 244 sync measurements. It divides each step's time by that step's median in its run,
and marks the probe slow when it is more than 1.25x its median (1.160 ms). Both lines are chosen here, not given by
anything.

| Probe before | Measurements | Step / its run's median, on average | More than 1.10x |
|---|---|---|---|
| Slow (more than 1.25x) | 8 | 1.190 | 4 |
| Not slow | 236 | 1.008 | 8 |

The 12 measurements more than 1.10x their run's median are 1-token passes (8), 2-token passes (2) and 3-token passes
(2), in `head_frag`, `head_mma`, `head_fragh`, `original_1` and `original_3`. The worst two: a 2-token pass at 1.57x
(`head_mma`, probe 1.444 ms, just under the line) and a 1-token pass at 1.53x (`head_frag`, probe 2.295 ms: MLX's own
matmul, with no Locals code in it, was twice as slow at that moment). So a slow moment is not a 2-token cost. The
probe catches 4 of the 12; it covers only 35 ms, taken before each measurement's fresh cache and warm steps.

`early_head_fragh`, the run without the probe, had such moments too: its 2-token passes ranged 12.51 to 15.37 ms, and
its first 1-token pass in rounds 1 to 4 took 11.91 to 12.57 ms, against 9.98 to 10.50 for the last.

A neighbour process holding 12 GB of GPU memory and doing no GPU work (`neighbour_12gb`) left the 2-token pass where
the unchanged runs put it: 12.07 ms sync, 8.92 ms pipelined.

## What this says

- On this Mac with the code unchanged, a 2-token pass costs 1.17x to 1.19x of a 1-token pass in sync and 1.12x to
  1.15x on the GPU, on both models, in 6 runs (and 1.18x to 1.19x in the 3 early runs). The issue's 1.34x and 1.39x
  did not come back.
- Every run of a model gives every shape the same kernel at 2 rows, from the shipped preset, and nothing re-times it.
  Every other kernel family is slower in the model here.
- The slow moments these runs saw hit 1-, 2- and 3-token passes, and at some of them MLX's own matmul was slow too.
  They are not a 2-token cost.
- What slowed the 2-token passes in two of the issue's runs is not found. Those runs differ from these in one known
  way: the owner's locals server on port 8321 (another install, mlx 0.32.4) was loaded beside them. Its log
  (`~/.locals/server.log`) was last written at 22:20:07, by a request; it was still running at the gap test
  (`../mtp/README.md`, committed at 23:23), so between those times it neither restarted nor finished a request. The
  issue's runs ended at 22:20:47 (12.57 ms), 22:33:33 (13.53 ms) and 22:35:02 (14.05 ms): the file times (`stat`) of
  their logs `run.log`, `run_orig12.log` and `run_zwyrol.log` in that session's scratch folder. A running server was
  not repeated here, because only one GPU job runs at a time on this Mac.
- Nothing in `locals/` was changed.

## Also seen

- **Two MLX versions share one tuning file** ([issue #24](https://github.com/kaziko/Locals-Qwen-Vision/issues/24)).
  The owner's server (mlx 0.32.4) and this repo's `.venv` (mlx 0.32.3) both use
  `~/.cache/locals/tune_v6_applegpu_g14s.json`, and a start under one version marks the shapes timed under the other
  as stale, so an idle server times them again. Not a cause here: the file has no Qwen3.5-4B shapes.
- **Building the graph of a custom kernel costs more Python.** At 2 rows, a pass whose matmuls all take MLX's kernel
  builds its graph in 1.33 ms (`rows2_mlx`), against 1.60 to 1.86 ms with `multi` (the unchanged runs). In a sync pass
  this adds to the time; pipelined, it hides behind the GPU.
