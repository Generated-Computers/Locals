# MTP drafts: Qwen3.5's own draft layer

The measurements behind [issue #18](https://github.com/kaziko/Locals-Qwen-Vision/issues/18). Run on an M2 Max
(30-core GPU, 32 GB), 2026-11-06, at commit `f31121c` (its `locals/` is the same as `0e40a3b`'s), with the repo's
`.venv`: mlx 0.32.3, mlx-lm 0.32.0. This Mac's kernel choices were timed on mlx 0.32.4.

## What it measures

Qwen3.5-4B's checkpoint has one multi-token prediction (MTP) layer: from the main model's hidden state at
position i and the embedding of token i+1, it predicts token i+2. The 4-bit MLX files
(`mlx-community/Qwen3.5-4B-MLX-4bit` and `Kaziko/Qwen3.5-4B-Zwyrol`) do not have it.

- `download_mtp.py` fetches only the layer's 15 `mtp.*` tensors (241.2 MB, BF16) from `Qwen/Qwen3.5-4B`, with
  HTTP range requests, and saves them as one safetensors file.
- `mtp_test.py` loads the main model with `Runtime.load` (float16, fused layers, small-batch kernels, as the
  server runs it on this Mac) and writes a greedy reply, thinking off, at most 192 tokens, to each of 12 prompts.
  Then, teacher-forced over prompt + reply, it counts how often the layer's guess equals the main model's
  greedy token. It prints first that the main model, teacher-forced, gives its own greedy reply back.
- It scores 8 wirings: norm weights used as w or as 1+w, the hidden state before or after the final norm, and
  the order of the concatenation. For the best one, it runs a float16 and a 4-bit layer, and a second guess
  (the layer fed its own output again over the same sequence: an approximation of a chained draft).
- Last, it times a main step of 1, 2 and 3 tokens and one guess (the layer, the output head, argmax), with one
  `mx.eval` per step and no pipelining. The 1-token step is timed first and last, as the noise floor.

## Commands

Run from this folder, one model at a time (never two GPU jobs at once):

```
<repo>/.venv/bin/python download_mtp.py mtp_bf16.safetensors       # 241 MB; *.safetensors is ignored here
LOCALS_DISK_CACHE=0 <repo>/.venv/bin/python mtp_test.py mtp_bf16.safetensors results/zwyrol.json Kaziko/Qwen3.5-4B-Zwyrol > results/zwyrol.txt 2>&1
LOCALS_DISK_CACHE=0 <repo>/.venv/bin/python mtp_test.py mtp_bf16.safetensors results/original.json mlx-community/Qwen3.5-4B-MLX-4bit > results/original.txt 2>&1
```

`<repo>` is the clone's folder; in the result files it stands for the path the run printed.

## Results

| | Zwyrol (`zwyrol.txt`) | Original (`original.txt`) |
|---|---|---|
| Main model gives its own greedy reply back | 1,633 of 1,635 | 1,685 of 1,690 |
| Best wiring | norms 1+w, hidden after the final norm, `[embedding, hidden]` | the same |
| First guess right, float16 layer | 83.7% (1,358 / 1,623) | 85.7% (1,438 / 1,678) |
| First guess right, 4-bit layer | 82.7% (1,343 / 1,623) | 85.4% (1,433 / 1,678) |
| Second guess right, given the first (4-bit) | 73.5% (979 / 1,332) | 74.8% (1,063 / 1,421) |
| The same, with the hidden state before the final norm (float16) | 82.0% | 84.5% |
| The six other wirings (float16) | 0.0% to 0.2% | 0.0% to 0.5% |
| Main step, 1 token (first, last) | 10.20, 10.14 ms | 10.47, 10.06 ms |
| Main step, 2 tokens | 14.05 ms | 13.53 ms |
| Main step, 3 tokens | 15.91 ms | 15.53 ms |
| One guess, float16 / 4-bit layer | 2.02 / 1.57 ms | 2.02 / 1.58 ms |

The by-prompt rates are in each result file. `results/original_10_prompts.*` is the first run, made with an earlier
version of `mtp_test.py`: the original model only, 10 prompts (without the two Polish tool-call prompts), and
the draft logits from the tied embedding (the script now takes the model's output head, Zwyrol's own
`lm_head`). In that run: first guess right 85.2% (float16) and 85.0% (4-bit); steps of 1, 2 and 3 tokens
10.35, 12.57 and 15.65 ms.

The 2-token step moved from run to run (1.23x to 1.39x of a 1-token step over the three runs) while the 3-token
step stayed at 1.53x to 1.57x. The "estimate" lines at the end of each result file use that run's own times; issue #18
has the estimate table and its assumptions.

## Gap test: the layer's KV with positions missing

`mtp_gap_test.py` asks how much the hit rate drops when the layer's own KV lacks positions: after a prompt-cache
hit (the main model's cache comes back, the layer's KV does not) and after steps the layer did not run. Same Mac
and `.venv`, 2026-11-06, at commit `6c972cc` (`locals/` the same as `f31121c`'s), Zwyrol, the best wiring, 4-bit
layer. The server on port 8321 was running beside it, idle; this test times nothing.

Teacher-forced over prompt + reply, as above. A missing key position is masked out of the layer's attention (each
query still sees itself), and the real positions are kept. 14 prompts: the 12 above, and two long ones, the first
5,000 characters of `locals/lookup.py` and the first 6,000 of `docs/engine/README.md` at that commit (the prompt
changes when those files change).

```
LOCALS_DISK_CACHE=0 <repo>/.venv/bin/python mtp_gap_test.py mtp_bf16.safetensors results/gap_zwyrol.json Kaziko/Qwen3.5-4B-Zwyrol > results/gap_zwyrol.txt 2>&1
```

First draft right (`results/gap_zwyrol.txt`):

| What the layer sees | All reply tokens | First 16 reply tokens | Reply tokens 48 on |
|---|---|---|---|
| Every earlier position (`full`) | 81.8% (1,640 / 2,005) | 87.1% (195 / 224) | 80.8% (1,106 / 1,369) |
| No prompt position before the last one (`no-prompt`) | 76.6% (1,535 / 2,005) | 73.2% (164 / 224) | 77.3% (1,058 / 1,369) |
| No reply position 16 to 47 (`gap32`) | 81.1% (1,626 / 2,005) | 87.1% (195 / 224) | 80.6% (1,104 / 1,369) |

- Checks: `full` with a boolean mask gives the same counts as with the `causal` mask (1,640 / 2,005), and the 12
  prompts of `mtp_test.py` give 1,343 / 1,623 again, as in `results/zwyrol.txt`.
- By group, from `results/gap_zwyrol.json`: the 3 tool calls 96.3% (104 / 108) with every position and 76.9%
  (83 / 108) without the prompt; the 2 long prompts 77.7% (297 / 382) and 70.2% (268 / 382).
- The tool-call replies (31 to 40 tokens) end before token 48, so `gap32` says nothing about them.

## Step 4: MTP drafts in the server, one request

Issue #18's step 4, with the engine at commit `f7efec9` (MTP drafts behind `LOCALS_MTP`, `locals/mtp.py`). Same M2 Max
(30-core GPU, 32 GB), mlx 0.32.3, mlx-lm 0.32.0, Zwyrol, 2026-11-07. `step4.sh` starts a server of its own from this
checkout on port 8399 (`LOCALS_DISK_CACHE=0`), checks that its `/health` build is the checkout's (`0fd1cbbd8ed2` in
every run), runs the decode sweep of `docs/measurements/m2-max.md`'s appendix at 1 request (`decode_sweep.py`, with
the port as a parameter: 256 tokens, end of text ignored, greedy, a warm-up and 2 rounds) and
`locals bench --agents --n 1` (a warm-up and 2 rounds, the server's temperature 0.2), saves `/stats`, and stops only its
own server. Nothing else ran on the GPU. The unchanged server (MTP off) ran twice first, as the noise floor.

```
bash step4.sh off-a && bash step4.sh off-b && bash step4.sh on1 1 && bash step4.sh on2 2
```

Each round on its own line (`results/step4/<label>_sweep.txt` and `_agents.txt`):

| Run | Decode sweep, 1 request | `bench --agents --n 1` (41 tokens, 11 forced) |
|---|---|---|
| MTP off (`off-a`) | 110.2, 110.1 tok/s | 51, 52 tok/s |
| MTP off (`off-b`) | 110.8, 109.7 tok/s | 51, 52 tok/s |
| MTP on, 1 draft (`on1`) | 108.6, 108.2 tok/s | 54, 51 tok/s |
| MTP on, 2 drafts (`on2`) | 107.5, 109.3 tok/s | 50, 50 tok/s |

Drafts over each whole run (warm-ups included), from `results/step4/<label>_stats.json`:

| Run | MTP drafts kept | Prompt-lookup drafts kept | Steps |
|---|---|---|---|
| `off-a` | | 33 of 85 | 841 |
| `on1` | 293 of 348 (84.2%) | 39 of 72 | 536 |
| `on2` | 360 of 567 (63.5%) | 36 of 93 | 474 |

- Live, the first draft is kept as often as the teacher-forced rate above says (84.2% against 82.7%), and the steps
  fall from 841 to 536. The time does not: a `_solo` pass reads its draft back to the CPU before it runs and its
  picks after, so each pass waits for the CPU twice, and the pipelining of plain steps is lost. The issue's estimate
  for this case ("full output head, each step waits for the CPU") was 1.00x for 1 draft; measured, 0.98x.
- Greedy replies (`greedy_check.py`, in process, the 14 prompts of `tests/test_mtp_model.py`, at most 192 tokens):
  11 are the same with MTP drafts, 3 are not (`results/greedy_check.txt`). Each of the 3 first differs where the two
  tokens' logits, teacher-forced, are equal or one float16 step apart (24.062 / 24.078; 30.297 / 30.297; 23.641 /
  23.625): a pass over several tokens rounds differently from a one-token step. In that run the MTP drafts were kept
  629 of 755 times (83.3%) with 1 draft, 844 of 1,197 (70.5%) with 2.
