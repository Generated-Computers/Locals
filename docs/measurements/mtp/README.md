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
