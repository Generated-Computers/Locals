# Measurements

How the speed numbers in the main README were measured, what else was measured,
and how to repeat it. Unless a section says otherwise, every number comes from a
MacBook Pro with an M2 Pro chip (19-core GPU) and 16 GB of memory, running
Qwen3.5-4B (`mlx-community/Qwen3.5-4B-MLX-4bit`) with mlx 0.32 and mlx-lm 0.32.
Only one GPU job ran at a time, and kernel tuning had finished before any timing
(a process that has just tuned its kernels is slower until the next start).

Part of the [technical documentation](../README.md). How the mechanisms named here
work is described in [Batch engine](../engine/README.md) and
[Prompt cache](../prompt-cache/README.md).

## Decode throughput by batch size

The table at the top of the main README. Sixteen short chat questions ("Explain
how a refrigerator works.", ...) were decoded at batch sizes 1, 2, 4, 8 and 16,
256 tokens each with the end-of-text token ignored, so every row does the same
work. Locals is its batch engine (`HybridBatchEngine`) in process; mlx-lm is its
`BatchGenerator`. The number is tokens written per second by all rows together.

| Rows at once | mlx-lm | Locals | Ratio |
|---|---|---|---|
| 1 | 61 | 64 | 1.0x |
| 2 | 82 | 116 | 1.4x |
| 4 | 87 | 160 | 1.8x |
| 8 | 87 | 271 | 3.1x |
| 16 | 157 | 341 | 2.2x |

The same test at 8 rows for the other tested models (tokens per second):

| Model | mlx-lm | Locals |
|---|---|---|
| Qwen3.5-4B | 87 | 271 |
| LFM2.5-2.6B | 144 | 404 |
| MiniCPM5-2B | 160 | 478 |
| Gemma 4 E4B | 80 | 224 |

Which of Locals' differences from mlx-lm produce these numbers, measured by
switching them off one at a time, is in [Why Locals is faster](../why-faster/README.md).

Dense 4-bit models gain 2.3x to 3.3x at 8 rows. A mixture-of-experts model in
the mxfp4 format (Mellum) gained nothing (1.0x): the tuned small-batch kernels
cover 4-bit affine layers in groups of 64 (`_tunable` in `locals/kernels.py`),
and mxfp4 layers stay on MLX's own kernels.

## On an M2 Max with 96 GB

The same test through the HTTP server on an Apple M2 Max (38-core GPU, 96 GB),
14 and 15 October 2026: 8 requests at once, 256 tokens each with the end-of-text
token ignored, the median of two rounds (tokens per second):

| Model | mlx-lm | Locals | Ratio |
|---|---|---|---|
| Qwen3.5-4B | 152 | 403 | 2.7x |
| Qwen3.5-9B | 72 | 220 | 3.0x |
| Qwen3.8-27B | 25 | 78 | 3.1x |

Bigger models gain more. mlx-lm itself is faster on this Mac than on the M2 Pro
(152 tokens per second for the 4B at 8 requests, against 87), so the 4B's ratio
is lower than its 3.1x there. Every batch size from 1 to 16, `locals bench`'s
numbers, memory and the script: [M2 Max measurements](m2-max.md).

## Serving benchmark with the InferenceX client

To check the numbers through the HTTP server with a client we did not write, we
used the benchmark client from SemiAnalysis's InferenceX
([github.com/SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX),
`infx/bench_serving/benchmark_serving.py`), unchanged, the way its fixed-length
runs use it: backend `vllm` (the `/v1/completions` endpoint), the `random`
dataset with `--random-range-ratio 0.8` (lengths drawn between 80% and 100% of the
target), `--ignore-eos`, `--request-rate inf` with `--max-concurrency` N, and
2 x N warmup requests. InferenceX sends 10 x N prompts per point; we sent
3 x N (at least 4) to fit the runs into one night. Each point ran against a fresh
`locals.server` on port 8341, with `--temp 0` and `LOCALS_DISK_CACHE=0`.

The metrics are InferenceX's: output throughput (tokens per second for all
requests), interactivity (1000 / median time per output token, tokens per second
for one user), time to first token (TTFT), time per output token (TPOT) and the
gaps between streamed tokens (ITL).

### 1,000 tokens in, 1,000 out

| Requests at once | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| stock mlx-lm engine, tokens/s | 52 | 66 | 70 | 69 |
| Locals, tokens/s | 60 | 98 | 125 | 177 |
| Locals, tokens/s per user | 65.6 | 53.9 | 33.9 | 24.4 |
| Locals, median TTFT | 1.40 s | 1.43 s | 1.38 s | 1.56 s |

The stock engine is mlx-lm's `BatchGenerator` behind the same server (model ids
`mlx/...`), as Locals configures it (prefill in batches of 4 prompts and 512-token
chunks, which keeps its logits in memory).

### 8,000 tokens in, 1,000 out, with and without mixed passes

When a request with a long prompt arrives while others are decoding, Locals now
computes the running rows' next tokens in the same passes as the new prompt
(mixed passes, `LOCALS_MIX_CHUNK`). Before, every running row waited until the
whole prompt was read.

| Requests at once | 2 | 4 | 8 |
|---|---|---|---|
| tokens/s, waiting for the prefill | 44.1 | 52.3 | 57.1 |
| tokens/s, mixed passes | 43.8 | 49.9 | 57.4 |
| longest gaps (p99.9 ITL), waiting | 0.26 s | 11.3 s | 12.0 s |
| longest gaps (p99.9 ITL), mixed passes | 1.6 s | 2.9 s | 3.1 s |

Throughput is the same within noise; what changes is that no stream stops for
the 12 seconds an 8,000-token prompt takes to read. At 2 requests the "waiting"
run had too few long-prompt arrivals for a 12 s gap to reach the p99.9 of its
token gaps. At 8 requests the delay is spread over all streams: the median time
per token rose (98 ms to 122 ms) while its p99 fell (178 ms to 148 ms).

## Where the time goes

**Decode steps.** One decode step at about 1,000 tokens of context took 25.7 ms
for 4 rows (156 tokens/s) and 31.8 ms for 8 rows (252 tokens/s). Reading the
4-bit weights once costs about 11 ms at this chip's memory bandwidth, so a step
is not bound by the weights alone; at one row the MLPs take 7.8 of 17.6 ms.

**Prefill.** Reading a prompt is compute bound: about 700 to 740 tokens per
second at the start of a prompt, whatever the chunk length between 960 and 2,048
tokens (one MLP costs 25 microseconds per row, linear in the rows). Smaller chunks
are slower: 394 tokens/s at 16 tokens per pass, 512 at 32, 562 at 64, 627 at 256,
684 at 512. Prompts arriving together are prefilled in one batch; with 1,000-token
prompts and 8 requests at once, prefill took 18% of the server's time and
decoding 78%.

**Prefill and decoding do not overlap.** We ran a prefill on its own MLX stream
while the batch decoded on the default stream: both together took as long as one
after the other (0.98x to 0.99x). The GPU does not run the two at the same time,
so the only way to keep rows decoding during a prefill is to put both in the same
passes. In such a mixed pass the decode rows add almost nothing (-8 to +4 ms
next to a 420 to 1,450 ms prompt chunk) when the pass's rows (decode rows plus
prompt tokens) fill whole matmul tiles; a few rows past a tile cost 60 to 120 ms
more. In our timings the boundaries were multiples of 32 rows, and of 64 from
128 rows on, where Locals switches to a dense matmul (`PREFILL_M`). Locals sizes mixed passes at 1,024 rows and uses
them only for prompts longer than one pass.

**Memory of a prefill.** One prefill forward of 8 rows x 1,024 tokens needed about
4 GB of activations on this model; 16 rows x 1,024 ran the GPU out of memory. Capping the rows
times the chunk length at 2,048 tokens keeps it near 1 GB (8 x 256 or 16 x 128)
with the same prefill rate (750 tokens/s).

## Long contexts

One growing prompt, prefilled through the engine in chunks, with the decode speed
of one row measured at each length:

| Context | Prefill rate in that stretch | Decode | Peak GPU memory |
|---|---|---|---|
| 16k | 611 tokens/s | 55 tokens/s | 4.46 GB |
| 32k | 439 | 46 | 4.62 GB |
| 64k | 297 | 36 | 5.50 GB |
| 96k | 217 | 29 | 6.49 GB |
| 128k | 168 | 24 | 7.62 GB |
| 160k | 141 | 21 | 8.74 GB |

Qwen3.5-4B keeps 32 KB of attention cache per token (8 of its 32 layers use full
attention; the other 24 keep a fixed recurrent state), so memory grows by about
1 GB per 32,000 tokens. The server's automatic limit for this model on 16 GB is
145,408 tokens (`service.context_fit`).

Through the server, a fact placed in the middle of a document made of the
repository's own source files was found at every length tried:

| Prompt tokens | Time to first token | Found |
|---|---|---|
| 28,814 | 50 s | yes |
| 59,324 | 133 s | yes |
| 110,749 | 348 s | yes |
| 128,761 | 446 s | yes |
| 139,387 | 501 s | yes |

A request of 158,012 tokens was refused with `context_length_exceeded`.

**Follow-up questions.** The second turn of a conversation used to read the whole
conversation again, because the chat template writes the earlier reply
differently from the turn that produced it. With the reply checkpoint
(`LOCALS_REPLY_CHECKPOINT`, on by default) it continues from where that reply
started:

| Conversation | Second turn before | Second turn now | Tokens from the cache |
|---|---|---|---|
| 60,564 tokens | 137.4 s | 1.5 s | 60,560 |
| 138,352 tokens | about 500 s (reading it again) | 2.4 s | 138,348 |

## Multi-turn conversations

Four conversations at once, each a series of turns: a shared system prompt, a
growing history, a tool-result-like message of 300 to 3,000 tokens each turn, a
reply of 64 to 768 tokens, and 0.2 to 2 seconds between turns, until the context
passed 24,000 tokens. The history carried the server's own replies, as coding
agents send them. Four minutes per run:

| | Reply checkpoint off | On |
|---|---|---|
| Turns finished | 23 | 36 |
| Output tokens/s | 39.6 | 64.2 |
| Prompt tokens from the cache | 28% | 83% |
| Median time to first token | 10.6 s | 5.2 s |

## Agent workloads

Sixteen coding agents that call tools, on Qwen3.5-2B, finished in 29 seconds with
Locals and 75 with mlx-lm. Forty programming problems from LiveCodeBench, solved
eight at a time, took 8 minutes instead of 20; single streams were within 5% of
each other. On four tasks that each span several files, `locals run` (a team of
agents, see [Team of agents](../agents/README.md)) passed all eight attempts in
about three minutes each, and a single agent in the same setup passed four of
eight in about five minutes each.

## Repeating the serving benchmark

Start a server on a spare port and run the InferenceX client against it, from a
checkout of InferenceX (`inferencex-e2e/`) in a Python environment with `aiohttp`,
`numpy`, `tqdm` and `transformers`:

```bash
LOCALS_DISK_CACHE=0 python -m locals.server --port 8341 --temp 0
```

```bash
python -m infx.bench_serving.benchmark_serving \
    --model mlx-community/Qwen3.5-4B-MLX-4bit --served-model-name locals/qwen3.5-4b \
    --backend vllm --base-url http://127.0.0.1:8341 --endpoint /v1/completions \
    --dataset-name random --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.8 \
    --num-prompts 24 --max-concurrency 8 --request-rate inf --ignore-eos \
    --num-warmups 16 --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir results
```

For the stock engine, start the server with `--engines mlx --preload mlx` and use
`--served-model-name mlx/qwen3.5-4b`. Restart the server between points, so the
prompt cache of one point does not serve the next.
