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

## Against Ollama and Magnitude

The same InferenceX client against three servers on the same M2 Pro, one at a
time on the GPU: Locals, Ollama 0.40.5 and Magnitude 0.2.9, each with its own
4-bit build of the model:

| | Qwen3.5-4B | Gemma 4 E4B |
|---|---|---|
| Locals | `mlx-community/Qwen3.5-4B-MLX-4bit` | `mlx-community/gemma-4-e4b-it-4bit` |
| Ollama | `qwen3.5:4b` (its MLX build, NVFP4 weights) | `gemma4:e4b-it-qat` (GGUF, Q4_0) |
| Magnitude | `qwen3.5-4b:gguf:q4` | `gemma-4-e4b-it-qat:gguf:q4` |

Chat completions (backend `openai-chat`), the client's `random` prompts (seed
0, lengths 80% to 100% of the target), temperature 0, thinking off
(`reasoning_effort: none`), 3 x N requests per point (2 x N with 8,000-token
prompts) after 2 x N warmup requests. Replies end where the model ends them:
Ollama does not support `ignore_eos` and Magnitude refuses it, so Locals ran
without it too. Each engine runs a different build of the model, so reply
lengths still differ; the time to the first token and the time per output token
do not depend on them, and output tokens per second are given with the average
reply length. The client ran unchanged except for what the other two servers
needed: `max_tokens` sent beside `max_completion_tokens` (Ollama reads only
`max_tokens`), `ignore_eos` and `max_completion_tokens` left out for Magnitude
(it refuses fields it does not know), and SSE comment lines skipped (Magnitude's
keepalive comments stopped the client). Ollama and Magnitude were measured on
16 and 17 October 2026, Locals on 18 October (commit `26f2482`).

### Qwen3.5-4B, 1,000 tokens in

| Requests at once | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| First token, Locals | 1.4 s | 1.5 s | 1.4 s | 1.6 s |
| First token, Ollama | 2.1 s | 9.4 s | 18.9 s | 41.8 s |
| First token, Magnitude | 1.4 s | 2.7 s | 3.3 s | 16.1 s |
| Time per token, Locals | 16 ms | 19 ms | 36 ms | 58 ms |
| Time per token, Ollama | 21 ms | 21 ms | 21 ms | 22 ms |
| Time per token, Magnitude | 24 ms | 35 ms | 79 ms | 109 ms |
| Tokens/s (reply length), Locals | 47 (259) | 79 (387) | 86 (362) | 111 (330) |
| Tokens/s (reply length), Ollama | 33 (171) | 36 (288) | 33 (220) | 34 (236) |
| Tokens/s (reply length), Magnitude | 37 (353) | 50 (666) | 45 (497) | 42 (405) |

Ollama answers one request at a time with this model: each reply is written at
full speed (21 ms a token) while the others wait, so the first token comes later
the more requests there are. Locals' numbers are lower than the fixed-length ones
above (111 against 177 tokens per second at 8 requests) because replies end at
different times, so the batch is not always full, and a new prompt is read each
time one ends.

### Qwen3.5-4B, 8,000 tokens in

| Requests at once | 1 | 4 |
|---|---|---|
| First token, Locals | 10.9 s | 17.6 s |
| First token, Ollama | 16.4 s | 72.5 s |
| First token, Magnitude | 11.5 s | 124.5 s |
| Time per token, Locals | 18 ms | 77 ms |
| Time per token, Ollama | 23 ms | 23 ms |
| Time per token, Magnitude | 26 ms | 120 ms |
| Tokens/s (reply length), Locals | 18 (282) | 30 (470) |
| Tokens/s (reply length), Ollama | 14 (270) | 15 (317) |
| Tokens/s (reply length), Magnitude | 24 (399) | 3.5 (143) |

With 4 requests at once, 4 of Magnitude's 8 replies came back empty; Locals and
Ollama answered all of them. For one request, Magnitude's higher tokens per
second come from its longer replies (399 tokens against 282): its first token
and its time per token are both slower than Locals'.

### Gemma 4 E4B, 1,000 tokens in

| Requests at once | 1 | 4 | 8 |
|---|---|---|---|
| First token, Locals | 0.9 s | 0.9 s | 1.1 s |
| First token, Ollama | 1.6 s | 2.2 s | 2.0 s |
| First token, Magnitude | 0.8 s | 3.8 s | 12.7 s |
| Time per token, Locals | 20 ms | 39 ms | 57 ms |
| Time per token, Ollama | 21 ms | 80 ms | 98 ms |
| Time per token, Magnitude | 18 ms | 53 ms | 101 ms |
| Tokens/s (reply length), Locals | 41 (187) | 81 (225) | 105 (208) |
| Tokens/s (reply length), Ollama | 35 (169) | 38 (171) | 58 (246) |
| Tokens/s (reply length), Magnitude | 50 (220) | 55 (183) | 45 (165) |

For one Gemma request alone Magnitude is a little faster than Locals (18 ms a
token against 20). With 4 or 8 requests Locals writes 1.5 to 2.3 times as many
tokens per second as either, and its first token stays near one second.

This comparison found a bug in Locals: before commit `26f2482`, Gemma 4's replies
to prompts longer than its 512-token sliding window, read several at a time,
could begin with the end of turn (with 4 or 8 requests at once, 6 of 36 replies
came back empty; none when each prompt was sent alone). The numbers above are
after the fix.

## Against llama.cpp

The same client and prompts against llama.cpp's own server (`llama-server` from
the official macOS build `b11724`) on the same M2 Pro, running the GGUF files
Ollama downloaded above (`qwen3.5:4b-q4_K_M`, Q4_K_M, and `gemma4:e4b-it-qat`,
Q4_0), with `-ngl 99 -fa on -np 8 -c 32768` (`-c 81920` for 8,000-token prompts)
and `--chat-template-kwargs '{"enable_thinking": false}'`. Its speculative
decoding options were left at their defaults (off); Locals' drafting is on by
default. Locals ran again in the same session, right after llama.cpp, on 19
October 2026 (commit `ca0306e`); its numbers matched the day before's within about 1%.
llama.cpp supports `ignore_eos`, so Qwen3.5-4B also ran with replies of a fixed
length, every reply as long on both servers.

### Qwen3.5-4B, 1,000 tokens in, replies of natural length

| Requests at once | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| First token, Locals | 1.4 s | 1.5 s | 1.4 s | 1.6 s |
| First token, llama.cpp | 1.7 s | 1.8 s | 2.1 s | 2.1 s |
| Time per token, Locals | 16 ms | 19 ms | 37 ms | 59 ms |
| Time per token, llama.cpp | 21 ms | 43 ms | 76 ms | 85 ms |
| Tokens/s (reply length), Locals | 47 (259) | 79 (387) | 86 (360) | 110 (325) |
| Tokens/s (reply length), llama.cpp | 43 (507) | 44 (456) | 46 (546) | 71 (383) |

### Qwen3.5-4B, 1,000 tokens in, replies of 820 to 1,024 tokens

| Requests at once | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Tokens/s, Locals | 86 | 97 | 123 | 174 |
| Tokens/s, llama.cpp | 44 | 43 | 46 | 117 |
| First token, Locals | 1.4 s | 1.5 s | 1.5 s | 1.7 s |
| First token, llama.cpp | 1.8 s | 1.8 s | 2.1 s | 0.9 s |

With one request, Locals' 86 tokens per second come partly from its drafts: past
the end of a reply that `ignore_eos` skips, the model repeats itself, and drafts
taken from earlier text are accepted. The natural-length table is the fair one
for a single request (47 against 43). From 2 requests on, both run batches, and
llama.cpp gains little until 8 requests on this model.

### Qwen3.5-4B, 8,000 tokens in

| Requests at once | 1 | 4 |
|---|---|---|
| First token, Locals | 11.0 s | 12.6 s |
| First token, llama.cpp | 13.5 s | 19.0 s |
| Time per token, Locals | 18 ms | 77 ms |
| Time per token, llama.cpp | 22 ms | 139 ms |
| Tokens/s (reply length), Locals | 18 (282) | 30 (470) |
| Tokens/s (reply length), llama.cpp | 24 (534) | 20 (404) |

### Gemma 4 E4B, 1,000 tokens in

| Requests at once | 1 | 4 | 8 |
|---|---|---|---|
| First token, Locals | 1.0 s | 0.9 s | 1.1 s |
| First token, llama.cpp | 1.6 s | 1.8 s | 1.9 s |
| Time per token, Locals | 19 ms | 39 ms | 61 ms |
| Time per token, llama.cpp | 21 ms | 80 ms | 88 ms |
| Tokens/s (reply length), Locals | 41 (188) | 81 (225) | 107 (210) |
| Tokens/s (reply length), llama.cpp | 35 (169) | 39 (171) | 67 (246) |

llama.cpp's Gemma numbers match Ollama's above (35.2 tokens/s with one request,
the same 676 tokens in all): the same model file, and Ollama runs Gemma on
llama.cpp's code.

### One request at a time

The requests of [Against mlx-lm's own server](#against-mlx-lms-own-server), with
llama.cpp started with `-np 1 -c 65536`:

| Request | llama.cpp | Locals |
|---|---|---|
| Rewrite a 100-line Python file with one rename (689 tokens back) | 48 tokens/s | 153 tokens/s |
| OpenCode's first request (its system prompt and tools), cold | first token 13.4 s | 11.9 s |
| A question about a 48,614-token document | 129 s | 101 s |
| Its follow-up | 0.5 s | 0.7 s |

Both answer a follow-up from their prompt cache; llama.cpp's is a little quicker
here. llama.cpp keeps no prompts across a restart unless `--slot-save-path` is
set and a client saves its slots; Locals keeps system prompts on disk (1.7 s for
OpenCode's first request after a restart, above). The rewrite was 128 tokens per
second on Locals the day before, against mlx-lm's server measured with it: this
Mac's speed varies from day to day, so only rows measured in one session are
compared.

## Against mlx-lm's own server

The same requests to mlx-lm's server (`python -m mlx_lm server`, mlx-lm 0.32.2,
`--chat-template-args '{"enable_thinking": false}'`, its defaults otherwise) and
to Locals (`python -m locals.server --temp 0`, commit `26f2482`), one server at a
time, Qwen3.5-4B, temperature 0, streamed, 18 October 2026:

| Request | mlx-lm | Locals |
|---|---|---|
| One request: rewrite a 100-line Python file with one rename (689 tokens back) | 43 tokens/s, 16.1 s | 128 tokens/s, 6.9 s |
| OpenCode's first request (its system prompt and tools, 7,283 tokens), cold | first token 16.3 s | 11.9 s |
| The same right after a restart | 16.3 s (as cold: nothing is kept) | 1.7 s (7,152 tokens from the disk cache) |
| A question about a 16,751-token document, then a follow-up | 39.7 s, then 0.34 s | 26.5 s, then 0.26 s |
| The same with 48,531 tokens | 136 s, then out of GPU memory | 101 s, then 0.56 s |
| The same with 58,344 tokens | 172 to 176 s, then out of GPU memory (twice) | 155 s, then 0.93 s |

The rewrite is faster for one request alone because the reply repeats the file,
and Locals checks drafts taken from the prompt in the same pass as the next token
(see [Batch engine](../engine/README.md)). The restart row is the disk cache of
system prompts: mlx-lm keeps prompts in memory only. Both servers answer a
follow-up from their prompt cache; past about 48,000 tokens on 16 GB, mlx-lm's
generation thread stopped with Metal's "Insufficient Memory" on the follow-up
(three of three tries) and the request failed, while Locals' memory guard kept
the conversation and answered. The edit row is the second of three rounds (the
rounds differed by less than 1%); the others ran once.

## On an M5 Pro

Measured by a Locals user on their MacBook Pro (M5 Pro, 16-core GPU, 24 GB, macOS 27, MLX 0.32.3) with the M2
Max report's decode sweep: n requests at once to `/v1/completions`, 256 tokens each, end of text ignored,
temperature 0, two rounds per point, the median. Locals at commit `d55465b` (18 October 2026), before the
kernels for the M5's neural accelerators and before batches of 8 to 16 requests were padded to faster sizes.
Tokens per second (mlx-lm / Locals):

| Requests at once | Qwen3.5-4B | Ratio | Qwen3.5-9B | Ratio |
|---|---|---|---|---|
| 1 | 86 / 90 | 1.05x | 52 / 54 | 1.05x |
| 2 | 151 / 161 | 1.07x | 87 / 93 | 1.06x |
| 4 | 232 / 253 | 1.09x | 129 / 131 | 1.02x |
| 8 | 258 / 382 | 1.48x | 145 / 229 | 1.58x |
| 16 | 333 / 508 | 1.53x | 200 / 307 | 1.53x |

Locals' own speed is close to the M2 Max's (382 against 403 tokens/s at 8 requests on the 4B), while mlx-lm's
is much higher (258 against 152): MLX's 4-bit kernels use the M5's neural accelerators. The 4B ran with both
engines in one server, the 9B with one engine per server. Later rounds on the same Mac: padding 12 requests to
16 made them 10-13% faster on the 4B and 19-20% on the 9B (commit `80c8828`), and Qwen3.8-27B wrote 18.5,
33.0, 49.0, 78.2 and 76.1 tokens per second with 1, 2, 4, 8 and 16 requests at once (commit `9f49cdc`, Locals
alone).

## A client-side check of the speed

The numbers above are checked here in a way that trusts nothing Locals reports, with the scripts in
[audit/](audit/). The table at the top was measured inside the process, with the end-of-text token ignored;
these go through each server's own HTTP API, as an app sees them:

- **Baseline:** mlx-lm's own server (`python -m mlx_lm server`, mlx-lm 0.32.2, its defaults: up to 32 requests
  decoded together), not the stock engine inside Locals.
- **Counting:** the client counts the tokens itself (the streamed text tokenized again with the model's
  tokenizer) and times them on its own clock, from sending the requests to the last byte. Each server's own
  count was recorded too, and matched the client's on both.
- **No cache:** every request starts with a nonce no other request had, so no prompt cache can help.
- **Same work:** temperature 0, 256 tokens a reply, essay prompts; every reply on both servers ran to 256.

Qwen3.5-4B on the M2 Pro (16 GB), one server at a time, Locals at commit `42b4f94` with the kernel choices it
ships (a new user's), 21 October 2026. Tokens per second, end to end:

| Requests at once | mlx-lm's server | Locals | Ratio | Locals with mlx-lm's arithmetic |
|---|---|---|---|---|
| 1 | 51 | 65 | 1.27x | 61 |
| 2 | 79 | 116 | 1.47x | 92 |
| 4 | 88 | 150 | 1.70x | 98 |
| 8 | 75 | 244 | 3.24x | 85 |
| 16 | 124 | 310 | 2.49x | 148 |

The last column is Locals with `LOCALS_DTYPE=bfloat16 LOCALS_FUSE=0 LOCALS_KERNELS=0`: MLX's own matmul
kernels, no fused layers, bfloat16. Its batch engine alone gives 10 to 20%, and it slows down from 4 to 8
requests as mlx-lm does; the kernels written for a few rows at a time give the rest.

**Are the replies the same?** Both servers build the same prompts (the same token counts) and write coherent
text with no repeated runs. Of the 31 replies, 3 are the same token for token; the others agree for a median of
50 tokens, then part at a near-tie ("facilitating" against "accelerating"). One at a time, with no batching, 2
of 8 are the same over 128 tokens. To see which is nearer the exact computation, every reply was scored by the
same model computed in float32 (`reference.py`, teacher forcing):

| Replies written by | The float32 model's own top choice | Mean log-probability |
|---|---|---|
| mlx-lm's server (bfloat16) | 98.79% of the tokens | -0.3792 |
| Locals (float16, its kernels) | 99.79% | -0.3707 |
| Locals with mlx-lm's arithmetic | 98.80% | -0.3794 |

The replies differ by rounding, and Locals' default rounds less: float16 keeps three more bits than bfloat16.
With mlx-lm's arithmetic, Locals scores what mlx-lm scores.

**Answers that can be checked** (`quality_audit.py`): 48 word problems with computed answers, eight at a
time, 400 tokens each. Both servers got the same 36 right, and missed the same 12 (all one kind of problem,
where the model was still checking its work when the 400 tokens ran out); 28 of the 48 replies are the same
text. mlx-lm took 183 s, Locals 63 s.

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
178,176 tokens (`service.context_fit`); it was 145,408 when these runs were made,
with a 2.5 GiB margin.

Through the server, a fact placed in the middle of a document made of the
repository's own source files was found at every length tried:

| Prompt tokens | Time to first token | Found |
|---|---|---|
| 28,814 | 50 s | yes |
| 59,324 | 133 s | yes |
| 110,749 | 348 s | yes |
| 128,761 | 446 s | yes |
| 139,387 | 501 s | yes |
| 169,971 | 711 s | yes (21 October 2026, the limit of 178,176; its follow-up came in 2.0 s from the prompt cache) |

A request of 158,012 tokens was refused with `context_length_exceeded` (by that
limit of then).

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
