# Benchmarks

Part of the [technical documentation](../README.md) of Locals: every result measured so far, in one place, with how
each was measured.

Unless a section says otherwise: a MacBook Pro with an M2 Pro (19-core GPU) and
16 GB, Qwen3.5-4B (`mlx-community/Qwen3.5-4B-MLX-4bit`), MLX 0.32 and mlx-lm
0.32, one job on the GPU at a time, and kernel tuning finished before any timing.
Every table says how it was measured; the scripts and the raw details are in
[Measurements](https://github.com/Generated-Computers/Locals/blob/main/docs/measurements/README.md).

## Agent work

<img alt="16 coding agents calling tools: mlx-lm 74.6 s, Locals 29.3 s. 40 LiveCodeBench problems: mlx-lm 19.7 min, Locals 7.93 min. 48 word problems: mlx-lm 183 s, Locals 63 s. Rewriting a 100-line file: mlx-lm 16.1 s, Locals 6.9 s. First request after a restart: mlx-lm 16.3 s, Locals 1.7 s." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/agents.svg#gh-light-mode-only" width="100%">
<img alt="16 coding agents calling tools: mlx-lm 74.6 s, Locals 29.3 s. 40 LiveCodeBench problems: mlx-lm 19.7 min, Locals 7.93 min. 48 word problems: mlx-lm 183 s, Locals 63 s. Rewriting a 100-line file: mlx-lm 16.1 s, Locals 6.9 s. First request after a restart: mlx-lm 16.3 s, Locals 1.7 s." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/agents-dark.svg#gh-dark-mode-only" width="100%">

## Every Mac measured so far

On every Mac measured so far (Qwen3.5-4B, tokens per second, mlx-lm / Locals):

| Mac | 1 request | 8 requests at once | 16 requests at once |
|---|---|---|---|
| M2 Pro, 16 GB (in the process) | 61 / 64 (1.0x) | 87 / 271 (3.1x) | 157 / 341 (2.2x) |
| M2 Pro, 16 GB (through each server's API, counted by the client) | 51 / 65 (1.3x) | 75 / 244 (3.2x) | 124 / 310 (2.5x) |
| M2 Max, 96 GB (through the server) | 91 / 104 (1.15x) | 152 / 403 (2.7x) | 267 / 530 (2.0x) |
| M5 Pro, 24 GB (through the server) | 86 / 90 (1.05x) | 258 / 382 (1.5x) | 333 / 508 (1.5x) |

## 1. How many requests at once (M2 Pro)

**In the process.** Sixteen short chat questions decoded at 1 to 16 rows at once,
256 tokens each with the end-of-text token ignored, so every row does the same
work. Tokens written per second by all rows together:

| Requests at once | mlx-lm | Locals | Ratio |
|---|---|---|---|
| 1 | 61 | 64 | 1.0x |
| 2 | 82 | 116 | 1.4x |
| 4 | 87 | 160 | 1.8x |
| 8 | 87 | 271 | 3.1x |
| 16 | 157 | 341 | 2.2x |

**Through each server's API, counted by the client.** The same comparison done in
a way that trusts nothing Locals reports: mlx-lm's own server
(`python -m mlx_lm server`, up to 32 requests decoded together) against Locals'
server, the client counting the streamed tokens itself with the model's
tokenizer, every request opening with a nonce so no prompt cache helps,
temperature 0, 256 tokens a reply (21 October 2026):

| Requests at once | mlx-lm's server | Locals | Ratio | Locals with mlx-lm's arithmetic |
|---|---|---|---|---|
| 1 | 51 | 65 | 1.27x | 61 |
| 2 | 79 | 116 | 1.47x | 92 |
| 4 | 88 | 150 | 1.70x | 98 |
| 8 | 75 | 244 | 3.24x | 85 |
| 16 | 124 | 310 | 2.49x | 148 |

The last column is Locals with MLX's own matmul kernels, no fused layers and
bfloat16 (`LOCALS_DTYPE=bfloat16 LOCALS_FUSE=0 LOCALS_KERNELS=0`): the batch
engine alone gives 10 to 20%, and the kernels written for a few rows at a time
give the rest.

**Each tested model**, 8 requests at once, in the process (tokens per second):

| Model | Size | mlx-lm | Locals | Ratio |
|---|---|---|---|---|
| Qwen3.5-4B | 2.8 GB | 87 | 271 | 3.1x |
| Qwen3.5-2B | 1.6 GB | 188 | 576 | 3.1x |
| LFM2.5-2.6B | 1.5 GB | 144 | 404 | 2.8x |
| MiniCPM5-2B | 1.3 GB | 160 | 478 | 3.0x |
| Gemma 4 E4B | 4.8 GB | 80 | 224 | 2.8x |

<img alt="M2 Pro · 16 GB: Qwen3.5-2B 3.1x (188 to 576 tokens/s), MiniCPM5-2B 3.0x (160 to 478 tokens/s), LFM2.5-2.6B 2.8x (144 to 404 tokens/s), Qwen3.5-4B 3.1x (87 to 271 tokens/s), Gemma 4 E4B 2.8x (80 to 224 tokens/s). M2 Max · 96 GB: Qwen3.5-4B 2.66x (152 to 403 tokens/s), Qwen3.5-9B 3.03x (72 to 220 tokens/s), Qwen3.8-27B 3.13x (25 to 78 tokens/s). M5 Pro · 24 GB: Qwen3.5-4B 1.48x (258 to 382 tokens/s), Qwen3.5-9B 1.58x (145 to 229 tokens/s)." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/models.svg#gh-light-mode-only" width="100%">
<img alt="M2 Pro · 16 GB: Qwen3.5-2B 3.1x (188 to 576 tokens/s), MiniCPM5-2B 3.0x (160 to 478 tokens/s), LFM2.5-2.6B 2.8x (144 to 404 tokens/s), Qwen3.5-4B 3.1x (87 to 271 tokens/s), Gemma 4 E4B 2.8x (80 to 224 tokens/s). M2 Max · 96 GB: Qwen3.5-4B 2.66x (152 to 403 tokens/s), Qwen3.5-9B 3.03x (72 to 220 tokens/s), Qwen3.8-27B 3.13x (25 to 78 tokens/s). M5 Pro · 24 GB: Qwen3.5-4B 1.48x (258 to 382 tokens/s), Qwen3.5-9B 1.58x (145 to 229 tokens/s)." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/models-dark.svg#gh-dark-mode-only" width="100%">

Dense 4-bit models gain 2.3x to 3.3x at 8 rows. A mixture-of-experts model in the
mxfp4 format (Mellum) gained nothing (1.0x): the tuned kernels cover 4-bit
affine layers in groups of 64, and other formats stay on MLX's kernels.

## 2. Other Macs

### M2 Max, 96 GB

Through the HTTP server with both engines in one process, n requests at once to
`/v1/completions`, 256 tokens each, end of text ignored, temperature 0, the
median of two rounds (14 and 15 October 2026). Tokens per second, mlx-lm / Locals:

| Requests at once | Qwen3.5-4B | Ratio | Qwen3.5-9B | Ratio | Qwen3.8-27B | Ratio |
|---|---|---|---|---|---|---|
| 1 | 91 / 104 | 1.15x | 49 / 56 | 1.15x | 18 / 20 | 1.14x |
| 2 | 121 / 177 | 1.46x | 60 / 86 | 1.43x | 21 / 30 | 1.43x |
| 4 | 142 / 244 | 1.71x | 69 / 125 | 1.81x | 24 / 42 | 1.80x |
| 8 | 152 / 403 | 2.66x | 72 / 220 | 3.03x | 25 / 78 | 3.13x |
| 16 | 267 / 530 | 1.99x | 138 / 300 | 2.18x | 47 / 106 | 2.25x |

`locals bench --agents` (tool-calling chat requests of natural length, the median
of two rounds), tokens per second, mlx-lm / Locals:

| Requests at once | Qwen3.5-4B | Qwen3.5-9B | Qwen3.8-27B |
|---|---|---|---|
| 1 | 34 / 62 (1.81x) | | |
| 2 | 61 / 113 (1.84x) | | |
| 4 | 69 / 140 (2.03x) | | |
| 8 | 80 / 131 (1.63x) | 41 / 108 (2.60x) | 11 / 24 (2.09x) |
| 16 | 95 / 142 (1.49x) | 49 / 77 (1.56x) | 13 / 23 (1.73x) |

GPU memory with both engines loaded: 11.6 GB (4B), 16.1 GB (9B) and 43.9 GB (27B)
at the peak. No request failed or was held back for memory.

### M5 Pro, 24 GB

Measured by a Locals user on their MacBook Pro (M5 Pro, 16-core GPU, 24 GB,
macOS 27, MLX 0.32.3), with the M2 Max's method above. Locals at commit
`d55465b` (18 October 2026), before the kernels for the M5's neural accelerators
and before batches of 8 to 16 requests were padded to faster sizes. Tokens per
second, mlx-lm / Locals:

| Requests at once | Qwen3.5-4B | Ratio | Qwen3.5-9B | Ratio |
|---|---|---|---|---|
| 1 | 86 / 90 | 1.05x | 52 / 54 | 1.05x |
| 2 | 151 / 161 | 1.07x | 87 / 93 | 1.06x |
| 4 | 232 / 253 | 1.09x | 129 / 131 | 1.02x |
| 8 | 258 / 382 | 1.48x | 145 / 229 | 1.58x |
| 16 | 333 / 508 | 1.53x | 200 / 307 | 1.53x |

`locals bench --agents` on the same Mac: Qwen3.5-4B 1.09x, 1.20x, 1.40x, 1.13x
and 1.11x at 1, 2, 4, 8 and 16 requests; Qwen3.5-9B 1.32x at 8 and 1.27x at 16.

Locals' own speed is close to the M2 Max's (382 against 403 tokens/s at 8
requests on the 4B), while mlx-lm's is much higher (258 against 152): MLX's
4-bit kernels use the M5's neural accelerators. What the later rounds on the
same Mac found:

- **What each part gives** (Qwen3.5-4B, tokens per second at 1, 2, 4, 8 and 16
  requests; 19 October, with other software running, so 5 to 8% below the runs
  above):

  | Variant | 1 | 2 | 4 | 8 | 16 |
  |---|---|---|---|---|---|
  | Locals | 88 | 156 | 228 | 355 | 459 |
  | without Locals' kernels | 89 | 157 | 220 | 245 | 392 |
  | without fused projections | 80 | 135 | 195 | 266 | 352 |
  | without the Gated DeltaNet kernel | 85 | 149 | 218 | 353 | 451 |
  | in float16 instead of bfloat16 | 88 | 156 | 214 | 366 | 462 |
  | mlx-lm | 84 | 141 | 207 | 231 | 313 |

  Locals' kernels give 45% at 8 requests and fused projections 33%. Unlike on
  the M2, float16 is not faster than bfloat16 here (the M5 computes bfloat16
  natively).
- **One decode step's 4-bit matrix products**, MLX's kernels against Locals'
  choice (ms, Qwen3.5-4B, bfloat16): 1 row 11.61 / 11.54, 4 rows 16.44 / 16.18,
  6 rows 24.79 / 19.80 (1.25x), 8 rows 30.05 / 20.07 (1.50x), 12 rows 43.35 /
  32.46 (1.34x), 16 rows 36.25 / 29.75 (1.22x), 32 rows 46.26 / 45.77. On the
  9B: 1.40x at 6 rows, 1.73x at 8, 1.46x at 12.
- **Reading prompts.** MLX's 4-bit matrix product is 1.11 to 1.26 times faster
  than dequantizing and multiplying densely on every layer shape (the M2 is the
  other way round), so Locals times both on each chip and keeps the faster one.
  First token for one 8,192-token prompt: 3.91 s (Locals) against 4.87 s
  (mlx-lm); four at once, the last one's first token after 16.6 s against 19.6 s.
- **Padding 12 requests to 16**, where the chip's kernels are faster at 16 rows:
  10 to 13% faster on the 4B and 19 to 20% on the 9B (213 to 254 tokens/s with
  12 requests). Padding other batch sizes gained nothing worth keeping.
- **Qwen3.8-27B** on 24 GB: 18.5, 33.0, 49.0, 78.2 and 76.1 tokens per second
  with 1, 2, 4, 8 and 16 requests at once, none failed (Locals alone).
- **MLX's recurrent-layer op for reading prompts** (it has a version for the
  M5): 3.6 to 3.8% faster on the 4B, chosen there by timing.
- **A 4-bit matrix kernel for the M5's neural accelerators** (`nax`, built and
  measured on this Mac; experimental, and only on the M5). One decode step's
  4-bit matrix products, ms, the best of today's kernels against the new one:

  | Rows | Qwen3.5-4B: today | with `nax` | Qwen3.5-9B: today | with `nax` |
  |---|---|---|---|---|
  | 1 | 10.58 | 10.58 (MLX's kept) | 17.74 | 17.74 (MLX's kept) |
  | 4 | 13.00 | 11.43 | 24.56 | 19.66 |
  | 6 | 14.57 | 11.76 | 27.06 | 20.06 |
  | 8 | 14.65 | 11.85 | 27.25 | 20.00 |
  | 12 | 24.25 | 12.01 | 47.73 | 20.29 |
  | 16 | 24.32 | 12.11 | 47.93 | 20.64 |
  | 32 | 28.27 | 15.94 | 53.30 | 29.96 |

  Timed shape by shape; whole decode steps and replies have not been measured
  yet. `LOCALS_NAX=0` turns it off.

### M1, M3 and M4

An M4 Pro (24 GB) has Flint's measurements (section 11). Qwen3.5-4B on it, through the server (`locals bench`, its
kernel tuning not finished): 81 / 78 tokens/s for one request (mlx-lm / Locals), 181 / 199 for four, 192 / 207 for
eight. The rest are not measured yet. On M1 (like the M2) Locals computes in float16, which those
GPUs handle natively; M3 and later keep bfloat16. Results from these chips are
welcome as [issues](https://github.com/Generated-Computers/Locals/issues), with
`locals bench` and `locals tune --report`.

## 3. Against other servers

All on the M2 Pro, one server at a time on the GPU, each with its own 4-bit
build of the model (Locals: MLX; Ollama 0.40.5: its MLX build of Qwen3.5-4B
with NVFP4 weights, and GGUF Q4_0 for Gemma; Magnitude 0.2.9: GGUF Q4;
llama.cpp `b11724`: the GGUF files Ollama downloads, Q4_K_M and Q4_0). The
client is SemiAnalysis's InferenceX benchmark client, chat completions,
random prompts, temperature 0, thinking off, replies of natural length.

<img alt="Qwen3.5-4B, 1,000-token prompts, 8 requests at once on an M2 Pro. Time to the first token: Locals 1.6 s, llama.cpp 2.1 s, Magnitude 16.1 s, Ollama 41.8 s. Tokens per second: Locals 110, llama.cpp 71, Magnitude 42, Ollama 34." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/servers.svg#gh-light-mode-only" width="100%">
<img alt="Qwen3.5-4B, 1,000-token prompts, 8 requests at once on an M2 Pro. Time to the first token: Locals 1.6 s, llama.cpp 2.1 s, Magnitude 16.1 s, Ollama 41.8 s. Tokens per second: Locals 110, llama.cpp 71, Magnitude 42, Ollama 34." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/servers-dark.svg#gh-dark-mode-only" width="100%">

### Ollama and Magnitude

Qwen3.5-4B, 1,000-token prompts:

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

Ollama answers this model one request at a time: each reply is written at full
speed while the others wait, so the first token comes later the more requests
there are.

Qwen3.5-4B, 8,000-token prompts:

| Requests at once | 1 | 4 |
|---|---|---|
| First token, Locals | 10.9 s | 17.6 s |
| First token, Ollama | 16.4 s | 72.5 s |
| First token, Magnitude | 11.5 s | 124.5 s |
| Tokens/s (reply length), Locals | 18 (282) | 30 (470) |
| Tokens/s (reply length), Ollama | 14 (270) | 15 (317) |
| Tokens/s (reply length), Magnitude | 24 (399) | 3.5 (143) |

With 4 requests at once, 4 of Magnitude's 8 replies came back empty; Locals and
Ollama answered all of them.

Gemma 4 E4B, 1,000-token prompts:

| Requests at once | 1 | 4 | 8 |
|---|---|---|---|
| First token, Locals | 0.9 s | 0.9 s | 1.1 s |
| First token, Ollama | 1.6 s | 2.2 s | 2.0 s |
| First token, Magnitude | 0.8 s | 3.8 s | 12.7 s |
| Tokens/s (reply length), Locals | 41 (187) | 81 (225) | 105 (208) |
| Tokens/s (reply length), Ollama | 35 (169) | 38 (171) | 58 (246) |
| Tokens/s (reply length), Magnitude | 50 (220) | 55 (183) | 45 (165) |

For one Gemma request alone Magnitude is a little faster (18 ms a token against
20); with 4 or 8 requests Locals writes 1.5 to 2.3 times as many tokens per
second as either.

### llama.cpp

Qwen3.5-4B, 1,000-token prompts:

| Requests at once | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| First token, Locals / llama.cpp | 1.4 / 1.7 s | 1.5 / 1.8 s | 1.4 / 2.1 s | 1.6 / 2.1 s |
| Time per token, Locals / llama.cpp | 16 / 21 ms | 19 / 43 ms | 37 / 76 ms | 59 / 85 ms |
| Tokens/s, natural reply length, Locals / llama.cpp | 47 / 43 | 79 / 44 | 86 / 46 | 110 / 71 |
| Tokens/s, replies of 820 to 1,024 tokens, Locals / llama.cpp | 86 / 44 | 97 / 43 | 123 / 46 | 174 / 117 |

With 8,000-token prompts and 4 at once: first token 12.6 s against 19.0 s, 30
against 20 tokens per second. Gemma 4 E4B with 8 at once: 107 against 67 tokens
per second. One request at a time:

| Request | llama.cpp | Locals |
|---|---|---|
| Rewrite a 100-line Python file with one rename (689 tokens back) | 48 tokens/s | 153 tokens/s |
| OpenCode's first request (its system prompt and tools), cold | first token 13.4 s | 11.9 s |
| A question about a 48,614-token document | 129 s | 101 s |
| Its follow-up | 0.5 s | 0.7 s |

### mlx-lm's own server

| Request | mlx-lm | Locals |
|---|---|---|
| Rewrite a 100-line Python file with one rename (689 tokens back) | 43 tokens/s, 16.1 s | 128 tokens/s, 6.9 s |
| OpenCode's first request (its system prompt and tools, 7,283 tokens), cold | first token 16.3 s | 11.9 s |
| The same right after a restart | 16.3 s (nothing is kept) | 1.7 s (7,152 tokens from the disk cache) |
| A question about a 16,751-token document, then a follow-up | 39.7 s, then 0.34 s | 26.5 s, then 0.26 s |
| The same with 48,531 tokens | 136 s, then out of GPU memory | 101 s, then 0.56 s |
| The same with 58,344 tokens | 172 to 176 s, then out of GPU memory (twice) | 155 s, then 0.93 s |

A file rewrite is faster even alone because the reply repeats the file, and
Locals checks pieces of it taken from the prompt in the same pass as the next
word. (The rewrite was 128 tokens/s on the day of this table and 153 on the day
of llama.cpp's: this Mac's speed varies from day to day, so only rows measured
in one session are compared.)

### The stock engine behind the same server

mlx-lm's batch engine behind Locals' own server (`--engines mlx`), InferenceX's
fixed-length runs, 1,000 tokens in and 1,000 out:

| Requests at once | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Stock mlx-lm engine, tokens/s | 52 | 66 | 70 | 69 |
| Locals, tokens/s | 60 | 98 | 125 | 177 |
| Locals, median first token | 1.40 s | 1.43 s | 1.38 s | 1.56 s |

## 4. Agents, end to end

**Eight agent tools on real coding tasks.** Locals' end-to-end suite runs the
tools themselves, unchanged, against a Locals server through a recording proxy:
OpenCode, Hermes Agent, OpenClaw, Kilo Code, Cline, OpenHands, Aider and Claude
Code, on small real projects (fix a bug, rename a function across files, add a
feature, answer a question about the code). Qwen3.5-4B, 21 October 2026: all 18
tasks done, 43 of 43 checks passed.

| Tool | Tasks done | Wall time per task | First token, median | Prompt from the cache |
|---|---|---|---|---|
| OpenCode | 4 of 4 | 18 to 44 s | 0.7 to 1.8 s | 90 to 97% |
| Hermes Agent | 2 of 2 | 55 and 69 s | 0.9 and 1.0 s | 87 and 90% |
| OpenClaw | 2 of 2 | 38 and 60 s | 1.2 and 0.9 s | 93 and 96% |
| Kilo Code | 2 of 2 | 52 and 38 s | 0.6 and 0.9 s | 85 and 98% |
| Cline | 2 of 2 | 29 and 49 s | 0.9 and 0.7 s | 82 and 96% |
| OpenHands | 2 of 2 | 79 and 61 s | 3.5 and 2.4 s | 87 and 99% |
| Aider | 2 of 2 | 15 and 16 s | 4.5 and 4.9 s | one request each |
| Claude Code | 2 of 2 | 60 and 116 s | 1.3 and 1.0 s | 79 and 88% |

Other scenarios of the same run:

- **Many at once.** Six OpenCode agents at once finished in 69 s, one alone in
  22 s: 1.9 times the work per minute. Four different tools at once (two
  OpenCode, Hermes Agent, OpenClaw) all finished, in 87 to 125 s, the median
  first token 1.1 s. GPU memory peaked at 8.4 GB.
- **A five-turn session** with OpenCode: from the second turn on, the first
  token came after 0.2 to 0.3 s, with 99 to 100% of an 11,000-token prompt from
  the cache.
- **A new user**: the installer took 9 s, the first `locals code --run` had the
  server ready after 23 s and the task done after 51 s.
- **A restart**: OpenCode's first request took 12.2 s cold and 1.7 s after a
  restart (7,152 of 7,276 prompt tokens from the disk cache).
- **Offline**: `locals start` in 6 s with the network off; OpenCode's task done
  in 26 s.
- **Interruptions**: Ctrl-C in a writing agent stops its requests on the server
  within 0.5 s; a client hanging up during a 20,000-token prompt frees the
  server, which answers the next request 1.7 s later.

**From the paper** (Qwen3.5-2B, every reply constrained to one well-formed tool
call; stock: mlx-lm's batch generator with llguidance for the grammar):

| | 16 agents, stock | 16 agents, Locals | 8 agents, stock | 8 agents, Locals |
|---|---|---|---|---|
| Wall time | 74.6 s | 29.3 s (2.55x) | 19.4 s | 22.1 s (0.87x) |
| Throughput | 125.5 tokens/s | 271.1 tokens/s (2.16x) | 108.6 tokens/s | 170.6 tokens/s (1.57x) |
| Agents finished | 9 of 16 | 12 of 16 | 5 of 8 | 6 of 8 |

| Workload (the whole engine against mlx-lm with llguidance) | mlx-lm | Locals | Gain |
|---|---|---|---|
| 8 concurrent tool-call requests through the server | 134 tokens/s | 410 tokens/s | 3.1x |
| 18 JSON jobs, 8 at once, the schema in the prompt | 6.12 s | 4.66 s | 1.31x |
| The same with only the keys in the prompt (Locals writes the syntax) | 6.12 s | 3.83 s | 1.60x |
| 40 LiveCodeBench problems, 8 at once (Qwen3.5-4B) | 1,181 s | 476 s | 2.5x |

**A team of agents** (`locals run`): on four tasks that each span several files,
it passed all eight attempts in about three minutes each; one agent alone, in
the same setup, passed four of eight in about five minutes each.

## 5. Long conversations and the prompt cache

**How long.** Qwen3.5-4B keeps 32 KB of attention cache per token (8 of its 32
layers use full attention; the other 24 keep a fixed state), so memory grows by
about 1 GB per 32,000 tokens. On a 16 GB Mac the server allows up to 178,176
tokens. One growing prompt, read in chunks:

| Context | Reading speed at that length | Writing speed (one request) | Peak GPU memory |
|---|---|---|---|
| 16,000 | 611 tokens/s | 55 tokens/s | 4.46 GB |
| 32,000 | 439 | 46 | 4.62 GB |
| 64,000 | 297 | 36 | 5.50 GB |
| 96,000 | 217 | 29 | 6.49 GB |
| 128,000 | 168 | 24 | 7.62 GB |
| 160,000 | 141 | 21 | 8.74 GB |

**A fact in the middle of a long document**, through the server (the document
made of the repository's own source files): found at every length tried.

| Prompt tokens | First token | Found |
|---|---|---|
| 28,814 | 50 s | yes |
| 59,324 | 133 s | yes |
| 110,749 | 348 s | yes |
| 128,761 | 446 s | yes |
| 139,387 | 501 s | yes |
| 169,971 | 711 s | yes, and its follow-up came in 2.0 s |

**Follow-up questions.** The next turn continues from where the previous reply
began instead of reading the conversation again:

| Conversation | Next turn before | Next turn now | Tokens from the cache |
|---|---|---|---|
| 60,564 tokens | 137.4 s | 1.5 s | 60,560 |
| 138,352 tokens | about 500 s | 2.4 s | 138,348 |

**Four conversations at once**, each a series of turns with a tool result of 300
to 3,000 tokens and a reply each turn, for four minutes:

| | Before | Now |
|---|---|---|
| Turns finished | 23 | 36 |
| Output tokens/s | 39.6 | 64.2 |
| Prompt tokens from the cache | 28% | 83% |
| Median time to the first token | 10.6 s | 5.2 s |

**No pause while a long prompt is read.** When an 8,000-token prompt arrives
while others are answering, their next tokens are computed in the same passes:
the longest gap in their streams went from 12 s to 3 s, with the same
throughput. A 3,000-token request that arrives during a 30,000-token one gets its
first token after 7 s instead of 41.

## 6. Quality of the answers

Locals computes the same model; replies differ from mlx-lm's only by rounding.
Scored by the same model computed in float32 (teacher forcing, the replies of
the client-side check above):

| Replies written by | The float32 model's own top choice | Mean log-probability |
|---|---|---|
| mlx-lm's server (bfloat16) | 98.79% of the tokens | -0.3792 |
| Locals (float16, its kernels) | 99.79% | -0.3707 |
| Locals with mlx-lm's arithmetic | 98.80% | -0.3794 |

Locals' default rounds less: float16 keeps three more bits than bfloat16.

- **Word problems with computed answers** (48, eight at a time, 400 tokens each):
  both got 36 right and missed the same 12; mlx-lm took 183 s, Locals 63 s.
- **LiveCodeBench** (from the paper): 12 of 40 problems passed with Locals and
  13 with mlx-lm, in 476 s against 1,181 s.
- **JSON under a token limit** (from the paper): with Locals' budget rule every
  JSON output is valid under every limit that admits one, 90 of 90, against 3 of
  90 without it; for recursive schemas 54 of 61 against 2 of 61.
- **Tool calls and JSON always parse**: replies are written under a grammar on
  the GPU, so a call names only the tools the agent offered, and a reply cut off
  by its length limit still closes its call.

## 7. Where the speed comes from

Each part of Locals switched off on its own (M2 Pro, in the process, tokens per
second):

<img alt="Tokens per second with 8 requests at once, Qwen3.5-4B on an M2 Pro, each part of Locals switched off on its own: Locals 272.7; Locals' model on mlx-lm's scheduler 272.9; without the Gated DeltaNet kernel 267.6; without fused projections 246.8; without pipelining 236.5; without the small-batch kernels 119.2; in bfloat16 instead of float16 97.9; everything above off 79.3; plain mlx-lm 86.2." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/ablation.svg#gh-light-mode-only" width="100%">
<img alt="Tokens per second with 8 requests at once, Qwen3.5-4B on an M2 Pro, each part of Locals switched off on its own: Locals 272.7; Locals' model on mlx-lm's scheduler 272.9; without the Gated DeltaNet kernel 267.6; without fused projections 246.8; without pipelining 236.5; without the small-batch kernels 119.2; in bfloat16 instead of float16 97.9; everything above off 79.3; plain mlx-lm 86.2." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/ablation-dark.svg#gh-dark-mode-only" width="100%">

| Variant | 1 request | 4 | 8 | 16 |
|---|---|---|---|---|
| Plain mlx-lm | 60.5 | 87.0 | 86.2 | 156.8 |
| Locals | 65.6 | 160.8 | 272.7 | 345.8 |
| Locals' model, scheduled by mlx-lm's batch engine | 67.0 | 162.1 | 272.9 | 344.6 |
| without the small-batch kernels | 67.0 | 118.1 | 119.2 | 213.0 |
| in bfloat16 instead of float16 | 65.6 | 93.0 | 97.9 | 317.0 |
| without fused projections | 63.7 | 154.5 | 246.8 | 332.8 |
| without the Gated DeltaNet kernel | 64.3 | 158.0 | 267.6 | 343.0 |
| without pipelining | 55.2 | 138.1 | 236.5 | 300.6 |
| all of the above off | 53.7 | 78.0 | 79.3 | 146.5 |

The speed-up is in how the model is computed for a few rows at a time, not in
the scheduling: Locals' model under mlx-lm's scheduler is just as fast.

1. **Kernels for 4-bit matrix products with a few rows** (the largest part).
   MLX's kernel for one row reads the weights once per row, and from about 6
   rows it switches to tiles of 32 rows, which a batch of 8 fills a quarter.
   Locals picks the fastest of several kernels for each exact shape by timing
   them on your Mac's GPU, and caches the choice.
2. **float16 on M1 and M2**, whose GPUs emulate bfloat16: 273 against 98 tokens/s
   with 8 requests. Kept only where a probe gives the same answer.
3. **Fused projections**: query, key and value in one product, gate and up in
   another: 4 to 10% at 4 to 16 requests.
4. **One kernel for a Gated DeltaNet step** (Qwen3.5's recurrent layers) instead
   of about fourteen: 1 to 2%.

**Where the time goes.** One decode step at about 1,000 tokens of context takes
25.7 ms for 4 rows and 31.8 ms for 8; reading the 4-bit weights once costs about
11 ms at this chip's memory bandwidth. Reading a prompt is limited by
arithmetic: about 700 tokens/s at its start, whatever the chunk length from 960
to 2,048 tokens. The GPU does not run a prompt's reading and other rows'
decoding at the same time on two streams (0.98x to 0.99x), so Locals puts both
in the same passes instead.

## 8. Sampling settings

A request's own `temperature`, `top_p`, `top_k` and `seed` apply to it, in one
pass for every row of the batch. 8 requests at once, Qwen3.5-4B, on a tuned
kernel cache (21 October 2026, tokens per second):

| Requests | mlx-lm-style server-wide settings (before) | Each request's own (now) |
|---|---|---|
| Greedy (temperature 0) | 249 | 247 |
| temperature 0.6, top_p 0.95, top_k 20 | 243 (the filters were ignored) | 225 |
| Eight different settings in one batch | 242 | 225 |
| Seeded (`seed`), temperature 0.6 | 243 | 238 |

The filters cost about 7% with eight requests at once, only for the requests
that ask for them. The same seeded request alone gives the same reply.

## 9. Reliability

- **End to end**: 43 of 43 checks with eight real agent tools (section 4).
- **A 30-minute soak**: bursts of 1 to 16 requests on all three APIs (OpenAI,
  Anthropic, Ollama), streamed and not, with clients hanging up: 0 problems in
  58 bursts, and memory besides the prompt cache flat at 2.47 to 2.49 GB.
- **Malformed requests** (bodies that are not JSON, wrong types, nesting a few
  thousand levels deep, bodies that never finish): every one gets a 4xx answer,
  and the server stays healthy.
- **Memory**: a guard keeps the batch, the prompt cache and prompt reading
  within the GPU's memory, so many agents with long prompts do not end in an
  out-of-memory error; on the M5 Pro's 24 GB the 27B runs 16 requests at once.
- **Tests**: 1,117 fast tests (run on every push, on Python 3.11 to 3.13) and 135 that load real models.

## 10. Where Locals is not faster

- **One request alone, with nothing to reuse**: about the same as mlx-lm (64
  against 61 tokens per second on the M2 Pro). It is faster alone when the reply
  repeats the prompt (file rewrites) or the prompt is cached.
- **Prompt-bound work**: reading a prompt is limited by the GPU's arithmetic.
  With 8,000-token prompts and 8 requests at once the server writes 57 tokens
  per second; most of the time goes into reading.
- **Formats outside the tuned kernels**: 4-bit affine quantization in groups of
  64 is covered; mxfp4 (a mixture-of-experts model, Mellum) ran at mlx-lm's
  speed.
- **The M5's neural accelerators**: MLX's own kernels already use them, so the
  gain there is 1.5x, not 3x (before Locals' own `nax` kernel, which is
  experimental).
- **8 agents in the paper's suite**: 22.1 s against 19.4 s for the stock engine,
  though with more tokens per second and more agents finishing.

## 11. Recursive's Flint (M4 Pro, 24 GB)

Flint (40B, 2.5B active per token: 256 routed experts, 6 a token, and a shared expert) on a MacBook Pro with an M4
Pro and 24 GB, the GPU's limit raised to 20 GB (`sudo sysctl iogpu.wired_limit_mb=20480`),
`mlx-community/flint-mixed-3-4bit` (17.8 GB: experts at 3 bits), MLX 0.32.4. Three setups, each alone on the GPU
(two copies of the model do not fit): mlx-lm with Flint (0.32.3.dev, ml-explore/mlx-lm#1998); Locals as on main
before Flint's support (e6be78c), given the same model code; and Locals with it. Both Locals setups use the kernel
choices this Mac's idle tuner had cached (`locals tune` in full mode does not fit next to macOS in 24 GB).

**In the process: requests at once.** A 256-token prompt each (the package's own source), then 48 greedy steps
after 4 warm-up steps; tokens per second all rows write together:

| Requests at once | mlx-lm | Locals before | Locals | Locals / mlx-lm |
|---|---|---|---|---|
| 1 | 85.3 | 78.6 | 116.8 | 1.37x |
| 2 | 123.3 | 127.0 | 179.3 | 1.45x |
| 4 | 176.9 | 173.5 | 224.7 | 1.27x |
| 8 | 201.6 | 223.3 | 304.8 | 1.51x |
| 16 | 244.5 | 278.8 | 339.8 | 1.39x |

Before, Locals was slower than mlx-lm for one request (its kernel routing cost 4 us of Python a matrix product, and
it fused nothing of Flint's) and gained only from its batching at 8 and 16. Of the 1.38x at one request, 0.14x is
MLX's command buffers: handed a layer's whole expert tensors, the expert kernels ended one each (102 tokens/s before
they were given one-expert views, docs/runtime, "Flint"). At 8 and 16 rows the expert kernels read an expert once for
all the rows that chose it (265 and 288 tokens/s before); from 4 rows they convert each input chunk once for several
weight rows, and the layers end in fewer kernels (284 and 313 at 8 and 16 before, 212 at 4: docs/runtime, "Flint",
"Several rows"). Past 16 rows the experts go back to MLX's sorted gather, which reads each expert once for all the
tokens that chose it.

**In the process: prompt length.** One request: a prompt read from nothing in 2,048-token chunks, then 64 greedy
steps written after it (4 warm-up steps first; a warm-up prompt before all of it, so no kernel compiles in a timing):

| Prompt tokens | Prompt read, tokens/s: mlx-lm / Locals before / Locals | Writing after it, tokens/s: mlx-lm / Locals before / Locals | Peak GPU memory, GB |
|---|---|---|---|
| 1,024 | 755 / 747 / 766 (1.3 s) | 83.8 / 73.7 / 109.2 | 19.04 / 19.11 / 19.00 |
| 4,096 | 680 / 675 / 763 (5.4 s) | 80.3 / 73.7 / 102.7 | 19.53 / 19.54 / 19.74 |
| 8,192 | 635 / 645 / 710 (11.5 s) | 76.6 / 70.6 / 94.7 | 19.66 / 19.67 / 19.77 |
| 16,384 | 567 / 575 / 625 (26 s) | 67.7 / 64.3 / 84.2 | 19.88 / 19.92 / 20.02 |
| 32,768 | 468 / 470 / 502 (65 s) | 60.6 / 56.0 / 68.9 | 20.39 / 20.24 / 20.46 |

Prompts are read 7 to 12% faster from 4,096 tokens: the 35 sliding-window layers attend in bands of 512 queries
against the keys their 513-token window reaches, not a whole 2,048-token chunk under a mask (docs/runtime, "Flint");
the rest is the same as mlx-lm's (Flint's kernels are for few rows, and a prompt's experts run through MLX's sorted
gather). These timings include the output layer's logits for every prompt token, as mlx-lm's own loop computes them;
the engine reads a prompt without them, at 849, 844, 788, 694 and 560 tokens/s for these lengths. Through the
server, a 13,000-token prompt's first token came after 18.7 s and the reply went on at 74.5 tokens/s; a 30,000-token
one's after 55.1 s, at 63.7 tokens/s. With `LOCALS_KV_BITS=8` (8-bit keys and values in the full-attention layers) the
server takes up to 67,584 tokens instead of about 35,800, and found a codename hidden in 59,778 tokens of source in
177 s; writing is 4 to 13% slower (docs/runtime, "Flint"). Writing is 1.30x mlx-lm's at 1,024 tokens and 1.14x at
32,768, where reading the 14 full-attention layers' keys and values takes a growing share of each step. On a 24 GB Mac the server allows about
35,000 tokens next to Flint.

**Through each server's API, counted by the client.** mlx-lm's own server (`python -m mlx_lm server`, with Flint),
Locals's as on main and Locals's now, one at a time: requests sent at once, each a prompt of about 1,000 tokens of
its own (a nonce first, so no prompt cache helps), temperature 0, up to 256 tokens; the client counts the streamed
text's tokens with the model's tokenizer, after a warm-up request:

| At once | All done, s: mlx-lm / Locals before / Locals | Tokens/s end to end | First token, median s | One reply's writing, tokens/s |
|---|---|---|---|---|
| 1 | 4.79 / 4.91 / 3.71 | 52.8 / 52.1 / 68.1 (1.29x) | 1.12 / 1.10 / 1.10 | 68.7 / 66.8 / 96.5 |
| 4 | 11.6 / 11.1 / 10.1 | 87.8 / 92.6 / 97.9 (1.12x) | 4.40 / 4.39 / 4.46 | 35.2 / 35.1 / 41.6 |
| 8 | 21.1 / 19.4 / 17.9 | 96.4 / 105.4 / 114.0 (1.18x) | 8.79 / 6.58 / 7.05 | 20.6 / 21.0 / 23.6 |

End to end is from sending the requests to the last token, the prompts' reading included (8,000 tokens at about
700 tokens/s is most of it at 8 at once). mlx-lm's server reads every prompt before it writes; Locals answers the
first requests while it reads the others (mixed passes), so its first tokens come sooner and its replies spend
longer under way: counted from the first reply's first token, mlx-lm's server writes 165 tokens/s at 8 at once,
Locals 139.
Replies stream without a stall while another long prompt is read: at most 2.8 s, against 9.7 without mixed passes
(docs/runtime, "Flint"). Decoding alone, clients with short prompts and 512-token replies, counted while every reply
streams: 289 to 293 tokens/s at 8 at once, 216 at 4 and 320 to 324 at 16, by how busy the Mac's CPU is besides
(275.7 and 305.3 at 8 and 16 before the changes for several rows, 256.1 and 280.7 before the expert kernels read
each expert once for the rows that chose it).

**Agent tool calls.** Flint as a coding agent's model in the batch engine, one request at a time: six tools shaped
as OpenCode's (`read` with `filePath`, `offset`, `limit`; `write`; `edit`; `bash` with a required `description`;
`glob`; `grep`), a system prompt that says to act through them, 12 tasks ("Show me the first 40 lines of
/repo/src/server.py", "Run the test suite in /repo with pytest", ...), `tool_choice` auto. A call is broken when it
names a parameter its tool lacks or leaves out a required one (the model card reports 41 of 85 OpenCode calls
failing on mlx-vlm for wrong parameter names):

| Grammar | Spans | Broken calls, greedy | Broken calls, sampled | Passes for the greedy replies | Writing, tokens/s |
|---|---|---|---|---|---|
| Untyped (the default) | a token a step | 4 of 12 | | 667 for 643 tokens | 99.2 |
| Untyped | in one pass | 4 of 12 | 11 of 39 | 667 for 643 | 99.3 |
| Typed (`--typed-tools`) | a token a step | 0 of 12 | | 672 for 648 | 98.8 |
| Typed | in one pass | 0 of 12 | 0 of 39 | 603 for 644 | 99.5 |

Sampled: the model card's settings (temperature 1.0, top_p 0.95, top_k 20), three seeds a task, 36 requests. The
untyped grammar's broken calls were `write` and `read` with `path` for `filePath`, and `bash` without its
`description`. The typed grammar writes the parameter names (`<arg_key>filePath</arg_key><arg_value>` after `read`
is five tokens the model does not choose), and one request alone feeds such a span in one pass: 10% fewer passes,
though a pass of several tokens reads each token's experts, so the time is about the same.
Writing time is wall time less the prompt's reading; the untyped grammar has no span of 3 tokens or more, so its
two rows differ only by noise. The typed automaton for these six tools took 3.9 s to build, once per tool set.

## How the measurements were made

- **Decode speed**: n requests at once, 256 tokens each with the end-of-text token
  ignored, so every request does the same work; tokens written per second by all
  of them together; the median of two rounds. "In the process" is the batch engine
  alone; "through the server" goes over HTTP as an app does.
- **The client-side check** counts the streamed tokens on the client with the
  model's own tokenizer and its own clock, with a nonce at the start of every
  prompt, against mlx-lm's own server: the scripts are in
  [docs/measurements/audit](https://github.com/Generated-Computers/Locals/blob/main/docs/measurements/audit/).
- **Other servers** were measured with SemiAnalysis's InferenceX client,
  unchanged but for what each server needed, one server at a time on the GPU.
- **Agents**: the end-to-end suite in
  [tests/e2e](https://github.com/Generated-Computers/Locals/blob/main/tests/e2e/README.md)
  runs the real tools through a recording proxy.
- **On your own Mac**: `locals bench` measures Locals against mlx-lm, and
  `locals tune --report` shows what the kernels gain on your chip.

Every table, its date, commit and command:
[Measurements](https://github.com/Generated-Computers/Locals/blob/main/docs/measurements/README.md),
[the M2 Max](https://github.com/Generated-Computers/Locals/blob/main/docs/measurements/m2-max.md),
[why Locals is faster](https://github.com/Generated-Computers/Locals/blob/main/docs/why-faster/README.md).
