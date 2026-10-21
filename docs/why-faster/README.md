# Why Locals is faster than mlx-lm

Part of the [technical documentation](../README.md) of Locals. This page answers
one question with measurements: what does Locals do differently from mlx-lm, and
how much does each difference count when many requests arrive at once?

The short answer: with 8 requests at once, Locals writes 273 tokens per second and
plain mlx-lm 86, and the whole difference comes from Locals' code. It is not in
how requests are scheduled: both batch concurrent requests, and both overlap the
CPU's work with the GPU's. It is in how the model is computed for a batch of a
few rows. mlx-lm's 4-bit matrix multiplications get no faster per row between 2
and 8 rows, so 8 requests at once run about as fast as 4. Locals replaces them
with GPU kernels written and tuned for small batches, computes in float16 on Macs
whose GPU has no native bfloat16 (M1, M2), and fuses the model's projections into
fewer, larger multiplications. Around that, it keeps conversations between turns
(the prompt cache), constrains tool calls on the GPU, and guards memory, which
matter for agents more than for a single chat.

## The measurement

Qwen3.5-4B (`mlx-community/Qwen3.5-4B-MLX-4bit`) on a MacBook Pro with an M2 Pro
(19-core GPU) and 16 GB, mlx 0.32 and mlx-lm 0.32. Sixteen short chat questions
are decoded at batch sizes 1, 4, 8 and 16, 256 tokens each, with the end-of-text
token ignored so that every row does the same work; the number is tokens written
per second by all rows together. "Plain mlx-lm" is `mlx_lm.load` with
`generate_step` for one stream and `BatchGenerator` for several. "Locals" is
`Runtime.load` with the single-stream engine for one stream and
`HybridBatchEngine` for several. Each variant below switches one thing off and
runs in its own process.

| Variant (tokens/s) | 1 request | 4 | 8 | 16 |
|---|---|---|---|---|
| Plain mlx-lm | 60.5 | 87.0 | 86.2 | 156.8 |
| Locals | 65.6 | 160.8 | 272.7 | 345.8 |
| Locals' model, scheduled by mlx-lm's `BatchGenerator` instead of Locals' engine | 67.0 | 162.1 | 272.9 | 344.6 |
| Locals without the small-batch kernels | 67.0 | 118.1 | 119.2 | 213.0 |
| Locals in bfloat16 instead of float16 | 65.6 | 93.0 | 97.9 | 317.0 |
| Locals without fused projections | 63.7 | 154.5 | 246.8 | 332.8 |
| Locals without the Gated DeltaNet kernel | 64.3 | 158.0 | 267.6 | 343.0 |
| Locals without pipelining | 55.2 | 138.1 | 236.5 | 300.6 |
| Locals with all of the above off | 53.7 | 78.0 | 79.3 | 146.5 |

How to read it: the first two rows are the comparison, plain mlx-lm against
Locals, 1.8 to 3.2 times faster from 4 requests on. The third row asks where
that comes from. When Locals loads a model it replaces parts of the model's
computation (the kernels, the number format, the fused layers below), and then
runs it in its own batch engine. The third row keeps Locals' model but hands it
to mlx-lm's batch scheduler instead, and it is just as fast. So the speed-up is
in Locals' model computation, not in its scheduler; mlx-lm's scheduler with
mlx-lm's own model is the first row, 86 tokens/s with 8 requests. The rows after
it remove one piece at a time from Locals; with all of them off, Locals' engine
is as slow as plain mlx-lm (79 tokens/s with 8 requests).

## The differences that make decoding faster

### 1. Kernels for 4-bit matmuls with a few rows (the largest part)

Decoding with B requests multiplies every weight matrix by B rows, one per
request. MLX's quantized matrix-vector kernel works on one row per threadgroup,
so with several rows it reads the weights once for each row; from about 6 rows
on (on the M2 Pro, for large matrices) MLX switches to a kernel that works in
tiles of 32 rows, which a batch of 8 fills only a quarter (as measured and noted
in `kernels.py`). Both waste most of their work at the 2 to 16 rows that agent
workloads produce, which is why plain mlx-lm writes 87 tokens per second with 4
requests and 86 with 8.

Locals routes every 4-bit matmul with few rows (affine quantization in groups of
64) to the fastest of several kernels for that exact shape: MLX's own, a
multi-row matrix-vector kernel (`qmv_multi.py`), one that dequantizes straight
into the matrix units' fragments (`qmv_frag.py`), and one built from MLX's matrix
unit pieces (`mma_kernel.py`). The choice is made by timing them on the Mac's own
GPU, once per shape (`kernels.py`: `_choose`, `tune_model`), and cached; a preset
for the GPU these measurements ran on (`applegpu_g14s`) ships with the package,
and the server tunes missing shapes in the background while it is idle. Without these kernels, Locals falls to 118
tokens/s at 4 and 8 requests.

More in [Runtime and kernels](../runtime/README.md).

### 2. float16 on GPUs without native bfloat16

The MLX checkpoints of Qwen3.5 keep their quantization scales and activations in
bfloat16, which the M1 and M2 GPUs (`applegpu_g13`, `applegpu_g14`) emulate rather
than compute natively. On those GPUs Locals casts them to float16
(`runtime.set_compute_dtype`) when every value fits float16's range and a probe
input gives the same next token and nearly the same logits as before, or when
float16 is at least as close as bfloat16 to a float32 run of the probe (Gemma 4
E4B); otherwise the model stays in bfloat16, as do models whose own code treats float16
specially (Gemma 3). In bfloat16, the same Locals runs at 93 and 98 tokens/s with 4 and 8
requests instead of 161 and 273: the small-batch kernels gain little without it.
On GPUs with native bfloat16 (M3 and later) Locals keeps bfloat16. On an M5
Pro the whole of Locals was measured at 1.05x mlx-lm with one request and 1.5x
with eight to sixteen ([Measurements](../measurements/README.md#on-an-m5-pro));
M3 and M4 chips are not measured yet.

### 3. Fused projections

The query, key and value projections of each attention layer become one matrix
multiplication, the gate and up projections of each MLP another, and Qwen3.5's
four input projections of a linear-attention layer a third (`fuse.py`). Fewer,
larger multiplications mean fewer kernel launches per step and better use of the
GPU. Worth 4% to 10% at 4 to 16 requests.

### 4. One kernel for a Gated DeltaNet step (Qwen3.5)

24 of Qwen3.5-4B's 32 layers are linear-attention (Gated DeltaNet) layers. For one
decode token, mlx-lm runs about fourteen small kernels per such layer; Locals
runs one (`gdn_kernel.py`). Worth 1% to 2% here; the launches it saves matter
most for a single stream.

### What both do: batching and pipelining

Both mlx-lm's `BatchGenerator` and Locals' `HybridBatchEngine` decode all running
requests together, one token per request per step, and let requests join and
leave between steps (continuous batching). Both also launch the next step on the
GPU before the CPU has read the previous one, so the CPU's bookkeeping overlaps
the GPU's work; switched off in Locals, this costs 13% to 16%. These are not
differences from mlx-lm, and on this test Locals' scheduler is as fast as
mlx-lm's, not faster.

## The differences that matter for agents

Coding agents send long, growing prompts, call tools, and come back with the same
conversation every turn. These parts of Locals do not change the decode numbers
above; they decide how much of a real workload is decoding at all.

**Reading prompts.** Through the server, with 1,000-token prompts, the first
token came after 1.40 s with Locals and 2.22 s with the stock mlx-lm engine
(InferenceX's client, one request at a time). The same float16 compute and fused
projections apply, and from 128 rows on Locals dequantizes the 4-bit weights and
uses MLX's dense matrix multiplication (`kernels.PREFILL_M`). New prompts that
arrive together are prefilled in one batch, in chunks sized to stay within
memory. A long prompt that arrives while others decode is read in the same passes
as their next tokens (mixed passes), so they keep streaming; this does not add
throughput. See [Batch engine](../engine/README.md).

**Keeping conversations.** A model like Qwen3.5 keeps a recurrent state that
cannot be cut back to an earlier point, so a cached conversation is only reusable
from exactly where it was stored. Locals stores snapshots where the next request
will continue: at the end of the system prompt, at the start of the last message,
and where the reply begins as the template will write it next time (it writes an
earlier reply differently from the open one). With four multi-turn conversations
at once, 83% of the prompt tokens came from the cache instead of 28%, and a
follow-up question on a 138,000-token document started in 2.4 s instead of
reading it again. See [Prompt cache](../prompt-cache/README.md).

**Tool calls and JSON on the GPU.** Requests that call tools or ask for JSON are
decoded under a grammar whose transition table lives on the GPU, so the mask for
every row is applied in the same pass; tokens the grammar determines (field
names, brackets, the tool-call syntax) are written without running the model.
The stock engine applies llguidance's mask on the CPU for each row and step. See
[Constrained decoding](../decoding/README.md).

**Memory.** Admission control and a growth guard keep the batch, the prompt
cache and prefill within the GPU's memory, so many agents with long prompts do
not end in an out-of-memory error that fails every request in flight. The
context limit is sized from the memory as well (about 178,000 tokens for
Qwen3.5-4B on 16 GB).

## Where Locals is not faster

**One request.** 65.6 against 60.5 tokens/s: about the same.

**Models outside the tuned kernels.** The small-batch kernels cover 4-bit affine
quantization in groups of 64; other formats stay on MLX's kernels. A
mixture-of-experts model in the mxfp4 format (Mellum) ran at the same speed as
with mlx-lm.

**Prefill-bound work.** Reading a prompt is limited by the GPU's arithmetic:
about 700 tokens/s at the start of a prompt on this chip, falling to 168 at
128,000 tokens. With 8,000-token prompts and 8 requests at once, the server wrote
57 tokens per second; most of the time went into reading prompts.

**Newer chips.** All numbers here come from an M2 Pro. On M3 and later, which
compute bfloat16 natively, the float16 difference does not apply, and the
balance of the other parts may differ: on an M5 Pro, mlx-lm itself is much
faster (MLX has kernels for its neural accelerators), and Locals gains 1.5x
with eight requests at once, not 3x.

## Repeating the measurement

The table comes from one script that loads the model the way each variant needs,
in its own process, one after another; the switches it uses are
`Runtime.load(model, fuse=False)`, `Runtime.load(model, small_m_kernel=False)`,
`Runtime.load(model, dtype="bfloat16")`, `locals.fuse.USE_GDN_KERNEL = False`,
and, for pipelining, `mx.async_eval` replaced by `mx.eval` so that every step
waits for the GPU. See [Measurements](../measurements/README.md) for the other
benchmarks.
