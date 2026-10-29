# Model runtime and GPU kernels

Part of the [technical documentation](../README.md) of Locals. File and function names refer to the `locals` package.

This part of the package turns a stock mlx-lm model into the model Locals decodes with. It changes how the model computes, not what it computes (`locals/`lm.py`). Loading runs four steps, in the same order in `Runtime.__init__` (`runtime.py`), which also runs `guard_attention` after the first (passes of a query length MLX's attention kernel cannot run on this GPU are split in two), and in `locals.lm.patch` (`lm.patch`):

1. `set_compute_dtype`: with the default `dtype="auto"`, on M1 and M2 GPUs, cast the bf16 arrays (weights, quantization scales and biases) to fp16, unless the model's source is fp16-special, a value exceeds 6.0e4, or the probe check fails; in those cases it stays in bf16.
2. `fuse_model`: concatenate projections that read the same input, and swap in fused layer classes.
3. `enable_small_m`: send the model's 4-bit quantized matmuls through Locals's router.
4. `tune_model`: pick the fastest kernel for each matmul shape and row count. The choice is cached on disk for each GPU architecture.

After these steps, `mx.set_wired_limit` is set to the GPU's `max_recommended_working_set_size`.

## Files

| File | What it holds |
|---|---|
| `runtime.py` | `Runtime` (load, patch, constraint and prompt caches), `set_compute_dtype`, `native_bf16` |
| `lm.py` | Drop-in replacement for mlx-lm: `load`, `patch`, and mlx-lm's `generate`, `stream_generate`, `batch_generate` and `BatchGenerator` re-exported unchanged |
| `fuse.py` | Projection fusion, `FusedAttention`, `FusedFlintAttention`, `FusedMLP`, `SplitProjection`, `FusedGatedDeltaNet`, `FusedGatedAttention`, `FusedHybridBody` |
| `models/` | Architectures the pinned mlx-lm lacks, registered where mlx-lm imports models (`models.register`): Flint; DiffuSpace's Clover (its causal pass as Qwen3.5-MoE's text model, `Model`, and the whole block-diffusion model, `BlockModel`) |
| `moe.py` | Flint's and Qwen3.5-MoE's mixture-of-experts blocks: the routers and the expert kernels, fused expert gate/up, the combines, the layers' steps |
| `qk_kernel.py` | Per-head q/k RMS norms and RoPE in one kernel (Flint's attention) |
| `qkv.py` | 8-bit keys and values for full-attention layers (`LOCALS_KV_BITS=8`): the attention kernels and the batch cache |
| `kernels.py` | Matmul routing (`enable_small_m`, `_linear_switch`), the dense prefill path, row packing, `qmm_small`, the autotuner and its cache, and output-layer helpers |
| `qmv_multi.py` | The `multi` and `v3` multi-row matrix-vector kernels |
| `qmv_frag.py` | The `frag` and `fragh` kernels, which dequantize straight into 8x8 MMA fragments, and `fragq`, which feeds them the weights as 1 + q/16 and applies the scales per group |
| `mma_kernel.py` | `qmm_mma`: MLX's tiled quantized GEMM rebuilt with 8- or 16-row tiles |
| `nax_kernel.py` | The `nax` kernels: 4-bit matmuls of 1 to 32 rows on the M5's neural accelerators (Metal 4 tensor operations) |
| `gdn_kernel.py` | One Metal kernel for a Gated DeltaNet decode step |
| `norm_kernel.py` | Fused residual add + RMS norm, the bits of MLX's add and norm (Flint's and Qwen3.5-MoE's layers; switched off for dense Qwen3.5) |
| `blockdiff.py`, `blockdiff_service.py` | Clover's block diffusion (`--blocks`): DiffuSpace's decoding, and the server's service for it |
| `clover_fast.py`, `moe_tiles.py` | Kernels for Clover's preview pass: its routing with the block coreset, its GDN convolution, a block's experts in tiles |
| `convert_clover.py` | Clover's MLX build from its bf16 checkpoint, a file at a time |
| `attention.py` | Ragged batch decode attention (present, switched off) |
| `defaults.py` | Model short names, the default model for this Mac, and checks run before a download |
| `presets/` | Shipped tuning results for each GPU architecture |

## Loading: `Runtime` and `locals.lm`

`Runtime.load(path, **kw)` resolves short names with `defaults.model_id` and then calls `mlx_lm.load` (`runtime.Runtime`). The constructor takes these arguments (`runtime.Runtime.__init__`):

| Argument | Default | Meaning |
|---|---|---|
| `fuse` | `True` | Run `fuse_model` |
| `small_m_kernel` | `True` | Passed to `enable_small_m(small_m_kernel, model)`, which always patches the layer classes. True turns on routing (tuned small-row kernels and the dense prefill path) and runs `tune_model`. False keeps MLX's kernels with row packing only, and sets the process-wide switch off for every model patched so far |
| `prompt_cache_bytes` | `2 << 30` | Byte limit of the prompt (KV) cache |
| `dtype` | `"auto"` | Passed to `set_compute_dtype` |
| `tune` | `None` | Tuning mode. Falls back to `LOCALS_TUNE`, then to `"auto"` (`runtime.Runtime.__init__`) |

The server's engine loads with `tune=os.environ.get("LOCALS_TUNE", "background")` instead (`service.BatchService._run_locals`).

After patching, `Runtime` rejects any model without `model.embed_tokens`. It runs the decoder families that have one: Llama, Qwen, Gemma, Mistral, Phi-3 and similar (`runtime.Runtime.__init__`). It then builds the token vocabulary, the byte trie and an `Engine`, plus two caches:

- **Compiled constraints**, least recently used first. The cache holds at most `LOCALS_MAX_COMPILED` entries (default 16) and at most `LOCALS_COMPILED_GB` GB of estimated tables (default 1.5) (`runtime.Runtime.__init__`, `runtime.Runtime._cput`). The limit exists because one token-table constraint takes 60-100 MB on the GPU (a 40-state regex on Qwen3.5's 248k vocabulary). Kept forever, a few dozen of them filled 4-5 GB, and the batch's memory check then admitted one request at a time (`runtime.Runtime.__init__`). Each constraint's device tables are built at compile time, not inside the first request (`runtime._prepare_device_tables`).
- **Prompt cache**: mlx-lm's `LRUPromptCache(max_size=16, max_bytes=prompt_cache_bytes)`. Locals's KV caches never change arrays in place, so a cache hit shares arrays instead of copying them (`runtime.Runtime.__init__`).

`Runtime.sample()` shares the prompt through tree attention, so it needs attention in every layer. It raises an error on models with recurrent layers, such as Qwen3.5 (`runtime.Runtime.sample`).

`locals.lm` is mlx-lm with the model patched. `load(path_or_hf_repo, *args, dtype="auto", tune=None, **kwargs)` calls `mlx_lm.load` and then `patch` (`lm.load`). `patch(model, dtype="auto", fuse=True, kernels=True, tune=None, log=None)` runs the four load steps in place (`lm.py`). `patch` is also exported as `locals.patch` (`__init__.__getattr__`).

## Compute dtype: fp16 or bf16

Apple GPUs before the M3 generation emulate bf16 arithmetic. On those GPUs (architecture `applegpu_g13*` = M1, `applegpu_g14*` = M2), MLX's fp16 kernels run up to 1.7x faster than its bf16 kernels. `native_bf16()` returns False on them (`runtime.native_bf16`). The code comment also notes that on Qwen3.5 checkpoints, fp16 is closer to a float32 run than bf16 is (`runtime.set_compute_dtype`).

`set_compute_dtype(model, dtype="auto")` casts only the bf16 arrays: activation-dtype weights, and quantization scales and biases. float32 arrays such as `A_log` are kept as they are (`runtime.set_compute_dtype`, `runtime.set_compute_dtype`). It returns the dtype in use.

| `dtype` | Behaviour |
|---|---|
| `"auto"` | Cast to fp16 only when all of these hold: the model has bf16 arrays, the GPU has no native bf16, `_fp16_special(model)` is false, and the safety checks below pass. Otherwise stay in bf16 (`runtime.set_compute_dtype`) |
| `"float16"` | Cast with no checks |
| `None` or `"bfloat16"` | No cast (`runtime.set_compute_dtype`) |

Safety checks under `"auto"`:

1. **Range.** If any bf16 array holds a value with magnitude above `6.0e4`, the model stays in bf16. 65504 is fp16's largest finite value, so larger values would become inf (`runtime.set_compute_dtype`; the code comment there claims the cast is exactly reversible below that limit). It is not exact for very small values: fp16 has fewer exponent bits than bf16. That is why a failed probe restores the saved bf16 arrays instead of casting back (`runtime.set_compute_dtype`, `runtime.set_compute_dtype`).
2. **Probe.** A fixed 12-token probe runs before and after the cast. fp16 is kept if the new last-position logits are finite, have the same argmax, and differ by at most `0.05 * max|ref| + 0.5`; failing that, it is kept if a float32 run of the probe (built from the saved bf16 arrays) has fp16's argmax and fp16's largest error against it is no larger than bf16's, the Gemma 4 E4B case (`runtime.set_compute_dtype`). On failure, the saved bf16 arrays themselves are put back, because casting fp16 back to bf16 loses small values (`runtime.set_compute_dtype`).

`_fp16_special` keeps a model in bf16 when the source of its text model's module contains the substring `float16`. The heuristic is meant to catch models that treat fp16 specially. The test also matches `bfloat16`, so any module that mentions bf16 stays in bf16 as well. In the installed mlx-lm 0.32.0 that includes deepseek_v3, deepseek_v32, minimax, mimo_v2_flash and mistral4. Gemma 3 clips every residual add through fp32 in fp16. In bf16 it runs its plain path, which is also faster: 68 against 53 tok/s for Gemma 3 4B on an M2 (`runtime.py`).

The chosen dtype matters downstream. Tuning results are keyed by activation dtype, the `fragh` kernel is fp16-only, and the dense prefill path dequantizes into the activation dtype (see below).

## Projection fusion (`fuse.py`)

q/k/v become one quantized matmul, and gate/up become another. Quantization groups run along the input dimension, so concatenating weights along the output dimension keeps the quantized format exact. Fewer, larger matmuls mean fewer kernel launches per step (less CPU encode time) and better GPU occupancy for the small k/v projections (`fuse.py`).

`_fusable(mods)` requires every module to be a `QuantizedLinear` with the same `(bits, group_size, mode)`, and a bias on all of them or none (`fuse._fusable`, `fuse._same_quant`). For example, mlx-lm's mixed quantization recipes give `v_proj` 6 bits next to 4-bit q/k, and such layers are left unfused. `_concat_quantized` builds the fused layer without calling `QuantizedLinear`'s constructor. The constructor quantizes random weights, which draws from the global random state, and seeded sampling would then differ from mlx-lm's (`fuse.py`).

What `fuse_model(model)` changes in each layer (`fuse.fuse_model`):

| Target | Condition | Result | Models |
|---|---|---|---|
| Attention q/k/v | Class `Attention` from module `qwen2` or `llama`; q/k/v quantized and fusable | `qkv_proj`; class becomes a subclass of `FusedAttention` | Llama, Qwen2-style |
| `GatedDeltaNet` `in_proj_{qkv,z,b,a}` | Has `in_proj_qkv`; all four fusable, or qkv/z and b/a each (b and a quantized apart: Clover's at 8 bits) | `in_proj_all`, one matmul or a `SplitProjection` of two; `FusedGatedDeltaNet` | Qwen3.5 linear-attention layers |
| `Qwen3NextAttention` q/k/v | Fusable, no biases | `qkv_proj`; `FusedGatedAttention` | Qwen3.5 / Qwen3-Next full-attention layers |
| MLP gate/up | `gate_proj` and `up_proj` quantized, fusable, and `_swiglu_mlp` passes | `gate_up_proj`; `FusedMLP` | SwiGLU MLPs |
| `Qwen3NextSparseMoeBlock` | mlx-lm's (module `qwen3_next`) | `FusedQwenMoe` (`moe.fuse_qwen_moe`: router and shared-expert gate one matmul, experts' gate/up, shared expert's gate/up) | Qwen3.5-MoE, Qwen3-Next, Clover's causal pass |
| Qwen3.5 `DecoderLayer` | its MLP a `FusedQwenMoe` | `FusedQwenLayer` (the MoE residual add in the expert combine) | Qwen3.5-MoE |
| Text model body | `Qwen3_5TextModel` and hidden size divisible by 128 | `FusedHybridBody` | Qwen3.5: inactive for dense models (see below); Qwen3.5-MoE's layers step through it (`_moe_steps`) |

A fused class is a new subclass named `"Fused" + <original name>`, with the fused class first in the method resolution order. The instance keeps all its other attributes (`fuse._fuse_mlp`, `fuse.py`, `fuse.py`, `fuse.py`). `fuse_model` returns a count of the layers it walked.

- `_swiglu_mlp` accepts an MLP class only if two checks pass (`fuse.py`):
  - **Source check:** `__call__` uses `swiglu(` or `silu(`, and no `gelu`, `relu`, `tanh` or `sigmoid(`. Gemma's MLPs use GELU.
  - **Numerical check:** the fused computation on a copy of the layer's own weights must match the original within `0.02 * (1 + max|ref|)` on a `(1, 3, dims)` input. The input comes from a fixed key, not the global random state.

  The verdict is cached per class in `_MLP_KINDS`.
- LFM2 names its MLP `feed_forward`, so `mlp` is `None` and nothing is fused there (`fuse.fuse_model`).
- In `FusedGatedAttention`, the query projection has width `2 * h * hd`: it carries queries and an output gate. The output is multiplied by `sigmoid(gate)` (`fuse.FusedGatedAttention._mix`).
- `FusedGatedDeltaNet` and `FusedGatedAttention` split their `__call__` into three parts: input projection, `_mix`, output projection. With that split, `locals/mixed.py` can run the projections for a batch's decode rows and a new prompt chunk in one pass, so the weights are read once. Only `_mix` runs per group (`fuse.py`, `fuse.py`, `mixed.forward`). `mixed.supported` requires a body with `embed_tokens`, `norm`, `fa_idx` and `ssm_idx`, and layers from module `qwen3_5` or `qwen3_next` with the pre-norm attributes. Each layer's mixer must be an instance of `FusedGatedDeltaNet` (linear layers) or `FusedGatedAttention` (`mixed.py`). The MLP also runs once over both groups (`mixed.forward`).

## Gated DeltaNet decode kernel (`gdn_kernel.py`)

For one decode token, stock mlx-lm runs about fourteen small kernels per linear-attention layer. At batch 1, those launches cost more than their arithmetic (`gdn_kernel.py`). `gdn_decode` does the whole step in one kernel. Each threadgroup owns one (sequence, value head) and runs four stages (`gdn_kernel.py`):

1. conv + silu over the head's q, k and v channels, writing the new conv state
2. RMS-normalize q and k
3. the delta-rule recurrence, with each simdgroup owning `RPS = Dv / NSG` rows of the head's state: 16 rows with Dv = 128 and NSG = 8. Each lane holds a float4 of each row. The module docstring's "per 4 rows" does not match the code.
4. gated RMS norm of the output

The grid is `(32, NSG, B * Hv)` with `NSG=8` (`gdn_kernel.gdn_decode`, `gdn_kernel.gdn_decode`). Rounding to the activation dtype happens where mlx-lm's operations round, so results match the stock path up to summation order (`gdn_kernel.py`): beta is MLX's `mx.sigmoid` of b and the decay's softplus MLX's `logaddexp(x, 0)`, each of their steps rounded in the activation dtype (`gdn_kernel._HEADER`); the convolution's SiLU is x times that sigmoid; the gated norm is MLX's `rms_norm` (`w * T(x * r)`) times SiLU(z) in float32 (mlx-lm's `precise_swiglu`); every exp a precise one, as MLX's own kernels compute it (a custom kernel compiles with fast math). Until these, the kernel rounded the sigmoid and softplus once (in bf16 that differs from MLX's in 17% and 15% of values), and a small Qwen3.5-MoE's decode steps drifted a few percent from stock's; now they are its logits bit for bit at one row (`tests/test_qwen_moe.py`), no slower. For S > 1 the state rows stay in registers across steps. A single step streams them instead, which keeps register pressure as in a plain decode kernel (`gdn_kernel.py`). The `states=True` variant (`locals_gdn_step3_states`) also writes the recurrent state after every token, which is needed to rewind a partially accepted draft (`gdn_kernel._kernel`).

`FusedGatedDeltaNet._mix` uses the kernel only when all of these hold (`fuse.py`):

- `USE_GDN_KERNEL` is True (default).
- `S <= _GDN_MAX_S` (4), or up to 16 tokens when `cache.want_states` is set.
- There is no mask, and the cache holds both a conv state and a recurrent state.
- `cache.lengths` is `None`. Ragged padded batches set lengths and take the masked stock path (`hybrid_batch.HybridBatchEngine._forward`).
- The recurrent state is float32, and the conv state has the projection's dtype.
- `_gdn_ok` was set at fusion time: the norm eps is `1e-6`, the conv weight has shape `(C, KC, 1)`, and `supported(Hk, Hv, Dk, Dv, KC)` is true. That function requires `Dk == Dv == 128`, `Hv % Hk == 0` and `KC >= 2` (`fuse.fuse_model`, `gdn_kernel.supported`).

When `want_states` is on and S > 1, the layer stores `cache.token_states` and `cache.token_conv`. `engine.rewind` uses them to roll back rejected draft tokens (`fuse.FusedGatedDeltaNet._mix`, `engine.rewind`). The `set_want_states` helper is defined at `engine.py`. Otherwise the stock recurrence (`gated_delta_update`) runs. Its q/k norm uses eps `1e-6 / Dk` so that it matches the reference l2norm, as mlx-lm 0.32 and later do (`fuse.FusedGatedDeltaNet._mix`).

**Prompts and MLX's op.** An unpadded prompt (no mask, no `cache.lengths`, more than 4 tokens) can go through MLX's own `mx.fast.gated_delta_update` instead, a chunked algorithm of matrix products with a version for the M5's neural accelerators. `kernels.gdn_fast` decides per layer shape and dtype, from timings `locals tune` makes with the whole model reading 512- and 2,048-token prompts each way (`gdn_model_tune`, keys `GDN{Hk}x{Hv}x{Dk}x{Dv}xT{512|2048}`); the op is kept where it is at least 3% faster. Timed alone, the op is 1.3-1.9x faster on an M2 Pro, but inside Qwen3.5-4B it was not (909 against 898 ms for 512 tokens, 3,389 against 3,366 ms for 2,048), so it is timed whole. Untimed shapes keep mlx-lm's kernel. With a mask the op is 12-50x slower and is never used. `LOCALS_GDN_FAST=1` or `0` forces it; `locals tune --report` shows the timings.

## Flint: a mixture of experts (`models/flint.py`, `moe.py`, `qk_kernel.py`)

Recursive's Flint is a 40B mixture of experts with 2.5B parameters active per token: 49 layers, the first two with a
dense SwiGLU MLP, the other 47 with 256 routed experts (6 per token, by sigmoid scores renormalized to sum to 1) and a
shared expert; grouped-query attention (32 query heads, 4 key/value heads of 128) with q/k RMS norms and a learned sink
per head, RoPE only on the sliding-window layers (3 of 4, a window of 513). mlx-lm has it only in an unreleased pull
request (ml-explore/mlx-lm#1998), so `models/flint.py` carries the same computation in MLX operations, and
`models.register()` puts it at `mlx_lm.models.flint` before `Runtime.load`, `locals.lm.load` and the stock engine load
a model. Where the installed mlx-lm has its own, that one stays: `mlx/<model>` is then still stock mlx-lm. With MLX
0.32.4 the module's logits equal the pull request's bit for bit.

`fuse_model` recognizes both modules' classes (by class and module name) and makes, in each layer:

| Target | Result |
|---|---|
| Attention q/k/v | `qkv_proj`, `FusedFlintAttention`; q/k norms and RoPE through `qk_kernel.qk_norm_rope` (16-bit activations, heads a multiple of 128), which also writes v as its own array; `_mix` (between the projections) for `mixed.py` |
| `SparseMoeBlock` | `FusedFlintMoe`: router logits from `moe.router`, the choice from `moe.route_topk`; few tokens through `moe.experts_up` / `moe.experts_down`, more through `FusedSwitchGLU` |
| `switch_mlp` gate/up | One `QuantizedSwitchLinear` of twice the rows (`moe.fuse_switch_glu`): quantization groups run along the input, so it is exact |
| Shared expert | gate/up fused as any SwiGLU MLP (`FusedMLP`) |

**Why.** On an M4 Pro a one-token Flint step is about 600 dependent kernel launches deep, and each costs about 5 us
before it reads anything, whatever its size (a residual add: 5.5 us, an RMS norm: 6.6 us). The 1.3 GB a token reads
take 5.9 ms at the 218 GB/s this M4 Pro reads at best (a kernel summing a 256 MB buffer; MLX's qmv reaches 202 on a
large 4-bit matrix, 125 to 155 on Flint's 5 MB attention projections); stock mlx-lm's step took 11.7 ms. The kernels here remove launches on that chain:

- **Router** (`router_logits`): one simdgroup per expert in 64 small threadgroups, the bf16 weight read as it is and
  every dot product summed in float32 (as `x.astype(f32) @ W.astype(f32).T`, within 1.5e-7). MLX's float32
  matrix-vector product took 42 us a layer, after casting the 1 MB weight to float32 each step: 2.0 ms of the step,
  now 0.6. Up to `ROUTER_MAX_ROWS` (16) rows; a prompt takes the stock path.
- **Choice** (`route_topk`): sigmoid, top-k by k rounds of a simdgroup argmax (ties to the lower index), and the
  renormalization, in one kernel instead of five. The same experts as the reference on every row tested.
- **Experts** (`experts_up`, `experts_down`): gather, slices, swiglu, gather, weighted sum, cast and the shared
  expert's add were eight dependent kernels, 70 us a layer at one token. `experts_up` writes swiglu(gate, up) for every
  (token, expert) pair; `experts_down` sums the down projections with the routing weights and adds the shared expert's
  output. A lane takes a group of 64 weights; a value is masked in place against inputs pre-divided by its power of
  two, as MLX's qmv does (shifting each value made the first version compute-bound: 74 us against MLX's 42). On an M4
  Pro at one token: 43.7 and 35.6 us against 42 + 5 (swiglu) and 45 for the stock kernels. They round where the stock
  graph does, so values differ by a last bit where a sum runs in another order. Up to `EXPERT_MAX_ROWS` (16) tokens;
  more take `FusedSwitchGLU`, whose sorted gather reads each expert once for all its tokens.
- **q/k** (`qk_norm_rope`): both norms and both rotations in one kernel instead of four. It saved only 0.1 ms: the q
  and k branches did not wait on each other, and MLX's encoder overlaps them. The norms equal `mx.fast.rms_norm`
  exactly (the normalized value is rounded before the weight's product); the rotation equals `mx.fast.rope` in 99% of
  values and is a last bit off in the rest.

**Command buffers.** MLX ends a command buffer once the inputs encoded into it pass `MLX_MAX_MB_PER_BUFFER` (50 MB on
an M4 Pro), counting each input at its whole size. The expert kernels were handed a layer's whole expert tensors (350
MB, of which a token reads 8), so each of them ended a command buffer: about 94 a step, and the GPU waited between them
(the main thread spent 28% of a step waiting for it, 13% after). `moe._experts_buffers` hands them a view of the first
expert of each (the same buffer, which the kernels index past it): 8.75 ms a step instead of 10.2, the same tokens,
114.6 tokens/s for one request instead of 102 and 264 instead of 255 at eight. Raising the limit itself does as well
for one request, but a command buffer then keeps more of a prompt's temporaries alive: reading 8 x 256 tokens, or a
32k prompt, ran out of GPU memory on 24 GB.

**Memory.** Two things held memory that stock mlx-lm does not: the projections fused away stayed in MLX's buffer cache
(2.5 GB of Flint's; for Flint, `fuse_model` now ends with `mx.clear_cache()`), and v, a view of the fused projection, kept the
whole projection of each sliding-window layer alive in the ring that keeps a prompt's first chunk as it is handed it
(0.7 GB for a 2k chunk; `qk_norm_rope` writes v as its own array). After a 2,048-token chunk the patched model holds
what stock mlx-lm holds, 18.04 GB.

**Mixed passes** (`mixed.py`): decode rows and a prompt chunk in one pass work for Flint as for Qwen3.5. Through the
server, three replies streaming while an 8,000-token prompt arrived stalled for at most 2.8 s, against 9.7 s (the whole
prompt) with `LOCALS_MIX_CHUNK=0`; the prompt's first token came after 10.4 s against 9.8, the three wrote 104 tokens/s
together against 107.

**A batch's experts read once** (`moe.expert_order`). The expert kernels took a simdgroup per (token, expert) pair,
so rows that chose the same expert (16% of a batch's choices at 8 rows of Flint's routing, 27% at 16) each read it.
They now take the pairs in the order of their experts (one argsort a layer, none for one token): the simdgroup at the
start of each run of one expert loads its weight words once and serves the run, the others return; the down
projection writes each pair's output and a combine sums a token's pairs in their order, so the outputs are bit for
bit the same. Alone, at Flint's shapes, up and down take 334 us a layer at 8 rows instead of 367 and 620 at 16
instead of 700; even at one row 2% less (the down projection's pairs run side by side). Decoding: 284 tokens/s at 8
rows instead of 265, 313 at 16 instead of 288, 117 at one instead of 115; through the server, every reply streaming,
276 at 8 clients instead of 256 and 305 at 16 instead of 281. MLX's sorted gather (each expert once) was slower at
8 and 16 rows than either: 233 and 277.

**Several rows: the arithmetic, and fewer kernels a layer.** In the paper's terms (Breaking the Space Barrier,
section 3) a step costs a weight stream, one read of the weights it needs; for a mixture of experts that is the dense
weights plus the union of the experts the batch chose: at 8 rows of Flint's routing 40 of 256 a layer, 2.6 GB a step,
12 ms at 218 GB/s, against 0.4 GB at one row. The paper notes that its GPU's step is bound by arithmetic from four
rows, and Flint's expert kernels are: at 8 rows
they took 15.7 ms a step, with the loads alone (the dot products made trivial) 11.8 and the arithmetic alone (no
weight loads) 13.7. Two changes cut the arithmetic, each row's sums in the same order (the same bits):

- From `EXPERT_ROWS_FROM` (4) tokens a simdgroup of `experts_up` takes 2 hidden units, so each chunk of the input is
  converted and prescaled once for 4 weight rows, and a lane of `experts_down` 4 output rows (3 or 4 units, or 8
  rows, spill registers: 18.7 ms).
- A 3-bit value becomes a float without an int-to-float conversion (`qdot8`): masked in place, ORed into the mantissa
  of 2^23 and 2^23 subtracted, exact below 2^23; the value in bits 21 to 23 is shifted down 3 and its input divided by
  2^18 instead of 2^21, the same product.

The experts then take 14.2 ms a step at 8 rows, 8.1 instead of 8.6 at 4, 26.2 instead of 29.3 at 16, the same at
one. Then the kernels between them, each a launch and a wait on the one before (about 2 us a dependent kernel at 8
rows), all with the same bits:

- `route_topk(order=True)` also orders the slots by expert: one threadgroup holds every token (up to
  `ORDER_MAX_ROWS`, 16), and its threads rank the slots, so no sort kernel a layer.
- The MoE layer's residual add is the combine's last step (`experts_down(residual=...)`, `FusedFlintLayer`).
- The attention's residual add and the post-attention norm are one kernel (`norm_kernel.add_rms`), laid out as MLX's
  own RMS norm (4 values a thread, a simdgroup sum, the partial sums summed by one simdgroup, a precise rsqrt):
  the bits of `x + r` and `mx.fast.rms_norm`, checked on 42M values.
- Each layer's input norm is computed by the kernel that ends the layer before it (`FusedFlintBody`): the combine,
  laid out the same way, writes the layer's output and the next layer's norm of it; after the two dense layers,
  `add_rms`.

An 8-row step went from 926 kernels to 733. Decoding in the process (the package's source as 256-token prompts, 48
steps): 305 tokens/s at 8 rows instead of 284, 225 at 4 instead of 211, 179 at 2 instead of 170, 340 at 16 instead
of 314, 117 at one as before; the same tokens and logits at 1 to 16 rows over 41 steps. The batch engine alone
writes 300 at 8 rows (224 at 4). Through the server, with the clients on the same Mac and every reply streaming, 289
to 293 at 8 clients (276 before), 216 at 4 (204), 320 to 324 at 16 (305): a busy CPU slows the GPU here (the engine
alone at 8 rows: 293 with four processes parsing JSON beside it, 299 without), and the server's steps are 1% longer
than the engine's alone, not more.

**Few-row matmuls: the weights as 1 + q/16** (`qmv_frag.launch_q`, the `fragq` family). At 8 rows the dense matmuls
(the attention's projections, the shared expert, the output layer) took 8.5 ms of a 26.4 ms step, twice what their
weights take to read. Each weight serves one 8 x 8 matrix product there, and `qmv_frag` spends half again the
product's own arithmetic dequantizing its two weights a lane (shift, mask, int-to-float, the scale's fma, twice): this
GPU's matrix units run at its scalar rate (2.85 T multiply-adds/s), from the same ALUs. All of Metal's mixed products
run at that rate (bf16 x bf16 into float among them), so `fragq` feeds the matrix units a nibble as it is: in the top
of a bf16 (or fp16) mantissa it is 1 + q/16, exact, and two of them 16 bits apart in one word take one shift, one
mask and one or; a k-order in which each lane's two weights are such a pair (no copy of the weights in another
layout: there is no room for one); the activations as they are; and a group's scale and bias on a float accumulator,
as MLX's qmv factors them. A quarter of the product's arithmetic instead of a half. Timed as the model runs them, each
call waiting on the one before, at 8 rows: q/k/v 44.4 us against `qmv_frag`'s 51.1, the attention's output 41.1
against 45.5, the shared expert's down projection 13.0 against 14.3, the output layer 968 against 1,124. Timed the
tuner's old way, back to back, the kernel was 5 to 15% faster on every shape and the model 2% slower at 8 rows (it
chose 4 tiles a simdgroup, which overlap well and wait badly), so the tuner now times chains (below,
"Autotuning"). With its choices, decoding at 8 rows went from 302 to 317 tokens/s, at 16 from 339 to 352, at 4 from
224 to 227 (three interleaved rounds each); one row is MLX's kernel as before. Through the server, 8 clients 305
tokens/s instead of 289 to 293, 4 clients 221, 16 clients 338.

**Prompts in bands** (`FusedFlintAttention._banded`). A 2,048-token chunk in a sliding-window layer was one attention
of 2,048 queries against the ring's 512 keys and the chunk's 2,048 under a mask, four in five of the scores masked.
In bands of `FLINT_BAND` (512) queries against the keys their window reaches (slices of the keys and of the mask), the
scores are the same (equal logits) and a prompt is read 11% faster from 4k tokens: 844 tokens/s instead of 760 at 4k,
694 instead of 627 at 16k, 560 instead of 504 at 32k (the body, in 2,048-token chunks). `LOCALS_FLINT_BAND=0` turns it
off.

**Prompt chunks.** The engine reads a prompt in 1,024-token chunks, halved as the context grows (`PREFILL_SCORES`: 512
past 15k, 256 past 31k) so that a chunk's attention scores stay small. Flint's prompt attention holds none (MLX's fused
causal attention in the full layers, the bands in the window layers), and 2,048-token chunks read 5% faster than
1,024 (851 against 799 tokens/s at 4k, 560 against 535 at 32k, 0.3 GB more at the peak). `fuse_model` sets the
model's `prefill_chunk_tokens` (2,048) and `prefill_scores` (2^26), which the engine takes over its defaults: 2,048
tokens to about 30k, 1,024 to about 63k. Through the server the first token of a 30k-token prompt came after 54.3 s
instead of 58.1; a 35k prompt, the most a 24 GB Mac's server takes, read without error, and a 60k one with
`LOCALS_KV_BITS=8` in 159 s instead of 177.

**The batch caches' mask.** The engine decodes even one request through mlx-lm's batch caches, whose mask for one
query a row only marks left padding. With no row padded it is all true, and it costs MLX's attention 40% (510 us
against 358 a layer at 30k keys): one request wrote 61.7 tokens/s at 30k through batch caches, 72.4 through its own.
`fuse._unpadded` reads the padding on the host (already evaluated, but for the step after rows join or leave) and the
full-attention layers then attend without the mask: 72.1. A padded batch keeps it. The window layers' masks (513
keys) cost about 1%, and stay.

**Passes of 5 to 8 tokens** (`fuse.attend_few`). MLX's attention runs a one-query kernel up to 4 queries and its matrix
kernel from 9; from 5 to 8 it takes a path twice as slow as 9: a full-attention layer at 1,024 keys took 306 us for 8
queries against 159 for 9, at 8,192 keys 2.1 ms against 1.2, a window layer 173 against 96 us. Forced spans of a
typed tool call (5 or 6 tokens) hit it in all 49 layers. Such a pass is padded in front to 9 queries: MLX's
causal mask aligns the last query with the last key, so the real queries keep their keys; an array mask gets copies
of its first row in front; the padding's outputs are dropped. A pass of 8 tokens after 1,024 takes 38.6 ms instead of
47.2.

**Long contexts.** One growing prompt (the package's own source), read 2,048 tokens at a time, one request writing:

| Context | Reading, mlx-lm / Locals | Writing, mlx-lm / Locals | Peak GPU memory, mlx-lm / Locals |
|---|---|---|---|
| 2,048 | 642 / 647 tok/s | 79 / 90 tok/s | 19.54 / 19.86 GB |
| 8,192 | 545 / 551 | 74 / 84 | 20.29 / 20.45 |
| 16,384 | 456 / 459 | 67 / 74 | 20.51 / 20.71 |
| 24,576 | 383 / 390 | 63 / 71 | 20.77 / 20.90 |
| 32,768 | 329 / 336 | 59 / 59 | 21.02 / 21.12 |

Reading a prompt is as fast as mlx-lm's (these kernels are for few rows). At 32,768 tokens both pass the 20 GB the GPU
was given and the gain is gone; the server lets a 24 GB Mac's Flint take about 35,000 tokens. Through the server, the
first token of a 13,094-token prompt came after 20.7 s and the reply went on at 74 tokens/s.

**8-bit keys and values** (`qkv.py`, `LOCALS_KV_BITS=8`). Flint's 14 full-attention layers keep 28 KB a token in
bf16, so a 24 GB Mac holds about 35,800 tokens next to the model; its config asks for 8-bit keys and values
(`kv_key_bits`, `kv_value_bits`, which mlx-lm's loader drops). With the switch on, those layers keep MLX's affine 8-bit
groups of 64 (15 KB a token; the sliding-window layers stay bf16, 513 tokens each) and the server takes requests up
to 67,584 tokens: a codename hidden in 37,844 and in 59,778 tokens of source was found (87 and 177 s). mlx-lm's own
quantized cache attends with three kernels and no attention sinks, which Flint has on every head, so `qkv.attend`
is a kernel of its own: Flint's 8 query heads a key/value head are an 8 x 128 matrix, so 8 keys at a time give an
8 x 8 tile of scores on the GPU's matrix units with no sums across lanes; each lane unpacks its own two elements of
each fragment straight from the 8-bit words (no threadgroup memory), with the head dimensions permuted so that a lane
reads whole words; a head's softmax row lives in four lanes; a second kernel joins the blocks with the sink. A
prompt unpacks what it reads and goes through MLX's attention. `BatchQuantizedKVCache` is mlx-lm's BatchKVCache with
every buffer a (words, scales, biases) triple, and the batch engine counts its bytes a token.

One 32,768-key layer (14 of them chained), against MLX's bf16 attention at 389 us: a simdgroup a head, summing
across lanes each key, 1,240 us; the 8 heads in one simdgroup, 950; on the matrix units through threadgroup memory,
1,310 (one threadgroup a core); fragments filled in place, 889; dims permuted for whole words, 498; the join in
parallel, 479. At 8,192 keys: 173 against 120. Each new token's keys and values are quantized and stored at every
step (two quantize kernels and six writes a layer), which is most of what the switch costs at short contexts:

| Context | Writing, 8-bit / bf16 | Keys and values, 8-bit / bf16 |
|---|---|---|
| 1,024 | 83.8 / 96.5 tokens/s | 0.06 / 0.07 GB |
| 8,192 | 75.8 / 84.8 | 0.17 / 0.28 GB |
| 16,384 | 67.7 / 77.2 | 0.29 / 0.51 GB |
| 32,768 | 59.2 / 61.4 | 0.54 / 0.98 GB |

So it is a switch for long work, off by default: twice the context for 4 to 13% of the writing speed.

**Forced spans in one pass, typed calls.** One request alone gets its forced spans in one pass (`hybrid_batch._solo`),
which the engine refused for Flint because sliding-window caches cannot be rewound. A forced span never is: its
tokens are certain, and only the pass's last position is sampled. So `_solo_ok` now takes `BatchRotatingKVCache`
and `qkv.BatchQuantizedKVCache` layers for spans, and drafts (which a rejection rewinds) stay off for them
(`_solo_drafts_ok`). Flint's tool calls then gain from `--typed-tools` (`FlintXML.typed_call`, docs/decoding): the
grammar writes parameter names, so `<arg_key>filePath</arg_key><arg_value>` after `read` is a span of five tokens.
On 12 OpenCode-like tasks the untyped grammar let Flint break 4 of 12 greedy calls and 11 of 39 sampled ones
(`path` for `filePath`, `bash` without its required `description`); the typed one none, with 10% fewer passes
(603 against 672), and 99.5 tokens/s against 98.8 when the spans go a token a step (docs/benchmarks, section 11).
A pass of several tokens reads every token's experts, so a pass over a sampled token and its span of five costs
about three one-token steps (34 ms against 10), not one.
The typed grammar fixes the schema's order, so a model that would write `path` before `pattern` writes `pattern`
first; with `--walk`, which large tool sets need, spans go a token a step.

Tried and left out:

- **Prompt-lookup drafts** (`hybrid_batch._solo`), left out: this work keeps to decoding as it is, not speculative.
  They need the window layers rewound (built and measured: a ring of the window plus 64 slots per request, a linear
  batch cache dropping its oldest slots 64 at a time, as fast as mlx-lm's rings) and drafts weighed against their
  pass, which reads every draft token's experts (16 tokens 61 ms, against 8.7 for a step): by how far back the lookup's
  match goes (14% of a draft kept at a bare 3-token match, 98% from 40 on), one request then wrote a file rewrite 1.41x
  faster and a quoted document 1.67x (1.86x sampled), prose unchanged; under Qwen3.5's own width rule an agent's
  thinking was 6% slower. A pass of several tokens rounds differently from one-token steps, so a greedy reply with
  drafts left the one without them at a near tie.
- A fused residual add + RMS norm body (`norm_kernel.add_rms`) for one row: no faster here either (8.84 against 8.70
  ms a step); a custom kernel's call costs the CPU more than MLX's two, and the step is as much the CPU's as the
  GPU's. With several rows the GPU bounds the step, and `FusedFlintBody` (above) is in: unchanged at one row.
- **The choice inside the expert kernels**, each simdgroup computing its token's top 6 from the logits (one launch a
  layer fewer): 9.23 ms against 8.80. Every simdgroup waits on the choice before its first weight load.
- **Four barrier stages instead of six** (the shared expert's kernels ordered with `mx.depends` next to the router,
  the choice and the experts' gate/up, so MLX's encoder needs fewer barriers): no change (8.75 / 8.81 against 8.68 /
  8.84 ms), nor with each layer's MoE half under `mx.compile`, which cut the CPU's graph building from 3.1 to 1.7 ms a
  step: the GPU then bounds the step at about 8.7 ms. Through the server, whose engine adds its own work on the CPU,
  one reply wrote 107.9 and 106.9 tokens/s with it against 108.1 and 107.4 without. At 8 rows, where the GPU bounds
  the step, the shared expert's kernels beside the router, the choice and the experts' gate/up (5 barrier stages a
  layer instead of 7) were slower: 294.6 tokens/s against 302.0.
- **For the expert kernels at 8 rows** (14.4 ms a step as kept): a run table from `route_topk` (at a run's first
  slot its expert and end, so a simdgroup reads one word instead of following the order and the choices), 15.9 against
  15.2 ms for the choice and the experts together, every simdgroup's weight loads then starting at once; the input
  prescaled once a layer by a kernel of its own (14.8); each row's raw weight words kept and unpacked in the loop
  (16.2); vector loads of the weights and the input (14.6); Metal's fast or relaxed math (14.6 and 14.7, other bits).
- **For `qmv_frag` at 8 rows**: the nibbles masked in place against scales prescaled for each position (within 5%
  either way), converted without int-to-float (3 to 6% slower), the next chunk's inputs loaded ahead (2 to 18% slower:
  registers); for the shared expert's shapes other tiles and split-K, within 5%. Its time is set by the matrix units
  and the dequantization they share an ALU with, which `fragq` cuts (above, "Few-row matmuls").
- **For `fragq`**: the next group's inputs, scales and biases loaded ahead (2 to 26% slower), 32-bit offsets instead
  of a pointer a tile (the same). Its threadgroup's input sums cost it 0 to 3 us in a chain.
- **A few-row kernel of Locals's own for 2 to 6 rows** (a lane one weight word of each of R rows, consecutive lanes
  consecutive words, the inputs prescaled once a word for every row): as fast as MLX's at 2 rows, slower from 4
  (registers). From 5 rows `fragq` wins, from 2 to 4 mostly MLX's own.
- `MLX_MAX_OPS_PER_BUFFER` at 100 and 400: at 4 and 8 rows slower than the default (211 and 299 against 222 and
  303 tokens/s).
- **Experts shared by rows read once**: the grid ordered so the simdgroups of one weight row for every token run side
  by side (262 against 265 tokens/s at 8 rows), and a down kernel that reads each distinct expert once for all its
  tokens (the same output; 155 against 142 us at 8 rows, 324 against 260 at 16: finding the shared experts in each
  simdgroup cost more than the reads it saved).
- **Faster 8-bit attention**: four score accumulators (448 against 477 us a layer at 32k), MLX's GQA two-pass layout
  with the scale and bias folded into the dot products (553), keys on lanes with one reduction per 32 keys (461), and
  the matrix-unit kernel on bf16 keys (520 against MLX's 394). All are bound by arithmetic, not by the bytes read.
- `MLX_MAX_OPS_PER_BUFFER` from 20 to 1,000: within noise once the expert kernels stopped ending command buffers.
- **Faster few-row matmuls** for a batch's attention projections, shared expert and output layer, which run near
  this GPU's matrix-unit rate (2.85 T multiply-adds/s, float or half alike; q/k/v at 16 rows is 168M of them, 59 us at
  best, against `qmv_frag`'s 106): `qmv_frag`'s tiles and splits swept again (within 10% of the tuner's), its half
  fragments (`launch_h`: 73 against 75 us at 8 rows, 117 against 106 at 16), the activations staged once a
  threadgroup in fragment order (no faster), two accumulators a tile (88 against 74), the input split among a
  threadgroup's simdgroups (68 to 73 against 76 to 78 at 8 rows, slower at 16). A kernel of another design,
  `fragq` (above, "Few-row matmuls"), took 5% off an 8-row step.
- The router at 8 and 16 rows split among 2 to 8 simdgroups an expert: slower (37 against 35 us at 8 rows).
- Experts' input prescaled once a threadgroup instead of in every simdgroup: the same output, slower (66.6 against
  50.9 us): the arithmetic it saved was hidden behind the loads, the barrier it added is not.
- **Faster prompt reading.** Where a 2,048-token chunk's time goes (the body, each part swapped for zeros):

  | Context before it | Chunk | Routed experts | Shared expert | Full-attention layers' attention | Window layers' attention |
  |---|---|---|---|---|---|
  | 0 | 2.41 s | 1.05 s | 0.23 | 0.08 | 0.25 |
  | 8,192 | 3.13 | 1.05 | 0.23 | 0.78 | 0.24 |
  | 16,384 | 3.88 | 1.06 | 0.23 | 1.54 | 0.26 |
  | 28,672 | 5.17 | 1.01 | 0.24 | 2.83 | 0.33 |

  The full-attention layers at 28k do 10^12 operations a layer in 202 ms, 4.9 TFLOPS, about 85% of what this GPU's
  matrix units do at most (2.85 T multiply-adds/s measured; a bf16 GEMM, 5.9 TFLOPS): no exact kernel gains much there.
  The routed experts go through MLX's sorted gather_qmm, whose tiles are 16 rows of one expert (its time steps with
  every 16 rows an expert: 8 rows cost what 16 do, 40 what 48 do; 4.8 TFLOPS in whole tiles, against 5.3 for one
  quantized GEMM of the same arithmetic): Flint's routing gives an expert 1 to 675 rows of a chunk (median 35), so
  15.7% of the rows it computes are padding, at most 0.15 s of a chunk. A grouped GEMM of Locals's own cannot take it
  back: `qmv_frag`, its best few-row kernel, at its best case (4-bit weights, cheaper to unpack than Flint's 3-bit;
  whole tiles; no gather; 256 experts' gate/up stacked into one 262,144 x 2,048 matrix) does 4.08 TFLOPS on 8-row
  tiles and 4.55 on 16-row ones, so 8-row tiles (padding down to about 7%) would give at most 3.8 and 16-row tiles
  for full experts with 8-row ones for the rest at most 4.2, against MLX's 4.0 to 4.2 with the padding; 3-bit
  unpacking, the gather and the tile schedule only take from that. An 8-row tile in MLX's own kernel for an expert's
  last rows is the change that would. Projections dequantized once a chunk for a bf16 GEMM are 6 to 7% faster alone (q/k/v 7.58 against 8.16
  ms at 2,048 rows), within noise in the model (831 tokens/s at 4k against 844); 4,096-token chunks 2.5% faster for
  0.4 GB more at the peak.

Measured on an M4 Pro (24 GB, the GPU limit raised to 20 GB), mlx-community/flint-mixed-3-4bit, 256-token prompts,
decode steps pipelined as the engines run them (all of it, and through the servers: docs/benchmarks, section 11):

| Requests at once | stock mlx-lm | Locals before | Locals |
|---|---|---|---|
| 1 | 85 tok/s | 79 | 118 |
| 4 | 177 | 174 | 212 |
| 8 | 202 | 223 | 284 |
| 16 | 245 | 279 | 313 |

**Accuracy.** Teacher-forced over 256 decode steps of real text, the patched model's choices against stock mlx-lm's
are as far apart as stock mlx-lm's against itself with the prompt read in two chunks instead of one: top-1 agreement
95.3% and mean KL 0.048, against 95.3% and 0.038 (per component 0.027 to 0.048); the loss moved by -0.02 to +0.03.
A 3-bit mixture of experts in bf16 is that sensitive to the order of a sum: a token whose sixth and seventh experts
score within rounding of each other routes differently.

## Qwen3.5-MoE and Clover (`moe.py`, `fuse.py`, `models/clover_moe.py`)

mlx-lm's Qwen3.5-MoE (`qwen3_5_moe`: Qwen3.6-35B-A3B's network, 40 layers, three Gated DeltaNet layers to each
full-attention one, 256 routed experts with 8 a token chosen by softmax, and a shared expert behind a sigmoid gate)
gets a fused block, as Flint's does. DiffuSpace's Clover is a block-diffusion model trained on that network; its
causal pass is Qwen3.5-MoE's forward, and that is how Locals runs it: `models/clover_moe.Model`, which `mlx_lm.load`
builds for `model_type: clover_moe`, is mlx-lm's Qwen3.5 text model, the same parameters under the same names without
the block-diffusion ones (`BLOCK_ONLY`). So both engines serve Clover a token a step as any model, with batching, the
grammars of tool calls and JSON schemas, the prompt cache and mixed passes, and the stock engine runs stock mlx-lm.
`BlockModel` is the whole model, for decoding in blocks (`--blocks`, below).

**Why a token a step on a Mac.** A one-token pass reads the weights a token needs, about 1.9 GB, and is bound by
memory; a block's preview (DiffuSpace's decoding) runs every matrix over all of the block's rows, which this GPU's
arithmetic (2.85 trillion multiply-adds a second) does not hide. On an M4 Pro a pass of the causal model takes 10.7 ms
for one row, 20.8 for 4 and 34.2 for 8, a preview of a 32-token block about 80, and a block of 32 resolves 1 to 2.7
tokens a step at the model's 0.95 threshold (prose: about 1). Decoding as DiffuSpace's Hugging Face code does wrote
14.6 tokens/s with stock MLX and 20.2 with Locals's kernels for the preview; the causal pass 93.6 (stock mlx-lm:
79.5).

| Target | Result |
|---|---|
| `Qwen3NextSparseMoeBlock` | `FusedQwenMoe`: the router and the shared expert's gate one matmul (`gates`, the E logits then the gate's; both 8-bit in Clover's build), the choice one kernel (`route_softmax`), a few tokens' experts in Flint's `experts_up` / `experts_down`, and a combine that adds the shared expert times its gate (`combine_gated`) and the layer's residual |
| Qwen3.5's `DecoderLayer` with that block | `FusedQwenLayer`; with `_moe_steps` (`LOCALS_MOE_STEPS`, on) the mixer's residual add in the post-mixer norm's kernel (`add_rms`) and the next layer's input norm in the combine (`FusedHybridBody`): two kernels a layer fewer |
| `GatedDeltaNet` whose b and a are quantized apart | `in_proj_all` a `SplitProjection` (qkv and z one matmul, b and a another, the outputs concatenated), so that `FusedGatedDeltaNet` and the decode kernel apply: Clover's build keeps b and a at 8 bits, and four projections of two widths fused nothing |

**The bits of mlx-lm's block.** mlx-lm computes the block in bf16: the softmax (precise) of the router's logits,
rounded; `argpartition`, here a stable ascending sort whose last k it takes, so a tie at the k-th place goes to the
higher index and the k come in ascending order; their renormalization, a running bf16 sum and a division; the k
outputs times their weights, each product and each partial sum rounded (MLX sums a 16-bit axis of 8 that way); the
shared expert times `mx.sigmoid` of its gate. `route_softmax` holds a token's logits in MLX's softmax layout (4
consecutive a lane, a pass of the simdgroup per 128) so that its normalizer is MLX's sum, and chooses and orders as
argpartition does: the same experts and weights, bit for bit (tested with ties, 32 to 512 experts); `combine_gated`
gives the block's bits on the same expert outputs. On Clover's own layers and hidden states, 1 to 8 rows: the same
experts every time, 99.95% of the block's outputs bit for bit, the rest a last bit off (the expert kernels sum each
dot product in another order than gather_qmm).

**MLX's rounding in two older kernels.** `mx.sigmoid` in bf16 rounds exp|x|, one plus it, its inverse and one minus
that; `nn.softplus` (`logaddexp`) and `nn.silu` round each step the same way, and MLX's own kernels use precise
`exp` and `rsqrt` where a custom kernel, compiled with fast math, gets fast ones. The expert kernels' swiglu (Flint's
too) and the GDN decode kernel's beta, decay, convolution SiLU and norms rounded once or used the fast functions,
which differs from MLX in 15 to 17% of bf16 values; they now compute as MLX does (`moe.mlx_sigmoid`,
`gdn_kernel._HEADER`), at no measured cost. A small Clover's patched decode steps give stock mlx-lm's logits bit for
bit at one row; before, they drifted a few percent, and on a random router a near tie flipped on any of it
(`tests/test_qwen_moe.py`).

**Where a one-token step went** (Clover, M4 Pro, pipelined steps, each part swapped for nothing in turn):

| | Stock mlx-lm | Locals before | Locals |
|---|---|---|---|
| Step | 12.6 ms | 12.3 ms | 10.7 ms |
| MoE blocks (40) | | 5.11 ms | 3.89 ms |
| GDN layers (30) | | 4.32 ms | 3.76 ms |
| Attention layers (10) | | about 1.7 ms | 1.13 ms |
| Output layer (8-bit, 248k rows, 540 MB) | | 2.3 ms | 2.3 ms |

Before, Locals's patch fused only the attention's q/k/v: the GDN layers' b and a at 8 bits next to 4-bit qkv and z
kept those layers on mlx-lm's fourteen kernels, and the MoE block was mlx-lm's (about twenty kernels a layer). The
weights a step reads take about 7.8 ms at the 250 GB/s this GPU's matrix-vector products reach on them (the GDN
layers' qkvz, 14 MB, in 56 us; a token's experts' gate/up, 7.3 MB, in about 35); each GDN layer also reads and writes
its 2 MB state (22 us a layer in the kernel). Those kernels run near that; most of what is left is the launches of
about 600 dependent kernels, and the output layer at 8 bits.

**Measured.** In the process, a 256-token prompt each (the package's own source), 48 greedy steps after 4 warm-up
steps, tokens per second all rows write together, the best of two runs, stock and Locals alternated:

| Requests at once | mlx-lm | Locals | Locals / mlx-lm |
|---|---|---|---|
| 1 | 79.5 | 93.6 | 1.18x |
| 2 | 117.5 | 146.8 | 1.25x |
| 4 | 150.2 | 192.2 | 1.28x |
| 8 | 179.8 | 247.9 | 1.38x |

Through the batch engine (requests of 256 tokens, the end ignored, from the first token to the last;
`LOCALS_SOLO=0`, so one-token steps only): 94.6 tokens/s for one request, 180.7 for 4 at once, 223.2 for 8.

**Tried and left out.** Larger command buffers (`MLX_MAX_MB_PER_BUFFER` 2,000, `MLX_MAX_OPS_PER_BUFFER` 1,000): no
faster. The route kernel's register arrays indexed only by constants (an index computed at run time can put an array
in thread memory): no faster here, the kernel's time is its launch. The shared expert's three kernels run beside the
routed experts' and cost about 9 us a layer in the step; folded into the expert kernels as one more slot, its
weights' read (1.6 MB, about 7 us) would take most of that back (it costs 0.15 ms of a one-row step, 1.5 of an
8-row one). At 8 rows the expert kernels read 2.8 GB a step (48 distinct experts a layer for 64 slots) at about 180
GB/s and are bound by their arithmetic, not their loads (the up kernel: 9.8 ms, its loads alone 7.4 at 237 GB/s, its
arithmetic alone 8.9): a 3-bit weight takes an and, an or and a subtraction to become a float, then a multiply-add.
Loading each lane's weights as vectors, no faster (the same bits); unpacking each weight once for all of a run's
slots (a third of an 8-row step's slots share an expert), slower at every width tried (10.5 ms against 9.6: the
registers it holds).

**The output layer's bits** (not changed: a choice of the build, which changes outputs). Requantized from the 8-bit
weights, on 4,096 positions of the package's runtime docs read teacher-forced:

| Output layer | Size | One row | Perplexity | Greedy token as at 8 bits | Mean KL from 8 bits |
|---|---|---|---|---|---|
| 8 bits (the build's) | 540 MB | 2.42 ms | 10.878 | | |
| 6 bits | 413 MB | 1.86 ms | 10.883 | 98.0% | 0.0016 |
| 4 bits | 286 MB | 1.31 ms | 11.057 | 91.6% | 0.017 |

At 6 bits a one-row step would be about 5% faster for 0.04% of perplexity.

### Clover in blocks (`blockdiff.py`, `blockdiff_service.py`, `clover_fast.py`, `moe_tiles.py`)

`locals serve --blocks [N]` (and `locals start --blocks`, `locals chat --blocks`) decodes Clover as DiffuSpace's
Hugging Face code does (`blockdiff.stream_generate`, LowConfidence): a block starts as N mask tokens; each step
previews it and resolves the open positions whose most likely token is above 0.95, or the most confident one; a
resolved block is committed (the causal pass) and the next starts. The model's response penalties apply (presence 0.9,
repetition 1.05; ramps toward the end tokens and `</think>` late in a long response), and greedy is the default, as
its generation_config says. N defaults to 8 (`defaults.MAC_BLOCK`; the model's blocks are 32): a preview costs about
what a prompt chunk of its size does and resolves about as many tokens whatever its size, so on an M4 Pro blocks of 8
write 38.0 tokens/s, of 16 27.3, of 32 20.2 (stock MLX: 14.6). Requests run one at a time (a step is already N rows),
each resolved block streamed, and the state after a reply kept for a next prompt that continues it; JSON schemas and
regexes are refused (a block is written at once, not a token a grammar can steer), and tool calls are parsed from the
reply as on the stock engine. `blockdiff.stream_speculative` is DiffuSpace's own serving mode (a preview drafts a
block, the causal pass checks it: 64.6 tokens/s at 4 drafted tokens, measured before the causal pass's kernels here);
it is not served.

The preview (`BlockModel(..., preview=True)`): the block reads the cached prefix and sees itself both ways; a GDN
layer adds a right convolution (the block's next three positions) to the causal one and, in place of the recurrence,
a softmax over the block's l2-normalized keys plus one bucket for the prefix, whose value is the clean state's
readout decayed by the position's gate; attention has no causal mask; each MoE layer routes within the block coreset,
the fewest experts that carry 90% of the block's summed top-8 routing mass. Its kernels: the routing with the coreset
in one threadgroup (`clover_fast.clover_route`); a block's experts in tiles of 4 slots a simdgroup, each weight chunk
unpacked once for the tile (`moe_tiles.py`: 1.3 to 1.45 ms a layer against SwitchGLU's 2.0); the GDN convolution in
one kernel and its softmax memory one compiled graph. A preview step's layers take 79 ms against stock MLX's 116.

## Switched-off kernels

- **Fused add + RMS norm** (`norm_kernel.add_rms`). It computes `h = x + r` and `rms_norm(h) * w` in one kernel, which halves the norm launches of a pre-norm layer (`norm_kernel.py`), laid out as MLX's own norm for one row so that both outputs are MLX's bits. It needs a hidden size divisible by 128, at most 4,096, and uses a quarter of it in threads per row (`norm_kernel.supported`). Flint's layers use it (`moe.FusedFlintLayer`, `moe.FusedFlintBody`), and Qwen3.5-MoE's (`moe.FusedQwenLayer.step`, through `FusedHybridBody` when `_moe_steps` is set). `FusedHybridBody` would use it for dense Qwen3.5 too, but `USE_ADD_RMS = False`: measured at one row, it was no faster than MLX's two kernels. With the flag off and no MoE steps, the body calls the stock `__call__` (`fuse.py`, `fuse.py`).
- **Ragged batch attention** (`attention.ragged_attention`). Adapted from MLX's `sdpa_vector`, it handles all rows of all requests in one launch. Row r sees the first `nkeys[r]` keys of KV slot `slots[r]`, and K/V are stored slot-major as `(n_slots, n_kv_heads, cap, D)` (`attention.py`). A GQA variant processes one (row, kv head) per threadgroup, so each K/V vector is read once (`attention.py`, `attention.ragged_attention`). `packed.py` calls it only when `USE_RAGGED` is set, no segment is a tree, and every segment has at most 64 tokens. The flag is False because per-request MLX SDPA keeps outputs identical to single-stream decoding (`packed.py`, `packed.packed_forward`).

## Quantized matmul routing (`kernels.py`)

**Why.** MLX's quantized matrix-vector kernel runs one input row per threadgroup, so M rows read the weights M times. From M >= 6 (measured on an M2 Pro, for large matrices), MLX switches to a 32-row tiled kernel instead (`kernels.py`). With BM = 32 and a 2 x 2 simdgroup layout, a pass over 6 to 32 tokens costs as much as 32 rows (`mma_kernel.py`). Both kernels waste work at small M. Small M is the common case in Locals: forced spans and small decode batches.

**Patching.** `enable_small_m(enabled=True, model=None)` sets the process-wide `_enabled` flag (`kernels.enable_small_m`). Given a model, it also changes the class of every module whose type is exactly `nn.QuantizedLinear` or `nn.QuantizedEmbedding` to a cached subclass, `LocalsQuantizedLinear` or `LocalsQuantizedEmbedding` (`_patched_class`, `kernels.py`). MLX's own classes are never modified, so other models in the same process (the server's stock engine, a library user's own model) stay as mlx-lm makes them. Called without a model, it only flips the switch for every model patched so far, which allows an A/B comparison (`kernels.enable_small_m`).

**Call path** of a patched `QuantizedLinear` (`kernels._linear_call_inner`):

```
_linear_switch
  _enabled    -> _linear_call       -> _maybe_packed -> _linear_call_inner
                                                          tuned to MLX's kernel, unpadded -> mlx-lm's original __call__
                                                          _dense_ok  -> _dense_matmul   (rows >= PREFILL_M)
                                                          _eligible  -> _small_matmul   (tuned kernel choice)
                                                          otherwise  -> mlx-lm's original __call__
  not enabled -> _linear_call_plain -> _maybe_packed -> mlx-lm's original __call__
```

The first line is a fast path, for Flint's layers only (`fuse_model` marks them `_locals_direct`; other models route as before): where `_CHOICE` (final choices only; a deferred one is not stored there) holds MLX's own kernel for the shape at these rows and `_PAD` keeps the rows as they are, the call goes straight to MLX. The routing after it cost 4 us of Python a call against 0.6 for MLX's (two reshapes, `_tunable` twice, a dtype's name, three lookups), 0.8 ms of a one-token Flint step, which building and encoding its graph bounds; with the fast path, 1.4 us.

A tied embedding head goes through `QuantizedEmbedding.as_linear` -> `_embed_switch`. It takes only the small-row path, with no dense path and no packing (`kernels._embed_as_linear`).

**Which layers qualify.** `_tunable(mod)` requires a quantized linear or embedding with 4 bits, group size 64, affine mode, biases, and K divisible by 64. Other group sizes, and mxfp4 and similar formats, stay on MLX's kernels: tuning them crashed the load (`kernels._tunable`). The weight layout is MLX's affine 4-bit: `w` is `(N, K/8)` uint32 with the low nibble first, and `scales`/`biases` are `(N, K/64)` (`kernels.py`). `_eligible(x, mod)` also requires `MIN_M` (1) <= M <= `MAX_M` (32), where M is the number of rows after flattening all leading dimensions, and fp16 or bf16 activations (`kernels._eligible`).

**Wide 8-bit layers** (`_tunable_head8`). An 8-bit layer of at least `HEAD8_MIN_N` (32,768) outputs, an output
layer kept at 8 bits (Clover's: 248,320 x 2,048, 540 MB), is tuned from `HEAD8_MIN_M` (5) rows, against MLX's kernel
only the `mma` tilings (MLX's loader in `mma_kernel.py` takes any width: `bits`). MLX reads it once up to 4 rows
(2.2 ms for Clover's) and then switches to its 32-row tiles: 3.8 ms at 6 rows, 4.6 at 8, 7.0 at 12, 6.3 at 16; `mma`
with 8- or 16-row tiles, 2.8, 2.8, 3.7 and 3.7 (the same tokens; outputs within bf16 rounding of MLX's). Its keys carry
the bits (`_mkey`: `248320x2048x8xbfloat16b8`; a 4-bit layer's keys are unchanged), it is never padded, and a
matrix of 64 MB or more is timed in place rather than copied (a copy of an output layer is 0.5 GB). Clover decodes 5.6%
faster at 8 rows (247.6 tokens/s instead of 234.5) and 3% at 6. `LOCALS_HEAD8=0` keeps such layers on MLX's kernel.

**Dense prefill path.** At `PREFILL_M = 128` rows or more, a qualifying layer can be computed as dequantize + dense GEMM in the activations' own dtype instead of MLX's quantized GEMM (`_dense_ok`, `_dense_matmul`). Which one is timed on the GPU once per layer shape at `PREFILL_TUNE_M = 1024` rows and cached with the other choices (`_dense_prefill`, `_tune_prefill`; keys `NxKxP`): dense must be 5% faster, since it also needs the dequantized weights as a temporary. On an M2 Pro with MLX 0.32 it is 6-11% faster for the large shapes and level for the small ones (it was 1.5x with older MLX); MLX's quantized GEMM uses the neural accelerators of M5 GPUs (`quantized_nax.h` in MLX's kernels). The server times untimed shapes while idle and reads prompts densely until then; `locals tune` times them at once. `LOCALS_DENSE_PREFILL=1` always reads densely, `0` never. A bf16 model is not computed in fp16 on this path, because values above 65504 would become inf. Layers with more than `DENSE_MAX = 1 << 27` weights are excluded: a 248k x 2560 lm_head would need a 1.3 GB temporary.

**Row packing.** Inside `with packed_rows(B, L, lengths):` (a thread-local context), a linear layer applied to right-padded `(B, L, D)` activations gathers only the valid rows. It runs the matmul on those rows and scatters the result back into zeros, so padding costs no matmul work (`kernels.py`). Packing applies even when small-row routing is disabled (`_linear_call_plain`). Ragged batched passes use it. `hybrid_batch.py` is `HybridBatchEngine._forward`, the batch decode step, whose rows feed different numbers of tokens (called at `hybrid_batch.HybridBatchEngine.run` and `:1355`). `regions.py` is `RegionScorer.logprobs`, which scores candidate paths in one right-padded pass.

## Kernel families

The autotuner chooses among these for each (N, K, M, dtype). A choice is a tuple whose first element names the family. `_launcher` turns a choice into a closure with the template and grid precomputed, cached by `(choice, M, K, N, dtype, bits)` (`kernels.py`).

| Family | Choice tuple | File | How it works |
|---|---|---|---|
| `mlx` | `("mlx",)` | MLX | `mx.quantized_matmul`, the baseline |
| `multi` | `("multi", VPT, R, NSG)` | `qmv_multi.py` | MLX's `qmv_fast` layout, with M input rows held in registers. Each weight word is loaded and converted once, then applied to all M rows. Nibbles are used in place with MLX's mask trick, x pre-divided by 1, 16, 256, 4096 (`qmv_multi.py`) |
| `v3` | `("v3", VPT, R, NSG)` | `qmv_multi.py` | No int-to-float conversions, which are slow on Apple GPUs. A masked nibble OR-ed into the mantissa of 2^23 gives `2^23 + q * 16^p` exactly, and a subtraction leaves the value, using full-rate instructions only (`qmv_multi.py`) |
| `frag` | `("frag", TN, NSG, SPLIT)` | `qmv_frag.py` | Computes `Y^T = W X^T` with simdgroup 8x8 matrices, dequantizing into the A-fragments in registers. No threadgroup memory and no barriers. The order of k is permuted so that each lane finds its 32 weights for a 128-wide chunk in one `uint4`, and every weight is read once (`qmv_frag.py`) |
| `fragh` | `("fragh", TN, NSG, SPLIT)` | `qmv_frag.py` | `frag` with half-precision fragments. Two nibbles become a `half2` through the exponent of 1024 (`v \| 0x64006400`), and one `half2` fma applies scale and bias. Activations are clamped to +/-65504 (`qmv_frag.py`, `qmv_frag.py`) |
| `fragq` | `("fragq", TN, NSG)` | `qmv_frag.py` (`launch_q`) | 8x8 matrices whose A side is the weights as 1 + q/16 in the activations' 16-bit type: a nibble in the top of the mantissa, exact, two of them (16 bits apart in one word) by one shift, one mask and one or. The B side is the activations as they are. A group's MMAs accumulate c = xsum + sum q x / 16 in float, and y += 16 s c + (b - 16 s) xsum, MLX's own qmv factoring (each group's input sums are summed once a threadgroup). A lane's two weights of MMA j are nibbles j % 4 and j % 4 + 4 of one word because column c of A stands for k = 8 (c / 2 + 4 (j / 4)) + j % 4 + 4 (c % 2). `launch` spends half again an MMA's arithmetic on dequantizing its two weights; this a quarter (`qmv_frag.py`) |
| `small` | `("small",)` | `kernels.py` (`qmm_small`) | Scalar FMA kernel that reads and dequantizes each weight once for all M rows. Measured on an M2 Pro, it beats MLX for 3 <= M <= 6 on large matrices and loses at M = 8 (`kernels.py`). Launch shape from `small_config(N)` (`kernels.small_config`) |
| `mma` | `("mma", BM, BN, BK, WM, WN, split)` | `mma_kernel.py` | MLX's `QuantizedBlockLoader` and `steel::BlockMMA` with BM = 8 (or 16) instead of 32. Every weight tile is dequantized once and applied to up to BM rows. MLX's headers are inlined from the installed wheel because custom kernels get no include path (`mma_kernel.py`) |
| `nax` | `("nax", "r", SN, WN, WK, TK, SPLIT)` or `("nax", BM, BN, BK, WM, WN, SPLIT)` | `nax_kernel.py` | The M5's neural accelerators (`mpp::tensor_ops::matmul2d`, 16 x 32 x 16 products). The register variant (`locals_qmm_naxr`, the one tuning picks): each simdgroup owns 32 or 64 output columns, reads its weights straight into registers (K renumbered so a thread loads 4·TK consecutive 4-bit weights of a row at once) and dequantizes them into the accelerator's operand, whose accumulators stay in their cooperative tensors for the whole K loop; WK simdgroups split K and add up once. The tile variant is MLX's `qmm_t_nax_tgp_impl` with 16-row tiles. Measured on an M5 Pro (macOS 27, MLX 0.32.3): one decode step's 4-bit matmuls 1.6-3.4x MLX's at 6 to 32 rows, timed per shape (whole decode steps not measured yet). Only where `nax_kernel.available()`: it compiles and checks a small case once; elsewhere (no Metal 4 tensor operations, as on macOS 15), or with `LOCALS_NAX=0`, it is no candidate, and a cached `nax` choice is read as `mlx` (`kernels._choose`) |

Notes on the families:

- In `frag`, `fragh` and `fragq`, M is a runtime input (`mcount`), and only `MT = (M + 7) // 8` is compiled in. One compiled kernel therefore serves eight row counts (`qmv_frag.py`, `qmv_frag.launch`).
- `fragq` takes up to 16 rows of float16 or bfloat16 activations. Its outputs are not `frag`'s bits: it sums `s * sum q x + b * sum x` as MLX's qmv does, as close to the exact sum as MLX's kernel is (`test_fragq_is_the_exact_sum_as_closely_as_mlxs_kernel`).
- `SPLIT > 1` (split-K, used by `frag`, `fragh` and `mma`) divides K across threadgroups, writes float32 partials and sums them afterwards. This helps small matrices fill the GPU (`mma_kernel.qmm_mma`, `qmv_frag.launch.f`).
- Because `fragh` clamps activations to the fp16 range, it is a candidate only for fp16 activations. A cached `fragh` choice read back for bf16 activations is replaced by `mlx` (`kernels._choose`, `kernels._tune_shape`).

## Autotuning

### Per-shape tuning (`_choose`, `_tune_shape`, `_time_shape`)

`_choose(x2, mod)` looks up the choice in this order (`kernels._choose`):

1. The in-memory map `_CHOICE[(N, K, M, dtype)]`.
2. The tuning dictionary, under the key `"{N}x{K}x{M}"`, with `"x{dtype}"` appended when the dtype is not `float16`.
3. If the background tuner is deferring work (`_DEFER` set, `_TUNING` not set), it returns `("mlx",)` without remembering it, so a later lookup can find the tuned result.
4. Otherwise it tunes the shape now, under `_TUNE_LOCK`.

`_time_shape` builds the candidate list from these rules (`kernels.py`; `_tune_shape` only names the shape in its crash marker and calls it):

| Family | Row counts M | Other conditions | Candidates |
|---|---|---|---|
| `mlx` | all | none | `("mlx",)` |
| `multi` | M <= 4 | `K % (VPT*32) == 0`, `N % R == 0` | `_MULTI_CANDIDATES` = (8,4,2), (16,4,2), (8,2,2) |
| `v3` | 3 to 6 | `K % (VPT*32) == 0`, `N % R == 0` | `_V3_CANDIDATES` = (8,8,2), (8,4,2), (16,4,2), (16,4,4), (16,2,2) |
| `frag` | 1 to 32 | `K % (128*SPLIT) == 0`, `N % (8*TN) == 0`; SPLIT > 1 only when K >= 4096 | `_FRAG_CANDIDATES` = (2,2,1), (4,2,1), (2,4,1), (1,2,2), (2,8,1), (1,4,1) |
| `fragh` | 1 to 32 | as `frag`, fp16 activations only | same as `frag` |
| `fragq` | 1 to 16 | `K % 64 == 0`, `N % (8*TN) == 0`, fp16 or bf16 activations (`supported_q`) | `_FRAGQ_CANDIDATES` = (1,2), (1,4), (2,2), (2,4), (2,8), (4,4) |
| `small` | 3 to 6 (`MIN_M_SMALL`, `MAX_M_SMALL`) | N >= 4096 or K >= 4096 | `("small",)` |
| `mma` | 2 to 32 | `N % BN`, `K % (BK*split)`, `(K/split) % 64` and `BN % (8*WN)` all zero | `_MMA_CANDIDATES[8]` for M <= 8, `[16]` above |
| `nax` | 2 to 32, where `nax_kernel.available()` | `nax_kernel.supported(M, K, N, cfg)`: the tile variant needs a 32-wide side per simdgroup (`tile_matmad_nax` computes nothing for one 16 x 16 fragment by one); the register variant N % (SN·WN), K % (16·TK·SPLIT), at most 32 KB of threadgroup memory and 1,024 threads | `nax_kernel.CANDIDATES`: 8 tile and 13 register configurations |

At M <= 2, `v3` is not a candidate: it wins in isolation there but loses about 1% inside the model (`kernels._tune_shape`).

The tuning procedure (`kernels._tune_shape`):

1. **Correctness.** Every candidate runs on a random input (fixed key `M * 131 + K`, not the global random state). Its output must be within `0.05 * (1 + max|ref|)` of MLX's. A NaN or an exception removes the candidate.
2. **Timing.** Candidates are timed on `max(1, min(16, (64 << 20) // nbytes))` copies of the weights. The aim (code comment) is that the weights come from DRAM as inside a model, not from the GPU's system cache. In fact the copies total at most about 64 MB, unless a single copy is already larger, and well below that for layers under 4 MB, where the 16-copy cap applies. The calls form a chain: each call's input waits on the last call's output (an add of a zero slice of it), as a model's layers wait on each other, `max(16, 2 * ncopy)` calls, the graph built before the clock starts (`_chain_us`); the chain's own adds, timed alone, are taken off. Calls timed back to back without waiting overlapped one's start with another's end, which rewarded few large threadgroups: at 8 rows of Flint's q/k/v projection, `fragq` with 4 tiles a simdgroup was then the fastest and 14% slower than with 1 inside the model, and choices made that way made Flint's 8-row step 2% slower where those made from chains made it 5% faster. Four interleaved rounds run, and each candidate keeps its best round.
3. **Decision.** A non-MLX kernel must be at least 10% faster than MLX over the whole chain, its adds included (`< 0.9 *` MLX's time): a small shape's own time is not much more than noise. Otherwise the choice is `("mlx",)`.
4. **Storage.** The choice is stored under the shape key. The best candidate of each family is also stored under `"{key}|{family}"` for the whole-step tuner. The cache file is rewritten.

### Whole-step tuning (`model_tune`, full mode only)

Per-shape timings in isolation miss interactions inside a model, such as cache reuse across layers and overlap of consecutive kernels. For example, at M = 4 `v3` loses to `fragh` on every shape alone, yet it makes the Qwen3.5-2B decode step about 4% faster (`kernels.model_tune`).

For each M from 2 to 32, `model_tune` gathers several options: the per-shape mix, plus each family whose per-shape winner exists for every shape of the model. It times a real batch-M decode step with each option, after a 64-token prefill per row, for 16 steps per round over 5 rounds, and takes the median. A family overrides the per-shape mix only if it is at least 2% faster (`< 0.98 *`) (`kernels.py`).

Timing is pipelined, as the engines decode: step s+1 is built and queued while step s runs. Synchronous timing ranks families differently. On the 0.8B model at M = 8 in fp16, synchronous timing picked `fragh`, which was 17% slower pipelined than `frag` (`kernels.model_tune`). Finished row counts are recorded under `"__model_tuned_pipe__" + md5(shapes)[:10]` so they are not repeated (`kernels.model_tune`, `kernels.model_tune`).

### Tuning modes (`tune_model`)

| Mode | What runs | Where it is the default |
|---|---|---|
| `auto` | If the cache or preset covers every shape for 1 to 16 rows (`tune_pending(max_m=16) == 0`), only `warmup`. Otherwise the `quick` procedure | `Runtime`, `locals.lm` |
| `background` | Serve at once. Untuned shapes run on MLX's kernel, and the server tunes them while idle. Shapes tuned already (for example from a preset) are compiled at load | The server engine (`service.BatchService._run_locals`) |
| `quick` | `pretune(max_m=16)`: every shape for 1 to 16 rows, then `gdn_model_tune`, then `warmup` | `locals tune --mode quick` |
| `full` | `pretune()` up to 32 rows, then `gdn_model_tune`, then `model_tune`, then `warmup` | `locals tune` (`app.cmd_tune`) |
| `off` | Nothing ahead of time. Untuned shapes are tuned on first use | none |

Sources: `kernels.tune_model`. Any other value raises `ValueError`. The log line at `kernels.tune_model` gives quick tuning on an M2 Pro as 2 minutes for Qwen3.5-4B and 7 for Gemma 4 E4B. The full procedure takes 10-20 minutes and gives the best batched throughput (`runtime.Runtime`).

Two consequences for row counts above 16:

- In `auto` mode, a shape with 17 to 32 rows that is not cached gets tuned the first time it occurs, because deferral is off (`kernels.tune_model`, `kernels._choose`).
- In `background` mode those row counts stay on MLX's kernels (`kernels.tune_model`). `pending_work` defaults to `max_m=16`.

`warmup(model, ms)` compiles and runs the chosen kernel of every shape for M in `ms` at load. `tune_model` calls it with 1 to 16 after every mode except `off`, which returns before any warmup. The function's own default is `ms=range(1, 9)`., because each new kernel instance costs about 50 ms of Metal compilation. A kernel that fails to compile or run falls back to `("mlx",)`, which covers a cached choice that the installed mlx no longer compiles (`kernels.warmup`).

### Background tuning while idle

When `tune_model` returns with `_DEFER` set, the service builds `kernels.pending_work(rt.model)`: the untuned (layer, rows) pairs, smallest batches first (`service.BatchService._run_locals`, `kernels.pending_work`). When the request queue is empty (and the other engine, if loaded, has nothing in flight or queued, and no other process marks the GPU busy), `_take` calls `_tune_while_idle`, which tunes one pair with `tune_one` in the engine's own thread, so the GPU is not shared. A request that arrives meanwhile waits for at most that one shape (`service.BatchService`). Another Locals server on the same Mac counts too: each process with a request in flight or queued, or a shape being timed, touches its own file in the user's temporary folder (`locals-gpu/busy-<pid>`) once a second (a thread started by `service._start_heartbeat`, `kernels.mark_busy`), and an idle tuner leaves its shapes while another live process marked itself in the last 3 seconds (`kernels.others_busy`), looking again a second later. A lock in that folder (`tuning.lock`) lets one process time a shape at a time. (A tuner saw only its own engines, and timed kernels next to another server's work, keeping the skewed timings.) `tune_one` sets `_TUNING` so that `_choose` really tunes instead of deferring (`kernels.py`). From then on the shape runs on its tuned kernel. When the list is empty, the server logs that tuning is finished. While a shape is timed (`_tune_shape`), a marker file next to the cache names it (`<cache>.tuning-<pid>`). A Metal error MLX raises in a command buffer's completion handler aborts the process, which no `except` catches, so the shape would be tuned again, and abort the server again, at every start. `_load_tune` reads markers left by dead processes (`_crashed_shapes`): each counts in `__crashed__`, and a shape that stopped a process twice keeps MLX's kernel. One death (a kill, a power cut) only tunes the shape again. SIGTERM (`locals stop`, a model switch, a restart) is not a death of this kind: the server's handler (`server._on_sigterm`) removes its marker and stops new ones (`kernels.stop_timing`), then lets the signal end the process at once, as before. (Two stops during idle tuning used to pin a shape to MLX's kernel, with a message about crashes.) `_load_tune` also removes `.tune-*.json` temporary files older than 10 minutes, which a save cut short by a kill leaves (`_drop_old_temps`).

### Cache file and presets

- **File name:** `tune_v6_<architecture>.json`, where the architecture is `mx.device_info()["architecture"]`. Results are therefore per GPU architecture (`kernels.py`).
- **Location:** `LOCALS_TUNE_CACHE` if set. It names the file itself, not a directory. Otherwise `~/.cache/locals/<file name>` (`kernels.py`). The path (environment variable and GPU architecture) is fixed when `kernels.py` is imported, so set `LOCALS_TUNE_CACHE` before importing locals. The file itself is read lazily, once per process, by `_load_tune`.
- **Load order** (`_load_tune`, `kernels.py`):
  1. The shipped preset for this architecture is loaded first.
  2. The user's cache is loaded on top and overrides it; a dict-valued key such as `__times__` is merged entry by entry.
  3. A user cache that does not parse is renamed to `.bad`, not silently replaced.
  4. If there is no readable user cache (none, or one just renamed to `.bad`) and the preset is non-empty, a user cache is written. Every write leaves out what the preset says (`_beyond_preset`), so the user's file holds only what this Mac measured or changed (a crash pin, for example). A copy of the preset in it overrode later presets: a release that corrected an entry never reached anyone who had run Locals once. (Files written before this keep their copies.)
- **Writes** (`_save_tune`) merge with the file as it is on disk, under an `fcntl` lock on `<cache>.lock` (another process's results are kept), then replace the whole file atomically, via a temporary file flushed to disk and `os.replace`. Writes and tuning both take `_TUNE_LOCK`, an `RLock`, because the server runs two engines (`kernels.py`, `kernels.py`).
- **Format:** a JSON object. Shape keys (`"2560x10240x4"`, `"…x4xbfloat16"`) map to choice lists such as `["frag", 2, 2, 1]`. `"…xP"` keys hold the prompt-reading choice (`["dense"]` or `["mlx"]`). `"__times__"` holds what tuning measured, per shape key and family in microseconds, which `locals tune --report` (`tune_report`) sums per batch size. `"…|family"` keys hold per-family winners. `"__model_tuned_pipe__…"` keys hold lists of finished row counts. `"__mlx__"` is the MLX version the file's timings were taken with. When a load finds another one, the shapes this Mac timed (matmul shapes and how prompts are read) are listed in `"__stale__"` (`_mark_stale`, with a log line): they keep their choice, and the idle tuner (`pending_work`, `tune_one`) and `locals tune` (`pretune`) time them again, each leaving the list once timed (`_unstale`; a merge keeps only what both files still list). Choices timed under an older MLX used to stay for good. A file from before this gets the version and nothing stale. A shape whose timings have no `fragq` (timed before the family existed, where it applies) counts as stale the same way without being listed (`_missing_family`): it keeps its choice until a tuner times it again, and then it has. Keys that start with `__` are kept as they are, and all others are read as tuples (`kernels._read_tune`).
- **Presets:** `locals/presets/tune_v6_applegpu_g14s.json` (M2 Pro and M2 Max) and `tune_v6_applegpu_g17s.json` (M5 Pro: so far only how prompts are read, measured on one for Qwen3.5-4B and 9B) are shipped. It is packaged through `[tool.setuptools.package-data]` (`pyproject.toml:58-59`). Setting `LOCALS_NO_PRESET=1` hides presets, so a start behaves as on a GPU without one (`kernels._preset_path`).

To re-tune from scratch, stop every Locals server, delete the cache file (or point `LOCALS_TUNE_CACHE` at a new path) and set `LOCALS_NO_PRESET=1`. A server left running writes its copy of the cache back at its next save.

## Output-layer helpers (`kernels.py`)

| Function | Purpose |
|---|---|
| `text_model(model)` | The decoder inside a multimodal wrapper (`language_model`), or the model itself (`kernels.text_model`) |
| `tied_embeddings(lm)`, `head_module(lm)` | Whether the vocabulary projection is the embedding table: the flag may be on the instance (Gemma 3), in `args` (Llama, Qwen), or `lm_head` may be missing. Returns the module that projects (`kernels.tied_embeddings`) |
| `post_logits(lm, out)` | Reproduces what the model's forward does after the projection: soft-capping (Gemma 2, 3n, 4), Cohere's `logit_scale`, Granite's `logits_scaling`. These are monotonic, so greedy and constrained choices never depend on them (`kernels.post_logits`) |
| `head_logits`, `forward_last`, `forward_hidden` | `head_logits` projects the hidden states it is given and applies `post_logits`. `forward_last` runs the body and projects only the last position, falling back to the model's own forward when it has no `model` or `args`. `forward_hidden` returns the last position's final normed hidden state (1, 1, D) and does not project (`kernels.py`, `kernels.py`, `kernels.py`) |
| `head_check(model, log)` | Compares the model's own forward against body + `head_logits` on a 6-token probe. A difference above `0.02 * max|ref| + 0.05` is logged; the function returns the max difference, or NaN on an exception. Nothing in `locals/`, `tests/` or `bench/` calls it (`kernels.head_check`) |
| `head_rows(model, h, ids)` | Exact logits for only the vocabulary rows `ids`. A constrained step often allows a handful of tokens, and gathering those rows of the 4-bit head skips most of its weight traffic. It uses the layer's own quantization mode, so an mxfp4 head works too (`kernels.head_rows`); the engine calls it for sparse steps (`engine.Engine._generate_pipelined.head`) |

## Model names and checks before download (`defaults.py`)

`MODELS` holds the short names Locals is tested with (`defaults.py`). Any mlx-lm chat model id or local path also works.

| Short name | Repository |
|---|---|
| `qwen3.5-4b` | `mlx-community/Qwen3.5-4B-MLX-4bit` |
| `qwen3.5-2b` | `mlx-community/Qwen3.5-2B-MLX-4bit` |
| `gemma-4-e4b` | `mlx-community/gemma-4-e4b-it-4bit` |
| `lfm2.5-2.6b` | `LiquidAI/LFM2.5-2.6B-MLX-4bit` (1.6 GB: 8 GB Macs) |
| `minicpm5-2b` | `mlx-community/MiniCPM5-2B-mlx-4Bit` (1.4 GB) |

`ALIASES` maps more names onto these: `qwen` and `qwen3.5` to the 4B; `gemma`, `gemma4`, `gemma-4`, `gemma4-e4b` and `gemma-4-e4b-it` to Gemma 4 E4B; `lfm` and `lfm2.5`; `minicpm` and `minicpm5` (`defaults.py`).

- **`default_model()`** reads the Mac's memory with `sysctl -n hw.memsize`. It picks `qwen3.5-2b` when memory is known and below 12 GB, and `qwen3.5-4b` otherwise, including when memory is unknown (`defaults.memory_gb`, `defaults.default_model`). The 4B (3 GB of weights, 64k context) and its prompt cache do not fit next to macOS in 8 GB. `DEFAULT_MODEL` is computed once, at import (`defaults.py`).
- **`model_id(name)`** resolves a `--model` value; every command uses it as the argparse `type` (for example `server.main`). Resolution order (`defaults.model_id`):
  1. A short name or alias, matched case-insensitively, becomes its repository; so does an Ollama name of one (`qwen3.5:4b`, `gemma4:e4b`: `:` becomes `-`, and `:latest` is dropped). A short name wins over a folder of the same name in the current directory.
  2. An existing path becomes its absolute path, because the background server runs elsewhere and later commands may run from other folders. Symlinks are kept.
  3. Anything else is returned unchanged.
- **`short_name(model)`** gives the model's name in the API (`locals/<short name>`). It takes the last path component, strips a trailing `-4bit` or `-mlx-4bit` in any capitalization, and lowercases the rest: `MiniCPM5-2B-mlx-4Bit` becomes `minicpm5-2b` (`defaults.short_name`). It is used by the server (`server.Server.__init__`).
- **`too_big(model, remote=False)`** returns a reason the model cannot run, or `None` (`defaults.too_big`).
  - It fails when weights + `HEADROOM_GB` (2.0) exceed `gpu_limit_gb()`, which is MLX's `max_recommended_working_set_size`.
  - The headroom covers the KV cache of a few agents' prompts, the prefill, and the server's margins. Without it, macOS swaps, or Metal stops the process; on a 16 GB Mac that can be a GPU driver panic that takes the whole machine down (`defaults.py`).
  - `weights_gb` sums the top-level `*.safetensors` files of a local folder or of the Hugging Face cache, without network access. With `remote=True` it falls back to the file sizes Hugging Face lists (`defaults.py`).
  - An unknown size or limit passes the check. `LOCALS_IGNORE_MEMORY=1` skips it.
- **`gguf_hint(model, remote=True)`** detects a GGUF model: a `.gguf` file, or a folder or Hugging Face repo with `.gguf` weights and no safetensors. Locals runs MLX models only, and mlx-lm loads no GGUF (`defaults.gguf_hint`).
  - A repo already in the cache with safetensors is accepted without contacting Hugging Face, so startup works offline.
  - For a hint, it strips `gguf` and quantization suffixes (`Q4_K_M`, `IQ…`, `F16`, `BF16`, `F32`) from the name. It searches Hugging Face for MLX builds (`filter="mlx"`, sorted by downloads, at most 3) under that name, and under the part after the first `_` (bartowski's `Qwen_Qwen3.5-0.8B` naming).
  - If nothing is found, it suggests mlx-community or `mlx_lm.convert --hf-path ORIGINAL -q`.
- **Where the checks run:** `locals serve` (in `server.main`) calls `gguf_hint(...) or too_big(..., remote=True)` before downloading anything; the app's `check_model` (through `fetch_model`) calls `gguf_hint(...) or ollama_hint(...) or too_big(..., remote=True)` and also refuses a name that is neither a folder nor a Hugging Face repository, a gated repository and a typo; and exit with the message if either returns one (`server.main`, `app.fetch_model`).
- **`build_id()`** fingerprints the installed code: a SHA-1 of the names, sizes and modification times of `locals/*.py`, except the client-only `plan.py` and `solve.py`, truncated to 12 hex digits (`defaults.build_id`). The app compares it with a running server's to detect a server started before an upgrade (`app._ensure_server`).

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `LOCALS_TUNE` | `auto` (library), `background` (server) | Tuning mode when none is passed (`runtime.Runtime.__init__`, `lm.patch`, `service.BatchService._run_locals`) |
| `LOCALS_TUNE_CACHE` | `~/.cache/locals/tune_v6_<arch>.json` | Path of the tuning cache file (`kernels.py`) |
| `LOCALS_NO_PRESET` | unset | Any non-empty value hides the shipped presets (`kernels._preset_path`) |
| `LOCALS_MAX_COMPILED` | `16` | Most compiled constraints kept (`runtime.Runtime.__init__`) |
| `LOCALS_COMPILED_GB` | `1.5` | GB of constraint tables kept (`runtime.Runtime._cput`) |
| `LOCALS_IGNORE_MEMORY` | unset | Skip the `too_big` memory check (`defaults.py`) |
| `LOCALS_DTYPE` | `auto` | The compute dtype `Runtime` uses when `dtype="auto"`: `float16` or `bfloat16` forces one (to measure a new GPU; `auto` picks float16 only on GPUs without native bfloat16) |
| `LOCALS_FUSE` | `1` | `0`: no fused projections |
| `LOCALS_KERNELS` | `1` | `0`: MLX's matmul kernels for few rows (no tuned kernels) |
| `LOCALS_GDN_KERNEL` | `1` | `0`: mlx-lm's Gated DeltaNet decode instead of `gdn_kernel.py` (`fuse.USE_GDN_KERNEL`) |
| `LOCALS_DENSE_PREFILL` | timed | `1`: prompts always through dequantize + dense GEMM; `0`: through MLX's quantized GEMM |
| `LOCALS_PAD_ROWS` | `1` | `0`: never compute a batch at more rows. By default batches of 8 to 16 are computed at a larger size, up to twice the rows, where the tuned timings show it at least 7% faster (`_pad_rows`; on an M5 Pro 12 requests computed as 16 ran 10-20% faster). `all`: other sizes too (the timings also pad 2 rows to 3, 4 to 5 and 24 to 32 on some M5 layers; measured end to end, within noise) |
| `LOCALS_EXPERT_KERNELS` | `1` | `0`: Flint's and Qwen3.5-MoE's routed experts through `FusedSwitchGLU` (gather_qmm) at every row count (`moe.USE_EXPERT_KERNELS`) |
| `LOCALS_HEAD8` | `1` | `0`: wide 8-bit layers (an output layer at 8 bits) on MLX's kernel at every row count (`kernels.HEAD8`) |
| `LOCALS_MOE_STEPS` | `1` | `0`: Qwen3.5-MoE's layers without the residual adds in the norms and the next norm in the expert combine (`fuse.USE_MOE_STEPS`) |
| `LOCALS_QK_KERNEL` | `1` | `0`: Flint's q/k norms and RoPE as four MLX kernels (`qk_kernel.USE_QK_KERNEL`) |
| `LOCALS_KV_BITS` | `16` | `8`: Flint's full-attention keys and values in 8 bits (`qkv.py`): about twice the context, 4 to 13% slower writing |
| `LOCALS_GDN_FAST` | timed | `1`: unpadded prompts through MLX's `mx.fast.gated_delta_update` in Qwen3.5's recurrent layers; `0`: through mlx-lm's kernel (`kernels.gdn_fast`) |

## Code switches

| Constant | Value | Where |
|---|---|---|
| `USE_GDN_KERNEL` | `True` unless `LOCALS_GDN_KERNEL=0` | `fuse.py` |
| `_GDN_MAX_S` | `4`; the 16-token limit with `want_states` is a literal in the condition at `fuse.FusedGatedDeltaNet._mix`. `gdn_kernel.MAX_S = 16` (`gdn_kernel.py`) is defined but never referenced | `fuse.py`, `fuse.FusedGatedDeltaNet._mix` |
| `USE_ADD_RMS` | `False` | `fuse.py` |
| `USE_RAGGED` | `False` | `packed.py` |
| `MIN_M`, `MAX_M` | `1`, `32` | `kernels.py` |
| `MIN_M_SMALL`, `MAX_M_SMALL` | `3`, `6` | `kernels.py` |
| `PREFILL_M` | `128` | `kernels.py` |
| `DENSE_MAX` | `1 << 27` | `kernels.py` |

## Things to keep in mind when changing this code

- Never draw from MLX's global random state during load or tuning. Seeded sampling must match mlx-lm's. That is why `_concat_quantized` avoids the constructor, and why the probes use fixed keys (`fuse.py`, `fuse._swiglu_mlp`, `kernels._tune_shape`).
- `enable_small_m` matches exact types only. A subclass of `QuantizedLinear` may compute differently and is left alone (`kernels.py`).
- Tuning keys carry the activation dtype. Different GPU architectures already use separate cache files (`kernels.py`). The dtype in the key separates a bf16 model from an fp16 model on the same GPU, for example Gemma 3 (kept in bf16) next to Qwen3.5 (fp16) on an M2. `fragh` must never run on bf16 activations. (`kernels._choose`, `kernels._choose`).
- A new kernel family needs four things: a branch in `_launcher`, candidate rules in `_time_shape`, an entry in `model_tune`'s `families` if it should compete there, and correctness under the tolerance check. A new choice can also break a cached file on another mlx version; `warmup` falls back to MLX in that case.
