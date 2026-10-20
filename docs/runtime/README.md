# Model runtime and GPU kernels

Part of the [technical documentation](../README.md) of Locals. File and function names refer to the `locals` package.

This part of the package turns a stock mlx-lm model into the model Locals decodes with. It changes how the model computes, not what it computes (`locals/lm.py:18`). Loading runs four steps, in the same order in `Runtime.__init__` (`runtime.py:137-145`) and in `locals.lm.patch` (`lm.py:59-65`):

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
| `fuse.py` | Projection fusion, `FusedAttention`, `FusedMLP`, `FusedGatedDeltaNet`, `FusedGatedAttention`, `FusedHybridBody` |
| `kernels.py` | Matmul routing (`enable_small_m`, `_linear_switch`), the dense prefill path, row packing, `qmm_small`, the autotuner and its cache, and output-layer helpers |
| `qmv_multi.py` | The `multi` and `v3` multi-row matrix-vector kernels |
| `qmv_frag.py` | The `frag` and `fragh` kernels, which dequantize straight into 8x8 MMA fragments |
| `mma_kernel.py` | `qmm_mma`: MLX's tiled quantized GEMM rebuilt with 8- or 16-row tiles |
| `gdn_kernel.py` | One Metal kernel for a Gated DeltaNet decode step |
| `norm_kernel.py` | Fused residual add + RMS norm (present, switched off) |
| `attention.py` | Ragged batch decode attention (present, switched off) |
| `defaults.py` | Model short names, the default model for this Mac, and checks run before a download |
| `presets/` | Shipped tuning results for each GPU architecture |

## Loading: `Runtime` and `locals.lm`

`Runtime.load(path, **kw)` resolves short names with `defaults.model_id` and then calls `mlx_lm.load` (`runtime.py:168-173`). The constructor takes these arguments (`runtime.py:133-134`):

| Argument | Default | Meaning |
|---|---|---|
| `fuse` | `True` | Run `fuse_model` |
| `small_m_kernel` | `True` | Passed to `enable_small_m(small_m_kernel, model)`, which always patches the layer classes. True turns on routing (tuned small-row kernels and the dense prefill path) and runs `tune_model`. False keeps MLX's kernels with row packing only, and sets the process-wide switch off for every model patched so far |
| `prompt_cache_bytes` | `2 << 30` | Byte limit of the prompt (KV) cache |
| `dtype` | `"auto"` | Passed to `set_compute_dtype` |
| `tune` | `None` | Tuning mode. Falls back to `LOCALS_TUNE`, then to `"auto"` (`runtime.py:144`) |

The server's engine loads with `tune=os.environ.get("LOCALS_TUNE", "background")` instead (`service.py:280`).

After patching, `Runtime` rejects any model without `model.embed_tokens`. It runs the decoder families that have one: Llama, Qwen, Gemma, Mistral, Phi-3 and similar (`runtime.py:146-150`). It then builds the token vocabulary, the byte trie and an `Engine`, plus two caches:

- **Compiled constraints**, least recently used first. The cache holds at most `LOCALS_MAX_COMPILED` entries (default 16) and at most `LOCALS_COMPILED_GB` GB of estimated tables (default 1.5) (`runtime.py:161-162`, `runtime.py:238-248`). The limit exists because one token-table constraint takes 60-100 MB on the GPU (a 40-state regex on Qwen3.5's 248k vocabulary). Kept forever, a few dozen of them filled 4-5 GB, and the batch's memory check then admitted one request at a time (`runtime.py:156-160`). Each constraint's device tables are built at compile time, not inside the first request (`runtime.py:43-58`).
- **Prompt cache**: mlx-lm's `LRUPromptCache(max_size=16, max_bytes=prompt_cache_bytes)`. Locals's KV caches never change arrays in place, so a cache hit shares arrays instead of copying them (`runtime.py:163-166`).

`Runtime.sample()` shares the prompt through tree attention, so it needs attention in every layer. It raises an error on models with recurrent layers, such as Qwen3.5 (`runtime.py:290-294`).

`locals.lm` is mlx-lm with the model patched. `load(path_or_hf_repo, *args, dtype="auto", tune=None, **kwargs)` calls `mlx_lm.load` and then `patch` (`lm.py:69-75`). `patch(model, dtype="auto", fuse=True, kernels=True, tune=None, log=None)` runs the four load steps in place (`lm.py:41-66`). `patch` is also exported as `locals.patch` (`__init__.py:57-60`).

## Compute dtype: fp16 or bf16

Apple GPUs before the M3 generation emulate bf16 arithmetic. On those GPUs (architecture `applegpu_g13*` = M1, `applegpu_g14*` = M2), MLX's fp16 kernels run up to 1.7x faster than its bf16 kernels. `native_bf16()` returns False on them (`runtime.py:61-65`). The code comment also notes that on Qwen3.5 checkpoints, fp16 is closer to a float32 run than bf16 is (`runtime.py:87-88`).

`set_compute_dtype(model, dtype="auto")` casts only the bf16 arrays: activation-dtype weights, and quantization scales and biases. float32 arrays such as `A_log` are kept as they are (`runtime.py:85-86`, `runtime.py:107`). It returns the dtype in use.

| `dtype` | Behaviour |
|---|---|
| `"auto"` | Cast to fp16 only when all of these hold: the model has bf16 arrays, the GPU has no native bf16, `_fp16_special(model)` is false, and the safety checks below pass. Otherwise stay in bf16 (`runtime.py:94-96`) |
| `"float16"` | Cast with no checks |
| `None` or `"bfloat16"` | No cast (`runtime.py:97-98`) |

Safety checks under `"auto"`:

1. **Range.** If any bf16 array holds a value with magnitude above `6.0e4`, the model stays in bf16. 65504 is fp16's largest finite value, so larger values would become inf (`runtime.py:101-103`; the code comment there claims the cast is exactly reversible below that limit). It is not exact for very small values: fp16 has fewer exponent bits than bf16. That is why a failed probe restores the saved bf16 arrays instead of casting back (`runtime.py:106`, `runtime.py:115`).
2. **Probe.** A fixed 12-token probe runs before and after the cast. fp16 is kept only if the new last-position logits are finite, have the same argmax, and differ by at most `0.05 * max|ref| + 0.5` (`runtime.py:104-111`). On failure, the saved bf16 arrays themselves are put back, because casting fp16 back to bf16 loses small values (`runtime.py:112-116`).

`_fp16_special` keeps a model in bf16 when the source of its text model's module contains the substring `float16`. The heuristic is meant to catch models that treat fp16 specially. The test also matches `bfloat16`, so any module that mentions bf16 stays in bf16 as well. In the installed mlx-lm 0.32.0 that includes deepseek_v3, deepseek_v32, minimax, mimo_v2_flash and mistral4. Gemma 3 clips every residual add through fp32 in fp16. In bf16 it runs its plain path, which is also faster: 68 against 53 tok/s for Gemma 3 4B on an M2 (`runtime.py:68-81`).

The chosen dtype matters downstream. Tuning results are keyed by activation dtype, the `fragh` kernel is fp16-only, and the dense prefill path dequantizes into the activation dtype (see below).

## Projection fusion (`fuse.py`)

q/k/v become one quantized matmul, and gate/up become another. Quantization groups run along the input dimension, so concatenating weights along the output dimension keeps the quantized format exact. Fewer, larger matmuls mean fewer kernel launches per step (less CPU encode time) and better GPU occupancy for the small k/v projections (`fuse.py:3-7`).

`_fusable(mods)` requires every module to be a `QuantizedLinear` with the same `(bits, group_size, mode)`, and a bias on all of them or none (`fuse.py:21-24`, `fuse.py:262-263`). For example, mlx-lm's mixed quantization recipes give `v_proj` 6 bits next to 4-bit q/k, and such layers are left unfused. `_concat_quantized` builds the fused layer without calling `QuantizedLinear`'s constructor. The constructor quantizes random weights, which draws from the global random state, and seeded sampling would then differ from mlx-lm's (`fuse.py:27-44`).

What `fuse_model(model)` changes in each layer (`fuse.py:269-320`):

| Target | Condition | Result | Models |
|---|---|---|---|
| Attention q/k/v | Class `Attention` from module `qwen2` or `llama`; q/k/v quantized and fusable | `qkv_proj`; class becomes a subclass of `FusedAttention` | Llama, Qwen2-style |
| `GatedDeltaNet` `in_proj_{qkv,z,b,a}` | Has `in_proj_qkv`; all four fusable | `in_proj_all`; `FusedGatedDeltaNet` | Qwen3.5 linear-attention layers |
| `Qwen3NextAttention` q/k/v | Fusable, no biases | `qkv_proj`; `FusedGatedAttention` | Qwen3.5 / Qwen3-Next full-attention layers |
| MLP gate/up | `gate_proj` and `up_proj` quantized, fusable, and `_swiglu_mlp` passes | `gate_up_proj`; `FusedMLP` | SwiGLU MLPs |
| Text model body | `Qwen3_5TextModel` and hidden size divisible by 256 | `FusedHybridBody` | Qwen3.5 (inactive, see below) |

A fused class is a new subclass named `"Fused" + <original name>`, with the fused class first in the method resolution order. The instance keeps all its other attributes (`fuse.py:104`, `fuse.py:294`, `fuse.py:308`, `fuse.py:314`). `fuse_model` returns a count of the layers it walked.

- `_swiglu_mlp` accepts an MLP class only if two checks pass (`fuse.py:75-97`):
  - **Source check:** `__call__` uses `swiglu(` or `silu(`, and no `gelu`, `relu`, `tanh` or `sigmoid(`. Gemma's MLPs use GELU.
  - **Numerical check:** the fused computation on a copy of the layer's own weights must match the original within `0.02 * (1 + max|ref|)` on a `(1, 3, dims)` input. The input comes from a fixed key, not the global random state.

  The verdict is cached per class in `_MLP_KINDS`.
- LFM2 names its MLP `feed_forward`, so `mlp` is `None` and nothing is fused there (`fuse.py:282`).
- In `FusedGatedAttention`, the query projection has width `2 * h * hd`: it carries queries and an output gate. The output is multiplied by `sigmoid(gate)` (`fuse.py:208-226`).
- `FusedGatedDeltaNet` and `FusedGatedAttention` split their `__call__` into three parts: input projection, `_mix`, output projection. With that split, `locals/mixed.py` can run the projections for a batch's decode rows and a new prompt chunk in one pass, so the weights are read once. Only `_mix` runs per group (`fuse.py:118-120`, `fuse.py:203-204`, `mixed.py:60-67`). `mixed.supported` requires a body with `embed_tokens`, `norm`, `fa_idx` and `ssm_idx`, and layers from module `qwen3_5` or `qwen3_next` with the pre-norm attributes. Each layer's mixer must be an instance of `FusedGatedDeltaNet` (linear layers) or `FusedGatedAttention` (`mixed.py:20-38`). The MLP also runs once over both groups (`mixed.py:68`).

## Gated DeltaNet decode kernel (`gdn_kernel.py`)

For one decode token, stock mlx-lm runs about fourteen small kernels per linear-attention layer. At batch 1, those launches cost more than their arithmetic (`gdn_kernel.py:3-6`). `gdn_decode` does the whole step in one kernel. Each threadgroup owns one (sequence, value head) and runs four stages (`gdn_kernel.py:7-12`):

1. conv + silu over the head's q, k and v channels, writing the new conv state
2. RMS-normalize q and k
3. the delta-rule recurrence, with each simdgroup owning `RPS = Dv / NSG` rows of the head's state: 16 rows with Dv = 128 and NSG = 8. Each lane holds a float4 of each row. The module docstring's "per 4 rows" does not match the code.
4. gated RMS norm of the output

The grid is `(32, NSG, B * Hv)` with `NSG=8` (`gdn_kernel.py:185`, `gdn_kernel.py:200-201`). Rounding to the activation dtype happens at the same points as in mlx-lm, so results match the stock path up to summation order (`gdn_kernel.py:14-15`). For S > 1 the state rows stay in registers across steps. A single step streams them instead, which keeps register pressure as in a plain decode kernel (`gdn_kernel.py:42-44`). The `states=True` variant (`locals_gdn_step2_states`) also writes the recurrent state after every token, which is needed to rewind a partially accepted draft (`gdn_kernel.py:163-174`).

`FusedGatedDeltaNet._mix` uses the kernel only when all of these hold (`fuse.py:124-135`):

- `USE_GDN_KERNEL` is True (default).
- `S <= _GDN_MAX_S` (4), or up to 16 tokens when `cache.want_states` is set.
- There is no mask, and the cache holds both a conv state and a recurrent state.
- `cache.lengths` is `None`. Ragged padded batches set lengths and take the masked stock path (`hybrid_batch.py:601-603`).
- The recurrent state is float32, and the conv state has the projection's dtype.
- `_gdn_ok` was set at fusion time: the norm eps is `1e-6`, the conv weight has shape `(C, KC, 1)`, and `supported(Hk, Hv, Dk, Dv, KC)` is true. That function requires `Dk == Dv == 128`, `Hv % Hk == 0` and `KC >= 2` (`fuse.py:303-306`, `gdn_kernel.py:178-179`).

When `want_states` is on and S > 1, the layer stores `cache.token_states` and `cache.token_conv`. `engine.rewind` uses them to roll back rejected draft tokens (`fuse.py:144-148`, `engine.py:545-563`). The `set_want_states` helper is defined at `engine.py:566-571`. Otherwise the stock recurrence (`gated_delta_update`) runs. Its q/k norm uses eps `1e-6 / Dk` so that it matches the reference l2norm, as mlx-lm 0.32 and later do (`fuse.py:184-189`).

**Prompts and MLX's op.** An unpadded prompt (no mask, no `cache.lengths`, more than 4 tokens) can go through MLX's own `mx.fast.gated_delta_update` instead, a chunked algorithm of matrix products with a version for the M5's neural accelerators. `kernels.gdn_fast` decides per layer shape and dtype, from timings `locals tune` makes with the whole model reading 512- and 2,048-token prompts each way (`gdn_model_tune`, keys `GDN{Hk}x{Hv}x{Dk}x{Dv}xT{512|2048}`); the op is kept where it is at least 3% faster. Timed alone, the op is 1.3-1.9x faster on an M2 Pro, but inside Qwen3.5-4B it was not (909 against 898 ms for 512 tokens, 3,389 against 3,366 ms for 2,048), so it is timed whole. Untimed shapes keep mlx-lm's kernel. With a mask the op is 12-50x slower and is never used. `LOCALS_GDN_FAST=1` or `0` forces it; `locals tune --report` shows the timings.

## Switched-off kernels

- **Fused add + RMS norm** (`norm_kernel.add_rms`). It computes `h = x + r` and `rms_norm(h) * w` in one kernel, which would halve the norm launches of a pre-norm layer (`norm_kernel.py:1-7`). It needs a hidden size divisible by 256 and uses 256 threads per row (`norm_kernel.py:67-80`). `FusedHybridBody` would use it, but `USE_ADD_RMS = False`: measured, it was no faster than MLX's two kernels. With the flag off, the body calls the stock `__call__` (`fuse.py:229`, `fuse.py:241-242`).
- **Ragged batch attention** (`attention.ragged_attention`). Adapted from MLX's `sdpa_vector`, it handles all rows of all requests in one launch. Row r sees the first `nkeys[r]` keys of KV slot `slots[r]`, and K/V are stored slot-major as `(n_slots, n_kv_heads, cap, D)` (`attention.py:1-8`). A GQA variant processes one (row, kv head) per threadgroup, so each K/V vector is read once (`attention.py:100-101`, `attention.py:205-215`). `packed.py` calls it only when `USE_RAGGED` is set, no segment is a tree, and every segment has at most 64 tokens. The flag is False because per-request MLX SDPA keeps outputs identical to single-stream decoding (`packed.py:70`, `packed.py:106`).

## Quantized matmul routing (`kernels.py`)

**Why.** MLX's quantized matrix-vector kernel runs one input row per threadgroup, so M rows read the weights M times. From M >= 6 (measured on an M2 Pro, for large matrices), MLX switches to a 32-row tiled kernel instead (`kernels.py:3-10`). With BM = 32 and a 2 x 2 simdgroup layout, a pass over 6 to 32 tokens costs as much as 32 rows (`mma_kernel.py:3-5`). Both kernels waste work at small M. Small M is the common case in Locals: forced spans and small decode batches.

**Patching.** `enable_small_m(enabled=True, model=None)` sets the process-wide `_enabled` flag (`kernels.py:523-536`). Given a model, it also changes the class of every module whose type is exactly `nn.QuantizedLinear` or `nn.QuantizedEmbedding` to a cached subclass, `LocalsQuantizedLinear` or `LocalsQuantizedEmbedding` (`_patched_class`, `kernels.py:513-520`). MLX's own classes are never modified, so other models in the same process (the server's stock engine, a library user's own model) stay as mlx-lm makes them. Called without a model, it only flips the switch for every model patched so far, which allows an A/B comparison (`kernels.py:524-529`).

**Call path** of a patched `QuantizedLinear` (`kernels.py:477-506`):

```
_linear_switch
  _enabled    -> _linear_call       -> _maybe_packed -> _linear_call_inner
                                                          _dense_ok  -> _dense_matmul   (rows >= PREFILL_M)
                                                          _eligible  -> _small_matmul   (tuned kernel choice)
                                                          otherwise  -> mlx-lm's original __call__
  not enabled -> _linear_call_plain -> _maybe_packed -> mlx-lm's original __call__
```

A tied embedding head goes through `QuantizedEmbedding.as_linear` -> `_embed_switch`. It takes only the small-row path, with no dense path and no packing (`kernels.py:499-510`).

**Which layers qualify.** `_tunable(mod)` requires a quantized linear or embedding with 4 bits, group size 64, affine mode, biases, and K divisible by 64. Other group sizes, and mxfp4 and similar formats, stay on MLX's kernels: tuning them crashed the load (`kernels.py:387-392`). The weight layout is MLX's affine 4-bit: `w` is `(N, K/8)` uint32 with the low nibble first, and `scales`/`biases` are `(N, K/64)` (`kernels.py:12-13`). `_eligible(x, mod)` also requires `MIN_M` (1) <= M <= `MAX_M` (32), where M is the number of rows after flattening all leading dimensions, and fp16 or bf16 activations (`kernels.py:395-402`).

**Dense prefill path.** At `PREFILL_M = 128` rows or more, a qualifying layer can be computed as dequantize + dense GEMM in the activations' own dtype instead of MLX's quantized GEMM (`_dense_ok`, `_dense_matmul`). Which one is timed on the GPU once per layer shape at `PREFILL_TUNE_M = 1024` rows and cached with the other choices (`_dense_prefill`, `_tune_prefill`; keys `NxKxP`): dense must be 5% faster, since it also needs the dequantized weights as a temporary. On an M2 Pro with MLX 0.32 it is 6-11% faster for the large shapes and level for the small ones (it was 1.5x with older MLX); MLX's quantized GEMM uses the neural accelerators of M5 GPUs (`quantized_nax.h` in MLX's kernels). The server times untimed shapes while idle and reads prompts densely until then; `locals tune` times them at once. `LOCALS_DENSE_PREFILL=1` always reads densely, `0` never. A bf16 model is not computed in fp16 on this path, because values above 65504 would become inf. Layers with more than `DENSE_MAX = 1 << 27` weights are excluded: a 248k x 2560 lm_head would need a 1.3 GB temporary.

**Row packing.** Inside `with packed_rows(B, L, lengths):` (a thread-local context), a linear layer applied to right-padded `(B, L, D)` activations gathers only the valid rows. It runs the matmul on those rows and scatters the result back into zeros, so padding costs no matmul work (`kernels.py:425-453`). Packing applies even when small-row routing is disabled (`_linear_call_plain`). Ragged batched passes use it. `hybrid_batch.py:609` is `HybridBatchEngine._forward`, the batch decode step, whose rows feed different numbers of tokens (called at `hybrid_batch.py:1298` and `:1355`). `regions.py:107` is `RegionScorer.logprobs`, which scores candidate paths in one right-padded pass.

## Kernel families

The autotuner chooses among these for each (N, K, M, dtype). A choice is a tuple whose first element names the family. `_launcher` turns a choice into a closure with the template and grid precomputed, cached by `(choice, M, K, N, dtype)` (`kernels.py:233-288`).

| Family | Choice tuple | File | How it works |
|---|---|---|---|
| `mlx` | `("mlx",)` | MLX | `mx.quantized_matmul`, the baseline |
| `multi` | `("multi", VPT, R, NSG)` | `qmv_multi.py` | MLX's `qmv_fast` layout, with M input rows held in registers. Each weight word is loaded and converted once, then applied to all M rows. Nibbles are used in place with MLX's mask trick, x pre-divided by 1, 16, 256, 4096 (`qmv_multi.py:3-8`) |
| `v3` | `("v3", VPT, R, NSG)` | `qmv_multi.py` | No int-to-float conversions, which are slow on Apple GPUs. A masked nibble OR-ed into the mantissa of 2^23 gives `2^23 + q * 16^p` exactly, and a subtraction leaves the value, using full-rate instructions only (`qmv_multi.py:127-134`) |
| `frag` | `("frag", TN, NSG, SPLIT)` | `qmv_frag.py` | Computes `Y^T = W X^T` with simdgroup 8x8 matrices, dequantizing into the A-fragments in registers. No threadgroup memory and no barriers. The order of k is permuted so that each lane finds its 32 weights for a 128-wide chunk in one `uint4`, and every weight is read once (`qmv_frag.py:1-20`) |
| `fragh` | `("fragh", TN, NSG, SPLIT)` | `qmv_frag.py` | `frag` with half-precision fragments. Two nibbles become a `half2` through the exponent of 1024 (`v \| 0x64006400`), and one `half2` fma applies scale and bias. Activations are clamped to +/-65504 (`qmv_frag.py:190-201`, `qmv_frag.py:263-264`) |
| `small` | `("small",)` | `kernels.py` (`qmm_small`) | Scalar FMA kernel that reads and dequantizes each weight once for all M rows. Measured on an M2 Pro, it beats MLX for 3 <= M <= 6 on large matrices and loses at M = 8 (`kernels.py:6-10`). Launch shape from `small_config(N)` (`kernels.py:412-414`) |
| `mma` | `("mma", BM, BN, BK, WM, WN, split)` | `mma_kernel.py` | MLX's `QuantizedBlockLoader` and `steel::BlockMMA` with BM = 8 (or 16) instead of 32. Every weight tile is dequantized once and applied to up to BM rows. MLX's headers are inlined from the installed wheel because custom kernels get no include path (`mma_kernel.py:1-12`) |

Notes on the families:

- In `frag` and `fragh`, M is a runtime input (`mcount`), and only `MT = (M + 7) // 8` is compiled in. One compiled kernel therefore serves eight row counts (`qmv_frag.py:41`, `qmv_frag.py:169`).
- `SPLIT > 1` (split-K, used by `frag`, `fragh` and `mma`) divides K across threadgroups, writes float32 partials and sums them afterwards. This helps small matrices fill the GPU (`mma_kernel.py:151-153`, `qmv_frag.py:178-179`).
- Because `fragh` clamps activations to the fp16 range, it is a candidate only for fp16 activations. A cached `fragh` choice read back for bf16 activations is replaced by `mlx` (`kernels.py:317-318`, `kernels.py:336`).

## Autotuning

### Per-shape tuning (`_choose`, `_tune_shape`)

`_choose(x2, mod)` looks up the choice in this order (`kernels.py:306-325`):

1. The in-memory map `_CHOICE[(N, K, M, dtype)]`.
2. The tuning dictionary, under the key `"{N}x{K}x{M}"`, with `"x{dtype}"` appended when the dtype is not `float16`.
3. If the background tuner is deferring work (`_DEFER` set, `_TUNING` not set), it returns `("mlx",)` without remembering it, so a later lookup can find the tuned result.
4. Otherwise it tunes the shape now, under `_TUNE_LOCK`.

`_tune_shape` builds the candidate list from these rules (`kernels.py:331-344`):

| Family | Row counts M | Other conditions | Candidates |
|---|---|---|---|
| `mlx` | all | none | `("mlx",)` |
| `multi` | M <= 4 | `K % (VPT*32) == 0`, `N % R == 0` | `_MULTI_CANDIDATES` = (8,4,2), (16,4,2), (8,2,2) |
| `v3` | 3 to 6 | `K % (VPT*32) == 0`, `N % R == 0` | `_V3_CANDIDATES` = (8,8,2), (8,4,2), (16,4,2), (16,4,4), (16,2,2) |
| `frag` | 1 to 32 | `K % (128*SPLIT) == 0`, `N % (8*TN) == 0`; SPLIT > 1 only when K >= 4096 | `_FRAG_CANDIDATES` = (2,2,1), (4,2,1), (2,4,1), (1,2,2) |
| `fragh` | 1 to 32 | as `frag`, fp16 activations only | same as `frag` |
| `small` | 3 to 6 (`MIN_M_SMALL`, `MAX_M_SMALL`) | N >= 4096 or K >= 4096 | `("small",)` |
| `mma` | 2 to 32 | `N % BN`, `K % (BK*split)`, `(K/split) % 64` and `BN % (8*WN)` all zero | `_MMA_CANDIDATES[8]` for M <= 8, `[16]` above |

At M <= 2, `v3` is not a candidate: it wins in isolation there but loses about 1% inside the model (`kernels.py:334`).

The tuning procedure (`kernels.py:345-384`):

1. **Correctness.** Every candidate runs on a random input (fixed key `M * 131 + K`, not the global random state). Its output must be within `0.05 * (1 + max|ref|)` of MLX's. A NaN or an exception removes the candidate.
2. **Timing.** Candidates are timed on `max(1, min(16, (64 << 20) // nbytes))` copies of the weights. The aim (code comment) is that the weights come from DRAM as inside a model, not from the GPU's system cache. In fact the copies total at most about 64 MB, unless a single copy is already larger, and well below that for layers under 4 MB, where the 16-copy cap applies. Each timing uses `reps = max(1, 8 // ncopy)` and is divided by the copy count. Four interleaved rounds run, and each candidate keeps its best round.
3. **Decision.** A non-MLX kernel must be at least 10% faster than MLX (`< 0.9 *` MLX's time). Otherwise the choice is `("mlx",)`.
4. **Storage.** The choice is stored under the shape key. The best candidate of each family is also stored under `"{key}|{family}"` for the whole-step tuner. The cache file is rewritten.

### Whole-step tuning (`model_tune`, full mode only)

Per-shape timings in isolation miss interactions inside a model, such as cache reuse across layers and overlap of consecutive kernels. For example, at M = 4 `v3` loses to `fragh` on every shape alone, yet it makes the Qwen3.5-2B decode step about 4% faster (`kernels.py:648-656`).

For each M from 2 to 32, `model_tune` gathers several options: the per-shape mix, plus each family whose per-shape winner exists for every shape of the model. It times a real batch-M decode step with each option, after a 64-token prefill per row, for 16 steps per round over 5 rounds, and takes the median. A family overrides the per-shape mix only if it is at least 2% faster (`< 0.98 *`) (`kernels.py:670-721`).

Timing is pipelined, as the engines decode: step s+1 is built and queued while step s runs. Synchronous timing ranks families differently. On the 0.8B model at M = 8 in fp16, synchronous timing picked `fragh`, which was 17% slower pipelined than `frag` (`kernels.py:698-701`). Finished row counts are recorded under `"__model_tuned_pipe__" + md5(shapes)[:10]` so they are not repeated (`kernels.py:667-668`, `kernels.py:722-724`).

### Tuning modes (`tune_model`)

| Mode | What runs | Where it is the default |
|---|---|---|
| `auto` | If the cache or preset covers every shape for 1 to 16 rows (`tune_pending(max_m=16) == 0`), only `warmup`. Otherwise the `quick` procedure | `Runtime`, `locals.lm` |
| `background` | Serve at once. Untuned shapes run on MLX's kernel, and the server tunes them while idle. Shapes tuned already (for example from a preset) are compiled at load | The server engine (`service.py:280`) |
| `quick` | `pretune(max_m=16)`: every shape for 1 to 16 rows, then `warmup` | `locals tune --mode quick` |
| `full` | `pretune()` up to 32 rows, then `model_tune`, then `warmup` | `locals tune` (`app.py:1248-1259`) |
| `off` | Nothing ahead of time. Untuned shapes are tuned on first use | none |

Sources: `kernels.py:834-882`. Any other value raises `ValueError`. The log line at `kernels.py:872-874` gives quick tuning on an M2 Pro as 2 minutes for Qwen3.5-4B and 7 for Gemma 4 E4B. The full procedure takes 10-20 minutes and gives the best batched throughput (`runtime.py:126-128`).

Two consequences for row counts above 16:

- In `auto` mode, a shape with 17 to 32 rows that is not cached gets tuned the first time it occurs, because deferral is off (`kernels.py:851`, `kernels.py:321-325`).
- In `background` mode those row counts stay on MLX's kernels (`kernels.py:857`). `pending_work` defaults to `max_m=16`.

`warmup(model, ms)` compiles and runs the chosen kernel of every shape for M in `ms` at load. `tune_model` calls it with 1 to 16 after every mode except `off`, which returns before any warmup. The function's own default is `ms=range(1, 9)`., because each new kernel instance costs about 50 ms of Metal compilation. A kernel that fails to compile or run falls back to `("mlx",)`, which covers a cached choice that the installed mlx no longer compiles (`kernels.py:729-754`).

### Background tuning while idle

When `tune_model` returns with `_DEFER` set, the service builds `kernels.pending_work(rt.model)`: the untuned (layer, rows) pairs, smallest batches first (`service.py:286-287`, `kernels.py:786-801`). When the request queue is empty, `_take` calls `_tune_while_idle`, which tunes one pair with `tune_one` in the engine's own thread, so the GPU is not shared. A request that arrives meanwhile waits for at most that one shape (`service.py:205-232`). `tune_one` sets `_TUNING` so that `_choose` really tunes instead of deferring (`kernels.py:804-812`). From then on the shape runs on its tuned kernel. When the list is empty, the server logs that tuning is finished.

### Cache file and presets

- **File name:** `tune_v6_<architecture>.json`, where the architecture is `mx.device_info()["architecture"]`. Results are therefore per GPU architecture (`kernels.py:133`).
- **Location:** `LOCALS_TUNE_CACHE` if set. It names the file itself, not a directory. Otherwise `~/.cache/locals/<file name>` (`kernels.py:134`). The path (environment variable and GPU architecture) is fixed when `kernels.py` is imported, so set `LOCALS_TUNE_CACHE` before importing locals. The file itself is read lazily, once per process, by `_load_tune`.
- **Load order** (`_load_tune`, `kernels.py:169-198`):
  1. The shipped preset for this architecture is loaded first.
  2. The user's cache is loaded on top and overrides it.
  3. A user cache that does not parse is renamed to `.bad`, not silently replaced.
  4. If there is no readable user cache (none, or one just renamed to `.bad`) and the preset is non-empty, the preset is written out as the user cache. Later writes also save the merged preset + user dictionary.
- **Writes** (`_save_tune`) replace the whole file atomically, via a temporary file and `os.replace`. Writes and tuning both take `_TUNE_LOCK`, an `RLock`, because the server runs two engines (`kernels.py:161`, `kernels.py:201-219`).
- **Format:** a JSON object. Shape keys (`"2560x10240x4"`, `"…x4xbfloat16"`) map to choice lists such as `["frag", 2, 2, 1]`. `"…xP"` keys hold the prompt-reading choice (`["dense"]` or `["mlx"]`). `"__times__"` holds what tuning measured, per shape key and family in microseconds, which `locals tune --report` (`tune_report`) sums per batch size. `"…|family"` keys hold per-family winners. `"__model_tuned_pipe__…"` keys hold lists of finished row counts. Keys that start with `__` are kept as they are, and all others are read as tuples (`kernels.py:164-166`).
- **Presets:** `locals/presets/tune_v6_applegpu_g14s.json` (M2 Pro and M2 Max) and `tune_v6_applegpu_g17s.json` (M5 Pro: so far only how prompts are read, measured on one for Qwen3.5-4B and 9B) are shipped. It is packaged through `[tool.setuptools.package-data]` (`pyproject.toml:58-59`). Setting `LOCALS_NO_PRESET=1` hides presets, so a start behaves as on a GPU without one (`kernels.py:145-156`).

To re-tune from scratch, delete the cache file (or point `LOCALS_TUNE_CACHE` at a new path) and set `LOCALS_NO_PRESET=1`.

## Output-layer helpers (`kernels.py`)

| Function | Purpose |
|---|---|
| `text_model(model)` | The decoder inside a multimodal wrapper (`language_model`), or the model itself (`kernels.py:631-633`) |
| `tied_embeddings(lm)`, `head_module(lm)` | Whether the vocabulary projection is the embedding table: the flag may be on the instance (Gemma 3), in `args` (Llama, Qwen), or `lm_head` may be missing. Returns the module that projects (`kernels.py:545-560`) |
| `post_logits(lm, out)` | Reproduces what the model's forward does after the projection: soft-capping (Gemma 2, 3n, 4), Cohere's `logit_scale`, Granite's `logits_scaling`. These are monotonic, so greedy and constrained choices never depend on them (`kernels.py:563-579`) |
| `head_logits`, `forward_last`, `forward_hidden` | `head_logits` projects the hidden states it is given and applies `post_logits`. `forward_last` runs the body and projects only the last position, falling back to the model's own forward when it has no `model` or `args`. `forward_hidden` returns the last position's final normed hidden state (1, 1, D) and does not project (`kernels.py:582-586`, `kernels.py:636-645`, `kernels.py:539-542`) |
| `head_check(model, log)` | Compares the model's own forward against body + `head_logits` on a 6-token probe. A difference above `0.02 * max|ref| + 0.05` is logged; the function returns the max difference, or NaN on an exception. Nothing in `locals/`, `tests/` or `bench/` calls it (`kernels.py:589-606`) |
| `head_rows(model, h, ids)` | Exact logits for only the vocabulary rows `ids`. A constrained step often allows a handful of tokens, and gathering those rows of the 4-bit head skips most of its weight traffic. It uses the layer's own quantization mode, so an mxfp4 head works too (`kernels.py:614-628`); the engine calls it for sparse steps (`engine.py:703`) |

## Model names and checks before download (`defaults.py`)

`MODELS` holds the short names Locals is tested with (`defaults.py:7-13`). Any mlx-lm chat model id or local path also works.

| Short name | Repository |
|---|---|
| `qwen3.5-4b` | `mlx-community/Qwen3.5-4B-MLX-4bit` |
| `qwen3.5-2b` | `mlx-community/Qwen3.5-2B-MLX-4bit` |
| `gemma-4-e4b` | `mlx-community/gemma-4-e4b-it-4bit` |
| `lfm2.5-2.6b` | `LiquidAI/LFM2.5-2.6B-MLX-4bit` (1.6 GB: 8 GB Macs) |
| `minicpm5-2b` | `mlx-community/MiniCPM5-2B-mlx-4Bit` (1.4 GB) |

`ALIASES` maps more names onto these: `qwen` and `qwen3.5` to the 4B; `gemma`, `gemma4`, `gemma-4`, `gemma4-e4b` and `gemma-4-e4b-it` to Gemma 4 E4B; `lfm` and `lfm2.5`; `minicpm` and `minicpm5` (`defaults.py:139-141`).

- **`default_model()`** reads the Mac's memory with `sysctl -n hw.memsize`. It picks `qwen3.5-2b` when memory is known and below 12 GB, and `qwen3.5-4b` otherwise, including when memory is unknown (`defaults.py:16-23`, `defaults.py:131-135`). The 4B (3 GB of weights, 64k context) and its prompt cache do not fit next to macOS in 8 GB. `DEFAULT_MODEL` is computed once, at import (`defaults.py:138`).
- **`model_id(name)`** resolves a `--model` value; every command uses it as the argparse `type` (for example `server.py:1482`). Resolution order (`defaults.py:155-166`):
  1. A short name or alias, matched case-insensitively, becomes its repository. A short name wins over a folder of the same name in the current directory.
  2. An existing path becomes its absolute path, because the background server runs elsewhere and later commands may run from other folders. Symlinks are kept.
  3. Anything else is returned unchanged.
- **`short_name(model)`** gives the model's name in the API (`locals/<short name>`). It takes the last path component, strips a trailing `-4bit` or `-mlx-4bit` in any capitalization, and lowercases the rest: `MiniCPM5-2B-mlx-4Bit` becomes `minicpm5-2b` (`defaults.py:147-152`). It is used by the server (`server.py:151`).
- **`too_big(model, remote=False)`** returns a reason the model cannot run, or `None` (`defaults.py:69-81`).
  - It fails when weights + `HEADROOM_GB` (2.0) exceed `gpu_limit_gb()`, which is MLX's `max_recommended_working_set_size`.
  - The headroom covers the KV cache of a few agents' prompts, the prefill, and the server's margins. Without it, macOS swaps, or Metal stops the process; on a 16 GB Mac that can be a GPU driver panic that takes the whole machine down (`defaults.py:26-29`).
  - `weights_gb` sums the top-level `*.safetensors` files of a local folder or of the Hugging Face cache, without network access. With `remote=True` it falls back to the file sizes Hugging Face lists (`defaults.py:42-66`).
  - An unknown size or limit passes the check. `LOCALS_IGNORE_MEMORY=1` skips it.
- **`gguf_hint(model, remote=True)`** detects a GGUF model: a `.gguf` file, or a folder or Hugging Face repo with `.gguf` weights and no safetensors. Locals runs MLX models only, and mlx-lm loads no GGUF (`defaults.py:84-128`).
  - A repo already in the cache with safetensors is accepted without contacting Hugging Face, so startup works offline.
  - For a hint, it strips `gguf` and quantization suffixes (`Q4_K_M`, `IQ…`, `F16`, `BF16`, `F32`) from the name. It searches Hugging Face for MLX builds (`filter="mlx"`, sorted by downloads, at most 3) under that name, and under the part after the first `_` (bartowski's `Qwen_Qwen3.5-0.8B` naming).
  - If nothing is found, it suggests mlx-community or `mlx_lm.convert --hf-path ORIGINAL -q`.
- **Where the checks run:** `locals serve` and the app's model fetch both call `gguf_hint(...) or too_big(..., remote=True)` before downloading anything, and exit with the message if either returns one (`server.py:1519-1523`, `app.py:465-471`).
- **`build_id()`** fingerprints the installed code: a SHA-1 of the names, sizes and modification times of `locals/*.py`, except the client-only `plan.py` and `solve.py`, truncated to 12 hex digits (`defaults.py:169-181`). The app compares it with a running server's to detect a server started before an upgrade (`app.py:554-557`).

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `LOCALS_TUNE` | `auto` (library), `background` (server) | Tuning mode when none is passed (`runtime.py:144`, `lm.py:64`, `service.py:280`) |
| `LOCALS_TUNE_CACHE` | `~/.cache/locals/tune_v6_<arch>.json` | Path of the tuning cache file (`kernels.py:134`) |
| `LOCALS_NO_PRESET` | unset | Any non-empty value hides the shipped presets (`kernels.py:148`) |
| `LOCALS_MAX_COMPILED` | `16` | Most compiled constraints kept (`runtime.py:162`) |
| `LOCALS_COMPILED_GB` | `1.5` | GB of constraint tables kept (`runtime.py:245`) |
| `LOCALS_IGNORE_MEMORY` | unset | Skip the `too_big` memory check (`defaults.py:74`) |
| `LOCALS_DTYPE` | `auto` | The compute dtype `Runtime` uses when `dtype="auto"`: `float16` or `bfloat16` forces one (to measure a new GPU; `auto` picks float16 only on GPUs without native bfloat16) |
| `LOCALS_FUSE` | `1` | `0`: no fused projections |
| `LOCALS_KERNELS` | `1` | `0`: MLX's matmul kernels for few rows (no tuned kernels) |
| `LOCALS_GDN_KERNEL` | `1` | `0`: mlx-lm's Gated DeltaNet decode instead of `gdn_kernel.py` (`fuse.USE_GDN_KERNEL`) |
| `LOCALS_DENSE_PREFILL` | timed | `1`: prompts always through dequantize + dense GEMM; `0`: through MLX's quantized GEMM |
| `LOCALS_PAD_ROWS` | `1` | `0`: never compute a batch at more rows. By default batches of 8 to 16 are computed at a larger size, up to twice the rows, where the tuned timings show it at least 7% faster (`_pad_rows`; on an M5 Pro 12 requests computed as 16 ran 10-20% faster). `all`: other sizes too (the timings also pad 2 rows to 3, 4 to 5 and 24 to 32 on some M5 layers; measured end to end, within noise) |
| `LOCALS_GDN_FAST` | timed | `1`: unpadded prompts through MLX's `mx.fast.gated_delta_update` in Qwen3.5's recurrent layers; `0`: through mlx-lm's kernel (`kernels.gdn_fast`) |

## Code switches

| Constant | Value | Where |
|---|---|---|
| `USE_GDN_KERNEL` | `True` unless `LOCALS_GDN_KERNEL=0` | `fuse.py:108` |
| `_GDN_MAX_S` | `4`; the 16-token limit with `want_states` is a literal in the condition at `fuse.py:126`. `gdn_kernel.MAX_S = 16` (`gdn_kernel.py:182`) is defined but never referenced | `fuse.py:108`, `fuse.py:126` |
| `USE_ADD_RMS` | `False` | `fuse.py:229` |
| `USE_RAGGED` | `False` | `packed.py:70` |
| `MIN_M`, `MAX_M` | `1`, `32` | `kernels.py:131-132` |
| `MIN_M_SMALL`, `MAX_M_SMALL` | `3`, `6` | `kernels.py:417-418` |
| `PREFILL_M` | `128` | `kernels.py:456` |
| `DENSE_MAX` | `1 << 27` | `kernels.py:459` |

## Things to keep in mind when changing this code

- Never draw from MLX's global random state during load or tuning. Seeded sampling must match mlx-lm's. That is why `_concat_quantized` avoids the constructor, and why the probes use fixed keys (`fuse.py:28-30`, `fuse.py:90`, `kernels.py:345`).
- `enable_small_m` matches exact types only. A subclass of `QuantizedLinear` may compute differently and is left alone (`kernels.py:535`).
- Tuning keys carry the activation dtype. Different GPU architectures already use separate cache files (`kernels.py:133`). The dtype in the key separates a bf16 model from an fp16 model on the same GPU, for example Gemma 3 (kept in bf16) next to Qwen3.5 (fp16) on an M2. `fragh` must never run on bf16 activations. (`kernels.py:313`, `kernels.py:317-318`).
- A new kernel family needs four things: a branch in `_launcher`, candidate rules in `_tune_shape`, an entry in `model_tune`'s `families` if it should compete there, and correctness under the tolerance check. A new choice can also break a cached file on another mlx version; `warmup` falls back to MLX in that case.
