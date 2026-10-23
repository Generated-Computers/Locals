# Batch engine: how requests are scheduled

Part of the [technical documentation](../README.md) of Locals. File and function names refer to the `locals` package.

`HybridBatchEngine.serve` in `locals/hybrid_batch.py` is the loop behind the `locals` engine. It uses continuous batching: requests join and leave a running batch between steps. A single background thread in `BatchService` (`locals/service.py`) owns the model and calls `serve`. HTTP handlers queue jobs; a job with a schema or regex (except with `--walk`) is queued by a `locals-compile` thread once `Runtime.precompile` has made its grammar's host tables, one grammar at a time, so the loop only copies them to the GPU (see [Server](../server/README.md)). This section covers what that loop does on each step:

- when rows join and leave the batch
- how prompts are prefilled
- how decoding continues while a long prompt is prefilled
- how GPU memory stays under a limit
- how sampling and constraints work for each row

## Files

| File | What the scheduler uses from it |
|---|---|
| `locals/hybrid_batch.py` | `HybridBatchEngine` (`serve`, admission, prefill, `_launch`, memory guard, combined constraint tables); `prefill_chunk`, `PREFILL_SCORES`, `PREFILL_TOKENS`; the cache helpers `_reserve`, `_filter`, `_append_rows`; `chain_cache` (the prompt cache); `DiskPromptCache` |
| `locals/mixed.py` | `supported` and `forward`: one forward pass over the decode rows plus a chunk of a new prompt |
| `locals/packed.py` | `Request` (one generation), `BatchStats`, the `BatchEngine` base class |
| `locals/engine.py` | `Controller`, `LocalsController` (constraint state and forced spans), `sample_logits`, `sample_rows` |
| `locals/service.py` | `BatchService`: the worker thread, jobs turned into `Request`s, `memory_limit`, recovery from engine errors |

`HybridBatchEngine` subclasses `BatchEngine` but does not call its `__init__`. The base class's fixed-slot `PackedKV` buffer is therefore never allocated. The batch state lives in mlx-lm's batch caches instead: `BatchKVCache` for attention layers and `ArraysCache` for recurrent layers. Rows join with `merge`/`extend` and leave with `filter`. The base class's `_commit` and `_tables` are used only by the older `run(mode="ragged")` and `_run_sync` paths, which the service does not use.

## What a row carries

`Request` (`packed.py`) holds `prompt`, `controller`, `max_tokens` (default 256), `out`, `text`, `finish`, `passes` and `forced`. It also has `slot` and `feed`, which only the base class and the ragged and `_run_sync` paths use. The service and the loop attach more attributes at run time:

| Attribute | Set by | Meaning |
|---|---|---|
| `pend` | `arrive`, `_commit_drip` | Forced tokens still to feed, one per step |
| `fed` | `arrive`, `attach`, every step | Every token fed after `prompt[:-1]`, starting with `prompt[-1]`. The row's cache holds exactly `prompt[:-1] + fed`. |
| `temp` | service | This request's temperature (`None`: use the service's) |
| `on_token` | service | Callback that receives each step's committed tokens (streaming) |
| `cancel` | `Job.cancel`; `source` (copies `job.cancelled`); `_after_commit` (when `on_token` raises) | The client went away, or the callback failed |
| `checkpoints` | service, from `job.checkpoints` (computed by `Server._checkpoints`) | `(position, kind)` pairs at which a copy of the prompt's cache is stored |
| `cached_tokens` | admission | Prompt tokens served from the prompt cache |
| `reply_state` | `_keep_reply` | Recurrent state kept at a `"reply"` checkpoint |

The values `finish` can take are `"stop"`, `"length"`, `"cancelled"` and `"error: on_token: ..."`.

A prompt is prefilled as `prompt[:-1]`. Its last token is fed by the row's first decode step (`attach` appends `prompt[-1]` to `xs` and `fed`). The prefill therefore runs the model body without the output layer. In `_run`'s words, those logits "(chunk x 248k) are not needed and cost ~1 GB at 2048".

## The loop: chained and boundary steps

`serve(source, temp, lru, lru_key, on_done)` runs until `source(True)` returns an empty list. With `block=True` the service's `source` returns an empty list once the service is closed. It also returns one when every job it took was a prime or was rejected as invalid, because both are finished inside `source` and produce no `Request`. `serve` then returns, and `_run_locals`'s `while not self._closed` loop calls it again. That resets its tables, `self.counters` and `tokens_out`.

Every step is drip mode: a uniform `(B, 1)` batch with one token per row. Each row feeds its next queued forced token, or else the token it sampled in the previous step. That sampled token is still on the device. Because the inputs never need the host, step s+1 can be launched before step s has been read.

```
chained step:   build x(s+1) on device -> launch s+1 -> read s (GPU busy with s+1) -> commit s
boundary step:  read s (GPU drains) -> commit, retire, join, admit -> launch s+1 from host tokens
```

### Chained step (the common case)

1. Rows that have queued forced tokens feed `pend.pop(0)`. All other rows feed `cur`, the step-s sample, which has not been read yet. The two are combined with `mx.where`.
2. If the batch has constrained rows, the device constraint states advance by that input: `g2 = g if free else self._advance(rows, x, g)`.
3. Step s+1 is launched with `_launch` and `mx.async_eval`.
4. Only then does `cur.tolist()` read step s. The GPU is already busy with s+1.
5. The host commits each live row's sample with `_commit_drip`, then runs the streaming and cancellation hook `_after_commit`.

A row is *live* if it had no queued forced tokens when the step was built. The sample of a row that is not live is discarded, because the forced token is what comes next. Every 32 steps a chained step also makes room for KV growth (see "Memory guard").

### What triggers a boundary

A boundary step runs when any of the following holds:

- `cur is None`: nothing is in flight (start-up, or the batch has emptied).
- `dead` is non-empty: a row finished (end-of-sequence token, constraint finished, `max_tokens`) or was cancelled at its last commit.
- A request is queued and `len(rows) + len(filling) < n_slots`.
- A prompt in `filling` is fully prefilled and can join.
- Any request in `filling` was cancelled.

The batch's rows change only at boundaries. The per-boundary flag `free` (no constrained rows) therefore stays valid through the chained steps that follow.

A request deferred for memory leaves the queue condition true. While it waits, every step is a boundary, so the loop runs unpipelined; the counter `deferred` counts it once. The growth trim (`_growth_trim`, every 32 steps) runs on boundary steps, chained steps and a lone row's passes alike: run only on chained steps, it stopped while a request waited for memory, when memory is tightest.

### Boundary step, in order

1. `cur.tolist()` waits for the in-flight step.
2. Feeds are decided before committing, as in a chained step. A live sample is fed now, and any forced span it produces follows on later steps.
3. Live rows are committed (`_commit_drip`, `_after_commit`).
4. Dead rows are retired (`retire`). Then `_filter` keeps the remaining rows in the cache, and `g` is indexed the same way.
5. Feeds are appended to `fed`, and `g` advances by them.
6. Cancelled queued requests and cancelled `filling` entries are dropped (see "Cancellation").
7. If `filling` is non-empty but no rows remain, the remaining prompts are prefilled at once and joined.
8. Fully prefilled `filling` entries join the batch. Each mixed pass reads a chunk of the entry with the fewest tokens left (`_launch`), so a short prompt that arrives while a long one is read does not wait for all of it: a 3k-token request waited 41 s behind a 31k-token one in arrival order, 7 s now, and the long one finished as before (46 s, 44 s).
9. Admission runs.
10. If there are still no rows, `cur = None` and the loop continues. At the top of the next iteration, if `arrive(False)` brought nothing and `filling` is empty, the combined tables are reset and `source(True)` blocks.
11. Otherwise the next step is launched from host-known tokens, `mx.array(xs)`.

The token budget passed for masking is `max_tokens - (len(fed) - 1)` at a boundary and `max_tokens - len(fed)` in a chained step. The comment at the boundary explains why: "a row's next committed sample follows exactly the tokens it has fed after its prompt, so out has len(fed) - 1 tokens then".

## Drip mode and forced spans

When `_commit_drip` commits a token, it asks the controller for a forced span. It calls `forced(y, heal=False)` for a `LocalsController`. The span is appended to `out` right away and queued in `r.pend`. Later steps feed it one token per step, so the batch never widens.

The span is planned without one-token healing because the sampled token has already been fed by the time the span is known (`run_drip` docstring). A row that finishes during a chained step has already been fed into step s+1. That wasted row is dropped at the next boundary.

## Admission

The engine has `n_slots` slots. The default is 16 in `HybridBatchEngine`, in `BatchService`, and for the server's `--slots` flag. Slots are counted as `len(rows) + len(filling)`: a prompt being prefilled in mixed passes already holds one.

Admission is first-in, first-out. It takes `queue[0]` while slots are free, and it stops at the first request that does not fit, so smaller requests behind it wait as well. The fit check applies only when something is already running or admitted (`rows or new or filling`). An empty engine always admits the head of the queue.

The fit check works as follows:

- `_merge_tokens(rows, new_lens)` estimates the extra KV positions that merging adds at its peak. Every row is padded to the longest one, and `_append_rows` copies one attention layer of the merged batch at a time. The estimate is `after - n*top + after // _kv_layers`, where `after = (n + len(new_lens)) * max(top, *new_lens)`.
- `_fits(lru, r, held, rows, tokens)` estimates what `r` needs: `(len(prompt) * (1 + copies) + held) * _kv_bytes_per_token() + _admit_bytes(rows, tokens) + replies * _reply_bytes`. It compares this with `memory_limit - (active memory - prompt-cache bytes)`. In other words, it asks whether the request would fit once the prompt cache has been emptied. `copies` counts the request's non-`"reply"` checkpoints, each of which keeps a copy of the prompt's cache. `rows` are the requests admitted in this round with `r`, whose prompts (`tokens` of them) are read together.
- `_admit_bytes(rows, tokens)` is what those requests need besides their KV. Each brings its fixed state (`_row_state`: every cache layer that is not a growing KV cache, measured once on a one-row cache after one token: 51.5 MB for Qwen3.5-4B's recurrent layers, about 154 MB for a 27B), held 2.2 times over while the batch decodes (14 Qwen3.5-4B requests admitted together added 1.68 GB, 120 MB each: the step being computed and the one queued behind it each hold their own), and a first KV block of 256 positions. Measured right after an admission, a request adds only its state once (52 MB), as no step is in flight yet, so the factor is not learned there. Reading the prompts needs `_read_bytes(tokens)`: each prefill pass records what it needed above the memory in use before it (`_note_read`, per power of two of tokens), and the room kept is the most a recorded pass of as many tokens or more needed, a sixth more; 1.5 GiB (`PREFILL_RESERVE`) until one was recorded, or always with `LOCALS_PREFILL_RESERVE_GB`. Reading is no straight line in the tokens (Qwen3.5-4B: 0.14 GB for 16 tokens, about 1.1 GB for 128 to 256, 2.1 GB for 2048), so nothing is extrapolated. The fixed state was not counted before: 16 requests at once on a 27B on 24 GB ran Metal out of memory. With 0.9 GB above what is in use, 16 short requests on Qwen3.5-4B now run 2.5 times as fast as one at a time and stay under the limit (`tests/test_memory_guard.py`).

The admitted requests are then prepared:

1. **Cached prefixes are looked up first, for all new requests.** The comment explains the order: "making room for one must not drop the next one's". `DiskPromptCache.restore` loads a system prompt saved before a restart into the prompt cache when memory holds nothing as long. Then `lru.fetch_nearest_cache` returns the longest cached prefix and the tokens still to prefill (`rest`).
2. **`_make_room` drops prompt-cache entries** until the whole admission fits. The comment explains why whole prompts count: each cached prefix is "a lazy copy, made during the prefill".
3. **The engine chooses between mixed passes and a prefill at once.** It uses mixed passes when `_mix_ok` is set, rows are already decoding, and the admitted prompts' remaining tokens total more than `mix_chunk`. If so, every admitted request becomes a `filling` entry. Otherwise the prompts are prefilled together immediately:
   - For each checkpoint between the cached prefix and the prompt's end, the prompt is prefilled up to that position with `_run`. A deep copy is stored in the prompt cache under that kind (`"system"` or `"user"`); this happens only if `_room_for_copy` allows it. `"system"` snapshots are also saved with `DiskPromptCache.save`. A `"reply"` checkpoint keeps only the recurrent state (`_keep_reply`).
   - `_prefill_batch` then prefills everyone's remainder, and `attach` joins the rows.

Small prompts admitted while a long one is still in `filling` are prefilled at once and join ahead of it.

`attach` does the following:

- extends the combined constraint table (`_table_add`) and, if the table was rebuilt, moves the running rows' states
- appends the new rows to the batch cache (`_append_rows`), or uses them as the cache if the batch was empty
- adds the new rows' start states to `g`
- adds each new row's `prompt[-1]` to the next feed

Checkpoints are only requested for models whose cache cannot be cut back to a shorter prefix. `server.py` `_needs_checkpoints` checks for this (Qwen3.5's recurrent layers, Gemma's sliding windows). `_checkpoints` places them at the end of the system part (`"system"`): the system message with its tools plus the user turn's header, found where two renderings with different user messages first differ. It also places them at the start of the last message (`"user"`), and where the reply starts as the next request will render it (`"reply"`). A candidate is used only if its rendered text is at least 2048 characters and a prefix of the prompt, and its token position n satisfies 1024 <= n < len(prompt), with one checkpoint per position.

## Prefill

### `_prefill_batch(prompts, others, chunk=1024, starts=None, stop=None)`

This prefills several prompts in one right-padded batch and returns a batch cache. `starts[i]` is a `(cache, remaining tokens)` pair, typically from the prompt cache. Without one, `_prefill_start` provides the start.

- **The starts are handed over.** The caches are merged into the batch one layer at a time (`_merge_layerwise`), each request's layer released once copied, and the `starts` list is emptied; callers drop their own references before the call. Before, the callers held every request's cache until the rows had joined, so the prompts' KV existed twice at the join while admission counted it once: three 26.7k-token prompts read up to their reply checkpoints and joined together took a 4B on a 16 GB Mac to 12.6 GB, past the GPU's 11.45 GB (with three 20k-token prompts, 6.75 GB in use right after the join against 4.76 now).

- **Rows with nothing to prefill** are separated out, for example the second of two requests that share a whole prompt. Only the rows that need work are prefilled; the others are merged in afterwards. The comment gives the reason: padding such rows "cost most of an admission (8 of 10 rows, 236 tokens each)".
- **One prompt that continues a cached prefix** grows its own KV buffers to the final length (`_reserve`) and wraps them as a batch of one (`_wrap_one`). It does not copy the prefix into a new batch buffer. The comment's example: "a 36k-token prefix of a full-attention 3B model is 4 GB: the copies did not fit next to each other on a 16 GB Mac".
- **Ragged lengths** are handled with `prepare(right_padding=...)` on `BatchKVCache` and `prepare(lengths=...)` on `ArraysCache`, followed by `finalize()`.
- **Chunks** are sized by `prefill_chunk(done + i, len(prompts), chunk, self._prefill_scores)`, where `done` is the longest reused prefix and `chunk` defaults to `self._prefill_base` (1024, or the model's `prefill_chunk_tokens`). After the first chunk, the KV buffers are reserved for the full length.

### `_prefill_start` and shared-prefix snapshots

`_prefill_start(tokens, others)` starts from a deep copy of the longest snapshot in `self.snapshots` that the prompt begins with. It then looks at the longest common prefix with `others` (the rest of the queue and the later prompts in the same admission). If that prefix is at least `min_share` tokens (default 64) and extends past the snapshot, it prefills up to there and saves a new snapshot. The method returns the cache and the tokens still to prefill.

Only `max_snapshots` snapshots are kept (`LOCALS_MAX_SNAPSHOTS`, default 4, at least 1). The least recently used is dropped first, because a snapshot that is reused is moved to the end of the `OrderedDict`. The comment explains the limit: keeping them all "grew by gigabytes over a night of agents". `_trim_for` clears all snapshots when memory is short.

### `_run(cache, toks, chunk=2048, stop=None)`

This prefills a single sequence through the body only. It is used for checkpoints in a prefill at once, for a filling entry's remaining checkpoints (`_fill_checkpoints`, when no rows are left), for shared-prefix snapshots in `_prefill_start`, and for primes. The rest of a filling entry after its last checkpoint goes through `_prefill_batch`, not `_run`. It reserves the KV buffers after the first chunk.

### Chunk size: `prefill_chunk(done, rows, base, scores)`

Two limits shrink the chunk. Start from `base` and halve:

| Constant | Value | Rule | Why (from the code comments) |
|---|---|---|---|
| `PREFILL_SCORES` | `1 << 24` | Halve while `c > 128` and `rows * c * (done + c) > PREFILL_SCORES` | A chunk of c tokens after n cached ones builds `rows x c x (n + c)` attention scores per head (2^24 fp16 scores is 0.5 GB on Qwen3.5-4B's 16 heads). The chunk is 2048 tokens at the start of a prompt, 256 at 48k and 128 beyond 100k. Without the bound, a 47k-token prompt's last chunks needed 3.5 GB above the model, and a 64k one did not fit on 16 GB. |
| `PREFILL_TOKENS` | `2048` | Halve while `c > 16` and `rows * c > PREFILL_TOKENS` | Activations grow with the total tokens in a forward: about 1 GB at 2048 and 4.3 GB at 8 x 1024. At 16 x 1024 the GPU ran out of memory when sixteen requests arrived together. The prefill rate is the same from about 1k tokens up (750 tok/s for 1 x 2048, 8 x 256, 16 x 128 or 32 x 64), so capping the total costs nothing. |

The score rule stops halving at 128. Beyond that point it no longer bounds the scores.

A model whose prompt attention holds no chunk's scores can say so: `fuse.fuse_model` sets Flint's `prefill_scores` (2^26, used in place of `PREFILL_SCORES`) and `prefill_chunk_tokens` (2048, the base of `_prefill_batch`), which the engine reads at construction (`_prefill_scores`, `_prefill_base`). Flint's full layers use MLX's fused causal attention and its window layers attend in bands, and 2,048-token chunks read 5% faster than 1,024 (docs/runtime, "Flint").

### `_reserve(cache, upto)`

This grows every attention layer's KV buffer to `upto` positions now, one layer at a time, evaluating each layer before moving on. mlx-lm would otherwise grow the buffers by concatenation as chunks arrive, with all layers in one evaluated graph. A model with attention in every layer then briefly holds its whole KV cache twice: "on a 16 GB Mac [that] ran a 21k-token prompt out of GPU memory".

Two more helpers follow the same one-layer-at-a-time pattern:

- `_grow_one_layer_at_a_time` patches `update_and_fetch` on `BatchKVCache` and `KVCache` so that a buffer of 4096 or more positions is evaluated as soon as it grows.
- `_filter` and `_append_rows` evaluate each layer as they go; `_append_rows` also releases each layer of the rows it appended.

## Mixed passes (`locals/mixed.py`)

### Why

Prefilling a new prompt on its own stops every running row for the whole prefill. On a 4B model on an M2 Pro, a prefill takes 1.5 s for 1k tokens and 12 s for 8k (the module docstring of mixed.py; the figure 1.47 does not appear anywhere in the repo).

Running the prefill on a second MLX stream does not help: the GPU does not run a second stream's work alongside the first (measured: two streams take as long as one after the other). The code gives no ratio; [Measurements](../measurements/README.md) gives "0.98x to 0.99x". Instead, `mixed.forward` computes the running rows' next tokens in the same forward pass as a chunk of the new prompt. The rows keep decoding while the prompt is prefilled, for little more than the prefill's own cost.

### `forward(body, xd, cache_d, xp, cache_p)`

This takes the decode tokens `xd` `(B, 1)` with the batch cache, and the prompt tokens `xp` `(1, S)` with that prompt's single-sequence cache. The two groups' embeddings are joined into one `(1, B + S, D)` sequence. In every layer, the input projection (`in_proj_all` or `qkv_proj`), the output projection and the MLP run once over all `B + S` tokens, so their weights are read once. Only the per-sequence part, `mixer._mix` (attention over each group's cache, the Gated DeltaNet recurrence), runs separately for each group, with its own cache and mask. The function returns the decode rows' normed hidden states.

### `supported(body)`

Mixed passes need Qwen3.5-style layers (module `qwen3_5` or `qwen3_next`) whose mixers are the fused `FusedGatedDeltaNet` and `FusedGatedAttention` from `locals/fuse.py`, plus `fa_idx`, `ssm_idx`, `embed_tokens` and `norm` on the body. The result is computed once per engine:

```
_mix_ok = mix_chunk > 0 and supported(...)
```

### Filling entries

Each prompt being prefilled this way is a dict in `filling`:

| Key | Meaning |
|---|---|
| `r` | The request |
| `cache` | Its single-sequence cache: the prompt-cache prefix, or a new cache |
| `pos` | Where the next chunk starts |
| `end` | `len(prompt) - 1` |
| `reserved` | KV positions already reserved |
| `cps` | Sorted `(position, kind)` checkpoints that lie strictly between `pos` and `end` |

### `_launch`

`_launch` picks the entry with the fewest tokens left among those with `pos < end`. If there is none, it runs a plain decode step (`_launch1`). Only one prompt advances per step.

Otherwise it does the following:

- Reserves KV up to `stop` (the next checkpoint, or else `end`) once the filling cache's attention layers already hold keys. A fresh cache (no cached prefix) is not reserved before its first chunk, only from the second pass on.
- Takes a chunk of `n = max(1, min(mix_chunk - B, stop - pos, prefill_chunk(pos, 1, mix_chunk)))` tokens.
- Runs `forward` and samples the decode rows from its output.
- Puts the filling cache's arrays in `_fill_eval`, so they are evaluated with the step (`mx.async_eval(cur, g, self._fill_eval)`).

If the chunk reaches a checkpoint, the step is evaluated synchronously and `_checkpoint` stores the copy. Chunks never cross a checkpoint, so the copy is taken at exactly that position.

**Tile alignment.** The first term, `mix_chunk - B`, makes decode rows plus prompt tokens add up to exactly `mix_chunk`. The code comment explains why this matters: "one row past a multiple of 32 or 64 costs a whole tile more, more than the decode step the pass saves". The default of 1024 is a multiple of both. A chunk cut short by a checkpoint, by the prompt's end, or by the score bound is not aligned.

**`LOCALS_MIX_CHUNK`** (default 1024) sets `mix_chunk`; 0 turns mixed passes off. The comment at its definition says that at 1024 a mixed pass is "as fast as a prefill on its own", and the rows get a token every pass (1.5 s on a 4B model on an M2 Pro) instead of none for the whole prefill (12 s for 8k tokens).

**The one-pass threshold.** An admission whose remaining tokens fit in one pass (at most `mix_chunk`) is prefilled at once. The comment gives two reasons: the rows would get only a single token in the meantime, and one batch prefills several prompts faster than passes run one after another.

### Joining and leaving

- **Joining.** Any complete entry forces a boundary, and every complete entry joins there, wherever it sits in the list. It passes them through `_prefill_batch(..., starts=[(cache, [])])`, which only merges them (a single entry is wrapped without copying), and calls `attach(..., count=False)`. Mixed passes already counted their tokens in `prefill_tokens`.
- **No rows left.** If every row finishes while prompts are still in `filling`, the rest is prefilled at once: `_fill_checkpoints`, then `_prefill_batch` with each entry's remaining tokens.
- **Cancellation.** A `filling` entry whose request is cancelled forces a boundary. There the entry is removed and finished as `"cancelled"`, and `_fill_eval` is cleared.

`LOCALS_MIXLOG=1` prints each pass: the number of rows, the prompt tokens in the pass, the position, and the milliseconds since the previous pass.

## One request alone (`_solo`)

With a single row, a pass over a few tokens costs little more than a pass over one: the weights are read once
either way, and that is what limits one stream (about 66 tokens/s for Qwen3.5-4B on an M2 Pro, near the
71 the memory bandwidth allows). So the only running request gets steps that feed more than its next token,
and every token they keep is one a one-token step would have produced:

- **Its forced span at once.** Tokens the grammar decides (tool-call syntax, JSON keys) are fed in one pass
  instead of one per step, and only the last position is sampled.
- **A prompt-lookup draft, verified in the same pass.** `_solo_draft` takes what followed the most recent
  earlier occurrence of the row's last 3 tokens (from an index kept on the request, `lk_index`, updated as
  tokens are committed, so no scan of a long prompt per step). Under a grammar the draft is first walked
  through it (`config`, `step_config`, `finished_at`) and cut at the first token it does not allow, and each
  draft position is masked with the grammar's state there (`_solo_masked`). `_solo_verify` keeps draft tokens
  while each is the argmax (greedy), or with probability p(d) under the request's own distribution, drawing
  from p without d on a rejection (sampled: the tokens follow p exactly). Rejected tokens are rewound
  (`_solo_rewind`: KV layers `trim`, Gated DeltaNet layers the per-token state the fused kernel recorded, so
  drafts need `_gdn_ok` layers and float32 states). A forced span after an accepted draft token ends the
  draft there; the span is fed next.

**When.** At a boundary, when one row runs, nothing waits and no prompt is filling (`_solo_ok`: no walk or
pushdown constraints). Sliding-window caches (Gemma, Flint) and 8-bit ones (`qkv.py`) are not rewound, so those
models get forced spans in one pass and no drafts (`_solo_drafts_ok`). After a chained step, `solo_next` asks for a
boundary when the only row has a forced span of 3 or more tokens or a draft: entering costs about one step,
since the pipeline drains. Without a span or a draft, `_solo` returns and the row goes on in pipelined
steps. It also returns as soon as `source` yields a request, which then joins as usual.

**Drafts that keep failing.** After a draft rejected outright, the row gets no draft for 1 token, doubling to
32 while drafts keep failing (`solo_miss`, `solo_resume`); one token under a grammar is too often the start
of a repeated JSON key, so drafts there need 2 tokens or more.

Measured through the server, one request, Qwen3.5-4B, greedy, the same answers with `LOCALS_SOLO=0` and 1:

| Workload | One-token steps | Alone |
|---|---|---|
| An edit that repeats a 70-line file | 66 tok/s | 108 tok/s (1.64x) |
| A `write` tool call that saves that file (the agents' tool-call grammar) | 54 tok/s | 91 tok/s (1.70x) |
| A short tool call | 37 tok/s | 38 tok/s |
| A JSON extraction (a list of objects) | 53 tok/s | 52 tok/s |

`LOCALS_SOLO=0` turns it off.

## Retiring rows

At a boundary, `retire(i, r)` moves a finished row's cache into the prompt cache (if there is one) and calls `on_done(r)`. How the row is taken out depends on whether it is the only row:

- **The only row in the batch.** The batch cache is about to be dropped, so `_only_row` hands its buffers to the prompt cache as views, without a copy. A copy would hold the KV cache twice.
- **Otherwise.** `_trim_for` first makes room for the row's copy plus one attention layer of the padded batch, plus 512 MB. Then `c.extract(i)` takes the row out. The comment explains the layer: when the finished rows are filtered out, one attention layer of the batch cache is copied once more, padded to the longest row.

In both cases, the row's `ArraysCache` arrays are copied (`a + 0`), because "a row view would pin the whole batch buffer".

The row is stored under `prompt[:-1] + fed`, exactly the tokens its cache holds. When the request kept a `"reply"` state and `_reply_entry` can build an entry, a different entry is stored instead. `_reply_entry` needs every layer to be either a kept recurrent state or a plain `KVCache` holding at least the reply's start. Otherwise it returns None (for example for a rotating window), and the whole row is stored as usual: the kept recurrent state plus the row's KV cut to the reply's start (`_reply_entry`), under `prompt[:reply_state[0]]` with type `"assistant"`. The template writes an earlier reply differently from the open turn, so the whole row is no prefix of the next request.

After all dead rows are retired, `_filter(cache, keep)` keeps the survivors one layer at a time, evaluating each layer. Without this, every layer's old and new copies would coexist until the next step was evaluated.

The prompt cache is `chain_cache(...)`, an mlx-lm `LRUPromptCache`. A new entry drops the shorter entries it extends that cannot be wanted again: earlier `"assistant"` ends, and earlier `"user"` checkpoints when the new entry is itself a `"user"` one. It never drops `"system"` entries. This applies only to caches that cannot be trimmed.

## Memory guard

A Metal out-of-memory error fails every request in flight, while a prompt-cache miss only costs a prefill (`_make_room` docstring). The guard therefore spends prompt-cache entries to keep running rows safe.

| Piece | What it does |
|---|---|
| `memory_limit` | GPU bytes the engine may use. The engine default is 0 (no guard). `BatchService` sets it to `max_recommended_working_set_size` minus a margin (`service.memory_plan`, or `LOCALS_MEMORY_MARGIN_GB`): MLX's pool of freed buffers (`mx.set_cache_limit`: 1 GiB, or an eighth of what the weights leave when that is under 8 GiB, at least 256 MiB: 0.43 GiB for a 27B on 24 GB) and 0.5 GiB for what the admission estimate misses. A 4B on 16 GB keeps 1.5 GiB. It was 2.5 GiB while the estimate left out each request's fixed state and kept 1.5 GiB for reads that take up to 2.1 GB; with those counted, 2.5 made room by emptying the prompt cache (four agents at once: 71-88% of prompts from it against 87-95%). With 1.5 GiB on a 16 GB Mac, 16 x 2k, 8 x 8k and 4 x 30k-token prompts at once all finished, at most 9.06 of 11.45 GB in use. |
| `_kv_bytes_per_token()` | KV bytes one token adds: the number of layers whose `make_prompt_cache` entry is a `KVCache` × the keys-plus-values width × 2 bytes. Sliding-window `RotatingKVCache` layers are not `KVCache` instances, so they are not counted. The width is the larger of `2 * kv_heads * head_dim` from the config and the widest `k_proj` + `v_proj`; the comment cites Gemma 4, where the config's figure was half the real width. Recurrent state is fixed in size and not counted. |
| `_trim_for(lru, need)` | Makes `need` more bytes fit under the limit, in this order: `gc.collect()` and `mx.clear_cache()`; then clear the shared-prefix snapshots; then `lru.trim_to(n_bytes=max(0, lru.nbytes - over))`, only if the prompt cache is not empty. mlx-lm's `CacheOrder.pop` evicts the oldest entry of the first kind, in the order `"assistant"`, `"user"`, `"system"`, that has at least as many entries as the next kind. This is a preference balanced by entry counts, not a strict order. With 1 reply and 3 "user" entries, a "user" entry goes first. With no replies, 1 "user" and 2 "system" entries, a "system" entry goes first. Increments the counter `cache_trims`. |
| `_make_room(lru, tokens, copies, held, rows=1)` | Calls `_trim_for` for `(tokens * (1 + copies) + held) * kv + _admit_bytes(rows, tokens)` before an admission: each new row's fixed state and 256 positions of KV, and the room a read of that size took, plus a sixth (1.5 GiB, `PREFILL_RESERVE`, until one has been recorded). |
| `_fits` | The admission check described under "Admission". |
| `_room_for_copy(lru, positions)` | Before a checkpoint copy: trims, then skips the copy if `positions * kv + _admit_bytes(0, PREFILL_TOKENS)` still does not fit. The copy shares the request's KV buffers, so the request's next write duplicates them. |
| Growth trim | Every 32 steps (`_growth_trim`: boundary steps, chained steps and a lone row's passes, which count 64 positions), calls `_trim_for(lru, len(rows) * 32 * kv)`. The rows' KV grows past what admission counted, so room for the next 32 steps comes out of the prompt cache in advance. `LOCALS_GROWTH_TRIM=0` turns this off (default `"1"`). |
| Retire trim | See "Retiring rows". |

`LOCALS_MEMLOG=1` prints active and peak GPU memory, prompt-cache size and the limit around each admission, reservation and trim. `LOCALS_PREFILL_LOG=1` prints each batch prefill (rows, prompt lengths, tokens to prefill) and each checkpoint.

## Sampling for each row

`BatchService` copies `top_p` and `top_k` onto the engine. The service's `temp` is passed to `serve`. A request's own `temp`, `top_p`, `top_k` and `seed` go onto its row (`Request.temp`, `.top_p`, `.top_k`, `.seed`) and override the engine's.

`_pick` first applies the constraint mask, then calls `_sample`:

- If no row has settings of its own, every row is sampled by `sample_logits` with the engine's.
- Otherwise `_specs` gives each row (temperature, top_p, top_k, key) and `sample_each` samples every row in one pass: the filters' mask for each row's own top_p and top_k from one sort of the vocabulary (`_keep`), then one draw, the Gumbel maximum at each row's temperature, where a seeded row's noise comes from its key. The key is `row_key(seed, n)` for the row's n-th draw (`Request.draws`), so a seeded request's draws do not depend on the other rows or on when it came in. The same request alone gives the same reply; next to other requests, a batch of another size can run other kernels, and a near tie between two tokens can go the other way.
- `sample_logits` takes the argmax at temperature 0. Otherwise it follows mlx-lm's rule: top-p on the untempered distribution, then top-k, then a categorical draw at `temp`. A sampled run of Locals and one of stock mlx-lm therefore follow the same distribution. The filters keep what mlx-lm's keep (a float32 sum taken in another order can move a top-p cut by one token), in one sort instead of mlx-lm's two: 3.6 ms instead of 5.3 ms a step for 8 rows of a 248k vocabulary on an M2 Max. With 8 requests at once that ask for top_p 0.95 and top_k 20, Qwen3.5-4B decodes 225 tokens/s against 247 for greedy ones (before, the server ignored the requests' filters: 243).
- The single-row speculative path (`_solo_verify`) uses the row's own top_p, top_k and seed too.
- The stock engine gives a request with settings of its own a sampler of its own (`service._job_sampler`); with a seed, its draws come from `row_key` as well.

## Constrained rows in the batch

All constraints in the batch share one combined table, so every row is masked with a single gather:

- `self._tab = (allowed, cover, tlen, offs)`
- Global state 0 is a free state (all tokens allowed) used by unconstrained rows.
- Each distinct constraint (`id(c.c)`) gets an offset, and a row's global state is `offset + q`.
- The mask is `allowed[g] & (tlen >= cover[g])`. The second term is the canonical boundary rule: at a state whose remaining forced bytes form one pre-token block, the next token must cover the whole block.
- `budget_mask` adds a further mask for each `LocalsController` row given its tokens left, allowing only tokens after which a valid completion still fits.

The row states `g` live on the device. `_advance` moves them by the fed token through each constraint's transition table (`_gpu_T()`); free rows stay at 0. During a forced span, the device state follows the fed tokens one at a time, while the host controller has already moved to the span's end. The rows are not live during that time, so their masked samples are discarded anyway.

`_table_add` only adds constraints, so known offsets stay valid. A state is a row over the whole vocabulary (262k tokens on Gemma 4), and with agents keeping the engine busy the table grew until Metal ran out of memory. So when a call brings a constraint that is not yet in the table, and more than half of the table's states (the free state included in the total) belong to constraints with no running or incoming row, `_table_add` rebuilds the table from the live ones and returns the old offsets. A call with no new constraint never rebuilds. `attach` then shifts the running rows' `g`. When the loop goes idle, all tables are reset.

With `BatchService(walk=True)`, `WalkController` rows use `_walk_add` instead. That table stores automaton states with 256 entries each, and a kernel recomputes each step's `(B, V)` rows. A batch cannot mix walk and table controllers. Pushdown (`PDAController`) constraints raise `NotImplementedError` in the batch engine. In table mode (`walk=False`), `service.py` rejects recursive schemas as invalid before they reach the engine. The walk path (`rt.compile_walk`) has no such check.

## Cancellation

`Job.cancel()` sets `r.cancel`. How quickly it takes effect depends on where the request is:

- **Waiting in the queue.** Removed at the next boundary, finished as `"cancelled"`, never prefilled. The comment notes that "a 50k-token prompt held every row up".
- **Being prefilled in mixed passes.** Forces a boundary, where it is removed (see "Mixed passes").
- **Decoding.** `_after_commit` marks the row dead at its next live commit, and it is retired at the following boundary.

If a caller's `on_token` callback raises, only that row ends, with `finish = "error: on_token: <ExceptionType>: <message>"`.

## Engine errors and recovery

Some faults belong to one request: a regex or schema that does not compile, or a recursive schema. `source` catches these and finishes that job as `"invalid: ..."`, so they never reach `serve`.

Any other exception escapes `serve`, for example a Metal out-of-memory error. `BatchService._run_locals` catches it and then:

1. prints the traceback
2. fails every job taken from the queue (`_inflight`, whether or not it reached the batch) with `f"error: {e!r}"` (the exception's repr)
3. clears the request map and `eng._fill_eval` (a prompt being prefilled in mixed passes loses its cache too)
4. trims the prompt cache to half its bytes
5. runs `gc.collect()` and `mx.clear_cache()`
6. calls `serve` again

`serve` starts from fresh local state (rows, cache, queue, filling) and resets its tables. An exception outside `serve` after loading is handled by `_main`, which fails every in-flight and later job with that error.

## Settings

| Setting | Default | Where |
|---|---|---|
| `n_slots` | 16 | `HybridBatchEngine(...)`, `BatchService(slots=...)`, server `--slots` |
| `share_prefix`, `min_share`, `tail_ok` | `True`, 64, 32 | `HybridBatchEngine(...)` (`tail_ok` is used only by the ragged `run` path and by `_run_sync`, both through `_prefill`) |
| `LOCALS_MIX_CHUNK` | 1024 | Rows plus prompt tokens in a mixed pass; 0 turns mixed passes off |
| `LOCALS_MAX_SNAPSHOTS` | 4 | Shared-prefix snapshots kept |
| `LOCALS_GROWTH_TRIM` | `1` | `0` turns off the trim every 32 steps |
| `LOCALS_SOLO` | `1` | `0`: one token per step for a request alone too (no forced spans in one pass, no drafts) |
| `LOCALS_MEMLOG`, `LOCALS_MIXLOG`, `LOCALS_PREFILL_LOG` | unset (off) | Logging, described above. Any non-empty value turns them on, `0` included, because the code tests `bool(os.environ.get(...))` |
| `PREFILL_SCORES`, `PREFILL_TOKENS` | `1 << 24`, 2048 | Module constants in `hybrid_batch.py` (a model's `prefill_scores` and `prefill_chunk_tokens` take precedence) |
| `memory_limit` | 0 in the engine | Set by `BatchService` as above |

## Counters

`self.counters`, also exposed by `BatchService.stats()`. `serve` creates a new dict (and resets `tokens_out`) every time it starts. The counters therefore restart from zero after an engine error, and after any `source(True)` call that yields no request (a prime or an invalid job while idle). `mixed_steps`, `deferred` and `cache_trims` appear only once they have been incremented:

| Counter | Meaning |
|---|---|
| `steps`, `rows` | Steps launched, and the row-steps they contained |
| `prefill_s` | Wall time from the start of each admission or join to the end of its `attach`. It includes prompt-cache lookup, disk restore, trims and checkpoint copies. Mixed passes are not counted; only the join that merges them is. |
| `prefill_tokens` | Prompt tokens prefilled |
| `admissions` | Number of `attach` calls |
| `idle_s` | Time spent blocked in `source(True)` |
| `mixed_steps` | Mixed passes run |
| `deferred` | Requests held back for memory (each counted once, however many steps it waits) |
| `cache_trims` | Times `_trim_for` evicted prompt-cache entries |
| `solo_steps`, `drafted` | Steps of a request alone (`_solo`), and the draft tokens they kept |

The service wraps `eng._launch` to publish `active` and `tokens_out` on every step.
