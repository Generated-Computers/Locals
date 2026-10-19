# Prompt cache and memory

Part of the [technical documentation](../README.md) of Locals. File and function names refer to the `locals` package.

Agents resend the whole conversation on every call. The engine keeps the model's cache of earlier prompts in GPU memory, so each request only prefills what it adds. This section covers four things: what gets kept, where a kept prefix starts and ends, what gets dropped first when memory runs short, and how the context limit and the cache budget are worked out from the Mac's GPU memory.

## The pieces

| Piece | Code | What it holds |
|---|---|---|
| Prompt cache | `chain_cache()` returns `ChainCache`, a subclass of mlx-lm's `LRUPromptCache` (`hybrid_batch.py`). It is created in `BatchService._run_locals` with `max_size=256` and `max_bytes=cache_bytes`. | Model caches keyed by token sequence. Each entry has a type: `system`, `user` or `assistant`. |
| Checkpoints | `Server._checkpoints` picks the positions. `HybridBatchEngine.serve` and `_checkpoint` take the copies. | Snapshots of a prompt's cache at chosen positions inside it. |
| Reply state | `_keep_reply_state`, `_reply_entry` (`hybrid_batch.py`) | The recurrent state where a reply starts, joined at retire with KV views of the finished row. |
| Disk cache | `DiskPromptCache` (`hybrid_batch.py`) | System-prompt snapshots on the SSD, kept across restarts. |
| Shared-prefix snapshots | `HybridBatchEngine.snapshots`, at most `LOCALS_MAX_SNAPSHOTS` (default 4) | Prefixes shared by requests that arrive together. They are the first thing dropped when memory is short. |
| Memory guard | `memory_limit`, `_fits`, `_make_room`, `_trim_for`, `_room_for_copy` | `_fits` holds back an admission that would not fit even with an empty prompt cache. The other three drop cache entries, or skip a checkpoint copy, before an allocation that would not fit. |
| Token streams | `Server._streams`, `_tokens_for`, `_remember` (`server.py`) | Exact token ids of recent prompts and replies. |

## Why some caches cannot be cut back to a prefix

A plain attention layer (`KVCache`) stores one key and one value per position. To cut it back to a shorter prefix, mlx-lm only lowers its `offset` (`KVCache.trim`), and `is_trimmable()` returns `True`. Two other layer kinds cannot be cut back:

- **Recurrent layers (Qwen3.5's Gated DeltaNet).** `qwen3_5.py` gives each linear layer an `ArraysCache(size=2)`. `cache[0]` is a convolution window over the last inputs and `cache[1]` is the recurrent state. Both are overwritten on every token. The state after n tokens has all n folded into it, so the state at any earlier position is gone. `ArraysCache` does not override `_BaseCache.is_trimmable()`, which returns `False`. On Qwen3.5-4B, 8 of the 32 layers are full attention and the other 24 are recurrent.
- **Sliding windows (Gemma).** The layers between full-attention ones use `RotatingKVCache(max_size=sliding_window)`. Once the window wraps, older positions are overwritten. `is_trimmable()` is `offset < max_size`.

`can_trim_prompt_cache` requires every layer to be trimmable, so one such layer makes the whole entry untrimmable. That has three consequences:

1. **Only exact or shorter entries are usable.** `LRUPromptCache.fetch_nearest_cache` can use an entry longer than the new prompt only by trimming a copy of it. For these models it falls back to an exact match or the longest stored prefix of the prompt. A cached conversation that differs from the new prompt anywhere before its end is of no use.
2. **Prefixes are stored on purpose.** The engine stores entries at the positions the next requests are likely to share. These are the checkpoints, described below.
3. **Stale prefixes are not removed.** mlx-lm's `insert_cache` drops the shorter entries a new entry extends only when the cache is trimmable. For a hybrid model, every agent step would leave behind another copy of the whole history. `ChainCache` fixes this (see "Eviction" below).

`Server._needs_checkpoints(engine)` turns checkpoints on only if some layer of `make_prompt_cache(model)` is not exactly a `KVCache`. A plain KV cache is already trimmed to any shared prefix by the prompt cache. Checkpoints of it would just hold the prompt's KV two more times; the docstring's example is 5 GB for a 20k-token prompt on an 8B model.

## How a request finds its cache and leaves one behind

In `HybridBatchEngine.serve`:

1. **Key.** The key is `lru_key` (`"locals"`) and the tokens are `prompt[:-1]`. The last prompt token is fed by the first decode step.
2. **Lookup.** For each new request, `self.disk.restore(...)` runs first, then `lru.fetch_nearest_cache(...)`. The lookup returns a deep copy of the entry and the tokens still to prefill, and records `r.cached_tokens`. All new requests look up their prefixes before any room is made, so making room for one request cannot drop the next one's prefix.
3. **Prefill.** `_make_room` drops entries until the admission fits. The prompt is then prefilled, and checkpoints are taken on the way.
4. **Retire.** When a request finishes, `retire(i, r)` turns its row of the batch cache into an entry:
   - If it is the only row, its buffers move to the prompt cache as they are (`_only_row`). The batch cache is dropped next, and a copy would hold the KV twice.
   - Otherwise the engine calls `_trim_for` and then `c.extract(i)`. In both cases the row's recurrent arrays are then copied with `a + 0`, because a view would pin the whole batch buffer.
   - The row is stored as an `assistant` entry under `prompt[:-1] + r.fed`, or the reply entry is stored instead (see below).

**Token streams.** The server re-renders each prompt from the client's messages, so the token ids have to stay stable. Encoding the rendered text from scratch may not give back the ids the model actually produced, and the lookup would then miss. `_remember` keeps up to 64 recent `{text, ids}` pairs for replies that came back exactly. `_tokens_for` reuses the ids of the longest remembered text that the new prompt starts with, and encodes only the rest.

## Checkpoints

`Server.prepare` computes `cps = self._checkpoints(...)` only for the `locals` engine, and only when `_needs_checkpoints` is true. The result is a list of `(n, kind)` pairs. It reaches the engine through `BatchService.submit(checkpoints=...)` as `Request.checkpoints`. Each position comes from rendering part of the conversation with the model's own chat template:

| Kind | Position | How it is rendered | Entry type | Why |
|---|---|---|---|---|
| `system` | The start of the line where the session's own part of the system message begins (`_session_start`), else the end of the system message (tools included) plus the user-turn header | Used only when there are at least two messages and the first is a system message. The end: render `[system, user "\x01A"]` and `[system, user "\x02B"]` and take the point where they differ (a template may refuse a conversation with no user message; Qwen3.5 does). The session's part: the first of the markers agents write there (`<env>`, `Working directory:`, `Today's date:` and the like, `_SESSION_MARKERS`), else where the system message first differs from a recent session's (the last 16 kept). When that point is within 4096 characters of the end, the end is not taken as well. | `system`, also saved to disk | Every session of an agent starts with it. Agents end it with the project's folder and the date: OpenCode's 7,222-token prompt has its folder at token 7,157, so the end is reached only by sessions in the same folder on the same day, and a new project's first request read the whole prompt again (about 7 s on an M2 Pro). |
| `user` | Start of the last message | Render `msgs[:-1]` with no generation prompt. Used only with more than two messages, or with two when the first is not a system message. | `user` | The history before the last message stays put. This holds even for clients such as OpenClaw that move a context message to the end on every turn. |
| `reply` | Where this request's reply starts, as the next request will render it | Render `msgs + [assistant "\x01A", user "\x03"]` and the same with `"\x02B"`, and take the point where they differ | `assistant` | See "The reply checkpoint". |

**Per-session text in the middle.** Some agents write what changes from session to session before parts that do
not: OpenHands puts the date and time, to the microsecond, 36% into its 39,000-character system message, and the
project's folder into its file editor's description (tools come first in Qwen3.5's template). `_volatile_to_end`
moves such blocks (`_VOLATILE`, `_TOOL_VOLATILE`) to the end of the system message, where the model still reads
them. Two OpenHands sessions in different folders then share 13,714 of their 13,795 prompt tokens; they shared 1,555.

**Which candidates are kept.** A candidate is skipped if its text is shorter than 2048 characters ("a short prefix is cheaper to prefill again") or if the prompt text does not start with it. The candidate text is then encoded. Its position is the first index where those ids and the prompt's ids differ, because the last token may merge with what follows; this is computed in one pass. A position is kept only if it is at least 1024 tokens, lies before the prompt's end, and is not already taken.

**How the engine takes them.** Checkpoints between the cached prefix and the prompt's end (`done < cp < len(r.prompt) - 1`) are taken in sorted order:

1. Prefill `prompt[done:cp]` with `self._run`.
2. For a `system` or `user` checkpoint, insert `copy.deepcopy(c0)` under `prompt[:cp]` with that type, if `_room_for_copy` allows it. For a `reply` checkpoint, call `_keep_reply` instead. A `system` checkpoint is also written to disk.

When a prompt is prefilled in mixed passes next to decoding rows (`mix_chunk`, `locals/mixed.py`), each chunk stops at the next checkpoint and `_checkpoint` does the same work there, with one difference. When `_room_for_copy` refuses the copy, `_checkpoint` returns before the disk save, so that `system` snapshot is not written to disk. The non-mixed path writes a `system` snapshot to disk even when it skipped the in-memory copy. If the rows finish first, `_fill_checkpoints` runs the remaining checkpoints at once.

**Memory cost.** `_fits` counts each `system` or `user` checkpoint (`_copies(r)`) as one more full copy of the prompt's KV (`len(r.prompt) * (1 + copies)`). `_make_room` counts each one only as a copy of the uncached part (`len(rest) * _copies(r)`). Before each copy, `_room_for_copy` asks again for the whole prompt's KV plus 1.5 GiB. Each `reply` checkpoint (`_replies(r)`) counts as `_reply_bytes`, the largest reply state kept so far.

## The reply checkpoint

**The problem.** A chat template writes an earlier assistant turn differently from the open turn that produced it; Qwen3.5 drops the empty think block. So request k's prompt plus its reply is not a prefix of request k+1. With a cache that cannot be cut back, the next turn fell back to the `user` checkpoint and read the last question and the whole reply again on every turn. The comment's example: a follow-up question on a 60k-token document took 137 s.

**Qwen3.5 with thinking off.** Its template writes the think block of an assistant turn only while the turn is the current one, and drops it from every earlier reply once the user writes again. So at each new user message, the conversation stopped being a prefix of what was cached at the previous turn's first reply: all of that turn's tool calls and results were read again (in an OpenCode session, 1,100 tokens after the first turn, 1,950 after the second, growing with the session). `_render` therefore renders with a variant of such a template (`_stable_template`: the condition `loop.index0 > ns.last_query_index` also holds for a reply without reasoning), so the empty block stays where the reply was generated after it. Replies with reasoning are written as the template writes them. The next turn's first request is then a continuation of the last one: 5,646 of its 5,673 tokens came from the cache, where 4,534 did before. `LOCALS_KEEP_THINK=0` turns the variant off.

**The position.** The `reply` candidate ends where the two renderings differ: where the reply's content starts in the next request's rendering. The open prompt continues past that point (for example, with the empty think block), which is why the position falls inside the prompt. A template that writes both turns the same way gives a position at the prompt's end. The `n < len(prompt)` rule drops it, and the finished row is then a prefix of the next request as it stands.

**What is kept.** The copy is small, so `_room_for_copy` is not consulted:

- **At prefill time,** `_keep_reply_state(r, n, cache)` stores `r.reply_state = (n, kept)`. `kept` holds deep copies of every layer that is not a `KVCache` or `BatchKVCache`: the recurrent state ("a few MB per layer"), and a sliding-window layer too. It holds `None` for each attention layer. `_keep_reply` records the largest such state in `_reply_bytes`.
- **At retire,** `_reply_entry(r, row)` builds the entry. Each kept copy is used as it is. Each attention layer becomes a new `KVCache` whose keys and values are views `[..., :n, :]` of the finished row's buffers, with no copy. It returns `None` if a layer is neither kind, or is an attention layer that has not reached `n`.
- **The entry replaces the row.** `retire` inserts the reply entry under `prompt[:n]` as `assistant`. The comment gives two reasons: the row is no prefix of the next request, and the views would count twice in the prompt cache's bytes next to the row. Without a reply entry, the whole row is inserted as usual.
- **Regenerated replies.** If the cached prefix found for a request ends exactly at its `reply` position, the state is taken straight from that cached prefix.

**Measured effect** (the 60k numbers are from the comments in service.py:322 and server.py:601; the others are from the message of commit 76f5502; none of these sources names the model):

| Case | Without | With |
|---|---|---|
| Follow-up question on a 60k-token document | 137 s | 1.5 s |
| Follow-up question on a 138k-token document | 8 minutes | 2.4 s |
| Prompts cached, four multi-turn conversations | 28% | 83% |

`LOCALS_REPLY_CHECKPOINT=0` turns it off. The automatic cache budget is at least `context_fit() * kv` (one conversation as long as fits) but never more than 16 GiB, so that a follow-up can continue from it (see "Memory sizing").

## Eviction

**mlx-lm's order (`LRUPromptCache`, mlx-lm 0.32.0).** Entries live in a trie keyed by token sequence. Each type has its own deque in insertion order, and the types are `assistant`, `user`, `system`.

- **Insertion order, not use order.** Inserting the same tokens again moves the entry to the back. `fetch_nearest_cache` does not change the order, so within a type the oldest inserted entry goes first, not the least recently used one.
- **Which type loses an entry (`CacheOrder.pop`).** The types are checked in the order `assistant`, `user`, `system`. A type gives up its oldest entry if it has at least as many entries as the next type; otherwise the check moves on. `system` is the fallback. This balances the types rather than giving strict priority. For example, with 1 `assistant`, 3 `user` and 2 `system` entries, a `user` entry goes first.
- **Bounds.** After an insert, at most one entry is evicted to stay within `max_size` (256). Then entries are evicted until the bytes are within `max_bytes`. `trim_to(n_bytes=...)` uses the same order.

**ChainCache's prefix rules.** These apply after mlx-lm's own insert, and only for an untrimmable entry that is not `system`:

- First it checks that the new entry survived the insert's own eviction.
- It then pops every shorter entry on the new entry's path (`pop_prefixes`) and drops it if:
  - it is an `assistant` entry: the conversation has moved past that reply; or
  - it is a `user` entry and the new entry is `user` too: it was the previous request's checkpoint.
- Every other prefix is put back in the trie, with its eviction position unchanged.
- A `system` entry is never dropped. A finished reply (`assistant`) does not drop its own request's `user` checkpoint, because a client that moves a message to the end on every turn (OpenClaw) continues from that checkpoint, not from the reply.
- Inserting a `system` entry drops nothing. For trimmable caches, mlx-lm already drops all prefixes.

The docstring gives the reason: with Qwen3.5, every agent step left another copy of the agent's whole history (9k tokens, 290 MB). The cache filled with them, and memory trims emptied it.

**The memory guard.** `_trim_for(lru, need)` acts when active memory plus `need` exceeds `memory_limit`. It escalates in this order:

1. `gc.collect()` and `mx.clear_cache()`.
2. Clear the shared-prefix snapshots.
3. `lru.trim_to(n_bytes=lru.nbytes - over)`, in the eviction order above. This counts `cache_trims`.

It is called:

- before each admission (`_make_room`);
- before each checkpoint copy (`_room_for_copy`);
- at retire when other rows are running;
- every 32 decode steps for the rows' growth (`LOCALS_GROWTH_TRIM`, on by default).

After an engine error such as a Metal out-of-memory error, the service halves the cache (`lru.trim_to(n_bytes=lru.nbytes // 2)`) and serves again.

## Disk cache of system prompts

**Why.** A restarted server should not prefill an agent's system prompt again. The docstring's numbers: Hermes' is 14k tokens and OpenClaw's 8.5k, which takes 10-30 s on a 4B model.

| Setting | Default | Effect |
|---|---|---|
| `LOCALS_DISK_CACHE` | `1` | `0` turns the disk cache off. |
| `LOCALS_DISK_CACHE_DIR` | `~/.cache/locals/prompts` | Root directory. |
| `LOCALS_DISK_CACHE_GB` | `4` | Size cap. After each save, the files with the oldest mtime are removed until the total is under the cap. |

**Directory.** Each model gets one folder under the root. Its name joins three parts with `--`:

- the model id with `/` replaced by `--` (its last 80 characters);
- the compute dtype;
- `_weights_fingerprint`: 12 hex digits of a SHA-1 over the mlx-lm version, the resolved model path, and the names, sizes and mtimes of its `.json` and `.safetensors` files.

On start, other folders for the same model and dtype are deleted. The reason given in the code: a saved cache of other weights would give wrong output, or fail the batch at its merge. The directory is mode `0700` and the files `0600`, because system prompts and tool definitions belong to this user only.

**Saving.** A snapshot is saved at each `system` checkpoint the engine takes. Each snapshot is two files: `<name>.safetensors`, written by `save_prompt_cache`, and `<name>.tokens.json`. The name is the first 20 hex digits of a SHA-1 of the token list. Both files are written under temporary names and renamed with `os.replace`, so a crash or a second server never leaves half a file.

**Restoring.** Before each new request's lookup, `restore` finds the longest saved token sequence that the prompt starts with and that is shorter than the prompt.

- It does nothing if memory already holds an exact or shorter entry at least that long.
- Otherwise it loads the file with `load_prompt_cache`, inserts it as `system`, and touches the file, so the size cap removes the least recently used files first.
- A file that fails to load is deleted.

## Copy-on-write and checkpoint copies near the memory limit

According to the code's comments, `copy.deepcopy` of a cache does not duplicate the KV buffers at once. This applies to a checkpoint copy and to the prefix returned by `fetch_nearest_cache`. The copy shares the buffers, and the request's next write makes MLX copy them. Three things follow:

- **Checkpoint copies.** A copy is free when it is made. As soon as the request goes on, the prompt's KV exists twice. `_room_for_copy(lru, len(r.prompt))` asks for the whole prompt's KV plus 1.5 GB. It first trims the prompt cache, then skips the copy if the request still would not fit under `memory_limit`. Near the context limit two copies of a prompt do not fit next to the model. A skipped copy only costs a later prefill, while a Metal out-of-memory error fails every request in flight.
- **Cached prefixes.** At admission, a request whose prefix came from the cache still counts its whole prompt ("each request's cached prefix is a lazy copy, made during the prefill").
- **Reply entries.** These use slices (views) of the finished row's buffers, so they make no copy. A view keeps its base buffer alive, so the entry holds on to the whole row's KV buffers, including the reply's positions past n. The prompt cache counts only the views' `nbytes`.

## Memory sizing

| Quantity | Code | Value |
|---|---|---|
| GPU limit | `mx.device_info()["max_recommended_working_set_size"]` | Depends on the Mac. |
| MLX pool of freed buffers | `mx.set_cache_limit(1 << 30)` | 1 GiB. By default MLX keeps up to the whole memory limit. |
| `memory_limit` | `limit - 1.5 GiB` | The limit minus 1.5 GiB (`LOCALS_MEMORY_MARGIN_GB`): 1 GiB for the pool, and 0.5 GiB for what the admission estimate misses, now that it counts each request's fixed state and what reading a prompt takes (see [Batch engine](../engine/README.md)). |
| `weights_bytes` | `mx.get_active_memory()` right after `Runtime.load` | |
| KV bytes per token | `_kv_bytes_per_token()` | (number of `KVCache` layers) × (2 × KV heads × `head_dim`, raised to the widest `k_proj` + `v_proj` output) × 2 bytes. The widest-projection rule exists because Gemma 4's full-attention layers have 512-wide heads where the config says 256. Recurrent state has a fixed size and is not counted. Qwen3.5-4B: 32 KB per token. |
| `context_fit()` | `(memory_limit - weights_bytes - (3 << 29)) // kv`, rounded down to a multiple of 1024 | The 1.5 GiB is a prefill's activations. Qwen3.5-4B on a 16 GB Mac: 145,408 tokens. |
| `max_context` | `--max-context`. Without it, 65,536 is raised to `context_fit()` once the engine loads (`_cap_context`), then capped at the model's own context length. | A prompt of `max_context` tokens or more gets `ContextTooLong`. So does one that leaves fewer than `min(max_tokens, 256)` tokens for the reply. The stock `mlx` engine keeps 65,536, because it has no memory guard. |
| `cache_bytes` | `--cache-gb`. Without it, `min(max(free, longest, 1.5 GiB), 16 GiB)`, where `free = limit - active - 5 GiB` and `longest = context_fit() * kv`. | Room for one conversation as long as fits (`longest`), up to the 16 GiB cap, so its follow-up can continue from it. |

`cache_bytes` is only the cache's own cap; the hard rule is `memory_limit`. At the automatic context limit, a single request's KV can take all the memory left beside the weights, and the guard empties the prompt cache for it.

**Admission.** `_fits(lru, r, held)` checks a request before it joins. Its estimate is `(len(prompt) * (1 + copies) + held) * kv + 1.5 GiB + replies * _reply_bytes`. `held` comes from `_merge_tokens`: every row padded to the longest one, plus one attention layer of the merged batch copied at a time. The estimate is compared with `memory_limit - (active - lru.nbytes)`, which assumes the prompt cache could be emptied. If it does not fit while other rows run, the request waits for a row to finish (counter `deferred`). With nothing running, the first request is always admitted. `_make_room` then drops the cache entries the admission needs.

## Settings and observing

| Setting | Default | Effect |
|---|---|---|
| `--cache-gb` | Sized from free GPU memory | Prompt cache cap in GiB. |
| `--max-context` | What fits, at least 65,536, and at most the model's own context length | Prompt plus reply tokens per request. |
| `LOCALS_REPLY_CHECKPOINT` | `1` | `0`: no reply checkpoints. |
| `LOCALS_KEEP_THINK` | `1` | `0`: Qwen3.5's template as it is, earlier replies without their empty think block (see "The reply checkpoint"). |
| `LOCALS_DISK_CACHE`, `LOCALS_DISK_CACHE_DIR`, `LOCALS_DISK_CACHE_GB` | `1`, `~/.cache/locals/prompts`, `4` | See "Disk cache of system prompts". |
| `LOCALS_GROWTH_TRIM` | `1` | `0`: no trimming every 32 steps for decode growth. |
| `LOCALS_MAX_SNAPSHOTS` | `4` | Number of shared-prefix snapshots kept. |
| `LOCALS_MEMLOG` | unset | Any non-empty value turns it on, even `0`. Prints GPU memory before and after each admission, at every `_trim_for` call (not only the calls that trim), and when KV buffers are reserved. |
| `LOCALS_PREFILL_LOG` | unset | Prints a `[prefill] ...` line for each batch prefill, and `[checkpoint] kind at n (disk on/off)` at each checkpoint position. In the non-mixed path the line is printed even when `_room_for_copy` skipped the copy. |

`GET /stats` reports, under `engines.locals`:

- `prompt_cache_mb` and `prompt_cache_entries`;
- `prompt_cache_by_kind` (entries and MB per type);
- `prompt_tokens` and `cached_tokens`;
- the counters `cache_trims` and `deferred`, next to the engine's other counters (`steps`, `rows`, `prefill_s`, `prefill_tokens`, `admissions`, `idle_s`, `mixed_steps`). `cache_trims` and `deferred` appear only after the first trim or deferral. All counters start again from zero each time `serve()` restarts, for example after an engine error.

Under `memory` it reports `gpu_active_mb`, `gpu_peak_mb`, `gpu_cache_mb` and `rss_mb`.
