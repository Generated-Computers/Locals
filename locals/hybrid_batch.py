"""Batched constrained decoding for any mlx-lm model, including hybrid
(recurrent + attention) models such as Qwen3.5, built on mlx-lm's batch caches.

Each step every active request feeds its sampled token plus its forced span, so
feeds are ragged. They are right-padded into one (B, Lmax) batch: attention
layers get ``right_padding`` (BatchKVCache rolls the padding to the left in
``finalize``), recurrent layers get per-request ``lengths`` (masked positions
leave the Gated DeltaNet state unchanged). Requests join with ``merge`` /
``extend`` and leave with ``filter``. Prompts that share a long prefix (the
nodes of a prompt trie) are prefilled once and the snapshot is copied.
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache, make_prompt_cache

from .engine import LocalsController, sample_logits, sample_rows
from .kernels import head_logits, packed_rows, text_model
from .packed import NEG_INF, BatchEngine, BatchStats, Request, _common


def _merge(caches):
    return [caches[0][i].merge([c[i] for c in caches]) for i in range(len(caches[0]))]


_MEMLOG = bool(os.environ.get("LOCALS_MEMLOG"))  # print GPU memory around each admission

# Query x key positions one prefill chunk may score per attention head. A
# chunk of c tokens after n cached ones builds (rows x c x (n + c)) scores per
# head (fp16, 16 heads on Qwen3.5-4B: 2^24 of them is 0.5 GB), so the chunk
# halves as the context grows: 2048 tokens at the start of a prompt, 256 at
# 48k, 128 beyond 100k. Without this, a 47k-token prompt's last chunks needed
# 3.5 GB above the model and a 64k one did not fit next to it on 16 GB.
PREFILL_SCORES = 1 << 24


def _reserve(cache, upto: int) -> None:
    """Grow every attention layer's KV buffer to hold ``upto`` positions now, one
    layer at a time. Left to mlx-lm, the buffer grows by concatenation as chunks
    arrive, all layers within one evaluated step: a model with attention in
    every layer (Llama, Qwen3, Mistral) briefly holds its whole KV cache twice,
    which on a 16 GB Mac ran a 21k-token prompt out of GPU memory."""
    for c in cache:
        k = getattr(c, "keys", None)
        if k is None or not isinstance(c, (BatchKVCache, KVCache)):
            continue
        step = getattr(c, "step", 256)
        need = -(-upto // step) * step
        have = k.shape[2]
        if have >= need:
            continue
        v = c.values
        if _MEMLOG and c is cache[0]:
            print(f"[mem] reserve {need} positions: active {mx.get_active_memory() / 1e9:.2f} GB", flush=True)
        c.keys = mx.concatenate([k, mx.zeros(k.shape[:2] + (need - have, k.shape[3]), k.dtype)], axis=2)
        c.values = mx.concatenate([v, mx.zeros(v.shape[:2] + (need - have, v.shape[3]), v.dtype)], axis=2)
        mx.eval(c.keys, c.values)
        del k, v


def _wrap_one(kv):
    """A single-stream KV cache as a batch-of-one cache layer, sharing its buffers."""
    b = BatchKVCache([0])
    if kv.keys is not None:
        b.keys, b.values = kv.keys, kv.values
        b.offset = b.offset + kv.offset
        b._idx = kv.offset
    return b


def _only_row(c):
    """A batch-of-one cache layer as a single-stream one, sharing its buffers."""
    if isinstance(c, BatchKVCache) and c.keys is not None:
        pad = int(c.left_padding.tolist()[0])
        k = KVCache()
        k.keys, k.values = c.keys[..., pad:c._idx, :], c.values[..., pad:c._idx, :]
        k.offset = c._idx - pad
        return k
    return c.extract(0)


def _grow_one_layer_at_a_time(cls) -> None:
    """When a KV buffer grows during decoding (every 256 positions), evaluate the
    new buffer at once, so that only one layer's old and new copies coexist
    instead of all layers' (the step's graph is otherwise evaluated as a whole)."""
    if getattr(cls, "_locals_grow_patch", False):
        return
    orig = cls.update_and_fetch

    def update_and_fetch(self, keys, values):
        k = getattr(self, "keys", None)
        idx = self._idx if hasattr(self, "_idx") else self.offset
        grow = k is not None and (idx + keys.shape[2]) > k.shape[2] and k.shape[2] >= 4096
        out = orig(self, keys, values)
        if grow:
            mx.eval(self.keys, self.values)
        return out

    cls.update_and_fetch = update_and_fetch
    cls._locals_grow_patch = True


_grow_one_layer_at_a_time(BatchKVCache)
_grow_one_layer_at_a_time(KVCache)


def _eval_layer(c) -> None:
    mx.eval([a for a in (getattr(c, "keys", None), getattr(c, "values", None), *(getattr(c, "cache", None) or []))
             if isinstance(a, mx.array)])


def _filter(cache, idx) -> None:
    """Keep the batch rows ``idx`` in every cache layer, one layer at a time. A
    filter copies the layer's buffers; left lazy, every layer's old and new
    copies coexist until the next step is evaluated, so the whole KV cache was
    briefly held twice (the memory guard emptied the prompt cache to make room
    for it each time an agent's turn ended)."""
    for c in cache:
        c.filter(idx)
        _eval_layer(c)


def _append_rows(cache, merged) -> None:
    """Append newly prefilled rows to the batch cache, one layer at a time (as _filter)."""
    for a, b in zip(cache, merged):
        a.extend(b)
        _eval_layer(a)


_GROWTH_TRIM = os.environ.get("LOCALS_GROWTH_TRIM", "1") != "0"


class DiskPromptCache:
    """System-prompt snapshots on the SSD, so that a restarted server does not prefill an agent's
    system prompt again (Hermes' is 14k tokens, OpenClaw's 8.5k: 10-30 s on a 4B model). A
    snapshot is saved when the server first makes one (the end of a long system prompt), and
    loaded into the prompt cache when a request starts with it and the memory has nothing as
    long. The directory is per model and compute dtype; the oldest files go past ``max_gb``."""

    def __init__(self, directory, max_gb: float = 4.0):
        from pathlib import Path

        import os

        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.dir, 0o700)  # system prompts and tool definitions: this user's only
        self.max_bytes = int(max_gb * (1 << 30))
        self.index = {}  # file name -> token tuple
        for f in self.dir.glob("*.tokens.json"):
            try:
                self.index[f.name[: -len(".tokens.json")]] = tuple(json.loads(f.read_text()))
            except (OSError, ValueError):
                pass

    def _name(self, tokens) -> str:
        import hashlib

        return hashlib.sha1(json.dumps(list(tokens)).encode()).hexdigest()[:20]

    def save(self, tokens, cache) -> None:
        from mlx_lm.models.cache import save_prompt_cache

        import os

        name = self._name(tokens)
        if name in self.index:
            return
        try:  # written under temporary names and renamed: a crash or a second server never leaves half a file
            tmp = self.dir / f".{name}.{os.getpid()}.tmp.safetensors"
            save_prompt_cache(str(tmp), cache)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.dir / f"{name}.safetensors")
            tmp = self.dir / f".{name}.{os.getpid()}.tmp.json"
            tmp.write_text(json.dumps(list(tokens)))
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.dir / f"{name}.tokens.json")
            self.index[name] = tuple(tokens)
            self._prune()
        except Exception as e:  # noqa: BLE001  # a cache that cannot be saved is only a slower restart
            import sys

            print(f"locals: the system prompt was not saved to disk ({type(e).__name__}: {e})", file=sys.stderr, flush=True)

    def _prune(self) -> None:
        files = sorted(self.dir.glob("*.safetensors"), key=lambda f: f.stat().st_mtime)
        total = sum(f.stat().st_size for f in files)
        while files and total > self.max_bytes:
            f = files.pop(0)
            total -= f.stat().st_size
            name = f.name[: -len(".safetensors")]
            for g in (f, self.dir / f"{name}.tokens.json"):
                g.unlink(missing_ok=True)
            self.index.pop(name, None)

    def restore(self, lru, key, prompt) -> int:
        """Load the longest saved snapshot that the prompt starts with into ``lru``, unless
        ``lru`` already holds as much; returns its length (0: nothing loaded)."""
        best, n = None, 0
        for name, toks in self.index.items():
            if n < len(toks) < len(prompt) and tuple(prompt[: len(toks)]) == toks:
                best, n = name, len(toks)
        if best is None:
            return 0
        res = lru._trie.search(key, prompt)  # what memory holds, without copying it
        if len(res.exact or res.shorter or []) >= n:
            return 0
        try:
            from mlx_lm.models.cache import load_prompt_cache

            cache = load_prompt_cache(str(self.dir / f"{best}.safetensors"))
            lru.insert_cache(key, list(prompt[:n]), cache, cache_type="system")
            (self.dir / f"{best}.safetensors").touch()
            return n
        except Exception:  # noqa: BLE001  # a file that does not load (cut short, an older format): forgotten
            for g in (self.dir / f"{best}.safetensors", self.dir / f"{best}.tokens.json"):
                g.unlink(missing_ok=True)
            self.index.pop(best, None)
            return 0


def prefill_chunk(done: int, rows: int = 1, base: int = 2048) -> int:
    """Tokens of the next prefill chunk after ``done`` cached positions."""
    c = base
    while c > 128 and rows * c * (done + c) > PREFILL_SCORES:
        c //= 2
    return c


def _cached(cache) -> int:
    """Positions already in a (single-stream) cache: the offset of its attention layers."""
    for c in cache:
        off = getattr(c, "offset", None)
        if isinstance(off, int):
            return off
    return 0


def chain_cache(max_size: int, max_bytes: int):
    """mlx-lm's LRUPromptCache, without the stale copies of a conversation where the cache cannot be
    cut back (a recurrent state; mlx-lm drops the prefixes a new entry extends only for trimmable
    caches). With Qwen3.5 every agent step left another copy of the agent's whole history (9k
    tokens, 290 MB): the cache filled with them, and memory trims emptied it. A new entry drops the
    shorter ones it extends that cannot be wanted again:

    * the end of an earlier reply ("assistant"): the conversation has gone on past it;
    * an earlier start-of-last-message checkpoint ("user"), when the new entry is one too: the
      checkpoint of the request before. A finished reply does not drop the checkpoint of its own
      request: a client that moves a message to the end on every turn (OpenClaw) continues from
      that checkpoint, not from the reply;
    * never a system-prompt entry: every new session starts from it."""
    from mlx_lm.models.cache import LRUPromptCache, can_trim_prompt_cache

    class ChainCache(LRUPromptCache):
        def insert_cache(self, model, tokens, prompt_cache, *, cache_type="assistant"):
            super().insert_cache(model, tokens, prompt_cache, cache_type=cache_type)
            if cache_type == "system" or can_trim_prompt_cache(prompt_cache):
                return
            node = self._trie._trie.get(model)
            for t in tokens:  # still there (a small cache may have evicted it at once)?
                node = node.get(t) if node is not None else None
            if node is None or "__value__" not in node:
                return
            for n, entry in self._trie.pop_prefixes(model, tokens):
                if not (entry.cache_type == "assistant" or (entry.cache_type == "user" and cache_type == "user")):
                    self._trie.add(model, tokens[:n], entry)  # kept, its place in the LRU unchanged
                    continue
                self._n_bytes -= entry.nbytes
                self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
                self._lru.remove(model, tokens[:n])

    return ChainCache(max_size=max_size, max_bytes=max_bytes)


class HybridBatchEngine(BatchEngine):
    def __init__(self, model, tokenizer, n_slots: int = 16, share_prefix: bool = True, min_share: int = 64, tail_ok: int = 32):
        self.tail_ok = tail_ok
        self.model = model
        self.tokenizer = tokenizer
        self.n_slots = n_slots
        self.share_prefix = share_prefix
        self.min_share = min_share
        # Shared-prefix snapshots of requests that arrived together (most recent last). Few are kept:
        # each holds a whole prefix's KV, and a server that kept them all grew by gigabytes over a night
        # of agents, until the prompt cache was trimmed to nothing on every admission.
        self.snapshots: "OrderedDict[Tuple[int, ...], list]" = OrderedDict()
        self.max_snapshots = max(1, int(os.environ.get("LOCALS_MAX_SNAPSHOTS", "4")))
        self.shared_tokens = 0
        self.profile = False
        self.prof = {"forward": 0.0, "sample": 0.0, "host": 0.0}
        self.trees = False
        self.top_p, self.top_k = 0.0, 0  # sampling filters at temp > 0 (see engine.sample_logits)
        self.tree_count = 0
        self.lm = text_model(model)
        # GPU bytes the engine may use (0: no guard). Before a prefill, prompt-cache
        # entries are dropped if the prefill would not fit next to them.
        self.memory_limit = 0
        self._kv_tok = None
        self._kv_layers = 1
        self.disk = None  # a DiskPromptCache: system-prompt snapshots kept across restarts

    # -- prefill with prefix snapshots -------------------------------------------------
    def _prefill(self, tokens: List[int], others: List[List[int]], tail_ok: int = 0):
        """Returns (cache, tail): KV/state for tokens[:len - len(tail)]. With
        ``tail_ok`` > 0, a short remainder after a reused snapshot is returned
        as ``tail`` to be fed in the first decode step instead of prefilled
        here (the snapshot itself is returned uncopied; merging copies it)."""
        best_key, best = None, 0
        if self.share_prefix:
            for key in self.snapshots:
                n = len(key)
                if n > best and n <= len(tokens) and tuple(tokens[:n]) == key:
                    best_key, best = key, n
        if best_key is not None:
            self.shared_tokens += best
            self.snapshots.move_to_end(best_key)
            if 0 < len(tokens) - best <= tail_ok:
                return self.snapshots[best_key], tokens[best:]
            cache = copy.deepcopy(self.snapshots[best_key])
        else:
            cache = make_prompt_cache(self.model)
        pos = best
        if self.share_prefix:
            lcp = max((_common(tokens, o) for o in others), default=0)
            if lcp >= self.min_share and lcp > pos:
                self._run(cache, tokens[pos:lcp])
                self._snapshot(tuple(tokens[:lcp]), cache)
                pos = lcp
        rest = tokens[pos:]
        if tail_ok and 0 < len(rest) <= tail_ok and pos > 0:
            snap = self.snapshots[tuple(tokens[:pos])]
            return snap, rest
        self._run(cache, rest)
        return cache, []

    def _snapshot(self, key: Tuple[int, ...], cache) -> None:
        self.snapshots[key] = copy.deepcopy(cache)
        while len(self.snapshots) > self.max_snapshots:
            self.snapshots.popitem(last=False)

    def _prefill_start(self, tokens: List[int], others: List[List[int]]):
        """Start cache (a copy of the longest usable prefix snapshot, creating a
        shared-prefix snapshot if worthwhile) and the tokens still to prefill."""
        best_key, best = None, 0
        if self.share_prefix:
            for key in self.snapshots:
                n = len(key)
                if n > best and n <= len(tokens) and tuple(tokens[:n]) == key:
                    best_key, best = key, n
        if best_key is not None:
            self.shared_tokens += best
            self.snapshots.move_to_end(best_key)
            cache = copy.deepcopy(self.snapshots[best_key])
        else:
            cache = make_prompt_cache(self.model)
        pos = best
        if self.share_prefix:
            lcp = max((_common(tokens, o) for o in others), default=0)
            if lcp >= self.min_share and lcp > pos:
                self._run(cache, tokens[pos:lcp])
                self._snapshot(tuple(tokens[:lcp]), cache)
                pos = lcp
        return cache, tokens[pos:]

    def _kv_bytes_per_token(self) -> int:
        """KV-cache bytes one token adds (attention layers only; recurrent state is fixed)."""
        if self._kv_tok is None:
            a = self.lm.args
            a = getattr(a, "text_config", None) or a
            get = (lambda k, d=None: a.get(k, d)) if isinstance(a, dict) else (lambda k, d=None: getattr(a, k, d))
            n_kv = sum(1 for c in make_prompt_cache(self.model) if isinstance(c, KVCache))
            heads = get("num_key_value_heads") or get("num_attention_heads") or 8
            dim = get("head_dim") or (get("hidden_size", 4096) // (get("num_attention_heads") or 32))
            per = 2 * heads * dim  # keys and values of one layer, from the config
            # the widest attention layer's own projections, where they are separate (Gemma 4's full-attention
            # layers have 512-wide heads where the config says 256: the estimate was half the truth)
            for _, m in self.lm.named_modules():
                k, v = getattr(m, "k_proj", None), getattr(m, "v_proj", None)
                if k is not None and v is not None and hasattr(k, "weight") and hasattr(v, "weight"):
                    per = max(per, int(k.weight.shape[0]) + int(v.weight.shape[0]))
            self._kv_layers = max(1, n_kv)
            self._kv_tok = self._kv_layers * per * 2
        return self._kv_tok

    def _merge_tokens(self, rows, new_lens: List[int]) -> int:
        """KV positions that merging new rows (prompt lengths ``new_lens``) into
        the running batch holds at its peak, beyond their own prefill: every row
        padded to the longest one, and one attention layer of the merged batch
        copied at a time (_append_rows)."""
        n = len(rows)
        top = max((len(q.prompt) + len(q.out) for q in rows), default=0)
        after = (n + len(new_lens)) * max([top] + list(new_lens))
        self._kv_bytes_per_token()  # sets _kv_layers
        return after - n * top + after // self._kv_layers

    def _fits(self, lru, r, held: int) -> bool:
        """Whether admitting ``r`` (its whole prompt uncached, at worst) can fit
        under ``memory_limit`` once the prompt cache is emptied."""
        if not self.memory_limit:
            return True
        copies = len(getattr(r, "checkpoints", None) or [])
        need = (len(r.prompt) * (1 + copies) + held) * self._kv_bytes_per_token() + (1536 << 20)
        return need <= self.memory_limit - (mx.get_active_memory() - (lru.nbytes if lru is not None else 0))

    def _make_room(self, lru, tokens: int, copies: int, held: int) -> None:
        """Drop least-valuable prompt-cache entries until an admission fits under
        ``memory_limit``: a Metal out-of-memory error fails every request in
        flight, a cache miss only costs a prefill. The admission needs KV for
        ``tokens`` new prompt tokens plus ``copies`` checkpoint copies of them,
        ``held`` more positions (the merge into the running batch:
        _merge_tokens) and the prefill transient (attention scores, bounded by
        PREFILL_SCORES)."""
        if not self.memory_limit or lru is None:
            return
        self._trim_for(lru, (tokens * (1 + copies) + held) * self._kv_bytes_per_token() + (1536 << 20))

    def _trim_for(self, lru, need: int) -> None:
        """Drop prompt-cache entries (least valuable first) until ``need`` more
        bytes fit under ``memory_limit``."""
        if not self.memory_limit or lru is None:
            return
        over = mx.get_active_memory() + need - self.memory_limit
        if over > 0:
            import gc

            gc.collect()  # finished rows' caches can sit in reference cycles until the collector runs
            mx.clear_cache()  # MLX's pool of freed buffers counts against the GPU limit too
            over = mx.get_active_memory() + need - self.memory_limit
        if over > 0 and self.snapshots:
            self.snapshots.clear()
            mx.clear_cache()
            over = mx.get_active_memory() + need - self.memory_limit
        if over > 0:
            if len(lru):
                lru.trim_to(n_bytes=max(0, lru.nbytes - over))
                mx.clear_cache()
                self.counters["cache_trims"] = self.counters.get("cache_trims", 0) + 1
        if _MEMLOG:
            print(f"[mem] room for {need / 1e9:.2f} GB: active {mx.get_active_memory() / 1e9:.2f} GB, prompt cache "
                  f"{lru.nbytes / 1e9:.2f} GB in {len(lru)}, limit {self.memory_limit / 1e9:.2f}", flush=True)

    def _prefill_batch(self, prompts: List[List[int]], others: List[List[int]], chunk: int = 1024, starts=None):
        """Prefill several prompts in one right-padded batch; returns a batch
        cache. ``starts[i]``, if given, is a (cache, remaining tokens) pair."""
        caches, rests = [], []
        for i, p in enumerate(prompts):
            if starts is not None and starts[i] is not None:
                c, rest = starts[i]
            else:
                c, rest = self._prefill_start(p, others + prompts[i + 1 :])
            caches.append(c)
            rests.append(list(rest))
        busy = [i for i, r in enumerate(rests) if r]
        if busy and len(busy) < len(rests) and not (len(caches) == 1):
            # Rows whose whole prompt is a shared snapshot (the second of two requests with one prompt)
            # have nothing to prefill: padding them to the others' length cost most of an admission
            # (8 of 10 rows, 236 tokens each). Prefill the rows that need it, then merge all.
            sub = self._prefill_batch([prompts[i] for i in busy], others, chunk, starts=[(caches[i], rests[i]) for i in busy])
            done_rows = {i: [c.extract(j) for c in sub] for j, i in enumerate(busy)}
            cache = _merge([done_rows.get(i, caches[i]) for i in range(len(caches))])
            for c in cache:
                if isinstance(c, ArraysCache):
                    c.left_padding = None
            return cache
        starts = caches
        lens = [len(r) for r in rests]
        L = max(lens)
        if len(caches) == 1 and _cached(caches[0]) > 0 and any(isinstance(c, KVCache) and c.keys is not None for c in caches[0]):
            # One prompt continuing a cached prefix: grow its own buffers to the final length
            # (layer by layer) and use them as the batch cache, instead of copying the whole
            # prefix into a new batch buffer first (a 36k-token prefix of a full-attention 3B
            # model is 4 GB: the copies did not fit next to each other on a 16 GB Mac).
            _reserve(caches[0], _cached(caches[0]) + L)
            cache = [_wrap_one(c) if isinstance(c, KVCache) else c.merge([c]) for c in caches[0]]
        else:
            cache = _merge(starts)
        del caches, starts
        for c in cache:
            if isinstance(c, ArraysCache):
                c.left_padding = None  # merge of empty caches sets zeros, which would mask out `lengths`
        if L == 0:
            return cache
        x = mx.array([r + [0] * (L - len(r)) for r in rests])
        ragged = min(lens) < L
        if ragged:
            for c in cache:
                if isinstance(c, BatchKVCache):
                    c.prepare(right_padding=[L - n for n in lens])
                elif isinstance(c, ArraysCache):
                    c.prepare(lengths=lens)
        done = max(len(p) - len(r) for p, r in zip(prompts, rests))  # longest reused prefix
        if os.environ.get("LOCALS_PREFILL_LOG"):
            print(f"[prefill] {len(prompts)} rows, prompts {[len(p) for p in prompts]}, to prefill {lens}", flush=True)
        base = next((c._idx for c in cache if isinstance(c, BatchKVCache)), 0)  # positions already in the batch cache
        i = 0
        while i < L:
            n = prefill_chunk(done + i, len(prompts), chunk)
            self.lm.model(x[:, i : i + n], cache)
            mx.eval([c.state for c in cache])
            if i == 0 and n < L:
                _reserve(cache, base + L)
            i += n
        if ragged:
            for c in cache:
                c.finalize()
            mx.eval([c.state for c in cache])
        return cache

    def _run(self, cache, toks: List[int], chunk: int = 2048) -> None:
        """Prefill ``toks`` into ``cache``. The body only, without the output
        layer: its logits (chunk x 248k) are not needed and cost ~1 GB at 2048."""
        done, i = _cached(cache), 0
        while i < len(toks):
            n = prefill_chunk(done + i, 1, chunk)
            self.lm.model(mx.array([toks[i : i + n]]), cache)
            mx.eval([c.state for c in cache])
            if i == 0 and n < len(toks):
                _reserve(cache, done + len(toks))
            i += n

    # -- decoding ------------------------------------------------------------------------
    def _forward(self, cache, feeds: List[List[int]]) -> mx.array:
        L = max(len(f) for f in feeds)
        lens = [len(f) for f in feeds]
        x = mx.array([f + [0] * (L - len(f)) for f in feeds])
        # Padding bookkeeping only when feeds are ragged: with lengths set, the
        # recurrent layers take the masked path (and skip the fused kernel).
        if min(lens) < L:
            for c in cache:
                if isinstance(c, BatchKVCache):
                    c.prepare(right_padding=[L - n for n in lens])
                elif isinstance(c, ArraysCache):
                    c.prepare(lengths=lens)
        with packed_rows(len(feeds), L, lens):
            h = self.lm.model(x, cache)  # (B, L, D), normed
        for c in cache:
            c.finalize()
        hl = h[mx.arange(len(feeds)), mx.array([n - 1 for n in lens])][:, None, :]
        return head_logits(self.lm, hl)[:, 0]

    def _head(self, hl: mx.array) -> mx.array:
        return head_logits(self.lm, hl)[:, 0]

    # -- walk mode: no token tables; the rows of the current states are recomputed each step
    def _walk_rows(self, g):
        """(B, V) int16 next global states for the rows' global states g (cached per g object)."""
        from .walk import _kernel

        cached = getattr(self, "_wrows", None)
        if cached is not None and cached[0] is g:
            return cached[1]
        table, acc, cover, tlen, eos, offs, packed = self._walk
        gq = g.reshape(-1).astype(mx.int32)
        B = int(gq.shape[0])
        V = int(eos.shape[0])
        (rows,) = _kernel()(inputs=[gq, table, packed.tbytes, packed.offs], template=[("V", V)], grid=(V, B, 1), threadgroup=(256, 1, 1),
                            output_shapes=[(B, V)], output_dtypes=[mx.int16])
        self._wrows = (g, rows)
        return rows

    def _pick(self, logits: mx.array, g, temp: float, budget=None) -> mx.array:
        """Sample one token per row under the rows' constraint states ``g``
        (global indices into the combined table, a device array); g=None: unmasked.
        ``budget``: (rows, tokens left per row) for budget-aware masking; a row
        with its own ``temp`` (a request's temperature) samples at that."""
        if budget is not None and any(getattr(r, "temp", None) is not None for r in budget[0]):
            temp = [temp if getattr(r, "temp", None) is None else r.temp for r in budget[0]]
        if getattr(self, "_walk", None) is not None and g is not None:
            table, acc, cover, tlen, eos, offs, packed = self._walk
            rows = self._walk_rows(g)
            free = (g == 0)[:, None]
            m = (rows != -1) | (acc[g][:, None] & eos[None, :]) | free
            m = m & (tlen[None, :] >= cover[g][:, None])
            logits = mx.where(m, logits, NEG_INF)
            return self._sample(logits, temp)
        if self._tab is not None and g is not None:
            allowed, cover, tlen, offs = self._tab
            m = allowed[g] & (tlen[None, :] >= cover[g][:, None])
            if budget is not None:
                rows, left = budget
                bms = [r.controller.budget_mask(g[i] - offs[id(r.controller.c)], n)
                       if isinstance(r.controller, LocalsController) else None
                       for i, (r, n) in enumerate(zip(rows, left))]
                if any(b is not None for b in bms):
                    ones = mx.ones((logits.shape[-1],), dtype=mx.bool_)
                    m = m & mx.stack([ones if b is None else b for b in bms])
            logits = mx.where(m, logits, NEG_INF)
        return self._sample(logits, temp)

    def _sample(self, logits: mx.array, temp) -> mx.array:
        if isinstance(temp, list):
            return sample_rows(logits, temp, self.top_p, self.top_k)
        return sample_logits(logits, temp, self.top_p, self.top_k)

    def _walk_next(self, x: mx.array, g) -> mx.array:
        rows = self._walk_rows(g)
        nxt = mx.take_along_axis(rows, x.reshape(-1, 1).astype(mx.int32), axis=1)[:, 0].astype(mx.int32)
        return mx.where(g == 0, 0, nxt)

    def _next_states(self, rows: List[Request], y: mx.array, g) -> mx.array:
        """Global constraint state of every row after token y (device only)."""
        if getattr(self, "_walk", None) is not None and g is not None:
            return self._walk_next(y, g)
        if self._tab is None:
            return None
        offs = self._tab[3]
        out = []
        for i, r in enumerate(rows):
            off = offs[id(r.controller.c)]
            out.append(r.controller._gpu_T()[y[i], g[i] - off].astype(mx.int32) + off)
        return mx.stack(out)

    def _chainable(self, rows: List[Request]) -> bool:
        return all(not r.controller.may_force() for r in rows)

    # -- drip mode ---------------------------------------------------------------------
    def _commit_drip(self, r, y) -> bool:
        """Commit a sampled token; a forced span goes to the row's queue and is
        fed one token per step. Returns done."""
        if y in r.controller.eos_ids:
            r.finish = "stop"
            return True
        r.controller.accept(y)
        r.out.append(y)
        if r.controller.finished():
            r.finish = "stop"
            return True
        if len(r.out) >= r.max_tokens:
            r.finish = "length"
            return True
        extra, replaces = r.controller.forced(y, heal=False) if isinstance(r.controller, LocalsController) else r.controller.forced(y)
        assert not replaces
        if extra:
            r.out.extend(extra)
            r.forced += len(extra)
            r.pend = list(extra)
            if r.controller.finished():
                r.finish = "stop"
                return True
        return False

    def _after_commit(self, r, n0: int, dead: set) -> None:
        """Streaming and cancellation hooks: ``r.on_token(new_tokens)`` is
        called with the tokens committed this step (sampled and forced), and a
        row whose ``r.cancel`` flag is set is retired at the next commit."""
        cb = getattr(r, "on_token", None)
        if cb is not None and len(r.out) > n0:
            try:
                cb(r.out[n0:])
            except Exception as e:  # noqa: BLE001  # a caller's callback that fails ends its own row, not the batch
                r.cancel = True
                r.cb_error = f"error: on_token: {type(e).__name__}: {e}"
        if getattr(r, "cancel", False) and id(r) not in dead:
            r.finish = getattr(r, "cb_error", None) or "cancelled"
            r.text = self.tokenizer.decode(r.out)
            dead.add(id(r))

    def _advance(self, rows: List[Request], x: mx.array, g):
        """Global constraint states after feeding x (device), or None."""
        if g is None:
            return None
        if getattr(self, "_walk", None) is not None:
            return self._walk_next(x, g)
        offs = self._tab[3]
        out = []
        for i, r in enumerate(rows):
            if not isinstance(r.controller, LocalsController):
                out.append(g[i])  # free state
                continue
            off = offs[id(r.controller.c)]
            out.append(r.controller._gpu_T()[x[i], g[i] - off].astype(mx.int32) + off)
        return mx.stack(out)

    def _launch1(self, cache, x: mx.array, g, temp: float, budget=None) -> mx.array:
        h = self.lm.model(x[:, None], cache)
        return self._pick(self._head(h[:, -1:, :]), g, temp, budget)

    def run_drip(self, requests: List[Request], temp: float = 0.0) -> BatchStats:
        """Pipelined continuous batching with one token per row per step.

        A forced span is fed one token per step instead of widening the step,
        so every step is a uniform (B, 1) batch whose inputs are known without
        the host: row i feeds its next queued forced token, or else the token
        it sampled last step (on the device). Constraint states live on the
        device and advance by the transition table. The host commits step s
        while step s+1 runs; it syncs only to drop finished rows or admit new
        ones. Forced spans are planned without one-token healing, since the
        sampled token is already fed when the span is known."""
        pending = list(requests)

        def source(block: bool):
            out = pending[:]
            pending.clear()
            return out

        return self.serve(source, temp)

    # -- online serving -----------------------------------------------------------------
    def _table_add(self, ctls) -> Optional[dict]:
        """Grow the combined mask table with any new constraints. State 0 is a
        free state (everything allowed) for unconstrained requests; offsets of
        known constraints never change, so device states stay valid, until most
        of the table belongs to finished requests: then it is rebuilt from the
        constraints of ``ctls`` (the running and new rows), and the old offsets
        are returned for the caller to move its rows' states (else None). A
        state is a row of the whole vocabulary (262k tokens on Gemma 4): with
        agents never letting the engine go idle, the table grew by every tool
        set and steered reply and ran Metal out of memory."""
        from .engine import _token_lengths
        from .walk import WalkController

        walkers = [c for c in ctls if isinstance(c, WalkController)]
        if walkers:
            if any(isinstance(c, LocalsController) for c in ctls):
                raise NotImplementedError("a batch mixes table and walk controllers")
            return self._walk_add(walkers)
        tab = getattr(self, "_otab", None)
        if tab is None:
            tab = self._otab = {"uniq": [], "offs": {}, "n": 1, "tlen": None}
        new = [c for c in ctls if isinstance(c, LocalsController) and id(c.c) not in tab["offs"]]
        if not new and self._tab is not None:
            return None
        old = None
        live = {id(c.c) for c in ctls if isinstance(c, LocalsController)}
        if 2 * sum(u.c.n_states for u in tab["uniq"] if id(u.c) not in live) > tab["n"]:
            old = tab["offs"]
            tab = self._otab = {"uniq": [], "offs": {}, "n": 1, "tlen": tab["tlen"], "tlen_real": tab.get("tlen_real")}
            new = [c for c in ctls if isinstance(c, LocalsController)]
        seen = set()
        new = [c for c in new if not (id(c.c) in seen or seen.add(id(c.c)))]
        for c in new:
            tab["offs"][id(c.c)] = tab["n"]
            tab["n"] += c.c.n_states
            tab["uniq"].append(c)
        n_model = text_model(self.model).model.embed_tokens.weight.shape[0]
        allowed = [mx.ones((1, n_model), dtype=mx.bool_)] + [u.allowed for u in tab["uniq"]]
        cover = [mx.zeros((1,), mx.int32)] + [u._gpu_cover() if u.canonical else mx.zeros((u.c.n_states,), mx.int32) for u in tab["uniq"]]
        if tab["uniq"] and not tab.get("tlen_real"):
            tab["tlen"], tab["tlen_real"] = _token_lengths(tab["uniq"][0].planner.vocab), True
        elif tab["tlen"] is None:
            tab["tlen"] = mx.zeros((n_model,), mx.int32)  # only the free state (cover 0) uses it
        self._tab = None  # the old table goes before the new one is built
        self._tab = (mx.concatenate(allowed, axis=0), mx.concatenate(cover), tab["tlen"], tab["offs"])
        mx.eval(self._tab[0], self._tab[1])
        return old

    def _walk_add(self, walkers) -> None:
        """Grow the combined automaton (not token) table: block i holds
        constraint i's byte table with its next states offset to global ids;
        global state 0 is the free state. Rows are recomputed per step by the
        walk kernel, so the memory is 256 entries per automaton state."""
        from .engine import _token_lengths

        wt = getattr(self, "_wtab", None)
        if wt is None:
            wt = self._wtab = {"uniq": [], "offs": {}, "n": 1}
        new = [c for c in walkers if id(c.wc) not in wt["offs"]]
        seen = set()
        new = [c for c in new if not (id(c.wc) in seen or seen.add(id(c.wc)))]
        if not new and getattr(self, "_walk", None) is not None:
            return
        for c in new:
            wt["offs"][id(c.wc)] = wt["n"]
            wt["n"] += c.wc.dfa.n
            wt["uniq"].append(c)
        import numpy as np

        from .regex import DEAD

        blocks = [np.full((1, 256), -1, dtype=np.int32)]
        acc = [np.zeros(1, dtype=bool)]
        cov = [np.zeros(1, dtype=np.int32)]
        for c in wt["uniq"]:
            base = wt["offs"][id(c.wc)]
            t = c.wc.dfa.table[:, :256].astype(np.int32)
            blocks.append(np.where(t == DEAD, -1, t + base))
            acc.append(c.wc.dfa.accepting.astype(bool))
            cov.append(np.array(c.cover) if c.canonical else np.zeros(c.wc.dfa.n, dtype=np.int32))
        if wt["n"] >= 2**15:
            raise ValueError("too many automaton states for int16 walk rows")
        table = mx.array(np.concatenate(blocks, axis=0).astype(np.int16).reshape(-1))
        u0 = wt["uniq"][0]
        self._walk = (table, mx.array(np.concatenate(acc)), mx.array(np.concatenate(cov).astype(np.int32)),
                      _token_lengths(u0.planner.vocab), u0.wc.eos_vec, wt["offs"], u0.wc.packed)
        self._wrows = None
        mx.eval(table)

    def _gstate(self, r) -> int:
        c = r.controller
        if getattr(self, "_walk", None) is not None:
            from .walk import WalkController

            return self._walk[5][id(c.wc)] + c.q if isinstance(c, WalkController) else 0
        return self._tab[3][id(c.c)] + c.q if isinstance(c, LocalsController) else 0

    def serve(self, source, temp: float = 0.0, lru=None, lru_key="locals", on_done=None) -> BatchStats:
        """Continuous batching over requests that arrive while it runs.

        ``source(block)`` returns newly arrived requests; with ``block`` it may
        wait, and an empty list then ends the loop. With an ``lru`` prompt cache
        (mlx-lm's LRUPromptCache), a request starts from the longest cached
        prefix of its prompt, and every finished row's cache is stored under
        exactly the tokens it holds, so a multi-turn client (an agent) only
        prefills what it appended since its last call."""
        self._tab = None
        self._otab = None
        self._walk = None
        self._wtab = None
        self._wrows = None
        queue: List[Request] = []
        self._check_controllers = True
        rows: List[Request] = []
        dead: set = set()
        cache, g, cur = None, None, None  # g: states masking the in-flight sample
        steps = nrows = 0
        prefill_s = 0.0
        self.tokens_out = 0
        self.counters = {"steps": 0, "rows": 0, "prefill_s": 0.0, "prefill_tokens": 0, "admissions": 0, "idle_s": 0.0}
        t0 = time.perf_counter()

        def arrive(block):
            if block:
                ti = time.perf_counter()
                got = source(True)
                self.counters["idle_s"] += time.perf_counter() - ti
            else:
                got = source(block)
            for r in got:
                r.pend, r.fed = [], []
                if type(r.controller).__name__ == "PDAController":
                    raise NotImplementedError("recursive (pushdown) constraints run on the single-stream engine only")
            return got

        def retire(i, r):
            if lru is not None:
                if len(rows) == 1:
                    # the only row: the batch cache is dropped next, so its buffers move to the
                    # prompt cache as they are (a copy would hold the KV cache twice at once)
                    row = [_only_row(c) for c in cache]
                else:
                    # the row's copy, and one attention layer of the batch cache copied once more when the
                    # finished rows are filtered out of it (_filter: padded to the longest row)
                    longest = max(len(q.prompt) + len(q.out) for q in rows)
                    kv = self._kv_bytes_per_token()
                    self._trim_for(lru, (len(r.prompt) + len(r.out)) * kv + len(rows) * longest * kv // self._kv_layers + (512 << 20))
                    row = [c.extract(i) for c in cache]
                for c in row:  # own copies: a row view would pin the whole batch buffer
                    if isinstance(c, ArraysCache):
                        c.cache = [None if a is None else a + 0 for a in c.cache]
                mx.eval([c.state for c in row])
                lru.insert_cache(lru_key, list(r.prompt[:-1]) + r.fed, row)
            if on_done is not None:
                on_done(r)

        while True:
            queue += arrive(False)
            if not rows and not queue and cur is None:
                # idle: the combined tables (which only grow) start again from the next requests' constraints
                self._tab = self._otab = self._walk = self._wtab = self._wrows = None
                got = arrive(True)
                if not got:
                    break
                queue += got
            self.active = len(rows)
            boundary = cur is None or dead or (queue and len(rows) < self.n_slots)
            if boundary:
                ys = cur.tolist() if cur is not None else []
                # Feeds are decided before committing, as in a chained step: a
                # live sample is fed now and its forced span follows later.
                live = [not r.pend for r in rows]
                xs = [r.pend.pop(0) if r.pend else yv for r, yv in zip(rows, ys)]
                for r, yv, lv in zip(rows, ys, live):
                    if lv and id(r) not in dead:
                        r.passes += 1
                        n0 = len(r.out)
                        if self._commit_drip(r, yv):
                            r.text = self.tokenizer.decode(r.out)
                            dead.add(id(r))
                        self.tokens_out += len(r.out) - n0
                        self._after_commit(r, n0, dead)
                keep = [i for i, r in enumerate(rows) if id(r) not in dead]
                if len(keep) != len(rows):
                    for i, r in enumerate(rows):
                        if id(r) in dead:
                            retire(i, r)
                    if keep:
                        idx = mx.array(keep)
                        _filter(cache, idx)
                        g = g[idx] if g is not None else None
                    else:
                        cache, g = None, None
                    xs = [xs[i] for i in keep]
                    rows = [rows[i] for i in keep]
                    dead.clear()
                for r, x in zip(rows, xs):
                    r.fed.append(x)
                if rows and g is not None:
                    g = self._advance(rows, mx.array(xs, dtype=mx.int32), g)
                for q in [q for q in queue if getattr(q, "cancel", False)]:  # its client went away while it waited:
                    queue.remove(q)  # finished here, never prefilled (a 50k-token prompt held every row up)
                    q.finish = "cancelled"
                    if on_done is not None:
                        on_done(q)
                new = []
                while queue and len(rows) + len(new) < self.n_slots:
                    # a request that would not fit next to the running rows waits for one to finish
                    held0 = self._merge_tokens(rows, [len(q.prompt) for q in new + [queue[0]]])
                    if (rows or new) and not self._fits(lru, queue[0], held0 + sum(len(q.prompt) for q in new)):
                        self.counters["deferred"] = self.counters.get("deferred", 0) + 1
                        break
                    new.append(queue.pop(0))
                if new:
                    tp = time.perf_counter()
                    if _MEMLOG:
                        print(f"[mem] before admission: active {mx.get_active_memory() / 1e9:.2f} GB, peak since last "
                              f"{mx.get_peak_memory() / 1e9:.2f} GB, rows {len(rows)}", flush=True)
                        mx.reset_peak_memory()
                    old = self._table_add([r.controller for r in rows + new])
                    if old is not None and g is not None:  # the table was rebuilt: the running rows' states move
                        offs = self._tab[3]
                        g = g + mx.array([offs[id(r.controller.c)] - old[id(r.controller.c)]
                                          if isinstance(r.controller, LocalsController) else 0 for r in rows], dtype=g.dtype)
                    starts = []
                    found = []
                    for r in new:  # every cached prefix first: making room for one must not drop the next one's
                        if lru is not None and self.disk is not None:  # a system prompt saved before a restart
                            self.disk.restore(lru, lru_key, list(r.prompt[:-1]))
                        c0, rest = (lru.fetch_nearest_cache(lru_key, list(r.prompt[:-1])) if lru is not None else (None, list(r.prompt[:-1])))
                        r.cached_tokens = len(r.prompt) - 1 - len(rest) if c0 is not None else 0
                        found.append((c0, rest))
                    # each request's cached prefix is a lazy copy, made during the prefill: the whole prompt counts
                    self._make_room(lru, sum(len(r.prompt) + len(rest) * len(getattr(r, "checkpoints", None) or [])
                                             for r, (_, rest) in zip(new, found)), 0,
                                    self._merge_tokens(rows, [len(r.prompt) for r in new]))
                    for r, (c0, rest) in zip(new, found):
                        done = r.cached_tokens
                        for cp, kind in sorted(getattr(r, "checkpoints", None) or []):
                            if lru is None or not done < cp < len(r.prompt) - 1:
                                continue
                            # prefill to the checkpoint, keep a copy, go on from there
                            c0 = c0 if c0 is not None else make_prompt_cache(self.model)
                            self._run(c0, list(r.prompt[done:cp]))
                            lru.insert_cache(lru_key, list(r.prompt[:cp]), copy.deepcopy(c0), cache_type=kind)
                            if kind == "system" and self.disk is not None:
                                self.disk.save(list(r.prompt[:cp]), c0)
                            if os.environ.get("LOCALS_PREFILL_LOG"):
                                print(f"[checkpoint] {kind} at {cp} (disk {'on' if self.disk is not None else 'off'})", flush=True)
                            done, rest = cp, list(r.prompt[cp:-1])
                        starts.append((c0, rest) if c0 is not None else None)
                    merged = self._prefill_batch([r.prompt[:-1] for r in new], [q.prompt[:-1] for q in queue], starts=starts)
                    if cache is None:
                        cache = merged
                    else:
                        _append_rows(cache, merged)
                    mx.eval([c.state for c in cache])
                    del merged, starts, found, c0  # the per-request prefill caches: merged copies now live in the batch
                    prefill_s += time.perf_counter() - tp
                    self.counters["prefill_s"] += time.perf_counter() - tp
                    self.counters["admissions"] += 1
                    if _MEMLOG:
                        print(f"[mem] admitted {[len(r.prompt) for r in new]} (cached {[r.cached_tokens for r in new]}): active "
                              f"{mx.get_active_memory() / 1e9:.2f} GB, peak {mx.get_peak_memory() / 1e9:.2f} GB, "
                              f"prompt cache {lru.nbytes / 1e9 if lru is not None else 0:.2f} GB", flush=True)
                        mx.reset_peak_memory()
                    self.counters["prefill_tokens"] += sum(len(r.prompt) - 1 - getattr(r, "cached_tokens", 0) for r in new)
                    gn = mx.array([self._gstate(r) for r in new], dtype=mx.int32)
                    g = gn if g is None else mx.concatenate([g, gn])
                    rows = rows + new
                    xs = xs + [r.prompt[-1] for r in new]
                    for r in new:
                        r.fed.append(r.prompt[-1])
                if not rows:
                    cur = None
                    continue
                if g is None:
                    g = mx.array([self._gstate(r) for r in rows], dtype=mx.int32)
                free = not any(isinstance(r.controller, LocalsController) or type(r.controller).__name__ == "WalkController" for r in rows)
                # A row's next committed sample follows exactly the tokens it has
                # fed after its prompt, so out has len(fed) - 1 tokens then.
                left = [r.max_tokens - (len(r.fed) - 1) for r in rows]
                cur = self._launch1(cache, mx.array(xs, dtype=mx.int32), None if free else g, temp, (rows, left))
                mx.async_eval(cur, g)
                steps += 1
                nrows += len(rows)
                self.counters["steps"] += 1
                self.counters["rows"] += len(rows)
                continue
            # Chained step: launch s+1 before reading s.
            live = [not r.pend for r in rows]
            hm = [bool(r.pend) for r in rows]
            if any(hm):
                hv = [r.pend.pop(0) if r.pend else 0 for r in rows]
                x = mx.where(mx.array(hm), mx.array(hv, dtype=mx.int32), cur.astype(mx.int32))
            else:
                hv = None
                x = cur.astype(mx.int32)
            g2 = g if free else self._advance(rows, x, g)
            left = [r.max_tokens - len(r.fed) for r in rows]  # this step's feed is appended below
            nxt = self._launch1(cache, x, None if free else g2, temp, (rows, left))
            mx.async_eval(nxt, g2)
            steps += 1
            nrows += len(rows)
            self.counters["steps"] += 1
            self.counters["rows"] += len(rows)
            if lru is not None and self.memory_limit and self.counters["steps"] % 32 == 0 and _GROWTH_TRIM:
                # the rows' KV grows every step, past what their admission counted: room for the next 32
                # steps comes out of the prompt cache now, before Metal runs out (which fails every row)
                self._trim_for(lru, len(rows) * 32 * self._kv_bytes_per_token())
            ys = cur.tolist()
            for i, (r, yv, lv) in enumerate(zip(rows, ys, live)):
                r.fed.append(hv[i] if hm[i] else yv)
                if lv and id(r) not in dead:
                    r.passes += 1
                    n0 = len(r.out)
                    if self._commit_drip(r, yv):
                        r.text = self.tokenizer.decode(r.out)
                        dead.add(id(r))
                    self.tokens_out += len(r.out) - n0
                    self._after_commit(r, n0, dead)
            cur, g = nxt, g2
        mx.synchronize()
        self.active = 0
        return BatchStats(time.perf_counter() - t0, prefill_s, steps, nrows, self.tokens_out)

    def run(self, requests: List[Request], temp: float = 0.0, pipeline: bool = True, mode: str = "drip") -> BatchStats:
        """Continuous batching. ``mode="drip"`` (default): one token per row per
        step, fully pipelined (see run_drip). ``mode="ragged"``: forced spans
        widen the step. With ``pipeline``, a ragged-mode step whose feeds are
        determined without the host (no live row can reach a forced span) is
        launched from the previous step's device tokens and constraint states
        before the host reads them, so graph building and host bookkeeping
        overlap GPU work. Outputs are identical to the unpipelined loop."""
        if not pipeline:
            return self._run_sync(requests, temp)
        if mode == "drip":
            return self.run_drip(requests, temp)
        self._tab = self._tables(requests)
        offs = self._tab[3] if self._tab is not None else None
        queue = list(requests)
        rows: List[Request] = []  # requests in cache row order (may include dead rows)
        dead: set = set()
        cache = None
        steps = nrows = 0
        prefill_s = 0.0
        t0 = time.perf_counter()
        cur = None  # (y, g_next) of the launched, uncommitted step
        while True:
            if cur is None:
                # Host-built step: drop finished rows, admit new requests, feed spans.
                live = [i for i, r in enumerate(rows) if id(r) not in dead]
                if len(live) != len(rows):
                    if live:
                        idx = mx.array(live)
                        _filter(cache, idx)
                    else:
                        cache = None
                    rows = [rows[i] for i in live]
                    dead.clear()
                if queue and len(rows) < self.n_slots:
                    tp = time.perf_counter()
                    new, caches = [], []
                    while queue and len(rows) + len(new) < self.n_slots:
                        r = queue.pop(0)
                        others = [q.prompt[:-1] for q in queue]
                        c, tail = self._prefill(r.prompt[:-1], others, tail_ok=self.tail_ok)
                        caches.append(c)
                        r.feed = tail + [r.prompt[-1]]
                        new.append(r)
                    merged = _merge(caches)
                    if cache is None:
                        cache = merged
                    else:
                        _append_rows(cache, merged)
                    rows.extend(new)
                    prefill_s += time.perf_counter() - tp
                if not rows:
                    break
                feeds = [r.feed for r in rows]
                nrows += sum(len(f) for f in feeds)
                g = None
                if self._tab is not None:
                    g = mx.array([offs[id(r.controller.c)] + r.controller.q for r in rows], dtype=mx.int32)
                y = self._pick(self._forward(cache, feeds), g, temp)
                cur = (y, self._next_states(rows, y, g))
                mx.async_eval(*[a for a in cur if a is not None])
                steps += 1
            y, gn = cur
            # Chain the next step before reading this one.
            nxt = None
            if not dead and not (queue and len(rows) < self.n_slots) and self._chainable(rows):
                h = self.lm.model(y[:, None], cache)
                y2 = self._pick(self._head(h[:, -1:, :]), gn, temp)
                nxt = (y2, self._next_states(rows, y2, gn))
                mx.async_eval(*[a for a in nxt if a is not None])
                nrows += len(rows)
                steps += 1
            ys = y.tolist()
            for r, yv in zip(rows, ys):
                if id(r) in dead:
                    continue
                r.passes += 1
                if self._commit(r, yv):
                    r.text = self.tokenizer.decode(r.out)
                    dead.add(id(r))
                elif nxt is not None and r.feed != [yv]:
                    raise RuntimeError("chained step fed a row that needed a forced span")
            cur = nxt
        mx.synchronize()
        return BatchStats(time.perf_counter() - t0, prefill_s, steps, nrows, sum(len(r.out) for r in requests))

    def _run_sync(self, requests: List[Request], temp: float = 0.0) -> BatchStats:
        self._tab = self._tables(requests)
        queue = list(requests)
        active: List[Request] = []
        cache = None
        steps = rows = 0
        prefill_s = 0.0
        t0 = time.perf_counter()
        while queue or active:
            if queue and len(active) < self.n_slots:
                tp = time.perf_counter()
                new, caches = [], []
                while queue and len(active) + len(new) < self.n_slots:
                    r = queue.pop(0)
                    others = [q.prompt[:-1] for q in queue]
                    c, tail = self._prefill(r.prompt[:-1], others, tail_ok=self.tail_ok)
                    caches.append(c)
                    r.feed = tail + [r.prompt[-1]]
                    new.append(r)
                merged = _merge(caches)
                if cache is None:
                    cache = merged
                else:
                    _append_rows(cache, merged)
                active.extend(new)
                prefill_s += time.perf_counter() - tp
            feeds = [r.feed for r in active]
            rows += sum(len(f) for f in feeds)
            ta = time.perf_counter()
            logits = self._forward(cache, feeds)
            if self.profile:
                mx.eval(logits)
            tb = time.perf_counter()
            steps += 1
            if self._tab is None:  # unconstrained requests
                ys = sample_logits(logits, temp, self.top_p, self.top_k).tolist()
            else:
                ys = self._sample_states([(r.controller, r.controller.q) for r in active], logits, temp).tolist()
            tc = time.perf_counter()
            keep, still = [], []
            for i, (r, y) in enumerate(zip(active, ys)):
                r.passes += 1
                if self._commit(r, y):
                    r.text = self.tokenizer.decode(r.out)
                else:
                    keep.append(i)
                    still.append(r)
            if len(still) != len(active):
                if still:
                    idx = mx.array(keep)
                    _filter(cache, idx)
                else:
                    cache = None
            active = still
            td = time.perf_counter()
            self.prof["forward"] += tb - ta
            self.prof["sample"] += tc - tb
            self.prof["host"] += td - tc
        mx.synchronize()
        return BatchStats(time.perf_counter() - t0, prefill_s, steps, rows, sum(len(r.out) for r in requests))
