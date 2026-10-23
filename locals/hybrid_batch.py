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
from mlx_lm.models.cache import ArraysCache, BatchKVCache, BatchRotatingKVCache, KVCache, make_prompt_cache

from .engine import LocalsController, row_key, sample_each, sample_logits
from .kernels import head_logits, packed_rows, text_model
from .packed import NEG_INF, BatchEngine, BatchStats, Request, _common


def _merge(caches):
    return [caches[0][i].merge([c[i] for c in caches]) for i in range(len(caches[0]))]


def _prepare_ragged(cache, lens: List[int]) -> None:
    """Rows right-padded to the longest: every layer told each row's own length, as mlx-lm's batch generator does.
    A sliding-window layer (Gemma's) needs the lengths too, or the padding evicts a shorter row's last real
    tokens from its window and the row continues from the padding: past the 512-token window, Gemma 4 E4B's
    replies to prompts read together began with the end of turn (11 of 24 empty, none alone)."""
    L = max(lens)
    pads = [L - n for n in lens]
    for c in cache:
        if isinstance(c, BatchKVCache):
            c.prepare(right_padding=pads)
        elif isinstance(c, BatchRotatingKVCache):
            c.prepare(lengths=lens, right_padding=pads)
        elif isinstance(c, ArraysCache):
            c.prepare(lengths=lens)


# What an admission keeps room for besides KV: each new request's fixed state (Qwen3.5's recurrent layers: 52 MB
# a request for the 4B, 154 MB for a 27B, which the check did not count: 16 requests at once on a 27B on 24 GB ran
# Metal out of memory) and reading its prompt. Reading needs memory that is no straight line in the tokens
# (Qwen3.5-4B: 0.14 GB for 16 tokens, 1.1 GB for 128 to 256, 2.1 GB for 2048), so the engine records what each
# read needed (_note_read) and keeps room for the most a read of as many tokens or more needed; PREFILL_RESERVE
# (1.5 GB) until it has seen one. LOCALS_PREFILL_RESERVE_GB keeps a fixed size instead.
_RESERVE_ENV = os.environ.get("LOCALS_PREFILL_RESERVE_GB")
PREFILL_RESERVE = int(float(_RESERVE_ENV or "1.5") * (1 << 30))
_PEAK = [0]  # the highest GPU memory seen before the engine reset MLX's peak counter (peak_memory())


def peak_memory() -> int:
    """The process's peak GPU memory, across the engine's resets of MLX's counter."""
    return max(_PEAK[0], mx.get_peak_memory())


def _reset_peak() -> None:
    _PEAK[0] = max(_PEAK[0], mx.get_peak_memory())
    mx.reset_peak_memory()


_MEMLOG = bool(os.environ.get("LOCALS_MEMLOG"))  # print GPU memory around each admission
_MIXLOG = bool(os.environ.get("LOCALS_MIXLOG"))  # print each mixed pass (locals/mixed.py)

# Query x key positions one prefill chunk may score per attention head. A
# chunk of c tokens after n cached ones builds (rows x c x (n + c)) scores per
# head (fp16, 16 heads on Qwen3.5-4B: 2^24 of them is 0.5 GB), so the chunk
# halves as the context grows: 2048 tokens at the start of a prompt, 256 at
# 48k, 128 beyond 100k. Without this, a 47k-token prompt's last chunks needed
# 3.5 GB above the model and a 64k one did not fit next to it on 16 GB.
PREFILL_SCORES = 1 << 24
# A model whose prompt attention never holds a chunk's scores (fuse.py sets its ``prefill_scores``: Flint's, MLX's
# fused causal attention in the full layers and bands in the window layers) keeps longer chunks further: a 32k prompt
# in 2,048-token chunks read at 560 tokens/s against 506 in chunks halved to 512, 0.5 GB more at the peak.
# Tokens of all rows in one prefill forward. Its activations grow with them (Qwen3.5-4B: about 1 GB
# at 2048, 4.3 GB at 8 rows x 1024, past the GPU's memory at 16 x 1024: sixteen requests arriving
# together failed with a Metal out-of-memory error); the prefill rate is the same from ~1k tokens up
# (750 tok/s for 1 x 2048, 8 x 256, 16 x 128 or 32 x 64), so more rows take shorter chunks.
PREFILL_TOKENS = 2048


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


def _copies(r) -> int:
    """Copies of a request's prompt cache its checkpoints keep (a "reply" one keeps only recurrent state)."""
    return sum(1 for _, kind in getattr(r, "checkpoints", None) or [] if kind != "reply")


def _replies(r) -> int:
    return sum(1 for _, kind in getattr(r, "checkpoints", None) or [] if kind == "reply")


def _keep_reply_state(r, n: int, cache) -> None:
    """At a "reply" checkpoint (``n``: where the reply starts as the next request renders the
    conversation): keep a copy of the cache layers that cannot be cut back to a prefix (recurrent
    state, a few MB per layer); _reply_entry adds the KV prefix of the finished row to them."""
    r.reply_state = (n, [copy.deepcopy(c) if not isinstance(c, (KVCache, BatchKVCache)) else None for c in cache])


def _reply_entry(r, row):
    """The prompt-cache entry for ``r.prompt[:n]`` (see _keep_reply_state): the finished row's KV cut
    to n positions (views of its buffers: no copy) and the recurrent state kept at n. None if the
    request kept none or a layer is of another kind (a rotating window cannot be cut back)."""
    state = getattr(r, "reply_state", None)
    if state is None:
        return None
    n, kept = state
    out = []
    for c, k in zip(row, kept):
        if k is not None:
            out.append(k)
        elif type(c) is KVCache and c.keys is not None and c.offset >= n:
            v = KVCache()
            v.keys, v.values, v.offset = c.keys[..., :n, :], c.values[..., :n, :], n
            out.append(v)
        else:
            return None
    return out


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
    """Append newly prefilled rows to the batch cache, one layer at a time (as _filter), each of ``merged``'s
    layers released once copied."""
    for i, a in enumerate(cache):
        a.extend(merged[i])
        _eval_layer(a)
        merged[i] = None


def _merge_layerwise(caches) -> list:
    """One batch cache of several requests' own caches (lists), a layer at a time, each request's layer released
    (set to None) once copied: merged whole, every request's cache and its copy existed at once, and three
    26.7k-token prompts joining at once took a 4B on a 16 GB Mac to 12.6 GB, past the GPU's limit."""
    out = []
    for i in range(len(caches[0])):
        layer = caches[0][i].merge([c[i] for c in caches])
        if isinstance(layer, ArraysCache):
            layer.left_padding = None  # (as _prefill_batch: a merge of empty caches sets zeros, which mask out lengths)
        _eval_layer(layer)
        for c in caches:
            if isinstance(c, list):
                c[i] = None
        out.append(layer)
    return out


_GROWTH_TRIM = os.environ.get("LOCALS_GROWTH_TRIM", "1") != "0"
# One request running alone: its forced spans fed in one pass, and prompt-lookup drafts verified in the same
# pass as the next token (HybridBatchEngine._solo). LOCALS_SOLO=0: one token per step, as with many requests.
_SOLO = os.environ.get("LOCALS_SOLO", "1") != "0"


def _fsync(path) -> None:
    """``path``'s data on the disk itself (renamed after a power cut, a file came back empty or half written)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


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
        for f in self.dir.glob(".*.tmp.*"):  # a save that a crash or a full disk cut short (pid in the name)
            try:
                os.kill(int(f.name.split(".")[2]), 0)  # (only asks whether that process still runs)
            except (ValueError, IndexError, ProcessLookupError):
                f.unlink(missing_ok=True)
            except PermissionError:
                pass
        for f in self.dir.glob("*.tokens.json"):
            try:
                self.index[f.name[: -len(".tokens.json")]] = tuple(json.loads(f.read_text()))
            except (OSError, ValueError):
                pass

    def _name(self, tokens) -> str:
        import hashlib

        return hashlib.sha1(json.dumps(list(tokens)).encode()).hexdigest()[:20]

    def save(self, tokens, cache) -> None:
        """Never raises: a snapshot that cannot be saved is only a slower restart. (The folder deleted under a running
        server, by the user or by another server's pruning, raised out of the engine: every request in flight
        failed, half the prompt cache went, and again at each new system prompt.)"""
        from mlx_lm.models.cache import save_prompt_cache

        import os
        import shutil

        name = self._name(tokens)
        if name in self.index and (self.dir / f"{name}.safetensors").exists():
            return
        tmps = []
        try:
            size = sum(a.nbytes for c in cache for a in (c.state if isinstance(c.state, (list, tuple)) else [c.state])
                       if isinstance(a, mx.array))
            if size > self.max_bytes:  # (it would push out every snapshot, itself too: LOCALS_DISK_CACHE_GB=0 did)
                return
            self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            if shutil.disk_usage(self.dir).free < 2 * size + (1 << 30):  # a nearly full disk: no snapshot (only slower)
                return
            tmps = [self.dir / f".{name}.{os.getpid()}.tmp.safetensors", self.dir / f".{name}.{os.getpid()}.tmp.json"]
            # written under temporary names, flushed to the disk and renamed: a crash or a power cut leaves no half
            # file under the real name, and a second server sharing the folder none either
            save_prompt_cache(str(tmps[0]), cache)
            _fsync(tmps[0])
            os.chmod(tmps[0], 0o600)
            os.replace(tmps[0], self.dir / f"{name}.safetensors")
            tmps[1].write_text(json.dumps(list(tokens)))
            _fsync(tmps[1])
            os.chmod(tmps[1], 0o600)
            os.replace(tmps[1], self.dir / f"{name}.tokens.json")
            self.index[name] = tuple(tokens)
            self._prune()
        except Exception as e:  # noqa: BLE001  # a cache that cannot be saved is only a slower restart
            import sys

            for t in tmps:  # (left behind, each failed save kept up to a snapshot's size of a full disk, for good)
                t.unlink(missing_ok=True)
            print(f"locals: the system prompt was not saved to disk ({type(e).__name__}: {e})", file=sys.stderr, flush=True)

    def _prune(self) -> None:
        """The oldest snapshots go past ``max_bytes``, across every model's folder (the cap was per model, and a
        model served once kept its 4 GB for good); a folder left empty goes too."""
        root = self.dir.parent
        files = []
        for f in root.glob("*/*.safetensors"):
            try:
                files.append((f.stat().st_mtime, f.stat().st_size, f))
            except OSError:
                pass
        files.sort()
        total = sum(size for _, size, _ in files)
        while files and total > self.max_bytes:
            _, size, f = files.pop(0)
            total -= size
            name = f.name[: -len(".safetensors")]
            for g in (f, f.parent / f"{name}.tokens.json"):
                g.unlink(missing_ok=True)
            if f.parent == self.dir:
                self.index.pop(name, None)
            elif not any(f.parent.iterdir()):
                f.parent.rmdir()

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


def prefill_chunk(done: int, rows: int = 1, base: int = 2048, scores: int = PREFILL_SCORES) -> int:
    """Tokens of the next prefill chunk after ``done`` cached positions."""
    c = base
    while c > 128 and rows * c * (done + c) > scores:
        c //= 2
    while c > 16 and rows * c > PREFILL_TOKENS:
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
        # rows of a mixed pass, decode rows and prompt tokens (locals/mixed.py; 0: a new prompt is prefilled
        # while the rows wait). 1024: as fast as a prefill on its own; the rows get a token every pass (1.5 s on
        # a 4B model on an M2 Pro) instead of none for the whole prefill (12 s for 8k tokens).
        self.mix_chunk = int(os.environ.get("LOCALS_MIX_CHUNK", "1024"))
        self._reply_bytes = 0  # a kept reply state's size (recurrent layers), once one was kept
        self._mix_ok = None
        self._fill_eval = []
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
        self._prefill_scores = getattr(self.lm, "prefill_scores", PREFILL_SCORES)
        self._prefill_base = getattr(self.lm, "prefill_chunk_tokens", 1024)  # a prompt's chunk where it may be longest
        # GPU bytes the engine may use (0: no guard). Before a prefill, prompt-cache
        # entries are dropped if the prefill would not fit next to them.
        self.memory_limit = 0
        self._kv_tok = None
        self._kv_layers = 1
        self._row_bytes = None  # fixed state bytes per request (_row_state)
        self._next_trim = 0  # the step of the next growth trim (_growth_trim)
        self._reads: Dict[int, int] = {}  # tokens per prefill pass (power of 2) -> most bytes one needed
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
            caches = make_prompt_cache(self.model)
            n_kv = sum(1 for c in caches if isinstance(c, KVCache))
            # 8-bit keys and values (locals/qkv.py): a byte each, plus a bf16 scale and bias a group of 64
            n_q = sum(1 for c in caches if getattr(c, "bits", None) == 8 and getattr(c, "group_size", None) == 64)
            heads = get("num_key_value_heads") or get("num_attention_heads") or 8
            dim = get("head_dim") or (get("hidden_size", 4096) // (get("num_attention_heads") or 32))
            per = 2 * heads * dim  # keys and values of one layer, from the config
            # the widest attention layer's own projections, where they are separate (Gemma 4's full-attention
            # layers have 512-wide heads where the config says 256: the estimate was half the truth)
            for _, m in self.lm.named_modules():
                k, v = getattr(m, "k_proj", None), getattr(m, "v_proj", None)
                if k is not None and v is not None and hasattr(k, "weight") and hasattr(v, "weight"):
                    per = max(per, int(k.weight.shape[0]) + int(v.weight.shape[0]))
            self._kv_layers = max(1, n_kv + n_q)
            self._kv_tok = max(1, n_kv) * per * 2 if not n_q else n_kv * per * 2 + n_q * (per + per * 4 // 64)
        return self._kv_tok

    def _row_state(self) -> int:
        """Bytes of the state each request holds whatever its length: every cache layer that is not a growing KV
        cache (recurrent and convolution state; a sliding window at its full size), measured once on a one-row
        cache after one token."""
        if self._row_bytes is None:
            self._row_bytes = 0
            try:
                cache = make_prompt_cache(self.model)
                self.lm.model(mx.array([[100]]), cache)
                mx.eval([c.state for c in cache])
                fixed = 0
                for c in cache:
                    if isinstance(c, KVCache):
                        continue
                    seen, arrs = set(), []
                    for group in (getattr(c, "state", ()), getattr(c, "cache", ())):
                        for a in (group if isinstance(group, (list, tuple)) else [group]):
                            if isinstance(a, mx.array) and id(a) not in seen:
                                seen.add(id(a))
                                arrs.append(a)
                    n = sum(a.nbytes for a in arrs)
                    size, k = getattr(c, "max_size", None), getattr(c, "keys", None)
                    if size and isinstance(k, mx.array) and k.ndim == 4 and k.shape[2]:
                        n = n * size // k.shape[2]  # a sliding window, at its full size
                    fixed += n
                self._row_bytes = fixed
                if _MEMLOG:
                    print(f"[mem] fixed state per request {fixed / 1e6:.1f} MB", flush=True)
            except Exception:  # noqa: BLE001  # a model the probe cannot run: no fixed state counted, as before
                pass
        return self._row_bytes

    def _note_read(self, tokens: int, need: int) -> None:
        """A prefill pass of ``tokens`` tokens (all rows) needed ``need`` bytes above what was in use before it."""
        b = 16
        while b < tokens:
            b *= 2
        self._reads[b] = max(self._reads.get(b, 0), need)

    def _read_bytes(self, tokens: int) -> int:
        """Room for reading ``tokens`` prompt tokens: the most a recorded read of as many tokens or more needed,
        with a sixth more; PREFILL_RESERVE where none was recorded (or LOCALS_PREFILL_RESERVE_GB is set)."""
        if _RESERVE_ENV:
            return PREFILL_RESERVE
        t = min(max(tokens, 1), PREFILL_TOKENS)
        seen = [v for b, v in self._reads.items() if b >= t]
        return max(seen) * 7 // 6 if seen else PREFILL_RESERVE

    def _admit_bytes(self, rows: int, tokens: int) -> int:
        """Memory ``rows`` new requests need besides their KV: each one's fixed state, held 2.2 times over while the
        batch decodes (Qwen3.5-4B: 14 requests admitted together added 1.68 GB, 120 MB each, against 52 MB of
        state: the step being computed and the one queued behind it each hold their own), and a first block of KV
        (256 positions); and reading ``tokens`` of their prompts. Measured at the admission instead, the cost read
        52 MB a request (no step in flight yet: half of what decoding holds), so it is not learned there."""
        per_row = int(2.2 * self._row_state()) + 256 * self._kv_bytes_per_token()
        return rows * per_row + self._read_bytes(tokens)

    def _merge_tokens(self, rows, new_lens: List[int]) -> int:
        """KV positions that merging new rows (prompt lengths ``new_lens``) into
        the running batch holds at its peak, beyond their own prefill: every row
        padded to the longest one, and one attention layer of the merged batch
        copied at a time (_append_rows). Counting the new rows' KV only once
        (their caches are handed to the merge a layer at a time) let three
        26.7k-token prompts in beside a running one and took a 4B on a 16 GB Mac
        to 11.07 GB against a 9.84 GB limit: reading prompts up to their
        checkpoints holds more than the KV and one pass."""
        n = len(rows)
        top = max((len(q.prompt) + len(q.out) for q in rows), default=0)
        after = (n + len(new_lens)) * max([top] + list(new_lens))
        self._kv_bytes_per_token()  # sets _kv_layers
        return after - n * top + after // self._kv_layers

    def _growth_trim(self, n_rows: int, positions: int = 32) -> None:
        """Every 32 steps: the rows' KV grows every step, past what their admission counted, so room for the next
        ``positions`` of each comes out of the prompt cache now, before Metal runs out (which fails every row).
        Called by every kind of step: chained, boundary (all of them while a request waits for memory: the trim
        never ran then, when memory is tightest) and a lone row's passes."""
        lru = getattr(self, "_serve_lru", None)
        if lru is not None and self.memory_limit and _GROWTH_TRIM and self.counters["steps"] >= self._next_trim:
            self._next_trim = self.counters["steps"] + 32  # (a multiple of 32 can be passed by in a lone row's passes)
            self._trim_for(lru, n_rows * positions * self._kv_bytes_per_token())

    def _keep_reply(self, r, n: int, cache) -> None:
        _keep_reply_state(r, n, cache)
        self._reply_bytes = max(self._reply_bytes, sum(c.nbytes for c in r.reply_state[1] if c is not None))

    def _room_for_copy(self, lru, positions: int) -> bool:
        """Whether a checkpoint's copy of a request's cache can be kept: the copy shares the request's KV
        buffers, so the request's next write copies them, and ``positions`` (its prompt) of KV then exist
        twice. Near the context limit that is past the GPU's memory; the copy is then left out (it is only
        a prompt-cache entry)."""
        if not self.memory_limit:
            return True
        need = positions * self._kv_bytes_per_token() + self._admit_bytes(0, PREFILL_TOKENS)
        self._trim_for(lru, need)
        return mx.get_active_memory() + need <= self.memory_limit

    def _fits(self, lru, r, held: int, rows: int = 1, tokens: int = 0) -> bool:
        """Whether admitting ``r`` (its whole prompt uncached, at worst) can fit under ``memory_limit`` once the
        prompt cache is emptied, with ``rows`` new requests in all (``r`` and those admitted with it) whose
        prompts, ``tokens`` of them, are read together."""
        if not self.memory_limit:
            return True
        copies = _copies(r)
        need = ((len(r.prompt) * (1 + copies) + held) * self._kv_bytes_per_token()
                + self._admit_bytes(rows, tokens or len(r.prompt)) + _replies(r) * self._reply_bytes)
        return need <= self.memory_limit - (mx.get_active_memory() - (lru.nbytes if lru is not None else 0))

    def _make_room(self, lru, tokens: int, copies: int, held: int, rows: int = 1) -> None:
        """Drop least-valuable prompt-cache entries until an admission fits under
        ``memory_limit``: a Metal out-of-memory error fails every request in
        flight, a cache miss only costs a prefill. The admission needs KV for
        ``tokens`` new prompt tokens plus ``copies`` checkpoint copies of them,
        ``held`` more positions (the merge into the running batch:
        _merge_tokens) and the prefill transient (attention scores, bounded by
        PREFILL_SCORES)."""
        if not self.memory_limit or lru is None:
            return
        self._trim_for(lru, (tokens * (1 + copies) + held) * self._kv_bytes_per_token() + self._admit_bytes(rows, tokens))

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

    def _prefill_batch(self, prompts: List[List[int]], others: List[List[int]], chunk: Optional[int] = None, starts=None,
                       stop=None):
        """Prefill several prompts in one right-padded batch; returns a batch
        cache. ``starts[i]``, if given, is a (cache, remaining tokens) pair; the
        list is emptied once its caches are merged (the caller hands them over:
        held there too, every prompt's KV existed twice at the join).
        ``stop()``, asked between chunks: true gives up the prefill (returns None)."""
        chunk = chunk or self._prefill_base
        given = starts
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
            sub = self._prefill_batch([prompts[i] for i in busy], others, chunk, starts=[(caches[i], rests[i]) for i in busy], stop=stop)
            if sub is None:
                return None
            done_rows = {i: [c.extract(j) for c in sub] for j, i in enumerate(busy)}
            del sub
            cache = _merge_layerwise([done_rows.get(i, caches[i]) for i in range(len(caches))])
            if given is not None:
                given.clear()
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
            cache = _merge_layerwise(starts)
        if given is not None:
            given.clear()
        del caches, starts
        for c in cache:
            if isinstance(c, ArraysCache):
                c.left_padding = None  # merge of empty caches sets zeros, which would mask out `lengths`
        if L == 0:
            return cache
        x = mx.array([r + [0] * (L - len(r)) for r in rests])
        ragged = min(lens) < L
        if ragged:
            _prepare_ragged(cache, lens)
        done = max(len(p) - len(r) for p, r in zip(prompts, rests))  # longest reused prefix
        if os.environ.get("LOCALS_PREFILL_LOG"):
            print(f"[prefill] {len(prompts)} rows, prompts {[len(p) for p in prompts]}, to prefill {lens}", flush=True)
        base = next((c._idx for c in cache if isinstance(c, BatchKVCache)), 0)  # positions already in the batch cache
        i = 0
        while i < L:
            n = prefill_chunk(done + i, len(prompts), chunk, self._prefill_scores)
            a0 = mx.get_active_memory()
            _reset_peak()
            self.lm.model(x[:, i : i + n], cache)
            mx.eval([c.state for c in cache])
            self._note_read(len(prompts) * n, mx.get_peak_memory() - a0)
            if i == 0 and n < L:
                _reserve(cache, base + L)
            i += n
            if stop is not None and i < L and stop():
                return None
        if ragged:
            for c in cache:
                c.finalize()
            mx.eval([c.state for c in cache])
        return cache

    def _run(self, cache, toks: List[int], chunk: int = 2048, stop=None) -> bool:
        """Prefill ``toks`` into ``cache``. The body only, without the output
        layer: its logits (chunk x 248k) are not needed and cost ~1 GB at 2048.
        False if ``stop()`` (asked between chunks) gave it up."""
        done, i = _cached(cache), 0
        while i < len(toks):
            n = prefill_chunk(done + i, 1, chunk, self._prefill_scores)
            a0 = mx.get_active_memory()
            _reset_peak()
            self.lm.model(mx.array([toks[i : i + n]]), cache)
            mx.eval([c.state for c in cache])
            self._note_read(n, mx.get_peak_memory() - a0)
            if i == 0 and n < len(toks):
                _reserve(cache, done + len(toks))
            i += n
            if stop is not None and i < len(toks) and stop():
                return False
        return True

    # -- decoding ------------------------------------------------------------------------
    def _forward(self, cache, feeds: List[List[int]]) -> mx.array:
        L = max(len(f) for f in feeds)
        lens = [len(f) for f in feeds]
        x = mx.array([f + [0] * (L - len(f)) for f in feeds])
        # Padding bookkeeping only when feeds are ragged: with lengths set, the
        # recurrent layers take the masked path (and skip the fused kernel).
        if min(lens) < L:
            _prepare_ragged(cache, lens)
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
        with its own sampling (a request's temperature, top_p, top_k or seed) samples by that."""
        if budget is not None:
            temp = self._specs(budget[0], temp)
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
        if isinstance(temp, list):  # per row: (temperature, top_p, top_k, key)
            return sample_each(logits, temp)
        return sample_logits(logits, temp, self.top_p, self.top_k)

    def _specs(self, rows, temp):
        """Per-row sampling when a row has its own (a request's temperature, top_p, top_k or seed): a list of
        (temperature, top_p, top_k, key or None); else ``temp`` (the engine's settings for every row). A seeded
        row's key counts its draws, so its draws do not depend on the batch around it."""
        if not any(getattr(r, a, None) is not None for r in rows for a in ("temp", "top_p", "top_k", "seed")):
            return temp
        out = []
        for r in rows:
            t = temp if getattr(r, "temp", None) is None else r.temp
            key = None
            if getattr(r, "seed", None) is not None and t > 0:
                key = row_key(r.seed, r.draws)
                r.draws += 1
            out.append((t, self.top_p if getattr(r, "top_p", None) is None else r.top_p,
                        self.top_k if getattr(r, "top_k", None) is None else r.top_k, key))
        return out

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

    # -- one request alone: several tokens per pass ------------------------------------------
    #
    # With one row, a pass over several tokens costs little more than a pass over one (the weights are
    # read once either way), so the only running request gets steps that feed more than its next token:
    # its forced span at once (tool-call syntax, JSON keys), or a prompt-lookup draft (the continuation
    # of the most recent earlier occurrence of its last tokens; an agent rewriting a file repeats it)
    # verified in the same pass. Accepted draft tokens are the ones one-token steps would have sampled;
    # rejected ones are rewound. Without a span or a draft, the row goes back to the pipelined steps.

    def _solo_ok(self, r, cache) -> bool:
        if not _SOLO or cache is None or getattr(self, "_walk", None) is not None:
            return False
        if any(type(c) not in (BatchKVCache, ArraysCache) for c in cache):
            # sliding-window layers (Gemma, Flint) and 8-bit ones (locals/qkv.py) are not rewound, which a draft needs
            # and a forced span does not: those get their spans in one pass, and no drafts (_solo_drafts_ok)
            from .qkv import BatchQuantizedKVCache

            if any(type(c) not in (BatchKVCache, ArraysCache, BatchRotatingKVCache, BatchQuantizedKVCache) for c in cache):
                return False
            if self._solo_drafts_ok(cache):
                return False
        name = type(r.controller).__name__
        return name not in ("WalkController", "PDAController")

    def _solo_drafts_ok(self, cache) -> bool:
        """Drafts need the per-token recurrent states the fused Gated DeltaNet kernel records (to rewind)."""
        from . import fuse

        n = getattr(self, "_solo_gdn", None)
        if n is None:  # the fused GDN layers, or 0 when any layer cannot record its states
            lin = [getattr(l, "linear_attn", None) for l in self.lm.model.layers]
            lin = [m for m in lin if m is not None]
            n = self._solo_gdn = len(lin) if lin and all(getattr(m, "_gdn_ok", False) for m in lin) else 0
        rec = [c for c in cache if isinstance(c, ArraysCache)]
        # every recurrent cache must be a fused GDN layer's: LFM2's convolutions (one state) and Mamba layers
        # record nothing to rewind to (they crashed every request in flight on the first rejected draft)
        if not n or len(rec) != n or not fuse.USE_GDN_KERNEL:
            return False
        return all(len(c.cache) >= 2 and c.cache[1] is not None and c.cache[1].dtype == mx.float32 and c.left_padding is None
                   and c.lengths is None for c in rec)

    def _solo_draft(self, r, cache) -> List[int]:
        """A prompt-lookup draft for an unconstrained row: what followed the most recent earlier occurrence of its
        last 3 tokens (locals/lookup.py's rule, from an index kept on the request instead of a scan of the whole
        prompt every step); its width follows the recent acceptance."""
        constrained = isinstance(r.controller, LocalsController)
        if (hasattr(r.controller, "step_config") and not constrained) or not self._solo_drafts_ok(cache):
            return []
        if len(r.out) < getattr(r, "solo_resume", 0):  # recent drafts were rejected: plain pipelined steps for a while
            return []
        stream = list(r.prompt) + r.out
        idx = getattr(r, "lk_index", None)
        if idx is None:
            idx, r.lk_index, r.lk_done = {}, {}, 2
            idx = r.lk_index
        for i in range(r.lk_done, len(stream) - 1):  # keys ending before the last token: earlier occurrences only
            idx[(stream[i - 2], stream[i - 1], stream[i])] = i + 1
        r.lk_done = max(r.lk_done, len(stream) - 1)
        j = idx.get(tuple(stream[-3:])) if len(stream) >= 3 else None
        if j is None:
            return []
        acc = getattr(r, "solo_acc", None) or []
        width = 4 if not acc else max(1, min(8, int(round(2 * sum(acc[-8:]) / len(acc[-8:]) + 1))))
        eos = set(r.controller.eos_ids) or set(self.tokenizer.eos_token_ids)
        draft = stream[j:j + width]
        for i, tok in enumerate(draft):
            if tok in eos:
                draft = draft[:i]
                break
        if constrained and draft:  # only the part the grammar allows (an agent's tool call: its arguments)
            cfg, kept = r.controller.config(), []
            for tok in draft:
                cfg = r.controller.step_config(cfg, tok)
                if cfg is None:
                    break
                kept.append(tok)
                if r.controller.finished_at(cfg):
                    break
            draft = kept if len(kept) >= 2 else []  # one token under a grammar is too often a repeated key's start
        return draft[: max(0, r.max_tokens - len(r.out) - 1)]

    def _solo_verify(self, logits: mx.array, draft: List[int], temp: float, r=None) -> List[int]:
        """The accepted draft tokens and the token after them. Greedy: draft tokens while each equals the argmax.
        Sampled: draft token d is kept with probability p(d) under the request's distribution, else the token is
        drawn from p without d (speculative sampling: the tokens follow p exactly). ``r``: the request, for its own
        top_p, top_k and seed."""
        if not temp:
            preds = mx.argmax(logits, axis=-1).tolist()
            out = []
            for j, d in enumerate(draft):
                if preds[j] != d:
                    return out + [preds[j]]
                out.append(d)
            return out + [preds[len(draft)]]
        from .engine import _filtered

        top_p = self.top_p if getattr(r, "top_p", None) is None else r.top_p
        top_k = self.top_k if getattr(r, "top_k", None) is None else r.top_k
        lp = _filtered(logits.astype(mx.float32), top_p, top_k) * (1.0 / temp)
        p = mx.softmax(lp, axis=-1)
        k = len(draft)
        k1 = k2 = None
        if getattr(r, "seed", None) is not None:
            k1, k2 = mx.random.split(row_key(r.seed, r.draws))
            r.draws += 1

        def draw(x):
            return int((mx.random.categorical(x) if k2 is None else mx.random.categorical(x, key=k2)).item())

        u = mx.random.uniform(shape=(k,)) if k1 is None else mx.random.uniform(shape=(k,), key=k1)
        keep = (u < p[mx.arange(k), mx.array(draft)]).tolist()
        out = []
        for j, d in enumerate(draft):
            if not keep[j]:
                rest = mx.where(mx.arange(lp.shape[-1]) == d, NEG_INF, lp[j])
                return out + [draw(rest)]
            out.append(d)
        return out + [draw(lp[k])]

    def _solo_masked(self, r, logits: mx.array, g, draft: List[int]) -> mx.array:
        """Logits at the sampled position and each draft position under the grammar's state there (as _pick
        masks one-token steps): the state after the fed tokens, then after each draft token."""
        states = [g]
        for tok in draft:
            states.append(self._advance([r], mx.array([tok], dtype=mx.int32), states[-1]))
        G = mx.concatenate(states)
        allowed, cover, tlen, offs = self._tab
        m = allowed[G] & (tlen[None, :] >= cover[G][:, None])
        left = r.max_tokens - len(r.out)
        bms = [r.controller.budget_mask(G[i] - offs[id(r.controller.c)], left - i) for i in range(len(states))]
        if any(b is not None for b in bms):
            ones = mx.ones((logits.shape[-1],), dtype=mx.bool_)
            m = m & mx.stack([ones if b is None else b for b in bms])
        return mx.where(m, logits, NEG_INF)

    def _solo_rewind(self, cache, drop: int) -> None:
        """Drop the last ``drop`` fed tokens of a batch-of-one cache (recurrent layers: the state the fused
        kernel recorded after the last kept token)."""
        for c in cache:
            if isinstance(c, ArraysCache):
                ts = getattr(c, "token_states", None)
                if ts is None:
                    raise RuntimeError("a recurrent layer recorded no per-token states to rewind to")
                keep = ts.shape[0] - drop
                old, qkv = c.token_conv
                n_keep = old.shape[1]
                c.cache[1] = ts[keep - 1]
                c.cache[0] = mx.contiguous(mx.concatenate([old, qkv[:, :keep]], axis=1)[:, -n_keep:])
            else:
                c.trim(drop)

    def _solo(self, r, cache, x: int, g, temp: float, arrive, dead: set):
        """Steps for the only running row while it has a forced span or a draft to feed. Returns (requests that
        arrived meanwhile, the next token to feed, its constraint state); x None: the row is done."""
        t_r = temp if getattr(r, "temp", None) is None else r.temp
        constrained = isinstance(r.controller, LocalsController)
        while True:
            got = arrive(False)
            if got:
                return got, x, g
            span = list(r.pend)
            draft = [] if span else self._solo_draft(r, cache)
            if not span and not draft:
                return [], x, g
            r.pend = []
            toks = [x] + span + draft
            for c in cache:
                if isinstance(c, ArraysCache):
                    c.want_states = bool(draft)
                    c.token_states = None
            try:
                h = self.lm.model(mx.array([toks], dtype=mx.int32), cache)
            finally:
                for c in cache:
                    if isinstance(c, ArraysCache):
                        c.want_states = False
            self.counters["steps"] += 1
            self.counters["rows"] += 1
            self.counters["solo_steps"] = self.counters.get("solo_steps", 0) + 1
            self._growth_trim(1, 64)  # (a pass feeds up to 33 tokens)
            logits = head_logits(self.lm, h[:, len(span):, :])[0]  # the last certain token, then each draft token
            gs = g
            if constrained and span:
                for tok in span:
                    gs = self._advance([r], mx.array([tok], dtype=mx.int32), gs)
            r.fed.extend(span)
            if draft:
                picked = self._solo_verify(self._solo_masked(r, logits, gs, draft) if constrained else logits, draft, t_r, r)
                a = len(picked) - 1
            else:
                y = self._pick(logits[-1:], gs if constrained else None, t_r, ([r], [r.max_tokens - len(r.out)]))
                picked, a = [int(y.item())], 0
            n0 = len(r.out)
            kept = a  # draft tokens that stay in the cache
            nxt = None  # the next token to feed
            for j, tok in enumerate(picked):
                r.passes += 1
                if self._commit_drip(r, tok):
                    r.text = self.tokenizer.decode(r.out)
                    dead.add(id(r))
                    kept = min(kept, j)  # the draft tokens after the last one committed stay out of the cache
                    break
                if j < a and r.pend:  # the grammar forces a span after this draft token: the rest of the draft goes
                    kept = j + 1
                    nxt = r.pend.pop(0)
                    break
            if draft:
                if kept < len(draft):
                    self._solo_rewind(cache, len(draft) - kept)
                for c in cache:  # the per-token states (48 MB a token on Qwen3.5-4B) are not kept past the pass
                    if isinstance(c, ArraysCache):
                        c.token_states = c.token_conv = None
                r.fed.extend(draft[:kept])
                r.solo_acc = (getattr(r, "solo_acc", None) or []) + [kept]
                self.counters["drafted"] = self.counters.get("drafted", 0) + kept
                # a draft step costs the pipelining of that step: after a draft rejected outright, none for 1 token,
                # doubling up to 32 while they keep failing (a JSON list's repeated keys: 0.83x without this)
                r.solo_miss = getattr(r, "solo_miss", 0) + 1 if kept == 0 else 0
                if r.solo_miss:
                    r.solo_resume = len(r.out) + min(32, 1 << (r.solo_miss - 1))
            self.tokens_out += len(r.out) - n0
            self._after_commit(r, n0, dead)
            if id(r) in dead:
                return [], None, None
            if nxt is None:
                nxt = picked[-1]
                g_after = gs
                if constrained and draft:  # the state after the kept draft tokens
                    for tok in draft[:kept]:
                        g_after = self._advance([r], mx.array([tok], dtype=mx.int32), g_after)
            else:
                g_after = gs
                for tok in draft[:kept]:
                    g_after = self._advance([r], mx.array([tok], dtype=mx.int32), g_after)
            x = nxt
            r.fed.append(x)
            g = self._advance([r], mx.array([x], dtype=mx.int32), g_after) if constrained else g

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

    def _launch(self, cache, x: mx.array, g, temp: float, budget, filling: List[dict], lru, lru_key) -> mx.array:
        """The next step: a plain decode step, or, while a prompt is being prefilled (``filling``), a
        mixed pass with its next chunk (locals/mixed.py). ``self._fill_eval``: what the caller evaluates
        with the step (the filling cache)."""
        # the prompt with the fewest tokens left first: a 3k-token request that arrived during a 31k-token one waited
        # for all of it (a minute) in arrival order; the long one loses the short one's few seconds
        e = min((e for e in filling if e["pos"] < e["end"]), key=lambda e: e["end"] - e["pos"], default=None)
        if e is None:
            self._fill_eval = []
            return self._launch1(cache, x, g, temp, budget)
        from .mixed import forward

        r, c = e["r"], e["cache"]
        stop = e["cps"][0][0] if e["cps"] else e["end"]
        if e["reserved"] < stop and any(isinstance(l, KVCache) and l.keys is not None for l in c):
            self._fill_eval = []  # (it holds the old buffers, which then lived on next to the grown ones)
            _reserve(c, stop)  # the KV buffers for the prompt (or up to its next checkpoint) at once
            e["reserved"] = stop
        # the pass's rows (decode rows and prompt tokens) a whole number of matmul tiles: one row past
        # a multiple of 32 or 64 costs a whole tile more, more than the decode step the pass saves
        n = min(self.mix_chunk - int(x.shape[0]), stop - e["pos"], prefill_chunk(e["pos"], 1, self.mix_chunk, self._prefill_scores))
        n = max(n, 1)
        h = forward(self.lm.model, x[:, None], cache, mx.array([r.prompt[e["pos"]: e["pos"] + n]]), c)
        out = self._pick(self._head(h), g, temp, budget)
        e["pos"] += n
        self.counters["mixed_steps"] = self.counters.get("mixed_steps", 0) + 1
        self.counters["prefill_tokens"] += n
        self._fill_eval = [a for l in c for a in (getattr(l, "keys", None), getattr(l, "values", None), *(getattr(l, "cache", None) or []))
                           if isinstance(a, mx.array)]
        if e["cps"] and e["pos"] == e["cps"][0][0]:
            mx.eval(out, self._fill_eval)
            self._checkpoint(e, lru, lru_key)
        if _MIXLOG:
            now = time.perf_counter()
            print(f"[mix] rows {int(x.shape[0])} + {n} prompt tokens (at {e['pos'] - n} of {e['end']}), "
                  f"{(now - getattr(self, '_mix_t', now)) * 1000:.0f} ms since the last pass", flush=True)
            self._mix_t = now
        return out

    def _checkpoint(self, e: dict, lru, lru_key) -> None:
        """A filling request reached its next checkpoint: keep a copy of its cache there."""
        cp, kind = e["cps"].pop(0)
        while e["cps"] and e["cps"][0][0] <= cp:  # one snapshot per position
            e["cps"].pop(0)
        r, c = e["r"], e["cache"]
        if kind == "reply":
            self._keep_reply(r, cp, c)
            return
        if not self._room_for_copy(lru, len(r.prompt)):
            return
        lru.insert_cache(lru_key, list(r.prompt[:cp]), copy.deepcopy(c), cache_type=kind)
        if kind == "system" and self.disk is not None:
            self.disk.save(list(r.prompt[:cp]), c)
        if os.environ.get("LOCALS_PREFILL_LOG"):
            print(f"[checkpoint] {kind} at {cp} (mixed; disk {'on' if self.disk is not None else 'off'})", flush=True)

    def _fill_checkpoints(self, e: dict, lru, lru_key, stop=None) -> bool:
        """Prefill a filling request up to its remaining checkpoints at once (no rows decode alongside). False if
        ``stop()`` (asked between chunks) gave it up."""
        while e["cps"]:
            cp = e["cps"][0][0]
            if not self._run(e["cache"], list(e["r"].prompt[e["pos"]:cp]), stop=stop):
                return False
            e["pos"] = cp
            self._checkpoint(e, lru, lru_key)
        return True

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
        # Requests whose prompts are prefilled in mixed passes, a chunk per step, while the rows decode
        # (locals/mixed.py): each {"r", "cache", "pos", "end", "cps", "reserved"}, joined once complete.
        filling: List[dict] = []
        self._fill_eval = []
        if getattr(self, "_mix_ok", None) is None:
            from . import mixed

            self._mix_ok = self.mix_chunk > 0 and mixed.supported(self.lm.model)
        dead: set = set()
        cache, g, cur = None, None, None  # g: states masking the in-flight sample
        steps = nrows = 0
        prefill_s = 0.0
        self.tokens_out = 0
        self.counters = {"steps": 0, "rows": 0, "prefill_s": 0.0, "prefill_tokens": 0, "admissions": 0, "idle_s": 0.0}
        self._next_trim = 0  # (the counters start again: a restart after an engine error kept the old step count)
        self._serve_lru = lru
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
                reply = _reply_entry(r, row)
                if reply is None:
                    lru.insert_cache(lru_key, list(r.prompt[:-1]) + r.fed, row)
                else:
                    # where the next request continues instead of the whole row: the template writes the reply
                    # differently from the open turn, so the row is no prefix of the next request; and its KV,
                    # views of the row's buffers, would count twice in the prompt cache's bytes next to the row
                    lru.insert_cache(lru_key, list(r.prompt[: r.reply_state[0]]), reply, cache_type="assistant")
                r.reply_state = None
            if on_done is not None:
                on_done(r)

        def attach(new, merged, tp, count):
            """Join prefilled requests (their caches ``merged``, a batch of them) to the running rows."""
            nonlocal cache, g, rows, xs, prefill_s
            old = self._table_add([r.controller for r in rows + new])
            if old is not None and g is not None:  # the table was rebuilt: the running rows' states move
                offs = self._tab[3]
                g = g + mx.array([offs[id(r.controller.c)] - old[id(r.controller.c)]
                                  if isinstance(r.controller, LocalsController) else 0 for r in rows], dtype=g.dtype)
            if cache is None:
                cache = merged
            else:
                _append_rows(cache, merged)
            mx.eval([c.state for c in cache])
            prefill_s += time.perf_counter() - tp
            self.counters["prefill_s"] += time.perf_counter() - tp
            self.counters["admissions"] += 1
            if _MEMLOG:
                print(f"[mem] admitted {[len(r.prompt) for r in new]} (cached {[r.cached_tokens for r in new]}): active "
                      f"{mx.get_active_memory() / 1e9:.2f} GB, peak {mx.get_peak_memory() / 1e9:.2f} GB, "
                      f"prompt cache {lru.nbytes / 1e9 if lru is not None else 0:.2f} GB", flush=True)
                _reset_peak()
            if count:  # (mixed passes count their chunks as they go)
                self.counters["prefill_tokens"] += sum(len(r.prompt) - 1 - getattr(r, "cached_tokens", 0) for r in new)
            gn = mx.array([self._gstate(r) for r in new], dtype=mx.int32)
            g = gn if g is None else mx.concatenate([g, gn])
            rows = rows + new
            xs = xs + [r.prompt[-1] for r in new]
            for r in new:
                r.fed.append(r.prompt[-1])

        xs: List[int] = []
        solo_next = False  # the only row has a forced span or a draft: a boundary, then steps of several tokens
        while True:
            queue += arrive(False)
            if not rows and not queue and cur is None and not filling:
                # idle: the combined tables (which only grow) start again from the next requests' constraints
                self._tab = self._otab = self._walk = self._wtab = self._wrows = None
                self._fill_eval = []
                self.active = 0
                got = arrive(True)
                if not got:
                    break
                queue += got
            self.active = len(rows)
            boundary = cur is None or dead or solo_next or (queue and len(rows) + len(filling) < self.n_slots) or (
                filling and any(e["pos"] >= e["end"] or getattr(e["r"], "cancel", False) for e in filling))
            solo_next = False
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
                gone = [e for e in filling if getattr(e["r"], "cancel", False)]  # their clients left during the prefill
                for e in gone:
                    filling.remove(e)
                    e["r"].finish = "cancelled"
                    if on_done is not None:
                        on_done(e["r"])
                if gone:
                    self._fill_eval = []  # (the next step sets it again)
                if filling and not rows:  # no rows decode alongside any more: the rest of the prompts at once
                    tp = time.perf_counter()
                    ready, filling[:] = list(filling), []
                    self._fill_eval = []
                    self.counters["prefill_tokens"] += sum(e["end"] - e["pos"] for e in ready)  # mixed passes counted the rest
                    # every client gone (Esc while its long prompt is read): the reading stops between chunks, as at
                    # an admission, instead of going on to the prompt's end for nobody with the queue waiting
                    left_all = lambda: all(getattr(e["r"], "cancel", False) for e in ready)  # noqa: E731
                    merged = None
                    if all(self._fill_checkpoints(e, lru, lru_key, stop=left_all) for e in ready) and not left_all():
                        merged = self._prefill_batch(
                            [e["r"].prompt[:-1] for e in ready], [], starts=[(e.pop("cache"), list(e["r"].prompt[e["pos"]:e["end"]])) for e in ready],
                            stop=left_all)
                    if merged is not None:
                        attach([e["r"] for e in ready], merged, tp, count=False)
                    else:
                        for e in ready:
                            e["r"].finish = "cancelled"
                            if on_done is not None:
                                on_done(e["r"])
                        self.counters["prefills_given_up"] = self.counters.get("prefills_given_up", 0) + 1
                    del ready, merged
                # a for-loop variable outlives its loop: it would hold a cancelled or joined request's cache
                e = gone = None
                if any(e["pos"] >= e["end"] for e in filling):  # prefilled in mixed passes: join the batch
                    tp = time.perf_counter()
                    ready = [e for e in filling if e["pos"] >= e["end"]]
                    filling[:] = [e for e in filling if e["pos"] < e["end"]]
                    self._fill_eval = []  # (it holds the last-read prompt's arrays; the next step sets it again)
                    attach([e["r"] for e in ready], self._prefill_batch(
                        [e["r"].prompt[:-1] for e in ready], [], starts=[(e.pop("cache"), []) for e in ready]), tp, count=False)
                    del ready
                new = []
                while queue and len(rows) + len(filling) + len(new) < self.n_slots:
                    # a request that would not fit next to the running rows waits for one to finish
                    held0 = self._merge_tokens(rows, [len(e["r"].prompt) for e in filling] + [len(q.prompt) for q in new + [queue[0]]])
                    if (rows or new or filling) and not self._fits(lru, queue[0], held0 + sum(len(q.prompt) for q in new),
                                                                   rows=len(new) + 1, tokens=sum(len(q.prompt) for q in new + [queue[0]])):
                        if not getattr(queue[0], "held", False):  # (once a request: while it waits, every step asks)
                            queue[0].held = True
                            self.counters["deferred"] = self.counters.get("deferred", 0) + 1
                        break
                    new.append(queue.pop(0))
                if new:
                    tp = time.perf_counter()
                    if _MEMLOG:
                        print(f"[mem] before admission: active {mx.get_active_memory() / 1e9:.2f} GB, peak since last "
                              f"{mx.get_peak_memory() / 1e9:.2f} GB, rows {len(rows)}", flush=True)
                        _reset_peak()
                    starts = []
                    found = []
                    for r in new:  # every cached prefix first: making room for one must not drop the next one's
                        if lru is not None and self.disk is not None:  # a system prompt saved before a restart
                            self.disk.restore(lru, lru_key, list(r.prompt[:-1]))
                        c0, rest = (lru.fetch_nearest_cache(lru_key, list(r.prompt[:-1])) if lru is not None else (None, list(r.prompt[:-1])))
                        r.cached_tokens = len(r.prompt) - 1 - len(rest) if c0 is not None else 0
                        found.append((c0, rest))
                    # each request's cached prefix is a lazy copy, made during the prefill: the whole prompt counts
                    self._make_room(lru, sum(len(r.prompt) + len(rest) * _copies(r)
                                             + _replies(r) * self._reply_bytes // max(1, self._kv_bytes_per_token())
                                             for r, (_, rest) in zip(new, found)), 0,
                                    self._merge_tokens(rows, [len(e["r"].prompt) for e in filling] + [len(r.prompt) for r in new]),
                                    rows=len(new))
                    if self._mix_ok and rows and sum(len(rest) for _, rest in found) > self.mix_chunk:
                        # the running rows decode on while these prompts are prefilled, a chunk in each step. Prompts
                        # that fit one pass are prefilled together at once instead: the rows would get a single token
                        # meanwhile, and one batch prefills several prompts faster than passes one after another.
                        for r, (c0, rest) in zip(new, found):
                            end = len(r.prompt) - 1
                            cps = {}  # one snapshot per position, the first kind in sorted order (as below)
                            for cp, kind in sorted(getattr(r, "checkpoints", None) or []):
                                if kind == "reply" and lru is not None and c0 is not None and cp == end - len(rest):
                                    self._keep_reply(r, cp, c0)  # the cached prefix ends there (a regenerated reply)
                                elif lru is not None and end - len(rest) < cp < end:
                                    cps.setdefault(cp, kind)
                            filling.append({"r": r, "cache": c0 if c0 is not None else make_prompt_cache(self.model),
                                            "pos": end - len(rest), "end": end, "reserved": 0, "cps": sorted(cps.items())})
                        new, found = [], []
                    # every client of these prompts gone (Esc in the agent while it waits for a long prompt): the
                    # prefill stops between chunks instead of reading the prompts to their end for nobody
                    left_all = lambda: all(getattr(q, "cancel", False) for q in new)  # noqa: E731
                    for r, (c0, rest) in zip(new, found):
                        if left_all():
                            break
                        done = r.cached_tokens
                        for cp, kind in sorted(getattr(r, "checkpoints", None) or []):
                            if kind == "reply" and lru is not None and c0 is not None and cp == done:
                                self._keep_reply(r, cp, c0)  # the cached prefix ends there (a regenerated reply)
                                continue
                            if lru is None or not done < cp < len(r.prompt) - 1:
                                continue
                            # prefill to the checkpoint, keep a copy, go on from there
                            c0 = c0 if c0 is not None else make_prompt_cache(self.model)
                            if not self._run(c0, list(r.prompt[done:cp]), stop=left_all):
                                break
                            if kind == "reply":
                                self._keep_reply(r, cp, c0)
                            elif self._room_for_copy(lru, len(r.prompt)):
                                lru.insert_cache(lru_key, list(r.prompt[:cp]), copy.deepcopy(c0), cache_type=kind)
                            if kind == "system" and self.disk is not None:
                                self.disk.save(list(r.prompt[:cp]), c0)
                            if os.environ.get("LOCALS_PREFILL_LOG"):
                                print(f"[checkpoint] {kind} at {cp} (disk {'on' if self.disk is not None else 'off'})", flush=True)
                            done, rest = cp, list(r.prompt[cp:-1])
                        starts.append((c0, rest) if c0 is not None else None)
                    merged = None
                    found = c0 = rest = None  # (the caches are handed to the prefill in starts)
                    if new and not left_all():
                        merged = self._prefill_batch([r.prompt[:-1] for r in new], [q.prompt[:-1] for q in queue], starts=starts,
                                                     stop=left_all)
                    if merged is not None:
                        attach(new, merged, tp, count=True)
                    elif new:
                        for q in new:
                            q.finish = "cancelled"
                            if on_done is not None:
                                on_done(q)
                        self.counters["prefills_given_up"] = self.counters.get("prefills_given_up", 0) + 1
                    del merged
                    # the per-request prefill caches: merged copies now live in the batch (a name left bound here
                    # would hold its request's whole KV next to the batch's copy for as long as it decodes)
                    starts = found = c0 = rest = None
                if not rows:
                    cur = None
                    continue
                if g is None:
                    g = mx.array([self._gstate(r) for r in rows], dtype=mx.int32)
                if len(rows) == 1 and not queue and not filling and self._solo_ok(rows[0], cache):
                    got, x1, g1 = self._solo(rows[0], cache, xs[0], g, temp, arrive, dead)
                    if x1 is None:  # done: it is retired at the next boundary
                        cur = None
                        continue
                    queue += got
                    xs, g = [x1], g1
                free = not any(isinstance(r.controller, LocalsController) or type(r.controller).__name__ == "WalkController" for r in rows)
                # A row's next committed sample follows exactly the tokens it has
                # fed after its prompt, so out has len(fed) - 1 tokens then.
                left = [r.max_tokens - (len(r.fed) - 1) for r in rows]
                cur = self._launch(cache, mx.array(xs, dtype=mx.int32), None if free else g, temp, (rows, left), filling, lru, lru_key)
                mx.async_eval(cur, g, self._fill_eval)
                steps += 1
                nrows += len(rows)
                self.counters["steps"] += 1
                self.counters["rows"] += len(rows)
                self._growth_trim(len(rows))  # (every step is a boundary while a request waits for memory)
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
            nxt = self._launch(cache, x, None if free else g2, temp, (rows, left), filling, lru, lru_key)
            mx.async_eval(nxt, g2, self._fill_eval)
            steps += 1
            nrows += len(rows)
            self.counters["steps"] += 1
            self.counters["rows"] += len(rows)
            self._growth_trim(len(rows))
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
            if _SOLO and len(rows) == 1 and not queue and not filling and not dead and self._solo_ok(rows[0], cache):
                # entering costs about a step (the pipeline drains): a forced span of 3 or more tokens, or a draft
                solo_next = len(rows[0].pend) >= 3 or bool(self._solo_draft(rows[0], cache))
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
