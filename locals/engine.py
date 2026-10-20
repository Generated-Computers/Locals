"""MLX generation loop with pluggable constraint controllers.

A controller masks logits, consumes each sampled token, and may return forced
tokens to append in the same forward pass as the sampled token. Appending m
known tokens costs one weight-streaming pass instead of m decode steps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache, make_prompt_cache

from .constraint import CompiledConstraint
from .forced import ForcedPlanner
from .kernels import forward_hidden, forward_last, head_dense, head_rows, text_model
from .regex import DEAD
from .vocab import TokenVocab

NEG_INF = float("-inf")


def _filtered(logits: mx.array, top_p: float = 0.0, top_k: int = 0) -> mx.array:
    """Log-probabilities after mlx-lm's filters, in its order: top-p on the untempered
    distribution, then top-k (mlx_lm.sample_utils.make_sampler)."""
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    if 0 < top_p < 1:
        from mlx_lm.sample_utils import apply_top_p

        lp = apply_top_p(lp, top_p)
    if top_k and top_k < lp.shape[-1]:
        from mlx_lm.sample_utils import apply_top_k

        lp = apply_top_k(lp, top_k)
    return lp


def sample_logits(logits: mx.array, temp: float, top_p: float = 0.0, top_k: int = 0, key=None) -> mx.array:
    """One token per row of ``logits`` (any leading shape): the argmax at temperature 0, else
    mlx-lm's rule (top-p, then top-k, on the untempered distribution; then a categorical draw at
    ``temp``), so a sampled run of ours and one of stock mlx-lm follow the same distribution (Qwen3.5
    recommends top_k 20 / top_p 0.95 in thinking mode, where greedy decoding loops)."""
    if temp == 0:
        return mx.argmax(logits, axis=-1)
    lp = _filtered(logits, top_p, top_k) * (1.0 / temp)
    return mx.random.categorical(lp, key=key) if key is not None else mx.random.categorical(lp)


def sample_rows(logits: mx.array, temps, top_p: float = 0.0, top_k: int = 0) -> mx.array:
    """One token per row of ``logits`` (B, V), row i at temperature ``temps[i]``
    (0: its argmax), the same top-k / top-p filters for every row."""
    t = [float(x) for x in temps]
    if all(x == t[0] for x in t):
        return sample_logits(logits, t[0], top_p, top_k)
    greedy = mx.argmax(logits, axis=-1)
    if not any(t):
        return greedy
    scale = mx.array([x if x > 0 else 1.0 for x in t], dtype=mx.float32)[:, None]
    drawn = mx.random.categorical(_filtered(logits.astype(mx.float32), top_p, top_k) / scale)
    return mx.where(mx.array([x == 0 for x in t]), greedy, drawn)


class ConstraintExhausted(RuntimeError):
    """No token is allowed in the current configuration (a pushdown
    constraint at its depth limit): decoding stops with finish="exhausted"."""

# Per constraint and DFA state: (visits where a forced span was possible, visits
# where one followed). The pipelined engine dispatches the next body early in
# states that rarely force.
import weakref as _weakref

_FORCE_STATS = _weakref.WeakKeyDictionary()  # one entry per constraint (or controller), gone with it


def _force_stats(owner) -> dict:
    try:
        return _FORCE_STATS.setdefault(owner, {})
    except TypeError:  # not weakly referable: statistics for this run only
        return {}


EARLY_BODY = True


class Controller:
    eos_ids: Sequence[int] = ()
    supports_gpu = True  # masks and state can be computed lazily on the GPU

    def gpu_state(self):
        return None

    def gpu_mask(self, qg):
        return None

    def gpu_next(self, y, qg):
        return None

    def may_force(self) -> bool:
        """Whether some token allowed now can lead to a forced span."""
        return False

    def apply(self, logits: mx.array) -> mx.array:
        return logits

    def accept(self, token: int) -> None:
        pass

    def forced(self, token: int) -> Tuple[List[int], bool]:
        """Tokens to feed after ``token``; the flag says they replace it."""
        return [], False

    def finished(self) -> bool:
        return False

    # Budget-aware masking hooks (LocalsController implements them).
    def budget_mask(self, qg, r: Optional[int]):
        return None

    def budget_ids(self, q: int, r: Optional[int], ids):
        return ids

    def heal_ok(self, r: int) -> bool:
        return True


class LocalsController(Controller):
    """Our constraint: GPU mask table, host DFA state, canonical forced spans.

    ``canonical`` enables the boundary rule: at a state whose remaining forced
    bytes are one pre-token block, the next token must cover the whole block.
    """

    def __init__(
        self,
        constraint: CompiledConstraint,
        planner: Optional[ForcedPlanner],
        use_forced: bool = True,
        canonical: bool = False,
    ):
        self.c = constraint
        self.planner = planner
        self.use_forced = use_forced and planner is not None
        self.canonical = canonical and planner is not None
        self.allowed = _device_table(constraint)
        self.eos_ids = constraint.eos_ids
        self.q = constraint.dfa.start
        if self.canonical:
            self.tok_len = _token_lengths(planner.vocab)

    def apply(self, logits):
        m = self.allowed[self.q]
        if self.canonical:
            cover = self.planner.cover_rule(self.q)
            if cover:
                m = m & (self.tok_len >= cover)
        return mx.where(m, logits, NEG_INF)

    def accept(self, token):
        nq = int(self.c.next_state[token, self.q])
        if nq == DEAD:
            raise RuntimeError(f"token {token} not allowed at state {self.q}")
        self.q = nq

    def forced(self, token, heal: bool = True):
        if not self.use_forced:
            return [], False
        plan = self.planner.plan(token if heal else None, self.q)
        if plan is None:
            return [], False
        self.q = plan.q_end
        return plan.tokens, plan.replaces_prev

    def finished(self):
        return bool(self.c.only_eos[self.q])

    # GPU-resident state: the DFA state lives on the device, so the next
    # step's mask needs no host round trip.
    def gpu_state(self):
        return mx.array(self.q, dtype=mx.int32)

    def gpu_mask(self, qg):
        m = self.allowed[qg]
        if self.canonical:
            m = m & (self.tok_len >= self._gpu_cover()[qg])
        return m

    def gpu_next(self, y, qg):
        return self._gpu_T()[y, qg].astype(mx.int32)

    def may_force(self):
        return self.use_forced and bool(_may_force(self.c)[self.q])

    SPARSE_MAX = 4096

    def sparse_ids(self, q: int):
        """Allowed token ids at state q (boundary rule applied) as a device
        array, or None when too many are allowed for a sparse head."""
        k = ("sparse", id(self.c), id(self.planner), self.canonical, q)
        if k not in _TABLES:
            ids = np.nonzero(self.c.allowed[q])[0]
            if self.canonical:
                cov = self.planner.cover_rule(q)
                if cov:
                    ids = ids[self._tok_len_np()[ids] >= cov]
            _TABLES[k] = ((self.c, self.planner), mx.array(ids.astype(np.int32)) if 0 < len(ids) <= self.SPARSE_MAX else None)
        return _TABLES[k][1]

    def _tok_len_np(self):
        return _tok_len_np(self.planner.vocab)

    def _gpu_T(self):
        k = ("T", id(self.c))
        if k not in _TABLES:
            _TABLES[k] = (self.c, mx.array(_host_T(self.c)))
            _TABLES.pop(("T_host", id(self.c)))  # (the device's copy is the one read)
        return _TABLES[k][1]

    def _gpu_cover(self):
        k = ("cover", id(self.planner))
        if k not in _TABLES:
            cov = np.array([self.planner.cover_rule(q) for q in range(self.c.n_states)], dtype=np.int32)
            _TABLES[k] = (self.planner, mx.array(cov))
        return _TABLES[k][1]

    # Budget-aware masking (locals/budget.py): with r tokens left, only tokens
    # after which a valid completion still fits. Where even that is impossible
    # (need(q) > r, e.g. max_tokens below the shortest output) the budget mask
    # is dropped, which is the behaviour without it.
    budget = True

    def budget_tables(self):
        k = ("budget", id(self.c), id(self.planner), self.use_forced, self.canonical)
        if k not in _TABLES:
            bt = _host_budget(self.c, self.planner, self.use_forced, self.canonical)
            eos = np.zeros(self.c.next_state.shape[0], dtype=bool)
            eos[list(self.eos_ids)] = True
            dev = (mx.array(bt.cost), mx.array(bt.need), mx.array(eos))
            _TABLES[k] = ((self.c, self.planner), (bt, dev))
        return _TABLES[k][1]

    def budget_active(self, r: Optional[int]) -> bool:
        if not self.budget or r is None:
            return False
        bt, _ = self.budget_tables()
        return r < bt.max_cost or bt.has_inf

    def budget_mask(self, qg, r: Optional[int]):
        """Device bool mask over the vocabulary for r tokens left at device
        state qg, or None when it would allow every allowed token."""
        if not self.budget_active(r):
            return None
        _, (cost, need, eos) = self.budget_tables()
        fits = cost[self._gpu_T()[:, qg].astype(mx.int32)] <= r
        return fits | eos | (need[qg] > r)

    def budget_ids(self, q: int, r: Optional[int], ids):
        """Filter a sparse-head id array (device) the same way, at host state q."""
        if ids is None or not self.budget_active(r):
            return ids
        bt, _ = self.budget_tables()
        if bt.need[q] > r:
            return ids
        k = ("budget_ids", id(self.c), id(self.planner), self.use_forced, self.canonical, q, r)  # costs depend on use_forced
        if k not in _TABLES:
            a = np.array(ids)
            nq = self.c.next_state[a, q]
            ok = np.isin(a, self.eos_ids) | ((nq != DEAD) & (bt.cost[np.where(nq == DEAD, 0, nq)] <= r))
            _TABLES[k] = ((self.c, self.planner), mx.array(a[ok].astype(np.int32)) if ok.any() else ids)
        return _TABLES[k][1]

    def heal_ok(self, r: int) -> bool:
        """Whether a forced span may be planned with healing at r tokens left."""
        return not self.budget or r > self.budget_tables()[0].heal_threshold

    # -- configurations as values (for verifying drafts, locals/lookup.py) ----------
    def config(self):
        return self.q

    def set_config(self, cfg):
        self.q = cfg

    def step_config(self, cfg, token: int):
        """Configuration after ``token`` from ``cfg``, or None if not allowed."""
        if token in self.eos_ids:
            return None
        nq = int(self.c.next_state[token, cfg])
        if nq == DEAD:
            return None
        if self.canonical:
            cov = self.planner.cover_rule(cfg)
            if cov and len(self.planner.vocab.token_bytes[token] or b"") < cov:
                return None
        return nq

    def device_state(self, cfg):
        return mx.array(cfg, dtype=mx.int32)

    def finished_at(self, cfg) -> bool:
        return bool(self.c.only_eos[cfg])


def _tok_len_np(vocab) -> np.ndarray:
    k = ("toklen_np", id(vocab))
    if k not in _TABLES:
        _TABLES[k] = (vocab, np.array([len(b) if b else 0 for b in vocab.token_bytes], dtype=np.int32))
    return _TABLES[k][1]


def _host_T(c: CompiledConstraint) -> np.ndarray:
    """The transition table as the device holds it (int16 where it fits), made on the host."""
    k = ("T_host", id(c))
    if k not in _TABLES:
        T = np.where(c.next_state == DEAD, c.dfa.start, c.next_state)  # (DEAD is unreachable under the mask)
        _TABLES[k] = (c, T.astype(np.int16 if c.n_states < 2**15 else np.int32))
    return _TABLES[k][1]


def _host_budget(c: CompiledConstraint, planner, use_forced: bool, canonical: bool):
    """locals/budget.py's tables for a controller with these settings."""
    from .budget import budget_tables

    use_forced, canonical = use_forced and planner is not None, canonical and planner is not None
    k = ("budget_host", id(c), id(planner), use_forced, canonical)
    if k not in _TABLES:
        tl = _tok_len_np(planner.vocab) if planner is not None else None
        _TABLES[k] = ((c, planner), budget_tables(c, planner if use_forced else None, planner if canonical else None, tl))
    return _TABLES[k][1]


def _may_force(c: CompiledConstraint) -> np.ndarray:
    k = ("may_force", id(c))
    if k not in _TABLES:
        fb = np.array([bool(b) for b in c.forced_bytes] + [False])
        T = np.where(c.next_state == DEAD, c.n_states, c.next_state)
        _TABLES[k] = (c, (fb[T] & c.allowed.T).any(axis=0))
    return _TABLES[k][1]


class _TableCache:
    """Tables computed per constraint (device masks, transition tables, budget tables). Callers store
    ``(owner, table)`` under a key that names the owner by ``id`` (the constraint, its planner, the
    vocabulary; a tuple of them). An entry keeps no strong reference to its owners: it goes when they
    do, so an id is never reused while its entry exists, and a server does not keep every constraint
    it ever saw (a plain dict held them all: gigabytes over a long run). The per-state entries
    ("sparse", "budget_ids", "pda_sparse": one per state and tokens left, they pile up for a
    long-lived grammar) are also kept least recently used under ``LOCALS_TABLE_CACHE_GB`` (0.5)."""

    PER_STATE = ("sparse", "budget_ids", "pda_sparse")

    def __init__(self):
        import collections
        import os
        import threading

        self._d = {}
        self._lru = collections.OrderedDict()  # per-state keys, least recently used first
        self._size = {}
        self._refs = {}
        self.total = 0
        self.limit = int(float(os.environ.get("LOCALS_TABLE_CACHE_GB", "0.5")) * (1 << 30))
        self._lock = threading.RLock()

    @staticmethod
    def _nbytes(v) -> int:
        if isinstance(v, (tuple, list)):
            return sum(_TableCache._nbytes(x) for x in v)
        return int(getattr(v, "nbytes", 0) or 0)

    def _per_state(self, k) -> bool:
        return isinstance(k, tuple) and bool(k) and k[0] in self.PER_STATE

    def __contains__(self, k) -> bool:
        return k in self._d

    def __getitem__(self, k):
        with self._lock:
            v = self._d[k]
            if k in self._lru:
                self._lru.move_to_end(k)
            return v

    def _drop(self, k) -> None:
        with self._lock:
            if self._d.pop(k, None) is not None:
                self.total -= self._size.pop(k, 0)
                self._lru.pop(k, None)
            self._refs.pop(k, None)

    def __setitem__(self, k, v) -> None:
        import weakref

        owner, table = v if isinstance(v, tuple) and len(v) == 2 else (None, v)
        owners = [o for o in (owner if isinstance(owner, tuple) else (owner,)) if o is not None]
        refs = []
        for o in owners:
            try:
                refs.append(weakref.ref(o, lambda _, k=k: self._drop(k)))
            except TypeError:  # not weakly referable: kept with the entry
                refs.append(o)
        with self._lock:
            self._drop(k)
            self._d[k] = (None, table)
            self._refs[k] = refs
            if self._per_state(k):
                n = self._nbytes(table)
                self._size[k] = n
                self.total += n
                self._lru[k] = True
                while self.total > self.limit and len(self._lru) > 1:
                    self._drop(next(iter(self._lru)))

    def get(self, k, default=None):
        with self._lock:
            return self[k] if k in self._d else default

    def pop(self, k) -> None:
        self._drop(k)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._lru.clear()
            self._size.clear()
            self._refs.clear()
            self.total = 0


_TABLES = _TableCache()


def _device_table(constraint):
    k = id(constraint)
    if k not in _TABLES:
        _TABLES[k] = (constraint, mx.array(constraint.allowed))
    return _TABLES[k][1]


_LENS = {}


def _token_lengths(vocab):
    k = id(vocab)
    if k not in _LENS:
        _LENS[k] = (vocab, mx.array(np.array([len(b) if b else 0 for b in vocab.token_bytes], dtype=np.int32)))
    return _LENS[k][1]


class LLGuidanceController(Controller):
    """Baseline: llguidance matcher, optionally with its fast-forward tokens."""

    supports_gpu = False

    def __init__(self, llg_tokenizer, grammar: str, eos_ids, use_ff: bool = False):
        import llguidance
        import llguidance.mlx as llmlx
        import llguidance.numpy as llnp

        self._mlx = llmlx
        self.m = llguidance.LLMatcher(llg_tokenizer, grammar)
        if self.m.is_error():
            raise RuntimeError(self.m.get_error())
        self.mask = llnp.allocate_token_bitmask(1, llg_tokenizer.vocab_size)
        self._fill = llnp.fill_next_token_bitmask
        self.use_ff = use_ff
        self.eos_ids = eos_ids

    def apply(self, logits):
        self._fill(self.m, self.mask, 0)
        return self._mlx.apply_token_bitmask(logits, self.mask)

    def accept(self, token):
        if not self.m.consume_token(token):
            raise RuntimeError(self.m.get_error())

    def forced(self, token):
        if not self.use_ff:
            return [], False
        ff = self.m.compute_ff_tokens()
        if ff:
            self.m.consume_tokens(ff)
        return list(ff), False

    def finished(self):
        return self.m.is_stopped()


def _spec_ok(model, rotating: bool = False) -> bool:
    """Whether the model's caches can be snapshotted and restored (the pipelined
    loop discards a prebuilt step that way); ``rotating``: sliding-window caches
    (Gemma) count too, which can be restored but not rewound past their window."""
    if not hasattr(model, "make_cache"):
        return True
    from mlx_lm.models.cache import ArraysCache, RotatingKVCache

    ok = (KVCache, ArraysCache) + ((RotatingKVCache,) if rotating else ())
    return all(type(c) in ok for c in model.make_cache())


class SpecKVCache(KVCache):
    """KV cache with functional updates, so a built-but-unevaluated step can be
    discarded by restoring the previous array handles (``snapshot/restore``)."""

    _starts = {}

    def __deepcopy__(self, memo):
        # Arrays are never mutated in place, so copies can share them.
        c = SpecKVCache()
        c.keys, c.values, c.offset = self.keys, self.values, self.offset
        return c

    def update_and_fetch(self, keys, values):
        prev = self.offset
        L = keys.shape[2]
        if self.keys is None or prev + L > self.keys.shape[2]:
            B, H, _, dk = keys.shape
            dv = values.shape[3]
            n = (self.step + L - 1) // self.step
            new_k = mx.zeros((B, H, n * self.step, dk), keys.dtype)
            new_v = mx.zeros((B, H, n * self.step, dv), values.dtype)
            if self.keys is None:
                self.keys, self.values = new_k, new_v
            else:
                k, v = self.keys, self.values
                if prev % self.step != 0:
                    k, v = k[..., :prev, :], v[..., :prev, :]
                self.keys = mx.concatenate([k, new_k], axis=2)
                self.values = mx.concatenate([v, new_v], axis=2)
        start = SpecKVCache._starts.get(prev)
        if start is None:
            start = SpecKVCache._starts[prev] = mx.array([0, 0, prev, 0], dtype=mx.int32)
        self.keys = mx.slice_update(self.keys, keys, start, (0, 1, 2, 3))
        self.values = mx.slice_update(self.values, values, start, (0, 1, 2, 3))
        self.offset += L
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


def _snapshot(cache):
    # KV layers: functional arrays + offset. Sliding-window layers: mlx arrays are
    # immutable, so holding the arrays and the write position is a snapshot. Recurrent
    # layers (ArraysCache): mlx-lm replaces their state arrays on update, so the
    # references suffice.
    out = []
    for c in cache:
        if isinstance(c, SpecKVCache):
            out.append((c.keys, c.values, c.offset))
        elif hasattr(c, "_idx"):  # RotatingKVCache
            out.append((c.keys, c.values, c.offset, c._idx))
        else:
            out.append(list(c.cache))
    return out


def _restore(cache, snap):
    for c, sn in zip(cache, snap):
        if isinstance(c, SpecKVCache):
            c.keys, c.values, c.offset = sn
        elif hasattr(c, "_idx"):
            c.keys, c.values, c.offset, c._idx = sn
        else:
            c.cache = list(sn)


def rewind(cache, drop: int) -> None:
    """Drop the last ``drop`` fed tokens of a single-stream cache after a
    multi-token pass: KV layers trim their offset; recurrent layers take the
    state the fused kernel recorded after the last kept token (fuse.py sets
    ``token_states`` and ``token_conv`` when ``want_states`` is on)."""
    if drop <= 0:
        return
    for c in cache:
        if isinstance(c, SpecKVCache):
            c.offset -= drop
        else:
            ts = getattr(c, "token_states", None)
            if ts is None:
                raise RuntimeError("recurrent layer has no per-token states to rewind to")
            keep = ts.shape[0] - drop
            old, qkv = c.token_conv
            n_keep = old.shape[1]
            c.cache[1] = ts[keep - 1]
            c.cache[0] = mx.contiguous(mx.concatenate([old, qkv[:, :keep]], axis=1)[:, -n_keep:])


def set_want_states(cache, want: bool) -> None:
    """Whether the next multi-token pass records per-token recurrent states
    (kept until the next such pass overwrites them, for rewinding)."""
    for c in cache:
        if not isinstance(c, SpecKVCache):
            c.want_states = want


def make_spec_cache(model):
    """The model's cache with every plain KVCache replaced by SpecKVCache."""
    if hasattr(model, "make_cache"):
        base = model.make_cache()
        return [SpecKVCache() if type(c) is KVCache else c for c in base]
    return [SpecKVCache() for _ in model.layers]


def cache_offset(cache) -> int:
    for c in cache:
        if isinstance(c, SpecKVCache):
            return c.offset
    raise ValueError("no KV layer in cache")


def cache_trimmable(cache) -> bool:
    return all(isinstance(c, SpecKVCache) for c in cache)


@dataclass
class GenResult:
    tokens: List[int]
    text: str
    prompt_tokens: int
    prefill_s: float
    decode_s: float
    passes: int
    forced_tokens: int
    pass_sizes: List[int] = field(default_factory=list)
    finish: str = ""
    fed: List[int] = field(default_factory=list)  # every token whose KV is in ``cache``
    cache: Optional[list] = None
    cached_tokens: int = 0  # prompt tokens reused from a prompt cache

    @property
    def tok_per_s(self) -> float:
        return len(self.tokens) / self.decode_s if self.decode_s > 0 else 0.0


class Engine:
    def __init__(self, model, tokenizer, prefill_step: int = 2048, sparse_head: bool = True):
        self.model = model
        self.sparse_head = sparse_head
        self.tokenizer = tokenizer  # mlx_lm TokenizerWrapper
        self.prefill_step = prefill_step

    def _prefill(self, cache, prompt: List[int], start: int = 0) -> None:
        i = start
        while len(prompt) - i > 1:
            n = min(self.prefill_step, len(prompt) - 1 - i)
            self.model(mx.array([prompt[i : i + n]]), cache=cache)
            mx.eval([c.state for c in cache])
            i += n

    def generate(
        self,
        prompt: List[int],
        controller: Optional[Controller] = None,
        max_tokens: int = 256,
        temp: float = 0.0,
        seed: int = 0,
        pipeline: bool = True,
        cache: Optional[list] = None,
        on_token=None,
        sampler=None,
        lookup: bool = False,
        top_p: float = 0.0,
        top_k: int = 0,
    ) -> GenResult:
        """``cache`` may hold KV for a prefix of ``prompt`` (at most len-1 tokens).
        ``lookup``: greedy decoding with prompt-lookup drafts (exact).
        ``on_token(token)`` is called for every output token as soon as it is
        known; ``sampler(logprobs) -> token`` overrides greedy/temperature."""
        ctl = controller or Controller()
        if lookup and cache is None and temp == 0 and sampler is None and ctl.supports_gpu and _spec_ok(self.model):
            return self._generate_lookup(prompt, ctl, max_tokens, cache, on_token)  # (experimental; no cached prefix)
        if pipeline and ctl.supports_gpu and _spec_ok(self.model, rotating=True):
            return self._generate_pipelined(prompt, ctl, max_tokens, temp, seed, cache, on_token, sampler, top_p, top_k)
        # the synchronous engine (host-side masks, caches the pipelined one cannot snapshot): no cached prefix,
        # and the request's temperature, top-p and top-k (a custom sampler needs the pipelined one); the tokens go
        # to on_token when it is done
        res = self._generate_sync(prompt, ctl, max_tokens, temp, seed, top_p, top_k)
        if on_token is not None:
            for t in res.tokens:
                on_token(t)
        return res

    def _generate_pipelined(self, prompt, ctl, max_tokens, temp, seed, cache=None, on_token=None, sampler=None,
                            top_p: float = 0.0, top_k: int = 0) -> GenResult:
        """Pipelined decode with a GPU-resident constraint state.

        While the GPU runs step n, the host builds step n+1 for the case that
        no forced span follows. If no forced span can follow the current
        state, that step is committed before waiting on step n (full overlap).
        Otherwise the host waits, and either commits the prebuilt step or
        discards it (restoring the cache handles) and feeds the forced span.
        """
        eos = set(ctl.eos_ids) or set(self.tokenizer.eos_token_ids)
        reused = 0
        if cache is not None:
            reused = cache_offset(cache)
            if reused >= len(prompt):
                if cache_trimmable(cache):
                    for c in cache:
                        c.trim(reused - len(prompt) + 1)
                    reused = cache_offset(cache)
                else:  # recurrent state cannot be rewound
                    cache, reused = None, 0
        if cache is None:
            cache = make_spec_cache(self.model)
        key = [mx.random.key(seed)]
        t0 = time.perf_counter()
        self._prefill(cache, prompt, start=reused)
        mx.synchronize()
        t1 = time.perf_counter()
        fed = list(prompt[:-1])

        sparse = self.sparse_head and hasattr(ctl, "sparse_ids") and sampler is None  # a custom sampler sees every logit
        self.sparse_steps = 0

        def body(x):
            return forward_hidden(self.model, x, cache)

        def head(h, qg, q_host=None, r=None):
            # r: output tokens left before this step's sample (budget-aware masking).
            ids = ctl.sparse_ids(q_host) if (sparse and q_host is not None) else None
            ids = ctl.budget_ids(q_host, r, ids) if ids is not None else None
            if ids is not None:
                self.sparse_steps += 1
                lg = head_rows(self.model, h, ids)
                if temp == 0:
                    y = ids[mx.argmax(lg)]
                else:
                    key[0], sub = mx.random.split(key[0])
                    y = ids[sample_logits(lg, temp, top_p, top_k, key=sub)]
                return y, ctl.gpu_next(y, qg)
            logits = head_dense(self.model, h)
            m = ctl.gpu_mask(qg)
            bm = ctl.budget_mask(qg, r)
            if bm is not None:
                m = bm if m is None else m & bm
            if m is not None:
                logits = mx.where(m, logits, NEG_INF)
            if sampler is not None:
                y = sampler((logits - mx.logsumexp(logits))[None])[0]
            elif temp == 0:
                y = mx.argmax(logits)
            else:
                key[0], sub = mx.random.split(key[0])
                y = sample_logits(logits, temp, top_p, top_k, key=sub)
            return y, ctl.gpu_next(y, qg)

        def launch(x, qg, q_host=None, r=None):
            return head(body(x), qg, q_host, r)

        y, qn = launch(mx.array([[prompt[-1]]]), ctl.gpu_state(), getattr(ctl, "q", None), max_tokens)
        mx.async_eval(y) if qn is None else mx.async_eval(y, qn)
        fed.append(prompt[-1])
        out: List[int] = []
        passes, forced_n = 1, 0
        sizes = [1]
        finish = "length"
        emitted = [0]

        def emit():
            if on_token is not None:
                while emitted[0] < len(out):
                    on_token(out[emitted[0]])
                    emitted[0] += 1

        stats = _force_stats(getattr(ctl, "c", ctl))
        while True:
            emit()
            can_force = ctl.may_force()
            snap = _snapshot(cache) if can_force else None
            early = False
            if can_force:
                h2 = body(y.reshape(1, 1))  # head is attached once the state is known
                # States that rarely force in practice run the body now, overlapping
                # the host's wait; a forced span then restores the snapshot.
                seen, forced_count = stats.get(ctl.q, (0, 0))
                early = EARLY_BODY and seen >= 3 and forced_count < 0.25 * seen
                if early:
                    mx.async_eval(h2)
                q_before = ctl.q
            else:
                y2, q2 = launch(y.reshape(1, 1), qn, None, max_tokens - len(out) - 1)
                mx.async_eval(y2) if q2 is None else mx.async_eval(y2, q2)
            yv = int(y.item())
            spec_fed = can_force is False  # the prebuilt step is committed and feeds yv
            if spec_fed:
                fed.append(yv)
            if yv in eos:
                finish = "stop"
                break
            try:
                ctl.accept(yv)
            except ConstraintExhausted:
                finish = "exhausted"
                break
            out.append(yv)
            if len(out) >= max_tokens:
                break
            if ctl.finished():
                finish = "stop"
                break
            if not can_force:
                extra, replaces = [], False
            elif ctl.heal_ok(max_tokens - len(out) + 1):
                extra, replaces = ctl.forced(yv)
            else:
                extra, replaces = ctl.forced(yv, heal=False)
            if can_force:
                seen, forced_count = stats.get(q_before, (0, 0))
                stats[q_before] = (seen + 1, forced_count + bool(extra))
            if extra:
                _restore(cache, snap)
                if can_force:
                    del h2
                else:
                    del y2, q2
                feed = list(extra) if replaces else [yv] + list(extra)
                if replaces:
                    out.pop()
                    forced_n += len(extra) - 1
                else:
                    forced_n += len(extra)
                out.extend(extra)
                if ctl.finished():
                    finish = "stop"
                    break
                y, qn = launch(mx.array([feed]), ctl.gpu_state(), getattr(ctl, "q", None), max_tokens - len(out))
                mx.async_eval(y) if qn is None else mx.async_eval(y, qn)
                fed.extend(feed)
                sizes.append(len(feed))
            else:
                if can_force:
                    del snap  # release old cache handles so updates can be done in place
                    y2, q2 = head(h2, qn, getattr(ctl, "q", None), max_tokens - len(out))
                    mx.async_eval(y2) if q2 is None else mx.async_eval(y2, q2)
                    fed.append(yv)
                y, qn = y2, q2
                sizes.append(1)
            passes += 1
        emit()
        if can_force and not spec_fed and snap is not None:
            _restore(cache, snap)  # the prebuilt step was never committed
        mx.synchronize()
        t2 = time.perf_counter()
        assert cache_offset(cache) == len(fed), (cache_offset(cache), len(fed))
        text = self.tokenizer.decode(out)
        return GenResult(out, text, len(prompt), t1 - t0, t2 - t1, passes, forced_n, sizes, finish, fed, cache, reused)

    def _generate_lookup(self, prompt, ctl, max_tokens, cache=None, on_token=None, n: int = 3, k: int = 8) -> GenResult:
        """The pipelined greedy loop with prompt-lookup drafts (locals/lookup.py
        holds the synchronous reference and the rationale).

        A draft is the continuation of the most recent earlier occurrence of
        the stream's last n tokens. It rides on the step the pipeline builds
        before the previous token y is known: that step feeds [y] + draft,
        the draft assuming y equals the continuation's first token. At y's
        sync a false assumption drops the draft (the cache is rewound while
        the pass still runs; position 0 stays valid, so only the pass's extra
        width was spent). A true one is verified at the next sync: the
        accepted prefix is exact greedy output, the rest is rewound to the
        per-token recurrent state the fused kernel recorded. Draft width
        follows recent acceptance; a step without a draft is the ordinary
        pipelined step. A pass that carried a draft is resolved before the
        next one is launched (its accepted length sets the cache position)."""
        from .lookup import _head_all, propose

        eos = set(ctl.eos_ids) or set(self.tokenizer.eos_token_ids)
        constrained = hasattr(ctl, "step_config")
        if cache is None:
            cache = make_spec_cache(self.model)
        t0 = time.perf_counter()
        self._prefill(cache, prompt)
        mx.synchronize()
        t1 = time.perf_counter()
        fed = list(prompt[:-1])
        stream = list(prompt)
        lm = text_model(self.model)
        acc_hist: List[int] = []
        st_ = {"drafted": 0, "draft_passes": 0, "assume_miss": 0}

        def width() -> int:
            if not acc_hist:
                return min(k, 4)
            h = acc_hist[-8:]
            return max(1, min(k, int(round(2 * sum(h) / len(h) + 1))))

        def guess_for(known):
            """Draft tokens after the token being fed. If that token is not
            known yet, the continuation's first token is the assumption."""
            m = width()
            if known is not None:
                return None, propose(stream + [known], n, m, eos)
            cont = propose(stream, n, m + 1, eos)
            return (cont[0], cont[1:]) if cont else (None, [])

        def body(x):
            set_want_states(cache, x.shape[1] > 1)
            h = lm.model(x, cache)
            set_want_states(cache, False)
            return h

        def chain(q0, g):
            """Device constraint states along the draft, starting from q0 (the
            state after the token the pass feeds first)."""
            st = [q0]
            for t in g:
                st.append(ctl.gpu_next(mx.array(t, dtype=mx.int32), st[-1]) if constrained else None)
            return st

        sparse = self.sparse_head and hasattr(ctl, "sparse_ids")

        def head_all(h, states, r, q_host=None):
            if h.shape[1] == 1 and sparse and q_host is not None:
                ids = ctl.budget_ids(q_host, r, ctl.sparse_ids(q_host))
                if ids is not None:
                    lg = head_rows(self.model, h[:, -1:, :], ids)
                    return ids[mx.argmax(lg)].reshape(1)
            logits = _head_all(self.model, h)
            if constrained:
                masks = []
                for i, st in enumerate(states):
                    m = ctl.gpu_mask(st)
                    bm = ctl.budget_mask(st, r - i)
                    if bm is not None:
                        m = bm if m is None else m & bm
                    masks.append(m)
                logits = mx.where(mx.stack(masks), logits, NEG_INF)
            return mx.argmax(logits, axis=-1)

        def x_of(y_dev, g):
            x = y_dev.reshape(1, 1)
            return mx.concatenate([x, mx.array([g], dtype=x.dtype)], axis=1) if g else x

        # --- first pass: the prompt's last token plus a draft of what follows it
        out: List[int] = []
        passes = forced_n = 0
        sizes: List[int] = []
        finish = "length"
        emitted = [0]

        def emit():
            if on_token is not None:
                while emitted[0] < len(out):
                    on_token(out[emitted[0]])
                    emitted[0] += 1

        g = propose(stream, n, width(), eos)  # the prompt's last token is known: no assumption
        if constrained and g:
            cfg, kept = ctl.config(), []
            for t in g:
                cfg = ctl.step_config(cfg, t)
                if cfg is None:
                    break
                kept.append(t)
            g = kept
        st = chain(ctl.gpu_state(), g)  # the state after the prompt's last token is the start state
        preds = head_all(body(mx.array([[prompt[-1]] + g])), st, max_tokens)
        y = preds[0]
        qn = ctl.gpu_next(y, ctl.gpu_state()) if constrained else None  # state after y
        mx.async_eval(preds, y, *([qn] if qn is not None else []))
        fed.append(prompt[-1])
        passes, sizes = 1, [1 + len(g)]
        pend = dict(preds=preds, g=g, states=st)  # the pass whose result is `preds`
        y_known: Optional[int] = None
        stats = _force_stats(getattr(ctl, "c", ctl))
        can_force, spec_fed, snap = False, True, None
        while True:
            emit()
            if pend["g"]:
                # --- a draft pass: resolve it now (its accepted length sets the cache position)
                pv = pend["preds"].tolist()
                g_, states_ = pend["g"], pend["states"]
                a = 0
                while a < len(g_) and pv[a] == g_[a]:
                    a += 1
                acc_hist.append(a)
                st_["draft_passes"] += 1
                st_["drafted"] += a
                rewind(cache, len(g_) - a)
                stop = False
                for t in g_[:a]:
                    fed.append(t)
                    ctl.accept(t)
                    out.append(t)
                    stream.append(t)
                    if ctl.finished():
                        finish, stop = "stop", True
                        break
                    if len(out) >= max_tokens:
                        stop = True
                        break
                if stop:  # resolved: nothing is pending any more (the end must not rewind or restore it again)
                    pend = dict(preds=None, g=[], states=None)
                    can_force = False
                    break
                y_known = pv[a]
                y = mx.array(y_known, dtype=mx.int32)
                qn = ctl.gpu_next(y, states_[a]) if constrained else None
                pend = dict(preds=None, g=[], states=None)
            can_force = ctl.may_force()
            snap = _snapshot(cache) if can_force else None
            early = False
            assume, g = guess_for(y_known)
            g = g[: max(0, max_tokens - len(out) - 2)]
            x = x_of(y, g)
            st = chain(qn, g)
            if can_force:
                h2 = body(x)
                seen, forced_count = stats.get(ctl.q, (0, 0))
                early = EARLY_BODY and seen >= 3 and forced_count < 0.25 * seen
                if early:
                    mx.async_eval(h2)
                q_before = ctl.q
            else:
                p2 = head_all(body(x), st, max_tokens - len(out) - 1)  # y is not in out yet: one token less, as below
                y2 = p2[0]
                q2 = ctl.gpu_next(y2, st[0]) if constrained else None
                mx.async_eval(p2, y2, *([q2] if q2 is not None else []))
            yv = int(y.item()) if y_known is None else y_known
            y_known = None
            spec_fed = can_force is False
            if spec_fed:
                fed.append(yv)
            if g and assume is not None and assume != yv:
                # the draft assumed another token: drop it, position 0 stays valid
                st_["assume_miss"] += 1
                rewind(cache, len(g))
                g, st = [], st[:1]
            if yv in eos:
                finish = "stop"
                break
            try:
                ctl.accept(yv)
            except ConstraintExhausted:
                finish = "exhausted"
                break
            out.append(yv)
            stream.append(yv)
            if len(out) >= max_tokens:
                break
            if ctl.finished():
                finish = "stop"
                break
            if not can_force:
                extra, replaces = [], False
            elif ctl.heal_ok(max_tokens - len(out) + 1):
                extra, replaces = ctl.forced(yv)
            else:
                extra, replaces = ctl.forced(yv, heal=False)
            if can_force:
                seen, forced_count = stats.get(q_before, (0, 0))
                stats[q_before] = (seen + 1, forced_count + bool(extra))
            if extra:
                _restore(cache, snap)
                del h2
                if replaces:
                    out.pop()
                    stream.pop()
                    forced_n += len(extra) - 1
                else:
                    forced_n += len(extra)
                out.extend(extra)
                stream.extend(extra)
                if ctl.finished():
                    finish = "stop"
                    break
                feed = list(extra) if replaces else [yv] + list(extra)
                g = propose(stream, n, width(), eos)[: max(0, max_tokens - len(out) - 1)]
                if constrained and g:
                    cfg, kept = ctl.config(), []
                    for t in g:
                        cfg = ctl.step_config(cfg, t)
                        if cfg is None:
                            break
                        kept.append(t)
                    g = kept
                qg = ctl.gpu_state()
                st = chain(qg, g)
                h = body(mx.array([feed + g]))
                preds = head_all(h[:, len(feed) - 1 :, :], st, max_tokens - len(out), ctl.q if not g else None)
                y = preds[0]
                qn = ctl.gpu_next(y, qg) if constrained else None
                mx.async_eval(preds, y, *([qn] if qn is not None else []))
                fed.extend(feed)
                sizes.append(len(feed) + len(g))
                pend = dict(preds=preds, g=g, states=st)
            else:
                if can_force:
                    snap = None  # release old cache handles so updates can be done in place
                    p2 = head_all(h2[:, : len(st), :], st, max_tokens - len(out), ctl.q if not g else None)
                    y2 = p2[0]
                    q2 = ctl.gpu_next(y2, st[0]) if constrained else None
                    mx.async_eval(p2, y2, *([q2] if q2 is not None else []))
                    fed.append(yv)
                y, qn = y2, q2
                sizes.append(1 + len(g))
                pend = dict(preds=p2, g=g, states=st)
            passes += 1
        emit()
        if can_force and not spec_fed and snap is not None:
            _restore(cache, snap)  # the prebuilt pass was never committed
        elif pend["g"]:
            rewind(cache, len(pend["g"]))  # a committed pass whose draft was never resolved
        mx.synchronize()
        t2 = time.perf_counter()
        assert cache_offset(cache) == len(fed), (cache_offset(cache), len(fed))
        text = self.tokenizer.decode(out)
        res = GenResult(out, text, len(prompt), t1 - t0, t2 - t1, passes, forced_n, sizes, finish, fed, cache, 0)
        res.drafted = st_["drafted"]
        res.draft_passes = st_["draft_passes"]
        res.assume_miss = st_["assume_miss"]
        return res

    def generate_tree(self, prompt, ctl, max_tokens=256, max_candidates=8, budget=16) -> GenResult:
        """Greedy constrained decoding with exact finite-region trees
        (see ``treedec``). Synchronous; trees are used when a decision has at
        most ``max_candidates`` candidates and the pass stays within
        ``budget`` tokens."""
        from .treedec import candidate_branches, keep_branch, tree_forward

        eos = set(ctl.eos_ids)
        cache = make_spec_cache(self.model)
        t0 = time.perf_counter()
        self._prefill(cache, prompt)
        mx.synchronize()
        t1 = time.perf_counter()
        feed = [prompt[-1]]
        out: List[int] = []
        passes = forced_n = trees = 0
        sizes: List[int] = []
        finish = "length"

        def take(y: int) -> Optional[List[int]]:
            """Commit y like sequential decoding; return the next feed or None to stop."""
            nonlocal finish, forced_n
            if y in eos:
                finish = "stop"
                return None
            ctl.accept(y)
            out.append(y)
            if len(out) >= max_tokens:
                return None
            if ctl.finished():
                finish = "stop"
                return None
            extra, replaces = ctl.forced(y)
            nxt = [y]
            if extra:
                if replaces:
                    out.pop()
                    nxt = list(extra)
                    forced_n += len(extra) - 1
                else:
                    nxt = [y] + list(extra)
                    forced_n += len(extra)
                out.extend(extra)
                if ctl.finished():
                    finish = "stop"
                    return None
            return nxt

        while True:
            branches = candidate_branches(ctl, max_candidates, budget - len(feed))
            if branches:
                base = cache[0].offset
                lg_main, lg_br, starts = tree_forward(self.model, cache, feed, branches)
                passes += 1
                trees += 1
                sizes.append(len(feed) + sum(len(b.tokens) for b in branches))
                y = int(mx.argmax(ctl.apply(lg_main[None])[0]).item())
                bi = next((i for i, b in enumerate(branches) if b.token == y), None)
                if bi is None:  # EOS (or a candidate without a branch)
                    keep_branch(cache, base, len(feed), 0, 0)
                    nxt = take(y)
                    if nxt is None:
                        break
                    feed = nxt
                    continue
                b = branches[bi]
                keep_branch(cache, base, len(feed), starts[bi], len(b.tokens))
                nxt = take(y)
                if nxt is None:
                    break
                assert nxt == b.tokens, (nxt, b.tokens)
                y2 = int(mx.argmax(ctl.apply(lg_br[bi][None])[0]).item())
                nxt = take(y2)
                if nxt is None:
                    break
                feed = nxt
            else:
                logits = forward_last(self.model, mx.array([feed]), cache)[0]
                passes += 1
                sizes.append(len(feed))
                y = int(mx.argmax(ctl.apply(logits[None])[0]).item())
                nxt = take(y)
                if nxt is None:
                    break
                feed = nxt
        mx.synchronize()
        t2 = time.perf_counter()
        res = GenResult(out, self.tokenizer.decode(out), len(prompt), t1 - t0, t2 - t1, passes, forced_n, sizes, finish)
        res.trees = trees
        return res

    def _generate_sync(self, prompt, ctl, max_tokens, temp, seed, top_p: float = 0.0, top_k: int = 0) -> GenResult:
        eos = set(ctl.eos_ids) or set(self.tokenizer.eos_token_ids)
        cache = make_prompt_cache(self.model)
        key = mx.random.key(seed)
        t0 = time.perf_counter()
        self._prefill(cache, prompt)
        mx.synchronize()
        t1 = time.perf_counter()
        feed = [prompt[-1]]
        out: List[int] = []
        passes = forced_n = 0
        sizes: List[int] = []
        finish = "length"
        while len(out) < max_tokens:
            logits = forward_last(self.model, mx.array([feed]), cache)[0]
            passes += 1
            sizes.append(len(feed))
            logits = ctl.apply(logits[None])[0]
            bm = ctl.budget_mask(ctl.gpu_state(), max_tokens - len(out))
            if bm is not None:
                logits = mx.where(bm, logits, NEG_INF)
            if temp == 0:
                y = mx.argmax(logits)
            else:
                key, sub = mx.random.split(key)
                y = sample_logits(logits, temp, top_p, top_k, key=sub)
            y = int(y.item())
            if y in eos:
                finish = "stop"
                break
            try:
                ctl.accept(y)
            except ConstraintExhausted:
                finish = "exhausted"
                break
            out.append(y)
            feed = [y]
            if ctl.finished():
                finish = "stop"
                break
            extra, replaces = ctl.forced(y) if ctl.heal_ok(max_tokens - len(out) + 1) else ctl.forced(y, heal=False)
            if extra:
                if replaces:
                    out.pop()
                    feed = list(extra)
                    forced_n += len(extra) - 1
                else:
                    feed = [y] + list(extra)
                    forced_n += len(extra)
                out.extend(extra)
                if ctl.finished():
                    finish = "stop"
                    break
        mx.synchronize()
        t2 = time.perf_counter()
        text = self.tokenizer.decode(out)
        return GenResult(out, text, len(prompt), t1 - t0, t2 - t1, passes, forced_n, sizes, finish)
