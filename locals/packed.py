"""Packed multi-request forward pass and a batched constrained decoder.

Every active request contributes a different number of new tokens per step
(its sampled token plus any forced span). All of them are packed into one flat
row batch, so each weight matrix is streamed once per step for the whole
batch; with the small-M kernels a pass over 8-16 rows costs about 2x a single
row on this hardware. Positions are per row (RoPE accepts per-row offsets),
KV is written with one scatter into a per-layer buffer holding one fixed
slot per request, and attention runs per request against its own slot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import mlx.core as mx
import numpy as np

from .engine import NEG_INF, Controller

# ----------------------------------------------------------------------------
# KV buffer and packed forward


class PackedKV:
    def __init__(self, model, n_slots: int, cap: int):
        attn = model.model.layers[0].self_attn
        self.n_heads = attn.n_heads
        self.n_kv = attn.n_kv_heads
        self.head_dim = getattr(attn, "head_dim", None) or attn.q_proj["weight"].shape[0] // attn.n_heads
        dt = mx.float16
        # Flat rows: row (slot * n_kv + head) * cap + pos holds one key/value vector.
        rows = n_slots * self.n_kv * cap
        self.K = [mx.zeros((rows, self.head_dim), dt) for _ in model.model.layers]
        self.V = [mx.zeros((rows, self.head_dim), dt) for _ in model.model.layers]
        mx.eval(self.K, self.V)
        self.cap = cap
        self.n_slots = n_slots
        self.length = [0] * n_slots

    def view(self, li: int, slot: int, n: int):
        """(1, n_kv, n, D) views of the first n entries of a slot."""
        shape = (self.n_slots, self.n_kv, self.cap, self.head_dim)
        k = self.K[li].reshape(shape)[slot : slot + 1, :, :n, :]
        v = self.V[li].reshape(shape)[slot : slot + 1, :, :n, :]
        return k, v

    def slot_major(self, li: int):
        shape = (self.n_slots, self.n_kv, self.cap, self.head_dim)
        return self.K[li].reshape(shape), self.V[li].reshape(shape)

    @property
    def nbytes(self) -> int:
        return sum(k.nbytes + v.nbytes for k, v in zip(self.K, self.V))


def _qkv(attn, x, R):
    H, Hk = attn.n_heads, attn.n_kv_heads
    if hasattr(attn, "qkv_proj"):
        qkv = attn.qkv_proj(x).reshape(R, H + 2 * Hk, 1, -1)
        return qkv[:, :H], qkv[:, H : H + Hk], qkv[:, H + Hk :]
    q = attn.q_proj(x).reshape(R, H, 1, -1)
    k = attn.k_proj(x).reshape(R, Hk, 1, -1)
    v = attn.v_proj(x).reshape(R, Hk, 1, -1)
    return q, k, v


USE_RAGGED = False  # per-request MLX SDPA keeps outputs identical to single-stream


def packed_forward(model, kv: PackedKV, segments: Sequence, want_logits: bool = True):
    """segments: [(slot, tokens)]. Appends each segment's tokens to its slot and
    returns logits for the last token of every segment, shape (S, V)."""
    toks: List[int] = []
    pos: List[int] = []
    meta = []
    trees = []
    out_rows: List[int] = []
    for seg in segments:
        slot, t = seg[0], seg[1]
        tree = seg[2] if len(seg) > 2 else None
        start = len(toks)
        base = kv.length[slot]
        toks.extend(t)
        if tree is None:
            pos.extend(range(base, base + len(t)))
            out_rows.append(start + len(t) - 1)
        else:
            pos_rel, _, rows = tree
            pos.extend(base + p for p in pos_rel)
            out_rows.extend(start + r for r in rows)
        meta.append((slot, start, len(t), base + len(t)))
        trees.append(tree)
    R = len(toks)
    pos_a = mx.array(pos, dtype=mx.int32)
    Hk = kv.n_kv
    # Flat write rows: for each packed row and kv head, (slot * Hk + h) * cap + pos.
    row_slot = [slot for (slot, s, L, e) in meta for _ in range(L)]
    # Storage position: consecutive in the slot (RoPE positions may differ for trees).
    store = [p for (slot, s, L, e) in meta for p in range(e - L, e)]
    widx = mx.array(
        [(sl * Hk + h) * kv.cap + p for sl, p in zip(row_slot, store) for h in range(Hk)], dtype=mx.int32
    )
    ragged = USE_RAGGED and all(t is None for t in trees) and all(L <= 64 for (_, _, L, _) in meta)
    masks = []
    for (slot, s0, L, e), tree in zip(meta, trees):
        if tree is None:
            masks.append("causal" if L > 1 else None)
        else:
            import numpy as _np

            local = tree[1]
            full = _np.concatenate([_np.ones((L, e - L), dtype=bool), local], axis=1)
            masks.append(mx.array(full)[None, None])
    if ragged:
        from .attention import ragged_attention

        r_slots = mx.array(row_slot, dtype=mx.int32)
        r_nkeys = mx.array([p + 1 for p in pos], dtype=mx.int32)
        r_scale = mx.array([body_scale(model)], dtype=mx.float32)
    body = model.model
    h = body.embed_tokens(mx.array(toks)[None])
    for li, layer in enumerate(body.layers):
        attn = layer.self_attn
        x = layer.input_layernorm(h)
        q, k, v = _qkv(attn, x, R)
        q = attn.rope(q, offset=pos_a)
        k = attn.rope(k, offset=pos_a)
        kv.K[li][widx] = k.reshape(R * Hk, -1)
        kv.V[li][widx] = v.reshape(R * Hk, -1)
        if ragged:
            Ks, Vs = kv.slot_major(li)
            o = ragged_attention(q.reshape(R, attn.n_heads, -1), Ks, Vs, r_slots, r_nkeys, r_scale).reshape(1, R, -1)
        else:
            outs = []
            for (slot, s, L, e), msk in zip(meta, masks):
                qb = q[s : s + L, :, 0, :].transpose(1, 0, 2)[None]
                kb, vb = kv.view(li, slot, e)
                ob = mx.fast.scaled_dot_product_attention(qb, kb, vb, scale=attn.scale, mask=msk)
                outs.append(ob[0].transpose(1, 0, 2).reshape(L, -1))
            o = (mx.concatenate(outs, axis=0) if len(outs) > 1 else outs[0])[None]
        h = h + attn.o_proj(o)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    for slot, s, L, e in meta:
        kv.length[slot] = e
    if not want_logits:
        return None
    h = body.norm(h)
    hl = h[:, mx.array(out_rows, dtype=mx.int32), :]
    from .kernels import head_logits

    return head_logits(model, hl)[0]


def body_scale(model) -> float:
    return float(model.model.layers[0].self_attn.scale)


def _tree_layout(main: List[int], branches) -> dict:
    """Flattened tokens, relative positions, ancestor mask and logit rows for
    main followed by a tree of branches (preorder: each branch contiguous)."""
    import numpy as _np

    m = len(main)
    toks = list(main)
    pos = list(range(m))
    starts = []
    br = [-1] * m
    for i, b in enumerate(branches):
        starts.append(len(toks) - m)
        toks.extend(b.tokens)
        pos.extend(range(m, m + len(b.tokens)))
        br.extend([i] * len(b.tokens))
    L = len(toks)
    br = _np.array(br)
    qi = _np.arange(L)[:, None]
    kj = _np.arange(L)[None, :]
    mask = (kj <= qi) & ((br[:, None] == br[None, :]) | (br[None, :] < 0))
    rows = [m - 1] + [m + s0 + len(b.tokens) - 1 for s0, b in zip(starts, branches)]
    return {"tokens": toks, "pos": pos, "mask": mask, "rows": rows, "starts": starts, "branches": branches, "m": m}


def prefill_slot(model, kv: PackedKV, slot: int, tokens: List[int], chunk: int = 1024) -> None:
    """Write KV for ``tokens`` into ``slot`` (all but the caller's last token)."""
    kv.length[slot] = 0
    for i in range(0, len(tokens), chunk):
        packed_forward(model, kv, [(slot, tokens[i : i + chunk])], want_logits=False)
        mx.eval(kv.K, kv.V)


def copy_prefix(kv: PackedKV, src: int, dst: int, n: int) -> None:
    """Copy the first n KV entries of slot src into slot dst (prefix sharing)."""
    for K, V in zip(kv.K, kv.V):
        for h in range(kv.n_kv):
            a = (src * kv.n_kv + h) * kv.cap
            b = (dst * kv.n_kv + h) * kv.cap
            K[b : b + n] = K[a : a + n]
            V[b : b + n] = V[a : a + n]
    kv.length[dst] = n


def keep_slot_range(kv: PackedKV, slot: int, src: int, dst: int, n: int) -> None:
    """Move n KV entries within a slot from storage position src to dst."""
    if n == 0 or src == dst:
        return
    import numpy as _np

    idx_s, idx_d = [], []
    for h in range(kv.n_kv):
        b = (slot * kv.n_kv + h) * kv.cap
        idx_s.extend(range(b + src, b + src + n))
        idx_d.extend(range(b + dst, b + dst + n))
    si = mx.array(idx_s, dtype=mx.int32)
    di = mx.array(idx_d, dtype=mx.int32)
    for K, V in zip(kv.K, kv.V):
        K[di] = K[si]
        V[di] = V[si]


def prefill_many(model, kv: PackedKV, items, budget: int = 2048) -> None:
    """Prefill several (slot, tokens) remainders, packing as many as fit in
    ``budget`` rows into each forward pass."""
    batch, rows = [], 0
    for slot, toks in items:
        for i in range(0, len(toks), budget):
            piece = toks[i : i + budget]
            if rows + len(piece) > budget and batch:
                packed_forward(model, kv, batch, want_logits=False)
                mx.eval(kv.K, kv.V)
                batch, rows = [], 0
            batch.append((slot, piece))
            rows += len(piece)
    if batch:
        packed_forward(model, kv, batch, want_logits=False)
        mx.eval(kv.K, kv.V)


def _common(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ----------------------------------------------------------------------------
# Batched constrained decoding


@dataclass
class Request:
    prompt: List[int]
    controller: Controller
    max_tokens: int = 256
    out: List[int] = field(default_factory=list)
    text: str = ""
    finish: str = ""
    slot: int = -1
    feed: List[int] = field(default_factory=list)
    passes: int = 0
    forced: int = 0


@dataclass
class BatchStats:
    wall_s: float
    prefill_s: float
    steps: int
    rows: int
    tokens: int

    @property
    def tok_per_s(self) -> float:
        return self.tokens / self.wall_s if self.wall_s else 0.0


class BatchEngine:
    def __init__(self, model, tokenizer, n_slots: int = 8, cap: int = 2048, share_prefix: bool = True,
                 trees: bool = False, max_candidates: int = 8, tree_budget: int = 12, row_cap: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.trees = trees
        self.max_candidates = max_candidates
        self.tree_budget = tree_budget
        self.row_cap = row_cap
        self.tree_count = 0
        self.kv = PackedKV(model, n_slots, cap)
        self.share_prefix = share_prefix
        # Tokens whose KV each slot currently holds (kept after a request
        # finishes, until the slot is reused): the nodes of a prompt trie.
        self.slot_tokens: List[List[int]] = [[] for _ in range(n_slots)]
        self.shared_tokens = 0
        self.profile = False
        self.prof = {"forward": 0.0, "sample": 0.0, "host": 0.0}

    def _tables(self, requests):
        """One combined mask table over every constraint in the batch, so all
        requests are masked with a single gather (global state = offset + q)."""
        from .engine import LocalsController, _token_lengths

        ctls = [r.controller for r in requests]
        if not all(isinstance(c, LocalsController) for c in ctls):
            return None
        uniq, offs = [], {}
        for c in ctls:
            if id(c.c) not in offs:
                offs[id(c.c)] = sum(u.c.n_states for u in uniq)
                uniq.append(c)
        allowed = mx.concatenate([u.allowed for u in uniq], axis=0)
        cover = mx.concatenate([u._gpu_cover() if u.canonical else mx.zeros((u.c.n_states,), mx.int32) for u in uniq])
        return allowed, cover, _token_lengths(uniq[0].planner.vocab), offs

    def _commit(self, r, y) -> bool:
        """Commit a sampled token exactly as sequential decoding; returns done."""
        if y in r.controller.eos_ids:
            r.finish = "stop"
            return True
        r.controller.accept(y)
        r.out.append(y)
        r.feed = [y]
        if r.controller.finished():
            r.finish = "stop"
            return True
        if len(r.out) >= r.max_tokens:
            r.finish = "length"
            return True
        extra, replaces = r.controller.forced(y)
        if extra:
            if replaces:
                r.out.pop()
                r.feed = list(extra)
            else:
                r.feed = [y] + list(extra)
            r.out.extend(extra)
            r.forced += len(extra)
            if r.controller.finished():
                r.finish = "stop"
                return True
        return False

    def _sample_states(self, states, logits, temp):
        allowed, cover, tlen, offs = self._tab
        g = mx.array([offs[id(c.c)] + q for c, q in states], dtype=mx.int32)
        m = allowed[g] & (tlen[None, :] >= cover[g][:, None])
        logits = mx.where(m, logits, NEG_INF)
        return mx.argmax(logits, axis=-1) if temp == 0 else mx.random.categorical(logits / temp)

    def _sample(self, active, logits, temp):
        if self._tab is not None:
            allowed, cover, tlen, offs = self._tab
            g = mx.array([offs[id(r.controller.c)] + r.controller.q for r in active], dtype=mx.int32)
            m = allowed[g] & (tlen[None, :] >= cover[g][:, None])
            logits = mx.where(m, logits, NEG_INF)
            return mx.argmax(logits, axis=-1) if temp == 0 else mx.random.categorical(logits / temp)
        ys = []
        for i, r in enumerate(active):
            li = logits[i]
            qg = r.controller.gpu_state()
            m = r.controller.gpu_mask(qg) if qg is not None else None
            if m is not None:
                li = mx.where(m, li, NEG_INF)
            ys.append(mx.argmax(li) if temp == 0 else mx.random.categorical(li / temp))
        return mx.stack(ys)

    def run(self, requests: List[Request], temp: float = 0.0) -> BatchStats:
        self._tab = self._tables(requests)
        kv = self.kv
        queue = list(requests)
        active: List[Request] = []
        free = list(range(kv.n_slots))
        steps = rows = 0
        prefill_s = 0.0
        t0 = time.perf_counter()
        while queue or active:
            # Admit waiting requests into free slots (least recently used
            # first, so finished slots keep their prefixes available longest).
            admitted = []
            while queue and free:
                r = queue.pop(0)
                if len(r.prompt) + r.max_tokens + 64 > kv.cap:
                    raise ValueError("request exceeds slot capacity")
                r.slot = free.pop(0)
                admitted.append(r)
            if admitted:
                tp = time.perf_counter()
                todo = []
                pending = set()  # slots whose KV is scheduled but not yet computed
                for r in admitted:
                    body = r.prompt[:-1]
                    best, src = 0, -1
                    if self.share_prefix:
                        for s2 in range(kv.n_slots):
                            if s2 == r.slot:
                                continue
                            c = _common(self.slot_tokens[s2], body)
                            if c > best:
                                best, src = c, s2
                    if best >= 16:
                        if src in pending:
                            prefill_many(self.model, kv, todo)
                            todo, pending = [], set()
                        copy_prefix(kv, src, r.slot, best)
                        self.shared_tokens += best
                    else:
                        best = 0
                        kv.length[r.slot] = 0
                    self.slot_tokens[r.slot] = list(body)
                    if len(body) > best:
                        todo.append((r.slot, body[best:]))
                        pending.add(r.slot)
                prefill_many(self.model, kv, todo)
                prefill_s += time.perf_counter() - tp
                for r in admitted:
                    r.feed = [r.prompt[-1]]
                    active.append(r)
            segs = []
            rows_now = sum(len(r.feed) for r in active)
            for r in active:
                r.tree = None
                if self.trees:
                    from .treedec import candidate_branches

                    br = candidate_branches(r.controller, self.max_candidates, self.tree_budget)
                    extra_rows = sum(len(x.tokens) for x in br) if br else 0
                    if br and rows_now + extra_rows <= self.row_cap:
                        rows_now += extra_rows
                        r.tree = _tree_layout(r.feed, br)
                if r.tree is None:
                    segs.append((r.slot, r.feed))
                else:
                    segs.append((r.slot, r.tree["tokens"], (r.tree["pos"], r.tree["mask"], r.tree["rows"])))
                    self.tree_count += 1
            rows += sum(len(sg[1]) for sg in segs)
            bases = [kv.length[r.slot] for r in active]
            ta = time.perf_counter()
            logits = packed_forward(self.model, kv, segs)
            if self.profile:
                mx.eval(logits)
            tb = time.perf_counter()
            steps += 1
            # One sampling pass: main rows at the current state, branch-end
            # rows at each branch's end state (known in advance).
            states, owners = [], []
            for i, r in enumerate(active):
                states.append((r.controller, r.controller.q))
                if r.tree is not None:
                    for b in r.tree["branches"]:
                        states.append((r.controller, b.q_end))
            ys = self._sample_states(states, logits, temp).tolist()
            tc = time.perf_counter()
            still = []
            k = 0
            for r, base in zip(active, bases):
                y = ys[k]
                k += 1
                y2s = None
                if r.tree is not None:
                    y2s = ys[k : k + len(r.tree["branches"])]
                    k += len(r.tree["branches"])
                r.passes += 1
                done = self._commit(r, y)
                if r.tree is not None:
                    m = r.tree["m"]
                    bi = next((i for i, b in enumerate(r.tree["branches"]) if b.token == y), None)
                    if bi is None:
                        kv.length[r.slot] = base + m
                    else:
                        b = r.tree["branches"][bi]
                        keep_slot_range(kv, r.slot, base + m + r.tree["starts"][bi], base + m, len(b.tokens))
                        kv.length[r.slot] = base + m + len(b.tokens)
                        if not done:
                            assert r.feed == b.tokens, (r.feed, b.tokens)
                            done = self._commit(r, y2s[bi])
                if done:
                    r.text = self.tokenizer.decode(r.out)
                    free.append(r.slot)
                    self.slot_tokens[r.slot] = self.slot_tokens[r.slot][: len(r.prompt) - 1]
                else:
                    still.append(r)
            active = still
            td = time.perf_counter()
            self.prof["forward"] += tb - ta
            self.prof["sample"] += tc - tb
            self.prof["host"] += td - tc
        mx.synchronize()
        wall = time.perf_counter() - t0
        return BatchStats(wall, prefill_s, steps, rows, sum(len(r.out) for r in requests))
