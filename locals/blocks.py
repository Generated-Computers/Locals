"""Decision-bound decoding: independent blocks in parallel, certified exact.

A structured output's language factors over cut vertices of its automaton
(block product, Thm A17.1): in ``{"k1":V1,"k2":V2,...}`` every key is a
forced chain and the values are the blocks. Sequential decoding spends one
weight stream per sampled token, sum over all blocks. Here every block is
decoded as its own branch, all branches in one batch that shares the prompt,
so the wave costs as many weight streams as the *longest* block. The
branches' conditioning is an approximation (a value does not see earlier
values), so the assembled output is then *certified* in one pass: the
sequential decoder's masked argmax is computed at every sampled position of
the assembled token sequence (the automaton states along it are the
transition table composed position by position, A14.2); the longest prefix
on which it agrees is exactly what sequential greedy decoding produces. At
the first disagreement the correct token is known from the same pass; the
cache there is recomputed rather than stored (the SC2 trade-off: recompute
along the unique path), the block is continued from it, and the rest is
certified again. Waves are bounded by the number of blocks.

Exactness is token-for-token with the sequential decoder that uses the same
commit rules (forced spans without healing, canonical boundary rule); the
number of weight streams is the hardware-independent cost.
"""

from __future__ import annotations

import copy
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import LRUPromptCache

from .engine import NEG_INF, LocalsController, _token_lengths, make_spec_cache
from .hybrid_batch import HybridBatchEngine
from .kernels import text_model
from .packed import Request
from .regex import DEAD


# --- the block plan --------------------------------------------------------------------------

def _dummy(s: dict, alt: int = 0) -> str:
    """A valid value of schema ``s`` (two different ones for alt = 0, 1)."""
    if "enum" in s:
        return json.dumps(s["enum"][alt % len(s["enum"])])
    if "const" in s:
        return json.dumps(s["const"])
    t = s.get("type")
    if isinstance(t, list):
        t = t[0]
    if t == "string":
        return '""' if alt == 0 else '"x"'
    if t == "integer":
        return "0" if alt == 0 else "1"
    if t == "number":
        return "0" if alt == 0 else "1.5"
    if t == "boolean":
        return "true" if alt == 0 else "false"
    if t == "null":
        return "null"
    if t == "array":
        n = max(int(s.get("minItems", 0)), alt)
        return "[" + ",".join(_dummy(s["items"], alt) for _ in range(n)) + "]"
    if t == "object":
        props = s.get("properties", {})
        req = s.get("required", list(props))
        return "{" + ",".join(f'"{k}":{_dummy(v, alt)}' for k, v in props.items() if k in req) + "}"
    raise ValueError(f"no dummy for schema {s}")


@dataclass
class Block:
    key: str
    start: int  # the state at which the sequential decoder samples the block's first token (after the committed part of the key's chain)
    states: frozenset  # the automaton states inside the block; leaving them ends the block
    skeleton: List[int]  # tokens fed before the first sample when the block is decoded alone


@dataclass
class BlockPlan:
    blocks: List[Block]


def _flat_object(s: dict) -> bool:
    return s.get("type") == "object" and "properties" in s and set(s.get("required", [])) == set(s["properties"]) \
        and not s.get("additionalProperties") and len(s["properties"]) > 0


def _render(node: dict, alt: int, buf: list, leaves: list, path: tuple):
    """Render a dummy document of ``node``; for every leaf value record
    (path, schema, offset after its key name, offset where its key starts)."""
    def emit(x):
        buf.append(x)
        buf[0] += len(x)
    emit("{")
    keys = list(node["properties"])
    for i, k in enumerate(keys):
        v = node["properties"][k]
        key_start = buf[0]
        emit('"' + k)
        if _flat_object(v):
            emit('":')
            _render(v, alt, buf, leaves, path + (k,))
        else:
            leaves.append([path + (k,), v, buf[0], key_start])
            emit('":' + _dummy(v, alt))
        emit("," if i < len(keys) - 1 else "}")


def _dummy_doc(schema: dict, alt: int):
    buf = [0]
    leaves: list = []
    _render(schema, alt, buf, leaves, ())
    return "".join(buf[1:]), leaves


def block_plan(rt, schema: dict, skeleton: str = "bare") -> Optional[BlockPlan]:
    """The blocks of an object schema: its leaf values, with nested objects
    whose properties are all required flattened (their keys are forced
    chains too). None if the schema is not of that shape. A block starts at
    the state at which the sequential decoder samples its first token (after
    the committed part of the key's chain, by the stable-tail rule); its
    states are the automaton states reachable from the key name without
    passing the exit (the state at the next leaf's key, or the end); every
    state is checked to be independent of the earlier values (the automaton
    is minimal). A block decoded alone sees ``{"parent":{"key`` as context."""
    if not _flat_object(schema):
        return None
    c = rt.compile(schema)
    dfa = c.constraint.dfa
    table = dfa.table
    hf = rt.tokenizer._tokenizer
    text0, leaves0 = _dummy_doc(schema, 0)
    text1, leaves1 = _dummy_doc(schema, 1)
    blocks = []
    for j, (path, leaf, off_name, key_start) in enumerate(leaves0):
        s_name = dfa.run(text0[:off_name].encode(), dfa.start)
        if s_name == DEAD or dfa.run(text1[: leaves1[j][2]].encode(), dfa.start) != s_name:
            return None
        if j + 1 < len(leaves0):
            exit_state = dfa.run(text0[: leaves0[j + 1][3]].encode(), dfa.start)
            exit_alt = dfa.run(text1[: leaves1[j + 1][3]].encode(), dfa.start)
        else:
            exit_state = dfa.run(text0.encode(), dfa.start)
            exit_alt = dfa.run(text1.encode(), dfa.start)
        if exit_state == DEAD or exit_alt != exit_state:
            return None
        plan0 = c.planner.plan(None, s_name)
        committed = list(plan0.tokens) if plan0 is not None else []
        start = plan0.q_end if plan0 is not None else s_name
        seen = {s_name}
        stack = [s_name]
        while stack:
            q = stack.pop()
            for nq in set(int(x) for x in table[q, :256] if x != DEAD):
                if nq != exit_state and nq not in seen:
                    seen.add(nq)
                    stack.append(nq)
        if exit_state in seen:
            return None
        context = "".join('{"' + p + '":' for p in path[:-1]) + '{"' + path[-1]
        sk = list(hf.encode(context, add_special_tokens=False)) + committed
        blocks.append(Block(".".join(path), start, frozenset(seen), sk))
    return BlockPlan(blocks)


# --- controllers ----------------------------------------------------------------------------------

class BlockController(LocalsController):
    """The full constraint, started inside one block and finished when the
    state leaves the block. Spans are planned without healing (the drip
    engine's rule) and the sampled tokens are recorded."""

    def __init__(self, compiled, start: int, states: frozenset):
        super().__init__(compiled.constraint, compiled.planner, True, True)
        self.q = start
        self.block = states
        self.sampled: List[int] = []

    def accept(self, token):
        super().accept(token)
        self.sampled.append(token)

    def forced(self, token, heal: bool = True):
        return super().forced(token, heal=False)

    def finished(self):
        return self.q not in self.block or super().finished()


class NoHealController(LocalsController):
    """The sequential reference: the same commit rules as the batch engine."""

    def forced(self, token, heal: bool = True):
        return super().forced(token, heal=False)


# --- the decoder --------------------------------------------------------------------------------

@dataclass
class BlockResult:
    tokens: List[int]
    text: str
    passes: int  # weight streams after the prompt: wave steps + certification passes + recomputations
    wave_steps: int
    cert_passes: int
    waves: int
    accepted_first: int  # sampled tokens certified in the first pass
    sampled: int
    decode_s: float
    prefill_s: float
    per_wave: List[dict] = field(default_factory=list)
    wave_s: float = 0.0
    cert_s: float = 0.0


class BlockDecoder:
    def __init__(self, rt, n_slots: int = 16, skeleton: str = "bare", wave_cap: int = 24, wave_mode: str = "ragged"):
        self.rt = rt
        self.wave_mode = wave_mode
        self.t_wave = self.t_cert = 0.0
        self.lm = text_model(rt.model)
        self.eng = HybridBatchEngine(rt.model, rt.tokenizer, n_slots=n_slots, min_share=16)
        self.skeleton = skeleton
        self.wave_cap = wave_cap  # steps of the first wave; longer blocks continue from the certified prefix
        self._plans: Dict[str, Optional[BlockPlan]] = {}
        self._hist: Dict[Tuple[str, str], List[float]] = {}  # (schema, block) -> [count, mean sampled tokens]

    def _budget(self, key: str, blk: Block) -> int:
        """Tokens a block may sample in the first wave: the cap, or, once the
        block has been seen, 1.5x its mean length + 2 (a runaway then stops
        early and continues from the certified prefix instead)."""
        h = self._hist.get((key, blk.key))
        if h is None or h[0] < 1:
            return self.wave_cap
        return max(2, min(self.wave_cap, int(1.5 * h[1] + 2)))

    def _learn(self, key: str, plan: BlockPlan, sampled: List[List[int]]) -> None:
        for blk, s in zip(plan.blocks, sampled):
            h = self._hist.setdefault((key, blk.key), [0, 0.0])
            h[1] = (h[1] * h[0] + len(s)) / (h[0] + 1)
            h[0] += 1

    def plan(self, schema: dict) -> Optional[BlockPlan]:
        key = json.dumps(schema, sort_keys=True)
        if key not in self._plans:
            self._plans[key] = block_plan(self.rt, schema, self.skeleton)
        return self._plans[key]

    # -- passes ------------------------------------------------------------------------------------
    def _head(self, hl):
        from .kernels import head_logits

        return head_logits(self.lm, hl)

    def _prefill(self, prompt: List[int], chunk: int = 2048):
        """Cache holding all of ``prompt`` and the logits after its last token."""
        cache = make_spec_cache(self.rt.model)
        i = 0
        while len(prompt) - 1 - i > 0:
            n = min(chunk, len(prompt) - 1 - i)
            self.lm.model(mx.array([prompt[i : i + n]]), cache)
            mx.eval([c.state for c in cache])
            i += n
        h = self.lm.model(mx.array([prompt[-1:]]), cache)
        logits = self._head(h[:, -1, :])[0]
        mx.eval(logits, *[c.state for c in cache])
        return cache, logits

    def _certify(self, cache_P, ctl_proto: LocalsController, D: List[int], checks: List[Tuple[int, int]], where: Dict[int, Tuple[int, int]]):
        """One pass over D from the prompt's cache; returns, for every block
        whose samples disagree with the sequential argmax, (the first such
        position, the argmax there). Only the first entry is certain: the
        later ones are computed in a context that the first fix may change,
        so they are continued as drafts and certified again. ``checks`` are
        (position, state before it)."""
        t = time.perf_counter()
        cache = copy.deepcopy(cache_P)
        h = self.lm.model(mx.array([D[:-1]]), cache)  # (1, S, D): row i predicts D[i + 1]
        rows = mx.array([k - 1 for k, _ in checks], dtype=mx.int32)
        logits = self._head(h[0, rows])  # (K, V)
        qs = mx.array([q for _, q in checks], dtype=mx.int32)
        m = ctl_proto.allowed[qs]
        if ctl_proto.canonical:
            m = m & (ctl_proto.tok_len[None, :] >= ctl_proto._gpu_cover()[qs][:, None])
        pred = mx.argmax(mx.where(m, logits, NEG_INF), axis=-1).tolist()
        self.t_cert += time.perf_counter() - t
        out = []
        failed = set()
        for (k, _), p in zip(checks, pred):
            j = where[k][0]
            if j in failed:
                continue  # after a block's first mismatch its later positions are replaced anyway
            if p != D[k]:
                out.append((k, p))
                failed.add(j)
        return out

    # -- assembly ------------------------------------------------------------------------------------
    def _assemble(self, compiled, plan: BlockPlan, T0: List[int], q_T0: int, sampled: List[List[int]], run_start: List[Optional[int]]):
        """Replay the sampled tokens of the blocks through the full controller
        with the sequential commit rules, as far as they fit. Returns the
        token sequence, the (position, state) of every sampled token after
        T0, for each such position its (block, index), the controller at the
        end, and the first block that still needs decoding (or None)."""
        ctl = NoHealController(compiled.constraint, compiled.planner, True, True)
        ctl.q = q_T0
        D = list(T0)
        checks: List[Tuple[int, int]] = []
        where: Dict[int, Tuple[int, int]] = {}
        pending = None
        for j, blk in enumerate(plan.blocks):
            if ctl.finished():
                break
            if ctl.q not in blk.states:  # a previous token carried past this block's start
                if sampled[j]:
                    pending = (j, "inconsistent")
                    break
                continue
            if run_start[j] is not None and run_start[j] != ctl.q:
                pending = (j, "inconsistent")  # the samples were produced from another state: decode again from here
                break
            for i, t in enumerate(sampled[j]):
                checks.append((len(D), ctl.q))
                where[len(D)] = (j, i)
                ctl.accept(t)
                D.append(t)
                extra, rep = ctl.forced(t)
                assert not rep
                D.extend(extra)
                if ctl.q not in blk.states:
                    break
            if ctl.q in blk.states and not ctl.finished():
                pending = (j, "incomplete")  # the block is not finished (the wave cap, or no samples yet)
                break
        return D, checks, where, ctl, pending

    # -- waves -------------------------------------------------------------------------------------
    def _wave(self, compiled, prompts: List[List[int]], starts: List[int], blocks: List[Block], cache_P, prompt, max_tokens):
        """Decode the blocks as one batch from the prompt's cache (a snapshot
        the engine copies per row). ``drip`` feeds one token per row per step;
        ``ragged`` feeds a sampled token with its forced span in one step.
        ``max_tokens`` is one number or one per block."""
        caps = max_tokens if isinstance(max_tokens, list) else [max_tokens] * len(prompts)
        reqs = [Request(list(p), BlockController(compiled, s, b.states), m) for p, s, b, m in zip(prompts, starts, blocks, caps)]
        self.eng.snapshots = OrderedDict({tuple(prompt): cache_P})
        t = time.perf_counter()
        if self.wave_mode == "ragged":
            st = self.eng.run(reqs, mode="ragged")
            steps = st.steps
        else:
            queue = list(reqs)

            def source(block):
                got, queue[:] = list(queue), []
                return got

            self.eng.serve(source)
            steps = self.eng.counters["steps"]
        self.t_wave += time.perf_counter() - t
        return [r.controller.sampled for r in reqs], steps, [r.finish for r in reqs]

    def generate(self, prompt: List[int], schema: dict, max_tokens: int = 320, max_waves: int = 24) -> BlockResult:
        plan = self.plan(schema)
        if plan is None:
            raise ValueError("schema is not a flat all-required object")
        compiled = self.rt.compile(schema)
        t0 = time.perf_counter()
        cache_P, logits0 = self._prefill(prompt)
        mx.synchronize()
        t1 = time.perf_counter()
        # step 0: the first token is sampled at the start state (from the prefill's logits)
        ctl = NoHealController(compiled.constraint, compiled.planner, True, True)
        y0 = int(mx.argmax(ctl.apply(logits0[None])[0]).item())
        ctl.accept(y0)
        extra, _ = ctl.forced(y0)
        T0 = [y0] + list(extra)
        q_T0 = ctl.q
        self.t_wave = self.t_cert = 0.0
        # wave 1: every block as a branch; block 0 continues the real prefix
        n = len(plan.blocks)
        sampled: List[List[int]] = [[] for _ in range(n)]
        run_start: List[Optional[int]] = [None] * n
        prompts, starts, blocks, idx = [], [], [], []
        for j, blk in enumerate(plan.blocks):
            if j == 0:
                if q_T0 in blk.states:
                    prompts.append(list(prompt) + T0)
                    starts.append(q_T0)
                    blocks.append(blk)
                    idx.append(j)
            else:
                prompts.append(list(prompt) + blk.skeleton)
                starts.append(blk.start)
                blocks.append(blk)
                idx.append(j)
        wave_steps = cert_passes = recomputes = 0
        per_wave = []
        waves = 0
        skey = json.dumps(schema, sort_keys=True)
        if prompts:
            got, steps, fin = self._wave(compiled, prompts, starts, blocks, cache_P, prompt, [self._budget(skey, b) for b in blocks])
            waves += 1
            wave_steps += steps
            for j, s0, s in zip(idx, starts, got):
                sampled[j] = list(s)
                run_start[j] = s0
            per_wave.append({"branches": len(prompts), "steps": steps, "finish": fin})
        accepted_first = None
        total_checks = 0
        while True:
            D, checks, where, rctl, pending = self._assemble(compiled, plan, T0, q_T0, sampled, run_start)
            mism = []
            if checks:
                mism = self._certify(cache_P, rctl, D, checks, where)
                cert_passes += 1
            k0 = mism[0][0] if mism else None
            n_ok = len(checks) if k0 is None else sum(1 for kk, _ in checks if kk < k0)
            if accepted_first is None:
                accepted_first = n_ok
            if not mism and pending is None:
                total_checks = len(checks)
                break
            if waves >= max_waves:
                total_checks = len(checks)
                break
            # every block that failed continues from its first mismatch with the
            # correct token; an unfinished block continues from its end; all in one wave
            prompts, starts, blocks, idx = [], [], [], []
            fixed = set()
            for k, y in mism:
                j, i = where[k]
                q_k = next(q for kk, q in checks if kk == k)
                c2 = NoHealController(compiled.constraint, compiled.planner, True, True)
                c2.q = q_k
                c2.accept(y)
                extra, _ = c2.forced(y)
                head = D[:k] + [y] + list(extra)
                sampled[j] = sampled[j][:i] + [y]
                run_start[j] = None
                fixed.add(j)
                blk = plan.blocks[j]
                if c2.q in blk.states and not c2.finished():
                    prompts.append(list(prompt) + head)
                    starts.append(c2.q)
                    blocks.append(blk)
                    idx.append(j)
            if pending is not None and pending[0] not in fixed:
                j, reason = pending
                blk = plan.blocks[j]
                if reason == "inconsistent":
                    sampled[j] = []
                run_start[j] = None
                if rctl.q in blk.states and not rctl.finished():
                    prompts.append(list(prompt) + D)
                    starts.append(rctl.q)
                    blocks.append(blk)
                    idx.append(j)
            waves += 1
            if prompts:
                got, steps, fin = self._wave(compiled, prompts, starts, blocks, cache_P, prompt, max_tokens)
                wave_steps += steps
                recomputes += 1  # the branches' prefill from the prompt's cache re-derives their prefixes
                for j, g in zip(idx, got):
                    sampled[j] = sampled[j] + list(g)
                per_wave.append({"branches": len(prompts), "steps": steps, "finish": fin, "blocks": [b.key for b in blocks]})
            else:
                per_wave.append({"branches": 0, "steps": 0})
        self._learn(skey, plan, sampled)
        mx.synchronize()
        t2 = time.perf_counter()
        return BlockResult(D, self.rt.tokenizer.decode(D), wave_steps + cert_passes + recomputes, wave_steps, cert_passes, waves,
                           accepted_first, total_checks, t2 - t1, t1 - t0, per_wave, self.t_wave, self.t_cert)


def sequential_reference(rt, prompt: List[int], schema: dict, max_tokens: int = 320):
    """Single-stream greedy decoding with the batch engine's commit rules."""
    c = rt.compile(schema)
    ctl = NoHealController(c.constraint, c.planner, True, True)
    return rt.engine.generate(prompt, ctl, max_tokens=max_tokens)
