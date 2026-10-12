"""Budget-aware masking (locals/budget.py).

Tokenizer-only property tests simulate the engines' stopping rule (stop right
after a sampled token once len(out) >= max_tokens; forced spans count toward
len(out)) with random and adversarial token choices under the budget mask:
every run must end in an accepting state within max_tokens, and need(start)
must be the true minimum (the cheapest run reaches it). Model tests run the
pipelined, synchronous and drip batch engines with tight max_tokens.
"""

import json
import random
import re

import numpy as np
import pytest

pytest.importorskip("mlx_lm")

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
HYBRID = "mlx-community/Qwen3.5-0.8B-MLX-4bit"


@pytest.fixture(scope="module")
def tok():
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer

    try:
        path = snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip("tokenizer not cached")
    return load_tokenizer(path)._tokenizer


@pytest.fixture(scope="module")
def compiled(tok):
    from outlines_core.json_schema import build_regex_from_schema
    from workloads import WORKLOADS

    from locals.constraint import compile_constraint
    from locals.forced import ForcedPlanner
    from locals.vocab import ByteTrie, TokenVocab

    vocab = TokenVocab.from_hf(tok, 151936, [151645, 151643])
    trie = ByteTrie.build(vocab)
    out = []
    for w in WORKLOADS:
        c = compile_constraint(build_regex_from_schema(json.dumps(w["schema"]), ""), vocab, trie)
        out.append((w["name"], c, ForcedPlanner(tok, vocab, c)))
    return out


def _simulate(ctl, max_tokens, pick):
    """One run of the engines' loop at DFA level; returns (valid, len(out))."""
    from locals.regex import DEAD

    c, bt = ctl.c, ctl.budget_tables()[0]
    tl = ctl._tok_len_np()
    q, n = c.dfa.start, 0
    while True:
        r = max_tokens - n
        ok = c.allowed[q].copy()
        if ctl.canonical:
            cv = ctl.planner.cover_rule(q)
            if cv:
                ok &= tl >= cv
        if bt.need[q] <= r:
            nq = c.next_state[:, q]
            fits = (nq != DEAD) & (bt.cost[np.where(nq == DEAD, 0, nq)] <= r)
            fits[list(c.eos_ids)] = True
            ok &= fits
        ids = np.nonzero(ok)[0]
        assert len(ids), "budget mask left no token"
        t = pick(ids, q)
        if t in c.eos_ids:
            return bool(c.dfa.accepting[q]), n
        q = int(c.next_state[t, q])
        n += 1
        if n >= max_tokens:
            return bool(c.dfa.accepting[q]), n
        if c.only_eos[q]:
            return True, n
        if ctl.use_forced and c.forced_bytes[q]:
            p = ctl.planner.plan(None, q)
            if p is not None and p.tokens:
                n += len(p.tokens)
                q = p.q_end
                if c.only_eos[q]:
                    return True, n


@pytest.mark.parametrize("canonical", [False, True])
def test_budget_runs_always_valid(compiled, canonical):
    from locals.engine import LocalsController

    rng = random.Random(0)
    for name, c, pl in compiled:
        ctl = LocalsController(c, pl, use_forced=True, canonical=canonical)
        bt = ctl.budget_tables()[0]
        need0 = int(bt.need[c.dfa.start])
        eos = set(c.eos_ids)
        # adversarial: never stop voluntarily (no EOS), otherwise uniform
        def wander(ids, q):
            non = [int(i) for i in ids if int(i) not in eos]
            return rng.choice(non) if non else int(ids[0])

        for mt in list(range(need0, need0 + 12)) + [need0 + 25, need0 + 60]:
            for _ in range(6):
                valid, n = _simulate(ctl, mt, wander)
                assert valid, (name, mt)
                assert n <= mt, (name, mt, n)


def test_need_is_the_true_minimum(compiled):
    from locals.engine import LocalsController

    for name, c, pl in compiled:
        ctl = LocalsController(c, pl, use_forced=True, canonical=True)
        bt = ctl.budget_tables()[0]
        need0 = int(bt.need[c.dfa.start])

        def cheapest(ids, q):
            nq = c.next_state[ids, q]
            cost = np.where(np.isin(ids, c.eos_ids), 0, bt.cost[np.where(nq < 0, 0, nq)])
            return int(ids[int(np.argmin(cost))])

        valid, n = _simulate(ctl, need0, cheapest)
        assert valid and n <= need0, (name, need0, n)
        # below the minimum nothing can be guaranteed: the mask falls back
        assert bt.need[c.dfa.start] >= 2


# ---------------------------------------------------------------------------- model tests


@pytest.fixture(scope="module")
def rt():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    from locals.runtime import Runtime

    return Runtime.load(MODEL)


def _cases(rt):
    from outlines_core.json_schema import build_regex_from_schema
    from workloads import WORKLOADS, make_prompt

    for w in WORKLOADS:
        c = rt.compile(w["schema"])
        rx = re.compile(build_regex_from_schema(json.dumps(w["schema"]), ""))
        yield w, c, rx, make_prompt(rt.tokenizer, w)


@pytest.mark.parametrize("pipeline", [True, False])
def test_engine_outputs_valid_under_tight_budget(rt, pipeline):
    from locals.engine import LocalsController

    for w, c, rx, p in _cases(rt):
        need0 = int(LocalsController(c.constraint, c.planner, True, True).budget_tables()[0].need[c.constraint.dfa.start])
        for mt in (need0, need0 + 3, need0 + 10):
            ctl = LocalsController(c.constraint, c.planner, use_forced=True, canonical=True)
            res = rt.engine.generate(p, ctl, max_tokens=mt, pipeline=pipeline)
            assert rx.fullmatch(res.text), (w["name"], mt, res.text)
            assert len(res.tokens) <= mt, (w["name"], mt, len(res.tokens))


def test_budget_off_by_default_is_unchanged_with_room(rt):
    """With ample max_tokens the budget mask never activates: identical output."""
    from locals.engine import LocalsController

    for w, c, rx, p in list(_cases(rt))[:3]:
        a = rt.engine.generate(p, LocalsController(c.constraint, c.planner, True, True), max_tokens=320)
        ctl = LocalsController(c.constraint, c.planner, True, True)
        ctl.budget = False
        b = rt.engine.generate(p, ctl, max_tokens=320)
        assert a.tokens == b.tokens


def test_drip_batch_valid_under_tight_budget():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(HYBRID, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    from outlines_core.json_schema import build_regex_from_schema
    from workloads import WORKLOADS, make_prompt

    from locals.engine import LocalsController
    from locals.hybrid_batch import HybridBatchEngine
    from locals.packed import Request
    from locals.runtime import Runtime

    rt2 = Runtime.load(HYBRID)
    reqs, rxs = [], []
    for w in WORKLOADS:
        c = rt2.compile(w["schema"])
        need0 = int(LocalsController(c.constraint, c.planner, True, True).budget_tables()[0].need[c.constraint.dfa.start])
        for extra in (0, 4):
            ctl = LocalsController(c.constraint, c.planner, use_forced=True, canonical=True)
            reqs.append(Request(make_prompt(rt2.tokenizer, w), ctl, need0 + extra))
            rxs.append(re.compile(build_regex_from_schema(json.dumps(w["schema"]), "")))
    HybridBatchEngine(rt2.model, rt2.tokenizer, n_slots=8).run(reqs)
    for r, rx in zip(reqs, rxs):
        assert rx.fullmatch(r.text), (r.max_tokens, r.text)
        assert len(r.out) <= r.max_tokens
