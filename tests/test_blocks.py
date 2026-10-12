"""Decision-bound decoding (locals/blocks.py), few-path regions
(locals/regions.py) and lazy rows (locals/lazy.py) on Qwen3.5-2B; skipped if
the model is not cached.

  * the block plan's states are independent of the earlier values and the
    blocks partition the document; the certified output is token-identical
    to the sequential decoder with the same commit rules, with fewer weight
    streams
  * region enumeration equals brute force; the batched scorer equals
    single-row scoring
  * lazy rows equal the full table's columns for every visited state and
    give the same output
"""

import os
import json

import pytest

pytest.importorskip("mlx_lm")

MODEL = os.environ.get("LOCALS_TEST_MODEL", "mlx-community/Qwen3.5-2B-MLX-4bit")


@pytest.fixture(scope="module")
def rt():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    from locals.runtime import Runtime

    return Runtime.load(MODEL)


def test_block_plan_and_exactness(rt):
    from workloads import prompt_text
    from workloads import WORKLOADS

    from locals.blocks import BlockDecoder, sequential_reference

    dec = BlockDecoder(rt, wave_cap=12)
    for w in [x for x in WORKLOADS if x["name"] in ("person", "tool_call", "product")]:
        plan = dec.plan(w["schema"])
        assert plan is not None
        keys = [b.key for b in plan.blocks]
        assert len(keys) == len(set(keys))
        for a in plan.blocks:
            for b in plan.blocks:
                if a is not b:
                    assert not (a.states & b.states), (a.key, b.key)
        ids = rt.encode_chat(prompt_text(w, True))
        ref = sequential_reference(rt, ids, w["schema"])
        res = dec.generate(ids, w["schema"])
        assert res.tokens == ref.tokens, (w["name"], res.text, rt.tokenizer.decode(ref.tokens))
        assert res.passes < ref.passes
        assert json.loads(res.text)


def test_regions(rt):
    import numpy as np

    from locals.regions import RegionScorer, canonical_tokens, region_completions
    from locals.regex import DEAD

    labels = ["billing", "billing_refund", "technical", "technical_bug", "other"]
    schema = {"type": "object", "properties": {"category": {"enum": labels}}, "required": ["category"]}
    c = rt.compile(schema)
    dfa = c.constraint.dfa
    q0 = dfa.run(b'{"category":"', dfa.start)
    exit_state = dfa.run(b'{"category":"other"', dfa.start)
    comps = region_completions(dfa, q0, [exit_state])
    assert sorted(w[:-1].decode() for w in comps) == sorted(labels)
    # brute force: every byte string of length <= 15 that reaches the exit
    brute = set()
    frontier = [(q0, b"")]
    for _ in range(15):
        nxt = []
        for q, w in frontier:
            for b in range(256):
                nq = int(dfa.table[q, b])
                if nq == DEAD:
                    continue
                if nq == exit_state:
                    brute.add(w + bytes([b]))
                else:
                    nxt.append((nq, w + bytes([b])))
        frontier = nxt
    assert brute == set(comps)
    hf = rt.tokenizer._tokenizer
    ids = rt.encode_chat("Ticket: I was charged twice. Classify it.") + hf.encode('{"category":"', add_special_tokens=False)
    scorer = RegionScorer(rt.model)
    cache = scorer.prefill(ids[:-1])
    cands = [canonical_tokens(hf, w) for w in comps]
    batched = scorer.logprobs(cache, ids[-1], cands)
    single = [scorer.logprobs(scorer.prefill(ids[:-1]), ids[-1], [cd])[0] for cd in cands]
    assert np.allclose(batched, single, atol=0.05), (batched, single)


def test_lazy_rows(rt):
    import numpy as np
    from workloads import prompt_text
    from workloads import WORKLOADS

    from locals.blocks import NoHealController as LocalsController
    from locals.lazy import LazyController, compile_lazy
    from outlines_core.json_schema import build_regex_from_schema

    w = WORKLOADS[0]
    rx = build_regex_from_schema(json.dumps(w["schema"]), "")
    lz, planner = compile_lazy(rt, rx)
    c = rt.compile(regex=rx)
    ids = rt.encode_chat(prompt_text(w, True))
    res_l = rt.engine.generate(ids, LazyController(lz, planner, True, True), max_tokens=200, pipeline=False)
    res_f = rt.engine.generate(ids, LocalsController(c.constraint, c.planner, True, True), max_tokens=200, pipeline=False)
    assert res_l.tokens == res_f.tokens
    assert 0 < lz.computed <= len(res_l.tokens)
    for q, (nxt, allowed) in lz._rows.items():
        assert np.array_equal(nxt.astype(np.int32), c.constraint.next_state[:, q])
        assert np.array_equal(np.array(allowed), c.constraint.allowed[q])


def test_walk_kernel(rt):
    import json as _json

    import numpy as np
    from workloads import prompt_text
    from outlines_core.json_schema import build_regex_from_schema
    from workloads import WORKLOADS

    from locals.blocks import NoHealController
    from locals.regex import DEAD
    from locals.walk import WalkController, compile_walk

    w = WORKLOADS[1]
    rx = build_regex_from_schema(_json.dumps(w["schema"]), "")
    wc, planner = compile_walk(rt, rx)
    c = rt.compile(regex=rx)
    # the kernel's rows equal the table's columns for every state
    import mlx.core as mx

    qs = mx.array(list(range(wc.n_states)), dtype=mx.int32)
    rows = np.array(wc.rows(qs)).astype(np.int32)
    T = c.constraint.next_state
    assert rows.shape == (wc.n_states, T.shape[0])
    assert np.array_equal(rows.T, T), int((rows.T != T).sum())
    # and the pipelined engine decodes the same tokens with either controller
    ids = rt.encode_chat(prompt_text(w, True))
    a = rt.engine.generate(ids, WalkController(wc, planner, True, True), max_tokens=200)
    b = rt.engine.generate(ids, NoHealController(c.constraint, c.planner, True, True), max_tokens=200)
    assert a.tokens == b.tokens
    assert wc.nbytes < 8_000_000
