"""Exactness checks that run the real 1.5B model (slow; skipped if not cached).

  * pipelined and synchronous decoding give identical greedy outputs
  * lossless forced spans give the same greedy output as plain constrained decoding
  * every constrained output matches the regex
  * tree-KV branches reproduce single-sequence greedy decoding
"""

import json
import re

import pytest

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


@pytest.fixture(scope="module")
def rt():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    from locals.runtime import Runtime

    return Runtime.load(MODEL)


def test_pipelined_equals_sync(rt):
    from workloads import WORKLOADS, make_prompt

    from locals.engine import LocalsController

    for w in WORKLOADS[:3]:
        c = rt.compile(w["schema"])
        p = make_prompt(rt.tokenizer, w)
        a = rt.engine.generate(p, LocalsController(c.constraint, c.planner, True, True), max_tokens=200, pipeline=True)
        b = rt.engine.generate(p, LocalsController(c.constraint, c.planner, True, True), max_tokens=200, pipeline=False)
        assert a.tokens == b.tokens
        free_a = rt.engine.generate(p, None, max_tokens=60, pipeline=True)
        free_b = rt.engine.generate(p, None, max_tokens=60, pipeline=False)
        assert free_a.tokens == free_b.tokens


def test_lossless_forced_equals_plain_and_outputs_valid(rt):
    from outlines_core.json_schema import build_regex_from_schema
    from workloads import WORKLOADS, make_prompt

    from locals.engine import LocalsController
    from locals.forced import ForcedPlanner

    for w in WORKLOADS:
        c = rt.compile(w["schema"])
        lossless = ForcedPlanner(rt.tokenizer._tokenizer, rt.vocab, c.constraint, commit_stable_tail=False, boundary_rule=False)
        p = make_prompt(rt.tokenizer, w)
        plain = rt.engine.generate(p, LocalsController(c.constraint, lossless, use_forced=False), max_tokens=256)
        forced = rt.engine.generate(p, LocalsController(c.constraint, lossless, use_forced=True), max_tokens=256)
        assert plain.text == forced.text
        assert forced.passes < plain.passes
        rx = re.compile(build_regex_from_schema(json.dumps(w["schema"]), ""))
        full = rt.generate(p, schema=w["schema"], max_tokens=256)
        assert rx.fullmatch(full.text), full.text
        json.loads(full.text)


def test_tree_branches_equal_single_greedy(rt):
    p = rt.encode_chat("List three colors of the rainbow.")
    ref = rt.engine.generate(p, None, max_tokens=30)
    tr = rt.sample(p, n=3, max_tokens=30, temp=0.0)
    for b in tr.branches:
        assert b == ref.tokens
