"""Prompt-lookup draft verification (locals/lookup.py, Engine._generate_lookup).

  * the fused GDN kernel's per-token states equal sequential single-token
    steps, and a rewound cache continues exactly like one that never saw the
    rejected tokens
  * greedy lookup decoding equals greedy decoding on regex, pushdown and free
    prompts (token-identical), on Qwen3.5-2B; skipped if not cached
"""

import os

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


def test_rewind_matches_sequential(rt):
    import mlx.core as mx

    from locals.engine import make_spec_cache, rewind, set_want_states
    from locals.kernels import text_model

    lm = text_model(rt.model)
    prompt = rt.encode_chat("Count from one to twenty in words.")
    a, b = make_spec_cache(rt.model), make_spec_cache(rt.model)
    x = mx.array([prompt])
    lm.model(x, a)
    lm.model(x, b)
    toks = [17, 220, 3070, 264, 1102, 25, 5, 4, 3]
    # a: one 9-token pass, keep 3, then feed the other 6 again as single tokens
    set_want_states(a, True)
    lm.model(mx.array([toks]), a)
    set_want_states(a, False)
    rewind(a, 6)
    outs_a = [lm.model(mx.array([[t]]), a) for t in toks[3:]]
    # b: the same 9 tokens one at a time
    outs_b = [lm.model(mx.array([[t]]), b) for t in toks]
    # c: the 9-token pass without a rewind: the fp16 kernel-path noise floor
    # between a multi-token pass and single-token steps (different matmul kernels)
    c = make_spec_cache(rt.model)
    lm.model(x, c)
    set_want_states(c, True)
    yc = lm.model(mx.array([toks]), c)
    err = lambda p, q: float(mx.max(mx.abs(p.astype(mx.float32) - q.astype(mx.float32))))
    floor = max(err(yc[:, i : i + 1], outs_b[i]) for i in range(3, 9))
    head = lm.model.embed_tokens.as_linear
    for i, (ya, yb) in enumerate(zip(outs_a, outs_b[3:])):
        assert err(ya, yb) <= 2 * floor + 0.05, (i, err(ya, yb), floor)
        assert int(mx.argmax(head(ya)[0, -1]).item()) == int(mx.argmax(head(yb)[0, -1]).item())
    ka = next(cc for cc in a if hasattr(cc, "offset"))
    kb = next(cc for cc in b if hasattr(cc, "offset"))
    assert ka.offset == kb.offset


def test_lookup_equals_greedy(rt):
    from workloads import JOBS, recursive_prompt as prompt_text
    from workloads import WORKLOADS, make_prompt

    cases = [(make_prompt(rt.tokenizer, w), w["schema"]) for w in WORKLOADS[:3]]
    cases += [(rt.encode_chat(prompt_text(s, t, 0)), s) for n, s, t in JOBS if n in ("thread", "fs")]
    cases += [(rt.encode_chat("Write a Python class LRUCache with get and put methods and a short usage example."), None)]
    drafted = 0
    for ids, schema in cases:
        mk = lambda: rt.compile(schema).controller() if schema else None
        a = rt.engine.generate(ids, mk(), max_tokens=200)
        b = rt.engine.generate(ids, mk(), max_tokens=200, lookup=True)
        assert a.tokens == b.tokens
        assert a.finish == b.finish
        drafted += b.drafted
    assert drafted > 10  # the drafts did something (2B: ~60 drafted tokens on these cases, 4B: ~25)
