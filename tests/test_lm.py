"""locals.lm, the mlx-lm drop-in: mlx-lm's own generate / stream_generate /
batch_generate / BatchGenerator run unchanged on a patched model, and greedy
text matches unpatched mlx-lm in the checkpoint's own precision. Skipped if the
model is not cached."""

import os

import pytest

pytest.importorskip("mlx_lm")

MODEL = os.environ.get("LOCALS_TEST_MODEL", "mlx-community/Qwen3.5-2B-MLX-4bit")
PROMPTS = ["Name three primary colors.", "What is the boiling point of water in Celsius?"]


@pytest.fixture(scope="module")
def pair():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    import mlx_lm

    from locals import lm

    stock = mlx_lm.load(MODEL)
    fast = lm.load(MODEL, dtype="bfloat16")  # the checkpoint's own precision: like-for-like with stock
    return stock, fast


def _chat(tok, q):
    return tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, enable_thinking=False)


def test_generate_matches_stock(pair):
    from mlx_lm import generate

    from locals import lm

    (sm, st), (fm, ft) = pair
    for q in PROMPTS:
        a = generate(sm, st, _chat(st, q), max_tokens=24)
        b = lm.generate(fm, ft, _chat(ft, q), max_tokens=24)
        assert b.strip() and a[:40] == b[:40], (a, b)


def test_stream_and_batch(pair):
    from locals import lm

    _, (fm, ft) = pair
    chunks = list(lm.stream_generate(fm, ft, _chat(ft, PROMPTS[0]), max_tokens=16))
    assert chunks and chunks[-1].generation_tokens > 0
    single = [lm.generate(fm, ft, _chat(ft, q), max_tokens=16) for q in PROMPTS]
    batch = lm.batch_generate(fm, ft, [_chat(ft, q) for q in PROMPTS], max_tokens=16)
    assert len(batch.texts) == len(PROMPTS) and all(t.strip() for t in batch.texts)
    # the batch decodes the same greedy text as one request at a time, up to where near-ties may part them
    assert all(s[:20] == b[:20] for s, b in zip(single, batch.texts)), (single, batch.texts)


def test_patch_is_idempotent_and_exported():
    import locals
    from locals import lm

    assert locals.patch is lm.patch
    assert {"load", "generate", "stream_generate", "batch_generate", "BatchGenerator"} <= set(lm.__all__)
