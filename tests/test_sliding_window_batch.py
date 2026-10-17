"""Prompts read together continue as they would alone on a model with sliding-window layers (Gemma 4 E4B, a
512-token window): rows right-padded to the longest must not let the padding evict a shorter row's last real
tokens from its window. Without the rows' lengths, 11 of 24 benchmark prompts past the window got empty replies
(the end of turn first) when read 8 at a time, and none alone. Slow; skipped if the model is not cached."""

import random

import pytest

pytestmark = pytest.mark.model

GEMMA = "mlx-community/gemma-4-e4b-it-4bit"


@pytest.fixture(scope="module")
def eng():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(GEMMA, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Gemma 4 E4B not cached")
    from locals.hybrid_batch import HybridBatchEngine
    from locals.runtime import Runtime

    rt = Runtime.load(GEMMA)
    return HybridBatchEngine(rt.model, rt.tokenizer, share_prefix=False)  # (each prompt read in full every time)


def _prompts(eng, lens):
    rng = random.Random(0)
    words = "the a of river stone garden window quickly blue seven argue paper lamp north".split()
    ids = []
    for n in lens:
        text = " ".join(rng.choice(words) for _ in range(n))
        ids.append(eng.tokenizer.encode(text)[:n])
    return ids


def _alone(eng, p, feed):
    cache = eng._prefill_batch([p[:-1]], [])
    return eng._forward(cache, [[p[-1]] + feed])[0]


def _close(a, b):
    """The same logits up to float16's batching noise: a row padded to a longer one is 0.16-0.64 off its own
    run (two copies of one 980-token prompt, unpadded, 0.36); with the padding in its window, 13.8-25.4."""
    import mlx.core as mx

    assert float(mx.max(mx.abs(a - b))) < 2.0, float(mx.max(mx.abs(a - b)))


def test_prompts_past_the_window_read_together_continue_as_alone(eng):
    prompts = _prompts(eng, [640, 980, 830])  # all past the 512-token window, of different lengths
    alone = [_alone(eng, p, []) for p in prompts]
    cache = eng._prefill_batch([p[:-1] for p in prompts], [])
    together = eng._forward(cache, [[p[-1]] for p in prompts])
    for a, b in zip(alone, together):
        _close(a, b)


def test_ragged_feeds_past_the_window(eng):
    """A pass in which rows feed different numbers of tokens (forced spans, drafts) is right-padded too."""
    prompts = _prompts(eng, [700, 900])
    feeds = [eng.tokenizer.encode(" river stone garden")[-3:], []]
    alone = [_alone(eng, p, f) for p, f in zip(prompts, feeds)]
    cache = eng._prefill_batch([p[:-1] for p in prompts], [])
    together = eng._forward(cache, [[p[-1]] + f for p, f in zip(prompts, feeds)])
    for a, b in zip(alone, together):
        _close(a, b)
