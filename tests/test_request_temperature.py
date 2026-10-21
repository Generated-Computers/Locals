"""A request's own temperature, in a batch with others (BatchService on our
engine): temperature 0 is greedy even when the service samples by default and
the same batch holds a sampled request. Skipped if the model is not cached."""

import os

import pytest

pytest.importorskip("mlx_lm")

MODEL = os.environ.get("LOCALS_TEST_MODEL", "mlx-community/Qwen3.5-2B-MLX-4bit")


@pytest.fixture(scope="module")
def service():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    from locals.service import BatchService

    svc = BatchService("locals", MODEL, slots=4, temp=1.0)  # samples by default
    yield svc
    svc.close()


def _prompt(tok, q):
    return tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, enable_thinking=False)


def test_greedy_request_in_a_sampling_batch(service):
    tok = service.tokenizer
    p = _prompt(tok, "Name three rivers in Europe, comma separated.")
    alone = service.submit(p, max_tokens=24, temp=0.0).wait()
    hot = [service.submit(_prompt(tok, "Write one surprising word."), max_tokens=24, temp=1.5) for _ in range(2)]
    mixed = service.submit(p, max_tokens=24, temp=0.0)  # shares the batch with the sampled rows
    mixed.wait()
    for h in hot:
        h.wait()
    assert alone.tokens == mixed.tokens, (alone.text, mixed.text)


def test_a_seeded_request_gives_the_same_reply(service):
    """A request's seed was ignored: the same request twice, sampled, gave two replies."""
    tok = service.tokenizer
    p = _prompt(tok, "Write a sentence about a cat, in your own words.")
    a = service.submit(p, max_tokens=24, temp=1.5, seed=7).wait()
    b = service.submit(p, max_tokens=24, temp=1.5, seed=7).wait()
    others = [service.submit(p, max_tokens=24, temp=1.5, seed=s).wait() for s in (8, 9)]
    assert a.tokens == b.tokens, (a.text, b.text)
    assert any(o.tokens != a.tokens for o in others)


def test_top_k_one_is_the_greedy_reply(service):
    """A request's own top_k reaches its row: top_k 1 at a high temperature is the greedy reply."""
    tok = service.tokenizer
    p = _prompt(tok, "Name three rivers in Europe, comma separated.")
    greedy = service.submit(p, max_tokens=24, temp=0.0).wait()
    k1 = service.submit(p, max_tokens=24, temp=1.5, top_k=1).wait()
    assert greedy.tokens == k1.tokens, (greedy.text, k1.text)
