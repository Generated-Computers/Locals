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
