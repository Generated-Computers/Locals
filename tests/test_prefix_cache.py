"""Continuing from the prompt cache gives the tokens a fresh prefill gives.

The batch engine reuses a cached prefix without copying it (its buffers become
the batch cache, grown once to the prompt's length) and moves a finished batch
of one into the prompt cache without copying: greedy outputs of a follow-up
request that hits the cache must equal those of a server with an empty cache.
Skipped if the model is not cached."""

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

    svc = BatchService("locals", MODEL, slots=4, temp=0.0)
    yield svc
    svc.close()


def _prompt(tok, turns):
    doc = "\n".join(f"Line {i}: the inventory of shelf {i % 7} was counted on day {i % 30} and found complete." for i in range(220))
    msgs = [{"role": "system", "content": "You answer questions about the document.\n" + doc}]
    for q, a in turns:
        msgs.append({"role": "user", "content": q})
        if a is not None:
            msgs.append({"role": "assistant", "content": a})
    return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)


def test_follow_up_from_cache_equals_fresh(service):
    """The follow-up continues the first request's token stream, its end-of-turn
    token included, as a chat's next turn does (a hybrid model's recurrent state
    cannot be cut back, so only a continuation of a cached stream reuses it)."""
    from locals.service import BatchService

    tok = service.tokenizer
    first = _prompt(tok, [("Which shelf is on line 12?", None)])
    a = service.submit(first, max_tokens=24).wait()
    assert a.finish == "stop"
    turn = "<|im_end|>\n<|im_start|>user\nAnd on line 40? Answer briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    follow = list(first) + list(a.tokens) + list(tok.encode(turn, add_special_tokens=False))
    b = service.submit(follow, max_tokens=24).wait()
    assert b.cached > len(first) // 2, (b.cached, len(first))  # the follow-up continued from the cache
    fresh = BatchService("locals", MODEL, slots=4, temp=0.0, cache_gb=0.001)  # effectively no prompt cache
    try:
        c = fresh.submit(follow, max_tokens=24).wait()
    finally:
        fresh.close()
    assert c.cached == 0 and b.tokens == c.tokens, (b.text, c.text)
