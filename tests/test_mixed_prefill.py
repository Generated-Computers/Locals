"""Prompts prefilled in mixed passes (locals/mixed.py) while other rows decode give the replies a
prefill on its own gives, the decoding row goes on meanwhile, and checkpoints on the way reach the
prompt cache. Qwen3.5-2B; skipped if the model is not cached."""

import os
import threading

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


def _prompt(tok, topic, n):
    doc = "\n".join(f"Note {i}: the {topic} log entry {i} was checked on day {i % 30} and found in order." for i in range(n))
    msgs = [{"role": "user", "content": doc + f"\n\nIn one sentence, what are these notes about?"}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)


def _run(svc, mix: bool, checkpoint=None):
    from locals import mixed

    eng = svc._eng
    eng._mix_ok = mix and mixed.supported(eng.lm.model)
    eng.mix_chunk = 128  # several passes for these prompts (longer than one pass: mixed)
    svc._lru.trim_to(n_bytes=0)  # no prefix of an earlier run
    before = eng.counters.get("mixed_steps", 0)
    tok = svc.tokenizer
    started = threading.Event()
    seen = []

    def on_a(t):
        seen.extend(t)
        if len(seen) >= 4:
            started.set()

    a = svc.submit(_prompt(tok, "warehouse", 30), max_tokens=160, on_token=on_a)
    assert started.wait(120)
    pb, pc = _prompt(tok, "kitchen", 70), _prompt(tok, "garden", 55)
    b = svc.submit(pb, max_tokens=24, checkpoints=[(checkpoint, "user")] if checkpoint else None)
    c = svc.submit(pc, max_tokens=24)
    for j in (a, b, c):
        j.wait(600)
    return [a.tokens, b.tokens, c.tokens], eng.counters.get("mixed_steps", 0) - before, len(pb)


def test_mixed_prefill_equals_prefill_alone(service):
    plain, n_plain, _ = _run(service, mix=False)
    mixed_out, n_mixed, _ = _run(service, mix=True)
    assert n_plain == 0 and n_mixed >= 4  # the prompts went through mixed passes
    for p, m in zip(plain[1:], mixed_out[1:]):  # the prompts prefilled alongside the decoding row
        assert len(m) == len(p) and sum(x == y for x, y in zip(p, m)) >= 0.9 * len(p), (p, m)
        assert p[:8] == m[:8]
    a_plain, a_mixed = plain[0], mixed_out[0]  # the row that kept decoding meanwhile
    assert len(a_mixed) == len(a_plain) and a_plain[:16] == a_mixed[:16]


def test_checkpoint_reached_in_mixed_passes(service):
    _, _, n = _run(service, mix=True, checkpoint=None)
    cp = 300  # inside the second prompt, past a pass boundary
    assert n > cp + 50
    service._lru.trim_to(n_bytes=0)
    _, steps, _ = _run(service, mix=True, checkpoint=cp)
    assert steps >= 4
    kinds = service._lru.stats_by_type()
    assert kinds.get("user", {}).get("n_sequences", 0) >= 1


def test_a_short_prompt_is_read_before_a_long_one(service):
    """Prompts read beside a decoding row go shortest first: a short one that arrives during a long one does not
    wait for all of it (a 3k-token request waited 41 s behind a 31k-token one)."""
    import time

    from locals import mixed

    eng = service._eng
    eng._mix_ok = mixed.supported(eng.lm.model)
    eng.mix_chunk = 128
    service._lru.trim_to(n_bytes=0)
    tok = service.tokenizer
    started = threading.Event()
    a = service.submit(_prompt(tok, "warehouse", 30), max_tokens=400, on_token=lambda t: started.set())
    assert started.wait(120)
    long, short = service.submit(_prompt(tok, "kitchen", 400), max_tokens=4), None
    time.sleep(0.5)
    short = service.submit(_prompt(tok, "garden", 20), max_tokens=4)
    short.wait(600)
    long.wait(600)
    a.wait(600)
    assert short.t_done < long.t_done, (short.t_done - long.t_done)
