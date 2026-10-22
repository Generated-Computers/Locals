"""set_compute_dtype on a GPU without native bf16: fp16 when it is at least as close to a float32 run as the
model's own bf16 (no model: a stand-in whose output in each dtype is given)."""

import mlx.core as mx
import mlx.nn as nn

from locals import runtime


class _Model(nn.Module):
    """Logits that depend only on the dtype its one weight is in."""

    def __init__(self, by_dtype):
        super().__init__()
        self.w = mx.ones((4,), dtype=mx.bfloat16)
        self._by = by_dtype

    def __call__(self, ids):
        return mx.array(self._by[self.w.dtype])[None, None, :].astype(mx.float32)


def _choose(monkeypatch, by_dtype):
    monkeypatch.setattr(runtime, "native_bf16", lambda: False)
    monkeypatch.setattr(runtime, "_fp16_special", lambda model: False)
    m = _Model(by_dtype)
    return runtime.set_compute_dtype(m, "auto"), m.w.dtype


def test_fp16_is_kept_when_it_is_closer_to_float32_than_bf16(monkeypatch):
    # bf16 is 4.0 off on the largest logit, fp16 0.25: the old check (fp16 against bf16) refused it
    got = _choose(monkeypatch, {mx.bfloat16: [26.0, 1, 0, 0], mx.float16: [22.25, 1, 0, 0], mx.float32: [22.0, 1, 0, 0]})
    assert got == ("float16", mx.float16)


def test_bf16_is_kept_when_fp16_is_further_from_float32(monkeypatch):
    got = _choose(monkeypatch, {mx.bfloat16: [22.0, 1, 0, 0], mx.float16: [26.0, 1, 0, 0], mx.float32: [22.1, 1, 0, 0]})
    assert got == ("bfloat16", mx.bfloat16)


def test_fp16_that_changes_the_top_token_is_refused(monkeypatch):
    got = _choose(monkeypatch, {mx.bfloat16: [5.0, 1, 0, 0], mx.float16: [0.5, 9.0, 0, 0], mx.float32: [5.0, 1, 0, 0]})
    assert got == ("bfloat16", mx.bfloat16)


def test_fp16_close_to_bf16_is_kept_without_a_float32_run(monkeypatch):
    calls = []
    by = {mx.bfloat16: [10.0, 1, 0, 0], mx.float16: [10.1, 1, 0, 0]}  # no float32 entry: it must not be needed

    class Counting(_Model):
        def __call__(self, ids):
            calls.append(self.w.dtype)
            return super().__call__(ids)

    monkeypatch.setattr(runtime, "native_bf16", lambda: False)
    monkeypatch.setattr(runtime, "_fp16_special", lambda model: False)
    m = Counting(by)
    assert runtime.set_compute_dtype(m, "auto") == "float16" and mx.float32 not in calls


def test_a_model_that_declares_itself_float16_safe_takes_fp16_without_the_probe(monkeypatch):
    """A model that keeps its own values in float16's range (Flint) says so: fp16 without running the model, even
    where the probe would keep bf16 or its source mentions float16; on a GPU with native bf16 the probe decides."""
    by = {mx.bfloat16: [22.0, 1, 0, 0], mx.float16: [26.0, 1, 0, 0], mx.float32: [22.1, 1, 0, 0]}  # the probe: bf16

    class Safe(_Model):
        float16_safe = True

        def __call__(self, ids):
            raise AssertionError("the probe ran")

    monkeypatch.setattr(runtime, "native_bf16", lambda: False)
    monkeypatch.setattr(runtime, "_fp16_special", lambda model: True)
    m = Safe(by)
    assert runtime.set_compute_dtype(m, "auto") == "float16" and m.w.dtype == mx.float16
    monkeypatch.setattr(runtime, "native_bf16", lambda: True)
    m = Safe(by)
    assert runtime.set_compute_dtype(m, "auto") == "bfloat16" and m.w.dtype == mx.bfloat16
