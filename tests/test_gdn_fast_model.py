"""Qwen3.5-4B reading a prompt with MLX's mx.fast.gated_delta_update in its recurrent layers (kernels.gdn_fast) gives
what it gives with mlx-lm's kernel, and the choice is looked up under the keys tuning stores it at."""

import mlx.core as mx
import pytest

from locals import kernels

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
pytestmark = [pytest.mark.model, pytest.mark.skipif(not hasattr(mx.fast, "gated_delta_update"), reason="MLX without the op")]


@pytest.fixture(scope="module")
def rt():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3.5-4B not cached")
    from locals.runtime import Runtime

    import gc

    yield Runtime.load(MODEL, tune="off")
    gc.collect()  # (the next module measures the memory in use)
    mx.clear_cache()


def test_same_answers_and_keys(rt, monkeypatch):
    from mlx_lm.models.cache import make_prompt_cache

    lm = kernels.text_model(rt.model)
    x = mx.array([rt.tokenizer.encode("Explain, step by step, how a compiler turns source code into a program. " * 40)])
    seen = []
    real = kernels.gdn_fast
    monkeypatch.setattr(kernels, "gdn_fast", lambda *a: seen.append(a) or real(*a))
    out = {}
    for fast in (False, True):
        monkeypatch.setattr(kernels, "_GDN_FORCE", fast)
        out[fast] = lm(x, cache=make_prompt_cache(lm))[0, -1].astype(mx.float32)
        mx.eval(out[fast])
    assert int(mx.argmax(out[True])) == int(mx.argmax(out[False]))
    assert float(mx.max(mx.abs(out[True] - out[False]))) < 0.05 * float(mx.max(mx.abs(out[False])))
    stored = {kernels._gdn_key(Hk, Hv, Dk, Dv, str(dt).split(".")[-1], kernels._gdn_size(x.shape[1])) for _, Hk, Hv, Dk, Dv, dt in kernels._gdn_layers(rt.model)}
    looked = {kernels._gdn_key(Hk, Hv, Dk, Dv, str(dt).split(".")[-1], kernels._gdn_size(S)) for Hk, Hv, Dk, Dv, dt, S in seen}
    assert seen and looked == stored, (looked, stored)
