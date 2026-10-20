"""The stock engine (mlx/...) keeps serving after an error inside its loop, as ours does: one error left it failing
every later request until the server restarted."""

import os

import pytest

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"


@pytest.mark.model
def test_the_stock_engine_serves_on_after_an_error():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3.5-4B not cached")
    from locals.service import BatchService

    saved = os.environ.get("LOCALS_DISK_CACHE")
    os.environ["LOCALS_DISK_CACHE"] = "0"
    svc = BatchService("stock", MODEL, slots=4, temp=0.0)
    try:
        tok = svc.tokenizer

        def boom(tokens):
            raise RuntimeError("a failure inside the engine's loop")

        bad = svc.submit(tok.encode("Hello"), max_tokens=8, on_token=boom)
        bad.wait(600)
        assert bad.finish.startswith("error")
        good = svc.submit(tok.encode("What is 2 + 2?"), max_tokens=8)
        good.wait(600)
        assert good.finish in ("length", "stop") and len(good.tokens) > 0, good.finish
    finally:
        svc.close()
        if saved is None:
            os.environ.pop("LOCALS_DISK_CACHE", None)
        else:
            os.environ["LOCALS_DISK_CACHE"] = saved
