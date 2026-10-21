"""Test configuration.

Tests that run a real model are marked ``model`` (slow, and skipped when the
weights are not in the local Hugging Face cache). The fast suite is
``pytest -m "not model"``; it needs only a tokenizer.
"""

import pytest

# Test modules that load model weights.
MODEL_TEST_FILES = {
    "test_blocks.py",
    "test_budget.py",
    "test_engine_model.py",
    "test_lm.py",
    "test_lookup.py",
    "test_mixed_prefill.py",
    "test_pda.py",
    "test_prefix_cache.py",
    "test_q35_kernels.py",
    "test_request_temperature.py",
    "test_server_attention.py",
    "test_server.py",
    "test_walk_batch.py",
}


@pytest.fixture(autouse=True)
def _no_real_user_files(tmp_path_factory, monkeypatch):
    """No test reads or writes the user's own OpenCode config, ~/.locals or the prompt cache on
    disk (a test of `locals code` once rewrote ~/.config/opencode/opencode.json)."""
    monkeypatch.setenv("LOCALS_DISK_CACHE", "0")
    try:
        from locals import app
    except Exception:  # noqa: BLE001
        return
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setattr(app, "LOCALS_DIR", home / ".locals")
    monkeypatch.setattr(app, "OPENCODE_CONFIG", home / ".config" / "opencode" / "opencode.json")
    monkeypatch.setattr(app, "OPENCODE_BIN", home / ".opencode" / "bin" / "opencode")
    # what a server on the default port runs (the user's own may be up): none, unless a test says so
    monkeypatch.setattr(app, "_serving_short", lambda port: None, raising=False)


@pytest.fixture(autouse=True, scope="session")
def _tuning_cache_copy(tmp_path_factory):
    """Kernel tuning reads and writes a copy of the user's cache: its timings (a model test tunes nothing it
    has), but what a test tunes, under a loaded machine, never reaches ~/.cache/locals (a run once rewrote it)."""
    import os
    import shutil

    try:
        from locals import kernels
    except Exception:  # noqa: BLE001
        yield
        return
    copy = tmp_path_factory.mktemp("tune") / os.path.basename(kernels._TUNE_PATH)
    if os.path.exists(kernels._TUNE_PATH):
        shutil.copy(kernels._TUNE_PATH, copy)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kernels, "_TUNE_PATH", str(copy))
        mp.setattr(kernels, "_TUNE", None)
        mp.setenv("LOCALS_TUNE_CACHE", str(copy))  # (servers and tools a test starts)
        yield


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.fspath.basename in MODEL_TEST_FILES or "model" in item.keywords:
            item.add_marker(pytest.mark.model)


@pytest.fixture(autouse=True, scope="module")
def _free_gpu_memory_between_modules():
    """Each test module loads its own model copies; release them (and MLX's
    buffer cache) before the next module, or a 4B suite outgrows a 16 GB Mac's
    GPU limit (Metal "Insufficient Memory" in the later modules)."""
    yield
    import gc

    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """pytest keeps each test's fixture values in ``item.funcargs`` until the
    session ends, which keeps every module's model alive; drop them."""
    yield
    item.funcargs.clear()
