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
