"""A model whose weights do not fit next to the GPU memory a server needs is refused before its
download and before loading (no model)."""

import locals.defaults as d


def test_too_big_names_the_sizes(monkeypatch, tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 1024)
    monkeypatch.delenv("LOCALS_IGNORE_MEMORY", raising=False)
    monkeypatch.setattr(d, "gpu_limit_gb", lambda: 10.7)
    monkeypatch.setattr(d, "weights_gb", lambda m, remote=False: 15.0)
    msg = d.too_big("mlx-community/Qwen3.8-27B-4bit", remote=True)
    assert msg and "15.0 GB" in msg and "10.7 GB" in msg and "LOCALS_IGNORE_MEMORY" in msg
    monkeypatch.setattr(d, "weights_gb", lambda m, remote=False: 2.8)
    assert d.too_big("mlx-community/Qwen3.5-4B-MLX-4bit") is None


def test_unknown_size_or_override_is_not_refused(monkeypatch):
    monkeypatch.setattr(d, "gpu_limit_gb", lambda: 10.7)
    monkeypatch.setattr(d, "weights_gb", lambda m, remote=False: None)
    assert d.too_big("some/model") is None
    monkeypatch.setattr(d, "weights_gb", lambda m, remote=False: 40.0)
    monkeypatch.setenv("LOCALS_IGNORE_MEMORY", "1")
    assert d.too_big("some/model") is None


def test_weights_of_a_local_folder(tmp_path):
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"\0" * (1 << 20))
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"\0" * (1 << 20))
    (tmp_path / "config.json").write_text("{}")
    assert abs(d.weights_gb(str(tmp_path)) - 2 / 1024) < 1e-9


def test_a_gguf_model_is_named_as_such(tmp_path, monkeypatch):
    import huggingface_hub

    class NoHub:  # offline: no MLX versions to name
        def list_models(self, **kw):
            return []

    monkeypatch.setattr(huggingface_hub, "HfApi", NoHub)
    f = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    f.write_bytes(b"GGUF")
    assert "is a GGUF model" in d.gguf_hint(str(f)) and "mlx_lm.convert" in d.gguf_hint(str(f))
    assert "is a GGUF model" in d.gguf_hint(str(tmp_path))  # a folder of .gguf files
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert d.gguf_hint(str(tmp_path)) is None  # MLX weights there too
    assert d.gguf_hint("mlx-community/Qwen3.5-4B-MLX-4bit", remote=False) is None
