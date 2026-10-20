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


def test_a_gguf_model_is_pointed_at_the_mlx_version_of_the_same_size(tmp_path, monkeypatch):
    """By downloads alone, a 4B GGUF was pointed at lmstudio-community/Qwen3.5-9B-MLX-4bit."""
    import huggingface_hub
    from types import SimpleNamespace

    class Hub:  # Hugging Face's answer, most downloaded first
        def list_models(self, **kw):
            return [SimpleNamespace(id=i) for i in ("lmstudio-community/Qwen3.5-9B-MLX-4bit", "someone/Qwen3.5-4B-mlx-8bit",
                                                    "mlx-community/Qwen3.5-4B-MLX-4bit")]

    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    f = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    f.write_bytes(b"GGUF")
    assert "(--model mlx-community/Qwen3.5-4B-MLX-4bit)" in d.gguf_hint(str(f))


def test_ollamas_names_for_the_models():
    """Someone coming from Ollama types its names: qwen3.5:4b, gemma4:e4b, qwen3.5 (its latest)."""
    assert d.model_id("qwen3.5:4b") == d.MODELS["qwen3.5-4b"] and d.model_id("qwen3.5:2b") == d.MODELS["qwen3.5-2b"]
    assert d.model_id("gemma4:e4b") == d.MODELS["gemma-4-e4b"] and d.model_id("qwen3.5:latest") == d.MODELS["qwen3.5-4b"]
    assert d.model_id("llama3.2:3b") == "llama3.2:3b"  # not one of these: as given (and refused with a reason)


def test_an_ollama_name_is_pointed_at_its_mlx_version(monkeypatch):
    """An Ollama name locals does not know failed with "Repo id must use alphanumeric chars" (its download)."""
    import huggingface_hub
    from types import SimpleNamespace

    class Hub:
        def list_models(self, search, **kw):
            assert search in ("llama-3.2", "llama3.2")
            return [SimpleNamespace(id=i) for i in ("mlx-community/Llama-3.2-1B-Instruct-4bit",
                                                    "someone/Llama-3.2-3B-Instruct-8bit", "mlx-community/Llama-3.2-3B-Instruct-4bit")]

    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    why = d.ollama_hint("llama3.2:3b")
    assert "is an Ollama model name" in why and "(--model mlx-community/Llama-3.2-3B-Instruct-4bit)" in why
    assert d.ollama_hint("mlx-community/Qwen3.5-4B-MLX-4bit") is None and d.ollama_hint("qwen") is None


def test_the_name_a_model_has_in_the_api():
    """locals/<name>: the repository or folder name without its 4-bit suffix, in any capitalization."""
    assert d.short_name("mlx-community/Qwen3.5-4B-MLX-4bit") == "qwen3.5-4b"
    assert d.short_name("mlx-community/MiniCPM5-2B-mlx-4Bit") == "minicpm5-2b"
    assert d.short_name("LiquidAI/LFM2.5-2.6B-MLX-4bit") == "lfm2.5-2.6b"
    assert d.short_name("mlx-community/gemma-4-e4b-it-4bit") == "gemma-4-e4b-it"
    assert d.short_name("mlx-community/Qwen3-0.6B-8bit") == "qwen3-0.6b-8bit"  # other precisions keep theirs
    assert d.short_name("/Users/me/models/FrogNano-4B/") == "frognano-4b"
