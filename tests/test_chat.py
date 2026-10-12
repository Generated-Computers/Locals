"""`locals chat` builds every turn in the model's own chat template (CPU only: the
tokenizers of the cached models). It used to append Qwen's ChatML markers to any
model's conversation from the second turn on."""

import json
from pathlib import Path

import pytest

pytest.importorskip("mlx_lm")


@pytest.mark.parametrize("model,end,tail", [
    ("mlx-community/gemma-4-e4b-it-4bit", "<turn|>", "<|turn>user\nand you?<turn|>\n<|turn>model\n"),
    ("mlx-community/Qwen3.5-4B-MLX-4bit", "<|im_end|>", "<|im_start|>user\nand you?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
])
def test_second_turn_in_the_models_template(model, end, tail):
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer

    from locals.chat import Conversation

    try:
        path = Path(snapshot_download(model, local_files_only=True))
    except Exception:
        pytest.skip("model not cached")
    tok = load_tokenizer(path, eos_token_ids=json.loads((path / "config.json").read_text()).get("eos_token_id"))
    conv = Conversation(tok, "Be brief.")
    first = conv.prompt("hi", [])
    reply = tok.encode("Hello there!", add_special_tokens=False) + [tok._tokenizer.convert_tokens_to_ids(end)]
    history = list(first) + reply
    conv.add("hi", reply)
    second = conv.prompt("and you?", history)
    text = tok.decode(second)
    assert text.endswith(tail) and text.count("Hello there!") == 1
    assert "<|im_start|>" not in text or "im_start" in tail
    assert second[:len(history)] == history  # the cached conversation is reused
    conv.reset()
    assert "Hello there!" not in tok.decode(conv.prompt("new", []))
