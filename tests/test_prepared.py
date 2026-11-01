"""A prepared prompt (locals.prepared): a rendered prompt as the model reads it. Needs Qwen3.5-4B's tokenizer
(cached), no weights."""

import base64
import io

import pytest

from locals import vision

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
IMAGE = vision.IMAGE_MARK


@pytest.fixture(scope="module")
def tok():
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(MODEL, local_files_only=True, allow_patterns=["*.json", "*.jinja", "tokenizer*"])
    except Exception:  # noqa: BLE001
        pytest.skip("tokenizer not cached")
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(path)


@pytest.fixture(scope="module")
def picture():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (200, 10, 10)).save(buf, "PNG")
    pic, = vision.load(["data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()])
    return pic


def _encode(tok, text):
    return list(tok._tokenizer.encode(text, add_special_tokens=False))


def test_a_text_prompt_is_its_tokens(tok):
    from locals.prepared import PreparedPrompt

    p = PreparedPrompt.of("Hello there, how are you?", tok)
    assert p.tokens == _encode(tok, "Hello there, how are you?")
    assert p.pictures == () and p.spans == ()


def test_a_text_prompt_goes_on_from_a_remembered_stream(tok):
    """The ids the model saw for the text it continues (a reply's own tokens), then the rest's."""
    from locals.prepared import PreparedPrompt

    seen = [101, 102, 103]  # (not what the text encodes to: the model's own ids are kept)
    p = PreparedPrompt.of("Hello there, how are you?", tok, remembered=lambda text: ("Hello there", seen))
    assert p.tokens == seen + _encode(tok, ", how are you?")


def test_an_image_prompt_has_each_images_own_tokens(tok, picture):
    """In place of the image's one placeholder: as many image tokens as the image has. A remembered stream is not
    read for it (its ids hold another image's tokens)."""
    from locals.prepared import PreparedPrompt

    pad = tok.convert_tokens_to_ids(vision.IMAGE_TOKEN)
    text = "Look: " + IMAGE + " What is it?"
    p = PreparedPrompt.of(text, tok, (picture,), remembered=lambda text: pytest.fail("an image prompt reads no stream"))
    assert picture.tokens > 1
    assert p.tokens.count(pad) == picture.tokens
    assert len(p.tokens) == len(_encode(tok, text)) - 1 + picture.tokens
    (start, end), = p.spans
    assert p.tokens[start:end] == [pad] * picture.tokens and pad not in p.tokens[:start] + p.tokens[end:]
    assert p.pictures == (picture,)


LONG = "".join(f"Rule {i}: read a file before you edit it, and keep replies short.\n" for i in range(150))


def test_a_checkpoint_is_where_a_text_prefix_ends_in_the_prompts_tokens(tok):
    from locals.prepared import PreparedPrompt

    p = PreparedPrompt.of(LONG + "Now fix the failing test, please.", tok)
    n = len(_encode(tok, LONG))
    assert n >= 1024 and p.tokens[:n] == _encode(tok, LONG)
    assert p.checkpoint_at(LONG) == n


def test_a_checkpoint_after_an_image_counts_the_images_tokens(tok, picture):
    from locals.prepared import PreparedPrompt

    head = LONG + "Look: " + IMAGE + " What is it?\n"
    p = PreparedPrompt.of(head + "Answer in one word.", tok, (picture,))
    assert p.checkpoint_at(head) == len(_encode(tok, head)) - 1 + picture.tokens
    assert p.checkpoint_at(LONG) == len(_encode(tok, LONG))  # (before the image: nothing to count)


def test_a_short_prefix_another_text_or_the_whole_prompt_is_no_checkpoint(tok, picture):
    """A prefix under 1024 tokens is cheaper to read again than to keep; the whole prompt is no prefix of it."""
    from locals.prepared import PreparedPrompt

    text = LONG + "Now fix the failing test, please."
    p = PreparedPrompt.of(text, tok)
    assert p.checkpoint_at(LONG[:3000]) is None and len(_encode(tok, LONG[:3000])) < 1024
    assert p.checkpoint_at("Something else entirely. " + LONG) is None
    assert p.checkpoint_at(text) is None
    head = "Look: " + IMAGE + " What is it?\n"  # (under 1024 with the image's tokens too)
    q = PreparedPrompt.of(head + LONG, tok, (picture,))
    assert len(_encode(tok, head)) - 1 + picture.tokens < 1024 and q.checkpoint_at(head) is None


def test_only_a_text_prompts_reply_is_remembered(tok, picture):
    """A remembered stream is read for the next request's text (an image prompt's ids hold its image's tokens)."""
    from locals.prepared import PreparedPrompt

    assert PreparedPrompt.of("Hello there.", tok).rememberable
    assert not PreparedPrompt.of("Look: " + IMAGE, tok, (picture,)).rememberable
