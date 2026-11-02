"""Gemma 4 reads an image through the server's own path: the processor's patches in prepare, the tower's features on
the engine's thread, the PLE chunk and no M-RoPE in the prefill, the reply decoded from the text model's own rope.

The model this runs on is the 5-bit E2B (2B effective, 5.1 GB on disk); the same code path reads E4B's tower."""

import base64
import io

import mlx.core as mx
import pytest

MODEL = "mlx-community/gemma-4-e2b-it-5bit"
pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def server():
    from locals.defaults import cached_snapshot

    if cached_snapshot(MODEL, ["config.json", "*.safetensors"]) is None:
        pytest.skip(f"{MODEL} not cached")
    from locals.server import Server

    srv = Server(MODEL, engines=("locals",), preload=("locals",))
    yield srv
    srv.unload()


def _card(lines, size=(700, 300)) -> str:
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("RGB", size, "white")
    g = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 34)
    except OSError:
        pytest.skip("no Helvetica to draw the text with")
    for i, line in enumerate(lines):
        g.text((30, 20 + i * 60), line, fill="black", font=font)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _ask(srv, messages, max_tokens: int = 40) -> str:
    req = {"model": "locals", "max_tokens": max_tokens, "temperature": 0, "messages": messages}
    return srv.complete(req)["choices"][0]["message"]["content"]


def _turn(url: str, question: str) -> list:
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                        {"type": "text", "text": question}]}]


def test_gemma_reads_the_image(server):
    url = _card(["Invoice 7319", "Date: 2026-10-31", "Total: 1,284.50 PLN", "Customer: Stefan Nowak"])
    assert server.reads_images()
    assert server.loaded("locals").vision_family == "gemma"
    assert "Stefan Nowak" in _ask(server, _turn(url, "What is the customer's name?"))


def test_gemma_reads_two_images_and_a_follow_up(server):
    a = _card(["Card A: red circle", "number 41"])
    b = _card(["Card B: blue square", "number 97"])
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": a}},
                                        {"type": "image_url", "image_url": {"url": b}},
                                        {"type": "text", "text": "Which number is on the card with the blue "
                                                                 "square? Answer with the number only."}]}]
    first = _ask(server, msgs, max_tokens=12)
    assert "97" in first
    # the follow-up turn reuses the image prompt's cache: the same picture is not read again
    msgs = msgs + [{"role": "assistant", "content": first},
                   {"role": "user", "content": "And the red circle's number?"}]
    assert "41" in _ask(server, msgs, max_tokens=12)


def test_gemma_reads_text_again_after_an_image(server):
    # (no M-RoPE is left installed: the next text prompt decodes with the layers' own rope)
    assert _ask(server, [{"role": "user", "content": "Reply with just the word: ok"}], max_tokens=8).strip(" .!") == "ok"
