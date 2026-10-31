"""Qwen3.5 reads an image through the server's own path (the image taken from the request in prepare, the tower and
the M-RoPE prefill on the engine's thread), and unloading gives the model's memory back while the server stays."""

import base64
import io

import mlx.core as mx
import pytest

# Qwen3.5 (M-RoPE over 64 of 256 dims) and Qwen3-VL (over all 128, and deepstack features in its first layers)
MODELS = ["mlx-community/Qwen3.5-4B-MLX-4bit", "mlx-community/Qwen3-VL-8B-Instruct-4bit"]
pytestmark = pytest.mark.model


@pytest.fixture(scope="module", params=MODELS)
def server(request):
    from locals.defaults import cached_snapshot

    if cached_snapshot(request.param, ["config.json", "*.safetensors"]) is None:  # (its README may be missing)
        pytest.skip(f"{request.param} not cached")
    from locals.server import Server

    srv = Server(request.param, engines=("locals",), preload=("locals",))
    yield srv
    srv.unload()


def _invoice() -> str:
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("RGB", (900, 500), "white")
    g = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except OSError:
        pytest.skip("no Helvetica to draw the text with")
    for i, line in enumerate(["Invoice 7319", "Date: 2026-10-31", "Total: 1,284.50 PLN", "Customer: Stefan Nowak"]):
        g.text((40, 30 + i * 85), line, fill="black", font=font)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _ask(srv, url: str, question: str) -> str:
    req = {"model": "locals", "max_tokens": 60, "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": question}]}]}
    return srv.complete(req)["choices"][0]["message"]["content"]


def test_reads_the_image_and_unloads(server):
    url = _invoice()
    assert server.reads_images()
    assert "Stefan Nowak" in _ask(server, url, "What is the customer's name?")

    loaded = mx.get_active_memory()
    r = server.unload()
    assert r["status"] == "unloaded" and r["engines"] == ["locals"]
    assert "locals" not in server._svc and not server.stats()["engines"]["locals"]["loaded"]
    assert "vision" in server.ollama_show("")["capabilities"]  # (from the checkpoint: the model is not loaded)
    assert mx.get_active_memory() < loaded * 0.1  # (the weights, prompt cache and vision tower all let go)

    # the next request loads the model again
    assert "7319" in _ask(server, url, "What is the invoice number?")
    assert "locals" in server._svc
