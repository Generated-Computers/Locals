"""Gemma 4's image path: the resize and patch order its processor uses, the prompt tokens its template's one
``<|image|>`` expands to, the PLE chunk the text model reads with an image, and the tower's arithmetic.

Needs no weights: the checkpoint's ``config.json`` and tokenizer alone (cached with ``*.json``/``*.jinja``), and a
tiny tower whose numbers a reference computes beside it."""

import io
import math

import mlx.core as mx
import numpy as np
import pytest

from locals import vision

MODEL = "mlx-community/gemma-4-e2b-it-5bit"

# transformers' get_aspect_ratio_preserving_size, the oracle for vision.gemma_resize (image_processing_gemma4):
# the largest aspect-preserving size at most `max_patches`, both sides a multiple of pooling * patch.
def _reference_resize(height: int, width: int, patch_size: int = 16, max_patches: int = 280 * 9,
                      pooling_kernel_size: int = 3):
    total_px = height * width
    target_px = max_patches * patch_size**2
    factor = math.sqrt(target_px / total_px)
    side_mult = pooling_kernel_size * patch_size
    target_height = int(math.floor(factor * height / side_mult)) * side_mult
    target_width = int(math.floor(factor * width / side_mult)) * side_mult
    if target_height == 0 and target_width == 0:
        raise ValueError("0 x 0")
    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)
    return target_height, target_width


def _png(w: int, h: int, encode=None) -> bytes:
    from PIL import Image

    if encode is None:
        a = np.zeros((h, w, 3), np.uint8)
    else:
        a = encode(w, h)
    buf = io.BytesIO()
    Image.fromarray(a, "RGB").save(buf, "PNG")
    return buf.getvalue()


def test_gemma_resize_matches_the_processor():
    for h, w in [(800, 1000), (100, 100), (10, 100), (2160, 3840), (48, 48), (672, 960), (1, 300),
                 (3000, 100), (17, 23)]:
        assert vision.gemma_resize(h, w) == _reference_resize(h, w), (h, w)
    # a 960x672 image is exactly the budget: 60x42 patches = 280 soft tokens, so the processor does not resize it
    assert vision.gemma_resize(672, 960) == (672, 960)


def test_gemma_preprocess_patches_are_raster_order_channels_last():
    # each pixel says where it is (R = x, G = y, B = 7), so a patch's vector reads as its pixels
    raw = _png(960, 672, lambda w, h: np.stack([np.tile(np.arange(w), (h, 1)) % 256,
                                                np.repeat(np.arange(h), w).reshape(h, w) % 256,
                                                np.full((h, w), 7)], axis=-1).astype(np.uint8))
    pic = vision.preprocess(raw, vision.GEMMA)
    assert pic.family == vision.GEMMA and pic.merge == 9
    assert pic.grid == (42, 60) and pic.tokens == 280  # 42 * 60 / 9
    x = pic.patches
    assert x.shape == (42 * 60, 16 * 16 * 3)
    # patch 0 starts at (y=0, x=0) and runs x first inside the patch, channels last, rescaled to [0, 1]
    assert np.allclose(x[0, :6], np.array([0, 0, 7, 1, 0, 7]) / 255.0)
    # patch 1 starts 16 columns right, patch 60 is the next row
    assert np.allclose(x[1, :3], np.array([16, 0, 7]) / 255.0)
    assert np.allclose(x[60, :3], np.array([0, 16, 7]) / 255.0)
    # the last patch is the bottom-right one
    assert np.allclose(x[-1, -3:], np.array([959 % 256, 671 % 256, 7]) / 255.0)
    assert float(x.max()) <= 1.0


def test_gemma_expand_wraps_the_soft_tokens_with_boi_and_eoi():
    pic = vision.Picture(None, (6, 9), family=vision.GEMMA, merge=9)  # 54 patches: 6 soft tokens
    assert pic.tokens == 6 and pic.wrap == 2
    ids, spans = vision.expand([1, 2, 99, 3], 99, [pic], boi=7, eoi=8)
    assert ids == [1, 2, 7] + [99] * 6 + [8, 3] and spans == [(3, 9)]
    # expanded_at counts the two tokens around the soft ones
    for n, at in [(0, 0), (1, 1), (2, 2), (3, 10), (4, 11)]:
        assert vision.expanded_at(n, spans, [pic]) == at
    # a Qwen picture expands as before: no wrappers
    q = vision.Picture(None, (4, 6))
    ids, spans = vision.expand([1, 99, 2], 99, [q])
    assert ids == [1] + [99] * 6 + [2]
    assert vision.expanded_at(2, spans, [q]) == 7


def test_gemma_expand_refuses_a_missing_boi_or_eoi():
    pic = vision.Picture(None, (6, 9), family=vision.GEMMA, merge=9)
    with pytest.raises(ValueError, match="<\\|image>"):
        vision.expand([99], 99, [pic])


def test_gemma_take_images_uses_its_own_mark():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    out, sources = vision.take_images(msgs, vision.GEMMA)
    assert sources == ["data:image/png;base64,AAAA"]
    text = out[0]["content"][1]["text"]
    assert text == "<|image|>" and text != vision.IMAGE_MARK
    from locals import oai

    joined = oai.text_of(out[0]["content"])
    assert joined == "what is this?<|image|>"  # (no newline between an image and its neighbours, as Qwen's)


def test_gemma_prompt_tokens_are_the_checkpoints():
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(MODEL, local_files_only=True, allow_patterns=["*.json", "*.jinja"])
    except Exception:  # noqa: BLE001
        pytest.skip("tokenizer not cached")
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(path)
    assert vision.prompt_tokens(tok, vision.GEMMA) == (258880, 255999, 258882)  # image, boi, eoi
    assert vision.prompt_tokens(tok, vision.QWEN)[1:] == (None, None)


def test_gemma_prepared_prompt_expands_to_the_processor_layout(tmp_path):
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(MODEL, local_files_only=True, allow_patterns=["*.json", "*.jinja"])
    except Exception:  # noqa: BLE001
        pytest.skip("tokenizer not cached")
    from mlx_lm.utils import load_tokenizer

    from locals.prepared import PreparedPrompt

    tok = load_tokenizer(path)
    pic = vision.Picture(None, (6, 9), key=1, family=vision.GEMMA, merge=9)
    p = PreparedPrompt.of("<|image|>what is this?", tok, (pic,))
    assert p.placed == tuple(tok._tokenizer.encode("<|image|>what is this?", add_special_tokens=False))
    assert p.tokens[:4] == [255999, 258880, 258880, 258880]  # boi, then the soft tokens
    assert p.tokens[1 + 6] == 258882  # eoi after the six soft tokens
    assert p.spans == ((1, 7),)
    assert p.tokens[1 + 6 + 1:] == list(p.placed[1:])  # the text follows eoi
    # a checkpoint after the image counts its boi/eoi (a long enough prompt to be worth keeping)
    text = "<|image|>" + "word " * 1200 + "end"
    long = PreparedPrompt.of(text, tok, (pic,))
    prefix = text[:len("<|image|>") + len("word ") * 1200]
    n = len(tok._tokenizer.encode(prefix, add_special_tokens=False))
    at = long.checkpoint_at(prefix)
    assert at is not None and at >= 1024
    # (the last token may merge with what follows: the checkpoint backs off at most one token)
    assert at in (vision.expanded_at(n, long.spans, long.pictures),
                  vision.expanded_at(n - 1, long.spans, long.pictures))


def test_gemma_family_detection():
    from pathlib import Path

    for name, expected in [("gemma-4-e2b-it-5bit", vision.GEMMA), ("Qwen3.5-4B-MLX-4bit", vision.QWEN)]:
        from huggingface_hub import snapshot_download

        try:
            p = snapshot_download("mlx-community/" + name, local_files_only=True, allow_patterns=["config.json"])
        except Exception:  # noqa: BLE001
            continue
        assert vision.family(Path(p)) == expected
    assert vision.family(Path("/nonexistent")) == vision.QWEN


def test_encoder_detects_gemma_and_divides_features_by_the_embed_scale(tmp_path, monkeypatch):
    from pathlib import Path

    from huggingface_hub import snapshot_download

    try:
        p = Path(snapshot_download(MODEL, local_files_only=True, allow_patterns=["config.json"]))
    except Exception:  # noqa: BLE001
        pytest.skip("config not cached")
    src = p / "config.json"
    (tmp_path / "config.json").write_bytes(src.read_bytes())
    enc = vision.Encoder(str(tmp_path), mx.float32)
    assert enc.family == vision.GEMMA
    assert enc.mrope_section is None
    # a tower that only answers its scale: encode divides the features by sqrt(text hidden), as the model scales them
    seen = []

    class Tower:
        feature_scale = 4.0

        def image(self, pixels, grid):
            seen.append(grid)
            return mx.ones((grid[0] * grid[1] // 9, 8)), []

    monkeypatch.setattr(enc, "tower", lambda: Tower())
    pic = vision.Picture(np.zeros((9, 768), np.float32), (3, 3), family=vision.GEMMA, merge=9)
    out, levels = enc.encode([pic])
    assert seen == [(3, 3)] and levels == []
    assert out[0].shape == (1, 8) and np.allclose(np.array(out[0]), 0.25)


def test_the_tower_matches_a_numpy_reference_built_from_transformers_formulas():
    """Patch embedder, the 16 blocks and the pool, against the same weights read through numpy in the order
    transformers' Gemma4VisionModel runs them (norm placements, scale 1 attention over every patch, the axial RoPE,
    the 3x3 pool times sqrt(hidden) in float32). torch is not installed here, so the reference is numpy."""
    from locals.models.gemma4_vl import Encoder, PatchEmbedder, VisionConfig, _pool

    cfg = VisionConfig(hidden_size=48, num_hidden_layers=2, num_attention_heads=4, head_dim=12,
                       intermediate_size=96, position_embedding_size=32, text_hidden_size=40)
    pe, enc = PatchEmbedder(cfg), Encoder(cfg)
    h, w = 3, 6  # 18 patches, two 3x3 pool rows
    rng = np.random.default_rng(7)
    pixels = rng.random((h * w, 3 * 16 * 16)).astype(np.float32) * 0.5 + 0.25
    positions = mx.stack([mx.tile(mx.arange(w, dtype=mx.int32), h),
                          mx.repeat(mx.arange(h, dtype=mx.int32), w)], axis=-1)
    rope = enc.rope(h, w, mx.float32)
    got = np.array(_pool(enc(pe(mx.array(pixels), positions), rope), h, w, 3, cfg.hidden_size, mx.float32))

    def arr(m):
        return np.array(m, dtype=np.float32)

    def rms(x, weight=None, eps=cfg.rms_norm_eps):
        y = x.astype(np.float32)
        y = y / np.sqrt((y**2).mean(-1, keepdims=True) + eps)
        return (y * weight) if weight is not None else y

    def rotate(part, cos, sin):  # rotate_half within a part; its cos/sin are the axis's frequencies, twice
        q = part.shape[-1] // 2
        a, b = part[..., :q], part[..., q:]
        c = cos[:, None, :]
        s = sin[:, None, :]
        return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)

    # the patch embedder: 2 * (pixel - 0.5), one linear, the learned (x, y) position embeddings added
    x = 2 * (pixels - 0.5)
    x = x @ arr(pe.input_proj.weight).T
    table = arr(pe.position_embedding_table)
    xs = np.tile(np.arange(w), h)
    ys = np.repeat(np.arange(h), w)
    x = x + table[0][xs] + table[1][ys]

    cos_x, sin_x, cos_y, sin_y = (arr(t) for t in rope)
    for layer in enc.layers:
        attn = layer.self_attn
        r = x
        y = rms(x, arr(layer.input_layernorm.weight))
        q = rms((y @ arr(attn.q_proj.linear.weight).T).reshape(h * w, 4, 12), arr(attn.q_norm.weight))
        k = rms((y @ arr(attn.k_proj.linear.weight).T).reshape(h * w, 4, 12), arr(attn.k_norm.weight))
        v = rms((y @ arr(attn.v_proj.linear.weight).T).reshape(h * w, 4, 12))  # (no scale)
        q = rotate(q[..., :6], cos_x, sin_x), rotate(q[..., 6:], cos_y, sin_y)
        k = rotate(k[..., :6], cos_x, sin_x), rotate(k[..., 6:], cos_y, sin_y)
        q, k = np.concatenate(q, axis=-1), np.concatenate(k, axis=-1)
        scores = np.einsum("thd,shd->hts", q, k) * 1.0  # (the reference's scaling: none)
        scores = np.exp(scores - scores.max(-1, keepdims=True))
        scores /= scores.sum(-1, keepdims=True)
        out = np.einsum("hts,shd->thd", scores, v).reshape(h * w, 48)
        x = r + rms(out @ arr(attn.o_proj.linear.weight).T, arr(layer.post_attention_layernorm.weight))
        r = x
        y = rms(x, arr(layer.pre_feedforward_layernorm.weight))
        gate = y @ arr(layer.mlp.gate_proj.linear.weight).T
        up = y @ arr(layer.mlp.up_proj.linear.weight).T
        act = 0.5 * gate * (1 + np.tanh(np.sqrt(2 / np.pi) * (gate + 0.044715 * gate**3)))  # gelu_pytorch_tanh
        x = r + rms((act * up) @ arr(layer.mlp.down_proj.linear.weight).T,
                    arr(layer.post_feedforward_layernorm.weight))
    want = x.reshape(h // 3, 3, w // 3, 3, -1).mean(axis=(1, 3)) * np.sqrt(cfg.hidden_size)
    assert got.shape == (2, 48)
    assert np.allclose(got, want, atol=2e-4), np.abs(got - want).max()


def test_per_layer_chunk_uses_the_pad_token_at_an_image():
    class Body:
        class config:
            pad_token_id = 0
            num_hidden_layers = 2
            hidden_size_per_layer_input = 3

        embed_tokens_per_layer_scale = 2.0

        def __init__(self):
            self.table = mx.arange(10 * 6, dtype=mx.float32).reshape(10, 6)
            self.seen = None

        def embed_tokens_per_layer(self, ids):
            self.seen = np.array(ids)
            return self.table[ids]

    body = Body()
    ids = [4, 5, 6, 7, 8]
    x = vision.per_layer_chunk(body, ids, [(1, 4)], 0, 5)
    assert body.seen.tolist() == [[4, 0, 0, 0, 8]]  # the image's three positions take the pad token
    assert x.shape == (1, 5, 2, 3)
    # the scale is applied before the unflatten (2.0 * the token's row)
    assert np.allclose(np.array(x[0, 0]), 2.0 * np.arange(24, 30).reshape(2, 3))
    assert np.allclose(np.array(x[0, 4]), 2.0 * np.arange(48, 54).reshape(2, 3))


def test_the_axial_rope_rotates_each_axis_half_within_itself():
    # transformers' apply_multidimensional_rope: the head splits into one 32-channel part per axis (x then y), and
    # each part is rotated within itself by rotate_half (channel i pairs with i+16), its cos/sin the axis's
    # frequencies repeated twice (H-H-W-W)
    x = np.zeros((1, 1, 12))
    for i in range(12):
        x[0, 0, i] = i + 1
    cos_x = np.full((1, 3), 0.5)
    sin_x = np.full((1, 3), 0.25)
    cos_y = np.full((1, 3), -0.5)
    sin_y = np.full((1, 3), 0.75)
    out = np.array(vision_rotate(mx.array(x), mx.array(cos_x), mx.array(sin_x), mx.array(cos_y), mx.array(sin_y)))

    def rotate(v, c, s):  # rotate_half within one part: (-v2, v1) * sin + v * cos
        h = len(v) // 2
        v1, v2 = v[:h], v[h:]
        return np.concatenate([v1 * c - v2 * s, v2 * c + v1 * s])

    want = np.concatenate([rotate(x[0, 0, :6], cos_x[0], sin_x[0]), rotate(x[0, 0, 6:], cos_y[0], sin_y[0])])
    assert np.allclose(out[0, 0], want)


def vision_rotate(*a):
    from locals.models.gemma4_vl import _rotate

    return _rotate(*a)


def test_the_pool_averages_3x3_blocks_and_scales():
    from locals.models.gemma4_vl import _pool

    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((6 * 3, 5)).astype(np.float32))  # 6 rows, 3 columns of patches
    out = np.array(_pool(x, 6, 3, 3, 5, mx.float32))
    assert out.shape == (2 * 1, 5)  # (6/3) * (3/3) blocks
    want = np.array(x).reshape(2, 3, 1, 3, 5).mean(axis=(1, 3)) * math.sqrt(5)
    assert np.allclose(out, want.reshape(2, 5))


# -- the engine's side: PLE with an image, and no M-RoPE -------------------------------------------


class _Cache:
    state = None


class _Body:
    """A Gemma-like text body: records what the engine feeds it."""

    class config:
        pad_token_id = 0
        num_hidden_layers = 2
        hidden_size_per_layer_input = 2

    embed_tokens_per_layer_scale = 1.0

    def __init__(self, per_layer=True):
        self.calls = []
        self.embed_tokens = lambda ids: mx.zeros((1, ids.shape[1], 3))
        self.embed_tokens_per_layer = (lambda ids: mx.zeros((1, ids.shape[1], 4))) if per_layer else None

    def __call__(self, x, cache, input_embeddings=None, per_layer_inputs=None):
        self.calls.append((x, input_embeddings.shape,
                           None if per_layer_inputs is None else per_layer_inputs.shape))


def _engine(body, monkeypatch, prefill_step=2):
    from locals import engine as eng
    from locals import vision

    class Wrapper:
        model = body
        layers = []

    monkeypatch.setattr(eng, "text_model", lambda model: Wrapper())
    monkeypatch.setattr(vision, "mrope", lambda *a, **k: pytest.fail("Gemma has no M-RoPE"))

    e = eng.Engine.__new__(eng.Engine)
    e.model, e.prefill_step = object(), prefill_step
    return e


def test_an_image_prompt_feeds_gemma_its_per_layer_inputs(monkeypatch):
    body = _Body()
    e = _engine(body, monkeypatch)
    feats = mx.zeros((2, 3))
    assert e.prefill_embedded([_Cache()], [1, 2, 3, 4, 5], [(1, 3)], [feats], None) is True
    assert [(c[0], c[1], c[2]) for c in body.calls] == [
        (None, (1, 2, 3), (1, 2, 2, 2)),  # the soft tokens of the image at 1..3 take the pad id, not an id of their own
        (None, (1, 2, 3), (1, 2, 2, 2)),
    ]


def test_a_text_model_without_ple_gets_no_per_layer_inputs(monkeypatch):
    body = _Body(per_layer=False)
    e = _engine(body, monkeypatch)
    assert e.prefill_embedded([_Cache()], [1, 2, 3, 4, 5], (), [], None) is True
    assert [c[2] for c in body.calls] == [None, None]


def test_generate_embedded_without_positions_skips_mrope(monkeypatch):
    from locals import engine as eng

    e = _engine(_Body(), monkeypatch)
    seen = {}

    def generate(prompt, ctl, *a, prefill=None, **k):
        seen["prefill"] = prefill
        return "done"

    monkeypatch.setattr(e, "_generate_sync", generate)
    assert e.generate_embedded([1, 2], [], [], None, max_tokens=1) == "done"
    assert callable(seen["prefill"])

