"""Images, for the models that read them.

A chat request with an image reaches this module twice. In the server's ``prepare``
(the HTTP thread) the image is decoded, resized and normalized into patch vectors, and its
``<|image_pad|>`` placeholder in the prompt is expanded to the number of tokens
the model will see. In the engine's thread, the vision tower turns those patches
into embeddings, the text model is prefilled with them, and generation runs with
M-RoPE positions: every image token carries its own (t, h, w), so the text model
knows where in the picture each token came from.

Qwen3.5 and Qwen3-VL are wired up (models/qwen3_5_vl.py); other models
keep the "this model reads text only" note in place of the picture.
"""

from __future__ import annotations

import base64
import io
import os
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import mlx.core as mx

from .defaults import cached_snapshot
from .oai import IMAGE_PARTS

# The text model's special tokens, from its config.json (Qwen3.5-4B)
IMAGE_TOKEN = "<|image_pad|>"
IMAGE_MARK = "<|vision_start|><|image_pad|><|vision_end|>"  # what the chat template writes for one image
# Pixels an image is resized to at most (LOCALS_VISION_MAX_PIXELS): one text token per 32x32 pixels, and the
# tower attends over all of an image's 16x16 patches at once. The checkpoint allows 16.7M pixels (16,384 tokens,
# 65,536 patches): a 2880x1800 screenshot was 5,000 tokens. 1M pixels is 1,024 tokens.
MAX_PIXELS = 1 << 20


@dataclass
class Picture:
    """One image, preprocessed: ``patches`` (patches, 2*16*16*3) pixels (numpy, float32) and the (h, w) patch grid."""

    patches: object
    grid: Tuple[int, int]

    @property
    def tokens(self) -> int:
        return self.grid[0] * self.grid[1] // 4


# -- preprocessing ------------------------------------------------------------------------------


def smart_resize(height: int, width: int, factor: int = 32, min_pixels: int = 65536,
                 max_pixels: int = 16777216) -> Tuple[int, int]:
    """The size the model was trained on: both sides a multiple of ``factor``, the pixel count within
    [min_pixels, max_pixels], the aspect ratio kept (transformers' ``smart_resize``)."""
    import math

    if height < 1 or width < 1:
        raise ValueError("the image has no pixels")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"an image's aspect ratio must be at most 200:1, this one is "
                         f"{max(height, width) / min(height, width):.0f}:1")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess(raw: bytes) -> Picture:
    """Image bytes -> the model's patch vectors. 16x16 patches over two identical frames (an image is a
    one-frame video), in spatial-merge order: block row, block column, in-block row, in-block column."""
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        w, h = im.size
        side = 32  # patch_size * spatial_merge_size
        cap = int(os.environ.get("LOCALS_VISION_MAX_PIXELS") or MAX_PIXELS)
        hb, wb = smart_resize(h, w, side, 65536, cap)
        if (hb, wb) != (h, w):
            im = im.resize((wb, hb), resample=Image.BICUBIC)
        pixels = np.asarray(im, dtype=np.float32) / 127.5 - 1.0  # (h, w, 3), mean/std 0.5

    # numpy, not MLX: this runs in the HTTP thread, and an MLX array made there is computed on that thread's
    # stream, which the engine's thread cannot use ("There is no Stream(gpu, 2) in current thread")
    hp, wp = hb // 16, wb // 16
    x = pixels.reshape(hp // 2, 2, 16, wp // 2, 2, 16, 3).transpose(0, 3, 1, 4, 2, 5, 6).reshape(-1, 1, 16, 16, 3)
    x = np.ascontiguousarray(np.broadcast_to(x, (x.shape[0], 2, 16, 16, 3)).reshape(x.shape[0], -1))
    return Picture(x, (hp, wp))


NO_FILES = "this server reads no files of its own for an image: send the image's bytes (a data URL)"
NO_URLS = "this server fetches no URLs for an image when other machines can reach it: send the image's bytes (a data URL)"


def _b64(s: str, validate: bool) -> bytes:
    try:
        return base64.b64decode(s, validate=validate)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"the image is not valid base64: {e}") from None


def decode(source: str, media_type: Optional[str] = None, local: bool = True) -> bytes:
    """Image bytes from what a client sends: a data URL, a bare base64 string (Ollama), an http(s) URL,
    or a path to a file on this Mac. URLs and files only for a server no other machine can reach (``local``):
    else a client could read this Mac's files, or have the server fetch from its network for it."""
    if not isinstance(source, str) or not source:
        raise ValueError("an image must be a data URL, a base64 string, an http(s) URL or a file path")
    if source.startswith("data:"):
        head, _, payload = source.partition(",")
        if "base64" in head:
            return base64.b64decode(payload, validate=False)
        return urllib.parse.unquote_to_bytes(payload)
    if source.startswith(("http://", "https://")):
        if not local:
            raise ValueError(NO_URLS)
        req = urllib.request.Request(source, headers={"User-Agent": "locals"})
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 (an explicit URL from the client)
            data = r.read((64 << 20) + 1)  # (no more than the limit read into memory)
        if len(data) > 64 << 20:
            raise ValueError("the image is larger than 64 MB")
        return data
    if source.startswith("file://"):
        if not local:
            raise ValueError(NO_FILES)
        return Path(source[7:]).read_bytes()
    if media_type and "/" in media_type and not os.path.exists(source):
        return _b64(source, validate=True)  # a bare base64 payload (Ollama's images list)
    if len(source) > 4096:
        return _b64(source, validate=False)
    if not local:
        raise ValueError(NO_FILES)
    try:
        return Path(source).expanduser().read_bytes()
    except OSError as e:
        raise ValueError(f"the image could not be read: {e}") from None


# -- the request ---------------------------------------------------------------------------------

def _source(part: dict) -> Optional[str]:
    v = part.get("image_url")
    if isinstance(v, dict):
        v = v.get("url")
    if v is None:
        v = part.get("image") or part.get("url")
    return v if isinstance(v, str) else None


def has_images(messages) -> bool:
    return any(isinstance(m, dict) and isinstance(m.get("content"), list)
               and any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS for p in m["content"])
               for m in messages or [])


def take_images(messages) -> Tuple[List[dict], List[str]]:
    """The messages with each image of a user or tool message as the chat template's ``IMAGE_MARK``
    text, and the images' sources in prompt order. Images elsewhere (a system message) keep the
    text-only note, as Qwen's template refuses them there."""
    out: List[dict] = []
    sources: List[str] = []
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else None
        if m.get("role") not in ("user", "tool") or not isinstance(content, list):
            out.append(m)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in IMAGE_PARTS:
                src = _source(part)
                if not src:
                    raise ValueError("an image part has no image: send image_url {\"url\": \"data:image/png;base64,...\"}")
                sources.append(src)
                parts.append({"type": "text", "text": IMAGE_MARK})
            else:
                parts.append(part)
        out.append({**m, "content": parts})
    return out, sources


def load(sources: Sequence[str], local: bool = True) -> List[Picture]:
    """Each source decoded and preprocessed; a bad one is the request's error (a ValueError naming it)."""
    out = []
    for i, src in enumerate(sources):
        try:
            out.append(preprocess(decode(src, local=local)))
        except Exception as e:  # noqa: BLE001  # PIL's UnidentifiedImageError, a URL that does not answer
            raise ValueError(f"image {i + 1} of {len(sources)}: {e}") from None
    return out


# -- the prompt ----------------------------------------------------------------------------------


def expand(ids: Sequence[int], image_token_id: int, pictures: Sequence[Picture]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """The prompt with each image's placeholder expanded to its token count; and where each image's
    tokens sit in it. The chat template writes one ``<|image_pad|>`` per image."""
    out: List[int] = []
    spans: List[Tuple[int, int]] = []
    i = 0
    for pic in pictures:
        while i < len(ids) and ids[i] != image_token_id:
            out.append(ids[i])
            i += 1
        if i >= len(ids):
            raise ValueError(f"the prompt has {len(spans)} image placeholder(s) for {len(pictures)} image(s); the "
                             "chat template writes <|vision_start|><|image_pad|><|vision_end|> for each")
        start = len(out)
        out.extend([image_token_id] * pic.tokens)
        spans.append((start, len(out)))
        i += 1
    rest = list(ids[i:])
    if image_token_id in rest:
        raise ValueError(f"the prompt has more image placeholders than the {len(pictures)} image(s) sent")
    out.extend(rest)
    return out, spans


def positions(spans: Sequence[Tuple[int, int]], grids: Sequence[Tuple[int, int]], total: int) -> mx.array:
    """M-RoPE positions (3, total) for a prompt: text tokens take one position each in all three rows,
    an image's tokens take (t, h, w) of their place in it, and the text after it continues past the
    image's longer side — transformers' ``get_rope_index``."""
    pos: List[List[int]] = [[], [], []]
    at = 0

    def text(n: int) -> None:
        nonlocal at
        if n <= 0:
            return
        for row in pos:
            row.extend(range(at, at + n))
        at += n

    for (start, end), (h, w) in zip(spans, grids):
        text(start - len(pos[0]))  # the text between the previous image and this one
        mh, mw = h // 2, w // 2
        rows = [i // mw for i in range(mh * mw)]  # the merged tokens, raster order (as the merger emits them)
        cols = [i % mw for i in range(mh * mw)]
        pos[0].extend([at] * (end - start))
        pos[1].extend([at + r for r in rows])
        pos[2].extend([at + c for c in cols])
        at += max(h, w) // 2
    text(total - len(pos[0]))
    return mx.array(pos, dtype=mx.int32)


def tail(prompt_positions: mx.array, n: int) -> mx.array:
    """The positions of ``n`` generated tokens after a prompt: one step each, continuing past the
    prompt's highest position (the M-RoPE delta of a multimodal prompt)."""
    last = int(mx.max(prompt_positions)) + 1
    step = mx.arange(last, last + n, dtype=mx.int32)
    return mx.concatenate([prompt_positions, mx.broadcast_to(step[None, :], (3, n))], axis=1)


# -- M-RoPE ----------------------------------------------------------------------------------------


class MRoPE:
    """A drop-in for a full-attention layer's ``nn.RoPE``: the rotation angles come from precomputed
    per-token (t, h, w) positions instead of from the token's index.

    The three axes are interleaved: frequency ``k`` takes its angle from h at 1, 4, 7, ... (``section[1]`` of
    them), from w at 2, 5, 8, ... (``section[2]``), and from t elsewhere. Qwen3.5's ``[11, 11, 10]`` splits the
    32 frequencies of its 64 rotated dims; Qwen3-VL's ``[24, 20, 20]`` the 64 of its whole 128-wide head. The
    angles are then repeated to fill the rotated dims. The slice by ``offset`` is what the layer's own rope
    would do with the KV cache's length."""

    def __init__(self, dims: int, base: float, positions: mx.array, section: Sequence[int] = (11, 11, 10)):
        if sum(section) != dims // 2:
            raise ValueError(f"mrope_section {list(section)} does not split this layer's {dims // 2} frequencies")
        self.dims = dims
        self.positions = positions  # (3, 1, total)
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        sel = [0] * (dims // 2)
        for axis, off in ((1, 1), (2, 2)):
            for i in range(section[axis]):
                sel[off + 3 * i] = axis
        self._sel = [mx.array([1.0 if a == k else 0.0 for a in sel], dtype=mx.float32) for k in range(3)]

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        n = x.shape[-2]
        pos = self.positions[:, :, offset: offset + n].astype(mx.float32)[..., None]  # (3, 1, n, 1)
        ang = pos * self.inv_freq  # (3, 1, n, dims/2)
        freq = ang[0] * self._sel[0] + ang[1] * self._sel[1] + ang[2] * self._sel[2]
        freq = mx.concatenate([freq, freq], axis=-1)  # (1, 1, n, dims)
        cos = mx.cos(freq).astype(x.dtype)
        sin = mx.sin(freq).astype(x.dtype)
        r, rest = x[..., : self.dims], x[..., self.dims:]  # (Qwen3.5 rotates 64 of a head's 256 dims, as nn.RoPE does)
        half = self.dims // 2
        rotated = mx.concatenate([-r[..., half:], r[..., :half]], axis=-1)
        out = r * cos + rotated * sin
        return mx.concatenate([out, rest], axis=-1) if rest.shape[-1] else out


@contextmanager
def mrope(model, positions: mx.array, section: Sequence[int]):
    """Run with M-RoPE in every full-attention layer of ``model``, then put the layers' own rope back.
    The layers are shared with the batch engine, so the swap never outlives this block."""
    from .kernels import text_model

    body = getattr(text_model(model), "model", None)
    swapped = []
    for layer in getattr(body, "layers", []):
        attn = getattr(layer, "self_attn", None)
        rope = getattr(attn, "rope", None)
        if rope is None or not hasattr(rope, "dims"):
            continue  # linear-attention layers rotate nothing
        attn.rope = MRoPE(rope.dims, rope.base, positions, section)
        swapped.append((attn, rope))
    try:
        yield
    finally:
        for attn, rope in swapped:
            attn.rope = rope


# -- the tower ---------------------------------------------------------------------------------------


class Encoder:
    """The vision tower of this model, built and loaded on the first image (its 300 MB stay out of
    memory until something needs them)."""

    def __init__(self, model_id: str, dtype, log=None):
        self.model_id = model_id
        self.dtype = dtype
        self.log = log
        self._tower = None
        self._dir: Optional[Path] = None
        self._config = None
        self._tried = False

    def directory(self) -> Optional[Path]:
        if self._dir is not None:
            return self._dir
        p = Path(self.model_id).expanduser()
        if p.is_dir():
            self._dir = p
            return p
        self._dir = cached_snapshot(self.model_id, ["config.json", "*.safetensors"])
        return self._dir

    def available(self) -> bool:
        """Whether this checkpoint has a vision tower we can read (a config and its weights)."""
        if self._tried:
            return self._tower is not None
        if self._config is None:
            from .models.qwen3_5_vl import vision_config

            d = self.directory()
            self._config = (vision_config(d) if d is not None else None) or False
        return bool(self._config)

    def tower(self):
        if self._tower is not None:
            return self._tower
        if self._tried or not self.available():
            raise ValueError(f"{self.model_id} has no vision tower: it reads text only")
        self._tried = True  # (one try: a checkpoint whose config names a tower but whose files lack it)
        from .models.qwen3_5_vl import load_tower

        if self.log:
            self.log(f"loading {self.model_id}'s vision tower ...")
        self._tower = load_tower(self.directory(), self.dtype, self._config)
        if self._tower is None:
            raise ValueError(f"{self.model_id}'s vision weights are not in its files: it reads text only")
        return self._tower

    @property
    def mrope_section(self) -> Tuple[int, int, int]:
        return self._config.mrope_section  # (read after encode: the config is loaded)

    def encode(self, pictures: Sequence[Picture]) -> Tuple[List[mx.array], List[List[mx.array]]]:
        """One (tokens, out_hidden) embedding block per image, in the text model's dtype; and for each deepstack
        layer of the tower (Qwen3-VL's three, Qwen3.5's none) one such block per image."""
        tower = self.tower()
        out, deep = [], []
        for pic in pictures:
            feats, ds = tower.image(mx.array(pic.patches).astype(self.dtype), pic.grid)
            out.append(feats.astype(self.dtype))
            deep.append([d.astype(self.dtype) for d in ds])
        levels = [[d[k] for d in deep] for k in range(len(deep[0]) if deep else 0)]
        mx.eval(out, levels)
        return out, levels


def place(n: int, spans: Sequence[Tuple[int, int]], features: Sequence[mx.array]) -> mx.array:
    """(1, n, D): each image's features at its span, zeros elsewhere (what a deepstack layer adds)."""
    d, dt = features[0].shape[-1], features[0].dtype
    parts, at = [], 0
    for (start, end), feats in zip(spans, features):
        if start > at:
            parts.append(mx.zeros((start - at, d), dtype=dt))
        parts.append(feats)
        at = end
    if n > at:
        parts.append(mx.zeros((n - at, d), dtype=dt))
    return mx.concatenate(parts, axis=0)[None]


class _Added:
    """A text layer that adds ``add[:, at:at+n]`` to its output, ``at`` being where the chunk it runs starts."""

    def __init__(self, layer, add: mx.array, state: dict):
        self.layer, self.add, self.state = layer, add, state

    def __call__(self, x, *a, **k):
        out = self.layer(x, *a, **k)
        at = self.state["at"]
        return out + self.add[:, at: at + out.shape[1]].astype(out.dtype)

    def __getattr__(self, name):  # (a layer's other attributes, read by the model's own code)
        return getattr(self.layer, name)


@contextmanager
def deepstack(model, adds: Sequence[mx.array]):
    """Qwen3-VL's deepstack, while a prompt is read: after text layer ``i`` its image tokens get the tower's
    ``i``-th deepstack features (``adds[i]``, from ``place``) added, as transformers' ``_deepstack_process``.
    Yields the state whose ``at`` the caller sets to each chunk's start; the layers are put back after."""
    from .kernels import text_model

    state = {"at": 0}
    layers = text_model(model).model.layers
    kept = [layers[i] for i in range(len(adds))]
    try:
        for i, add in enumerate(adds):
            layers[i] = _Added(kept[i], add, state)
        yield state
    finally:
        for i, layer in enumerate(kept):
            layers[i] = layer


def embed_prompt(body, ids: Sequence[int], spans: Sequence[Tuple[int, int]], features: Sequence[mx.array]) -> mx.array:
    """The prompt's input embeddings: the token embeddings, with each image's span replaced by the
    tower's features (the text model then reads the picture where the placeholders were)."""
    emb = body.embed_tokens(mx.array([list(ids)]))
    if not spans:
        return emb
    parts = []
    at = 0
    for (start, end), feats in zip(spans, features):
        parts.append(emb[:, at:start])
        parts.append(feats[None])
        at = end
    parts.append(emb[:, at:])
    return mx.concatenate(parts, axis=1)
