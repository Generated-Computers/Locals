"""Images, for the models that read them.

A chat request with an image reaches this module twice. In the server's ``prepare``
(the HTTP thread) the image is decoded, resized and normalized into patch vectors, and its
``<|image_pad|>`` placeholder in the prompt is expanded to the number of tokens
the model will see. In the engine's thread, the vision tower turns those patches
into embeddings, the text model is prefilled with them, and generation runs with
M-RoPE positions: every image token carries its own (t, h, w), so the text model
knows where in the picture each token came from.

Qwen3.5, Qwen3-VL (models/qwen3_5_vl.py) and Gemma 4 (models/gemma4_vl.py) are
wired up; other models keep the "this model reads text only" note in place of
the picture.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import mlx.core as mx

from .defaults import cached_snapshot
from .models.qwen3_5_vl import rotate_half
from .oai import GEMMA_IMAGE_MARK, IMAGE_MARK as QWEN_IMAGE_MARK  # (what the chat template writes for one image)
from .oai import IMAGE_MARKS, IMAGE_PARTS

QWEN, GEMMA = "qwen", "gemma"  # the two vision towers this server knows
IMAGE_MARK = QWEN_IMAGE_MARK  # (the Qwen mark, under the name callers knew)
# The text model's special tokens, from its config.json (Qwen3.5-4B / Gemma 4)
IMAGE_TOKEN = "<|image_pad|>"
# per family: the chat template's one placeholder, and the tokens that wrap its soft tokens (Gemma's
# processor writes <|image> <|image|>*N <image|>; only the N in the middle take the tower's features)
MARKS = {QWEN: QWEN_IMAGE_MARK, GEMMA: GEMMA_IMAGE_MARK}
PLACEHOLDERS = {QWEN: (IMAGE_TOKEN, None, None), GEMMA: (GEMMA_IMAGE_MARK, "<|image>", "<image|>")}
# Pixels an image is resized to at most (LOCALS_VISION_MAX_PIXELS): one text token per 32x32 pixels, and the
# tower attends over all of an image's 16x16 patches at once. The checkpoint allows 16.7M pixels (16,384 tokens,
# 65,536 patches): a 2880x1800 screenshot was 5,000 tokens. 1M pixels is 1,024 tokens.
MAX_PIXELS = 1 << 20
# Gemma 4's processor: 16-pixel patches pooled 3x3, at most 280 soft tokens, both sides a multiple of 48.
GEMMA_PATCH, GEMMA_POOL, GEMMA_MAX_SOFT_TOKENS = 16, 3, 280


@dataclass
class Picture:
    """One image, preprocessed: ``patches`` (patches, 16*16*3) pixels (numpy, float32; None when only counted),
    the (h, w) patch grid, and ``key``, the image's id in prompt-cache keys (``cache_key``: None, not cached).
    ``family`` says whose tower reads it and ``merge`` how many patches make one text token (Qwen: 2 frames of
    16x16 pixels a patch, merged 2x2; Gemma: one frame, pooled 3x3)."""

    patches: object
    grid: Tuple[int, int]
    key: Optional[int] = None
    family: str = QWEN
    merge: int = 4

    @property
    def tokens(self) -> int:
        return self.grid[0] * self.grid[1] // self.merge

    @property
    def wrap(self) -> int:
        """Tokens the expansion adds around the soft ones (Gemma's boi/eoi)."""
        return self.lead + self.tail

    @property
    def lead(self) -> int:
        """Tokens the expansion adds before the soft ones (Gemma's boi)."""
        return 1 if self.family == GEMMA else 0

    @property
    def tail(self) -> int:
        """Tokens the expansion adds after the soft ones (Gemma's eoi)."""
        return 1 if self.family == GEMMA else 0


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


def family(model_dir) -> str:
    """Which vision tower a checkpoint's config names: Gemma 4's, or else Qwen3.5/3-VL's (the default)."""
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
    except (OSError, ValueError, TypeError):
        return QWEN
    return GEMMA if isinstance(cfg, dict) and cfg.get("model_type") in ("gemma4",) else QWEN


def gemma_resize(height: int, width: int, patch_size: int = GEMMA_PATCH,
                 max_patches: int = GEMMA_MAX_SOFT_TOKENS * GEMMA_POOL**2,
                 pooling_kernel_size: int = GEMMA_POOL) -> Tuple[int, int]:
    """The size Gemma 4's processor resizes to: the aspect ratio kept, at most ``max_patches`` patches, both
    sides a multiple of ``pooling_kernel_size * patch_size`` (transformers' ``get_aspect_ratio_preserving_size``,
    which rounds *down*: a small image is enlarged to almost fill the patch budget)."""
    import math

    if height < 1 or width < 1:
        raise ValueError("the image has no pixels")
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side = pooling_kernel_size * patch_size
    target_height = int(math.floor(ideal_height / side)) * side
    target_width = int(math.floor(ideal_width / side)) * side
    if target_height == 0 and target_width == 0:
        raise ValueError(f"attempting to resize to a 0 x 0 image: both sides of {height}x{width} are under {side}")
    max_side = (max_patches // pooling_kernel_size**2) * side
    if target_height == 0:
        target_height = side
        target_width = min(int(math.floor(width / height)) * side, max_side)
    elif target_width == 0:
        target_width = side
        target_height = min(int(math.floor(height / width)) * side, max_side)
    return target_height, target_width


def _resized(h: int, w: int, family: str = QWEN) -> Tuple[int, int]:
    if family == GEMMA:
        return gemma_resize(h, w)
    cap = int(os.environ.get("LOCALS_VISION_MAX_PIXELS") or MAX_PIXELS)
    return smart_resize(h, w, 32, 65536, cap)  # 32: patch_size * spatial_merge_size


def _key(raw: bytes, grid: Tuple[int, int]) -> int:
    import hashlib

    h = hashlib.blake2b(raw, digest_size=7)
    h.update(repr(grid).encode())  # (the same bytes resized to another grid are other features)
    return int.from_bytes(h.digest(), "big") + 1


def measure(raw: bytes, family: str = QWEN) -> Picture:
    """The image's grid from its header alone, no patches: for a token count (PIL reads no pixels for ``size``)."""
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as im:
        w, h = im.size
    hb, wb = _resized(h, w, family)
    patch, merge = (GEMMA_PATCH, GEMMA_POOL**2) if family == GEMMA else (16, 4)
    grid = (hb // patch, wb // patch)
    return Picture(None, grid, _key(raw, grid), family, merge)


def preprocess(raw: bytes, family: str = QWEN) -> Picture:
    """Image bytes -> the model's patch vectors. Qwen: 16x16 patches over two identical frames (an image is a
    one-frame video), in spatial-merge order. Gemma: 16x16 patches in raster order, channels last inside a patch,
    rescaled to [0, 1] (its patch embedder applies ``2 * (pixel - 0.5)`` itself)."""
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        w, h = im.size
        hb, wb = _resized(h, w, family)
        if (hb, wb) != (h, w):
            im = im.resize((wb, hb), resample=Image.BICUBIC)

    # numpy, not MLX: this runs in the HTTP thread, and an MLX array made there is computed on that thread's
    # stream, which the engine's thread cannot use ("There is no Stream(gpu, 2) in current thread")
    if family == GEMMA:
        pixels = np.asarray(im, dtype=np.float32) / 255.0  # (h, w, 3)
        hp, wp = hb // GEMMA_PATCH, wb // GEMMA_PATCH
        x = pixels.reshape(hp, GEMMA_PATCH, wp, GEMMA_PATCH, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 16 * 16 * 3)
        return Picture(np.ascontiguousarray(x, dtype=np.float32), (hp, wp), _key(raw, (hp, wp)), GEMMA, GEMMA_POOL**2)

    pixels = np.asarray(im, dtype=np.float32) / 127.5 - 1.0  # (h, w, 3), mean/std 0.5
    hp, wp = hb // 16, wb // 16
    x = pixels.reshape(hp // 2, 2, 16, wp // 2, 2, 16, 3).transpose(0, 3, 1, 4, 2, 5, 6).reshape(-1, 1, 16, 16, 3)
    x = np.ascontiguousarray(np.broadcast_to(x, (x.shape[0], 2, 16, 16, 3)).reshape(x.shape[0], -1))
    return Picture(x, (hp, wp), _key(raw, (hp, wp)))


NO_FILES = "this server reads no files of its own for an image: send the image's bytes (a data URL)"
NO_URLS = "this server fetches no URLs for an image when other machines can reach it: send the image's bytes (a data URL)"


def _b64(s: str, validate: bool) -> bytes:
    try:
        return base64.b64decode(s, validate=validate)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"the image is not valid base64: {e}") from None


def decode(source: str, local: bool = True) -> bytes:
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


def take_images(messages, family: str = QWEN) -> Tuple[List[dict], List[str]]:
    """The messages with each image of a user or tool message as the chat template's mark for one image, and the
    images' sources in prompt order. Images elsewhere (a system message) keep the text-only note, as the templates
    refuse them there."""
    mark = MARKS.get(family, QWEN_IMAGE_MARK)
    out: List[dict] = []
    sources: List[str] = []
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list) or m.get("role") not in ("user", "tool"):
            out.append(m)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in IMAGE_PARTS:
                src = _source(part)
                if not src:
                    raise ValueError("an image part has no image: send image_url {\"url\": \"data:image/png;base64,...\"}")
                sources.append(src)
                parts.append({"type": "text", "text": mark})
            else:
                parts.append(part)
        out.append({**m, "content": parts})
    return out, sources


def load(sources: Sequence[str], local: bool = True, count: bool = False, family: str = QWEN) -> List[Picture]:
    """Each source decoded and preprocessed (with ``count``, only measured) for ``family``'s tower; a bad one is
    the request's error (a ValueError naming it)."""
    out = []
    for i, src in enumerate(sources):
        try:
            raw = decode(src, local=local)
            out.append(measure(raw, family) if count else preprocess(raw, family))
        except Exception as e:  # noqa: BLE001  # PIL's UnidentifiedImageError, a URL that does not answer
            raise ValueError(f"image {i + 1} of {len(sources)}: {e}") from None
    return out


# -- the prompt ----------------------------------------------------------------------------------


def prompt_tokens(tokenizer, family: str = QWEN) -> Tuple[int, Optional[int], Optional[int]]:
    """(the placeholder's id, the id that opens an image's soft tokens, the id that closes them). Qwen's template
    writes its own start/end tags, so only Gemma has the extra pair."""
    token, boi, eoi = PLACEHOLDERS[family]
    to_id = tokenizer.convert_tokens_to_ids
    return to_id(token), (to_id(boi) if boi else None), (to_id(eoi) if eoi else None)


def expand(ids: Sequence[int], image_token_id: int, pictures: Sequence[Picture],
           boi: Optional[int] = None, eoi: Optional[int] = None) -> Tuple[List[int], List[Tuple[int, int]]]:
    """The prompt with each image's placeholder expanded to its token count; and where each image's tokens sit in
    it (the soft ones only: Gemma's ``<|image>`` / ``<image|>`` around them are text tokens of their own)."""
    out: List[int] = []
    spans: List[Tuple[int, int]] = []
    i = 0
    for pic in pictures:
        while i < len(ids) and ids[i] != image_token_id:
            out.append(ids[i])
            i += 1
        if i >= len(ids):
            raise ValueError(f"the prompt has {len(spans)} image placeholder(s) for {len(pictures)} image(s); the "
                             "chat template writes one of its image tokens for each")
        if pic.family == GEMMA:
            if boi is None or eoi is None:
                raise ValueError("a Gemma 4 prompt needs the <|image> and <image|> tokens around its image's tokens")
            out.append(boi)
        start = len(out)
        out.extend([image_token_id] * pic.tokens)
        spans.append((start, len(out)))
        if pic.family == GEMMA:
            out.append(eoi)
        i += 1
    rest = list(ids[i:])
    if image_token_id in rest:
        raise ValueError(f"the prompt has more image placeholders than the {len(pictures)} image(s) sent")
    out.extend(rest)
    return out, spans


def expanded_at(n: int, spans: Sequence[Tuple[int, int]], pictures: Sequence[Picture] = ()) -> int:
    """Where position ``n`` of the prompt before ``expand`` (one placeholder per image) is in the expanded one,
    whose images' tokens sit at ``spans`` (Gemma's boi/eoi count too)."""
    shift = 0
    for k, (start, end) in enumerate(spans):
        wrap = pictures[k].wrap if k < len(pictures) else 0
        lead = pictures[k].lead if k < len(pictures) else 0
        if start - shift - lead < n:  # (the placeholder's own index in the prompt before expand)
            shift += end - start - 1 + wrap
    return n + shift


def cache_key(prompt: Sequence[int], spans: Sequence[Tuple[int, int]], pictures: Sequence[Picture]) -> Optional[List[int]]:
    """The prompt's key in the prompt cache: its tokens, with each image's pad tokens as minus the image's ``key``
    (no token id is negative). Under the pad tokens themselves an entry would be the next image's of the same size
    too. None if an image has no key."""
    if any(p.key is None for p in pictures):
        return None
    out = list(prompt)
    for (start, end), pic in zip(spans, pictures):
        out[start:end] = [-pic.key] * (end - start)
    return out


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



def rope_delta(prompt_positions: mx.array) -> int:
    """Past the prompt, a token's three M-RoPE positions are one number, its index plus this (negative where an
    image's tokens take fewer positions than tokens). M-RoPE with t = h = w is the layer's own rope at that
    position, so the reply decodes as a text row whose rope offset is shifted by this much."""
    return int(mx.max(prompt_positions)) + 1 - prompt_positions.shape[-1]


class ImagePrompt:
    """An image prompt (GLOSSARY.md) as the batch engine reads it: ``prompt`` with each image's tokens at its
    ``spans``. ``key`` is its cache key (None: an image has no id, the prompt is not cached), ``rope_delta`` how far
    its rope offset is shifted as its reply decodes, and ``cancelled()`` whether its client has left. The prompt
    cache keeps the row without that shift (an entry's offset is its length: _only_row, and mlx_lm's
    BatchKVCache.extract, set it so) and a follow-up turn adds its own; a cache whose extract kept the offset
    (BatchRotatingKVCache's) would shift it twice. Made on the engine's thread: its positions are MLX arrays."""

    def __init__(self, prompt: Sequence[int], spans: Sequence[Tuple[int, int]], pictures: Sequence[Picture],
                 encoder, prefill, cancelled=lambda: False):
        self.prompt, self.spans, self.pictures = list(prompt), list(spans), list(pictures)
        self.family = pictures[0].family if pictures else QWEN
        # Gemma 4 has no M-RoPE: its image tokens take ordinary positions, so the reply's rope needs no shift
        self._pos = (None if self.family == GEMMA else
                     positions(self.spans, [p.grid for p in self.pictures], len(self.prompt)))  # (3, prompt)
        self.key = cache_key(self.prompt, self.spans, self.pictures)
        self.rope_delta = 0 if self._pos is None else rope_delta(self._pos)
        self._encoder, self._prefill, self.cancelled = encoder, prefill, cancelled
        self._read_from = {}  # spans, features and deepstack of the images not in the cached prefix

    def read(self, cache, start: int, stop: int) -> Optional[str]:
        """Read ``prompt[start:stop]`` into ``cache``, which holds ``prompt[:start]``: the tower's features go where
        the images' tokens are, every token at its own M-RoPE position. The tower reads the images from the first
        ``start`` on, once. None when read, else how the request ends: one request's fault (the tower's weights, an
        image it cannot read, a prefill out of memory) is ``"error: ..."``, that request's alone; a cancel, asked
        before the tower and between the prefill's chunks, is ``"cancelled"``. The features are kept from one part to
        the next and let go of when the read ends, however it ends (the request outlives its prefill)."""
        if self.cancelled():  # (its client left while it waited: the tower need not read its images)
            read, why = False, "cancelled"
        else:
            try:
                enc = self._encoder()
                if not self._read_from:
                    keep = [k for k, (_, end) in enumerate(self.spans) if end > start]
                    features, deep = enc.encode([self.pictures[k] for k in keep]) if keep else ([], [])
                    self._read_from.update(spans=[self.spans[k] for k in keep], features=features, deep=deep)
                r = self._read_from
                pos = None if self._pos is None else self._pos[:, None, :]
                section = None if self._pos is None else enc.mrope_section
                read = self._prefill(cache, self.prompt, r["spans"], r["features"], pos, r["deep"],
                                     section, cancel=self.cancelled, start=start, stop=stop)
            except Exception as e:  # noqa: BLE001
                from .service import _reason  # (the request errors' wording, the one the client reads for a text job)

                read, why = False, f"error: {_reason(e)}"
            else:
                why = None if read else "cancelled"
        if not read or stop >= len(self.prompt) - 1:  # (the last prompt token is the reply's first decode step's)
            self._read_from.clear()  # the read has ended: the features are not kept while the reply decodes
        return why


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
        out = r * cos + rotate_half(r) * sin
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
        self._family: Optional[str] = None
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

    def _detect(self) -> None:
        if self._config is not None:
            return
        d = self.directory()
        fam = family(d) if d is not None else QWEN
        if fam == GEMMA:
            from .models.gemma4_vl import has_tower, vision_config
        else:
            from .models.qwen3_5_vl import has_tower, vision_config

        self._config = (vision_config(d) if d is not None else None) or False
        self._family = fam
        if self._config and not has_tower(d):
            self._config = False

    def available(self) -> bool:
        """Whether this checkpoint has a vision tower we can read (a config and its weights). A config that names one
        is not enough: a checkpoint without the tower's weights said it read images, and its first image request
        was an error where the next got the text-only note."""
        if self._tried:
            return self._tower is not None
        self._detect()
        return bool(self._config)

    @property
    def family(self) -> str:
        self._detect()
        return self._family or QWEN

    def tower(self):
        if self._tower is not None:
            return self._tower
        if self._tried or not self.available():
            raise ValueError(f"{self.model_id} has no vision tower: it reads text only")
        self._tried = True  # (one try: a checkpoint whose config names a tower but whose files lack it)
        if self.family == GEMMA:
            from .models.gemma4_vl import load_tower
        else:
            from .models.qwen3_5_vl import load_tower

        if self.log:
            self.log(f"loading {self.model_id}'s vision tower ...")
        self._tower = load_tower(self.directory(), self.dtype, self._config)
        if self._tower is None:
            raise ValueError(f"{self.model_id}'s vision weights are not in its files: it reads text only")
        return self._tower

    @property
    def mrope_section(self) -> Optional[Tuple[int, int, int]]:
        """The text model's M-RoPE split, or None for a tower whose text model has no M-RoPE (Gemma 4)."""
        if self.family == GEMMA:
            return None
        return self._config.mrope_section  # (read after encode: the config is loaded)

    def encode(self, pictures: Sequence[Picture]) -> Tuple[List[mx.array], List[List[mx.array]]]:
        """One (tokens, out_hidden) embedding block per image, in the text model's dtype; and for each deepstack
        layer of the tower (Qwen3-VL's three, Qwen3.5's and Gemma 4's none) one such block per image. Gemma's
        features are divided by the text model's embed scale: the model multiplies every input embedding by it,
        and transformers does not scale image features."""
        tower = self.tower()
        scale = getattr(tower, "feature_scale", 1.0)
        out, deep = [], []
        for pic in pictures:
            feats, ds = tower.image(mx.array(pic.patches).astype(self.dtype), pic.grid)
            if scale != 1.0:
                feats = feats / scale
            out.append(feats.astype(self.dtype))
            deep.append([d.astype(self.dtype) for d in ds])
        levels = [[d[k] for d in deep] for k in range(len(deep[0]) if deep else 0)]
        mx.eval(out, levels)
        return out, levels


def overlay(x: mx.array, at: int, spans: Sequence[Tuple[int, int]], features: Sequence[mx.array], add: bool = False) -> mx.array:
    """``x`` (1, n, D), the prompt's tokens ``at`` to ``at + n``: where an image's span overlaps them, its features
    instead (or, with ``add``, added). A chunk is read at a time, so no array the length of the prompt is made."""
    n = x.shape[1]
    parts, i = [], 0
    for (start, end), feats in zip(spans, features):
        s, e = max(start, at), min(end, at + n)
        if s >= e:
            continue
        f = feats[s - start: e - start][None]
        parts += [x[:, i: s - at], x[:, s - at: e - at] + f.astype(x.dtype) if add else f]
        i = e - at
    if not parts:
        return x
    parts.append(x[:, i:])
    return mx.concatenate(parts, axis=1)


class _Added:
    """A text layer that adds its deepstack features to the image tokens of the chunk it runs (``at``: where the
    chunk starts in the prompt)."""

    def __init__(self, layer, spans, features, state: dict):
        self.layer, self.spans, self.features, self.state = layer, spans, features, state

    def __call__(self, x, *a, **k):
        return overlay(self.layer(x, *a, **k), self.state["at"], self.spans, self.features, add=True)

    def __getattr__(self, name):  # (a layer's other attributes, read by the model's own code)
        return getattr(self.layer, name)


@contextmanager
def deepstack(model, spans: Sequence[Tuple[int, int]], levels: Sequence[Sequence[mx.array]]):
    """Qwen3-VL's deepstack, while a prompt is read: after text layer ``i`` its image tokens get the tower's
    ``i``-th deepstack features (``levels[i]``, one block per image, at ``spans``) added, as transformers'
    ``_deepstack_process``. Yields the state whose ``at`` the caller sets to each chunk's start; the layers are
    put back after."""
    from .kernels import text_model

    state = {"at": 0}
    if not levels:  # (Gemma 4's tower has none: the text model's own code is left untouched)
        yield state
        return
    lm = text_model(model)
    body = getattr(lm, "model", None) or lm  # (mlx-lm's Gemma 4 wrapper's language model is the body itself)
    layers = body.layers
    kept = [layers[i] for i in range(len(levels))]
    try:
        for i, level in enumerate(levels):
            layers[i] = _Added(kept[i], spans, level, state)
        yield state
    finally:
        for i, layer in enumerate(kept):
            layers[i] = layer


def embed_chunk(body, ids: Sequence[int], spans: Sequence[Tuple[int, int]], features: Sequence[mx.array],
                at: int, n: int) -> mx.array:
    """The input embeddings of the prompt's tokens ``at`` to ``at + n``: the token embeddings, with the image
    spans they overlap replaced by the tower's features (the text model then reads the picture where the
    placeholders were)."""
    return overlay(body.embed_tokens(mx.array([list(ids[at: at + n])])), at, spans, features)


def per_layer_chunk(body, ids: Sequence[int], spans: Sequence[Tuple[int, int]], at: int, n: int) -> mx.array:
    """Gemma 4's per-layer token inputs (PLE) of the prompt's tokens ``at`` to ``at + n``, shape
    (1, n, layers, hidden_per_layer). At an image's soft tokens the id is the pad token: transformers gives the
    text model ``llm_input_ids``, which hold the pad id where the image's features go (they are input embeddings,
    not a vocabulary token), so the per-layer table is looked up with it there and with the token elsewhere."""
    chunk = list(ids[at: at + n])
    pad = getattr(getattr(body, "config", None), "pad_token_id", 0)
    for start, end in spans:
        for k in range(max(start, at), min(end, at + n)):
            chunk[k - at] = pad
    cfg = body.config
    x = body.embed_tokens_per_layer(mx.array([chunk])) * body.embed_tokens_per_layer_scale
    return x.reshape(1, len(chunk), cfg.num_hidden_layers, cfg.hidden_size_per_layer_input)
