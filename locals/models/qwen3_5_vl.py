"""Qwen3.5-VL's vision tower in MLX.

mlx-lm's ``qwen3_5`` keeps only the text half of a checkpoint: its ``sanitize``
drops every ``vision_tower.*`` weight (and ``Model`` has no ``visual`` module).
That is all a text server needs until an image arrives. This module rebuilds
the tower from the same files, lazily, and turns preprocessed pixels into the
``(tokens, out_hidden)`` features the text model's ``<|image_pad|>`` positions
take.

The arithmetic is transformers' ``Qwen3_5VisionModel``, step for step: 16x16
patches over 2 frames through one convolution, a learned 36x36 position table
resampled bilinearly to the image's own grid, axial 2D rope over half the head
dim, ``depth`` pre-norm blocks that attend within one image, then a 2x2 patch
merger that narrows 1024 to the text model's 2560.

Patches are ordered by spatial-merge block, the order the reference's
``get_vision_interpolation_indices_and_weights`` emits (block row, block
column, in-block row, in-block column): the merger's ``view(-1, 4 * hidden)``
then groups exactly one 2x2 block into one text token.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# Qwen3.5 (and its MoE) and Qwen3-VL share this tower. Qwen3-VL's also merges the output of three of its middle
# blocks (deepstack_visual_indexes [8, 16, 24]), which the text model adds to the image's tokens after its first three
# layers (vision.deepstack); its M-RoPE splits 64 frequencies [24, 20, 20], Qwen3.5's 32 [11, 11, 10].
# (qwen3_vl_moe is the same arithmetic, untested: not listed.)
VISION_MODELS = ("qwen3_5", "qwen3_5_moe", "qwen3_vl")


@dataclass
class VisionConfig:
    """The checkpoint's ``vision_config``, with the defaults Qwen3.5 leaves out."""

    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2560
    rope_theta: float = 10000.0
    eps: float = 1e-6  # LayerNorm's, in the tower and in the merger (not nn.LayerNorm's 1e-5)
    deepstack_visual_indexes: Tuple[int, ...] = ()
    mrope_section: Tuple[int, int, int] = (11, 11, 10)  # the text model's, from its rope parameters

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge_size**2

    @property
    def grid_per_side(self) -> int:
        return int(self.num_position_embeddings**0.5)

    @classmethod
    def from_dict(cls, d: dict) -> "VisionConfig":
        names = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in names}
        if isinstance(kw.get("deepstack_visual_indexes"), list):
            kw["deepstack_visual_indexes"] = tuple(kw["deepstack_visual_indexes"])
        rope = d.get("rope_parameters") or {}
        if isinstance(rope, dict) and rope.get("rope_theta"):
            kw["rope_theta"] = float(rope["rope_theta"])
        return cls(**kw)


def vision_config(model_dir: Path) -> Optional[VisionConfig]:
    """The tower's config from a checkpoint's ``config.json``, or None for a text-only model."""
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict) or cfg.get("model_type") not in VISION_MODELS:
        return None
    v = cfg.get("vision_config")
    if not isinstance(v, dict) or not v.get("hidden_size"):
        return None
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    section = rope.get("mrope_section") if isinstance(rope, dict) else None
    if not (isinstance(section, list) and len(section) == 3 and all(isinstance(n, int) and n > 0 for n in section)):
        return None  # (no M-RoPE: the positions this code gives image tokens would mean nothing to the model)
    if isinstance(rope, dict) and rope.get("mrope_interleaved") is False:
        return None  # Qwen2.5-VL's split in blocks, not built here
    head = text.get("head_dim") or (text.get("hidden_size", 0) // max(1, text.get("num_attention_heads", 1)))
    partial = rope.get("partial_rotary_factor") or text.get("partial_rotary_factor") or 1.0
    if head and sum(section) * 2 != int(head * partial):
        return None  # (a split that does not cover the rotated dims: vision.MRoPE would refuse it on the first image)
    out = VisionConfig.from_dict(v)
    if (out.patch_size, out.temporal_patch_size, out.spatial_merge_size) != (16, 2, 2):
        return None  # (vision.preprocess cuts 16x16 patches over two frames, merged 2x2: other values, other tokens)
    out.mrope_section = tuple(section)
    return out


def rotate_half(x: mx.array) -> mx.array:
    """RoPE's rotation: the second half of the last axis, negated, then the first."""
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _bilinear_pos(table: mx.array, side: int, h: int, w: int) -> mx.array:
    """``table`` (side*side, D) resampled to (h, w, D), as ``F.interpolate(mode="bilinear", align_corners=True)``.

    The reference expresses this as per-patch gather indices and weights; with the
    table this small (36x36) gathering is what it amounts to."""
    ys = mx.linspace(0, side - 1, h)  # align_corners=True: endpoints map to 0 and side-1
    xs = mx.linspace(0, side - 1, w)
    y0 = mx.floor(ys).astype(mx.int32)
    x0 = mx.floor(xs).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, side - 1)
    x1 = mx.minimum(x0 + 1, side - 1)
    dy = (ys - y0.astype(mx.float32))[:, None]
    dx = (xs - x0.astype(mx.float32))[None, :]
    base = y0[:, None] * side + x0[None, :]
    w00 = (1 - dy) * (1 - dx)
    w01 = (1 - dy) * dx
    w10 = dy * (1 - dx)
    w11 = dy * dx
    flat = table.reshape(-1, table.shape[-1])
    out = (flat[base.reshape(-1)] * w00.reshape(-1, 1) + flat[(base + (x1 - x0)[None, :]).reshape(-1)] * w01.reshape(-1, 1)
           + flat[(base + (y1 - y0)[:, None] * side).reshape(-1)] * w10.reshape(-1, 1)
           + flat[(base + (y1 - y0)[:, None] * side + (x1 - x0)[None, :]).reshape(-1)] * w11.reshape(-1, 1))
    return out.reshape(h, w, -1)


class PatchEmbed(nn.Module):
    """One convolution over (2 frames, 16, 16, 3) patch vectors (MLX's NDHWC layout)."""

    def __init__(self, cfg: VisionConfig):
        self.shape = (cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size, cfg.in_channels)
        self.proj = nn.Conv3d(cfg.in_channels, cfg.hidden_size,
                              kernel_size=(cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size),
                              stride=(cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size), bias=True)

    def __call__(self, patches: mx.array) -> mx.array:
        return self.proj(patches.reshape(-1, *self.shape)).reshape(patches.shape[0], -1)


class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig):
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim**-0.5
        self.qkv = nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=True)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        s = x.shape[0]
        q, k, v = (t.reshape(1, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
                   for t in mx.split(self.qkv(x), 3, axis=-1))
        q = q.astype(mx.float32)
        k = k.astype(mx.float32)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        out = mx.fast.scaled_dot_product_attention(q.astype(x.dtype), k.astype(x.dtype), v, scale=self.scale)
        return self.proj(out.transpose(0, 2, 1, 3).reshape(s, -1))


class VisionMLP(nn.Module):
    def __init__(self, cfg: VisionConfig):
        self.linear_fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, cfg: VisionConfig):
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.eps)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.eps)
        self.attn = VisionAttention(cfg)
        self.mlp = VisionMLP(cfg)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    """2x2 patches (4 x hidden) -> one text token (out_hidden). A deepstack merger (``postshuffle``) norms the four
    patches together, after they are joined; the final merger norms each patch before."""

    def __init__(self, cfg: VisionConfig, postshuffle: bool = False):
        self.hidden_size = cfg.hidden_size * cfg.merge_unit
        self.postshuffle = postshuffle
        self.norm = nn.LayerNorm(self.hidden_size if postshuffle else cfg.hidden_size, eps=cfg.eps)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, cfg.out_hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x.reshape(-1, self.hidden_size)) if self.postshuffle else self.norm(x).reshape(-1, self.hidden_size)
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class VisionTower(nn.Module):
    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)
        self.blocks = [VisionBlock(cfg) for _ in range(cfg.depth)]
        self.merger = PatchMerger(cfg)
        if cfg.deepstack_visual_indexes:
            self.deepstack_merger_list = [PatchMerger(cfg, postshuffle=True) for _ in cfg.deepstack_visual_indexes]

    # -- positions ---------------------------------------------------------------------------
    def _rope(self, h: int, w: int, dtype) -> Tuple[mx.array, mx.array]:
        """Axial (h, w) rope: head_dim//4 frequencies per axis, each axis over half the head."""
        side_dim = self.cfg.head_dim // 2
        inv = 1.0 / (self.cfg.rope_theta ** (mx.arange(0, side_dim, 2, dtype=mx.float32) / side_dim))
        m = self.cfg.spatial_merge_size
        # block-major order, as the patches are: (block_row, block_col, in_row, in_col)
        hp = mx.broadcast_to(mx.arange(h).reshape(h // m, m)[:, None, :, None], (h // m, w // m, m, m)).reshape(-1)
        wp = mx.broadcast_to(mx.arange(w).reshape(w // m, m)[None, :, None, :], (h // m, w // m, m, m)).reshape(-1)
        fh = hp.astype(mx.float32)[:, None] * inv
        fw = wp.astype(mx.float32)[:, None] * inv
        freq = mx.concatenate([fh, fw], axis=-1)
        freq = mx.concatenate([freq, freq], axis=-1)[None, None]  # (1, 1, tokens, head_dim)
        return mx.cos(freq).astype(dtype), mx.sin(freq).astype(dtype)

    # -- one image ---------------------------------------------------------------------------
    def image(self, patches: mx.array, grid: Tuple[int, int]) -> Tuple[mx.array, List[mx.array]]:
        """``patches``: (tokens, 2*16*16*3) pixels; ``grid``: (h, w) in 16-pixel patches. The image's
        (tokens, out_hidden) features, and one more such block per deepstack layer (none for Qwen3.5)."""
        h, w = grid
        x = self.patch_embed(patches)
        pos = _bilinear_pos(self.pos_embed.weight, self.cfg.grid_per_side, h, w)
        m = self.cfg.spatial_merge_size
        pos = pos.reshape(h // m, m, w // m, m, -1).transpose(0, 2, 1, 3, 4).reshape(h * w, -1)
        x = x + pos.astype(x.dtype)
        cos, sin = self._rope(h, w, x.dtype)
        deep = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin)
            if i in self.cfg.deepstack_visual_indexes:
                deep.append(self.deepstack_merger_list[self.cfg.deepstack_visual_indexes.index(i)](x))
        return self.merger(x), deep


_TOWER = ("vision_tower.", "model.visual.")  # the tower's tensors: mlx-vlm's names, transformers'


def has_tower(path: Path) -> bool:
    """Whether a checkpoint's files hold tower tensors, read from their safetensors headers (no tensor loaded). The
    files, not the index: an index can name a shard that is not downloaded, and _weights reads the files."""
    for f in sorted(Path(path).glob("*.safetensors")):
        try:
            with open(f, "rb") as fh:
                names = json.loads(fh.read(int.from_bytes(fh.read(8), "little")))
        except (OSError, ValueError):
            continue
        if isinstance(names, dict) and any(k.startswith(_TOWER) for k in names):
            return True
    return False


def _weights(path: Path) -> List[Tuple[str, mx.array]]:
    """Every tower tensor of a checkpoint, named as this module's parameters are."""
    out: List[Tuple[str, mx.array]] = []
    for f in sorted(Path(path).glob("*.safetensors")):
        for k, v in mx.load(str(f)).items():
            for prefix in _TOWER:
                if k.startswith(prefix):
                    out.append((k[len(prefix):], v))
                    break
    return out


def load_tower(model_dir: Path, dtype, cfg: VisionConfig) -> Optional[VisionTower]:
    """The tower for a checkpoint (``cfg``: its vision_config), its weights cast to the text model's dtype (None: the
    files hold no vision weights)."""
    weights = _weights(model_dir)
    if not weights:
        return None
    tower = VisionTower(cfg)
    # transformers' layout (out, channels, frames, h, w) as MLX's (out, frames, h, w, channels); an mlx-vlm conversion
    # (mlx-community's) is in MLX's already
    weights = [(k, v.transpose(0, 2, 3, 4, 1) if k == "patch_embed.proj.weight" and v.ndim == 5
                and v.shape[1] == cfg.in_channels and v.shape[-1] != cfg.in_channels else v) for k, v in weights]
    if dtype in (mx.float16, mx.bfloat16):
        weights = [(k, v.astype(dtype) if v.dtype in (mx.float16, mx.bfloat16, mx.float32) else v) for k, v in weights]
    tower.load_weights(weights)
    mx.eval(tower.parameters())
    return tower
