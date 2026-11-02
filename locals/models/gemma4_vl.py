"""Gemma 4's vision tower in MLX.

mlx-lm's ``gemma4`` keeps only the text half of a checkpoint: its ``sanitize`` drops every
``vision_tower.*`` weight (and ``Model`` has no ``vision_tower`` module). That is all a text
server needs until an image arrives. This module rebuilds the tower from the same files,
lazily, and turns preprocessed pixels into the ``(tokens, text_width)`` features the text
model's ``<|image|>`` positions take.

The arithmetic is transformers' ``Gemma4VisionModel``, step for step: 16x16 patches through
one linear, a learned 2-D position table looked up by each patch's (x, y), 16 bidirectional
pre-norm RMSNorm blocks whose attention rotates an axial (H-H-W-W) RoPE over the whole head
with scale 1, a 3x3 average pool times ``sqrt(hidden_size)`` (in float32), then the
``embed_vision`` RMSNorm (no scale) and its quantized projection to the text model's width.

Two details are easy to get wrong and are named here:

* The pool is over the *real* patches only. The reference pads every image to the processor's
  2520 patches and pools that grid, then drops the padded outputs; for one image with no
  padding the same features are the 3x3 means of its own grid, in raster order.
* The text model multiplies every input embedding by ``sqrt(text hidden)`` (``embed_scale``),
  and transformers does *not* scale image features. ``VisionTower.feature_scale`` says by how
  much ``Encoder.encode`` divides them, so what reaches the text model is unscaled.

No normalized pixels: the image processor rescales to [0, 1] and the patch embedder applies
``2 * (pixel - 0.5)``, which is what ``vision.preprocess`` leaves for it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# The checkpoint's model_type. E2B and E4B share this tower (their text widths differ).
VISION_MODELS = ("gemma4",)

# Checkpoint name -> our parameter name: vision_tower.* keeps its suffix, embed_vision.* its prefix.
_TOWER = "vision_tower."
_EMBED = "embed_vision."
_CLAMPS = ("input_max", "input_min", "output_max", "output_min")  # (a tower linear's bounds: ClippedLinear)


@dataclass
class VisionConfig:
    """The checkpoint's ``vision_config`` plus the text-side numbers this module needs."""

    hidden_size: int = 768
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: int = 3072
    patch_size: int = 16
    pooling_kernel_size: int = 3
    position_embedding_size: int = 10240
    rms_norm_eps: float = 1e-6
    rope_theta: float = 100.0
    hidden_activation: str = "gelu_pytorch_tanh"
    standardize: bool = False
    use_clipped_linears: bool = True
    # the embed_vision projection's layout, as the checkpoint's tensors have it (load_tower fills this in)
    embed_quantized: bool = False
    # text side
    text_hidden_size: int = 1536
    pad_token_id: int = 0
    # the quantized embed_vision projection (the tower's own linears are plain bf16)
    quant_bits: int = 5
    quant_group_size: int = 64
    quant_mode: str = "affine"

    @property
    def dim(self) -> int:
        return self.head_dim or self.hidden_size // self.num_attention_heads

    @property
    def kv_heads(self) -> int:
        return self.num_key_value_heads or self.num_attention_heads

    @property
    def merge_unit(self) -> int:
        return self.pooling_kernel_size**2

    @property
    def feature_scale(self) -> float:
        """The text model's ``embed_scale``: image features are divided by it before they are overlaid."""
        return self.text_hidden_size**0.5

    @classmethod
    def from_dict(cls, vision: dict, full: dict) -> "VisionConfig":
        names = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in vision.items() if k in names}
        rope = vision.get("rope_parameters") or {}
        if isinstance(rope, dict) and rope.get("rope_theta"):
            kw["rope_theta"] = float(rope["rope_theta"])
        text = full.get("text_config") if isinstance(full.get("text_config"), dict) else {}
        if text.get("hidden_size"):
            kw["text_hidden_size"] = int(text["hidden_size"])
        pad = text.get("pad_token_id", full.get("pad_token_id"))
        if isinstance(pad, int):
            kw["pad_token_id"] = pad
        quant = full.get("quantization") or full.get("quantization_config") or {}
        if isinstance(quant, dict):
            if isinstance(quant.get("bits"), int):
                kw["quant_bits"] = quant["bits"]
            if isinstance(quant.get("group_size"), int):
                kw["quant_group_size"] = quant["group_size"]
            if isinstance(quant.get("mode"), str):
                kw["quant_mode"] = quant["mode"]
        return cls(**kw)


def vision_config(model_dir: Path) -> Optional[VisionConfig]:
    """The tower's config from a checkpoint's ``config.json``, or None for a text-only (or other) model."""
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(cfg, dict) or cfg.get("model_type") not in VISION_MODELS:
        return None
    v = cfg.get("vision_config")
    if not isinstance(v, dict) or not v.get("hidden_size"):
        return None
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    if not text.get("hidden_size"):
        return None  # (the projection's output width is the text model's)
    out = VisionConfig.from_dict(v, cfg)
    if (out.patch_size, out.pooling_kernel_size) != (16, 3):
        return None  # (vision.preprocess cuts 16x16 patches pooled 3x3: other values, other tokens)
    if out.dim % 4:
        return None  # (axial RoPE splits the head into two halves of two quarters)
    return out


# -- the tower ---------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """``Gemma4RMSNorm``: mean square over the last axis, in float32, optional learned scale."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight if self.with_scale else None, self.eps)


class ClippedLinear(nn.Module):
    """``Gemma4ClippableLinear``: the checkpoint bounds each tower linear's input and output
    (``use_clipped_linears``: ``input_min`` / ``input_max`` / ``output_min`` / ``output_max``, finite in these
    files: layer 0's gate projection holds [-2.45, 2.44], layer 15's o_projection [-2.27, 2.25]), and transformers
    clamps with them, so an activation past one is a feature the reference never produces. A checkpoint that says
    no carries no such tensors, and then this is a plain linear (``linear`` is the module's name in both)."""

    def __init__(self, cfg: VisionConfig, in_dim: int, out_dim: int):
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.clipped = cfg.use_clipped_linears
        if self.clipped:
            inf = float("inf")
            self.input_min = mx.array(-inf, dtype=mx.float32)
            self.input_max = mx.array(inf, dtype=mx.float32)
            self.output_min = mx.array(-inf, dtype=mx.float32)
            self.output_max = mx.array(inf, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        if not self.clipped:
            return self.linear(x)
        x = mx.clip(x, self.input_min.astype(x.dtype), self.input_max.astype(x.dtype))
        y = self.linear(x)
        return mx.clip(y, self.output_min.astype(y.dtype), self.output_max.astype(y.dtype))


class PatchEmbedder(nn.Module):
    """Linear over each 16x16 patch plus the learned (x, y) position embeddings."""

    def __init__(self, cfg: VisionConfig):
        self.patch_size = cfg.patch_size
        self.input_proj = nn.Linear(3 * cfg.patch_size**2, cfg.hidden_size, bias=False)
        self.position_embedding_table = mx.zeros((2, cfg.position_embedding_size, cfg.hidden_size),
                                                 dtype=mx.float32)

    def __call__(self, pixels: mx.array, positions: mx.array) -> mx.array:
        x = 2 * (pixels - 0.5)  # the processor rescales to [0, 1]; the reference scales here
        x = x.astype(self.input_proj.weight.dtype)
        x = self.input_proj(x)
        table = self.position_embedding_table
        return x + table[0][positions[:, 0]] + table[1][positions[:, 1]]


def _rotate(x: mx.array, cos_x, sin_x, cos_y, sin_y) -> mx.array:
    """Axial 2-D RoPE as transformers applies it: the head's first half rotates with x, its second half
    with y, each half in two quarters (``rotate_half``), frequencies duplicated H-H-W-W."""
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    q = half // 2
    a1, a2 = a[..., :q], a[..., q:]
    b1, b2 = b[..., :q], b[..., q:]
    a = mx.concatenate([a1 * cos_x - a2 * sin_x, a2 * cos_x + a1 * sin_x], axis=-1)
    b = mx.concatenate([b1 * cos_y - b2 * sin_y, b2 * cos_y + b1 * sin_y], axis=-1)
    return mx.concatenate([a, b], axis=-1)


class Attention(nn.Module):
    def __init__(self, cfg: VisionConfig):
        self.num_heads = cfg.num_attention_heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.dim
        self.q_proj = ClippedLinear(cfg, cfg.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = ClippedLinear(cfg, cfg.hidden_size, self.kv_heads * self.head_dim)
        self.v_proj = ClippedLinear(cfg, cfg.hidden_size, self.kv_heads * self.head_dim)
        self.o_proj = ClippedLinear(cfg, self.num_heads * self.head_dim, cfg.hidden_size)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps, with_scale=False)

    def __call__(self, x: mx.array, cos_x, sin_x, cos_y, sin_y) -> mx.array:
        s = x.shape[0]
        cos_x, sin_x, cos_y, sin_y = (a[:, None, :] for a in (cos_x, sin_x, cos_y, sin_y))  # (tokens, 1, freqs)
        q = self.q_norm(self.q_proj(x).reshape(s, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(s, self.kv_heads, self.head_dim))
        v = self.v_norm(self.v_proj(x).reshape(s, self.kv_heads, self.head_dim))
        q = _rotate(q, cos_x, sin_x, cos_y, sin_y).transpose(1, 0, 2)[None]
        k = _rotate(k, cos_x, sin_x, cos_y, sin_y).transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]
        if self.kv_heads != self.num_heads:
            rep = self.num_heads // self.kv_heads
            k = mx.repeat(k, rep, axis=1)
            v = mx.repeat(v, rep, axis=1)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)  # (the reference's scaling; no mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(s, -1))


class MLP(nn.Module):
    def __init__(self, cfg: VisionConfig):
        self.gate_proj = ClippedLinear(cfg, cfg.hidden_size, cfg.intermediate_size)
        self.up_proj = ClippedLinear(cfg, cfg.hidden_size, cfg.intermediate_size)
        self.down_proj = ClippedLinear(cfg, cfg.intermediate_size, cfg.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: VisionConfig):
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, x: mx.array, cos_x, sin_x, cos_y, sin_y) -> mx.array:
        residual = x
        x = self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cos_x, sin_x, cos_y, sin_y))
        x = residual + x
        residual = x
        x = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        return residual + x


class Encoder(nn.Module):
    def __init__(self, cfg: VisionConfig):
        self.layers = [EncoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.cfg = cfg

    def rope(self, h: int, w: int, dtype) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Axial (x, y) RoPE: ``head_dim // 4`` frequencies per axis, each over half the head."""
        dim = self.cfg.dim
        spatial = dim // 2
        inv = 1.0 / (self.cfg.rope_theta ** (mx.arange(0, spatial, 2, dtype=mx.float32) / spatial))
        xs = mx.tile(mx.arange(w, dtype=mx.float32), h)[:, None]
        ys = mx.repeat(mx.arange(h, dtype=mx.float32), w)[:, None]
        fx = xs * inv
        fy = ys * inv
        return (mx.cos(fx).astype(dtype), mx.sin(fx).astype(dtype),
                mx.cos(fy).astype(dtype), mx.sin(fy).astype(dtype))

    def __call__(self, x: mx.array, rope) -> mx.array:
        for layer in self.layers:
            x = layer(x, *rope)
        return x


class MultimodalEmbedder(nn.Module):
    """``Gemma4MultimodalEmbedder``: RMSNorm without scale, then the projection to the text model's width. The
    projection is quantized in mlx-community's conversions and plain in an unquantized one (or at other bits than
    the file's default): ``cfg.embed_quantized``, which ``load_tower`` reads from the tensors."""

    def __init__(self, cfg: VisionConfig):
        self.embedding_pre_projection_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, with_scale=False)
        if cfg.embed_quantized:
            self.embedding_projection = nn.QuantizedLinear(cfg.hidden_size, cfg.text_hidden_size, bias=False,
                                                           group_size=cfg.quant_group_size, bits=cfg.quant_bits,
                                                           mode=cfg.quant_mode)
        else:
            self.embedding_projection = nn.Linear(cfg.hidden_size, cfg.text_hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.embedding_projection(self.embedding_pre_projection_norm(x))


def _pool(x: mx.array, h: int, w: int, k: int, hidden: int, dtype) -> mx.array:
    """3x3 average of the patch grid, times ``sqrt(hidden)`` in float32 (as the reference pooler)."""
    y = x.astype(mx.float32).reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))
    y = y.reshape(-1, x.shape[-1]) * (hidden**0.5)
    return y.astype(dtype)


class VisionTower(nn.Module):
    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self.patch_embedder = PatchEmbedder(cfg)
        self.encoder = Encoder(cfg)
        self.embed_vision = MultimodalEmbedder(cfg)
        self.feature_scale = cfg.feature_scale

    def image(self, pixels: mx.array, grid: Tuple[int, int]) -> Tuple[mx.array, List[mx.array]]:
        """``pixels``: (patches, 16*16*3) values in [0, 1]; ``grid``: (h, w) in 16-pixel patches. The image's
        (tokens, text width) features, and no deepstack blocks (the list Gemma 4 does not have)."""
        h, w = grid
        positions = mx.stack([mx.tile(mx.arange(w, dtype=mx.int32), h),
                              mx.repeat(mx.arange(h, dtype=mx.int32), w)], axis=-1)
        x = self.patch_embedder(pixels.astype(mx.float32), positions)
        dtype = x.dtype
        x = self.encoder(x, self.encoder.rope(h, w, dtype))
        x = _pool(x, h, w, self.cfg.pooling_kernel_size, self.cfg.hidden_size, dtype)
        # (the reference standardizes here only if its config says so; these checkpoints do not)
        return self.embed_vision(x), []


# -- reading a checkpoint ----------------------------------------------------------------------


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


_NUMPY = {"F16": "float16", "F32": "float32", "F64": "float64", "I8": "int8", "U8": "uint8", "U16": "uint16",
          "I16": "int16", "U32": "uint32", "I32": "int32", "I64": "int64", "BOOL": "bool"}  # safetensors' names


def _tensor(f, header_len: int, meta: dict) -> mx.array:
    """One tensor, read by itself: its bytes from where the header says they are. BF16 through numpy's uint16
    and a view (numpy has no bfloat16), everything else as its own type."""
    import numpy as np

    begin, end = meta["data_offsets"]
    f.seek(8 + header_len + begin)
    raw = f.read(end - begin)
    shape = tuple(meta["shape"])
    if meta["dtype"] == "BF16":
        return mx.array(np.frombuffer(raw, dtype=np.uint16).reshape(shape)).view(mx.bfloat16)
    return mx.array(np.frombuffer(raw, dtype=np.dtype(_NUMPY[meta["dtype"]])).reshape(shape))


def _weights(path: Path) -> List[Tuple[str, mx.array]]:
    """Every tower tensor of a checkpoint, named as this module's parameters are. Read tensor by tensor:
    loading the whole shard (mx.load) would hold the text weights a second time, 4 GB on a 16 GB Mac."""
    out: List[Tuple[str, mx.array]] = []
    for f in sorted(Path(path).glob("*.safetensors")):
        with open(f, "rb") as fh:
            try:
                header_len = int.from_bytes(fh.read(8), "little")
                header = json.loads(fh.read(header_len))
            except (OSError, ValueError):  # (a shard we cannot read: the next one may hold the tower)
                continue
            if not isinstance(header, dict):
                continue
            for k, meta in header.items():
                if k == "__metadata__":
                    continue
                if k.startswith(_TOWER):
                    out.append((k[len(_TOWER):], _tensor(fh, header_len, meta)))
                elif k.startswith(_EMBED):
                    out.append((k, _tensor(fh, header_len, meta)))
    return out


def _embed_layout(cfg: VisionConfig, weights) -> VisionConfig:
    """``cfg`` with the ``embed_vision`` projection's layout as the checkpoint's tensors have it: mlx-lm's rule
    (``f"{path}.scales" in weights``), so an unquantized conversion loads as a plain linear and one packed at other
    bits than the file's default loads at its own. The shape of the packed weight and of the scales say the bits and
    the group size, and the scales' dtype says the mode (only affine keeps them in the weight's own dtype)."""
    from dataclasses import replace

    w = dict(weights)
    prefix = _EMBED + "embedding_projection."
    if prefix + "scales" not in w:
        return replace(cfg, embed_quantized=False)
    in_dim = cfg.hidden_size
    group_size = in_dim // w[prefix + "scales"].shape[-1]
    bits = (w[prefix + "weight"].shape[-1] * 32) // in_dim
    if w[prefix + "scales"].dtype != mx.uint8:
        mode = "affine"
    elif (bits, group_size) == (4, 16):
        mode = "nvfp4"
    elif (bits, group_size) == (4, 32):
        mode = "mxfp4"
    elif (bits, group_size) == (8, 32):
        mode = "mxfp8"
    else:
        raise ValueError(f"{prefix} is packed at {bits} bits with group size {group_size}, which mlx cannot read")
    return replace(cfg, embed_quantized=True, quant_bits=bits, quant_group_size=group_size, quant_mode=mode)


def load_tower(model_dir: Path, dtype, cfg: VisionConfig) -> Optional[VisionTower]:
    """The tower for a checkpoint (``cfg``: its vision_config), its floating weights cast to the text model's
    dtype (None: the files hold no vision weights). What the configuration alone cannot say -- whether the
    projection is quantized, at which bits, and whether the linears clamp -- comes from the tensors."""
    weights = _weights(model_dir)
    if not weights:
        return None
    cfg = _embed_layout(cfg, weights)
    if not cfg.use_clipped_linears:  # (a checkpoint that clamps none carries no bounds to load)
        weights = [(k, v) for k, v in weights if not k.endswith(_CLAMPS)]
    tower = VisionTower(cfg)
    cast = []
    for k, v in weights:
        if v.dtype in (mx.float16, mx.bfloat16, mx.float32) and dtype in (mx.float16, mx.bfloat16):
            v = v.astype(dtype)
        cast.append((k, v))
    tower.load_weights(cast)
    mx.eval(tower.parameters())
    return tower
