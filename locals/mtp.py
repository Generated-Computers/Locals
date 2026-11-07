"""Qwen3.5's MTP layer as a draft source for a request running alone (issue #18; GLOSSARY.md: Draft, Draft source,
MTP layer).

From the main model's hidden state at position i (after its final norm) and the embedding of token i+1, the layer
predicts token i+2 (docs/measurements/mtp/README.md: 82.7% of first drafts right on Zwyrol, teacher-forced, with
the layer in 4 bits). Its draft logits come from the main model's own output head. The 4-bit MLX files of
Qwen3.5-4B do not have the layer; ``LOCALS_MTP`` names a file with its 15 ``mtp.*`` tensors
(docs/measurements/mtp/download_mtp.py fetches them from ``Qwen/Qwen3.5-4B``).

The layer attends over a KV of its own: one more entry at the end of each request's cache list
(docs/adr/0001-mtp-kv-in-the-cache-list.md), a plain ``KVCache`` (``BatchKVCache`` in a batch), so every place that
copies, saves, merges, filters, trims or rewinds the cache carries it with no change. Its keys have one column more
than the layer's head: 1 where the slot holds the layer's KV for that position, 0 where it is a hole (a position the
layer did not run on: a batched step, a token whose next token is not known yet). A hole still takes its position, so
the entry's offset equals the main cache's, and its mark goes wherever the keys go. The layer's attention masks the
holes out.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import BatchKVCache, KVCache

from .kernels import head_logits, text_model

DRAFTS_ENV = "LOCALS_MTP_DRAFTS"


class Layer(nn.Module):
    """The checkpoint's ``mtp.*`` module: two input norms, the projection of [embedding, hidden] and one
    full-attention decoder layer, then its norm (the wiring settled in docs/measurements/mtp/README.md)."""

    def __init__(self, args):
        super().__init__()
        from mlx_lm.models.qwen3_5 import DecoderLayer

        H, eps = args.hidden_size, args.rms_norm_eps
        self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=eps)
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.layers = [DecoderLayer(args, layer_idx=args.full_attention_interval - 1)]  # a full-attention layer
        self.norm = nn.RMSNorm(H, eps=eps)


def why_off(model) -> Optional[str]:
    """Why MTP drafts cannot run on ``model`` (None: they can). They are for Qwen3.5-4B only: a dense Qwen3.5 text
    model with 32 layers of width 2560 (its config's MTP layer is not in mlx-lm's model arguments, so the file
    ``LOCALS_MTP`` names is the check that the layer exists: it must load into this model's layer shapes)."""
    lm = text_model(model)
    args = getattr(lm, "args", None)
    body = getattr(lm, "model", None)
    if args is None or not any(c.__name__ == "Qwen3_5TextModel" for c in type(body).__mro__):
        return "the model is not Qwen3.5"
    if getattr(args, "num_experts", 0):
        return "the model is not Qwen3.5-4B (a mixture of experts)"
    if (args.hidden_size, args.num_hidden_layers) != (2560, 32):
        return f"the model is not Qwen3.5-4B ({args.num_hidden_layers} layers of width {args.hidden_size})"
    return None


def fingerprint(path) -> str:
    """A mark for the MTP file (its path, size and time), for the disk-cache folder's name: a cache list with the
    MTP entry never meets files saved without one, or with another file's."""
    p = Path(path).expanduser().resolve()
    st = p.stat()
    return hashlib.sha1(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:8]


def drafts_setting() -> int:
    """MTP drafts per pass (LOCALS_MTP_DRAFTS): 1 or 2."""
    v = os.environ.get(DRAFTS_ENV, "1").strip()
    if v not in ("1", "2"):
        raise ValueError(f"{DRAFTS_ENV}={v!r}: 1 or 2")
    return int(v)


class _Flagged:
    """The entry as the layer's attention sees it: the new keys go in with their flag column, and come back
    without it."""

    def __init__(self, entry, flags: mx.array):
        self.entry, self.flags = entry, flags

    @property
    def offset(self):
        return self.entry.offset

    def update_and_fetch(self, keys, values):
        f = mx.broadcast_to(self.flags[:, None, :, None].astype(keys.dtype), keys.shape[:3] + (1,))
        k, v = self.entry.update_and_fetch(mx.concatenate([keys, f], axis=-1), values)
        return k[..., :-1], v


def _written(entry) -> int:
    """Positions written in the entry's buffer (a batch entry's count its left padding)."""
    if entry.keys is None:
        return 0
    return entry._idx if isinstance(entry, BatchKVCache) else entry.offset


class MTP:
    """The layer, loaded for one main model, and what it does to a cache list's MTP entry."""

    def __init__(self, model, layer: Layer, drafts: int = 1):
        self.lm = text_model(model)
        self.layer = layer
        self.drafts = drafts
        args = self.lm.args
        self.heads, self.head_dim = args.num_key_value_heads, args.head_dim
        self.dtype = None

    @classmethod
    def load(cls, model, path, drafts: int = 1, bits: Optional[int] = 4) -> "MTP":
        """The layer from ``path`` (the 15 ``mtp.*`` tensors, norm weights stored as w and used as 1+w, as mlx-lm's
        ``sanitize`` does for the main model's norms), in the main model's compute dtype, quantized to ``bits``
        (group 64; None: not quantized)."""
        lm = text_model(model)
        raw = mx.load(str(path))
        weights = [(k[len("mtp."):], v + 1.0 if v.ndim == 1 else v) for k, v in raw.items() if k.startswith("mtp.")]
        layer = Layer(lm.args)
        layer.load_weights(weights, strict=True)
        emb = lm.model.embed_tokens
        dtype = getattr(emb, "scales", getattr(emb, "weight", None)).dtype
        layer.set_dtype(dtype)
        if bits:
            nn.quantize(layer, group_size=64, bits=bits)
        mx.eval(layer.parameters())
        out = cls(model, layer, drafts)
        out.dtype = dtype
        return out

    @property
    def nbytes(self) -> int:
        """The layer's weights in memory."""
        from mlx.utils import tree_flatten

        return sum(v.nbytes for _, v in tree_flatten(self.layer.parameters()))

    # -- the entry ------------------------------------------------------------------------------------------------
    @staticmethod
    def entry() -> KVCache:
        return KVCache()

    def holes(self, entry, rows: int, n: int) -> None:
        """``n`` holes at the end of every row of the entry."""
        if n <= 0:
            return
        dt = self.dtype or mx.float16
        entry.update_and_fetch(mx.zeros((rows, self.heads, n, self.head_dim + 1), dt),
                               mx.zeros((rows, self.heads, n, self.head_dim), dt))

    def catch_up(self, entry, upto: int, rows: int) -> None:
        """Holes up to ``upto`` written positions (the main cache's: a pass the layer did not see)."""
        self.holes(entry, rows, upto - _written(entry))

    def _mask(self, entry, flags: mx.array) -> mx.array:
        """(B, 1, L, T): each new position sees the real slots before it and itself (if real), never a hole or
        a batch row's left padding."""
        B, L = flags.shape
        t0 = _written(entry)
        old = (entry.keys[:, 0, :t0, -1] > 0) if t0 else mx.zeros((B, 0), dtype=mx.bool_)
        if isinstance(entry, BatchKVCache) and t0:
            old = old & (mx.arange(t0)[None, :] >= entry.left_padding[:, None])
        real = mx.concatenate([old, flags], axis=1)  # (B, T)
        T = t0 + L
        causal = mx.arange(T)[None, :] <= (t0 + mx.arange(L))[:, None]  # (L, T)
        return real[:, None, None, :] & causal[None, None]

    def fill(self, entry, hidden: mx.array, nexts: mx.array, flags: mx.array) -> mx.array:
        """Run the layer over the entry's next L positions: ``hidden`` (B, L, D) the main model's hidden states there
        (after its final norm), ``nexts`` (B, L) the token after each, ``flags`` (B, L) where ``nexts`` is known (a
        False slot is written as a hole). Returns the layer's output (B, L, D), before its norm."""
        lay = self.layer
        e = self.lm.model.embed_tokens(nexts)
        x = lay.fc(mx.concatenate([lay.pre_fc_norm_embedding(e), lay.pre_fc_norm_hidden(hidden.astype(e.dtype))], axis=-1))
        mask = self._mask(entry, flags)
        return lay.layers[0](x, mask=mask, cache=_Flagged(entry, flags))

    def logits(self, out: mx.array) -> mx.array:
        """Draft logits of the layer's output: its norm, then the main model's output head."""
        return head_logits(self.lm, self.layer.norm(out))

    @staticmethod
    def tokens(logits: mx.array) -> mx.array:
        """The draft tokens of draft logits (..., V): their argmax."""
        return mx.argmax(logits, axis=-1)

    def draft(self, entry, out: mx.array, r, n: int) -> List[int]:
        """``n`` (1 or 2) chained drafts from ``out`` (1, 1, D), the layer's output at the entry's last slot: the first
        from its logits, the second from the layer run once more on its own output (normed) and the first draft's
        embedding, at the next slot, which is cut off again (the entry's offset stays the main cache's)."""
        d1 = self.tokens(self.logits(out[:, -1:, :])).reshape(1, 1)
        if n < 2:
            return d1.reshape(-1).tolist()
        out2 = self.fill(entry, self.layer.norm(out[:, -1:, :]), d1, mx.ones((1, 1), dtype=mx.bool_))
        d2 = self.tokens(self.logits(out2)).reshape(1, 1)
        entry.trim(1)
        return mx.concatenate([d1, d2], axis=1).reshape(-1).tolist()
