"""Flint (RecursiveSI/flint): a mixture-of-experts model with 39.8B parameters, about 2.5B of them used per token.

Each MoE layer has 256 experts and a token uses 6, so the experts (95% of the weights) stay in the weight files
and are read when a token is routed to them: the GPU holds the rest of the model (about 1.3 GB at 4 bits) and a
cache of recently used experts (locals.defaults.expert_cache_gb), and macOS keeps more of the files in its own cache.
That runs the 22.6 GB INT4 build (RecursiveSI/flint-int4) on a Mac whose GPU cannot hold it.

The architecture is Qwen3-MoE's with five changes (from the model card and Recursive's SGLang kit):
- a learned per-head logit in each attention softmax's denominator (``softmax_offset``, an attention sink);
- 35 layers attend to the current token and the 512 before it (``sliding_window`` 513), 14 to everything;
- RoPE on the sliding layers only (``no_rope_layer_indices`` lists the others);
- routing by sigmoid scores: the top 6, their scores renormalized to sum to one;
- an ungated shared expert added to every MoE block; layers 0 and 1 are dense SwiGLU MLPs.
"""
import json
import os
import struct
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

MODEL_PATHS: List[Path] = []  # the folder being loaded (set by locals.models.register around mlx-lm's load_model)
READ_AHEAD = int(os.environ.get("LOCALS_EXPERT_READ_AHEAD", "1"))  # MoE layers whose experts are guessed and read ahead
WHOLE = 96  # a pass that needs more of a layer's 256 experts (reading a prompt) reads the layer's expert tensors whole
_PARTS = [(p, k) for p in ("gate_proj", "up_proj", "down_proj") for k in ("weight", "scales", "biases")]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    layer_types: list
    no_rope_layer_indices: list
    mlp_only_layers: list
    sliding_window: int = 513
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 262144


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer: int):
        super().__init__()
        self.n_heads, self.n_kv_heads, d = args.num_attention_heads, args.num_key_value_heads, args.head_dim
        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * d, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv_heads * d, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.n_kv_heads * d, bias=False)
        self.o_proj = nn.Linear(self.n_heads * d, args.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.softmax_offset = mx.zeros((self.n_heads,))  # the sink logit of each head
        self.rope = None if layer in set(args.no_rope_layer_indices) else nn.RoPE(d, traditional=False, base=args.rope_theta)
        self.scale = d ** -0.5

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask, sinks=self.softmax_offset)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, dims: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dims, hidden, bias=False)
        self.up_proj = nn.Linear(dims, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class ExpertStore:
    """The experts of every MoE layer, read from the safetensors files one expert at a time: each of an expert's
    nine arrays (gate, up and down projection: 4-bit weights, scales, biases) is a contiguous slice of a tensor that
    stacks the layer's 256 experts.

    Decoding keeps recently used experts on the GPU in slots: per layer, nine arrays of ``slots`` experts each,
    allocated once, the least recently used slot taken for an expert that is not in one. A read expert is copied
    into its slot in place, so a token makes no new GPU buffers (on the M2 Pro a fresh 1.7 MB buffer cost 0.6 ms
    under memory pressure, a fifth of the time of a token's misses) and gather_qmm reads the slots directly.
    Experts not in a slot are read with pread on a few threads at once, which an SSD serves faster than one read
    after another (and macOS keeps what was read in its own file cache); ``prefetch`` starts reading the experts
    a layer is expected to need while the GPU works on the layer before. A pass that needs more experts than a
    layer has slots stacks them instead (many requests at once), and one that needs most of them (reading a
    prompt) reads the layer's tensors whole."""

    def __init__(self, folder: Path, budget_bytes: int, readers: int = 8):
        self.budget = budget_bytes
        self.hits = self.misses = self.ahead = 0
        self.pending: "OrderedDict[tuple, Any]" = OrderedDict()  # (layer, expert) -> its read, started ahead
        self.spec = {}  # (layer, part) -> (file descriptor, start, bytes per expert, shape per expert, dtype)
        self.where = {}  # tensor name -> its file
        self.ranges = {}  # layer -> [(file descriptor, start, bytes)] of its expert tensors
        self.experts = {}  # layer -> experts in its tensors
        self.warming = set()
        self.fds = []
        for f in sorted(folder.glob("model*.safetensors")):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            fd = os.open(f, os.O_RDONLY)
            self.fds.append(fd)
            for name, t in header.items():
                if name == "__metadata__" or ".mlp.switch_mlp." not in name:
                    continue
                layer = int(name.split(".layers.")[1].split(".")[0])
                proj, kind = name.split(".switch_mlp.")[1].split(".")
                begin, end = t["data_offsets"]
                experts = t["shape"][0]
                self.spec[(layer, (proj, kind))] = (fd, 8 + n + begin, (end - begin) // experts, t["shape"][1:], t["dtype"])
                self.where[name] = f
                self.ranges.setdefault(layer, []).append((fd, 8 + n + begin, end - begin))
                self.experts[layer] = experts
        self.pool = ThreadPoolExecutor(max_workers=readers, thread_name_prefix="locals-experts")
        layers = sorted(self.experts)
        per = self.expert_bytes(layers[0]) if layers else 1
        self.slots = max(1, min(min(self.experts.values(), default=1), budget_bytes // max(1, per * len(layers))))
        self.dtype = None
        self.arrays = {}  # layer -> its nine slot arrays (made on first use, in the compute dtype)
        self.lru = {}  # layer -> OrderedDict expert -> slot, least recently used first
        self.free = {}  # layer -> free slots

    def expert_bytes(self, layer: int) -> int:
        return sum(self.spec[(layer, part)][2] for part in _PARTS)

    @property
    def bytes(self) -> int:
        """GPU memory of the slots made so far."""
        return sum(a.nbytes for arrs in self.arrays.values() for a in arrs)

    def _raw(self, layer: int, e: int) -> np.ndarray:
        """An expert's nine arrays as one buffer (read straight into it)."""
        buf = np.empty(self.expert_bytes(layer), dtype=np.uint8)
        view, o = memoryview(buf), 0
        for part in _PARTS:
            fd, start, size, _, _ = self.spec[(layer, part)]
            os.preadv(fd, [view[o:o + size]], start + e * size)
            o += size
        return buf

    def _arrays(self, layer: int, buf: np.ndarray, dtype) -> list:
        whole = mx.array(buf)  # one copy per expert; its nine arrays are views of it
        parts, o = [], 0
        for part in _PARTS:
            size, shape, kind = self.spec[(layer, part)][2:]
            a = whole[o:o + size]
            o += size
            if kind == "U8":  # 4-bit weights, two to a byte: MLX's packing in 32-bit words
                a = a.view(mx.uint32).reshape(shape[0], -1)
            else:  # BF16 scales and biases, in the compute dtype
                a = a.view(mx.bfloat16).reshape(shape).astype(dtype)
            parts.append(a)
        return parts

    def _reads(self, layer: int, missing: list) -> dict:
        """The buffers of ``missing``: those read ahead (done or under way), the others read now, in parallel."""
        ahead = {e: self.pending.pop((layer, e)) for e in missing if (layer, e) in self.pending}
        self.ahead += len(ahead)
        later = [e for e in missing if e not in ahead]
        raw = dict(zip(later, self.pool.map(lambda e: self._raw(layer, e), later) if len(later) > 1
                       else [self._raw(layer, e) for e in later]))
        raw.update({e: f.result() for e, f in ahead.items()})
        return raw

    def _slot_arrays(self, layer: int, dtype) -> list:
        if self.dtype is None or self.dtype != dtype:  # first use, or the runtime changed the compute dtype (its probe): start over
            self.arrays.clear()
            self.lru.clear()
            self.free.clear()
            self.dtype = dtype
        arrs = self.arrays.get(layer)
        if arrs is None:
            arrs = []
            for part in _PARTS:
                size, shape, kind = self.spec[(layer, part)][2:]
                if kind == "U8":
                    arrs.append(mx.zeros((self.slots, shape[0], size // shape[0] // 4), dtype=mx.uint32))
                else:
                    arrs.append(mx.zeros((self.slots, *shape), dtype=dtype))
            mx.eval(arrs)
            self.arrays[layer] = arrs
            self.lru[layer] = OrderedDict()
            self.free[layer] = list(range(self.slots - 1, -1, -1))
        return arrs

    def in_slots(self, layer: int, experts: list, dtype):
        """For a pass needing at most ``slots`` experts: the layer's nine slot arrays with ``experts`` in them,
        and each expert's slot. None for a pass that needs more."""
        if len(experts) > self.slots:
            return None
        arrs = self._slot_arrays(layer, dtype)
        lru, free = self.lru[layer], self.free[layer]
        for e in experts:
            if e in lru:
                lru.move_to_end(e)
                self.hits += 1
        missing = [e for e in experts if e not in lru]
        if missing:
            raw = self._reads(layer, missing)
            wanted = set(experts)
            for e in missing:
                if free:
                    s = free.pop()
                else:  # the least recently used slot this pass does not need
                    old = next(k for k in lru if k not in wanted)
                    s = lru.pop(old)
                start = mx.array([s, 0, 0])
                for j, part in enumerate(self._arrays(layer, raw[e], dtype)):
                    arrs[j] = mx.slice_update(arrs[j], part[None], start, (0, 1, 2))  # in place (the old array has no other user)
                lru[e] = s
                self.misses += 1
        return arrs, [lru[e] for e in experts]

    def stacked(self, layer: int, experts: list, dtype) -> list:
        """The nine arrays of ``experts`` (in this order), each stacked along a first axis: for a pass that needs
        more experts than a layer has slots. Those in slots are taken from there; nothing is kept."""
        arrs = self.arrays.get(layer) if self.dtype is not None and self.dtype == dtype else None
        lru = self.lru.get(layer, {}) if arrs is not None else {}
        raw = self._reads(layer, [e for e in experts if e not in lru])
        self.hits += len(experts) - len(raw)
        self.misses += len(raw)
        cols = [[] for _ in _PARTS]
        for e in experts:
            parts = [a[lru[e]] for a in arrs] if e in lru else self._arrays(layer, raw[e], dtype)
            for c, part in zip(cols, parts):
                c.append(part)
        return [mx.stack(c) for c in cols]

    def whole(self, layer: int, dtype) -> list:
        """The layer's nine stacked expert tensors, read whole when used (MLX's lazy load of the file), for a pass
        that needs most of the experts: not kept, and no per-expert work."""
        loaded, out = {}, []
        for proj, kind in _PARTS:
            name = f"model.layers.{layer}.mlp.switch_mlp.{proj}.{kind}"
            f = self.where[name]
            if f not in loaded:
                loaded[f] = mx.load(str(f))  # (its arrays are read only if evaluated: just these nine)
            a = loaded[f][name]
            out.append(a.view(mx.uint32) if kind == "weight" else a.astype(dtype))
        return out

    def warm(self, layer: int) -> None:
        """Read ``layer``'s expert tensors into macOS's file cache in the background, for whole() to find there."""
        if layer not in self.ranges or layer in self.warming:
            return
        self.warming.add(layer)

        def read():
            buf = bytearray(8 << 20)
            view = memoryview(buf)
            for fd, start, size in self.ranges[layer]:
                o = 0
                while o < size:
                    n = min(len(buf), size - o)
                    os.preadv(fd, [view[:n]], start + o)
                    o += n
            self.warming.discard(layer)

        self.pool.submit(read)

    def prefetch(self, layer: int, experts) -> None:
        """Start reading ``experts`` of ``layer`` unless they are in slots or on their way."""
        lru = self.lru.get(layer, {})
        for e in experts:
            key = (layer, e)
            if e not in lru and key not in self.pending:
                self.pending[key] = self.pool.submit(self._raw, layer, e)
        while len(self.pending) > 512:  # reads guessed for a pass that took other experts
            self.pending.popitem(last=False)


class StreamedExperts(nn.Module):
    """SwitchGLU over experts the ExpertStore reads: the routed experts of this pass, stacked, and mx.gather_qmm."""

    def __init__(self, layer: int):
        super().__init__()
        self.layer = layer
        self.store: Optional[ExpertStore] = None

    def __call__(self, x: mx.array, inds: mx.array) -> mx.array:
        mx.eval(inds)  # which experts this pass needs, before they are read
        flat = np.array(inds).reshape(-1)
        experts, inv = np.unique(flat, return_inverse=True)
        if len(experts) > WHOLE:  # most of the layer's experts (reading a prompt): the layer's tensors, read whole
            self.store.warm(self.layer + 1)
            gw, gs, gb, uw, us, ub, dw, ds, db = self.store.whole(self.layer, x.dtype)
            idx = inds.astype(mx.uint32)
        else:
            got = self.store.in_slots(self.layer, experts.tolist(), x.dtype)
            if got is not None:  # decoding: the slots, indexed by each token's experts' slots
                (gw, gs, gb, uw, us, ub, dw, ds, db), slots = got
                idx = mx.array(np.asarray(slots, dtype=np.uint32)[inv].reshape(inds.shape))
            else:  # more experts than a layer has slots (many requests at once): stacked for this pass
                gw, gs, gb, uw, us, ub, dw, ds, db = self.store.stacked(self.layer, experts.tolist(), x.dtype)
                idx = mx.array(inv.reshape(inds.shape).astype(np.uint32))
        x = mx.expand_dims(x, (-2, -3))
        do_sort = idx.size >= 64
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, idx)

        def qmm(a, w, s, b):
            return mx.gather_qmm(a, w, s, b, rhs_indices=idx, transpose=True, group_size=64, bits=4,
                                 sorted_indices=do_sort)

        h = swiglu(qmm(x, gw, gs, gb), qmm(x, uw, us, ub))
        y = qmm(h, dw, ds, db)
        if do_sort:
            y = _scatter_unsort(y, inv_order, inds.shape)
        return y.squeeze(-2)


class SparseMoe(nn.Module):
    def __init__(self, args: ModelArgs, layer: int):
        super().__init__()
        self.k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.experts = StreamedExperts(layer)
        self._next = ()  # the next MoE layers' routers and indices, to guess their experts (set by Model.sanitize)

    def __call__(self, x: mx.array) -> mx.array:
        scores = mx.sigmoid(self.gate(x).astype(mx.float32))
        inds = mx.argpartition(-scores, kth=self.k - 1, axis=-1)[..., : self.k]
        if self._next and inds.size <= 16 * self.k:  # decoding: the next layers' experts, guessed from this input, read ahead
            guesses = [mx.argpartition(-gate(x), kth=self.k - 1, axis=-1)[..., : self.k] for gate, _ in self._next]
            mx.eval(inds, *guesses)
            for (_, layer), guess in zip(self._next, guesses):
                self.experts.store.prefetch(layer, np.unique(np.array(guess)).tolist())
        else:
            mx.eval(inds)
        w = mx.take_along_axis(scores, inds, axis=-1)
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
        y = self.experts(x, inds)
        y = (y * w[..., None].astype(y.dtype)).sum(axis=-2)
        return y + self.shared_expert(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer: int):
        super().__init__()
        self.self_attn = Attention(args, layer)
        dense = layer in set(args.mlp_only_layers)
        self.mlp = MLP(args.hidden_size, args.intermediate_size) if dense else SparseMoe(args, layer)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class FlintModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layer_types = args.layer_types
        self.swa_idx = self.layer_types.index("sliding_attention")
        self.fa_idx = self.layer_types.index("full_attention")

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        full_mask = create_attention_mask(h, cache[self.fa_idx])
        swa_mask = create_attention_mask(h, cache[self.swa_idx], window_size=self.args.sliding_window)
        for layer, c, kind in zip(self.layers, cache, self.layer_types):
            h = layer(h, full_mask if kind == "full_attention" else swa_mask, c)
        return self.norm(h)


class Model(nn.Module):
    # Each pass over a prompt reads every expert of a layer (about 7 s on the M2 Pro, whatever its length), so
    # prompts go in few long passes: 4096 tokens (a pass of 8192 peaked at 3.5 GB of GPU memory, too much next
    # to the slots on a 16 GB Mac). The fused attention makes no score matrix to bound (HybridBatchEngine).
    prefill_tokens = 4096
    prefill_scores = None

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FlintModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None) -> mx.array:
        return self.lm_head(self.model(inputs, cache, input_embeddings))

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights: dict) -> dict:
        """The experts stay in their files (an ExpertStore reads them); the INT4 build keeps 4-bit weights as bytes,
        which MLX packs in 32-bit words."""
        out = {}
        for name, w in weights.items():
            if ".mlp.switch_mlp." in name:
                continue
            if name.endswith(".weight") and w.dtype == mx.uint8:
                w = w.view(mx.uint32)
            out[name] = w
        if not MODEL_PATHS:
            raise RuntimeError("Flint's experts are read from its folder: load it with locals (locals.models.register)")
        from ..defaults import expert_cache_gb

        store = ExpertStore(MODEL_PATHS[-1], int(expert_cache_gb() * 2**30))
        moe = [i for i, layer in enumerate(self.model.layers) if isinstance(layer.mlp, SparseMoe)]
        for n, i in enumerate(moe):
            self.model.layers[i].mlp.experts.store = store
            # the MoE layers after this one (a plain tuple attribute: not a child module, not a parameter path)
            ahead = tuple((self.model.layers[j].mlp.gate, j) for j in moe[n + 1:n + 1 + READ_AHEAD])
            object.__setattr__(self.model.layers[i].mlp, "_next", ahead)
        self.expert_store = store
        return out

    def make_cache(self) -> List[Any]:
        return [KVCache() if kind == "full_attention" else RotatingKVCache(max_size=self.args.sliding_window)
                for kind in self.model.layer_types]
