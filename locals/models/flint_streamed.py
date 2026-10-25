"""Flint (RecursiveSI/flint): a mixture-of-experts model with 39.8B parameters, about 2.5B of them used per token.

Each MoE layer has 256 experts and a token uses 6, so the experts (95% of the weights) stay in the weight files
and are read when a token is routed to them: the GPU holds the rest of the model (about 1.3 GB at 4 bits) and a
cache of experts (locals.defaults.expert_cache_gb). That runs the 22.6 GB INT4 build (RecursiveSI/flint-int4) on a
Mac whose GPU cannot hold it. A decode pass's MoE blocks run in Locals' own kernels (flint_kernels).

The architecture is Qwen3-MoE's with five changes (from the model card and Recursive's SGLang kit):
- a learned per-head logit in each attention softmax's denominator (``softmax_offset``, an attention sink);
- 35 layers attend to the current token and the 512 before it (``sliding_window`` 513), 14 to everything;
- RoPE on the sliding layers only (``no_rope_layer_indices`` lists the others);
- routing by sigmoid scores: the top 6, their scores renormalized to sum to one;
- an ungated shared expert added to every MoE block; layers 0 and 1 are dense SwiGLU MLPs.
"""
import errno
import fcntl
import heapq
import json
import os
import struct
import threading
import time
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
# The experts read ahead: layer l's MoE input through the routers of the next MoE layers guesses their experts,
# the more reliably the higher the guess ranks (in a pass's top 6, on the M2 Pro: the top 3 one layer ahead 99, 97
# and 90% of the time, the 6th 46%; the top 2 four layers ahead 95 and 84%), so each next layer gets fewer.
GUESSES = tuple(int(g) for g in os.environ.get("LOCALS_EXPERT_GUESSES", "5,4,3,2").split(",") if g.strip())
# Guessed reads running at once at most, so a read a layer waits for finds the SSD with few others in its queue:
# without the expert pack an expert is four reads, and 4 made a token 139 ms instead of 156 (2: 150, 8: 149);
# with it an expert is one read, and 2 made it 86 ms instead of 98 (1: 88, 3: 92, 4: 94). 0: no limit.
GUESS_READS = int(os.environ["LOCALS_EXPERT_GUESS_READS"]) if os.environ.get("LOCALS_EXPERT_GUESS_READS") else None
FUSED = os.environ.get("LOCALS_FLINT_KERNELS", "1") != "0"  # decode passes' MoE blocks in Locals' kernels (flint_kernels)
# Decode steps that ask the host only on a missing expert (FlintModel._optimistic): "auto" (when most layers find
# their experts in slots), "always", or "0" (never)
OPTIMISTIC = os.environ.get("LOCALS_FLINT_OPTIMISTIC") or "auto"
OPTIMISTIC = "auto" if OPTIMISTIC == "1" else OPTIMISTIC
REFINE = int(os.environ.get("LOCALS_EXPERT_REFINE", "6"))  # guesses of the next layer from a partial result (0: off)
KEEP_GPU_BUSY = os.environ.get("LOCALS_KEEP_GPU_BUSY", "1") != "0"  # while a pass waits for reads (ExpertStore._wait)
DECAY = 0.5 ** (1 / 32)  # an expert's use counts half as much 32 passes over its layer later (which slot to take)
WINDOW_BLOCK = 256  # queries a sliding layer's attention takes at once over a long pass (Attention._windowed)
SETS_IDLE_S = 2.0  # seconds of decoding without a prompt pass after which the layer sets are let go (drop_sets)
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
        self.window = args.sliding_window if args.layer_types[layer] == "sliding_attention" else None

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
        if self.window is not None and isinstance(mask, mx.array) and L >= 2 * WINDOW_BLOCK:
            out = self._windowed(q, k, v, mask)
        else:
            out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask, sinks=self.softmax_offset)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))

    def _windowed(self, q: mx.array, k: mx.array, v: mx.array, mask: mx.array) -> mx.array:
        """A sliding layer's attention over many queries, in blocks of WINDOW_BLOCK queries, each over the keys its
        window reaches: the mask leaves a query's other keys out, so this is the same softmax (bit for bit on the
        M2 Pro). One call with the window as a mask array computes every query against every key: over a
        4,096-token pass 57 ms a layer, in blocks of 256 queries 11.6 ms. The queries are the keys' last L."""
        L, T = q.shape[2], k.shape[2]
        outs = []
        for s in range(0, L, WINDOW_BLOCK):
            e = min(L, s + WINDOW_BLOCK)
            k0 = max(0, T - L + s - (self.window - 1))  # the first key the block's first query reaches
            k1 = T - L + e
            outs.append(mx.fast.scaled_dot_product_attention(q[:, :, s:e], k[:, :, k0:k1], v[:, :, k0:k1], scale=self.scale,
                                                             mask=mask[..., s:e, k0:k1], sinks=self.softmax_offset))
        return mx.concatenate(outs, axis=2)


class MLP(nn.Module):
    def __init__(self, dims: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dims, hidden, bias=False)
        self.up_proj = nn.Linear(dims, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def converted(arrs: list, dtype) -> list:
    """Nine expert arrays (weights, scales, biases of gate, up, down) with the scales and biases in ``dtype`` (on the
    GPU, when used): for MLX's quantized products, which take them in the input's type."""
    return [a if i % 3 == 0 or a.dtype == dtype else a.astype(dtype) for i, a in enumerate(arrs)]


class _Fill:
    """One expert's read into its slot: four reads (its three 4-bit weight arrays, then its six scales and biases)
    that reader threads take in turn, so the parts of one expert are read at once."""

    __slots__ = ("tasks", "next", "left", "done", "error", "cancelled", "slot", "victim", "put", "got", "guess")

    def __init__(self, tasks: list):
        self.tasks, self.next, self.left = tasks, 0, len(tasks)
        self.done, self.error, self.cancelled = threading.Event(), None, False
        self.slot = self.victim = self.put = self.got = None  # set by ExpertStore._take
        self.guess = False  # read ahead on a guess (and not yet wanted by its layer's pass)

    def wait(self) -> None:
        self.done.wait()
        if self.error is not None:
            raise self.error


def _pread(fd: int, dst, offset: int) -> None:
    """Read into ``dst``, a buffer or a list of them filled in order (one scatter read: an expert of the pack into
    its nine slot arrays)."""
    bufs = dst if isinstance(dst, list) else [dst]
    while bufs:
        n = os.preadv(fd, bufs, offset)
        if n <= 0:
            raise OSError(errno.EIO, f"the weight file ends {sum(len(b) for b in bufs)} bytes into an expert")
        offset += n
        while bufs and n >= len(bufs[0]):
            n -= len(bufs[0])
            bufs = bufs[1:]
        if bufs and n:
            bufs = [bufs[0][n:]] + bufs[1:]


_LIBC = None


def _libc():
    """libSystem through ctypes: memcpy (the GIL let go while it copies) and Mach's purgeable memory; None off macOS."""
    global _LIBC
    if _LIBC is None:
        import ctypes

        try:
            lib = ctypes.CDLL(None, use_errno=True)
            lib.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            lib.memcpy.restype = ctypes.c_void_p
            lib.mach_vm_allocate.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64, ctypes.c_int]
            lib.mach_vm_allocate.restype = ctypes.c_int
            lib.mach_vm_deallocate.argtypes = [ctypes.c_uint, ctypes.c_uint64, ctypes.c_uint64]
            lib.mach_vm_deallocate.restype = ctypes.c_int
            lib.vm_purgable_control.argtypes = [ctypes.c_uint, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
            lib.vm_purgable_control.restype = ctypes.c_int
            lib.task = ctypes.c_uint.in_dll(lib, "mach_task_self_").value
            _LIBC = lib
        except (AttributeError, OSError, ValueError):
            _LIBC = False
    return _LIBC or None


class _RamCache:
    """Experts evicted from the GPU's slots, kept in the Mac's memory outside the GPU's share: a miss found here is
    one memcpy into its slot (0.04 ms for an expert) instead of a read from the SSD (2 to 4 ms while reads queue).
    Each expert is a purgeable region, volatile while it waits: when macOS needs the memory it empties regions
    instead of writing them to swap, and an emptied one is a plain miss. The expert put in longest ago goes first."""

    NONVOLATILE, VOLATILE, EMPTY = 0, 1, 2

    def __init__(self, entry_bytes: int, capacity: int):
        import ctypes

        self.c = ctypes
        self.lib = _libc()
        self.size = -(-entry_bytes // 16384) * 16384
        self.capacity = capacity
        self.lru: "OrderedDict[tuple, int]" = OrderedDict()  # (layer, expert) -> address
        self.free: list = []  # regions holding no expert
        self.regions: list = []
        self.hits = self.puts = self.purged = 0

    def _state(self, addr: int, state: int) -> int:
        st = self.c.c_int(state)
        self.lib.vm_purgable_control(self.lib.task, addr, 0, self.c.byref(st))  # VM_PURGABLE_SET_STATE
        return st.value

    def region(self) -> Optional[int]:
        """A region to copy an expert into (its old contents dropped), or None."""
        if self.free:
            addr = self.free.pop()
        elif len(self.regions) < self.capacity:
            a = self.c.c_uint64(0)
            if self.lib.mach_vm_allocate(self.lib.task, self.c.byref(a), self.size, 0x1 | 0x2):  # ANYWHERE | PURGABLE
                return None
            addr = a.value
            self.regions.append(addr)
        elif self.lru:
            addr = self.lru.popitem(last=False)[1]
        else:
            return None
        self._state(addr, self.NONVOLATILE)
        return addr

    def put(self, key: tuple, addr: int) -> None:
        self._state(addr, self.VOLATILE)
        old = self.lru.pop(key, None)
        if old is not None:
            self.free.append(old)
        self.lru[key] = addr
        self.puts += 1

    def get(self, key: tuple) -> Optional[int]:
        """The region holding ``key``, taken out (keep it from being emptied until ``release``), or None."""
        addr = self.lru.pop(key, None)
        if addr is None:
            return None
        if self._state(addr, self.NONVOLATILE) == self.EMPTY:  # macOS took the memory back
            self.purged += 1
            self.release(addr)
            return None
        self.hits += 1
        return addr

    def release(self, addr: int) -> None:
        self._state(addr, self.VOLATILE)
        self.free.append(addr)

    def clear(self) -> None:
        for addr in self.lru.values():
            self.free.append(addr)
        self.lru.clear()

    def close(self) -> None:
        for addr in self.regions:
            self.lib.mach_vm_deallocate(self.lib.task, addr, self.size)
        self.regions, self.free = [], []
        self.lru.clear()


class _Reader:
    """Threads that read experts into their slots, the most urgent first: a read is queued with its layer's place
    in the model, so the reads a layer waits for come before those guessed for the layers after it, and a guess
    that the layer's routing did not confirm is dropped if no thread has started it."""

    def __init__(self, threads: int, guesses: int = 0):
        self.heap: list = []
        self.seq = 0
        self.stopped = False
        self.guesses = guesses  # guessed reads running at once at most (0: no limit), leaving the SSD to wanted ones
        self.running_guesses = 0
        self.cv = threading.Condition(threading.Lock())
        for i in range(threads):
            threading.Thread(target=self._work, name=f"locals-experts-{i}", daemon=True).start()

    def submit(self, priority: int, tasks: list, guess: bool = False) -> _Fill:
        fill = _Fill(tasks)
        fill.guess = guess
        with self.cv:
            heapq.heappush(self.heap, (priority, self.seq, fill))
            self.seq += 1
            self.cv.notify(len(tasks))
        return fill

    def want(self, fill: _Fill) -> None:
        """``fill`` was a guess its layer's pass now wants: past the limit on guesses."""
        with self.cv:
            if fill.guess:
                fill.guess = False
                self.cv.notify_all()

    def cancel(self, fill: _Fill) -> bool:
        """Drop ``fill`` if no part of it was started (then its slot is free once a thread passes it)."""
        with self.cv:
            if fill.next == 0 and not fill.cancelled:
                fill.cancelled = True
                return True
            return False

    def stop(self) -> None:
        with self.cv:
            self.stopped = True
            self.cv.notify_all()

    def _work(self) -> None:
        while True:
            with self.cv:
                while not self.heap:
                    if self.stopped:
                        return
                    self.cv.wait()
                fill = self.heap[0][2]
                if fill.guess and self.guesses and self.running_guesses >= self.guesses and not fill.cancelled:
                    self.cv.wait(0.002)  # (a wanted read, queued first, or a guess's end, wakes it sooner)
                    continue
                if fill.cancelled:
                    heapq.heappop(self.heap)
                    fill.left -= len(fill.tasks) - fill.next
                    fill.next = len(fill.tasks)
                    if fill.left == 0:
                        fill.done.set()
                    continue
                task = fill.tasks[fill.next]
                fill.next += 1
                if fill.next == len(fill.tasks):
                    heapq.heappop(self.heap)
                guess = fill.guess
                self.running_guesses += guess
            try:
                for fd, offset, dst, n in task:
                    if fd < 0:  # a copy in memory: (-1, destination, source, bytes)
                        _libc().memcpy(offset, dst, n)
                    else:  # (fd, offset, destination, None): the file's bytes as they are
                        _pread(fd, dst, offset)
            except Exception as e:  # noqa: BLE001 (the waiting pass raises it)
                fill.error = e
            with self.cv:
                fill.left -= 1
                self.running_guesses -= guess
                if fill.left == 0:
                    fill.done.set()
                if guess:
                    self.cv.notify()


class _Layer:
    """One MoE layer's slots: which expert each holds, the reads under way, and how much each expert was used."""

    __slots__ = ("arrays", "dst", "addr", "where", "free", "fills", "fresh", "score", "unit", "n")

    def __init__(self, arrays: list, dst: list, addr: list, slots: int, experts: int):
        self.n = slots
        self.arrays, self.dst = arrays, dst  # the nine slot arrays; dst[slot]: where each part of a slot is written
        self.addr = addr  # addr[slot]: the address of each part of the slot
        self.where = {}  # expert -> its slot (read, or being read)
        self.free = list(range(slots - 1, -1, -1))
        self.fills = {}  # expert -> its read under way
        self.fresh = set()  # experts guessed for the layer's next pass: not evicted before it
        self.score = np.zeros(experts)  # each expert's uses, each counted DECAY times less a pass later (in self.unit)
        self.unit = 1.0


class ExpertStore:
    """The experts of every MoE layer, read from the safetensors files one expert at a time: each of an expert's
    nine arrays (gate, up and down projection: 4-bit weights, scales, biases) is a contiguous slice of a tensor that
    stacks the layer's 256 experts.

    Decoding keeps experts on the GPU in slots: per layer, nine arrays of ``slots`` experts each, allocated once.
    A missing expert is read from the file straight into a slot (preadv into the slot's memory, which the GPU
    shares with the CPU): no buffer is made and no GPU work is done to place it (copying read experts into slots
    on the GPU took 63 ms of a token's 257 on the M2 Pro). The slot taken is the one whose expert was used least,
    each use counting half as much 32 passes later (on the M2 Pro this kept 61% of the experts a token used
    where the least recently used slot kept 57.5%). The reads bypass macOS's file cache (F_NOCACHE): the slots
    hold what was read, and the file cache would hold the same experts twice. Reader threads read several experts
    at once and the parts of each at once, which an SSD serves faster than one read after another, the reads a
    layer waits for first; ``prefetch`` reads the experts a later layer is expected to need while the GPU works on
    the layers before. A pass that needs more experts than a layer has slots stacks them instead (many requests
    at once), and one that needs most of them (reading a prompt) reads the layer's tensors whole.

    Writing a slot is safe when the CPU does it: the model waits for the GPU at every MoE layer's routing, every
    result computed from a layer's slots is evaluated before one of them is written (``fence``), and a slot being
    filled is used only after its read is done."""

    def __init__(self, folder: Path, budget_bytes: int, readers: int = 16, ram_bytes: int = 0):
        self.budget = budget_bytes
        self.hits = self.misses = self.ahead = 0
        self.ready = self.passes = 0  # MoE blocks of decode passes, and those that found every expert in a slot
        self.spec = {}  # (layer, part) -> (file descriptor, start, bytes per expert, shape per expert, dtype)
        self.where = {}  # tensor name -> its file
        self.experts = {}  # layer -> experts in its tensors
        self.fds = []
        for f in sorted(folder.glob("model*.safetensors")):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            fd = os.open(f, os.O_RDONLY)
            if hasattr(fcntl, "F_NOCACHE"):
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
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
                self.experts[layer] = experts
        self.pack = {}  # layer -> the descriptor of its expert-major file (models/pack.py), if written
        if os.environ.get("LOCALS_EXPERT_PACK", "1") != "0":
            from . import pack as _pack

            found = _pack.find(folder)
            for layer in (sorted(self.experts) if found is not None else []):
                fd = os.open(found / f"layer-{layer}.bin", os.O_RDONLY)
                if hasattr(fcntl, "F_NOCACHE"):
                    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                self.fds.append(fd)
                self.pack[layer] = fd
        self.pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="locals-experts")  # stacked()
        self.reader = _Reader(readers, GUESS_READS if GUESS_READS is not None else (2 if self.pack else 4))
        layers = sorted(self.experts)
        per = self.expert_bytes(layers[0]) if layers else 1
        self.slots = max(1, min(min(self.experts.values(), default=1), budget_bytes // max(1, per * len(layers))))
        self.arrays = {}  # layer -> its nine slot arrays (made on first use, in the compute dtype)
        self.layers = {}  # layer -> its _Layer
        self.uses = {}  # layer -> results computed from its slots, maybe not yet evaluated (see fence)
        self._filler = None  # (x, w) of the product that keeps the GPU busy while a pass waits for reads (_wait)
        self._sets = None  # two sets of a whole layer's expert tensors, for passes over prompts (whole)
        self._sets_used = 0.0
        self.buf_of = {}  # layer -> (set, its read): a layer read into a set ahead of its pass
        self.buf_uses = {}  # set -> results computed from it, maybe not yet evaluated
        self._used_set = 0
        ram = int(ram_bytes // max(1, per)) if layers and _libc() is not None else 0
        self.ram = _RamCache(per, ram) if ram > 0 else None  # experts evicted from the slots, in the Mac's memory

    def close(self) -> None:
        self.reader.stop()
        if getattr(self, "ram", None) is not None:
            self.ram.close()
        self.pool.shutdown(wait=False)
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds = []

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 (interpreter shutdown)
            pass

    def expert_bytes(self, layer: int) -> int:
        return sum(self.spec[(layer, part)][2] for part in _PARTS)

    @property
    def bytes(self) -> int:
        """GPU memory of the slots made so far."""
        return sum(a.nbytes for arrs in self.arrays.values() for a in arrs)

    @property
    def planned_bytes(self) -> int:
        """GPU memory of the slots once every layer's are made."""
        return sum(self.slots * self.expert_bytes(layer) for layer in self.experts)

    def reset(self) -> None:
        """Every slot emptied and made again on next use (with ``slots`` as it is then)."""
        for L in self.layers.values():
            for f in list(L.fills.values()):
                self.reader.cancel(f)
                f.done.wait()
        if self.ram is not None:
            self.ram.clear()
        had = bool(self.arrays)
        self.arrays.clear()
        self.layers.clear()
        self.uses.clear()
        if had:  # (the old slots back to macOS now, not kept in MLX's pool next to the new ones)
            mx.clear_cache()

    def reserve(self, dtype=None) -> None:
        """Make every layer's slots now (so the memory they take is in use from the start)."""
        for layer in sorted(self.experts):
            self._slot_arrays(layer)

    def _raw(self, layer: int, e: int) -> np.ndarray:
        """An expert's nine arrays as one buffer (read straight into it)."""
        buf = np.empty(self.expert_bytes(layer), dtype=np.uint8)
        view, o = memoryview(buf), 0
        if layer in self.pack:  # (the pack's layout is this buffer's)
            _pread(self.pack[layer], view, e * len(buf))
            return buf
        for part in _PARTS:
            fd, start, size, _, _ = self.spec[(layer, part)]
            _pread(fd, view[o:o + size], start + e * size)
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

    def _slot_arrays(self, layer: int, dtype=None) -> list:
        """The layer's nine slot arrays, made on first use: the files' bytes as they are (4-bit weights in MLX's
        32-bit words, bfloat16 scales and biases), whatever the compute dtype; kernels read them as they are and
        MLX's ops convert them on the GPU (``converted``)."""
        arrs = self.arrays.get(layer)
        if arrs is None:
            arrs, host = [], []
            for part in _PARTS:
                size, shape, kind = self.spec[(layer, part)][2:]
                if kind == "U8":
                    a = mx.zeros((self.slots, shape[0], size // shape[0] // 4), dtype=mx.uint32)
                    mx.eval(a)
                    h = np.asarray(memoryview(a))
                else:  # BF16: 16-bit slots seen as bfloat16
                    raw = mx.zeros((self.slots, *shape), dtype=mx.uint16)
                    a = raw.view(mx.bfloat16)
                    mx.eval(raw, a)
                    h = np.asarray(memoryview(raw))
                arrs.append(a)
                host.append(h)
            dst = [[memoryview(h[s]).cast("B") for h in host] for s in range(self.slots)]
            addr = [[h[slot].ctypes.data for h in host] for slot in range(self.slots)]
            self.arrays[layer] = arrs
            self.layers[layer] = _Layer(arrs, dst, addr, self.slots, self.experts[layer])
        return arrs

    def used(self, layer: int, result: mx.array) -> None:
        """``result`` was computed from ``layer``'s slots (see fence)."""
        uses = self.uses.setdefault(layer, [])
        uses.append(result)
        if len(uses) > 16:  # (results a caller evaluated long ago: let them go)
            self.fence(layer)

    def fence(self, layer: int) -> None:
        """Before a slot of ``layer`` is written: every result computed from its slots is evaluated. A caller may
        leave a pass's result unevaluated while it starts the next one (results kept lazy across calls), and the
        pending pass would read the slot after the new expert is written into it."""
        uses = self.uses.pop(layer, None)
        if uses:
            mx.eval(uses)

    def _settle(self, L: _Layer) -> dict:
        """Reads of ``L`` that ended: their experts are in their slots, or (a failed read) not kept; the experts
        they evicted are in the RAM cache. Returns the failed ones' errors."""
        failed = {}
        for e, f in [(e, f) for e, f in L.fills.items() if f.done.is_set()]:
            del L.fills[e]
            self._ended(f)
            if f.error is not None:
                failed[e] = f.error
                L.free.append(L.where.pop(e))
        return failed

    def _ended(self, f: _Fill) -> None:
        if f.put is not None:
            key, addr = f.put
            if f.error is None and not f.cancelled:
                self.ram.put(key, addr)
            else:
                self.ram.release(addr)
        if f.got is not None:
            self.ram.release(f.got)

    def _cancel(self, layer: int, L: _Layer, e: int, f: _Fill) -> bool:
        """Drop ``e``'s read if no part of it started: its slot is as it was, so the expert it would have evicted
        keeps it (or it is free again), and a RAM cache region it would have used is not."""
        if not self.reader.cancel(f):
            return False
        del L.fills[e]
        s = L.where.pop(e)
        if f.victim is not None and f.victim not in L.where:
            L.where[f.victim] = s
        else:
            L.free.append(s)
        if f.put is not None:
            self.ram.release(f.put[1])
        if f.got is not None:
            self.ram.put((layer, e), f.got)
        return True

    def _victim(self, L: _Layer, keep: set) -> Optional[int]:
        """The expert in a slot used least (decayed), not in ``keep``, not being read and not guessed for the
        layer's next pass."""
        cands = [e for e in L.where if e not in keep and e not in L.fills and e not in L.fresh]
        if not cands:
            return None
        return cands[int(np.argmin(L.score[np.array(cands)]))]

    def _take(self, layer: int, e: int, keep: set, wait: bool = True, guess: bool = False) -> Optional[int]:
        """A slot for ``e``, its read started: a free one, else the one whose expert was used least and is not in
        ``keep``; with ``wait``, the next one a read under way leaves if every slot is taken. None (without
        ``wait``) if there is none to take. The evicted expert is copied to the RAM cache first, and ``e`` comes
        from there if it is in it."""
        L = self.layers[layer]
        self.fence(layer)
        victim = None
        while True:
            self._settle(L)
            if L.free:
                s = L.free.pop()
                break
            victim = self._victim(L, keep)
            if victim is not None:
                s = L.where.pop(victim)
                break
            if not wait:
                return None
            busy = [f for x, f in L.fills.items() if x not in keep]
            if not busy:
                raise RuntimeError(f"no slot of layer {layer} to read expert {e} into")
            busy[0].done.wait()
        L.where[e] = s
        tasks = [[], [], [], []]  # the gate, up and down weights each a task, the six scales and biases one more
        put = got = None
        if self.ram is not None:
            out = self.ram.region() if victim is not None else None
            if out is not None:
                put, o = ((layer, victim), out), 0
                for j, part in enumerate(_PARTS):
                    size = self.spec[(layer, part)][2]
                    tasks[j // 3 if j % 3 == 0 else 3].append((-1, out + o, L.addr[s][j], size))
                    o += size
            got = self.ram.get((layer, e))
        if got is None and layer in self.pack:  # one read of the expert, scattered into the nine parts of its slot
            tasks = [[item for t in tasks for item in t] + [(self.pack[layer], e * self.expert_bytes(layer), list(L.dst[s]), None)]]
        else:
            o = 0
            for j, part in enumerate(_PARTS):
                fd, start, size = self.spec[(layer, part)][:3]
                if got is not None:
                    item = (-1, L.addr[s][j], got + o, size)
                else:
                    item = (fd, start + e * size, L.dst[s][j], None)
                tasks[j // 3 if j % 3 == 0 else 3].append(item)
                o += size
            tasks = [t for t in tasks if t]
        f = self.reader.submit(layer, tasks, guess)
        f.slot, f.victim, f.put, f.got = s, victim, put, got
        L.fills[e] = f
        return s

    def table(self, layer: int, dtype) -> mx.array:
        """The layer's slot of each expert, -1 for one without a slot or still being read (int32, on the GPU): for
        a pass that looks its experts up itself (Model's optimistic decode)."""
        self._slot_arrays(layer)
        L = self.layers[layer]
        self._settle(L)
        t = np.full(self.experts[layer], -1, dtype=np.int32)
        for e, slot in L.where.items():
            if e not in L.fills:
                t[e] = slot
        return mx.array(t)

    def in_slots(self, layer: int, experts: list, dtype):
        """For a pass needing at most ``slots`` experts: the layer's nine slot arrays with ``experts`` in them,
        and each expert's slot. None for a pass that needs more."""
        got = self.begin(layer, experts, dtype)
        if got is None:
            return None
        arrs, slots, pending = got
        self.finish(layer, pending)
        return arrs, slots

    def begin(self, layer: int, experts: list, dtype):
        """Like in_slots, without waiting: (the nine slot arrays, each expert's slot, the experts whose reads are
        under way). Their slots are usable once ``finish`` returns; the others' at once."""
        if self._sets is not None and not self.buf_of and time.monotonic() - self._sets_used > SETS_IDLE_S:
            self.drop_sets()
        arrs = self._slot_arrays(layer)
        L = self.layers[layer]
        if len(experts) > L.n:  # (the layer's slots as made: ``slots`` may have changed since; reset() remakes them)
            return None
        wanted = set(experts)
        for e, f in list(L.fills.items()):  # guesses this pass's routing did not confirm: dropped if not started
            if e not in wanted:
                self._cancel(layer, L, e, f)
        L.fresh.clear()
        L.unit /= DECAY
        if L.unit > 2.0 ** 64:
            L.score /= L.unit
            L.unit = 1.0
        for e in experts:
            if e in L.where:
                if e in L.fills:
                    self.ahead += 1  # read ahead: its read may still be under way
                    self.reader.want(L.fills[e])
                else:
                    self.hits += 1
            else:
                self._take(layer, e, wanted)
                self.misses += 1
            L.score[e] += L.unit
        self._settle(L)
        return arrs, [L.where[e] for e in experts], [e for e in experts if e in L.fills]

    def finish(self, layer: int, pending: list) -> None:
        """Wait for the reads of ``pending`` (from begin); a failed one raises."""
        L = self.layers[layer]
        self._wait([L.fills[e] for e in pending if e in L.fills])
        failed = self._settle(L)
        bad = [e for e in pending if e in failed]
        if bad:
            raise failed[bad[0]]

    def _wait(self, fills: list) -> None:
        """Wait for ``fills``, the GPU kept busy meanwhile with a tiny product on a stream of its own: an Apple GPU
        left idle for a millisecond runs its next work at a lower clock (on the M2 Pro a layer's work then took
        0.63 ms instead of 0.30), and a decode waiting for an expert's read leaves it idle that long at every
        layer. LOCALS_KEEP_GPU_BUSY=0: plain waits."""
        pending = [f for f in fills if not f.done.is_set()]
        if not pending:
            return
        if not KEEP_GPU_BUSY:
            for f in pending:
                f.done.wait()
            return
        if self._filler is None:
            self._side = mx.new_stream(mx.gpu)
            with mx.stream(self._side):
                self._filler = (mx.zeros((1, 512), dtype=mx.float16), mx.zeros((512, 512), dtype=mx.float16))
                mx.eval(self._filler)
        x, w = self._filler
        with mx.stream(self._side):
            while not all(f.done.is_set() for f in pending):
                mx.eval(x @ w)

    def stacked(self, layer: int, experts: list, dtype) -> list:
        """The nine arrays of ``experts`` (in this order), each stacked along a first axis: for a pass that needs
        more experts than a layer has slots. Those in slots are taken from there; nothing is kept."""
        L = self.layers.get(layer)
        where = {}
        if L is not None:
            for e in experts:
                f = L.fills.get(e)
                if f is not None:
                    f.done.wait()
            self._settle(L)
            where = {e: s for e, s in L.where.items() if e not in L.fills}
        missing = [e for e in experts if e not in where]
        raw = dict(zip(missing, self.pool.map(lambda e: self._raw(layer, e), missing) if len(missing) > 1
                       else [self._raw(layer, e) for e in missing]))
        self.hits += len(experts) - len(raw)
        self.misses += len(raw)
        cols = [[] for _ in _PARTS]
        for e in experts:
            parts = converted([a[where[e]] for a in L.arrays], dtype) if e in where else self._arrays(layer, raw[e], dtype)
            for c, part in zip(cols, parts):
                c.append(part)
        return [mx.stack(c) for c in cols]

    def _layer_set(self) -> list:
        """Two sets of a whole layer's nine expert tensors (every expert, the files' bytes), made once: whole() reads
        a layer into one while the GPU computes the layer before from the other."""
        if self._sets is None:
            layer = sorted(self.experts)[0]
            E = self.experts[layer]
            sets = []
            for _ in range(2):
                arrs, dst = [], []
                for part in _PARTS:
                    size, shape, kind = self.spec[(layer, part)][2:]
                    if kind == "U8":
                        a = mx.zeros((E, shape[0], size // shape[0] // 4), dtype=mx.uint32)
                        mx.eval(a)
                        h = np.asarray(memoryview(a))
                    else:
                        raw = mx.zeros((E, *shape), dtype=mx.uint16)
                        a = raw.view(mx.bfloat16)
                        mx.eval(raw, a)
                        h = np.asarray(memoryview(raw))
                    arrs.append(a)
                    dst.append((memoryview(h).cast("B"), h.ctypes.data))
                sets.append((arrs, dst))
            self._sets = sets
        return self._sets

    def _read_layer(self, layer: int, k: int) -> _Fill:
        """Start reading ``layer``'s whole expert tensors into set ``k``, in 8 MB reads on the reader threads; the
        experts in the layer's slots are copied from there instead (a prompt pass reads every expert, at the SSD's
        3.3 GB/s on the M2 Pro: 6.6 s for the 21.3 GB of a pass)."""
        _, dst = self._layer_set()[k]
        L = self.layers.get(layer)
        have = {e: s for e, s in L.where.items() if e not in L.fills} if L is not None else {}
        E = self.experts[layer]
        tasks, copies = [], []
        chunk = 8 << 20
        for j, part in enumerate(_PARTS):
            fd, start, size = self.spec[(layer, part)][:3]
            mv, base = dst[j]
            e = 0
            while e < E:
                if e in have:
                    copies.append((-1, base + e * size, L.addr[have[e]][j], size))
                    e += 1
                    continue
                f = e
                while f < E and f not in have:
                    f += 1
                if layer not in self.pack:
                    for o in range(e * size, f * size, chunk):  # experts e to f - 1, read
                        tasks.append([(fd, start + o, mv[o:o + min(chunk, f * size - o)], None)])
                e = f
        if layer in self.pack:  # runs of 4 experts, each read whole and scattered into the nine tensors
            per = self.expert_bytes(layer)
            sizes = [self.spec[(layer, part)][2] for part in _PARTS]
            todo = [e for e in range(E) if e not in have]
            i = 0
            while i < len(todo):
                j = i
                while j < len(todo) and j - i < 4 and todo[j] == todo[i] + (j - i):
                    j += 1
                bufs = [dst[q][0][e * sizes[q]:(e + 1) * sizes[q]] for e in todo[i:j] for q in range(len(_PARTS))]
                tasks.append([(self.pack[layer], todo[i] * per, bufs, None)])
                i = j
        for i in range(0, len(copies), 16):  # (the copies in tasks of 16)
            tasks.append(copies[i:i + 16])
        return self.reader.submit(layer, tasks)

    def whole(self, layer: int, dtype) -> list:
        """The layer's nine stacked expert tensors (scales and biases in ``dtype``), for a pass that needs most of the
        experts (reading a prompt): read into one of two reused sets while the GPU computes the layer before from the
        other, and the next MoE layer's read started now. Direct reads into the set's memory: no buffer made per
        layer and no copy through macOS's file cache (MLX's own load of the files took 7 s a pass over a 1,024-token
        prompt)."""
        sets = self._layer_set()
        self._sets_used = time.monotonic()
        got = self.buf_of.pop(layer, None)
        if got is None:  # (not read ahead: a pass that starts here)
            k = 0 if all(v[0] != 0 for v in self.buf_of.values()) else 1
            self.buf_fence(k)
            got = (k, self._read_layer(layer, k))
        k, fill = got
        layers = sorted(self.experts)
        i = layers.index(layer)
        if i + 1 < len(layers) and layers[i + 1] not in self.buf_of:
            self.buf_fence(1 - k)  # (the layer before's results from that set: computed before this layer's routing)
            self.buf_of[layers[i + 1]] = (1 - k, self._read_layer(layers[i + 1], 1 - k))
        self._wait([fill])
        fill.wait()
        self._used_set = k
        return converted(sets[k][0], dtype)

    def drop_sets(self) -> None:
        """Give the layer sets' memory back (0.84 GB for Flint) once decoding has gone on without a prompt pass:
        they are made again for the next one."""
        for k in range(2):
            self.buf_fence(k)
        self._sets = None
        mx.clear_cache()

    def buf_used(self, result: mx.array) -> None:
        """``result`` was computed from the set whole() last returned."""
        self.buf_uses.setdefault(self._used_set, []).append(result)

    def buf_fence(self, k: int) -> None:
        uses = self.buf_uses.pop(k, None)
        if uses:
            mx.eval(uses)

    def revise(self, layer: int, experts) -> None:
        """A better guess of ``layer``'s experts: the guessed reads for it not started and not in ``experts`` are
        dropped (their slots back as they were), and ``experts`` read ahead."""
        L = self.layers.get(layer)
        if L is None:
            return
        keep = set(experts)
        for e, f in list(L.fills.items()):
            if f.guess and e not in keep:
                self._cancel(layer, L, e, f)
                L.fresh.discard(e)
        self.prefetch(layer, experts)

    def prefetch(self, layer: int, experts) -> None:
        """Start reading ``experts`` of ``layer`` into slots unless they are in one (or on their way); they are
        kept until the layer's next pass."""
        L = self.layers.get(layer)
        if L is None:
            return
        keep = set(experts)
        for e in experts:
            if e not in L.where:
                if self._take(layer, e, keep | L.fresh, wait=False, guess=True) is None:
                    break  # no slot to spare: the rest are read when needed
            L.fresh.add(e)


def _swiglu_fp16(g: mx.array, u: mx.array, down) -> mx.array:
    """down(swiglu(g, u)) in float16 without overflow. Flint's layer 46 makes the residual stream's massive
    activation (about 24,000) through one expert whose SwiGLU product reaches about 68,000, past float16's 65,504:
    the only value of the model outside its range (every other one stays under 24,000, where float16 also rounds
    to 16 against bfloat16's 128). The product is taken in float32, and a row above 2^9 is divided by the power of
    2 that brings it under (exactly), the down projection's output multiplied back: MLX's quantized products add a
    group's 64 inputs in float16 (for the biases), which overflowed for inputs of 2^15 in its gather kernel."""
    h = swiglu(g.astype(mx.float32), u.astype(mx.float32))
    m = mx.max(mx.abs(h), axis=-1, keepdims=True)
    scale = mx.power(2.0, mx.maximum(mx.ceil(mx.log2(mx.maximum(m, 1.0) / 512.0)), 0.0))
    return (down((h / scale).astype(mx.float16)).astype(mx.float32) * scale).astype(mx.float16)


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
        uses_slots = whole = False
        if len(experts) > WHOLE:  # most of the layer's experts (reading a prompt): the layer's tensors, read whole
            gw, gs, gb, uw, us, ub, dw, ds, db = self.store.whole(self.layer, x.dtype)
            whole = True
            idx = inds.astype(mx.uint32)
        else:
            got = self.store.in_slots(self.layer, experts.tolist(), x.dtype)
            if got is not None:  # decoding: the slots, indexed by each token's experts' slots
                (gw, gs, gb, uw, us, ub, dw, ds, db), slots = converted(got[0], x.dtype), got[1]
                idx = mx.array(np.asarray(slots, dtype=np.uint32)[inv].reshape(inds.shape))
                uses_slots = True
            else:  # more experts than a layer has slots (many requests at once): stacked for this pass
                gw, gs, gb, uw, us, ub, dw, ds, db = self.store.stacked(self.layer, experts.tolist(), x.dtype)
                idx = mx.array(inv.reshape(inds.shape).astype(np.uint32))
                uses_slots = True
        x = mx.expand_dims(x, (-2, -3))
        do_sort = idx.size >= 64
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, idx)

        def qmm(a, w, s, b):
            return mx.gather_qmm(a, w, s, b, rhs_indices=idx, transpose=True, group_size=64, bits=4,
                                 sorted_indices=do_sort)

        g, u = qmm(x, gw, gs, gb), qmm(x, uw, us, ub)
        if x.dtype == mx.float16:  # (see _swiglu_fp16)
            y = _swiglu_fp16(g, u, lambda h: qmm(h, dw, ds, db))
        else:
            y = qmm(swiglu(g, u), dw, ds, db)
        if do_sort:
            y = _scatter_unsort(y, inv_order, inds.shape)
        if uses_slots:
            self.store.used(self.layer, y)
        if whole:
            self.store.buf_used(y)
        return y.squeeze(-2)


class SparseMoe(nn.Module):
    def __init__(self, args: ModelArgs, layer: int):
        super().__init__()
        self.k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.experts = StreamedExperts(layer)
        self._next = ()  # (router, index, guesses) of the next MoE layers, to guess their experts (Model.sanitize)
        self._refine = None  # (norm, router, index) of the next MoE layer, to guess again from a partial result

    def _ahead(self, rows: int) -> list:
        """(router, layer, guesses per token) of the layers read ahead by a pass of ``rows`` tokens: the guesses
        are shared out among the pass's tokens (each token's best guesses); a pass of 8 tokens with 5 + 4 + 3 + 2
        guesses each read 2.1 GB, of which 0.54 were experts it then used."""
        if rows > 16:
            return []
        ahead = [(gate, layer, rank // rows if n else max(1, rank // rows)) for n, (gate, layer, rank) in enumerate(self._next)]
        return [a for a in ahead if a[2] > 0]

    def _fusable(self, x: mx.array, rows: int, ahead: list) -> bool:
        if not FUSED or rows > 16 or x.dtype not in (mx.float16, mx.bfloat16) or x.shape[-1] % 512:
            return False
        se = self.shared_expert
        mods = [se.gate_proj, se.up_proj, se.down_proj]
        E = self.gate.weight.shape[0]
        return (len(ahead) <= 4 and all(r <= 8 for *_, r in ahead) and self.k <= 8 and E % 32 == 0 and E <= 1024
                and all(isinstance(m, nn.QuantizedLinear) and m.bits == 4 and m.group_size == 64 and "bias" not in m
                        and m.scales.dtype == x.dtype for m in mods)
                and se.gate_proj.weight.shape[0] % 512 == 0
                and all(g.weight.dtype == x.dtype for g in [self.gate] + [a[0] for a in ahead]))

    def __call__(self, x: mx.array, residual: Optional[mx.array] = None) -> mx.array:
        rows = x.size // x.shape[-1]
        ahead = self._ahead(rows)
        if residual is not None and self._fusable(x, rows, ahead):
            return self._fused(x, residual, rows, ahead)
        # (on float32 logits, as the decode router kernel selects: bf16 logits tie often, and which way a tie went
        # changed with MLX's rounding, 0.32.3 to 0.32.4)
        g = self.gate
        logits = g(x).astype(mx.float32) if "scales" in g else x.astype(mx.float32) @ g.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        inds = mx.argpartition(-scores, kth=self.k - 1, axis=-1)[..., : self.k]
        if ahead:  # decoding: the next layers' experts, guessed from this input, read ahead
            guesses = [mx.argpartition(-gate(x), kth=rank - 1, axis=-1)[..., :rank] for gate, _, rank in ahead]
            mx.eval(inds, *guesses)
            for (_, layer, _), guess in zip(ahead, guesses):
                self.experts.store.prefetch(layer, np.unique(np.array(guess)).tolist())
        else:
            mx.eval(inds)
        w = mx.take_along_axis(scores, inds, axis=-1)
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
        y = self.experts(x, inds)
        y = (y * w[..., None].astype(y.dtype)).sum(axis=-2)
        y = y + self.shared_expert(x)
        return y if residual is None else residual + y

    def _route(self, x2: mx.array, ahead: list, table=None, okin=None):
        """The layer's router and the read-ahead routers in one product, the selections in one more (flint_kernels):
        (experts, weights, guesses, slots, ok); slots and ok are meaningful with ``table``."""
        from . import flint_kernels as fk

        logits = fk.routers(x2, [self.gate.weight] + [g.weight for g, _, _ in ahead])
        return fk.route(logits, self.k, [r for *_, r in ahead], table, okin)

    def _read_ahead(self, guesses: np.ndarray, ahead: list, after: int = -1) -> None:
        """Start reading the guessed experts of the layers in ``ahead`` after layer ``after``."""
        o = 0
        for _, nxt, r in ahead:
            if nxt > after:
                self.experts.store.prefetch(nxt, np.unique(guesses[:, o:o + r]).tolist())
            o += r

    def _fused(self, x: mx.array, residual: mx.array, rows: int, ahead: list) -> mx.array:
        """A decode pass's MoE block in Locals' kernels: routing (``_route``), then the experts (``_compute``)."""
        D = x.shape[-1]
        inds, w, guesses, _, _ = self._route(x.reshape(rows, D), ahead)
        mx.eval(inds, guesses)  # the layer's one wait for the GPU: which experts, before they are read
        if ahead:
            self._read_ahead(np.array(guesses), ahead)
        return self._compute(x, residual, rows, inds, w)

    def _compute(self, x: mx.array, residual: mx.array, rows: int, inds: mx.array, w: mx.array) -> mx.array:
        """residual + the routed experts ``inds`` (evaluated), weighted by ``w``, + the shared expert: the experts
        in slots and the shared one computed while the others are read, the others added after."""
        from . import flint_kernels as fk

        shape, D = x.shape, x.shape[-1]
        x2, r2 = x.reshape(rows, D), residual.reshape(rows, D)
        store = self.experts.store
        layer = self.experts.layer
        experts, inv = np.unique(np.array(inds).reshape(-1), return_inverse=True)
        got = store.begin(layer, experts.tolist(), x.dtype)
        if got is None:  # more experts than slots (many requests at once): stacked, MLX's kernels
            y = self.experts(x, inds.reshape(*shape[:-1], self.k))
            y = (y * w.reshape(*shape[:-1], self.k)[..., None].astype(y.dtype)).sum(axis=-2)
            return residual + y + self.shared_expert(x)
        arrs, slots, pending = got
        store.passes += 1
        store.ready += not pending
        idx = mx.array(np.asarray(slots, dtype=np.uint32)[inv].reshape(rows, self.k))
        shared = self._shared()
        if not pending:
            y = fk.experts(x2, r2, arrs, shared, idx, _ones(rows, self.k), w)
            store.used(layer, y)
            return y.reshape(shape)
        late = np.isin(experts, pending)[inv].reshape(rows, self.k)
        part, contrib = fk.experts(x2, r2, arrs, shared, idx, mx.array((~late).astype(np.uint32)), w, out_dtype=mx.float32,
                                   keep=True)
        store.used(layer, part)
        mx.async_eval(part)  # the experts in slots and the shared one, while the others are read
        refine = self._refine
        if refine is not None and REFINE and rows <= 16:
            # The residual with the shared expert and the experts in slots is the next layer's input but for the
            # missing experts and the next layer's attention: through its norm and router it names the next
            # layer's experts far better than this layer's input does (on Flint, the top 6 held 86% of them
            # against 78%; by rank 100, 99, 96, 90, 78% against 99, 96, 89, 78, 62%). Computed while this layer
            # waits for its reads, it revises the next layer's reads ahead.
            norm, gate, nxt = refine
            g = mx.argpartition(-gate(norm(part.astype(x.dtype))).astype(mx.float32), kth=REFINE - 1, axis=-1)[..., :REFINE]
            mx.eval(g)
            store.revise(nxt, np.unique(np.array(g)).tolist())
        store.finish(layer, pending)
        # the whole sum again from the residual, the experts in slots and the shared one taken from the first call:
        # in the same order as when every expert was in a slot, so the result does not depend on when reads end
        y = fk.experts(x2, r2, arrs, shared, idx, mx.array(np.where(late, 1, 2).astype(np.uint32)), w, out_dtype=x.dtype,
                       prev=contrib, shared_from_prev=True)
        store.used(layer, y)
        return y.reshape(shape)

    def _shared(self) -> list:
        se = self.shared_expert
        return [a for m in (se.gate_proj, se.up_proj, se.down_proj) for a in (m.weight, m.scales, m.biases)]

    def _optimistic(self, x: mx.array, residual: mx.array, rows: int, ahead: list, okin):
        """The block without waiting for the host: the experts are looked up in the layer's slot table on the GPU
        and computed if every one has a slot (and okin, the layer before's flags, is set), else the result is
        the residual and the flags are clear. Returns (result, flags, (experts, weights, guesses))."""
        from . import flint_kernels as fk

        shape, D = x.shape, x.shape[-1]
        store = self.experts.store
        layer = self.experts.layer
        table = store.table(layer, x.dtype)
        inds, w, guesses, slots, ok = self._route(x.reshape(rows, D), ahead, table, okin)
        y = fk.experts(x.reshape(rows, D), residual.reshape(rows, D), store.arrays[layer], self._shared(), slots,
                       _ones(rows, self.k), w, ok=ok)
        store.used(layer, y)
        return y.reshape(shape), ok, (inds, w, guesses)


_ONES = {}


def _ones(rows: int, k: int) -> mx.array:
    a = _ONES.get((rows, k))
    if a is None:
        a = _ONES[(rows, k)] = mx.ones((rows, k), dtype=mx.uint32)
        mx.eval(a)
    return a


def _snapshot(c) -> dict:
    """What a decode step changes in a KV cache (KVCache, RotatingKVCache and the batch ones): written back, the
    step is undone. A rotating cache's step overwrites the entry leaving its window, which no later step reads."""
    return {k: getattr(c, k) for k in ("keys", "values", "offset", "_idx", "_offset") if hasattr(c, k)}


def _restore(c, snap: dict) -> None:
    for k, v in snap.items():
        setattr(c, k, v)


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
        if isinstance(self.mlp, SparseMoe):  # (it adds its output to h itself: in its last kernel when decoding)
            return self.mlp(self.post_attention_layernorm(h), residual=h)
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
        self._window = 1  # MoE layers an optimistic decode step evaluates at once (adapted as it goes)
        self._fusable_for = {}
        # Optimistic steps save the host's turn at every layer whose experts are all in slots (31 -> 20 ms a token
        # on the M2 Pro with every expert in one), but read guessed experts later: where experts are read most
        # layers, they were slower (156 -> 166 ms with 55 slots, 119 -> 134 with 96). Decode steps take them once
        # 90% of the MoE blocks found every expert in a slot, until fewer than 80% do.
        self._ready = 0.0  # running share of decode passes' MoE blocks that found every expert in a slot
        self._windows = False

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        full_mask = create_attention_mask(h, cache[self.fa_idx])
        swa_mask = create_attention_mask(h, cache[self.swa_idx], window_size=self.args.sliding_window)
        masks = [full_mask if kind == "full_attention" else swa_mask for kind in self.layer_types]
        if OPTIMISTIC != "0" and h.shape[1] == 1 and cache[0] is not None and self._optimizable(h):
            store = self._store()
            p0, r0 = store.passes, store.ready
            # (two thresholds: no flapping between the two ways)
            if OPTIMISTIC == "always" or self._ready >= (0.8 if self._windows else 0.9):
                self._windows = True
                h = self._optimistic(h, cache, masks)
            else:
                self._windows = False
                for layer, c, m in zip(self.layers, cache, masks):
                    h = layer(h, m, c)
            if store.passes > p0:
                self._ready = 0.9 * self._ready + 0.1 * (store.ready - r0) / (store.passes - p0)
            return self.norm(h)
        for layer, c, m in zip(self.layers, cache, masks):
            h = layer(h, m, c)
        return self.norm(h)

    def _store(self):
        return next(l.mlp.experts.store for l in self.layers if isinstance(l.mlp, SparseMoe))

    def _optimizable(self, h: mx.array) -> bool:
        key = (h.shape[0], h.dtype)
        ok = self._fusable_for.get(key)
        if ok is None:
            ok = self._fusable_for[key] = all(
                l.mlp._fusable(h, h.shape[0], l.mlp._ahead(h.shape[0])) for l in self.layers if isinstance(l.mlp, SparseMoe))
        return ok

    def _optimistic(self, h: mx.array, cache: list, masks: list) -> mx.array:
        """A decode step that asks the host only when an expert is missing. Layers run in windows evaluated
        at once: each MoE block looks its experts up on the GPU (SparseMoe._optimistic) and clears a flag when
        one has no slot, and the blocks after it compute nothing. The host then reads the flags: the layers up
        to the first clear one are done; that layer's routing is valid (its input was), so its missing experts
        are read and its block computed as usual; the caches of the layers after it are restored and the step
        resumes there. The window grows while layers find their experts and shrinks when one waits."""
        n = len(self.layers)
        i = 0
        rows = h.shape[0]
        while i < n:
            if not isinstance(self.layers[i].mlp, SparseMoe):
                h = self.layers[i](h, masks[i], cache[i])
                i += 1
                continue
            win, okin, j = [], None, i
            while j < n and len(win) < self._window and isinstance(self.layers[j].mlp, SparseMoe):
                lay, c = self.layers[j], cache[j]
                snap = _snapshot(c)
                ha = h + lay.self_attn(lay.input_layernorm(h), masks[j], c)
                xn = lay.post_attention_layernorm(ha)
                ahead = lay.mlp._ahead(rows)
                h, okin, rec = lay.mlp._optimistic(xn, ha, rows, ahead, okin)
                win.append((j, snap, ha, xn, okin, rec, ahead))
                j += 1
            mx.eval(h, *[a for _, _, _, _, ok, rec, _ in win for a in (ok, rec[0], rec[2])])
            bad = next((k for k, wd in enumerate(win) if not np.array(wd[4]).all()), None)
            done = win if bad is None else win[:bad]
            store = self.layers[i].mlp.experts.store
            for j_, _, _, xn, _, rec, _ in done:  # their experts were in slots: the store counts the uses
                store.begin(j_, np.unique(np.array(rec[0])).tolist(), h.dtype)
            store.passes += len(done)
            store.ready += len(done)
            last = done[-1][0] if done else i - 1
            src = win if bad is None else win[:bad + 1]
            latest = {}  # each later layer's guesses from the closest layer before it
            for j_, _, _, _, _, rec, ahead in src:
                o = 0
                for _, nxt, r in ahead:
                    latest[nxt] = (rec[2], o, r)
                    o += r
            for nxt in sorted(latest):
                if nxt > (win[bad][0] if bad is not None else last):
                    g, o, r = latest[nxt]
                    store.prefetch(nxt, np.unique(np.array(g)[:, o:o + r]).tolist())
            if bad is None:
                self._window = min(n, 2 * self._window)
                i = j
                continue
            jb, _, ha, xn, _, rec, _ = win[bad]
            for jj, snap, *_ in win[bad + 1:]:
                _restore(cache[jj], snap)
            h = self.layers[jb].mlp._compute(xn, ha, rows, rec[0], rec[1])
            self._window = max(1, bad + 1)
            i = jb + 1
        return h


class Model(nn.Module):
    # In float16 its values stay in range (_swiglu_fp16) and it is closer to a float32 run than in bfloat16 (over
    # 1,800 positions on the M2 Pro: logit error median 0.77 against 2.0, float32's top token at 90.1% of them
    # against 88.2%); and on GPUs without bfloat16 arithmetic (M1, M2) a 4,096-token prompt pass took 8.0 s
    # against 10.1. Locals' runtime then takes float16 without its probe.
    float16_safe = True

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
        from ..defaults import expert_cache_gb, expert_ram_gb

        store = ExpertStore(MODEL_PATHS[-1], int(expert_cache_gb() * 2**30), ram_bytes=int(expert_ram_gb() * 2**30))
        moe = [i for i, layer in enumerate(self.model.layers) if isinstance(layer.mlp, SparseMoe)]
        for n, i in enumerate(moe):
            self.model.layers[i].mlp.experts.store = store
            # the MoE layers after this one (a plain tuple attribute: not a child module, not a parameter path)
            ahead = tuple((self.model.layers[j].mlp.gate, j, min(rank, self.args.num_experts))
                          for j, rank in zip(moe[n + 1:], GUESSES) if rank > 0)
            object.__setattr__(self.model.layers[i].mlp, "_next", ahead)
            if n + 1 < len(moe):
                j = moe[n + 1]
                object.__setattr__(self.model.layers[i].mlp, "_refine",
                                   (self.model.layers[j].post_attention_layernorm, self.model.layers[j].mlp.gate, j))
        self.expert_store = store
        return out

    def make_cache(self) -> List[Any]:
        return [KVCache() if kind == "full_attention" else RotatingKVCache(max_size=self.args.sliding_window)
                for kind in self.model.layer_types]
