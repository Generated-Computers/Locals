"""Metal kernels for the small-M regime on Apple GPUs.

MLX's quantized matrix-vector kernel runs one input row per threadgroup, so M
rows stream the weights M times; from M >= 6 (on M2 Pro, for large matrices)
it switches to a 32-row tiled kernel. Both are wasteful for small M, the
regime of forced spans and small decode batches. ``qmm_small`` reads each
weight once and dequantizes it once, then applies it to all M rows. Measured
on M2 Pro it beats MLX for 3 <= M <= 6 on large matrices and loses at M = 8
(a scalar-FMA kernel cannot match MLX's MMA tiles there), so it is only used
in that range.

Weight format: MLX affine 4-bit, group size 64. ``w`` is (N, K/8) uint32 with
the low nibble first; ``scales`` and ``biases`` are (N, K/64).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# v4: dequantize each weight once, reuse it for all M rows. Lane owns 16
# consecutive K values per 512-block, processed as 4 sub-chunks of 4 so that
# only R float4 weights and M float4 inputs are live at a time.
_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int out_row = (int(threadgroup_position_in_grid.x) * NSG + int(sg)) * R;
    if (out_row >= N) {
        return;
    }
    constexpr int Kw = K / 8;
    constexpr int Kg = K / GS;
    const device uint32_t* wp = w + out_row * Kw + lane * 2;
    const device T* sc = scales + out_row * Kg + lane / (GS / 16);
    const device T* bi = biases + out_row * Kg + lane / (GS / 16);

    float acc[M][R];
    for (int m = 0; m < M; m++) {
        for (int r = 0; r < R; r++) {
            acc[m][r] = 0.0f;
        }
    }

    for (int k = 0; k < K; k += 512) {
        const int kk = k + int(lane) * 16;
        if (kk < K) {
            uint32_t wv[R][2];
            float s[R];
            float b[R];
            for (int r = 0; r < R; r++) {
                wv[r][0] = wp[r * Kw];
                wv[r][1] = wp[r * Kw + 1];
                s[r] = float(sc[r * Kg]);
                b[r] = float(bi[r * Kg]);
            }
            float sx[M];
            for (int m = 0; m < M; m++) {
                sx[m] = 0.0f;
            }
            for (int j = 0; j < 4; j++) {
                float4 q[R];
                for (int r = 0; r < R; r++) {
                    uint32_t word = wv[r][j >> 1] >> ((j & 1) * 16);
                    q[r] = float4(word & 0xf, (word >> 4) & 0xf, (word >> 8) & 0xf, (word >> 12) & 0xf) * s[r];
                }
                for (int m = 0; m < M; m++) {
                    const device T* xm = x + m * K + kk + 4 * j;
                    float4 xv = float4(xm[0], xm[1], xm[2], xm[3]);
                    sx[m] += xv.x + xv.y + xv.z + xv.w;
                    for (int r = 0; r < R; r++) {
                        acc[m][r] += dot(q[r], xv);
                    }
                }
            }
            for (int m = 0; m < M; m++) {
                for (int r = 0; r < R; r++) {
                    acc[m][r] += b[r] * sx[m];
                }
            }
        }
        wp += 64;
        sc += 512 / GS;
        bi += 512 / GS;
    }

    for (int m = 0; m < M; m++) {
        for (int r = 0; r < R; r++) {
            float v = simd_sum(acc[m][r]);
            if (lane == 0) {
                y[m * N + out_row + r] = static_cast<T>(v);
            }
        }
    }
"""

_KERNEL = mx.fast.metal_kernel(
    name="locals_qmm_small",
    input_names=["x", "w", "scales", "biases"],
    output_names=["y"],
    source=_SOURCE,
)


def qmm_small(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, group_size: int = 64, R: int = 4, NSG: int = 2) -> mx.array:
    M, K = x.shape
    N = w.shape[0]
    if N % R:
        R = 1
    n_tg = (N + NSG * R - 1) // (NSG * R)
    (y,) = _KERNEL(
        inputs=[x, w, scales, biases],
        template=[("T", x.dtype), ("M", M), ("K", K), ("N", N), ("GS", group_size), ("R", R), ("NSG", NSG)],
        grid=(n_tg * 32 * NSG, 1, 1),
        threadgroup=(32 * NSG, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )
    return y


# --- autotuned dispatch ------------------------------------------------------
#
# For each (N, K, M) the first call benchmarks MLX's kernel, qmm_small, and a
# few qmm_mma tilings, and caches the fastest choice (also on disk). M ranges
# over 2..16; larger M stays on MLX's 32-row tiles.

import json as _json
import os as _os
import time as _time

MIN_M = 1
MAX_M = 32
_TUNE_FILE = "tune_v6_%s.json" % mx.device_info()["architecture"]
_TUNE_PATH = _os.environ.get("LOCALS_TUNE_CACHE") or _os.path.expanduser("~/.cache/locals/" + _TUNE_FILE)
_TUNE = None
_MULTI_CANDIDATES = [(8, 4, 2), (16, 4, 2), (8, 2, 2)]  # (VPT, R, NSG), M <= 4
_V3_CANDIDATES = [(8, 8, 2), (8, 4, 2), (16, 4, 2), (16, 4, 4), (16, 2, 2)]  # (VPT, R, NSG), 3 <= M <= 6
_FRAG_CANDIDATES = [(2, 2, 1), (4, 2, 1), (2, 4, 1), (1, 2, 2)]  # (TN, NSG, SPLIT)
_MMA_CANDIDATES = {
    8: [(8, 32, 64, 1, 2, 1), (8, 32, 64, 1, 4, 1), (8, 64, 64, 1, 8, 1), (8, 32, 32, 1, 2, 1), (8, 32, 64, 1, 2, 2)],
    16: [(16, 32, 64, 2, 2, 1), (16, 64, 64, 2, 2, 1), (16, 32, 64, 1, 4, 1), (16, 64, 64, 1, 8, 1)],
}


def _preset_path():
    """Tuning results shipped with the package for this GPU architecture, if any
    (``LOCALS_NO_PRESET=1`` hides them: a first start as on a GPU without them)."""
    if _os.environ.get("LOCALS_NO_PRESET"):
        return None
    try:
        from importlib.resources import files

        p = files("locals").joinpath("presets", _TUNE_FILE)
        return str(p) if p.is_file() else None
    except Exception:
        return None


import threading as _threading

_TUNE_LOCK = _threading.RLock()  # tuning and the cache file: one thread at a time (the server has two engines)


def _read_tune(path):
    with open(path) as f:
        return {k: (v if k.startswith("__") else tuple(v)) for k, v in _json.load(f).items()}


def _load_tune():
    """The tuning choices: the shipped preset for this GPU, overridden by the user's own cache. A
    cache file that does not parse (a write cut short) is set aside, not silently replaced."""
    global _TUNE
    with _TUNE_LOCK:
        if _TUNE is None:
            tune = {}
            preset = _preset_path()
            if preset:
                try:
                    tune.update(_read_tune(preset))
                except Exception:  # noqa: BLE001
                    pass
            user = None
            if _os.path.exists(_TUNE_PATH):
                try:
                    user = _read_tune(_TUNE_PATH)
                except Exception:  # noqa: BLE001
                    try:
                        _os.replace(_TUNE_PATH, _TUNE_PATH + ".bad")
                    except OSError:
                        pass
            tune.update(user or {})
            _TUNE = tune
            crashed = _crashed_shapes(tune)
            if (user is None and tune) or crashed:
                try:
                    _save_tune()
                except OSError:
                    pass
    return _TUNE


def _marker(pid: int) -> str:
    return f"{_TUNE_PATH}.tuning-{pid}"


def _alive(pid: int) -> bool:
    try:
        _os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # (another user's process)
        return True
    return True


def _crashed_shapes(tune: dict) -> bool:
    """Shapes a dead process was timing (its marker file is left: a Metal error MLX raises in a command buffer's
    completion handler aborts the process, and no except catches it). Each counts in ``__crashed__``; a shape
    that stopped a process twice keeps MLX's kernel, where it would be tuned again, and abort the server again,
    at every start. One death (a kill, a power cut) only tunes the shape again. Whether anything changed."""
    import glob
    import sys

    changed = False
    for m in glob.glob(glob.escape(_TUNE_PATH) + ".tuning-*"):
        try:
            pid = int(m.rsplit("-", 1)[1])
            with open(m) as f:
                key = f.read().strip()
        except (ValueError, OSError):
            continue
        if pid == _os.getpid() or _alive(pid):
            continue
        try:
            _os.unlink(m)
        except OSError:
            continue  # (another process took it)
        if not key:
            continue
        crashed = tune.setdefault("__crashed__", {})
        crashed[key] = crashed.get(key, 0) + 1
        changed = True
        if crashed[key] >= 2 and not isinstance(tune.get(key), tuple):
            tune[key] = ("mlx",)
            print(f"locals: timing the tuned kernels for the shape {key} stopped locals twice; that shape keeps MLX's "
                  f"kernel", file=sys.stderr, flush=True)
    return changed


def _merge_tune(mine: dict, theirs: dict) -> dict:
    """Two tuning caches as one: ``mine`` (this process's) wins a key both have; the timings (``__times__``) and
    the lists of whole-step-tuned row counts are merged inside."""
    out = dict(theirs)
    for k, v in mine.items():
        w = out.get(k)
        if isinstance(v, dict) and isinstance(w, dict):
            out[k] = {**w, **v}
        elif k.startswith("__") and isinstance(v, list) and isinstance(w, list):
            out[k] = list(dict.fromkeys([*w, *v]))
        else:
            out[k] = v
    return out


def _save_tune():
    """The cache, written atomically (a crash never leaves half a file) and merged with the file as it is now
    under a lock: another process tuning at the same time (`locals tune` next to a server tuning while idle)
    rewrote the whole file from its own copy, so the last writer erased the other's results. This process
    takes the other's keys too."""
    import fcntl
    import tempfile

    with _TUNE_LOCK:
        d = _os.path.dirname(_TUNE_PATH)
        _os.makedirs(d, exist_ok=True)
        with open(_TUNE_PATH + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                theirs = _read_tune(_TUNE_PATH) if _os.path.exists(_TUNE_PATH) else {}
            except Exception:  # noqa: BLE001  (a file that does not parse: ours replaces it)
                theirs = {}
            data = _merge_tune(_TUNE, theirs)
            _TUNE.update({k: v for k, v in data.items() if k not in _TUNE or isinstance(v, (dict, list))})
            fd, tmp = tempfile.mkstemp(dir=d, prefix=".tune-", suffix=".json")
            try:
                with _os.fdopen(fd, "w") as f:
                    _json.dump(data, f)
                    f.flush()
                    _os.fsync(f.fileno())  # (renamed after a power cut, the file could come back empty: tuning lost)
                _os.replace(tmp, _TUNE_PATH)
            except BaseException:
                try:
                    _os.unlink(tmp)
                except OSError:
                    pass
                raise


def _time_us(fn, reps=12):
    for _ in range(3):
        mx.eval(fn())
    mx.synchronize()
    t0 = _time.perf_counter()
    outs = [fn() for _ in range(reps)]
    mx.eval(outs)
    mx.synchronize()
    return (_time.perf_counter() - t0) / reps * 1e6


_LAUNCH = {}


def _launcher(choice, M, K, N, dtype):
    """A cached closure with the kernel's template and grid pre-computed."""
    key = (choice, M, K, N, dtype)
    f = _LAUNCH.get(key)
    if f is not None:
        return f
    kind = choice[0]
    if kind in ("frag", "fragh"):
        from .qmv_frag import launch, launch_h

        f = (launch if kind == "frag" else launch_h)(M, K, N, dtype, *choice[1:])
    elif kind == "v3":
        from .qmv_multi import launch_v3

        f = launch_v3(M, K, N, dtype, *choice[1:])
    elif kind == "multi":
        from .qmv_multi import _kernel as _mk, launch_params

        template, grid, tg = launch_params(M, K, N, dtype, *choice[1:])
        kern = _mk()

        def f(x2, w, sc, bi):
            (y,) = kern(inputs=[x2, w, sc, bi], template=template, grid=grid, threadgroup=tg,
                        output_shapes=[(M, N)], output_dtypes=[dtype])
            return y
    elif kind == "mma":
        from .mma_kernel import _kernel

        BM, BN, BK, WM, WN, split = choice[1:]
        out_dtype = dtype if split == 1 else mx.float32
        kern = _kernel()
        template = [("T", dtype), ("U", out_dtype), ("BM", BM), ("BN", BN), ("BK", BK), ("WM", WM), ("WN", WN),
                    ("K", K), ("N", N), ("M", M), ("SPLIT", split)]
        grid = (N // BN * 32 * WM * WN, (M + BM - 1) // BM, split)
        tg = (32 * WM * WN, 1, 1)
        shapes = [(split * M, N)]

        def f(x2, w, sc, bi):
            (y,) = kern(inputs=[x2, w, sc, bi], template=template, grid=grid, threadgroup=tg,
                        output_shapes=shapes, output_dtypes=[out_dtype])
            if split > 1:
                y = y.reshape(split, M, N).sum(axis=0).astype(dtype)
            return y
    elif kind == "small":
        R, NSG = small_config(N)

        def f(x2, w, sc, bi):
            return qmm_small(x2, w, sc, bi, 64, R=R, NSG=NSG)
    else:
        def f(x2, w, sc, bi):
            return mx.quantized_matmul(x2, w, sc, bi, transpose=True, group_size=64, bits=4)
    _LAUNCH[key] = f
    return f


def _run(choice, x2, mod):
    M, K = x2.shape
    return _launcher(choice, M, K, mod["weight"].shape[0], x2.dtype)(x2, mod["weight"], mod["scales"], mod["biases"])


_CHOICE = {}


# Background tuning: a shape not tuned yet runs on MLX's own kernel instead of being tuned in
# the middle of a request; the server tunes the pending shapes one by one while it has no work
# (locals/service.py), and each takes its tuned kernel from then on.
_DEFER = False
_TUNING = False  # set while the background tuner itself chooses


def _choose(x2, mod):
    M, K = x2.shape
    N = mod["weight"].shape[0]
    dt = str(x2.dtype).split(".")[-1]
    c = _CHOICE.get((N, K, M, dt))
    if c is not None:
        return c
    key = f"{N}x{K}x{M}" + ("" if dt == "float16" else f"x{dt}")
    tune = _load_tune()
    if key in tune:
        c = tuple(tune[key])
        if dt != "float16" and c[0] == "fragh":  # computes in fp16: never for bf16 activations
            c = ("mlx",)
        _CHOICE[(N, K, M, dt)] = c
        return c
    if _DEFER and not _TUNING:
        return ("mlx",)
    with _TUNE_LOCK:
        c = _CHOICE.get((N, K, M, dt))
        return c if c is not None else _tune_shape(x2, mod, M, K, N, dt, key, tune)


def _tune_shape(x2, mod, M, K, N, dt, key, tune):
    """_time_shape, with the shape named in this process's marker file while it runs (_crashed_shapes)."""
    marker = _marker(_os.getpid())
    try:
        _os.makedirs(_os.path.dirname(marker), exist_ok=True)
        with open(marker, "w") as f:
            f.write(key)
    except OSError:
        marker = None
    try:
        return _time_shape(x2, mod, M, K, N, dt, key, tune)
    finally:
        if marker:
            try:
                _os.unlink(marker)
            except OSError:
                pass


def _time_shape(x2, mod, M, K, N, dt, key, tune):
    from . import qmv_frag, qmv_multi

    cands = [("mlx",)]
    if M <= 4:
        cands += [("multi", *c) for c in _MULTI_CANDIDATES if qmv_multi.supported(M, K, N, c[0], c[1])]
    if 3 <= M <= 6:  # at M <= 2 it wins in isolation but loses ~1% inside the model
        cands += [("v3", *c) for c in _V3_CANDIDATES if K % (c[0] * 32) == 0 and N % c[1] == 0]
    for kind in ("frag", "fragh") if dt == "float16" else ("frag",):  # fragh clamps to fp16's range
        cands += [(kind, *c) for c in _FRAG_CANDIDATES if qmv_frag.supported(M, K, N, c[0], c[2]) and (c[2] == 1 or K >= 4096)]
    if M >= 2:
        if MIN_M_SMALL <= M <= MAX_M_SMALL and (N >= 4096 or K >= 4096):
            cands.append(("small",))
        for BM, BN, BK, WM, WN, split in _MMA_CANDIDATES[8 if M <= 8 else 16]:
            if N % BN or K % (BK * split) or (K // split) % 64 or BN % (8 * WN):
                continue
            cands.append(("mma", BM, BN, BK, WM, WN, split))
    xr = mx.random.normal(x2.shape, key=mx.random.key(M * 131 + K)).astype(x2.dtype)  # not the global random state
    ref = _run(("mlx",), xr, mod).astype(mx.float32)
    tol = 0.05 * (1.0 + float(mx.max(mx.abs(ref))))
    ok = []
    for c in cands:
        try:
            got = _run(c, xr, mod).astype(mx.float32)
            if c[0] != "mlx" and not (float(mx.max(mx.abs(got - ref))) <= tol):  # a NaN fails too
                continue
            ok.append(c)
        except Exception:
            continue
    # Time over enough weight copies (>= 64 MB) that they come from DRAM, as
    # inside a model, not from the GPU's system cache; interleaved rounds, best
    # round per candidate, for robustness to machine noise.
    nbytes = sum(mod[k].nbytes for k in ("weight", "scales", "biases"))
    ncopy = max(1, min(16, (64 << 20) // max(nbytes, 1)))
    copies = [{k: mx.array(mod[k]) + 0 if k != "weight" else mx.array(mod[k]) | 0 for k in ("weight", "scales", "biases")}
              for _ in range(ncopy)]
    mx.eval(copies)

    def sweep(c):
        f = _launcher(c, M, K, N, x2.dtype)
        return [f(xr, cp["weight"], cp["scales"], cp["biases"]) for cp in copies]

    times = {c: float("inf") for c in ok}
    for _ in range(4):
        for c in ok:
            times[c] = min(times[c], _time_us(lambda: sweep(c), reps=max(1, 8 // ncopy)) / ncopy)
    del copies
    mlx_t = times[("mlx",)]
    for fam in {c[0] for c in ok}:
        tune[f"{key}|{fam}"] = min((c for c in ok if c[0] == fam), key=lambda c: times[c])
    tune.setdefault("__times__", {})[key] = {fam: round(min(times[c] for c in ok if c[0] == fam), 1) for fam in {c[0] for c in ok}}
    challenger = min((c for c in ok if c[0] != "mlx"), key=lambda c: times[c], default=None)
    choice = challenger if challenger is not None and times[challenger] < 0.9 * mlx_t else ("mlx",)
    best = (times[choice], choice)
    tune[key] = best[1]
    _CHOICE[(N, K, M, dt)] = best[1]
    _save_tune()
    return best[1]


def _tunable(mod) -> bool:
    """A layer the tuned kernels compute: 4-bit affine quantization in groups of 64 (other group sizes,
    and mxfp4 and the like, stay on MLX's kernels; tuning them crashed the load)."""
    return (isinstance(mod, (nn.QuantizedLinear, nn.QuantizedEmbedding)) and getattr(mod, "bits", None) == 4
            and getattr(mod, "group_size", None) == 64 and getattr(mod, "mode", "affine") == "affine"
            and mod.get("biases") is not None and (mod["weight"].shape[1] * 8) % 64 == 0)


def _eligible(x: mx.array, mod) -> bool:
    if not _tunable(mod):
        return False
    K = x.shape[-1]
    if K % 64:
        return False
    M = x.size // K
    return MIN_M <= M <= MAX_M and x.dtype in (mx.float16, mx.bfloat16)


_PAD: dict = {}  # (N, K, M, dtype) -> the rows M rows of this shape are computed at (_pad_rows)


def _pad_rows(N: int, K: int, M: int, dt: str) -> int:
    """The rows to compute M rows of this layer shape at: up to 2M rows where the tuned timings say the chosen
    kernel there is at least 7% faster. On an M5 Pro a Qwen3.5-4B decode step took 32.5 ms at 12 rows and 29.8 ms
    at 16 (9B: 64.9 and 55.5): there MLX switches to another kernel between them, and 12 requests padded to 16 ran
    10-13% faster end to end (9B: 19-20%). On an M2 Pro no larger size is more than 3% faster, so nothing is
    padded. Outside 8 to 16 rows the same timings also padded 2 rows to 3, 4 to 5 and 24 to 32 on some layers, by
    gains the M5 Pro's sweeps did not show (2 to 6 requests: -2.8% to +4.4%, within the 4% two servers of the same
    code differ by; 24: +0.7% on a 9B, -0.4% on a 4B), so only 8 to 16 rows are padded unless ``LOCALS_PAD_ROWS=all``.
    Needs the timings tuning keeps; ``LOCALS_PAD_ROWS=0`` never pads."""
    hit = _PAD.get((N, K, M, dt))
    if hit is not None:
        return hit
    best = M
    mode = _os.environ.get("LOCALS_PAD_ROWS", "1")
    if mode != "0" and (8 <= M <= 16 or mode == "all"):
        tune = _load_tune()
        times = tune.get("__times__", {})

        def t(m):
            k = f"{N}x{K}x{m}" + ("" if dt == "float16" else f"x{dt}")
            c, tt = tune.get(k), times.get(k)
            return tt.get(c[0], tt.get("mlx")) if c is not None and tt else None

        tm = t(M)
        if tm:
            for m2 in range(M + 1, min(MAX_M, 2 * M) + 1):
                t2 = t(m2)
                if t2 and t2 < 0.93 * tm and (best == M or t2 < t(best)):
                    best = m2
    _PAD[(N, K, M, dt)] = best
    return best


def _small_matmul(x, mod):
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    M, K = x2.shape
    P = _pad_rows(mod["weight"].shape[0], K, M, str(x2.dtype).split(".")[-1])
    if P > M:  # more rows run faster on this GPU: compute them, keep these
        xp = mx.concatenate([x2, mx.zeros((P - M, K), x2.dtype)])
        y = _run(_choose(xp, mod), xp, mod)[:M]
    else:
        y = _run(_choose(x2, mod), x2, mod)
    return y.reshape(*shape[:-1], y.shape[-1])


def small_config(N: int):
    """(rows per simdgroup, simdgroups per threadgroup) for qmm_small."""
    return (2, 4) if N >= 4096 else (2, 2)


MIN_M_SMALL = 3
MAX_M_SMALL = 6

_orig_linear_call = nn.QuantizedLinear.__call__
_orig_embed_as_linear = nn.QuantizedEmbedding.as_linear
_enabled = False


# Row packing for right-padded batches: while active, a linear layer applied to
# the padded (B, L, D) activations runs only on the valid rows and scatters the
# result back, so padding costs no matmul work.
_PACK = _threading.local()  # .ctx: (B, L, valid flat indices, count) in the thread building the graph


class packed_rows:
    def __init__(self, B: int, L: int, lengths):
        self.ctx = (B, L, mx.array([b * L + i for b, n in enumerate(lengths) for i in range(n)], dtype=mx.int32), sum(lengths))

    def __enter__(self):
        self.prev = getattr(_PACK, "ctx", None)
        _PACK.ctx = self.ctx
        return self

    def __exit__(self, *a):
        _PACK.ctx = self.prev


def _maybe_packed(self, x, call):
    ctx = getattr(_PACK, "ctx", None)
    if ctx is not None and x.ndim == 3 and x.shape[0] == ctx[0] and x.shape[1] == ctx[1] and ctx[3] < x.shape[0] * x.shape[1]:
        B, L, idx, R = ctx
        xs = x.reshape(B * L, -1)[idx][None]
        ys = call(self, xs)[0]
        y = mx.zeros((B * L, ys.shape[-1]), ys.dtype)
        y[idx] = ys
        return y.reshape(B, L, -1)
    return call(self, x)


PREFILL_M = 128  # from this many rows, dequantize and use the dense GEMM where that is faster on this GPU
PREFILL_TUNE_M = 1024  # the rows that choice is timed at: one prefill chunk


DENSE_MAX = 1 << 27  # weights (N x K) dequantized at most: an lm_head of 248k x 2560 would be a 1.3 GB temporary
_DENSE: dict = {}  # (N, K, dtype) -> True: dequantize + dense GEMM; False: MLX's quantized GEMM


def _dense_ok(x, mod) -> bool:
    if not _tunable(mod) or mod["weight"].shape[0] * mod["weight"].shape[1] * 8 > DENSE_MAX:
        return False
    return x.dtype in (mx.float16, mx.bfloat16) and x.size // x.shape[-1] >= PREFILL_M and _dense_prefill(mod, x.dtype)


def _prefill_key(N: int, K: int, dt: str) -> str:
    return f"{N}x{K}xP" + ("" if dt == "float16" else f"x{dt}")


def _dense_prefill(mod, dtype) -> bool:
    """Whether a prompt is read through this layer as dequantize + dense GEMM rather than MLX's quantized
    GEMM: timed on this GPU once per shape (cached with the other choices). On an M2 Pro with MLX 0.32 the
    dense GEMM is 3-11% faster at 1024 rows (it was 1.5x with older MLX); MLX's quantized GEMM uses the M5's
    neural accelerators. ``LOCALS_DENSE_PREFILL``: 1 always dense, 0 never."""
    w = mod["weight"]
    N, K, dt = w.shape[0], w.shape[1] * 8, str(dtype).split(".")[-1]
    hit = _DENSE.get((N, K, dt))
    if hit is not None:
        return hit
    force = _os.environ.get("LOCALS_DENSE_PREFILL")
    if force in ("0", "1"):
        _DENSE[(N, K, dt)] = force == "1"
        return _DENSE[(N, K, dt)]
    key = _prefill_key(N, K, dt)
    tune = _load_tune()
    if key in tune:
        _DENSE[(N, K, dt)] = tune[key][0] == "dense"
        return _DENSE[(N, K, dt)]
    if _DEFER and not _TUNING:
        return True  # not timed yet: as before (the server times it while idle)
    with _TUNE_LOCK:
        if (N, K, dt) not in _DENSE:
            _tune_prefill(mod, N, K, dtype, key, tune)
        return _DENSE[(N, K, dt)]


GDN_SIZES = (512, 2048)  # prompt tokens per pass the recurrent-layer choice is timed at (whole model)
_GDN: dict = {}  # (Hk, Hv, Dk, Dv, dtype, size) -> True: MLX's mx.fast.gated_delta_update
_GDN_FORCE = None  # while timing the whole model: True / False for every layer


def _gdn_size(S: int) -> int:
    return GDN_SIZES[-1] if S >= 1024 else GDN_SIZES[0]


def _gdn_key(Hk, Hv, Dk, Dv, dt, size) -> str:
    return f"GDN{Hk}x{Hv}x{Dk}x{Dv}xT{size}" + ("" if dt == "float16" else f"x{dt}")


def gdn_fast(Hk: int, Hv: int, Dk: int, Dv: int, dtype, S: int) -> bool:
    """Whether a Gated DeltaNet layer (Qwen3.5) reads ``S`` prompt tokens, unpadded, with MLX's built-in
    mx.fast.gated_delta_update (a chunked algorithm of matrix products, with a version for the M5's neural
    accelerators) rather than mlx-lm's step-by-step kernel. Chosen by timing the whole model reading a
    prompt (gdn_model_tune, in locals tune): timed alone, the op is 1.3-1.9x faster on an M2 Pro, yet
    inside the 4B reading 4,096 and 8,192 tokens it was 0.94x and 0.93x. Untimed: mlx-lm's kernel. With
    padding (a mask) the op is 12-50x slower and never used. ``LOCALS_GDN_FAST``: 1 always, 0 never."""
    if _GDN_FORCE is not None:
        return _GDN_FORCE
    dt = str(dtype).split(".")[-1]
    size = _gdn_size(S)
    hit = _GDN.get((Hk, Hv, Dk, Dv, dt, size))
    if hit is not None:
        return hit
    force = _os.environ.get("LOCALS_GDN_FAST")
    if force in ("0", "1"):
        fast = force == "1" and hasattr(mx.fast, "gated_delta_update")
    else:
        c = _load_tune().get(_gdn_key(Hk, Hv, Dk, Dv, dt, size))
        fast = c is not None and c[0] == "fast"
    _GDN[(Hk, Hv, Dk, Dv, dt, size)] = fast
    return fast


def gdn_model_tune(model, log=None) -> dict:
    """Time ``model`` reading 512- and 2,048-token prompts with mlx-lm's recurrent kernel and with MLX's op,
    best of two runs each, and keep the op where it is at least 3% faster. Returns {size: (kernel s, op s)}."""
    global _GDN_FORCE
    layers = _gdn_layers(model)
    if not layers or not hasattr(mx.fast, "gated_delta_update"):
        return {}
    tune = _load_tune()
    keys = {T: [_gdn_key(Hk, Hv, Dk, Dv, str(dt).split(".")[-1], T) for _, Hk, Hv, Dk, Dv, dt in layers] for T in GDN_SIZES}
    if all(k in tune for ks in keys.values() for k in ks):
        return {}
    from mlx_lm.models.cache import make_prompt_cache

    lm = text_model(model)
    body = getattr(lm, "model", None)
    if body is None:
        return {}
    res = {}
    try:
        for T in GDN_SIZES:
            x = mx.array([[(i * 7919) % 20000 + 100 for i in range(T)]])
            best = {}
            for _ in range(2):
                for fast in (False, True):
                    _GDN_FORCE = fast
                    cache = make_prompt_cache(lm)
                    mx.eval(body(x[:, :16], cache))  # (compiled and warm)
                    mx.synchronize()
                    t0 = _time.perf_counter()
                    mx.eval(body(x, make_prompt_cache(lm)))
                    mx.synchronize()
                    best[fast] = min(best.get(fast, float("inf")), _time.perf_counter() - t0)
            res[T] = (best[False], best[True])
            for k in keys[T]:
                tune[k] = ("fast",) if best[True] < 0.97 * best[False] else ("kernel",)
                tune.setdefault("__times__", {})[k] = {"kernel": round(best[False] * 1e6, 1), "fast": round(best[True] * 1e6, 1)}
    finally:
        _GDN_FORCE = None
        _GDN.clear()
    _save_tune()
    if log:
        log("locals: prompt reading in the recurrent layers: " + ", ".join(
            f"{T} tokens {k * 1e3:.0f} ms with mlx-lm's kernel, {f * 1e3:.0f} ms with MLX's op" for T, (k, f) in res.items()))
    return res


def _gdn_layers(model) -> list:
    """(layer, Hk, Hv, Dk, Dv, dtype) for each distinct Gated DeltaNet shape of ``model``."""
    seen, out = set(), []
    for _, m in model.named_modules():
        if hasattr(m, "num_v_heads") and hasattr(m, "head_k_dim") and hasattr(m, "A_log"):
            dims = (m.num_k_heads, m.num_v_heads, m.head_k_dim, m.head_v_dim, m.A_log.dtype)
            act = next((v.dtype for _, v in m.parameters().items() if hasattr(v, "dtype") and v.dtype in (mx.float16, mx.bfloat16)), None)
            norm = getattr(getattr(m, "norm", None), "weight", None)
            act = norm.dtype if norm is not None else (act or mx.float16)
            if (dims[:4], act) not in seen:
                seen.add((dims[:4], act))
                out.append((m, *dims[:4], act))
    return out


def _tune_prefill(mod, N, K, dtype, key, tune) -> None:
    x = mx.random.normal((PREFILL_TUNE_M, K), key=mx.random.key(K + 7)).astype(dtype)
    sc, bi = mod["scales"].astype(dtype), mod["biases"].astype(dtype)

    def quant():
        return mx.quantized_matmul(x, mod["weight"], mod["scales"], mod["biases"], transpose=True, group_size=64, bits=4)

    def dense():
        return x @ mx.dequantize(mod["weight"], sc, bi, group_size=64, bits=4).T

    tq = td = float("inf")
    for _ in range(3):  # interleaved rounds, the best of each
        tq, td = min(tq, _time_us(quant, reps=4)), min(td, _time_us(dense, reps=4))
    dense_wins = td < 0.95 * tq  # (the dense path also needs the dequantized weights as a temporary)
    tune[key] = ("dense",) if dense_wins else ("mlx",)
    tune.setdefault("__times__", {})[key] = {"mlx": round(tq, 1), "dense": round(td, 1)}
    _DENSE[(N, K, str(dtype).split(".")[-1])] = dense_wins
    _save_tune()


def _dense_matmul(x, mod):
    """Large-M 4-bit matmul as dequantize + GEMM in the activations' own type, where it is faster than MLX's
    quantized GEMM on this GPU (_dense_prefill): the dequantized weights cost little traffic next to
    M x K x N FLOPs. (A bf16 model is not computed in fp16: values past 65504 would become inf.)"""
    w = mx.dequantize(mod["weight"], mod["scales"].astype(x.dtype), mod["biases"].astype(x.dtype), group_size=64, bits=4)
    return x @ w.T


def _linear_call_inner(self, x):
    if _dense_ok(x, self):
        y = _dense_matmul(x, self)
        if "bias" in self:
            y = y + self["bias"]
        return y
    if _eligible(x, self):
        y = _small_matmul(x, self)
        if "bias" in self:
            y = y + self["bias"]
        return y
    return _orig_linear_call(self, x)


def _linear_call(self, x):
    return _maybe_packed(self, x, _linear_call_inner)


def _linear_call_plain(self, x):
    return _maybe_packed(self, x, _orig_linear_call)


def _embed_as_linear(self, x):
    if _eligible(x, self):
        return _small_matmul(x, self)
    return _orig_embed_as_linear(self, x)


def _linear_switch(self, x):
    return (_linear_call if _enabled else _linear_call_plain)(self, x)


def _embed_switch(self, x):
    return (_embed_as_linear if _enabled else _orig_embed_as_linear)(self, x)


_PATCHED: dict = {}  # base class -> its locals subclass


def _patched_class(base):
    if base not in _PATCHED:
        body = {"as_linear": _embed_switch} if issubclass(base, nn.QuantizedEmbedding) else {"__call__": _linear_switch}
        _PATCHED[base] = type("Locals" + base.__name__, (base,), body)
    return _PATCHED[base]


def enable_small_m(enabled: bool = True, model=None) -> None:
    """Route the 4-bit quantized matmuls with few rows of locals's models to the autotuned fastest kernel
    (MLX, qmm_small, qmm_mma, ...), and their padded batches to row packing; ``enabled=False`` keeps
    MLX's kernels with the packing. With ``model``, that model's layers become locals's (their class a
    subclass); without, the switch changes for the models patched so far (an A/B comparison). mlx's
    own classes are never changed: other models in the process (the server's stock engine, a library
    user's own) stay as mlx-lm makes them."""
    global _enabled
    _enabled = enabled
    if model is None:
        return
    for _, mod in model.named_modules():
        if type(mod) in (nn.QuantizedLinear, nn.QuantizedEmbedding):  # mlx's own (a subclass may compute differently)
            mod.__class__ = _patched_class(type(mod))


def forward_hidden(model, tokens: mx.array, cache) -> mx.array:
    """Final normed hidden state of the last position, shape (1, 1, D)."""
    lm = text_model(model)
    return lm.model(tokens, cache)[:, -1:, :]


def tied_embeddings(lm) -> bool:
    """Whether the vocabulary projection is the embedding table. Models keep the
    flag on the instance (Gemma 3, set while loading), in ``args`` (Llama, Qwen),
    or drop ``lm_head`` altogether."""
    if not hasattr(lm, "lm_head"):
        return True
    t = lm.__dict__.get("tie_word_embeddings") if isinstance(lm.__dict__, dict) else None
    if t is None:
        t = getattr(lm, "tie_word_embeddings", None)
    if t is None:
        t = getattr(getattr(lm, "args", None), "tie_word_embeddings", False)
    return bool(t)


def head_module(lm):
    return lm.model.embed_tokens if tied_embeddings(lm) else lm.lm_head


def post_logits(lm, out: mx.array) -> mx.array:
    """What the model's own forward does to the logits after the vocabulary
    projection: soft-capping (Gemma 2, 3n, 4), a logit scale (Cohere), a
    logits scaling divisor (Granite). Monotonic, so greedy and constrained
    choices never depend on it; sampling at temperature > 0 does."""
    args = getattr(lm, "args", None)
    cap = getattr(lm, "final_logit_softcapping", None)
    if cap is None:
        cap = getattr(args, "final_logit_softcapping", None)
    if cap:
        out = mx.tanh(out / cap) * cap
    mod = type(lm).__module__.rsplit(".", 1)[-1]
    if mod in ("cohere", "cohere2"):
        out = out * lm.model.args.logit_scale
    if mod.startswith("granite") and getattr(lm, "logits_scaling", None):
        out = out / lm.logits_scaling
    return out


def head_logits(lm, h: mx.array) -> mx.array:
    """Logits of hidden states ``h`` (..., D), as the model's own forward computes them."""
    mod = head_module(lm)
    out = mod.as_linear(h) if mod is lm.model.embed_tokens else mod(h)
    return post_logits(lm, out)


def head_check(model, log=None) -> float:
    """Max |logit difference| between the model's own forward and body + head_logits
    on a short probe (0 when they agree; logged when they do not)."""
    lm = text_model(model)
    probe = mx.array([[(i * 7919) % 1000 + 100 for i in range(6)]])
    try:
        ref = model(probe)[:, -1].astype(mx.float32)
        ours = head_logits(lm, lm.model(probe))[:, -1].astype(mx.float32)
        err = float(mx.max(mx.abs(ref - ours)))
    except Exception as e:  # noqa: BLE001
        if log:
            log(f"locals: could not check this model's output layer ({type(e).__name__}: {e})")
        return float("nan")
    tol = 0.02 * float(mx.max(mx.abs(ref))) + 0.05
    if err > tol and log:
        log(f"locals: this model's forward post-processes logits in a way locals does not reproduce (max diff {err:.3f}); "
            "greedy and constrained decoding are unaffected, sampling at temperature > 0 may differ")
    return err


def head_dense(model, h: mx.array) -> mx.array:
    lm = text_model(model)
    return head_logits(lm, h)[0, -1]


def head_rows(model, h: mx.array, ids: mx.array) -> mx.array:
    """Logits for the vocabulary rows ``ids`` only (exact), shape (len(ids),).

    Constrained decisions often allow a handful of tokens; gathering just
    those rows of the 4-bit head skips most of its weight traffic."""
    lm = text_model(model)
    mod = head_module(lm)
    if not isinstance(mod, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        return head_dense(model, h)[ids]
    w, sc, bi = mod["weight"][ids], mod["scales"][ids], mod.get("biases")
    bi = bi[ids] if bi is not None else None
    # the layer's own mode: an mxfp4 head (Mellum2.1's MLX build) has no biases, and affine's call failed
    out = mx.quantized_matmul(h[0], w, sc, bi, transpose=True, group_size=mod.group_size, bits=mod.bits,
                              mode=getattr(mod, "mode", "affine"))[-1]
    return post_logits(lm, out)


def text_model(model):
    """The decoder-only language model inside a (possibly multimodal) wrapper."""
    return getattr(model, "language_model", model)


def forward_last(model, tokens: mx.array, cache) -> mx.array:
    """Logits for the last position only: run the body on every token but the
    vocabulary projection on the final hidden state."""
    model = text_model(model)
    body = getattr(model, "model", None)
    args = getattr(model, "args", None)
    if body is None or args is None:
        return model(tokens, cache=cache)[:, -1, :]
    h = body(tokens, cache)[:, -1:, :]
    return head_logits(model, h)[:, -1, :]


def model_tune(model, ms=range(2, MAX_M + 1), families=("v3", "multi", "fragh", "frag", "mma"), steps: int = 16, rounds: int = 5, verbose: bool = False) -> dict:
    """Pick one kernel family per row count M by timing the real decode step.

    Per-shape timings in isolation miss interactions inside a model (cache
    reuse across layers, overlap of consecutive kernels): e.g. at M = 4 the
    scalar v3 kernel loses to fragh on every shape alone yet makes the Qwen3.5-2B
    step ~4% faster. For each M, every family whose per-shape winner exists for
    all shapes is timed on a batch-M decode step; the fastest (>= 2% better
    than the per-shape mix) overrides the per-shape choices."""
    import time as _t

    lm = text_model(model)
    shapes = set()
    for _, mod in model.named_modules():
        if _tunable(mod):
            shapes.add((mod["weight"].shape[0], mod["weight"].shape[1] * 8, str(mod["scales"].dtype).split(".")[-1]))
    tune = _load_tune()
    import hashlib

    sig = "__model_tuned_pipe__" + hashlib.md5(repr(sorted(shapes)).encode()).hexdigest()[:10]
    done = list(tune.get(sig, []))
    picked = {}
    for M in ms:
        if M in done:
            continue
        keys = {(N, K, dt): f"{N}x{K}x{M}" + ("" if dt == "float16" else f"x{dt}") for N, K, dt in shapes}
        if not all(k in tune for k in keys.values()):
            continue
        base = {sk: tuple(tune[k]) for sk, k in keys.items()}
        options = {"per-shape": base}
        for fam in families:
            fk = {sk: tune.get(f"{k}|{fam}") for sk, k in keys.items()}
            if all(v is not None for v in fk.values()):
                options[fam] = {sk: tuple(v) for sk, v in fk.items()}
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(lm)
        lm(mx.array([[(i * 7919 + b) % 20000 + 100 for i in range(64)] for b in range(M)]), cache=cache)
        mx.eval([c.state for c in cache])
        x = mx.array([[(b * 31) % 20000 + 50] for b in range(M)])
        def step():
            h = lm.model(x, cache)
            y = head_logits(lm, h)
            return mx.argmax(y[:, -1], -1)

        res = {name: [] for name in options}
        for _ in range(rounds):
            for name, opt in options.items():
                for (N, K, dt), c in opt.items():
                    _CHOICE[(N, K, M, dt)] = c
                # Pipelined, as the engines decode: step s+1 is built and
                # queued while step s runs. Synchronous timing ranks the
                # families differently (0.8B, M=8, fp16: sync picked fragh,
                # 17% slower pipelined than frag).
                y = step()
                mx.async_eval(y)
                mx.eval(y)
                t0 = _t.perf_counter()
                for i in range(steps):
                    y2 = step()
                    mx.async_eval(y2)
                    mx.eval(y)
                    y = y2
                mx.eval(y)
                res[name].append((_t.perf_counter() - t0) / steps)
        med = {name: sorted(v)[len(v) // 2] for name, v in res.items()}
        best = min(med, key=med.get)
        if verbose:
            print(f"model_tune M={M}: " + ", ".join(f"{k} {v * 1e3:.2f} ms" for k, v in sorted(med.items(), key=lambda kv: kv[1])), flush=True)
        chosen = options[best] if med[best] < 0.98 * med["per-shape"] else base
        for (N, K, dt), c in chosen.items():
            _CHOICE[(N, K, M, dt)] = c
            tune[keys[(N, K, dt)]] = list(c)
        picked[M] = best if chosen is not base else "per-shape"
        done.append(M)
        tune[sig] = done
        _save_tune()
        del cache
    return picked


def warmup(model, ms=range(1, 9)) -> int:
    """Compile the chosen kernel of every matmul shape for M in ``ms`` now,
    instead of on first use in the middle of a request (each new kernel
    instance costs ~50 ms of Metal compilation). Returns instances run."""
    outs, seen, failed = [], set(), set()
    for _, mod in model.named_modules():
        if not _tunable(mod):
            continue
        N, K = mod["weight"].shape[0], mod["weight"].shape[1] * 8
        dt = mod["scales"].dtype
        for M in ms:
            x = mx.zeros((M, K), dt)
            c = _choose(x, mod)
            key = (c, M if c[0] not in ("frag", "fragh") else (M + 7) // 8, K, N, dt)
            if key in failed:  # the same kernel instance as one that failed (frag shares one across 8 row counts)
                _CHOICE[(N, K, M, str(dt).split(".")[-1])] = ("mlx",)
                continue
            if key in seen:
                continue
            seen.add(key)
            try:  # each chosen kernel compiled and run once now: one that fails here falls back to MLX's
                mx.eval(_run(c, x, mod))
            except Exception:  # noqa: BLE001  # (a cached choice the installed mlx no longer compiles)
                failed.add(key)
                _CHOICE[(N, K, M, str(dt).split(".")[-1])] = ("mlx",)
    return len(seen)


def pretune(model, max_m: int = MAX_M, log=None) -> int:
    """Autotune every quantized matmul shape of ``model`` for M = 2..max_m
    ahead of time (results are cached on disk). Returns shapes tuned.
    ``log``: given a progress line now and then (every 10% and 20 s at most)."""
    tune = _load_tune()
    seen, work = set(), []
    for _, mod in model.named_modules():
        if _tunable(mod):
            w = mod["weight"]
            N, K = w.shape[0], w.shape[1] * 8
            if (N, K) in seen or K % 64:
                continue
            seen.add((N, K))
            dt = mod["scales"].dtype
            name = str(dt).split(".")[-1]
            for M in range(MIN_M, max_m + 1):
                work.append((mod, M, K, dt, f"{N}x{K}x{M}" + ("" if name == "float16" else f"x{name}") not in tune))
            if N * K <= DENSE_MAX:  # how prompts are read through it (_dense_prefill)
                work.append((mod, PREFILL_TUNE_M, K, dt, _prefill_key(N, K, name) not in tune))
    total, done = sum(w[-1] for w in work), 0
    t0 = last = _time.perf_counter()
    for mod, M, K, dt, pending in work:
        if M >= PREFILL_M:
            _dense_prefill(mod, dt)
        else:
            _choose(mx.zeros((M, K), dt), mod)
        done += pending
        now = _time.perf_counter()
        if log and pending and done < total and done * 10 // total > (done - 1) * 10 // total and now - last >= 20:
            log(f"locals: tuning {done * 100 // total}% ({(now - t0) / 60:.1f} min)")
            last = now
    return len(seen)


def pending_work(model, max_m: int = 16) -> list:
    """The (layer, rows) pairs of ``model`` whose matmul shape is not tuned yet, smallest batches first."""
    tune = _load_tune()
    seen, work = set(), []
    for _, mod in model.named_modules():
        if _tunable(mod):
            w = mod["weight"]
            N, K = w.shape[0], w.shape[1] * 8
            if (N, K) in seen or K % 64:
                continue
            seen.add((N, K))
            name = str(mod["scales"].dtype).split(".")[-1]
            for M in range(MIN_M, max_m + 1):
                if f"{N}x{K}x{M}" + ("" if name == "float16" else f"x{name}") not in tune:
                    work.append((mod, M))
            if N * K <= DENSE_MAX and _prefill_key(N, K, name) not in tune:
                work.append((mod, PREFILL_TUNE_M))  # how prompts are read through it, after the batch sizes
    return sorted(work, key=lambda w: w[1])


def tune_one(mod, M: int) -> tuple:
    """Tune one (layer, rows) shape now (the background tuner, in the engine's own thread)."""
    global _TUNING
    _TUNING = True
    try:
        if M >= PREFILL_M:
            return ("dense",) if _dense_prefill(mod, mod["scales"].dtype) else ("mlx",)
        K = mod["weight"].shape[1] * 8
        return _choose(mx.zeros((M, K), mod["scales"].dtype), mod)
    finally:
        _TUNING = False


def tune_pending(model, max_m: int = MAX_M) -> int:
    """How many (matmul shape, batch size) combinations of ``model`` the
    tuning cache (or the shipped preset) does not cover yet."""
    tune = _load_tune()
    seen, missing = set(), 0
    for _, mod in model.named_modules():
        if _tunable(mod):
            w = mod["weight"]
            N, K = w.shape[0], w.shape[1] * 8
            if (N, K) in seen or K % 64:
                continue
            seen.add((N, K))
            dt = str(mod["scales"].dtype).split(".")[-1]
            for M in range(MIN_M, max_m + 1):
                key = f"{N}x{K}x{M}" + ("" if dt == "float16" else f"x{dt}")
                missing += key not in tune
    return missing


def tune_report(model) -> str:
    """What tuning measured for ``model`` on this GPU (locals tune --report): per batch size, the time of
    one decode step's 4-bit matmuls with MLX's kernels against the chosen ones (each layer shape counted
    as often as the model has it), the fastest of Locals' own, and per shape how prompts are read. Shapes
    taken from a shipped preset have no timings."""
    from collections import Counter

    tune = _load_tune()
    times = tune.get("__times__", {})
    shapes = Counter()
    for _, mod in model.named_modules():
        if _tunable(mod):
            w = mod["weight"]
            if (w.shape[1] * 8) % 64 == 0:
                shapes[(w.shape[0], w.shape[1] * 8, str(mod["scales"].dtype).split(".")[-1])] += 1
    dts = sorted({dt for _, _, dt in shapes})
    out = [f"GPU {mx.device_info()['architecture']}, MLX {mx.__version__}, {', '.join(dts)}, "
           f"{sum(shapes.values())} quantized layers in {len(shapes)} shapes; cache {_TUNE_PATH}",
           "", "rows  MLX's kernels  chosen  speed-up  fastest of Locals' own  chosen kernels (layers)"]
    for M in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        t_mlx = t_ch = t_own = 0.0
        fams, timed = Counter(), 0
        for (N, K, dt), n in shapes.items():
            key = f"{N}x{K}x{M}" + ("" if dt == "float16" else f"x{dt}")
            t, c = times.get(key), tune.get(key)
            if not t or "mlx" not in t or c is None:
                continue
            timed += n
            own = min((v for f, v in t.items() if f != "mlx"), default=t["mlx"])
            t_mlx, t_ch, t_own = t_mlx + n * t["mlx"], t_ch + n * t.get(c[0], t["mlx"]), t_own + n * own
            fams[c[0]] += n
        if not timed:
            out.append(f"{M:4d}  (not timed here)")
            continue
        note = "" if timed == sum(shapes.values()) else f", {timed} of {sum(shapes.values())} layers timed"
        pads = Counter(_pad_rows(N, K, M, dt) for (N, K, dt), n in shapes.items() for _ in range(n))
        pads.pop(M, None)
        if pads:
            note += "; computed at " + ", ".join(f"{m} rows ({k} layers)" for m, k in sorted(pads.items()))
        out.append(f"{M:4d}  {t_mlx / 1e3:9.2f} ms  {t_ch / 1e3:6.2f} ms  {t_mlx / t_ch:6.2f}x  {t_own / 1e3:9.2f} ms "
                   f"({t_mlx / t_own:.2f}x)    " + ", ".join(f"{f} {k}" for f, k in fams.most_common()) + note)
    out += ["", f"reading prompts ({PREFILL_TUNE_M} rows): MLX's quantized GEMM against dequantize + dense GEMM"]
    for (N, K, dt), n in sorted(shapes.items()):
        key = _prefill_key(N, K, dt)
        t, c = times.get(key), tune.get(key)
        if N * K > DENSE_MAX:
            continue
        if not t:
            out.append(f"  {N}x{K} ({n} layers): not timed here")
            continue
        out.append(f"  {N}x{K} ({n} layers): MLX {t['mlx'] / 1e3:.2f} ms, dense {t['dense'] / 1e3:.2f} ms -> {c[0]}")
    gdn = _gdn_layers(model)
    if gdn:
        out += ["", "reading prompts in the recurrent layers (unpadded): mlx-lm's kernel against MLX's mx.fast.gated_delta_update"]
        for _, Hk, Hv, Dk, Dv, dt in gdn:
            for b in GDN_SIZES:
                key = _gdn_key(Hk, Hv, Dk, Dv, str(dt).split(".")[-1], b)
                t, c = times.get(key), tune.get(key)
                out.append(f"  {Hv} heads, the whole model reading {b:5d} tokens: " + (
                    f"{t['kernel'] / 1e3:.0f} ms with the kernel, {t['fast'] / 1e3:.0f} ms with MLX's op -> {c[0]}" if t and c else "not timed here"))
    return "\n".join(out)


def tune_model(model, mode: str = "auto", log=None) -> dict:
    """The one-time kernel autotuning a loaded model goes through.

    ``"auto"``: if the cache or a shipped preset already covers the model, do
    nothing but compile the chosen kernels (seconds); otherwise run the quick
    procedure. ``"quick"``: tune every matmul shape for batch sizes up to 16
    (a few minutes). ``"full"``: every shape up to 32 rows plus whole-step
    timing per batch size (10--20 minutes; the best batched throughput).
    ``"off"``: nothing ahead of time (untuned shapes are tuned on first use).
    Results are cached in ``~/.cache/locals`` (or ``$LOCALS_TUNE_CACHE``)."""
    import sys as _sys

    log = log or (lambda s: print(s, file=_sys.stderr, flush=True))
    mode = (mode or "auto").lower()
    if mode not in ("auto", "background", "quick", "full", "off"):
        raise ValueError("tune must be one of auto, background, quick, full, off")
    global _DEFER
    _DEFER = False  # a later load in this process tunes as its own mode says
    if mode == "off":
        return {"mode": "off"}
    if mode in ("auto", "background"):
        missing = tune_pending(model, max_m=16)  # the background tuner and the quick procedure go up to 16 rows
        if missing == 0:  # everything up to 16 rows cached (or shipped): compile the chosen kernels, nothing to tune
            _DEFER = mode == "background"  # a server never tunes in the middle of a request (more rows stay on MLX's)
            warmup(model, range(1, 17))
            return {"mode": "cached"}
        if mode == "background":
            _DEFER = True  # serve now on MLX's kernels; the server tunes while it is idle
            warmup(model, range(1, 17))  # the shapes tuned already (a shipped preset): compiled now, checked
            pending = len(pending_work(model))
            log(f"locals: no tuned kernels for this model on {mx.device_info()['architecture']} yet: serving now, and tuning "
                f"{pending} kernel shapes while the server is idle (cached afterwards). locals stop, then locals tune --model <model>, does the "
                f"full procedure at once (10-20 minutes, the best batched throughput).")
            return {"mode": "background", "pending": pending}
        else:
            mode = "quick"
        pending = tune_pending(model, max_m=16) if mode == "quick" else 0
        if pending:  # the quick procedure has work to do
            log(f"locals: tuning the matrix kernels for this model on {mx.device_info()['architecture']}: {pending} kernel "
                f"shapes, once (cached afterwards; on an M2 Pro 2 minutes for Qwen3.5-4B, 7 for Gemma 4 E4B). "
                f"For the full procedure, 10-20 minutes and the best batched throughput: locals tune --model <model>")
    if mode == "quick":
        pretune(model, max_m=16, log=log)
        gdn_model_tune(model, log=log)
        warmup(model, range(1, 17))
        return {"mode": "quick"}
    pretune(model)
    gdn_model_tune(model, log=log)
    picked = model_tune(model)
    warmup(model, range(1, 17))  # compile now, not in the first requests
    return {"mode": "full", "picked": picked}
