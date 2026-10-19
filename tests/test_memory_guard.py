"""The memory guard's estimate for admitting requests next to running ones: each request's own fixed state (Qwen3.5's
recurrent layers, which a 27B on 24 GB ran out of memory on), and room for reading a prompt learned from the reads
the engine made. MLX's pool of freed buffers sized by the room a model leaves; prompts joining the batch merged a layer
at a time.
Batches computed at more rows where the tuned kernels are faster there, and prompts read by MLX's recurrent op where
it is faster."""

import os
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from locals import hybrid_batch as hb
from locals import kernels


def _engine(row_state=0, kv=0):
    e = SimpleNamespace(_reads={})
    for name in ("_note_read", "_read_bytes", "_admit_bytes"):
        setattr(e, name, getattr(hb.HybridBatchEngine, name).__get__(e))
    e._row_state = lambda: row_state
    e._kv_bytes_per_token = lambda: kv
    return e


def test_room_for_reading_is_learned_from_the_reads(monkeypatch):
    monkeypatch.setattr(hb, "_RESERVE_ENV", None)
    e = _engine()
    assert e._read_bytes(10) == hb.PREFILL_RESERVE  # nothing seen yet: the fixed reserve
    e._note_read(12, 120 << 20)
    assert e._read_bytes(10) == (120 << 20) * 7 // 6  # a read of as many tokens or more was seen
    assert e._read_bytes(1000) == hb.PREFILL_RESERVE  # none that large yet
    e._note_read(2048, 2100 << 20)
    assert e._read_bytes(1000) == (2100 << 20) * 7 // 6 and e._read_bytes(5000) == (2100 << 20) * 7 // 6  # one pass
    monkeypatch.setattr(hb, "_RESERVE_ENV", "0.5")
    assert e._read_bytes(10) == hb.PREFILL_RESERVE  # LOCALS_PREFILL_RESERVE_GB: fixed


def test_each_new_request_brings_its_state(monkeypatch):
    monkeypatch.setattr(hb, "_RESERVE_ENV", None)
    e = _engine(row_state=154 << 20, kv=66 << 10)  # a 27B's recurrent state and KV per token
    e._note_read(16, 50 << 20)
    one = e._admit_bytes(1, 15)
    assert one == int(2.2 * (154 << 20)) + 256 * (66 << 10) + (50 << 20) * 7 // 6
    assert e._admit_bytes(8, 15) - one == 7 * (int(2.2 * (154 << 20)) + 256 * (66 << 10))


def test_buffer_pool_and_margin_follow_the_room_left(monkeypatch):
    from locals.service import memory_plan

    monkeypatch.delenv("LOCALS_MEMORY_MARGIN_GB", raising=False)
    G = 1 << 30
    assert memory_plan(int(11.45e9), int(2.4e9)) == (G, G + (512 << 20))  # a 4B on 16 GB: as before
    pool, margin = memory_plan(int(19.07e9), int(15.4e9))  # a 27B on 24 GB: 3.7 GB left
    assert pool == (int(19.07e9) - int(15.4e9)) // 8 and margin == pool + (512 << 20)
    assert memory_plan(int(5e9), int(4.9e9))[0] == 256 << 20  # at least 256 MB
    monkeypatch.setenv("LOCALS_MEMORY_MARGIN_GB", "2")
    assert memory_plan(int(19.07e9), int(15.4e9))[1] == 2 * G


def _tuned(tmp_path, monkeypatch, times):
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_TUNE", {**{f"1024x512x{m}": ("mlx",) for m in times},
                                           "__times__": {f"1024x512x{m}": {"mlx": t} for m, t in times.items()}})
    monkeypatch.setattr(kernels, "_PAD", {})
    monkeypatch.setattr(kernels, "_CHOICE", {})


def test_rows_are_padded_where_more_run_faster(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALS_PAD_ROWS", raising=False)
    _tuned(tmp_path, monkeypatch, {8: 60.0, 12: 100.0, 15: 95.0, 16: 80.0, 24: 85.0})
    assert kernels._pad_rows(1024, 512, 12, "float16") == 16  # 20% faster, and faster than 24 rows
    assert kernels._pad_rows(1024, 512, 8, "float16") == 8  # nothing faster within twice 8 rows
    assert kernels._pad_rows(1024, 512, 15, "float16") == 16 and kernels._pad_rows(1024, 512, 16, "float16") == 16
    layer = nn.QuantizedLinear(512, 1024, bits=4, group_size=64)
    layer.set_dtype(mx.float16)
    x = mx.random.normal((12, 512), key=mx.random.key(5)).astype(mx.float16)
    ref = mx.quantized_matmul(x, layer["weight"], layer["scales"], layer["biases"], transpose=True, group_size=64, bits=4)
    y = kernels._small_matmul(x, layer)
    assert y.shape == ref.shape and float(mx.max(mx.abs(y - ref))) < 0.05 * (1 + float(mx.max(mx.abs(ref))))
    monkeypatch.setattr(kernels, "_PAD", {})
    monkeypatch.setenv("LOCALS_PAD_ROWS", "0")
    assert kernels._pad_rows(1024, 512, 12, "float16") == 12


def test_no_padding_without_a_large_enough_gain(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALS_PAD_ROWS", raising=False)
    _tuned(tmp_path, monkeypatch, {12: 100.0, 16: 96.0})  # an M2 Pro: 4% apart
    assert kernels._pad_rows(1024, 512, 12, "float16") == 12
    assert kernels._pad_rows(1024, 512, 13, "float16") == 13  # untimed rows: as they are


def test_fewer_than_eight_rows_stay_unless_asked(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALS_PAD_ROWS", raising=False)
    _tuned(tmp_path, monkeypatch, {4: 100.0, 5: 80.0})  # as an M5 Pro's timings had on some layers
    assert kernels._pad_rows(1024, 512, 4, "float16") == 4
    monkeypatch.setattr(kernels, "_PAD", {})
    monkeypatch.setenv("LOCALS_PAD_ROWS", "all")
    assert kernels._pad_rows(1024, 512, 4, "float16") == 5


def test_recurrent_op_choice(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALS_GDN_FAST", raising=False)
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_TUNE", {kernels._gdn_key(16, 32, 128, 128, "bfloat16", 2048): ("fast",),
                                           kernels._gdn_key(16, 32, 128, 128, "bfloat16", 512): ("kernel",)})
    monkeypatch.setattr(kernels, "_GDN", {})
    assert kernels.gdn_fast(16, 32, 128, 128, mx.bfloat16, 4096)  # long prompts: timed at 2,048 tokens
    assert not kernels.gdn_fast(16, 32, 128, 128, mx.bfloat16, 300)  # shorter ones: at 512
    assert not kernels.gdn_fast(16, 32, 128, 128, mx.float16, 4096)  # untimed: mlx-lm's kernel
    monkeypatch.setattr(kernels, "_GDN", {})
    monkeypatch.setenv("LOCALS_GDN_FAST", "0")
    assert not kernels.gdn_fast(16, 32, 128, 128, mx.bfloat16, 4096)


@pytest.mark.skipif(not hasattr(mx.fast, "gated_delta_update"), reason="MLX without mx.fast.gated_delta_update")
def test_recurrent_op_matches_mlx_lm():
    """fuse.FusedGatedDeltaNet's call of MLX's op gives what mlx-lm's gated_delta_update gives."""
    from mlx_lm.models.gated_delta import compute_g, gated_delta_update

    B, S, Hk, Hv, D = 1, 96, 2, 4, 64
    r = lambda i, *shape: mx.random.normal(shape, key=mx.random.key(i))  # noqa: E731
    q, k = r(1, B, S, Hk, D) * 0.1, r(2, B, S, Hk, D) * 0.1
    v, a, b = r(3, B, S, Hv, D), r(4, B, S, Hv), r(5, B, S, Hv)
    A_log, dt_bias, state = r(6, Hv) * 0.5, r(7, Hv) * 0.5, r(8, B, Hv, D, D) * 0.1
    ref, ref_state = gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, None)
    out, out_state = mx.fast.gated_delta_update(q, k, v, compute_g(A_log, a, dt_bias), mx.sigmoid(b), state, None)
    assert float(mx.max(mx.abs(out - ref))) < 2e-3 * (1 + float(mx.max(mx.abs(ref))))
    assert float(mx.max(mx.abs(out_state - ref_state))) < 2e-3 * (1 + float(mx.max(mx.abs(ref_state))))


MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"


@pytest.fixture(scope="module")
def service():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3.5-4B not cached")
    from locals.service import BatchService

    saved = os.environ.get("LOCALS_DISK_CACHE")
    os.environ["LOCALS_DISK_CACHE"] = "0"  # (no system prompts written to this user's disk cache)
    svc = BatchService("locals", MODEL, slots=16, temp=0.0)
    yield svc
    svc.close()
    if saved is None:
        os.environ.pop("LOCALS_DISK_CACHE", None)
    else:
        os.environ["LOCALS_DISK_CACHE"] = saved


@pytest.mark.model
def test_tight_memory_batches_without_passing_the_limit(service, monkeypatch):
    """16 requests at once with 0.9 GB above what is in use (as a 27B on 24 GB has): some are held back, the others
    run together, all finish, and GPU memory stays under the guard's limit. With the old fixed 1.5 GB reserve they
    ran one at a time; with 0.5 GB all 16 were admitted, and memory went 0.95 GB past the limit."""
    monkeypatch.setattr(hb, "_RESERVE_ENV", None)
    tok = service.tokenizer
    warm = service.submit(tok.encode("Hello"), max_tokens=2)
    warm.wait(600)
    eng = service._eng
    import gc

    gc.collect()  # (a model another test left must not be freed under this test's limit)
    assert 40e6 < eng._row_state() < 70e6  # Qwen3.5-4B: 51.5 MB of recurrent state a request
    eng.memory_limit = mx.get_active_memory() + int(0.9e9)
    before = dict(eng.counters)
    hb._reset_peak()
    prompts = ["Explain how a refrigerator works.", "Why is the sky blue?", "What does a compiler do?", "What is a prime number?"]
    jobs = [service.submit(tok.encode(prompts[i % 4]), max_tokens=96) for i in range(16)]
    for j in jobs:
        j.wait(900)
    peak = mx.get_peak_memory()
    assert all(len(j.tokens) > 0 for j in jobs)
    assert peak <= eng.memory_limit + int(0.1e9), (peak / 1e9, eng.memory_limit / 1e9)
    assert eng.counters.get("deferred", 0) > before.get("deferred", 0)  # some waited
    assert eng.counters["admissions"] - before.get("admissions", 0) < 16  # and some ran together


def test_layerwise_merge_matches_and_releases():
    """Requests' own caches merged into a batch a layer at a time: what the whole merge gives, each request's
    layer released once copied (held to the end, every prompt's KV existed twice when they joined)."""
    from mlx_lm.models.cache import ArraysCache, KVCache

    def one(n, seed):
        kv, rec = KVCache(), ArraysCache(2)
        k = mx.random.normal((1, 2, n, 8), key=mx.random.key(seed))
        kv.update_and_fetch(k, k * 2)
        rec[0], rec[1] = mx.ones((1, 3, 4)) * seed, mx.ones((1, 2, 5, 5)) * seed
        return [kv, rec]

    ref = hb._merge([one(5, 1), one(9, 2)])
    caches = [one(5, 1), one(9, 2)]
    got = hb._merge_layerwise(caches)
    assert all(layer is None for c in caches for layer in c)
    for a, b in zip(ref, got):
        for x, y in zip(a.state, b.state):
            if isinstance(x, mx.array):
                assert x.shape == y.shape and bool(mx.array_equal(x, y))
