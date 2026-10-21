"""A GPU without tuned kernels: shapes run on MLX's kernel at once, and are tuned one at a
time while the server is idle (no model needed: a small quantized layer)."""

import mlx.core as mx
import mlx.nn as nn

from locals import kernels


def test_untuned_shapes_wait_for_the_idle_tuner(tmp_path, monkeypatch):
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_TUNE", {})
    monkeypatch.setattr(kernels, "_CHOICE", {})
    monkeypatch.setattr(kernels, "_DEFER", True)
    layer = nn.QuantizedLinear(512, 1024, bits=4, group_size=64)
    model = nn.Sequential(layer)
    work = kernels.pending_work(model, max_m=3)
    assert [m for _, m in work] == [1, 2, 3, kernels.PREFILL_TUNE_M]  # smallest batches first, then how prompts are read
    x = mx.zeros((1, 512), layer["scales"].dtype)
    assert kernels._choose(x, layer) == ("mlx",)  # untuned: MLX's kernel, nothing tuned in the request
    key = "1024x512x1" + ("" if layer["scales"].dtype == mx.float16 else "x" + str(layer["scales"].dtype).split(".")[-1])
    assert key not in kernels._load_tune()
    picked = kernels.tune_one(*work[0])
    assert kernels._load_tune()[key] == picked and kernels._choose(x, layer) == picked
    assert [m for _, m in kernels.pending_work(model, max_m=3)] == [2, 3, kernels.PREFILL_TUNE_M]
    assert (tmp_path / "tune.json").exists()


def _fp16_layer():
    layer = nn.QuantizedLinear(512, 1024, bits=4, group_size=64)
    layer.set_dtype(mx.float16)
    return layer


def test_how_prompts_are_read_is_timed_on_this_gpu(tmp_path, monkeypatch):
    """Dequantize + dense GEMM or MLX's quantized GEMM for a prompt: timed once per shape (MLX's uses the M5's
    neural accelerators; on an M2 Pro the dense one is 3-11% faster), untimed in a request on the server."""
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_TUNE", {})
    monkeypatch.setattr(kernels, "_DENSE", {})
    monkeypatch.setattr(kernels, "_DEFER", True)
    monkeypatch.delenv("LOCALS_DENSE_PREFILL", raising=False)
    layer = _fp16_layer()
    x = mx.zeros((256, 512), mx.float16)
    assert kernels._dense_ok(x, layer)  # not timed yet: dense, as before
    key = kernels._prefill_key(1024, 512, "float16")
    assert key not in kernels._load_tune()
    kernels._DENSE.clear()
    picked = kernels.tune_one(layer, kernels.PREFILL_TUNE_M)
    tune = kernels._load_tune()
    assert tune[key] == picked and set(tune["__times__"][key]) == {"mlx", "dense"}
    assert kernels._dense_ok(x, layer) == (picked == ("dense",))
    assert not kernels._dense_ok(mx.zeros((64, 512), mx.float16), layer)  # few rows: never dense
    assert kernels.PREFILL_TUNE_M not in [m for _, m in kernels.pending_work(nn.Sequential(layer), max_m=0)]
    xr = mx.random.normal((256, 512), key=mx.random.key(3)).astype(mx.float16)
    ref = mx.quantized_matmul(xr, layer["weight"], layer["scales"], layer["biases"], transpose=True, group_size=64, bits=4)
    for dense in (True, False):  # either way, the layer computes the same
        kernels._DENSE[(1024, 512, "float16")] = dense
        y = kernels._linear_call_inner(layer, xr)
        assert float(mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)))) < 0.05 * (1 + float(mx.max(mx.abs(ref))))


def test_the_prompt_reading_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_TUNE", {})
    layer, x = _fp16_layer(), mx.zeros((256, 512), mx.float16)
    for value, dense in (("0", False), ("1", True)):
        monkeypatch.setattr(kernels, "_DENSE", {})
        monkeypatch.setenv("LOCALS_DENSE_PREFILL", value)
        assert kernels._dense_ok(x, layer) is dense
    assert not (tmp_path / "tune.json").exists()  # nothing timed


def test_the_tuning_report(tmp_path, monkeypatch):
    """locals tune --report: per batch size MLX's kernels against the chosen ones, and how prompts are read."""
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_TUNE", {})
    monkeypatch.setattr(kernels, "_CHOICE", {})
    monkeypatch.setattr(kernels, "_DENSE", {})
    monkeypatch.setattr(kernels, "_DEFER", False)
    monkeypatch.delenv("LOCALS_DENSE_PREFILL", raising=False)
    layer = _fp16_layer()
    model = nn.Sequential(layer, _fp16_layer())  # one shape, two layers
    for mod, M in kernels.pending_work(model, max_m=4):
        kernels.tune_one(mod, M)
    text = kernels.tune_report(model)
    assert "2 quantized layers in 1 shapes" in text
    rows = {line.split()[0]: line for line in text.splitlines() if line[:4].strip().isdigit()}
    assert "x" in rows["4"] and "ms" in rows["4"] and "not timed" in rows["8"]
    assert "1024x512 (2 layers): MLX" in text


def test_an_idle_tuner_waits_while_another_server_works(tmp_path, monkeypatch):
    """A server tuning while idle saw only its own engines: next to another locals server serving on the same GPU it
    timed kernels anyway, and kept the skewed timings. Each busy process marks a file; the tuner waits for them."""
    import os
    import subprocess
    import time

    from locals import service

    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_BUSY_DIR", str(tmp_path))
    other = tmp_path / f"busy-{os.getppid()}"  # a live process, busy a moment ago
    other.write_text("")
    assert kernels.others_busy()
    os.utime(other, (time.time() - 10, time.time() - 10))
    assert not kernels.others_busy()  # its last mark is old: idle now
    dead = subprocess.Popen(["true"])
    dead.wait()
    (tmp_path / f"busy-{dead.pid}").write_text("")
    assert not kernels.others_busy() and not (tmp_path / f"busy-{dead.pid}").exists()
    kernels.mark_busy(now=True)
    assert (tmp_path / f"busy-{os.getpid()}").exists() and not kernels.others_busy()  # (its own: not another's)

    svc = service.BatchService.__new__(service.BatchService)
    svc._tune_work, svc._inflight = [("layer", 1)], {}
    tuned = []
    monkeypatch.setattr(kernels, "tune_one", lambda mod, M: tuned.append((mod, M)))
    os.utime(other, None)
    svc._tune_while_idle()
    assert tuned == [] and svc._tune_work == [("layer", 1)]  # left for later, not dropped
    os.utime(other, (time.time() - 10, time.time() - 10))
    svc._tune_after = 0.0
    svc._tune_while_idle()
    assert tuned == [("layer", 1)] and not svc._tune_work


def test_shapes_timed_under_another_mlx_are_timed_again(tmp_path, monkeypatch):
    """The tune cache was not keyed on MLX's version: choices timed under an older MLX stayed for good after an
    upgrade. They are stale now: still used, and timed again by the idle tuner (and locals tune)."""
    import json

    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_preset_path", lambda: None)
    monkeypatch.setattr(kernels, "_TUNE", None)
    monkeypatch.setattr(kernels, "_CHOICE", {})
    monkeypatch.setattr(kernels, "_DEFER", True)
    layer = nn.QuantizedLinear(512, 1024, bits=4, group_size=64)
    dt = str(layer["scales"].dtype).split(".")[-1]
    key = "1024x512x1" + ("" if dt == "float16" else f"x{dt}")
    (tmp_path / "tune.json").write_text(json.dumps({"__mlx__": "0.0.1", key: ["mlx"], "__times__": {key: {"mlx": 1.0}}}))
    tune = kernels._load_tune()
    assert tune["__stale__"] == [key] and tune["__mlx__"] == mx.__version__
    assert json.loads((tmp_path / "tune.json").read_text())["__mlx__"] == mx.__version__
    x = mx.zeros((1, 512), layer["scales"].dtype)
    assert kernels._choose(x, layer) == ("mlx",)  # still used, not timed in a request
    work = kernels.pending_work(nn.Sequential(layer), max_m=1)
    assert [m for _, m in work] == [1, kernels.PREFILL_TUNE_M]
    kernels.tune_one(*work[0])
    assert key not in kernels._load_tune()["__stale__"] and kernels._load_tune()["__times__"][key] != {"mlx": 1.0}
    assert key not in json.loads((tmp_path / "tune.json").read_text())["__stale__"]
    assert [m for _, m in kernels.pending_work(nn.Sequential(layer), max_m=1)] == [kernels.PREFILL_TUNE_M]


def test_a_cache_from_before_versions_gets_one_and_nothing_is_stale(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_preset_path", lambda: None)
    monkeypatch.setattr(kernels, "_TUNE", None)
    (tmp_path / "tune.json").write_text(json.dumps({"1024x512x1": ["mlx"]}))
    tune = kernels._load_tune()
    assert not tune.get("__stale__") and json.loads((tmp_path / "tune.json").read_text())["__mlx__"] == mx.__version__
    # what both processes still call stale stays stale: one that timed a shape again took it off
    assert kernels._merge_tune({"__stale__": ["a"]}, {"__stale__": ["a", "b"]})["__stale__"] == ["a"]
