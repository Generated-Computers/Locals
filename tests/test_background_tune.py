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
