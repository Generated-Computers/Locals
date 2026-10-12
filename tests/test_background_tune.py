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
    assert [m for _, m in work] == [1, 2, 3]  # smallest batches first
    x = mx.zeros((1, 512), layer["scales"].dtype)
    assert kernels._choose(x, layer) == ("mlx",)  # untuned: MLX's kernel, nothing tuned in the request
    key = "1024x512x1" + ("" if layer["scales"].dtype == mx.float16 else "x" + str(layer["scales"].dtype).split(".")[-1])
    assert key not in kernels._load_tune()
    picked = kernels.tune_one(*work[0])
    assert kernels._load_tune()[key] == picked and kernels._choose(x, layer) == picked
    assert [m for _, m in kernels.pending_work(model, max_m=3)] == [2, 3]
    assert (tmp_path / "tune.json").exists()
