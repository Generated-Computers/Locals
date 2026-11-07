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


def _old_version_file(tmp_path, monkeypatch, stale):
    """A one-file cache (``LOCALS_TUNE_CACHE``) last written under another MLX, with three timed shapes."""
    import json

    path = tmp_path / "tune.json"
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(path))
    monkeypatch.setattr(kernels, "_preset_path", lambda: None)
    monkeypatch.setattr(kernels, "_TUNE", None)
    data = {"__mlx__": "0.0.1", "1024x512x1": ["mlx"], "1024x512x2": ["mlx"], "1024x512xP": ["dense"]}
    if stale is not None:
        data["__stale__"] = stale
    path.write_text(json.dumps(data))
    return path


def test_a_version_change_marks_every_timed_shape_stale_whatever_the_file_listed(tmp_path, monkeypatch, capsys):
    """The save in the same load intersected the new marks with the file's ``__stale__`` list: a file that held one
    (even ``[]``) kept that list, so a real MLX upgrade marked nothing new, and the log line counted shapes that
    were not saved as stale."""
    import json

    for stale in (None, ["1024x512x2"], []):
        path = _old_version_file(tmp_path, monkeypatch, stale)
        tune = kernels._load_tune()
        every = ["1024x512x1", "1024x512x2", "1024x512xP"]
        assert tune["__stale__"] == every, stale
        saved = json.loads(path.read_text())
        assert saved["__stale__"] == every and saved["__mlx__"] == mx.__version__, stale
        assert "the 3 kernel shapes" in capsys.readouterr().err, stale  # (the count that was saved)


def test_two_saves_under_one_version_keep_a_shape_one_of_them_timed_again_off_the_stale_list(tmp_path, monkeypatch):
    """The other process timed 1024x512x2 again and saved; this one still lists it as stale when it saves."""
    import json

    path = tmp_path / "tune.json"
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(path))
    monkeypatch.setattr(kernels, "_PRESET", {})
    shapes = {"1024x512x1": ["mlx"], "1024x512x2": ["frag", 2, 2, 1]}
    path.write_text(json.dumps({"__mlx__": mx.__version__, **shapes, "__stale__": ["1024x512x1"]}))
    monkeypatch.setattr(kernels, "_TUNE", {"__mlx__": mx.__version__, "1024x512x1": ("mlx",), "1024x512x2": ("mlx",),
                                           "__stale__": ["1024x512x1", "1024x512x2"]})
    kernels._save_tune()
    assert json.loads(path.read_text())["__stale__"] == ["1024x512x1"]
    assert kernels._TUNE["__stale__"] == ["1024x512x1"]


def _version_dir(tmp_path, monkeypatch):
    """The default cache directory, as without ``LOCALS_TUNE_CACHE``: this MLX's file is absent."""
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / f"{kernels._TUNE_STEM}_mlx{mx.__version__}.json"))
    monkeypatch.setattr(kernels, "_TUNE_PATH_FROM_ENV", None)
    monkeypatch.setattr(kernels, "_preset_path", lambda: None)
    monkeypatch.setattr(kernels, "_TUNE", None)
    return kernels._TUNE_STEM


def test_a_new_mlx_version_seeds_its_file_from_the_old_shared_one_all_stale(tmp_path, monkeypatch, capsys):
    import json

    _version_dir(tmp_path, monkeypatch)
    old = tmp_path / kernels._TUNE_FILE
    old.write_text(json.dumps({"__mlx__": mx.__version__, "1024x512x1": ["frag", 2, 2, 1], "1024x512xP": ["dense"],
                               "__stale__": [], "__times__": {"1024x512x1": {"frag": 3.0}}}))
    before = old.read_bytes()
    tune = kernels._load_tune()
    assert tune["1024x512x1"] == ("frag", 2, 2, 1) and tune["1024x512xP"] == ("dense",)  # the choices stay
    assert tune["__stale__"] == ["1024x512x1", "1024x512xP"]  # (even one it says this MLX timed: a shared file)
    saved = json.loads(open(kernels._TUNE_PATH).read())
    assert saved["__stale__"] == ["1024x512x1", "1024x512xP"] and saved["__mlx__"] == mx.__version__
    assert saved["1024x512x1"] == ["frag", 2, 2, 1]
    assert old.read_bytes() == before
    assert "the 2 kernel shapes" in capsys.readouterr().err


def test_the_seed_is_the_other_version_file_written_last(tmp_path, monkeypatch):
    import json
    import os

    stem = _version_dir(tmp_path, monkeypatch)
    for i, (version, choice) in enumerate([("0.0.2", ["frag", 4, 2, 1]), ("0.0.1", ["mlx"]), (None, ["dense"])]):
        name = f"{stem}_mlx{version}.json" if version else f"{stem}.json"
        (tmp_path / name).write_text(json.dumps({"__mlx__": version or "0.0.0", "1024x512x1": choice}))
        os.utime(tmp_path / name, (1_000_000 + 100 * (3 - i),) * 2)  # 0.0.2's written last, the shared one first
    tune = kernels._load_tune()
    assert tune["1024x512x1"] == ("frag", 4, 2, 1) and tune["__stale__"] == ["1024x512x1"]


def test_a_seed_that_does_not_parse_is_passed_over_for_the_next_newest(tmp_path, monkeypatch):
    import json
    import os

    stem = _version_dir(tmp_path, monkeypatch)
    good, cut = tmp_path / f"{stem}.json", tmp_path / f"{stem}_mlx0.0.2.json"
    good.write_text(json.dumps({"__mlx__": "0.0.1", "1024x512x1": ["dense"]}))
    cut.write_text('{"1024x512x1": ["fr')  # (a write cut short)
    os.utime(good, (1_000_000,) * 2)
    tune = kernels._load_tune()
    assert tune["1024x512x1"] == ("dense",) and tune["__stale__"] == ["1024x512x1"]
    assert cut.read_text() == '{"1024x512x1": ["fr'  # (another version's file: not set aside from here)


def test_a_shape_pinned_after_two_crashes_is_never_marked_stale(tmp_path, monkeypatch):
    """A shape that stopped locals twice keeps MLX's kernel (_crashed_shapes). Marked stale, the idle tuner timed it
    again under the MLX that had aborted twice, and nothing took it off the list, so the server aborted at every
    start. Neither a version change, nor a seed, nor its second crash leaves it stale."""
    import json
    import subprocess

    pinned, timed = "1024x512x1", "1024x512x2"
    for seeded in (False, True):
        if seeded:
            _version_dir(tmp_path, monkeypatch)
            path = tmp_path / kernels._TUNE_FILE
        else:
            path = _old_version_file(tmp_path, monkeypatch, None)
        path.write_text(json.dumps({"__mlx__": "0.0.1", pinned: ["mlx"], timed: ["mlx"], "__crashed__": {pinned: 2}}))
        assert kernels._load_tune()["__stale__"] == [timed], seeded
    path = tmp_path / "tune.json"  # this MLX's own file lists a shape stale, and its second crash comes now
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(path))
    monkeypatch.setattr(kernels, "_TUNE_PATH_FROM_ENV", str(path))
    monkeypatch.setattr(kernels, "_TUNE", None)
    path.write_text(json.dumps({"__mlx__": mx.__version__, pinned: ["frag", 2, 2, 1], timed: ["mlx"],
                                "__stale__": [pinned, timed], "__crashed__": {pinned: 1}}))
    dead = subprocess.Popen(["true"])
    dead.wait()
    (tmp_path / f"tune.json.tuning-{dead.pid}").write_text(pinned)
    tune = kernels._load_tune()
    assert tune["__crashed__"] == {pinned: 2} and tune["__stale__"] == [timed]
    assert json.loads(path.read_text())["__stale__"] == [timed]


def test_a_seed_this_mlx_wrote_that_never_saw_a_version_change_keeps_its_shapes(tmp_path, monkeypatch, capsys):
    """One install, one MLX: the old shared file names this MLX and has no ``__stale__`` list (only a version change
    writes one), so every shape in it was timed under this MLX. The first start under versioned names re-timed them
    all."""
    import json

    _version_dir(tmp_path, monkeypatch)
    (tmp_path / kernels._TUNE_FILE).write_text(json.dumps({"__mlx__": mx.__version__, "1024x512x1": ["frag", 2, 2, 1]}))
    tune = kernels._load_tune()
    assert tune["1024x512x1"] == ("frag", 2, 2, 1) and not tune.get("__stale__")
    saved = json.loads(open(kernels._TUNE_PATH).read())  # (this MLX's file is written: the next start reads it)
    assert saved["1024x512x1"] == ["frag", 2, 2, 1] and not saved.get("__stale__")
    assert "kernel shapes" not in capsys.readouterr().err
