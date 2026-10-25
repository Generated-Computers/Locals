"""Fusion and kernel routing on tiny quantized layers (no model): projections fuse only when their
quantization matches, fusing does not draw from the global random state, routing changes only the
model it is given, and the tuning cache survives a cut-short file."""

import json

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from locals import fuse, kernels  # noqa: E402


def _q(out_dims, in_dims=128, bits=4, bias=False):
    lin = nn.Linear(in_dims, out_dims, bias=bias)
    return nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)


def test_mixed_quantization_does_not_fuse():
    assert fuse._fusable([_q(64), _q(32)])
    assert not fuse._fusable([_q(64), _q(32, bits=6)])  # mlx-lm's mixed recipes: v_proj at 6 bits
    assert not fuse._fusable([_q(64, bias=True), _q(32)])
    with pytest.raises(ValueError):
        fuse._concat_quantized([_q(64), _q(32, bits=8)])


def test_a_fused_layer_computes_what_the_parts_did_and_leaves_the_random_state():
    a, b = _q(64), _q(32)
    x = mx.random.normal((2, 128), key=mx.random.key(3))
    mx.random.seed(7)
    before = mx.random.uniform(shape=(4,))
    mx.random.seed(7)
    fused = fuse._concat_quantized([a, b])
    after = mx.random.uniform(shape=(4,))
    assert mx.array_equal(before, after)
    assert mx.allclose(fused(x), mx.concatenate([a(x), b(x)], axis=-1), atol=1e-4)


def test_routing_changes_only_the_model_given():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _q(64)

    mine, other = M(), M()
    kernels.enable_small_m(True, mine)
    assert type(mine.proj) is not nn.QuantizedLinear and isinstance(mine.proj, nn.QuantizedLinear)
    assert type(other.proj) is nn.QuantizedLinear and nn.QuantizedLinear.__call__ is kernels._orig_linear_call
    kernels.enable_small_m(False)  # the switch for the models patched so far (an A/B comparison)
    assert type(mine.proj).__call__ is kernels._linear_switch and kernels._enabled is False
    kernels.enable_small_m(True)


def test_a_cut_short_tuning_cache_is_set_aside(tmp_path, monkeypatch):
    path = tmp_path / "tune.json"
    path.write_text('{"1x64x2": ["mlx"], "2x64')
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(path))
    monkeypatch.setattr(kernels, "_TUNE", None)
    monkeypatch.setattr(kernels, "_preset_path", lambda: None)
    assert kernels._load_tune() == {"__mlx__": mx.__version__}  # (the version its timings are taken with)
    assert (tmp_path / "tune.json.bad").exists()
    kernels._TUNE["3x64x2"] = ("mlx",)
    kernels._save_tune()
    assert json.loads(path.read_text()) == {"3x64x2": ["mlx"], "__mlx__": mx.__version__}


def test_a_model_that_does_not_fit_fp16_gets_its_own_arrays_back(monkeypatch):
    from locals import runtime

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(2200, 8)
            self.out = nn.Linear(8, 4)

        def __call__(self, x):
            return self.out(self.embed(x))

    m = Tiny()
    m.embed.weight = mx.full((2200, 8), 300.0)
    m.out.weight = mx.full((4, 8), 300.0)  # 300 * 300 * 8 = 720k: inf in fp16
    m.out.bias = mx.array([1e-7, 0.0, 0.0, 0.0])  # below fp16's range: lost by a cast there and back
    m.set_dtype(mx.bfloat16)
    before = {k: mx.array(v) for k, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(m.parameters())}
    monkeypatch.setattr(runtime, "native_bf16", lambda: False)
    monkeypatch.setattr(runtime, "_fp16_special", lambda model: False)
    assert runtime.set_compute_dtype(m, "auto") == "bfloat16"
    from mlx.utils import tree_flatten

    for k, v in tree_flatten(m.parameters()):
        assert v.dtype == mx.bfloat16 and mx.array_equal(v, before[k]), k


def test_fusing_a_model_whose_layers_have_no_mlp():
    """LFM2's layers call their MLP feed_forward: loading it stopped at layer.mlp."""
    import mlx.nn as nn

    from locals.fuse import fuse_model

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.feed_forward = nn.Linear(4, 4)

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer(), Layer()]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Body()

    assert fuse_model(Model()) == 2


def test_head_rows_on_an_mxfp4_head():
    """Mellum2.1's MLX build is mxfp4: its head has no biases, and head_rows called affine's matmul."""
    import mlx.core as mx
    import mlx.nn as nn

    from locals.kernels import head_rows

    class LM(nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.lm_head = nn.Linear(64, 32, bias=False)
            nn.quantize(self, group_size=32, bits=4, mode=mode)

    for mode in ("affine", "mxfp4"):
        m = LM(mode)
        h = mx.random.normal((1, 1, 64))
        ids = mx.array([3, 7, 9])
        assert mx.allclose(head_rows(m, h, ids), m.lm_head(h)[0, -1][ids]), mode


def test_a_cached_nax_choice_where_nax_does_not_run_is_mlxs(tmp_path, monkeypatch):
    """A tune file with NAX choices on a GPU (or an MLX) without them: MLX's kernel, not a compile error in a request."""
    import mlx.core as mx
    import mlx.nn as nn

    from locals import kernels, nax_kernel

    monkeypatch.setattr(nax_kernel, "_AVAILABLE", False)
    monkeypatch.setattr(kernels, "_CHOICE", {})
    layer = nn.QuantizedLinear(256, 128, bits=4, group_size=64)
    dt = str(layer["scales"].dtype).split(".")[-1]
    monkeypatch.setattr(kernels, "_TUNE", {"128x256x20" + ("" if dt == "float16" else f"x{dt}"): ("nax", "r", 32, 1, 4, 8, 1)})
    assert kernels._choose(mx.zeros((20, 256), layer["scales"].dtype), layer) == ("mlx",)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_fragq_is_the_exact_sum_as_closely_as_mlxs_kernel(dtype):
    """qmv_frag.launch_q (the weights as 1 + q/16, the scale and bias per group in float) against the exact float32
    product of the inputs and the dequantized weights: no further from it than MLX's own kernel is, give or take a
    rounding of the output, at 1 to 16 rows, with and without whole tiles of rows."""
    from locals import qmv_frag

    dt = getattr(mx, dtype)
    for N, K in [(256, 512), (1040, 2048)]:
        w, s, b = mx.quantize((mx.random.normal((N, K), key=mx.random.key(N)) * 0.02).astype(dt), group_size=64, bits=4)
        s, b = s.astype(dt), b.astype(dt)
        deq = mx.dequantize(w, s, b, group_size=64, bits=4).astype(mx.float32)
        for M in (1, 3, 8, 9, 16):
            x = (mx.random.normal((M, K), key=mx.random.key(M)) * 0.5).astype(dt)
            exact = x.astype(mx.float32) @ deq.T
            ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4).astype(mx.float32)
            room = float(mx.abs(ref - exact).max()) + 2 ** -7 * float(mx.abs(exact).max())
            for TN, NSG in [(1, 4), (2, 8)]:
                assert qmv_frag.supported_q(M, K, N, TN)
                y = qmv_frag.launch_q(M, K, N, dt, TN, NSG)(x, w, s, b)
                assert y.shape == (M, N) and y.dtype == dt
                assert float(mx.abs(y.astype(mx.float32) - exact).max()) <= room


def test_a_shape_timed_before_a_kernel_family_is_timed_again(tmp_path, monkeypatch):
    """A shape whose timings have no fragq (timed before it existed) keeps its choice in requests, and the idle
    tuner times it again; once timed with it, it is not pending."""
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(tmp_path / "tune.json"))
    monkeypatch.setattr(kernels, "_preset_path", lambda: None)
    monkeypatch.setattr(kernels, "_TUNE", None)
    monkeypatch.setattr(kernels, "_CHOICE", {})
    monkeypatch.setattr(kernels, "_DEFER", True)
    layer = nn.QuantizedLinear(512, 1024, bits=4, group_size=64)
    layer.set_dtype(mx.bfloat16)
    key = "1024x512x8xbfloat16"
    f32 = "1024x512x8xfloat32"  # (activations fragq does not take: never stale for it)
    assert not kernels._missing_family({"__times__": {f32: {"mlx": 1.0}}}, f32)
    (tmp_path / "tune.json").write_text(json.dumps({"__mlx__": mx.__version__, key: ["frag", 2, 4, 1],
                                                    "__times__": {key: {"frag": 9.0, "mlx": 20.0}}}))
    tune = kernels._load_tune()
    assert not tune.get("__stale__") and kernels._missing_family(tune, key)
    assert kernels._choose(mx.zeros((8, 512), layer["scales"].dtype), layer) == ("frag", 2, 4, 1)  # kept in requests
    work = kernels.pending_work(nn.Sequential(layer), max_m=8)
    assert 8 in [m for _, m in work]
    kernels.tune_one(layer, 8)
    assert "fragq" in kernels._load_tune()["__times__"][key] and not kernels._missing_family(kernels._load_tune(), key)
    assert 8 not in [m for _, m in kernels.pending_work(nn.Sequential(layer), max_m=8)]
