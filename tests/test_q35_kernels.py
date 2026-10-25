"""Qwen3.5 fusion and small-M kernels match the stock model (slow; skipped if
the model is not cached).

  * every kernel kind the autotuner can pick agrees with mx.quantized_matmul
  * the fused, kernel-routed model gives the same logits as the stock model
    for batch sizes 1, 3 and 8 (within bf16 noise) and the same greedy tokens
"""

import os
import pytest

MODEL = os.environ.get("LOCALS_TEST_MODEL", "mlx-community/Qwen3.5-2B-MLX-4bit")


_PAIR = {}  # one (stock, fused) pair at a time: two 4B pairs (4 x 2.6 GB) do not fit a 16 GB Mac's GPU


def _pair(dtype: str):
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    if dtype not in _PAIR:
        import gc

        import mlx.core as mx
        from mlx_lm import load

        from locals.runtime import set_compute_dtype

        _PAIR.clear()
        gc.collect()
        mx.clear_cache()
        pair = load(MODEL)[0], load(MODEL)[0]
        if dtype == "float16":
            for m in pair:
                assert set_compute_dtype(m, "float16") == "float16"
        _PAIR[dtype] = pair
    return _PAIR[dtype]


@pytest.fixture(scope="module", autouse=True)
def _release_pair():
    yield
    _PAIR.clear()


@pytest.fixture
def models():
    return _pair("bfloat16")


@pytest.fixture
def models_fp16():
    return _pair("float16")


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("M", [1, 2, 3, 5, 8, 11, 16])
def test_kernels_match_mlx(M, dtype):
    import mlx.core as mx
    import mlx.nn as nn

    from locals import qmv_frag, qmv_multi

    dt = getattr(mx, dtype)
    for N, K in [(512, 2048), (2048, 6144), (8224, 2048)]:
        lin = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=4)
        lin.set_dtype(dt)
        x = mx.random.normal((M, K)).astype(dt)
        ref = mx.quantized_matmul(x, lin.weight, lin.scales, lin.biases, transpose=True, group_size=64, bits=4).astype(mx.float32)
        tol = 0.02 * (1 + float(mx.max(mx.abs(ref))))
        outs = []
        for TN, NSG, SP in [(1, 2, 1), (2, 4, 1), (1, 2, 2)]:
            if qmv_frag.supported(M, K, N, TN, SP):
                outs.append(qmv_frag.qmv_frag(x, lin.weight, lin.scales, lin.biases, TN, NSG, SP))
                outs.append(qmv_frag.launch_h(M, K, N, x.dtype, TN, NSG, SP)(x, lin.weight, lin.scales, lin.biases))
        for TN, NSG in [(1, 4), (2, 8), (4, 4)]:
            if qmv_frag.supported_q(M, K, N, TN):
                outs.append(qmv_frag.launch_q(M, K, N, x.dtype, TN, NSG)(x, lin.weight, lin.scales, lin.biases))
        if M <= 6:
            for VPT, R, NSG in [(8, 8, 2), (16, 4, 2)]:
                outs.append(qmv_multi.launch_v3(M, K, N, x.dtype, VPT, R, NSG)(x, lin.weight, lin.scales, lin.biases))
        if M <= 4:
            for VPT, R, NSG in [(8, 4, 2), (16, 4, 2)]:
                if qmv_multi.supported(M, K, N, VPT, R):
                    outs.append(qmv_multi.qmv_multi(x, lin.weight, lin.scales, lin.biases, VPT, R, NSG))
        assert outs
        for y in outs:
            assert y.shape == (M, N)
            assert float(mx.max(mx.abs(y.astype(mx.float32) - ref))) <= tol


def test_fused_model_matches_stock(models):
    _fused_model_matches_stock(*models)


def test_fused_model_matches_stock_fp16(models_fp16):
    _fused_model_matches_stock(*models_fp16)


def _fused_model_matches_stock(stock, fused):
    """Logits within the stock model's own noise floor: stock decoding a batch
    of 8 against stock decoding the same rows one at a time (other matmul
    kernels). On Qwen3.5-4B in bf16 that floor reaches ~5 logits, against ~0.4
    in fp16, so a fixed tolerance tuned on 2B cannot hold."""
    import mlx.core as mx

    from locals.fuse import fuse_model
    from locals.kernels import enable_small_m, text_model

    assert fuse_model(fused) > 0
    enable_small_m(True, fused)  # the fused model's layers only: the stock model stays as mlx-lm makes it
    lm_s, lm_f = text_model(stock), text_model(fused)
    assert any(type(l.linear_attn).__name__.startswith("Fused") for l in lm_f.layers if l.is_linear)
    for B in [1, 3, 8]:
        prompt = mx.array([[(i * 7919 + 13 * b) % 20000 + 100 for i in range(40)] for b in range(B)])
        cs, cf = lm_s.make_cache(), lm_f.make_cache()
        rows = [lm_s.make_cache() for _ in range(B)]
        ls = lm_s(prompt, cache=cs)[:, -1]
        for b in range(B):
            lm_s(prompt[b : b + 1], cache=rows[b])
        lf = lm_f(prompt, cache=cf)[:, -1]
        toks = mx.argmax(ls, axis=-1)[:, None]
        errs, floors, flips = [], [], []
        for _ in range(6):
            ls = lm_s(toks, cache=cs)[:, -1].astype(mx.float32)
            lr = mx.concatenate([lm_s(toks[b : b + 1], cache=rows[b])[:, -1] for b in range(B)]).astype(mx.float32)
            lf = lm_f(toks, cache=cf)[:, -1].astype(mx.float32)
            errs.append(float(mx.max(mx.abs(ls - lf))))
            floors.append(float(mx.max(mx.abs(ls - lr))))
            top_s, top_f = mx.argmax(ls, axis=-1), mx.argmax(lf, axis=-1)
            gap = mx.sort(ls, axis=-1)[:, -1] - mx.sort(ls, axis=-1)[:, -2]
            # greedy choices agree wherever the stock top-2 margin exceeds the noise
            flips.append(bool(mx.any((top_s != top_f) & (gap >= max(0.25, 2 * floors[-1])))))
            scale = float(mx.max(mx.abs(ls)))
            toks = top_s[:, None]
        bound = max(0.05 * scale + 0.5, 1.5 * max(floors))
        assert max(errs) <= bound, (B, errs, floors)
        assert not any(flips), (B, errs, floors)


def test_gdn_kernel_matches_stock_layer(models):
    _gdn_kernel_matches_stock_layer(models[1])


def test_gdn_kernel_matches_stock_layer_fp16(models_fp16):
    _gdn_kernel_matches_stock_layer(models_fp16[1])


def _gdn_kernel_matches_stock_layer(fused):
    import mlx.core as mx

    from locals import fuse
    from locals.kernels import text_model

    fuse.fuse_model(fused)  # idempotent for already-fused layers
    lm = text_model(fused)
    layer = next(l for l in lm.layers if l.is_linear)
    lin = layer.linear_attn
    assert lin._gdn_ok
    B = 3
    ca, cb = lm.make_cache()[0], lm.make_cache()[0]
    dt = lm.model.norm.weight.dtype
    x = mx.random.normal((B, 5, lm.args.hidden_size)).astype(dt) * 0.5
    lin(x, cache=ca)
    lin(x, cache=cb)
    for step, S in enumerate([1, 3, 1, 2, 5, 8, 1]):
        xt = mx.random.normal((B, S, lm.args.hidden_size)).astype(dt) * 0.5
        fuse.USE_GDN_KERNEL = True
        ya = lin(xt, cache=ca)
        fuse.USE_GDN_KERNEL = False
        yb = lin(xt, cache=cb)
        fuse.USE_GDN_KERNEL = True
        for a, b in [(ya, yb), (ca[0], cb[0]), (ca[1], cb[1])]:
            a, b = a.astype(mx.float32), b.astype(mx.float32)
            err = float(mx.max(mx.abs(a - b)))
            assert err <= 0.03 * (1 + float(mx.max(mx.abs(b)))), (step, err)


def test_long_prefill_matches_stock(models):
    """Prompts of >= PREFILL_M tokens take the dequantize + fp16 GEMM path."""
    import mlx.core as mx

    from locals.fuse import fuse_model
    from locals.kernels import PREFILL_M, enable_small_m, text_model

    stock, fused = models
    fuse_model(fused)
    lm_s, lm_f = text_model(stock), text_model(fused)
    L = 2 * PREFILL_M + 37
    prompt = mx.array([[(i * 7919) % 20000 + 100 for i in range(L)] for _ in range(2)])
    enable_small_m(False)
    ls = lm_s(prompt, cache=lm_s.make_cache())[:, -1].astype(mx.float32)
    enable_small_m(True)
    lf = lm_f(prompt, cache=lm_f.make_cache())[:, -1].astype(mx.float32)
    enable_small_m(False)
    err = float(mx.max(mx.abs(ls - lf)))
    assert err < 0.05 * float(mx.max(mx.abs(ls))) + 0.5, err
    assert bool(mx.all(mx.argmax(ls, -1) == mx.argmax(lf, -1)))
