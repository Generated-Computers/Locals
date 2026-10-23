"""Flint (locals/models/flint.py, locals/moe.py, locals/qk_kernel.py): the model loads on an mlx-lm without it, and
its fused paths compute what the stock layers compute, on a small random Flint (no weights to download)."""

import importlib
import sys

import mlx.core as mx
import mlx.nn as nn
import pytest

from locals import moe
from locals.models import flint


def _args(**kw):
    a = dict(model_type="flint", hidden_size=256, num_hidden_layers=4, intermediate_size=512, num_attention_heads=4,
             num_key_value_heads=2, head_dim=128, num_experts=16, num_experts_per_tok=4, moe_intermediate_size=128,
             shared_expert_intermediate_size=128, mlp_only_layers=[0], sliding_window=9, rms_norm_eps=1e-6,
             vocab_size=512, layer_types=["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
             rope_parameters={"rope_theta": 10000.0})
    a.update(kw)
    return flint.ModelArgs(**a)


def _model(seed=0, bits=4, expert_bits=3):
    """A small Flint quantized as mlx-community's build is: experts at ``expert_bits``, the router unquantized."""
    mx.random.seed(seed)
    m = flint.Model(_args())
    m.update(nn.utils.tree_map(lambda p: (mx.random.normal(p.shape) * 0.05).astype(mx.bfloat16), m.parameters()))
    for layer in m.layers:  # norms near one, sinks of either sign
        for n in (layer.input_layernorm, layer.post_attention_layernorm, layer.self_attn.q_norm, layer.self_attn.k_norm):
            n.weight = (1 + n.weight).astype(mx.bfloat16)
        layer.self_attn.softmax_offset = mx.random.normal((4,)).astype(mx.bfloat16)

    def pred(path, mod):
        if path.endswith("mlp.gate") or not hasattr(mod, "to_quantized"):
            return False
        return {"group_size": 64, "bits": expert_bits} if "switch_mlp" in path else {"group_size": 64, "bits": bits}

    nn.quantize(m, class_predicate=pred)
    m.update(nn.utils.tree_map(lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p, m.parameters()))
    mx.eval(m.parameters())
    return m


def test_register_puts_flint_where_mlx_lm_imports_models():
    from locals.models import register

    saved = sys.modules.pop("mlx_lm.models.flint", None)
    try:
        register()
        mod = importlib.import_module("mlx_lm.models.flint")
        assert hasattr(mod, "Model") and hasattr(mod, "ModelArgs")
    finally:
        if saved is not None:
            sys.modules["mlx_lm.models.flint"] = saved


def test_route_topk_picks_the_reference_experts():
    for T in (1, 7, 300):
        logits = mx.random.normal((T, 256), key=mx.random.key(T)) * 3
        inds, w = moe.route_topk(logits, 6)
        ref_i, ref_w = flint.route(logits, 6)
        order, ref_order = mx.argsort(inds, axis=-1), mx.argsort(ref_i, axis=-1)
        assert mx.array_equal(mx.take_along_axis(inds, order, -1), mx.take_along_axis(ref_i, ref_order, -1).astype(mx.uint32))
        assert mx.allclose(mx.take_along_axis(w, order, -1), mx.take_along_axis(ref_w, ref_order, -1), atol=1e-6)


def test_router_logits_are_float32_sums():
    w = (mx.random.normal((256, 2048), key=mx.random.key(1)) * 0.02).astype(mx.bfloat16)
    for T in (1, 5, 16):
        x = mx.random.normal((T, 1, 2048), key=mx.random.key(T)).astype(mx.bfloat16)
        ref = x.astype(mx.float32) @ w.astype(mx.float32).T
        got = moe.router(x, w)
        assert got.dtype == mx.float32 and got.shape == (T, 1, 256)
        assert float(mx.abs(got - ref).max()) <= 1e-5 * float(mx.abs(ref).max())


@pytest.mark.parametrize("bits", [3, 4])
def test_expert_kernels_compute_what_switch_glu_does(bits):
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    def qsw(shape, key):
        m = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
        nn.Module.__init__(m)
        m.group_size, m.bits, m.mode, m.global_scale = 64, bits, "affine", None
        w, s, b = mx.quantize(mx.random.normal(shape, key=key) * 0.03, 64, bits)
        m.weight, m.scales, m.biases = w, s.astype(mx.bfloat16), b.astype(mx.bfloat16)
        return m

    E, D, I, K = 32, 512, 256, 4
    gu, dn = qsw((E, 2 * I, D), mx.random.key(2)), qsw((E, D, I), mx.random.key(3))
    for T in (1, 3, 16):
        x = mx.random.normal((T, D), key=mx.random.key(T)).astype(mx.bfloat16)
        inds, w = moe.route_topk(mx.random.normal((T, E), key=mx.random.key(10 + T)), K)
        shared = (mx.random.normal((T, D), key=mx.random.key(20 + T)) * 0.1).astype(mx.bfloat16)
        g_u = gu(mx.expand_dims(x, (-2, -3)), inds)
        h_ref = nn.silu(g_u[..., :I]) * g_u[..., I:]
        y_ref = (dn(h_ref, inds).squeeze(-2) * w[..., None]).sum(axis=-2).astype(mx.bfloat16) + shared
        h = moe.experts_up(x, inds, gu)
        y = moe.experts_down(h, inds, w, dn, shared)
        # the same up to summation order: a last-bit difference where the stock kernel adds in another order
        assert float(mx.abs(h.astype(mx.float32) - h_ref.reshape(T * K, I).astype(mx.float32)).max()) <= \
            2 ** -6 * float(mx.abs(h_ref).max())
        assert float(mx.abs(y.astype(mx.float32) - y_ref.astype(mx.float32)).max()) <= 2 ** -6 * float(mx.abs(y_ref).max())


def test_qk_norm_rope_is_mlx_norm_then_rope():
    from locals.qk_kernel import qk_norm_rope

    m = _model()
    for layer in m.layers[:2]:  # a full-attention layer (no RoPE) and a sliding one
        a = layer.self_attn
        a.head_dim = 128
        for B, L, off in ((1, 1, 300), (3, 1, mx.array([4, 0, 77])), (2, 6, 0)):
            qkv = (mx.random.normal((B, L, (4 + 2 * 2) * 128), key=mx.random.key(B * 10 + L)) * 2).astype(mx.bfloat16)
            q0 = a.q_norm(qkv[..., :512].reshape(B, L, 4, 128)).transpose(0, 2, 1, 3)
            k0 = a.k_norm(qkv[..., 512:768].reshape(B, L, 2, 128)).transpose(0, 2, 1, 3)
            if a.rope is not None:
                q0, k0 = a.rope(q0, offset=off), a.rope(k0, offset=off)
            q, k, v = qk_norm_rope(qkv, a, off)
            assert mx.array_equal(v, qkv[..., 768:].reshape(B, L, 2, 128).transpose(0, 2, 1, 3))
            tol = 0.02 * float(mx.abs(q0).max())  # (RoPE: a last-bit difference in some values, from the angle)
            assert float(mx.abs(q.astype(mx.float32) - q0.astype(mx.float32)).max()) <= tol
            assert float(mx.abs(k.astype(mx.float32) - k0.astype(mx.float32)).max()) <= tol


def test_mixed_pass_is_a_decode_step_and_a_prefill():
    """locals/mixed.py on Flint: decode rows and a prompt chunk in one pass give the rows what a decode step gives
    them, and leave the prompt's cache as reading the chunk alone does."""
    from mlx_lm.models.cache import make_prompt_cache

    import locals
    from locals import mixed

    m = _model(seed=7)
    locals.patch(m, tune="off", kernels=False)
    body = m.model
    assert mixed.supported(body)
    seqs = [[int(t) for t in mx.random.randint(0, 512, (n,), key=mx.random.key(n)).tolist()] for n in (20, 11)]
    prompt = [int(t) for t in mx.random.randint(0, 512, (14,), key=mx.random.key(3)).tolist()]

    def batch_cache():
        rows = []
        for s in seqs:
            c = make_prompt_cache(m)
            mx.eval(m(mx.array([s]), cache=c))
            rows.append(c)
        return [rows[0][i].merge([r[i] for r in rows]) for i in range(len(rows[0]))]

    xd = mx.array([[5], [6]])
    cd_ref, cd = batch_cache(), batch_cache()
    cp_ref, cp = make_prompt_cache(m), make_prompt_cache(m)
    for c in (cp_ref, cp):
        mx.eval(m(mx.array([prompt[:9]]), cache=c))
    ref_rows = body(xd, cd_ref)
    body(mx.array([prompt[9:]]), cp_ref)
    rows = mixed.forward(body, xd, cd, mx.array([prompt[9:]]), cp)
    assert float(mx.abs(rows.astype(mx.float32) - ref_rows.astype(mx.float32)).max()) <= 0.05 * float(mx.abs(ref_rows).max())
    nxt = mx.array([[prompt[-1]]])  # both prompt caches go on alike
    a, b = m(nxt, cache=cp_ref)[0, -1].astype(mx.float32), m(nxt, cache=cp)[0, -1].astype(mx.float32)
    assert float(mx.abs(a - b).max()) <= 0.05 * float(mx.abs(a).max()) + 0.05


def test_patched_flint_decodes_as_the_stock_one():
    """The fused model (q/k/v, router, experts, shared expert, q/k norm-rope) against the stock one, greedily, on
    a prompt pass and on decode steps (the one-token kernels)."""
    from mlx_lm.models.cache import make_prompt_cache

    import locals

    stock, fused = _model(seed=4), _model(seed=4)
    locals.patch(fused, tune="off", kernels=False)
    assert type(fused.layers[1].mlp).__name__.startswith("Fused") and hasattr(fused.layers[1].self_attn, "qkv_proj")
    prompt = mx.array([[5, 17, 300, 2, 9, 44, 120, 7, 7, 1, 499, 3]])
    ref, got = stock(prompt).astype(mx.float32), fused(prompt).astype(mx.float32)
    assert float(mx.abs(ref - got).max()) <= 0.05 * float(mx.abs(ref).max())
    caches = [make_prompt_cache(stock), make_prompt_cache(fused)]
    ys = [mx.argmax(m(prompt, cache=c)[:, -1:], axis=-1) for m, c in zip((stock, fused), caches)]
    same = 0
    for _ in range(12):  # past the sliding window of 9
        a, b = (m(y, cache=c) for m, y, c in zip((stock, fused), ys, caches))
        assert float(mx.abs(a - b).max()) <= 0.05 * float(mx.abs(a).max()) + 0.05
        ys = [mx.argmax(a[:, -1:], axis=-1)] * 2  # both go on from the stock token
        same += 1
    assert same == 12
