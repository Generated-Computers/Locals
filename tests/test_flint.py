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


@pytest.mark.parametrize("H,KVH", [(32, 4), (4, 2)])  # 8 query heads a key/value head (the matrix kernel), and 2
def test_8bit_attention_is_mlx_attention_over_the_same_keys(H, KVH):
    """qkv.attend against MLX's attention over the 8-bit keys and values unpacked, with sinks, left padding and a
    buffer larger than its valid part: the same to a last bit of the output."""
    from locals import qkv

    HD = 128
    sinks = (mx.random.normal((H,), key=mx.random.key(0)) * 2).astype(mx.bfloat16)
    for B, n, cap, pads in ((1, 1, 256, [0]), (1, 13, 256, [0]), (1, 2100, 2304, [0]), (3, 700, 768, [0, 37, 300])):
        q = mx.random.normal((B, H, 1, HD), key=mx.random.key(n)).astype(mx.bfloat16)
        kq = mx.quantize((mx.random.normal((B, KVH, cap, HD), key=mx.random.key(n + 1)) * 2).astype(mx.bfloat16), group_size=64, bits=8)
        vq = mx.quantize(mx.random.normal((B, KVH, cap, HD), key=mx.random.key(n + 2)).astype(mx.bfloat16), group_size=64, bits=8)
        start = mx.array(pads, dtype=mx.int32)
        got = qkv.attend(q, kq, vq, n, sinks, 0.0884, start)
        mask = mx.arange(n)[None, None, None, :] >= start[:, None, None, None]
        ref = mx.fast.scaled_dot_product_attention(q, qkv.dequantized(kq)[..., :n, :], qkv.dequantized(vq)[..., :n, :],
                                                   scale=0.0884, mask=mask, sinks=sinks)
        assert float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max()) <= 2 ** -7 * (1 + float(mx.abs(ref).max()))


def test_8bit_cache_in_a_batch_decodes_as_alone(monkeypatch):
    """Flint's full-attention layers in 8-bit caches (LOCALS_KV_BITS=8): two requests merged into a batch decode as
    each does alone, a third joins (extend), one leaves (filter), and a row taken out (extract) goes on as alone."""
    from mlx_lm.models.cache import make_prompt_cache

    import locals
    from locals.qkv import BatchQuantizedKVCache, QuantizedKVCache8

    monkeypatch.setenv("LOCALS_KV_BITS", "8")
    m = _model(seed=8)
    locals.patch(m, tune="off", kernels=False)
    assert isinstance(make_prompt_cache(m)[0], QuantizedKVCache8)
    seqs = [[int(t) for t in mx.random.randint(0, 512, (n,), key=mx.random.key(n)).tolist()] for n in (30, 13, 21)]

    def alone(s, steps):
        c = make_prompt_cache(m)
        mx.eval(m(mx.array([s]), cache=c))
        return c, [m(mx.array([[t]]), cache=c)[0, -1].astype(mx.float32) for t in steps]

    def close(a, b):
        return float(mx.abs(a - b).max()) <= 0.05 * (1 + float(mx.abs(b).max()))

    steps = list(range(1, 11))
    refs = [alone(s, steps)[1] for s in seqs]
    rows = []
    for s in seqs[:2]:
        c = make_prompt_cache(m)
        mx.eval(m(mx.array([s]), cache=c))
        rows.append(c)
    batch = [rows[0][i].merge([r[i] for r in rows]) for i in range(len(rows[0]))]
    assert isinstance(batch[0], BatchQuantizedKVCache)
    for i in range(5):
        out = m(mx.array([[steps[i]], [steps[i]]]), cache=batch)[:, -1].astype(mx.float32)
        assert close(out[0], refs[0][i]) and close(out[1], refs[1][i])
    third, _ = alone(seqs[2], steps[:5])  # the third request, prefilled alone up to here, joins
    joined = [third[i].merge([third[i]]) for i in range(len(third))]
    for c, j in zip(batch, joined):
        c.extend(j)
    for i in range(5, 8):
        out = m(mx.array([[steps[i]]] * 3), cache=batch)[:, -1].astype(mx.float32)
        assert all(close(out[r], refs[r][i]) for r in range(3))
    for c in batch:
        c.filter(mx.array([1, 2]))  # the first request is done
    out = m(mx.array([[steps[8]]] * 2), cache=batch)[:, -1].astype(mx.float32)
    assert close(out[0], refs[1][8]) and close(out[1], refs[2][8])
    back = [c.extract(1) for c in batch]  # the third request, back to a cache of its own
    assert isinstance(back[0], QuantizedKVCache8)
    assert close(m(mx.array([[steps[9]]]), cache=back)[0, -1].astype(mx.float32), refs[2][9])


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


@pytest.mark.parametrize("kv_bits", ["16", "8"])
def test_a_span_in_one_pass_is_its_tokens_one_at_a_time(kv_bits, monkeypatch):
    """One request alone gets a forced span (tool-call syntax, a typed call's parameter names) in one pass
    (HybridBatchEngine._solo): over Flint's batch caches, sliding-window rings past their window and full-attention
    layers in bf16 or 8 bits, that pass gives the logits one-token steps give and leaves the caches as they do. No
    drafts: a rejected draft would need those caches rewound."""
    from types import SimpleNamespace

    from mlx_lm.models.cache import BatchRotatingKVCache, make_prompt_cache

    import locals
    from locals.hybrid_batch import HybridBatchEngine

    monkeypatch.setenv("LOCALS_KV_BITS", kv_bits)
    m = _model(seed=9)
    locals.patch(m, tune="off", kernels=False)
    prompt = mx.array([[int(t) for t in mx.random.randint(0, 512, (30,), key=mx.random.key(3)).tolist()]])
    span = [11, 400, 23, 7, 7, 260, 3]

    def batch_of_one():
        c = make_prompt_cache(m)
        mx.eval(m(prompt, cache=c))
        return [x.merge([x]) for x in c]

    drip, one = batch_of_one(), batch_of_one()
    assert any(isinstance(c, BatchRotatingKVCache) for c in one)
    ref = mx.concatenate([m(mx.array([[t]]), cache=drip)[:, -1] for t in span]).astype(mx.float32)
    got = m(mx.array([span]), cache=one)[0].astype(mx.float32)
    assert float(mx.abs(got - ref).max()) <= 0.05 * (1 + float(mx.abs(ref).max()))
    for t in (5, 99, 300):  # one-token steps after the span, past the window again
        a, b = (m(mx.array([[t]]), cache=c)[0, -1].astype(mx.float32) for c in (drip, one))
        assert float(mx.abs(a - b).max()) <= 0.05 * (1 + float(mx.abs(a).max()))
    eng = HybridBatchEngine.__new__(HybridBatchEngine)
    eng.lm = SimpleNamespace(model=m)
    r = SimpleNamespace(controller=SimpleNamespace())
    assert eng._solo_ok(r, one) and not eng._solo_drafts_ok(one)


def test_banded_window_attention_is_the_masked_one(monkeypatch):
    """A prompt chunk in the sliding-window layers, a few queries at a time against the keys their window reaches
    (FusedFlintAttention._banded), gives the scores of the whole chunk under the window's mask: from an empty ring
    and from a full one."""
    from mlx_lm.models.cache import make_prompt_cache

    import locals
    from locals import fuse

    m = _model(seed=11)
    locals.patch(m, tune="off", kernels=False)
    toks = [[int(t) for t in mx.random.randint(0, 512, (n,), key=mx.random.key(n)).tolist()] for n in (12, 20)]

    def run(band):
        monkeypatch.setattr(fuse, "FLINT_BAND", band)
        c = make_prompt_cache(m)
        return [m(mx.array([t]), cache=c).astype(mx.float32) for t in toks]

    for got, ref in zip(run(4), run(0)):
        assert float(mx.abs(got - ref).max()) <= 1e-3 * (1 + float(mx.abs(ref).max()))


def test_flint_reads_prompts_in_longer_chunks():
    """Flint's prompt attention holds no chunk's scores (MLX's fused causal attention, bands in the window layers), so
    fuse_model lets the engine keep 2,048-token chunks where hybrid_batch.PREFILL_SCORES halves them."""
    import locals
    from locals.hybrid_batch import PREFILL_SCORES, prefill_chunk
    from locals.kernels import text_model

    m = _model(seed=2)
    locals.patch(m, tune="off", kernels=False)
    lm = text_model(m)
    assert lm.prefill_chunk_tokens == 2048 and lm.prefill_scores > PREFILL_SCORES
    assert prefill_chunk(16384, 1, 2048, lm.prefill_scores) == 2048 and prefill_chunk(16384, 1, 2048) == 512


def test_batch_rows_decode_as_alone_with_and_without_padding():
    """A batch of rows of different lengths (left-padded) keeps its mask; a batch whose rows are not padded (one row,
    or rows left after the longer one finished) attends without it (fuse._unpadded); both decode as each row alone."""
    from mlx_lm.models.cache import make_prompt_cache

    import locals
    from locals import fuse

    m = _model(seed=12)
    locals.patch(m, tune="off", kernels=False)
    seqs = [[int(t) for t in mx.random.randint(0, 512, (n,), key=mx.random.key(40 + n)).tolist()] for n in (26, 14)]
    steps = [3, 77, 401, 9]

    def alone(s):
        c = make_prompt_cache(m)
        mx.eval(m(mx.array([s]), cache=c))
        return c, [m(mx.array([[t]]), cache=c)[0, -1].astype(mx.float32) for t in steps]

    def close(a, b):  # (the same here; a padded row attending without its mask is 0.03 off)
        return float(mx.abs(a - b).max()) <= 2e-3 * (1 + float(mx.abs(b).max()))

    refs = [alone(s)[1] for s in seqs]
    rows = []
    for s in seqs:
        c = make_prompt_cache(m)
        mx.eval(m(mx.array([s]), cache=c))
        rows.append(c)
    batch = [rows[0][i].merge([r[i] for r in rows]) for i in range(len(rows[0]))]
    full = [c for c in batch if type(c).__name__ == "BatchKVCache"]
    assert full and not fuse._unpadded(full[0])  # the shorter row is padded
    for i in range(2):
        out = m(mx.array([[steps[i]], [steps[i]]]), cache=batch)[:, -1].astype(mx.float32)
        assert close(out[0], refs[0][i]) and close(out[1], refs[1][i])
    for c in batch:
        c.filter(mx.array([1]))  # the longer row is done: the other's padding is trimmed
    assert fuse._unpadded(full[0])
    for i in range(2, 4):
        assert close(m(mx.array([[steps[i]]]), cache=batch)[0, -1].astype(mx.float32), refs[1][i])


@pytest.mark.parametrize("L", [5, 6, 7, 8])
def test_a_pass_of_5_to_8_queries_padded_to_9_attends_as_unpadded(L):
    """fuse.attend_few pads 5 to 8 queries in front to 9 (MLX's slow path for 5 to 8): the real queries get what
    MLX's attention gives them unpadded, under the causal mask, a window's array mask (2-D and per row), or none."""
    from mlx_lm.models.base import create_causal_mask

    from locals.fuse import attend_few

    B, H, KVH, D, n = 2, 8, 2, 128, 40
    q = mx.random.normal((B, H, L, D), key=mx.random.key(L)).astype(mx.bfloat16)
    k = mx.random.normal((B, KVH, n, D), key=mx.random.key(10 + L)).astype(mx.bfloat16)
    v = mx.random.normal((B, KVH, n, D), key=mx.random.key(20 + L)).astype(mx.bfloat16)
    sinks = mx.random.normal((H,), key=mx.random.key(30)).astype(mx.bfloat16)
    window = create_causal_mask(L, n - L, window_size=9)
    for mask in ("causal", None, window, mx.broadcast_to(window[None, None], (B, 1, L, n))):
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.09, mask=mask, sinks=sinks).astype(mx.float32)
        got = attend_few(q, k, v, 0.09, mask, sinks).astype(mx.float32)
        assert got.shape == ref.shape
        assert float(mx.abs(got - ref).max()) <= 1e-2 * (1 + float(mx.abs(ref).max()))
