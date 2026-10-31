"""Qwen3.5-MoE's fused paths (locals/moe.py: FusedQwenMoe, route_softmax, combine_gated; fuse.py: SplitProjection,
the layers' steps) and Clover's causal pass (locals/models/clover_moe.py: Model), on a small random Clover quantized
as its MLX build is (experts at 3 bits; the router, the shared expert's gate and the GDN a/b projections at 8): what
mlx-lm's Qwen3.5 layers compute, the expert choice and the combine bit for bit."""

import copy
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from locals import moe
from locals.models import clover_moe

CFG = dict(model_type="clover_moe", hidden_size=256, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
           head_dim=128, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=128,
           linear_value_head_dim=128, linear_conv_kernel_dim=4, num_experts=32, num_experts_per_tok=4,
           moe_intermediate_size=128, shared_expert_intermediate_size=128, full_attention_interval=4,
           layer_types=["linear_attention"] * 3 + ["full_attention"], rms_norm_eps=1e-6, vocab_size=512,
           mask_token_id=511, block_size=8,
           rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25})


def _randomize(m, dtype):
    m.update(nn.utils.tree_map(lambda p: (mx.random.normal(p.shape) * 0.1).astype(dtype), m.parameters()))
    for layer in m.layers:  # norms near one, the recurrence's decay in range
        for n in (layer.input_layernorm, layer.post_attention_layernorm):
            n.weight = (1 + n.weight).astype(dtype)
        if layer.is_linear:
            layer.linear_attn.A_log = mx.random.uniform(-1, 1, layer.linear_attn.A_log.shape)
        else:
            a = layer.self_attn
            a.q_norm.weight, a.k_norm.weight = (1 + a.q_norm.weight).astype(dtype), (1 + a.k_norm.weight).astype(dtype)
    m.model.norm.weight = (1 + m.model.norm.weight).astype(dtype)
    m.model.embed_tokens.weight = m.model.embed_tokens.weight * 10  # hidden states of a useful size


def _causal(seed=0):
    """Clover's causal pass, small, quantized as the MLX build: experts 3 bits, router / shared gate / a / b 8."""
    mx.random.seed(seed)
    m = clover_moe.Model(clover_moe.ModelArgs.from_dict(copy.deepcopy(CFG)))
    _randomize(m, mx.bfloat16)
    for layer in m.layers:  # a router as decisive as a trained one: few near-ties, which a last bit would flip
        layer.mlp.gate.weight = layer.mlp.gate.weight * 8

    def pred(path, mod):
        if not hasattr(mod, "to_quantized"):
            return False
        if path.endswith(("mlp.gate", "shared_expert_gate", "in_proj_a", "in_proj_b")):
            return {"group_size": 64, "bits": 8}
        return {"group_size": 64, "bits": 3 if "switch_mlp" in path else 4}

    nn.quantize(m, class_predicate=pred)
    m.update(nn.utils.tree_map(lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 and p.ndim > 1 else p, m.parameters()))
    mx.eval(m.parameters())
    return m


def _stock_choice(logits, k):
    g = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
    s = mx.take_along_axis(g, inds, axis=-1)
    return inds.astype(mx.uint32), (s / s.sum(axis=-1, keepdims=True)).astype(mx.float32)


@pytest.mark.parametrize("E,k", [(32, 4), (256, 8), (512, 10)])
def test_route_softmax_is_mlx_lms_choice(E, k):
    """The experts mlx-lm's Qwen3NextSparseMoeBlock picks, in its order (ascending probability; a tie at the k-th
    place to the higher index, as argpartition's stable sort leaves it), and its renormalized weights, bit for bit;
    with more logits fused after the router's (the shared expert's gate) as without."""
    for T in (1, 5, 16, 40):
        lg = (mx.random.normal((T, E + 1), key=mx.random.key(T + E)) * 3).astype(mx.bfloat16)
        ties = mx.argsort(-lg[:, :E], axis=-1)[:, k - 1 : k + 1]  # the k-th and k+1-th equal: a tie at the k-th place
        lg = mx.put_along_axis(lg, ties, mx.take_along_axis(lg, ties[:, :1], axis=-1), axis=-1) if T > 1 else lg
        want_i, want_w = _stock_choice(lg[:, :E], k)
        got_i, got_w = moe.route_softmax(lg, E, k)
        assert mx.array_equal(got_i, want_i) and mx.array_equal(got_w, want_w)
        inds, weights, order = moe.route_softmax(lg[:, :E], E, k, order=True)
        assert mx.array_equal(inds, want_i)
        flat = inds.reshape(-1)[order]
        assert sorted(order.tolist()) == list(range(T * k)) and all(a <= b for a, b in zip(flat.tolist(), flat.tolist()[1:]))


def test_route_softmax_names_real_experts_for_nan_logits():
    lg = mx.random.normal((3, 64), key=mx.random.key(1)).astype(mx.bfloat16)
    lg = lg.at[0, 5].add(mx.nan).at[1].add(mx.nan)
    inds, w = moe.route_softmax(lg, 64, 8)
    assert int(inds.max()) < 64 and len(set(inds[2].tolist())) == 8


def test_gated_combine_is_the_blocks_arithmetic():
    """combine_gated against mlx-lm's block on the same expert outputs: the weighted sum over the k (each product and
    partial sum in bf16), the shared expert times the sigmoid of its gate, the residual, the next norm: bit for bit."""
    T, K, D, E = 5, 4, 256, 32
    ys = (mx.random.normal((T * K, D), key=mx.random.key(2)) * 2).astype(mx.bfloat16)
    lg = (mx.random.normal((T, E + 1), key=mx.random.key(3)) * 3).astype(mx.bfloat16)
    _, w = moe.route_softmax(lg, E, K)
    shared = mx.random.normal((T, D), key=mx.random.key(4)).astype(mx.bfloat16)
    res = (mx.random.normal((T, D), key=mx.random.key(5)) * 8).astype(mx.bfloat16)
    norm = nn.RMSNorm(D, eps=1e-6)
    norm.weight = (1 + 0.2 * mx.random.normal((D,), key=mx.random.key(6))).astype(mx.bfloat16)
    want = (ys.reshape(T, K, D) * w.astype(mx.bfloat16)[..., None]).sum(axis=-2) + mx.sigmoid(lg[:, E : E + 1]) * shared
    assert mx.array_equal(moe.combine_gated(ys, w, shared, lg, E), want)
    assert mx.array_equal(moe.combine_gated(ys, w, shared, lg, E, residual=res), res + want)
    y, n = moe.combine_gated(ys, w, shared, lg, E, residual=res, norm=norm)
    assert mx.array_equal(y, res + want) and mx.array_equal(n, mx.fast.rms_norm(res + want, norm.weight, 1e-6))


def test_fused_block_is_mlx_lms_block():
    """FusedQwenMoe against the block it replaces: the same experts, the outputs within the last bits the expert
    kernels' sums in another order give; past EXPERT_MAX_ROWS rows (a prompt) the block's own computation exactly."""
    stock, fused = _causal(seed=1), _causal(seed=1)
    a, b = stock.layers[1].mlp, fused.layers[1].mlp
    assert moe.fuse_qwen_moe(b) and isinstance(b, moe.FusedQwenMoe) and "gates" in b
    for rows in (1, 4, 16, 24):
        x = (mx.random.normal((1, rows, 256), key=mx.random.key(rows)) * 3).astype(mx.bfloat16)
        want, got = a(x), b(x)
        if rows > moe.EXPERT_MAX_ROWS:
            assert mx.array_equal(got, want)
        else:
            err = float(mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max())
            assert err <= 0.02 * float(mx.abs(want).max()) + 1e-3
        res = (mx.random.normal(x.shape, key=mx.random.key(50 + rows)) * 4).astype(mx.bfloat16)
        assert mx.array_equal(b(x, residual=res), res + b(x)) if rows <= moe.EXPERT_MAX_ROWS else True


def test_patched_clover_decodes_as_the_stock_one():
    """locals.patch on Clover's causal pass (a/b quantized apart from qkv/z: SplitProjection and the GDN decode kernel;
    the fused MoE blocks; the layers' steps) against stock mlx-lm's Qwen3.5 layers: a prompt pass, then greedy
    decode steps, one row and three."""
    from mlx_lm.models.cache import make_prompt_cache

    import locals
    from locals import mixed
    from locals.fuse import FusedGatedDeltaNet, SplitProjection

    stock, fused = _causal(seed=4), _causal(seed=4)
    locals.patch(fused, tune="off", kernels=False)
    lin = fused.layers[0].linear_attn
    assert isinstance(lin, FusedGatedDeltaNet) and isinstance(lin.in_proj_all, SplitProjection) and lin._gdn_ok
    assert all(isinstance(l, moe.FusedQwenLayer) for l in fused.layers) and fused.model._moe_steps
    assert mixed.supported(fused.model)
    for B in (1, 3):
        prompt = mx.random.randint(0, 511, (B, 12), key=mx.random.key(B))
        ref, got = stock(prompt).astype(mx.float32), fused(prompt).astype(mx.float32)
        assert float(mx.abs(ref - got).max()) <= 0.05 * float(mx.abs(ref).max())
        caches = [make_prompt_cache(stock), make_prompt_cache(fused)]
        ys = [mx.argmax(m(prompt, cache=c)[:, -1:], axis=-1) for m, c in zip((stock, fused), caches)]
        same = 0
        for _ in range(12):
            a, b = (m(y, cache=c) for m, y, c in zip((stock, fused), ys, caches))
            assert float(mx.abs(a - b).max()) <= 0.05 * float(mx.abs(a).max()) + 0.05
            same += int(mx.array_equal(mx.argmax(a[:, -1], axis=-1), mx.argmax(b[:, -1], axis=-1)))
            ys = [mx.argmax(a[:, -1:], axis=-1)] * 2  # both go on from the stock tokens
        assert same >= 11  # (a random model's near-ties may differ once)


def test_layer_steps_are_the_layers_one_by_one():
    """The patched body (each residual add in the norm after it, the next layer's input norm in the expert combine)
    against the stock body's loop over the same patched layers: the same logits, bit for bit."""
    from mlx_lm.models.cache import make_prompt_cache

    import locals

    m = _causal(seed=6)
    locals.patch(m, tune="off", kernels=False)
    stock_body = next(c for c in type(m.model).__mro__ if c.__name__ == "Qwen3_5TextModel")
    prompt = mx.array([[5, 17, 300, 2, 9, 44, 120, 7]])
    caches = [make_prompt_cache(m), make_prompt_cache(m)]
    outs = [m.lm_head(stock_body.__call__(m.model, prompt, caches[0])), m(prompt, cache=caches[1])]
    assert mx.array_equal(*outs)
    y = mx.argmax(outs[0][:, -1:], axis=-1)
    for _ in range(6):
        outs = [m.lm_head(stock_body.__call__(m.model, y, caches[0])), m(y, cache=caches[1])]
        assert mx.array_equal(*outs)
        y = mx.argmax(outs[0][:, -1:], axis=-1)


def test_clover_loads_as_qwen3_5s_text_model_and_in_blocks_as_the_whole_model(tmp_path, monkeypatch):
    """mlx-lm's loader builds Clover's causal pass (mlx-lm's Qwen3.5 text model, the block-diffusion weights left
    out), whose logits are the whole model's commit pass; blockdiff.load builds the whole model, in blocks of 8."""
    from mlx_lm.models import qwen3_5
    from mlx_lm.utils import load_model

    from locals import blockdiff
    from locals.models import register

    mx.random.seed(3)
    whole = clover_moe.BlockModel(clover_moe.BlockArgs.from_dict(copy.deepcopy(CFG)))
    _randomize(whole, mx.float32)
    weights = dict(nn.utils.tree_flatten(whole.parameters()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(CFG))
    register()
    m, _ = load_model(tmp_path)
    assert type(m) is clover_moe.Model and isinstance(m, qwen3_5.TextModel)
    names = dict(nn.utils.tree_flatten(m.parameters()))
    assert not any(k.endswith(clover_moe.BLOCK_ONLY) for k in names) and len(names) == len(weights) - 3 * 4
    toks = mx.array([[1, 5, 9, 30, 7, 2, 2, 100]])
    np.testing.assert_allclose(np.array(m(toks)), np.array(whole(toks)), rtol=1e-4, atol=1e-4)
    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", lambda path, *a, **k: "tokenizer")
    b, tok, settings = blockdiff.load(str(tmp_path))
    assert type(b) is clover_moe.BlockModel and tok == "tokenizer" and settings.block_size == 8
    assert blockdiff.load(str(tmp_path), block_size=4)[2].block_size == 4


def test_server_decodes_in_blocks_only_for_a_block_diffusion_model(tmp_path):
    from locals.server import Server

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    with pytest.raises(ValueError, match="not a block-diffusion model"):
        Server(str(tmp_path), engines=("locals",), preload=(), blocks=8, log=lambda s: None)


def test_another_qwen3_5_moe_keeps_its_paths():
    """Clover's fast paths are Clover's only: the same network as mlx-lm's own qwen3_5_moe (another Qwen3.5-MoE build)
    keeps its MoE blocks and layers, its GDN layers unfused where b and a are quantized apart, and its layers unmarked
    (main's tuning keys, MLX's kernel for its 8-bit output layer)."""
    from mlx_lm.models import qwen3_5_moe

    import locals
    from locals import fuse

    text = {k: v for k, v in copy.deepcopy(CFG).items() if k not in ("layer_types", "mask_token_id", "block_size")}
    other = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs(model_type="qwen3_5_moe", text_config=dict(text, model_type="qwen3_5_moe_text")))
    mx.random.seed(2)
    _randomize(other.language_model, mx.bfloat16)

    def pred(path, mod):
        if not hasattr(mod, "to_quantized"):
            return False
        if path.endswith(("mlp.gate", "shared_expert_gate", "in_proj_a", "in_proj_b", "lm_head")):
            return {"group_size": 64, "bits": 8}
        return {"group_size": 64, "bits": 3 if "switch_mlp" in path else 4}

    nn.quantize(other, class_predicate=pred)
    locals.patch(other, tune="off", kernels=False)
    body = other.language_model.model
    assert all(type(l.mlp).__name__ == "Qwen3NextSparseMoeBlock" and not isinstance(l, moe.FusedQwenLayer) for l in body.layers)
    assert all(hasattr(l.linear_attn, "in_proj_qkv") for l in body.layers if l.is_linear)  # (as main: not fused)
    assert not getattr(body, "_moe_steps", False)
    assert not any(vars(m).get("_locals_direct") or vars(m).get("_locals_head8") for _, m in other.named_modules())
    clover = _causal(seed=2)
    locals.patch(clover, tune="off", kernels=False)
    lin = clover.layers[0].linear_attn
    assert isinstance(lin.in_proj_all, fuse.SplitProjection) and lin._gdn_exact
    assert vars(clover.lm_head).get("_locals_head8") and vars(clover.layers[0].mlp.gates).get("_locals_direct") == "clover"
