"""Clover (locals/models/clover_moe.py, locals/blockdiff.py) on a small random model (no weights to download):
its preview ops against NumPy ports of DiffuSpace's Hugging Face code (modeling_clover_moe.py,
sampling_clover_moe.py), its commit pass against mlx-lm's Qwen3.5-MoE, and the caches a preview must leave."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from locals import blockdiff
from locals.models import clover_moe


def _args(**kw):
    a = dict(model_type="clover_moe", hidden_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
             head_dim=64, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=32,
             linear_value_head_dim=32, linear_conv_kernel_dim=4, num_experts=16, num_experts_per_tok=4,
             moe_intermediate_size=64, shared_expert_intermediate_size=64,
             layer_types=["linear_attention"] * 3 + ["full_attention"], rms_norm_eps=1e-6, vocab_size=256,
             mask_token_id=255, block_size=8, rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.25})
    a.update(kw)
    return clover_moe.BlockArgs(**a)


def _model(seed=0, dtype=mx.float32, **kw):
    mx.random.seed(seed)
    m = clover_moe.BlockModel(_args(**kw))
    m.update(nn.utils.tree_map(lambda p: (mx.random.normal(p.shape) * 0.1).astype(dtype), m.parameters()))
    for layer in m.layers:
        for n in (layer.input_layernorm, layer.post_attention_layernorm):
            n.weight = (1 + n.weight).astype(dtype)
        if layer.is_linear:
            g = layer.linear_attn
            g.A_log = mx.random.uniform(-1, 1, g.A_log.shape)
            g.bd_softmax_logit_scale = mx.random.uniform(5, 15, g.bd_softmax_logit_scale.shape)
            g.bd_softmax_history_bias = mx.random.uniform(-2, 6, g.bd_softmax_history_bias.shape)
        else:
            a = layer.self_attn
            a.q_norm.weight, a.k_norm.weight = (1 + a.q_norm.weight).astype(dtype), (1 + a.k_norm.weight).astype(dtype)
    m.model.norm.weight = (1 + m.model.norm.weight).astype(dtype)
    m.model.embed_tokens.weight = m.model.embed_tokens.weight * 10  # hidden states of a useful size
    mx.eval(m.parameters())
    return m


def _silu(x):
    return x / (1 + np.exp(-x))


def _softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def ref_conv(history, current, w, right):
    """bd_gdn_conv1d for one block a request, all positions valid, in HF's layouts: history (B, C, K-1),
    current (B, L, C), conv weight (C, 1, K), right weight (C, 1, K-1)."""
    B, L, C = current.shape
    K = w.shape[-1]
    x = np.concatenate([history, current.transpose(0, 2, 1)], axis=-1)
    out = np.zeros((B, C, L), dtype=np.float64)
    for t in range(L):
        out[:, :, t] = (x[:, :, t : t + K] * w[None, :, 0, :]).sum(-1)
    for d in range(1, min(K, L)):
        out[:, :, :-d] += current[:, d:, :].transpose(0, 2, 1) * right[None, :, 0, d - 1, None]
    return _silu(out).transpose(0, 2, 1)


def ref_memory(q, k, v, g, state, scale, logit_scale, history_bias, history_c, block_size=32):
    """bd_gdn_local_memory, all positions valid: q, k (B, L, Hk, Dk), v (B, L, Hv, Dv), g (B, L, Hv) (log decay),
    state (B, Hv, Dv, Dk)."""
    B, L, Hv = v.shape[:3]
    q = np.repeat(q, Hv // q.shape[2], axis=2)
    k = np.repeat(k, Hv // k.shape[2], axis=2)
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    history = np.einsum("blhk,bhvk->blhv", q * scale, state) * np.exp(g)[..., None]
    logits = np.einsum("blhk,bshk->bhls", q, k) * logit_scale[None, :, None, None]
    bucket = history_bias + history_c * math.log(L / block_size)
    bucket = np.broadcast_to(bucket[None, :, None, None], (B, Hv, L, 1))
    w = _softmax(np.concatenate([logits, bucket], axis=-1))
    local = np.einsum("bhls,bshv->blhv", w[..., :-1], v)
    return w[..., -1].transpose(0, 2, 1)[..., None] * history + local


def ref_coreset(logits, top_k, top_p):
    """coreset_torch, all positions valid: logits (B, L, E)."""
    B, L, E = logits.shape
    ids = np.argsort(-logits, axis=-1, kind="stable")[..., :top_k]
    scores = np.take_along_axis(_softmax(logits), ids, axis=-1)
    totals = np.zeros((B, E))
    for b in range(B):
        np.add.at(totals[b], ids[b].reshape(-1), scores[b].reshape(-1))
    probs = totals / np.maximum(totals.sum(-1, keepdims=True), 1e-9)
    order = np.argsort(-probs, axis=-1, kind="stable")
    sp = np.take_along_axis(probs, order, axis=-1)
    keep = (np.cumsum(sp, -1) - sp) < top_p
    keep[:, 0] = True
    allowed = np.zeros_like(keep)
    np.put_along_axis(allowed, order, keep, axis=-1)
    return np.where(allowed[:, None, :], logits, -np.inf)


def test_preview_conv_is_the_reference_right_convolution():
    m = _model()
    g = m.layers[0].linear_attn
    rng = np.random.default_rng(0)
    for L in (8, 3, 1):
        hist = rng.normal(size=(2, 3, g.conv_dim)).astype(np.float32)
        cur = rng.normal(size=(2, L, g.conv_dim)).astype(np.float32)
        got = np.array(g._preview_conv(mx.array(hist), mx.array(cur)))
        w = np.array(g.conv1d.weight).transpose(0, 2, 1)  # MLX (C, K, 1) -> HF (C, 1, K)
        want = ref_conv(hist.transpose(0, 2, 1), cur, w, np.array(g.conv_right_weight))
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_preview_memory_is_the_reference_softmax_with_a_history_bucket():
    m = _model()
    g = m.layers[0].linear_attn
    rng = np.random.default_rng(1)
    for L in (8, 4):  # 4: the bucket's log(L / block_size) term
        q = rng.normal(size=(1, L, g.num_k_heads, g.head_k_dim)).astype(np.float32)
        k = rng.normal(size=(1, L, g.num_k_heads, g.head_k_dim)).astype(np.float32)
        v = rng.normal(size=(1, L, g.num_v_heads, g.head_v_dim)).astype(np.float32)
        a = rng.normal(size=(1, L, g.num_v_heads)).astype(np.float32)
        state = rng.normal(size=(1, g.num_v_heads, g.head_v_dim, g.head_k_dim)).astype(np.float32)
        got = np.array(g._preview_memory(mx.array(q), mx.array(k), mx.array(v), mx.array(a), mx.array(state)))
        logg = -np.exp(np.array(g.A_log)) * np.log1p(np.exp(a + np.array(g.dt_bias)))
        want = ref_memory(q, k, v, logg, state, g.head_k_dim**-0.5, np.array(g.bd_softmax_logit_scale),
                          np.array(g.bd_softmax_history_bias), np.array(g.bd_softmax_history_c), block_size=8)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("top_p", [0.9, 0.5, 0.999])
def test_block_coreset_keeps_the_experts_the_reference_keeps(top_p):
    rng = np.random.default_rng(2)
    logits = (rng.normal(size=(2, 8, 16)) * 2).astype(np.float32)
    got = np.array(clover_moe.block_coreset(mx.array(logits), 4, top_p))
    want = ref_coreset(logits.astype(np.float64), 4, top_p)
    assert (np.isinf(got) == np.isinf(want)).all()
    np.testing.assert_array_equal(got[~np.isinf(got)], logits[~np.isinf(want)])
    assert (~np.isinf(got[0, 0])).sum() < 16  # it does cut


def test_commit_pass_is_mlx_lms_qwen3_5_moe():
    """Clover's causal pass is Qwen3.5-MoE's forward: the same weights give the same logits."""
    from mlx_lm.models import qwen3_5_moe

    m = _model()
    a = m.args
    text = dict(model_type="qwen3_5_moe_text", hidden_size=a.hidden_size, num_hidden_layers=a.num_hidden_layers,
                num_attention_heads=a.num_attention_heads, num_key_value_heads=a.num_key_value_heads, head_dim=a.head_dim,
                linear_num_value_heads=a.linear_num_value_heads, linear_num_key_heads=a.linear_num_key_heads,
                linear_key_head_dim=a.linear_key_head_dim, linear_value_head_dim=a.linear_value_head_dim,
                linear_conv_kernel_dim=a.linear_conv_kernel_dim, num_experts=a.num_experts,
                num_experts_per_tok=a.num_experts_per_tok, moe_intermediate_size=a.moe_intermediate_size,
                shared_expert_intermediate_size=a.shared_expert_intermediate_size, rms_norm_eps=a.rms_norm_eps,
                vocab_size=a.vocab_size, full_attention_interval=4, intermediate_size=64,
                rope_parameters={"rope_type": "default", "rope_theta": a.rope_theta, "partial_rotary_factor": 0.25})
    q = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs(model_type="qwen3_5_moe", text_config=text))
    ours = dict(nn.utils.tree_flatten(m.parameters()))
    theirs = {k: v for k, v in nn.utils.tree_flatten(q.parameters())}
    load = {k: ours[k[len("language_model."):]] for k in theirs}
    q.load_weights(list(load.items()))
    toks = mx.array([[1, 5, 9, 30, 7, 2, 2, 100, 41, 3, 77]])
    want = q(toks)
    h = m.model(toks)
    got = m.lm_head(h)
    np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-3, atol=1e-3)
    # and with caches, a piece at a time
    ca, cb = m.make_cache(), q.make_cache()
    for lo, hi in ((0, 4), (4, 5), (5, 11)):
        got = m.lm_head(m.model(toks[:, lo:hi], ca))
        want = q(toks[:, lo:hi], cache=cb)
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-3, atol=1e-3)


def _snapshot(cache):
    out = []
    for c in cache:
        if hasattr(c, "keys"):
            out.append((c.offset, np.array(c.keys[..., : c.offset, :]), np.array(c.values[..., : c.offset, :])))
        else:
            out.append(tuple(np.array(a) for a in c.cache))
    return out


def test_a_preview_leaves_the_caches_and_the_commit_after_it_is_the_commit_without():
    m = _model()
    prompt = mx.array([[1, 5, 9, 30, 7, 2, 2, 100, 41, 3, 77]])
    block = mx.array([[255, 4, 255, 255, 8, 255, 255, 255]])
    resolved = mx.array([[6, 4, 12, 13, 8, 2, 19, 60]])
    ca, cb = m.make_cache(), m.make_cache()
    m.model(prompt, ca), m.model(prompt, cb)
    before = _snapshot(ca)
    for blk in (block, resolved):
        mx.eval(m(blk, ca, preview=True))
        after = _snapshot(ca)
        for x, y in zip(before, after):
            for u, v in zip(x, y):
                np.testing.assert_array_equal(u, v)
    ha, hb = m.model(resolved, ca), m.model(resolved, cb)
    np.testing.assert_array_equal(np.array(ha), np.array(hb))
    for x, y in zip(_snapshot(ca), _snapshot(cb)):
        for u, v in zip(x, y):
            np.testing.assert_array_equal(u, v)


def test_preview_attention_sees_the_prefix_and_the_whole_block():
    m = _model()
    layer = m.layers[3]
    attn = layer.self_attn
    rng = np.random.default_rng(3)
    cache = m.make_cache()[3]
    prefix = mx.array(rng.normal(size=(1, 5, 128)).astype(np.float32))
    blk = mx.array(rng.normal(size=(1, 8, 128)).astype(np.float32))
    attn(prefix, "causal", cache)
    got = attn(blk, None, cache, preview=True)
    assert cache.offset == 5
    # by hand: every block query against the 5 prefix keys and all 8 block keys, no mask
    x = mx.concatenate([prefix, blk], axis=1)
    h, kv, hd = attn.num_attention_heads, attn.num_key_value_heads, attn.head_dim
    q, gate = mx.split(attn.q_proj(x).reshape(1, 13, h, -1), 2, axis=-1)
    k = attn.k_norm(attn.k_proj(x).reshape(1, 13, kv, -1)).transpose(0, 2, 1, 3)
    v = attn.v_proj(x).reshape(1, 13, kv, -1).transpose(0, 2, 1, 3)
    q = attn.rope(attn.q_norm(q).transpose(0, 2, 1, 3))[:, :, 5:]
    k = mx.repeat(attn.rope(k), h // kv, axis=1)
    v = mx.repeat(v, h // kv, axis=1)
    p = mx.softmax((q @ k.swapaxes(-1, -2)) * attn.scale, axis=-1)
    out = (p @ v).transpose(0, 2, 1, 3).reshape(1, 8, -1) * mx.sigmoid(gate[:, 5:].reshape(1, 8, -1))
    np.testing.assert_allclose(np.array(got), np.array(attn.o_proj(out)), rtol=1e-4, atol=1e-4)


def ref_penalties(logits, counts, length, block, mask_id, s, stop_ids):
    """CloverPenaltyState.apply on a [block, vocab] array."""
    logits = logits.copy()
    seen = counts > 0
    if s.frequency_penalty:
        logits -= counts * s.frequency_penalty
    if s.presence_penalty:
        logits -= seen * s.presence_penalty
    if s.repetition_penalty != 1:
        vals = logits[:, seen]
        logits[:, seen] = np.where(vals < 0, vals * s.repetition_penalty, vals / s.repetition_penalty)
    pos = length + np.arange(1, logits.shape[0] + 1)

    def ramp(start, window, strength):
        return np.clip((pos - start) / window, 0, 1) * strength

    if s.length_penalty and stop_ids:
        logits[:, stop_ids] += ramp(s.length_penalty_start, s.length_penalty_window, s.length_penalty)[:, None]
    if s.think_end_bias:
        close = s.think_end_token_id
        visible = ((block != mask_id) & (block == close)).any()
        if not (seen[close] or visible):
            logits[:, close] += ramp(s.think_end_start, s.think_end_window, s.think_end_bias)
    return logits


@pytest.mark.parametrize("closed", [False, True])
def test_penalties_are_the_references(closed):
    s = blockdiff.Settings(presence_penalty=0.9, frequency_penalty=0.3, repetition_penalty=1.05, length_penalty=10.0,
                           length_penalty_start=5, length_penalty_window=16, think_end_bias=10.0, think_end_start=10,
                           think_end_window=8, think_end_token_id=7)
    rng = np.random.default_rng(4)
    V, B = 64, 8
    pen = blockdiff.Penalties(s, V, [3, 5])
    history = [1, 2, 2, 9, 40, 40, 40] + ([7] if closed else [])
    pen.commit(history[:4])
    pen.commit(history[4:])
    counts = np.bincount(history, minlength=V).astype(np.float64)
    block = np.array([63, 12, 63, 63, 4, 63, 63, 63])
    logits = rng.normal(size=(B, V)) * 3
    want = ref_penalties(logits, counts, len(history), block, 63, s, [3, 5])
    rows = [0, 2, 3, 5, 6, 7]
    got = pen.apply(mx.array(logits[rows].astype(np.float32)), rows, block.tolist(), 63)
    np.testing.assert_allclose(np.array(got), want[rows], rtol=1e-5, atol=1e-5)


class _Detok:
    def reset(self):
        self.tokens, self.last = [], 0

    def add_token(self, t):
        self.tokens.append(t)

    def finalize(self):
        pass

    @property
    def last_segment(self):
        seg = " ".join(map(str, self.tokens[self.last:]))
        self.last = len(self.tokens)
        return seg


class _Tok:
    eos_token_ids = {254}

    def __init__(self):
        self.detokenizer = _Detok()


def test_a_threshold_of_one_resolves_a_position_a_step_and_zero_the_block_at_once():
    m = _model()
    prompt = [1, 5, 9, 30, 7, 2]
    one = list(blockdiff.stream_generate(m, _Tok(), prompt, blockdiff.Settings(max_tokens=16, block_size=8, threshold=1.0,
                                                                              eos_token_ids=[254])))
    assert sum(len(r.tokens) for r in one) == one[-1].generation_tokens
    assert all(r.steps == len(r.tokens) or r.finish_reason == "stop" for r in one)
    assert one[-1].finish_reason in ("stop", "length")
    zero = list(blockdiff.stream_generate(m, _Tok(), prompt, blockdiff.Settings(max_tokens=16, block_size=8, threshold=0.0,
                                                                               eos_token_ids=[254])))
    assert all(r.steps == 1 for r in zero)
    for r in one + zero:
        assert 255 not in r.tokens  # never the mask token


def test_greedy_blocks_are_what_a_preview_of_the_block_as_it_stands_gives():
    """Each resolved position took the argmax of a preview over the block as it stood: replaying the steps by hand
    gives the same tokens."""
    m = _model(seed=3)
    prompt = [1, 5, 9, 30, 7, 2]
    s = blockdiff.Settings(max_tokens=8, block_size=8, threshold=0.5, eos_token_ids=[254])
    (r,) = list(blockdiff.stream_generate(m, _Tok(), prompt, s))
    cache = m.make_cache()
    m.model(mx.array(prompt)[None], cache)
    block, pending = [255] * 8, list(range(8))
    while pending:
        logits = np.array(m(mx.array(block)[None], cache, preview=True)[0].astype(mx.float32))
        logits[:, 255] = -np.inf
        p = _softmax(logits[pending])
        tok, conf = p.argmax(-1), p.max(-1)
        take = [j for j in range(len(pending)) if conf[j] > 0.5] or [int(conf.argmax())]
        for j in take:
            block[pending[j]] = int(tok[j])
        pending = [q for j, q in enumerate(pending) if j not in take]
    kept = block[: block.index(254) + 1] if 254 in block else block
    assert r.tokens == kept


def _quantized(seed=0, expert_bits=3):
    m = _model(seed, dtype=mx.bfloat16)
    nn.quantize(m, class_predicate=lambda p, mod: hasattr(mod, "to_quantized") and (
        {"group_size": 64, "bits": expert_bits} if "switch_mlp" in p else {"group_size": 64, "bits": 4}))
    m.eval()
    return m


@pytest.mark.parametrize("bits", [3, 4])
def test_tile_kernels_are_moe_pys_bits(bits):
    from locals import moe, moe_tiles

    m = _quantized(expert_bits=bits)
    sw = m.layers[0].mlp.switch_mlp
    gu = moe._concat_switch([sw.gate_proj, sw.up_proj])
    rng = np.random.default_rng(5)
    for T, E in ((32, 6), (32, 16), (8, 16), (1, 16)):  # rows, experts they choose among (runs long and short)
        x = mx.array(rng.normal(size=(T, 128)).astype(np.float32)).astype(mx.bfloat16)
        inds = mx.array(np.stack([rng.permutation(E)[:4] for _ in range(T)]).astype(np.uint32))
        w = mx.softmax(mx.array(rng.normal(size=(T, 4)).astype(np.float32)), axis=-1)
        order = moe.expert_order(inds)
        want_h = moe.experts_up(x, inds, gu, order, exact=True)  # (MLX's sigmoid, as Clover's kernels round)
        want = moe.experts_down(want_h, inds, w, sw.down_proj, None, order)
        for tile in (1, 3, 4, 8):
            tiles = moe_tiles.tile_lengths(inds, order, tile)
            h = moe_tiles.experts_up(x, inds, gu, order, tiles, tile=tile)
            assert mx.array_equal(h, want_h), (T, E, tile)
            assert mx.array_equal(moe_tiles.experts_down(h, inds, w, sw.down_proj, None, order, tiles, tile=tile), want)


def test_tile_lengths_cut_each_run_in_tiles():
    from locals import moe_tiles

    inds = mx.array([[0, 2], [2, 5], [2, 0], [2, 7], [2, 9], [2, 3]], dtype=mx.uint32)  # expert 2: a run of 6
    order = mx.argsort(inds.reshape(-1)).astype(mx.uint32)
    se = np.array(inds.reshape(-1))[np.array(order)]
    n = np.array(moe_tiles.tile_lengths(inds, order, 4))
    assert n.sum() == se.size and (n <= 4).all()
    for p in np.nonzero(n)[0]:
        assert (se[p : p + n[p]] == se[p]).all()  # a tile is one expert's


def test_fused_moe_is_the_modules():
    from locals.fuse import fuse_model

    m = _quantized()
    ref = _quantized()
    fuse_model(m)
    assert type(m.layers[0].mlp).__name__ == "FusedSparseMoe"
    assert type(m.layers[0].linear_attn).__name__ == "FusedCloverGatedDeltaNet" and hasattr(m.layers[0].linear_attn, "in_proj_qkvz")
    assert type(m.layers[3].self_attn).__name__ == "FusedCloverAttention"
    assert hasattr(m.layers[0].mlp.shared_expert, "gate_up_proj")
    rng = np.random.default_rng(6)
    for L in (32, 8, 100):  # 100: the prompt path
        x = mx.array(rng.normal(size=(1, L, 128)).astype(np.float32)).astype(mx.bfloat16)
        for preview in (True, False):
            got = m.layers[1].mlp(x, preview)
            want = ref.layers[1].mlp(x, preview)
            np.testing.assert_allclose(np.array(got.astype(mx.float32)), np.array(want.astype(mx.float32)), rtol=0.02, atol=0.02)


@pytest.mark.parametrize("coreset", [None, 0.9, 0.5])
def test_route_kernel_is_the_modules_routing(coreset):
    from locals import clover_fast, moe_tiles

    rng = np.random.default_rng(7)
    for R in (32, 7, 1):
        logits = mx.array((rng.normal(size=(R, 64)) * 2).astype(np.float32)).astype(mx.bfloat16)
        inds, w, order, tiles = clover_fast.clover_route(logits, 4, coreset, 4)
        lf = logits.astype(mx.float32)[None]
        if coreset is not None:
            lf = clover_moe.block_coreset(lf, 4, coreset)
        want = np.argsort(-np.array(lf[0]), axis=-1, kind="stable")[:, :4]
        np.testing.assert_array_equal(np.array(inds), want)  # in order of logit, ties to the lower expert
        ww = np.array(mx.softmax(mx.take_along_axis(lf[0], mx.array(want), axis=-1), axis=-1))
        np.testing.assert_allclose(np.array(w), ww, rtol=1e-5, atol=1e-6)
        se = np.array(inds).reshape(-1)[np.array(order)]
        assert (np.diff(se.astype(np.int64)) >= 0).all() and sorted(np.array(order).tolist()) == list(range(R * 4))
        np.testing.assert_array_equal(np.array(tiles), np.array(moe_tiles.tile_lengths(inds, order, 4)))


def test_preview_conv_kernel_is_the_ops():
    from locals import clover_fast

    m = _model(dtype=mx.bfloat16)
    g = m.layers[0].linear_attn
    rng = np.random.default_rng(8)
    for L in (8, 3, 1):
        hist = mx.array(rng.normal(size=(2, 3, g.conv_dim)).astype(np.float32)).astype(mx.bfloat16)
        cur = mx.array(rng.normal(size=(2, L, g.conv_dim)).astype(np.float32)).astype(mx.bfloat16)
        want = g._preview_conv(hist, cur)
        got = clover_fast._preview_conv_kernel(hist, cur, g.conv1d.weight, g.conv_right_weight)
        np.testing.assert_allclose(np.array(got.astype(mx.float32)), np.array(want.astype(mx.float32)), rtol=1e-2, atol=1e-2)
        assert float(mx.mean((got != want).astype(mx.float32))) < 0.02  # (a bf16 rounding apart, now and then)


def test_fused_model_previews_and_commits_as_the_module():
    from locals.fuse import fuse_model

    m, ref = _quantized(seed=4), _quantized(seed=4)
    fuse_model(m)
    prompt = mx.array([[1, 5, 9, 30, 7, 2, 2, 100, 41, 3, 77]])
    block = mx.array([[255, 4, 255, 255, 8, 255, 255, 255]])
    ca, cb = m.make_cache(), ref.make_cache()
    m.model(prompt, ca), ref.model(prompt, cb)
    for preview in (True, False):
        got, want = m(block, ca, preview=preview), ref(block, cb, preview=preview)
        a, b = np.array(got.astype(mx.float32)), np.array(want.astype(mx.float32))
        assert np.abs(a - b).max() < 0.1 * np.abs(b).max()
        assert (a.argmax(-1) == b.argmax(-1)).mean() >= 0.75


def test_compiled_preview_memory_is_the_modules():
    from locals import clover_fast

    m = _model()
    g = m.layers[0].linear_attn
    rng = np.random.default_rng(9)
    for L in (8, 4):
        q = mx.array(rng.normal(size=(1, L, g.num_k_heads, g.head_k_dim)).astype(np.float32))
        k = mx.array(rng.normal(size=(1, L, g.num_k_heads, g.head_k_dim)).astype(np.float32))
        v = mx.array(rng.normal(size=(1, L, g.num_v_heads, g.head_v_dim)).astype(np.float32))
        a = mx.array(rng.normal(size=(1, L, g.num_v_heads)).astype(np.float32))
        s = mx.array(rng.normal(size=(1, g.num_v_heads, g.head_v_dim, g.head_k_dim)).astype(np.float32))
        want = g._preview_memory(q, k, v, a, s)
        got = clover_fast.FusedCloverGatedDeltaNet._preview_memory(g, q, k, v, a, s)
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-5, atol=1e-6)


class _WordTok(_Tok):
    def decode(self, ids):
        return " ".join(map(str, ids))

    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split()]


@pytest.fixture
def service(monkeypatch):
    from locals import blockdiff
    from locals.blockdiff_service import BlockDiffusionService

    settings = blockdiff.Settings(max_tokens=24, block_size=8, threshold=0.5, eos_token_ids=[254])
    # (made in the service's thread, as a model it loads: MLX streams are a thread's own; float32, so that reading a
    # prompt in other pieces, which runs other matmul kernels, cannot flip a near tie)
    monkeypatch.setattr(blockdiff, "load", lambda path, **kw: (_model(seed=5), _WordTok(), settings))
    svc = BlockDiffusionService("locals", "tiny-clover")
    yield svc
    svc.close()


def test_service_streams_blocks_and_reuses_the_cache_of_a_continued_prompt(service):
    groups = []
    first = service.submit([1, 5, 9, 30, 7, 2], max_tokens=16, on_token=groups.append).wait(60)
    assert first.done and first.finish in ("stop", "length") and first.tokens
    assert sum(groups, []) == first.tokens and all(len(g) <= 8 for g in groups)
    assert first.text == " ".join(map(str, first.tokens))
    kept = service._kept[0]
    assert kept[:6] == [1, 5, 9, 30, 7, 2] and len(kept) >= 6 + len(first.tokens)
    more = kept + [11, 12, 13]
    second = service.submit(more, max_tokens=16).wait(60)
    assert second.cached == len(kept)
    service._kept = ([], None)  # the same prompt read whole: the same reply
    again = service.submit(more, max_tokens=16).wait(60)
    assert again.cached == 0 and again.tokens == second.tokens


def test_service_refuses_grammars_and_stops_when_cancelled(service):
    job = service.submit([1, 2, 3], regex="[a-z]+")
    assert job.done and job.finish.startswith("invalid:")
    job = service.submit([1, 2, 3], schema={"type": "object"})
    assert job.finish.startswith("invalid:")
    job = service.submit([1, 2, 3], max_tokens=24, on_token=lambda t: job.cancel())
    job.wait(60)
    assert job.finish == "cancelled" and len(job.tokens) <= 8


def test_converted_checkpoint_loads_into_the_model():
    """The Hugging Face layout (fused expert gate/up, conv1d (C, 1, K), zero-centred norms) through convert_clover's
    renaming and quantization loads into the module, strict: each quantized weight dequantizes to the float model's
    (within 3-bit error; gate and up swapped would be ~1.4 off), every other parameter is the float model's."""
    import io

    from locals import convert_clover as cc

    m = _model(dtype=mx.bfloat16)
    hf = {}
    params = dict(nn.utils.tree_flatten(m.parameters()))
    for k, v in params.items():
        if k.endswith("switch_mlp.gate_proj.weight"):
            up = params[k.replace("gate_proj", "up_proj")]
            hf[k.replace("switch_mlp.gate_proj.weight", "experts.gate_up_proj")] = mx.concatenate([v, up], axis=1)
        elif k.endswith("switch_mlp.up_proj.weight"):
            continue
        elif k.endswith("switch_mlp.down_proj.weight"):
            hf[k.replace("switch_mlp.down_proj.weight", "experts.down_proj")] = v
        elif k.endswith("conv1d.weight"):
            hf[k] = v.moveaxis(1, 2)
        elif k.endswith(cc.ZERO_CENTRED):
            hf[k] = v - 1
        else:
            hf[k] = v
    out, quant = {}, {}
    for k, v in hf.items():
        cc.convert(k, v, out, quant, io.StringIO())
    assert quant["model.layers.0.mlp.switch_mlp.gate_proj"]["bits"] == 3 and quant["lm_head"]["bits"] == 8
    q = clover_moe.BlockModel(_args())
    nn.quantize(q, class_predicate=lambda p, mod: quant.get(p, False) if hasattr(mod, "to_quantized") else False)
    q.load_weights(list(out.items()), strict=True)
    loaded = dict(nn.utils.tree_flatten(q.parameters()))
    seen = set()
    for name, mod in q.named_modules():
        if hasattr(mod, "bits") and "scales" in mod:
            w = mx.dequantize(mod["weight"], mod["scales"], mod["biases"], group_size=mod.group_size, bits=mod.bits)
            ref = params[name + ".weight"].astype(mx.float32)
            rel = float(mx.linalg.norm((w.astype(mx.float32) - ref).reshape(-1)) / mx.linalg.norm(ref.reshape(-1)))
            assert rel < 0.25, (name, rel)
            seen |= {name + ".weight", name + ".scales", name + ".biases"}
    for k, v in params.items():
        if k not in seen:
            assert mx.array_equal(loaded[k], v), k


def test_rewind_leaves_the_caches_of_a_pass_over_the_kept_tokens():
    m = _model()
    prompt = mx.array([[1, 5, 9, 30, 7, 2, 2, 100]])
    more = mx.array([[6, 4, 12, 13, 8, 2, 19]])
    for keep in (0, 1, 3, 7):
        ca, cb = m.make_cache(), m.make_cache()
        m.model(prompt, ca), m.model(prompt, cb)
        for layer, c in zip(m.layers, ca):
            if layer.is_linear:
                c.save_replay = True
        m.model(more, ca)
        m.rewind(ca, more.shape[1] - keep)
        if keep:
            m.model(more[:, :keep], cb)
        for x, y in zip(_snapshot(ca), _snapshot(cb)):
            for u, v in zip(x, y):
                np.testing.assert_allclose(u, v, rtol=1e-4, atol=1e-5)
        # and the next pass reads the same caches
        nxt = mx.array([[3, 3]])
        np.testing.assert_allclose(np.array(m.model(nxt, ca)), np.array(m.model(nxt, cb)), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("width", [2, 4, 8])
def test_exact_self_speculation_writes_what_autoregressive_decoding_writes(width):
    m = _model(seed=6)
    prompt = [1, 5, 9, 30, 7, 2]
    s = blockdiff.Settings(max_tokens=20, block_size=8, eos_token_ids=[254])
    ar = [t for r in blockdiff.stream_speculative(m, _Tok(), prompt, s, width=1) for t in r.tokens]
    spec = [t for r in blockdiff.stream_speculative(m, _Tok(), prompt, s, width=width, rule="exact") for t in r.tokens]
    assert len(ar) == 20 or ar[-1] == 254
    assert spec == ar
