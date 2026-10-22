"""Flint (locals/models/flint_streamed.py) on a tiny random model saved as the INT4 build saves it: 4-bit weights as bytes,
experts stacked per layer. Its experts are read from the files (ExpertStore); these tests check that they give
what the experts computed in memory give, that decoding with caches gives what one pass over the prompt gives,
and that the attention sinks are a logit in the softmax's denominator."""
import json

import mlx.core as mx
import numpy as np
import pytest

from locals import models

models.register()
from locals.models import flint_streamed as flint  # noqa: E402



def test_only_recursives_4bit_build_loads_with_its_experts_in_its_files():
    """defaults.streams: Flint's 4-bit build as Recursive exports it (weights as bytes) streams on any Mac; a build
    in MLX's 32-bit words (mlx-community's, its experts at 3 bits or not) loads whole (locals/models/flint.py)."""
    from locals.defaults import streams

    mixed = {**CFG, "quantization": {"bits": 4, "group_size": 64, "model.layers.1.mlp.switch_mlp.gate_proj": {"bits": 3, "group_size": 64}}}
    assert streams(CFG, "U8")
    assert not streams(CFG, "U32") and not streams(mixed, "U8") and not streams(mixed, "U32")
    assert not streams({**CFG, "quantization": {"bits": 3, "group_size": 64}}, "U8")
    assert not streams({**CFG, "model_type": "qwen3_moe"}, "U8")


def test_a_flint_folder_loads_the_module_its_build_needs(tmp_path, monkeypatch):
    """mlx-lm's load_model (locals.models.register) takes flint_streamed for the 4-bit build's bytes, and the memory
    check counts its weights without the experts; any other Flint folder is mlx-lm's to load (locals/models/flint.py)."""
    import mlx_lm.utils as mlx_utils
    from locals.defaults import streamed, weights_gb

    _save(tmp_path)
    assert streamed(tmp_path)
    model, _ = mlx_utils.load_model(tmp_path)
    assert type(model) is flint.Model and model.expert_store is not None
    monkeypatch.setenv("LOCALS_EXPERT_CACHE_GB", "0")
    files = sum(f.stat().st_size for f in tmp_path.glob("*.safetensors")) / 2**30
    assert weights_gb(str(tmp_path)) < files
    cfg = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**cfg, "quantization": {"bits": 3, "group_size": 64}}))
    assert not streamed(tmp_path) and weights_gb(str(tmp_path)) == files

CFG = {
    "model_type": "flint", "architectures": ["FlintForCausalLM"], "hidden_size": 128, "num_hidden_layers": 4,
    "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
    "rms_norm_eps": 1e-6, "vocab_size": 96, "num_experts": 8, "num_experts_per_tok": 2,
    "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64,
    "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
    "no_rope_layer_indices": [0, 3], "mlp_only_layers": [0], "sliding_window": 5, "rope_theta": 10000.0,
    "quantization": {"bits": 4, "group_size": 64},
}


def _q(w, bits=4):
    wq, s, b = mx.quantize(w, group_size=64, bits=bits)
    return wq.view(mx.uint8), s.astype(mx.bfloat16), b.astype(mx.bfloat16)


def _save(folder):
    """Random weights in the INT4 build's layout; returns the dense expert weights for a reference."""
    rng = np.random.default_rng(0)
    r = lambda *s: mx.array(rng.normal(0, 0.08, s).astype(np.float32))  # noqa: E731
    w, D, H, KV, d = {}, 128, 4, 2, 32

    def lin(name, out, inp, bits=4):
        w[f"{name}.weight"], w[f"{name}.scales"], w[f"{name}.biases"] = _q(r(out, inp), bits)

    lin("model.embed_tokens", 96, D, bits=8)
    lin("lm_head", 96, D)
    w["model.norm.weight"] = mx.ones((D,), dtype=mx.bfloat16)
    for i in range(4):
        p = f"model.layers.{i}"
        for n, o, k in (("q_proj", H * d, D), ("k_proj", KV * d, D), ("v_proj", KV * d, D), ("o_proj", D, H * d)):
            lin(f"{p}.self_attn.{n}", o, k)
        w[f"{p}.self_attn.q_norm.weight"] = (1 + r(d)).astype(mx.bfloat16)
        w[f"{p}.self_attn.k_norm.weight"] = (1 + r(d)).astype(mx.bfloat16)
        w[f"{p}.self_attn.softmax_offset"] = r(H).astype(mx.bfloat16) * 10
        w[f"{p}.input_layernorm.weight"] = mx.ones((D,), dtype=mx.bfloat16)
        w[f"{p}.post_attention_layernorm.weight"] = mx.ones((D,), dtype=mx.bfloat16)
        if i == 0:
            lin(f"{p}.mlp.gate_proj", 128, D), lin(f"{p}.mlp.up_proj", 128, D), lin(f"{p}.mlp.down_proj", D, 128)
            continue
        w[f"{p}.mlp.gate.weight"] = r(8, D).astype(mx.bfloat16) * 5
        for n, o, k in (("gate_proj", 64, D), ("up_proj", 64, D), ("down_proj", D, 64)):
            lin(f"{p}.mlp.shared_expert.{n}", o, k)
            stack = [_q(r(o, k)) for _ in range(8)]
            for j, kind in enumerate(("weight", "scales", "biases")):
                w[f"{p}.mlp.switch_mlp.{n}.{kind}"] = mx.stack([s[j] for s in stack])
    half = len(w) // 2
    names = sorted(w)
    mx.save_safetensors(str(folder / "model-00000.safetensors"), {n: w[n] for n in names[:half]})
    mx.save_safetensors(str(folder / "model-00001.safetensors"), {n: w[n] for n in names[half:]})
    (folder / "config.json").write_text(json.dumps(CFG))
    return w


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    import mlx_lm.utils as mlx_utils

    folder = tmp_path_factory.mktemp("flint")
    w = _save(folder)
    model, _ = mlx_utils.load_model(folder)
    return model, w


def _in_memory(model, w):
    """What the experts give computed in memory from the stacked tensors (mlx-lm's SwitchGLU way)."""

    def call(self, x, inds):
        p = f"model.layers.{self.layer}.mlp.switch_mlp"
        t = lambda n, k: w[f"{p}.{n}.{k}"]  # noqa: E731
        conv = lambda n: (t(n, "weight").view(mx.uint32), t(n, "scales").astype(x.dtype), t(n, "biases").astype(x.dtype))  # noqa: E731
        xx = mx.expand_dims(x, (-2, -3))
        qmm = lambda a, n: mx.gather_qmm(a, *conv(n), rhs_indices=inds, transpose=True, group_size=64, bits=4)  # noqa: E731
        g, u = qmm(xx, "gate_proj"), qmm(xx, "up_proj")
        if x.dtype == mx.float16:  # (the product in float32, as the model takes it in float16)
            return flint._swiglu_fp16(g, u, lambda h: qmm(h, "down_proj")).squeeze(-2)
        return qmm(flint.swiglu(g, u), "down_proj").squeeze(-2)

    return call


def _reference_moe(model, w):
    """A MoE block as MLX's own ops compute it from the experts in memory, routed on float32 logits as the router
    kernel routes (bf16 logits tie often, and a tie may go either way)."""
    experts = _in_memory(model, w)

    def call(self, x, residual=None):
        logits = x.astype(mx.float32) @ self.gate.weight.astype(mx.float32).T
        inds = mx.argpartition(-logits, kth=self.k - 1, axis=-1)[..., : self.k]
        s = mx.sigmoid(mx.take_along_axis(logits, inds, axis=-1))
        y = experts(self.experts, x, inds)
        y = (y * (s / s.sum(axis=-1, keepdims=True))[..., None].astype(y.dtype)).sum(axis=-2) + self.shared_expert(x)
        return y if residual is None else residual + y

    return call


def test_streamed_experts_give_what_the_experts_in_memory_give(tiny, monkeypatch):
    model, w = tiny
    tokens = mx.array([[3, 17, 42, 5, 88, 9, 61, 2, 33]])
    streamed = model(tokens)
    monkeypatch.setattr(flint.StreamedExperts, "__call__", _in_memory(model, w))
    full = model(tokens)
    assert mx.allclose(streamed, full, atol=1e-4).item()
    store = model.expert_store
    assert store.misses > 0 and store.bytes > 0  # experts were read from the files and kept


def test_decoding_with_caches_gives_what_one_pass_gives(tiny):
    """Past the sliding window (5): the rotating caches of the sliding layers and the full ones of the others."""
    model, _ = tiny
    tokens = mx.array([[3, 17, 42, 5, 88, 9, 61, 2, 33, 70, 11, 4]])
    one_pass = model(tokens)[0, -1]
    cache = model.make_cache()
    model(tokens[:, :6], cache=cache)
    for t in range(6, 12):
        last = model(tokens[:, t:t + 1], cache=cache)[0, -1]
    assert mx.allclose(last, one_pass, atol=2e-3).item()


def test_the_sink_is_a_logit_in_the_softmax_denominator(tiny):
    model, _ = tiny
    attn = model.model.layers[0].self_attn  # a full-attention layer without RoPE
    x = mx.random.normal((1, 4, 128), key=mx.random.key(1))
    got = attn(x, mask="causal")
    B, L, H, KV, d = 1, 4, 4, 2, 32
    q = attn.q_norm(attn.q_proj(x).reshape(B, L, H, d)).transpose(0, 2, 1, 3)
    k = attn.k_norm(attn.k_proj(x).reshape(B, L, KV, d)).transpose(0, 2, 1, 3)
    v = attn.v_proj(x).reshape(B, L, KV, d).transpose(0, 2, 1, 3)
    k, v = mx.repeat(k, H // KV, axis=1), mx.repeat(v, H // KV, axis=1)
    s = (q @ k.transpose(0, 1, 3, 2)) * d ** -0.5
    s = mx.where(mx.tril(mx.ones((L, L))).astype(mx.bool_), s, -mx.inf)
    sink = attn.softmax_offset.reshape(1, H, 1, 1).astype(s.dtype)
    m = mx.maximum(s.max(axis=-1, keepdims=True), sink)
    e = mx.exp(s - m)
    p = e / (e.sum(axis=-1, keepdims=True) + mx.exp(sink - m))
    want = attn.o_proj((p @ v).transpose(0, 2, 1, 3).reshape(B, L, -1))
    assert mx.allclose(got, want, atol=1e-3).item()


def test_few_slots_evict_without_losing_the_experts_a_pass_uses(tiny, monkeypatch):
    """With 3 slots a layer (a token uses 2 of 8 experts), decoding evicts the least recently used slot a pass
    does not need; a pass that needs more experts than there are slots stacks them. Both give what the experts
    in memory give."""
    model, w = tiny
    store = model.expert_store
    monkeypatch.setattr(store, "slots", 3)
    store.reset()  # (start over: the earlier tests made slots of the default size)
    tokens = mx.array([[3, 17, 42, 5, 88, 9, 61, 2, 33]])
    cache = model.make_cache()
    got = [model(tokens[:, i:i + 1], cache=cache)[0, -1] for i in range(9)]
    misses = store.misses
    many = model(tokens)  # 9 tokens: up to 8 experts a layer, more than its 3 slots
    assert all(a.shape[0] == 3 for a in store.arrays[1])
    assert store.misses > misses
    monkeypatch.setattr(flint.StreamedExperts, "__call__", _in_memory(model, w))
    cache = model.make_cache()
    want = [model(tokens[:, i:i + 1], cache=cache)[0, -1] for i in range(9)]
    assert all(mx.allclose(a, b, atol=1e-4).item() for a, b in zip(got, want))
    assert mx.allclose(many, model(tokens), atol=1e-4).item()


def test_a_pass_that_needs_most_experts_reads_the_layers_tensors_whole(tiny, monkeypatch):
    """Reading a prompt routes its tokens to most of a layer's experts: the layer's stacked tensors are read
    whole (and the next layer's warmed) instead of expert by expert, with the same result."""
    model, w = tiny
    tokens = mx.array([[3, 17, 42, 5, 88, 9, 61, 2, 33, 70, 11, 4]])
    monkeypatch.setattr(flint, "WHOLE", 1)
    read = []
    monkeypatch.setattr(flint.ExpertStore, "stacked", lambda *a: read.append(a) or pytest.fail("read expert by expert"))
    whole = model(tokens)
    monkeypatch.setattr(flint.StreamedExperts, "__call__", _in_memory(model, w))
    assert mx.allclose(whole, model(tokens), atol=1e-4).item()


def test_in_float16_the_slots_keep_the_files_bf16_and_the_experts_give_what_they_give_in_memory(tmp_path, monkeypatch):
    """Locals computes in float16 on M1 and M2: the slots keep an expert's bf16 scales and biases as the files have
    them (converted on the GPU when used), and the experts give what they give in memory in float16."""
    import mlx_lm.utils as mlx_utils

    w = _save(tmp_path)
    model, _ = mlx_utils.load_model(tmp_path)
    model.set_dtype(mx.float16)
    tokens = mx.array([[3, 17, 42, 5, 88, 9, 61, 2, 33]])
    cache = model.make_cache()
    got = [model(tokens[:, i:i + 1], cache=cache)[0, -1] for i in range(9)]
    assert model.expert_store.arrays[1][1].dtype == mx.bfloat16
    monkeypatch.setattr(flint.StreamedExperts, "__call__", _in_memory(model, w))
    cache = model.make_cache()
    want = [model(tokens[:, i:i + 1], cache=cache)[0, -1] for i in range(9)]
    assert all(mx.allclose(a, b, atol=2e-3).item() for a, b in zip(got, want))


WIDE = dict(CFG, hidden_size=512, intermediate_size=512, head_dim=128, moe_intermediate_size=512, num_experts=32,
            shared_expert_intermediate_size=512, num_hidden_layers=3, layer_types=["full_attention", "sliding_attention", "full_attention"],
            no_rope_layer_indices=[0, 2])


def _save_wide(folder):
    """A model as wide as the decode kernels need (hidden, expert and shared widths multiples of 512, experts a
    multiple of 32)."""
    rng = np.random.default_rng(1)
    r = lambda *s: mx.array(rng.normal(0, 0.05, s).astype(np.float32))  # noqa: E731
    w, D, H, KV, d = {}, 512, 4, 2, 128

    def lin(name, out, inp, bits=4):
        w[f"{name}.weight"], w[f"{name}.scales"], w[f"{name}.biases"] = _q(r(out, inp), bits)

    lin("model.embed_tokens", 96, D, bits=8)
    lin("lm_head", 96, D)
    w["model.norm.weight"] = mx.ones((D,), dtype=mx.bfloat16)
    for i in range(3):
        p = f"model.layers.{i}"
        for n, o, k in (("q_proj", H * d, D), ("k_proj", KV * d, D), ("v_proj", KV * d, D), ("o_proj", D, H * d)):
            lin(f"{p}.self_attn.{n}", o, k)
        w[f"{p}.self_attn.q_norm.weight"] = (1 + r(d)).astype(mx.bfloat16)
        w[f"{p}.self_attn.k_norm.weight"] = (1 + r(d)).astype(mx.bfloat16)
        w[f"{p}.self_attn.softmax_offset"] = r(H).astype(mx.bfloat16) * 10
        w[f"{p}.input_layernorm.weight"] = mx.ones((D,), dtype=mx.bfloat16)
        w[f"{p}.post_attention_layernorm.weight"] = mx.ones((D,), dtype=mx.bfloat16)
        if i == 0:
            lin(f"{p}.mlp.gate_proj", 512, D), lin(f"{p}.mlp.up_proj", 512, D), lin(f"{p}.mlp.down_proj", D, 512)
            continue
        w[f"{p}.mlp.gate.weight"] = r(32, D).astype(mx.bfloat16)  # (logits of a few units: no sigmoid at 1.0, no ties)
        for n, o, k in (("gate_proj", 512, D), ("up_proj", 512, D), ("down_proj", D, 512)):
            lin(f"{p}.mlp.shared_expert.{n}", o, k)
            stack = [_q(r(o, k)) for _ in range(32)]
            for j, kind in enumerate(("weight", "scales", "biases")):
                w[f"{p}.mlp.switch_mlp.{n}.{kind}"] = mx.stack([s[j] for s in stack])
    mx.save_safetensors(str(folder / "model-00000.safetensors"), w)
    (folder / "config.json").write_text(json.dumps(WIDE))
    return w


@pytest.mark.parametrize("slots", [None, 3])
@pytest.mark.parametrize("optimistic", [True, False])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_the_decode_kernels_give_what_the_experts_in_memory_give(tmp_path, monkeypatch, dtype, optimistic, slots):
    """Decoding one token at a time, a model as wide as Flint's kernels need routes with the fused router kernels
    and computes its MoE blocks with the fused expert kernels, the experts already in slots and the shared one
    first and the ones being read after; optimistically, each step's windows of layers look their experts up on
    the GPU and resume after the first layer that misses one (its caches after it restored). With 3 slots a
    layer, layers miss experts all the time. It gives what MLX's own ops on the experts in memory give."""
    import mlx_lm.utils as mlx_utils

    monkeypatch.setattr(flint, "OPTIMISTIC", "always" if optimistic else "0")
    w = _save_wide(tmp_path)
    model, _ = mlx_utils.load_model(tmp_path)
    model.set_dtype(dtype, predicate=lambda t: t == mx.bfloat16)
    store = model.expert_store
    if slots:
        store.slots = slots
        store.reset()
    tokens = [3, 17, 42, 5, 88, 9, 61, 2, 33, 70, 11, 4]
    calls = []
    route = flint.SparseMoe._route
    monkeypatch.setattr(flint.SparseMoe, "_route", lambda self, *a: calls.append(1) or route(self, *a))
    restores = []
    restore = flint._restore
    monkeypatch.setattr(flint, "_restore", lambda *a: restores.append(1) or restore(*a))
    cache = model.make_cache()
    got = [model(mx.array([[t]]), cache=cache)[0, -1] for t in tokens]
    assert len(calls) >= 2 * len(tokens)  # both MoE layers, every token, took the kernels
    assert store.misses > 0 and store.hits > 0
    if optimistic and slots:
        assert restores  # a window went past a layer that missed an expert
    monkeypatch.setattr(flint.SparseMoe, "__call__", _reference_moe(model, w))
    monkeypatch.setattr(flint, "OPTIMISTIC", "0")
    cache = model.make_cache()
    want = [model(mx.array([[t]]), cache=cache)[0, -1] for t in tokens]
    tol = 0.05 if dtype == mx.bfloat16 else 0.01
    for a, b in zip(got, want):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        assert float(mx.max(mx.abs(a - b))) <= tol * (1 + float(mx.max(mx.abs(b))))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_experts_evicted_to_the_ram_cache_come_back_from_it(tmp_path, monkeypatch, dtype):
    """With 3 slots a layer of 32 experts, decoding evicts experts all the time; with a RAM cache they are copied
    there first, and a later miss copies them back instead of reading the file. The logits are bit for bit those
    of the same decode reading every expert from the file."""
    import mlx_lm.utils as mlx_utils

    _save_wide(tmp_path)
    tokens = [3, 17, 42, 5, 88, 9, 61, 2, 33, 70, 11, 4] * 3
    runs = []
    for ram in ("0.5", "0"):
        monkeypatch.setenv("LOCALS_EXPERT_RAM_GB", ram)
        model, _ = mlx_utils.load_model(tmp_path)
        model.set_dtype(dtype, predicate=lambda t: t == mx.bfloat16)
        store = model.expert_store
        assert (store.ram is not None) == (ram != "0")
        store.slots = 3
        store.reset()
        cache = model.make_cache()
        runs.append([model(mx.array([[t]]), cache=cache)[0, -1] for t in tokens])
        if store.ram is not None:
            assert store.ram.puts > 50 and store.ram.hits > 50
    assert all(mx.array_equal(a, b).item() for a, b in zip(*runs))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_a_prompt_pass_copies_the_experts_in_slots_and_reads_the_others(tmp_path, monkeypatch, dtype):
    """A pass over a prompt reads its layers' expert tensors whole into two reused sets, the next layer while
    the GPU computes this one; the experts decoding left in slots are copied from there instead of read. It gives
    what the experts in memory give."""
    import mlx_lm.utils as mlx_utils

    w = _save_wide(tmp_path)
    model, _ = mlx_utils.load_model(tmp_path)
    model.set_dtype(dtype, predicate=lambda t: t == mx.bfloat16)
    store = model.expert_store
    cache = model.make_cache()
    for t in [3, 17, 42, 5]:  # decoding fills slots
        mx.eval(model(mx.array([[t]]), cache=cache))
    assert any(store.layers[l].where for l in store.layers)
    copies = []
    read = flint.ExpertStore._read_layer

    def counted(self, layer, k):
        f = read(self, layer, k)
        copies.append(sum(1 for task in f.tasks for item in task if item[0] < 0))
        return f

    monkeypatch.setattr(flint.ExpertStore, "_read_layer", counted)
    monkeypatch.setattr(flint, "WHOLE", 1)
    tokens = mx.array([[3, 17, 42, 5, 88, 9, 61, 2, 33, 70, 11, 4, 19, 23, 7, 50, 64, 31]])
    got = model(tokens)
    assert sum(copies) > 0
    monkeypatch.setattr(flint.SparseMoe, "__call__", _reference_moe(model, w))
    want = model(tokens)
    tol = {mx.bfloat16: 0.05, mx.float16: 0.01, mx.float32: 1e-3}[dtype]
    assert float(mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32)))) <= tol * (1 + float(mx.max(mx.abs(want.astype(mx.float32)))))


def test_float16_keeps_a_swiglu_product_past_its_range():
    """In float16 an expert's SwiGLU product past 65,504 (Flint's layer 46 reaches about 68,000) is taken in
    float32 and scaled by a power of 2 for the down projection: finite, and what float32 gives."""
    rng = np.random.default_rng(3)
    w = mx.array(rng.normal(0, 0.002, (64, 128)).astype(np.float32))  # (the output stays in range, as Flint's does)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    g = mx.array([[40.0] * 128, [0.5] * 128])  # silu(40) * 2000 = 80,000 in the first row
    u = mx.array([[2000.0] * 128, [2.0] * 128])
    down = lambda h: mx.quantized_matmul(h, wq, s.astype(h.dtype), b.astype(h.dtype), transpose=True, group_size=64, bits=4)  # noqa: E731
    got = flint._swiglu_fp16(g.astype(mx.float16), u.astype(mx.float16), down)
    want = down(flint.swiglu(g, u))
    assert got.dtype == mx.float16 and bool(mx.all(mx.isfinite(got)))
    assert not bool(mx.all(mx.isfinite(down(flint.swiglu(g.astype(mx.float16), u.astype(mx.float16))))))  # (plain float16)
    for a, b in zip(got.astype(mx.float32), want):  # (float16's own rounding of the weights, per row)
        assert float(mx.max(mx.abs(a - b))) <= 0.02 * float(mx.max(mx.abs(b)))


def test_a_sliding_layers_long_pass_attends_in_blocks_over_each_ones_window(tiny, monkeypatch):
    """Over a pass longer than two blocks of queries, a sliding layer's attention runs block by block over the keys
    each block's window reaches (the mask's slices): what one attention over all the keys with the mask gives, in
    one pass and in passes through the rotating cache (whose keys then start with the window before)."""
    model, _ = tiny
    tokens = mx.array([[3, 17, 42, 5, 88, 9, 61, 2, 33, 70, 11, 4, 19, 23, 7, 50, 64, 31, 8, 90]])
    monkeypatch.setattr(flint, "WINDOW_BLOCK", 10 ** 6)
    one = model(tokens)
    cache = model.make_cache()
    chunks = [model(tokens[:, :9], cache=cache), model(tokens[:, 9:], cache=cache)]
    windowed = []
    attn = flint.Attention._windowed
    monkeypatch.setattr(flint.Attention, "_windowed", lambda self, *a: windowed.append(1) or attn(self, *a))
    monkeypatch.setattr(flint, "WINDOW_BLOCK", 3)
    assert mx.allclose(model(tokens), one, atol=1e-5).item()
    cache = model.make_cache()
    got = [model(tokens[:, :9], cache=cache), model(tokens[:, 9:], cache=cache)]
    assert all(mx.allclose(a, b, atol=1e-5).item() for a, b in zip(got, chunks))
    assert windowed


def test_with_an_expert_pack_the_experts_are_one_read_each_and_give_the_same(tmp_path, monkeypatch):
    """models/pack.py writes the experts expert by expert; the store then reads an expert with one scatter read
    into its slot (and a prompt pass, runs of them into the layer's tensors). Decoding and a prompt pass give, bit
    for bit, what they give reading the safetensors files."""
    import mlx_lm.utils as mlx_utils

    from locals.models import pack

    folder = tmp_path / "model"
    folder.mkdir()
    _save_wide(folder)
    monkeypatch.setenv("LOCALS_EXPERT_PACK_DIR", str(tmp_path / "packs"))
    tokens = [3, 17, 42, 5, 88, 9, 61, 2, 33, 70, 11, 4] * 2
    prompt = mx.array([[3, 17, 42, 5, 88, 9, 61, 2, 33, 70, 11, 4, 19, 23, 7, 50]])
    monkeypatch.setattr(flint, "WHOLE", 1)  # (the prompt pass, more experts than slots: whole layers; tokens: slots)
    runs = []
    for packed in (False, True):
        if packed:
            assert pack.find(folder) is None
            pack.write(folder, log=lambda m: None)
            assert pack.find(folder) is not None
        model, _ = mlx_utils.load_model(folder)
        store = model.expert_store
        assert bool(store.pack) == packed
        store.slots = 3
        store.reset()
        cache = model.make_cache()
        out = [model(mx.array([[t]]), cache=cache)[0, -1] for t in tokens]
        out.append(model(prompt))
        runs.append(out)
    assert all(mx.array_equal(a, b).item() for a, b in zip(*runs))


def test_the_layer_sets_go_while_decoding_and_come_back_for_the_next_prompt(tmp_path, monkeypatch):
    """A prompt pass makes the two layer sets; decoding with no prompt pass for SETS_IDLE_S lets them go, and the
    next prompt pass makes them again, with the same results."""
    import mlx_lm.utils as mlx_utils

    _save_wide(tmp_path)
    model, _ = mlx_utils.load_model(tmp_path)
    store = model.expert_store
    monkeypatch.setattr(flint, "WHOLE", 1)
    monkeypatch.setattr(flint, "SETS_IDLE_S", 0.0)
    prompt = mx.array([[3, 17, 42, 5, 88, 9, 61, 2, 33, 70, 11, 4, 19, 23, 7, 50, 64, 31]])
    first = model(prompt)
    mx.eval(first)
    assert store._sets is not None
    cache = model.make_cache()
    mx.eval(model(mx.array([[3]]), cache=cache))  # a decode step (the fused path: slots)
    assert store._sets is None
    again = model(prompt)
    assert store._sets is not None and mx.array_equal(first, again).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_experts_split_between_two_calls_give_the_bits_of_one(dtype):
    """The experts in slots first, the others after their reads: the second call sums every contribution in the
    experts' order (those of the first call taken from it), so the result does not depend on which reads had
    ended - bit for bit what one call over all of them gives."""
    from locals.models import flint_kernels as fk

    rng = np.random.default_rng(5)
    D, EI, S, rows = 512, 512, 6, 3
    q = lambda *shape: _q(mx.array(rng.normal(0, 0.05, shape).astype(np.float32)))  # noqa: E731
    routed = []
    for o, k in ((EI, D), (EI, D), (D, EI)):
        parts = [q(o, k) for _ in range(S)]
        routed += [mx.stack([p[i] for p in parts]) for i in range(3)]
    routed = [a.view(mx.uint32) if a.dtype == mx.uint8 else a for a in routed]
    shared = [a.view(mx.uint32) if a.dtype == mx.uint8 else a.astype(dtype) for o, k in ((EI, D), (EI, D), (D, EI)) for a in q(o, k)]
    x = mx.array(rng.normal(0, 0.5, (rows, D)).astype(np.float32)).astype(dtype)
    res = mx.array(rng.normal(0, 1, (rows, D)).astype(np.float32)).astype(dtype)
    slots = mx.array(np.stack([rng.choice(S, 2, replace=False) for _ in range(rows)]).astype(np.uint32))
    w = mx.array(rng.random((rows, 2)).astype(np.float32))
    one = fk.experts(x, res, routed, shared, slots, mx.ones((rows, 2), dtype=mx.uint32), w)
    first = mx.array(np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint32))
    part, contrib = fk.experts(x, res, routed, shared, slots, first, w, out_dtype=mx.float32, keep=True)
    two = fk.experts(x, res, routed, shared, slots, mx.where(first == 1, 2, 1).astype(mx.uint32), w, prev=contrib,
                     shared_from_prev=True)
    assert mx.array_equal(one, two).item()
