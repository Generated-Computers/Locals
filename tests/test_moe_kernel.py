"""The grouped expert kernels (locals/moe_kernel.py): mlx-lm's SwitchGLU computed with each expert read once a
decode step, in MLX's affine 4-bit and mxfp4 formats; and the runtime patch that uses them for decode steps."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

from locals import moe_kernel as mk


def _glu(D, I, E, mode, group, dtype, seed=0):
    mx.random.seed(seed)
    m = SwitchGLU(D, I, E)
    m.set_dtype(dtype)
    nn.quantize(m, group_size=group, bits=4, mode=mode, class_predicate=lambda p, l: hasattr(l, "to_quantized"))
    mx.eval(m.parameters())
    return m


def _arrays(m):
    return tuple((l["weight"], l["scales"], l.get("biases")) for l in (m.gate_proj, m.up_proj, m.down_proj))


@pytest.mark.parametrize("mode,group", [("affine", 64), ("mxfp4", 32)])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("rows", [1, 5, 33])
def test_grouped_experts_give_what_mlx_lms_switchglu_gives(mode, group, dtype, rows):
    D, I, E, k = 512, 384, 16, 4  # (I not a multiple of 256: the last block of a lane's loop is partial)
    m = _glu(D, I, E, mode, group, dtype)
    rng = np.random.default_rng(rows)
    x = mx.array(rng.normal(0, 0.5, (rows, D)).astype(np.float32)).astype(dtype)
    inds = mx.array(np.stack([rng.choice(E, k, replace=False) for _ in range(rows)]).astype(np.uint32))
    want = m(x[:, None, :], inds[:, None, :]).reshape(rows, k, D).astype(mx.float32)
    got = mk.switch_glu(x, inds, *_arrays(m), mode, group).astype(mx.float32)
    assert got.shape == want.shape
    tol = 0.02 if dtype == mx.bfloat16 else 2e-3
    assert float(mx.max(mx.abs(got - want))) <= tol * (1 + float(mx.max(mx.abs(want))))


def test_pairs_are_grouped_by_expert():
    rng = np.random.default_rng(1)
    inds = rng.integers(0, 12, 40).astype(np.uint32)
    order, starts, experts, nseg = (np.array(a) for a in mk.group(mx.array(inds), 12))
    n = int(nseg[0])
    assert n == len(np.unique(inds)) and starts[n] == len(inds)
    assert sorted(order.tolist()) == list(range(len(inds)))
    for s in range(n):
        seg = order[starts[s]:starts[s + 1]]
        assert len(seg) and all(inds[p] == experts[s] for p in seg)
    assert list(experts[:n]) == sorted(experts[:n])


def test_the_runtime_patch_uses_the_kernels_for_decode_steps_only(monkeypatch):
    m = _glu(512, 256, 8, "affine", 64, mx.float16)
    calls = []
    real = mk.switch_glu
    monkeypatch.setattr(mk, "switch_glu", lambda *a, **kw: calls.append(a[0].shape[0]) or real(*a, **kw))

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.glu = m

    holder = Holder()
    rng = np.random.default_rng(2)
    small = mx.array(rng.normal(0, 0.5, (4, 1, 512)).astype(np.float16))
    big = mx.array(rng.normal(0, 0.5, (1, mk.MAX_ROWS + 1, 512)).astype(np.float16))
    i_small = mx.array(rng.integers(0, 8, (4, 1, 2)).astype(np.uint32))
    i_big = mx.array(rng.integers(0, 8, (1, mk.MAX_ROWS + 1, 2)).astype(np.uint32))
    want_small, want_big = m(small, i_small), m(big, i_big)
    assert mk.enable(holder) == 1 and type(m).__name__ == "GroupedSwitchGLU"
    got_small, got_big = m(small, i_small), m(big, i_big)
    assert calls == [4]  # the decode step only; a long pass keeps MLX's kernels
    assert got_small.shape == want_small.shape and mx.allclose(got_small, want_small, atol=2e-3).item()
    assert mx.array_equal(got_big, want_big).item()


def test_flints_experts_keep_their_own_kernels():
    """Flint (locals/models/flint.py) has its own expert kernels (locals/moe.py): its SwitchGLUs stay as they are,
    even unfused."""
    from locals.models import flint

    args = flint.ModelArgs(model_type="flint", hidden_size=512, num_hidden_layers=2, intermediate_size=512,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=128, num_experts=8, num_experts_per_tok=2,
                           moe_intermediate_size=256, shared_expert_intermediate_size=256, mlp_only_layers=[0],
                           layer_types=["full_attention", "sliding_attention"], sliding_window=9, rms_norm_eps=1e-6,
                           vocab_size=64)
    model = flint.Model(args)
    nn.quantize(model, group_size=64, bits=4, class_predicate=lambda p, l: hasattr(l, "to_quantized") and not p.endswith("mlp.gate"))
    sw = model.layers[1].mlp.switch_mlp
    assert mk._mats(sw) is not None  # (a SwitchGLU the kernels could compute)
    assert mk.enable(model) == 0 and type(sw) is SwitchGLU
