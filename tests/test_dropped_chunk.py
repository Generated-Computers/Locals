"""A prompt chunk the GPU drops (macOS's watchdog ends a Metal command buffer while the display needs the GPU) is read
again from where it began: hybrid_batch._marks and _rewind, HybridBatchEngine._read_chunk; on a small random LFM3,
whose caches are of all three kinds (full attention, a sliding window, a convolution's state), alone and in a batch."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import ArraysCache, BatchRotatingKVCache, KVCache, RotatingKVCache, make_prompt_cache

from locals.hybrid_batch import HybridBatchEngine, _merge_layerwise
from locals.models import lfm3

ERR = "[METAL] Command buffer execution failed: Internal Error (0000000e:Internal Error)."
KINDS = ["conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention"]


def _model():
    mx.random.seed(3)
    m = lfm3.Model(lfm3.ModelArgs(
        model_type="lfm3", vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=len(KINDS),
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, layer_types=KINDS, conv_L_cache=7, sliding_window=8,
        rope_parameters={"full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1e6}, "sliding_attention": {}}))
    m.update(nn.utils.tree_map(lambda p: mx.random.normal(p.shape) * 0.1, m.parameters()))
    mx.eval(m.parameters())
    return m


def _spoil(c):
    """What a dropped command buffer may leave: the chunk's writes not done, or done in part."""
    if isinstance(c, ArraysCache):
        c[0] = mx.full(c[0].shape, float("nan"), dtype=c[0].dtype)
    elif isinstance(c, (RotatingKVCache, BatchRotatingKVCache)):
        c.keys = mx.full(c.keys.shape, float("nan"), dtype=c.keys.dtype)
        c.left_padding = getattr(c, "left_padding", None)
    else:  # a full-attention buffer, the chunk written past its old offset
        end = c.offset if isinstance(c, KVCache) else c._idx
        c.keys[..., end - 3:end, :] = float("nan")
    mx.eval([c.state])


class _Drops:
    """HybridBatchEngine._sync, but on the calls in ``fail`` the chunk's results are spoiled and the GPU's error raised."""

    def __init__(self, fail):
        self.fail, self.calls = set(fail), 0

    def _sync(self, cache):
        self.calls += 1
        mx.eval([c.state for c in cache])
        if self.calls in self.fail:
            for c in cache:
                _spoil(c)
            raise RuntimeError(ERR)


def _read(model, cache, toks, chunk, drops):
    i = 0
    while i < toks.shape[1]:
        n = min(chunk, toks.shape[1] - i)
        i += HybridBatchEngine._read_chunk(drops, cache, lambda k: model.model(toks[:, i:i + k], cache), n)
    return model(mx.array([[7]] * toks.shape[0]), cache)[:, -1]


def _batch_cache(model, rows):
    cache = _merge_layerwise([make_prompt_cache(model) for _ in range(rows)])
    for c in cache:
        if isinstance(c, ArraysCache):
            c.left_padding = None  # (as the engine's prefill sets it)
    return cache


@pytest.mark.parametrize("rows", [1, 2])
def test_a_dropped_chunk_is_read_again_and_the_prompt_ends_as_it_would_have(rows, capsys):
    m = _model()
    toks = mx.random.randint(0, 256, (rows, 70), key=mx.random.key(5))
    new = (lambda: make_prompt_cache(m)) if rows == 1 else (lambda: _batch_cache(m, rows))
    clean = _read(m, new(), toks, 16, _Drops([]))
    for fail in ([1], [3], [3, 4], [2, 5]):  # the first chunk too (no buffer yet), and twice in a row
        drops = _Drops(fail)
        got = _read(m, new(), toks, 16, drops)
        assert float(mx.abs(got - clean).max()) <= 1e-4 * (1 + float(mx.abs(clean).max())), fail
        assert drops.calls > 70 // 16 + len(fail)  # (the chunks read again were halved)
    assert "the GPU dropped a prompt chunk" in capsys.readouterr().err


def test_a_chunk_that_grew_a_buffer_is_not_read_again():
    """A full-attention buffer the chunk replaced with a larger one (mlx-lm grows it 256 positions at a time) may not
    hold what it copied from the old one: the error goes on to the engine, as before."""
    m = _model()
    toks = mx.random.randint(0, 256, (1, 300), key=mx.random.key(6))
    with pytest.raises(RuntimeError, match="Command buffer"):
        _read(m, make_prompt_cache(m), toks, 100, _Drops([3]))  # 200 positions in a 256 buffer, then 300


def test_other_errors_are_not_caught():
    m = _model()

    class Oom(_Drops):
        def _sync(self, cache):
            raise RuntimeError("[metal::malloc] Attempting to allocate 99999999999 bytes")

    with pytest.raises(RuntimeError, match="malloc"):
        _read(m, make_prompt_cache(m), mx.zeros((1, 20), dtype=mx.int32), 16, Oom([]))


def test_gives_up_after_its_retries():
    from locals import hybrid_batch

    m = _model()
    with pytest.raises(RuntimeError, match="Command buffer"):
        _read(m, make_prompt_cache(m), mx.zeros((1, 20), dtype=mx.int32), 16, _Drops(range(1, hybrid_batch.GPU_RETRIES + 2)))


@pytest.fixture(autouse=True)
def _no_pause(monkeypatch):
    from locals import hybrid_batch

    monkeypatch.setattr(hybrid_batch, "RETRY_PAUSE", 0.0)


def test_the_marks_cost_no_copy():
    """Each chunk is written into the same full-attention buffers, marks or not (a buffer held by them would be copied
    whole at every chunk: 3 GB a chunk at 245k tokens)."""
    from locals.hybrid_batch import _buffers

    m = _model()
    seen = []

    class Watch(_Drops):
        def _sync(self, cache):
            mx.eval([c.state for c in cache])
            seen.append(_buffers(cache[1]))

    _read(m, make_prompt_cache(m), mx.random.randint(0, 256, (1, 200), key=mx.random.key(8)), 16, Watch([]))
    assert len(seen) == 13 and len(set(seen)) == 1  # (200 positions: all in mlx-lm's first 256-position buffer)
