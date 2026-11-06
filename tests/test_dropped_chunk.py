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


def _model(kinds=KINDS, dtype=mx.float32):
    mx.random.seed(3)
    m = lfm3.Model(lfm3.ModelArgs(
        model_type="lfm3", vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=len(kinds),
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, layer_types=kinds, conv_L_cache=7, sliding_window=8,
        rope_parameters={"full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1e6}, "sliding_attention": {}}))
    m.update(nn.utils.tree_map(lambda p: (mx.random.normal(p.shape) * 0.1).astype(dtype), m.parameters()))
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
            mx.synchronize()  # (else MLX may still hold a buffer, and the spoiling write copy it: see _read_chunk)
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
        assert drops.calls == 5 + len(fail)  # (70 tokens in 16s; a dropped chunk read again whole: 16 is the least)
    assert "the GPU dropped a prompt chunk" in capsys.readouterr().err


def test_a_chunk_with_no_room_in_its_buffers_is_read_again():
    """A full-attention buffer with no room for the chunk (mlx-lm grows it 256 positions at a time) is grown before
    the chunk, not in it: the chunk is written into the grown buffer, and read again if dropped."""
    m = _model()
    toks = mx.random.randint(0, 256, (1, 300), key=mx.random.key(6))
    clean = _read(m, make_prompt_cache(m), toks, 100, _Drops([]))
    drops = _Drops([3])
    got = _read(m, make_prompt_cache(m), toks, 100, drops)  # 200 positions in a 256 buffer, then 300
    assert float(mx.abs(got - clean).max()) <= 1e-4 * (1 + float(mx.abs(clean).max()))
    assert drops.calls == 5  # (100, 100, the third dropped, then read again in 50s)


def test_a_chunk_with_room_in_its_buffers_is_written_into_them():
    """A buffer of a size mlx-lm does not grow to (an LRU row from extract, a trimmed entry) with room for the chunk
    is not grown first: that would copy it whole."""
    from locals.hybrid_batch import _buffers

    m = _model()
    toks = mx.random.randint(0, 256, (1, 290), key=mx.random.key(6))
    cache = make_prompt_cache(m)
    _read(m, cache, toks[:, :249], 100, _Drops([]))  # (250 positions with _read's last token)
    for c in (cache[1], cache[5]):
        c.keys, c.values = (mx.concatenate([a, mx.zeros(a.shape[:2] + (44, a.shape[3]), a.dtype)], axis=2)
                            for a in (c.keys, c.values))  # 300 positions
    before = _buffers(cache[1])
    HybridBatchEngine._read_chunk(_Drops([]), cache, lambda k: m.model(toks[:, 250:250 + k], cache), 40)
    assert cache[1].keys.shape[2] == 300 and _buffers(cache[1]) == before


def test_a_chunk_written_into_a_copied_buffer_is_not_read_again():
    """A full-attention buffer that something else holds is copied by the chunk, in the dropped command buffer, maybe
    not at all: the error goes on to the engine."""
    from locals.hybrid_batch import _buffers

    m = _model()
    toks = mx.random.randint(0, 256, (1, 32), key=mx.random.key(6))
    cache = make_prompt_cache(m)
    HybridBatchEngine._read_chunk(_Drops([]), cache, lambda k: m.model(toks[:, :k], cache), 16)
    before = _buffers(cache[1])
    held = cache[1].keys[..., :1, :]  # (a view of the buffer: MLX cannot write the chunk into it in place)
    mx.eval(held)
    with pytest.raises(RuntimeError, match="Command buffer"):
        HybridBatchEngine._read_chunk(_Drops([1]), cache, lambda k: m.model(toks[:, 16:16 + k], cache), 16)
    assert _buffers(cache[1]) != before  # (the chunk was written into a copy)


def test_a_short_chunk_read_again_counts_its_own_tokens():
    m = _model()
    toks = mx.random.randint(0, 256, (1, 6), key=mx.random.key(7))
    cache = make_prompt_cache(m)
    assert HybridBatchEngine._read_chunk(_Drops([1]), cache, lambda k: m.model(toks[:, :k], cache), 6) == 6
    assert cache[1].offset == 6


def test_a_prefix_cut_from_a_longer_buffer_is_read_on():
    """A conversation's next turn goes on from where the previous reply began: the finished row's key and value
    buffers cut to that position (hybrid_batch._reply_entry), strided views. The marks find them, and the turn's
    prompt is read on from them (the marks once failed on such a view: the engine restarted, and every prompt-cache
    entry was lost)."""
    from types import SimpleNamespace

    from locals.hybrid_batch import _keep_reply_state, _marks, _reply_entry, _rewind

    m = _model()
    toks = mx.random.randint(0, 256, (1, 90), key=mx.random.key(9))
    clean = _read(m, make_prompt_cache(m), toks, 16, _Drops([]))
    cache, r = make_prompt_cache(m), SimpleNamespace()
    m.model(toks[:, :40], cache)
    _keep_reply_state(r, 40, cache)
    m.model(mx.random.randint(0, 256, (1, 20), key=mx.random.key(10)), cache)  # the reply
    mx.eval([c.state for c in cache])
    entry = _reply_entry(r, cache)
    assert not memoryview(entry[1].keys).c_contiguous
    assert _rewind(entry, _marks(entry))  # (nothing read in between: the same buffers)
    got = _read(m, entry, toks[:, 40:], 16, _Drops([]))
    assert float(mx.abs(got - clean).max()) <= 1e-4 * (1 + float(mx.abs(clean).max()))


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


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_the_marks_cost_no_copy(dtype):
    """Each chunk is written into the same full-attention buffers, marks or not (a buffer held by them would be copied
    whole at every chunk: 3 GB a chunk at 245k tokens)."""
    from locals.hybrid_batch import _buffers

    m = _model(dtype=dtype)
    seen = []

    class Watch(_Drops):
        def _sync(self, cache):
            mx.eval([c.state for c in cache])
            seen.append(_buffers(cache[1]))

    _read(m, make_prompt_cache(m), mx.random.randint(0, 256, (1, 200), key=mx.random.key(8)), 16, Watch([]))
    assert len(seen) == 13 and len(set(seen)) == 1  # (200 positions: all in mlx-lm's first 256-position buffer)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])  # (bfloat16, as most checkpoints: no buffer format)
@pytest.mark.parametrize("fail", [[], [1], [2]])
def test_a_prompt_reads_on_from_a_reply_entry(fail, dtype):
    """A follow-up turn starts from a reply's prompt-cache entry, whose full-attention keys and values are cut back to
    where the reply starts (views of the finished row's buffers, not contiguous): its chunks are read like any other,
    and a dropped chunk is read again, the first too (its buffers, which have no room, are grown before it)."""
    from types import SimpleNamespace

    from locals.hybrid_batch import _keep_reply_state, _reply_entry

    m = _model(["conv", "full_attention", "conv", "full_attention"], dtype)  # (a sliding window cannot be cut back)
    toks = mx.random.randint(0, 256, (1, 90), key=mx.random.key(9))
    row, r = make_prompt_cache(m), SimpleNamespace()
    _read(m, row, toks[:, :30], 16, _Drops([]))
    _keep_reply_state(r, 31, row)  # (_read ended with one more token)
    _read(m, row, mx.random.randint(0, 256, (1, 20), key=mx.random.key(10)), 16, _Drops([]))  # the reply
    entry = _reply_entry(r, row)
    assert entry is not None and entry[1].keys.dtype == dtype
    clean = _read(m, make_prompt_cache(m), mx.concatenate([toks[:, :30], mx.array([[7]]), toks[:, 30:]], axis=1), 16,
                  _Drops([]))
    drops = _Drops(fail)
    got = _read(m, entry, toks[:, 30:], 16, drops)
    tol = 1e-4 if dtype == mx.float32 else 2e-2  # (the chunks end at other places than in the cold read)
    assert float(mx.abs(got - clean).max()) <= tol * (1 + float(mx.abs(clean).max()))
    assert drops.calls == 4 + len(fail)  # (60 tokens in 16s; a dropped chunk read again whole: 16 is the least)


def test_a_chunk_read_at_once_after_an_eval_is_written_in_place():
    """mx.eval returns before MLX lets go of the buffers its command buffers read (in their completion handlers, on
    another thread): a chunk written into one still held would copy it whole. _read_chunk waits for them first
    (without that wait this failed 30 times in 30 on an M2 Pro)."""
    from locals.hybrid_batch import _buffers

    m = _model()
    cache, new = make_prompt_cache(m), mx.zeros((1, 2, 1, 16))
    toks = mx.random.randint(0, 256, (1, 8), key=mx.random.key(11))
    for _ in range(20):
        m.model(toks, cache)  # a chunk read the plain way: its command buffers read the buffers, through views of them
        mx.eval([c.state for c in cache])
        kv = cache[1]
        before = _buffers(kv)
        HybridBatchEngine._read_chunk(_Drops([]), [kv], lambda k: kv.update_and_fetch(new, new), 1)  # (one layer: soon)
        assert _buffers(kv) == before
