"""hybrid_batch._filter and _append_rows: rows leave and join the batch one cache layer at a time (no model)."""

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, BatchKVCache

from locals.hybrid_batch import _append_rows, _filter


def _kv(pads):
    c = BatchKVCache(pads)
    k = mx.random.normal((len(pads), 2, 8, 4))
    c.update_and_fetch(k, k * 2)
    return c


def test_filter_keeps_the_same_rows_as_mlx_lm():
    a, b = _kv([0, 3, 1]), _kv([0, 3, 1])
    b.keys, b.values = a.keys + 0, a.values + 0
    r = ArraysCache(1)
    r.cache = [mx.arange(3 * 5).reshape(3, 5)]
    idx = mx.array([1, 2])
    _filter([a, r], idx)
    b.filter(idx)
    assert mx.array_equal(a.keys, b.keys) and mx.array_equal(a.values, b.values)
    assert a.left_padding.tolist() == b.left_padding.tolist() and a._idx == b._idx
    assert r.cache[0].tolist() == [[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]


def test_append_rows_matches_mlx_lm():
    a, b, new = _kv([0, 2]), _kv([0, 2]), _kv([1])
    b.keys, b.values = a.keys + 0, a.values + 0
    _append_rows([a], [new])
    b.extend(new)
    assert mx.array_equal(a.keys, b.keys) and mx.array_equal(a.values, b.values)
    assert a.left_padding.tolist() == b.left_padding.tolist() and a.offset.tolist() == b.offset.tolist()


def test_prefill_chunks_bound_the_tokens_of_a_forward():
    """Sixteen 1k-token prompts arriving together were prefilled in 16 x 1024-token chunks; the activations
    of one such forward ran a 16 GB Mac's GPU out of memory. Rows share PREFILL_TOKENS per forward."""
    from locals.hybrid_batch import PREFILL_SCORES, PREFILL_TOKENS, prefill_chunk

    assert prefill_chunk(0, 1, 2048) == 2048  # one prompt: as before
    for rows in (1, 2, 4, 8, 16, 32, 64):
        for done in (0, 1000, 30000):
            c = prefill_chunk(done, rows, 1024)
            assert rows * c <= max(PREFILL_TOKENS, rows * 16)
            assert c <= 128 or rows * c * (done + c) <= PREFILL_SCORES
    assert prefill_chunk(0, 16, 1024) == 128
