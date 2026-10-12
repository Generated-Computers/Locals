"""engine.sample_logits: the one sampling rule of both engines (no model)."""

import mlx.core as mx
import numpy as np

from locals.engine import NEG_INF, sample_logits


def _logits():
    rng = np.random.default_rng(0)
    return mx.array(rng.normal(size=(3, 50)).astype(np.float32) * 3)


def test_greedy_is_argmax():
    lg = _logits()
    assert sample_logits(lg, 0.0).tolist() == mx.argmax(lg, axis=-1).tolist()


def test_top_k_keeps_only_k():
    lg = _logits()
    rows = np.array(lg)
    keep = [set(np.argsort(-r)[:5]) for r in rows]
    draws = [sample_logits(lg, 1.0, top_k=5, key=mx.random.key(i)).tolist() for i in range(60)]
    for d in draws:
        for b, t in enumerate(d):
            assert t in keep[b]


def test_top_p_keeps_nucleus():
    lg = _logits()
    p = np.exp(np.array(lg))  # mlx-lm's rule: the nucleus of the untempered distribution, then the temperature
    p /= p.sum(-1, keepdims=True)
    nucleus = []
    for r in p:
        o = np.argsort(-r)
        c = np.cumsum(r[o])
        n = int(np.searchsorted(c, 0.9)) + 1
        nucleus.append(set(o[:n]))
    for i in range(60):
        for b, t in enumerate(sample_logits(lg, 0.6, top_p=0.9, key=mx.random.key(i)).tolist()):
            assert t in nucleus[b]


def test_masked_rows_never_sampled():
    lg = _logits()
    m = mx.array(np.arange(50) % 2 == 0)
    lg = mx.where(m[None, :], lg, NEG_INF)
    for i in range(40):
        for t in sample_logits(lg, 1.0, top_p=0.95, top_k=20, key=mx.random.key(i)).tolist():
            assert t % 2 == 0


def test_small_row_with_top_k():
    # a sparse head row shorter than top_k must still sample
    lg = mx.array([[0.1, 2.0, -1.0]])
    t = sample_logits(lg, 0.7, top_k=20, top_p=0.95, key=mx.random.key(1)).tolist()
    assert t[0] in (0, 1, 2)
