"""The engine keeps a few shared-prefix snapshots, not every one it made: each holds a
prefix's KV, and a server that kept them all grew by gigabytes over a night of agents."""

from collections import OrderedDict

import pytest

pytest.importorskip("mlx_lm")

from locals.hybrid_batch import HybridBatchEngine  # noqa: E402


def test_snapshots_are_bounded_and_recent_ones_stay():
    eng = object.__new__(HybridBatchEngine)
    eng.snapshots, eng.max_snapshots = OrderedDict(), 3
    for i in range(10):
        eng._snapshot((i,) * 4, [i])
    assert list(eng.snapshots) == [(7,) * 4, (8,) * 4, (9,) * 4]
    assert eng.snapshots[(9,) * 4] == [9]
