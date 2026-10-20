"""Idle-time kernel tuning waits while another engine of the server works (locals bench serves two at once): its
timings, kept for good, were taken on a shared GPU, and the other engine's numbers came out low."""

import queue

from locals import service


def _svc():
    s = service.BatchService.__new__(service.BatchService)
    s._q, s._inflight = queue.Queue(), {}
    return s


def test_tuning_while_idle_waits_for_the_other_engine(monkeypatch):
    tuned = []
    monkeypatch.setattr("locals.kernels.tune_one", lambda mod, M: tuned.append(M))
    ours, stock = _svc(), _svc()
    monkeypatch.setattr(service, "_SERVICES", service.weakref.WeakSet([ours, stock]))
    ours._tune_work = [("mod", 4), ("mod", 8)]
    stock._inflight[1] = object()  # the stock engine is answering
    ours._tune_while_idle()
    assert tuned == []
    stock._inflight.clear()
    ours._tune_while_idle()
    assert tuned == [4]
