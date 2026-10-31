"""Unloading while requests come and go (no model: stub services, engines and workers)."""

import contextlib
import queue
import threading

import mlx.core as mx

from locals import engine as eng
from locals import vision
from locals.server import Server
from locals.service import BatchService, Job, Unloaded


def _service(**kw) -> BatchService:
    svc = BatchService.__new__(BatchService)
    svc.engine, svc.model_id = "locals", "m"
    svc.tokenizer = None
    svc._q = queue.Queue()
    svc._compile_lock = threading.Lock()
    svc._inflight = {}
    svc._vision_q = []
    svc._closed = svc._unloaded = False
    svc.__dict__.update(kw)
    return svc


class _Rt:
    def __init__(self, during=None):
        self.during = during

    def ready(self, schema, regex, recursive_ok):
        return None

    def precompile(self, schema, regex, recursive_ok):
        if self.during:
            self.during()
        return "grammar"


def test_a_grammar_made_while_the_model_unloads_ends_cancelled():
    """Its client sends it again (as every queued job of an unload), not "the server is shutting down"."""
    svc = _service()
    job = Job(prompt=[1], schema={"type": "object"})

    def unload():
        svc._unloaded = svc._closed = True

    svc._compile_then_queue(_Rt(during=unload), job)
    assert job.done and job.finish == "cancelled"
    assert svc._q.empty()


def test_a_grammar_queued_just_as_the_service_closes_is_not_left_waiting():
    """Closed between the check and the put (after unload emptied the queue): finished, not waiting forever."""
    svc = _service()

    class Late(queue.Queue):
        def put(self, item, *a, **k):
            svc._unloaded = svc._closed = True
            super().put(item, *a, **k)

    svc._q = Late()
    job = Job(prompt=[1], schema={"type": "object"})
    svc._compile_then_queue(_Rt(), job)
    assert job.done and job.finish == "cancelled"


def test_a_grammar_made_while_the_server_stops_says_so():
    svc = _service()
    job = Job(prompt=[1], schema={"type": "object"})

    def stop():
        svc._closed = True

    svc._compile_then_queue(_Rt(during=stop), job)
    assert job.finish == "error: the server is shutting down"


def test_unload_lets_the_model_go_only_once_the_worker_has_stopped(monkeypatch):
    """A worker still in a long prefill when close() gives up keeps the weights: unload waits for it, so
    the model loaded again (after the engine's load lock this holds) is never a second copy beside it."""
    seen = []

    class Worker:
        joins = 0

        def is_alive(self):
            return self.joins < 3

        def join(self, timeout=None):
            self.joins += 1
            seen.append(svc._rt is not None)

    svc = _service(_rt="the weights", _eng="engine", _lru="cache", _vision_enc=None, _tune_work=None, _thread=Worker())
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    svc.unload()
    assert seen == [True, True, True]  # each join: the model still held
    assert svc._rt is None and svc._unloaded


class _Body:
    """The text model, read in chunks: the client leaves after the first."""

    def __init__(self, gone):
        self.calls, self.gone = 0, gone

    def __call__(self, x, cache, input_embeddings=None):
        self.calls += 1
        self.gone.set()


def test_an_image_prompt_stops_between_prefill_chunks_when_cancelled(monkeypatch):
    gone = threading.Event()
    body = _Body(gone)
    monkeypatch.setattr(eng, "text_model", lambda model: type("M", (), {"model": body})())
    monkeypatch.setattr(eng, "make_prompt_cache", lambda model: [])
    monkeypatch.setattr(eng, "forward_last", lambda model, x, cache: mx.array([[0.0, 1.0, 0.0]]))
    monkeypatch.setattr(vision, "embed_chunk", lambda body, prompt, spans, features, at, n: mx.zeros((1, n, 4)))
    monkeypatch.setattr(vision, "mrope", lambda model, positions, section: contextlib.nullcontext())
    monkeypatch.setattr(vision, "deepstack", lambda model, spans, levels: contextlib.nullcontext({}))
    e = eng.Engine.__new__(eng.Engine)
    e.model, e.prefill_step = None, 2
    e.tokenizer = type("T", (), {"eos_token_ids": [0], "decode": staticmethod(lambda ids: "")})()
    res = e.generate_embedded(list(range(1, 11)), [], None, None, max_tokens=5, cancel=gone.is_set)
    assert body.calls == 1  # 9 tokens to read, 2 a chunk: 5 chunks, stopped after the first
    assert res.finish == "cancelled" and res.tokens == []


def test_a_text_completion_prepared_before_an_unload_goes_to_the_model_loaded_again(monkeypatch):
    """/v1/completions submitted to the service an unload just closed: the model loaded again takes it
    (as the chat path), not a 500 from Unloaded."""
    class Old:
        tokenizer = None

        def submit(self, *a, **k):
            raise Unloaded("the model was unloaded")

    class New:
        tokenizer = type("T", (), {"decode": staticmethod(lambda ids: "hi" * len(ids))})()

        def submit(self, prompt, on_token=None, **k):
            job = Job(prompt=list(prompt))
            job.tokens, job.text, job.finish = [7], "hi", "stop"
            on_token([7])
            job._event.set()
            return job

    srv = Server.__new__(Server)
    srv._active = {}
    srv.service = lambda engine: New()
    monkeypatch.setattr(Server, "_account", lambda self, p, job, n, calls, t0, stopped=None: {"completion_tokens": n})
    p = {"engine": "locals", "svc": Old(), "prompt": [1, 2], "max_tokens": 5, "model": "m", "stop": []}
    out = srv.text_completion({}, p)
    assert out["choices"][0]["text"] == "hi"
    assert isinstance(p["svc"], New)
