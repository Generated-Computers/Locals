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
    p = {"engine": "locals", "svc": Old(), "tok": New.tokenizer, "prompt": [1, 2], "max_tokens": 5, "model": "m",
         "stop": []}
    out = srv.text_completion({}, p)
    assert out["choices"][0]["text"] == "hi"
    assert isinstance(p["svc"], New)


class _Tok:
    """The tokenizer a request was prepared with."""

    decode = staticmethod(lambda ids: "hi" * len(ids))
    _tokenizer = type("HF", (), {"encode": staticmethod(lambda s, add_special_tokens=False: [ord(c) for c in s])})()


class _Unloaded:
    """A service an unload has finished: its tokenizer is dropped, and it takes no request."""

    tokenizer = None

    def submit(self, *a, **k):
        raise Unloaded("the model was unloaded")


class _Loaded:
    tokenizer = _Tok()

    def __init__(self):
        self.prompts = []

    def submit(self, prompt, on_token=None, **k):
        self.prompts.append(list(prompt))
        job = Job(prompt=list(prompt))
        job.tokens, job.text, job.finish = [7], "hi", "stop"
        on_token([7])
        job._event.set()
        return job


def _chat_server(new):
    srv = Server.__new__(Server)
    srv._active = {"k": {}}
    srv.service = lambda engine: new
    return srv


def _chat():
    return {"engine": "locals", "svc": _Unloaded(), "tok": _Tok(), "prompt": [1, 2], "max_tokens": 5, "schema": None,
            "regex": None, "thinking": False, "tools": [], "stop": [], "ignore_eos": True, "text": "q"}


def test_a_chat_reply_prepared_before_an_unload_is_written_by_the_model_loaded_again():
    """Its tokens are read with the tokenizer the request was prepared with, not the unloaded service's (None)."""
    new = _Loaded()
    srv, out = _chat_server(new), []
    job, text, _ = srv._decode(_chat(), "", out.append, None, "k")
    assert text == "hi" and job.finish == "stop"
    assert new.prompts == [[1, 2]]


def test_a_chat_reply_after_a_prefix_prepared_before_an_unload_is_written_by_the_model_loaded_again():
    new = _Loaded()
    srv, out = _chat_server(new), []
    _, text, _ = srv._decode(_chat(), "ab", out.append, None, "k")
    assert text == "abhi"
    assert new.prompts == [[1, 2, ord("a"), ord("b")]]


def test_a_reply_cancelled_by_an_unload_is_remembered_with_the_tokenizer_it_was_prepared_with():
    srv = Server.__new__(Server)
    srv._lock, srv._raw, srv._streams = threading.Lock(), {}, {"locals": []}
    job = Job(prompt=[1, 2])
    job.tokens, job.text, job.finish = [7], "hi", "cancelled"
    srv._remember("locals", _chat(), job, [], prefix="ab")
    assert srv._streams["locals"] == [{"text": "qabhi", "ids": [1, 2, ord("a"), ord("b"), 7]}]


def test_an_image_chat_prepared_across_an_unload_is_read_with_its_images_by_the_model_loaded_again():
    """/unload lands while an image request is prepared, after the server took the model for it: the model loaded
    again reads the request's image, not a text-only note in its place. Needs Qwen3.5-4B's tokenizer (cached)."""
    import base64
    import io

    import pytest
    from huggingface_hub import snapshot_download
    from PIL import Image

    model = "mlx-community/Qwen3.5-4B-MLX-4bit"
    try:
        path = snapshot_download(model, local_files_only=True, allow_patterns=["*.json", "*.jinja", "tokenizer*"])
    except Exception:  # noqa: BLE001
        pytest.skip("tokenizer not cached")
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(path)

    class Loaded:
        """A service that reads images, until an unload finishes it (BatchService drops its tokenizer and tower)."""

        unloaded = False

        @property
        def tokenizer(self):
            return None if self.unloaded else tok

        def vision(self):
            return None if self.unloaded else "the vision tower"

        def unload(self):
            self.unloaded = True
            return 0.0

        def submit(self, prompt, on_token=None, **k):
            assert self.unloaded
            raise Unloaded("the model was unloaded")

    class Again(Loaded):
        def __init__(self):
            self.got = []

        def submit(self, prompt, on_token=None, **k):
            self.got.append((list(prompt), k.get("pictures"), k.get("spans")))
            job = Job(prompt=list(prompt))
            job.tokens, job.text, job.finish = [7], "a red square", "stop"
            on_token([7])
            job._event.set()
            return job

    srv = Server(model, engines=("locals",), preload=(), log=lambda s: None)
    first, again = Loaded(), Again()
    srv._svc["locals"] = first
    cap, order = srv._cap_context, []

    def unload_now():  # (prepare's first step after it took the model)
        cap()
        if not order:
            order.append(srv.unload()["status"])
            srv._svc["locals"] = again  # another request loaded the model again

    srv._cap_context = unload_now
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 10, 10)).save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    p = srv.prepare({"model": "locals/qwen3.5-4b", "max_tokens": 8, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is in the picture?"}, {"type": "image_url", "image_url": {"url": url}}]}]})
    assert order == ["unloaded"] and first.unloaded  # the unload landed inside prepare
    srv._active = {"k": {}}
    srv._decode(p, "", lambda c: None, None, "k")
    (prompt, pictures, spans), = again.got
    assert pictures and len(pictures) == 1, "the image was dropped: the model loaded again read a text-only prompt"
    assert spans and prompt.count(tok.convert_tokens_to_ids(vision.IMAGE_TOKEN)) > 1  # the image's own tokens
