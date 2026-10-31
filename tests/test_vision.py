"""Images (locals/vision.py): the request's images taken out and decoded, the prompt's placeholders expanded,
M-RoPE positions, and the patch order the tower and its merger expect (no model: the arithmetic alone)."""

import base64
import io

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from locals import apis, oai, vision
from locals import engine as eng


def _png(w: int, h: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 200, 30)).save(buf, "PNG")
    return buf.getvalue()


def test_mrope_with_text_positions_is_the_layers_own_rope():
    # Text alone has the same position in t, h and w: M-RoPE must then rotate as the layer's nn.RoPE does, the
    # 64 rotated dims of Qwen3.5's 256 (partial rotary) and the other 192 left as they are, at any cache offset
    rope = nn.RoPE(64, traditional=False, base=10_000_000)
    x = mx.random.normal((1, 4, 7, 256)).astype(mx.float16)
    offset = 5
    pos = mx.broadcast_to(mx.arange(32, dtype=mx.int32)[None, None], (3, 1, 32))
    got = vision.MRoPE(64, 10_000_000, pos)(x, offset=offset)
    want = rope(x, offset=offset)
    assert got.shape == want.shape
    assert float(mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max()) < 2e-2
    assert mx.array_equal(got[..., 64:], x[..., 64:])


def test_mrope_takes_h_and_w_from_their_interleaved_frequencies():
    # mrope_section [11, 11, 10], interleaved: frequency 1, 4, ..., 31 from h; 2, 5, ..., 29 from w; the rest from t
    m = vision.MRoPE(64, 10_000.0, mx.zeros((3, 1, 1), dtype=mx.int32))
    sel = [int(np.argmax([float(m._sel[a][k]) for a in range(3)])) for k in range(32)]
    assert [k for k in range(32) if sel[k] == 1] == list(range(1, 32, 3))
    assert [k for k in range(32) if sel[k] == 2] == list(range(2, 30, 3))
    assert sel.count(0) == 11


def test_expand_and_positions():
    pic = vision.Picture(np.zeros((4 * 6, 1536), np.float32), (4, 6))  # 4x6 patches: a 2x3 grid of text tokens
    assert pic.tokens == 6
    ids, spans = vision.expand([1, 2, 99, 3, 4], 99, [pic])
    assert ids == [1, 2] + [99] * 6 + [3, 4] and spans == [(2, 8)]
    pos = vision.positions(spans, [pic.grid], len(ids)).tolist()
    assert pos[0] == [0, 1, 2, 2, 2, 2, 2, 2, 5, 6]  # t: the image's one frame; the text after it goes on past max(2, 3)
    assert pos[1] == [0, 1, 2, 2, 2, 3, 3, 3, 5, 6]  # h: the rows
    assert pos[2] == [0, 1, 2, 3, 4, 2, 3, 4, 5, 6]  # w: the columns
    tail = vision.tail(mx.array(pos, dtype=mx.int32), 2).tolist()
    assert [r[-2:] for r in tail] == [[7, 8]] * 3


def test_expand_refuses_a_count_that_does_not_match():
    pic = vision.Picture(np.zeros((4, 1536), np.float32), (2, 2))
    with pytest.raises(ValueError, match="placeholder"):
        vision.expand([1, 2], 99, [pic])
    with pytest.raises(ValueError, match="more image placeholders"):
        vision.expand([99, 1, 99], 99, [pic])


def test_patches_are_in_spatial_merge_order():
    # each pixel holds its own patch coordinates: patch k must be block row, block column, in-block row, in-block column
    from PIL import Image

    h, w = 256, 320  # 16 x 20 patches, already a multiple of 32 and over the minimum pixel count
    arr = np.zeros((h, w, 3), np.uint8)
    for py in range(h // 16):
        for px in range(w // 16):
            arr[py * 16:(py + 1) * 16, px * 16:(px + 1) * 16] = (py * 10, px * 10, 0)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    pic = vision.preprocess(buf.getvalue())
    assert pic.grid == (16, 20) and pic.patches.shape == (320, 2 * 16 * 16 * 3)
    first = pic.patches.reshape(320, 2, 16, 16, 3)[:, 0, 0, 0, :2]  # each patch's first pixel, R and G
    coords = np.rint((first + 1.0) * 127.5 / 10).astype(int)
    want = [(br * 2 + ir, bc * 2 + ic) for br in range(8) for bc in range(10) for ir in range(2) for ic in range(2)]
    assert [tuple(c) for c in coords] == want


def test_smart_resize_keeps_the_trained_sizes():
    assert vision.smart_resize(400, 640, 32, 65536, 1 << 20) == (384, 640)
    h, w = vision.smart_resize(1800, 2880, 32, 65536, 1 << 20)
    assert h % 32 == 0 and w % 32 == 0 and h * w <= 1 << 20
    assert vision.smart_resize(20, 20, 32, 65536, 1 << 20) == (256, 256)
    with pytest.raises(ValueError, match="aspect ratio"):
        vision.smart_resize(1, 500, 32, 65536, 1 << 20)


def test_take_images_and_decode():
    data = _png(64, 48)
    url = "data:image/png;base64," + base64.b64encode(data).decode()
    msgs = [{"role": "system", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "what is it?"}, {"type": "image_url", "image_url": {"url": url}}]}]
    assert vision.has_images(msgs) and not vision.has_images(msgs[:1])
    out, sources = vision.take_images(msgs)
    assert sources == [url]
    assert oai.text_of(out[1]["content"]) == "what is it?\n" + vision.IMAGE_MARK
    assert msgs[1]["content"][1]["type"] == "image_url"  # (the request itself is left as it was)
    assert vision.decode(url) == data
    assert vision.decode(base64.b64encode(data).decode(), media_type="image/png") == data  # Ollama's bare base64
    pics = vision.load([url])
    assert pics[0].tokens == pics[0].grid[0] * pics[0].grid[1] // 4
    with pytest.raises(ValueError, match="image 1 of 1"):
        vision.load(["data:image/png;base64,bm90IGFuIGltYWdl"])


def test_a_server_others_reach_reads_no_local_files_and_fetches_no_urls(tmp_path, monkeypatch):
    f = tmp_path / "x.png"
    f.write_bytes(_png(8, 8))
    assert vision.decode(str(f)) == f.read_bytes()
    with pytest.raises(ValueError, match="reads no files"):
        vision.decode(str(f), local=False)
    with pytest.raises(ValueError, match="reads no files"):
        vision.decode("file://" + str(f), local=False)
    fetched = []
    monkeypatch.setattr(vision.urllib.request, "urlopen", lambda *a, **k: fetched.append(a))
    for url in ("http://127.0.0.1:8080/x.png", "https://169.254.169.254/latest/meta-data/"):
        with pytest.raises(ValueError, match="fetches no URLs"):
            vision.decode(url, local=False)
    assert fetched == []


def test_a_fetched_image_is_read_no_further_than_the_limit(monkeypatch):
    asked = []

    class Body:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            asked.append(n)
            return b"x" * ((64 << 20) + 1)

    monkeypatch.setattr(vision.urllib.request, "urlopen", lambda *a, **k: Body())
    with pytest.raises(ValueError, match="larger than 64 MB"):
        vision.decode("http://127.0.0.1/big.png")
    assert asked == [(64 << 20) + 1]  # a bounded read, not the whole body


def test_anthropic_tool_result_images_become_image_parts():
    png = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
    chat = apis.anthropic_to_chat({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "screenshot:"}, png]},
        {"type": "tool_result", "tool_use_id": "t2", "is_error": True, "content": [png]},
        {"type": "tool_result", "tool_use_id": "t3", "is_error": True, "content": "no such file"},
        {"type": "tool_result", "tool_use_id": "t4", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
    ]}]}, "locals/q")
    t1, t2, t3, t4 = chat["messages"]
    assert t1 == {"role": "tool", "tool_call_id": "t1", "content": [
        {"type": "text", "text": "screenshot:"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}
    assert t2["content"] == [{"type": "text", "text": "Error: "}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    assert t3["content"] == "Error: no such file"  # text only: a string, as before
    assert t4["content"] == "a\nb"
    # the image reaches take_images (it reads tool messages), and a text-only model gets the note
    msgs, sources = vision.take_images([t1])
    assert sources == ["data:image/png;base64,AAAA"]
    assert oai.text_of(t1["content"]) == "screenshot:\n" + oai.IMAGE_NOTE


def test_anthropic_images_become_image_parts():
    chat = apis.anthropic_to_chat({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        {"type": "text", "text": "what is it?"}]}]}, "locals/q")
    content = chat["messages"][0]["content"]
    assert content[0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    assert content[1] == {"type": "text", "text": "what is it?"}
    # a model that reads text only still gets the note in the image's place
    assert oai.text_of(content) == oai.IMAGE_NOTE + "\nwhat is it?"


@pytest.mark.parametrize("model_type, vision, text, reads", [
    ("qwen3_5", {"hidden_size": 1024, "deepstack_visual_indexes": []},
     {"head_dim": 256, "rope_parameters": {"mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25}}, True),
    ("qwen3_vl", {"hidden_size": 1152, "deepstack_visual_indexes": [8, 16, 24]},
     {"head_dim": 128, "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20]}}, True),
    # a split that does not cover the rotated dims, or none: the image tokens' positions would mean nothing
    ("qwen3_5", {"hidden_size": 1024}, {"head_dim": 256, "rope_parameters": {"mrope_section": [24, 20, 20], "partial_rotary_factor": 0.25}}, False),
    ("qwen3_vl", {"hidden_size": 1152}, {"head_dim": 128, "rope_scaling": {"rope_type": "default"}}, False),
    ("qwen3_vl", {"hidden_size": 1152}, {"head_dim": 128, "rope_scaling": {"mrope_interleaved": False, "mrope_section": [24, 20, 20]}}, False),
    ("qwen3_5", None, {}, False),  # a text-only checkpoint (Qwen3.8)
    ("gemma4", {"hidden_size": 768}, {}, False),
])
def test_only_checkpoints_this_code_computes_right_read_images(tmp_path, model_type, vision, text, reads):
    import json

    from locals.models.qwen3_5_vl import vision_config

    cfg = {"model_type": model_type, "text_config": text}
    if vision is not None:
        cfg["vision_config"] = vision
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    got = vision_config(tmp_path)
    assert (got is not None) == reads
    if got is not None:
        assert list(got.mrope_section) == text.get("rope_parameters", text.get("rope_scaling"))["mrope_section"]


def test_mrope_split_for_qwen3_vl_and_a_wrong_one():
    m = vision.MRoPE(128, 5_000_000.0, mx.zeros((3, 1, 1), dtype=mx.int32), (24, 20, 20))
    sel = [int(np.argmax([float(m._sel[a][k]) for a in range(3)])) for k in range(64)]
    assert [k for k in range(64) if sel[k] == 1] == list(range(1, 60, 3))  # section[1] = 20 for h; t keeps 24
    assert [k for k in range(64) if sel[k] == 2] == list(range(2, 60, 3))
    with pytest.raises(ValueError, match="does not split"):
        vision.MRoPE(64, 10_000.0, mx.zeros((3, 1, 1), dtype=mx.int32), (24, 20, 20))


def test_deepstack_adds_to_the_image_tokens_of_the_chunk_read():
    # after layer i, the image's tokens of the chunk being read get adds[i]; the layers come back after
    class Body:
        def __init__(self):
            self.layers = [lambda x, *a: x, lambda x, *a: x * 2, lambda x, *a: x]

    class Model:
        model = Body()

    m = Model()
    feats = [mx.ones((2, 3)), mx.full((1, 3), 5.0)]
    add = vision.place(8, [(2, 4), (6, 7)], feats)
    assert add.shape == (1, 8, 3) and add[0, :, 0].tolist() == [0, 0, 1, 1, 0, 0, 5, 0]
    kept = list(m.model.layers)
    with vision.deepstack(m, [add]) as st:
        st["at"] = 4  # a chunk of tokens 4..7
        h = mx.zeros((1, 4, 3))
        for layer in m.model.layers:
            h = layer(h)
        assert h[0, :, 0].tolist() == [0, 0, 10, 0]  # added after layer 0, doubled by layer 1
    assert m.model.layers == kept


class _Healing(eng.Controller):
    """Each token is followed by a forced token 7 that, when healing is allowed, replaces it."""

    def __init__(self):
        self.n = 0

    def forced(self, token, heal=True):
        return [7], heal

    def accept(self, token):
        self.n += 1

    def finished(self):
        return self.n >= 3


def _sync_engine(monkeypatch):
    monkeypatch.setattr(eng, "make_prompt_cache", lambda model: [])
    monkeypatch.setattr(eng, "forward_last", lambda model, x, cache: mx.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]))
    e = eng.Engine.__new__(eng.Engine)
    e.model = None
    e.tokenizer = type("T", (), {"eos_token_ids": [0], "decode": staticmethod(lambda ids: " ".join(map(str, ids)))})()
    return e


def test_a_streamed_reply_is_the_reply_the_client_gets(monkeypatch):
    """A healed token replaces one already sent: with ``on_token`` (an image request streamed), the
    sync path must not heal, so the stream and the final reply are the same tokens (as the batch engine)."""
    e = _sync_engine(monkeypatch)
    streamed = []
    res = e._generate_sync([1, 2], _Healing(), 20, 0.0, 0, prefill=lambda cache: None, on_token=streamed.extend)
    assert streamed == res.tokens
    assert res.tokens[:2] == [3, 7]
    # not streamed: healing still replaces the sampled token
    res = e._generate_sync([1, 2], _Healing(), 20, 0.0, 0, prefill=lambda cache: None)
    assert res.tokens[:2] == [7, 7]


def test_an_image_request_that_fails_ends_only_itself(monkeypatch):
    """An error in generate_embedded (memory the prefill could not get) is the request's error,
    not an exception out of the worker, which would fail every job in flight."""
    from locals.service import BatchService, Job

    svc = BatchService.__new__(BatchService)
    svc.tokenizer = None
    svc._inflight = {}
    svc.prompt_tokens = svc.tokens_out = 0
    svc.temp, svc.top_p, svc.top_k, svc.seed = 0.0, 0.0, 0, 0
    svc.model_id = "m"
    enc = type("Enc", (), {"encode": lambda self, pics: (None, ()), "mrope_section": (11, 11, 10)})()
    svc.vision = lambda: enc
    monkeypatch.setattr(vision, "positions", lambda spans, grids, n: mx.zeros((3, n)))
    monkeypatch.setattr(vision, "tail", lambda pos, n: pos)

    def oom(*a, **k):
        raise RuntimeError("[metal::malloc] Attempting to allocate 9 GB")

    rt = type("Rt", (), {"engine": type("E", (), {"generate_embedded": staticmethod(oom)})()})()
    other = Job(prompt=[5])
    svc._inflight[id(other)] = other
    job = Job(prompt=[1, 2, 3])
    svc._inflight[id(job)] = job
    svc._vision(rt, job, None)
    assert job.done and job.finish.startswith("error: ") and "allocate" in job.finish
    assert not other.done and id(other) in svc._inflight
    assert svc.prompt_tokens == 0


def _batch_service(monkeypatch, compiled):
    """A service with a stub runtime whose grammars count their compiles (no model, no worker thread)."""
    import queue
    import threading

    from locals.service import BatchService

    svc = BatchService.__new__(BatchService)
    svc.engine, svc.model_id, svc.walk = "locals", "m", False
    svc.tokenizer = None
    svc._q = queue.Queue()
    svc._compile_lock = threading.Lock()
    svc._inflight, svc._owner, svc._vision_q = {}, {}, []
    svc._closed = svc._unloaded = False
    svc.prompt_tokens = svc.tokens_out = 0
    svc._free = svc._endless = eng.Controller
    monkeypatch.setattr(eng, "LocalsController", lambda c, planner, *a: ("grammar", c))

    class Rt:
        def compile(self, schema, regex=None, ready=None, recursive_ok=False):
            compiled.append(schema)
            return type("C", (), {"constraint": len(compiled), "planner": None})()

    return svc, Rt()


def test_an_image_job_held_for_an_idle_batch_keeps_its_grammar(monkeypatch):
    """Taken while rows decode, an image job is held; released at the idle call, it runs with the controller made
    when it was taken. Its grammar was made again there, on the engine thread, while every reply waited."""
    from locals.service import Job

    compiled, ran = [], []
    svc, rt = _batch_service(monkeypatch, compiled)
    svc._vision = lambda rt, job, ctl: ran.append((job, ctl))
    job = Job(prompt=[1, 2], schema={"type": "object"}, pictures=["image"])
    svc._q.put(job)
    assert svc._source(rt, block=False) == []
    assert ran == [] and [j for j, _ in svc._vision_q] == [job]
    assert svc._source(rt, block=True) == []
    assert ran == [(job, ("grammar", 1))]
    assert compiled == [{"type": "object"}]


def test_while_an_image_job_waits_the_batch_takes_no_new_rows(monkeypatch):
    """Text that arrives after the image job waits for it: the running rows end, and the idle batch runs the
    image job. Taking text meanwhile, a busy server never let it run."""
    from locals.service import Job

    svc, rt = _batch_service(monkeypatch, [])
    ran = []
    svc._vision = lambda rt, job, ctl: ran.append(job)
    image = Job(prompt=[1], pictures=["image"])
    svc._q.put(image)
    assert svc._source(rt, block=False) == []
    text = Job(prompt=[2, 3])
    svc._q.put(text)
    for _ in range(3):  # the rows still running: nothing new joins them
        assert svc._source(rt, block=False) == []
    assert ran == [] and not svc._q.empty()
    reqs = svc._source(rt, block=True)  # the batch is idle: the image job first, then the text joins
    assert ran == [image]
    assert [r.prompt for r in reqs] == [[2, 3]]


def test_an_image_job_cancelled_while_it_waits_does_not_run_the_tower(monkeypatch):
    from locals.service import Job

    svc, rt = _batch_service(monkeypatch, [])
    read = []
    svc.vision = lambda: type("Enc", (), {"encode": lambda self, pics: read.append(pics)})()
    job = Job(prompt=[1], pictures=["image"])
    svc._inflight[id(job)] = job
    svc._q.put(job)
    svc._source(rt, block=False)
    job.cancel()
    assert svc._source(rt, block=True) == []
    assert job.done and job.finish == "cancelled"
    assert read == [] and not svc._inflight
