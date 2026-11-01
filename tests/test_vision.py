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


def _png(w: int, h: int, colour=(10, 200, 30)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, "PNG")
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
    assert vision.take_images(["hi"]) == (["hi"], [])  # (not an object: the server's shape check answers it)
    assert oai.text_of(out[1]["content"]) == "what is it?" + vision.IMAGE_MARK
    assert msgs[1]["content"][1]["type"] == "image_url"  # (the request itself is left as it was)
    assert vision.decode(url) == data
    pics = vision.load([url])
    assert pics[0].tokens == pics[0].grid[0] * pics[0].grid[1] // 4
    with pytest.raises(ValueError, match="image 1 of 1"):
        vision.load(["data:image/png;base64,bm90IGFuIGltYWdl"])


def test_an_images_cache_key_is_its_bytes_and_size():
    """Two images of one size have the same pad tokens: in the prompt cache's key each is the image's own id, from
    its bytes and its grid (the same bytes resized to another grid are other features)."""
    one, two = (base64.b64encode(_png(64, 48)).decode(), base64.b64encode(_png(64, 48, (1, 2, 3))).decode())
    a, a2, b = vision.load(["data:image/png;base64," + one, "data:image/png;base64," + one, "data:image/png;base64," + two])
    assert a.key == a2.key and a.key != b.key and a.key > 0 and a.grid == b.grid
    assert vision.load(["data:image/png;base64," + one], count=True)[0].key == a.key
    assert vision.cache_key([5, 9, 9, 9, 6], [(1, 4)], [a]) == [5, -a.key, -a.key, -a.key, 6]
    assert vision.cache_key([5, 9, 6], [(1, 2)], [vision.Picture(None, (2, 2))]) is None  # (no key: not cached)


def test_a_checkpoint_moves_past_the_image_tokens_before_it():
    """Checkpoints are found on the prompt with one placeholder per image; each then moves by the tokens the images
    before it expanded to."""
    spans = [(1, 5), (7, 23)]  # [a, P, b, c, P, d] with P expanded to 4 and 16 tokens
    assert [vision.expanded_at(n, spans) for n in range(7)] == [0, 1, 5, 6, 7, 23, 24]


@pytest.mark.parametrize("size", [(64, 48), (640, 480), (2000, 1000), (37, 2100)])
def test_a_count_reads_only_the_images_size(size, monkeypatch):
    """count_tokens: the token count of a full load, from the image's header (two 4032x3024 JPEGs: 250 ms a count
    before, 30 ms after, docs/measurements/vision/results/count_tokens.txt)."""
    url = "data:image/png;base64," + base64.b64encode(_png(*size)).decode()
    full = vision.load([url])
    monkeypatch.setattr(vision, "preprocess", lambda raw: pytest.fail("a count preprocesses no image"))
    counted = vision.load([url], count=True)
    assert counted[0].grid == full[0].grid and counted[0].patches is None
    with pytest.raises(ValueError, match="image 1 of 1"):
        vision.load(["data:image/png;base64,bm90IGFuIGltYWdl"], count=True)


def test_an_image_is_joined_to_its_text_as_qwens_template_joins_it():
    """Qwen's template writes ``AAA<|vision_start|><|image_pad|><|vision_end|>BBB``: a newline after the image,
    which the model was not trained with, changed Qwen3-VL-8B's answers on two of six test pictures. Text parts
    are still joined with a newline."""
    url = "data:image/png;base64," + base64.b64encode(_png(64, 48)).decode()
    img = {"type": "image_url", "image_url": {"url": url}}
    content = [{"type": "text", "text": "AAA"}, img, {"type": "text", "text": "BBB"}, {"type": "text", "text": "CCC"}, img]
    out, _ = vision.take_images([{"role": "user", "content": content}])
    mark = vision.IMAGE_MARK
    assert oai.text_of(out[0]["content"]) == "AAA" + mark + "BBB\nCCC" + mark
    assert oai.text_of([img, img, {"type": "text", "text": "x"}]) == oai.IMAGE_NOTE + "\n" + oai.IMAGE_NOTE + "\nx"


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
    # other patch values: vision.preprocess cuts 16x16 patches over two frames, and the merger takes 2x2 of them
    ("qwen3_vl", {"hidden_size": 1152, "patch_size": 14},
     {"head_dim": 128, "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20]}}, False),
    ("qwen3_vl", {"hidden_size": 1152, "spatial_merge_size": 4},
     {"head_dim": 128, "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20]}}, False),
    ("qwen3_vl", {"hidden_size": 1152, "temporal_patch_size": 1},
     {"head_dim": 128, "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20]}}, False),
    ("qwen3_vl", {"hidden_size": 1152, "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2},
     {"head_dim": 128, "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20]}}, True),
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


@pytest.mark.parametrize("files, reads", [
    ({"model.safetensors": ["language_model.model.embed_tokens.weight", "vision_tower.blocks.0.attn.qkv.weight"]}, True),
    ({"model-00001-of-00002.safetensors": ["model.language_model.norm.weight"],  # transformers' names, the tower in
      "model-00002-of-00002.safetensors": ["model.visual.merger.norm.weight"]}, True),  # the second shard
    ({"model.safetensors": ["language_model.model.embed_tokens.weight"]}, False),  # a text-only conversion
    ({}, False),
])
def test_a_config_that_names_a_tower_reads_images_only_with_its_weights(tmp_path, files, reads):
    """A checkpoint whose config names a tower but whose files lack it said it read images: its first image request
    was an error, and the next got the text-only note."""
    import json

    cfg = {"model_type": "qwen3_vl", "vision_config": {"hidden_size": 1152, "deepstack_visual_indexes": [8, 16, 24]},
           "text_config": {"head_dim": 128, "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20]}}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    for name, keys in files.items():
        mx.save_safetensors(str(tmp_path / name), {k: mx.zeros((2,)) for k in keys})
    enc = vision.Encoder(str(tmp_path), mx.float16)
    assert enc.available() == reads


def test_a_model_not_loaded_says_whether_it_reads_images(tmp_path):
    """/api/show listed vision only while the model was loaded: a client that asked before the first request, or
    after an unload, was told the model reads text only. The checkpoint decides now; the stock engine never reads them."""
    import json

    from locals.server import Server

    cfg = {"model_type": "qwen3_vl", "vision_config": {"hidden_size": 1152, "deepstack_visual_indexes": [8, 16, 24]},
           "text_config": {"head_dim": 128, "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20]}}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"vision_tower.blocks.0.attn.qkv.weight": mx.zeros((2,))})
    for engines, reads in ((("locals",), True), (("stock",), False)):
        srv = Server.__new__(Server)
        srv.model_id, srv.engines, srv._svc = str(tmp_path), engines, {}
        assert srv.reads_images() == reads


def test_mrope_split_for_qwen3_vl_and_a_wrong_one():
    m = vision.MRoPE(128, 5_000_000.0, mx.zeros((3, 1, 1), dtype=mx.int32), (24, 20, 20))
    sel = [int(np.argmax([float(m._sel[a][k]) for a in range(3)])) for k in range(64)]
    assert [k for k in range(64) if sel[k] == 1] == list(range(1, 60, 3))  # section[1] = 20 for h; t keeps 24
    assert [k for k in range(64) if sel[k] == 2] == list(range(2, 60, 3))
    with pytest.raises(ValueError, match="does not split"):
        vision.MRoPE(64, 10_000.0, mx.zeros((3, 1, 1), dtype=mx.int32), (24, 20, 20))


def test_deepstack_adds_to_the_image_tokens_of_the_chunk_read():
    # after layer i, the image's tokens of the chunk being read get level i's features; the layers come back after
    class Body:
        def __init__(self):
            self.layers = [lambda x, *a: x, lambda x, *a: x * 2, lambda x, *a: x]

    class Model:
        model = Body()

    m = Model()
    feats = [mx.ones((2, 3)), mx.full((1, 3), 5.0)]
    kept = list(m.model.layers)
    with vision.deepstack(m, [(2, 4), (6, 7)], [feats]) as st:
        st["at"] = 4  # a chunk of tokens 4..7
        h = mx.zeros((1, 4, 3))
        for layer in m.model.layers:
            h = layer(h)
        assert h[0, :, 0].tolist() == [0, 0, 10, 0]  # added after layer 0, doubled by layer 1
    assert m.model.layers == kept


@pytest.mark.parametrize("step", [1, 2, 3, 5, 16])
def test_a_prompt_read_in_chunks_gets_the_whole_prompts_embeddings(step):
    """Each chunk's embeddings, with the image features of the spans it overlaps, are the whole prompt's at its
    tokens, however the chunks cut the images; so are the deepstack features added to it."""
    rng = np.random.default_rng(0)
    table = mx.array(rng.standard_normal((32, 4)).astype(np.float32))
    body = type("Body", (), {"embed_tokens": staticmethod(lambda ids: table[ids])})()
    ids = list(range(1, 15))
    spans = [(2, 5), (5, 6), (9, 13)]  # two images next to each other, and one more
    feats = [mx.array(rng.standard_normal((e - s, 4)).astype(np.float32)) for s, e in spans]
    whole = np.array(table[mx.array(ids)])
    deep = np.zeros_like(whole)
    for (s, e), f in zip(spans, feats):
        whole[s:e] = np.array(f)
        deep[s:e] = np.array(f)
    hidden = mx.array(rng.standard_normal(whole.shape).astype(np.float32))
    for at in range(0, len(ids), step):
        n = min(step, len(ids) - at)
        assert np.array_equal(np.array(vision.embed_chunk(body, ids, spans, feats, at, n))[0], whole[at: at + n])
        added = vision.overlay(hidden[None, at: at + n], at, spans, feats, add=True)
        assert np.array_equal(np.array(added)[0], np.array(hidden)[at: at + n] + deep[at: at + n])


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
    svc._inflight, svc._owner = {}, {}
    svc._closed = svc._unloaded = False
    svc.prompt_tokens = svc.tokens_out = 0
    svc._free = svc._endless = eng.Controller
    monkeypatch.setattr(eng, "LocalsController", lambda c, planner, *a: ("grammar", c))

    class Rt:
        engine = type("E", (), {"prefill_embedded": lambda self, *a, **k: True})()

        def compile(self, schema, regex=None, ready=None, recursive_ok=False):
            compiled.append(schema)
            return type("C", (), {"constraint": len(compiled), "planner": None})()

    return svc, Rt()


def _image_job(**kw):
    from locals.service import Job

    pic = vision.Picture(None, (4, 4), key=7)  # (a 4x4 patch grid: 4 tokens over 2 positions)
    return Job(prompt=[1, 9, 9, 9, 9, 2, 3], pictures=[pic], spans=[(1, 5)], **kw)


def test_an_image_job_joins_the_batch_while_rows_decode(monkeypatch):
    """Taken while other rows decode (block False), an image job is a batch request at once, with its grammar,
    next to the text taken with it. It used to wait for an idle batch, which a busy server never had."""
    from locals.service import Job

    compiled = []
    svc, rt = _batch_service(monkeypatch, compiled)
    image = _image_job(schema={"type": "object"}, checkpoints=[(3, "system")])
    text = Job(prompt=[2, 3], checkpoints=[(1, "system")])
    svc._q.put(image)
    svc._q.put(text)
    reqs = svc._source(rt, block=False)
    assert [r.prompt for r in reqs] == [image.prompt, text.prompt]
    assert reqs[0].controller == ("grammar", 1) and compiled == [{"type": "object"}]
    assert isinstance(reqs[0].image, vision.ImagePrompt) and reqs[1].image is None
    # the image prompt's rope delta and cache key (test_an_image_prompts_cache_key_...); a text row's are 0 and its prompt
    assert reqs[0].rope_delta == -2 and reqs[1].rope_delta == 0
    assert reqs[0].cache_key == [1, -7, -7, -7, -7, 2, 3] and reqs[1].cache_key == [2, 3]
    assert reqs[0].checkpoints == [(3, "system")] and reqs[1].checkpoints == [(1, "system")]


def test_an_image_jobs_read_after_an_unload_says_the_model_reads_text_only(monkeypatch):
    """The tower is asked for at each read, not when the job is taken: one gone meanwhile (an unload, a checkpoint
    without its weights) ends the request with what the model cannot do."""
    svc, rt = _batch_service(monkeypatch, [])
    svc._q.put(_image_job())
    (r,) = svc._source(rt, block=False)
    svc.vision = lambda: None
    assert r.image.read(None, 0, 6) == "error: m reads text only"


def _image_prompt(encoder=None, prefill=None, cancelled=lambda: False, **kw):
    """The image prompt of ``_image_job`` (a 4x4-grid image at 1..5 of 7 tokens), on a stub tower and prefill."""
    job = _image_job()
    return vision.ImagePrompt(kw.get("prompt", job.prompt), kw.get("spans", job.spans), kw.get("pictures", job.pictures),
                              encoder=encoder or (lambda: None), prefill=prefill or (lambda *a, **k: True),
                              cancelled=cancelled)


def test_an_image_prompts_cache_key_names_its_images_and_its_rope_runs_behind_its_length():
    ip = _image_prompt()
    # each image's pad tokens as the image's own id, so its checkpoints are kept in the prompt cache too
    assert ip.key == [1, -7, -7, -7, -7, 2, 3]
    # 1 text + the image's 2 positions + 2 text: the reply's first token is at 5, its index 7, so the rope goes -2
    assert ip.rope_delta == -2
    # an image with no id: the prompt is not cached
    assert _image_prompt(pictures=[vision.Picture(None, (4, 4))]).key is None


def _encoder(encode, section=(11, 11, 10)):
    return type("Enc", (), {"encode": lambda self, pics: encode(pics), "mrope_section": section})()


def test_an_image_prompt_read_reads_the_tower_then_the_prompt():
    read = []
    enc = _encoder(lambda pics: ("features", ["deep"]))
    ip = _image_prompt(encoder=lambda: enc, prefill=lambda cache, *a, **k: read.append((cache, a, k)) or True)
    assert ip.read("cache", 0, 6) is None
    (cache, (prompt, spans, features, pos, deep, section), k), = read
    assert cache == "cache" and prompt == [1, 9, 9, 9, 9, 2, 3] and spans == [(1, 5)] and features == "features"
    assert pos.shape == (3, 1, 7) and deep == ["deep"] and section == (11, 11, 10)
    assert (k["start"], k["stop"]) == (0, 6)


def test_an_image_prompt_read_from_a_cached_prefix_runs_the_tower_on_the_images_after_it():
    """From a cached prefix (``start``), and in parts between checkpoints, the tower reads only the images the
    prefix does not hold, once, and each part of the prompt is read from where the last one stopped."""
    towered, read = [], []
    enc = _encoder(lambda pics: towered.append([p.key for p in pics]) or (["f2"], [["d2"]]))
    two = dict(prompt=[1, 9, 9, 9, 9, 2, 9, 9, 9, 9, 3, 4], spans=[(1, 5), (6, 10)],
               pictures=[vision.Picture(None, (4, 4), key=1), vision.Picture(None, (4, 4), key=2)])
    prefill = lambda cache, prompt, spans, features, pos, deep, section, **k: (  # noqa: E731
        read.append((spans, features, deep, k["start"], k["stop"])) or True)
    ip = _image_prompt(encoder=lambda: enc, prefill=prefill, **two)
    assert ip.read("cache", 5, 8) is None and ip.read("cache", 8, 11) is None
    assert towered == [[2]]
    assert read == [([(6, 10)], ["f2"], [["d2"]], 5, 8), ([(6, 10)], ["f2"], [["d2"]], 8, 11)]
    # a prefix that holds every image: no tower at all
    ip = _image_prompt(encoder=lambda: enc, prefill=prefill, **two)
    assert ip.read("cache", 10, 11) is None and towered == [[2]]
    assert read[-1] == ([], [], [], 10, 11)


def test_an_image_prompt_that_fails_to_read_returns_its_error():
    """An error reading the image prompt (memory the prefill could not get, a tower it has no weights for) is the
    request's end, returned to the batch engine, not an exception out of serve(), which would fail every job."""
    def oom(*a, **k):
        raise RuntimeError("[metal::malloc] Attempting to allocate 9 GB")

    ip = _image_prompt(encoder=lambda: _encoder(lambda pics: (None, ())), prefill=oom)
    end = ip.read(None, 0, 6)
    assert end.startswith("error: RuntimeError") and "allocate" in end

    def text_only():
        raise ValueError("m reads text only")

    assert _image_prompt(encoder=text_only).read(None, 0, 6) == "error: m reads text only"


def test_an_image_prompt_cancelled_while_it_waits_does_not_run_the_tower():
    towered = []
    ip = _image_prompt(encoder=lambda: _encoder(lambda pics: towered.append(pics) or ([], [])),
                       cancelled=lambda: True)
    assert ip.read(None, 0, 6) == "cancelled" and towered == []


def test_an_image_prompt_whose_prefill_stops_for_a_cancel_says_so():
    ip = _image_prompt(encoder=lambda: _encoder(lambda pics: ([], [])), prefill=lambda *a, **k: False)
    assert ip.read(None, 0, 6) == "cancelled"


@pytest.mark.parametrize("left, second, stop, end", [  # each way ImagePrompt.read returns, past a first part
    (False, lambda: True, 6, None),  # read to its end
    (False, lambda: False, 3, "cancelled"),  # a cancel inside the prefill, between its chunks
    (True, lambda: True, 3, "cancelled"),  # a cancel between two parts: the read returns before the tower
    (False, lambda: 1 / 0, 3, "error: ZeroDivisionError: division by zero"),  # the tower's or the prefill's error
])
def test_an_image_prompt_lets_go_of_the_towers_features_when_its_read_ends(left, second, stop, end):
    """The features are kept between the parts of a read (a checkpoint: the first part here), and let go of however
    the read ends: the request (Job._req) outlives its prefill, and must not hold the images' features meanwhile."""
    import gc
    import weakref

    class Features:
        pass

    made = []

    def encode(pics):
        made.append(Features())
        return [made[-1]], []

    outcomes = [lambda: True, second]  # the prefill's, part by part
    gone = []
    ip = _image_prompt(encoder=lambda: _encoder(encode), prefill=lambda *a, **k: outcomes.pop(0)(),
                       cancelled=lambda: bool(gone))
    assert ip.read(None, 0, 2) is None  # a part before a checkpoint
    held = weakref.ref(made.pop())
    gc.collect()
    assert held() is not None
    if left:
        gone.append(True)  # (its client left)
    assert ip.read(None, 2, stop) == end
    gc.collect()
    assert held() is None


def test_an_image_rows_rope_past_the_prompt_is_mrope():
    """Past the prompt, an image row's M-RoPE is the layer's own rope at its index plus rope_delta, also as one
    row of a batch whose offsets differ per row (what the batch engine's attention layers do)."""
    rope = nn.RoPE(64, traditional=False, base=10_000_000)
    pos = vision.positions([(3, 19)], [(8, 8)], 24)  # 3 text, a 4x4-token image over 4 positions, 5 text
    delta = vision.rope_delta(pos)
    assert delta == 3 + 4 + 5 - 24
    full = vision.tail(pos, 6)[:, None, :]
    x = mx.random.normal((1, 4, 1, 256)).astype(mx.float16)
    for i in (23, 24, 29):  # the prompt's last token (fed by the first decode step), then reply tokens
        want = vision.MRoPE(64, 10_000_000, full)(x, offset=i)
        got = rope(x, offset=i + delta)
        assert float(mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max()) < 2e-2
        rows = rope(mx.concatenate([x, x]), offset=mx.array([i + 100, i + delta]))  # (a text row beside it)
        assert float(mx.abs(rows[1:].astype(mx.float32) - want.astype(mx.float32)).max()) < 2e-2


def test_the_rope_shift_moves_with_its_row():
    """_shift_rope changes only the attention layers' rope offsets: the row's shift stays with it when other rows
    leave (filter) and join (extend), and its mask and writes (_idx, left padding) are not touched."""
    from mlx_lm.models.cache import ArraysCache, BatchKVCache

    from locals.hybrid_batch import _shift_rope

    def layer(n_rows, length):
        c = BatchKVCache([0] * n_rows)
        c.update_and_fetch(mx.zeros((n_rows, 1, length, 4)), mx.zeros((n_rows, 1, length, 4)))
        return c

    kv, rec = layer(2, 5), ArraysCache(1)
    _shift_rope([kv, rec], [0, -7])
    assert kv.offset.tolist() == [5, -2] and kv._idx == 5 and kv.left_padding.tolist() == [0, 0]
    kv.filter(mx.array([1]))
    assert kv.offset.tolist() == [-2]
    kv.extend(layer(1, 3))
    assert kv.offset.tolist() == [-2, 3]
