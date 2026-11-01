"""An image prompt's reply decodes in the batch engine (HybridBatchEngine.serve with Request.image)
and gives the tokens of the synchronous engine's image path (Engine.generate_embedded, M-RoPE through the reply), alone
and beside text rows; the text rows give what they give without it. A small random Qwen3 with features made up for its
image tokens: no tower. And its prompt-cache entries, under the image's id, make a follow-up turn read only what it
adds."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models import qwen3

from locals import vision
from locals.engine import Controller, Engine
from locals.hybrid_batch import DiskPromptCache, HybridBatchEngine, chain_cache
from locals.packed import Request

N = 12  # reply tokens
SPANS = [(5, 21)]  # 16 image tokens (an 8x8 patch grid) over 4 positions
PROMPT = [3, 17, 40, 41, 42] + [99] * 16 + [60, 61, 62, 63, 64, 65, 66, 67, 68]
TEXT = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
KEY = 77  # the image's id in prompt-cache keys (vision.Picture.key)


class _Tok:
    eos_token_ids = (1,)

    def decode(self, ids):
        return " ".join(map(str, ids))


class _Free(Controller):
    eos_ids = (1,)


@pytest.fixture(scope="module")
def model():
    mx.random.seed(11)
    m = qwen3.Model(qwen3.ModelArgs(
        model_type="qwen3", vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-6, rope_theta=10_000.0,
        max_position_embeddings=4096, tie_word_embeddings=True))
    m.update(nn.utils.tree_map(lambda p: mx.random.normal(p.shape) * 0.5, m.parameters()))
    mx.eval(m.parameters())
    return m


def _image(model, prompt=PROMPT):
    feats = [mx.random.normal((16, 64), key=mx.random.key(2))]
    pos = vision.positions(SPANS, [(8, 8)], len(prompt))
    return feats, pos


def _reference(model, prompt=PROMPT):
    feats, pos = _image(model, prompt)
    full = vision.tail(pos, N + 8)[:, None, :]
    res = Engine(model, _Tok()).generate_embedded(prompt, SPANS, feats, full, _Free(), max_tokens=N,
                                                  section=(6, 5, 5))
    return res.tokens


def _image_request(model, read, prompt=PROMPT, key=KEY, checkpoints=None, cancelled=lambda: False):
    """``read`` gets each part of the prompt read, (start, stop); ``cancelled()``: its client has left."""
    feats, _ = _image(model, prompt)
    eng = Engine(model, _Tok())
    r = Request(list(prompt), _Free(), N)
    enc = type("Enc", (), {"encode": lambda self, pics: (feats, []), "mrope_section": (6, 5, 5)})()

    def prefill(cache, *a, start, stop, **k):
        read.append((start, stop))
        return eng.prefill_embedded(cache, *a, start=start, stop=stop, **k)

    r.image = vision.ImagePrompt(prompt, SPANS, [vision.Picture(None, (8, 8), key=key)], lambda: enc, prefill,
                                 cancelled)
    r.checkpoints = checkpoints
    return r


def _kept(lru):
    """The copies the prompt cache gets, as (tokens, cache type, cache)."""
    kept = []
    insert = lru.insert_cache
    lru.insert_cache = lambda m, tokens, c, cache_type="assistant": kept.append((list(tokens), cache_type, c)) or insert(
        m, tokens, c, cache_type=cache_type)
    return kept


def _serve(model, waves, lru=None, disk=None, room=True):
    """Serve ``waves`` (lists of requests): the first at once, each next one when the rows have decoded 3 tokens.
    ``disk``: the engine's DiskPromptCache; ``room``: False, the memory is short for every checkpoint's copy."""
    eng = HybridBatchEngine(model, _Tok(), n_slots=4, share_prefix=False)
    eng.disk = disk
    if not room:
        eng._room_for_copy = lambda lru, positions: False
    waves = [list(w) for w in waves]
    first = waves[0]
    done = []

    def source(block):
        if waves and (waves[0] is first or all(len(r.out) >= 3 for r in first)):
            got = waves.pop(0)
            for r in got:
                r.pend, r.fed = [], []
            return got
        return []

    eng.serve(source, lru=lru, on_done=done.append)
    return done


def test_the_image_row_decodes_as_the_synchronous_image_path(model):
    want = _reference(model)
    assert len(want) == N  # (a long enough reply to tell positions apart)
    read = []
    (r,) = _serve(model, [[_image_request(model, read)]])
    assert read == [(0, len(PROMPT) - 1)] and r.finish == "length"
    assert r.out == want


def test_beside_text_rows_the_image_row_and_the_text_rows_keep_their_tokens(model):
    want = _reference(model)
    (alone,) = _serve(model, [[Request(list(TEXT), _Free(), N)]])
    # admitted together, and joining rows already decoding (its rope shift then goes into a batch of others)
    for waves in ([[Request(list(TEXT), _Free(), N), _image_request(model, [])]],
                  [[Request(list(TEXT), _Free(), N)], [_image_request(model, [])]],
                  [[_image_request(model, [])], [Request(list(TEXT), _Free(), N)]]):
        done = _serve(model, waves)
        text = next(r for r in done if r.image is None)
        image = next(r for r in done if r.image is not None)
        assert image.out == want
        assert text.out == alone.out


def test_an_image_rows_prompt_cache_entry_is_under_its_key(model):
    """Not under its pad tokens: an entry there would be the next image's of the same size."""
    lru = chain_cache(max_size=8, max_bytes=1 << 30)
    (r,) = _serve(model, [[_image_request(model, [])]], lru=lru)
    assert len(lru) == 1
    assert lru.fetch_nearest_cache("locals", list(PROMPT[:-1]))[1] == list(PROMPT[5:-1])  # (the text before it only)
    c, rest = lru.fetch_nearest_cache("locals", r.cache_key + r.out)  # (the row holds the reply fed back, all N tokens)
    assert c is not None and rest == []


@pytest.mark.parametrize("beside", [False, True])
def test_an_image_rows_prompt_cache_entry_holds_no_rope_delta(model, beside):
    """The entry's rope offset is its length: the row's M-RoPE delta goes when the row leaves the batch (alone, by
    _only_row; beside a row still decoding, by mlx_lm's BatchKVCache.extract), and the next turn adds its own. A
    cache that kept the offset instead (BatchRotatingKVCache.extract) would read a follow-up at shifted positions."""
    from mlx_lm.models.cache import KVCache

    lru = chain_cache(max_size=8, max_bytes=1 << 30)
    image = _image_request(model, [])
    assert image.rope_delta != 0  # (else an offset that kept it would pass too)
    wave = [image] + ([Request(list(TEXT), _Free(), N + 4)] if beside else [])  # (the text row outlives the image row)
    _serve(model, [wave], lru=lru)
    c, rest = lru.fetch_nearest_cache("locals", image.cache_key + image.out)
    assert c is not None and rest == []
    layers = [layer for layer in c if isinstance(layer, KVCache)]
    assert layers and all(layer.offset == layer.keys.shape[2] for layer in layers)


def test_the_next_turn_reads_only_what_follows_the_cached_turn(model):
    """A follow-up (the turn, its reply, a question) starts from the turn's cached row: the tower and the prompt read
    only the new tokens, and its reply is the one of the whole prompt read cold."""
    lru = chain_cache(max_size=8, max_bytes=1 << 30)
    (first,) = _serve(model, [[_image_request(model, [])]], lru=lru)
    follow = PROMPT + first.out + [70, 71, 72]
    read = []
    (r,) = _serve(model, [[_image_request(model, read, follow)]], lru=lru)
    assert read == [(len(PROMPT) + N, len(follow) - 1)] and r.cached_tokens == len(PROMPT) + N
    assert r.out == _reference(model, follow)
    # another image of the same size: only the text before the image is reused
    read = []
    (other,) = _serve(model, [[_image_request(model, read, follow, key=KEY + 1)]], lru=lru)
    assert read == [(SPANS[0][0], len(follow) - 1)] and other.cached_tokens == SPANS[0][0]


def test_an_image_rows_checkpoints_are_kept_under_its_key(model):
    """Read in parts, the prompt stops at each checkpoint (past the image here), whose copy goes into the prompt cache
    under the key; and its reply is the one of the prompt read in one part."""
    lru = chain_cache(max_size=8, max_bytes=1 << 30)
    kept = _kept(lru)  # (a KV-only model's row then replaces the checkpoint's copy: it can be cut back)
    read = []
    (r,) = _serve(model, [[_image_request(model, read, checkpoints=[(24, "user")])]], lru=lru)
    assert read == [(0, 24), (24, len(PROMPT) - 1)] and r.out == _reference(model)
    assert [k[:2] for k in kept] == [(r.cache_key[:24], "user"), (r.cache_key[:-1] + r.fed, "assistant")]


def test_an_image_prompts_system_checkpoints_are_saved_to_disk_under_its_key(tmp_path, model):
    """As a text prompt's: a system message is text only, so its checkpoint is before the first image, where the key is
    the prompt's tokens; one past an image (made up here) is saved under the key, with the image's id."""
    disk = DiskPromptCache(tmp_path / "m", 1.0)
    lru = chain_cache(max_size=8, max_bytes=1 << 30)
    (r,) = _serve(model, [[_image_request(model, [], checkpoints=[(3, "system"), (24, "system")])]], lru=lru, disk=disk)
    assert r.finish == "length"
    assert sorted(disk.index.values()) == [tuple(PROMPT[:3]), tuple(r.cache_key[:24])]
    assert -KEY in r.cache_key[:24]


def _same_cache(a, b, n):
    """Both caches hold the same first ``n`` positions, bit for bit."""
    from mlx_lm.models.cache import KVCache

    pairs = [(x, y) for x, y in zip(a, b) if isinstance(x, KVCache)]
    assert pairs
    return all(mx.array_equal(x.keys[..., :n, :], y.keys[..., :n, :]).item()
               and mx.array_equal(x.values[..., :n, :], y.values[..., :n, :]).item() for x, y in pairs)


def _text_made(model, n):
    """The checkpoint at ``n`` of a text prompt that begins as PROMPT does, before its image."""
    lru = chain_cache(max_size=8, max_bytes=1 << 30)
    kept = _kept(lru)
    text = Request(PROMPT[:SPANS[0][0]] + [60, 61, 62], _Free(), N)
    text.checkpoints = [(n, "system")]
    _serve(model, [[text]], lru=lru)
    ((_, _, c),) = [k for k in kept if len(k[0]) == n]
    return c


def test_an_image_prompts_checkpoint_before_its_image_holds_what_a_text_prompts_does(model):
    """The text before the first image is read as a text prompt's (the layers' own rope, not M-RoPE, whose rounding
    differs), so a text request that starts from the checkpoint decodes what it would cold; the image's reply is still
    the synchronous image path's."""
    lru = chain_cache(max_size=8, max_bytes=1 << 30)
    kept = _kept(lru)
    read = []
    (image,) = _serve(model, [[_image_request(model, read, checkpoints=[(3, "system")])]], lru=lru)
    assert read == [(3, len(PROMPT) - 1)] and image.out == _reference(model)  # (to 3: _token_reader)
    ((tokens, _, c_image),) = [k for k in kept if len(k[0]) == 3]
    assert tokens == PROMPT[:3]
    assert _same_cache(c_image, _text_made(model, 3), 3)



def test_an_image_prompt_whose_client_left_reads_nothing_before_its_image(model):
    """Its client gone while it waited: the text before the image is not read, and no checkpoint is kept."""
    lru = chain_cache(max_size=8, max_bytes=1 << 30)
    kept = _kept(lru)
    read = []
    (r,) = _serve(model, [[_image_request(model, read, checkpoints=[(3, "system")], cancelled=lambda: True)]], lru=lru)
    assert r.finish == "cancelled" and r.out == [] and read == [] and kept == []


def test_an_image_prompt_whose_text_before_its_image_fails_to_read_is_finished_there(model, monkeypatch):
    """As a fault in its image's read: that request's error, not the batch's."""
    def oom(self, cache, toks, chunk=2048, stop=None):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(HybridBatchEngine, "_run", oom)
    lru = chain_cache(max_size=8, max_bytes=1 << 30)
    image = _image_request(model, [], checkpoints=[(3, "system")])
    _serve(model, [[image]], lru=lru)
    assert image.finish == "error: RuntimeError: out of memory" and image.out == []

@pytest.mark.parametrize("image", [False, True])
def test_a_system_checkpoint_is_saved_to_disk_when_its_copy_is_left_out(tmp_path, model, image):
    """Memory too short for a checkpoint's copy in the prompt cache: the system prompt still goes to disk (a file, no
    GPU memory), for a text prompt and an image prompt alike."""
    disk = DiskPromptCache(tmp_path / "m", 1.0)
    lru = chain_cache(max_size=8, max_bytes=1 << 30)
    if image:
        r = _image_request(model, [], checkpoints=[(3, "system")])
    else:
        r = Request(list(TEXT), _Free(), N)
        r.checkpoints = [(3, "system")]
    _serve(model, [[r]], lru=lru, disk=disk, room=False)
    assert list(disk.index.values()) == [tuple(r.prompt[:3])]
    assert lru.stats_by_type()["system"]["n_sequences"] == 0


def test_an_image_prompt_that_ends_at_its_read_is_finished_there(model):
    def oom(*a, **k):
        raise RuntimeError("out of memory")

    r = Request(list(PROMPT), _Free(), N)
    r.image = vision.ImagePrompt(PROMPT, SPANS, [vision.Picture(None, (8, 8), key=KEY)],
                                 lambda: type("Enc", (), {"encode": lambda self, pics: ([], []), "mrope_section": (6, 5, 5)})(), oom)
    text = Request(list(TEXT), _Free(), N)
    done = _serve(model, [[r, text]])
    assert r.finish == "error: RuntimeError: out of memory" and r.out == []
    assert text.finish == "length" and len(text.out) == N and done[0] is r
