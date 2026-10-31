"""An image prompt's reply decodes in the batch engine (HybridBatchEngine.serve with Request.embed and rope_delta)
and gives the tokens of the synchronous engine's image path (Engine.generate_embedded, M-RoPE through the reply), alone
and beside text rows; the text rows give what they give without it. A small random Qwen3 with features made up for its
image tokens: no checkpoint, no tower."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models import qwen3

from locals import vision
from locals.engine import Controller, Engine
from locals.hybrid_batch import HybridBatchEngine, chain_cache
from locals.packed import Request

N = 12  # reply tokens
SPANS = [(5, 21)]  # 16 image tokens (an 8x8 patch grid) over 4 positions
PROMPT = [3, 17, 40, 41, 42] + [99] * 16 + [60, 61, 62, 63, 64, 65, 66, 67, 68]
TEXT = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


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


def _image(model):
    feats = [mx.random.normal((16, 64), key=mx.random.key(2))]
    pos = vision.positions(SPANS, [(8, 8)], len(PROMPT))
    return feats, pos


def _reference(model):
    feats, pos = _image(model)
    full = vision.tail(pos, N + 8)[:, None, :]
    res = Engine(model, _Tok()).generate_embedded(PROMPT, SPANS, feats, full, _Free(), max_tokens=N,
                                                  section=(6, 5, 5))
    return res.tokens


def _image_request(model, read):
    feats, pos = _image(model)
    eng = Engine(model, _Tok())
    r = Request(list(PROMPT), _Free(), N)

    def embed(cache):
        read.append(len(PROMPT))
        return None if eng.prefill_embedded(cache, PROMPT, SPANS, feats, pos[:, None, :], section=(6, 5, 5)) else "cancelled"

    r.embed, r.rope_delta = embed, vision.rope_delta(pos)
    return r


def _serve(model, waves, lru=None):
    """Serve ``waves`` (lists of requests): the first at once, each next one when the rows have decoded 3 tokens."""
    eng = HybridBatchEngine(model, _Tok(), n_slots=4, share_prefix=False)
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
    assert read == [len(PROMPT)] and r.finish == "length"
    assert r.out == want


def test_beside_text_rows_the_image_row_and_the_text_rows_keep_their_tokens(model):
    want = _reference(model)
    (alone,) = _serve(model, [[Request(list(TEXT), _Free(), N)]])
    # admitted together, and joining rows already decoding (its rope shift then goes into a batch of others)
    for waves in ([[Request(list(TEXT), _Free(), N), _image_request(model, [])]],
                  [[Request(list(TEXT), _Free(), N)], [_image_request(model, [])]],
                  [[_image_request(model, [])], [Request(list(TEXT), _Free(), N)]]):
        done = _serve(model, waves)
        text = next(r for r in done if r.embed is None)
        image = next(r for r in done if r.embed is not None)
        assert image.out == want
        assert text.out == alone.out


def test_an_image_row_leaves_no_prompt_cache_entry(model):
    """Its pad tokens name no prompt: an entry under them would be the next image's of the same size."""
    lru = chain_cache(max_size=8, max_bytes=1 << 30)
    _serve(model, [[_image_request(model, []), Request(list(TEXT), _Free(), N)]], lru=lru)
    assert len(lru) == 1
    assert lru.fetch_nearest_cache("locals", list(PROMPT[:-1]))[0] is None


def test_an_image_prompt_that_ends_at_its_read_is_finished_there(model):
    r = _image_request(model, [])
    r.embed = lambda cache: "error: RuntimeError: out of memory"
    text = Request(list(TEXT), _Free(), N)
    done = _serve(model, [[r, text]])
    assert r.finish == "error: RuntimeError: out of memory" and r.out == []
    assert text.finish == "length" and len(text.out) == N and done[0] is r
