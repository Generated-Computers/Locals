"""MTP drafts (issue #18, docs/adr/0001-mtp-kv-in-the-cache-list.md): the MTP layer's KV is one more entry at the end
of a request's cache list, its offset equal to the main cache's wherever the cache goes, and its drafts never change a
reply. Runs on a tiny random Qwen3.5 (four layers, the fused kernels' head sizes) and a random MTP layer, so it needs
no download; the real model's checks are in test_mtp_model.py."""

import mlx.core as mx
import mlx.nn as nn
import pytest

EOS = 1


@pytest.fixture(scope="module")
def tiny():
    from mlx_lm.models import qwen3_5 as Q

    from locals.fuse import fuse_model

    args = Q.TextModelArgs(model_type="qwen3_5", hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=128, vocab_size=512,
                           linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=128,
                           linear_value_head_dim=128, tie_word_embeddings=False)
    mx.random.seed(0)
    m = Q.TextModel(args)
    nn.quantize(m, group_size=64, bits=4)
    assert fuse_model(m) == 4
    return m


@pytest.fixture(scope="module")
def mtp_file(tiny, tmp_path_factory):
    """A random MTP layer for the tiny model, saved as the checkpoint stores it (``mtp.`` names, norms as w)."""
    from locals.mtp import Layer

    mx.random.seed(1)
    layer = Layer(tiny.args)
    path = tmp_path_factory.mktemp("mtp") / "mtp.safetensors"
    from mlx.utils import tree_flatten

    flat = {f"mtp.{k}": (v - 1.0 if v.ndim == 1 else v) for k, v in tree_flatten(layer.parameters())}
    mx.save_safetensors(str(path), flat)
    return path


class Tok:
    eos_token_ids = [EOS]

    def decode(self, t):
        return " ".join(map(str, t))


def _free():
    from locals.engine import Controller

    class Free(Controller):
        eos_ids = [EOS]

    return Free()


PROMPT = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]


def _engine(tiny, mtp=None):
    from locals.hybrid_batch import HybridBatchEngine

    return HybridBatchEngine(tiny, Tok(), n_slots=4, share_prefix=False, mtp=mtp)


def _serve(eng, reqs, lru=None):
    from locals.packed import Request

    reqs = [Request(list(p), _free(), n) for p, n in reqs]
    waiting = list(reqs)

    def source(block):
        out = waiting[:]
        waiting.clear()
        return out

    eng.serve(source, lru=lru)
    return reqs


def _lru():
    from locals.hybrid_batch import chain_cache

    return chain_cache(max_size=16, max_bytes=1 << 30)


def _offsets(cache):
    """Positions each attention entry of a single-row cache holds."""
    from mlx_lm.models.cache import KVCache

    return [c.offset for c in cache if isinstance(c, KVCache)]


def test_mtp_off_keeps_the_cache_list_of_the_model(tiny):
    from mlx_lm.models.cache import make_prompt_cache

    lru = _lru()
    (r,) = _serve(_engine(tiny), [(PROMPT, 8)], lru)
    cache = lru.fetch_nearest_cache("locals", PROMPT + r.out + [0] * 20)[0]
    assert len(cache) == len(make_prompt_cache(tiny))


def test_mtp_on_adds_one_entry_with_the_main_offset(tiny, mtp_file):
    from mlx_lm.models.cache import make_prompt_cache

    from locals.mtp import MTP

    lru = _lru()
    (r,) = _serve(_engine(tiny, MTP.load(tiny, mtp_file)), [(PROMPT, 8)], lru)
    cache = lru.fetch_nearest_cache("locals", PROMPT + r.out + [0] * 20)[0]
    assert len(cache) == len(make_prompt_cache(tiny)) + 1
    offs = _offsets(cache)
    assert len(set(offs)) == 1, offs


def _real(entry):
    """Which slots of a single-row MTP entry hold the layer's KV (1) and which are holes (0)."""
    return [int(v > 0) for v in entry.keys[0, 0, : entry.offset, -1].tolist()]


def test_a_read_prompt_is_in_the_mtp_entry_up_to_its_last_token(tiny, mtp_file):
    """Every prompt position the read covers gets the layer's KV: the token after each is known, the last one's too
    (the prompt's last token, which the first decode step feeds)."""
    from locals.mtp import MTP

    lru = _lru()
    (r,) = _serve(_engine(tiny, MTP.load(tiny, mtp_file)), [(PROMPT, 3)], lru)
    cache = lru.fetch_nearest_cache("locals", PROMPT + r.out + [0] * 20)[0]
    assert _real(cache[-1])[: len(PROMPT) - 1] == [1] * (len(PROMPT) - 1)


def _oracle(tiny, mtp_file, truth, wrong, drafts):
    """The MTP layer, run as it is, but with its draft tokens taken from ``truth`` (the reply with MTP off) and the
    ones ``wrong(pass, i)`` names changed: the passes then keep all, some or none of the draft. Each draft records
    the entry's written positions next to the positions the request has fed (its own bookkeeping)."""
    from locals.mtp import MTP, _written

    class Oracle(MTP):
        seen = []

        def draft(self, entry, out, r, n):
            got = super().draft(entry, out, r, n)
            assert len(got) == n
            k = len(Oracle.seen)
            Oracle.seen.append((_written(entry), len(r.prompt) - 1 + len(r.fed) - 1))
            m = len(r.out)
            return [(truth[m + i] + 1) % 500 + 2 if wrong(k, i) else truth[m + i] for i in range(n) if m + i < len(truth)]

    base = MTP.load(tiny, mtp_file, drafts=drafts)
    o = Oracle(tiny, base.layer, drafts)
    o.dtype = base.dtype
    return o


LONG = [5, 9, 3, 8, 2, 7, 4, 6, 33, 41, 57, 12, 99, 70, 18, 64, 23, 81, 45, 30]


@pytest.mark.parametrize("drafts,wrong", [
    (1, lambda k, i: False),          # every draft kept
    (1, lambda k, i: k % 2 == 1),     # every other draft rejected outright
    (2, lambda k, i: False),
    (2, lambda k, i: i == 1),         # the second draft rejected: part kept
    (2, lambda k, i: k % 3 == 2),     # both rejected every third pass
])
def test_mtp_drafts_keep_the_reply_and_the_entry_at_the_main_offset(tiny, mtp_file, drafts, wrong):
    (off,) = _serve(_engine(tiny), [(LONG, 40)])
    mtp = _oracle(tiny, mtp_file, off.out, wrong, drafts)
    lru = _lru()
    eng = _engine(tiny, mtp)
    (on,) = _serve(eng, [(LONG, 40)], lru)
    assert on.out == off.out
    assert mtp.seen and all(a == b for a, b in mtp.seen), mtp.seen
    assert eng.counters["draft_mtp_tokens"] > 0
    assert 0 < eng.counters["draft_mtp_accepted"] <= eng.counters["draft_mtp_tokens"]
    cache = lru.fetch_nearest_cache("locals", LONG + on.out + [0] * 20)[0]
    assert len(set(_offsets(cache))) == 1, _offsets(cache)
    assert sum(_real(cache[-1])[len(LONG):]) > len(on.out) // 2  # (the solo passes wrote the layer's KV)


def test_batched_rows_and_a_restored_prefix_keep_the_entry_at_the_main_offset(tiny, mtp_file):
    """Two requests decode together (batched steps leave holes), then one starts from a cached prefix of the first
    (the prompt cache's copy, its entry with it): the same replies as with MTP off, every kept cache at one offset."""
    from locals.mtp import MTP

    other = [40, 41, 42, 43, 44, 45, 46, 47, 48]

    def run(mtp):
        lru = _lru()
        eng = _engine(tiny, mtp)
        a, b = _serve(eng, [(LONG, 24), (other, 30)], lru)
        (c,) = _serve(eng, [(LONG + a.out + [77, 78], 20)], lru)
        return lru, eng, (a, b, c)

    _, _, ref = run(None)
    lru, eng, got = run(MTP.load(tiny, mtp_file))
    assert [r.out for r in got] == [r.out for r in ref]
    assert got[2].cached_tokens > len(LONG)  # (it did start from the first one's cache)
    assert eng.counters["steps"] > eng.counters.get("solo_steps", 0)  # (batched steps ran)
    for r in got[1:]:  # (the first one's entry went: the third one's extends it, chain_cache)
        cache = lru.fetch_nearest_cache("locals", r.prompt + r.out + [0] * 20)[0]
        assert cache is not None
        assert len(set(_offsets(cache))) == 1, _offsets(cache)


def test_a_prompt_read_in_mixed_passes_fills_the_entry(tiny, mtp_file):
    """A prompt read a chunk at a time beside a decoding row (locals/mixed.py): its entry gets the layer's KV at every
    position the read covers, and the replies are MTP off's."""
    from locals.mtp import MTP

    longp = [(7 * i) % 400 + 3 for i in range(90)]

    def run(mtp):
        lru = _lru()
        eng = _engine(tiny, mtp)
        eng.mix_chunk = 16
        from locals.packed import Request

        first = Request(list(LONG), _free(), 60)
        second = Request(list(longp), _free(), 10)
        sent = []

        def source(block):
            if not sent:
                sent.append(1)
                return [first]
            if len(sent) == 1 and len(first.out) >= 2:
                sent.append(1)
                return [second]
            return []

        eng.serve(source, lru=lru)
        return lru, eng, first, second

    _, _, f0, s0 = run(None)
    lru, eng, f1, s1 = run(MTP.load(tiny, mtp_file))
    assert (f1.out, s1.out) == (f0.out, s0.out)
    assert eng.counters.get("mixed_steps", 0) > 0
    row = lru.fetch_nearest_cache("locals", LONG + f1.out + [0] * 20)[0]  # (the row that decoded beside the read)
    assert len(set(_offsets(row))) == 1, _offsets(row)
    cache = lru.fetch_nearest_cache("locals", longp + s1.out + [0] * 20)[0]
    assert len(set(_offsets(cache))) == 1, _offsets(cache)
    assert _real(cache[-1])[: len(longp) - 1] == [1] * (len(longp) - 1)


def test_the_memory_guard_counts_the_mtp_entry(tiny, mtp_file):
    """One more attention layer next to the tiny model's one: twice the KV bytes a token (Qwen3.5-4B: 9 / 8)."""
    from locals.mtp import MTP

    off, on = _engine(tiny)._kv_bytes_per_token(), _engine(tiny, MTP.load(tiny, mtp_file))._kv_bytes_per_token()
    assert on == 2 * off


def test_a_saved_cache_keeps_its_entry_and_holes(tiny, mtp_file, tmp_path):
    """The disk cache saves and loads the list as it is: the entry, its offset and its hole marks."""
    from locals.hybrid_batch import DiskPromptCache
    from locals.mtp import MTP

    lru = _lru()
    eng = _engine(tiny, MTP.load(tiny, mtp_file))
    (r,) = _serve(eng, [(PROMPT, 6)], lru)
    key = PROMPT + r.out
    cache = lru.fetch_nearest_cache("locals", key + [0] * 20)[0]
    tokens = PROMPT[:-1] + r.fed
    disk = DiskPromptCache(tmp_path / "d")
    disk.save(tokens, cache)
    back = _lru()
    assert DiskPromptCache(tmp_path / "d").restore(back, "locals", tokens + [0, 0]) == len(tokens)
    loaded = back.fetch_nearest_cache("locals", tokens + [0, 0])[0]
    assert len(loaded) == len(cache)
    assert _offsets(loaded) == _offsets(cache)
    assert _real(loaded[-1]) == _real(cache[-1]) and 0 in _real(cache[-1])  # (the last fed token's slot: a hole)


def test_the_single_request_engine_gets_no_entry(tiny, mtp_file):
    from mlx_lm.models.cache import make_prompt_cache

    from locals.engine import make_spec_cache
    from locals.mtp import MTP

    _engine(tiny, MTP.load(tiny, mtp_file))
    assert len(make_spec_cache(tiny)) == len(make_prompt_cache(tiny)) == len(tiny.layers)


def test_another_model_keeps_mtp_off_with_one_line(tiny, mtp_file, monkeypatch, capsys):
    from locals.service import _load_mtp

    monkeypatch.setenv("LOCALS_MTP", str(mtp_file))
    assert _load_mtp(tiny, False) is None
    assert _load_mtp(tiny, True) is None
    lines = [l for l in capsys.readouterr().err.splitlines() if "LOCALS_MTP" in l]
    assert len(lines) == 2
    assert "not Qwen3.5-4B" in lines[0] and "Qwenputer" in lines[1]
    monkeypatch.delenv("LOCALS_MTP")
    assert _load_mtp(tiny, False) is None and capsys.readouterr().err == ""


def test_mtp_on_and_off_never_share_disk_cache_files(mtp_file, tmp_path):
    from locals.service import _disk_folder

    off = _disk_folder(tmp_path, "m/x", "float16")
    off.mkdir()
    on = _disk_folder(tmp_path, "m/x", "float16", mtp_file)
    on.mkdir()
    assert on != off
    assert _disk_folder(tmp_path, "m/x", "float16") == off and on.exists()  # (neither one's start deletes the other)
    assert _disk_folder(tmp_path, "m/x", "float16", mtp_file) == on and off.exists()


def _image_request(tiny, prompt, spans, n):
    """An image prompt on the tiny model: made-up features for its image tokens (no tower), as test_vision_batch."""
    from locals import vision
    from locals.engine import Engine
    from locals.packed import Request

    feats = [mx.random.normal((16, 256), key=mx.random.key(2)).astype(mx.float32)]
    eng = Engine(tiny, Tok())
    r = Request(list(prompt), _free(), n)
    enc = type("Enc", (), {"encode": lambda self, pics: (feats, []), "mrope_section": (6, 5, 5)})()
    r.image = vision.ImagePrompt(prompt, spans, [vision.Picture(None, (8, 8), key=77)], lambda: enc,
                                 eng.prefill_embedded, lambda: False)
    return r


def test_an_image_prompt_gets_no_mtp_drafts_and_keeps_its_reply(tiny, mtp_file):
    """The M-RoPE swap covers only the model's own layers: the image row's entry is holes past the text before its
    image, its reply is MTP off's, and the layer drafts nothing for it."""
    from locals.mtp import MTP

    prompt = [3, 17, 40, 41, 42] + [99] * 16 + [60, 61, 62, 63, 64, 65, 66, 67, 68]

    def run(mtp):
        lru = _lru()
        eng = _engine(tiny, mtp)
        r = _image_request(tiny, prompt, [(5, 21)], 30)
        waiting = [r]
        eng.serve(lambda block: [waiting.pop()] if waiting else [], lru=lru)
        return lru, eng, r

    _, _, ref = run(None)
    lru, eng, got = run(MTP.load(tiny, mtp_file))
    assert got.out == ref.out and len(got.out) == 30
    assert eng.counters.get("draft_mtp_tokens", 0) == 0
    cache = lru.fetch_nearest_cache("locals", got.cache_key + got.out)[0]
    assert len(set(_offsets(cache))) == 1, _offsets(cache)
