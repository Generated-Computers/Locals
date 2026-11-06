"""Parked rows (hybrid_batch: a row stays in the batch and reads nothing while its controller waits, Qwenputer's
reply while its computer makes a run). The fused Gated DeltaNet step's held flags: a held row keeps its conv and
recurrent state, the other rows get the plain step's values, bit for bit."""

import mlx.core as mx
import numpy as np
import pytest

from locals.gdn_kernel import gdn_decode


def test_held_rows_keep_their_state_the_others_step_as_ever():
    Hk, Hv, Dk, Dv, KC, B = 16, 32, 128, 128, 4, 5
    C = 2 * Hk * Dk + Hv * Dv
    P = C + Hv * Dv + 2 * Hv
    rnd = np.random.RandomState(0)
    proj = mx.array(rnd.randn(B, 1, P).astype(np.float32)).astype(mx.bfloat16)
    conv = mx.array(rnd.randn(B, KC - 1, C).astype(np.float32)).astype(mx.bfloat16)
    conv_w = mx.array(rnd.randn(C, KC).astype(np.float32) * 0.3).astype(mx.bfloat16)
    A_log = mx.array(rnd.randn(Hv).astype(np.float32) * 0.1)
    dt_bias = mx.array(rnd.randn(Hv).astype(np.float32) * 0.1)
    state = mx.array(rnd.randn(B, Hv, Dv, Dk).astype(np.float32) * 0.1)
    norm_w = mx.array(rnd.randn(Dv).astype(np.float32)).astype(mx.bfloat16)
    args = (proj, conv, conv_w, A_log, dt_bias, state, norm_w, Hk, Hv, Dk, Dv)
    y0, c0, s0 = gdn_decode(*args)
    held = [0, 1, 0, 0, 1]
    y1, c1, s1 = gdn_decode(*args, held=mx.array(held, dtype=mx.int32))
    mx.eval(y0, c0, s0, y1, c1, s1)
    f = lambda a: np.array(a.astype(mx.float32))  # noqa: E731
    for b, h in enumerate(held):
        if h:
            assert np.array_equal(f(c1[b]), f(conv[b])) and np.array_equal(f(s1[b]), f(state[b]))
            assert not f(y1[b]).any()
        else:
            for a, z in ((y1, y0), (c1, c0), (s1, s0)):
                assert np.array_equal(f(a[b]), f(z[b]))


# -- the engine: a parked row goes on as if it had read its span at once ------------------------------------------------

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
PROMPTS = ["Write a short paragraph about lighthouses.", "List five prime numbers and say why each is prime.",
           "Explain what a hash table is in two sentences."]


@pytest.fixture(scope="module")
def rt():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip(f"{MODEL} is not in the local cache")
    from locals.runtime import Runtime

    return Runtime.load(MODEL, tune="off")


def _serve(rt, at, span, park, after=None, wait_s=None):
    """Three requests; the first, once it wrote ``at`` tokens, goes on with ``span``: fed as a forced span (``park``
    False), or parked and let go with it once the other two wrote ``after`` more tokens (or after ``wait_s`` seconds, the
    first request alone)."""
    import threading

    from locals.engine import Controller
    from locals.hybrid_batch import HybridBatchEngine
    from locals.packed import Request

    eng = HybridBatchEngine(rt.model, rt.tokenizer, n_slots=4, share_prefix=False)
    assert eng.can_park()
    eng.parks = park
    state = {"written": None}

    class Parks(Controller):
        eos_ids = list(rt.vocab.eos_ids)

        def __init__(self):
            self.n, self.parked, self.ready = 0, False, None

        def accept(self, token):
            self.n += 1
            if self.n == at and park:
                self.parked = True
                if wait_s is not None:
                    threading.Timer(wait_s, lambda: (setattr(self, "ready", list(span)), eng.wake())).start()

        def forced(self, token):
            return (list(span), False) if self.n == at and not park else ([], False)

        def release(self):
            got, self.ready = self.ready, None
            if got is not None:
                self.parked = False
            return got

    class Counts(Controller):
        eos_ids = list(rt.vocab.eos_ids)

        def accept(self, token):
            first = reqs[0].controller  # (the two rows count together: about two tokens a step)
            if first.parked and state["written"] is None:
                state["written"] = 0
            if state["written"] is not None:
                state["written"] += 1
                if state["written"] == after:
                    first.ready = list(span)

    n = 1 if wait_s is not None else len(PROMPTS)
    reqs = [Request(rt.encode_chat(p), Parks() if i == 0 else Counts(), 80) for i, p in enumerate(PROMPTS[:n])]
    for r in reqs:
        r.pend, r.fed = [], []
    sent = [False]

    def source(block):
        if sent[0]:
            return []
        sent[0] = True
        return list(reqs)

    eng.serve(source, temp=0.0)
    return [r.out for r in reqs], eng.counters


@pytest.mark.model
@pytest.mark.parametrize("case", ["span shorter than the wait", "span longer than the wait", "no span", "alone"])
def test_a_parked_row_goes_on_as_if_it_read_its_span_at_once(rt, case):
    span = rt.tokenizer.encode(" <tool_response> 1234567 </tool_response> So the result is", add_special_tokens=False)
    kw = {"span shorter than the wait": dict(after=4 * len(span)), "span longer than the wait": dict(after=3),
          "no span": dict(after=6), "alone": dict(wait_s=0.3)}[case]
    if case == "no span":
        span = []
    ref, _ = _serve(rt, 12, span, False, **kw)
    got, counters = _serve(rt, 12, span, True, **kw)
    assert got == ref
    absorbed = counters.get("absorbed", 0)  # (the span read in one pass but its last token, as many as the parked slots)
    if case == "span shorter than the wait":
        assert absorbed == len(span) - 1
    elif case == "span longer than the wait":
        assert 0 < absorbed < len(span) - 1


@pytest.mark.model
def test_a_waiting_request_takes_a_finished_rows_slot_in_place(rt):
    """The parked rows' engine: a request waiting for a slot takes a finished row's in place (its keys and values
    written into the slot, its recurrent states copied), and every request's tokens are the ones it gets without."""
    from locals.engine import Controller
    from locals.hybrid_batch import HybridBatchEngine
    from locals.packed import Request

    class Free(Controller):
        eos_ids = []

    prompts = PROMPTS + ["Name three rivers in Europe.", "What is a prime number?"]

    def serve(parks):
        eng = HybridBatchEngine(rt.model, rt.tokenizer, n_slots=2, share_prefix=False)
        eng.parks = parks
        reqs = [Request(rt.encode_chat(p), Free(), n) for p, n in zip(prompts, (20, 35, 12, 28, 16))]
        for r in reqs:
            r.pend, r.fed = [], []
        sent = [False]

        def source(block):
            if sent[0]:
                return []
            sent[0] = True
            return list(reqs)

        eng.serve(source, temp=0.0)
        return [r.out for r in reqs], eng.counters

    ref, _ = serve(False)
    got, counters = serve(True)
    assert got == ref
    assert counters.get("slots_reused", 0) >= 2


@pytest.mark.model
def test_a_finished_rows_slot_waits_for_the_next_request(rt):
    """A request that comes a step after a row finished (its client's next one) still takes that row's slot in
    place: the slot waits a few steps, held; the tokens are the ones it gets without."""
    from locals.engine import Controller
    from locals.hybrid_batch import HybridBatchEngine
    from locals.packed import Request

    class Free(Controller):
        eos_ids = []

    def serve(parks):
        eng = HybridBatchEngine(rt.model, rt.tokenizer, n_slots=2, share_prefix=False)
        eng.parks = parks
        reqs = [Request(rt.encode_chat(p), Free(), n) for p, n in zip(PROMPTS, (14, 30, 20))]
        for r in reqs:
            r.pend, r.fed = [], []
        sent = [0]

        def source(block):
            if sent[0] == 0:
                sent[0] = 1
                return reqs[:2]
            if sent[0] == 1 and reqs[0].finish:  # (the first is done: the next comes)
                sent[0] = 2
                return reqs[2:]
            return []

        eng.serve(source, temp=0.0)
        return [r.out for r in reqs], eng.counters

    ref, _ = serve(False)
    got, counters = serve(True)
    assert got == ref
    assert counters.get("slots_reused", 0) == 1


def test_an_image_rows_slot_in_place_keeps_its_rope_shift():
    """An image row's offset is its rope position, shifted by its M-RoPE delta (hybrid_batch._shift_rope), not its
    length: placed into a finished row's slot, all its keys are copied (its length read from its left padding), and
    the slot takes its shifted offset."""
    from mlx_lm.models.cache import BatchKVCache

    from locals.hybrid_batch import HybridBatchEngine, _shift_rope

    def kv(n, seed):
        return mx.random.normal((1, 2, n, 4), key=mx.random.key(seed))

    cache = BatchKVCache([0, 0])
    cache.update_and_fetch(mx.concatenate([kv(10, 1)] * 2), mx.concatenate([kv(10, 2)] * 2))
    merged = BatchKVCache([0])
    k, v = kv(6, 3), kv(6, 4)
    merged.update_and_fetch(k, v)
    _shift_rope([merged], [-3])  # (an image's tokens take fewer positions than their count)
    assert HybridBatchEngine._place([cache], 0, [merged], 0)
    assert cache.offset.tolist() == [3, 10] and cache.left_padding.tolist() == [4, 0]
    assert mx.array_equal(cache.keys[0:1, :, 4:10, :], k) and mx.array_equal(cache.values[0:1, :, 4:10, :], v)
