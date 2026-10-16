"""One request running alone in the batch engine (HybridBatchEngine._solo): forced spans fed in one pass and
prompt-lookup drafts verified in the same pass give the tokens one-token steps give, and a request arriving
meanwhile joins as before. Needs Qwen3.5-4B in the local cache."""

import json

import pytest

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
pytestmark = pytest.mark.model

FILE = "\n".join(f"def f{i}(x):\n    return x * {i} + {i % 7}\n" for i in range(12))
EDIT = ("Here is a module:\n\n```python\n" + FILE + "```\n\nRename every function fN to gN and keep everything else. "
        "Reply with the whole module in one ```python block.")
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                                           "city": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
          "required": ["name", "age", "city", "tags"]}


@pytest.fixture(scope="module")
def rt():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip(f"{MODEL} is not in the local cache")
    from locals.runtime import Runtime

    return Runtime.load(MODEL, tune="auto")


def _serve(rt, make_requests, solo: bool, monkeypatch, temp=0.0, arrive_after=None):
    """Serve the requests; with ``arrive_after`` = n, the second request arrives once the first wrote n tokens."""
    from locals import hybrid_batch
    from locals.hybrid_batch import HybridBatchEngine

    monkeypatch.setattr(hybrid_batch, "_SOLO", solo)
    eng = HybridBatchEngine(rt.model, rt.tokenizer, n_slots=4, share_prefix=False)
    reqs = make_requests()
    for r in reqs:
        r.pend, r.fed = [], []
    waiting = list(reqs)
    first = waiting.pop(0)
    sent = [False]

    def source(block):
        if not sent[0]:
            sent[0] = True
            return [first] if arrive_after is not None else [first] + [waiting.pop(0) for _ in list(waiting)]
        if waiting and len(first.out) >= (arrive_after or 0):
            return [waiting.pop(0)]
        return []

    eng.serve(source, temp=temp)
    return reqs, eng.counters


def _free(rt, n):
    from locals.engine import Controller
    from locals.packed import Request

    class Free(Controller):
        eos_ids = list(rt.vocab.eos_ids)

    return Request(rt.encode_chat(EDIT), Free(), n)


def _json_request(rt, n):
    from locals.engine import LocalsController
    from locals.packed import Request

    c = rt.compile(SCHEMA)
    return Request(rt.encode_chat("Ana Silva, 29, lives in Lisbon; tags: vip, early. As JSON."),
                   LocalsController(c.constraint, c.planner, True, True), n)


def test_drafts_give_the_same_greedy_tokens(rt, monkeypatch):
    off, _ = _serve(rt, lambda: [_free(rt, 300)], False, monkeypatch)
    on, counters = _serve(rt, lambda: [_free(rt, 300)], True, monkeypatch)
    assert on[0].out == off[0].out
    assert counters.get("drafted", 0) > 50 and counters.get("solo_steps", 0) > 0  # the answer repeats the module


def test_a_constrained_request_alone_gives_the_same_json(rt, monkeypatch):
    # (this schema's forced spans are one token each: the boundary feeds that token, so no step of several
    # tokens is needed; the row goes back to the pipelined steps)
    off, _ = _serve(rt, lambda: [_json_request(rt, 120)], False, monkeypatch)
    on, _ = _serve(rt, lambda: [_json_request(rt, 120)], True, monkeypatch)
    assert on[0].out == off[0].out
    assert json.loads(rt.tokenizer.decode(on[0].out))["age"] == 29


def test_sampled_requests_alone_stay_valid_and_use_drafts(rt, monkeypatch):
    on, counters = _serve(rt, lambda: [_free(rt, 200)], True, monkeypatch, temp=0.7)
    assert len(on[0].out) > 50 and "def g" in rt.tokenizer.decode(on[0].out)
    assert counters.get("drafted", 0) > 0  # sampled drafts are accepted with the model's own probability
    on, _ = _serve(rt, lambda: [_json_request(rt, 120)], True, monkeypatch, temp=0.7)
    assert json.loads(rt.tokenizer.decode(on[0].out))["city"]


def test_a_request_arriving_meanwhile_joins_and_both_answers_are_the_same(rt, monkeypatch):
    make = lambda: [_free(rt, 200), _json_request(rt, 80)]  # noqa: E731
    off, _ = _serve(rt, make, False, monkeypatch, arrive_after=40)
    on, counters = _serve(rt, make, True, monkeypatch, arrive_after=40)
    assert [r.out for r in on] == [r.out for r in off]
    assert counters.get("solo_steps", 0) > 0


COPY_SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}, "text": {"type": "string"}},
               "required": ["title", "text"]}
SENTENCES = " ".join(f"Item {i} costs {i * 3} dollars and ships in {i % 5 + 1} days." for i in range(25))


def _copy_request(rt, n):
    from locals.engine import LocalsController
    from locals.packed import Request

    c = rt.compile(COPY_SCHEMA)
    prompt = rt.encode_chat("Put this text, copied exactly, into the text field, with the title Prices:\n\n" + SENTENCES)
    return Request(prompt, LocalsController(c.constraint, c.planner, True, True), n)


def test_drafts_under_a_grammar_give_the_same_tokens(rt, monkeypatch):
    off, _ = _serve(rt, lambda: [_copy_request(rt, 500)], False, monkeypatch)
    on, counters = _serve(rt, lambda: [_copy_request(rt, 500)], True, monkeypatch)
    assert on[0].out == off[0].out
    assert json.loads(rt.tokenizer.decode(on[0].out))["title"]
    assert counters.get("drafted", 0) > 50  # the field repeats the prompt
