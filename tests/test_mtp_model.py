"""MTP drafts on the real model, through the server (issue #18): with ``LOCALS_MTP`` set, greedy replies are
token-identical to MTP off, or first differ at a float16 tie (decided with the owner on #18), for the 14 prompts of docs/measurements/mtp/mtp_gap_test.py (three of them constrained
tool calls), a turn that starts from a prompt-cache hit, a request that ran in a batch with another one and then
alone, and an image prompt; and the layer's drafts are taken. Needs Kaziko/Qwen3.5-4B-Zwyrol in the local cache and
the MTP file (docs/measurements/mtp/download_mtp.py; LOCALS_TEST_MTP names another path)."""

import os
import threading
from pathlib import Path

import pytest

MODEL = "Kaziko/Qwen3.5-4B-Zwyrol"
MTP_FILE = Path(os.environ.get("LOCALS_TEST_MTP") or Path(__file__).parent.parent / "docs/measurements/mtp/mtp_bf16.safetensors")
pytestmark = pytest.mark.model

# the prompts of docs/measurements/mtp/mtp_test.py and mtp_gap_test.py (the two long ones: the first 5,000 characters
# of locals/lookup.py and 6,000 of docs/engine/README.md, as there)
_WEATHER = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather forecast for a city",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "day": {"type": "string"}},
                            "required": ["city"]}}}]
_DESK = [{"type": "function", "function": {"name": "read_screen", "description": "Read the text in a window on the screen",
          "parameters": {"type": "object", "properties": {"window": {"type": "string"}}, "required": ["window"]}}},
         {"type": "function", "function": {"name": "save_note", "description": "Save a short note for the owner",
          "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
         {"type": "function", "function": {"name": "agent_status", "description": "What a named coding agent is doing now",
          "parameters": {"type": "object", "properties": {"agent": {"type": "string"}}, "required": ["agent"]}}}]
_REPO = Path(__file__).parent.parent
PROMPTS = [
    ("en-explain", "Explain how a refrigerator works, in about 150 words.", None),
    ("code", "Write a Python function that reads a CSV file and returns the average of its 'price' column. Handle a missing file and bad numbers.", None),
    ("pl-email", "Napisz krótki e-mail do sąsiada z prośbą o podlewanie kwiatów podczas mojego wyjazdu w przyszłym tygodniu.", None),
    ("pl-explain", "Wyjaśnij prostymi słowami, czym jest inflacja i jak wpływa na oszczędności.", None),
    ("json", "Return a JSON object describing three fictional books, with the fields title, author, year and genres (a list). Output only the JSON.", None),
    ("summary", "Summarize this in three bullet points:\n\nThe city council met on Tuesday to discuss the new bike lanes. "
     "Residents of the old town said the lanes take away parking spaces that shops depend on. Cyclists said the lanes "
     "cut accidents by a third in the first year in the northern district. The council agreed to a six-month trial on "
     "two streets, with a parking study before and after, and will vote on a full plan in spring.", None),
    ("math", "A train leaves at 14:20 and travels 245 km at 70 km/h. When does it arrive? Show the steps.", None),
    ("code-convert", "Convert this to TypeScript with types:\n\nfunction total(items, tax) {\n  let sum = 0;\n  for (const it of items) {\n"
     "    sum += it.price * it.qty;\n  }\n  return Math.round(sum * (1 + tax) * 100) / 100;\n}", None),
    ("tool-call", "What will the weather be in Kraków tomorrow?", _WEATHER),
    ("chat", "I'm tired after a long day. Give me three quick ideas to relax tonight.", None),
    ("pl-tool-screen", "Co mam teraz otwarte na ekranie? Przeczytaj, co jest w oknie przeglądarki.", _DESK),
    ("pl-tool-note", "Zapisz notatkę: jutro o dziesiątej telefon do Ani w sprawie faktury, a potem powiedz mi, co robi agent od kodu.", _DESK),
    ("long-code", "Explain what this code does:\n\n" + (_REPO / "locals" / "lookup.py").read_text()[:5000], None),
    ("long-doc", "Summarize this document:\n\n" + (_REPO / "docs" / "engine" / "README.md").read_text()[:6000], None),
]


def _req(messages, max_tokens=192, tools=None):
    req = {"model": "locals", "max_tokens": max_tokens, "temperature": 0, "messages": messages,
           "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        req["tools"] = tools
    return req


# the history of the conversation case: the same in every run
_ANSWER = ("A refrigerator moves heat from inside to outside. A compressor squeezes a refrigerant gas, the hot gas gives "
           "its heat away in the coils at the back, and the liquid then expands and turns cold in the coils inside.")


def _reply(out):
    m = out["choices"][0]["message"]
    return (m.get("content") or "", [(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls") or []])


def _engine_stats(srv):
    return srv.stats()["engines"]["locals"]


def _workload(srv):
    """Every case's (prompt tokens, reply tokens), as the service finished it, and what the cases needed to show."""
    svc = srv.service("locals")
    done = []
    finish = svc._finish

    def record(job, tokens, why):
        done.append((job.max_tokens, list(job.prompt), list(tokens)))
        return finish(job, tokens, why)

    svc._finish = record
    got, seen = {}, {}
    for name, text, tools in PROMPTS:
        out = srv.complete(_req([{"role": "user", "content": text}], tools=tools))
        got[name] = done[-1][1:]
        seen[name] = (_reply(out), out["usage"].get("locals", {}).get("forced_tokens", 0))
    # a conversation: its next turn starts from the prompt cache
    turn = [{"role": "user", "content": PROMPTS[0][1]}, {"role": "assistant", "content": _ANSWER},
            {"role": "user", "content": "And how is a freezer different?"}]
    own = _reply(srv.complete(_req(turn, 60)))[0]
    got["turn"] = done[-1][1:]
    out = srv.complete(_req(turn + [{"role": "assistant", "content": own},  # (its own reply, as an agent sends it back)
                                    {"role": "user", "content": "Which one uses more power?"}], 120))
    got["next-turn"] = done[-1][1:]
    seen["next-turn-cached"] = out["usage"]["prompt_tokens_details"]["cached_tokens"]
    # two at once: the short one ends, and the long one goes on alone
    before = _engine_stats(srv)
    ts = [threading.Thread(target=srv.complete, args=(_req([{"role": "user", "content": q}], n),))
          for q, n in (("Name three colours.", 12), (PROMPTS[1][1], 191))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    after = _engine_stats(srv)
    got["short"] = next(d[1:] for d in done[-2:] if d[0] == 12)
    got["long"] = next(d[1:] for d in done[-2:] if d[0] == 191)
    seen["together"] = (after["rows"] - before["rows"]) - (after["steps"] - before["steps"])  # steps with both rows
    seen["alone-drafts"] = after.get("draft_mtp_tokens", 0) - before.get("draft_mtp_tokens", 0)
    from test_vision_model import _invoice

    srv.complete(_req([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _invoice()}},
                                                    {"type": "text", "text": "What is the total, and who is the customer?"}]}], 60))
    got["image"] = done[-1][1:]
    return got, seen, _engine_stats(srv)


@pytest.fixture(scope="module")
def runs():
    from locals.defaults import cached_snapshot

    if cached_snapshot(MODEL, ["config.json", "*.safetensors"]) is None:
        pytest.skip(f"{MODEL} not cached")
    if not MTP_FILE.exists():
        pytest.skip(f"no MTP file at {MTP_FILE}")
    from locals.server import Server

    out = {}
    for label, env in (("off", {}), ("on1", {"LOCALS_MTP": str(MTP_FILE), "LOCALS_MTP_DRAFTS": "1"}),
                       ("on2", {"LOCALS_MTP": str(MTP_FILE), "LOCALS_MTP_DRAFTS": "2"})):
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LOCALS_MTP", raising=False)
            for k, v in env.items():
                mp.setenv(k, v)
            srv = Server(MODEL, engines=("locals",), preload=("locals",))
            try:
                out[label] = _workload(srv)
            finally:
                srv.unload()
    return out


@pytest.fixture(scope="module")
def scores(runs):
    """The main model's logits after a prompt and the start of a reply, teacher-forced in one pass (the servers are
    unloaded by now: one model at a time)."""
    import mlx.core as mx

    from locals.kernels import head_logits, text_model
    from locals.runtime import Runtime

    lm = text_model(Runtime.load(MODEL).model)

    def at(prompt, reply):
        return head_logits(lm, lm.model(mx.array([prompt + reply]))[:, -1:, :])[0, 0].astype(mx.float32)

    return at


def _tie(logits, a: int, b: int) -> bool:
    """Whether tokens a and b are the two best and their logits at most one float16 step apart."""
    import math

    import mlx.core as mx

    top = mx.argsort(-logits)[:2].tolist()
    va, vb = logits[a].item(), logits[b].item()
    step = 2.0 ** (math.floor(math.log2(max(abs(va), abs(vb), 1e-3))) - 10)
    return set(top) == {a, b} and abs(va - vb) <= step


@pytest.mark.parametrize("setting", ["on1", "on2"])
@pytest.mark.parametrize("case", [p[0] for p in PROMPTS] + ["turn", "next-turn", "short", "long", "image"])
def test_greedy_replies_are_the_same_with_mtp_drafts(runs, scores, setting, case):
    """The same tokens, or (decided with the owner on issue #18) the first different one at a float16 tie: a pass
    over several tokens rounds differently from a one-token step, so a tie may break the other way. An image prompt
    is not teacher-forced here: its reply must be the same."""
    (prompt, off), (prompt_on, on) = runs["off"][0][case], runs[setting][0][case]
    if case == "next-turn" and prompt_on != prompt:  # (its history is the run's own reply to "turn": that one differed)
        assert runs["off"][0]["turn"][1] != runs[setting][0]["turn"][1]
        pytest.skip("the turn before differed at a tie, so this turn's prompt differs")
    assert prompt_on == prompt
    i = next((i for i, (a, b) in enumerate(zip(off, on)) if a != b), None)
    if i is None:
        assert len(on) == len(off)
        return
    assert case != "image", (i, off[i], on[i])
    assert _tie(scores(prompt, off[:i]), off[i], on[i]), (i, off[i], on[i])


def test_the_workload_reaches_each_case(runs):
    _, seen, _ = runs["off"]
    for k in ("tool-call", "pl-tool-screen", "pl-tool-note"):
        (_, calls), forced = seen[k]
        assert calls and forced > 0  # (a tool call, under the tool-call grammar)
    assert seen["next-turn-cached"] > 50  # (the next turn started from the cache of the turn before)
    assert seen["together"] > 0  # (the two ran in steps together)
    assert all(runs[s][1]["alone-drafts"] > 0 for s in ("on1", "on2"))  # (and the long one then drafted alone)


@pytest.mark.parametrize("setting", ["on1", "on2"])
def test_the_mtp_layer_drafts_and_its_drafts_are_kept(runs, setting):
    stats = runs[setting][2]
    assert stats.get("draft_mtp_tokens", 0) > 500
    assert stats["draft_mtp_accepted"] > 0.5 * stats["draft_mtp_tokens"] / (1 if setting == "on1" else 2)
    assert "draft_mtp_tokens" not in runs["off"][2]
