"""Qwenputer (locals/models/qwenputer.py, qwenputer_walk.py): Qwen3.5-4B with a RISC-V computer compiled into 11 layers
before its own, which runs the model's calc tool.

With Qwen3.5-4B's tokenizer cached: the presenter shows a calc exchange the model ran before its answer as reasoning
(thinking on or off), leaves out one after content, and passes another tool's call to the client.

With the chip pack made (``locals serve qwenputer`` makes it once; LOCALS_QWENPUTER_PACK names another folder): the
computer's ways of reading tokens (MLX operations, the device stepper, the block pass) agree on every state and
prediction, and from the firmware it makes the run the authors' reference machine made for a calc call (recorded in
data/qwenputer_calc_run.json), stepped on the GPU, one run or several at once."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from locals.models import qwenputer as Q

RUN = json.loads((Path(__file__).parent / "data" / "qwenputer_calc_run.json").read_text())


def test_names():
    from locals.defaults import model_id

    assert model_id("qwenputer") == Q.REPO and Q.is_qwenputer(model_id("qwenputer"))
    assert not Q.is_qwenputer(model_id("qwen"))


# -- the reply as the client reads it ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tok():
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(Q.BASE, "tokenizer.json"), str):
        pytest.skip("Qwen3.5-4B's tokenizer is not cached")
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    return load_tokenizer(Path(try_to_load_from_cache(Q.BASE, "tokenizer.json")).parent)


V0 = 248320
CALL = "<tool_call>\n<function=calc>\n<parameter=expression>\n1+1\n</parameter>\n</function>\n</tool_call>"
ECHO = "\n<tool_response>\n2\n</tool_response>\n<think>\n\n</think>\n\n"


def _show(tok, thinking, events):
    from locals.models.qwenputer_walk import Presenter

    p = Presenter(tok, thinking, V0)
    out = []
    for kind, text in events:
        if kind == "run":  # a run: trace tokens (here two), then the echoed text
            out += p.feed("run", [V0 + 7, V0 + 9] + tok.encode(ECHO, add_special_tokens=False))
        else:
            for t in tok.encode(text, add_special_tokens=False):
                out += p.feed("qwen", t)
    return tok.decode(out + p.finish())


def _parse(tok, text, thinking):
    import copy

    from locals import formats, oai

    fmt = formats.detect(tok)
    if not thinking:
        fmt = copy.copy(fmt)
        fmt.opens_thinking = True
    return oai.parse_reply(text, [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
                           fmt=fmt, thinking=thinking and not getattr(fmt, "opens_thinking", False))


def test_an_exchange_before_the_answer_is_reasoning(tok):
    shown = _show(tok, False, [("qwen", CALL), ("run", None), ("qwen", "It is 2.")])
    assert shown == "<think>" + CALL + "\n<tool_response>\n2\n</tool_response></think>It is 2."
    reasoning, content, calls = _parse(tok, shown, False)
    assert reasoning.startswith("<tool_call>") and reasoning.endswith("</tool_response>")
    assert content == "It is 2." and calls == []


def test_with_thinking_the_exchange_joins_the_models_reasoning(tok):
    shown = _show(tok, True, [("qwen", "Let me add.\n</think>\n\n"), ("qwen", CALL), ("run", None), ("qwen", "It is 2.")])
    assert shown == "Let me add.\n" + CALL + "\n<tool_response>\n2\n</tool_response></think>It is 2."
    reasoning, content, calls = _parse(tok, shown, True)
    assert "Let me add." in reasoning and "<tool_response>" in reasoning and content == "It is 2." and calls == []


def test_an_exchange_after_content_is_left_out(tok):
    assert _show(tok, False, [("qwen", "Sure. "), ("qwen", CALL), ("run", None), ("qwen", "It is 2.")]) == "Sure. It is 2."


def test_another_tools_call_is_the_clients(tok):
    call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
    shown = _show(tok, False, [("qwen", call)])
    assert shown == call
    _, content, calls = _parse(tok, shown, False)
    assert content == "" and [c["function"]["name"] for c in calls] == ["get_weather"]


# -- the computer ------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def comp():
    pack = Path(os.environ["LOCALS_QWENPUTER_PACK"]) if os.environ.get("LOCALS_QWENPUTER_PACK") else Q.pack_dir()
    if not (pack / "snapshot.safetensors").exists():
        pytest.skip("no Qwenputer chip pack on this Mac (locals serve qwenputer makes it)")
    from locals.models.qwenputer_walk import Computer

    return Computer(Q.Chip(pack))


def _at_call(comp):
    """The computer after the prompt and Qwen's calc call: the token it predicts there (GO)."""
    comp.start(RUN["prompt"])
    for t in RUN["call"][:-1]:
        assert comp.peek(t) == -1  # (Qwen's text: Qwen predicts)
    return comp.peek(RUN["call"][-1])


def _copy(comp):
    return tuple(mx.array(np.array(a)) for a in (comp.s, comp.c, comp.mk))


def test_the_ways_of_reading_agree(comp):
    """The run's first 300 tokens read by the MLX operations (equal to transformers' layers: see the module), the
    stepper (teacher-forced) and the block pass: the same predictions and states."""
    assert _at_call(comp) == RUN["run"][0]
    toks = RUN["run"][:300]
    s0, c0, m0 = _copy(comp)
    want = RUN["run"][1:301]

    block = comp.read(toks)
    sb, cb = np.array(comp.s), np.array(comp.c)
    assert block.tolist() == want

    comp.s, comp.c, comp.mk = (mx.array(np.array(a)) for a in (s0, c0, m0))
    bufs = comp.stepper.start(comp.s, comp.c, [toks[0]], mk=comp.mk)
    dep, pred, _ = comp.stepper.steps(bufs, len(toks), np.array([toks[1:] + [-1]], np.int32))
    mx.eval(dep)
    assert np.array(pred)[0, :len(toks)].tolist() == want
    assert np.array_equal(np.array(comp.s), sb) and np.array_equal(np.array(bufs["c"][bufs["parity"]]), cb)

    st = comp.chip.new_state(1)
    comp.w.unpack(mx.array(np.array(s0)), mx.array(np.array(c0)), st)
    preds = [int(comp.chip.predict(comp.chip(comp.chip.embed([[t]]), st))[0, -1]) for t in toks]
    assert preds == want
    s_mlx, c_mlx = comp.w.pack(st)
    assert np.array_equal(np.array(s_mlx), sb) and np.array_equal(np.array(c_mlx), cb)


def test_the_computer_makes_the_reference_run_stepped(comp):
    first = _at_call(comp)
    assert [first] + comp.step(first) == RUN["run"]


def test_a_kept_prompt_prefix_gives_the_same_state(comp):
    """Computer.start from the state kept after a prefix other prompts share: the state reading the whole prompt."""
    comp.start(RUN["prompt"])
    want = [np.array(a) for a in (comp.s, comp.c, comp.mk)]
    for _ in range(2):  # (the first keeps the prefix, the second starts from it)
        comp.start(RUN["prompt"], cuts=[300])
        assert all(np.array_equal(np.array(a), b) for a, b in zip((comp.s, comp.c, comp.mk), want))


# -- the controller --------------------------------------------------------------------------------------------------
def _pack():
    return Path(os.environ["LOCALS_QWENPUTER_PACK"]) if os.environ.get("LOCALS_QWENPUTER_PACK") else Q.pack_dir()


def test_a_held_call_gets_the_computers_run(comp, tok):
    """split: the row stops at the calc call's end; the computer reads the call, makes the run (in a pool with other
    replies' runs, one joining while another goes on: each its own), and the echo is what the reply goes on with."""
    from locals.models.qwenputer_walk import CalcController, Pool

    ctls = []
    for _ in range(2):
        ctl = CalcController(comp.dev, RUN["prompt"], tok, [], split=True)
        for t in RUN["call"]:
            ctl.accept(t)
        assert ctl.held and ctl.finished()
        ctls.append(ctl)
    firsts = [c.read_call() for c in ctls]
    assert firsts == [RUN["run"][0]] * 2
    pool = Pool(comp.dev)
    pool.add(ctls[0].comp, firsts[0], 0)
    done = dict(pool.advance() + pool.advance())  # (the second joins 64 steps into the first's run)
    pool.add(ctls[1].comp, firsts[1], 1)
    while len(pool):
        done.update(pool.advance())
    runs = [done[0], done[1]]
    assert runs == [RUN["run"]] * 2
    comp.start(RUN["prompt"])  # (alone: the states after the run are the pool's)
    comp.step(int(comp.read(RUN["call"])[-1]))
    for c in ctls:
        assert all(np.array_equal(np.array(a), np.array(b)) for a, b in zip((c.comp.s, c.comp.c, c.comp.mk), (comp.s, comp.c, comp.mk)))
    echo = ctls[0].ran(runs[0])
    assert echo == [t for t in RUN["run"] if t < V0] and not ctls[0].held
    assert "<tool_response>" in tok.decode(ctls[0].drain())


def test_a_run_on_a_cpu_core_is_the_gpus(comp, tmp_path):
    """The computer's run made on a CPU core (models/qwenputer_cpu.py, built with clang): the reference run, and the
    state, convolutions and masks the GPU's stepper leaves, bit for bit, on host copies and in place on the device's
    own buffers."""
    from locals.models.qwenputer_cpu import CPU, shared

    cpu = CPU.make(comp.dev, tmp_path)
    if cpu is None:
        pytest.skip("no C compiler (xcode-select --install)")
    first = _at_call(comp)
    s, c, mk = np.array(comp.s), np.array(comp.c), np.array(comp.mk)
    runs = []
    for threads in (1, 3):  # (alone, and its attention layers shared out to 3 threads)
        s1, c1, mk1 = s.copy(), c.copy(), mk.copy()
        rest, _ = cpu.run(s1, c1, mk1, first, threads=threads)
        runs.append((rest, s1, c1, mk1))
    # on the device's own buffers, as the server runs it: MLX arrays updated in place, nothing copied out or back
    on = _copy(comp)
    mx.eval(on)
    views = shared([(on[0], np.float32), (on[1], np.float32), (on[2], np.uint32)])
    assert views is not None
    rest, _ = cpu.run(*views, first)
    runs.append((rest, *on))
    # a run cut at max_tokens ends with -5, as the GPU's Pool ends one: a failure, not a run that ended (and so does
    # one stepped on the GPU alone)
    assert cpu.run(s.copy(), c.copy(), mk.copy(), first, max_tokens=50)[0] == RUN["run"][1:51] + [-5]
    assert _at_call(comp) == first and comp.step(first, max_tokens=48)[-1] == -5
    assert _at_call(comp) == first  # (back at the call)
    assert [first] + comp.step(first) == RUN["run"]
    for rest, s1, c1, mk1 in runs:
        assert [first] + rest == RUN["run"]
        for a, b in ((s1, comp.s), (c1, comp.c), (mk1, comp.mk)):
            assert np.array_equal(a, np.array(b))


# -- the toolbox -------------------------------------------------------------------------------------------------------
def test_a_question_is_offered_the_tools_it_asks_for():
    """The toolbox's tools a request is offered, by its last user message: arithmetic gets calc's prompt alone."""
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "docs" / "qwenputer"))
    from arithmetic import questions

    assert not any(Q.tools_for(q + " Use calc if you have it. End with the exact number.") for q, _ in questions(60))
    text, dates = ("count", "length", "reverse", "words", "count_letters"), ("weekday", "days_between", "add_days")
    assert Q.tools_for("How many r's are in strawberry?") == text
    assert Q.tools_for("How many r’s are in strawberry?") == text  # (a phone's apostrophe)
    assert Q.tools_for("How many vowels are in onomatopoeia?") == text
    assert Q.tools_for("Write the word Qwenputer backwards.") == text
    assert Q.tools_for("How many words are in this sentence?") == text
    assert Q.tools_for("Sort these numbers: 3, 1, 2") == ("sort",)
    assert Q.tools_for("What day of the week was 1969-07-20?") == dates
    assert Q.tools_for("How many days are there from 2026-01-01 to 2026-12-25?") == dates
    assert Q.tools_for("What date is 100 days after 2026-10-31?") == dates
    assert Q.tools_for("What is 255 in binary?") == ("to_base",)
    assert Q.tools_for("Convert 0xFF to decimal") == ("to_base",)
    assert Q.tools_for("Sort the letters of banana", have=("sort",)) == ("sort",)
    assert Q.tools_for("Put 3.5, 1 and 2 in increasing order") == ("sort",)
    # arithmetic in the words of a text, list or date question: calc alone
    for q in ("What do I need in order to compute 3*4?", "If I save $5 a day, how many days until I have $100?",
              "How many times larger is 48 than 6?"):
        assert Q.tools_for(q) == (), q


def test_a_long_token_is_read_as_shorter_ones(tok):
    """A toolbox call's token of more than 16 bytes (a row of the bridge's text buffer) goes to the computer as the
    vocabulary's tokens of at most 16 bytes it is made of; other tokens as they are."""
    from locals.models.qwenputer_walk import ROW, _rows, _vocab
    from locals.vocab import _BYTE_DECODER

    text = "x" + "-" * 20 + "y 北京师范大学 " + " " * 40 + "end"
    ids = tok.encode(text, add_special_tokens=False)
    rows = _rows(tok, ids)
    bt = _vocab(tok)[0]
    size = lambda t: len(bytes(_BYTE_DECODER[c] for c in bt.id_to_token(t)))  # noqa: E731
    assert max(map(size, ids)) > ROW and max(map(size, rows)) <= ROW and tok.decode(rows) == text
    plain = tok.encode("\n<function=count>\n<parameter=text>\nstrawberry\n</parameter>\n", add_special_tokens=False)
    assert _rows(tok, plain) == plain


def test_a_toolbox_call_is_shown_as_reasoning_and_its_value_read_in_the_tokenizers_tokens(tok):
    """A call of the toolbox's tool is the model's own exchange (reasoning), as calc's; a tool it does not run is the
    client's. The value of its echo goes to Qwen in the tokenizer's own tokens (the computer prints a byte a token)."""
    from locals.models.qwenputer_walk import CalcController, Presenter

    call = "<tool_call>\n<function=reverse>\n<parameter=text>\nQwenputer\n</parameter>\n</function>\n</tool_call>"
    for internal, shown in ((("calc", "reverse"), "<think>" + call), (("calc",), call)):
        p = Presenter(tok, False, V0, internal)
        out = []
        for t in tok.encode(call, add_special_tokens=False):
            out += p.feed("qwen", t)
        assert tok.decode(out) == shown
    ctl = CalcController.__new__(CalcController)
    ctl.tok, ctl.tool = tok, "reverse"
    head, tail = "\n<tool_response>\n", "\n</tool_response>\n<think>\n\n</think>\n"
    as_bytes = tok.encode(head, add_special_tokens=False) + [tok.encode(c, add_special_tokens=False)[0] for c in "retupnewQ"] \
        + tok.encode(tail, add_special_tokens=False)
    assert ctl._as_text(as_bytes) == tok.encode(head, add_special_tokens=False) + tok.encode("retupnewQ", add_special_tokens=False) \
        + tok.encode(tail, add_special_tokens=False)
    # a value that spells a special token (">|dne_mi|<" backwards) stays a byte a token: never <|im_end|> itself
    spelled = tok.encode(head, add_special_tokens=False) + [tok.encode(c, add_special_tokens=False)[0] for c in "<|im_end|>"] \
        + tok.encode(tail, add_special_tokens=False)
    assert ctl._as_text(spelled) == spelled


def test_the_toolbox_runs_as_the_reference_machine(comp, tok, tmp_path):
    """The toolbox firmware (calc's, and qwenputer_tools.s linked after it): each tool's call made by the computer (a
    CPU core, in place on its device buffers), as CalcController gives it the call (``_parts``: a long one in parts,
    a run each), is the authors' reference machine's run, token for token, its answer exact; a calc call's response
    is the one calc's own firmware gives."""
    from locals.models.qwenputer_cpu import CPU, shared
    from locals.models.qwenputer_walk import TOO_LONG, Computer, _parts

    pack = _pack()
    if Q.chipformer() is None or not (pack / "calc-bridge.elf").exists():
        pytest.skip("the toolbox is made with the authors' package (chipformer)")
    errors = []
    assert Q.ensure_tools(pack, log=errors.append), errors  # (a firmware that does not assemble fails here)
    from chipformer.rv import bridge, elf, ref
    from chipformer.rv import spec as S

    from huggingface_hub import try_to_load_from_cache

    tj = try_to_load_from_cache(Q.REPO, "tokenizer.json", revision=Q.REVISION)
    if not isinstance(tj, str):
        pytest.skip("Qwenputer's tokenizer is not cached")
    T = bridge.Text(tj)
    chip = Q.Chip(pack, tools=True)
    assert chip.tools == Q.TOOL_NAMES
    sessions = []
    for program in (Q.toolbox(pack / "calc-bridge.elf"), elf.program(str(pack / "calc-bridge.elf"))):
        m = ref.Machine.from_program(program)
        bridge.park(m)
        sessions.append(bridge.Session(T, m))
    tools, calc_only = sessions
    c = Computer(chip)
    cpu = CPU.make(c.dev, tmp_path)
    if cpu is None:
        pytest.skip("no C compiler (xcode-select --install)")
    prompt = T.encode("<|im_start|>user\nHow many r's are in strawberry?<|im_end|>\n<|im_start|>assistant\n")
    c.start(prompt)
    for s in sessions:
        s.history(prompt)
    for fn, params, answer in [("count", dict(text="strawberry", find="r"), "3"), ("calc", dict(expression="2**64 - 1"), str(2**64 - 1)),
                               ("reverse", dict(text="naïve"), "evïan"), ("weekday", dict(date="1969-07-20"), "Sunday"),
                               ("days_between", dict(start="2024-02-01", end="2024-03-01"), "29"),
                               ("sort", dict(numbers="5 3 -1 12 0"), "-1, 0, 3, 5, 12"), ("length", dict(text="naïve café"), "10"),
                               ("add_days", dict(date="2026-10-31", days="-366"), "2025-10-30"),
                               ("to_base", dict(number="0xFF", base="2"), "11111111"), ("words", dict(text="one two  three"), "3"),
                               ("count_letters", dict(text="Onomatopoeia", letters="a, e, i, o, u"), "8"),
                               ("sort", dict(numbers="4.99, 12.50, -3.75, 4.9, -4"), "-4, -3.75, 4.9, 4.99, 12.50"),
                               ("sort", dict(numbers="1,000, 250"), "error (numbers such as 12, -3, 4.5: no fractions, thousands separators, units or exponents)"),
                               ("add_days", dict(date="0000-03-01", days="-1"), "0000-02-29"), ("weekday", dict(date="\t2024-01-05\r"), "Friday"),
                               ("length", dict(text="a" + " " * 20 + "b"), "22"), ("reverse", dict(text="北京师范大学"), "学大范师京北"),
                               ("words", dict(text=" ".join(["cat", "dog", "bird"] * 70)), "210"),  # (3 parts)
                               ("sort", dict(numbers=", ".join(map(str, range(40, 0, -1)))), ", ".join(map(str, range(1, 41)))),
                               ("words", dict(text=" ".join(["internationalization"] * 300)), "300"),  # (its text past 0x4000)
                               ("words", dict(text=" ".join(["cat"] * 100)), None), ("calc", dict(expression="6*7"), "42"),
                               ("words", dict(text=" ".join(["cat"] * 100)), None), ("length", dict(text="abc"), "3"),
                               ("sort", dict(numbers=".5, -.5, −2, 0.25"), "−2, -.5, 0.25, .5"),
                               ("words", dict(text="a " * 1200), "error (the call is too long for the computer: at most about 1,000 tokens or 7,000 characters)"),
                               (TOO_LONG, {}, "error (the call is too long for the computer: at most about 1,000 tokens or 7,000 characters)")]:
        body = f"\n<function={fn}>\n" + "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items()) + "</function>\n"
        parts = [T.encode(body)] if fn == "calc" else _parts(tok, T.encode(body))
        for part in parts if answer is not None else parts[:1]:  # (None: a long call cut short after its first part)
            call = [T.TCO] + part + [T.TCC]
            n0 = len(tools.toks)
            tools.qwen(call)
            want = [chip.v0 + t if not S.is_text(t) else t - S.TEXT_TOKEN for t in tools.toks[n0 + len(call):]]
            first = int(c.read(call)[-1])
            mx.eval(c.s, c.c, c.mk)
            rest, _ = cpu.run(*shared([(c.s, np.float32), (c.c, np.float32), (c.mk, np.uint32)]), first)
            assert [first] + rest == want, fn
        if answer is None:
            c.read([T.NL])
            continue
        echo = T.decode_bytes([t for t in want if t < chip.v0]).decode()
        assert echo.split("\n")[2] == answer, (fn, echo)
        if fn == "calc":
            calc_only.qwen(call)
            assert T.decode_bytes(calc_only.runs[-1]).decode() == echo
        c.read([T.NL])
    # the controller as the server's computer thread drives it: a long call's parts, a run each (nothing shown, nothing
    # for Qwen to read), then the call's response
    from locals.models.qwenputer_walk import CalcController

    ctl = CalcController(c.dev, prompt, tok, [], split=True, internal=("calc", "words"))
    body = "\n<function=words>\n<parameter=text>\n" + " ".join(["cat", "dog", "bird"] * 70) + "\n</parameter>\n</function>\n"
    for t in tok.encode("<tool_call>" + body + "</tool_call>", add_special_tokens=False):
        ctl.accept(t)
    parts = 0
    while True:
        first = ctl.read_call()
        mx.eval(ctl.comp.s, ctl.comp.c, ctl.comp.mk)
        rest, _ = cpu.run(*shared([(ctl.comp.s, np.float32), (ctl.comp.c, np.float32), (ctl.comp.mk, np.uint32)]), first)
        echo = ctl.ran([first] + rest)
        if not ctl.parts:
            break
        assert echo == [] and ctl.held
        parts += 1
    assert parts == 2 and tok.decode(echo).split("\n")[2] == "210" and not ctl.held
    assert tok.decode(ctl.drain()).count("<tool_response>") == 1


def test_the_server_offers_the_toolbox_and_gives_a_turn_its_exchange_back(tok):
    """Server.prepare for Qwenputer (no GPU: a stand-in service with the tokenizer): a question that asks for a toolbox
    tool is offered it and its computer runs it; arithmetic is offered calc alone; a client's own tool of the same name
    is the client's; and a reply the client sends back without its reasoning, where the computer ran a tool, is written
    as the client sending it back with it would have it written."""
    from types import SimpleNamespace

    from locals.server import Server

    srv = Server("qwenputer", engines=("locals",), preload=(), log=lambda s: None)
    srv._svc["locals"] = SimpleNamespace(tokenizer=tok, qwenputer_tools=Q.TOOL_NAMES)
    ask = lambda msgs, **kw: srv.prepare(dict(model="locals/qwen3.5-4b-qwenputer", max_tokens=64, messages=msgs, **kw))  # noqa: E731
    q = {"role": "user", "content": "How many r's are in the word strawberry?"}
    p = ask([q])
    assert p["internal"] == ("calc", "count", "length", "reverse", "words", "count_letters") and '"name": "count"' in p["text"]
    p = ask([{"role": "user", "content": "What is 12345 * 6789?"}])
    assert p["internal"] == ("calc",) and '"name": "count"' not in p["text"]
    mine = {"type": "function", "function": {"name": "count", "description": "mine", "parameters": {"type": "object", "properties": {}}}}
    assert "count" not in ask([q], tools=[mine])["internal"]
    exchange = ("<tool_call>\n<function=count>\n<parameter=text>\nstrawberry\n</parameter>\n<parameter=find>\nr\n"
                "</parameter>\n</function>\n</tool_call>\n<tool_response>\n3\n</tool_response>")
    answer = 'There are 3 r\'s in the word "strawberry".'
    then = {"role": "user", "content": "Now write it backwards."}
    with_it = ask([q, {"role": "assistant", "content": answer, "reasoning_content": exchange}, then])["text"]
    without = ask([q, {"role": "assistant", "content": answer}, then])["text"]
    assert without != with_it
    from locals.server import _digest

    srv._exchanges[_digest(answer)] = None
    assert ask([q, {"role": "assistant", "content": answer}, then])["text"] == with_it
    # a tool's result sent as a user turn is not a question: the turn before it is the current one, left as sent
    result = {"role": "user", "content": "<tool_response>\nok\n</tool_response>"}
    assert ask([q, {"role": "assistant", "content": answer}, result])["text"] == ask([q, {"role": "assistant", "content": answer,
                                                                                    "reasoning_content": ""}, result])["text"]
    # a reply as the conversation's last message (a prefill) is left as sent: it is written with its reasoning
    assert ask([q, {"role": "assistant", "content": answer}])["text"] == ask([q, {"role": "assistant", "content": answer,
                                                                                "reasoning_content": ""}])["text"]


def test_a_reply_with_a_computers_exchange_leaves_no_request_in_flight(tok):
    """The server's requests in flight (/stats, locals status): a Qwenputer reply whose computer ran a tool is taken out
    at its end (its exchange's digest once took the name of the request's key, and the request stayed in for good)."""
    from locals.server import Server
    from locals.service import Job

    reply = ("<think><tool_call>\n<function=count>\n<parameter=text>\nstrawberry\n</parameter>\n<parameter=find>\nr\n"
             "</parameter>\n</function>\n</tool_call>\n<tool_response>\n3\n</tool_response>\n</think>\n\nThere are 3 r's.")
    toks = tok.encode(reply, add_special_tokens=False)

    def submit(prompt, max_tokens=16, on_token=None, **kw):
        job = Job(list(prompt), max_tokens=max_tokens, tokens=list(toks), text=reply, finish="stop")
        on_token(list(toks))
        job._event.set()
        return job

    srv = Server("qwenputer", engines=("locals",), preload=(), log=lambda s: None)
    srv._svc["locals"] = SimpleNamespace(tokenizer=tok, qwenputer_tools=Q.TOOL_NAMES, submit=submit)
    out = srv._run(dict(model="locals/qwen3.5-4b-qwenputer", max_tokens=64,
                        messages=[{"role": "user", "content": "How many r's are in the word strawberry?"}]))
    assert out["choices"][0]["message"]["content"] == "There are 3 r's." and srv._exchanges and srv._active == {}


def _stub_controller(tok, internal=("calc",), **computer):
    """A CalcController without a computer (its runs given by hand), parked at calls as the server's are."""
    from types import SimpleNamespace

    from locals.models.qwenputer_walk import CalcController

    tco, tcc = tok.convert_tokens_to_ids("<tool_call>"), tok.convert_tokens_to_ids("</tool_call>")
    dev = SimpleNamespace(chip=SimpleNamespace(v0=V0, info={"tool_call": (tco, tcc)}))
    prompt = tok.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    return CalcController(dev, prompt, tok, [], park=True, internal=internal, computer=computer)


def _response(tok, value):
    return [V0 + 7] + tok.encode(f"\n<tool_response>\n{value}\n</tool_response>\n<think>\n\n</think>\n\n", add_special_tokens=False)


def test_a_clients_call_waits_for_the_computers_check(tok):
    """With the client's tools' schemas (the server's "checks"), a call of the client's is held at its </tool_call>
    while the computer checks it: one that is right goes to the client as written, and the model goes on after it; one
    that is wrong is the computer's exchange (reasoning), its response read by the model, which writes the call again."""
    from locals.server import _check_schemas

    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}]
    head, checks = _check_schemas(tools)
    call = "<tool_call>\n<function=get_weather>\n<parameter=days>\nthree\n</parameter>\n</function>\n</tool_call>"
    for value, ok in (("ok", True), ("error (the required parameter city is missing)", False)):
        ctl = _stub_controller(tok, tools=head, checks=checks)
        parked = []
        ctl.on_park = lambda: parked.append(True)
        for t in tok.encode(call, add_special_tokens=False):
            ctl.accept(t)
        assert parked and ctl.parked and ctl.checking == "get_weather" and ctl.drain() == []
        echo = ctl.ran(_response(tok, value))
        shown = tok.decode(ctl.drain() + ctl.finish())
        if ok:
            assert echo == [] and shown == call
        else:
            assert tok.decode(echo).startswith("\n<tool_response>\nerror (the required parameter city")
            assert shown.startswith("<think>" + call + "\n<tool_response>\nerror") and ctl.check_tries == 1


def test_a_number_the_answer_copies_waits_for_the_computers_check(tok):
    """A number of the answer's of 6 digits or more, after the computer gave a result of as many, is held while
    copy_check reads it (what follows waits behind it): shown as the computer's result if it was miscopied; the
    reply's end waits for the check."""
    ctl = _stub_controller(tok, copy=True)
    asked = []
    ctl.on_copy = lambda c, results, number: asked.append((results, number))
    ctl.results = ["100000000000000000000000000000000000000"]
    ctl.pres.state = "content"
    for t in tok.encode("It is 100,000,000,000,000,000,000,000,000,000,000,000. That is 10^38", add_special_tokens=False):
        ctl.accept(t)
    assert asked == [("100000000000000000000000000000000000000", "100,000,000,000,000,000,000,000,000,000,000,000")]
    assert tok.decode(ctl.drain()) == "It is "
    ends = []
    assert ctl.finish_after(lambda: ends.append(True))
    ctl.copied("100,000,000,000,000,000,000,000,000,000,000,000,000")
    assert ends == [True]
    assert tok.decode(ctl.drain()) == "100,000,000,000,000,000,000,000,000,000,000,000,000. That is 10^38"
    # a short number, or one when the computer gave none as long, goes at once
    ctl = _stub_controller(tok, copy=True)
    ctl.on_copy = lambda *a: asked.append(a)
    ctl.results, ctl.pres.state = ["42"], "content"
    for t in tok.encode("The answer is 42 and 1234567.", add_special_tokens=False):
        ctl.accept(t)
    assert not ctl.finish_after(lambda: None) and tok.decode(ctl.drain()) == "The answer is 42 and 1234567." and len(asked) == 1


def test_a_number_too_long_to_check_is_shown_as_written(tok):
    """A number of the answer's longer than copy_check reads (a model writing ",000" on and on) is shown as it comes,
    not held to the reply's end (where the server's loop guard could not see it), and not checked."""
    ctl = _stub_controller(tok, copy=True)
    asked = []
    ctl.on_copy = lambda *a: asked.append(a)
    ctl.results, ctl.pres.state = ["100000000000000000000000000000000000000"], "content"
    for t in tok.encode("It is 100" + ",000" * 60, add_special_tokens=False):
        ctl.accept(t)
    assert len(tok.decode(ctl.drain())) > 150 and ctl.run_on
    for t in tok.encode(" and so on. Then 123,456,789. Done", add_special_tokens=False):
        ctl.accept(t)
    assert len(asked) == 1 and asked[0][2] == "123,456,789" and not ctl.run_on


def test_the_notes_of_earlier_replies_go_to_the_next_requests_computer(tok):
    """The server keeps the notes calls its computer ran in a reply (by the conversation before it and its content),
    and a request that continues the conversation gives the notes they left to its computer with its first notes
    call; a reply the client sends back with its reasoning has them there. Another conversation with a reply of the
    same words gets none of them."""
    from types import SimpleNamespace

    from locals import oai
    from locals.models.qwenputer_walk import _note_op, _notes_state
    from locals.server import Server, _turn_digests, _turn_key

    assert _note_op("remember", "\n<function=remember>\n<parameter=key>\nwifi\n</parameter>\n<parameter=value>\nhunter\t2\n"
                                "</parameter>\n</function>\n") == ("remember", "wifi", "hunter 2")
    assert _note_op("forget", "<parameter=key>\n\"wifi\"\n</parameter>") == ("forget", "wifi", "")  # (as notes.c reads it)
    assert _notes_state([("remember", "a", "1"), ("remember", "B", "2"), ("remember", "A", "3"), ("forget", "b", ""),
                         ("remember", "x" * 61, "4")]) == "remember\tA\t3"
    assert _notes_state([("remember", f"k{i}", "v" * 100) for i in range(30)]).count("\n") == 18  # (2,048 bytes)
    srv = Server("qwenputer", engines=("locals",), preload=(), log=lambda s: None)
    srv._svc["locals"] = SimpleNamespace(tokenizer=tok, qwenputer_tools=Q.TOOL_NAMES)
    later = "<tool_call>\n<function=forget>\n<parameter=key>\ngym\n</parameter>\n</function>\n</tool_call>\n<tool_response>\nforgot \"gym\"\n</tool_response>"
    msgs = [{"role": "user", "content": "Remember my wifi password: hunter2, and gym is Monday."},
            {"role": "assistant", "content": "Noted."}, {"role": "user", "content": "Forget the gym."},
            {"role": "assistant", "content": "Done.", "reasoning_content": later}, {"role": "user", "content": "What do you remember?"}]
    turns = _turn_digests(oai.convert_messages(msgs))
    srv._notes[_turn_key(turns[1], "Noted.")] = [("remember", "wifi", "hunter2"), ("remember", "gym", "Monday")]
    p = srv.prepare(dict(model="locals/qwen3.5-4b-qwenputer", max_tokens=64, messages=msgs))
    assert p["computer"]["notes"] == "remember\twifi\thunter2"
    assert set(p["internal"]) >= {"remember", "recall", "forget"}
    other = [{"role": "user", "content": "Remember that my pin is 4921."}, {"role": "assistant", "content": "Noted."},
             {"role": "user", "content": "What do you remember?"}]
    assert srv.prepare(dict(model="locals/qwen3.5-4b-qwenputer", max_tokens=64, messages=other))["computer"]["notes"] == ""
    # the key the reply is kept by is the one the next request looks for
    assert p["turn"] == _turn_digests(oai.convert_messages(msgs))[-1]


def test_a_clients_call_the_computer_cannot_check_goes_as_written(tok):
    """A call of the client's is checked only when the presenter holds it (one with a name, after the reasoning) and
    the computer can read it with its schema: a call longer than that, one inside the reasoning, and one with no
    function name go as written (the model is not told a right call is wrong); a check whose run fails lets the call
    go too."""
    from locals.server import _check_schemas

    tools = [{"type": "function", "function": {"name": "write", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path"]}}}]
    head, checks = _check_schemas(tools)

    def feed(text, thinking=False):
        ctl = _stub_controller(tok, tools=head, checks=checks)
        if thinking:
            ctl.pres.state = ctl.pres.before = "think"
            ctl.pres.opened = True
        for t in tok.encode(text, add_special_tokens=False):
            ctl.accept(t)
        return ctl

    long = "<tool_call>\n<function=write>\n<parameter=path>\na.py\n</parameter>\n<parameter=content>\n" + \
        "\n".join(f"x{i} = {i} * {i}" for i in range(400)) + "\n</parameter>\n</function>\n</tool_call>"
    ctl = feed(long)
    assert not ctl.parked and ctl.checking is None and tok.decode(ctl.drain() + ctl.finish()) == long
    short = "<tool_call>\n<function=write>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>"
    ctl = feed("I will write it.\n" + short + "\nNow on.", thinking=True)
    assert not ctl.parked and ctl.checking is None
    assert tok.decode(ctl.drain() + ctl.finish()) == "I will write it.\n" + short + "\nNow on.</think>"  # (all reasoning)
    nameless = '<tool_call>\n{"name": "write", "arguments": {"path": "a.py"}}\n</tool_call>'
    ctl = feed(nameless + "\nThen more.")
    assert not ctl.parked and tok.decode(ctl.drain() + ctl.finish()) == nameless + "\nThen more."
    ctl = feed(short)
    assert ctl.parked and ctl.checking == "write"
    assert ctl.ran([V0 + 7, -3]) == [] and ctl.checking is None and ctl.comp is None and not ctl.unread
    assert tok.decode(ctl.drain() + ctl.finish()) == short
    ctl = feed(short)  # (an error of the computer's own, not the call's)
    assert ctl.ran(_response(tok, "error (out of memory: the computer has about 15 KB for a call)")) == []
    assert tok.decode(ctl.drain() + ctl.finish()) == short and ctl.check_tries == 0


def test_runs_that_use_up_their_work_count_as_cut_runs(tok):
    """A program over run_python's steps (or a pattern over regex's) fails only after a run of about a minute, and the
    model writes another: two such runs in a reply, and a toolbox call gets an error without a run, as after two runs
    cut at MAX_RUN; calc still runs. Other errors do not count."""
    ctl = _stub_controller(tok, internal=("calc", "run_python", "words"))

    def call(fn):
        for t in tok.encode(f"<tool_call>\n<function={fn}>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
                            add_special_tokens=False):
            ctl.accept(t)
        assert ctl.parked
        ctl.parked = False

    for error in ("error (line 1: NameError: name 'x' is not defined)",
                  "19 10\n28 10\nerror (line 3: TimeoutError: the program takes more than the computer's 1,250 steps)",
                  "error (the pattern needs too many steps on this text for the computer: make it simpler)"):
        call("run_python")
        assert error in tok.decode(ctl.ran(_response(tok, error))) and not ctl.refused
    assert ctl.cut_runs == 2 and len(ctl.results) == 1  # (what a program printed before its error, as ever)
    call("words")
    assert ctl.read_call() == -1 and "twice in this reply" in tok.decode(ctl.ran([]))
    call("calc")
    ctl.comp = SimpleNamespace(read=lambda unread: np.array([V0 + 7]))
    assert ctl.read_call() == V0 + 7


def test_the_servers_schemas_for_checks_take_odd_schemas():
    """A schema with an odd "required" (true, a number, objects in it) gives checks without it, not an error."""
    from locals.server import _check_schemas

    for req in (True, 1, [{"a": 1}, "city"], None):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}}, "required": req}}}]
        head, checks = _check_schemas(tools)
        assert head == "tools\tf" and checks["f"] == ("city\t\tr" if req == [{"a": 1}, "city"] else "city\t\to")


def test_a_number_right_before_a_call_is_not_moved_after_it(tok):
    """A number of the answer's that a call follows at once ends there: shown before the call."""
    ctl = _stub_controller(tok, copy=True)
    ctl.on_copy = lambda *a: None
    ctl.results, ctl.pres.state = ["42"], "content"
    call = "<tool_call>\n<function=save>\n<parameter=v>\n1\n</parameter>\n</function>\n</tool_call>"
    for t in tok.encode("Total 123456789" + call, add_special_tokens=False):
        ctl.accept(t)
    assert tok.decode(ctl.drain() + ctl.finish()) == "Total 123456789" + call


def test_a_long_notes_replay_is_cut_to_what_fits_the_call(tok):
    """The notes given to a request's computer with its first notes call: the newest of them that fit the call
    (the computer reads a call of about 1,000 tokens at most), not a call too long to read."""
    from locals.models.qwenputer_walk import _parts

    ctl = _stub_controller(tok)
    ctl.internal = frozenset({"calc", "recall"})
    ctl.notes = "\n".join(f"remember\tkey number {i}\tvalue {i} " + "word " * 12 for i in range(120))
    read = []
    ctl.comp = SimpleNamespace(read=lambda toks: read.append(toks) or np.array([-1] * (len(toks) - 1) + [5]))
    for t in tok.encode("<tool_call>\n<function=recall>\n<parameter=key>\nkey number 119\n</parameter>\n</function>\n</tool_call>",
                        add_special_tokens=False):
        ctl.accept(t)
    ctl.read_call()
    text = tok.decode(read[0]) + "".join(tok.decode(x) for x in ctl.parts)  # (a call in parts)
    assert "<function=too_long>" not in text and "key number 119\t" in text and "key number 0\t" not in text
    assert len(ctl.parts) >= 1 and len(_parts(tok, tok.encode("x", add_special_tokens=False))) == 1


def test_a_run_cut_at_its_limit_is_an_error_for_the_model_and_the_computer_is_made_again(comp, tok, tmp_path):
    """A run that goes past MAX_RUN (here a limit of 2,000 tokens) is not the reply's failure: the model reads an error
    as the tool's response, and the reply's next call gets a computer made again from the reply so far. After two
    runs cut in a reply, a toolbox call gets an error without a run."""
    from locals.models.qwenputer_cpu import CPU, shared
    from locals.models.qwenputer_walk import CalcController

    pack = _pack()
    if Q.chipformer() is None or not Q.ensure_tools(pack):
        pytest.skip("the toolbox is made with the authors' package (chipformer)")
    from locals.models.qwenputer_walk import Device

    dev = Device(Q.Chip(pack, tools=True))
    cpu = CPU.make(dev, tmp_path)
    if cpu is None:
        pytest.skip("no C compiler (xcode-select --install)")
    prompt = tok.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    ctl = CalcController(dev, prompt, tok, [], split=True, internal=("calc", "run_python", "words"))

    def call(fn, args, limit=250_000):
        body = f"<tool_call>\n<function={fn}>\n" + "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items()) + "</function>\n</tool_call>"
        for t in tok.encode(body, add_special_tokens=False):
            ctl.accept(t)
        first = ctl.read_call()
        mx.eval(ctl.comp.s, ctl.comp.c, ctl.comp.mk)
        rest, _ = cpu.run(*shared([(ctl.comp.s, np.float32), (ctl.comp.c, np.float32), (ctl.comp.mk, np.uint32)]), first,
                          max_tokens=limit)
        return tok.decode(ctl.ran([first] + rest))

    echo = call("run_python", {"code": "t = 0\nfor i in range(200):\n    t += i\nprint(t)"}, limit=2000)
    assert "error (the computer's run went past its limit" in echo and ctl.comp is None
    for t in tok.encode("\nLet me count instead.", add_special_tokens=False):
        ctl.accept(t)
    assert call("words", {"text": "one two three"}).split("\n")[2] == "3"
    assert call("run_python", {"code": "print(sum(range(10)))"}).split("\n")[2] == "45"
    # cut again: the reply's next toolbox call gets an error without a run, and calc still runs
    assert "went past its limit" in call("run_python", {"code": "t = 0\nfor i in range(200):\n    t += i\nprint(t)"}, limit=2000)
    body = "<tool_call>\n<function=words>\n<parameter=text>\na b\n</parameter>\n</function>\n</tool_call>"
    for t in tok.encode(body, add_special_tokens=False):
        ctl.accept(t)
    runs = ctl.runs
    assert ctl.read_call() == -1 and "twice in this reply" in tok.decode(ctl.ran([])) and ctl.runs == runs
    assert call("calc", {"expression": "6*7"}).split("\n")[2] == "42"
