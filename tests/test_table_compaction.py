"""The batch engine's combined mask table is rebuilt from the running rows' constraints when most
of it belongs to finished requests (each state is a row of the whole vocabulary: a server whose
agents never let it go idle grew it until Metal ran out of memory). The running rows' states move
with it. Skipped if the model is not cached."""

import os
import re

import pytest

pytest.importorskip("mlx_lm")

MODEL = os.environ.get("LOCALS_TEST_MODEL", "mlx-community/Qwen3.5-2B-MLX-4bit")


@pytest.fixture(scope="module")
def rt():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    from locals.runtime import Runtime

    return Runtime.load(MODEL)


def test_table_rebuilt_while_a_row_runs(rt):
    from locals.engine import LocalsController
    from locals.hybrid_batch import HybridBatchEngine
    from locals.packed import Request

    def req(rx, text, n):
        c = rt.compile(regex=rx)
        return Request(rt.encode_chat(text), LocalsController(c.constraint, c.planner, True, True), n)

    long_rx = r"[a-z ]*\{"  # few states, and a row that runs to its token limit
    a = req(long_rx, "Write a long story in lowercase words.", 150)
    shorts = [(rf"(yes|no) [0-9]{{{k + 1}}}", req(rf"(yes|no) [0-9]{{{k + 1}}}", "Is seven prime? Then a number.", 20))
              for k in range(8)]
    eng = HybridBatchEngine(rt.model, rt.tokenizer, n_slots=4)
    rebuilt = []
    orig = eng._table_add

    def spy(ctls):
        old = orig(ctls)
        rebuilt.append(old is not None and not a.finish)  # rebuilt while the long row ran
        return old

    eng._table_add = spy
    checked = []
    launch = eng._launch1

    def check(cache, x, g, temp, budget=None):
        # right after a rebuild, the long row's state on the device is its own state at its new offset
        if rebuilt and rebuilt[-1] and g is not None and budget is not None and a in budget[0]:
            i = budget[0].index(a)
            checked.append(g.tolist()[i] == eng._tab[3][id(a.controller.c)] + a.controller.q)
            rebuilt[-1] = False  # later launches run a step ahead of the host
        return launch(cache, x, g, temp, budget)

    eng._launch1 = check
    first = [True]

    def source(block):
        if first[0]:
            first[0] = False
            return [shorts[0][1], a]  # the long row second: the rebuild moves its offset
        for i, (_, r) in enumerate(shorts[1:], 1):  # the next short request once the one before it is done
            if not getattr(r, "_sent", False) and shorts[i - 1][1].finish:
                r._sent = True
                return [r]
        return []

    eng.serve(source)
    assert checked and all(checked), checked
    assert re.fullmatch(r"[a-z ]*\{?", a.text) and len(a.out) > 100, a.text  # its states moved with the table
    for rx, r in shorts:
        if r.finish:
            assert re.fullmatch(rx, r.text), (rx, r.text)
