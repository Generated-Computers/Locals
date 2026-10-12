"""Constraint tables and forced-span planning against the real Qwen tokenizer.

Needs only the tokenizer files (downloaded with the model); no weights.
"""

import json
import random

import pytest

pytest.importorskip("mlx_lm")

from locals.constraint import compile_constraint
from locals.forced import ForcedPlanner
from locals.regex import DEAD
from locals.vocab import ByteTrie, TokenVocab

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


@pytest.fixture(scope="module")
def tok():
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer

    try:
        path = snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip("tokenizer not cached")
    return load_tokenizer(path)._tokenizer


@pytest.fixture(scope="module")
def setup(tok):
    vocab = TokenVocab.from_hf(tok, 151936, [151645, 151643])
    return vocab, ByteTrie.build(vocab)


SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "status": {"enum": ["pending", "approved", "rejected"]},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age", "status", "tags"],
}


def compiled(setup):
    from outlines_core.json_schema import build_regex_from_schema

    vocab, trie = setup
    return compile_constraint(build_regex_from_schema(json.dumps(SCHEMA), ""), vocab, trie)


def test_vocab_bytes_roundtrip(tok, setup):
    vocab, _ = setup
    rng = random.Random(0)
    for t in rng.sample(range(151643), 2000):
        b = vocab.token_bytes[t]
        assert b is not None
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            continue
        assert tok.decode([t]) == s


def test_token_table_matches_dfa(setup):
    vocab, _ = setup
    c = compiled(setup)
    rng = random.Random(1)
    for _ in range(20000):
        q, t = rng.randrange(c.n_states), rng.randrange(151643)
        assert int(c.next_state[t, q]) == c.dfa.run(vocab.token_bytes[t], q)


def test_plans_spell_forced_bytes_and_are_canonical(tok, setup):
    vocab, _ = setup
    c = compiled(setup)
    pl = ForcedPlanner(tok, vocab, c)
    checked = 0
    for q in range(c.n_states):
        if not c.forced_bytes[q]:
            continue
        p = pl.plan(None, q)
        if p is None:
            continue
        spelled = b"".join(vocab.token_bytes[i] for i in p.tokens)
        assert c.forced_bytes[q].startswith(spelled)
        # canonical: the tokenizer produces exactly these tokens for these bytes
        assert tok.encode(spelled.decode(), add_special_tokens=False) == p.tokens
        # the plan is valid under the constraint
        s = q
        for i in p.tokens:
            s = int(c.next_state[i, s])
            assert s != DEAD
        assert s == p.q_end
        checked += 1
    assert checked > 5


def test_stable_tail_commits_key_quote_before_letters(tok, setup):
    vocab, _ = setup
    c = compiled(setup)
    pl = ForcedPlanner(tok, vocab, c)
    # Start state is forced through '{"name":"' and string content follows,
    # which may begin with punctuation, so the last block '":"' must be held back.
    p = pl.plan(None, c.dfa.start)
    assert p is not None
    spelled = b"".join(vocab.token_bytes[i] for i in p.tokens)
    assert spelled == b'{"name'
    assert p.cover == 3  # next token must cover '":"'


def test_lossless_planner_holds_back_last_block(tok, setup):
    vocab, _ = setup
    c = compiled(setup)
    pl = ForcedPlanner(tok, vocab, c, commit_stable_tail=False, boundary_rule=False)
    for q in range(c.n_states):
        p = pl.plan(None, q)
        if p is not None:
            assert p.cover == 0
            assert p.committed_forced_bytes < len(c.forced_bytes[q])
