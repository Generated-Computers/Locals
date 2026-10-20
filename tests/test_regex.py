import json
import random
import re

import pytest

from locals.regex import compile_dfa, utf8_sequences

PATTERNS = [
    r"abc",
    r"a|b|cd",
    r"(ab)*c+",
    r"[a-c]{2,4}x?",
    r"-?(0|[1-9][0-9]*)(\.[0-9]+)?",
    r'"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\/bfnrt])*"',
    r"(true|false|null)",
    r"[^a-z]{0,3}",
    r"\d+\s\w*",
    r"é|日本|[α-ω]+",
    r"(a|ab)(c|bcd)(d*)",
    r"x{3}y{1,}z{0,2}",
]

ALPHABET = list("abcdxyz0129.-\"\\/ nfrt") + ["é", "日", "本", "β", "\x01", "\x7f", "\u0090", "😀"]


def rand_str(rng):
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 8)))


@pytest.mark.parametrize("pat", PATTERNS)
def test_dfa_matches_python_re(pat):
    dfa = compile_dfa(pat)
    rx = re.compile(pat, re.ASCII if "\\w" in pat or "\\d" in pat or "\\s" in pat else 0)
    rng = random.Random(hash(pat) & 0xFFFF)
    samples = [rand_str(rng) for _ in range(3000)]
    # Include positive samples by mutation-free generation from simple cases.
    samples += ["abc", "cd", "ababccc", "aax", "-0", "12.5", '"a\\n"', "true", "日本", "αβγ", "xxxyzz"]
    for s in samples:
        assert dfa.matches(s.encode("utf-8")) == bool(rx.fullmatch(s)), (pat, s)


def test_utf8_sequences_exact():
    rng = random.Random(0)
    for _ in range(200):
        lo = rng.randint(0, 0x10FFFF)
        hi = min(0x10FFFF, lo + rng.choice([0, 1, 50, 5000, 200000]))
        seqs = utf8_sequences(lo, hi)
        for _ in range(50):
            cp = rng.randint(max(0, lo - 10), min(0x10FFFF, hi + 10))
            if 0xD800 <= cp <= 0xDFFF:
                continue
            b = chr(cp).encode()
            inside = any(len(q) == len(b) and all(x <= c <= y for (x, y), c in zip(q, b)) for q in seqs)
            assert inside == (lo <= cp <= hi)


def test_json_schema_regex_compiles():
    from outlines_core.json_schema import build_regex_from_schema

    schema = {"type": "object", "properties": {"name": {"type": "string"}, "n": {"type": "integer"}}, "required": ["name", "n"]}
    pat = build_regex_from_schema(json.dumps(schema))
    dfa = compile_dfa(pat)
    assert dfa.matches(b'{"name":"x","n":3}')
    assert dfa.matches(b'{ "name" : "x" , "n" : -12 }')
    assert not dfa.matches(b'{"name":"x"}')


def test_minimizing_stops_once_past_the_cap(monkeypatch):
    """A grammar whose token table would not fit is refused while its DFA is minimized (a string with maxLength
    6000 took 214 s to be refused, inside the batch loop); the cap is exact."""
    from locals.constraint import WorkBudgetExceeded, compile_constraint
    from locals.regex import TooManyStates
    from locals.vocab import ByteTrie, TokenVocab

    pat = '"[a-z]{0,100}"'
    n = compile_dfa(pat).n
    assert compile_dfa(pat, max_minimal=n).n == n
    with pytest.raises(TooManyStates):
        compile_dfa(pat, max_minimal=n - 1)
    # a large DFA whose minimal one is small (an enum of ids: a trie of 5000 leaves) still compiles
    assert compile_dfa("(" + "|".join(f"id-{i:04d}" for i in range(5000)) + ")", max_minimal=20).n < 20
    toks = [bytes([b]) for b in b'"abcdefghijklmnopqrstuvwxyz'] + [b""]
    vocab = TokenVocab(toks, eos_ids=[len(toks) - 1], n_model=len(toks))
    monkeypatch.setenv("LOCALS_MAX_TABLE_GB", str(50 * 7 * vocab.n_model / 1e9))  # room for 50 states
    assert compile_constraint('"[a-z]{0,40}"', vocab, ByteTrie.build(vocab)).dfa.n <= 50
    with pytest.raises(WorkBudgetExceeded, match="over 50 states"):
        compile_constraint(pat, vocab, ByteTrie.build(vocab))
