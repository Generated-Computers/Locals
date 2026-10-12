"""Recursive schemas as a pushdown automaton (locals/grammar.py, pda.py, pdactl.py).

Tokenizer-only:
  * grammar: cyclic references become nonterminals, others are inlined;
    ambiguous grammars are refused
  * token tables equal byte-level stepping on every reachable configuration
    of a small vocabulary (three grammars, stacks up to depth 4), including
    the fail-closed limits (at most MAX_POP real pops / MAX_PUSH pushes per
    token)
  * with the real Qwen vocabulary, tokenized random trees walk through the
    tables and the allowed set at sampled prefixes equals brute force
Model (Qwen3.5-2B, skipped if not cached):
  * pipelined and synchronous decoding agree, outputs parse and nest deeper
    than the regex path's unrolling, forced spans cross returns
"""

import os
import itertools
import json
import random

import numpy as np
import pytest

pytest.importorskip("mlx_lm")

MODEL = os.environ.get("LOCALS_TEST_MODEL", "mlx-community/Qwen3.5-2B-MLX-4bit")

TREE = {"type": "object", "properties": {"name": {"type": "string"}, "children": {"type": "array", "items": {"$ref": "#"}}},
        "required": ["name", "children"]}
EXPR = {"$defs": {"expr": {"anyOf": [{"type": "number"}, {"type": "object", "properties": {
    "op": {"enum": ["+", "*", "-"]}, "args": {"type": "array", "items": {"$ref": "#/$defs/expr"}, "minItems": 2}},
    "required": ["op", "args"]}]}}, "$ref": "#/$defs/expr"}
WRAPPED = {"type": "object", "properties": {"root": {"$ref": "#/$defs/expr"}, "n": {"type": "integer"}},
           "required": ["root", "n"], "$defs": EXPR["$defs"]}
GRAMMARS = [("tree", TREE), ("expr", EXPR), ("wrapped", WRAPPED)]


def test_grammar_shapes():
    from locals.grammar import GrammarError, compile_grammar, is_recursive

    assert is_recursive(TREE) and is_recursive(EXPR) and is_recursive(WRAPPED)
    assert not is_recursive({"type": "object", "properties": {"a": {"$ref": "#/$defs/b"}}, "$defs": {"b": {"type": "string"}}})
    g = compile_grammar(TREE)
    assert g.names == ["#"] and "\\U00200000" in g.rules[0]
    g = compile_grammar(EXPR)
    assert g.names == ["#/$defs/expr"] and g.root == 0  # a bare root reference reuses the definition
    g = compile_grammar(WRAPPED)
    assert g.names == ["#", "#/$defs/expr"] and g.n_inlined == 0
    # two recursive alternatives that both start with "{" cannot be told apart by one byte
    amb = {"$defs": {"a": {"type": "object", "properties": {"x": {"type": "array", "items": {"$ref": "#/$defs/a"}}}, "required": ["x"]},
                     "b": {"type": "object", "properties": {"y": {"type": "array", "items": {"$ref": "#/$defs/b"}}}, "required": ["y"]}},
           "anyOf": [{"$ref": "#/$defs/a"}, {"$ref": "#/$defs/b"}]}
    from locals.pda import build_pda

    with pytest.raises(GrammarError):
        build_pda(compile_grammar(amb))


def _small_vocab():
    from locals.vocab import ByteTrie, TokenVocab

    alphabet = b'{}[]",:+*-.eE0123456789op agrs nmechild'
    toks = sorted(set([bytes([b]) for b in alphabet] + [
        b'{"op":"', b'","args":[', b'],', b']}', b'2,', b'2]', b'2.5', b'.5', b'1e+3', b'},{"op":"', b'"}', b'2]}', b'2]},',
        b'[{', b'{"', b'0.', b'00', b'2]]', b'2]]}', b'"name":"', b'","children":[', b'[]}', b'[]},{', b'"}]}', b'"},{"',
        b'[]}]}', b'[]}]}]}', b'3]}]}', b'3]},{"op":"', b']}]}', b'{"root":', b'},"n":', b'"n":']))
    vocab = TokenVocab(toks + [b""], eos_ids=[len(toks)], n_model=len(toks) + 1)
    return toks, vocab, ByteTrie.build(vocab)


@pytest.mark.parametrize("name,schema", GRAMMARS)
def test_tables_match_byte_stepping_on_reachable_configurations(name, schema):
    from locals.grammar import compile_grammar
    from locals.pda import MAX_POP, MAX_PUSH, build_pda, token_tables
    from locals.regex import DEAD

    toks, vocab, trie = _small_vocab()
    pda = build_pda(compile_grammar(schema))
    c = token_tables(pda, trie, vocab)
    root_nt = int(pda.nt_of[pda.root])

    def stacks(q, depth):
        def rec(nt, d):
            if d == 0:
                if nt == root_nt:
                    yield ()
                return
            for r in pda.returns_of[nt]:
                for rest in rec(pda.nt_of[r], d - 1):
                    yield rest + (r,)
        return list(rec(pda.nt_of[q], depth))

    checked = live = 0
    for q in range(pda.n):
        for depth in range(5):
            for st in stacks(q, depth):
                for t in range(len(toks)):
                    s1 = list(st)
                    got = c.advance(t, q, s1)
                    q2, s2, minh, ok = q, list(st), len(st), True
                    for b in toks[t]:
                        q2 = pda.step(q2, s2, b)
                        minh = min(minh, len(s2))
                        if q2 == DEAD:
                            ok = False
                            break
                    pops = (len(st) - minh) + (1 if q2 == pda.FIN else 0)
                    ref = q2 if ok and pops <= MAX_POP and len(s2) - minh <= MAX_PUSH else DEAD
                    assert got == ref, (name, toks[t], q, st, got, ref)
                    if got != DEAD:
                        assert s1 == s2, (name, toks[t], q, st, s1, s2)
                        live += 1
                    checked += 1
    assert live > 100 and checked > 5000


@pytest.fixture(scope="module")
def real_vocab():
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer

    try:
        path = snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip("tokenizer not cached")
    from locals.vocab import ByteTrie, TokenVocab

    tok = load_tokenizer(path)
    hf = tok._tokenizer
    vocab = TokenVocab.from_hf(hf, 248320, sorted(set(tok.eos_token_ids)))
    return hf, vocab, ByteTrie.build(vocab)


def test_real_vocabulary_traces_and_allowed_sets(real_vocab):
    from locals.pda import MAX_POP, MAX_PUSH, STACK_DEPTH, compile_pda_constraint
    from locals.regex import DEAD

    hf, vocab, trie = real_vocab
    c = compile_pda_constraint(TREE, vocab, trie)
    pda = c.pda
    rng = random.Random(1)

    def tree(d):
        kids = [tree(d - 1) for _ in range(rng.choice([0, 0, 1, 2, 3]))] if d > 0 else []
        return {"name": rng.choice(["a", "Maria Gonzalez", "x y", "root node", "q"]), "children": kids}

    toks = [t for t in range(vocab.n_model) if vocab.token_bytes[t]]
    tb = vocab.token_bytes

    def brute(q, stack):
        out = []
        for t in toks:
            q2, s, minh, ok = q, list(stack), len(stack), True
            for b in tb[t]:
                q2 = pda.step(q2, s, b)
                minh = min(minh, len(s))
                if q2 == DEAD:
                    ok = False
                    break
            pops = (len(stack) - minh) + (1 if q2 == pda.FIN else 0)
            if ok and pops <= MAX_POP and len(s) - minh <= MAX_PUSH and len(s) <= STACK_DEPTH:
                out.append(t)
        if pda.accepting[q] and not stack:
            out += list(c.eos_ids)
        return sorted(out)

    compared = 0
    for _ in range(4):
        doc = json.dumps(tree(4), separators=(",", ":"))
        ids = hf.encode(doc, add_special_tokens=False)
        q, stack = pda.root, []
        for k, t in enumerate(ids):
            if k in (0, 3, 9, len(ids) - 2) and compared < 12:
                assert c.allowed_ids(q, stack).tolist() == brute(q, stack), (doc[:60], q, stack)
                compared += 1
            q = c.advance(t, q, stack)
            assert q != DEAD, (k, tb[t], stack)
        assert q == pda.FIN
    assert compared >= 8


# ---------------------------------------------------------------------------- model tests


@pytest.fixture(scope="module")
def rt():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    from locals.runtime import Runtime

    return Runtime.load(MODEL)


def test_recursive_generation_is_valid_and_deterministic(rt):
    c = rt.compile(TREE)
    assert c.recursive
    p = rt.encode_chat("Produce a single chain of nested JSON nodes 5 levels deep: level1 contains level2, and so on to "
                       "level5, each node an object with a string 'name' and an array 'children' holding the next level "
                       "(empty at level5). Respond with the JSON only.")
    a = rt.engine.generate(p, c.controller(), max_tokens=300, pipeline=True)
    b = rt.engine.generate(p, c.controller(), max_tokens=300, pipeline=False)
    assert a.tokens == b.tokens
    assert a.finish == "stop" and a.forced_tokens > 0
    d = json.loads(a.text)
    depth = 1
    while d["children"]:
        d = d["children"][0]
        depth += 1
    assert depth >= 4  # the regex path unrolls to 3 levels


def test_expression_tree_with_lazy_returns(rt):
    c = rt.compile(EXPR)
    p = rt.encode_chat("Write the arithmetic expression (2 + 3) * (4 - 1) as a JSON expression tree. A node is either a "
                       "number or an object {\"op\": one of \"+\",\"*\",\"-\", \"args\": [subexpressions...]}. "
                       "Respond with the JSON only.")
    r = rt.engine.generate(p, c.controller(), max_tokens=200)
    assert r.finish == "stop", r.text
    d = json.loads(r.text)

    def ok(n):
        return isinstance(n, (int, float)) or (n["op"] in "+*-" and len(n["args"]) >= 2 and all(ok(x) for x in n["args"]))

    assert ok(d)


def test_budget_masking_under_tight_limits(rt):
    """Budget-aware masking on the pushdown path: from the first budget that
    yields a valid output upward, every budget yields a valid output; without
    the mask most tight budgets cut the output mid-value."""
    from workloads import valid

    c = rt.compile(TREE)
    p = rt.encode_chat("Build a tree of the Solar System as nested JSON: the root is 'Solar System'; its children are "
                       "Mercury, Venus and Earth; Earth has the child 'Moon'. Respond with the JSON only.")
    full = rt.engine.generate(p, c.controller(), max_tokens=300)
    assert full.finish == "stop"
    first_ok, ok_on, ok_off, n = None, 0, 0, 0
    for mt in range(6, len(full.tokens) + 1, 4):
        ctl = c.controller()
        r = rt.engine.generate(p, ctl, max_tokens=mt)
        try:
            good = valid(TREE, json.loads(r.text)) and len(r.tokens) <= mt
        except Exception:
            good = False
        if first_ok is None and good:
            first_ok = mt
        if first_ok is not None:
            assert good, (mt, r.text)
            n += 1
            ok_on += 1
        off = c.controller()
        off.budget = False
        r2 = rt.engine.generate(p, off, max_tokens=mt)
        try:
            ok_off += valid(TREE, json.loads(r2.text)) and len(r2.tokens) <= mt
        except Exception:
            pass
    assert first_ok is not None and n >= 3
    assert ok_off < ok_on
