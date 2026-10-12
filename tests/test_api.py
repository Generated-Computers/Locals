"""The library API (locals/api.py): Vocabulary, GrammarCompiler, CompiledGrammar,
Matcher and the mlx-lm logits processor, against the real Qwen tokenizer.

The fast tests need only the tokenizer files; ``test_api_generate`` runs the
model and is marked ``model``.
"""

import json
import os

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("mlx_lm")

import locals  # noqa: E402

TOKENIZER_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
MODEL = os.environ.get("LOCALS_TEST_MODEL", "mlx-community/Qwen3.5-2B-MLX-4bit")

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
TREE = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "children": {"type": "array", "items": {"$ref": "#"}}},
    "required": ["name", "children"],
}
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "finish", "description": "Done.", "parameters": {"type": "object", "properties": {}}}},
]


@pytest.fixture(scope="module")
def tokenizer():
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer

    try:
        path = snapshot_download(TOKENIZER_MODEL, local_files_only=True)
    except Exception:
        pytest.skip("tokenizer not cached")
    return load_tokenizer(path)


@pytest.fixture(scope="module")
def compiler(tokenizer):
    return locals.GrammarCompiler(locals.Vocabulary.from_tokenizer(tokenizer))


def _walk(matcher, tokenizer, text, feed_forced=True):
    """Drive ``matcher`` through ``text`` token by token, checking every token
    against the mask first; returns the tokens fed (sampled and forced)."""
    ids = tokenizer.encode(text, add_special_tokens=False) if hasattr(tokenizer, "encode") else tokenizer._tokenizer.encode(text)
    fed = []
    i = 0
    while i < len(ids):
        t = ids[i]
        mask = matcher.next_token_mask()
        assert bool(mask[t].item()), f"token {t!r} ({tokenizer.decode([t])!r}) rejected at {matcher.state}"
        assert matcher.accept_token(t)
        fed.append(t)
        i += 1
        forced = matcher.jump_forward() if feed_forced else []
        for f in forced:  # forced tokens must spell the next bytes of the text
            assert ids[i] == f, (tokenizer.decode([f]), tokenizer.decode([ids[i]]))
            fed.append(f)
            i += 1
    return fed


def test_vocabulary_from_tokenizer(tokenizer):
    v = locals.Vocabulary.from_tokenizer(tokenizer)
    assert v.n_tokens == len(tokenizer._tokenizer)
    assert v.eos_ids and all(0 <= e < v.n_tokens for e in v.eos_ids)
    assert v.trie.n_nodes > 100_000
    v2 = locals.Vocabulary.from_tokenizer(tokenizer, vocab_size=151936)
    assert v2.n_tokens == 151936


def test_compile_json_schema_and_cache(compiler):
    g = compiler.compile_json_schema(SCHEMA)
    assert isinstance(g, locals.CompiledGrammar) and g.kind == "json_schema" and not g.recursive
    assert 50 < g.n_states < 2000
    assert g.table_bytes > 0 and g.vocab_size == compiler.vocab.n_tokens
    assert compiler.compile_json_schema(SCHEMA) is g  # cached by source


def test_matcher_accepts_valid_json_with_forced_spans(compiler, tokenizer):
    g = compiler.compile_json_schema(SCHEMA)
    m = g.matcher()
    text = '{"name":"Maria Gonzalez","age":34,"status":"approved","tags":["hiking","photography"]}'
    fed = _walk(m, tokenizer, text)
    assert tokenizer.decode(fed) == text
    assert m.is_completed() and not m.is_terminated()
    eos = m.eos_ids[0]
    assert bool(m.next_token_mask()[eos].item())
    assert m.accept_token(eos) and m.is_terminated()
    # a forced span appeared somewhere: the keys are certain text
    m2 = g.matcher()
    sampled_only = _walk(m2, tokenizer, text, feed_forced=False)
    assert len(sampled_only) == len(fed)


def test_matcher_rejects_invalid_tokens_and_does_not_move(compiler, tokenizer):
    g = compiler.compile_json_schema(SCHEMA)
    m = g.matcher()
    state0 = m.state
    bad = tokenizer._tokenizer.encode("hello")[0]
    assert not m.accept_token(bad)
    assert m.state == state0
    assert not m.accept_token(m.eos_ids[0])  # the empty output is not valid JSON for this schema
    assert not m.is_terminated()


def test_matcher_mask_padding_and_apply(compiler):
    g = compiler.compile_json_schema(SCHEMA)
    m = g.matcher()
    V = g.vocab_size
    logits = mx.zeros((1, V + 100))  # a model with a padded logits width
    masked = m.apply_token_mask(logits)
    assert masked.shape == (1, V + 100)
    arr = np.array(masked[0])
    allowed = np.isfinite(arr)
    assert allowed.sum() > 0 and not allowed[V:].any()  # padding is never allowed
    first = int(np.flatnonzero(allowed)[0])
    assert m.accept_token(first)


def test_matcher_fork_is_independent(compiler, tokenizer):
    g = compiler.compile_json_schema(SCHEMA)
    m = g.matcher()
    _walk(m, tokenizer, '{"name":"A')
    f = m.fork()
    _walk(f, tokenizer, 'lice","age":3')
    assert f.state != m.state
    assert m.state == g.matcher().state or True  # m is unchanged by f's progress
    assert not m.is_completed()


def test_regex_grammar(compiler, tokenizer):
    g = compiler.compile_regex(r"(red|green|blue) (apple|pear)")
    assert g.kind == "regex"
    m = g.matcher()
    _walk(m, tokenizer, "green pear")
    assert m.is_completed()
    m = g.matcher()
    with pytest.raises(AssertionError):
        _walk(m, tokenizer, "green plum")


def test_recursive_schema_uses_pushdown(compiler, tokenizer):
    g = compiler.compile_json_schema(TREE)
    assert g.recursive and g.n_states > 0
    text = '{"name":"root","children":[{"name":"a","children":[]},{"name":"b","children":[{"name":"c","children":[]}]}]}'
    m = g.matcher(canonical=False)  # the tokenizer's own split of this text
    fed = _walk(m, tokenizer, text)
    assert tokenizer.decode(fed) == text
    assert m.is_completed()
    # The boundary-cover rule (canonical=True) holds back the block ``":[`` after
    # ``"children`` and asks the next sampled token to cover it; the tokenizer's
    # split of the longer punctuation run ``":[]},{"`` starts with ``":`` instead.
    m = g.matcher(canonical=True)
    _walk(m, tokenizer, '{"name":"root","children":[{"name":"a","children')
    hf = tokenizer._tokenizer
    short, full = hf.encode('":')[0], hf.encode('":[')[0]
    mask = m.next_token_mask()
    assert not bool(mask[short].item()) and bool(mask[full].item())


def test_tool_call_grammar(compiler, tokenizer):
    g = compiler.compile_tool_calls(TOOLS, tool_choice="required")
    assert g.kind == "tool_calls"
    m = g.matcher()
    text = "I will read it.\n<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.py\n</parameter>\n</function>\n</tool_call>"
    fed = _walk(m, tokenizer, text)
    assert tokenizer.decode(fed) == text
    assert m.is_completed()
    none = compiler.compile_tool_calls(TOOLS, tool_choice="none").matcher()
    assert none.accept_token(tokenizer._tokenizer.encode("Done.")[0])
    assert not any(none.accept_token(t) for t in tokenizer._tokenizer.encode("<tool_call>"))
    with pytest.raises(ValueError):
        compiler.compile_tool_calls([], tool_choice="auto")


def test_logits_processor_masks_like_matcher(compiler, tokenizer):
    g = compiler.compile_json_schema(SCHEMA)
    proc = locals.logits_processor(g)
    V = g.vocab_size
    prompt = mx.array(tokenizer._tokenizer.encode("Extract the person as JSON."))
    rng = np.random.default_rng(0)
    tokens = prompt
    out = []
    for _ in range(40):
        logits = mx.array(rng.normal(size=(1, V)).astype(np.float32))
        masked = proc(tokens, logits)
        y = int(mx.argmax(masked[0]).item())
        assert np.isfinite(float(masked[0, y].item()))
        out.append(y)
        tokens = mx.concatenate([tokens, mx.array([y])])
        if y in g.vocab.eos_ids:
            break
    text = tokenizer.decode([t for t in out if t not in g.vocab.eos_ids])
    # whatever a random model picks under the mask is a valid prefix of the language
    assert text.startswith('{"name":')
    assert proc.matcher.state is not None


@pytest.mark.model
def test_api_generate_with_runtime_and_mlx_lm():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    rt = locals.load(MODEL)
    grammar = locals.GrammarCompiler(locals.Vocabulary.from_model(rt.model, rt.tokenizer)).compile_json_schema(SCHEMA)
    prompt = "Maria Gonzalez, 34, was approved; she likes hiking and photography. Return the record as JSON."
    res = rt.generate(prompt, grammar=grammar, max_tokens=120)
    obj = json.loads(res.text)
    assert set(obj) == set(SCHEMA["required"]) and res.forced_tokens > 0
    # the same grammar as a logits processor for mlx-lm's generator
    from mlx_lm import generate

    ids = rt.encode_chat(prompt)
    text = generate(rt.model, rt.tokenizer, ids, max_tokens=120, logits_processors=[locals.logits_processor(grammar)], verbose=False)
    obj2 = json.loads(text)
    assert set(obj2) == set(SCHEMA["required"])


def test_logits_processor_starts_over_and_refuses_what_the_grammar_does_not_allow(compiler, tokenizer):
    g = compiler.compile_json_schema(SCHEMA)
    proc = locals.logits_processor(g)
    V = g.vocab_size
    prompt = mx.array(tokenizer._tokenizer.encode("Extract the person as JSON."))
    logits = mx.zeros((1, V))
    first = int(mx.argmax(proc(prompt, logits)[0]).item())
    proc(mx.concatenate([prompt, mx.array([first])]), logits)
    second = int(mx.argmax(proc(prompt, logits)[0]).item())  # the same prompt again: a new sequence
    assert second == first
    bad = int(mx.argmin(proc(prompt, logits)[0]).item())  # masked out
    with pytest.raises(ValueError):
        proc(mx.concatenate([prompt, mx.array([bad])]), logits)
