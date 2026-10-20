"""Every output a JSON schema allows is JSON, and means what the schema says: patterns are matched
against the string as JSON writes it (a top-level | used to split the value; "." could write a raw
quote or newline), {} property schemas are any value (they used to split the object's regex), and
regex syntax the engine cannot honour is an error, not a different language."""

import json
import random

import numpy as np
import pytest

pytest.importorskip("outlines_core")

from outlines_core.json_schema import build_regex_from_schema  # noqa: E402

from locals import regex as rx  # noqa: E402
from locals.grammar import GrammarError, json_safe_schema  # noqa: E402


def _dfa(schema):
    return rx.compile_dfa(build_regex_from_schema(json.dumps(json_safe_schema(schema)), ""))


CASES = [
    ({"type": "object", "properties": {"a": {"type": "string", "pattern": "x|y"}}, "required": ["a"]},
     ['{"a":"x"}', '{"a":"y"}'], ['{"a":"x}', '{"a":y"}']),
    ({"type": "string", "pattern": ".+"}, ['"a"', '"a\\"b"', '"\\\\"', '"\\u0001"', '"é"'], ['"a"b"', '"\\"', '"\x01"', '""']),
    ({"type": "string", "pattern": "\\s+"}, ['" "', '"\\n"', '"\\t"'], ['"\n"']),
    ({"type": "string", "pattern": "\""}, ['"\\""'], ['"""']),
    ({"type": "object", "properties": {"a": {}}, "required": ["a"]}, ['{"a":1}', '{"a":"s"}', '{"a":null}'], ['{"a":true', "null"]),
    ({"type": "string", "pattern": "[a-z]+@[a-z]+\\.com"}, ['"ab@cd.com"'], ['"ab@cdxcom"']),
    # allOf is one object that has all the parts' properties (outlines-core wrote {"a":1}{"b":"x"}, and ignored an
    # allOf next to "properties"); a part may name a definition (an OpenAPI or zod "base and extension" schema)
    ({"allOf": [{"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
                {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]}]},
     ['{"a":1,"b":"x"}'], ['{"a":1}{"b":"x"}', '{"a":1}']),
    ({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"],
      "allOf": [{"properties": {"b": {"type": "string"}}, "required": ["b"]}]}, ['{"a":1,"b":"x"}'], ['{"a":1}']),
    ({"$defs": {"Base": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}},
      "allOf": [{"$ref": "#/$defs/Base"}, {"properties": {"name": {"type": "string", "maxLength": 3}}, "required": ["name"]}]},
     ['{"id":3,"name":"bob"}'], ['{"id":3}', '{"id":3,"name":"bobby"}', '{"name":"bob"}']),
    ({"type": "object", "properties": {"s": {"allOf": [{"type": "string", "minLength": 2}, {"maxLength": 3}]}},
      "required": ["s"]}, ['{"s":"ab"}', '{"s":"abc"}'], ['{"s":"a"}', '{"s":"abcd"}']),
]


@pytest.mark.parametrize("schema,good,bad", CASES)
def test_patterns_and_empty_schemas(schema, good, bad):
    d = _dfa(schema)
    assert all(d.matches(g.encode()) for g in good)
    assert not any(d.matches(b.encode()) for b in bad)


def test_every_output_is_json():
    rnd = random.Random(0)
    for schema, _, _ in CASES + [({"type": "string", "pattern": "[^a]{2,5}"}, [], [])]:
        d = _dfa(schema)
        live = set(np.nonzero(d.accepting)[0].tolist())
        while True:  # the states from which an accepting one is reachable
            more = {q for q in range(d.n) if q not in live and any(int(t) in live for t in d.table[q] if t != rx.DEAD)}
            if not more:
                break
            live |= more
        for _ in range(100):
            q, out = d.start, bytearray()
            for _ in range(300):
                if d.accepting[q] and rnd.random() < 0.3:
                    break
                nxt = [b for b in range(256) if int(d.table[q, b]) in live]
                if not nxt:
                    break
                b = rnd.choice(nxt)
                out.append(b)
                q = int(d.table[q, b])
            if d.accepting[q]:
                json.loads(out.decode("utf-8"))


def test_schemas_that_cannot_be_honoured():
    with pytest.raises(GrammarError):
        json_safe_schema({"type": "object", "properties": {'a"b': {"type": "string"}}})
    with pytest.raises(GrammarError):
        json_safe_schema({"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 1})


def test_allof_parts_that_cannot_be_joined_are_refused():
    for bad in ({"allOf": [{"type": "string"}, {"type": "integer"}]},
                {"allOf": [{"type": "string", "pattern": "a+"}, {"pattern": "b+"}]},
                {"allOf": [{"enum": [1, 2]}, {"enum": [3]}]},
                {"$defs": {"A": {"allOf": [{"$ref": "#/$defs/A"}]}}, "allOf": [{"$ref": "#/$defs/A"}, {"type": "object"}]}):
        with pytest.raises(GrammarError, match="allOf"):
            json_safe_schema(bad)


def test_regex_syntax_is_honoured_or_refused():
    assert rx.compile_dfa("x{,2}").matches(b"xx") and not rx.compile_dfa("x{,2}").matches(b"xxx")
    for bad in ("a{3,1}", "[z-a]", "x{100000000}", "\\U00110000"):
        with pytest.raises(rx.RegexError):
            rx.compile_dfa(bad)
    assert issubclass(rx.BudgetExceeded, ValueError)  # a request error (400), not the server's


def test_integers_stay_within_their_bounds():
    """outlines-core leaves minimum and maximum out: a rating from 1 to 5 could be 7."""
    from locals.grammar import integer_regex, schema_regex

    rnd = random.Random(1)
    ranges = [(1, 5), (0, 100), (-40, -1), (13, 2017), (0, None), (10, None), (None, 0), (None, -3), (7, 7), (-1000, 999)]
    ranges += [tuple(sorted(rnd.randint(-3000, 3000) for _ in range(2))) for _ in range(40)]
    for lo, hi in ranges:
        d = rx.compile_dfa(integer_regex(lo, hi))
        start = (lo if lo is not None else (hi or 0) - 2000) - 30
        for v in range(start, (hi if hi is not None else (lo or 0) + 2000) + 30):
            assert d.matches(str(v).encode()) == ((lo is None or v >= lo) and (hi is None or v <= hi)), (lo, hi, v)
        assert not any(d.matches(b) for b in (b"-0", b"01", b"+1", b"1.0", b""))

    def dfa(schema):
        return rx.compile_dfa(schema_regex(json_safe_schema(schema)))

    d = dfa({"type": "object", "properties": {"rating": {"type": "integer", "minimum": 1, "maximum": 5},
                                              "t": {"type": ["integer", "null"], "exclusiveMinimum": -10, "exclusiveMaximum": 0}},
             "required": ["rating", "t"]})
    assert d.matches(b'{"rating":5,"t":-9}') and d.matches(b'{"rating":1,"t":null}')
    assert not any(d.matches(b) for b in (b'{"rating":7,"t":-1}', b'{"rating":0,"t":-1}', b'{"rating":3,"t":0}',
                                          b'{"rating":3,"t":-10}'))
    assert dfa({"type": "integer", "minimum": 0, "exclusiveMinimum": True}).matches(b"1")  # (draft 4)
    assert not dfa({"type": "integer", "minimum": 0, "exclusiveMinimum": True}).matches(b"0")
    # zod writes the safe-integer bounds for every int: no bound, not a 16-digit range (~50 states a field)
    zod = dfa({"type": "integer", "minimum": -9007199254740991, "maximum": 9007199254740991})
    assert zod.n == dfa({"type": "integer"}).n
    for bad in ({"type": "integer", "minimum": 5, "maximum": 2}, {"type": "integer", "minimum": "5"}):
        with pytest.raises(GrammarError):
            dfa(bad)
