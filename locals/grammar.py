"""JSON Schema -> recursive transition network (a grammar of regexes).

Definitions on a reference cycle become nonterminals; every other ``$ref`` is
inlined. A nonterminal is invoked from a regex by the call escape
``\\U00200000 + i`` (regex.CALL_BASE), which locals/pda.py turns into a push of
the return state. Leaf types (string, number, integer, boolean, null, enum,
const, formats, length bounds) reuse outlines-core's regexes; objects, arrays
and anyOf/oneOf are assembled here. Whitespace is compact (none), as in the
rest of the engine.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .regex import CALL_BASE


class GrammarError(ValueError):
    pass


@dataclass
class Grammar:
    rules: List[str]  # regex per nonterminal, calls as \U escapes
    names: List[str]  # definition path per nonterminal ("#" is the root)
    root: int = 0
    n_inlined: int = 0
    stats: dict = field(default_factory=dict)


def _call(i: int) -> str:
    return "\\U%08x" % (CALL_BASE + i)


# --- a schema made safe for the regex compilers -------------------------------------------------
#
# outlines-core builds a string's regex as  "  + pattern + "  with the pattern as it is, so
#   * a top-level | in a pattern splits the whole value ("x|y" accepted  "x  and  y"  );
#   * a pattern that can match a quote, a backslash or a control character (".", "[^a]", "\\s")
#     produces text that is not JSON ("a"b", a raw newline);
# and a property whose schema is {} gets an ungrouped alternation that splits the object's regex.
# json_safe_schema rewrites the schema so that every output is JSON and means what the schema says:
# a pattern is grouped and matched against the string's characters as JSON writes them (a quote
# as \", a newline as \n), so the decoded string is what the pattern matches.

_ANY = {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}, {"type": "null"}, {"type": "array"},
                  {"type": "object"}]}
_UNSAFE = [(0x00, 0x1F), (0x22, 0x22), (0x5C, 0x5C)]  # what a JSON string writes as an escape


def _json_pattern(pattern: str) -> str:
    """``pattern`` (anchored, as outlines-core and the engine read it) as a regex over the text of a
    JSON string's contents."""
    from . import regex as rx

    def safe(node):
        if isinstance(node, rx.Chars):
            plain = rx.complement(rx.normalize(rx.complement(node.ranges) + _UNSAFE))
            alts = [rx.Chars(plain)] if plain else []
            for lo, hi in _UNSAFE:
                for cp in range(lo, hi + 1):
                    if any(a <= cp <= b for a, b in node.ranges):
                        esc = json.dumps(chr(cp))[1:-1]  # \" \\ \n \u0001 ...
                        alts.append(rx.Cat([rx.Chars([(ord(c), ord(c))]) for c in esc]))
            if not alts:
                raise GrammarError(f"the pattern {pattern!r} has a class that matches nothing")
            return alts[0] if len(alts) == 1 else rx.Alt(alts)
        if isinstance(node, rx.Cat):
            return rx.Cat([safe(x) for x in node.items])
        if isinstance(node, rx.Alt):
            return rx.Alt([safe(x) for x in node.items])
        if isinstance(node, rx.Rep):
            return rx.Rep(safe(node.item), node.lo, node.hi)
        raise GrammarError(f"unsupported pattern {pattern!r}")

    def lit(cp: int) -> str:
        c = chr(cp)
        return c if c.isascii() and c.isalnum() else "\\U%08X" % cp

    def show(node) -> str:
        if isinstance(node, rx.Chars):
            r = node.ranges
            if len(r) == 1 and r[0][0] == r[0][1]:
                return lit(r[0][0])
            return "[" + "".join(lit(a) if a == b else f"{lit(a)}-{lit(b)}" for a, b in r) + "]"
        if isinstance(node, rx.Cat):
            return "".join(f"(?:{show(x)})" if isinstance(x, rx.Alt) else show(x) for x in node.items)
        if isinstance(node, rx.Alt):
            return "(?:" + "|".join(show(x) for x in node.items) + ")"
        if isinstance(node, rx.Rep):
            inner = show(node.item) if isinstance(node.item, rx.Chars) else f"(?:{show(node.item)})"
            if node.hi == node.lo:
                return inner + "{%d}" % node.lo
            return inner + "{%d,%s}" % (node.lo, "" if node.hi is None else node.hi)
        raise GrammarError(f"unsupported pattern {pattern!r}")

    try:
        tree = rx.parse(pattern)
    except rx.RegexError as e:
        raise GrammarError(f"pattern {pattern!r}: {e}") from None
    return "(?:" + show(safe(tree)) + ")"


_LOWER = ("minimum", "exclusiveMinimum", "minLength", "minItems", "minProperties", "minContains")
_UPPER = ("maximum", "exclusiveMaximum", "maxLength", "maxItems", "maxProperties", "maxContains")
_SUBSCHEMA = ("items", "additionalProperties", "contains", "propertyNames", "not")
_NOTES = ("title", "description", "default", "examples", "$comment", "$id", "$schema", "deprecated", "readOnly",
          "writeOnly")


def _merge_all_of(n: dict, root: dict, refs: frozenset = frozenset()) -> dict:
    """``n`` with its allOf merged into it: one schema that allows what all the parts allow (properties and
    required joined, bounds the tighter, a property in two parts as the allOf of both). outlines-core writes an
    allOf's parts one after the other ({"a":1}{"b":"x"}: no JSON), and ignores one next to "properties"."""
    out = {k: v for k, v in n.items() if k != "allOf"}
    for part in n["allOf"]:
        seen = refs
        while isinstance(part, dict) and isinstance(part.get("$ref"), str):  # (a part that names a definition)
            ref = part["$ref"]
            if ref in seen:
                raise GrammarError(f"allOf: {ref} is a part of itself")
            seen = seen | {ref}
            part = {**_resolve(root, ref), **{k: v for k, v in part.items() if k != "$ref"}}
        if part is True or part == {}:
            continue
        if not isinstance(part, dict):
            raise GrammarError("allOf: a part that allows nothing (false) or is not a schema")
        if isinstance(part.get("allOf"), list):
            part = _merge_all_of(part, root, seen)
        out = _merge2(out, part)
    return out


def _merge2(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if k not in out or out[k] == v or k in _NOTES:
            out.setdefault(k, v)
        elif k == "properties" and isinstance(v, dict) and isinstance(out[k], dict):
            props = dict(out[k])
            for name, sub in v.items():
                props[name] = sub if name not in props or props[name] == sub else {"allOf": [props[name], sub]}
            out[k] = props
        elif k == "required" and isinstance(v, list) and isinstance(out[k], list):
            out[k] = list(dict.fromkeys(out[k] + v))
        elif k == "type":
            ta = set(out[k]) if isinstance(out[k], list) else {out[k]}
            tb = set(v) if isinstance(v, list) else {v}
            both = (ta & tb) | {"integer" for x, y in ((ta, tb), (tb, ta)) if "integer" in x and "number" in y}
            if not both:
                raise GrammarError(f"allOf: its parts allow no type in common ({out[k]} and {v})")
            out[k] = sorted(both)[0] if len(both) == 1 else sorted(both)
        elif k == "enum" and isinstance(v, list) and isinstance(out[k], list):
            out[k] = [x for x in out[k] if x in v]
            if not out[k]:
                raise GrammarError("allOf: its parts' enums have no value in common")
        elif k in _LOWER and isinstance(v, (int, float)) and isinstance(out[k], (int, float)):
            out[k] = max(out[k], v)
        elif k in _UPPER and isinstance(v, (int, float)) and isinstance(out[k], (int, float)):
            out[k] = min(out[k], v)
        elif k == "additionalProperties" and (v is False or out[k] is False):
            out[k] = False
        elif k in _SUBSCHEMA:
            out[k] = v if out[k] is True else out[k] if v is True else {"allOf": [out[k], v]}
        else:
            raise GrammarError(f"allOf: its parts give {k} two values ({json.dumps(out[k])} and {json.dumps(v)}), "
                               "which is not supported")
    return out


def json_safe_schema(schema):
    """The schema (a dict or its JSON text) rewritten for the regex compilers (see above); the same
    type back. Property names that JSON would escape are refused, as are bounds that contradict; an
    allOf is merged into one schema."""
    as_text = isinstance(schema, str)
    node = json.loads(schema) if as_text else schema

    def walk(n, depth=0):
        if isinstance(n, list):
            return [walk(x, depth) for x in n]
        if not isinstance(n, dict):
            return n
        if isinstance(n.get("allOf"), list) and (len(n["allOf"]) > 1 or set(n) - {"allOf", *_NOTES}):
            if depth > 32:
                raise GrammarError("allOf nested too deeply")
            return walk(_merge_all_of(n, node), depth + 1)
        out = {}
        for k, v in n.items():
            if k == "pattern" and isinstance(v, str):
                out[k] = _json_pattern(v)
            elif k == "properties" and isinstance(v, dict):
                props = {}
                for name, sub in v.items():
                    if any(c in name for c in '"\\') or any(ord(c) < 0x20 for c in name):
                        raise GrammarError(f"property name {name!r}: quotes, backslashes and control characters are not supported")
                    props[name] = _ANY if sub is True or sub == {} else walk(sub, depth)
                out[k] = props
            elif k in ("enum", "const", "default", "examples"):
                out[k] = v  # values, not schemas
            else:
                out[k] = walk(v, depth)
        for lo, hi in (("minItems", "maxItems"), ("minLength", "maxLength"), ("minProperties", "maxProperties")):
            if isinstance(n.get(lo), int) and isinstance(n.get(hi), int) and n[lo] > n[hi]:
                raise GrammarError(f"{lo} {n[lo]} is more than {hi} {n[hi]}")
        return out

    out = walk(node)
    return json.dumps(out) if as_text else out


# --- integer bounds -------------------------------------------------------------------------------
#
# outlines-core leaves out minimum and maximum: {"type": "integer", "minimum": 1, "maximum": 5} allowed 7, or
# -3. schema_regex writes such an integer as a placeholder string outlines-core copies as it is, and puts the
# range's own regex in its place.

def _same_length(a: str, b: str) -> str:
    """Digit strings of len(a) == len(b) from a to b (leading zeros allowed: a suffix's digits)."""
    if a == b:
        return a
    if len(a) == 1:
        return f"[{a}-{b}]"
    k = len(a) - 1
    if a[1:] == "0" * k and b[1:] == "9" * k:
        return (f"[{a[0]}-{b[0]}]" if a[0] != b[0] else a[0]) + f"[0-9]{{{k}}}"
    if a[0] == b[0]:
        return a[0] + _same_length(a[1:], b[1:])
    alts = [a[0] + _same_length(a[1:], "9" * k)]
    if int(b[0]) - int(a[0]) > 1:
        lo, hi = int(a[0]) + 1, int(b[0]) - 1
        alts.append((f"[{lo}-{hi}]" if lo != hi else str(lo)) + f"[0-9]{{{k}}}")
    alts.append(b[0] + _same_length("0" * k, b[1:]))
    return "(" + "|".join(alts) + ")"


def _naturals(lo: int, hi):
    """Alternatives for the decimal integers (no leading zeros) from lo >= 0 to hi (None: no end)."""
    alts = []
    d_lo = len(str(lo))
    d_hi = len(str(hi)) if hi is not None else d_lo
    for d in range(d_lo, d_hi + 1):
        a = max(lo, 10 ** (d - 1) if d > 1 else 0)
        b = min(hi, 10 ** d - 1) if hi is not None else 10 ** d - 1
        if a <= b:
            alts.append(_same_length(str(a), str(b)))
    if hi is None:
        alts.append(f"[1-9][0-9]{{{d_lo},}}")
    return alts


def integer_regex(lo: Optional[int], hi: Optional[int]) -> str:
    """The JSON integers from lo to hi (either None: no bound) as a regex."""
    alts = []
    if hi is None or hi >= 0:
        alts += _naturals(max(lo, 0) if lo is not None else 0, hi)
    if lo is None or lo < 0:  # -v for v from max(1, -hi) to -lo
        alts += ["-" + x for x in _naturals(max(1, -hi) if hi is not None else 1, -lo if lo is not None else None)]
    return "(" + "|".join(alts) + ")"


_BOUNDS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")


def _integer_bounds(n: dict):
    """(lo, hi) of an integer schema with bounds (None for a side without one), or None when it has none or they
    are not numbers. Draft 4's boolean exclusiveMinimum/exclusiveMaximum are read too."""
    if not any(k in n for k in _BOUNDS):
        return None
    lo = hi = None
    for k in _BOUNDS:
        v = n.get(k)
        if v is None or isinstance(v, bool) and k.startswith("exclusive"):
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise GrammarError(f"{k} must be a number, not {json.dumps(v)}")
        if k == "minimum":
            b = math.ceil(v) + (1 if n.get("exclusiveMinimum") is True and v == math.ceil(v) else 0)
            lo = b if lo is None else max(lo, b)
        elif k == "maximum":
            b = math.floor(v) - (1 if n.get("exclusiveMaximum") is True and v == math.floor(v) else 0)
            hi = b if hi is None else min(hi, b)
        elif k == "exclusiveMinimum":
            b = math.floor(v) + 1
            lo = b if lo is None else max(lo, b)
        else:
            b = math.ceil(v) - 1
            hi = b if hi is None else min(hi, b)
    # a bound past JavaScript's safe integers is no bound (zod writes -9007199254740991 and 9007199254740991 for
    # every int; their regex costs ~50 automaton states, 90 MB of token table, a field)
    lo = None if lo is not None and lo <= -(2**53 - 1) else lo
    hi = None if hi is not None and hi >= 2**53 - 1 else hi
    if lo is not None and hi is not None and lo > hi:
        raise GrammarError(f"no integer is between the bounds ({lo} > {hi})")
    return (lo, hi) if lo is not None or hi is not None else None


def schema_regex(safe_schema, whitespace: str = "") -> str:
    """The regex of a schema made safe by json_safe_schema (a dict or its JSON text): outlines-core's, with
    integers kept within their minimum and maximum."""
    from outlines_core.json_schema import build_regex_from_schema

    node = json.loads(safe_schema) if isinstance(safe_schema, str) else safe_schema
    tag = "LOCALSINT" + hashlib.sha1(json.dumps(node, sort_keys=True).encode()).hexdigest()[:10]
    ranges = []

    def walk(n):
        if isinstance(n, list):
            return [walk(x) for x in n]
        if not isinstance(n, dict):
            return n
        t = n.get("type")
        if isinstance(t, list) and "integer" in t and any(k in n for k in _BOUNDS):  # Optional[int] with bounds
            return walk({"anyOf": [_for_type(n, x) for x in t]})
        if t == "integer" and "enum" not in n and "const" not in n:
            b = _integer_bounds(n)
            if b is not None:
                ranges.append(integer_regex(*b))
                return {"const": f"{tag}Z{len(ranges) - 1}Z"}
        return {k: (v if k in ("enum", "const", "default", "examples") else walk(v)) for k, v in n.items()}

    out = walk(node)
    rx = build_regex_from_schema(json.dumps(out), whitespace)
    for i, r in enumerate(ranges):
        rx = rx.replace(f'"{tag}Z{i}Z"', r)
    return rx


_TYPE_KEYS = {
    "object": {"properties", "required", "additionalProperties", "minProperties", "maxProperties"},
    "array": {"items", "prefixItems", "minItems", "maxItems", "uniqueItems"},
    "string": {"pattern", "minLength", "maxLength", "format"},
    "number": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"},
    "integer": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"},
}
_ANY_TYPE_KEYS = {"enum", "const", "title", "description", "default", "examples"}


def _for_type(sub: dict, t: str) -> dict:
    """``sub`` for one of the types its "type" lists: that type's keywords only."""
    keep = _ANY_TYPE_KEYS | _TYPE_KEYS.get(t, set())
    out = {k: v for k, v in sub.items() if k in keep}
    out["type"] = t
    return out


def _literal(text: str) -> str:
    """``text`` as a regex that matches it literally."""
    return "".join(c if c.isascii() and c.isalnum() else "\\U%08X" % ord(c) for c in text)


def _leaf(sub: dict) -> str:
    return schema_regex(sub)


def _resolve(root: dict, ref: str) -> dict:
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        raise GrammarError(f"only local references are supported: {ref}")
    node = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise GrammarError(f"unresolved reference {ref}")
        node = node[part]
    return node


def _refs_in(node) -> List[str]:
    out = []
    if isinstance(node, dict):
        if isinstance(node.get("$ref"), str):
            out.append(node["$ref"])
        for k, v in node.items():
            if k not in ("$ref", "enum", "const", "examples", "default"):
                out += _refs_in(v)
    elif isinstance(node, list):
        for v in node:
            out += _refs_in(v)
    return out


def _cyclic_refs(root: dict) -> List[str]:
    """References that lie on a cycle of the reference graph (Tarjan)."""
    graph: Dict[str, List[str]] = {}
    todo = ["#"]
    while todo:
        r = todo.pop()
        if r in graph:
            continue
        graph[r] = _refs_in(_resolve(root, r))
        todo += graph[r]
    index, low, on, stack, out = {}, {}, set(), [], []
    counter = [0]

    def visit(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in graph[v]:
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1 or v in graph[v]:
                out.extend(comp)

    for v in list(graph):
        if v not in index:
            visit(v)
    return out


_LEAF_KEYS = {"type", "enum", "const", "pattern", "format", "minLength", "maxLength", "minimum", "maximum",
              "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minItems", "maxItems", "items",
              "properties", "required", "additionalProperties", "anyOf", "oneOf", "allOf", "$ref", "title",
              "description", "default", "examples", "$defs", "definitions", "$schema", "$id"}


class _Compiler:
    def __init__(self, root: dict):
        self.root = root
        self.cyclic = _cyclic_refs(root)
        self.names = ["#"] + [r for r in self.cyclic if r != "#"] if "#" in self.cyclic else ["#"] + list(self.cyclic)
        self.ids = {r: i for i, r in enumerate(self.names)}
        self.n_inlined = 0
        self.depth = 0

    def compile(self) -> Grammar:
        meta = {"$ref", "$defs", "definitions", "$schema", "$id", "title", "description"}
        ref = self.root.get("$ref")
        if ref in self.ids and ref != "#" and set(self.root) <= meta:
            # the root is a bare reference to a recursive definition: use it directly
            self.names = [r for r in self.names if r != "#"]
            self.ids = {r: i for i, r in enumerate(self.names)}
            root = self.ids[ref]
        else:
            root = 0
        rules = [self.regex(_resolve(self.root, r), top=True) for r in self.names]
        return Grammar(rules, list(self.names), root, self.n_inlined,
                       {"nonterminals": len(self.names), "cyclic_refs": list(self.cyclic)})

    def regex(self, sub, top: bool = False) -> str:
        if sub is True or sub == {}:
            raise GrammarError("unconstrained values ({} / true) are not supported")
        if "$ref" in sub:
            ref = sub["$ref"]
            if ref in self.ids and not top:
                return _call(self.ids[ref])
            if ref in self.ids and top:
                # the definition itself: its body
                return self.regex({k: v for k, v in _resolve(self.root, ref).items() if k != "$ref"}, top=False) \
                    if len(sub) > 1 else self.regex(_resolve(self.root, ref), top=False)
            self.n_inlined += 1
            self.depth += 1
            if self.depth > 64:
                raise GrammarError("reference nesting too deep")
            try:
                return self.regex(_resolve(self.root, ref))
            finally:
                self.depth -= 1
        if "anyOf" in sub or "oneOf" in sub:
            alts = sub.get("anyOf") or sub.get("oneOf")
            return "(" + "|".join(self.regex(a) for a in alts) + ")"
        if "allOf" in sub:
            if len(sub["allOf"]) != 1:
                raise GrammarError("allOf with more than one schema is not supported")
            return self.regex(sub["allOf"][0])
        t = sub.get("type")
        if t == "object" or ("properties" in sub and t is None):
            return self.obj(sub)
        if t == "array" or ("items" in sub and t is None):
            return self.arr(sub)
        if isinstance(t, list):  # each type with only its own keywords ("null" with an object's properties failed)
            return "(" + "|".join(self.regex(_for_type(sub, x)) for x in t) + ")"
        if _refs_in(sub):
            raise GrammarError(f"references inside a {t} schema are not supported")
        return _leaf(sub)

    def obj(self, sub) -> str:
        if sub.get("additionalProperties") not in (None, False):
            raise GrammarError("additionalProperties is not supported")
        props = sub.get("properties", {})
        req = list(sub.get("required", []))
        parts = [('"' + _literal(json.dumps(k)[1:-1]) + '":' + self.regex(v), k in req) for k, v in props.items()]
        if not parts:
            return "\\{\\}"
        if any(r for _, r in parts):
            first = next(i for i, (_, r) in enumerate(parts) if r)
            out = "".join(f"({p},)?" for p, _ in parts[:first])
            out += parts[first][0]
            for p, r in parts[first + 1:]:
                out += f",{p}" if r else f"(,{p})?"
            return "\\{" + out + "\\}"
        # all optional: any non-empty subset in schema order, or nothing
        tails = []
        for i in range(len(parts)):
            rest = "".join(f"(,{p})?" for p, _ in parts[i + 1:])
            tails.append(parts[i][0] + rest)
        return "\\{(" + "|".join(tails) + ")?\\}"

    def arr(self, sub) -> str:
        items = sub.get("items")
        if items is None:
            raise GrammarError("arrays need an items schema")
        item = self.regex(items)
        lo, hi = int(sub.get("minItems", 0)), sub.get("maxItems")
        if hi is None:
            more = f"(,{item})*"
        else:
            more = f"(,{item}){{0,{int(hi) - 1}}}" if int(hi) > 1 else ""
        if lo == 0:
            body = f"({item}{more})?" if hi is None or int(hi) >= 1 else ""
        else:
            body = item + f"(,{item}){{{lo - 1},{'' if hi is None else int(hi) - 1}}}"
        return "\\[" + body + "\\]"


def is_recursive(schema) -> bool:
    if isinstance(schema, str):
        schema = json.loads(schema)
    return bool(_cyclic_refs(schema))


def compile_grammar(schema) -> Grammar:
    if isinstance(schema, str):
        schema = json.loads(schema)
    return _Compiler(schema).compile()
