"""Sibling-parallel decoding of recursive structures.

In a recursive schema whose recursion runs through array properties (a
node's ``children``), the items of such an array are separated by cut
vertices of the constraint (the ``,`` and the closing ``]``), so the
language is a product over sibling blocks (Thm A17.1) and each subtree can
be decoded as its own branch. Decoding proceeds level by level:

  wave 0   the root node with its recursive arrays limited to *leaf* items
           (items whose own recursive arrays are forced empty), on the
           single-stream engine with a non-recursive regex;
  wave k   every node produced in wave k-1 gets a branch whose prompt is the
           task prompt plus the current document serialized up to that
           node's recursive array's ``[``: the same prefix the sequential
           decoder would have there. The branch continues mid-document with
           the node's items, constrained to leaf items followed by ``]``, so
           a leaf closes its array as it would sequentially. All branches of
           a wave decode together on the batch engine, sharing the prompt
           and the common document prefix (prefilled once, reused across
           waves through the prompt cache). Their items become the next
           wave's nodes.

Latency follows the depth of the tree times the longest sibling instead of
the number of nodes. The conditioning differs from sequential decoding only
in that a branch does not see the subtrees its earlier siblings decode in
the same wave (their arrays are still empty in its prefix). The stitched
document always conforms to the schema, since every branch matches its own
sub-grammar. Branch counts are capped (fail-closed: nodes beyond the cap
keep empty arrays). Telling the model about the outline in an extra turn
was tried and made it fill every array, so the prefix is the document alone.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from outlines_core.json_schema import build_regex_from_schema

from .engine import LocalsController
from .grammar import GrammarError, _resolve, json_safe_schema
from .hybrid_batch import HybridBatchEngine
from .packed import Request


@dataclass
class TreeShape:
    node: dict  # the recursive definition, an object schema
    ref: str  # its reference ("#" or "#/$defs/x")
    keys: List[str]  # its array properties whose items are the definition
    root_is_node: bool


def analyze(schema: dict) -> TreeShape:
    """The supported shape: the root is the recursive object definition (or
    a bare reference to it), whose recursion runs only through array
    properties with ``items: {"$ref": <the definition>}``."""
    ref = "#"
    node = schema
    meta = {"$ref", "$defs", "definitions", "$schema", "$id", "title", "description"}
    if "$ref" in schema and set(schema) <= meta:
        ref = schema["$ref"]
        node = _resolve(schema, ref)
    if node.get("type") != "object" or "properties" not in node:
        raise GrammarError("sibling-parallel decoding needs an object definition at the root")
    keys = []
    for k, p in node["properties"].items():
        if p.get("type") == "array" and isinstance(p.get("items"), dict) and p["items"].get("$ref") in (ref, "#"):
            keys.append(k)
        elif json.dumps(p).find('"$ref"') >= 0 and (ref in json.dumps(p) or '"#"' in json.dumps(p)):
            raise GrammarError(f"recursion through property {k!r} is not an array of the definition")
    if not keys:
        raise GrammarError("no recursive array property")
    return TreeShape(node, ref, keys, ref == "#")


def _strip_defs(node: dict, schema: dict) -> dict:
    """The definition with its $defs attached, so inlined references resolve."""
    out = dict(node)
    for k in ("$defs", "definitions"):
        if k in schema:
            out[k] = schema[k]
    return out


def leaf_schema(shape: TreeShape, schema: dict) -> dict:
    s = copy.deepcopy(_strip_defs(shape.node, schema))
    for k in shape.keys:
        p = dict(s["properties"][k])
        p.pop("items", None)
        p.pop("minItems", None)
        p["maxItems"] = 0
        s["properties"][k] = p
    return s


def skeleton_schema(shape: TreeShape, schema: dict) -> dict:
    s = copy.deepcopy(_strip_defs(shape.node, schema))
    leaf = leaf_schema(shape, schema)
    for k in shape.keys:
        p = dict(s["properties"][k])
        p["items"] = leaf
        s["properties"][k] = p
    return s


def items_regex(shape: TreeShape, schema: dict, key: str) -> str:
    """Contents of a recursive array after its ``[``: leaf items, then ``]``."""
    leaf = build_regex_from_schema(json.dumps(json_safe_schema(leaf_schema(shape, schema))), "")
    p = shape.node["properties"][key]
    lo, hi = int(p.get("minItems", 0)), p.get("maxItems")
    more = f"(,{leaf})*" if hi is None else (f"(,{leaf}){{0,{int(hi) - 1}}}" if int(hi) > 1 else "")
    if lo == 0:
        body = f"({leaf}{more})?" if hi is None or int(hi) >= 1 else ""
    else:
        body = leaf + f"(,{leaf}){{{lo - 1},{'' if hi is None else int(hi) - 1}}}"
    return body + "\\]"


def _dumps(x) -> str:
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False)


_SENTINEL = "\u0000LOCALS_STOP"


def _prefix_text(root: Any, target: list) -> str:
    """The compact document up to and including the ``[`` of ``target``."""

    def swap(x):
        if x is target:
            return _SENTINEL
        if isinstance(x, dict):
            return {k: swap(v) for k, v in x.items()}
        if isinstance(x, list):
            return [swap(v) for v in x]
        return x

    text = _dumps(swap(root))
    i = text.find(json.dumps(_SENTINEL, ensure_ascii=False))
    assert i >= 0
    return text[:i] + "["


@dataclass
class TreeResult:
    text: str
    obj: Any
    waves: int
    branches: int
    tokens: int
    prefill_s: float
    wall_s: float
    per_wave: List[Tuple[int, float]] = field(default_factory=list)  # (branches, seconds)


def parallel_tree(rt, schema: dict, task_messages: List[dict], max_tokens: int = 200, max_waves: int = 12,
                  n_slots: int = 16, max_branches: int = 96) -> TreeResult:
    """``task_messages``: the chat messages of the task (system/user)."""
    t0 = time.perf_counter()
    shape = analyze(schema)
    tok = rt.tokenizer
    hf = tok._tokenizer
    prompt_ids = tok.apply_chat_template(task_messages, add_generation_prompt=True, enable_thinking=False)
    skel = rt.compile(skeleton_schema(shape, schema))
    res = rt.engine.generate(list(prompt_ids), skel.controller(), max_tokens=max_tokens)
    root = json.loads(res.text)
    tokens = len(res.tokens)
    prefill = res.prefill_s
    regexes = {k: rt.compile(regex=items_regex(shape, schema, k)) for k in shape.keys}
    eng = getattr(rt, "_sibling_engine", None)
    if eng is None or eng.n_slots < n_slots:
        eng = rt._sibling_engine = HybridBatchEngine(rt.model, rt.tokenizer, n_slots=n_slots, min_share=16)
    pending = [root]
    waves, branches, per_wave = 0, 0, []
    while pending and waves < max_waves and branches < max_branches:
        jobs = [(node, k) for node in pending for k in shape.keys][: max_branches - branches]
        reqs = []
        for node, k in jobs:
            ids = list(prompt_ids) + hf.encode(_prefix_text(root, node[k]), add_special_tokens=False)
            c = regexes[k]
            reqs.append(Request(ids, LocalsController(c.constraint, c.planner, True, True), max_tokens))
        tw = time.perf_counter()
        eng.snapshots.clear()
        queue = list(reqs)

        def source(block, queue=queue):
            got, queue[:] = list(queue), []
            return got

        st = eng.serve(source, lru=rt.prompt_cache)
        per_wave.append((len(reqs), time.perf_counter() - tw))
        prefill += st.prefill_s
        tokens += st.tokens
        branches += len(reqs)
        nxt = []
        for (node, k), r in zip(jobs, reqs):
            try:
                items = json.loads("[" + r.text)
            except Exception:
                items = []  # a branch cut by max_tokens: keep the node's array empty
            node[k] = items
            nxt += [x for x in items if isinstance(x, dict)]
        pending = nxt
        waves += 1
    text = json.dumps(root, separators=(",", ":"), ensure_ascii=False)
    return TreeResult(text, root, waves, branches, tokens, prefill, time.perf_counter() - t0, per_wave)
