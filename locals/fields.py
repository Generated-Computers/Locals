"""Block-parallel field decoding.

A JSON object's language factorizes over its fields: the forced key spans are
cut vertices of the constraint automaton, and the language is the product of
the value blocks (the block product of Thm A17.1). For extraction, each value
depends on the document far more than on the other values, so every field is
decoded as its own branch: prompt + ``{"key":`` + the value's grammar followed
by one terminator (``,`` or ``}``). All branches share the prompt's KV (a
prefix-tree of requests, prefilled once) and decode together in one batch, so
latency follows the longest field instead of the sum of all fields.

This changes the conditioning (a value does not see earlier values), so it is
an approximation of sequential decoding, unlike forced spans. The stitched
object always matches the schema: each value matches its own sub-grammar.
"""

from __future__ import annotations

import json
import time
from typing import List, Tuple

from .engine import LocalsController
from .grammar import json_safe_schema, schema_regex
from .hybrid_batch import HybridBatchEngine
from .packed import Request


def parallel_fields(rt, schema: dict, prompt_ids: List[int], max_tokens: int = 160) -> Tuple[str, float]:
    """Returns (json text, prefill seconds)."""
    props = schema["properties"]
    keys = list(props)
    reqs = []
    for k in keys:
        rx = schema_regex(json_safe_schema(props[k])) + "[,}]"
        c = rt.compile(regex=rx)
        prefix = rt.tokenizer.encode("{" + json.dumps(k) + ":", add_special_tokens=False)  # the key as JSON writes it
        reqs.append(Request(list(prompt_ids) + prefix, LocalsController(c.constraint, c.planner, True, True), max_tokens))
    eng = getattr(rt, "_field_engine", None)
    if eng is None or eng.n_slots < len(reqs):
        eng = rt._field_engine = HybridBatchEngine(rt.model, rt.tokenizer, n_slots=max(8, len(reqs)), min_share=16)
    eng.snapshots.clear()
    st = eng.run(reqs)
    if any(r.finish == "length" for r in reqs):  # a field cut at max_tokens: not the schema's output
        raise ValueError(f"a field ran over max_tokens ({max_tokens}); decode the object whole instead")
    vals = [r.text[:-1] if r.text[-1:] in ",}" else r.text for r in reqs]
    text = "{" + ",".join(f"{json.dumps(k)}:{v}" for k, v in zip(keys, vals)) + "}"
    return text, st.prefill_s
