"""Greedy replies with MTP drafts against without, in process (HybridBatchEngine.serve), for the 14 prompts of
tests/test_mtp_model.py; where they differ, the logits of the two tokens at the first difference, teacher-forced
over the prompt and the MTP-off reply up to there. Run from the repo root:
.venv/bin/python docs/measurements/mtp/greedy_check.py docs/measurements/mtp/mtp_bf16.safetensors"""

import sys, json, mlx.core as mx
sys.path.insert(0, "tests")
from test_mtp_model import PROMPTS
from locals.runtime import Runtime
from locals.hybrid_batch import HybridBatchEngine
from locals.engine import Controller
from locals.packed import Request
from locals.mtp import MTP
from locals.kernels import head_logits
rt = Runtime.load("Kaziko/Qwen3.5-4B-Zwyrol")
tok = rt.tokenizer
lm = rt.model.language_model if hasattr(rt.model, "language_model") else rt.model
class Free(Controller):
    eos_ids = list(rt.vocab.eos_ids)
mtps = {n: MTP.load(rt.model, sys.argv[1], drafts=n) for n in (1, 2)}
def run(p, mtp):
    eng = HybridBatchEngine(rt.model, tok, n_slots=4, share_prefix=False, mtp=mtp)
    r = Request(list(p), Free(), 192); w=[r]
    eng.serve(lambda b: [w.pop()] if w else [])
    return r.out, eng.counters
tot = {1: [0, 0], 2: [0, 0]}
for name, text, tools in PROMPTS:
    p = tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], tools=tools, add_generation_prompt=True, enable_thinking=False, tokenize=False), add_special_tokens=False)
    off, _ = run(p, None)
    for n in (1, 2):
        on, c = run(p, mtps[n])
        tot[n][0] += c.get("draft_mtp_accepted", 0); tot[n][1] += c.get("draft_mtp_tokens", 0)
        i = next((i for i, (x, y) in enumerate(zip(off, on)) if x != y), None)
        if i is None and len(on) != len(off):
            i = min(len(on), len(off))
        if i is None:
            print(f"{name:14s} drafts={n} same ({len(off)} tokens)", flush=True); continue
        lg = head_logits(lm, lm.model(mx.array([p + off[:i]]))[:, -1:, :])[0, 0].astype(mx.float32)
        a, b = lg[off[i]].item(), lg[on[i]].item() if i < len(on) else float("nan")
        top = mx.max(lg).item()
        print(f"{name:14s} drafts={n} first diff at {i}: off {off[i]} ({a:.3f}) on {on[i] if i < len(on) else None} ({b:.3f}), max {top:.3f}, gap {abs(a-b):.4f}", flush=True)
for n in (1, 2):
    print(f"drafts={n}: accepted {tot[n][0]} of {tot[n][1]} = {tot[n][0]/max(1,tot[n][1]):.1%}")
