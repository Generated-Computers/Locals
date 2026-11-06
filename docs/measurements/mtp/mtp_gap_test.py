"""Gap test for issue #18: how much does the MTP layer's hit rate drop when its KV is missing positions?
Same setup as docs/measurements/mtp/mtp_test.py (Runtime.load, greedy, thinking off, at most 192 tokens), the best
wiring only (norms 1+w, hidden after the final norm, concat [embedding, hidden]), 4-bit layer (group 64).
Teacher-forced over prompt + reply. A gap is a set of key positions the layer may not attend to (each query still
sees itself), with the real positions kept:
  full       every earlier position (as mtp_test.py)
  no-prompt  no prompt position before P-1 (the prompt came from the prompt cache, the MTP KV did not)
  gap32      no reply position in [P-1+16, P-1+48) (32 steps the layer did not run, e.g. batched steps)
Usage: mtp_gap_test.py MTP.safetensors OUT.json MODEL"""
import json, sys, time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

import locals as L
from locals.runtime import Runtime
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen3_5 import DecoderLayer

REPO = Path(L.__file__).resolve().parent.parent
print("locals from", L.__file__, "| mlx", mx.__version__, flush=True)
MODEL = sys.argv[3]
print("model", MODEL, flush=True)
rt = Runtime.load(MODEL)
model, tok = rt.model, rt.tokenizer
lm = model.language_model; tm = lm.model; args = lm.args
head = lm.lm_head if hasattr(lm, "lm_head") else tm.embed_tokens.as_linear
print("output head", "own lm_head" if hasattr(lm, "lm_head") else "tied embedding", flush=True)
EOS = set(tok.eos_token_ids) | {tok.convert_tokens_to_ids("<|im_end|>")}


class MTP(nn.Module):
    def __init__(self):
        super().__init__()
        H, eps = args.hidden_size, args.rms_norm_eps
        self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=eps)
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.layers = [DecoderLayer(args, layer_idx=args.full_attention_interval - 1)]
        self.norm = nn.RMSNorm(H, eps=eps)


raw = mx.load(sys.argv[1])
m = MTP()
m.load_weights([(k[len("mtp."):], (v + 1.0 if v.ndim == 1 else v)) for k, v in raw.items()], strict=True)
m.set_dtype(mx.float16)
nn.quantize(m, group_size=64, bits=4)
mx.eval(m.parameters())


def mtp_logits(hid, emb, mask):
    x = m.fc(mx.concatenate([m.pre_fc_norm_embedding(emb), m.pre_fc_norm_hidden(hid)], axis=-1))
    return head(m.norm(m.layers[0](x, mask=mask, cache=None)))


def post_norm_hidden(ids):
    h = tm.embed_tokens(ids)
    fa, ssm = create_attention_mask(h, None), create_ssm_mask(h, None)
    for layer in tm.layers:
        h = layer(h, mask=ssm if layer.is_linear else fa, cache=None)
    return tm.norm(h)


def greedy(ids, n):
    cache = make_prompt_cache(model); out = []
    logits = model(mx.array(ids)[None], cache=cache)
    for _ in range(n):
        y = int(mx.argmax(logits[0, -1]).item()); out.append(y)
        if y in EOS: break
        logits = model(mx.array([[y]]), cache=cache)
    return out


weather = [{"type": "function", "function": {"name": "get_weather", "description": "Weather forecast for a city and a day",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "day": {"type": "string"}},
                           "required": ["city", "day"]}}}]
desk = [{"type": "function", "function": {"name": "read_screen", "description": "Read the text of the window in front, or of a named app",
          "parameters": {"type": "object", "properties": {"app": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "save_note", "description": "Save a short note for the owner",
          "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
        {"type": "function", "function": {"name": "agent_status", "description": "What a named coding agent is doing now",
          "parameters": {"type": "object", "properties": {"agent": {"type": "string"}}, "required": ["agent"]}}}]
code_src = (REPO / "locals" / "lookup.py").read_text()[:5000]
doc_src = (REPO / "docs" / "engine" / "README.md").read_text()[:6000]
PROMPTS = [  # the 12 of mtp_test.py, then 2 long prompts (an agent's prompt is long)
    ("en-explain", "Explain how a refrigerator works, in about 150 words.", None),
    ("code", "Write a Python function that reads a CSV file and returns the average of its 'price' column. Handle a missing file and bad numbers.", None),
    ("pl-email", "Napisz krótki e-mail do sąsiada z prośbą o podlewanie kwiatów podczas mojego wyjazdu w przyszłym tygodniu.", None),
    ("pl-explain", "Wyjaśnij prostymi słowami, czym jest inflacja i jak wpływa na oszczędności.", None),
    ("json", "Return a JSON object describing three fictional books, with the fields title, author, year and genres (a list). Output only the JSON.", None),
    ("summary", "Summarize this in three bullet points:\n\nThe city council met on Tuesday to discuss the new bike lanes. "
     "Residents of the old town said the lanes take away parking spaces that shops depend on. Cyclists said the lanes "
     "cut accidents by a third in the first year in the northern district. The council agreed to a six-month trial on "
     "two streets, with a parking study before and after, and will vote on a full plan in spring.", None),
    ("math", "A train leaves at 14:20 and travels 245 km at 70 km/h. When does it arrive? Show the steps.", None),
    ("code-convert", "Convert this to TypeScript with types:\n\nfunction total(items, tax) {\n  let sum = 0;\n  for (const it of items) {\n"
     "    sum += it.price * it.qty;\n  }\n  return Math.round(sum * (1 + tax) * 100) / 100;\n}", None),
    ("tool-call", "What will the weather be in Kraków tomorrow?", weather),
    ("chat", "I'm tired after a long day. Give me three quick ideas to relax tonight.", None),
    ("pl-tool-screen", "Co mam teraz otwarte na ekranie? Przeczytaj, co jest w oknie przeglądarki.", desk),
    ("pl-tool-note", "Zapisz notatkę: jutro o dziesiątej telefon do Ani w sprawie faktury, a potem powiedz mi, co robi agent od kodu.", desk),
    ("long-code", "Here is a Python file:\n\n```python\n" + code_src + "\n```\n\nExplain what the function `propose` does and how the main loop uses its draft.", None),
    ("long-doc", "Here is a design document:\n\n" + doc_src + "\n\nSummarize it in five bullet points.", None),
]

seqs = []
for name, text, tools in PROMPTS:
    p = tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], tools=tools, add_generation_prompt=True,
                                           enable_thinking=False, tokenize=False), add_special_tokens=False)
    t0 = time.perf_counter(); r = greedy(p, 192); s = time.perf_counter() - t0
    seqs.append((name, p, r))
    print(f"{name:14s} prompt {len(p):5d} reply {len(r):3d} tokens ({s:.1f} s): {tok.decode(r)[:70]!r}", flush=True)


def mask_for(n, hide):
    """(n, n) boolean, True = may attend: causal, minus the hidden key positions, but always itself."""
    q = mx.arange(n)[:, None]; k = mx.arange(n)[None, :]
    keep = k <= q
    if hide is not None:
        lo, hi = hide
        keep = keep & ~((k >= lo) & (k < hi))
    return keep | (k == q)


VARIANTS = ["causal", "full", "no-prompt", "gap32"]
tot = {v: {} for v in VARIANTS}  # variant -> position set -> [hits, tried]
per = {}
for name, p, r in seqs:
    toks = p + r; P, Lt = len(p), len(toks)
    ids = mx.array(toks)[None]
    post = post_norm_hidden(ids)
    hid, emb = post[:, :-1], tm.embed_tokens(ids[:, 1:])
    n = hid.shape[1]
    hides = {"causal": "causal", "full": None, "no-prompt": (0, P - 1), "gap32": (P - 1 + 16, P - 1 + 48)}
    sets = {"reply": range(P - 1, Lt - 2), "first16": range(P - 1, min(P - 1 + 16, Lt - 2)),
            "after-gap": range(P - 1 + 48, Lt - 2)}
    per[name] = {"prompt": P, "reply": len(r)}
    for v in VARIANTS:
        mk = hides[v] if v == "causal" else mask_for(n, hides[v])
        g = mx.argmax(mtp_logits(hid, emb, mk)[0], axis=-1); mx.eval(g)
        g = g.tolist()
        for sname, rng in sets.items():
            hits = sum(int(g[i] == toks[i + 2]) for i in rng); tried = len(rng)
            t = tot[v].setdefault(sname, [0, 0]); t[0] += hits; t[1] += tried
            per[name][f"{v}/{sname}"] = [hits, tried]

print(flush=True)
for v in VARIANTS:
    print(f"{v:10s} " + " | ".join(f"{s} {h}/{t} = {h / max(t, 1):.1%}" for s, (h, t) in tot[v].items()), flush=True)
print("\nper prompt, reply positions (full vs no-prompt), and after the gap (full vs gap32):", flush=True)
for name, d in per.items():
    f, npf, fa, ga = d["full/reply"], d["no-prompt/reply"], d["full/after-gap"], d["gap32/after-gap"]
    print(f"  {name:14s} prompt {d['prompt']:5d}  reply {f[0]:3d}/{f[1]:3d} vs {npf[0]:3d}/{npf[1]:3d}"
          f"   after-gap {fa[0]:3d}/{fa[1]:3d} vs {ga[0]:3d}/{ga[1]:3d}", flush=True)
json.dump({"model": MODEL, "totals": tot, "per_prompt": per}, open(sys.argv[2], "w"), indent=1)
