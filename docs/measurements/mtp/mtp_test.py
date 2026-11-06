"""Test B: how often does Qwen3.5-4B's own MTP layer guess the token after next, and what would it cost?
The main model is loaded with Locals' Runtime (float16, fused layers, small-batch kernels: as the server runs it
on this Mac). Ten prompts get a greedy reply (no thinking); then, teacher-forced over prompt + reply, the MTP layer
gets (hidden state at i, embedding of token i+1) and predicts token i+2. A guess counts when it equals the main
model's own greedy token there. Variants settle what the checkpoint does not say: the hidden state before or after
the final norm, the order of the concatenation, and the norm weights stored as w or as 1+w.
Usage: mtp_test.py MTP.safetensors OUT.json"""
import json, sys, time
import mlx.core as mx
import mlx.nn as nn

import locals as L
from locals.runtime import Runtime
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.qwen3_5 import DecoderLayer

print("locals from", L.__file__, "| mlx", mx.__version__, flush=True)
MODEL = sys.argv[3] if len(sys.argv) > 3 else "mlx-community/Qwen3.5-4B-MLX-4bit"
print("model", MODEL, flush=True)
rt = Runtime.load(MODEL)
model, tok = rt.model, rt.tokenizer
lm = model.language_model; tm = lm.model; args = lm.args
dt = tm.embed_tokens.scales.dtype if hasattr(tm.embed_tokens, "scales") else mx.float16
head = lm.lm_head if hasattr(lm, "lm_head") else tm.embed_tokens.as_linear  # Zwyrol: untied, one row edited
print("compute dtype", rt.dtype, "| embed dtype", dt, "| output head", "own lm_head" if hasattr(lm, "lm_head") else "tied embedding", flush=True)
EOS = set(tok.eos_token_ids) | {tok.convert_tokens_to_ids("<|im_end|>")}

class MTP(nn.Module):
    def __init__(self):
        super().__init__()
        H, eps = args.hidden_size, args.rms_norm_eps
        self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=eps)
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.layers = [DecoderLayer(args, layer_idx=args.full_attention_interval - 1)]  # a full-attention layer
        self.norm = nn.RMSNorm(H, eps=eps)

raw = mx.load(sys.argv[1])
def make_mtp(plus1, bits=None):
    m = MTP()
    m.load_weights([(k[len("mtp."):], (v + 1.0 if (plus1 and v.ndim == 1) else v)) for k, v in raw.items()], strict=True)
    m.set_dtype(mx.float16)
    if bits: nn.quantize(m, group_size=64, bits=bits)
    mx.eval(m.parameters()); return m

def mtp_forward(m, hid, emb, hid_first, cache=None):
    a, b = m.pre_fc_norm_embedding(emb), m.pre_fc_norm_hidden(hid)
    x = m.fc(mx.concatenate([b, a] if hid_first else [a, b], axis=-1))
    mask = create_attention_mask(x, cache)
    h = m.layers[0](x, mask=mask, cache=cache)
    return h, head(m.norm(h))

def hidden_states(ids):
    h = tm.embed_tokens(ids)
    fa, ssm = create_attention_mask(h, None), create_ssm_mask(h, None)
    for layer in tm.layers:
        h = layer(h, mask=ssm if layer.is_linear else fa, cache=None)
    return h, tm.norm(h)

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
PROMPTS = [
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
]

seqs = []
for name, text, tools in PROMPTS:
    p = tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], tools=tools, add_generation_prompt=True,
                                           enable_thinking=False, tokenize=False), add_special_tokens=False)
    t0 = time.perf_counter(); r = greedy(p, 192); s = time.perf_counter() - t0
    seqs.append((name, p, r))
    print(f"{name:12s} prompt {len(p):4d} reply {len(r):3d} tokens ({s:.1f} s): {tok.decode(r)[:80]!r}", flush=True)

# teacher-forced pass: hidden states and the main model's own check of the replies
feats = []; agree = n_agree = 0
for name, p, r in seqs:
    ids = mx.array(p + r)[None]; P, Lt = len(p), len(p) + len(r)
    pre, post = hidden_states(ids)
    main = mx.argmax(head(post)[0], axis=-1)
    mx.eval(pre, post, main)
    for i in range(P - 1, Lt - 1):
        agree += int(main[i].item() == (p + r)[i + 1]); n_agree += 1
    feats.append((name, ids, P, Lt, pre, post))
print(f"precondition: teacher-forced main model = its own greedy reply on {agree} of {n_agree} tokens", flush=True)

def accept(m, which, hid_first, depth2=False):
    """first guesses right / tried, per prompt; with depth2, also P(second right | first right)."""
    per = {}; c1 = n1 = c2 = n2 = 0
    for name, ids, P, Lt, pre, post in feats:
        hid = pre if which == "pre" else post
        h1, lg1 = mtp_forward(m, hid[:, :-1], tm.embed_tokens(ids[:, 1:]), hid_first)
        g1 = mx.argmax(lg1[0], axis=-1); mx.eval(g1)
        g1 = g1.tolist(); toks = ids[0].tolist()
        ok1 = {i: g1[i] == toks[i + 2] for i in range(P - 1, Lt - 2)}
        per[name] = (sum(ok1.values()), len(ok1)); c1 += sum(ok1.values()); n1 += len(ok1)
        if depth2:
            h2in = h1 if which == "pre" else m.norm(h1)
            _, lg2 = mtp_forward(m, h2in[:, :-1], tm.embed_tokens(ids[:, 2:]), hid_first)
            g2 = mx.argmax(lg2[0], axis=-1).tolist()
            for i in range(P - 1, Lt - 3):
                if ok1[i]: n2 += 1; c2 += int(g2[i] == toks[i + 3])
    return c1, n1, per, c2, n2

res = {"variants": {}}
for plus1 in (True, False):
    m = make_mtp(plus1)
    for which in ("pre", "post"):
        for hid_first in (False, True):
            c1, n1, per, _, _ = accept(m, which, hid_first)
            key = f"norm {'1+w' if plus1 else 'w'} | hidden {which}-norm | concat {'[hidden, emb]' if hid_first else '[emb, hidden]'}"
            res["variants"][key] = c1 / n1
            print(f"{key:60s} first guess right {c1:4d} / {n1} = {c1 / n1:.1%}", flush=True)
best = max(res["variants"], key=res["variants"].get)
plus1 = "1+w" in best; which = "pre" if "pre-norm" in best else "post"; hid_first = "[hidden, emb]" in best
print(f"\nbest variant: {best}", flush=True)
for bits in (None, 4):
    m = make_mtp(plus1, bits)
    c1, n1, per, c2, n2 = accept(m, which, hid_first, depth2=True)
    lab = "4-bit" if bits else "float16"
    res[lab] = {"a1": c1 / n1, "a2_given_a1": c2 / max(n2, 1), "per_prompt": {k: v[0] / v[1] for k, v in per.items()}}
    print(f"MTP {lab}: first {c1}/{n1} = {c1 / n1:.1%}; second given first {c2}/{n2} = {c2 / max(n2, 1):.1%}", flush=True)
    print("   per prompt: " + ", ".join(f"{k} {v[0] / v[1]:.0%}" for k, v in per.items()), flush=True)

# cost: a main step with S tokens against one MTP guess, timed the same way (mx.eval each step, no pipelining)
def tm_step(S, n=30):
    cache = make_prompt_cache(model); mx.eval(model(feats[0][1], cache=cache))
    x = mx.array([[feats[0][1][0, 5].item()] * S])
    for _ in range(3): mx.eval(model(x, cache=cache))
    t = time.perf_counter()
    for _ in range(n): mx.eval(model(x, cache=cache))
    return (time.perf_counter() - t) / n
def mtp_step(m, n=30):
    c = KVCache(); hid = feats[0][4][:, :1]; e = tm.embed_tokens(feats[0][1][:, :1])
    for _ in range(3): mx.eval(mx.argmax(mtp_forward(m, hid, e, hid_first, c)[1], axis=-1))
    t = time.perf_counter()
    for _ in range(n): mx.eval(mx.argmax(mtp_forward(m, hid, e, hid_first, c)[1], axis=-1))
    return (time.perf_counter() - t) / n
runs = [(S, tm_step(S)) for S in (1, 2, 3, 1)]  # S=1 twice, first and last: the noise floor
print(f"\nmain step: " + ", ".join(f"S={S} {v * 1e3:.2f} ms" for S, v in runs), flush=True)
steps = {2: runs[1][1], 3: runs[2][1]}; s1 = min(runs[0][1], runs[3][1])
res["timing_ms"] = {"main_S1_first": runs[0][1] * 1e3, "main_S2": runs[1][1] * 1e3, "main_S3": runs[2][1] * 1e3, "main_S1_last": runs[3][1] * 1e3}
a1 = res["4-bit"]["a1"]; a2 = res["4-bit"]["a2_given_a1"]
for bits in (None, 4):
    m = make_mtp(plus1, bits); c = mtp_step(m); lab = "4-bit" if bits else "float16"
    res["timing_ms"][f"mtp_{lab}"] = c * 1e3
    r2, r3, cc = steps[2] / s1, steps[3] / s1, c / s1
    k1 = (1 + a1) / (r2 + cc); k2 = (1 + a1 + a1 * a2) / (r3 + 2 * cc)
    print(f"MTP {lab} guess: {c * 1e3:.2f} ms = {cc:.2f} of a main step | estimate, 1 draft: {k1:.2f}x, 2 drafts: {k2:.2f}x "
          f"(with 4-bit acceptance; before the pipelining loss)", flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
