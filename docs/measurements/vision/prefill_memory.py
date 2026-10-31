"""Usage: prefill_memory.py MODEL PICTURE MODE(exact|mem) OUT.json
exact: the invoice question, prefill_step 256 (a chunk ends inside the image), 60 tokens.
mem: about 36,000 tokens of text, then the picture, then a question; 8 tokens. Both record the peak memory of
Engine.generate_embedded and the reply's tokens."""
import base64, json, sys, time
from pathlib import Path
import mlx.core as mx
import locals as L
from locals import engine as eng
from locals.server import Server

model_id, pic, mode, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
print("locals from", L.__file__, flush=True)
rec = []
real = eng.Engine.generate_embedded
def wrapped(self, prompt, spans, features, *a, **k):
    if mode == "exact":
        self.prefill_step = 256
    mx.synchronize(); mx.clear_cache(); mx.reset_peak_memory()
    a0 = mx.get_active_memory()
    t = time.perf_counter()
    res = real(self, prompt, spans, features, *a, **k)
    mx.synchronize()
    rec.append({"prompt": len(prompt), "spans": [list(s) for s in spans], "prefill_step": self.prefill_step,
                "active_before_mb": round(a0 / 2**20), "peak_mb": round(mx.get_peak_memory() / 2**20),
                "peak_over_start_mb": round((mx.get_peak_memory() - a0) / 2**20), "s": round(time.perf_counter() - t, 2),
                "tokens": list(res.tokens)})
    return res
eng.Engine.generate_embedded = wrapped

srv = Server(model_id, engines=("locals",), preload=("locals",))
url = "data:image/png;base64," + base64.b64encode(pic.read_bytes()).decode()
if mode == "exact":
    parts = [{"type": "text", "text": "Read this carefully."}, {"type": "image_url", "image_url": {"url": url}},
             {"type": "text", "text": "What is the invoice number, the customer's name and the total?"}]
    max_tokens = 60
else:
    filler = "".join(f"Line {i}: the parcel {i * 7919 % 10007} left the depot at {i % 24:02d}:{i * 13 % 60:02d}.\n" for i in range(1500))
    parts = [{"type": "text", "text": filler}, {"type": "image_url", "image_url": {"url": url}},
             {"type": "text", "text": "What is the invoice number?"}]
    max_tokens = 8
req = {"model": "locals", "max_tokens": max_tokens, "temperature": 0, "messages": [{"role": "user", "content": parts}]}
for _ in range(2):
    r = srv.complete(req)
    rec[-1]["answer"] = r["choices"][0]["message"]["content"]
    print({k: v for k, v in rec[-1].items() if k != "tokens"}, flush=True)
Path(out).write_text(json.dumps({"locals": L.__file__, "model": model_id, "mode": mode, "runs": rec}, indent=1))
srv.unload()
