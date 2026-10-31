"""Usage: turn2_cost.py MODEL — a follow-up turn about two images: its time, its cached tokens, and its answer.
Each run empties the prompt cache, asks the first turn, then times the second."""
import base64
import io
import statistics
import sys
import time

import numpy as np
from PIL import Image

import locals as L
from locals.server import Server

model_id = sys.argv[1]
print("locals from", L.__file__, flush=True)
rng = np.random.default_rng(0)
def pic(w, h):
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([(x * 255 // w), (y * 255 // h), ((x + y) * 127 // (w + h))], -1).astype(np.int16)
    a = np.clip(a + rng.integers(-20, 20, a.shape), 0, 255).astype(np.uint8)
    b = io.BytesIO(); Image.fromarray(a).save(b, "PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()}}
filler = "".join(f"Line {i}: the parcel {i * 7919 % 10007} left the depot at {i % 24:02d}:{i * 13 % 60:02d}.\n" for i in range(150))
first = [{"role": "user", "content": [{"type": "text", "text": filler}, pic(1280, 800), pic(1024, 1024),
                                      {"type": "text", "text": "Describe both pictures in one sentence."}]}]
srv = Server(model_id, engines=("locals",), preload=("locals",))
ts, cached, answers = [], [], []
for _ in range(4):
    if srv.service("locals")._lru is not None:
        srv.service("locals")._lru.trim_to(n_sequences=0)
    a = srv.complete({"model": "locals", "max_tokens": 40, "temperature": 0, "messages": first})["choices"][0]["message"]["content"]
    follow = first + [{"role": "assistant", "content": a}, {"role": "user", "content": "Which one is brighter?"}]
    t = time.perf_counter()
    r = srv.complete({"model": "locals", "max_tokens": 1, "temperature": 0, "messages": follow})
    ts.append(time.perf_counter() - t)
    cached.append(r["usage"]["prompt_tokens_details"]["cached_tokens"]); answers.append(r["choices"][0]["message"]["content"])
print(f"{model_id}: turn 2 of {r['usage']['prompt_tokens']} tokens, cached {cached[1:]}, median {statistics.median(ts[1:])*1000:.0f} ms "
      f"(runs 2-4: {', '.join(f'{t*1000:.0f}' for t in ts[1:])}), first token {answers[1:]!r}", flush=True)
srv.unload()
