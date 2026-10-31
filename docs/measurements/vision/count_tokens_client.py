import base64
import io
import json
import statistics
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

port = sys.argv[1]
rng = np.random.default_rng(0)
w, h = 4032, 3024
y, x = np.mgrid[0:h, 0:w]
a = np.stack([(x * 255 // w), (y * 255 // h), ((x + y) * 127 // (w + h))], -1).astype(np.int16)
a = np.clip(a + rng.integers(-20, 20, a.shape), 0, 255).astype(np.uint8)
b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG", quality=90)
img = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(b.getvalue()).decode()}}
body = {"model": "claude-x", "max_tokens": 1, "messages": [{"role": "user", "content": [img, img, {"type": "text", "text": "What colours?"}]}]}
def post(path, d):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(d).encode(), {"content-type": "application/json"})
    t0 = time.perf_counter(); out = json.load(urllib.request.urlopen(r, timeout=600)); return out, time.perf_counter() - t0
ts = []
for _ in range(7):
    c, t = post("/v1/messages/count_tokens", body); ts.append(t)
m, _ = post("/v1/messages", body)
print(f"2 images 4032x3024 JPEG: count_tokens={c['input_tokens']} messages.usage.input_tokens={m['usage']['input_tokens']} "
      f"count median {statistics.median(ts)*1000:.0f} ms (min {min(ts)*1000:.0f}, max {max(ts)*1000:.0f}, n=7)")
