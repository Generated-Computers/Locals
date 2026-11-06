"""Fetch only the mtp.* tensors of Qwen/Qwen3.5-4B (BF16) with HTTP range requests, save them as one
safetensors file. Usage: download_mtp.py OUT.safetensors"""
import json, struct, sys, urllib.request
import numpy as np
import mlx.core as mx

REPO = "https://huggingface.co/Qwen/Qwen3.5-4B/resolve/main/"
UA = {"User-Agent": "locals-mtp-test"}

def get(url, a, b):
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={a}-{b - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) != b - a:
        raise RuntimeError(f"asked {b - a} bytes, got {len(data)} (range ignored?)")
    return data

idx = json.loads(urllib.request.urlopen(urllib.request.Request(REPO + "model.safetensors.index.json", headers=UA)).read())
want = {k: f for k, f in idx["weight_map"].items() if k.startswith("mtp.")}
out, total = {}, 0
for shard in sorted(set(want.values())):
    url = REPO + shard
    n = struct.unpack("<Q", get(url, 0, 8))[0]
    head = json.loads(get(url, 8, 8 + n)); base = 8 + n
    for k in sorted(k for k, f in want.items() if f == shard):
        t = head[k]; a, b = t["data_offsets"]
        assert t["dtype"] == "BF16", (k, t["dtype"])
        raw = np.frombuffer(get(url, base + a, base + b), dtype=np.uint16).reshape(t["shape"])
        arr = mx.array(raw).view(mx.bfloat16)
        out[k] = arr; total += b - a
        print(f"{k:55s} {str(t['shape']):16s} mean {arr.astype(mx.float32).mean().item():+.4f} "
              f"absmax {mx.abs(arr.astype(mx.float32)).max().item():.3f}", flush=True)
mx.save_safetensors(sys.argv[1], out)
print(f"{len(out)} tensors, {total / 1e6:.1f} MB, saved to {sys.argv[1]}")
