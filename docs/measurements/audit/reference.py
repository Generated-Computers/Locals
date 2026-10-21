"""Each server's replies scored by the same model computed in float32 (teacher forcing): how often the token a
server wrote is the reference's own top choice, and its mean log-probability. A greedy decoder with exact
arithmetic would score 100%; rounding flips near-ties, so neither does, and the closer one rounds less.

    python reference.py mlxlm.json locals.json
"""
import json
import sys

import mlx.core as mx
from mlx_lm import load

from locals.runtime import set_compute_dtype

model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
print("reference dtype:", set_compute_dtype(model, "float32"))
for path in sys.argv[1:]:
    d = json.load(open(path))
    agree = total = 0
    lp = 0.0
    for res in d["results"]:
        for r in res["replies"]:
            p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True,
                                        enable_thinking=False)
            out = r["tokens"]
            logits = model(mx.array([p + out]))[0, len(p) - 1:-1].astype(mx.float32)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            o = mx.array(out)
            agree += int((mx.argmax(logits, axis=-1) == o).sum())
            lp += float(mx.take_along_axis(logp, o[:, None], axis=-1).sum())
            total += len(out)
    print(f"{d['label']:10s}: {total} tokens; the reference's top choice {agree / total:.2%}; mean log-prob {lp / total:.4f}")
