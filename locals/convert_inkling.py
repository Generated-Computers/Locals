"""Make Inkling 2 Mini's build for Locals from Thinking Machines' release.

    python -m locals.convert_inkling [--src ~/models/Inkling-2-Mini] [--out ~/models/Inkling-2-Mini-MLX-nvfp4]

The release (thinkingmachines/Inkling-2-Mini, 19.5 GB) has the routed experts in NVFP4 in layers 0 to 19 and in
MXFP8 in layers 20 to 23, and everything else in bfloat16. Its maker asks for 32 GB; the build is 15.6 GB:

* the experts of layers 0 to 19 as released, bit for bit;
* the experts of layers 20 to 23 in NVFP4 like the others, each block of 16 with the scale that loses least
  (``--tail mxfp8`` keeps them as released: 17.5 GB, for a Mac with 32 GB);
* attention's six projections, the embedding and the output layer 8-bit (affine, groups of 64; ``--attention-bits``
  and ``--head-bits`` choose others);
* everything else as released, the routers and the loop's gates in float32 as the model computes them.

Text only: the vision tower is left out. The bundled draft model (``mtp.safetensors``, 0.64 GB) goes in as released
unless ``--no-draft`` (``--draft-only`` adds it to a build already made). ``--src`` is downloaded if it is not there.
"""

import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = "thinkingmachines/Inkling-2-Mini"
WEIGHTS = "model-00001-of-00001.safetensors"
DRAFT = "mtp.safetensors"
SMALL = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
         "chat_template.jinja"]
GS = 64
ATTENTION = tuple(f".self_attn.{n}_proj.weight" for n in ("q", "k", "v", "r", "o", "gate"))
HEAD = ("lm_head.weight", "model.language_model.embed_tokens.weight")

_codes = np.arange(127)
_e, _m = _codes >> 3, _codes & 7
E4M3 = mx.array(np.where(_e == 0, _m / 8 * 2.0**-6, (1 + _m / 8) * 2.0 ** (_e - 7)).astype(np.float32))  # by code
E2M1 = mx.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=mx.float32)  # the 4-bit values' magnitudes, by their low 3 bits
_EDGES = mx.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=mx.float32)  # halfway between them


def _fp4(x: mx.array):
    """The 4-bit codes of ``x`` (already divided by its scale) and the values they stand for."""
    mag = (mx.abs(x)[..., None] > _EDGES).sum(axis=-1)
    return mag + 8 * (x < 0), E2M1[mag] * mx.sign(x)


def nvfp4(w: mx.array, shifts=range(-3, 4)):
    """One expert's matrix (rows, n) in NVFP4 with its own global scale, as MLX reads it: the packed 4-bit weights
    (rows, n / 8) uint32, the 8-bit scales (rows, n / 16), the global scale, and the relative error. The scale of
    each block of 16 is the 8-bit float, among those next to the block's largest weight over 6, that loses least
    (MLX's quantizer takes the nearest: 10% more error on these weights)."""
    rows, n = w.shape
    g = mx.abs(w).max()
    k = g / 2688.0
    b = w.reshape(rows, n // 16, 16)
    near = mx.argmin(mx.abs((mx.abs(b).max(axis=-1) / 6 / k)[..., None] - E4M3), axis=-1).astype(mx.int32)
    best, code = None, None
    for d in shifts:
        c = mx.clip(near + d, 1, 126)
        s = E4M3[c][..., None] * k
        err = mx.square(_fp4(b / s)[1] * s - b).sum(axis=-1)
        if best is None:
            best, code = err, c
        else:
            better = err < best
            best, code = mx.where(better, err, best), mx.where(better, c, code)
        mx.eval(best, code)
    nib, _ = _fp4(b / (E4M3[code][..., None] * k))
    nib = np.array(nib.reshape(rows, n // 2, 2).astype(mx.uint8))
    packed = mx.array((nib[..., 0] | (nib[..., 1] << 4)).view(np.uint32))  # low nibble first
    rel = mx.sqrt(best.sum() / mx.square(w).sum())
    return packed, code.astype(mx.uint8), g, rel


def tail_nvfp4(weight: mx.array, scales: mx.array, log):
    """A layer's MXFP8 experts (the bytes as uint32, the scales of each 32) in NVFP4 with a scale per expert."""
    ws, ss, gs, rels = [], [], [], []
    for e in range(weight.shape[0]):
        w = mx.dequantize(weight[e], scales[e], None, group_size=32, bits=8, mode="mxfp8", dtype=mx.float32)
        q, s, g, rel = nvfp4(w)
        if e == 0:  # MLX reads back what was meant
            back = mx.dequantize(q, s, None, group_size=16, bits=4, mode="nvfp4", global_scale=g, dtype=mx.float32)
            mine = mx.sqrt(mx.square(back - w).sum() / mx.square(w).sum())
            assert abs(float(mine) - float(rel)) < 1e-4, (float(mine), float(rel))
        mx.eval(q, s, g, rel)
        ws.append(q), ss.append(s), gs.append(g), rels.append(float(rel))
    log(f"  NVFP4 from MXFP8: relative error {np.mean(rels):.4f} (worst expert {np.max(rels):.4f})")
    return mx.stack(ws), mx.stack(ss), mx.stack(gs).astype(mx.float32)


def add_draft(src: Path, out: Path, config: dict, draft_config: dict, log=print) -> None:
    """The release's draft model as the build's last weight file, as released (bfloat16, 0.64 GB), under ``mtp.``;
    the build's config names it (``draft_config``), and the model then has it (inkling2_loop.Model.mtp)."""
    weights = {"mtp." + k: v for k, v in mx.load(str(src / DRAFT)).items()}
    mx.save_safetensors(str(out / "model-draft.safetensors"), weights, metadata={"format": "mlx"})
    config["draft_config"] = draft_config
    log(f"wrote model-draft.safetensors: {sum(v.nbytes for v in weights.values()) / 1e9:.2f} GB")


def convert(src: Path, out: Path, attention_bits: int = 8, head_bits: int = 8, tail: str = "nvfp4",
            shard_bytes: float = 5e9, log=print, draft: bool = True):
    from .models import inkling2_loop as ink

    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    model = ink.Model(ink.ModelArgs.from_dict(config))
    weights = model.sanitize(mx.load(str(src / WEIGHTS)))  # (lazy: a tensor is read when it is converted)
    names = sorted(weights, key=lambda k: (int(k.split(".layers.")[1].split(".")[0]) if ".layers." in k else -1, k))

    quant, shard, shards, size, index = {}, {}, [], 0, {}

    def flush():
        nonlocal shard, size
        if shard:
            shards.append(shard)
            name = f"model-{len(shards):05d}.safetensors"
            mx.save_safetensors(str(out / name), shard, metadata={"format": "mlx"})
            index.update({k: name for k in shard})
            log(f"wrote {name}: {size / 1e9:.2f} GB ({time.time() - t0:.0f} s)")
            shard, size = {}, 0
            shards[-1] = None
            mx.clear_cache()

    def put(k, v):
        nonlocal size
        mx.eval(v)
        shard[k] = v
        size += v.nbytes

    done = set()
    for k in names:
        if k in done:
            continue
        v = weights[k]
        if ".mlp.experts." in k:
            base = k.rsplit(".", 1)[0]
            w, s = weights[base + ".weight"], weights[base + ".scales"]
            done.update({base + ".weight", base + ".scales", base + ".global_scale"})
            if base + ".global_scale" in weights:  # NVFP4: as released
                put(base + ".weight", w), put(base + ".scales", s), put(base + ".global_scale", weights[base + ".global_scale"])
            elif tail == "mxfp8":
                put(base + ".weight", w), put(base + ".scales", s)
            else:
                log(f"{base}:")
                w, s, g = tail_nvfp4(w, s, log)
                put(base + ".weight", w), put(base + ".scales", s), put(base + ".global_scale", g)
        elif k.endswith(ATTENTION) or k in HEAD:
            bits = head_bits if k in HEAD else attention_bits
            if bits >= 16:
                put(k, v)
            else:
                path = k[: -len(".weight")]
                wq, s, b = mx.quantize(v, group_size=GS, bits=bits)
                put(path + ".weight", wq), put(path + ".scales", s), put(path + ".biases", b)
                quant[path] = {"group_size": GS, "bits": bits}
        else:
            put(k, v)
        if size >= shard_bytes:
            flush()
    flush()

    n = len(shards)
    weight_map = {}
    for i in range(n):  # the shards' names with their count, as mlx-lm writes them
        old, new = f"model-{i + 1:05d}.safetensors", f"model-{i + 1:05d}-of-{n:05d}.safetensors"
        (out / old).rename(out / new)
        weight_map.update({k: new for k, f in index.items() if f == old})
    total = sum((out / f).stat().st_size for f in set(weight_map.values()))
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2))

    config.pop("vision_config", None)  # text only
    draft_config = config.pop("mtp_config", None)
    if draft and draft_config and (src / DRAFT).exists():
        add_draft(src, out, config, draft_config, log)
    bits = sorted({q["bits"] for q in quant.values()})
    common = max(bits, key=lambda b: sum(q["bits"] == b for q in quant.values())) if bits else 8
    config["quantization"] = {"group_size": GS, "bits": common, "mode": "affine",
                              **{p: q for p, q in sorted(quant.items()) if q["bits"] != common}}
    config["quantization_config"] = config["quantization"]
    (out / "config.json").write_text(json.dumps(config, indent=2))
    for f in SMALL[1:]:
        if (src / f).exists():
            shutil.copy(src / f, out / f)
    (out / "README.md").write_text(
        f"# Inkling 2 Mini, MLX, NVFP4 experts\n\nConverted from https://huggingface.co/{REPO} with Locals "
        "(`python -m locals.convert_inkling`). Text only: the vision tower is left out.\n\n"
        "- Routed experts of layers 0 to 19: NVFP4 with a scale per expert, as released, bit for bit.\n"
        + ("- Routed experts of layers 20 to 23: MXFP8, as released.\n" if tail == "mxfp8" else
           "- Routed experts of layers 20 to 23 (MXFP8 in the release): NVFP4, each block's scale the one that loses least.\n")
        + f"- Attention's projections: {attention_bits}-bit affine, groups of 64. The embedding and the output layer: "
        f"{head_bits}-bit.\n- Everything else as released.\n\nApache-2.0, as the release.\n")
    log(f"done: {total / 1e9:.2f} GB in {out} ({time.time() - t0:.0f} s)")
    return total


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="python -m locals.convert_inkling", description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default="~/models/Inkling-2-Mini", help="the release's folder (downloaded if missing)")
    ap.add_argument("--out", default="~/models/Inkling-2-Mini-MLX-nvfp4", help="the build's folder")
    ap.add_argument("--attention-bits", type=int, default=8, choices=(4, 5, 6, 8, 16), help="16: as released (bfloat16)")
    ap.add_argument("--head-bits", type=int, default=8, choices=(4, 5, 6, 8, 16))
    ap.add_argument("--tail", default="nvfp4", choices=("nvfp4", "mxfp8"), help="the experts of layers 20 to 23")
    ap.add_argument("--no-draft", action="store_true", help="leave the draft model (0.64 GB) out of the build")
    ap.add_argument("--draft-only", action="store_true", help="add the draft model to a build already made")
    args = ap.parse_args(argv)
    src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
    if not (src / WEIGHTS).exists():
        from huggingface_hub import snapshot_download

        snapshot_download(REPO, local_dir=str(src), allow_patterns=SMALL + [WEIGHTS, DRAFT])
    if args.draft_only:
        config = json.loads((out / "config.json").read_text())
        add_draft(src, out, config, json.loads((src / "config.json").read_text())["mtp_config"])
        for key in ("quantization", "quantization_config"):  # (mlx-lm writes both)
            config[key] = config["quantization"]
        (out / "config.json").write_text(json.dumps(config, indent=2))
        return
    convert(src, out, args.attention_bits, args.head_bits, args.tail, draft=not args.no_draft)


if __name__ == "__main__":
    main()
