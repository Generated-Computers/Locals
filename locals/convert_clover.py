"""Make Clover-35B-A3B-Preview's MLX build: download the bf16 checkpoint one file at a time, quantize it, delete it.

    python -m locals.convert_clover [--out ~/models/Clover-35B-A3B-Preview-MLX-mixed-3-4bit]

The bf16 checkpoint is 69.3 GB in 28 files; the build 16.7 GB. Each file is downloaded, written in MLX's layout (the
experts' fused gate/up split into switch_mlp gate and up, conv1d moved to MLX's (C, K, 1), the zero-centred norms
stored as 1 + w), quantized, saved as one output file and deleted, so the disk holds the build and at most three
bf16 files at once. Resumable: a file whose output is done is skipped.

Recipe (affine, group 64): routed experts 3-bit, their down projection 4-bit in the 20 layers mlx-lm's mixed_3_4
recipe gives more bits; the embedding, lm_head, routers, the shared-expert gates and the GDN a/b projections 8-bit;
every other linear layer 4-bit. Other parameters as stored.
"""

import json
import os
import queue
import shutil
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")  # no second copy of what is downloaded
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import mlx.core as mx
from huggingface_hub import hf_hub_download

REPO = "Diffuspace/Clover-35B-A3B-Preview"
REV = "4b0402b65e523d0183a14c431955438714cb16ff"
GS = 64
LAYERS = 40
SMALL = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
         "chat_template.jinja", "model.safetensors.index.json", "LICENSE", "ATTRIBUTIONS.md"]
ZERO_CENTRED = (".input_layernorm.weight", ".post_attention_layernorm.weight", "model.norm.weight",
                ".q_norm.weight", ".k_norm.weight")
EIGHT_BIT = ("model.embed_tokens.weight", "lm_head.weight", ".mlp.gate.weight", ".shared_expert_gate.weight",
             ".in_proj_a.weight", ".in_proj_b.weight")


def more_bits(i: int) -> bool:  # mlx-lm's mixed_3_4 choice of layers
    return i < LAYERS // 8 or i >= 7 * LAYERS // 8 or (i - LAYERS // 8) % 3 == 2


def layer_of(name: str) -> int:
    return int(name.split(".")[2])


def quantize(prefix, w, bits, out, quant, log):
    wq, s, b = mx.quantize(w, group_size=GS, bits=bits)
    mx.eval(wq, s, b)
    back = mx.dequantize(wq, s, b, group_size=GS, bits=bits)
    wf = w.astype(mx.float32)
    err = mx.sqrt(mx.sum(mx.square(back.astype(mx.float32) - wf)) / mx.maximum(mx.sum(mx.square(wf)), 1e-30))
    out[prefix + ".weight"], out[prefix + ".scales"], out[prefix + ".biases"] = wq, s, b
    quant[prefix] = {"group_size": GS, "bits": bits}
    log.write(f"{prefix} bits={bits} shape={tuple(w.shape)} rel_err={float(err):.5f}\n")


def convert(name, a, out, quant, log):
    if name.endswith(".mlp.experts.gate_up_proj"):
        base = name[: -len("experts.gate_up_proj")] + "switch_mlp."
        mid = a.shape[-2] // 2
        quantize(base + "gate_proj", a[..., :mid, :], 3, out, quant, log)
        quantize(base + "up_proj", a[..., mid:, :], 3, out, quant, log)
    elif name.endswith(".mlp.experts.down_proj"):
        base = name[: -len("experts.down_proj")] + "switch_mlp."
        quantize(base + "down_proj", a, 4 if more_bits(layer_of(name)) else 3, out, quant, log)
    elif name.endswith("conv1d.weight"):
        out[name] = a.moveaxis(2, 1) if a.shape[-1] != 1 else a  # HF (C, 1, K) -> MLX (C, K, 1)
    elif name.endswith(ZERO_CENTRED):
        out[name] = (a.astype(mx.float32) + 1.0).astype(a.dtype)
    elif a.ndim == 2 and name.endswith(".weight"):  # every linear layer and the embedding
        quantize(name[: -len(".weight")], a, 8 if name.endswith(EIGHT_BIT) else 4, out, quant, log)
    else:
        out[name] = a


def fetch(fname, staging):
    for attempt in range(6):
        try:
            return hf_hub_download(REPO, fname, revision=REV, local_dir=staging)
        except Exception as e:  # network hiccups: try again from where it stopped
            print(f"download {fname} failed ({e!r}); retry {attempt + 1}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"could not download {fname}")


def convert_all(staging, outdir):
    staging, outdir = Path(staging), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = outdir / "_convert"
    meta.mkdir(exist_ok=True)
    for f in SMALL:
        if not (meta / f).exists():
            shutil.copy(fetch(f, str(staging)), meta / f)
    index = json.loads((meta / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted(set(index.values()), key=lambda f: (not f.startswith("bd_"), f))
    n = len(files)
    outname = {f: f"model-{i + 1:05d}-of-{n:05d}.safetensors" for i, f in enumerate(files)}
    todo = [f for f in files if not (meta / (outname[f] + ".done")).exists()]
    print(f"{len(files)} files, {len(todo)} to do", flush=True)

    q: "queue.Queue" = queue.Queue(maxsize=1)

    def downloader():
        for f in todo:
            t = time.time()
            p = fetch(f, str(staging))
            size = os.path.getsize(p)
            print(f"downloaded {f} {size / 1e9:.2f} GB in {time.time() - t:.0f} s ({size / 1e6 / max(time.time() - t, 1e-3):.0f} MB/s)", flush=True)
            q.put((f, p))
        q.put(None)

    threading.Thread(target=downloader, daemon=True).start()
    with open(meta / "quant_log.txt", "a") as log:
        while (item := q.get()) is not None:
            f, p = item
            t = time.time()
            weights = mx.load(p)
            out, quant = {}, {}
            for name in sorted(weights):
                convert(name, weights[name], out, quant, log)
                mx.eval(*[out[k] for k in out])
            log.flush()
            mx.save_safetensors(str(outdir / outname[f]), out, metadata={"format": "mlx"})
            (meta / (outname[f] + ".quant.json")).write_text(json.dumps(quant))
            (meta / (outname[f] + ".keys.json")).write_text(json.dumps(sorted(out)))
            (meta / (outname[f] + ".done")).write_text("")
            del weights, out
            os.remove(p)
            mx.clear_cache()
            print(f"converted {f} -> {outname[f]} in {time.time() - t:.0f} s; free disk "
                  f"{shutil.disk_usage(str(outdir)).free / 1e9:.1f} GB", flush=True)

    # config, index, tokenizer
    quant, weight_map = {}, {}
    for f in files:
        quant.update(json.loads((meta / (outname[f] + ".quant.json")).read_text()))
        for k in json.loads((meta / (outname[f] + ".keys.json")).read_text()):
            weight_map[k] = outname[f]
    cfg = json.loads((meta / "config.json").read_text())
    cfg.pop("auto_map", None)
    qcfg = {"group_size": GS, "bits": 4, "mode": "affine",
            **{k: v for k, v in sorted(quant.items()) if v["bits"] != 4}}
    cfg["quantization"] = qcfg
    cfg["quantization_config"] = qcfg
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))
    total = sum((outdir / o).stat().st_size for o in outname.values())
    (outdir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2))
    for f in SMALL:
        if f not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(meta / f, outdir / f)
    (outdir / "README.md").write_text(
        f"# Clover-35B-A3B-Preview, MLX, mixed 3/4-bit\n\nConverted from https://huggingface.co/{REPO} "
        f"(revision {REV}) with Locals (python -m locals.convert_clover).\n\nAffine quantization, group size 64: routed experts "
        "3-bit, their down projection 4-bit in layers 0-4, 7, 10, ..., 34 and 35-39; embedding, lm_head, "
        "routers, shared-expert gates and the GDN a/b projections 8-bit; every other linear layer 4-bit.\n"
        "The model is Apache-2.0 (see LICENSE and ATTRIBUTIONS.md).\n")
    print(f"done: {total / 1e9:.2f} GB in {outdir}", flush=True)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="python -m locals.convert_clover", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="~/models/Clover-35B-A3B-Preview-MLX-mixed-3-4bit", help="the build's folder")
    ap.add_argument("--staging", default=None, help="where bf16 files wait their turn (default: OUT/_download)")
    args = ap.parse_args(argv)
    out = Path(args.out).expanduser()
    convert_all(Path(args.staging).expanduser() if args.staging else out / "_download", out)


if __name__ == "__main__":
    main()
