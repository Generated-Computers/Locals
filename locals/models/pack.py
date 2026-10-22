"""An expert-major copy of a streamed model's experts (``locals.models.flint_streamed.ExpertStore`` reads it if present).

The INT4 build stacks a layer's 256 experts in each of nine tensors (gate, up and down projection: 4-bit weights,
scales, biases), so one expert is nine slices in nine places of the file, and reading it takes four or more
requests. Laid out expert by expert, an expert is one contiguous read, which an SSD serves faster: on the M2 Pro
0.76 ms for one expert alone against 1.28 ms as four reads at once, and 4.4 GB/s against 2.6 with eight in flight.
One file per layer: its experts in order, each the nine parts in ExpertStore's order. Written once (the layer's
tensors read whole, written expert by expert), next to nothing else: ~/.cache/locals/experts/<model>-<files>/."""
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Dict, Optional

PARTS = [(p, k) for p in ("gate_proj", "up_proj", "down_proj") for k in ("weight", "scales", "biases")]


def _tensors(folder: Path) -> Dict[int, list]:
    """layer -> [(file, start, bytes per expert, experts)] of its nine expert tensors, in PARTS order."""
    found: Dict[int, dict] = {}
    for f in sorted(folder.glob("model*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, t in header.items():
            if name == "__metadata__" or ".mlp.switch_mlp." not in name:
                continue
            layer = int(name.split(".layers.")[1].split(".")[0])
            proj, kind = name.split(".switch_mlp.")[1].split(".")
            b, e = t["data_offsets"]
            found.setdefault(layer, {})[(proj, kind)] = (f, 8 + n + b, (e - b) // t["shape"][0], t["shape"][0])
    return {layer: [parts[p] for p in PARTS] for layer, parts in found.items() if len(parts) == len(PARTS)}


def pack_dir(folder: Path) -> Path:
    """Where the pack of the model in ``folder`` lives (named after the folder's weight files: sizes and times)."""
    key = "|".join(f"{f.name}:{f.stat().st_size}:{int(f.stat().st_mtime)}" for f in sorted(folder.glob("model*.safetensors")))
    name = folder.resolve().name if folder.name not in ("", ".") else "model"
    parent = folder.resolve().parent.parent.name if folder.resolve().parent.name == "snapshots" else name
    root = Path(os.environ.get("LOCALS_EXPERT_PACK_DIR") or Path.home() / ".cache" / "locals" / "experts")
    return root / f"{parent[-60:]}-{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def find(folder: Path) -> Optional[Path]:
    """The pack of ``folder``'s model if it was written whole, else None."""
    d = pack_dir(folder)
    return d if (d / "done").exists() else None


def write(folder: Path, log=print) -> Path:
    """Write the pack (about the size of the experts: 21.4 GB for Flint); returns its folder."""
    d = pack_dir(folder)
    d.mkdir(parents=True, exist_ok=True)
    layers = _tensors(folder)
    total = sum(size * experts for parts in layers.values() for _, _, size, experts in parts)
    st = os.statvfs(d)
    if st.f_bavail * st.f_frsize < total + (2 << 30):
        raise OSError(f"the expert pack needs {total / 2**30:.1f} GB in {d}; {st.f_bavail * st.f_frsize / 2**30:.1f} GB free")
    for i, (layer, parts) in enumerate(sorted(layers.items())):
        out = d / f"layer-{layer}.bin"
        if out.exists() and out.stat().st_size == sum(size * experts for _, _, size, experts in parts):
            continue
        data = []
        for f, start, size, experts in parts:
            with open(f, "rb") as fh:
                fh.seek(start)
                data.append(memoryview(fh.read(size * experts)))
        E = parts[0][3]
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            for e in range(E):
                fh.write(b"".join(bytes(m[e * size:(e + 1) * size]) for m, (_, _, size, _) in zip(data, parts)))
        tmp.rename(out)
        log(f"locals: expert pack, layer {layer} ({i + 1} of {len(layers)})")
    (d / "done").write_text(json.dumps({"layers": sorted(layers), "bytes": total}))
    return d


def main(argv=None) -> None:
    """python -m locals.models.pack MODEL: write MODEL's expert pack (a Hugging Face id or a folder)."""
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="python -m locals.models.pack", description=main.__doc__)
    ap.add_argument("model")
    args = ap.parse_args(argv)
    folder = Path(args.model).expanduser()
    if not folder.exists():
        from ..defaults import cached_snapshot

        folder = cached_snapshot(args.model, ["*.json", "*.safetensors"])
        if folder is None:
            sys.exit(f"{args.model} is not downloaded")
    if not _tensors(folder):
        sys.exit(f"{args.model} has no stacked expert tensors to pack")
    have = find(folder)
    print(f"{have} (already written)" if have is not None else write(folder))


if __name__ == "__main__":
    main()
