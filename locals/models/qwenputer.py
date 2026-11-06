"""Qwenputer (divagr1925/Qwen3.5-4B-Qwenputer): Qwen3.5-4B with a RISC-V computer compiled into its weights.

The checkpoint is a stock ``Qwen3_5ForCausalLM``, 43 layers at hidden size 3968 (18.8 GB in bf16). The first 11
layers are the computer: an RV32IM + Zba core with a bignum coprocessor whose RAM, ROM and registers live in the
Gated DeltaNet states, its weights computed by a compiler (1.05M of them are not zero). Then come Qwen3.5-4B's own
32 layers, widened with zeros. The vocabulary is Qwen's plus 230,324 tokens of the computer's trace. When Qwen
writes a ``<tool_call>`` the computer layers capture its text; at ``</tool_call>`` the computer runs (its own tokens,
which Qwen's layers never see) and prints the tool response, which comes back as Qwen's tokens, one byte each.

What the checkpoint holds, as measured on it:

- Qwen's 32 layers are Qwen3.5-4B's own (quantized to 4 bits they equal mlx-community's Qwen3.5-4B-MLX-4bit bit
  for bit), so they run as that model does: same weights, kernels and caches.
- The computer layers read and write only the 1,408 dims after Qwen's 2,560, and of those 1,265 at most. Every
  row of the embedding carries 62 "ballast" dims of 2^40 there, so each RMSNorm of a computer layer divides by
  exactly 2^37: Qwen's part of the residual (|x|^2 < 2) is far below fp32's resolution of that sum, and the
  computer computes the same on any token whatever Qwen's dims hold.
- After the last computer layer, a text token's residual is Qwen's embedding row plus three marks the computer
  leaves: its output code (21 dims of +-2^-20, set where the computer predicts the next token), tau (2^-40) and
  mu (0). Their squares are below 2^-38, so Qwen's norms see Qwen's embedding row alone.

So this module keeps the computer apart (``Chip``): its layers compacted to their nonzero rows and columns, run
in bf16 as transformers runs them (the ops below follow ``modeling_qwen3_5``'s dtypes step by step), on a
residual of the computer's dims only. ``convert`` makes that "chip pack" from the checkpoint (some 60 MB, against
18.8 GB); Qwen's part loads as Qwen3.5-4B.

Decoding (the authors' ``chipformer/rv/chat.py``, RvChat): every token runs through the computer layers; a text
token also runs through Qwen's layers (whose cache and positions see text alone). The next token is the
computer's where its output code is set, else Qwen's. The firmware (the parked ``calc`` program) is a snapshot of
the computer layers' recurrent states, written in after the prompt.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np

REPO = "divagr1925/Qwen3.5-4B-Qwenputer"
REVISION = "2d9068508f2b680badb7abcded96fdd107e9ba5d"  # the checkpoint the chip pack is made from
BASE = "mlx-community/Qwen3.5-4B-MLX-4bit"  # Qwen's part: its 32 layers, quantized, are this model's bit for bit
# The authors' reference package (their RV32 machine, circuit compiler, text bridge): it loads the firmware into a
# chip pack, and drafts the computer's runs while Locals serves (``chipformer()``).
CHIPFORMER = "ca13310e2f9f788d1dd3feeea8213c1b0c23913c"
INSTALL = (f'pip install "chipformer @ git+https://github.com/divagr18/qwenputer@{CHIPFORMER}'
           '#subdirectory=experimental/riscv"')
FIRMWARE_URL = ("https://raw.githubusercontent.com/divagr18/qwenputer/" + CHIPFORMER
                + "/experimental/riscv/tests/data/rv/calc-bridge.elf")  # the calc firmware, built with -DBRIDGE
FIRMWARE_SHA256 = "b643ad61291a947fa6c7498de9170b1d4da062dc1f34810feba9c4eebdec862d"
# The toolbox: more tools the computer runs, in RV32 assembly (qwenputer_tools.s) linked after calc's firmware; calc's
# main loop resumes from its park at RESUME (lui a1, 0x8), which a jump to the toolbox's dispatcher replaces
TOOLS_SOURCE = Path(__file__).with_name("qwenputer_tools.s")
RESUME = 0x41484
# and the toolbox's tools in C (qwenputer_c/, compiled by its build.py into TOOLS_C), linked at fixed addresses: code
# from C_TEXT (c_main, which the dispatcher calls for a name of none of its own), read-only data from C_RODATA
TOOLS_C = Path(__file__).with_name("qwenputer_tools_c.s")
C_TEXT, C_RODATA = 0x48000, 0x22000

D_QWEN = 2560  # Qwen3.5-4B's hidden size: the computer's dims come after it
EPS = 1e-6
HEAD = 128  # key and value head size of the Gated DeltaNet layers
CONV = 4  # their convolution's width

# The trace vocabulary (chipformer/rv/spec.py): families of payload tokens, then 8 controls.
FAMILY_SIZES = [("V", 1 << 15), ("F0", 1 << 15), ("F1", 1 << 15), ("F2", 4), ("J", 1 << 16), ("R", 1 << 16),
                ("BYTE", 512), ("FLT", 8), ("HEX", 16), ("QS", 16), ("QD", 128), ("BD", 256)]
CONTROL_NAMES = ["NT", "ST0", "ST1", "SS0", "SS1", "HALT", "GO", "SC"]
OUT_NAMES = [n for n, _ in FAMILY_SIZES] + CONTROL_NAMES + ["Q"]  # the output code's family ids (rv/core.py)


def _families():
    out, base = {}, 0
    for n, s in FAMILY_SIZES:
        out[n] = (base, s)
        base += s
    return out, base


FAMILIES, CTRL = _families()
CHIP_VOCAB = CTRL + len(CONTROL_NAMES)


# -- the chip pack ------------------------------------------------------------------------------------------------
def convert(src, out) -> dict:
    """Make the chip pack from the Qwenputer checkpoint folder ``src``: ``out/chip.safetensors`` (the computer
    layers compacted, the embedding's computer part) and ``out/chip.json``. Reads one tensor at a time."""
    src, out = Path(src), Path(out)
    meta = json.loads((src / "chipformer.json").read_text())
    cfg = json.loads((src / "config.json").read_text())
    H = cfg["hidden_size"]
    n_chip, v0 = meta["chip_layers"], meta["chip_token_offset"]
    assert meta["layout"] == "rv1" and meta["chip_dims_offset"] == D_QWEN, meta
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    shards: Dict[str, dict] = {}

    def get(k) -> np.ndarray:
        f = index[k]
        if f not in shards:
            shards.clear()  # (one shard open at a time)
            shards[f] = mx.load(str(src / f))
        return np.array(shards[f][k].astype(mx.float32))

    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    kd, vd = nk * HEAD, nv * HEAD
    layers, used = [], set()
    for L in range(n_chip):
        p = f"model.layers.{L}."
        for n in ("input_layernorm", "post_attention_layernorm"):
            assert not get(p + n + ".weight").any(), "a computer layer's norm gain is not 1"
        a = {k: get(p + "linear_attn." + k + ".weight") for k in ("in_proj_qkv", "in_proj_z", "in_proj_b",
                                                                   "in_proj_a", "out_proj", "conv1d")}
        for k in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
            assert not a[k][:, :D_QWEN].any(), f"layer {L} {k} reads Qwen's dims"
        assert not a["out_proj"][:D_QWEN].any(), f"layer {L} writes Qwen's dims"
        heads = sorted(h for h in range(nv) if a["out_proj"][:, h * HEAD:(h + 1) * HEAD].any())
        layer = {"heads": heads}
        if heads:
            kh = sorted({h * nk // nv for h in heads})
            rows = ([k * HEAD + i for k in kh for i in range(HEAD)] + [kd + k * HEAD + i for k in kh for i in range(HEAD)]
                    + [2 * kd + h * HEAD + i for h in heads for i in range(HEAD)])
            vrows = [h * HEAD + i for h in heads for i in range(HEAD)]
            layer.update(
                key_heads=kh, kmap=[kh.index(h * nk // nv) for h in heads],
                qkv=_compact(a["in_proj_qkv"][rows]), z=_compact(a["in_proj_z"][vrows]),
                b=_compact(a["in_proj_b"][heads]), a=_compact(a["in_proj_a"][heads]),
                conv=a["conv1d"][rows, 0, :], out=_compact_out(a["out_proj"][:, vrows]),
                A_log=get(p + "linear_attn.A_log")[heads], dt_bias=get(p + "linear_attn.dt_bias")[heads],
                norm=get(p + "linear_attn.norm.weight"))
        g, u, d = (get(p + "mlp." + k + ".weight") for k in ("gate_proj", "up_proj", "down_proj"))
        assert not (g[:, :D_QWEN].any() or u[:, :D_QWEN].any() or d[:D_QWEN].any()), f"layer {L} MLP touches Qwen's dims"
        units = np.nonzero(g.any(1) & u.any(1) & d.any(0))[0]
        if len(units):
            layer.update(gate=_compact(g[units]), up=_compact(u[units]), down=_compact_out(d[:, units]))
        for k in ("qkv", "z", "b", "a", "gate", "up"):
            if k in layer:
                used.update(layer[k][1].tolist())
        for k in ("out", "down"):
            if k in layer:
                used.update(layer[k][1].tolist())
        layers.append(layer)

    # the embedding's computer part: text rows are one row plus 128 byte bits (but for 262 special tokens);
    # trace rows are, per family, a row plus one row per payload bit (but QS, QD and the controls)
    shards.clear()
    E = mx.load(str(src / index["model.embed_tokens.weight"]))["model.embed_tokens.weight"]
    cols = np.nonzero(np.array(mx.any(E[:, D_QWEN:] != 0, axis=0)))[0] + D_QWEN
    keep = meta["code_dims"] + [meta["tau_dim"], meta["mu_dim"]]
    used.update(cols.tolist())
    used.update(keep)
    dims = np.array(sorted(used), dtype=np.int32)
    pos = {int(d): i for i, d in enumerate(dims)}
    X = np.array(E[:, dims.tolist()].astype(mx.float32))  # (vocab, R)
    del E
    T, C = X[:v0], X[v0:v0 + CHIP_VOCAB]
    special = sorted(set(meta["printable"]) | set(meta["tool_call"]))
    plain_rows = np.ones(v0, bool)
    plain_rows[special] = False
    Tn = T[plain_rows]
    bitcols = np.nonzero((Tn != Tn[0]).any(0))[0]
    assert len(bitcols) == 128 and set(np.unique(Tn[:, bitcols]).tolist()) <= {0.0, 1.0}, len(bitcols)
    plain = Tn[0].copy()
    plain[bitcols] = 0
    bits = np.packbits(T[:, bitcols].astype(np.uint8) * plain_rows[:, None], axis=1)  # (v0, 16)
    check = np.broadcast_to(plain, Tn.shape).copy()
    check[:, bitcols] = Tn[:, bitcols]
    assert np.array_equal(check, Tn), "text rows are not one row plus byte bits"
    fam_base, fam_delta, explicit = [], [], {}
    for name, (base, size) in FAMILIES.items():
        rows = C[base:base + size]
        nb = (size - 1).bit_length()
        deltas = np.zeros((16, len(dims)), np.float32)
        for i in range(nb):
            deltas[i] = rows[1 << i] - rows[0]
        pred = rows[0][None] + sum(((np.arange(size) >> i) & 1)[:, None] * deltas[i][None] for i in range(nb))
        if not np.array_equal(pred, rows):
            for t in range(size):
                explicit[base + t] = rows[t]
        fam_base.append(rows[0])
        fam_delta.append(deltas)
    for t in range(CTRL, CHIP_VOCAB):
        explicit[t] = C[t]
    for name, x in [("plain", plain), ("fam_base", np.stack(fam_base)), ("fam_delta", np.stack(fam_delta)),
                    ("special_rows", T[special]), ("explicit_rows", np.stack([explicit[t] for t in sorted(explicit)]))]:
        assert np.array_equal(x.astype(np.float32), np.array(mx.array(x).astype(mx.bfloat16).astype(mx.float32))), name

    W: Dict[str, mx.array] = {}
    bf = lambda x: mx.array(x).astype(mx.bfloat16)  # noqa: E731
    i32 = lambda x: mx.array(np.asarray(x, dtype=np.int32))  # noqa: E731
    for L, layer in enumerate(layers):
        p = f"L{L}."
        for k in ("qkv", "z", "b", "a", "gate", "up"):
            if k in layer:
                w, cols_ = layer[k]
                W[p + k + ".w"], W[p + k + ".in"] = bf(w), i32([pos[int(c)] for c in cols_])
        for k in ("out", "down"):
            if k in layer:
                w, rows_ = layer[k]
                W[p + k + ".w"], W[p + k + ".to"] = bf(w), i32([pos[int(r)] for r in rows_])
        if layer["heads"]:
            W[p + "conv"] = bf(layer["conv"])
            W[p + "kmap"] = i32(layer["kmap"])
            W[p + "heads"] = i32(layer["heads"])  # (the checkpoint's value head of each: where its snapshot rows go)
            for k in ("A_log", "dt_bias", "norm"):
                W[p + k] = bf(layer[k])
    W["emb.plain"], W["emb.bitcols"], W["emb.bits"] = bf(plain), i32(bitcols), mx.array(bits)
    W["emb.special_ids"], W["emb.special_rows"] = i32(special), bf(T[special])
    W["emb.fam_base"], W["emb.fam_delta"] = bf(np.stack(fam_base)), bf(np.stack(fam_delta))
    W["emb.explicit_ids"] = i32(sorted(explicit))
    W["emb.explicit_rows"] = bf(np.stack([explicit[t] for t in sorted(explicit)]))
    W["dims"] = i32(dims)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "chip.safetensors"), W)
    info = {"format": 1, "source": str(src), "hidden": H, "layers": n_chip, "chip_token_offset": v0,
            "chip_vocab": meta["chip_vocab"], "code": [pos[d] for d in meta["code_dims"]], "tau": pos[meta["tau_dim"]],
            "mu": pos[meta["mu_dim"]], "nfam": meta["nfam"], "npay": meta["npay"], "tool_call": meta["tool_call"],
            "printable": meta["printable"], "eos": [248046, 248044], "dims": len(dims),
            "parameters": int(sum(v.size for k, v in W.items() if k.endswith(".w")))}
    (out / "chip.json").write_text(json.dumps(info, indent=1))
    return info



# -- the chip pack on this Mac -----------------------------------------------------------------------------------
def chipformer():
    """The authors' package if it can be imported (installed: ``INSTALL``; or a checkout's experimental/riscv folder
    named by LOCALS_CHIPFORMER), else None."""
    path = os.environ.get("LOCALS_CHIPFORMER")
    if path and path not in sys.path:
        sys.path.append(path)
    try:
        from chipformer.rv import bridge, core, elf, ref  # noqa: F401
    except ImportError:
        return None
    import chipformer

    return chipformer


def pack_dir() -> Path:
    return Path(os.environ.get("LOCALS_HOME", Path.home() / ".locals")) / "qwenputer" / REVISION[:12]


def ensure_pack(log=print) -> Path:
    """The chip pack, made on first use: the checkpoint (18.8 GB) converted, and its firmware loaded with the
    authors' package (``chipformer``). Afterwards the checkpoint is not needed (``hf cache delete`` frees it)."""
    d = pack_dir()
    if all((d / f).exists() for f in ("chip.safetensors", "chip.json", "snapshot.safetensors", "calc-bridge.elf")):
        return d
    if chipformer() is None:
        raise RuntimeError("Qwenputer's firmware is loaded by its authors' package, which is not installed: " + INSTALL)
    import hashlib
    import shutil
    import tempfile
    import urllib.request

    from huggingface_hub import snapshot_download

    log(f"locals: making Qwenputer's chip pack (once): downloading {REPO} (18.8 GB)")
    src = Path(snapshot_download(REPO, revision=REVISION))
    tmp = Path(tempfile.mkdtemp(prefix="pack-", dir=d.parent if d.parent.exists() else None))
    try:
        info = convert(src, tmp)
        firmware = tmp / "calc-bridge.elf"
        with urllib.request.urlopen(FIRMWARE_URL, timeout=60) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != FIRMWARE_SHA256:
            raise RuntimeError(f"the firmware at {FIRMWARE_URL} is not the one this pack was made for")
        firmware.write_bytes(data)
        add_snapshot(tmp, firmware_states(src, firmware))
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists():
            shutil.rmtree(d)
        tmp.rename(d)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    log(f"locals: Qwenputer's chip pack is in {d} ({info['parameters'] / 1e6:.0f}M weights); the checkpoint in the "
        "Hugging Face cache is no longer needed")
    return d


def toolbox(firmware):
    """The toolbox firmware (an asm.Program of the authors' package): calc's firmware (an ELF), the toolbox's code
    after its code and its read-only data after calc's, and calc's resume after its park turned into a jump to the
    toolbox's dispatcher (which goes on in calc's code, every register as calc parked, for a calc call)."""
    chipformer()
    from chipformer.rv import asm, elf
    from chipformer.rv import spec as S

    calc = elf.program(str(firmware))
    i = (RESUME - S.ROM_BASE) // 4
    if calc.text[i] != 0x000085B7 or calc.text[i - 1] != 0x73:  # (lui a1, 0x8 after the park's ecall)
        raise RuntimeError("calc's firmware is not the one the toolbox is linked to")
    code, rodata = TOOLS_SOURCE.read_text().replace("RESUME_NEXT", hex(RESUME + 4)).split("\n.rodata\n")
    pad = len(calc.rodata) + (-len(calc.rodata) % 4)
    n = len(calc.text)
    mine = asm.assemble(".text\n" + ".word 0\n" * n + code.replace(".text\n", "", 1) + f"\n.rodata\n.space {pad}\n" + rodata)
    text = list(calc.text) + mine.text[n:]
    text[i] = asm.enc_j(mine.labels["from_calc"] - RESUME, 0)
    ro = bytes(calc.rodata) + bytes(pad - len(calc.rodata)) + bytes(mine.rodata[pad:])
    words, data = [], bytearray()
    section = None
    for line in TOOLS_C.read_text().splitlines():
        if line in (".text", ".rodata"):
            section = line
        elif line.startswith(".word ") and section == ".text":
            words += [int(w, 16) for w in line[6:].split(",")]
        elif line.startswith(".byte ") and section == ".rodata":
            data += bytes(int(b) for b in line[6:].split(","))
    c_text, c_ro = (C_TEXT - S.ROM_BASE) // 4, C_RODATA - S.DROM_BASE
    if len(text) > c_text or len(ro) > c_ro:
        raise RuntimeError("the toolbox's assembly runs into its C")
    text += [0] * (c_text - len(text)) + words
    ro += bytes(c_ro - len(ro)) + bytes(data)
    return asm.Program(text, calc.data, {"_start": calc.entry}, data_base=calc.data_base, rodata=ro)


def tools_snapshot() -> str:
    """The toolbox's snapshot file in a chip pack (named for the toolbox's sources)."""
    import hashlib

    return f"snapshot-tools-{hashlib.sha256(TOOLS_SOURCE.read_bytes() + TOOLS_C.read_bytes()).hexdigest()[:8]}.safetensors"


def ensure_tools(pack, log=print) -> bool:
    """The toolbox's snapshot in the chip pack, made once (with the authors' package, from calc's firmware there and
    the checkpoint's config and tokenizer); False if it cannot be made here."""
    pack = Path(pack)
    if (pack / tools_snapshot()).exists():
        return True
    if chipformer() is None or not (pack / "calc-bridge.elf").exists():
        return False
    try:
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=["config.json", "tokenizer.json"]))
        add_snapshot(pack, firmware_states(src, toolbox(pack / "calc-bridge.elf")), tools_snapshot())
        for old in pack.glob("snapshot-tools-*.safetensors"):  # (an earlier toolbox's, 13 MB, unused for a week)
            if old.name != tools_snapshot() and time.time() - old.stat().st_mtime > 7 * 86400:
                old.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        log(f"locals: Qwenputer's toolbox could not be loaded ({e!r}); its computer runs calc only")
        return False
    log("locals: Qwenputer's computer has its toolbox: " + ", ".join(TOOL_NAMES))
    return True


def firmware_states(src, firmware) -> dict:
    """The parked firmware as the computer layers' recurrent states, with the authors' package: their circuit
    (core.build), machine (ref.Machine, parked at the firmware's first exit) and snapshot (core.snapshot), laid out
    as their rv/hf.py inject writes the checkpoint's cache: "L{n}" -> (value heads, key rows, value columns)."""
    chipformer()
    from chipformer.q35.ir import Head, Target
    from chipformer.rv import bridge, core, elf, ref

    cfg = json.loads((Path(src) / "config.json").read_text())
    c = core.build(Target(cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], cfg["intermediate_size"]),
                   bridge.Text(Path(src) / "tokenizer.json"))
    m = ref.Machine.from_program(firmware if hasattr(firmware, "text") else elf.program(str(firmware)))
    bridge.park(m)
    core.snapshot(c, m)
    r = c.target.value_heads // c.target.key_heads
    out = {}
    for L in range(c.n_layers()):
        st = np.zeros((cfg["linear_num_value_heads"], HEAD, HEAD), np.float32)
        heads = [i for i, n in enumerate(c.nodes) if isinstance(n, Head) and n.level == 2 * L + 1]
        for j, ni in enumerate(heads):
            init = c.nodes[ni].init or []
            for x in range(len(c.nodes[ni].vheads())):
                for row, vec in (init[x] if x < len(init) else {}).items():
                    st[j * r + x, row] = np.asarray(vec, np.float64)
        out[f"L{L}"] = st
    return out


def is_qwenputer(model_id: str) -> bool:
    from ..defaults import is_qwenputer

    return is_qwenputer(model_id)


def add_snapshot(pack, states, file: str = "snapshot.safetensors") -> None:
    """Write the firmware into a chip pack: ``states`` maps "L{n}" to the checkpoint's recurrent state of computer
    layer n after loading the parked firmware, (value heads, key rows, value columns) as the authors' rv/hf.py
    inject writes it (their core.snapshot of calc-bridge.elf). Kept per compacted head, in MLX's layout (value
    rows, key columns)."""
    pack = Path(pack)
    W = mx.load(str(pack / "chip.safetensors"))
    out = {}
    for name, st in states.items():
        st = np.asarray(st, np.float32)
        heads = np.array(W[f"{name}.heads"]) if f"{name}.heads" in W else np.zeros(0, np.int64)
        other = np.setdiff1d(np.arange(st.shape[0]), heads)
        assert not st[other].any(), f"{name}: the snapshot fills a head the computer does not use"
        if len(heads):
            out[name] = mx.array(np.ascontiguousarray(st[heads].transpose(0, 2, 1)))
    tmp = pack / f".{file}.{os.getpid()}.safetensors"  # (written whole, then named: a server starting next to another,
    mx.save_safetensors(str(tmp), out)                # or a write cut short, never leaves a part of one)
    os.replace(tmp, pack / file)


def _compact(w: np.ndarray):
    """(rows, hidden) -> (rows, used columns), the used columns' hidden indices."""
    cols = np.nonzero(w.any(0))[0]
    return w[:, cols], cols


def _compact_out(w: np.ndarray):
    """(hidden, k) -> (written rows, k), the written rows' hidden indices."""
    rows = np.nonzero(w.any(1))[0]
    return w[rows], rows


# -- the computer ---------------------------------------------------------------------------------------------------
f32, bf16 = mx.float32, mx.bfloat16


def _silu(x: mx.array) -> mx.array:  # torch's: x / (1 + exp(-x)), in fp32
    return x / (1 + mx.exp(-x))


def _softplus(x: mx.array) -> mx.array:  # torch's (threshold 20)
    return mx.where(x > 20, x, mx.log1p(mx.exp(mx.minimum(x, 20))))


def _rms(x: mx.array, n: int) -> mx.array:
    """Qwen3_5RMSNorm over the full hidden size n with gain 1 (the computer layers' norms): x is the residual's
    computer dims; Qwen's dims are below fp32's resolution of the ballast's 62 * 2^80."""
    xf = x.astype(f32)
    return (xf * mx.rsqrt((xf * xf).sum(-1, keepdims=True) / n + EPS)).astype(bf16)


class ChipState:
    """One batch's computer state: per layer with heads, the convolution's last 3 inputs (B, 3, C) and the
    recurrent state (B, heads, 128, 128) in fp32 (mlx-lm's layout: value rows, key columns)."""

    def __init__(self, conv: List[Optional[mx.array]], rec: List[Optional[mx.array]]):
        self.conv, self.rec = conv, rec


class Chip:
    def __init__(self, path, tools: bool = False):
        """A chip pack's computer; ``tools``: its firmware the toolbox (``ensure_tools``), if the pack has it, else
        calc's alone (``self.tools``: the tools the firmware runs besides calc)."""
        path = Path(path)
        self.info = json.loads((path / "chip.json").read_text())
        self.W = mx.load(str(path / "chip.safetensors"))
        self.n_layers, self.v0, self.H = self.info["layers"], self.info["chip_token_offset"], self.info["hidden"]
        self.code = mx.array(self.info["code"])
        self.nfam, self.npay = self.info["nfam"], self.info["npay"]
        self.printable = self.info["printable"]
        snap = path / "snapshot.safetensors"
        self.tools: tuple = ()
        if tools and (path / tools_snapshot()).exists():
            snap, self.tools = path / tools_snapshot(), TOOL_NAMES
        self.snapshot = mx.load(str(snap)) if snap.exists() else None
        e = {k[4:]: np.array(v.astype(f32)) if v.dtype == bf16 else np.array(v) for k, v in self.W.items() if k.startswith("emb.")}
        self._emb = e
        self._special = {int(t): i for i, t in enumerate(e["special_ids"])}
        self._explicit = {int(t): i for i, t in enumerate(e["explicit_ids"])}
        self._fam_of = np.zeros(CHIP_VOCAB, np.int32)
        for i, (base, size) in enumerate(FAMILIES.values()):
            self._fam_of[base:base + size] = i
        self._fam_base = np.array([b for b, _ in FAMILIES.values()])
        self._cache: Dict[int, np.ndarray] = {}
        self._step = None

    # embedding rows (the computer's dims), on the host
    def _row(self, t: int) -> np.ndarray:
        e = self._emb
        if t < self.v0:
            i = self._special.get(t)
            if i is not None:
                return e["special_rows"][i]
            r = e["plain"].copy()
            r[e["bitcols"]] = np.unpackbits(e["bits"][t])[:len(e["bitcols"])]
            return r
        c = t - self.v0
        i = self._explicit.get(c)
        if i is not None:
            return e["explicit_rows"][i]
        f = self._fam_of[c]
        p = c - self._fam_base[f]
        bits = (p >> np.arange(16)) & 1
        return e["fam_base"][f] + bits.astype(np.float32) @ e["fam_delta"][f]

    def embed(self, ids: List[List[int]]) -> mx.array:
        rows = []
        for seq in ids:
            for t in seq:
                r = self._cache.get(t)
                if r is None:
                    r = self._row(t)
                    if len(self._cache) < 1 << 16:
                        self._cache[t] = r
                rows.append(r)
        x = np.stack(rows).reshape(len(ids), len(ids[0]), -1)
        return mx.array(x).astype(bf16)

    def new_state(self, B: int) -> ChipState:
        conv, rec = [], []
        for L in range(self.n_layers):
            w = self.W.get(f"L{L}.conv")
            if w is None:
                conv.append(None)
                rec.append(None)
                continue
            nh = self.W[f"L{L}.kmap"].shape[0]
            conv.append(mx.zeros((B, CONV - 1, w.shape[0]), bf16))
            rec.append(mx.zeros((B, nh, HEAD, HEAD), f32))
        return ChipState(conv, rec)

    def inject(self, st: ChipState, rows: Optional[List[int]] = None) -> None:
        """The firmware: every computer head's recurrent state set to the snapshot (rows: those of the batch)."""
        if self.snapshot is None:
            raise RuntimeError("this chip pack has no firmware snapshot")
        for L in range(self.n_layers):
            if st.rec[L] is None:
                continue
            s = self.snapshot[f"L{L}"]  # (heads, 128 value rows, 128 key columns)
            B = st.rec[L].shape[0]
            if rows is None or len(rows) == B:
                st.rec[L] = mx.array(np.broadcast_to(np.array(s)[None], (B,) + s.shape))
            else:
                m = mx.array([i in set(rows) for i in range(B)])[:, None, None, None]
                st.rec[L] = mx.where(m, s[None], st.rec[L])

    def __call__(self, x: mx.array, st: ChipState) -> mx.array:
        """The computer layers on the residual's computer dims x (B, T, R) bf16; returns the residual after them."""
        if x.shape[1] == 1:  # a decoding step: one compiled graph
            if self._step is None:
                self._step = mx.compile(self._forward)
            x, st.conv, st.rec = self._step(x, st.conv, st.rec)
            return x
        x, st.conv, st.rec = self._forward(x, st.conv, st.rec)
        return x

    def _forward(self, x, conv, rec):
        st = ChipState(list(conv), list(rec))
        for L in range(self.n_layers):
            x = self._attn(L, x, st)
            x = self._mlp(L, x)
        return x, st.conv, st.rec

    def _lin(self, L: int, k: str, h: mx.array) -> mx.array:
        return h[..., self.W[f"L{L}.{k}.in"]] @ self.W[f"L{L}.{k}.w"].T

    def _add(self, L: int, k: str, x: mx.array, y: mx.array) -> mx.array:
        to = self.W[f"L{L}.{k}.to"]
        idx = mx.broadcast_to(to, x.shape[:-1] + to.shape)
        return mx.put_along_axis(x, idx, x[..., to] + y, axis=-1)

    def _attn(self, L: int, x: mx.array, st: ChipState) -> mx.array:
        W = self.W
        if f"L{L}.conv" not in W:
            return x
        from mlx_lm.models.gated_delta import gated_delta_kernel

        B, T, _ = x.shape
        h = _rms(x, self.H)
        qkv = self._lin(L, "qkv", h)  # (B, T, C) bf16
        z = self._lin(L, "z", h) if f"L{L}.z.w" in W else mx.zeros((B, T, W[f"L{L}.kmap"].shape[0] * HEAD), bf16)
        b = self._lin(L, "b", h) if f"L{L}.b.w" in W else mx.zeros((B, T, W[f"L{L}.kmap"].shape[0]), bf16)
        a = self._lin(L, "a", h) if f"L{L}.a.w" in W else mx.zeros((B, T, W[f"L{L}.kmap"].shape[0]), bf16)
        seq = mx.concatenate([st.conv[L], qkv], axis=1)  # (B, 3 + T, C)
        st.conv[L] = seq[:, -(CONV - 1):]
        w = W[f"L{L}.conv"].astype(f32)  # (C, 4)
        sf = seq.astype(f32)
        conv = sum(sf[:, j:j + T] * w[:, j] for j in range(CONV)).astype(bf16)
        conv = _silu(conv.astype(f32)).astype(bf16)
        kmap = W[f"L{L}.kmap"]
        nh = kmap.shape[0]
        nk = (conv.shape[-1] - nh * HEAD) // (2 * HEAD)
        q = conv[..., :nk * HEAD].reshape(B, T, nk, HEAD)
        k = conv[..., nk * HEAD:2 * nk * HEAD].reshape(B, T, nk, HEAD)
        v = conv[..., 2 * nk * HEAD:].reshape(B, T, nh, HEAD).astype(f32)
        q, k = q[:, :, kmap].astype(f32), k[:, :, kmap].astype(f32)
        q = q * mx.rsqrt((q * q).sum(-1, keepdims=True) + 1e-6)
        k = k * mx.rsqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        q = q / mx.array(HEAD ** 0.5, f32)
        beta = mx.sigmoid(b.astype(f32)).astype(bf16).astype(f32)
        g = -mx.exp(W[f"L{L}.A_log"].astype(f32)) * _softplus(a.astype(f32) + W[f"L{L}.dt_bias"].astype(f32))
        y, st.rec[L] = gated_delta_kernel(q, k, v, mx.exp(g), beta, st.rec[L], None)
        y = y.astype(bf16).astype(f32)  # (B, T, nh, 128)
        n = (y * mx.rsqrt((y * y).mean(-1, keepdims=True) + EPS)).astype(bf16)
        n = W[f"L{L}.norm"] * n
        n = (n.astype(f32) * _silu(z.reshape(B, T, nh, HEAD).astype(f32))).astype(bf16)
        return self._add(L, "out", x, n.reshape(B, T, nh * HEAD) @ W[f"L{L}.out.w"].T)

    def _mlp(self, L: int, x: mx.array) -> mx.array:
        if f"L{L}.gate.w" not in self.W:
            return x
        h = _rms(x, self.H)
        g = _silu(self._lin(L, "gate", h).astype(f32)).astype(bf16)
        return self._add(L, "down", x, (g * self._lin(L, "up", h)) @ self.W[f"L{L}.down.w"].T)

    # the output code: the token the computer predicts next, per position (-1 where Qwen predicts)
    def predict(self, h: mx.array) -> np.ndarray:
        code = np.array(h[..., self.code].astype(f32))  # (..., nfam + npay)
        set_ = (code != 0).any(-1)
        bits = (code > 0).astype(np.int64)
        fid = (bits[..., :self.nfam] << np.arange(self.nfam)).sum(-1)
        pay = (bits[..., self.nfam:] << np.arange(self.npay)).sum(-1)
        out = np.full(code.shape[:-1], -1, np.int64)
        for i in zip(*np.nonzero(set_)):
            out[i] = self.decode(int(fid[i]), int(pay[i]))
        return out

    def decode(self, fid: int, p: int) -> int:
        """A model token id from an output code (rv/core.py decode_code); -2 if the code names no token."""
        if not 0 <= fid < len(OUT_NAMES):
            return -2
        name = OUT_NAMES[fid]
        if name == "Q":
            return self.printable[p] if p < len(self.printable) else -2
        if name in CONTROL_NAMES:
            return self.v0 + CTRL + CONTROL_NAMES.index(name)
        base, size = FAMILIES[name]
        return self.v0 + base + p if p < size else -2

    def name(self, t: int) -> str:
        c = t - self.v0
        if c >= CTRL:
            return CONTROL_NAMES[c - CTRL]
        f = int(self._fam_of[c])
        return f"{FAMILY_SIZES[f][0]}({c - self._fam_base[f]})"


# -- decoding ---------------------------------------------------------------------------------------------------------
CALC_TOOL = {"type": "function", "function": {
    "name": "calc",
    "description": "Evaluate an arithmetic expression exactly, with numbers of up to 62 digits: + - * / and parentheses "
                   "on integers and fractions (exact: the result is an integer or a reduced fraction p/q), + - * on "
                   "decimals, and on integers also // (floor division), %, ** and gcd(a,b). Returns the exact value.",
    "parameters": {"type": "object", "properties": {
        "expression": {"type": "string", "description": "e.g. 123456789*987654321-42 or 1.25*-3.5 or 84/18"}},
        "required": ["expression"]}}}  # (the authors' rv/chat.py)


def _tool(name: str, description: str, optional=(), **params) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {k: {"type": "string", "description": d} for k, d in params.items()},
        "required": [k for k in params if k not in optional]}}}


# The toolbox's tools (qwenputer_tools.s): what a model miscounts or misjudges, done exactly by its computer
TOOL_SPECS = [
    _tool("count", "Count how many times a letter or text occurs in a text.", text="the text", find="the letter or text to count"),
    _tool("length", "Count the characters of a text.", text="the text"),
    _tool("reverse", "Write a text backwards.", text="the text"),
    _tool("sort", "Sort numbers in ascending order.", numbers="the numbers, separated by commas"),
    _tool("weekday", "The day of the week of a date.", date="YYYY-MM-DD"),
    _tool("days_between", "The number of days from one date to another.", start="YYYY-MM-DD", end="YYYY-MM-DD"),
    _tool("add_days", "The date a number of days after a date (before it, if the number is negative).", date="YYYY-MM-DD",
          days="the number of days, e.g. 100 or -30"),
    _tool("words", "Count the words of a text.", text="the text"),
    _tool("to_base", "Write a whole number in another base: binary is 2, octal 8, hexadecimal 16 (2 to 36). The number "
          "may be decimal, or start with 0x, 0b or 0o.", number="the number, e.g. 255 or 0xFF", base="the base, e.g. 2 or 16"),
    _tool("count_letters", "Count the characters of a text that are any of the given letters, e.g. its vowels (aeiou).",
          text="the text", letters="the letters, e.g. aeiou"),
    # (in C: qwenputer_c/)
    _tool("json_check", "Check whether a text is valid JSON: what kind of value it is, or where and why it is not.",
          text="the whole JSON, as written"),
    _tool("json_format", "Write JSON indented (pretty) or on one line (compact).", text="the whole JSON, as written",
          style="pretty or compact", optional=("style",)),
    _tool("json_get", "The value at a path in JSON: users[0].name, or users[*].email for every user's.",
          text="the whole JSON, as written", path="the path, e.g. users[0].name"),
    _tool("json_count", "How many items an array, or keys an object (and which), has at a path in JSON.",
          text="the whole JSON, as written", path="the path (none: the whole JSON)", optional=("path",)),
    _tool("column_stats", "The count, sum, average, median, minimum and maximum of a table's column (a CSV, tab-separated "
          "or Markdown table).", table="the whole table, every row as written", column="the column's name or number"),
    _tool("table_sort", "A table's rows sorted by a column.", table="the whole table, every row as written",
          column="the column's name or number", order="ascending or descending", optional=("order",)),
    _tool("table_filter", "The rows of a table whose column meets a condition.",
          table="the whole table, every row as written", column="the column's name or number",
          condition="e.g. > 100, = Paris or contains York"),
    _tool("table_totals", "A table's rows grouped by a column, with another column added up for each group.",
          table="the whole table, every row as written", group_by="the column to group by",
          column="the column to add up (none: count the rows)", optional=("column",)),
    _tool("compare_versions", "Which of two version numbers is newer (1.10.0 is newer than 1.9.3).", first="a version",
          second="another version"),
    _tool("sort_versions", "Version numbers sorted from oldest to newest.", versions="the versions, separated by commas",
          order="ascending or descending", optional=("order",)),
    _tool("compare_lists", "Which items two lists have in common, and which only one of them has (letter case ignored).",
          first="a list, its items separated by commas or one a line", second="another list"),
    _tool("truth_table", "A logic expression's truth table, and whether it is a tautology, satisfiable or a "
          "contradiction.", expression="e.g. (p and q) -> r, with not, and, or, xor, ->, <->"),
    _tool("logic_equivalent", "Whether two logic expressions are equivalent, with a counterexample if not.",
          first="an expression, e.g. not (p and q)", second="another expression"),
    _tool("regex_find", "The matches of a regular expression (Python's syntax, as re.findall: ^ and $ are the whole "
          "text's start and end; \\b a word's) in a text, and how many.", pattern="the regular expression",
          text="the text"),
    _tool("regex_replace", "A text with the matches of a regular expression replaced (\\1 for its first group).",
          pattern="the regular expression", text="the text", replacement="what to put in each match's place"),
    _tool("remember", "Keep a note for later in this conversation.", key="the note's name", value="what to remember"),
    _tool("recall", "A note kept earlier (key all: every note).", key="the note's name"),
    _tool("forget", "Drop a note.", key="the note's name"),
    _tool("stats", "Statistics of a list of numbers: count, sum, mean, median, mode, min, max, range and standard deviation.",
          numbers="the numbers, separated by commas or one a line"),
    _tool("percent_change", "The percentage change from one number to another.", old="the number before", new="the number after"),
    _tool("round_number", "A number rounded half up (as money is): to some decimals, or to tens, hundreds (negative places).",
          number="the number", places="decimals to keep, e.g. 2 (-2: to hundreds)"),
    _tool("compound_interest", "The amount and interest of a sum after compound interest.", principal="the sum",
          rate="the yearly rate in percent, e.g. 5", years="how many years",
          compounding="yearly, quarterly, monthly or daily", optional=("compounding",)),
    _tool("loan_payment", "A loan's (or mortgage's) monthly payment, its total and its interest.", principal="the sum lent",
          rate="the yearly rate in percent, e.g. 6.5", years="how many years"),
    _tool("is_prime", "Whether a whole number (up to 19 digits) is prime.", number="the number"),
    _tool("factor", "A whole number's prime factors (up to 18 digits; its factors found up to about 20,000, or "
          "100,000 for numbers below 4,294,967,296).", number="the number"),
    _tool("convert_units", "Convert a value between units of length, mass, volume, area, time, speed, data, energy, "
          "pressure or temperature.", value="the number", **{"from": "its unit, e.g. km, lb, °F, cups, GiB",
                                                          "to": "the unit wanted"}),
    _tool("roman_numeral", "A number as a roman numeral, or a roman numeral as a number (1 to 3999).",
          value="a number or a roman numeral"),
    _tool("number_to_words", "A number in English words.", number="the number, e.g. 1234567 or -4.5"),
    _tool("replace_text", "Replace every occurrence of a text in a text (as written, letter case too).", text="the text",
          find="what to replace", replace="what to put in its place"),
    _tool("change_case", "A text in another case.", text="the text",
          case="upper, lower, title, sentence, snake, camel, kebab, pascal or constant"),
    _tool("text_part", "A word, line, character or sentence of a text by its number (from 1; -1: the last; 2-5: a range).",
          text="the text", part="word, line, character or sentence", number="e.g. 3, -1 or 2-5"),
    _tool("sort_lines", "Lines (or comma-separated items) in alphabetical order.", text="the lines or items",
          order="ascending or descending", optional=("order",)),
    _tool("unique_items", "Lines (or comma-separated items) without their repeats (letter case ignored).", text="the lines or items"),
    _tool("word_frequency", "A text's most frequent words, with their counts.", text="the text", top="how many, e.g. 5",
          optional=("top",)),
    _tool("is_palindrome", "Whether a text reads the same backwards (letter case, spaces and punctuation ignored).", text="the text"),
    _tool("is_anagram", "Whether two texts have the same letters, as often.", first="a text", second="another text"),
    _tool("validate", "Whether a text is a valid email address, IPv4 address, URL, UUID, date, time or hex color.",
          text="the text", kind="email, ipv4, url, uuid, date, time or hex color"),
    _tool("encode", "A text in base64, hex, URL encoding, ROT13, Caesar, Morse or binary.", text="the text",
          encoding="base64, hex, url, rot13, caesar, morse or binary"),
    _tool("decode", "A text decoded from base64, hex, URL encoding, ROT13, Caesar, Morse or binary.", text="the encoded text",
          encoding="base64, hex, url, rot13, caesar, morse or binary"),
    _tool("edit_distance", "The edit (Levenshtein) distance between two texts of up to 80 characters.", first="a text",
          second="another text"),
    _tool("time_add", "A time (or date and time) after an amount of time.", time="e.g. 14:30, 2:30 pm or 2024-03-10 14:30",
          amount="e.g. 2h 45m, 90 minutes, -1 day"),
    _tool("time_between", "The time from one time (or date and time) to another.", start="e.g. 09:15 or 2024-03-10 09:15",
          end="the later one"),
    _tool("age", "Someone's age (or the time since a date) in years, months and days.", birth_date="YYYY-MM-DD",
          on_date="YYYY-MM-DD: the date to count to (today's)"),
    _tool("date_info", "A date's day of the week, day of the year, ISO week, quarter and whether its year is a leap year.",
          date="YYYY-MM-DD"),
    _tool("unix_time", "A Unix timestamp as a date and time (UTC), or a date and time as a Unix timestamp.",
          value="a timestamp, e.g. 1700000000, or YYYY-MM-DD HH:MM"),
    _tool("convert_timezone", "A time in another time zone (zones as offsets, e.g. UTC+5:30, or names: EST, CEST, JST).",
          time="e.g. 15:00 or 2024-03-10 15:00", **{"from": "its zone", "to": "the zone wanted"}),
    _tool("check_digit", "Whether a card number (Luhn), ISBN, EAN, UPC or IBAN is valid, and its check digit (or the one to "
          "append to a number without one).",
          number="the number", kind="luhn, isbn10, isbn13, ean13, upc or iban"),
    _tool("hash_text", "A text's SHA-256, CRC-32 or Adler-32 hash.", text="the text", algorithm="sha256, crc32 or adler32"),
    _tool("run_python", "Run a short Python program on your own exact computer and get what it prints. Whole numbers "
          "(to 2^62), strings, lists, dicts, loops, functions, math.gcd / isqrt / factorial / comb. No floats: use // "
          "and math.isqrt. Small programs only: about 1,250 steps (a loop of about 250 rounds).",
          code="the program; print the answer"),
]
TOOL_NAMES = tuple(t["function"]["name"] for t in TOOL_SPECS)
# Which of them a request is offered, by what its last user message asks: offered always, any extra tool changed how
# Qwen answered arithmetic (59 of 60 questions right with calc alone, 54 to 58 with one to six more: it went on to
# check calc's result by hand), so a question that asks for none of them gets the prompt it gets with calc alone
_MONTH = r"(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|june?|july?|aug(ust)?|sept?(ember)?|oct(ober)?|nov(ember)?|dec(ember)?)"
TOOL_GROUPS = [
    (("count", "length", "reverse", "words", "count_letters"),
     r"\b(letters?|characters?|backwards?|reversed?|palindromes?|spell(ed|ing)?|occurrences?|vowels?|consonants?|words?|"
     r"count|how many times\b.*\b(appears?|occurs?)|how many ['\"‘“]?\w['\"’”]?['’]?s)\b"),
    (("sort",), r"\b(sort(ed|ing)?|ascending|descending|increasing|decreasing|smallest to largest|largest to smallest|"
                r"in order(?! to))\b"),
    # (a date, or a day of the week: "how many days" alone is as often arithmetic, "how many days until I have $100?")
    (("weekday", "days_between", "add_days"), r"\b\d{4}-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b|\b(day of the week|weekday)\b|"
                                               rf"\b{_MONTH}\.? \d{{1,2}}(st|nd|rd|th)?\b|\b\d{{1,2}}(st|nd|rd|th)? (of )?{_MONTH}\b"),
    (("to_base",), r"\b(binary|hexadecimal|hex|octal|base[ -]?\d+|in base|0x[0-9a-f]+|0b[01]+|0o[0-7]+)\b"),
    (("json_check", "json_format", "json_get", "json_count"), r"\bjson\b|^\s*[\[{]\s*\"|\{\s*\"[^\"\n]+\"\s*:"),
    (("column_stats", "table_sort", "table_filter", "table_totals"),
     r"\b(csv|tsv|spreadsheet|columns?|(?<!truth )tables?)\b|\|[^\n]*\|\s*\n\s*\|?\s*:?-{3,}"),
    (("compare_versions", "sort_versions"), r"\bversions?\b|\bv?\d+\.\d+\.\d+"),
    (("compare_lists",), r"\b(in common|in both|common to both|both lists|on both lists|lists?\b.*\b(missing|only|differ|"
                         r"compare|not (in|on)))\b"),
    (("truth_table", "logic_equivalent"), r"\b(truth tables?|tautology|tautologies|contradiction|satisfiable|logically|"
                                          r"equivalent|boolean|propositional|de morgan)\b|[∧∨¬↔⊕]|<->|\w\s*->\s*\w"),
    (("regex_find", "regex_replace"), r"\b(regex(es|ps)?|regular expressions?|reg ?exp?s?)\b"),
    (("remember", "recall", "forget"), r"\b(remember|recall|forget|notes?|note (this|that|down)|what did i (say|tell))\b"),
    (("stats",), r"\b(average|mean|median|mode|standard deviation|std ?dev|variance|statistics)\b"),
    (("percent_change", "round_number", "compound_interest", "loan_payment"),
     r"\b(percent(age)? (change|increase|decrease|difference)|(increase|decrease|rise|drop|grow|fall)[sd]? by|round(ed|ing)? "
     r"(to|off|up|down)|compound(ed|ing)?|interest|loan|mortgage|monthly payments?)\b"),
    (("is_prime", "factor"), r"\b(primes?|factors?|factori[sz](e|ed|ation)|divisors?|divisible)\b"),
    (("convert_units",), r"\b(celsius|fahrenheit|kelvin|kilomet(er|re)s?|km|miles?|feet|foot|inch(es)?|yards?|"
                         r"pounds?|lbs?|kilograms?|kg|ounces?|grams?|lit(er|re)s?|gallons?|cups?|tablespoons?|teaspoons?|mph|"
                         r"km/h|knots?|acres?|hectares?|(giga|mega|kilo|tera)bytes?|[GMKT]i?B|joules?|calories?|kwh|psi)\b"),
    (("roman_numeral", "number_to_words"), r"\b(roman numerals?|roman|in words|spell(ed)? out|written out|[MDCLXVI]{4,})\b"),
    (("replace_text", "change_case", "text_part", "sort_lines", "unique_items", "word_frequency", "is_palindrome",
      "is_anagram", "edit_distance"),
     r"\b(replace|substitute|upper ?case|lower ?case|title case|snake[ _]?case|camel ?case|kebab|capitali[sz]e|"
     r"(first|last|second|third|fourth|fifth|\d+(st|nd|rd|th)) (word|line|letter|character|sentence)|"
     r"(word|line|letter|character|sentence) (number |#)?\d+|alphabetical(ly)?|"
     r"duplicates?|unique|most (common|frequent)|frequen(cy|cies)|palindromes?|anagrams?|edit distance|levenshtein)\b"),
    (("validate", "encode", "decode"), r"\b(valid|validate|base ?64|url[- ]?(en|de)cod\w*|percent[- ]encod\w*|rot ?13|caesar|"
                                       r"morse|encod(e|ed|ing)|decod(e|ed|ing)|email address|ip address|uuid|hex colou?r)\b"),
    (("time_add", "time_between", "age", "date_info", "unix_time", "convert_timezone"),
     r"\b\d{1,2}:\d{2}\b|\b(\d{1,2} ?(am|pm)|time ?zones?|utc|gmt|est|edt|pst|pdt|cet|cest|jst|unix|timestamp|epoch|how old|"
     r"age|born|iso week|week number|day of the year|leap year)\b"),
    (("check_digit", "hash_text"), r"\b(luhn|credit card|card number|isbn|ean|upc|barcode|iban|check digit|checksum|imei|"
                                   r"sha-?256|crc-?32|crc|adler|hash)\b"),
    (("run_python",), r"\b(python|programs?|code|script|algorithm|simulat(e|ion)|primes?|fibonacci|collatz|divisible|"
                      r"perfect (numbers?|squares?)|palindromic numbers?|sum of (all )?(the )?(digits|numbers|integers|"
                      r"multiples|squares|cubes|primes|even|odd)|multiples of|how many (numbers|integers|ways|pairs|triples))\b"),
]


def tools_for(text: str, have=TOOL_NAMES) -> tuple:
    """The toolbox's tools (of ``have``) a request whose last user message is ``text`` is offered."""
    import re

    out = []
    for names, pattern in TOOL_GROUPS:
        if re.search(pattern, text or "", re.IGNORECASE):
            out += [n for n in names if n in have]
    return tuple(out)


def generate(model, chip: Chip, prompt: List[int], max_tokens: int = 1000, max_chip: int = 60000, on_token=None,
             prefill_step: int = 2048) -> dict:
    """Greedy decoding of one prompt (the authors' RvChat.generate): ``model`` is Qwen3.5-4B (mlx-lm), ``chip``
    the computer. Returns the tokens (trace tokens included: ids >= chip.v0) and counts."""
    from mlx_lm.models.cache import make_prompt_cache

    import time

    cache = make_prompt_cache(model)
    st = chip.new_state(1)
    for i in range(0, len(prompt), prefill_step):
        part = prompt[i:i + prefill_step]
        logits = model(mx.array([part]), cache=cache)[:, -1]
        h = chip(chip.embed([part]), st)
        mx.eval(logits, h, st.conv, st.rec)
    chip.inject(st)
    v0 = chip.v0
    nxt = int(mx.argmax(logits[0, :v0]))  # the prompt's last position: Qwen predicts
    out, n_text, n_chip = [], 0, 0
    t0, t_chip = time.perf_counter(), 0.0
    while n_text < max_tokens and n_chip < max_chip:
        out.append(nxt)
        if on_token is not None:
            on_token(nxt)
        if nxt in chip.info["eos"]:
            break
        t1 = time.perf_counter()
        h = chip(chip.embed([[nxt]]), st)
        own = int(chip.predict(h)[0, -1])
        if nxt >= v0:  # a trace token: the computer layers only
            n_chip += 1
            t_chip += time.perf_counter() - t1
            if own < 0:
                raise RuntimeError(f"the computer gave up a position after {chip.name(nxt)}")
            nxt = own
            continue
        n_text += 1
        logits = model(mx.array([[nxt]]), cache=cache)[:, -1]
        nxt = own if own >= 0 else int(mx.argmax(logits[0, :v0]))
    return {"tokens": out, "text_tokens": n_text, "chip_tokens": n_chip, "seconds": time.perf_counter() - t0,
            "chip_seconds": t_chip}


# -- the decoding step in a few Metal kernels ---------------------------------------------------------------------
# One token through the computer layers is ~470 small MLX kernels (7 ms on an M2 Pro: launches, not weights). These
# do each sublayer in two or three: the norm and every input projection of the sublayer (with the convolution and its
# SiLU) in one, the recurrence and the gated norm in one, the output projection and the residual add in one. The
# weights are the same, as sparse rows (1.05M entries); every rounding to bf16 is where transformers rounds (the
# MLX path above, which equals transformers' on every value: tests/test_qwenputer.py). Each norm divides by the
# ballast's 2^37 whatever the order of its sum: outside the ballast the residual stays below 2^8.

_PROJ = """
uint r = thread_position_in_grid.x, b = thread_position_in_grid.y;
uint tid = thread_position_in_threadgroup.x, tg = threads_per_threadgroup.x;
threadgroup float part[256];
float s = 0.0f;
for (uint i = tid; i < R; i += tg) { float v = static_cast<float>(x[b * R + i]); s += v * v; }
part[tid] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint w = tg / 2; w > 0; w >>= 1) {
    if (tid < w) part[tid] += part[tid + w];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
float inv = metal::precise::rsqrt(part[0] / float(H) + 1e-6f);
if (r >= N) return;
float acc = 0.0f;
for (int j = ptr[r]; j < ptr[r + 1]; j++) {
    float h = static_cast<float>(static_cast<bfloat16_t>(static_cast<float>(x[b * R + col[j]]) * inv));
    acc += val[j] * h;
}
bfloat16_t y = static_cast<bfloat16_t>(acc);
if (CONVR > 0 && r < CONVR) {
    uint o = b * 3 * CONVR;
    bfloat16_t s0 = cs[o + r], s1 = cs[o + CONVR + r], s2 = cs[o + 2 * CONVR + r];
    float c = 0.0f;
    c = c + static_cast<float>(s0) * cw[4 * r];
    c = c + static_cast<float>(s1) * cw[4 * r + 1];
    c = c + static_cast<float>(s2) * cw[4 * r + 2];
    c = c + static_cast<float>(y) * cw[4 * r + 3];
    float cb = static_cast<float>(static_cast<bfloat16_t>(c));
    p[b * N + r] = static_cast<bfloat16_t>(cb / (1.0f + metal::precise::exp(-cb)));
    bool on = act[b] != 0;
    cso[o + r] = on ? s1 : s0;
    cso[o + CONVR + r] = on ? s2 : s1;
    cso[o + 2 * CONVR + r] = on ? y : s2;
} else if (GATE) {
    float gate = static_cast<float>(y);
    float sg = static_cast<float>(static_cast<bfloat16_t>(gate / (1.0f + metal::precise::exp(-gate))));
    float up = 0.0f;
    for (int j = ptr[N + r]; j < ptr[N + r + 1]; j++) {
        float h = static_cast<float>(static_cast<bfloat16_t>(static_cast<float>(x[b * R + col[j]]) * inv));
        up += val[j] * h;
    }
    p[b * N + r] = static_cast<bfloat16_t>(sg * static_cast<float>(static_cast<bfloat16_t>(up)));
} else {
    p[b * N + r] = y;
}
"""

# one threadgroup of 32 x 32 threads per value head: lane l of simdgroup g holds key columns 4l..4l+3 of state
# rows 4g..4g+3 (mlx-lm's gated delta kernel's split of a row, its sums in the same order)
_HEAD = """
uint lane = thread_position_in_threadgroup.x, g = thread_position_in_threadgroup.y;
uint h = threadgroup_position_in_grid.z % NH, b = threadgroup_position_in_grid.z / NH;
uint tid = g * 32 + lane;
threadgroup float qs[128], ks[128], ys[128], red[3];
uint kh = kmap[h];
uint base = b * N;
if (tid < 128) {
    qs[tid] = static_cast<float>(p[base + kh * 128 + tid]);
    ks[tid] = static_cast<float>(p[base + NK * 128 + kh * 128 + tid]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (g < 2) {
    threadgroup float* u = g == 0 ? qs : ks;
    float s = 0.0f;
    for (int i = 0; i < 4; i++) s += u[4 * lane + i] * u[4 * lane + i];
    s = simd_sum(s);
    if (lane == 0) red[g] = s;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float qn = 0.0f, kn = 0.0f;
if (tid < 128) {
    qn = qs[tid] * metal::precise::rsqrt(red[0] + 1e-6f) / 11.313708498984761f;
    kn = ks[tid] * metal::precise::rsqrt(red[1] + 1e-6f);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 128) { qs[tid] = qn; ks[tid] = kn; }
threadgroup_barrier(mem_flags::mem_threadgroup);
float bb = static_cast<float>(p[base + CONVR + NH * 128 + h]);
float aa = static_cast<float>(p[base + CONVR + NH * 128 + NH + h]);
float y0 = 1.0f / (1.0f + metal::precise::exp(metal::abs(bb)));
float beta = static_cast<float>(static_cast<bfloat16_t>((bb < 0) ? y0 : 1.0f - y0));
float xa = aa + static_cast<float>(dtb[h]);
float sp = (xa > 20.0f) ? xa : log1p(metal::precise::exp(metal::min(xa, 20.0f)));
float decay = metal::precise::exp(-metal::precise::exp(static_cast<float>(alog[h])) * sp);
for (int j = 0; j < 4; j++) {
    uint dv = 4 * g + j;
    uint off = ((b * NH + h) * 128 + dv) * 128 + 4 * lane;
    float v = static_cast<float>(p[base + 2 * NK * 128 + h * 128 + dv]);
    float st[4];
    float kv = 0.0f;
    for (int i = 0; i < 4; i++) { st[i] = si[off + i] * decay; kv += st[i] * ks[4 * lane + i]; }
    kv = simd_sum(kv);
    float delta = (v - kv) * beta;
    float out = 0.0f;
    bool on = act[b] != 0;
    for (int i = 0; i < 4; i++) { st[i] = st[i] + ks[4 * lane + i] * delta; out += st[i] * qs[4 * lane + i]; so[off + i] = on ? st[i] : si[off + i]; }
    out = simd_sum(out);
    if (lane == 0) ys[dv] = static_cast<float>(static_cast<bfloat16_t>(out));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (g == 0) {
    float s = 0.0f;
    for (int i = 0; i < 4; i++) s += ys[4 * lane + i] * ys[4 * lane + i];
    s = simd_sum(s);
    if (lane == 0) red[2] = s;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 128) {
    float z = static_cast<float>(p[base + CONVR + h * 128 + tid]);
    float n = static_cast<float>(static_cast<bfloat16_t>(ys[tid] * metal::precise::rsqrt(red[2] / 128.0f + 1e-6f)));
    n = static_cast<float>(static_cast<bfloat16_t>(static_cast<float>(nw[tid]) * n));
    o[b * NH * 128 + h * 128 + tid] = static_cast<bfloat16_t>(n * (z / (1.0f + metal::precise::exp(-z))));
}
"""

# the residual after an output projection: written dims get x + W y, the others x
_OUT = """
uint r = thread_position_in_grid.x, b = thread_position_in_grid.y;
if (r >= R) return;
bfloat16_t xr = x[b * R + r];
int w = wrow[r];
if (w < 0) { xo[b * R + r] = xr; return; }
float acc = 0.0f;
for (int j = ptr[w]; j < ptr[w + 1]; j++) acc += val[j] * static_cast<float>(y[b * K + col[j]]);
xo[b * R + r] = static_cast<bfloat16_t>(static_cast<float>(xr) + static_cast<float>(static_cast<bfloat16_t>(acc)));
"""

# the next token, decided on the GPU: the output code (its family id and payload) -> the token (a trace token, or
# Q: the text token the computer echoes) and its embedding row; a row stops where Qwen predicts (-1), on a code that
# names no token (-2) or where the computer leaves a position after its own token (-3); -4: stopped before
_NEXT = """
uint r = thread_position_in_grid.x, b = thread_position_in_grid.y;
if (r >= R) return;
int fid = 0, pay = 0;
bool set = false;
for (int i = 0; i < NFAM + NPAY; i++) {
    float c = static_cast<float>(h[b * R + code[i]]);
    set = set || (c != 0.0f);
    int bit = c > 0.0f ? 1 : 0;
    if (i < NFAM) fid |= bit << i; else pay |= bit << (i - NFAM);
}
int tok = -1, f = -1, e = -1;
if (set) {
    if (fid < NFAMS) {
        if (pay < fsize[fid]) { tok = V0 + fbase[fid] + pay; f = fid; e = expl[fbase[fid] + pay]; } else tok = -2;
    } else if (fid < NFAMS + NCTRL) {
        tok = V0 + CTRL + (fid - NFAMS); e = expl[CTRL + (fid - NFAMS)];
    } else if (fid == NFAMS + NCTRL && pay < NPRINT) {
        tok = printable[pay]; e = -2 - pay;
    } else tok = -2;
}
if (tok == -1 && cur[b] >= V0) tok = -3;
bool on = act[b] != 0;
bool go = on && tok >= 0;
if (r == 0) { nt[b] = on ? tok : -4; nact[b] = go ? 1 : 0; }
float v = 0.0f;
if (go) {
    if (e >= 0) v = static_cast<float>(erows[e * R + r]);
    else if (e <= -2) v = static_cast<float>(srows[prow[-2 - e] * R + r]);
    else {
        v = static_cast<float>(fb[f * R + r]);
        for (int i = 0; i < 16; i++)
            if ((pay >> i) & 1) v += static_cast<float>(fd[(f * 16 + i) * R + r]);
    }
}
xn[b * R + r] = static_cast<bfloat16_t>(v);
"""


def _csr(blocks):
    """Sparse rows from dense (rows, k) blocks, each with its column index list: (ptr, col, val)."""
    ptr, col, val = [0], [], []
    for w, cols in blocks:
        w = np.asarray(w, np.float32)
        for row in w:
            nz = np.nonzero(row)[0]
            col.extend(np.asarray(cols)[nz].tolist())
            val.extend(row[nz].tolist())
            ptr.append(len(col))
    return mx.array(np.array(ptr, np.int32)), mx.array(np.array(col, np.int32)), mx.array(np.array(val, np.float32))


class FusedStep:
    """``Chip``'s decoding step (one token a row) in Metal kernels; the same results."""

    def __init__(self, chip: "Chip"):
        W, self.R = chip.W, int(chip.W["dims"].shape[0])
        self.H = chip.H
        ar = lambda k: np.array(W[k].astype(f32)) if W[k].dtype == bf16 else np.array(W[k])  # noqa: E731
        self.layers = []
        for L in range(chip.n_layers):
            p, lay = f"L{L}.", {}
            if p + "conv" in W:
                nh = int(W[p + "kmap"].shape[0])
                C = int(W[p + "conv"].shape[0])
                blocks = [(ar(p + "qkv.w"), ar(p + "qkv.in"))]
                for k, n in (("z", nh * HEAD), ("b", nh), ("a", nh)):
                    blocks.append((ar(p + k + ".w"), ar(p + k + ".in")) if p + k + ".w" in W else (np.zeros((n, 0)), np.zeros(0, np.int64)))
                lay["proj"] = _csr(blocks)
                lay["n"] = C + nh * HEAD + 2 * nh
                lay.update(C=C, nh=nh, nk=(C - nh * HEAD) // (2 * HEAD), conv=mx.array(ar(p + "conv").reshape(-1)),
                           kmap=W[p + "kmap"], alog=W[p + "A_log"].astype(f32), dtb=W[p + "dt_bias"].astype(f32),
                           nw=W[p + "norm"].astype(f32), out=self._out(ar(p + "out.w"), ar(p + "out.to")))
            if p + "gate.w" in W:
                g, u = ar(p + "gate.w"), ar(p + "up.w")
                lay["mlp"] = _csr([(g, ar(p + "gate.in")), (u, ar(p + "up.in"))])
                lay["units"] = g.shape[0]
                lay["down"] = self._out(ar(p + "down.w"), ar(p + "down.to"))
            self.layers.append(lay)
        mk = mx.fast.metal_kernel
        self.k_proj = mk(name="qwenputer_proj", input_names=["x", "ptr", "col", "val", "cw", "cs", "act"],
                         output_names=["p", "cso"], source=_PROJ)
        self.k_head = mk(name="qwenputer_head", input_names=["p", "kmap", "alog", "dtb", "nw", "si", "act"],
                         output_names=["o", "so"], source=_HEAD)
        self.k_out = mk(name="qwenputer_out", input_names=["x", "y", "wrow", "ptr", "col", "val"],
                        output_names=["xo"], source=_OUT)
        self.k_next = mk(name="qwenputer_next", input_names=["h", "cur", "act", "code", "fsize", "fbase", "expl", "printable",
                                                             "prow", "erows", "srows", "fb", "fd"],
                         output_names=["nt", "nact", "xn"], source=_NEXT)
        e = chip._emb
        expl = np.full(CHIP_VOCAB, -1, np.int32)
        expl[e["explicit_ids"]] = np.arange(len(e["explicit_ids"]))
        srow = {int(t): i for i, t in enumerate(e["special_ids"])}
        self.tables = [chip.code.astype(mx.int32), mx.array([sz for _, sz in FAMILY_SIZES], mx.int32),
                       mx.array([b for b, _ in FAMILIES.values()], mx.int32), mx.array(expl),
                       mx.array(np.array(chip.printable, np.int32)), mx.array(np.array([srow[t] for t in chip.printable], np.int32)),
                       W["emb.explicit_rows"], W["emb.special_rows"], W["emb.fam_base"], W["emb.fam_delta"]]
        self.consts = [("R", self.R), ("NFAM", chip.nfam), ("NPAY", chip.npay), ("NFAMS", len(FAMILY_SIZES)),
                       ("NCTRL", len(CONTROL_NAMES)), ("NPRINT", len(chip.printable)), ("V0", chip.v0), ("CTRL", CTRL)]
        self.chip = chip

    def _out(self, w, to):
        wrow = np.full(self.R, -1, np.int32)
        wrow[to.astype(np.int64)] = np.arange(len(to))
        return (mx.array(wrow),) + _csr([(w, np.arange(w.shape[1]))])

    def __call__(self, x: mx.array, st: ChipState, act: Optional[mx.array] = None) -> mx.array:
        """x (B, 1, R) bf16 -> the residual after the computer layers; st is updated (but for rows whose
        ``act`` is 0: they keep their states)."""
        B = x.shape[0]
        if act is None:
            act = mx.ones((B,), mx.int32)
        x = x.reshape(B, self.R)
        R, tg = self.R, 256
        dummy = mx.zeros((1,), bf16)
        for L, lay in enumerate(self.layers):
            if "proj" in lay:
                n, C, nh = lay["n"], lay["C"], lay["nh"]
                ptr, col, val = lay["proj"]
                p, cs = self.k_proj(inputs=[x, ptr, col, val, lay["conv"], st.conv[L], act],
                                    template=[("R", R), ("H", self.H), ("N", n), ("CONVR", C), ("GATE", False)],
                                    grid=(-(-n // tg) * tg, B, 1), threadgroup=(tg, 1, 1),
                                    output_shapes=[(B, n), (B, 3, C)], output_dtypes=[bf16, bf16])
                o, so = self.k_head(inputs=[p, lay["kmap"], lay["alog"], lay["dtb"], lay["nw"], st.rec[L], act],
                                    template=[("N", n), ("NK", lay["nk"]), ("NH", nh), ("CONVR", C)],
                                    grid=(32, 32, nh * B), threadgroup=(32, 32, 1),
                                    output_shapes=[(B, nh * HEAD), st.rec[L].shape], output_dtypes=[bf16, f32])
                st.conv[L], st.rec[L] = cs, so
                wrow, ptr, col, val = lay["out"]
                x = self.k_out(inputs=[x, o, wrow, ptr, col, val], template=[("R", R), ("K", nh * HEAD)],
                               grid=(-(-R // tg) * tg, B, 1), threadgroup=(tg, 1, 1),
                               output_shapes=[(B, R)], output_dtypes=[bf16])[0]
            if "mlp" in lay:
                U = lay["units"]
                ptr, col, val = lay["mlp"]
                p = self.k_proj(inputs=[x, ptr, col, val, mx.zeros((1,), f32), dummy, act],
                                template=[("R", R), ("H", self.H), ("N", U), ("CONVR", 0), ("GATE", True)],
                                grid=(-(-U // tg) * tg, B, 1), threadgroup=(tg, 1, 1),
                                output_shapes=[(B, U), (1,)], output_dtypes=[bf16, bf16])[0]
                wrow, ptr, col, val = lay["down"]
                x = self.k_out(inputs=[x, p, wrow, ptr, col, val], template=[("R", R), ("K", U)],
                               grid=(-(-R // tg) * tg, B, 1), threadgroup=(tg, 1, 1),
                               output_shapes=[(B, R)], output_dtypes=[bf16])[0]
        return x.reshape(B, 1, R)

    def next(self, h: mx.array, cur: mx.array, act: mx.array):
        """(next tokens (B,), next act (B,), next input rows (B, 1, R)) from the residual after the computer layers."""
        B = h.shape[0]
        nt, nact, xn = self.k_next(inputs=[h.reshape(B, self.R), cur, act] + self.tables, template=self.consts,
                                   grid=(-(-self.R // 256) * 256, B, 1), threadgroup=(256, 1, 1),
                                   output_shapes=[(B,), (B,), (B, self.R)], output_dtypes=[mx.int32, mx.int32, bf16])
        return nt, nact, xn.reshape(B, 1, self.R)

    def run(self, st: ChipState, first: List[int], rows: Optional[List[int]] = None, max_tokens: int = 60000,
            sync: int = 32) -> List[List[int]]:
        """Free-run the computer from each row's token in ``first`` (rows: the batch rows that run, the others keep
        their states) until Qwen is to predict: per running row, the tokens it read after ``first``, the last of them
        the one after which Qwen predicts (a negative last entry: the computer failed, see _NEXT). Syncs with the
        host every ``sync`` steps (between them the GPU decides the tokens: 1.1 ms a step on an M2 Pro, against
        4.3 with a sync each)."""
        B = len(first)
        rows = list(range(B)) if rows is None else rows
        on = [i in set(rows) for i in range(B)]
        x = self.chip.embed([[t if on[i] else 0] for i, t in enumerate(first)])
        cur = mx.array(np.array(first, np.int32))
        act = mx.array(np.array(on, np.int32))
        out = {i: [] for i in rows}
        live, n, k = set(rows), 0, 4
        while live and n < max_tokens:
            got = []
            for _ in range(k):
                h = self(x, st, act)
                cur, act, x = self.next(h, cur, act)
                got.append(cur)
            mx.eval(got, act, st.rec, st.conv)
            n += k
            k = min(2 * k, sync)
            got = np.array(mx.stack(got, 1))
            for i in list(live):
                for t in got[i].tolist():
                    if t == -4:
                        break
                    if t < 0:
                        if t != -1:
                            out[i].append(t)
                        break
                    out[i].append(t)
                if got[i][-1] < 0:
                    live.discard(i)
        return [out[i] for i in rows]
