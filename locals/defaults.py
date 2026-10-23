"""Defaults shared by the commands."""

from pathlib import Path

# Short names for the models locals is tested with; any mlx-lm chat model id or
# local path works as well.
MODELS = {
    "qwen3.5-4b": "mlx-community/Qwen3.5-4B-MLX-4bit",
    "qwen3.5-2b": "mlx-community/Qwen3.5-2B-MLX-4bit",
    "gemma-4-e4b": "mlx-community/gemma-4-e4b-it-4bit",
    "lfm2.5-2.6b": "LiquidAI/LFM2.5-2.6B-MLX-4bit",  # 1.6 GB: 8 GB Macs
    "minicpm5-2b": "mlx-community/MiniCPM5-2B-mlx-4Bit",  # 1.4 GB
    # Recursive's 40B mixture of experts (2.5B active), experts at 3 bits: 17.8 GB, a 24 GB Mac with the GPU limit
    # raised (too_big says how); Recursive's own 4-bit build (RecursiveSI/flint-int4, 22.6 GB) wants 32 GB or more
    "flint": "mlx-community/flint-mixed-3-4bit",
}


# Measured on a MacBook Pro with an M2 Pro (16 GB): tokens per second with 8 requests at once, stock
# mlx-lm and locals, 256 tokens each (locals bench measures them on any Mac), and the weights in GB.
MEASURED_M2_PRO = {
    "qwen3.5-4b": {"weights_gb": 2.8, "mlx": 87, "locals": 271},
    "qwen3.5-2b": {"weights_gb": 1.6, "mlx": 188, "locals": 576},
    "gemma-4-e4b": {"weights_gb": 4.8, "mlx": 80, "locals": 224},
    "lfm2.5-2.6b": {"weights_gb": 1.5, "mlx": 144, "locals": 404},
    "minicpm5-2b": {"weights_gb": 1.3, "mlx": 160, "locals": 478},
}
# Models the M2 Pro's 16 GB cannot hold, measured as above on the Mac named ("on")
MEASURED_ELSEWHERE = {
    "flint": {"weights_gb": 16.6, "mlx": 202, "locals": 264, "on": "an M4 Pro (24 GB)"},
}
SHORT = {"qwen3.5-4b": "qwen", "qwen3.5-2b": "qwen3.5-2b", "gemma-4-e4b": "gemma", "lfm2.5-2.6b": "lfm",
         "minicpm5-2b": "minicpm", "flint": "flint"}
# Agent tasks done in the end-to-end suite (tests/e2e: real agent tools on real coding tasks, an M2 Pro), the same
# eight for both (OpenCode's four, Hermes's and OpenClaw's two; Qwen3.5-4B also did all 18 with eight tools):
# Gemma 4 E4B repeats itself on the longer ones (the server's loop guard cuts it) and gives up early (2 to 3 of 8).
AGENT_TASKS = {"qwen3.5-4b": (8, 8), "gemma-4-e4b": (3, 8)}


def agent_note(model: str) -> str:
    """A line for a model that does poorly as a coding agent (in AGENT_TASKS), else ""."""
    key = next((k for k, v in MODELS.items() if v == model), None)
    done, of = AGENT_TASKS.get(key, (None, None))
    if done is None or done * 2 >= of:
        return ""
    q = AGENT_TASKS["qwen3.5-4b"]
    return (f"note: {SHORT[key]} finished {done} of {of} coding-agent tasks in our tests, qwen {q[0]} of {q[1]}; "
            f"for agents use --model qwen")


def chip() -> str:
    """The Mac's chip ("Apple M2 Pro"), or "" if it cannot be read."""
    import subprocess

    try:
        return subprocess.run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def memory_gb() -> float:
    """The Mac's memory in GB (0 if it cannot be read)."""
    import subprocess

    try:
        return int(subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout) / 2**30
    except Exception:  # noqa: BLE001
        return 0.0


# GPU memory a model needs next to its weights: the KV cache of a few agents' prompts, the prefill and
# the server's margins. A model whose weights leave less than this does not run: macOS swaps it, or
# Metal stops it (and with it, on a 16 GB Mac, once the whole machine: a GPU driver panic).
HEADROOM_GB = 2.0


def gpu_limit_gb() -> float:
    """The GPU memory MLX may use on this Mac, in GB (0 if unknown)."""
    try:
        import mlx.core as mx

        return mx.device_info()["max_recommended_working_set_size"] / 2**30
    except Exception:  # noqa: BLE001
        return 0.0


def cached_snapshot(model: str, allow_patterns) -> "Path | None":
    """The folder of a model downloaded already (Hugging Face's cache), or None. Without the progress
    bar Hugging Face draws even for a lookup in the cache ("Fetching 9 files" on every start)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars

    quiet = are_progress_bars_disabled()
    disable_progress_bars()
    try:
        return Path(snapshot_download(model, local_files_only=True, allow_patterns=allow_patterns))
    except Exception:  # noqa: BLE001  # not downloaded (or not a repository id)
        return None
    finally:
        if not quiet:
            enable_progress_bars()


def weights_gb(model: str, remote: bool = False) -> "float | None":
    """The model's weights in GB: its local files (a folder, or the Hugging Face cache), else, with
    ``remote``, the sizes Hugging Face lists. None if unknown."""
    path = Path(model).expanduser()
    if not path.exists():
        path = cached_snapshot(model, ["*.json", "*.safetensors"])
    if path is not None:
        files = list(path.glob("*.safetensors"))
        if files:
            return sum(f.stat().st_size for f in files) / 2**30
    if not remote:
        return None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model, files_metadata=True, timeout=10)
        n = sum(s.size or 0 for s in info.siblings or [] if s.rfilename.endswith(".safetensors") and "/" not in s.rfilename)
        return n / 2**30 if n else None
    except Exception:  # noqa: BLE001
        return None


def too_big(model: str, remote: bool = False) -> "str | None":
    """Why ``model`` cannot run on this Mac (its weights and HEADROOM_GB exceed the GPU memory), or None.
    LOCALS_IGNORE_MEMORY=1 skips the check."""
    import os

    if os.environ.get("LOCALS_IGNORE_MEMORY"):
        return None
    w, limit = weights_gb(model, remote), gpu_limit_gb()
    if not w or not limit or w + HEADROOM_GB <= limit:
        return None
    raise_to = wired_limit_hint(w)
    if raise_to:
        return (f"{model} has {w:.1f} GB of weights; this Mac's GPU may use {limit:.1f} GB of its {memory_gb():.0f} GB, macOS's "
                f"default, and the model needs about {HEADROOM_GB:.0f} GB more to run. This Mac has the memory: let the GPU use "
                f"more until the next restart with\n  sudo sysctl iogpu.wired_limit_mb={raise_to}\nand quit large apps first "
                f"(browsers with many tabs). Or choose a smaller model (--model {_smaller(model)}), or set LOCALS_IGNORE_MEMORY=1 "
                f"to try anyway (macOS may swap, or stop the server).")
    return (f"{model} has {w:.1f} GB of weights; this Mac's GPU can use {limit:.1f} GB ({memory_gb():.0f} GB of memory), "
            f"and the model needs about {HEADROOM_GB:.0f} GB more to run. Choose a smaller model (--model {_smaller(model)}), or set "
            f"LOCALS_IGNORE_MEMORY=1 to try anyway (macOS may swap, or stop the server).")


MACOS_GB = 4.0  # memory to leave macOS and the apps a Mac keeps open when the GPU limit is raised


def wired_limit_hint(weights: float) -> int:
    """The GPU limit in MB (``sudo sysctl iogpu.wired_limit_mb=...``) at which a model of ``weights`` GB fits on
    this Mac, when it fits under macOS's default limit only with that raised; 0 if raising it does not help, or
    if it is raised already. macOS gives the GPU about two thirds of a 24 GB Mac (16 GB), and Flint needs 19."""
    import math

    need = weights + HEADROOM_GB + 0.5
    mem, limit = memory_gb(), gpu_limit_gb()
    if not mem or need > mem - MACOS_GB or need <= limit:
        return 0
    return int(math.ceil(need)) * 1024


def _smaller(model: str) -> str:
    """A model to suggest instead of one too big for this Mac: this Mac's default, or the smallest Qwen3.5."""
    return "qwen3.5-2b" if model in (DEFAULT_MODEL, MODELS.get("qwen3.5-4b")) else short_name(DEFAULT_MODEL)


def gguf_hint(model: str, remote: bool = True) -> "str | None":
    """If ``model`` is a GGUF model (a .gguf file, a folder or a Hugging Face repo with .gguf weights and
    no safetensors: llama.cpp's, Ollama's and LM Studio's format), what to use instead: locals runs MLX
    models (MLX reads only GGUF's rare Q4_0/Q8_0 tensors, under llama.cpp's names; mlx-lm loads none).
    The MLX versions of the same model on Hugging Face are named. None for any other model."""
    import re

    path = Path(model).expanduser()
    if path.is_file() or path.is_dir():
        files = [path] if path.is_file() else list(path.glob("*.gguf")) + list(path.glob("*.safetensors"))
        if not any(f.suffix == ".gguf" for f in files) or any(f.suffix == ".safetensors" for f in files):
            return None
        name = (path.stem if path.is_file() else path.name)
    elif remote and "/" in model and model.count("/") == 1:
        try:
            from huggingface_hub import HfApi

            # downloaded already (MLX weights in the cache): no request to Hugging Face, so it starts offline too
            cached = cached_snapshot(model, ["*.json", "*.safetensors"])
            if cached is not None and any(cached.glob("*.safetensors")):
                return None
            names = [s.rfilename for s in HfApi().model_info(model, timeout=10).siblings or []]
        except Exception:  # noqa: BLE001
            return None
        if not any(n.endswith(".gguf") for n in names) or any(n.endswith(".safetensors") for n in names):
            return None
        name = model.split("/")[1]
    else:
        return None
    base = re.sub(r"[-_.]?(i1-)?gguf$", "", name, flags=re.I)
    base = re.sub(r"[-_.](Q\d\w*|IQ\d\w*|F16|BF16|F32)$", "", base, flags=re.I)  # a file's quantization suffix
    found = []
    try:
        from huggingface_hub import HfApi

        for q in dict.fromkeys([base, base.split("_", 1)[-1]]):  # bartowski's "Qwen_Qwen3.5-0.8B"
            found += [m.id for m in HfApi().list_models(search=q, filter="mlx", sort="downloads", limit=12) if m.id not in found]
    except Exception:  # noqa: BLE001
        pass
    # the same model first (its size in the name: by downloads alone, a 9B came before the 4B asked for), then
    # mlx-community's conversions, then 4-bit ones
    want = base.split("_", 1)[-1].lower()
    found.sort(key=lambda mid: (want not in mid.split("/")[-1].lower(), not mid.startswith("mlx-community/"),
                                "4bit" not in mid.lower()))
    see = (f" MLX versions of it: {', '.join(found[:3])} (--model {found[0]})." if found else
           " Look for an MLX version of it on Hugging Face (mlx-community), or convert the original weights with "
           "mlx_lm.convert --hf-path ORIGINAL -q.")
    return f"{model} is a GGUF model (the format of llama.cpp, Ollama and LM Studio); locals runs MLX models.{see}"


def ollama_hint(model: str) -> "str | None":
    """If ``model`` is an Ollama model name (name:tag, no publisher; qwen3.5:4b and the others locals knows are
    taken by model_id), what to use instead: locals runs MLX models from Hugging Face, whose MLX versions of it are
    named, the size of the tag first. None for any other name. (Without it: "Repo id must use alphanumeric chars".)"""
    import re

    if ":" not in model or "/" in model or Path(model).expanduser().exists():
        return None
    name, tag = (model.lower().split(":", 1) + [""])[:2]
    size = tag if tag and tag != "latest" else ""
    names = list(dict.fromkeys([re.sub(r"([a-z])(\d)", r"\1-\2", name), name]))  # llama3.2 is Llama-3.2 there
    found = []
    try:
        from huggingface_hub import HfApi

        # with the size first: by downloads alone, qwen3's first 20 held no 4B (a 27B was named for qwen3:4b)
        for q in ([f"{n}-{size}" for n in names] if size else []) + names:
            found += [m.id for m in HfApi().list_models(search=q, filter="mlx", sort="downloads", limit=20) if m.id not in found]
    except Exception:  # noqa: BLE001
        pass
    if size:  # another size is another model: none of those is named
        found = [m for m in found if re.search(rf"(^|[-_.]){re.escape(size)}([-_.]|$)", m.split("/")[-1].lower())]
    found.sort(key=lambda mid: (not mid.startswith("mlx-community/"), "4bit" not in mid.lower()))
    see = (f" MLX versions of it on Hugging Face: {', '.join(found[:3])} (--model {found[0]})." if found else
           " Look for an MLX version of it on Hugging Face (huggingface.co/mlx-community).")
    return (f"{model} is an Ollama model name; locals runs MLX models from Hugging Face.{see} Or one of locals's own: "
            f"qwen, gemma, lfm, minicpm.")


def default_model() -> str:
    """The model for this Mac: Qwen3.5-4B (3 GB of weights, 64k context) from 16 GB of memory up,
    Qwen3.5-2B on a Mac with less (the 4B and its prompt cache do not fit next to macOS in 8 GB)."""
    gb = memory_gb()
    return MODELS["qwen3.5-2b"] if 0 < gb < 12 else MODELS["qwen3.5-4b"]


DEFAULT_MODEL = default_model()
ALIASES = {"qwen": "qwen3.5-4b", "qwen3.5": "qwen3.5-4b", "gemma": "gemma-4-e4b", "gemma4": "gemma-4-e4b",
           "gemma-4": "gemma-4-e4b", "gemma4-e4b": "gemma-4-e4b", "gemma-4-e4b-it": "gemma-4-e4b",
           "lfm": "lfm2.5-2.6b", "lfm2.5": "lfm2.5-2.6b", "minicpm": "minicpm5-2b", "minicpm5": "minicpm5-2b",
           "flint-40b": "flint", "flint-3bit": "flint"}
MODEL_HELP = ("model: qwen (Qwen3.5-4B, the default from 16 GB of memory), qwen3.5-2b (the default below), gemma (Gemma 4 "
              "E4B), lfm (LFM2.5-2.6B), minicpm (MiniCPM5-2B), flint (Recursive's Flint, 24 GB Macs and up), or any "
              "mlx-lm model id or local path")


def short_name(model: str) -> str:
    """The name a model has in the API (``locals/<short name>``): its repository or folder name without the
    4-bit suffix, in any capitalization ("MiniCPM5-2B-mlx-4Bit" -> "minicpm5-2b"), lower case."""
    import re

    # (and a mixed recipe's bits: flint-mixed-3-4bit is "flint")
    return re.sub(r"(-mlx)?(-mixed-\d+)?-4bit$", "", str(model).rstrip("/").split("/")[-1], flags=re.I).lower()


def model_id(name: str) -> str:
    """The model id for a short name (qwen, gemma, qwen3.5-4b, gemma-4-e4b, ..., and Ollama's names for them:
    qwen3.5:4b, gemma4:e4b); a local folder as its
    absolute path (the background server runs elsewhere, and later commands from other folders);
    other names unchanged. A short name wins over a folder of the same name in the current one."""
    name = str(name)  # a Path too, as mlx_lm.load takes
    key = name.strip().lower()
    if ":" in key and key not in ALIASES and key not in MODELS and not Path(name).expanduser().exists():
        ollama = key.removesuffix(":latest").replace(":", "-")  # Ollama's names: qwen3.5:4b, gemma4:e4b, qwen3.5
        if ollama in ALIASES or ollama in MODELS:
            key = ollama
    if key in ALIASES or key in MODELS:
        return MODELS[ALIASES.get(key, key)]
    path = Path(name).expanduser()
    if path.exists():
        return str(path.absolute())  # links kept (a link to a snapshot folder keeps its own name)
    return name


def build_id() -> str:
    """A fingerprint of the installed locals code (its files' names, sizes and times): a background
    server started before an upgrade, or before an edit, reports another one."""
    import hashlib

    here = Path(__file__).resolve().parent
    h = hashlib.sha1()
    for f in sorted(here.glob("*.py")) + sorted(here.glob("models/*.py")):
        if f.name in ("plan.py", "solve.py"):  # the client side of locals solve: the server does not run them
            continue
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns};".encode())
    return h.hexdigest()[:12]
