"""mlx-lm, faster: the same functions with the same arguments.

Change the import and keep your code::

    # from mlx_lm import load, generate, stream_generate, batch_generate
    from locals.lm import load, generate, stream_generate, batch_generate

    model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
    text = generate(model, tokenizer, prompt, max_tokens=256)

or keep mlx-lm's loader and patch the model you already have::

    from mlx_lm import load
    import locals
    model, tokenizer = load(...)
    locals.patch(model)

``patch`` changes how the model computes, not what it computes: 4-bit matrix
products of 2-16 rows go to kernels tuned for your GPU (mlx-lm's default
kernels are built for one row or many), q/k/v and gate/up projections are
fused, Qwen3.5's recurrent layers run as one kernel, and on M1/M2 GPUs (no
native bf16) the model computes in fp16. Several sequences at once (batch_generate,
mlx-lm's BatchGenerator, a server) gain the most; one sequence runs at the
same speed or a little faster. Greedy outputs are the same tokens wherever the
model's top two choices are not within rounding of each other.
"""

from __future__ import annotations

from typing import Optional

from mlx_lm import batch_generate, generate, stream_generate  # noqa: F401  (re-exported unchanged)
from mlx_lm import load as _mlx_load
from mlx_lm.generate import BatchGenerator  # noqa: F401
from mlx_lm.sample_utils import make_logits_processors, make_sampler  # noqa: F401

__all__ = ["load", "patch", "generate", "stream_generate", "batch_generate", "BatchGenerator", "make_sampler",
           "make_logits_processors"]


def patch(model, dtype: Optional[str] = "auto", fuse: bool = True, kernels: bool = True, tune: Optional[str] = None, log=None):
    """Make a loaded mlx-lm model faster in place; returns it.

    ``dtype``: "auto" computes in fp16 on GPUs without native bf16 (M1, M2) and
    keeps the checkpoint's precision elsewhere; "bfloat16" or "float16" force one.
    ``fuse``: fuse projections (and Qwen3.5's recurrent layers). ``kernels``:
    route small-batch 4-bit matrix products to the tuned kernels. ``tune``:
    "auto" (default; tunes once per GPU and model shape, about two minutes, then
    cached), "quick", "full" or "off" (also the LOCALS_TUNE environment variable).
    """
    import os

    import mlx.core as mx

    from .fuse import fuse_model
    from .kernels import enable_small_m, tune_model
    from .runtime import set_compute_dtype

    set_compute_dtype(model, dtype)
    if fuse:
        fuse_model(model)
    enable_small_m(kernels, model)
    if kernels:
        tune_model(model, mode=tune or os.environ.get("LOCALS_TUNE", "auto"), log=log)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    return model


def load(path_or_hf_repo: str, *args, dtype: Optional[str] = "auto", tune: Optional[str] = None, **kwargs):
    """``mlx_lm.load`` with the model patched (``patch``); same arguments and returns."""
    from .defaults import model_id
    from .models import register

    register()  # architectures this mlx-lm lacks (Flint)
    out = _mlx_load(model_id(path_or_hf_repo), *args, **kwargs)  # short names too: "qwen", "gemma"
    patch(out[0], dtype=dtype, tune=tune)
    return out
