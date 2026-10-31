"""Model architectures the installed mlx-lm does not have yet.

mlx-lm finds a model's code by importing ``mlx_lm.models.<model_type>``.
``register()`` puts each module here in that place, unless mlx-lm ships its
own: then mlx-lm's stays, so the stock engine (``mlx/<model>``) is still stock
mlx-lm. Locals's fast paths (``fuse.py``) patch either one.

Flint's 4-bit build as Recursive exports it (RecursiveSI/flint-int4, its
weights as bytes: ``defaults.streamed``) loads ``flint_streamed`` instead,
which reads the experts from the weight files as tokens need them (a 16 GB
Mac runs it): ``register()`` also has mlx-lm's ``load_model`` take that
module for such a folder.

LFM3's config.json points transformers' AutoConfig at Liquid's own Python
code, which transformers 5.19 does not have: loading the tokenizer asked in a
terminal whether to run it, and went on without after 15 s. ``register()``
gives transformers a plain config class for such a name instead, so it neither
asks nor runs that code (the tokenizer needs nothing from it).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_MODULES = ("flint",  # Recursive's Flint (mlx-lm has it from ml-explore/mlx-lm#1998, unreleased)
            "lfm3",  # Liquid's LFM3 (no mlx-lm release has it yet)
            "clover_moe")  # DiffuSpace's Clover (its causal pass as Qwen3.5-MoE's text model; and in blocks)
_CONFIGS = ("lfm3",)  # model types whose config.json names remote code for AutoConfig


def register() -> None:
    for name in _MODULES:
        target = "mlx_lm.models." + name
        if target in sys.modules:
            continue
        try:
            found = importlib.util.find_spec(target) is not None
        except ModuleNotFoundError:
            found = False
        if not found:
            sys.modules[target] = importlib.import_module("." + name, __name__)
    _register_streamed()
    _register_configs()


def _register_configs() -> None:
    try:
        from transformers import AutoConfig, PretrainedConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except ImportError:
        return
    for name in _CONFIGS:
        if name not in CONFIG_MAPPING:
            AutoConfig.register(name, type(name.capitalize() + "Config", (PretrainedConfig,),
                                           {"model_type": name, "__module__": __name__}))


def _register_streamed() -> None:
    """mlx-lm's ``load_model``, with flint_streamed's classes for a folder whose experts stay in its files; its
    sanitize() needs the folder, which mlx-lm does not pass. Any other folder loads as mlx-lm loads it."""
    import mlx_lm.utils as mlx_utils

    if getattr(mlx_utils.load_model, "_locals", False):
        return
    stock = mlx_utils.load_model

    def load_model(model_path, *args, **kwargs):
        from ..defaults import streamed

        if len(args) >= 4 or "get_model_classes" in kwargs or not streamed(Path(model_path)):
            return stock(model_path, *args, **kwargs)
        from . import flint_streamed

        kwargs["get_model_classes"] = lambda config: (flint_streamed.Model, flint_streamed.ModelArgs)
        flint_streamed.MODEL_PATHS.append(Path(model_path))
        try:
            return stock(model_path, *args, **kwargs)
        finally:
            flint_streamed.MODEL_PATHS.pop()

    load_model._locals = True
    mlx_utils.load_model = load_model
