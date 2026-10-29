"""Model architectures the installed mlx-lm does not have yet.

mlx-lm finds a model's code by importing ``mlx_lm.models.<model_type>``.
``register()`` puts each module here in that place, unless mlx-lm ships its
own: then mlx-lm's stays, so the stock engine (``mlx/<model>``) is still stock
mlx-lm. Locals's fast paths (``fuse.py``) patch either one.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

_MODULES = ("flint", "clover_moe")  # Recursive's Flint (mlx-lm has it from ml-explore/mlx-lm#1998, unreleased); DiffuSpace's Clover


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
