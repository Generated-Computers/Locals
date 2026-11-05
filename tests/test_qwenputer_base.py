"""LOCALS_QWENPUTER_BASE: Qwenputer's Qwen part can be another Qwen3.5-family model (models/qwenputer.py BASE); unset,
it is Qwen3.5-4B, whose 32 layers the checkpoint holds."""

import importlib

from locals.models import qwenputer as Q


def test_base_default_and_override(monkeypatch):
    try:
        monkeypatch.delenv("LOCALS_QWENPUTER_BASE", raising=False)
        assert importlib.reload(Q).BASE == Q.DEFAULT_BASE == "mlx-community/Qwen3.5-4B-MLX-4bit"
        monkeypatch.setenv("LOCALS_QWENPUTER_BASE", "mlx-community/Qwen3.8-27B-4bit")
        assert importlib.reload(Q).BASE == "mlx-community/Qwen3.8-27B-4bit"
    finally:
        monkeypatch.delenv("LOCALS_QWENPUTER_BASE", raising=False)
        importlib.reload(Q)
