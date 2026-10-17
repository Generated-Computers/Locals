"""A new project's first request reuses an agent's system prompt: agents end it with the project's folder and the
date (OpenCode's <env> block), so the checkpoint at the end of the system part is reached only by sessions in the
same folder; one at the start of that block is shared by every project. Needs Qwen3.5-4B's tokenizer (cached)."""

import json
import os
from pathlib import Path

import pytest

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
DATA = json.loads((Path(__file__).parent / "data" / "opencode_requests.json").read_text())


class _Svc:
    def __init__(self, tok):
        self.tokenizer = tok


@pytest.fixture(scope="module")
def server():
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True, allow_patterns=["*.json", "*.jinja", "tokenizer*"])
    except Exception:  # noqa: BLE001
        pytest.skip("tokenizer not cached")
    from mlx_lm.utils import load_tokenizer

    from locals.server import Server

    srv = Server(MODEL, engines=("locals",), preload=(), log=lambda s: None)
    srv._svc["locals"] = _Svc(load_tokenizer(snapshot_download(MODEL, local_files_only=True)))
    return srv


def _request(folder: str, system: str = DATA["system"], user: str = "Fix the failing test."):
    return {"model": "locals/qwen3.5-4b", "tools": DATA["tools"],
            "messages": [{"role": "system", "content": system.replace("/home/user/project", folder)},
                         {"role": "user", "content": user}]}


def _systems(p):
    return [n for n, kind in p["checkpoints"] if kind == "system"]


def test_two_projects_share_a_checkpoint_before_the_folder(server):
    a = server.prepare(_request("/Users/me/code/alpha"))
    b = server.prepare(_request("/Users/me/work/beta-service"))
    shared = len(os.path.commonprefix([a["prompt"], b["prompt"]]))
    common = set(_systems(a)) & set(_systems(b))
    assert common, (_systems(a), _systems(b))
    assert max(common) <= shared and max(common) > 0.95 * shared  # the whole shared part, up to the <env> line
    assert max(common) > 0.9 * len(a["prompt"])  # nearly the whole first request


def test_a_system_prompt_without_markers_is_split_where_sessions_differ(server):
    # an agent of its own: no <env> block, its sessions differ in the last paragraph
    base = DATA["system"].split("Here is some useful information")[0]
    p1 = server.prepare(_request("/x", system=base + "\nSession notes: project alpha, branch main."))
    p2 = server.prepare(_request("/x", system=base + "\nSession notes: project beta, branch dev."))
    p3 = server.prepare(_request("/x", system=base + "\nSession notes: project gamma, branch fix."))
    shared = len(os.path.commonprefix([p2["prompt"], p3["prompt"]]))
    assert set(_systems(p2)) & set(_systems(p3)), (_systems(p1), _systems(p2), _systems(p3))
    assert max(set(_systems(p2)) & set(_systems(p3))) > 0.95 * shared
