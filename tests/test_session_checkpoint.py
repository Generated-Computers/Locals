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


def test_the_next_turn_continues_the_previous_request(server, monkeypatch):
    """A new user message must not change how the previous turn is written: Qwen3.5's template dropped the empty
    think block of every earlier reply, so the next turn's first request stopped being a continuation of the
    last one at the turn's first reply, and its tool calls and results were read again."""
    from locals import server as srv_mod

    call = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": json.dumps({"filePath": "todo.py"})}}
    turn = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "Read todo.py."},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1", "content": "def add(x): ..."}]
    reply = "It defines add."
    nxt = turn + [{"role": "assistant", "content": reply}, {"role": "user", "content": "Now add a test."}]
    tools = [{"type": "function", "function": {"name": "read", "description": "Read a file.", "parameters": {
        "type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}}}]

    def prompts():
        a = server.prepare({"model": "locals/qwen3.5-4b", "messages": turn, "tools": tools})["text"]
        b = server.prepare({"model": "locals/qwen3.5-4b", "messages": nxt, "tools": tools})["text"]
        return a, b

    a, b = prompts()
    assert b.startswith(a + reply), (a[-200:], b[len(a) - 200:len(a) + 100])
    monkeypatch.setattr(srv_mod, "_KEEP_THINK", False)  # the template as it is: the continuation is lost
    a, b = prompts()
    assert not b.startswith(a)


def test_a_timestamp_in_the_middle_of_the_system_message_goes_to_its_end(server):
    """OpenHands writes the date and time (to the microsecond) 36% into its system message: every new session read
    the rest again. The block is moved to the end, and sessions share everything before it."""
    head, tail = DATA["system"][:3000], DATA["system"][3000:].split("Here is some useful information")[0]

    def system(stamp):
        return head + f"\n<CURRENT_DATETIME>\nThe current date and time is: {stamp}\n</CURRENT_DATETIME>\n\n" + tail

    a = server.prepare(_request("/x", system=system("2026-10-17T17:33:08.829311")))
    b = server.prepare(_request("/x", system=system("2026-10-17T17:33:47.825302")))
    shared = len(os.path.commonprefix([a["prompt"], b["prompt"]]))
    assert shared > 0.95 * len(a["prompt"])
    assert "2026-10-17T17:33:08.829311" in a["text"]  # the model still reads it
    common = set(_systems(a)) & set(_systems(b))
    assert common and max(common) > 0.95 * shared


def test_the_folder_in_a_tool_description_moves_to_the_system_message(server):
    """OpenHands names the project's folder in its file editor's description; tools come first in Qwen3.5's
    template, so two folders parted after 1,555 of 13,795 tokens."""
    def req(folder):
        r = _request("/x")
        r["tools"] = [dict(t) for t in DATA["tools"]]
        fn = dict(r["tools"][0]["function"])
        fn["description"] = fn["description"] + f"\n\nYour current working directory is: {folder}\nStart there."
        r["tools"][0] = {"type": "function", "function": fn}
        return r

    a, b = server.prepare(req("/Users/me/alpha")), server.prepare(req("/Users/me/beta"))
    shared = len(os.path.commonprefix([a["prompt"], b["prompt"]]))
    assert shared > 0.97 * len(a["prompt"]) and "/Users/me/alpha" in a["text"]
    assert a["text"].index("/Users/me/alpha") > a["text"].index("<tools>")  # now after the tools


def test_thinking_on_the_reply_checkpoint_is_where_the_next_request_continues(server):
    """Claude Code (thinking on) adds a user text after every tool result, so each earlier reply loses its thoughts
    in the next request: the reply checkpoint must be at the reply's start, inside the prompt. A stand-in reply
    without thoughts kept its empty think block (_stable_template) and the checkpoint fell past the prompt's end."""
    call = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": json.dumps({"filePath": "todo.py"})}}
    first = [{"role": "system", "content": DATA["system"]}, {"role": "user", "content": "Read todo.py."},
             {"role": "assistant", "content": "Reading.", "reasoning_content": "I should read it.", "tool_calls": [call]},
             {"role": "tool", "tool_call_id": "call_1", "content": "def add(x): ..."},
             {"role": "user", "content": "<total_tokens>9000 tokens left</total_tokens>"}]
    second = first + [{"role": "assistant", "content": "It defines add.", "reasoning_content": "Done."},
                      {"role": "user", "content": "Thanks."}]
    req = lambda msgs: {"model": "locals/qwen3.5-4b-thinking", "messages": msgs, "tools": DATA["tools"]}  # noqa: E731
    a, b = server.prepare(req(first)), server.prepare(req(second))
    replies = [n for n, kind in a["checkpoints"] if kind == "reply"]
    shared = len(os.path.commonprefix([a["prompt"], b["prompt"]]))
    assert replies and replies[0] <= shared and replies[0] > shared - 8, (a["checkpoints"], shared, len(a["prompt"]))


def test_gemma_keeps_the_thoughts_of_earlier_turns():
    """Gemma 4 writes thoughts with thinking off, and its template keeps them only for a call-making reply of the
    current turn: at the next user message the previous turn was read again from its first reply."""
    from huggingface_hub import snapshot_download

    gemma = "mlx-community/gemma-4-e4b-it-4bit"
    try:
        path = snapshot_download(gemma, local_files_only=True, allow_patterns=["*.json", "*.jinja", "tokenizer*"])
    except Exception:  # noqa: BLE001
        pytest.skip("Gemma 4's tokenizer not cached")
    from mlx_lm.utils import load_tokenizer

    from locals.server import Server

    srv = Server(gemma, engines=("locals",), preload=(), log=lambda s: None)
    srv._svc["locals"] = _Svc(load_tokenizer(path))
    call = {"id": "c1", "type": "function", "function": {"name": "read", "arguments": json.dumps({"filePath": "todo.py"})}}
    turn = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "Read todo.py."},
            {"role": "assistant", "content": "", "reasoning_content": "I should read the file first.", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": "def add(x): ..."}]
    nxt = turn + [{"role": "assistant", "content": "It defines add."}, {"role": "user", "content": "Now add a test."}]
    tools = [{"type": "function", "function": {"name": "read", "description": "Read a file.", "parameters": {
        "type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}}}]
    a = srv.prepare({"model": "locals/gemma-4-e4b-it", "messages": turn, "tools": tools})
    b = srv.prepare({"model": "locals/gemma-4-e4b-it", "messages": nxt, "tools": tools})
    assert b["text"].startswith(a["text"]), (a["text"][-300:], b["text"][len(a["text"]) - 300:len(a["text"]) + 100])
    assert "I should read the file first." in b["text"]
