"""A thinking reply replayed by its tool-call id renders as the model wrote it: Qwen3.8's template
keeps every turn's think block but no longer splits a reply's own "</think>" out of its content, so
the server splits it (no model: the templates of the cached tokenizers; skipped if not cached)."""

import pytest

from locals import oai

RAW = ("The file is there; run it.\n</think>\n\n<tool_call>\n<function=run>\n<parameter=command>\n"
       "python hello.py\n</parameter>\n</function>\n</tool_call>")
TOOLS = [{"type": "function", "function": {"name": "run", "description": "Run a command", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def _tok(repo):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip(f"{repo} tokenizer not cached")


def _turns():
    first = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "Run hello.py."}]
    second = first + [{"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
        "name": "run", "arguments": '{"command": "python hello.py"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "hello"}, {"role": "user", "content": "Thanks. Now list the files."}]
    return first, second


def test_qwen38_thinking_replay_continues_the_first_prompt():
    tok = _tok("mlx-community/Qwen3.8-27B-4bit")
    first, second = _turns()
    t1 = tok.apply_chat_template(oai.convert_messages(first), tools=TOOLS, add_generation_prompt=True, tokenize=False,
                                 enable_thinking=True)
    t2 = tok.apply_chat_template(oai.convert_messages(second, raw={oai.id_key("call_1"): RAW}), tools=TOOLS,
                                 add_generation_prompt=True, tokenize=False, enable_thinking=True)
    assert t1.endswith("<think>\n")
    assert t2.startswith(t1 + RAW), t2[len(t1) - 40:len(t1) + 200]


def test_qwen35_rendering_is_unchanged():
    tok = _tok("mlx-community/Qwen3.5-4B-MLX-4bit")
    _, second = _turns()
    split = oai.convert_messages(second, raw={oai.id_key("call_1"): RAW})
    whole = [dict(m) for m in split]
    whole[2] = {"role": "assistant", "content": RAW}  # what the server sent before: the template split it itself
    for msgs in (second[:3], second):  # the reply as the last turn, and before a new question
        n = len(msgs) + 1 if msgs is second[:3] else len(split)
        a = tok.apply_chat_template(split[:n], tools=TOOLS, tokenize=False, enable_thinking=True)
        b = tok.apply_chat_template(whole[:n], tools=TOOLS, tokenize=False, enable_thinking=True)
        assert a == b
