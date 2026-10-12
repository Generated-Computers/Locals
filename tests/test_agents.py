"""Agent harness pieces that need no model: the action grammar and parser,
the special-token grammar for llguidance, and tool sandboxing."""

import re
from pathlib import Path

import pytest

from locals.agents import ACTION_REGEX, TOOLS, AgentState, Task, _tool, parse_action


def test_action_regex_accepts_native_tool_calls_and_parses():
    call = ("<tool_call>\n<function=write_file>\n<parameter=path>\nmain.py\n</parameter>\n"
            "<parameter=content>\nprint('a < b')\nx = {\"k\": 1}\n</parameter>\n</function>\n</tool_call>")
    assert re.fullmatch(ACTION_REGEX, call)
    act = parse_action(call)
    assert act == {"tool": "write_file", "thought": "", "path": "main.py", "content": "print('a < b')\nx = {\"k\": 1}"}
    assert re.fullmatch(ACTION_REGEX, "<tool_call>\n<function=list_files>\n</function>\n</tool_call>")


def test_action_regex_rejects_unknown_tools_and_missing_parameters():
    assert not re.fullmatch(ACTION_REGEX, "<tool_call>\n<function=rm_rf>\n</function>\n</tool_call>")
    assert not re.fullmatch(ACTION_REGEX, "<tool_call>\n<function=write_file>\n<parameter=path>\na\n</parameter>\n</function>\n</tool_call>")
    assert not re.fullmatch(ACTION_REGEX, "Sure! <tool_call>\n<function=list_files>\n</function>\n</tool_call>")
    assert set(TOOLS) == {"list_files", "read_file", "write_file", "run_file", "run_python", "finish"}


def test_tools_stay_in_sandbox(tmp_path: Path):
    st = AgentState(0, Task("t", "t"), tmp_path / "a")
    st.dir.mkdir()
    assert _tool(st, {"tool": "write_file", "path": "x.py", "content": "print(6*7)"}, True).startswith("wrote")
    assert _tool(st, {"tool": "run_file", "path": "x.py"}, True) == "42"
    assert _tool(st, {"tool": "read_file", "path": "../../etc/passwd"}, True).startswith("error")
    # absolute paths are mapped into the sandbox
    assert _tool(st, {"tool": "write_file", "path": "/tmp/y.txt", "content": "hi"}, True).startswith("wrote")
    assert (st.dir / "y.txt").read_text() == "hi\n"
    assert _tool(st, {"tool": "run_python", "code": "print(1)"}, False).startswith("error")
    assert _tool(st, {"tool": "list_files"}, True) == "x.py\ny.txt"


def test_llguidance_grammar_references_special_tokens():
    llguidance = pytest.importorskip("llguidance")  # the `compare` extra

    from locals.service import _llg_grammar

    g = _llg_grammar(llguidance, ACTION_REGEX, {"<tool_call>": 248058, "</tool_call>": 248059})
    assert "<[248058]>" in g and "<[248059]>" in g
    plain = _llg_grammar(llguidance, r"[a-z]+", {"<tool_call>": 248058})
    assert "248058" not in plain
