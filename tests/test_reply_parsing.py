"""Replies parse the way the grammar writes them: a value may hold an example call or "</think>", a
parameter that may be a string stays a string, calls to tools not offered are not calls, repeated
data is not a loop, and Ollama histories keep their call ids."""

import json

from locals import apis, oai
from locals.server import _looping

TOOLS = [{"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "content": {"type": "string"},
    "note": {"anyOf": [{"type": "string"}, {"type": "null"}]}, "n": {"type": "integer"}}}}}]


def test_a_value_may_hold_a_whole_example_call():
    example = "<tool_call>\n<function=x>\n</function>\n</tool_call>"
    text = ("<tool_call>\n<function=write>\n<parameter=path>\nt.py\n</parameter>\n<parameter=content>\n"
            f"FIXTURE = '''{example}'''\n</parameter>\n</function>\n</tool_call>")
    _, _, calls = oai.parse_reply(text, TOOLS)
    assert len(calls) == 1 and example in json.loads(calls[0]["function"]["arguments"])["content"]


def test_think_close_in_a_call_without_thinking_is_text():
    text = ("<tool_call>\n<function=write>\n<parameter=path>\na.py\n</parameter>\n<parameter=content>\n"
            "s = t.split('</think>')[1]\n</parameter>\n</function>\n</tool_call>")
    reasoning, _, calls = oai.parse_reply(text, TOOLS, thinking=False)
    assert reasoning == "" and len(calls) == 1
    reasoning, content, _ = oai.parse_reply("thoughts</think>\n\nanswer", None, thinking=True)
    assert reasoning == "thoughts" and content == "answer"
    assert oai.split_reasoning("cut mid-thought", None, True) == ("cut mid-thought", "")


def test_values_keep_the_type_the_schema_allows():
    props = TOOLS[0]["function"]["parameters"]["properties"]
    assert oai.coerce("1.0", props["note"]) == "1.0" and oai.coerce("null", props["note"]) == "null"
    assert oai.coerce("7", props["n"]) == 7 and oai.coerce("nan", props["n"]) == "nan"
    assert oai.coerce('{"a": 1}', {}) == '{"a": 1}'  # no type: the text as written


def test_calls_to_tools_not_offered_are_not_calls():
    text = "<tool_call>\n<function=rm_everything>\n</function>\n</tool_call>"
    assert oai.parse_reply(text, TOOLS)[2] == []


def test_repeated_data_is_not_a_loop():
    assert not _looping([15, 11] * 40)  # "0, 0, 0, ..." in a reply
    assert _looping([15, 11] * 150) and _looping(list(range(45)) * 3)


def test_ollama_call_ids_are_stable_and_results_pair_by_position():
    req = {"messages": [{"role": "user", "content": "go"},
                        {"role": "assistant", "content": "", "tool_calls": [
                            {"function": {"name": "a", "arguments": {}}}, {"function": {"name": "b", "arguments": {}}}]},
                        {"role": "tool", "content": "A"}, {"role": "tool", "content": "B"}]}
    one, two = apis.ollama_to_chat(req, "m"), apis.ollama_to_chat(req, "m")
    assert one == two
    ids = [tc["id"] for tc in one["messages"][1]["tool_calls"]]
    assert [m["tool_call_id"] for m in one["messages"][2:]] == ids


def test_a_raw_generate_is_a_text_completion():
    out = apis.ollama_to_text({"prompt": "def f(", "raw": True, "stream": False, "options": {"num_predict": 20}}, "m")
    assert out == {"model": "m", "prompt": "def f(", "stream": False, "max_tokens": 20}
