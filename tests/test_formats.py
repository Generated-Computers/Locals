"""Tool-call formats (locals/formats.py): each format's reply grammar accepts
what its models write and nothing else, replies parse back to OpenAI calls,
streaming holds calls back, and detection picks the format from real chat
templates (those of the models in the local cache)."""

import json
from pathlib import Path
import re

import pytest

from locals import formats, oai
from locals.regex import compile_dfa

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather.", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read.", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]
NAMES = ["get_weather", "read_file"]


def _ok(fmt, text, tool_choice="auto", thinking=False):
    rx = fmt.reply_regex(NAMES, tool_choice, thinking)
    return bool(re.fullmatch(rx, text, re.S)), compile_dfa(rx, 400_000).matches(text.encode())


@pytest.mark.parametrize("name,call,texts_ok,texts_bad", [
    ("hermes", '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}\n</tool_call>',
     ["Just text.", "Let me check.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.py\"}}\n</tool_call>"],
     ['<tool_call>\n{"name": "rm_rf", "arguments": {}}\n</tool_call>', 'x <tool_call>\n{"name": "get_weather"}\n</tool_call>']),
    ("llama3_json", '{"name": "get_weather", "parameters": {"city": "Paris", "days": 2}}',
     ["Just text.", "Sure, here it is: {not a call}", 'text then {"name": "get_weather", "parameters": {}} (text: no call)'],
     ['{"name": "rm_rf", "parameters": {}}', '{"foo": 1}']),
    ("mistral", '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "read_file", "arguments": {"path": "a"}}]',
     ["Just text."],
     ['[TOOL_CALLS] [{"name": "rm", "arguments": {}}]', "[1, 2]",
      'Sure. [TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]']),  # 7B's calls start the reply
    ("mistral_v11", '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}[TOOL_CALLS]read_file[ARGS]{"path": "a"}',
     ["Just text.", "[1, 2] is a list.", 'Let me look.\n[TOOL_CALLS]read_file[ARGS]{"path": "a.py"}'],
     ['[TOOL_CALLS]rm[ARGS]{}', 'Text first. [TOOL_CALLS]rm[ARGS]{}']),
    ("gemma4", '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:2}<tool_call|>',
     ["Just text.", 'Checking.<|tool_call>call:read_file{path:<|"|>a "b" c.py<|"|>}<tool_call|>'],
     ['<|tool_call>call:rm_rf{}<tool_call|>', 'x<|tool_call>call:get_weather{city:Paris}<tool_call|>']),
])
def test_grammar_and_parse(name, call, texts_ok, texts_bad):
    fmt = formats.FORMATS[name]
    for t in [call] + texts_ok:
        assert _ok(fmt, t) == (True, True), t
    for t in texts_bad:
        assert _ok(fmt, t) == (False, False), t
    assert _ok(fmt, call, "required") == (True, True)
    assert _ok(fmt, "Just text.", "required") == (False, False)
    assert _ok(fmt, "Just text.", "none") == (True, True) and _ok(fmt, call, "none") == (False, False)
    reasoning, content, calls = oai.parse_reply(call, TOOLS, fmt=fmt)
    assert calls and calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"])["city"] == "Paris"
    assert not content.strip()
    r, c, none = oai.parse_reply("Just text.", TOOLS, fmt=fmt)
    assert c == "Just text." and not none


def test_grammar_sizes():
    """The reply automata stay small enough for token tables (the engine's default path)."""
    for name in ("hermes", "llama3_json", "mistral", "mistral_v11", "gemma4"):
        n = compile_dfa(formats.FORMATS[name].reply_regex(NAMES + [f"tool_{i}" for i in range(20)]), 400_000).n
        assert n < 1500, (name, n)


def test_thinking_hermes():
    fmt = formats.FORMATS["hermes"]
    text = 'Need the weather.</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome"}}\n</tool_call>'
    assert _ok(fmt, text, thinking=True) == (True, True)
    reasoning, content, calls = oai.parse_reply("<think>\n" + text, TOOLS, fmt=fmt)
    assert reasoning == "Need the weather." and calls[0]["function"]["name"] == "get_weather"


def test_reply_stream_per_format():
    def run(fmt, text, thinking=False):
        rs, out = oai.ReplyStream(thinking, fmt), []
        for i in range(1, len(text) + 1):
            out += rs.feed(text[:i])
        return "".join(d for k, d in out if k == "content"), "".join(d for k, d in out if k == "reasoning")

    llama = formats.FORMATS["llama3_json"]
    assert run(llama, '{"name": "get_weather", "parameters": {}}') == ("", "")
    assert run(llama, "Hello there.")[0] == "Hello there."
    hermes = formats.FORMATS["hermes"]
    assert run(hermes, 'Checking.\n<tool_call>\n{"name": "x", "arguments": {}}\n</tool_call>')[0] == "Checking.\n"
    assert run(hermes, "<think>\nhmm\n</think>\n\nAnswer.", thinking=True) == ("Answer.", "\nhmm\n")
    mistral = formats.FORMATS["mistral_v11"]
    assert run(mistral, '[TOOL_CALLS]x[ARGS]{}')[0] == ""


def test_fallback_prepare():
    fmt = formats.FORMATS["fallback"]
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                                 "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "17C"}, {"role": "tool", "tool_call_id": "c2", "content": "sunny"}]
    out, tools = fmt.prepare(msgs, oai.tool_specs(TOOLS))
    assert tools is None and out[0]["role"] == "system" and "get_weather" in out[0]["content"] and "Be brief." in out[0]["content"]
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]  # alternation kept, tool results merged
    assert '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>' in out[2]["content"]
    assert "17C" in out[3]["content"] and "sunny" in out[3]["content"]


def test_mistral_ids_and_llama_single_calls():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "call_abcdef123456", "type": "function",
                                                                "function": {"name": "a", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "call_abcdef123456", "content": "ok"}]
    out, _ = formats.FORMATS["mistral"].prepare(msgs, [])
    assert re.fullmatch(r"[A-Za-z0-9]{9}", out[0]["tool_calls"][0]["id"]) and out[0]["tool_calls"][0]["id"] == out[1]["tool_call_id"]
    assert re.fullmatch(r"[A-Za-z0-9]{9}", formats.FORMATS["mistral"].new_id())
    two = [{"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "a", "arguments": {}}},
                                                               {"id": "2", "function": {"name": "b", "arguments": {}}}]}]
    out, _ = formats.FORMATS["llama3_json"].prepare(two, [])
    assert [len(m["tool_calls"]) for m in out] == [1, 1]


@pytest.mark.parametrize("model,expected", [
    ("mlx-community/Qwen3.5-2B-MLX-4bit", "qwen_xml"),
    ("mlx-community/Qwen2.5-1.5B-Instruct-4bit", "hermes"),
    ("mlx-community/Qwen3-4B-Instruct-2507-4bit", "hermes"),
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "llama3_json"),
    ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", "mistral_legacy"),
    ("mlx-community/Ministral-3-3B-Instruct-2512-4bit", "mistral_v11"),
    ("mlx-community/gemma-3-4b-it-qat-4bit", "fallback"),
    ("mlx-community/gemma-4-e4b-it-4bit", "gemma4"),
    ("mlx-community/Llama-3.1-8B-Instruct-4bit", "llama3_json"),
    ("mlx-community/Qwen3-8B-4bit", "hermes"),
])
def test_detect_from_real_templates(model, expected):
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(model, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    from mlx_lm.utils import load_tokenizer

    assert formats.detect(load_tokenizer(model)).name == expected


def test_mistral_user_after_tool_result_renders():
    """Mistral templates raise when a user turn follows tool results (OpenCode sends that):
    the format inserts an assistant turn there."""
    from huggingface_hub import snapshot_download

    model = "mlx-community/Ministral-3-3B-Instruct-2512-4bit"
    try:
        snapshot_download(model, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(model)
    fmt = formats.detect(tok)
    call = {"id": "call1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "a"})}}
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call1", "content": "text"}, {"role": "user", "content": "and now?"}]
    m, tools = fmt.prepare(oai.convert_messages(msgs), oai.tool_specs(TOOLS))
    text = tok.apply_chat_template(m, tools=tools, add_generation_prompt=True, tokenize=False)
    assert "[TOOL_RESULTS]" in text and text.rstrip().endswith("[INST]and now?[/INST]")


RAW_CALLS = {  # one call to get_weather as each format's models write it
    "qwen_xml": "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
    "hermes": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
    "llama3_json": '{"name": "get_weather", "parameters": {"city": "Paris"}}',
    "mistral": '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]',
    "mistral_legacy": '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]',
    "mistral_v11": '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}',
    "gemma4": '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>',
    "fallback": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
}


@pytest.mark.parametrize("model", [
    "mlx-community/Qwen3.5-2B-MLX-4bit", "mlx-community/Qwen3-4B-Instruct-2507-4bit", "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit", "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
    "mlx-community/gemma-3-4b-it-qat-4bit", "mlx-community/gemma-4-e4b-it-4bit",
])
def test_tool_results_reach_the_prompt(model):
    """The server's path for a tool round trip: the model's call is parsed, the
    client sends it back with the result, and the next prompt is rendered (the
    call replayed as the model wrote it where the format allows). The result
    must be in that prompt, after the call, for every template (Gemma 4's writes
    results only after an assistant turn's tool_calls)."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(model, local_files_only=True)
    except Exception:
        pytest.skip("model not cached")
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(model)
    fmt = formats.detect(tok)
    raw_text = RAW_CALLS[fmt.name]
    _, content, calls = oai.parse_reply(raw_text, TOOLS, fmt=fmt)
    assert calls and json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    raw = {oai.id_key(c["id"]): raw_text for c in calls}
    msgs = [{"role": "system", "content": "You are a weather assistant."}, {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": content or None, "tool_calls": calls},
            {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"city": "Paris", "temp_c": 17, "sky": "drizzle"})}]
    for follow_up in (False, True):
        m = msgs + ([{"role": "user", "content": "And tomorrow?"}] if follow_up else [])
        conv = oai.convert_messages(m, raw=raw if fmt.replay_raw else None)
        conv, tools = fmt.prepare(conv, oai.tool_specs(TOOLS) if fmt.native_tools else oai.tool_specs(TOOLS))
        kw = {} if formats.thinking_style(tok) == "none" else {"enable_thinking": False}
        text = tok.apply_chat_template(conv, tools=tools, add_generation_prompt=True, tokenize=False, **kw)
        assert "drizzle" in text and "17" in text, (model, text[-600:])
        assert text.rfind("drizzle") > text.find("Weather in Paris?"), model


def test_closing_tags_in_free_text_are_not_calls():
    """A model that writes a closing tag in its text (no call opened) has made no call:
    the server's max_calls stop counts complete calls only."""
    call = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    for name in ("hermes", "fallback"):
        fmt = formats.FORMATS[name]
        text = "I would write </tool_call> here and </tool_call> there."
        assert fmt.count_complete(text) == 0 and fmt.cut_after(text, 1) == text
        assert fmt.count_complete(text + "\n" + call) == 1
        assert fmt.cut_after(text + "\n" + call + "\n" + call, 1) == text + "\n" + call
    g = formats.FORMATS["gemma4"]
    gcall = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
    assert g.count_complete("text <tool_call|> more") == 0 and g.count_complete("x" + gcall + gcall) == 2


@pytest.mark.parametrize("name,text,n_calls", [
    ("llama3_json", '{"error": "I cannot do that"}', 0),
    ("llama3_json", "{not json at all} but text", 0),
    ("llama3_json", 'Let me check.\n{"name": "get_weather", "parameters": {"city": "Paris"}}', 1),
    ("llama3_json", 'Plain answer: use {"name": "x"} carefully, it is not a call. Done.', 0),
    ("mistral_v11", "[1, 2, 3] are the numbers.", 0),
    ("mistral_v11", 'Let me fix it.\n[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}', 1),  # Ministral's text, then its call
    ("gemma4", 'call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>', 1),
    ("gemma4", "calling all stations: the weather is fine.", 0),
    ("gemma4", "call: me later, it is fine.", 0),
    ("gemma4", 'Checking.<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>', 1),
    ("hermes", "Just text.", 0),
])
def test_stream_shows_what_parse_returns(name, text, n_calls):
    """Streamed content plus what the server sends at the end of the reply (text the
    stream held back because it might have been a call) is the parsed content, and
    no call text leaks into it. Gemma 4 sometimes writes a call without <|tool_call>:
    that is a call too."""
    fmt = formats.FORMATS[name]
    rs = oai.ReplyStream(False, fmt)
    for i in range(1, len(text) + 1):
        rs.feed(text[:i])
    _, content, calls = oai.parse_reply(text, TOOLS, fmt=fmt)
    assert len(calls) == n_calls
    shown = rs.content_sent + (rs.unsent(content) if not calls else "")
    assert shown.rstrip() == content.rstrip()


@pytest.mark.parametrize("name", ["hermes", "fallback", "llama3_json", "mistral", "mistral_v11", "gemma4", None])
def test_server_required_call_starts_the_reply(name):
    """The server asks for calls only when a call is required or named (a small
    model given room to talk first can talk to the token limit: Gemma 4 under
    smolagents did); the library keeps text before calls where the format has it."""
    fmt = formats.FORMATS[name] if name else None
    call = {"hermes": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
            "fallback": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
            "llama3_json": '{"name": "get_weather", "parameters": {"city": "Paris"}}',
            "mistral": '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]',
            "mistral_v11": '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}',
            "gemma4": '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>',
            None: "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"}[name]
    for choice in ("required", {"type": "function", "function": {"name": "get_weather"}}):
        rx = oai.reply_regex(TOOLS, choice, fmt=fmt, preamble=False)
        assert compile_dfa(rx, 400_000).matches(call.encode())
        assert not compile_dfa(rx, 400_000).matches(("Let me check. " + call).encode())


@pytest.mark.parametrize("name", ["hermes", "mistral", "mistral_v11", "gemma4", None])
def test_parallel_tool_calls_false_allows_one_call(name):
    fmt = formats.FORMATS[name] if name else None
    two = {"hermes": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "A"}}\n</tool_call>\n<tool_call>\n{"name": "read_file", "arguments": {"path": "b"}}\n</tool_call>',
           "mistral": '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "A"}}, {"name": "read_file", "arguments": {"path": "b"}}]',
           "mistral_v11": '[TOOL_CALLS]get_weather[ARGS]{"city": "A"}[TOOL_CALLS]read_file[ARGS]{"path": "b"}',
           "gemma4": '<|tool_call>call:get_weather{city:<|"|>A<|"|>}<tool_call|><|tool_call>call:read_file{path:<|"|>b<|"|>}<tool_call|>',
           None: "<tool_call>\n<function=get_weather>\n<parameter=city>\nA\n</parameter>\n</function>\n</tool_call>\n"
                 "<tool_call>\n<function=read_file>\n<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>"}[name]
    for choice in ("auto", "required"):
        assert compile_dfa(oai.reply_regex(TOOLS, choice, fmt=fmt, preamble=False), 400_000).matches(two.encode())
        assert not compile_dfa(oai.reply_regex(TOOLS, choice, fmt=fmt, preamble=False, parallel=False), 400_000).matches(two.encode())


@pytest.mark.parametrize("text,name", [
    ('```tool_call\n{"name": "read_file", "arguments": {"path": "main.py"}}\n```', "read_file"),
    ('```json\n{\n  "tool_call": {\n    "name": "get_weather",\n    "arguments": {"city": "Paris"}\n  }\n}\n```', "get_weather"),
    ('I will read it first.\n```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```', "read_file"),
    ('{"name": "get_weather", "arguments": {"city": "Oslo"}}', "get_weather"),
    ('```json\n{"command": "ls", "timeout": 30}\n```', None),          # no tool named: text
    ('```json\n{"name": "rm_rf", "arguments": {}}\n```', None),          # not an offered tool: text
    ('Here is the code:\n```python\nprint(1)\n```\nRun it.', None),
])
def test_fallback_takes_json_block_calls(text, name):
    """Models without tool training (the prompt-described format, Gemma 3) write
    their calls as JSON blocks: one that ends the reply and names an offered tool
    is a call; the stream never shows it, and text stays text."""
    fmt = formats.FORMATS["fallback"]
    rs = oai.ReplyStream(False, fmt)
    for i in range(1, len(text) + 1):
        rs.feed(text[:i])
    _, content, calls = oai.parse_reply(text, TOOLS, fmt=fmt)
    assert [c["function"]["name"] for c in calls] == ([name] if name else [])
    shown = rs.content_sent + (rs.unsent(content) if not calls else "")
    assert shown.rstrip() == content.rstrip() and "{" not in rs.content_sent.replace("print(1)", "")


def test_gemma4_text_before_calls_is_replayed_as_a_thought():
    """Gemma 4's template writes an assistant message's text after its calls' results and
    then closes the model's turn; text written before calls goes in the thought channel
    (written before the calls) so that the turn stays open."""
    fmt = formats.FORMATS["gemma4"]
    call = {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"filePath": "a.py"}'}}
    msgs = [{"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "Let me read it first.", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": "x = 1"},
            {"role": "assistant", "content": " ", "reasoning_content": "It reads fine.", "tool_calls": [dict(call, id="c2")]},
            {"role": "tool", "tool_call_id": "c2", "content": "x = 1"},
            {"role": "assistant", "content": "Done: it was fine."}]
    out, _ = fmt.prepare(msgs, [])
    assert out[1]["content"] == "" and out[1]["reasoning_content"] == "Let me read it first."
    assert out[3]["content"] == " " and out[3]["reasoning_content"] == "It reads fine."  # nothing to move
    assert out[5] == msgs[5]  # a final answer keeps its text
    msgs[3] = dict(msgs[3], content="Again, to be sure.")
    out, _ = fmt.prepare(msgs, [])
    assert out[3]["reasoning_content"] == "It reads fine.\n\nAgain, to be sure." and out[3]["content"] == ""


QWEN38_EFFORT = """{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}{{ raise_exception('Unexpected reasoning effort') }}{%- endif %}"""


def test_reasoning_effort_maps_to_the_templates_values():
    from locals.formats import effort_for, reasoning_efforts

    allowed = reasoning_efforts(QWEN38_EFFORT)
    assert allowed == ["xhigh", "medium", "low"]
    assert [effort_for(a, allowed) for a in ("minimal", "low", "medium", "high", "xhigh", "HIGH")] == \
        ["low", "low", "medium", "xhigh", "xhigh", "xhigh"]
    assert reasoning_efforts("{{ messages }}") is None  # no switch: the request's value is not passed on
    assert effort_for("high", ["low", "medium", "high"]) == "high"  # gpt-oss's own scale


def test_minicpm5_calls():
    """MiniCPM5 writes <function name="..."><param name="...">...</param></function>; a value with <, & or a
    newline goes in CDATA. Without the format it got the fallback form and made no call in OpenCode."""
    from locals.formats import FORMATS, detect_from_template
    from locals.regex import compile_nfa, determinize, trim_and_minimize

    tpl = ('{%- if tools %}<function name="function-name"><param name="param-name">param-value</param></function>'
           "{%- endif %}")
    f = detect_from_template(tpl)
    assert f is FORMATS["minicpm_xml"]
    d = trim_and_minimize(determinize(compile_nfa(f.reply_regex(["bash", "write"], "auto"))))
    good = ('Let me look.\n<function name="bash"><param name="command">ls -la</param></function>\n'
            '<function name="write"><param name="content"><![CDATA[print("<hi>")\nx = 1 & 2 ]] ]]]]></param></function>')
    assert d.matches(good.encode())
    assert not d.matches(b'<function name="rm"><param name="x">1</param></function>')
    assert not d.matches(b'<function name="write"><param name="c"><![CDATA[a]]>b]]></param></function>')
    free, part = f.split(good)
    assert free == "Let me look.\n"
    assert f.parse_calls(part) == [("bash", {"command": "ls -la"}), ("write", {"content": 'print("<hi>")\nx = 1 & 2 ]] ]]'})]
    assert f.count_complete(good) == 2 and f.cut_after(good, 1).endswith("ls -la</param></function>")


def test_flint_calls():
    """Flint writes <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>, a non-string value as
    JSON, and opens its own <think> block. The template's <tool_call> alone made it the hermes form, whose grammar
    held Flint to JSON calls."""
    from locals.formats import FORMATS, detect_from_template
    from locals.regex import compile_nfa, determinize, trim_and_minimize

    tpl = ("For each function call, output the function name and arguments within the following XML format:\n"
           "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>")
    assert detect_from_template(tpl) is FORMATS["hermes"]  # (GLM's template says the same: it stays as it was)
    tpl = "{%- macro flint_text(t) -%}{{ t }}{%- endmacro -%}" + tpl
    f = detect_from_template(tpl)
    assert f is FORMATS["flint_xml"]
    d = trim_and_minimize(determinize(compile_nfa(f.reply_regex(["get_weather", "write"], "auto"))))
    good = ("<think>The user wants the weather. I could write <tool_call>x</tool_call> here: it is thought.</think>"
            "Let me check.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
            "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"
            "<tool_call>write<arg_key>content</arg_key><arg_value>print('</tool_call>')\nx = 1 < 2</arg_value></tool_call>")
    assert d.matches(good.encode())
    assert d.matches(b"<think>Easy.</think>It is sunny.") and d.matches(b"It is sunny.")
    assert not d.matches(good.replace("get_weather", "rm_rf").encode())
    assert not d.matches(b"It is <think>sunny</think>.")  # a thought block only opens the reply
    assert not d.matches(b'<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>')  # the hermes form
    tools = TOOLS + [{"type": "function", "function": {"name": "write", "parameters": {
        "type": "object", "properties": {"content": {"type": "string"}}}}}]
    reasoning, content, calls = oai.parse_reply(good, tools, fmt=f)
    assert reasoning.startswith("The user wants the weather.") and content == "Let me check."
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls] == [
        ("get_weather", {"city": "Paris", "days": 3}), ("write", {"content": "print('</tool_call>')\nx = 1 < 2"})]
    assert f.count_complete(good) == 2 and f.count_complete(good[:good.rfind("<tool_call>") + 20]) == 1  # streaming
    assert f.cut_after(good, 1).endswith("<arg_value>3</arg_value></tool_call>")
    r, c, cut = oai.parse_reply("<think>Still thinking when the token limit", tools, fmt=f)
    assert (r, c, cut) == ("Still thinking when the token limit", "", [])


def test_flint_template_renders_calls_the_grammar_writes():
    """The chat template writes an earlier turn's calls in the form the grammar gives the model now, so a replayed
    turn and a new reply look alike to it (a mismatch teaches the model the other form mid-conversation)."""
    try:
        from huggingface_hub import snapshot_download
        from mlx_lm.utils import load_tokenizer

        tok = load_tokenizer(Path(snapshot_download("mlx-community/flint-mixed-3-4bit", local_files_only=True)))
    except Exception:  # noqa: BLE001
        pytest.skip("Flint tokenizer not cached")
    from locals.regex import compile_nfa, determinize, trim_and_minimize

    f = formats.detect(tok)
    assert f is formats.FORMATS["flint_xml"] and formats.thinking_style(tok) == "model"
    msgs = [{"role": "user", "content": "Weather in Paris for 3 days?"},
            {"role": "assistant", "content": "", "reasoning_content": "Call the tool.",
             "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}}]}]
    text = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False)
    reply = text[text.rindex("<|assistant|>") + len("<|assistant|>"):]
    d = trim_and_minimize(determinize(compile_nfa(f.reply_regex(NAMES, "auto"))))
    assert d.matches(reply.encode()), reply
    from locals.vocab import TokenVocab

    v = TokenVocab.from_hf(tok._tokenizer, len(tok._tokenizer), tok.eos_token_ids)
    ids = tok._tokenizer.convert_tokens_to_ids(["<tool_call>", "<arg_key>", "</arg_value>", "<think>", "</think>"])
    assert [v.token_bytes[i] for i in ids] == [b"<tool_call>", b"<arg_key>", b"</arg_value>", b"<think>", b"</think>"]
    assert v.token_bytes[tok._tokenizer.convert_tokens_to_ids("<|observation|>")] is None  # a control token stays out


def test_lfm2_python_calls():
    """LFM2.5 writes a reply's calls as one Python-style list between <|tool_call_start|> and <|tool_call_end|>:
    strings in single quotes with backslash escapes, Python numbers and None, lists and objects as JSON."""
    from locals import oai
    from locals.formats import FORMATS, detect_from_template
    from locals.regex import compile_nfa, determinize, trim_and_minimize

    f = detect_from_template('{{- "<|tool_call_start|>[" + calls + "]<|tool_call_end|>" -}}')
    assert f is FORMATS["lfm2_python"]
    reply = ("The user wants the weather.</think>Let me check.<|tool_call_start|>[get_weather(city='Paris', days=3, units=None), "
             "run(cmd='echo \\'hi\\'\\nls', opts={\"a\": [1, true], \"b\": null})]<|tool_call_end|>")
    d = trim_and_minimize(determinize(compile_nfa(f.reply_regex(["get_weather", "run"], "auto", thinking=True))))
    assert d.matches(reply.encode()) and d.matches(b"Thinking.</think>It is sunny in Paris.")
    assert not d.matches(reply.replace("run(", "rm(").encode())
    tools = [{"type": "function", "function": {"name": n, "parameters": {"type": "object", "properties": {}}}} for n in ("get_weather", "run")]
    reasoning, content, calls = oai.parse_reply(reply, tools, fmt=f, thinking=True)
    assert reasoning == "The user wants the weather." and content == "Let me check."
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls] == [
        ("get_weather", {"city": "Paris", "days": 3, "units": None}), ("run", {"cmd": "echo 'hi'\nls", "opts": {"a": [1, True], "b": None}})]
    assert f.count_complete(reply) == 2 and f.count_complete(reply[:reply.find("run(") + 12]) == 1  # streaming
    assert f.cut_after(reply, 1).endswith("units=None)]<|tool_call_end|>")


def test_minicpm5_call_tokens_are_text_a_grammar_may_write():
    """MiniCPM5 writes <function, </function>, <param and </param> as special tokens (ids 18-21): left out of the
    grammar's vocabulary, a call could only be spelled letter by letter, and the model answered in prose."""
    import pytest

    try:
        from huggingface_hub import snapshot_download
        from mlx_lm.utils import load_tokenizer

        tok = load_tokenizer(Path(snapshot_download("mlx-community/MiniCPM5-2B-mlx-4Bit", local_files_only=True)))
    except Exception:  # noqa: BLE001
        pytest.skip("MiniCPM5 tokenizer not cached")
    from locals.vocab import TokenVocab

    v = TokenVocab.from_hf(tok._tokenizer, len(tok._tokenizer), tok.eos_token_ids)
    ids = tok._tokenizer.convert_tokens_to_ids(["<function", "</function>", "<param", "</param>"])
    assert [v.token_bytes[i] for i in ids] == [b"<function", b"</function>", b"<param", b"</param>"]
    assert v.token_bytes[tok._tokenizer.convert_tokens_to_ids("<|im_end|>")] is None  # a control token stays out


def test_lfm2_reads_every_call_its_grammar_allows():
    """LFM2's grammar lets the model write a tool's own name and any parameter name; Python's parser dropped
    Context7's resolve-library-id(...) and Grep(-n=True), and the call came back as the reply's text."""
    from locals.regex import compile_dfa

    f = formats.FORMATS["lfm2_python"]
    reply = ("<|tool_call_start|>[resolve-library-id(libraryName='react'), Grep(pattern='a, (b)', -n=True), "
             "run(cmd='x=1', n=-3, opts={\"a\": [1, true]})]<|tool_call_end|>")
    assert compile_dfa(f.reply_regex(["resolve-library-id", "Grep", "run"], "auto", False), 200_000).matches(reply.encode())
    assert f.parse_calls(reply) == [("resolve-library-id", {"libraryName": "react"}),
                                    ("Grep", {"pattern": "a, (b)", "-n": True}),
                                    ("run", {"cmd": "x=1", "n": -3, "opts": {"a": [1, True]}})]
    assert f.parse_calls("<|tool_call_start|>[run('positional')]<|tool_call_end|>") == []


def test_flint_typed_calls():
    """With the tools' schemas (the server's --typed-tools), Flint's grammar writes each call's parameter names
    itself: in the schema's order, the required ones present, each value of its type. The untyped grammar takes any
    name, as the model card's failing OpenCode calls had them (``path`` for ``filePath``)."""
    f = formats.FORMATS["flint_xml"]
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {
                 "filePath": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["filePath"]}}},
             {"type": "function", "function": {"name": "edit", "parameters": {"type": "object", "properties": {
                 "filePath": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"},
                 "replaceAll": {"type": "boolean"}, "mode": {"enum": ["fast", "safe"]}, "lines": {"type": "array", "items": {"type": "integer"}}},
                 "required": ["filePath", "oldString", "newString"]}}}]
    typed = compile_dfa(oai.reply_regex(tools, "auto", typed=True, fmt=f), 400_000)
    free = compile_dfa(oai.reply_regex(tools, "auto", fmt=f), 400_000)

    def call(name, *kv):
        return "<tool_call>" + name + "".join(f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in kv) + "</tool_call>"

    good = ("<think>Read it first.</think>" + call("read", ("filePath", "/a.py"), ("limit", 40)) +
            call("edit", ("filePath", "/a.py"), ("oldString", "x = 1 < 2"), ("newString", "x = 2"), ("replaceAll", "true"),
                 ("mode", "safe"), ("lines", "[1, 2]")))
    assert typed.matches(good.encode()) and free.matches(good.encode())
    assert typed.matches(b"No call needed.") and typed.matches(call("read", ("filePath", "/a.py")).encode())
    bad = [call("read", ("path", "/a.py")),  # the card's wrong name
           call("read", ("limit", "40")),  # the required filePath left out
           call("read", ("limit", "40"), ("filePath", "/a.py")),  # out of the schema's order
           call("read", ("filePath", "/a.py"), ("offset", "ten")),  # not an integer
           call("edit", ("filePath", "/a"), ("oldString", "a"), ("newString", "b"), ("mode", "quick")),  # not in the enum
           call("edit", ("filePath", "/a"), ("oldString", "a"), ("newString", "b"), ("lines", "1, 2")),  # not an array
           call("write", ("filePath", "/a"))]  # no such tool
    assert not any(typed.matches(b.encode()) for b in bad)
    assert free.matches(bad[0].encode()) and free.matches(bad[3].encode())
    _, _, calls = oai.parse_reply(good, tools, fmt=f)
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [
        {"filePath": "/a.py", "limit": 40},
        {"filePath": "/a.py", "oldString": "x = 1 < 2", "newString": "x = 2", "replaceAll": True, "mode": "safe", "lines": [1, 2]}]
    one = compile_dfa(oai.reply_regex(tools, {"type": "function", "function": {"name": "read"}}, typed=True, fmt=f), 400_000)
    assert one.matches(call("read", ("filePath", "/a.py")).encode())
    assert not one.matches(call("edit", ("filePath", "/a"), ("oldString", "a"), ("newString", "b")).encode())
