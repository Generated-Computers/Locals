"""OpenAI <-> Qwen tool-format layer (locals/oai.py): the not-containing
construction is exact, the reply grammar accepts what the template
describes and rejects malformed calls, replies parse and stream."""

import itertools
import json
import re
import random

from locals import oai
from locals.regex import compile_dfa

TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "read a file",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                                                     "required": ["path"]}}},
    {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}, "timeout": {"type": "number"}, "background": {"type": "boolean"},
        "env": {"type": "object"}, "args": {"type": "array"}}}}},
]


def test_not_containing_exact():
    for lits in (("<tool_call>",), ("</parameter>",), ("</think>",), ("<ab>",), ("<ab>", "<ac>", "<a>"), ("<tool_call>", "<function=")):
        dfa = compile_dfa(oai.not_containing(*lits))
        alphabet = sorted(set("".join(lits)) | {"x"})
        for n in range(0, 7):
            for s in itertools.product(alphabet, repeat=n):
                s = "".join(s)
                assert dfa.matches(s.encode()) == all(l not in s for l in lits), (lits, s)
        rng = random.Random(1)
        for _ in range(2000):
            s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
            assert dfa.matches(s.encode()) == all(l not in s for l in lits), (lits, s)
        assert dfa.matches("unicode ✓ < > <<".encode())


def test_reply_grammar():
    dfa = compile_dfa(oai.reply_regex(TOOLS))
    ok = ["", "just text", "Let me look.\n\n<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>",
          "<tool_call>\n<function=bash>\n<parameter=command>\nls -la | grep x < y\n</parameter>\n<parameter=timeout>\n3\n</parameter>\n</function>\n</tool_call>",
          "<tool_call>\n<function=read>\n</function>\n</tool_call>\n<tool_call>\n<function=bash>\n<parameter=command>\necho hi\n</parameter>\n</function>\n</tool_call>",
          "a < b and </think> in text is fine"]
    bad = ["<tool_call>\n<function=nope>\n</function>\n</tool_call>",  # unknown tool
           "<tool_call>\n<function=read>\n<parameter=path>\na\n</parameter>\n</function>",  # unclosed
           "<tool_call>\n<function=read>\n</function>\n</tool_call> trailing",  # suffix
           "text <tool_call> text", "<function=read>\n</function>", "text\n\n<function=read>\n<parameter=path>\na\n</parameter>\n</function>"]
    for s in ok:
        assert dfa.matches(s.encode()), s
    for s in bad:
        assert not dfa.matches(s.encode()), s
    assert dfa.n < 400
    req = compile_dfa(oai.reply_regex(TOOLS, tool_choice="required"))
    assert not req.matches(b"just text") and req.matches(ok[2].encode())
    one = compile_dfa(oai.reply_regex(TOOLS, tool_choice={"type": "function", "function": {"name": "bash"}}))
    assert not one.matches(ok[2].encode()) and one.matches(ok[3].encode())
    think = compile_dfa(oai.reply_regex(TOOLS, thinking=True))
    assert think.matches(b"I should read it.\n</think>\n\n" + ok[2].encode())
    assert not think.matches(ok[2].encode())
    obj = re.compile(oai.json_object_regex(), re.S)  # the model's usual pretty-printing, and compact
    assert obj.fullmatch('{\n  "a": 42,\n  "b": 17\n}') and obj.fullmatch('{"a":1}') and not obj.fullmatch("[1]")
    none = re.compile(oai.reply_regex(TOOLS, tool_choice="none"), re.S)
    assert none.fullmatch("Just text.") and not none.fullmatch("x<tool_call>\n<function=read>\n</function>\n</tool_call>")
    assert oai.reply_regex([]) is None


def test_parse_and_coerce():
    text = ("I'll run it.\n\n<tool_call>\n<function=bash>\n<parameter=command>\necho <a> </b>\n</parameter>\n<parameter=timeout>\n2.5\n"
            "</parameter>\n<parameter=background>\ntrue\n</parameter>\n<parameter=env>\n{\"A\": 1}\n</parameter>\n<parameter=args>\n[1, 2]\n"
            "</parameter>\n</function>\n</tool_call>")
    reasoning, content, calls = oai.parse_reply("hmm\n</think>\n\n" + text, TOOLS)
    assert reasoning == "hmm" and content == "I'll run it."
    assert len(calls) == 1 and calls[0]["function"]["name"] == "bash"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"command": "echo <a> </b>", "timeout": 2.5, "background": True, "env": {"A": 1}, "args": [1, 2]}
    # a call cut inside its function (stock engine, a length limit) is not a call: it would arrive
    # without its last arguments (a write_file without its content)
    cut = "<tool_call>\n<function=read>\n<parameter=path>\nx.py\n</parameter>\n<parameter=limit>\n5"
    _, _, calls = oai.parse_reply(cut, TOOLS)
    assert calls == []
    # one that ended after </function>, without </tool_call>, is complete
    ended = "<tool_call>\n<function=read>\n<parameter=path>\nx.py\n</parameter>\n</function>"
    _, _, calls = oai.parse_reply(ended, TOOLS)
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "x.py"}
    _, content, calls = oai.parse_reply("plain answer", TOOLS)
    assert content == "plain answer" and calls == []
    assert oai.coerce("7", {"type": "integer"}) == 7 and oai.coerce("x", {"type": "integer"}) == "x"
    assert oai.coerce("[1]", None) == "[1]"


def test_stream_holds_partial_tags():
    st = oai.ReplyStream(thinking=True)
    full = "think a bit</think>\n\nHello <b> world<tool_call>\n<function=read>\n</function>\n</tool_call>"
    got = []
    for n in range(1, len(full) + 1):
        got += st.feed(full[:n])
    reasoning = "".join(d for k, d in got if k == "reasoning")
    content = "".join(d for k, d in got if k == "content")
    assert reasoning == "think a bit" and content == "Hello <b> world"
    assert all(TOOL not in d for _, d in got for TOOL in ("<tool_call>", "</think>"))


def test_convert_messages():
    msgs = [{"role": "system", "content": "A"}, {"role": "system", "content": [{"type": "text", "text": "B"}]},
            {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "x"}}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{\"path\": \"a\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "text"}]
    out = oai.convert_messages(msgs)
    assert out[0] == {"role": "system", "content": "A\n\nB"}
    assert out[1]["content"] == "hi\n[image omitted]"
    assert out[2]["tool_calls"][0]["function"]["arguments"] == {"path": "a"}
    assert out[3]["role"] == "tool"
    assert oai.engine_of("mlx/qwen3.5-2b") == "mlx" and oai.engine_of("locals/qwen3.5-2b-thinking") == "locals"
    # only an explicit prefix picks the stock engine: a Hugging Face id or a bare id is ours
    assert oai.engine_of("mlx-community/Qwen3.5-4B-MLX-4bit") == "locals" and oai.engine_of("qwen3.5-4b") == "locals"
    assert oai.thinking_of("locals/qwen3.5-2b-thinking") and not oai.thinking_of("locals/qwen3.5-2b")
    assert oai.response_schema({"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}}) == {"type": "object"}


def test_loop_guard_any_period():
    """The server stops a reply that repeats one block of tokens three times, whatever
    the block's length (Gemma 4 looped on a 56-token cycle; the old guard only saw
    blocks of 8, 16, 32, 64 or 128 tokens), and leaves ordinary text alone."""
    from locals.server import _looping

    block = list(range(100, 156))  # 56 tokens
    reply = [1, 2, 3] + block * 3
    assert _looping(reply) and not _looping(reply[:-1]) and not _looping([1, 2, 3] + block * 2)
    assert _looping([7] * 24) and not _looping([7] * 23)  # one token over and over
    assert not _looping(list(range(1000)))
    assert not _looping([5, 6] * 11 + [9])  # a short repeat that stopped
    # without tools (plain chat) a short phrase asked for ten times is not a loop
    phrase = [11, 12, 13]  # "hello world\n"
    assert _looping(phrase * 10) and not _looping(phrase * 10, reps=6, min_span=64)
    assert _looping(block * 6, reps=6, min_span=64)


def test_request_temperature():
    """A request's temperature reaches its row: 0 is greedy whatever the server's
    default, rows of one batch may differ, and bad values are a 400."""
    import mlx.core as mx

    from locals.engine import sample_rows
    from locals.server import _temperature

    assert _temperature({}) is None and _temperature({"temperature": None}) is None
    assert _temperature({"temperature": 0}) == 0.0 and _temperature({"temperature": 0.7}) == 0.7
    assert _temperature({"temperature": 5}) == 2.0 and _temperature({"temperature": -1}) == 0.0
    try:
        _temperature({"temperature": "hot"})
        raise AssertionError("expected a ValueError")
    except ValueError:
        pass
    logits = mx.array([[0.0, 5.0, 1.0], [3.0, 0.0, 2.9]])
    assert sample_rows(logits, [0, 0]).tolist() == [1, 0]
    mx.random.seed(0)
    drawn = [sample_rows(logits, [0, 2.0]).tolist() for _ in range(64)]
    assert all(d[0] == 1 for d in drawn)  # the greedy row
    assert {d[1] for d in drawn} >= {0, 2}  # the hot row samples


def test_loop_guard_spares_call_arguments():
    """A file with a few identical lines is not a loop; free text saying the
    same thing three times is."""
    from locals import formats
    from locals.server import Server

    srv = Server.__new__(Server)
    srv.max_calls = 6
    fmt = formats.FORMATS["hermes"]
    line, head = list(range(500, 510)), list(range(100, 140))
    assert srv._guard(head + line * 3, "I will do it. I will do it. I will do it.", fmt) == "loop"
    in_call = '<tool_call>\n{"name": "write", "arguments": {"content": "a\\na\\n'
    assert srv._guard(head + line * 6, in_call, fmt) is None
    assert srv._guard(head + line * 13, in_call, fmt) == "loop"


def test_error_mid_stream_is_an_event_not_a_second_response():
    """An engine error after a streamed reply began ends the stream with an
    OpenAI-style error event; the server used to write a second HTTP response into
    it (OpenCode: "InvalidHTTPResponse", and the task ended)."""
    import http.client
    import json
    import threading

    from locals.server import _HTTPServer, make_handler

    class Stub:
        requests = errors = 0

        def log(self, s):
            pass

        def prepare(self, req):
            return {}

        def stream(self, req, write, p):
            write(b'data: {"choices": [{"delta": {"content": "Hel"}}]}\n\n')
            raise RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory")

    httpd = _HTTPServer(("127.0.0.1", 0), make_handler(Stub()))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
        c.request("POST", "/v1/chat/completions", json.dumps({"stream": True, "messages": [{"role": "user", "content": "hi"}]}),
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        body = r.read().decode()  # a corrupted chunked stream would raise here
        assert r.status == 200
        events = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
        assert events[-1] == "[DONE]" and "Insufficient Memory" in json.loads(events[-2])["error"]["message"]
    finally:
        httpd.shutdown()


def test_line_loop_guard():
    """An agent rephrasing one intention over and over (lines in changing order) is a
    loop; documents, code and numbered lists are not."""
    from locals.server import _line_looping

    loop = ("I'll use the `read` tool.\nLet's read the file.\nI'll proceed with the read call.\n"
            "Since I don't have any context, I'll read the file.\nI'll execute the read command.\n")
    assert _line_looping("thought\nThe user wants is_prime.\n" + loop * 8)
    assert not _line_looping(loop * 2)  # not yet: fewer than 30 lines
    assert not _line_looping("\n".join(f"- step {i}: check the file number {i} for errors" for i in range(60)))
    assert not _line_looping(open(__file__).read())


def _scripted_server(monkeypatch, fmt, replies, thinking):
    """A Server whose engine writes ``replies`` in turn (one token per character), and the p dict of a request."""
    from locals import server as S

    monkeypatch.setattr(S, "native_context", lambda m: None)

    class Tok:
        class _tokenizer:  # noqa: N801
            @staticmethod
            def encode(s, add_special_tokens=False):
                return [ord(c) for c in s]

        @staticmethod
        def decode(ids):
            return "".join(chr(i) for i in ids)

    class Job:
        def __init__(self, text, on_token):
            self.text, self.tokens = text, [ord(c) for c in text]
            self.done, self.finish, self.cached, self.t_submit, self.t_done = True, "stop", 0, 0.0, 0.0
            on_token(self.tokens)

        def cancel(self):
            pass

    class Svc:
        tokenizer = Tok()

        def __init__(self):
            self.replies, self.regexes = list(replies), []

        def submit(self, prompt, regex=None, on_token=None, **kw):
            self.regexes.append(regex)
            return Job(self.replies.pop(0), on_token)

    srv = S.Server("local/none", engines=("locals",), preload=(), log=lambda s: None)
    svc = Svc()
    tools = [{"type": "function", "function": {"name": n, "parameters": {"type": "object", "properties": {k: {"type": "string"}},
                                                                             "required": [k]}}} for n, k in (("bash", "command"), ("read", "filePath"))]
    tools = oai.tool_specs(tools)
    p = {"engine": "locals", "thinking": thinking, "svc": svc, "prompt": [1, 2, 3], "checkpoints": [], "text": "<prompt>", "tools": tools,
         "schema": None, "regex": oai.reply_regex(tools, "auto", thinking, fmt=fmt, preamble=False), "max_tokens": 4096,
         "model": "locals/x", "fmt": fmt, "tool_choice": "auto", "stop": [], "temp": None}
    prev = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                                 "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}
    req = {"model": "locals/x", "messages": [{"role": "user", "content": "list"}, prev, {"role": "tool", "tool_call_id": "c1", "content": "a.py"}]}
    return srv, svc, p, req


def _events(srv, req, p):
    out = []
    srv._run(req, out.append, p)
    deltas = []
    for line in b"".join(out).decode().splitlines():
        if line.startswith("data: {"):
            d = json.loads(line[6:])["choices"][0]["delta"]
            deltas.append(d)
    return deltas


def test_steered_reply_in_thinking_mode_streams_reasoning(monkeypatch):
    """Qwen3.5 with thinking on repeats its last call: the reply is decoded again
    after a steering text, which sits inside the think block the prompt opened.
    That text and the thinking after it stream as reasoning; </think> never
    reaches the content (it did: OpenCode showed the thinking as the answer)."""
    from locals import formats

    call = "<tool_call>\n<function={}>\n<parameter={}>\n{}\n</parameter>\n</function>\n</tool_call>"
    replies = ["List them again.\n</think>\n\n" + call.format("bash", "command", "ls"),
               "So I read a.py instead.\n</think>\n\n" + call.format("read", "filePath", "a.py")]
    srv, svc, p, req = _scripted_server(monkeypatch, formats.FORMATS["qwen_xml"], replies, thinking=True)
    deltas = _events(srv, req, p)
    content = "".join(d.get("content") or "" for d in deltas)
    reasoning = "".join(d.get("reasoning_content") or "" for d in deltas)
    assert "</think>" not in content and "already made" not in content and "So I read" not in content
    assert "already made" in reasoning and "So I read a.py instead." in reasoning
    calls = [d["tool_calls"][0]["function"] for d in deltas if d.get("tool_calls")]
    assert [c["name"] for c in calls] == ["read"]


def test_steered_gemma_reply_cannot_open_a_thought_block(monkeypatch):
    """Gemma 4's steered reply has begun with text, so its grammar no longer offers the
    thought block that may open a reply, and channel markers are never content
    (Gemma went on as if thinking and wrote <channel|> into its answer)."""
    from locals import formats

    fmt = formats.FORMATS["gemma4"]
    replies = ['<|channel>thought\nagain\n<channel|><|tool_call>call:bash{command:<|"|>ls<|"|>}<tool_call|>',
               'I will read it.<|tool_call>call:read{filePath:<|"|>a.py<|"|>}<tool_call|>']
    srv, svc, p, req = _scripted_server(monkeypatch, fmt, replies, thinking=True)
    deltas = _events(srv, req, p)
    lead = fmt.thought_regex()
    assert svc.regexes[0].startswith(lead) and svc.regexes[1] == svc.regexes[0][len(lead):]
    content = "".join(d.get("content") or "" for d in deltas)
    assert "already made" in content and "I will read it." in content
    assert "<channel|>" not in content and "<|channel>" not in content
    calls = [d["tool_calls"][0]["function"] for d in deltas if d.get("tool_calls")]
    assert [c["name"] for c in calls] == ["read"]
    rx = compile_dfa(fmt.free_text(), 100_000)
    assert rx.matches(b"plain words") and not rx.matches(b"thinking\n<channel|>\nmore")


def test_ignore_eos_and_usage_chunk_on_text_completions():
    """Benchmarks (InferenceX, vLLM's, aiperf) send ignore_eos for fixed output lengths and read the
    token count from a last chunk with no choices (stream_options.include_usage, as OpenAI streams it)."""
    import json
    import time
    from types import SimpleNamespace

    from locals.server import Server, _include_usage

    assert _include_usage({"stream_options": {"include_usage": True}})
    assert not _include_usage({"stream_options": "yes"}) and not _include_usage({})

    class Tok:
        def decode(self, toks):
            return "".join(chr(97 + t % 26) for t in toks)

    class Svc:
        tokenizer = Tok()

        def submit(self, prompt, max_tokens=16, on_token=None, temp=None, ignore_eos=False, **kw):
            self.asked = {"max_tokens": max_tokens, "ignore_eos": ignore_eos}
            toks = list(range(max_tokens))
            on_token(toks)
            t = time.perf_counter()
            return SimpleNamespace(done=True, finish="length", tokens=toks, text=Tok().decode(toks), cached=0,
                                   t_submit=t, t_done=t, cancel=lambda: None)

    srv = Server.__new__(Server)
    srv._active, srv.log = {}, (lambda s: None)
    srv.per_engine = {"locals": {k: 0 for k in ("requests", "prompt_tokens", "cached_tokens", "completion_tokens",
                                                 "forced_tokens", "passes", "decode_s", "tool_calls")}}
    for include in (True, False):
        svc = Svc()
        p = {"engine": "locals", "svc": svc, "prompt": [1, 2, 3], "max_tokens": 5, "model": "locals/m", "stop": [],
             "temp": 0.0, "ignore_eos": True, "include_usage": include, "regex": None, "schema": None}
        out = []
        srv.text_completion({}, p, write=out.append)
        events = [e[6:] for e in b"".join(out).decode().split("\n\n") if e.startswith("data: ")]
        assert svc.asked == {"max_tokens": 5, "ignore_eos": True}
        assert events[-1] == "[DONE]"
        chunks = [json.loads(e) for e in events[:-1]]
        assert "".join(c["choices"][0]["text"] for c in chunks if c["choices"]) == "abcde"
        last = chunks[-1]
        if include:
            assert last["choices"] == [] and last["usage"]["completion_tokens"] == 5
        else:
            assert last["choices"] and all(c["choices"] for c in chunks)


def test_keepalive_is_a_blank_line(monkeypatch):
    """A stream quiet for 15 s (a long prefill) writes a keepalive that strict clients skip: InferenceX's and
    vLLM's benchmark clients json-parse every non-empty line, and failed on the ": keepalive" comment."""
    import json
    import threading
    import time
    from types import SimpleNamespace

    from locals import server as srv_mod
    from locals.server import Server

    clock = [0.0]
    real = time.perf_counter

    def fake_clock():
        clock[0] += 4.0  # every reading 4 s later: the stream looks quiet for 15 s within a few polls
        return real() + clock[0]

    class Svc:
        tokenizer = SimpleNamespace(decode=lambda toks: "".join(chr(97 + t % 26) for t in toks))

        def submit(self, prompt, max_tokens=16, on_token=None, temp=None, ignore_eos=False, **kw):
            job = SimpleNamespace(done=False, finish="length", tokens=[0, 1], text="ab", cached=0, t_submit=0.0, t_done=0.0,
                                  cancel=lambda: None)

            def later():
                time.sleep(1.2)  # a few 0.25 s polls without tokens
                on_token([0, 1])
                job.done = True

            threading.Thread(target=later, daemon=True).start()
            return job

    monkeypatch.setattr(srv_mod.time, "perf_counter", fake_clock)
    srv = Server.__new__(Server)
    srv._active, srv.log = {}, (lambda s: None)
    srv.per_engine = {"locals": {k: 0 for k in ("requests", "prompt_tokens", "cached_tokens", "completion_tokens",
                                                 "forced_tokens", "passes", "decode_s", "tool_calls")}}
    p = {"engine": "locals", "svc": Svc(), "prompt": [1], "max_tokens": 2, "model": "locals/m", "stop": [], "temp": 0.0,
         "ignore_eos": False, "include_usage": True, "regex": None, "schema": None}
    out = []
    srv.text_completion({}, p, write=out.append)
    assert srv_mod.KEEPALIVE in out[: out.index(next(o for o in out if o.startswith(b"data: ")))]
    text = ""
    for line in b"".join(out).split(b"\n"):  # as InferenceX's client reads the stream
        line = line.strip()
        if not line:
            continue
        chunk = line.decode().removeprefix("data: ")
        if chunk != "[DONE]":
            d = json.loads(chunk)
            text += "".join(c["text"] for c in d["choices"])
    assert text == "ab"


def test_context_limit_is_what_fits_in_memory(monkeypatch):
    """With no --max-context, requests may be as long as the KV cache that fits next to the model (at
    least 64k, at most the model's own context); an explicit --max-context is kept."""
    from types import SimpleNamespace

    from locals import server as srv_mod
    from locals.server import Server
    from locals.service import BatchService

    svc = BatchService.__new__(BatchService)
    svc._eng = SimpleNamespace(memory_limit=int(8.17 * 2**30), _kv_bytes_per_token=lambda: 32768)
    svc.weights_bytes = int(2.21 * 2**30)
    fit = svc.context_fit()
    assert fit % 1024 == 0 and 140_000 < fit < 150_000  # Qwen3.5-4B on a 16 GB Mac

    def server(max_context, native, fit_tokens):
        s = Server.__new__(Server)
        s.log, s.model_id, s.engines = (lambda m: None), "m", ("locals",)
        s._auto_context = max_context is None
        s.max_context = max_context or 65536
        s._svc = {"locals": SimpleNamespace(context_fit=lambda: fit_tokens)}
        monkeypatch.setattr(srv_mod, "native_context", lambda model: native)
        s._cap_context()
        return s.max_context

    assert server(None, 262144, fit) == fit
    assert server(None, 32768, fit) == 32768  # the model's own limit
    monkeypatch.setattr(srv_mod, "native_context", lambda model: 262144)
    s = Server.__new__(Server)  # the stock engine (no memory guard) keeps the old limit
    s.log, s.model_id, s.engines = (lambda m: None), "m", ("locals", "mlx")
    s._auto_context, s.max_context, s._stock_context = True, 65536, 65536
    s._svc = {"locals": SimpleNamespace(context_fit=lambda: fit)}
    s._cap_context()
    assert s.context_for("locals") == fit and s.context_for("mlx") == 65536
    assert server(None, 262144, 20480) == 65536  # a small Mac: never below what it was
    assert server(40000, 262144, fit) == 40000  # asked for
