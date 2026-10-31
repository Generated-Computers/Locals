"""LFM3 (locals/models/lfm3.py): the model loads on an mlx-lm without it, its partial RoPE is Liquid's, and a cache
(convolution state, sliding-window ring, full keys) continues a prompt as one pass over it would, alone or in a
left-padded batch; on a small random LFM3 (no weights to download)."""

import importlib
import sys

import mlx.core as mx
import mlx.nn as nn
import pytest

from locals.models import lfm3

KINDS = ["conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "sliding_attention", "conv"]


def _args(**kw):
    a = dict(model_type="lfm3", vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=len(KINDS),
             num_attention_heads=4, num_key_value_heads=2, head_dim=32, layer_types=KINDS, conv_L_cache=7,
             sliding_window=5, norm_eps=1e-5, conv_bias=False, use_sandwich_norm=True, tie_word_embeddings=True,
             rope_parameters={"full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1e6, "rope_type": "default"},
                              "sliding_attention": {"rope_theta": 10000, "rope_type": "default"}})
    a.update(kw)
    return lfm3.ModelArgs(**a)


def _model(seed=0):
    """Float32 random weights, norms near one: a cache and a full pass then agree to rounding."""
    mx.random.seed(seed)
    m = lfm3.Model(_args())
    m.update(nn.utils.tree_map(lambda p: mx.random.normal(p.shape) * 0.08, m.parameters()))
    for layer in m.layers:
        norms = [layer.operator_norm, layer.ffn_norm, layer.post_operator_norm, layer.post_ffn_norm]
        if layer.is_attention_layer:
            norms += [layer.self_attn.q_norm, layer.self_attn.k_norm]
        for n in norms:
            n.weight = 1 + n.weight
    m.model.embedding_norm.weight = 1 + m.model.embedding_norm.weight
    mx.eval(m.parameters())
    return m


def test_register_puts_lfm3_where_mlx_lm_and_transformers_look():
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from locals.models import register

    saved = sys.modules.pop("mlx_lm.models.lfm3", None)
    try:
        register()
        mod = importlib.import_module("mlx_lm.models.lfm3")
        assert hasattr(mod, "Model") and hasattr(mod, "ModelArgs")
        # a config class of its own: AutoConfig neither asks whether to run Liquid's code nor runs it
        assert "lfm3" in CONFIG_MAPPING and CONFIG_MAPPING["lfm3"].model_type == "lfm3"
    finally:
        if saved is not None:
            sys.modules["mlx_lm.models.lfm3"] = saved


def _liquid_rope(x, pos, hd, theta, factor):
    """modeling_lfm3.py's RoPE: the zero-tail inverse frequencies, rotate_half, in float32."""
    turned = int(factor * hd) // 2
    inv = 1.0 / (theta ** (mx.arange(0, 2 * turned, 2, dtype=mx.float32) / hd))
    inv = mx.concatenate([inv, mx.zeros((hd // 2 - turned,))])
    f = pos[..., None].astype(mx.float32) * inv  # (B, L, hd / 2)
    emb = mx.concatenate([f, f], axis=-1)[:, None]  # (B, 1, L, hd)
    half = hd // 2
    rot = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * mx.cos(emb) + rot * mx.sin(emb)


@pytest.mark.parametrize("kind", ["full_attention", "sliding_attention"])
def test_rope_is_liquids_partial_and_full(kind):
    a = _args()
    attn = lfm3.Attention(a, KINDS.index(kind))
    p = a.rope_parameters[kind]
    x = mx.random.normal((2, 4, 6, 32), key=mx.random.key(3))
    for offset in (0, 37):
        pos = mx.broadcast_to(mx.arange(6) + offset, (2, 6))
        ref = _liquid_rope(x, pos, 32, p["rope_theta"], p.get("partial_rotary_factor", 1.0))
        assert float(mx.abs(attn.rope(x, offset) - ref).max()) < 1e-4
    offsets = mx.array([3, 50])  # a batch's rows at their own positions
    ref = _liquid_rope(x, mx.arange(6)[None] + offsets[:, None], 32, p["rope_theta"], p.get("partial_rotary_factor", 1.0))
    assert float(mx.abs(attn.rope(x, offsets) - ref).max()) < 1e-4
    if kind == "full_attention":  # the second half of each pair's frequencies stays where it is
        same = attn.rope(x, 37)
        assert mx.array_equal(same[..., 8:16], x[..., 8:16]) and mx.array_equal(same[..., 24:], x[..., 24:])


def test_a_cache_continues_a_prompt_as_one_pass_does():
    """Chunks, then tokens one at a time, past the window (5) and the convolution's width (7)."""
    from mlx_lm.models.cache import make_prompt_cache

    m = _model(seed=1)
    seq = mx.random.randint(0, 512, (1, 23), key=mx.random.key(7))
    full = m(seq)[0]
    c = make_prompt_cache(m)
    assert [type(x).__name__ for x in c[:4]] == ["ArraysCache", "KVCache", "ArraysCache", "RotatingKVCache"]
    got = [m(seq[:, :9], cache=c)[0], m(seq[:, 9:13], cache=c)[0]]
    got += [m(seq[:, i:i + 1], cache=c)[0] for i in range(13, 23)]
    assert float(mx.abs(mx.concatenate(got) - full).max()) <= 1e-4 * (1 + float(mx.abs(full).max()))


def test_batch_rows_decode_as_alone_with_left_padding():
    from mlx_lm.models.cache import make_prompt_cache

    m = _model(seed=2)
    seqs = [mx.random.randint(0, 512, (n,), key=mx.random.key(40 + n)).tolist() for n in (17, 6)]
    steps = [3, 77, 401, 9, 250, 18, 6]  # the shorter row passes its window

    def alone(s):
        c = make_prompt_cache(m)
        mx.eval(m(mx.array([s]), cache=c))
        return [m(mx.array([[t]]), cache=c)[0, -1] for t in steps]

    refs = [alone(s) for s in seqs]
    rows = []
    for s in seqs:
        c = make_prompt_cache(m)
        mx.eval(m(mx.array([s]), cache=c))
        rows.append(c)
    batch = [rows[0][i].merge([r[i] for r in rows]) for i in range(len(rows[0]))]
    for i, t in enumerate(steps):
        out = m(mx.array([[t], [t]]), cache=batch)[:, -1]
        for r in range(2):
            assert float(mx.abs(out[r] - refs[r][i]).max()) <= 1e-4 * (1 + float(mx.abs(refs[r][i]).max()))


def test_sanitize_takes_liquids_mlx_names_and_the_original_ones():
    m = lfm3.Model(_args())
    conv_hf = mx.zeros((128, 1, 7))  # transformers' layout
    conv_mlx = mx.zeros((128, 7, 1))
    out = m.sanitize({"language_model.model.layers.0.conv.conv.weight": conv_mlx,
                      "model.layers.2.conv.conv.weight": conv_hf,
                      "language_model.model.embedding_norm.weight": mx.ones((128,)),
                      "lm_head.weight": mx.zeros((512, 128))})
    assert set(out) == {"model.layers.0.conv.conv.weight", "model.layers.2.conv.conv.weight", "model.embedding_norm.weight"}
    assert out["model.layers.0.conv.conv.weight"].shape == (128, 7, 1) == out["model.layers.2.conv.conv.weight"].shape


def test_locals_patch_leaves_lfm3_computing_the_same():
    """No fused path is LFM3's yet: patching must leave each layer as it is (kernels aside)."""
    import locals

    stock, patched = _model(seed=5), _model(seed=5)
    locals.patch(patched, tune="off", kernels=False)
    seq = mx.random.randint(0, 512, (1, 12), key=mx.random.key(11))
    assert float(mx.abs(stock(seq) - patched(seq)).max()) <= 1e-5 * (1 + float(mx.abs(stock(seq)).max()))


# --- the server on LFM3's chat template (its tokenizer from the local cache) ------------------------------------

LFM3 = "LiquidAI/LFM3-1.7B-MLX-4bit"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather.", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def _tokenizer(repo):
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer

    from locals.models import register

    register()
    try:
        return load_tokenizer(Path(snapshot_download(repo, local_files_only=True)))
    except Exception:  # noqa: BLE001
        pytest.skip(f"{repo} not cached")


@pytest.fixture(scope="module")
def server():
    from locals.server import Server

    class Svc:
        tokenizer = _tokenizer(LFM3)

    srv = Server(LFM3, engines=("locals",), preload=(), log=lambda s: None)
    srv._svc["locals"] = Svc()
    return srv


def test_template_reads_as_qwen_calls_that_think_when_off_two_lines_apart():
    from locals import formats

    tok = _tokenizer(LFM3)
    assert formats.detect(tok).name == "qwen_xml" and formats.thinking_style(tok) == "open"
    assert formats.thinks_when_off(tok) and formats.call_gap(tok) == "\n\n" and formats.think_gap(tok) == ""


def test_qwen35_keeps_its_grammar():
    from locals import formats

    tok = _tokenizer("mlx-community/Qwen3.5-4B-MLX-4bit")
    assert not formats.thinks_when_off(tok) and formats.call_gap(tok) == "\n" and formats.think_gap(tok) == "\n\n"


def test_template_renders_calls_the_grammar_writes():
    """Two calls of an earlier turn, as the template shows them, are a reply the grammar can write now; with thinking
    on too (the reply starts in the block the prompt opened, and goes on right after ``</think>``)."""
    from locals import oai
    from locals.regex import compile_dfa

    tok = _tokenizer(LFM3)
    calls = [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": c}}} for c in ("Paris", "Tokyo")]
    text = tok.apply_chat_template([{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": calls}],
                                   tools=TOOLS, tokenize=False)
    reply = text[text.rindex("<think></think>") + len("<think></think>"):text.rindex("<|im_end|>")]
    assert compile_dfa(oai.reply_regex(TOOLS, "auto", sealed=True, gap="\n\n"), 400_000).matches(reply.encode()), reply
    assert not compile_dfa(oai.reply_regex(TOOLS, "auto"), 400_000).matches(reply.encode())  # (Qwen3.5's one newline)
    text = tok.apply_chat_template([{"role": "user", "content": "x"},
                                    {"role": "assistant", "content": "", "reasoning_content": "Call it.", "tool_calls": calls}],
                                   tools=TOOLS, tokenize=False)
    reply = text[text.rindex("<think>") + len("<think>"):text.rindex("<|im_end|>")]
    assert reply.startswith("Call it.</think><tool_call>")
    assert compile_dfa(oai.reply_regex(TOOLS, "auto", thinking=True, gap="\n\n", think_gap=""), 400_000).matches(reply.encode())


def test_lfm3_thinks_unless_told_not_to(server):
    """As its template does: with thinking off it reasoned in its answers (10 of 18 agent tasks done, 15 thinking)."""
    req = {"model": "locals/lfm3-1.7b", "messages": [{"role": "user", "content": "Hi"}]}
    assert server.prepare(req)["thinking"] and server.prepare(dict(req, model="locals/lfm3-1.7b-thinking"))["thinking"]
    assert server.prepare(req)["text"].endswith("<think>")
    for off in ({"chat_template_kwargs": {"enable_thinking": False}}, {"reasoning_effort": "none"}):
        p = server.prepare(dict(req, **off))
        assert not p["thinking"] and p["text"].endswith("<think></think>")


OFF = {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize("tools", [None, TOOLS])
def test_a_reply_with_thinking_off_holds_no_think_tags(server, tools):
    from locals.regex import compile_dfa

    req = {"model": "locals/lfm3-1.7b", "messages": [{"role": "user", "content": "Weather in Paris?"}], **OFF}
    if tools:
        req["tools"] = tools
    p = server.prepare(req)
    d = compile_dfa(p["regex"], 400_000)
    assert d.matches(b"It is sunny in Paris.")
    for bad in ("<think>Let me see.</think>It is sunny.", "It is sunny.</think>It is sunny."):
        assert not d.matches(bad.encode())
    if tools:
        assert d.matches(b"<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>")
    # thinking on: the reply starts inside the block the prompt opened, and closes it (no grammar without tools)
    on = server.prepare({k: v for k, v in req.items() if k != "chat_template_kwargs"})["regex"]
    assert on is None if not tools else compile_dfa(on, 400_000).matches(b"Let me see.</think>\n\nIt is sunny.")
    assert server.prepare(dict(req, ignore_eos=True))["regex"] is None  # a benchmark's fixed-length reply: as before


def test_a_thinking_conversation_stays_a_prefix_of_the_next_request(server):
    """Thinking on, a reply's reasoning stays in the next request as it was written, even after a user message (Claude
    Code's reminders next to its tool results): the next prompt starts with this one and the reply, so the whole
    conversation comes from the cache (without it, each step read the conversation again from its first reply)."""
    from locals import oai

    call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
    raw = "The user wants the weather.</think>" + call  # what the model wrote after the prompt's <think>
    server._raw.clear()
    server._raw[oai.id_key("toolu_1")] = raw
    task = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "Weather in Paris?"}]
    first = server.prepare({"model": "locals/lfm3-1.7b-thinking", "messages": task, "tools": TOOLS})["text"]
    turn = [{"role": "assistant", "content": "", "tool_calls": [{"id": "toolu_1", "type": "function",
                                                               "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"},
            {"role": "user", "content": "<system-reminder>Keep going.</system-reminder>"}]
    second = server.prepare({"model": "locals/lfm3-1.7b-thinking", "messages": task + turn, "tools": TOOLS})["text"]
    assert first.endswith("<think>") and second.startswith(first + raw)
    off = server.prepare({"model": "locals/lfm3-1.7b", "messages": task + turn, "tools": TOOLS, **OFF})["text"]
    assert "The user wants the weather." not in off  # thinking off: as the template writes it (no switch)
