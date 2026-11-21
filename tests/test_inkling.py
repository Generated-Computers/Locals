"""Thinking Machines' Inkling 2 Mini (locals/models/inkling2_loop.py, locals/inkling.py) on small random models."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from locals import inkling
from locals.models import inkling2_loop as ink

TEXT = dict(hidden_size=64, num_hidden_layers=8, vocab_size=128, unpadded_vocab_size=120, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=16,
            sliding_window_size=8, local_layer_ids=[0, 1, 2, 3, 4, 6], d_rel=4, rel_extent=8, log_scaling_n_floor=4,
            log_scaling_alpha=0.1, rms_norm_eps=1e-6, qk_l2_clip_threshold=3.0, rel_l2_clip_threshold=2.0,
            sconv_kernel_size=4, n_layers_per_attn_res_block=3, n_routed_experts=4, num_experts_per_tok=2,
            intermediate_size=64, mlp_gate_softcap=4.0, mlp_up_softcap=25.0, route_scale=2.0, final_logit_softcapping=30.0)
LOOP = dict(core_start=3, core_end=6, num_passes=3, gate_rank=8, is_varloop=False)


def _model(loop: bool = True, seed: int = 0, experts: str = "", fused: bool = False) -> ink.Model:
    """A random model in float32. ``experts``: "q" has them quantized as the release (NVFP4 with a scale an expert,
    MXFP8 in the last layer). The router's bias picks the experts, a different pair a layer: with random weights two
    scores are often a rounding apart, and a path that rounds differently would pick another."""
    mx.random.seed(seed)
    config = {"model_type": "inkling2_loop", "text_config": {**TEXT, **({"loop_config": LOOP} if loop else {})}}
    m = ink.Model(ink.ModelArgs.from_dict(config))
    weights = []
    for k, v in tree_flatten(m.parameters()):
        if k.endswith(("norm.weight", "l2_clip.scale")):
            v = 1 + 0.1 * mx.random.normal(v.shape)
        elif k.endswith("e_score_correction_bias"):
            v = mx.array([(i * 7 + len(k)) % 4 * 10.0 for i in range(4)])
        else:
            v = mx.random.normal(v.shape) * (v.shape[-1] if v.ndim > 1 else 16) ** -0.5
        weights.append((k, v))
    m.update(tree_unflatten(weights))
    if experts:
        def predicate(path, module):
            if ".mlp.experts." not in path:
                return False
            if ".layers.7." in path:
                return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
            return {"group_size": 16, "bits": 4, "mode": "nvfp4", "global_scale": True}

        nn.quantize(m, class_predicate=predicate)
    if "a" in experts:  # attention's projections 8-bit, as in a build (the kernels read those of one input as one matrix)
        nn.quantize(m, group_size=64, bits=8, class_predicate=lambda p, mod: p.endswith(
            tuple(f"self_attn.{n}_proj" for n in ("q", "k", "v", "r", "o", "gate"))) and isinstance(mod, nn.Linear))
    mx.eval(m.parameters())
    if fused:
        assert inkling.fuse(m) == 8
    return m


def _close(a, b, tol=1e-4):
    return float(mx.abs(a - b)[..., :120].max()) <= tol * (1 + float(mx.abs(b)[..., :120].max()))


def test_register_puts_inkling_where_mlx_lm_imports_models():
    import sys

    from locals.models import register

    register()
    assert sys.modules["mlx_lm.models.inkling2_loop"].Model is ink.Model


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("experts", ["", "q"])
def test_a_cache_continues_a_text_as_one_pass_does(loop, experts):
    """Chunks, then tokens one at a time, past the window (8), the convolutions' width (4) and the position where
    a global layer's queries start to scale (4); 48 visits' worth of caches in the looped model's 14."""
    m = _model(loop, experts=experts)
    toks = mx.random.randint(0, 120, (2, 24), key=mx.random.key(3))
    full = m(toks)
    assert full.shape == (2, 24, 128) and bool((full[..., 120:] == ink.SHUT_OUT).all())  # the padded vocabulary
    cache = m.make_cache()
    assert len(cache) == 2 * (14 if loop else 8)
    assert [type(c).__name__ for c in cache[:2]] == ["RotatingKVCache", "ArraysCache"]
    got = [m(toks[:, :6], cache=cache), m(toks[:, 6:11], cache=cache)] + [m(toks[:, i:i + 1], cache=cache) for i in range(11, 24)]
    assert _close(mx.concatenate(got, axis=1), full)


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("experts", ["", "q", "qa"])
@pytest.mark.parametrize("rows", [1, 2])
def test_a_step_in_kernels_is_the_reference_step(loop, experts, rows):
    """One token a row through inkling.py's kernels, from the reference's cache of a prompt: the reference's logits,
    step after step (the window going round, a global layer past its floor), and a cache the reference continues."""
    ref, fused = _model(loop, experts=experts), _model(loop, experts=experts, fused=True)
    toks = mx.random.randint(0, 120, (rows, 26), key=mx.random.key(5))
    a, b = ref.make_cache(), fused.make_cache()
    assert _close(fused(toks[:, :5], cache=b), ref(toks[:, :5], cache=a))
    for i in range(5, 22):
        assert _close(fused(toks[:, i:i + 1], cache=b), ref(toks[:, i:i + 1], cache=a)), i
    assert _close(ref(toks[:, 22:], cache=b), ref(toks[:, 22:], cache=a))  # the reference, on the kernels' cache


@pytest.mark.parametrize("fused", [False, True])
def test_batch_rows_decode_as_alone_with_and_without_padding(fused):
    """Several requests at once: rows of different lengths merged into one batch (the shorter left-padded, its
    window not yet full) decode as each alone; then the longer leaves and the other goes on."""
    m = _model(seed=4, experts="q", fused=fused)
    seqs = [[int(t) for t in mx.random.randint(0, 120, (n,), key=mx.random.key(40 + n)).tolist()] for n in (19, 5)]
    steps = [3, 77, 101, 9, 55, 18, 6]

    def prompt(s):
        c = m.make_cache()
        mx.eval(m(mx.array([s]), cache=c))
        return c

    refs = []
    for s in seqs:
        c = prompt(s)
        refs.append([m(mx.array([[t]]), cache=c)[0, -1] for t in steps])
    rows = [prompt(s) for s in seqs]
    batch = [rows[0][i].merge([r[i] for r in rows]) for i in range(len(rows[0]))]
    assert [type(c).__name__ for c in batch[:2]] == ["BatchRotatingKVCache", "ArraysCache"]
    for i in range(4):
        out = m(mx.array([[steps[i]], [steps[i]]]), cache=batch)[:, -1]
        assert _close(out[0], refs[0][i]) and _close(out[1], refs[1][i]), i
    for c in batch:
        c.filter(mx.array([1]))
    for i in range(4, 7):
        assert _close(m(mx.array([[steps[i]]]), cache=batch)[0, -1], refs[1][i]), i


def test_route_picks_the_references_experts():
    mx.random.seed(2)
    x = mx.random.normal((5, 64))
    w, bias = mx.random.normal((16, 64)) * 0.3, mx.random.normal((16,)) * 0.1
    scores = mx.sigmoid(x @ w.T)
    want = mx.argsort(-(scores + bias), axis=-1)[:, :4]
    inds, weights = inkling.route(x, w, bias, 4, 8.0)
    assert inds.tolist() == want.tolist()  # the highest first
    picked = mx.take_along_axis(scores, want, axis=-1)
    assert float(mx.abs(weights - picked / picked.sum(axis=-1, keepdims=True) * 8.0).max()) < 1e-5


def test_attend_reads_a_window_written_round_and_round():
    """The kernel's attention over a ring whose newest key is in the middle, with rows of different numbers of real
    keys, against the reference's attention over the same keys in order."""
    mx.random.seed(6)
    M, H, KV, hd, S, R = 2, 4, 2, 16, 24, 32
    q, keys, values = mx.random.normal((M, H, hd)), mx.random.normal((M, KV, S, hd)), mx.random.normal((M, KV, S, hd))
    bias, sinks, gate = mx.random.normal((M, H, R)), mx.random.normal((M, H)), mx.random.normal((M, H * hd))
    idx, counts = 9, mx.array([24, 13], dtype=mx.int32)
    got = inkling.attend(q, keys, values, bias, sinks, gate, idx, counts, hd**-0.5)
    order = [(idx + i) % S for i in range(S)]  # oldest first
    k_pos = mx.stack([mx.arange(S) - (S - n) for n in (24, 13)])  # (a padded row's first keys are before its text)
    q_pos = mx.array([[23], [12]])
    want = ink.attend(q[:, :, None], keys[:, :, order], values[:, :, order], bias[:, None], sinks[:, None], scale=hd**-0.5,
                      window=S, q_pos=q_pos, k_pos=k_pos)
    want = want.reshape(M, -1) * mx.sigmoid(gate)
    assert float(mx.abs(got - want).max()) < 1e-5


def test_sanitize_takes_the_releases_names_and_its_experts():
    """The release's layout: experts as bytes of 4-bit pairs with an amax and a denominator each, conv taps as
    (C, 1, K), the loop's queries in two tables, the routers in bfloat16."""
    m = _model(experts="q")
    weights = dict(tree_flatten(m.parameters()))
    release = {}
    for k, v in weights.items():
        k = k.replace("model.language_model.", "model.llm.")
        if k.endswith(".message_input.query"):
            release["model.llm.middle_cycle_cross_pass_queries"] = mx.concatenate([v[:, :1], v[:, 3:]], axis=1).flatten(0, 1)
            release["model.llm.middle_cycle_cross_pass_kv_queries"] = v[:, 1:3].flatten(0, 1)
        elif ".mlp.experts." in k:
            base, param = k.rsplit(".", 1)
            name = base.replace("gate_up_proj", "w13_weight").replace("down_proj", "w2_weight")
            if param == "weight":
                release[name] = v.view(mx.uint8)
            elif param == "scales":
                release[name + "_scale"] = v
            else:
                release[name + "_amax"], release[name + "_outer_scale_denominator"] = v * 7.5 / 2688.0, mx.array([7.5])
        elif k.endswith("conv1d.weight"):
            release[k.replace("self_attn.", "attn.").replace("post_attention_convolution", "attn_sconv")
                    .replace("post_mlp_convolution", "mlp_sconv").replace(".conv1d.weight", ".weight")] = v.swapaxes(1, 2)
        elif k.endswith("mlp.gate.weight"):
            release[k] = v.astype(mx.bfloat16)
        elif k == "lm_head.weight":
            release["model.llm.unembed.weight"] = v
        else:
            release[k] = v
    release["model.visual.proj_out.weight"] = mx.zeros((4, 4))  # text only: left out
    fresh = ink.Model(m.args)
    got = fresh.sanitize(release)
    renamed = {k for k in got if k.startswith("model.llm.")}
    assert not renamed or all(".self_attn." not in k for k in renamed)
    for k in ("model.language_model.message_input.query", "model.language_model.layers.0.mlp.experts.gate_up_proj.global_scale",
              "model.language_model.layers.0.post_mlp_convolution.conv1d.weight"):
        assert float(mx.abs(got[k].astype(mx.float32) - weights[k].astype(mx.float32)).max()) < 1e-6, k
    assert got["model.language_model.layers.1.mlp.gate.weight"].dtype == mx.float32
    assert got["model.language_model.layers.0.mlp.experts.gate_up_proj.weight"].dtype == mx.uint32
    experts = fresh.model.language_model.layers
    assert (experts[0].mlp.experts.gate_up_proj.mode, experts[7].mlp.experts.down_proj.mode) == ("nvfp4", "mxfp8")
    assert not any(k.startswith("model.visual") for k in got)


def test_the_converters_nvfp4_is_read_back_by_mlx_and_loses_less():
    from locals.convert_inkling import nvfp4

    mx.random.seed(8)
    w = mx.random.normal((48, 64)) * 0.03
    packed, scales, g, rel = nvfp4(w)
    back = mx.dequantize(packed, scales, None, group_size=16, bits=4, mode="nvfp4", global_scale=g, dtype=mx.float32)
    mine = float(mx.sqrt(mx.square(back - w).sum() / mx.square(w).sum()))
    assert abs(mine - float(rel)) < 1e-5
    q, s, *_ = mx.quantize(w, 16, 4, mode="nvfp4", global_scale=g)
    stock = mx.dequantize(q, s, None, group_size=16, bits=4, mode="nvfp4", global_scale=g, dtype=mx.float32)
    assert mine < float(mx.sqrt(mx.square(stock - w).sum() / mx.square(w).sum()))


def test_head_logits_are_the_models_own():
    """The engines call the body and the output layer apart (kernels.head_logits): softcapped, the padded
    vocabulary shut out, as the model's own forward."""
    from locals import kernels

    m = _model()
    toks = mx.array([[5, 9, 33, 70]])
    assert _close(kernels.head_logits(m, m.model(toks)), m(toks), 1e-5)
    assert m.model.embed_tokens.weight.shape[0] == 128
    assert _close(m.model(None, None, m.model.embed_tokens(toks)), m.model(toks), 1e-6)  # (the embedding with its norm)


# --- its replies: messages back to back (locals/formats.py, Inkling) --------------------------------------------------

REPLY = ("<|message_model|><|content_thinking|>Need the weather.<|end_message|><|message_model|><|content_text|>Let me check.<|end_message|>"
         '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris"}}<|end_message|>')
WEATHER = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def test_a_reply_is_its_reasoning_its_text_and_its_calls():
    import json
    import re

    from locals import formats, oai

    fmt = formats.detect_from_template("... <|content_invoke_tool_json|> ... reasoning_effort ...")
    assert fmt.name == "inkling"
    reasoning, content, calls = oai.parse_reply(REPLY, WEATHER, fmt)
    assert (reasoning, content) == ("Need the weather.", "Let me check.")
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls] == [("get_weather", {"city": "Paris"})]
    assert oai.parse_reply("<|message_model|><|content_text|>Paris.<|end_message|>", None, fmt) == ("", "Paris.", [])
    rx = fmt.reply_regex(["get_weather", "other"])
    assert re.fullmatch(rx, REPLY) and re.fullmatch(rx, "<|message_model|><|content_text|>Paris.<|end_message|>")
    assert not re.fullmatch(rx, "Paris.")  # text is a message too
    assert fmt.count_complete(REPLY) == 1 and fmt.cut_after(REPLY + REPLY[REPLY.index("<|message_model|>get"):], 1) == REPLY


@pytest.mark.parametrize("calls", [True, False])
def test_a_stream_shows_the_text_without_its_markers(calls):
    import re

    from locals import formats, oai

    stream, text, out = oai.ReplyStream(False, formats.FORMATS["inkling"], calls=calls), "", []
    for piece in re.findall(r"<\|[a-z_]+\|>|[^<]+", REPLY[: REPLY.index("<|message_model|>get")]):
        text += piece
        out += stream.feed(text)
    assert out == [("reasoning", "Need the weather."), ("content", "Let me check.")]


def test_a_turns_calls_go_to_the_template_as_objects():
    """Inkling's template raises on OpenAI's JSON-string arguments."""
    from locals import formats

    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]}]
    out, tools = formats.FORMATS["inkling"].prepare(msgs, WEATHER)
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"} and tools == WEATHER
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'  # (the request's own left as it was)


def test_a_grammar_writes_inklings_markers_as_their_tokens():
    """Spelled letter by letter (special tokens are no text to a grammar unless listed), the model lost the thread:
    after a tool result it called the tool again."""
    from locals.vocab import FORMAT_TOKENS

    assert {"<|message_model|>", "<|end_message|>", "<|content_thinking|>", "<|content_text|>",
            "<|content_invoke_tool_json|>"} <= FORMAT_TOKENS


# --- one request alone: spans in one pass, drafts taken back (hybrid_batch._solo) -------------------------------------

def _batch_of_one(m, prompt):
    c = m.make_cache()
    mx.eval(m(mx.array([prompt]), cache=c))
    return [x.merge([x]) for x in c]


@pytest.mark.parametrize("fused", [False, True])
def test_a_span_in_one_pass_is_its_tokens_one_at_a_time(fused):
    """A forced span (a tool call's syntax) goes in one pass over the batch-of-one caches, windows past their size:
    the logits of one-token steps, and the caches they leave."""
    m = _model(seed=9, experts="q", fused=fused)
    prompt = [int(t) for t in mx.random.randint(0, 120, (14,), key=mx.random.key(3)).tolist()]
    span = [11, 100, 23, 7, 7, 60, 3]
    drip, one = _batch_of_one(m, prompt), _batch_of_one(m, prompt)
    ref = mx.concatenate([m(mx.array([[t]]), cache=drip)[:, -1] for t in span])
    assert _close(m(mx.array([span]), cache=one)[0], ref)
    for t in (5, 99, 30):
        assert _close(m(mx.array([[t]]), cache=one)[0, -1], m(mx.array([[t]]), cache=drip)[0, -1])


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("prompt_len", [3, 14])
def test_a_drafts_end_is_taken_back(fused, prompt_len):
    """A pass feeds a token and a draft after it; the model confirms part of the draft and the rest is taken back
    (inkling.rewind): what follows is as if the rest had never been fed, in a window not yet full and in a full one."""
    from mlx_lm.models.cache import ArraysCache

    from locals.hybrid_batch import HybridBatchEngine

    m = _model(seed=10, experts="q", fused=fused)
    prompt = [int(t) for t in mx.random.randint(0, 120, (prompt_len,), key=mx.random.key(6)).tolist()]
    fed, dropped, after = [42, 8, 77], [19, 3, 101], [64, 5, 90, 12]
    want, got = _batch_of_one(m, prompt), _batch_of_one(m, prompt)
    mx.eval(m(mx.array([fed]), cache=want))
    eng = HybridBatchEngine.__new__(HybridBatchEngine)
    eng.lm = m
    assert eng._inkling() and eng._solo_drafts_ok(got)
    for c in got:
        if isinstance(c, ArraysCache):
            c.want_states = True
    mx.eval(m(mx.array([fed + dropped]), cache=got))
    eng._solo_rewind(got, len(dropped))
    for c in got:
        if isinstance(c, ArraysCache):
            c.want_states = False
    for t in after:
        assert _close(m(mx.array([[t]]), cache=got)[0, -1], m(mx.array([[t]]), cache=want)[0, -1]), t
    assert _close(m(mx.array([after]), cache=got)[0], m(mx.array([after]), cache=want)[0])  # (and a span after it)


def test_a_forced_marker_is_fed_as_its_token(monkeypatch):
    """A format's control token that the tokenizer marks special is one piece of a forced span, not its letters:
    Inkling's "<|message_model|>" forced letter by letter no longer read as the start of a message, and the model
    wrote its tool call again and again. With Qwen's tokenizer (the one the fast tests have) and one of its
    special tokens standing in."""
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer

    from locals import forced, vocab

    try:
        tok = load_tokenizer(snapshot_download("mlx-community/Qwen2.5-1.5B-Instruct-4bit", local_files_only=True))._tokenizer
    except Exception:
        pytest.skip("tokenizer not cached")
    marker = "<|im_start|>"
    assert tok.convert_tokens_to_ids(marker) in tok.all_special_ids

    class NoConstraint:
        forced_bytes = [b""]

    def pieces(text):
        p = forced.ForcedPlanner(tok, None, NoConstraint())
        starts = p._piece_starts(text.encode())
        return [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]

    assert marker not in pieces("x" + marker + "call")  # a special token is no text to a grammar...
    monkeypatch.setattr(vocab, "FORMAT_TOKENS", vocab.FORMAT_TOKENS | {marker})
    got = pieces("x" + marker + "call")  # ...unless it is a format's own
    assert marker in got and tok.encode(marker, add_special_tokens=False) == [tok.convert_tokens_to_ids(marker)]


# --- grammars: walked, and tool calls typed --------------------------------------------------------------------------

TOOLS = [{"type": "function", "function": {"name": "edit_file", "parameters": {
             "type": "object", "required": ["path", "old", "new"],
             "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"},
                            "count": {"type": "integer"}, "mode": {"enum": ["a", "b"]},
                            "tags": {"type": "array", "items": {"type": "string"}}}}}},
         {"type": "function", "function": {"name": "opt", "parameters": {
             "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}, "c": {"type": ["string", "null"]}}}}},
         {"type": "function", "function": {"name": "bare"}}]


def _call(name, args, header=None):
    import json

    return ("<|message_model|>" + (header or name) + "<|content_invoke_tool_json|>"
            + json.dumps({"name": name, "args": args}, separators=(",", ":"), ensure_ascii=False) + "<|end_message|>")


def test_a_call_is_typed_by_its_tools_schema():
    """A call's arguments are the tool's parameters, in the schema's order, the required ones there, each of its
    type, and the JSON's name is the header's: all but the values is forced text (fed in one pass)."""
    import re

    from locals import formats, oai

    fmt = formats.FORMATS["inkling"]
    rx = fmt.typed_call(TOOLS)
    for name, args in [("edit_file", {"path": "a.py", "old": "x\ny \"q\"", "new": "é"}),
                       ("edit_file", {"path": "a", "old": "b", "new": "c", "mode": "b", "tags": ["q", "r"]}),
                       ("edit_file", {"path": "a", "old": "b", "new": "c", "count": 3, "tags": []}),
                       ("opt", {}), ("opt", {"b": True}), ("opt", {"a": 1, "c": None}), ("opt", {"a": 1, "b": False, "c": "s"}),
                       ("bare", {}), ("bare", {"k": [1, {"z": None}]})]:
        assert re.fullmatch(rx, _call(name, args)), (name, args)
    for name, args in [("edit_file", {"path": "a", "old": "b"}),  # a required one missing
                       ("edit_file", {"old": "b", "path": "a", "new": "c"}),  # not the schema's order
                       ("edit_file", {"path": 3, "old": "b", "new": "c"}), ("opt", {"c": 1}), ("opt", {"z": 1}),
                       ("edit_file", {"path": "a", "old": "b", "new": "c", "mode": "c"})]:
        assert not re.fullmatch(rx, _call(name, args)), (name, args)
    assert not re.fullmatch(rx, _call("edit_file", {"path": "a", "old": "b", "new": "c"}, header="opt"))
    # the server's grammar of a reply is typed without --typed-tools, and a chosen tool is the only one
    reply = oai.reply_regex(TOOLS, "required", fmt=fmt, preamble=False)
    assert re.fullmatch(reply, _call("opt", {"a": 1}) + _call("bare", {})) and not re.fullmatch(reply, _call("opt", {"z": 1}))
    one = oai.reply_regex(TOOLS, {"type": "function", "function": {"name": "opt"}}, fmt=fmt, preamble=False)
    assert re.fullmatch(one, _call("opt", {"b": False})) and not re.fullmatch(one, _call("bare", {}))


def test_a_fused_model_asks_for_walked_grammars():
    """7 bytes a state a token over 201,024 tokens: a set of tools' tables were 1.3 GB beside 15.6 GB of weights."""
    assert getattr(_model(fused=True), "walk_constraints", False) and not getattr(_model(), "walk_constraints", False)


def test_a_walked_grammar_tries_a_drafts_tokens():
    from locals.regex import DEAD, compile_dfa
    from locals.walk import WalkController

    class Vocab:
        token_bytes = [b"ab", b"c", b"a", None, b"abc"]

    class WC:
        dfa, vocab = compile_dfa("(ab)+c"), Vocab()

    WC.only_eos = WC.dfa.accepting & ((WC.dfa.table != DEAD).sum(axis=1) == 0)
    ctl = WalkController.__new__(WalkController)
    ctl.wc, ctl.eos_ids, ctl.canonical, ctl.q = WC, [3], False, WC.dfa.start
    q = ctl.draft_step(ctl.q, 0)
    assert q is not None and ctl.draft_step(ctl.q, 1) is None and ctl.draft_step(q, 3) is None  # (the end is no draft)
    assert ctl.draft_step(q, 4) is not None and not ctl.draft_finished(q) and ctl.draft_finished(ctl.draft_step(q, 1))
    assert ctl.q == WC.dfa.start and not hasattr(ctl, "step_config")  # (the one-stream engine's sign of token tables)
