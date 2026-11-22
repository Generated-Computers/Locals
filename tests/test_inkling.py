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


DRAFT = dict(hidden_size=TEXT["hidden_size"], num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
             intermediate_size=48, rms_norm_eps=1e-6, rope_theta=1e4, local_layer_ids=[0, 2], sliding_window_size=8,
             block_size=8, markov_rank=8, markov_head_type="rnn_hidden", target_layer_ids=[2, 6, 11, 14])


def _model(loop: bool = True, seed: int = 0, experts: str = "", fused: bool = False, draft: bool = False,
           adapter: bool = False) -> ink.Model:
    """A random model in float32. ``experts``: "q" has them quantized as the release (NVFP4 with a scale an expert,
    MXFP8 in the last layer). The router's bias picks the experts, a different pair a layer: with random weights two
    scores are often a rounding apart, and a path that rounds differently would pick another."""
    mx.random.seed(seed)
    config = {"model_type": "inkling2_loop", "text_config": {**TEXT, **({"loop_config": LOOP} if loop else {})},
              **({"draft_config": DRAFT} if draft else {}), **({"adapter": {"kind": "direction"}} if adapter else {})}
    m = ink.Model(ink.ModelArgs.from_dict(config))
    weights = []
    for k, v in tree_flatten(m.parameters()):
        if k.endswith(("norm.weight", "l2_clip.scale")):
            v = 1 + 0.1 * mx.random.normal(v.shape)
        elif k.endswith(".mlp.ablate"):  # (a unit direction, each layer its own)
            v = mx.random.normal(v.shape)
            v = v / mx.linalg.norm(v)
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


@pytest.mark.parametrize("window", [0, 6])
def test_attentions_weights_in_one_kernel_are_the_references(window):
    """ink.attend with inkling.soft: the bias by distance, the mask (later keys, keys past the window, a shorter
    row's padding) and the softmax with its sink in one pass over the scores, for rows at their own positions."""
    mx.random.seed(21)
    B, Hq, Hkv, L, S, D, R = 2, 4, 2, 5, 13, 16, 8
    q, k, v = mx.random.normal((B, Hq, L, D)), mx.random.normal((B, Hkv, S, D)), mx.random.normal((B, Hkv, S, D))
    bias, sinks = mx.random.normal((B, L, Hq, R)), mx.random.normal((B, L, Hq))
    end = mx.array([40, 9])  # (the second row holds 9 tokens: its first 4 keys are padding)
    q_pos, k_pos = (end - L)[:, None] + mx.arange(L), (end - S)[:, None] + mx.arange(S)
    want = ink.attend(q, k, v, bias, sinks, scale=0.25, window=window, q_pos=q_pos, k_pos=k_pos)
    got = ink.attend(q, k, v, bias, sinks, scale=0.25, window=window, q_pos=q_pos, k_pos=k_pos, soft=inkling.soft)
    assert _close(got, want, 1e-5)
    one = ink.attend(q[:1], k[:1], v[:1], bias[:1], sinks[:1], scale=0.25, window=window, q_pos=mx.arange(35, 40),
                     k_pos=mx.arange(27, 40), soft=inkling.soft)
    assert _close(one, want[:1], 1e-5)


@pytest.mark.parametrize("fused", [False, True])
def test_a_long_chunk_in_blocks_of_queries_is_the_chunk_at_once(fused, monkeypatch):
    """A sliding layer scores a long chunk a block of queries at a time, over the keys that block sees, and a long
    pass is evaluated a visit at a time: the states are those of the chunk scored and evaluated at once."""
    m = _model(seed=4, experts="q", fused=fused)
    tokens = mx.random.randint(0, 120, (1, 60), key=mx.random.key(9))

    def read(block, each):
        monkeypatch.setattr(ink, "QUERY_BLOCK", block)
        monkeypatch.setattr(ink, "EVAL_TOKENS", each)
        cache = m.make_cache()
        return mx.concatenate([m.model(tokens[:, :11], cache), m.model(tokens[:, 11:49], cache), m.model(tokens[:, 49:], cache)], axis=1)

    want = read(1 << 20, 1 << 20)
    for block in (4, 7, 16):
        assert _close(read(block, 8), want)


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
    assert marker not in pieces("x" + marker + "call")  # (another format's, Gemma 4's say: as before)
    monkeypatch.setattr(vocab, "INKLING_TOKENS", vocab.INKLING_TOKENS | {marker})
    got = pieces("x" + marker + "call")  # ...unless it is Inkling's own
    assert marker in got and tok.encode(marker, add_special_tokens=False) == [tok.convert_tokens_to_ids(marker)]


def test_other_models_are_served_as_before_inkling():
    """What the engine does for Inkling 2 alone: a prompt that requests waiting together share is read once, a
    checkpoint's copy counts the windows it holds, usage says what drafts did, room is kept for a read of its size
    and the prompt cache is smaller beside its weights. A Qwen3.5 (gated delta and attention layers) reads each
    request's prompt, counts its copies and keeps room as it did before."""
    from mlx_lm.models import qwen3_5
    from mlx_lm.models.cache import LRUPromptCache

    from locals import hybrid_batch
    from locals.engine import Controller
    from locals.hybrid_batch import HybridBatchEngine
    from locals.packed import Request

    class Free(Controller):
        eos_ids = [127]

    class Tok:
        eos_token_ids = [127]

        def decode(self, ids):
            return " ".join(map(str, ids))

    text = dict(model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, vocab_size=128, linear_num_value_heads=4, linear_num_key_heads=2,
                linear_key_head_dim=16, linear_value_head_dim=16, full_attention_interval=2, tie_word_embeddings=True)
    m = qwen3_5.Model(qwen3_5.ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": text}))
    mx.eval(m.parameters())
    tokens = [int(t) for t in mx.random.randint(0, 120, (120,), key=mx.random.key(3)).tolist()]
    same, other = tokens[:100], tokens[:70] + tokens[100:120]
    eng = HybridBatchEngine(m, Tok(), n_slots=4, min_share=8)
    reqs = [Request(list(p), Free(), 6) for p in (same, same, other, same)]
    for r in reqs:
        r.pend, r.fed = [], []
    waiting = list(reqs)
    eng.serve(lambda block: [waiting.pop(0) for _ in list(waiting)], lru=LRUPromptCache(max_size=16), lru_key="tiny")
    assert eng.counters["prefill_tokens"] == 3 * (len(same) - 1) + len(other) - 1
    assert not eng._inkling() and eng._copies_state(3) == 0 and not hybrid_batch._reads_by_size(eng)
    assert getattr(eng.lm, "prompt_cache_bytes", None) is None  # (its prompt cache is sized as before)
    ink_eng = HybridBatchEngine(_model(fused=True), Tok(), n_slots=4)
    assert ink_eng._copies_state(3) > 0 and hybrid_batch._reads_by_size(ink_eng)


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


def _as_the_template(name, args):
    """A call as the model's template writes an earlier turn's: its arguments sorted by name (``dictsort``)."""
    return _call(name, dict(sorted(args.items())))


def test_a_call_is_typed_by_its_tools_schema():
    """A call's arguments are the tool's parameters, sorted as the model's template writes them, the required ones
    there, each of its type, and the JSON's name is the header's: all but the values is forced text (fed in one
    pass), and the reply is what the next request holds in its place."""
    import re

    from locals import formats, oai

    fmt = formats.FORMATS["inkling"]
    rx = fmt.typed_call(TOOLS)
    for name, args in [("edit_file", {"path": "a.py", "old": "x\ny \"q\"", "new": "é"}),
                       ("edit_file", {"path": "a", "old": "b", "new": "c", "mode": "b", "tags": ["q", "r"]}),
                       ("edit_file", {"path": "a", "old": "b", "new": "c", "count": 3, "tags": []}),
                       ("opt", {}), ("opt", {"b": True}), ("opt", {"a": 1, "c": None}), ("opt", {"a": 1, "b": False, "c": "s"}),
                       ("bare", {}), ("bare", {"k": [1, {"z": None}]})]:
        assert re.fullmatch(rx, _as_the_template(name, args)), (name, args)
    for name, args in [("edit_file", {"new": "c", "old": "b"}),  # a required one missing
                       ("edit_file", {"path": "a", "old": "b", "new": "c"}),  # the schema's order, not the template's
                       ("edit_file", {"new": "c", "old": "b", "path": 3}), ("opt", {"c": 1}), ("opt", {"z": 1}),
                       ("edit_file", {"mode": "c", "new": "c", "old": "b", "path": "a"})]:
        assert not re.fullmatch(rx, _call(name, args)), (name, args)
    assert not re.fullmatch(rx, _call("edit_file", {"new": "c", "old": "b", "path": "a"}, header="opt"))
    # the server's grammar of a reply is typed without --typed-tools, and a chosen tool is the only one
    reply = oai.reply_regex(TOOLS, "required", fmt=fmt, preamble=False)
    assert re.fullmatch(reply, _call("opt", {"a": 1}) + _call("bare", {})) and not re.fullmatch(reply, _call("opt", {"z": 1}))
    one = oai.reply_regex(TOOLS, {"type": "function", "function": {"name": "opt"}}, fmt=fmt, preamble=False)
    assert re.fullmatch(one, _call("opt", {"b": False})) and not re.fullmatch(one, _call("bare", {}))


def test_a_turn_sent_back_is_the_reply_it_came_from():
    """Clients send a turn with calls back without its reasoning, and its text trimmed; the model's template writes
    an earlier turn's reasoning. From the reply the server kept, the turn is written as the model wrote it, so the
    next request goes on from the finished row; a turn whose calls the client changed is the client's."""
    from locals import formats

    fmt = formats.FORMATS["inkling"]
    raw = ("<|message_model|><|content_thinking|> Read it first.\n<|end_message|><|message_model|><|content_text|>Reading. <|end_message|>"
           + _as_the_template("edit_file", {"path": "a", "old": "b", "new": {"z": 1, "a": [2]}}))
    call = lambda args: {"id": "c1", "type": "function", "function": {"name": "edit_file", "arguments": args}}  # noqa: E731
    sent = {"role": "assistant", "content": "Reading.", "tool_calls": [call({"path": "a", "new": {"a": [2], "z": 1}, "old": "b"})]}
    got = fmt.restore(sent, raw)
    assert got["reasoning_content"] == " Read it first.\n" and got["content"] == "Reading. "
    assert list(got["tool_calls"][0]["function"]["arguments"]["new"]) == ["z", "a"] and got["tool_calls"][0]["id"] == "c1"
    assert "reasoning_content" not in sent  # (the client's message is not written into)
    assert fmt.restore({**sent, "reasoning_content": "Read it first."}, raw)["reasoning_content"] == " Read it first.\n"
    for other in ({**sent, "tool_calls": [call({"path": "a2", "old": "b", "new": {"z": 1, "a": [2]}})]},  # another call
                  {**sent, "content": "Something else."}, {**sent, "reasoning_content": "My own thought."},
                  {**sent, "tool_calls": sent["tool_calls"] * 2}):
        assert fmt.restore(other, raw) is other
    bare = _as_the_template("opt", {"a": 1})  # (a reply that went straight to its call)
    got = fmt.restore({"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "function": {"name": "opt", "arguments": {"a": 1}}}]}, bare)
    assert got["content"] == "" and "reasoning_content" not in got


def test_requests_with_one_prompt_asked_at_once_read_it_once():
    """An agent's versions of one part are asked at once with one prompt: the first reads it, and the prompt cache has
    it for the others (and what two prompts share, for the second); the replies are those of requests asked alone."""
    from mlx_lm.models.cache import LRUPromptCache

    from locals.engine import Controller
    from locals.hybrid_batch import HybridBatchEngine
    from locals.packed import Request

    class Free(Controller):
        eos_ids = [127]

    class Tok:  # (the engine writes a finished reply's text)
        eos_token_ids = [127]

        def decode(self, ids):
            return " ".join(map(str, ids))

    m = _model(seed=12, experts="q", fused=True)
    tokens = [int(t) for t in mx.random.randint(0, 120, (60,), key=mx.random.key(3)).tolist()]
    same, other = tokens[:40], tokens[:30] + tokens[45:60]

    def serve(prompts, share=True):
        eng = HybridBatchEngine(m, Tok(), n_slots=4, share_prefix=share, min_share=8)
        reqs = [Request(list(p), Free(), 6) for p in prompts]
        for r in reqs:
            r.pend, r.fed = [], []
        waiting = list(reqs)
        eng.serve(lambda block: [waiting.pop(0) for _ in list(waiting)], lru=LRUPromptCache(max_size=16), lru_key="tiny")
        return [r.out for r in reqs], eng.counters["prefill_tokens"]

    alone = [serve([p], share=False)[0][0] for p in (same, other)]
    outs, read = serve([same, same, other, same])
    assert outs == [alone[0], alone[0], alone[1], alone[0]]
    assert read == (len(same) - 1) + (len(other) - 1 - 30)  # the prompt once, and the other's own end
    assert serve([same, same, other, same], share=False)[1] == 3 * (len(same) - 1) + len(other) - 1


def test_a_fused_model_asks_for_walked_grammars():
    """7 bytes a state a token over 201,024 tokens: a set of tools' tables were 1.3 GB beside 15.6 GB of weights."""
    assert getattr(_model(fused=True), "walk_constraints", False) and not getattr(_model(), "walk_constraints", False)


def test_a_fused_model_says_what_its_prompts_need():
    """The engine reads its prompts in the model's chunk, keeps room for a read of its size (a long conversation's
    next request has a few dozen tokens to read) and, where the weights leave little, a small prompt cache."""
    from locals.hybrid_batch import HybridBatchEngine

    m = _model(fused=True)
    assert m.prefill_chunk_tokens == inkling.PREFILL_CHUNK and m.read_room_by_size and m.prompt_cache_bytes < 2 << 30
    eng = HybridBatchEngine(m, None, n_slots=2)
    eng._note_read(40, 100 << 20), eng._note_read(inkling.PREFILL_CHUNK, 900 << 20), eng._note_read(2048, 2900 << 20)
    assert eng._read_bytes(40) < eng._read_bytes(6000) == (900 << 20) * 7 // 6  # (a chunk at most, whatever is left)
    cache, seen = m.make_cache(), []
    eng._read_chunk = lambda cache, feed, n: seen.append(n) or n
    eng._run(cache, list(range(3 * inkling.PREFILL_CHUNK)))
    assert seen == [inkling.PREFILL_CHUNK] * 3


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


# --- the draft model (locals/inkling_draft.py) -----------------------------------------------------------------------

def _draft_state(m, cache):
    own = len(cache) - len(m.mtp.layers)
    toks, probs = m.mtp.draft(7, cache[own:])
    return toks.tolist(), probs


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("prompt_len", [3, 14])
def test_the_draft_models_context_is_the_targets_passes(fused, prompt_len):
    """The draft model reads what the target's passes left in three caches after the target's own: the same draft
    whether the text went in as one pass, a token at a time (in kernels), or with a draft's end taken back; in a
    window not yet full and in a full one. And the target computes what it computes without a draft model."""
    from mlx_lm.models.cache import ArraysCache

    m, plain = _model(seed=11, experts="q", fused=fused, draft=True), _model(seed=11, experts="q", fused=fused)
    if not fused:
        m.mtp.bind(m)
    assert len(m.make_cache()) == len(plain.make_cache()) + 3
    prompt = [int(t) for t in mx.random.randint(0, 120, (prompt_len,), key=mx.random.key(8)).tolist()]
    fed, dropped = [42, 8, 77, 5], [19, 3, 101]
    whole = _batch_of_one(m, prompt + fed)
    steps, back = _batch_of_one(m, prompt), _batch_of_one(m, prompt)
    for t in fed:
        mx.eval(m(mx.array([[t]]), cache=steps))
    for c in back:
        if isinstance(c, ArraysCache):
            c.want_states = True
    mx.eval(m(mx.array([fed + dropped]), cache=back))
    inkling.rewind(back, len(dropped))
    assert m.mtp.draft(7, whole[len(whole) - 3:]) is None  # (a merged cache: no pass yet whose last token the head reads)
    for c in (whole, steps, back):
        mx.eval(m(mx.array([[9]]), cache=c))
    want_t, want_p = _draft_state(m, whole)
    assert len(want_t) == 7 and float(want_p.min()) > 0
    for other in (steps, back):
        got_t, got_p = _draft_state(m, other)
        assert got_t == want_t and _close(got_p, want_p, 1e-3)
    # the target computes what it computes with nothing reading its passes
    a, b = _batch_of_one(m, prompt), _batch_of_one(m, prompt)
    with_draft = m(mx.array([fed]), cache=a)[0]
    m.model.language_model.taps = None
    assert _close(with_draft, m(mx.array([fed]), cache=b)[0])


def test_a_build_without_the_draft_model_has_none(monkeypatch):
    m = _model(fused=True)
    assert "mtp" not in m and getattr(m.model.language_model, "taps", None) is None
    monkeypatch.setenv("LOCALS_INKLING_DRAFT", "0")
    off = _model(fused=True, draft=True)
    assert off.draft_off and len(off.make_cache()) == len(m.make_cache()) and off.model.language_model.taps is None


def test_the_build_downloads_only_what_its_folder_lacks(tmp_path, monkeypatch):
    """A release folder downloaded for a build without the draft model has no mtp.safetensors: --draft-only gets its
    0.64 GB alone (a folder deleted after the build too), and a build made again gets the draft model's file."""
    import json

    import huggingface_hub

    from locals import convert_inkling as conv

    asked = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, local_dir, allow_patterns: asked.append(allow_patterns))
    monkeypatch.setattr(conv, "add_draft", lambda *a, **k: None)
    monkeypatch.setattr(conv, "convert", lambda *a, **k: None)
    src, out = tmp_path / "release", tmp_path / "build"
    src.mkdir(), out.mkdir()
    (src / "config.json").write_text(json.dumps({"mtp_config": {}}))
    (out / "config.json").write_text(json.dumps({"quantization": {}}))
    (src / conv.WEIGHTS).write_bytes(b"")
    conv.main(["--src", str(src), "--out", str(out), "--draft-only"])
    assert asked[-1] == conv.SMALL + [conv.DRAFT]
    conv.main(["--src", str(src), "--out", str(out)])
    assert asked[-1] == conv.SMALL + [conv.DRAFT]
    (src / conv.DRAFT).write_bytes(b"")
    n = len(asked)
    conv.main(["--src", str(src), "--out", str(out), "--draft-only"])
    conv.main(["--src", str(src), "--out", str(out)])
    assert len(asked) == n  # (all there: nothing downloaded)
    (src / conv.WEIGHTS).unlink(), (src / conv.DRAFT).unlink()
    conv.main(["--src", str(src), "--out", str(out), "--draft-only"])
    assert asked[-1] == conv.SMALL + [conv.DRAFT]  # (the weights deleted after the build: not downloaded again)


@pytest.mark.parametrize("rows", [1, 2])
def test_a_build_with_an_adapter_clears_the_experts_output_of_its_direction(rows):
    """A build made with --adapter has a direction a layer (``mlp.ablate``): the experts' summed output is without
    it, in the reference and in the kernels (combine's own variant), a token at a time and a few tokens a pass."""
    ref, fused = _model(experts="q", adapter=True), _model(experts="q", adapter=True, fused=True)
    plain = _model(experts="q")
    layer = ref.model.language_model.layers[2]
    x = mx.random.normal((1, 3, 64))
    d = layer.mlp.ablate
    assert float(mx.abs(layer.mlp(x) @ d).max()) < 1e-4 < float(mx.abs(plain.model.language_model.layers[2].mlp(x) @ d).max())
    toks = mx.random.randint(0, 120, (rows, 24), key=mx.random.key(8))
    a, b = ref.make_cache(), fused.make_cache()
    assert _close(fused(toks[:, :5], cache=b), ref(toks[:, :5], cache=a))
    assert not _close(plain(toks[:, :5]), ref(toks[:, :5]))  # (it is another model than the one without)
    for i in range(5, 14):
        assert _close(fused(toks[:, i:i + 1], cache=b), ref(toks[:, i:i + 1], cache=a)), i
    if rows == 1:  # (a text's few tokens in one pass of the kernels)
        assert _close(fused(toks[:, 14:20], cache=b), ref(toks[:, 14:20], cache=a))


def test_an_adapter_file_is_read_and_one_of_another_shape_refused(tmp_path):
    """convert_inkling reads a GGUF adapter's metadata and float tensors (dimensions the other way round, as weights
    have them), and ``add_adapter`` takes only a LoRA for this architecture."""
    import struct

    import numpy as np

    from locals import convert_inkling as conv

    def text(t):
        return struct.pack("<Q", len(t)) + t.encode()

    def gguf(arch):
        a, b = np.arange(6, dtype=np.float32), np.ones(4, dtype=np.float16)
        head = b"GGUF" + struct.pack("<IQQ", 3, 2, 2)
        head += text("general.architecture") + struct.pack("<I", 8) + text(arch)
        head += text("adapter.type") + struct.pack("<I", 8) + text("lora")
        head += text("x.lora_a") + struct.pack("<IQQIQ", 2, 3, 2, 0, 0)  # (GGUF: 3 wide, 2 rows)
        head += text("x.lora_b") + struct.pack("<IQIQ", 1, 4, 1, 32)
        pad = (-len(head)) % 32
        return head + b"\0" * pad + a.tobytes() + b"\0" * 8 + b.tobytes()

    path = tmp_path / "a.gguf"
    path.write_bytes(gguf("inkling2_loop"))
    meta, tensors = conv.read_gguf(path)
    assert meta["adapter.type"] == "lora" and tensors["x.lora_a"].shape == (2, 3) and tensors["x.lora_a"][1, 2] == 5
    assert tensors["x.lora_b"].dtype == np.float32 and tensors["x.lora_b"].tolist() == [1, 1, 1, 1]
    path.write_bytes(gguf("llama"))
    with pytest.raises(ValueError, match="not a LoRA adapter for inkling2_loop"):
        conv.add_adapter(tmp_path, path, tmp_path / "out")
