"""Block diffusion decoding, for DiffuSpace's Clover (locals/models/clover_moe.py).

A response is written ``block_size`` positions at a time. A block starts as mask tokens. Each step
runs the model over it in preview mode (the block reads the cached prefix and sees itself both ways;
nothing is cached) and takes, at every position not resolved yet, the most likely token (or a sampled
one) and its probability. The positions whose probability is above ``threshold`` are resolved, or the
most confident one when none is. Once every position is, the block is run once more causally (commit),
which extends the caches, and the next block starts. A response ends at its first end token, once the
block that holds it is resolved.

The settings and the response penalties (presence, frequency and repetition over the response so far;
ramps toward the end tokens and ``</think>`` late in a long response) are those of DiffuSpace's Hugging
Face code (modeling_clover_moe.py, sampling_clover_moe.py); the defaults, the model's
generation_config.json, but blocks of 8 tokens (defaults.MAC_BLOCK; the model's are 32).

Locals serves Clover a token a step by its causal pass unless asked for blocks (``locals serve --blocks``):
on a Mac that is the faster way. This module is the other one, and ``stream_speculative`` DiffuSpace's
own serving mode (blocks drafted by a preview, checked by the causal pass).

    model, tokenizer, settings = load("~/models/Clover-35B-A3B-Preview-MLX-mixed-3-4bit")
    for r in stream_generate(model, tokenizer, prompt, settings):
        print(r.text, end="", flush=True)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import mlx.core as mx
import numpy as np

PREFILL_CHUNK = 2048  # prompt tokens a commit pass


@dataclass
class Settings:
    max_tokens: int = 24576
    block_size: int = 32
    threshold: float = 0.95
    temperature: float = 0.0  # 0: the most likely token (generation_config's do_sample false)
    top_p: float = 1.0
    top_k: int = 0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 0.0  # added to the end tokens' logits, ramping up from length_penalty_start
    length_penalty_start: int = 8192
    length_penalty_window: int = 8192
    think_end_bias: float = 0.0  # added to </think>'s logit, ramping up from think_end_start, until it is written
    think_end_start: int = 4096
    think_end_window: int = 4096
    think_end_token_id: Optional[int] = None
    eos_token_ids: Sequence[int] = ()
    pad_token_id: Optional[int] = None
    seed: Optional[int] = None

    @classmethod
    def from_generation_config(cls, cfg: dict, **overrides) -> "Settings":
        """The settings of a Hugging Face generation_config.json (Clover's names), then ``overrides``."""
        eos = cfg.get("eos_token_id")
        kw = dict(
            max_tokens=cfg.get("max_new_tokens"),
            temperature=float(cfg.get("temperature") or 1.0) if cfg.get("do_sample") else 0.0,
            top_p=cfg.get("top_p"),
            top_k=cfg.get("top_k") or 0,
            presence_penalty=cfg.get("presence_penalty"),
            frequency_penalty=cfg.get("frequency_penalty"),
            repetition_penalty=cfg.get("repetition_penalty"),
            length_penalty=cfg.get("dllm_length_penalty"),
            length_penalty_start=cfg.get("dllm_length_penalty_start"),
            length_penalty_window=cfg.get("dllm_length_penalty_window"),
            think_end_bias=cfg.get("dllm_think_end_bias"),
            think_end_start=cfg.get("dllm_think_end_start"),
            think_end_window=cfg.get("dllm_think_end_window"),
            think_end_token_id=cfg.get("dllm_think_end_token_id"),
            eos_token_ids=tuple([eos] if isinstance(eos, int) else eos or ()),
            pad_token_id=cfg.get("pad_token_id"),
        )
        s = cls(**{k: v for k, v in kw.items() if v is not None})
        return replace(s, **overrides)


def load(path: str, **overrides):
    """(model, tokenizer, settings) for a Clover checkpoint (a local folder or a Hugging Face repo): the whole model
    (BlockModel), its generation config, blocks of MAC_BLOCK tokens unless ``block_size`` is given."""
    from .defaults import MAC_BLOCK

    overrides.setdefault("block_size", MAC_BLOCK)
    from mlx_lm.utils import _download, load_model, load_tokenizer

    from .models import clover_moe, register

    register()
    local = Path(path).expanduser()
    folder = local if local.exists() else Path(_download(path))
    # the whole model (both passes): mlx-lm's loader builds the causal pass alone (clover_moe.Model)
    model, config = load_model(folder, get_model_classes=lambda config: (clover_moe.BlockModel, clover_moe.BlockArgs))
    tokenizer = load_tokenizer(folder, eos_token_ids=config.get("eos_token_id"))
    # the weights kept in memory (as locals.lm.patch does): unwired, a 24 GB Mac short of memory paged them out and
    # each step waited on them
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    gen = folder / "generation_config.json"
    cfg = json.loads(gen.read_text()) if gen.exists() else {}
    settings = Settings.from_generation_config(cfg, **overrides)
    if model.args.denoising_config and "threshold" not in overrides:
        settings = replace(settings, threshold=model.args.denoising_config.get("threshold", settings.threshold))
    return model, tokenizer, settings


class Penalties:
    """Clover's response penalties, over the tokens of the blocks resolved so far (never the prompt's)."""

    def __init__(self, s: Settings, vocab: int, stop_ids: Sequence[int]):
        self.s = s
        self.length = 0
        self.closed = False  # the response holds </think>
        self.active = bool(s.presence_penalty or s.frequency_penalty or s.repetition_penalty != 1
                           or s.length_penalty or s.think_end_bias)
        self.counts = mx.zeros((vocab,), dtype=mx.float32)
        stop = sorted({i for i in stop_ids if 0 <= i < vocab})
        self.stop = mx.zeros((vocab,), dtype=mx.float32).at[mx.array(stop)].add(1.0) if stop else None
        c = s.think_end_token_id
        self.close = mx.zeros((vocab,), dtype=mx.float32).at[mx.array([c])].add(1.0) if c is not None else None

    def commit(self, tokens: List[int]) -> None:
        self.length += len(tokens)
        if tokens:
            self.counts = self.counts.at[mx.array(tokens)].add(1.0)
        if self.s.think_end_token_id in tokens:
            self.closed = True

    def apply(self, logits: mx.array, rows: List[int], block: List[int], mask_id: int) -> mx.array:
        """``logits`` (len(rows), vocab, float32) of the block's positions ``rows``, penalized."""
        if not self.active:
            return logits
        s = self.s

        def ramp(start, window, strength):
            return [min(max((self.length + r + 1 - start) / window, 0.0), 1.0) * strength for r in rows]

        zeros = [0.0] * len(rows)
        stop = ramp(s.length_penalty_start, s.length_penalty_window, s.length_penalty) if self.stop is not None else zeros
        visible = any(t == s.think_end_token_id for t in block if t != mask_id)
        close = (ramp(s.think_end_start, s.think_end_window, s.think_end_bias)
                 if self.close is not None and not (self.closed or visible) else zeros)
        f32 = mx.float32
        return _penalize(logits, self.counts, mx.array(s.frequency_penalty, f32), mx.array(s.presence_penalty, f32),
                         mx.array(s.repetition_penalty, f32), mx.array(stop, f32)[:, None],
                         self.stop if self.stop is not None else mx.zeros_like(self.counts), mx.array(close, f32)[:, None],
                         self.close if self.close is not None else mx.zeros_like(self.counts))


@partial(mx.compile, shapeless=True)
def _penalize(logits, counts, frequency, presence, repetition, stop_ramp, stop, close_ramp, close):
    """The penalties in the reference's order, one graph (a strength of 0, or 1 for the repetition penalty, changes
    no value: adding 0 or dividing by 1 is exact)."""
    seen = counts > 0
    logits = logits - counts * frequency
    logits = logits - seen.astype(mx.float32) * presence
    logits = mx.where(seen, mx.where(logits < 0, logits * repetition, logits / repetition), logits)
    return logits + stop_ramp * stop + close_ramp * close


def predict(logits: mx.array, s: Settings, key: Optional[mx.array] = None):
    """Each row's token (the most likely, or sampled with the temperature, top-k and top-p) and its
    probability at temperature 1, which is what the threshold compares."""
    if s.temperature > 0:
        scaled = logits / s.temperature
        if s.top_k and 0 < s.top_k < scaled.shape[-1]:
            cutoff = mx.min(mx.topk(scaled, s.top_k, axis=-1), axis=-1, keepdims=True)
            scaled = mx.where(scaled < cutoff, -mx.inf, scaled)
        if s.top_p is not None and s.top_p < 1:
            order = mx.argsort(-scaled, axis=-1)
            ranked = mx.take_along_axis(scaled, order, axis=-1)
            probs = mx.softmax(ranked, axis=-1)
            ranked = mx.where((mx.cumsum(probs, axis=-1) - probs) > s.top_p, -mx.inf, ranked)
            scaled = mx.put_along_axis(mx.zeros_like(scaled), order, ranked, axis=-1)
        token = mx.random.categorical(scaled, key=key)
    else:
        token = mx.argmax(logits, axis=-1)
    picked = mx.take_along_axis(logits, token[:, None], axis=-1)[:, 0]
    return token, mx.exp(picked - mx.logsumexp(logits, axis=-1))


@dataclass
class Response:
    text: str  # the text of this block's tokens
    tokens: List[int]  # this block's tokens of the response (to the end token)
    steps: int  # preview passes this block took
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int  # the response so far
    generation_tps: float
    steps_total: int
    finish_reason: Optional[str] = None  # "stop", "length", or None while it goes on


def _eval_cache(cache) -> None:
    mx.eval([a for c in cache for a in (c.state[:2] if hasattr(c, "keys") else c.cache) if a is not None])


def stream_generate(model, tokenizer, prompt: Union[str, List[int]], settings: Optional[Settings] = None,
                    cache=None, cached: int = 0, ignore_eos: bool = False, commit_last: bool = False,
                    **overrides) -> Iterator[Response]:
    """One ``Response`` a resolved block. ``prompt``: text (tokenized as it is, so a chat prompt is
    the chat template's output) or token ids. ``cache`` (model.make_cache()) holds the first ``cached``
    prompt tokens already; it ends holding the prompt and the response's blocks but the last, or every
    response token with ``commit_last`` (for a next prompt that continues this one). ``ignore_eos``: end tokens
    do not end the response (a benchmark's fixed length)."""
    s = replace(settings or Settings(), **overrides)
    mask_id = model.args.mask_token_id
    ids = list(prompt) if not isinstance(prompt, str) else tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        raise ValueError("an empty prompt")
    eos = sorted(set(s.eos_token_ids) or set(tokenizer.eos_token_ids))
    pad = s.pad_token_id if s.pad_token_id is not None else (eos[0] if eos else 0)
    B = s.block_size
    if B <= 0 or model.args.block_size % B:
        raise ValueError(f"block_size must divide the model's ({model.args.block_size})")
    vocab = model.args.vocab_size
    no_mask = mx.zeros((vocab,), dtype=mx.float32).at[mx.array([mask_id])].add(-mx.inf)
    key = mx.random.key(s.seed) if s.seed is not None else None

    tic = time.perf_counter()
    if cache is None:
        cache, cached = model.make_cache(), 0
    if not 0 <= cached < len(ids):
        raise ValueError("the cache must hold fewer tokens than the prompt")
    for i in range(cached, len(ids), PREFILL_CHUNK):
        model.model(mx.array(ids[i : i + PREFILL_CHUNK])[None], cache)
        _eval_cache(cache)
    prompt_tps = (len(ids) - cached) / (time.perf_counter() - tic)

    pen = Penalties(s, vocab, eos + [pad])
    detok = tokenizer.detokenizer
    detok.reset()
    response: List[int] = []
    steps_total = 0
    tic = time.perf_counter()
    while len(response) < s.max_tokens:
        remaining = min(B, s.max_tokens - len(response))
        block = [mask_id] * B
        pending = list(range(B))
        steps = 0
        while pending:
            logits = model(mx.array(block)[None], cache, preview=True, rows=mx.array(pending))[0].astype(mx.float32)
            logits = pen.apply(logits + no_mask, pending, block, mask_id)
            if key is not None:
                key, sub = mx.random.split(key)
            else:
                sub = None
            token, conf = predict(logits, s, sub)
            token, conf = token.tolist(), conf.tolist()
            take = [j for j, c in enumerate(conf) if c > s.threshold] or [int(np.argmax(conf))]
            for j in take:
                block[pending[j]] = token[j]
            taken = set(take)
            pending = [p for j, p in enumerate(pending) if j not in taken]
            steps += 1
        steps_total += steps
        kept, finish = [], None
        for t in block[:remaining]:
            kept.append(t)
            if t in eos and not ignore_eos:
                finish = "stop"
                break
        response += kept
        pen.commit(kept)
        if finish is None and len(response) >= s.max_tokens:
            finish = "length"
        if finish is None:
            model.model(mx.array(block)[None], cache)  # commit: evaluated with the next block's first preview
        elif commit_last:
            model.model(mx.array(kept)[None], cache)
            _eval_cache(cache)
        for t in kept:
            if t not in eos:
                detok.add_token(t)
        if finish:
            detok.finalize()
        elapsed = time.perf_counter() - tic
        yield Response(detok.last_segment, kept, steps, len(ids), prompt_tps, len(response),
                       len(response) / max(elapsed, 1e-9), steps_total, finish)
        if finish:
            return


def accepted(logits: mx.array, drafts: mx.array, rule: str = "threshold", ratio: float = 1.0, delta: float = 0.6) -> mx.array:
    """How many of ``drafts`` (W - 1) the causal pass's ``logits`` (W, vocab; row j for the token after position j)
    accept, from the first: DiffuSpace's rules. ``exact``: the draft is row j's argmax. ``threshold``: its probability is
    at least min(ratio, delta * exp(-entropy of row j)), and above 0."""
    L = logits[: drafts.size]
    if rule == "exact":
        ok = mx.argmax(L, axis=-1) == drafts
    else:
        logp = L - mx.logsumexp(L, axis=-1, keepdims=True)
        p = mx.exp(logp)
        entropy = -mx.sum(mx.where(p > 0, p * logp, 0.0), axis=-1)
        pd = mx.exp(mx.take_along_axis(logp, drafts[:, None], axis=-1)[:, 0])
        ok = (pd > 0) & (pd >= mx.minimum(ratio, delta * mx.exp(-entropy)))
    return mx.sum(mx.cumprod(ok.astype(mx.int32)))


def stream_speculative(model, tokenizer, prompt: Union[str, List[int]], settings: Optional[Settings] = None,
                       width: int = 16, rule: str = "threshold", ratio: float = 1.0, delta: float = 0.6,
                       ignore_eos: bool = False, **overrides) -> Iterator[Response]:
    """Clover's self-speculative decoding (DiffuSpace's SGLang and llama.cpp backends): each round a preview of
    [the last token, width - 1 mask tokens] drafts the masked positions (argmax), one causal pass over [the last
    token, the drafts] checks them as an autoregressive model does (row j's logits for the token after j), the drafts
    accepted from the first (``accepted``) and one more token from the row after them are the round's output, and the
    caches drop the rest (``Model.rewind``). ``width=1``: no drafts, plain autoregressive decoding with the causal
    pass. One ``Response`` a round (``steps``: the passes it took)."""
    s = replace(settings or Settings(), **overrides)
    mask_id = model.args.mask_token_id
    ids = list(prompt) if not isinstance(prompt, str) else tokenizer.encode(prompt, add_special_tokens=False)
    eos = set(s.eos_token_ids) or set(tokenizer.eos_token_ids)
    pad = s.pad_token_id if s.pad_token_id is not None else (min(eos) if eos else 0)
    vocab = model.args.vocab_size
    no_mask = mx.zeros((vocab,), dtype=mx.float32).at[mx.array([mask_id])].add(-mx.inf)
    key = mx.random.key(s.seed) if s.seed is not None else None
    head = model.lm_head if not model.args.tie_word_embeddings else model.model.embed_tokens.as_linear

    def sample(row):
        nonlocal key
        sub = None
        if key is not None:
            key, sub = mx.random.split(key)
        return predict(row, s, sub)[0]

    tic = time.perf_counter()
    cache = model.make_cache()
    for i in range(0, len(ids), PREFILL_CHUNK):
        h = model.model(mx.array(ids[i : i + PREFILL_CHUNK])[None], cache)
    pen = Penalties(s, vocab, sorted(eos) + [pad])
    first = sample(pen.apply(head(h[:, -1:])[0].astype(mx.float32) + no_mask, [0], [], mask_id))
    pending = int(first.item())
    prompt_tps = len(ids) / (time.perf_counter() - tic)
    gdn = [c for layer, c in zip(model.layers, cache) if layer.is_linear]
    response: List[int] = []
    detok = tokenizer.detokenizer
    detok.reset()
    passes, tic = 1, time.perf_counter()
    new = [pending]
    while True:
        kept, finish = [], None
        for t in new:
            if len(response) >= s.max_tokens:
                finish = "length"
                break
            kept.append(t)
            response.append(t)
            if t in eos and not ignore_eos:
                finish = "stop"
                break
        if finish is None and len(response) >= s.max_tokens:
            finish = "length"
        pen.commit(kept)
        for t in kept:
            if t not in eos:
                detok.add_token(t)
        if finish:
            detok.finalize()
        elapsed = time.perf_counter() - tic
        yield Response(detok.last_segment, kept, 0, len(ids), prompt_tps, len(response), len(response) / max(elapsed, 1e-9),
                       passes, finish)
        if finish:
            return
        if width > 1:
            block = [pending] + [mask_id] * (width - 1)
            rows = list(range(1, width))
            lg = model(mx.array(block)[None], cache, preview=True, rows=mx.array(rows))[0].astype(mx.float32)
            drafts = mx.argmax(pen.apply(lg + no_mask, rows, block, mask_id), axis=-1)
            cand = mx.concatenate([mx.array([pending]), drafts.astype(mx.int32)])
        else:
            drafts, cand = mx.zeros((0,), dtype=mx.int32), mx.array([pending])
        for c in gdn:
            c.save_replay = True
        h = model.model(cand[None], cache)
        for c in gdn:
            c.save_replay = False
        L = pen.apply(head(h)[0].astype(mx.float32) + no_mask, list(range(width)), [pending], mask_id)
        n = int(accepted(L, drafts, rule, ratio, delta).item()) if width > 1 else 0
        bonus = int(sample(L[n : n + 1]).item())
        model.rewind(cache, width - 1 - n)
        passes += 2 if width > 1 else 1
        new = (drafts[:n].tolist() if n else []) + [bonus]
        pending = bonus


def generate(model, tokenizer, prompt: Union[str, List[int]], settings: Optional[Settings] = None,
             verbose: bool = False, **overrides) -> str:
    """The whole response's text (see ``stream_generate``)."""
    text, last = [], None
    for r in stream_generate(model, tokenizer, prompt, settings, **overrides):
        text.append(r.text)
        if verbose:
            print(r.text, end="", flush=True)
        last = r
    if verbose and last is not None:
        print(f"\n{'=' * 10}\nprompt: {last.prompt_tokens} tokens, {last.prompt_tps:.1f} tokens/s\n"
              f"response: {last.generation_tokens} tokens, {last.generation_tps:.1f} tokens/s, "
              f"{last.generation_tokens / max(last.steps_total, 1):.2f} tokens a step")
    return "".join(text)


BLOCK_DIFFUSION_TYPES = ("clover_moe",)


def is_block_diffusion(model: str) -> bool:
    """Whether ``model`` (a folder, or a Hugging Face repo already downloaded) is a block-diffusion model."""
    path = Path(model).expanduser()
    if not path.exists():
        try:
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(model, local_files_only=True, allow_patterns=["config.json"]))
        except Exception:  # noqa: BLE001
            return False
    try:
        return json.loads((path / "config.json").read_text()).get("model_type") in BLOCK_DIFFUSION_TYPES
    except (OSError, ValueError):
        return False


def chat(args) -> None:
    """``locals chat --blocks`` (a block-diffusion model in this process): the conversation rendered again each turn."""
    import sys

    from .fuse import fuse_model

    print(f"loading {args.model} ...", file=sys.stderr, flush=True)
    model, tokenizer, settings = load(args.model, **({"block_size": args.blocks} if getattr(args, "blocks", None) else {}))
    fuse_model(model)
    mx.clear_cache()
    system = [{"role": "system", "content": args.system}] if args.system else []
    msgs = list(system)
    print(f"chat ready (block diffusion: a block of {settings.block_size} tokens at a time). /reset, /quit", file=sys.stderr)
    while True:
        try:
            msg = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not msg:
            continue
        if msg == "/quit":
            return
        if msg == "/reset":
            msgs = list(system)
            print("(new conversation)")
            continue
        msgs.append({"role": "user", "content": msg})
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=bool(args.think))
        text, last = [], None
        try:
            for r in stream_generate(model, tokenizer, prompt, settings, max_tokens=args.max_tokens):
                print(r.text, end="", flush=True)
                text.append(r.text)
                last = r
        except KeyboardInterrupt:
            print("\n(stopped; this turn is not kept)")
            msgs.pop()
            continue
        reply = "".join(text)
        msgs.append({"role": "assistant", "content": ("<think>\n" if args.think else "") + reply})
        if last is not None and not args.no_stats:
            print(f"\n[{last.generation_tokens} tokens, {last.generation_tps:.1f} tok/s, "
                  f"{last.generation_tokens / max(last.steps_total, 1):.2f} tokens a step, prompt {last.prompt_tokens} "
                  f"tokens at {last.prompt_tps:.0f} tok/s]", file=sys.stderr, flush=True)


def main(argv=None):
    """python -m locals.blockdiff --model PATH --prompt "..." (a user message, in the chat template)"""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m locals.blockdiff", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="a Clover checkpoint for MLX (a folder or a Hugging Face repo)")
    ap.add_argument("--prompt", required=True, help="the user message, or @file to read it")
    ap.add_argument("--system", default=None)
    ap.add_argument("--no-think", action="store_true", help="an empty <think> block: answer without reasoning")
    ap.add_argument("--raw", action="store_true", help="the prompt as it is, no chat template")
    ap.add_argument("--max-tokens", type=int, default=None, help="(default: the model's, 24576)")
    ap.add_argument("--threshold", type=float, default=None, help="(default: the model's, 0.95)")
    ap.add_argument("--block-size", type=int, default=None, help="(default 8, the fastest on a Mac; the model's: 32)")
    ap.add_argument("--temp", type=float, default=None, help="0: the most likely token (the model's default)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-fuse", action="store_true", help="the model's own computation (no Locals kernels)")
    args = ap.parse_args(argv)
    over = {k: v for k, v in dict(max_tokens=args.max_tokens, threshold=args.threshold, block_size=args.block_size,
                                  temperature=args.temp, seed=args.seed).items() if v is not None}
    model, tokenizer, settings = load(args.model, **over)
    if not args.no_fuse:
        from .fuse import fuse_model

        fuse_model(model)
        mx.clear_cache()
    text = Path(args.prompt[1:]).read_text() if args.prompt.startswith("@") else args.prompt
    if not args.raw:
        msgs = ([{"role": "system", "content": args.system}] if args.system else []) + [{"role": "user", "content": text}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=not args.no_think)
    generate(model, tokenizer, text, settings, verbose=True)


if __name__ == "__main__":
    main()
