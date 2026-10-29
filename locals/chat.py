"""Interactive chat on the locals engine.

    locals chat                       # on the background server (started if it is not running)
    locals chat --local --model mlx-community/Qwen3.5-2B-MLX-4bit   # the model in this process

With a ChatML template (Qwen) the conversation is kept as one token stream
whose KV (and recurrent state) stays in the cache, so every turn only prefills
the new user message; other templates (Gemma, Llama, Mistral, ...) render the
conversation again each turn and reuse the cache where it still matches. Commands:
  /reset          start a new conversation
  /json <schema>  constrain the next reply to a JSON schema (file path or inline JSON)
  /n <N>          generate N alternative replies in parallel for the next message
  /stats          toggle the speed line after each reply
  /quit           exit
"""

import argparse
import json
import os
import sys
import time

from mlx_lm.sample_utils import make_sampler

from .defaults import DEFAULT_MODEL, MAC_BLOCK, MODEL_HELP, model_id
from .runtime import Runtime


class Conversation:
    """The prompt of each turn in the model's own chat template."""

    def __init__(self, tok, system=None, think=False):
        from . import formats

        self.tok = tok
        self.fmt = formats.detect(tok)
        self.chatml = "<|im_start|>" in formats.chat_template(tok)
        self.kw = {} if formats.thinking_style(tok) == "none" else {"enable_thinking": think}
        self.system = [{"role": "system", "content": system}] if system else []
        self.msgs = list(self.system)
        self.eos = set(tok.eos_token_ids)
        one = [{"role": "user", "content": "x"}]
        a = tok.apply_chat_template(one, add_generation_prompt=False, tokenize=False, **self.kw)
        b = tok.apply_chat_template(one, add_generation_prompt=True, tokenize=False, **self.kw)
        self.gen = b[len(a):] if b.startswith(a) else "<|im_start|>assistant\n"  # the template's generation prompt

    def prompt(self, msg, history):
        """Token ids of the prompt for user message ``msg``; ``history`` holds the tokens
        fed so far. ChatML continues them; other templates render the conversation again."""
        if history and self.chatml:
            delta = ("" if history[-1] in self.eos else "<|im_end|>") + "\n<|im_start|>user\n" + msg + "<|im_end|>\n" + self.gen
            return list(history) + self.tok.encode(delta, add_special_tokens=False)
        return self.tok.apply_chat_template(self.msgs + [{"role": "user", "content": msg}], add_generation_prompt=True, **self.kw)

    def add(self, msg, reply_ids):
        """Record a finished turn: the user message and the reply the model wrote."""
        from . import oai

        text = self.tok.decode([t for t in reply_ids if t not in self.eos])
        _, content = oai.split_reasoning(text, self.fmt)
        self.msgs += [{"role": "user", "content": msg}, {"role": "assistant", "content": content.strip()}]

    def reset(self):
        self.msgs = list(self.system)


def parallel_replies(rt, ids, n, max_tokens, temp, top_p):
    """n sampled replies to the same prompt, decoded as one batch
    (mlx-lm's pipelined BatchGenerator, running on our small-M kernels)."""
    from mlx_lm.generate import BatchGenerator

    gen = BatchGenerator(rt.model, stop_tokens=[[t] for t in rt.tokenizer.eos_token_ids],
                         sampler=make_sampler(temp, top_p))
    uids = gen.insert([ids] * n, [max_tokens] * n)
    out = {u: [] for u in uids}
    t_first = None
    try:
        while resps := gen.next_generated():
            if t_first is None:
                t_first = time.perf_counter()
            for r in resps:
                if r.finish_reason != "stop":
                    out[r.uid].append(r.token)
    finally:  # (Ctrl-C too)
        gen.close()
    dt = time.perf_counter() - (t_first or time.perf_counter())
    total = sum(len(v) for v in out.values())
    return [rt.tokenizer.decode(out[u]) for u in uids], {"tokens": total, "tps": total / dt if dt else 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="locals chat")
    ap.add_argument("--model", default=None, type=model_id, help=MODEL_HELP + " (default: the model used last)")
    ap.add_argument("--port", type=int, default=None, help="the server's port (default: the last one used, else 8321)")
    ap.add_argument("--system", default=None)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--think", action="store_true", help="let the model think before answering")
    ap.add_argument("--no-stats", action="store_true")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"],
                    help="compute dtype; auto = float16 on M1/M2 GPUs (no native bf16); implies --local")
    ap.add_argument("--local", action="store_true",
                    help="load the model in this process instead of using the background server (stop the server first)")
    ap.add_argument("--blocks", type=int, nargs="?", const=MAC_BLOCK, default=None,
                    help=f"a block-diffusion model (Clover) in this process: blocks of N tokens by diffusion (default {MAC_BLOCK}) "
                    "instead of a token a step; implies --local")
    args = ap.parse_args(argv)
    if args.port:  # (client.connect reads it, as for locals --port N chat)
        os.environ["LOCALS_PORT"] = str(args.port)
    args.local = args.local or args.dtype != "auto" or bool(args.blocks)

    if not args.local:  # on the background server: one copy of the model in memory, kept between commands
        from .client import chat

        return chat(args)
    args.model = args.model or DEFAULT_MODEL
    from .app import check_local_load

    check_local_load(args.model, "chat")
    if args.blocks:  # (Clover in blocks: its own decoding, locals/blockdiff.py)
        from .blockdiff import chat as blockdiff_chat
        from .blockdiff import is_block_diffusion

        if not is_block_diffusion(args.model):
            sys.exit(f"locals chat: --blocks: {args.model} is not a block-diffusion model (Clover)")
        return blockdiff_chat(args)
    print(f"loading {args.model} ...", file=sys.stderr, flush=True)
    rt = Runtime.load(args.model, dtype=args.dtype)
    tok = rt.tokenizer
    sampler = make_sampler(args.temp, args.top_p) if args.temp > 0 else None
    show_stats = not args.no_stats
    conv = Conversation(tok, args.system, args.think)
    history, cache = [], None  # tokens whose KV is in ``cache``
    schema = None
    n_alt = 1

    # warm-up: compile kernels and graph shapes before the first real turn
    rt.engine.generate(tok.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=True,
                                               enable_thinking=False), None, max_tokens=8, temp=args.temp, sampler=sampler)
    print("chat ready. /reset, /n <N>, /json <schema>, /stats, /quit", file=sys.stderr)
    while True:
        try:
            msg = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/reset":
            history, cache = [], None
            conv.reset()
            print("(new conversation)")
            continue
        if msg == "/stats":
            show_stats = not show_stats
            continue
        if msg.startswith("/n"):
            try:
                n_alt = max(1, int(msg[2:].strip() or 1))
            except ValueError:
                print("(/n takes a number: /n 3)")
                continue
            print(f"(next reply: {n_alt} alternatives in parallel)")
            continue
        if msg.startswith("/json"):
            arg = msg[5:].strip()
            try:
                if os.path.exists(arg):
                    with open(arg) as f:
                        schema = json.load(f)
                else:
                    schema = json.loads(arg)
                rt.compile(schema)  # a schema the engine cannot compile is reported now, not mid-reply
            except Exception as e:  # noqa: BLE001
                schema = None
                print(f"(not a usable schema: {type(e).__name__}: {e})")
                continue
            print("(next reply is constrained to the schema)")
            continue

        ids = conv.prompt(msg, history)
        if cache is not None and ids[:len(history)] != history:  # the template wrote the past differently
            history, cache = [], None
        if n_alt > 1:
            try:
                texts, stat = parallel_replies(rt, ids, n_alt, args.max_tokens, args.temp or 0.8, args.top_p)
            except KeyboardInterrupt:  # (it ended the chat)
                print("\n(stopped; this turn is not kept)")
                n_alt = 1
                continue
            for i, t in enumerate(texts):
                print(f"\n--- reply {i + 1}\n{t}")
            if show_stats:
                print(f"[{n_alt} replies, {stat['tokens']} tokens, {stat['tps']:.1f} tok/s aggregate, "
                      f"{stat['tps'] / n_alt:.1f} per reply]", file=sys.stderr, flush=True)
            # continue the conversation from reply 1 (re-prefilled next turn)
            conv.add(msg, tok.encode(texts[0], add_special_tokens=False))
            history = tok.encode(tok.decode(ids) + texts[0] + "<|im_end|>", add_special_tokens=False) if conv.chatml else []
            cache, n_alt = None, 1
            continue
        ctl = None
        if schema is not None:
            c = rt.compile(schema)
            ctl = c.controller()  # a PDAController for recursive schemas
        detok = tok.detokenizer
        detok.reset()

        def on_token(t):
            detok.add_token(t)
            seg = detok.last_segment
            if seg:
                print(seg, end="", flush=True)

        try:
            res = rt.engine.generate(ids, ctl, max_tokens=args.max_tokens, temp=args.temp,
                                     sampler=None if ctl else sampler, cache=cache, on_token=on_token)
        except KeyboardInterrupt:  # Ctrl-C stops this reply, not the chat
            print("\n(stopped; this turn is not kept)")
            history, cache, schema = [], None, None
            continue
        detok.finalize()
        print(detok.last_segment, end="", flush=True)
        schema = None
        conv.add(msg, res.tokens)
        history, cache = list(res.fed), res.cache
        if show_stats:
            n = len(res.tokens)
            print(f"\n[{n} tokens, {n / res.decode_s if res.decode_s else 0:.1f} tok/s, "
                  f"prefill {len(ids) - res.cached_tokens} new tokens in {res.prefill_s*1e3:.0f} ms, "
                  f"context {len(history)} tokens]", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
