"""Command line entry point.

    locals generate --prompt "Extract ..." --schema schema.json [--model ID]
    locals generate --prompt "Write a haiku" --n 4 --temp 0.9
"""

import argparse
import json
import os
import sys

from .defaults import DEFAULT_MODEL, MODEL_HELP, model_id
from .runtime import Runtime


def main(argv=None):
    ap = argparse.ArgumentParser(prog="locals generate")
    ap.add_argument("--model", default=None, type=model_id, help=MODEL_HELP + " (default: the model used last)")
    ap.add_argument("--port", type=int, default=None, help="the server's port (default: the last one used, else 8321)")
    ap.add_argument("--prompt", required=True, help="prompt text, or @file to read it")
    ap.add_argument("--schema", default=None, help="JSON schema file for structured output")
    ap.add_argument("--regex", default=None, help="regex constraint instead of a schema")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1, help="parallel samples sharing the prompt KV")
    ap.add_argument("--no-forced", action="store_true", help="disable forced-span compression")
    ap.add_argument("--no-fuse", action="store_true", help="keep separate q/k/v and gate/up matmuls")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"],
                    help="compute dtype; auto = float16 on M1/M2 GPUs (no native bf16)")
    ap.add_argument("--local", action="store_true", help="load the model in this process instead of using the background "
                    "server (implied by --dtype, --no-fuse, --no-forced; stop the server first)")
    args = ap.parse_args(argv)
    if args.port:  # (client.connect reads it, as for locals --port N chat)
        os.environ["LOCALS_PORT"] = str(args.port)

    # the inputs first: a wrong path or a broken schema used to surface as a traceback after the model loaded
    try:
        prompt = open(args.prompt[1:]).read() if args.prompt.startswith("@") else args.prompt
        schema = json.load(open(args.schema)) if args.schema else None
    except OSError as e:
        sys.exit(f"locals generate: cannot read {e.filename}: {e.strerror}")
    except ValueError as e:
        sys.exit(f"locals generate: {args.schema} is not valid JSON: {e}")
    if args.schema and args.regex:
        sys.exit("locals generate: give --schema or --regex, not both")
    if not (args.local or args.dtype != "auto" or args.no_fuse or args.no_forced):
        from .client import generate

        return generate(args, prompt, schema)  # on the background server (started if it is not running)
    args.model = args.model or DEFAULT_MODEL
    from .app import check_local_load

    check_local_load(args.model, "generate")
    print(f"loading {args.model} ...", file=sys.stderr, flush=True)
    rt = Runtime.load(args.model, fuse=not args.no_fuse, dtype=args.dtype)
    if args.n > 1:
        if args.schema or args.regex:
            sys.exit("--n with a constraint runs on the server: leave out --local, --dtype, --no-fuse and --no-forced")
        res = rt.sample(prompt, args.n, max_tokens=args.max_tokens, temp=args.temp or 0.8, seed=args.seed)
        for i, t in enumerate(res.texts):
            print(f"--- sample {i}\n{t}")
        if args.stats:
            print(f"[{res.tokens} tokens, prefill {res.prefill_s:.2f}s, decode {res.decode_s:.2f}s, "
                  f"{res.tokens / res.decode_s:.1f} tok/s aggregate, peak {res.peak_gb:.2f} GB]", file=sys.stderr)
        return
    res = rt.generate(prompt, schema=schema, regex=args.regex, max_tokens=args.max_tokens,
                      temp=args.temp, seed=args.seed, forced=not args.no_forced)
    print(res.text)
    if args.stats:
        print(f"[{len(res.tokens)} tokens in {res.passes} passes ({res.forced_tokens} forced), "
              f"prefill {res.prefill_s:.2f}s, decode {res.decode_s:.2f}s, {res.tok_per_s:.1f} tok/s]", file=sys.stderr)


if __name__ == "__main__":
    main()
