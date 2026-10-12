"""locals as a grammar library: compile once, use in any decode loop.

    python examples/02_grammar_library.py [model]

Three ways to use a compiled grammar:
  1. as a logits processor for mlx-lm's own `generate` (masks only);
  2. step by step with a Matcher, in a decode loop you write yourself;
  3. with our engine, which also feeds forced text in the same forward pass.
"""

import json
import sys

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache

import locals

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3.5-4B-MLX-4bit"
SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"},
                                           "population_millions": {"type": "number"}}, "required": ["city", "country", "population_millions"]}

model, tokenizer = load(MODEL)
compiler = locals.GrammarCompiler(locals.Vocabulary.from_model(model, tokenizer))
grammar = compiler.compile_json_schema(SCHEMA)
print(f"grammar: {grammar.n_states} states, {grammar.table_bytes / 1e6:.0f} MB of tables, recursive={grammar.recursive}")

messages = [{"role": "user", "content": "Give me one large European capital as JSON with city, country and population in millions."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=True)

# 1. mlx-lm's generator with the grammar as a logits processor
text = generate(model, tokenizer, prompt, max_tokens=80, logits_processors=[locals.logits_processor(grammar)], verbose=False)
print("\n1. mlx-lm + logits processor:", text, "->", json.loads(text))

# 2. your own decode loop with a Matcher: mask, sample, accept, jump forward
matcher = grammar.matcher()
cache = make_prompt_cache(model)
ids = list(prompt)
logits = model(mx.array([ids]), cache=cache)[:, -1, :]
out = []
while len(out) < 80 and not matcher.is_terminated():
    y = int(mx.argmax(matcher.apply_token_mask(logits[0])).item())
    if y in matcher.eos_ids:
        break
    matcher.accept_token(y, check=False)
    step = [y] + matcher.jump_forward()  # forced tokens ride along in the same forward pass
    out.extend(step)
    logits = model(mx.array([step]), cache=cache)[:, -1, :]
print("2. Matcher loop:", tokenizer.decode(out))

# 3. our engine, same grammar object
rt = locals.Runtime(model, tokenizer)
res = rt.generate(messages[0]["content"], grammar=grammar, max_tokens=80)
print(f"3. locals engine: {res.text} ({res.passes} passes, {res.forced_tokens} forced tokens)")
