"""Many agents, one model: BatchService.

    python examples/03_agents_pool.py [model] [n_agents]

Every agent runs in its own thread and sends its prompts to one BatchService.
The service owns the model: agents that wait for the model at the same moment
decode together as one batch (continuous batching), a conversation's prefix is
served from the prompt cache, so a later turn only prefills the new text, and
every reply is constrained, so it always parses. This is the machinery behind
`locals serve` and `locals agents`.
"""

import json
import sys
import threading
import time

import locals

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3.5-4B-MLX-4bit"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8

SYSTEM = ("You are a research assistant. Answer with one JSON object: 'answer' (one sentence) and "
          "'confidence' (a number between 0 and 1).")
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}, "confidence": {"type": "number"}},
          "required": ["answer", "confidence"]}
QUESTIONS = ["Why is the sky blue?", "What does a compiler do?", "How do vaccines work?", "What is a prime number?",
             "Why do we have leap years?", "What is photosynthesis?", "How does a hash table work?", "What causes tides?"]
FOLLOW_UP = "Now explain it to a ten-year-old."

svc = locals.BatchService("locals", MODEL, slots=16)  # loads the model in a worker thread; blocks until ready
tok = svc.tokenizer


def render(messages):
    return tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=True)


# Every agent's conversation starts with the same system prompt: prefill that prefix once.
a, b = (render([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]) for q in QUESTIONS[:2])
shared = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
svc.prime(a[:shared])

results = [None] * N


def agent(i):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}]
    first = svc.submit(render(messages), schema=SCHEMA, max_tokens=120).wait()
    # Turn 2 re-renders the conversation through the chat template. The cache serves the
    # longest stored prefix (here the primed system prompt). `locals serve` and `locals agents`
    # instead continue on the exact tokens the model produced, so the whole previous turn is
    # served and only the new message is prefilled.
    messages += [{"role": "assistant", "content": first.text}, {"role": "user", "content": FOLLOW_UP}]
    ids = render(messages)
    second = svc.submit(ids, schema=SCHEMA, max_tokens=120).wait()
    results[i] = (json.loads(first.text), json.loads(second.text), second.cached, len(ids))


threads = [threading.Thread(target=agent, args=(i,), daemon=True) for i in range(N)]
t0 = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.perf_counter() - t0

for i, (a, b, cached, n) in enumerate(results):
    print(f"agent {i}: {a['answer'][:60]!r} ({a['confidence']})\n         -> {b['answer'][:60]!r}  [turn 2: {cached}/{n} prompt tokens from cache]")
s = svc.stats()
print(f"\n{N} agents x 2 turns in {wall:.1f} s; {s['tokens_out']} output tokens ({s['tokens_out'] / wall:.0f} tok/s), "
      f"{s['cached_tokens']}/{s['prompt_tokens']} prompt tokens served from the cache")
svc.close()
