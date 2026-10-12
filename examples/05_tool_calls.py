"""Tool calls without a server: the grammar of the model's function-calling format.

    python examples/05_tool_calls.py [model]

`GrammarCompiler.compile_tool_calls(tools)` builds the grammar of a reply in
Qwen's native format: free text that cannot start a call, then calls whose
tags are literal and whose names are the offered tools. The tags are forced
text, so they are written without a forward pass, and every reply parses.
`locals.oai.parse_reply` turns the finished reply into OpenAI-style tool calls,
with argument values coerced to the types the tool declares.

`typed=True` also fixes parameter names and value types in the grammar. That
automaton is large (about a thousand states), so pair it with the walk kernel
(`locals serve --walk --typed-tools`) rather than token tables.
"""

import json
import sys

from mlx_lm import load

import locals
from locals import oai

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3.5-4B-MLX-4bit"
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city.",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                                                       "unit": {"enum": ["celsius", "fahrenheit"]}},
                                                     "required": ["city"]}}},
    {"type": "function", "function": {"name": "convert", "description": "Convert an amount between currencies.",
                                      "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from": {"type": "string"},
                                                                                       "to": {"type": "string"}},
                                                     "required": ["amount", "from", "to"]}}},
]

model, tokenizer = load(MODEL)
rt = locals.Runtime(model, tokenizer)
compiler = locals.GrammarCompiler(locals.Vocabulary.from_model(model, tokenizer))
grammar = compiler.compile_tool_calls(TOOLS)
print(f"reply grammar: {grammar.n_states} states, {grammar.table_bytes / 1e6:.0f} MB of token tables")

messages = [{"role": "user", "content": "What is the weather in Kyoto right now, and how much is 100 euros in yen?"}]
prompt = tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, enable_thinking=False, tokenize=True)
res = rt.generate(prompt, grammar=grammar, max_tokens=200)
print("\nraw reply:\n" + res.text)

reasoning, content, calls = oai.parse_reply(res.text, TOOLS)
print("\ncontent:", repr(content))
for c in calls:
    print("call:", c["function"]["name"], json.loads(c["function"]["arguments"]))
print(f"\n{len(res.tokens)} tokens in {res.passes} forward passes; {res.forced_tokens} tokens were forced by the grammar")
