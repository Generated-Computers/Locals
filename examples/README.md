# Examples

All examples take the model id as their first argument and default to
`mlx-community/Qwen3.5-4B-MLX-4bit` (about 2.5 GB, downloaded on first use).
`mlx-community/Qwen3.5-2B-MLX-4bit` is a faster choice for a first try.

| file | what it shows |
|---|---|
| `01_structured_output.py` | `locals.load(...)` and `Runtime.generate` with a JSON schema, a regex and a token limit |
| `02_grammar_library.py` | the grammar library: one compiled grammar used as an mlx-lm logits processor, in a hand-written decode loop with `Matcher`, and in our engine |
| `03_agents_pool.py` | many agents on one model with `BatchService`: continuous batching, prompt cache, constrained replies |
| `04_openai_client.py` | `locals serve` from an OpenAI-compatible client: streaming, `response_format`, tools |
| `05_tool_calls.py` | a tool-call grammar compiled for the model's own function-calling format, decoded and parsed without a server |
| `person.json` | the JSON schema used by the quick start (`locals generate --schema examples/person.json`) |

Run one GPU job at a time; a second model process on a 16 GB machine competes
for memory with the first.
