# Examples

Five short scripts show Locals used as a library and as a server. Run them from the repository root after `pip install -e .`, on an Apple-silicon Mac.

| File | What it shows |
|---|---|
| `01_structured_output.py` | `locals.load` and `Runtime.generate` with a JSON schema, a regex, and a tight token limit |
| `02_grammar_library.py` | One compiled grammar used three ways: as an mlx-lm logits processor, in a hand-written decode loop with `Matcher`, and in the Locals engine |
| `03_agents_pool.py` | Many agent threads sharing one model through `BatchService`: continuous batching, the prompt cache, constrained replies |
| `04_openai_client.py` | A running `locals serve` used from a plain-HTTP OpenAI client: streaming, `response_format`, tools |
| `05_tool_calls.py` | A tool-call grammar decoded and parsed into OpenAI-style calls without a server |
| `person.json` | The JSON schema used by the README's `locals generate` quick start |

## Before you run them

- **Model argument.** Examples 01, 02, 03 and 05 take a model id as their first argument. The default is `mlx-community/Qwen3.5-4B-MLX-4bit`, which mlx-lm downloads from Hugging Face on first use. `mlx-community/Qwen3.5-2B-MLX-4bit` is a faster choice for a first try.
- **Example 04 takes no model.** It talks to a server you start yourself (see below).
- **First load.** The first load of a new model shape tunes kernels (comment in `01_structured_output.py`), so the first run of a model is slower than later runs.
- **One GPU job at a time.** A second model process on a 16 GB machine competes for memory with the first. That includes a `locals serve` you left running.

## 01: structured output

```bash
python examples/01_structured_output.py [model]
```

The script loads the model once with `locals.load(MODEL)`, which returns a `Runtime`. It then makes three calls to `Runtime.generate`. A string prompt is rendered through the model's chat template before decoding.

1. **A JSON schema.** `rt.generate(prompt, schema=SCHEMA, max_tokens=200)` extracts a person record (name, age, email, status enum, tags) from a sentence.
   - It prints the text, then parses it with `json.loads`.
   - It then prints the token count, `passes` (forward passes), `forced_tokens` (tokens written without a pass), `tok_per_s` and `finish`.
   - Text the schema forces (keys, quotes, brackets) is written without running the model. That is why the pass count is below the token count.
2. **A regex.** `rt.generate("Pick a colour and a fruit.", regex=r"(red|green|blue) (apple|pear|plum)", max_tokens=8)`.
3. **A tight token limit.** The same schema with `max_tokens=28`. The output stays valid JSON: arrays close early and strings shorten.

## 02: the grammar library

```bash
python examples/02_grammar_library.py [model]
```

The model is loaded with mlx-lm's own `load`. The script compiles one grammar and then uses it three ways:

```python
grammar = locals.GrammarCompiler(locals.Vocabulary.from_model(model, tokenizer)).compile_json_schema(SCHEMA)
```

It first prints the grammar's `n_states`, `table_bytes` and `recursive`.

1. **mlx-lm with a logits processor.** It runs `generate(model, tokenizer, prompt, logits_processors=[locals.logits_processor(grammar)])`.
   - The processor only masks logits. mlx-lm samples one token per step, so forced text still costs a pass each.
   - One processor serves one sequence at a time.
   - A token the grammar does not allow raises `ValueError`.
2. **Your own decode loop with a `Matcher`.** Each step does the following:
   - mask the logits with `apply_token_mask`
   - take the argmax
   - `accept_token(y, check=False)`
   - append `jump_forward()`, the tokens the grammar now forces
   - feed `[y] + forced` to the model in one forward pass

   The loop ends at an EOS id (`matcher.eos_ids`), when `is_terminated()`, or after 80 tokens.
3. **The Locals engine.** `locals.Runtime(model, tokenizer).generate(text, grammar=grammar)` runs on the same model and grammar objects. It prints the pass and forced-token counts.

## 03: many agents, one model

```bash
python examples/03_agents_pool.py [model] [n_agents]     # n_agents defaults to 8
```

`locals.BatchService("locals", MODEL, slots=16)` loads the model in a worker thread and blocks until it is ready. Each agent runs in its own thread and calls `svc.submit(token_ids, schema=SCHEMA, max_tokens=120).wait()`. Requests that wait for the model at the same time decode together as one batch. Every reply follows a small schema (`answer`, `confidence`), so `json.loads` always succeeds.

The script shows three things:

- **Priming.** It renders two prompts, finds their common prefix (the shared system prompt), and prefills it once with `svc.prime(...)`.
- **Two turns per agent.** The second turn re-renders the whole conversation through the chat template and adds a follow-up question. It prints how many of turn 2's prompt tokens came from the cache (`second.cached`).
  - Here the cache can serve only the longest stored prefix, the primed system prompt, because re-rendering changes the tokens.
  - `locals serve` and `locals agents` instead continue on the exact tokens the model produced, so the whole previous turn is served from the cache.
- **Totals.** It prints wall time and output tokens per second, taken from `svc.stats()` (`tokens_out`, `cached_tokens`, `prompt_tokens`), then calls `svc.close()`.

## 04: an OpenAI-compatible client

```bash
locals serve                          # in another terminal: http://127.0.0.1:8321/v1
python examples/04_openai_client.py
```

The script uses only `urllib`, so it has no dependencies. The `openai` package works the same way with `base_url="http://127.0.0.1:8321/v1"` and any `api_key`. The address is hard-coded in `BASE`; change it if you started the server on another `--port`. If the server does not answer, the script exits with "is `locals serve` running?".

It reads `GET /models` and picks the first id that starts with `locals` and does not contain `thinking`. Then it sends three requests to `/chat/completions`:

1. **Streaming.** `"stream": true`. The server-sent events are printed as they arrive.
2. **`response_format`.** A `json_schema` response format; the reply is parsed with `json.loads`.
3. **Tools.** Two tools (`get_weather`, `convert`) and a question that needs both. The script prints each tool call with its parsed arguments, then the `usage` block. The reply is decoded under a grammar, so every call parses and names an offered tool.

Model ids choose the engine. `locals/...` is the Locals engine. `mlx/...` is stock mlx-lm with llguidance, for comparison, and is listed only when the server allows that engine: `locals serve --engines locals,mlx`. The default is `locals` only. Ids ending in `-thinking` exist only for models whose template can think.

## 05: tool calls without a server

```bash
python examples/05_tool_calls.py [model]
```

1. `GrammarCompiler.compile_tool_calls(TOOLS)` builds the grammar of a whole reply:
   - free text that cannot start a call
   - then calls whose tags are literal and whose function names are the offered tools

   The tags are forced text, so they cost no forward pass.
2. The prompt is rendered with `tools=TOOLS` through the chat template and decoded with `rt.generate(prompt_ids, grammar=grammar, max_tokens=200)`.
3. `locals.oai.parse_reply(res.text, TOOLS)` returns `(reasoning, content, calls)` in OpenAI form, with argument values coerced to the types each tool declares.

`compile_tool_calls` writes Qwen's native `<tool_call>` format, the `qwen_xml` format that Qwen3.5 models use. Run this example with a Qwen3.5 model. Other families write other formats (Hermes JSON, Llama JSON, Mistral, Gemma 4 and more), and the server picks the right one from the chat template.

`compile_tool_calls(..., typed=True)` also fixes parameter names and value types in the grammar. That automaton is large (about a thousand states, per the example's docstring). Pair it with the walk kernel (`locals serve --walk --typed-tools`) rather than token tables.

## `person.json`

This is the schema from the README's quick start: name, age, email, a status enum and a list of tags. Use it with the command-line generator:

```bash
locals generate --prompt "Ana Silva, 29, ana@example.com, application approved, tags: vip, early" \
    --schema examples/person.json
```

`locals generate` also takes these options:

| Option | Default | Effect |
|---|---|---|
| `--regex` | none | Constrain to a regular expression instead of a schema |
| `--max-tokens` | `512` | Token limit |
| `--temp` | `0.0` | Sampling temperature |
| `--model` | the model used last, else this Mac's default | Which model to use (without `--local`, the background server is started or switched to it) |

## The server from your own code

Start Locals with `locals start`; it answers at `http://127.0.0.1:8321` in the same
language as the OpenAI and Ollama services. These snippets use the official
`openai` Python package; any language with an OpenAI library works the same way.
The model name `locals` means the model the server runs. A full name such as
`locals/qwen3.5-4b` names one model, and the server refuses it with a message if
it runs another.

**A simple question.**

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8321/v1", api_key="local")
reply = client.chat.completions.create(
    model="locals",
    messages=[{"role": "user", "content": "Write a haiku about the sea."}],
)
print(reply.choices[0].message.content)
```

**Words as they are written**, instead of waiting for the whole answer:

```python
stream = client.chat.completions.create(
    model="locals",
    messages=[{"role": "user", "content": "Explain what a Python list is in two sentences."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

**Answers in a fixed format.** Give a JSON schema, and the answer always follows it:

```python
import json

schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"}},
    "required": ["name", "age", "city"],
}
reply = client.chat.completions.create(
    model="locals",
    messages=[{"role": "user", "content": "Maria Gonzalez is 34 and lives in Madrid."}],
    response_format={"type": "json_schema", "json_schema": {"name": "person", "schema": schema}},
)
print(json.loads(reply.choices[0].message.content))
# {'name': 'Maria Gonzalez', 'age': 34, 'city': 'Madrid'}
```

**Letting the model use your functions.** Describe a function, and the model
answers with a call to it, which your program then carries out:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "The current weather in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]
reply = client.chat.completions.create(
    model="locals",
    messages=[{"role": "user", "content": "Do I need an umbrella in Paris today?"}],
    tools=tools,
)
call = reply.choices[0].message.tool_calls[0]
print(call.function.name, call.function.arguments)  # get_weather {"city": "Paris"}
```

**Many requests at once.** This is where Locals is fastest. Send your requests
together, from threads or with an async client, and Locals answers them as one
batch. Here eight answers came back in 1.5 seconds:

```python
from concurrent.futures import ThreadPoolExecutor

def ask(question):
    reply = client.chat.completions.create(
        model="locals",
        messages=[{"role": "user", "content": question}],
    )
    return reply.choices[0].message.content

questions = [f"Write a one-line Python function that returns the square of {n}." for n in range(8)]
with ThreadPoolExecutor(max_workers=8) as pool:
    answers = list(pool.map(ask, questions))
```

**With the Ollama library**, or from any app that speaks Ollama's language:

```python
import ollama

reply = ollama.Client(host="http://127.0.0.1:8321").chat(
    model="qwen3.5-4b", messages=[{"role": "user", "content": "Say hello in French."}]
)
print(reply["message"]["content"])
```

**From the Terminal**, with no programming at all:

```bash
curl http://127.0.0.1:8321/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "locals", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Libraries built on OpenAI's interface, such as LangChain, LlamaIndex, pydantic-ai
and the OpenAI Agents SDK, work the same way: give them the address above and the
model name.
