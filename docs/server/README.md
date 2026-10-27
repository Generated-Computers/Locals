# The HTTP server and its APIs

Part of the [technical documentation](../README.md) of Locals. File and function names refer to the `locals` package.

The server is made of four files:

- `server.py` is an OpenAI-compatible HTTP server built on the standard library's `ThreadingHTTPServer`. It runs one thread per connection and uses no web framework. It accepts up to 128 waiting connections (`_HTTPServer.request_queue_size`): with the standard library's 5, macOS 27 reset connections when 8 or 16 requests arrived at once.
- `apis.py` translates Ollama's API to Chat Completions and back.
- `oai.py` converts OpenAI messages into what the model's chat template expects. It also builds the tool-call grammar and parses replies back into OpenAI form.
- `service.py` holds `BatchService`, the client API to one engine. A single worker thread owns the model and runs continuous batching. Every HTTP request becomes one row of a shared batch, so concurrent agents decode together.

## Files

| File | Main pieces |
|---|---|
| `server.py` | `Server`: prepares requests, runs the decode loop, the reply guards and accounting. `make_handler`: routing, CORS and host checks. `serve`, and `main` with the command-line flags |
| `apis.py` | `ollama_to_chat`, `ollama_to_text`, `chat_to_ollama`, `text_to_ollama`, `OllamaStream`, `ollama_model_entry` |
| `oai.py` | `convert_messages`, `tool_specs`, `reply_regex` / `call_regex` / `typed_call_regex`, `parse_reply`, `split_reasoning`, `ReplyStream`, `coerce`, `response_schema`, `json_object_regex`, `engine_of`, `thinking_of` |
| `service.py` | `BatchService`, `Job`, and the two engine loops `_run_locals` and `_run_stock` |

## How a request flows

```
Handler.do_POST
  -> _allowed()                        Host and Origin checks (403)
  -> body <= MAX_BODY, a JSON object
  -> Server.prepare(req)               chat: template, token ids, grammar, limits
     Server.prepare_text(req)          /v1/completions: raw prompt
  -> Server.complete / Server.stream -> Server._run -> Server._decode
     Server.text_completion
  -> BatchService.submit(...)          returns a Job; the engine's worker thread decodes it
  <- on_token(ids) into a queue.Queue  called from the worker thread
  <- SSE chunks, or one JSON object
```

1. **Checks.** `do_POST` refuses the following:
   - requests that fail `_allowed()`: 403;
   - bodies larger than `MAX_BODY` (`64 << 20` bytes), or chunks whose size is not hex digits (`-1` read to the end of the connection): 413;
   - a body that has not all arrived `BODY_TIMEOUT` (60 s) after it began, in all, not per read: 408; a `Transfer-Encoding` other than chunked: 501;
   - bodies that are not a JSON object, or nested past what Python's recursion takes: 400;
   - any answer given before the body was read closes the connection (`Connection: close`): the unread body was read as the next request.

   After the body, each write to the client may block for `SEND_TIMEOUT` (120 s) at most, and an idle keep-alive connection waits as long for its next request: a client that stops reading has its job cancelled (with no limit, the thread was held and the job decoded to its end). An error nobody caught is a 500 when no answer has begun (`_guarded`), and the server raises its open-file limit to 10240 at start (launchd's default of 256 reset new clients past it).
   - chat requests whose `messages` is not a non-empty list: 400.
2. **Preparation.** `Server.prepare` (chat) or `Server.prepare_text` (text completions) runs the request checks that can be made before decoding, all before any byte of the response is sent. A regex or schema that does not compile is found only later, in the worker (`invalid: ...`; see How the constraint is chosen) (`service.BatchService._run_locals.source`; `server.Server._decode`, `server.make_handler.Handler.do_POST`). The handler maps exceptions to statuses:
   - `ContextTooLong`: 400 with code `context_length_exceeded`;
   - `ValueError`, `TypeError`, `KeyError`, `AttributeError`: 400;
   - anything else: 500, with the traceback in the log.

   The result is a plan dict `p` that holds the engine, the service, the prompt token ids, the cache checkpoints, the rendered text, the tools, the schema and regex, `max_tokens`, the tool-call format, the stop sequences, the temperature, `ignore_eos` and `include_usage`.
3. **Submission.** `_decode` calls `BatchService.submit(prompt, schema=, regex=, max_tokens=, on_token=q.put, temp=, ignore_eos=, checkpoints=)`. That call queues a `Job` and returns at once.
4. **Decoding.** The worker thread calls `on_token` with each group of committed tokens. `_decode` then:
   - drains the queue, waiting up to 0.25 s at a time;
   - decodes the whole token list to text;
   - runs the reply guards and the stop sequences;
   - feeds the text to `oai.ReplyStream`, which yields `reasoning` and `content` deltas.

   Text that ends in an incomplete character (`�`) waits for more tokens.
5. **Finishing.** When the job is done, `_run` takes these steps in order:
   - parses the reply with `oai.parse_reply`;
   - applies the repeated-call steering;
   - records the exact token stream with `_remember`;
   - sends the tool calls, the finish chunk and the usage, then `data: [DONE]`.

### What `Server.prepare` does, in order

1. With `LOCALS_DEBUG` set, it saves the request (see Security).
2. `_unsupported` refuses with a 400 the options the server would otherwise ignore while the client relied on them:
   - `n` other than 1;
   - a truthy `logprobs` or `top_logprobs`. `logprobs: false` and `top_logprobs: 0` pass (`server._unsupported`).
3. It picks the engine with `oai.engine_of(model)` and thinking with `oai.thinking_of(model, --thinking)`.
   - A request for another model by the server's own names (`locals/<name>`, `mlx/<name>`, `stock/<name>`; an alias of the loaded model passes) is a 400 that says which model runs (`check_model`), and that the plain `locals` is always the model loaded: another locals command had switched the model under a connected agent, and the agent's requests were answered by the new model without a word. Any other name (`gpt-4o` in an app, Ollama's names) is the model loaded.
   - A bool in `chat_template_kwargs.enable_thinking` (the switch vLLM and SGLang use) overrides thinking.
   - `reasoning_effort: "none"`, or `reasoning.effort` in the Responses API shape, turns thinking off.
4. `Server.service(engine)` loads the engine on first use. Each engine has its own lock, so the other engine keeps serving during a load. After a failed load, the engine is not loaded again for 60 s. Requests in that window get a 500 that repeats the error.
5. `_cap_context()` fixes the context limit once (see Context limits).
6. `format_of(engine)` reads two things from the chat template, once per engine: the tool-call format (`formats.detect`) and the thinking style (`formats.thinking_style`).
7. Thinking is forced off in four cases:
   - The style is `"none"` (no switch) or `"switch"` (thoughts marked in a way the parser does not know).
   - The output is constrained (a `response_format` of type `json_object`, `json_schema` or `regex`, or `guided_regex`). The grammar writes the JSON from the first token, and an open think block would hold it.
   - `formats.offers_thinking` says the model has no thinking mode. Such a model never closes a think block.
   - An Anthropic request ends with an assistant prefill (`_prefill`): the reply goes on from it.
8. `oai.convert_messages` flattens the messages:
   - `system` and `developer` messages are merged into one at the front;
   - content parts are joined with newlines, with an image as "[an image was attached here; this model reads text only]" and a file as "[a file was attached here; this model reads only the text of the conversation]" (`oai.IMAGE_NOTE`, `oai.FILE_NOTE`);
   - tool-call arguments become dicts;
   - an empty user message is added when there is none, because templates require one.

   Tool-call turns the server wrote itself are replayed verbatim (see Exact token streams).
9. `oai.tool_specs` normalises `tools`. When tools are offered and the first message is a system message, `_cwd_hint` reads the path from its `Working directory: <path>` line and appends a new last paragraph: "The working directory is <cwd>. Every file path you pass to a tool must start with <cwd>/ (for example <cwd>/main.py); run commands there with: cd <cwd> && <command>." It adds nothing if the prompt already contains that sentence (`server.py`). Small models invent absolute paths, and the end of the system prompt is where they pay attention.
10. `fmt.prepare(msgs, tools)` lets the tool-call format adapt the messages to its template. The base `ToolFormat` returns them unchanged.
11. It builds the template kwargs:
    - `enable_thinking`, unless the style is `"none"`;
    - the client's `chat_template_kwargs`, filtered by `_template_switches`;
    - with thinking on, `reasoning_effort`. When the template lists the values it accepts, `formats.effort_for` maps the request's value to the nearest one. A template that lists only `xhigh`, `medium` and `low` raises on anything else.
12. `_render` calls `tokenizer.apply_chat_template(..., tokenize=False)`. A Jinja error raised by the template, such as a wrong role order or a missing field, becomes a 400.
13. With the `"always"` style and thinking off, it appends `fmt.think_close`, so the model answers at once. LFM2.5's template opens a think block whatever the switch says.
14. `_tokens_for` turns the text into token ids. Where the text continues a stored stream, it reuses that stream's ids.
15. It checks the context, sets `max_tokens`, picks the constraint, and, on our engine, computes the cache checkpoints.

## Endpoints

Paths are matched with any trailing slash removed.

| Method | Path | Answer |
|---|---|---|
| POST | `/v1/chat/completions`, `/chat/completions` | OpenAI Chat Completions, streamed or not |
| POST | `/v1/completions`, `/completions` | OpenAI text completions on a raw prompt, with no chat template |
| GET | `/v1/models`, `/models` | `{"object": "list", "data": Server.models()}` |
| GET | `/health` | `{"status": "ok", "engines": {name: loaded}, "version": __version__, "build": BUILD_ID}` |
| GET | `/stats`, `/v1/stats` | live counters from `Server.stats()` |
| GET | `/` | a status page (`STATUS_HTML`) that polls `/stats` every second. On the `--ollama-port` listener it returns the text `Ollama is running` instead |
| GET | `/api/version`, `/api/tags`, `/api/ps` | Ollama |
| POST | `/api/chat`, `/api/generate`, `/api/show`, `/api/pull`, `/api/embed`, `/api/embeddings` | Ollama (see below) |
| OPTIONS | any | 204 with CORS headers, after `_allowed()` |
| HEAD | any | 200 with an empty body. No host or origin check |

`GET /v1/models/<id>` (and `/models/<id>`) answers the loaded model's entry. `POST /v1/messages` and `/v1/messages/count_tokens` are the Anthropic API (below). Any other path gets a 404.

`BUILD_ID` comes from `defaults.build_id()`. It is a fingerprint of the names, sizes and modification times (`st_mtime_ns`) of the installed `locals/*.py` files, except `plan.py` and `solve.py`, which are the client side of `locals solve` (`defaults.build_id`). The CLI compares it with its own, so it can restart a background server that still runs older code.

`/stats` returns:

- `model`, `uptime_s`, `requests`, `errors`;
- `active`: one entry per request in flight, with its engine, prompt tokens, tokens so far and start time;
- `memory`: `gpu_active_mb`, `gpu_peak_mb`, `gpu_cache_mb`, `rss_mb`;
- per engine: `loaded`, and the server's counters (`requests`, `prompt_tokens`, `cached_tokens`, `completion_tokens`, `forced_tokens`, `passes`, `decode_s`, `tool_calls`).

The per-engine counters are merged with `BatchService.stats()`, which adds `active`, `tokens_out`, the engine's own counters, `prompt_cache_mb`, `prompt_cache_entries` and `prompt_cache_by_kind`. Keys the two share, `prompt_tokens` and `cached_tokens`, take the service's values.

Errors use OpenAI's shape: `{"error": {"message", "type", "param", "code"}}`.

- `type` is `invalid_request_error` for statuses below 500 and `server_error` from 500 up.
- `param` is `"messages"` only when a `code` is set.

## Model ids and engines

Two engines can serve the same model, and each request's model id picks one. `ENGINE_NAMES` maps API names to `BatchService` engines:

- `locals` maps to `"locals"`, our engine;
- `mlx` maps to `"stock"`, plain mlx-lm.

| Id | Engine | Notes |
|---|---|---|
| `locals/<short>` | ours | `<short>` is `defaults.short_name(model)`: the repository or folder name in lower case, without `-mlx-4bit` or `-4bit` |
| `mlx/<short>` | stock mlx-lm | as are ids that start with `stock/` or `stock-` |
| contains `think` (as in `...-thinking`) | either | turns thinking on |
| anything else | ours | including a Hugging Face id such as `mlx-community/Qwen3.5-4B-MLX-4bit`. The server has one model; an id naming another model in our own namespace (`locals/<short>` of another model) is refused (`check_model`) |

- A request without `model` is served as `locals/<short>`.
- Asking for an engine that is not in `--engines` is a 400: `engine 'mlx' is not enabled on this server (enabled: locals)`.

`/v1/models` lists two ids for each enabled engine:

- `<engine>/<short>`;
- `<engine>/<short>-thinking`, when the model can think or before any engine has loaded.

Each entry carries `context_length` and `max_model_len`, both equal to `context_for(engine)`. Once `format_of` has run for the engine (at preload, or on its first chat request; an engine loaded only by a `/v1/completions` request has no format yet), each entry also carries (`server.Server.__init__`, `server.Server.models`, `server.Server.prepare`, `server.Server.prepare_text`):

- `tool_call_format`: the format's name, such as `qwen_xml`;
- `parallel_tool_calls`: whether the format writes several calls in one reply.

## Chat request options

| Field | Effect |
|---|---|
| `model` | picks the engine and thinking (see above) |
| `messages` | required, a non-empty list |
| `tools` | OpenAI function tools. A bare `{name, description, parameters}` also works |
| `tool_choice` | `auto` (the default), `required`, `none`, or `{"type": "function", "function": {"name": ...}}` (`{"name": ...}` also works). On our engine, whenever tools are offered (with or without a `response_format`), `_check_tool_choice` turns any other value, or a name that is not one of the tools, into a 400. The stock engine does not check it (`server.Server.prepare`, `server._check_tool_choice`). |
| `parallel_tool_calls` | `false`: at most one call per reply (our engine's grammar) |
| `response_format` | type `text`, `json_object`, `json_schema` (schema in `json_schema.schema`) or `regex` (in `regex` or `pattern`). Any other type is a 400, so a reply is never unconstrained by mistake |
| `guided_regex` | vLLM's name for a regex constraint |
| `reasoning_effort`, `reasoning.effort` | `none` turns thinking off. Other values reach the template as `reasoning_effort` only when thinking is on, unless the client sets that key in `chat_template_kwargs`. When the template lists the values it accepts, the nearest one is used (`server.Server.prepare`) |
| `chat_template_kwargs` | `enable_thinking` (a bool) switches thinking. Other keys reach the template if allowed (see Security) |
| `stop` | a string or a list. At most 16 non-empty strings, applied only when the request offers no tools |
| `temperature` | clamped to 0 to 2. If absent, null or a bool, the server's `--temp` applies. Anything `float()` cannot convert is a 400; a numeric string such as "0.7" is accepted (`server._temperature`) |
| `max_completion_tokens`, else `max_tokens` | Zero or negative (such as `-1`) means the server's `--max-tokens`. Always capped by `--max-tokens` and by the room left in the context |
| `ignore_eos` | runs to `max_tokens` whatever the model ends with, as vLLM's option does for benchmarks' fixed output lengths. It also turns off the reply guards, and the tool grammar unless a call is required |
| `stream` | server-sent events |
| `stream_options.include_usage` | sends usage in a separate last chunk with `choices: []` |
| `n` | must be 1. Otherwise a 400 that says to send the request n times; those requests run together in one batch |
| `logprobs`, `top_logprobs` | 400 |

A request's own `top_p` (0 to 1; 0 and 1 mean no filter), `top_k` (a whole number; 0 means no filter) and `seed` (a whole number) apply to it (`_sampling`; another value is a 400); those it leaves out are the server's `--top-p` and `--top-k`, and no seed. With a seed, the same request alone gives the same reply (docs/engine, "Sampling for each row"). (Before, the server ignored all three.)

### How the constraint is chosen

`prepare` checks these in order:

1. `response_format.type == "json_object"` uses `oai.json_object_regex()`. That is outlines-core's regex for `{"type": "object"}`, with whitespace `JSON_OBJECT_WS` = `( |\n *)?`: a space, or a newline and an indent. In compact form the model, which writes `{` and a newline by habit, was pushed into keys such as `"/a"`.
2. `response_format.type == "regex"`, or `guided_regex`, uses that regex. It must be a non-empty string.
3. `json_schema` passes the schema to the engine, which compiles it.
4. On our engine, with no schema and tools offered, it uses `oai.reply_regex(tools, tool_choice, thinking, typed=--typed-tools, fmt=fmt, preamble=False, parallel=parallel_tool_calls is not False)`.
5. On our engine, with tools and a `response_format` (or `guided_regex`) at once, as the OpenAI Agents SDK sends for an agent with an output type, the reply is the calls or the output, as OpenAI's is: the alternation of `reply_regex(tools, "required", ...)` and the format's regex (`grammar.schema_regex` for a schema). `tool_choice: required` or a named function gives the calls only, `none` the output only. With the format's grammar alone, the model could not call: asked for the weather in Paris, it wrote a forecast as JSON.

This order has three consequences:

- On the stock engine, a `response_format` wins over tools.
- The stock engine never gets a tool grammar. With tools, its replies are free text, parsed afterwards.
- A regex or schema that does not compile ends the job with `invalid: ...`, which the server raises as `BadRequest`. A non-streamed request gets a 400. A streamed request gets an error event, because the grammar is compiled in the worker after the stream's headers have been sent.

**The tool grammar.** The reply is optional free text that cannot open a call, followed by calls as `tool_choice` requires:

- `auto`: zero or more calls;
- `required`: one or more;
- a named function: exactly that one;
- `none`: free text only. The tools stay in the prompt, so its cache still matches the turns that offered them.

Properties of the grammar:

- Every tag is a literal, so the engine forces it.
- Tool names are drawn from the offered tools.
- A reply cut by the token budget still closes its call.
- `preamble=False`: a reply that must call starts with the call, because a small model given room to talk first can talk until the token limit.
- With thinking on, the grammar starts inside the open `<think>` block.
- Formats other than `qwen_xml` build their own grammar with `fmt.reply_regex`.

By default, parameter names and values are free, and values are converted to the schema's types after parsing (`oai.coerce`). This keeps the automaton at a few hundred states. `--typed-tools` instead fixes parameter names, their order and simple value types (`typed_call_regex`). Required parameters are mandatory and the others optional. It applies to the `qwen_xml` format and to Flint's (`flint_xml`, whose `reply_regex` takes the schemas, `FlintXML.typed_call`); other formats build their grammar with `fmt.reply_regex` and ignore it (`oai.typed_call_regex`, `oai.reply_regex`). Use it with `--walk`, which handles grammars of any size.

## Text completions

`prepare_text` accepts `prompt` in three forms:

- a string, encoded without a chat template;
- a list with one string;
- a list of token ids. Every id must be inside the vocabulary, because an id past it reads past the embedding.

Other rules:

- When the template uses `bos_token` and encoding left it out (Gemma 4, Ministral), a BOS token is put first.
- `echo` and `suffix` are refused.
- `max_tokens` defaults to 256.
- There is no thinking, no tools and no `response_format`.
- `stop`, `temperature`, `ignore_eos` and `stream_options.include_usage` work as they do for chat.

Replies have `object: "text_completion"` and `choices[0].text`. Their `usage` has no `locals` block.

## Streaming

Streams are `text/event-stream` with `Transfer-Encoding: chunked`; the handler speaks HTTP/1.1. An HTTP/1.0 client, which has no chunked encoding, gets the stream as it is and `Connection: close`: the stream ends when the connection closes (`Handler._chunked`, `_stream_start`, `_part`, `_stream_end`). Each event is `data: {chunk}\n\n`, and the last one is `data: [DONE]`.

- **Role.** `"role": "assistant"` goes with the first delta, after the prefill, as vLLM and SGLang do. When the role was sent before the prefill, clients that time the first chunk measured no time to first token at all.
- **Reasoning.** With thinking on, `oai.ReplyStream` starts in reasoning mode. It sends `delta.reasoning_content` until the think close tag, then `delta.content`.
  - It also handles models that open the block themselves, such as Qwen3, and Gemma 4's `<|channel>thought ... <channel|>`.
  - A partial tag at the end of the text is held back until it is decided.
  - Non-streamed replies carry `message.reasoning_content`.
- **Content and tool calls.** Content is streamed up to the format's call opener. In Qwen3.5's format, tool calls are streamed while the model writes them (`oai.CallStream`): a call's first delta has its `index`, `id`, `type` and name once the name is written, and the next ones add to `function.arguments` as the JSON grows: a string value as it is written (less an end that may still be its close), another type once it is complete. The pieces add up to exactly the arguments the finished reply's parse gives (tested a character at a time), so a client that joins them gets the call it would have got whole. A long `write` call had shown nothing for up to 27 s. A call that may still turn out to repeat the previous turn's call, or one before it in the same reply, is held until it differs (the guards steer away from or drop those), and later calls wait with it; calls past `max_calls` and calls to tools not offered are not sent. After the reply ends, the rest of each call sent in part is sent, and calls not sent yet go whole (other formats: every call, as before). When a call was already sent and a later one repeats the previous turn's, the reply keeps the calls up to that one instead of being steered. Text that was held back because it looked like the start of a call, and turned out not to be one, is sent at that point (`ReplyStream.unsent`).
- **Keepalive.** After 15 s with nothing written, such as during a long prefill or while a tool call is being written, the stream writes `KEEPALIVE`, a bare newline. SSE readers skip it.
  - It replaced a `: keepalive` comment. Benchmark clients that JSON-parse every non-empty line (InferenceX's, vLLM's `benchmark_serving`) failed on that comment.
  - If the client has left, the write fails and the job is cancelled.
- **Finish chunk.** An empty delta with `finish_reason`:
  - `tool_calls`;
  - `length`, when the engine's finish was `length` or `exhausted`, or when the loop guard cut a call (`loop_in_call`) or a reply without tools;
  - `stop` otherwise.

  Without `include_usage`, this chunk also carries `usage`. With it, usage goes in a separate last chunk with `choices: []`, which is where clients such as vLLM's and InferenceX's benchmarks read it.
- **Usage fields.** `prompt_tokens`, `completion_tokens`, `total_tokens` and `prompt_tokens_details.cached_tokens` (prompt tokens served from the prompt cache). An extension object `locals` adds `engine`, `forced_tokens`, `passes`, `constrained` and `wall_s`.
- **Errors after the stream has begun.** Once the headers are out, an error is sent as an event `data: {"error": {"message", "type": "server_error", ...}}` (`invalid_request_error`, with the plain message, for the request's own fault, such as a grammar that does not compile). Then come `data: [DONE]` and a clean end of the chunked body, not a second HTTP response.
- **Engine errors.** The engine can fail before a reply has written any token, for example with a Metal out-of-memory error, which ends every request in flight. In that case `_decode` decodes the reply once more, since the engine has dropped its cache and serves again.
- **Disconnects.** Every request, streamed or not, is checked about once a second with `_alive_socket`, a non-blocking `poll` and peek at the socket (`select` refuses file descriptors above 1023). A write that fails mid-stream (the agent's Esc or Ctrl-C) ends the same way: `_decode` cancels the job it submitted whenever its loop exits by an exception, so the batch row stops at the next commit instead of decoding on to `max_tokens` unread. A stream that writes nothing for a while (a long prompt being read, a tool call held back) is found by the probe, not by the next keepalive 15 s later. In the engine, a prompt read synchronously (nothing else decoding) stops between chunks once every request it reads has been cancelled (`prefills_given_up` in the stats), so the next request does not wait for a prompt nobody reads.

## Reply guards

`Server._guard` runs on each token group, except with `ignore_eos`. It returns a reason to stop early, and the job is cancelled at once. The server also stops at stop sequences.

| Reason | When | Then |
|---|---|---|
| `max_calls` | the reply holds `--max-calls` complete calls (default 6) | the reply is cut after that many (`fmt.cut_after`) |
| `duplicate_call` | two calls in the reply have the same name and arguments | duplicates are dropped, and the text is cut to match |
| `loop` | the token stream ends in a repeated block (`_looping`). With tools offered and no call opener in the reply yet, also when the last 30 lines of at least 12 characters say at most 6 different things (`_line_looping`). Once an opener has appeared, the raised in-call thresholds apply for the rest of the reply (`server.Server._guard`, `server._line_looping`) | the reply ends there |
| `stop` | a stop sequence appeared | the text is cut at it |

The model can repeat itself forever, and the grammar cannot see it. `_looping(toks, reps=3, max_period=256, min_span=24, short_span=128)` looks for one block of up to 256 tokens repeated `reps` times at the end, over at least `min_span` tokens.

- A block shorter than 8 tokens must repeat over `short_span` tokens, because data repeats short runs on purpose.
- The thresholds rise where repetition can be legitimate:
  - inside a call's arguments, such as a file of identical lines: `reps=16, min_span=256, short_span=256`, and the reply is cut only after `LOOP_GRACE` (1024) more tokens of it (`_past_grace`): the reason is `loop_in_call`, the call is dropped and the finish is `length`;
  - in a reply without tools, such as "write it ten times": `reps=12, min_span=96, short_span=256`, with the same 1024-token grace, and the finish is `length`.
- With tools, the call checks and `_line_looping` see only the text after the reasoning, so drafts in thoughts are not counted as calls. The token-loop check `_looping` still runs on every token, reasoning included (`server.Server._guard`, `server.Server._decode`).

**Repeated calls across turns.** `_previous_calls` walks back from the end over tool results to the last assistant turn and collects its calls with their results. If a user message comes first, it returns nothing and steering does not apply (`server.py`). If the new reply repeats one of them (same name and arguments), `_run` decodes the turn again:

- It adds a steering prefix written for the model: "The X call above was already made and returned: ... Repeating it would give the same result, so I will do something different now: ", with the result shortened to 160 characters. The client receives it as `reasoning_content` when the prompt opened a think block (`_in_think`), otherwise as `content` (`server.Server._run`, `server.Server._steer_text`).
- It caps `max_tokens` at `STEERED_MAX_TOKENS` (2048). Gemma 4 wrote 4,096 tokens of prose after the steering text.
- It removes a leading thought block from the grammar (Gemma 4's `thought_regex`).

Steering is skipped for `llama3_json`, `mistral` and `mistral_legacy`, whose calls are the whole reply and cannot be preceded by text. For these formats the first repeat already ends the turn with the "I am stuck" message and returns no calls, unless `tool_choice` is `required` or a named function (`server.Server._run`).

If the steered reply repeats too, it is decoded up to twice more without the tools whose call repeated (for this reply only), unless `tool_choice` names a function. When no other tool is left, or the attempts run out, the turn ends with a message that asks the user how to proceed, and no calls are returned. The exception is a `tool_choice` of `required` or a named function: then the repeated call is returned as asked. With tools and a response format (`tool_choice` `auto`), a reply that repeats a call is decoded once more as the format's output instead (`p["output_regex"]`).

Two smaller rules:

- With `tool_choice: "none"`, a call written anyway (only the stock engine can) is returned as text.
- Without tools, a reply is never parsed for calls.

## Exact token streams and raw replies

On the next turn, the client sends the conversation back and the chat template renders it again. Encoding that text does not always give back the tokens the model produced, and then the prompt cache misses. `Server` keeps two stores, both guarded by `Server._lock`, to fix this.

**`_streams[engine]`** holds the last 64 finished replies. Each entry is `{"text": rendered prompt + raw reply, "ids": prompt ids + steering prefix ids + reply tokens}`.

- `_tokens_for` finds the longest stored text that is a prefix of the new prompt text. It reuses that entry's ids and encodes only the rest.
- Only uncut replies are stored (`stopped is None`), because only an uncut reply is a token stream the model can continue.

**`_raw`** maps a tool-call id to the raw reply text the call was parsed from. It keeps at most 4096 entries and drops the oldest 1024 once past that.

- `oai.convert_messages(raw=...)` renders an assistant turn whose first call id is known as that text, verbatim. The prompt then encodes to the tokens the model produced.
- Ids are compared with `oai.id_key`, which keeps letters and digits only, because some clients (OpenClaw) strip the rest.
- A thinking reply is split back into `reasoning_content` and `content`.
- Only formats with `replay_raw = True` use `_raw`. That is the `ToolFormat` default. Mistral v3 and v11 and Gemma 4 set it to `False`.
- What is stored is the reply as the client received it: cut after `max_calls`, without duplicate calls.

**Cache checkpoints.** Some models have a cache that cannot be cut back to a shorter prefix, such as Qwen3.5's recurrent layers or Gemma's sliding windows. `_needs_checkpoints` detects them by checking for any cache layer that is not a plain `KVCache`. For these models, our engine also stores the prompt's cache at up to four positions computed by `_checkpoints`:

- `"system"`: where this session's own part of the system message starts (`_session_start`: what it shares with recent sessions' system messages comes before), and its end too when that is more than 4096 characters further;
- `"user"`: the start of the last message;
- `"reply"`: where this reply starts once the next request renders it. Set `LOCALS_REPLY_CHECKPOINT=0` to turn it off.

Each position is found with the model's own template. `"system"` is where two renderings part: the system message followed by two different dummy user messages (this needs at least two messages, the first a system one). `"user"` is the rendering of every message but the last. `"reply"` is where two renderings part: the whole conversation followed by two different dummy assistant replies and a follow-up user message (`server.Server._checkpoints`). A position is used only if its text is a prefix of the prompt text, at least 2048 characters long, and matches at least 1024 of the prompt's tokens.

A plain KV cache does not need checkpoints. The prompt cache cuts it back to any shared prefix, so snapshots would only store the prompt's KV twice more.

## Context limits

Each request has a limit on prompt plus reply tokens, `context_for(engine)`.

**Our engine** uses `max_context`:

- With `--max-context`, that value.
- Otherwise it is sized once, when our engine loads: `max(65536, BatchService.context_fit())`.
- Where less than 65536 fits (a large model with full attention on a small Mac), clients are still told 65536 (Hermes refuses a model with less), but a request is held to the fit (`limit_for`, `_fit_limit`): past it, it gets `context_length_exceeded`. Admitted alone, nothing else checks a request's memory, and a Metal out-of-memory error fails every request in flight.
- In both cases it is then capped at the model's own context, `native_context(model_id)`, which reads `max_position_embeddings` from `config.json` or from its `text_config`.
- Qwen3.5-4B on a 16 GB Mac gets 178,176 tokens.

**The stock engine** uses `--max-context`, or 65536 without it, capped at the model's own context and at `max_context`. It is never sized from our engine's memory model, because the stock engine has no memory guard.

`BatchService.context_fit()` computes how much KV cache fits under the engine's memory limit, next to the weights and a prefill's activations: `(memory_limit - weights_bytes - (3 << 29)) // kv_bytes_per_token`, rounded down to a multiple of 1024. `3 << 29` is 1.5 GB.

`memory_limit` is Metal's `max_recommended_working_set_size` minus a margin (`service.memory_plan`): MLX's pool of freed buffers (1 GiB, or an eighth of the room a model leaves under 8 GiB, at least 256 MiB) and 0.5 GiB for what the admission estimate misses, 1.5 GiB for a 4B on 16 GB; `LOCALS_MEMORY_MARGIN_GB` sets it.

Until our engine loads (for example with `--preload none`), the limit is 65536, or the model's own context if that is smaller. `_cap_context` runs on every request and fixes the final value after the first load.

A chat request raises `ContextTooLong` in two cases:

- the prompt has `limit` tokens or more;
- the room left (`limit - prompt`) is less than `min(max_tokens, 256)`. A reply of a few tokens would only be cut, so the client is told to make room instead.

The answer is OpenAI's error, so agents compact their history and retry:

```json
{"error": {"message": "This model's maximum context length is N tokens. However, your messages resulted in M tokens. ...",
           "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded"}}
```

Otherwise `max_tokens = max(1, min(asked or --max-tokens, --max-tokens, room))`. A text completion fails only when the prompt alone reaches the limit.

## Anthropic's Messages API

Claude Code and the Anthropic SDKs speak Anthropic's Messages API, so the server answers `POST /v1/messages` and
`POST /v1/messages/count_tokens` as well (`locals launch claude` points Claude Code at it). `apis.anthropic_to_chat`
turns a request into a Chat Completions request, which goes through the same `prepare`, batch, prompt cache, grammar,
guards and hang-up handling:

| Messages API | Chat Completions |
|---|---|
| `system` (a string or text blocks) | one system message (blocks joined by a blank line) |
| `text`, `image`, `document` blocks | text; an image as `oai.IMAGE_NOTE`; a text document as its text, another as `oai.FILE_NOTE` |
| assistant `thinking`, `text`, `tool_use` blocks | `reasoning_content`, `content`, `tool_calls` (ids kept, so raw replay finds the reply) |
| user `tool_result` blocks | `tool` messages, in order, before the user's text; `is_error` as "Error: " before the result |
| `tools` (`name`, `description`, `input_schema`) | functions; Anthropic's own server tools (web search, code execution) are left out |
| `tool_choice` `auto` / `any` / `tool` / `none`, `disable_parallel_tool_use` | `auto` / `required` / that function / `none`, `parallel_tool_calls: false` |
| `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences` | the same |
| `thinking` `enabled` or `adaptive` / `disabled` | thinking on / off (`chat_template_kwargs.enable_thinking`) |

Any model name is served by the server's model (`chat_model`), and the reply carries the name asked for; our own names for another model are refused (`check_model`, as above; a 404 on Ollama's API).
`apis.AnthropicStream` reads the server's SSE output and writes the Messages API's events: `message_start` at once,
a content block per kind (`thinking` with `thinking_delta` and an empty `signature_delta`, `text` with `text_delta`,
each tool call as `tool_use` with `input_json_delta`), `ping` for the server's keepalives, `message_delta` (stop
reason `end_turn`, `tool_use` or `max_tokens`; usage with `cache_read_input_tokens` from the prompt cache) and
`message_stop`. Errors have Anthropic's shape (`{"type": "error", "error": {"type": ..., "message": ...}}`); a
prompt longer than the context is "prompt is too long: N tokens > M maximum", the words Claude Code compacts on.
`count_tokens` answers the prompt's length without generating. `GET /v1/models` carries Anthropic's fields
(`type`, `display_name`, `created_at`, `has_more`) beside OpenAI's. A request body may be gzip- or
deflate-compressed (`Content-Encoding`), as Claude Code can send it.

Claude Code asks for thinking and adds a user text (a `<total_tokens>` note) after every tool result, so in the
next request each earlier reply is written without its thoughts: the reply checkpoint, at the reply's start, is
where every request continues (16,363 of 16,934 tokens from the cache in a measured session).

## The Ollama API

Many local tools default to Ollama, so the server answers Ollama's API too:

- always, under `/api/...` on its own port;
- with `--ollama-port`, also on Ollama's own port. `locals serve --ollama` uses 11434. On that port, `GET /` answers `Ollama is running`.

Requests are translated to Chat Completions and go through the same `prepare`, batch, grammar and guards. `apis.OllamaStream` reads the server's SSE output and writes newline-delimited JSON (`application/x-ndjson`).

| Endpoint | Behaviour |
|---|---|
| `GET /api/version` | `{"version": "0.12.6"}` (`OLLAMA_VERSION`, which apps check) |
| `GET /api/tags` | one entry per enabled engine: `<short>:latest` for ours, `mlx/<short>:latest` for the stock engine |
| `GET /api/ps` | the same, for loaded engines only |
| `POST /api/show` | details, `capabilities` (`completion`, `tools`, and `thinking` when the loaded model offers it) and `model_info.<family>.context_length` |
| `POST /api/chat` | chat |
| `POST /api/generate` | `system` and `prompt` as a chat. With `raw`, the prompt goes untemplated through `/v1/completions`. `suffix` (fill-in-the-middle) is refused |
| `POST /api/pull` | `{"status": "success"}`: there is nothing to pull |
| `POST /api/embed`, `/api/embeddings` | 501, not served |

How requests are translated (`apis.ollama_to_chat`):

- **Model names.** Any other name maps to the server's one model (`Server.chat_model`); our own names for another model (`locals/<short>`, `mlx/<short>`, `stock/<short>`) are refused (`check_model`, a 404 on Ollama's API). `mlx/...` picks the stock engine when it is enabled. A name that contains `think` turns thinking on.
- **Load and unload.** A chat without messages, or a generate without a prompt, is Ollama's way to load a model. With `keep_alive` 0 it unloads one. Either is answered at once, with `done_reason` `load` or `unload`.
- **Options.** `options.num_predict` (when > 0) becomes `max_tokens`. `temperature`, `top_p`, `top_k`, `stop` and `seed` are copied (for `/api/generate` with `raw` too), and all of them apply.
- **Formats.** `format: "json"` becomes `json_object`. A schema object becomes `json_schema`.
- **Thinking.** `think: true` or `false` becomes `chat_template_kwargs.enable_thinking`. A level string (`low`, `medium`, `high`) turns thinking on and sets `reasoning_effort`.
- **Images.** The images are dropped. A message, or a generate prompt, that had any gets one note appended to its text: `IMAGE_NOTE`, "[an image was attached here; this model reads text only]" (`apis.ollama_to_chat`).
- **Tool call ids.** Ollama's tool calls have no ids, so they get stable ones (`call_<message>_<index>`), the same on every request. Random ids changed the history the template writes, so the prompt cache missed. Tool results without a `tool_call_id` are paired with the last assistant turn's calls by position.

Replies have:

- `message.thinking` for the reasoning;
- tool-call arguments as objects;
- `done_reason` set to `length` or `stop`;
- `prompt_eval_count`, `eval_count`, and durations in nanoseconds.

In a stream, tool calls are collected and sent in the final `done` line. Errors are `{"error": "..."}`.

## Security

The server has no authentication. It is meant to listen on loopback, which is the default (`--host 127.0.0.1`). The checks below defend against web pages that try to use it from the user's browser.

- **Host check** (`_host_ok`, against DNS rebinding). When the server is bound to a loopback address, the `Host` header must name this machine. Accepted names:
  - `localhost`, `127.0.0.1`, `::1`, `0.0.0.0`;
  - names ending in `.localhost`, `.local` or `.internal` (Docker's `host.docker.internal`, for Open WebUI in a container);
  - this machine's host name;
  - any loopback or private IP address;
  - entries of `LOCALS_HOSTS` (comma-separated; `*` allows any).

  Anything else gets 403 `this server answers to localhost only`. A request without a `Host` header passes. When the server is bound to any address other than `localhost`, `127.0.0.1` or `::1`, this check is off (`server._host_ok`).
- **Origin check** (`_origin_ok`). A request that carries an `Origin` header must come from one of these:
  - a page served from this machine: `http` or `https` with host `localhost`, `127.0.0.1` or `::1` (`_LOOPBACK`), or a name ending in `.localhost`, on any port. Other loopback or private addresses are not accepted as origins (`server._origin_ok`);
  - a desktop app or extension scheme: `app`, `file`, `tauri`, `vscode-webview`, `vscode-file`, `chrome-extension`, `moz-extension`, `safari-web-extension`;
  - an origin listed in `LOCALS_ORIGINS` (comma-separated; `*` allows any).

  Anything else gets a 403. Without this check, any site the user visits could use the model and read its replies, since a cross-site request needs no preflight. Requests without an `Origin` header, such as agents and curl, pass this check. CORS headers are sent only to allowed origins: `Access-Control-Allow-Origin` echoing the origin, `Vary: Origin`, any headers, and the methods `GET, POST, OPTIONS`.
- **Template switches** (`_template_switches`). `chat_template_kwargs` reach the template only as plain values: a bool, a number, or a string of at most 200 characters. Keys in `_TEMPLATE_DENY` are always dropped; they include `chat_template`, `tools`, `messages`, `add_generation_prompt`, `tokenize` and `enable_thinking`. A template sent by a client would run inside the server, where a loop in it holds a thread and the GIL, as in vLLM's CVE for the same switch.
- **Limits.**
  - Request bodies are at most `64 << 20` bytes.
  - A connection that sends nothing for 600 s is dropped (`Handler.timeout`).
  - Token ids sent to `/v1/completions` are checked against the vocabulary.
- **Debug dumps.** With `LOCALS_DEBUG` set, each request is saved as JSON in `LOCALS_DEBUG_DIR` (default `~/.cache/locals/requests`). The directory has mode 0700 and each file 0600, because prompts can hold secrets. Ollama requests are saved twice: as received, and after translation.

## BatchService

`BatchService(engine, model, slots=16, temp=0.0, seed=0, cache_gb=None, walk=False, top_p=0.0, top_k=0)` starts a daemon thread named `batch-<engine>`. That thread loads the model and runs the engine's loop.

- The constructor waits until the engine is ready, and raises the load error again if loading failed.
- `engine` is `"locals"` or `"stock"`.
- The server passes its own `--temp`, whose default is 0.2.

| Method | What it does |
|---|---|
| `submit(prompt, schema=None, max_tokens=512, regex=None, on_token=None, checkpoints=None, temp=None, ignore_eos=False)` | queues a `Job` and returns it. Safe to call from any thread. An empty prompt is a `ValueError` |
| `prime(tokens)` | prefills the tokens once and stores them in the prompt cache, for example a shared system prompt. Blocks until done |
| `context_fit()` | the number of tokens one request can have on this Mac with our engine, or None |
| `stats()` | live counters, and the prompt cache's size, its entries and the entries per kind |
| `close()` | stops the worker, waiting up to 30 s |

A `Job` holds the result:

- `tokens` and `text`;
- `finish`: `stop`, `length`, `exhausted`, `cancelled`, `primed`, `invalid: ...` or `error: ...`;
- `cached`: prompt tokens served from the cache;
- `t_submit` and `t_done`.

It has `done`, `wait(timeout)` and `cancel()`. A cancelled job ends at the engine's next commit, with `finish == "cancelled"`. `on_token` is called from the worker thread with each group of committed tokens.

How the worker runs:

- It takes every queued job each time it looks (`_take`).
- While idle, it tunes one pending kernel shape at a time (`_tune_while_idle`), so a request that arrives waits for one shape at most.
- At start, it seeds MLX's random generator with `seed` and caps MLX's buffer cache at `1 << 30` bytes. By default, MLX keeps freed buffers up to the whole memory limit.
- After loading, a fatal error fails every job in flight, and every later job, with `error: ...`.

### `"locals"`: our engine (`_run_locals`)

- **Model and batching.** It loads with `Runtime.load(model, tune=LOCALS_TUNE)` (default `background`) and runs `HybridBatchEngine(model, tokenizer, n_slots=slots)`: fused layers, small-batch kernels, and pipelined continuous batching.
- **Constraints.** A schema or regex is compiled per job.
  - By default, `rt.compile` builds token tables for a `LocalsController`. Their host part (the automaton, the token table, the budget tables: seconds for a large grammar) is made by `rt.precompile` in a thread of its own after `submit`, one grammar at a time, and the job is queued once it is done; the batch loop only copies the tables to the GPU. A grammar that fails keeps its error, so the same one fails at once next time. Made in the batch loop, a schema with `maxItems: 50` (1302 states) stopped every running reply for 8 s, 31 s on a 16 GB Mac.
  - A grammar whose token table would pass `LOCALS_MAX_TABLE_GB` is refused while its automaton is minimized, as soon as it has more states than fit (a string with `maxLength: 6000`: 7 s, from 214 s).
  - With `walk`, `rt.compile_walk` builds a `WalkController`, with no token tables and any grammar size.
  - Without `--walk`, recursive schemas (`$ref` cycles) are refused; the walk path has no such check.
  - A compile error ends only that job, with `invalid:`; the batch goes on.
  - Unconstrained jobs stop at the model's EOS ids. With `ignore_eos` they have no stop ids.
- **Memory.**
  - `memory_limit` is the recommended working set minus `memory_plan`'s margin (1.5 GiB for a 4B on 16 GB, see above).
  - The prompt cache (`chain_cache`, at most 256 entries) gets `cache_gb` when it is given. Otherwise it gets `min(max(free, longest, 1.5 GB), 16 << 30)`.
  - `free` is the working set minus the active memory after loading and `5 << 30` (5 GB) for running rows and prefill.
  - `longest` is the KV of one conversation of `context_fit()` tokens. A follow-up question on a long document then continues from it; the comment gives 1.5 s instead of 137 s at 60k tokens.
  - The `--cache-gb` help text says 4 GB, but the code subtracts `5 << 30`.
- **Disk prompt cache.** Unless `LOCALS_DISK_CACHE=0`, system prompts are kept across restarts.
  - Location: `LOCALS_DISK_CACHE_DIR` (default `~/.cache/locals/prompts`), up to `LOCALS_DISK_CACHE_GB` (default 4, shared by every model's folder; 0 turns it off). A save never fails a request: a folder deleted under a running server is made again, and a snapshot that cannot be written is skipped with a line in the log.
  - Each folder is named after the model, the dtype, and a fingerprint of the weight files and the mlx-lm version.
  - Folders left from other weights of the same model are removed.
- **Checkpoints.** `"system"` and `"user"` checkpoints are stored as copies of the cache in prompt-cache entries of their kind. A copy is skipped when `_room_for_copy` says there is no room, and `"system"` copies are also saved to the disk cache. A `"reply"` checkpoint keeps only the layers that cannot be cut back (recurrent state). When the row finishes, these are joined with the row's KV cut to that position and stored as an `"assistant"` entry. Entries are evicted in mlx-lm's balanced order between the kinds (see [Prompt cache](../prompt-cache/README.md)) (`hybrid_batch._copies`, `hybrid_batch.HybridBatchEngine._checkpoint`, `hybrid_batch.HybridBatchEngine.serve.retire`; `service.BatchService.submit`).
- **Errors.** When the engine loop raises, for example on Metal out of memory, the worker:
  - fails the jobs in flight;
  - halves the prompt cache;
  - clears MLX's buffer cache;
  - keeps serving.

### `"stock"`: plain mlx-lm (`_run_stock`)

- **Model and batching.** `mlx_lm.load`, then mlx-lm's `BatchGenerator` with `completion_batch_size=slots`, `prefill_batch_size=min(slots, 4)` and `prefill_step_size=512`. mlx-lm computes logits at every prompt position, and with a 248k vocabulary a 16 x 2048 prefill chunk needs about 16 GB for them alone.
- **Prompt cache.** `LRUPromptCache(max_size=256, max_bytes=cache_gb or 1.5 GB)`. Each job fetches its longest cached prefix and prefills only the rest.
  - Recurrent-layer cache rows are copied before they are stored. mlx-lm returns them as views of the whole batch buffer, so each cached conversation would hold every row's state (about 0.3 GB at 16 rows on Qwen3.5-2B).
- **Constraints.** llguidance logits processors (`LLGProcessor`).
  - A schema becomes a regex through `grammar.schema_regex` (outlines-core's, with integer bounds), after `json_safe_schema`.
  - llguidance never matches a special token against regex text. `_llg_grammar` therefore refers to special tokens at the regex's ends by id, in a lark grammar.
  - Grammars are cached per regex.
- **Sampling.** A request with its own temperature, `top_p`, `top_k` or `seed` gets a sampler of its own (`_job_sampler`); the settings it leaves out are the server's.
- **`ignore_eos`** removes the row's stop tokens only for an unconstrained job (`if job.ignore_eos and not procs`). A constrained job keeps its stop tokens (`service.BatchService._run_stock`).
- **Cancellation.** Cancelled jobs are removed from the generator.
- **Not supported:** checkpoints, the disk cache, and a memory guard.

## Command-line flags

`main()` is the entry point of `python -m locals.server`. `locals serve` (`app.cmd_serve`) remembers the model and port, then `os.execv`s `python -m locals.server` with the flags `app.server_cmd` builds. It accepts only a subset: `--model`, `--port`, `--engines`, `--preload`, `--slots`, `--temp`, `--top-p`, `--top-k`, `--max-tokens`, `--cache-gb`, `--max-context`, `--thinking`, `--max-calls`, `--walk`, `--typed-tools` and `--ollama`. `--host`, `--seed` and `--ollama-port` exist only on `python -m locals.server` (`app.server_cmd`, and the parsers in `app._main`). It also remembers the model and port for later commands, and adds `--ollama` as a shortcut for `--ollama-port 11434`.

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `DEFAULT_MODEL` | a short name (`qwen`, `qwen3.5-2b`, `gemma`, `lfm`, `lfm3`, `minicpm`), any mlx-lm model id, or a local path. The default is Qwen3.5-2B on a Mac with less than 12 GB of memory, and Qwen3.5-4B otherwise |
| `--host` | `127.0.0.1` | IPv6 addresses (`::1`) work |
| `--port` | `8321` | |
| `--ollama-port` | `0` (off) | also listen there as Ollama (11434). Must differ from `--port` |
| `--engines` | `locals` | engines that may be loaded, comma-separated: `locals`, `mlx` |
| `--preload` | `locals` | engines loaded at start, comma-separated, or `none` |
| `--slots` | `16` | largest batch per engine |
| `--temp` | `0.2` | default sampling temperature |
| `--top-p` | `0.0` (off) | nucleus sampling when the temperature is above 0 (Qwen3.5 thinking: 0.95) |
| `--top-k` | `0` (off) | top-k sampling when the temperature is above 0 (Qwen3.5 thinking: 20) |
| `--max-tokens` | `4096` | cap on completion tokens per request |
| `--cache-gb` | unset | prompt-cache size in GB, used by both engines. When unset, our engine sizes it from free GPU memory and the stock engine uses 1.5 GB (`max_bytes=self.cache_bytes or int(1.5 * (1 << 30))`) (`service.BatchService.__init__`, `service.BatchService._run_locals`, `service.BatchService._run_stock`) |
| `--max-context` | unset | prompt plus reply tokens per request (see Context limits) |
| `--thinking` | off | think before every reply |
| `--seed` | `0` | |
| `--max-calls` | `6` | tool calls kept per reply |
| `--walk` | off | constraints run by the walk kernel (our engine) |
| `--typed-tools` | off | tool calls with parameter names and types fixed (use with `--walk`) |

Before loading the model, `main` checks that:

- the port is free. If it is not, the error says whether a locals server or another program holds it;
- `--ollama-port`, when set, is free and differs from `--port`;
- the model is not a GGUF model and fits (`defaults.gguf_hint`, `defaults.too_big`).

Expected load errors, such as a model that is not on Hugging Face, or one that is not downloaded while offline, are reported in one line (`_load_error`).

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `LOCALS_ORIGINS` | empty | extra web origins allowed to call the server (`*`: any) |
| `LOCALS_HOSTS` | empty | extra `Host` names accepted on a loopback bind (`*`: any) |
| `LOCALS_DEBUG` | unset | save every request as JSON |
| `LOCALS_DEBUG_DIR` | `~/.cache/locals/requests` | where those files go |
| `LOCALS_REPLY_CHECKPOINT` | `1` | `0` turns off the `"reply"` cache checkpoint |
| `LOCALS_TUNE` | `background` | passed to `Runtime.load` as `tune` |
| `LOCALS_DISK_CACHE` | `1` | `0` turns off the disk prompt cache |
| `LOCALS_DISK_CACHE_DIR` | `~/.cache/locals/prompts` | where the disk prompt cache lives |
| `LOCALS_DISK_CACHE_GB` | `4` | size of the disk prompt cache |
