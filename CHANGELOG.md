# Changelog

## 0.2.2 (unreleased)

- OpenCode 2 works with Locals as OpenCode 1 does. `locals code` opened OpenCode 2's interface with `-m`, which it
  rejects ("Unrecognized flag: -m"); the model (and the agent `--agent` names) now goes in OPENCODE_CONFIG_CONTENT,
  which wins over a project's own opencode.json, and OpenCode's server runs for that launch (`--standalone`)
  instead of its shared background service. `locals swarm`, `locals solve` and `locals solve --plan` speak OpenCode
  2's server API (sessions, prompts and waits, interrupts, permission replies), and read its messages as version 1's.
  The agents are not given OpenCode 2's web search and code mode tools. Tested with OpenCode 1.18.32 and 2.1.2.
- **Recursive's Flint** (`--model flint`: mlx-community/flint-mixed-3-4bit, 17.8 GB, a 24 GB Mac and up). The model's
  code ships with Locals (mlx-lm has it only in an unreleased pull request), its tool calls have their own format
  (`<tool_call>NAME<arg_key>...</arg_key><arg_value>...</arg_value></tool_call>`; it was taken for the hermes form,
  whose grammar held Flint to JSON calls), and its reasoning, which it opens itself, comes back as reasoning. Kernels
  for its router, experts and q/k norms: on an M4 Pro with 24 GB, 117 tokens/s for one request where mlx-lm gives 85,
  319 for eight at once where it gives 202; prompts are read 7 to 12% faster than mlx-lm reads them from 4,096 tokens
  (docs/benchmarks, section 11).
- On a Mac with the memory for a model but macOS's default GPU limit below what it needs (Flint on 24 GB), the memory
  check says how to raise the limit (`sudo sysctl iogpu.wired_limit_mb=...`) instead of only naming a smaller model.
- Flint's matrix products tuned to MLX's own kernel skip Locals's routing: 4 us of Python a call, 0.8 ms of a
  one-token step.
- Decode rows and a prompt chunk in one pass for Flint too: replies streaming while an 8,000-token prompt is read
  stall for at most 2.8 s instead of 9.7.
- The projections `locals.patch` fuses away from Flint no longer stay in MLX's buffer cache (2.5 GB, which a 24 GB
  Mac swapped for).
- `LOCALS_KV_BITS=8`: Flint's full-attention keys and values in 8 bits, as its config asks, read by a kernel of
  their own on the GPU's matrix units with Flint's attention sinks; a 24 GB Mac takes 67,584-token requests instead
  of about 35,800, for 4 to 13% of the writing speed.
- `--typed-tools` types Flint's tool calls too: the grammar writes each call's parameter names in the schema's order,
  required ones present, values of their types. On 12 OpenCode-like tasks Flint broke 4 of 12 greedy calls and 11 of
  39 sampled ones without it (`path` for `filePath`, `bash` without its `description`), none with it.
- One request alone gets its forced spans in one pass over Flint's sliding-window and 8-bit caches too: those
  caches cannot be rewound, which drafts need and forced spans do not. Flint's typed calls take 10%
  fewer passes (about the same time: a pass of several tokens reads each token's experts).
- Flint's expert kernels no longer end a command buffer each: MLX ends one once its inputs pass 50 MB, counted at
  their whole size, and the kernels were handed a layer's whole expert tensors (350 MB). Handed one-expert views of
  the same buffers, a one-token step takes 8.75 ms instead of 10.2: 115 tokens/s for one request instead of 102, and
  8 to 9% more after long prompts.
- Flint's sliding-window layers read a prompt chunk in bands of 512 queries against the keys their window reaches,
  instead of the whole chunk under a mask: prompts 11% faster from 4,096 tokens (844 tokens/s instead of 760 at 4k,
  560 instead of 504 at 32k), the same logits.
- Flint reads prompts in 2,048-token chunks to about 30k tokens (a model's `prefill_chunk_tokens` and
  `prefill_scores`, set where its attention holds no chunk's scores): a 30k prompt's first token after 54 s instead
  of 58, a 60k one with `LOCALS_KV_BITS=8` after 159 s instead of 177.
- Flint's full-attention layers attend without the batch caches' mask when no row is padded (all true, and 40% of
  MLX's attention at 30k keys): after a 30k-token prompt Flint writes 63.7 tokens/s instead of 55.9, after 13k 74.5
  instead of 70.1.
- Flint's expert kernels take experts of 2, 3, 4, 5, 6 and 8 bits (mlx-lm's mixed recipes put some layers at 6), and
  its router kernel names real experts even for a row of NaN logits. A Flint request alone gets its forced spans in
  one pass once it is no longer left-padded.
- Flint's passes of 5 to 8 tokens (a typed tool call's forced spans) are padded to 9 queries in its attention:
  MLX's attention takes a path twice as slow from 5 to 8 queries (2.1 ms a full layer at 8k keys for 8, 1.2 for 9).
- Flint's expert kernels read an expert once for every row of a batch that chose it, the same outputs bit for bit:
  284 tokens/s at 8 requests at once instead of 265, 313 at 16 instead of 288 (305 instead of 281 through the server).
- Flint with several requests at once, the same tokens: from 4 rows the expert kernels, bound by their arithmetic as
  much as by the weights they read, convert each input chunk once for several weight rows and turn 3-bit values into
  floats without an int-to-float conversion; the expert choice also orders the experts' rows (no sort kernel), and
  each layer's residual adds and next norm are inside its kernels (an add and RMS norm kernel laid out as MLX's,
  the same bits). 305 tokens/s at 8 at once instead of 284, 225 at 4 instead of 212, 340 at 16 instead of 313.
- A kernel family for Flint's matmuls of up to 16 rows, `fragq`: the weights go into the GPU's matrix units as
  1 + q/16 in the activations' 16-bit type (two by one shift, mask and or), the scale and bias per group on a float
  accumulator, a quarter of the matrix product's arithmetic spent on dequantizing instead of a half. The tuner times
  Flint's layers in a chain of calls that wait on each other, as a model runs them (back to back without waiting it
  chose kernels that made Flint 2% slower), and times again its shapes timed before the family. Flint's layers have
  tuning entries of their own (`…xbfloat16flint`): every other model's layers, of the same shapes too, keep their
  kernels, timings and choices. Flint: 317 tokens/s at 8 at once instead
  of 302, 352 at 16 instead of 339, 227 at 4 instead of 224; through the server, 305 at 8 clients.
- Recursive's own 4-bit build of Flint (RecursiveSI/flint-int4, 22.6 GB) runs on a 16 GB Mac: the GPU holds its
  weights without the experts (1.2 GB) and slots for recently used experts (40% of GPU memory), and the rest are
  read from the weight files as tokens need them (locals/models/flint_streamed.py, which loads that build on any
  Mac; mlx-community's builds load whole as before). On the M2 Pro: about 6 tokens per second, a 6,000-token prompt
  read in 19 s, and OpenCode fixed a bug in a small project with it.
- Flint's decode passes run its MoE blocks in Metal kernels of their own: the layer's router and the routers that
  guess the next layers' experts in one product and one selection (0.27 ms of a layer's GPU time became 0.04),
  and the routed and shared experts in two kernels instead of about fourteen; the experts already on the GPU are
  computed while the others are read. Without reads a token takes 31 ms instead of 40.
- Where Flint's layers mostly find their experts on the GPU, a decode step asks the host only when one is missing:
  each layer looks its experts up on the GPU, and the step runs windows of layers at once, resuming after a layer
  that missed one (its later layers' KV caches restored). With every expert on the GPU, a token takes 20 ms
  instead of 31 (50 tokens per second on the M2 Pro). Steps switch to it when 90% of the layers find their
  experts (LOCALS_FLINT_OPTIMISTIC: auto, always, 0).
- Flint reads a prompt's layers into two reused buffers, the next layer while the GPU computes this one, and copies
  the experts already on the GPU instead of reading them: a 1,024-token pass after some decoding, 5.4 s instead of
  6.9. Its GPU memory holds the files' bytes in every compute dtype, converted on the GPU when used.
- On M1 and M2 Flint computes in float16: its one value past float16's range (an expert's SwiGLU product in layer
  46, about 68,000) is taken in float32 and scaled by a power of 2. It is closer to a float32 run than bfloat16
  (logit error median 0.77 against 2.0) and a 4,096-token prompt pass takes 8.0 s instead of 10.1. A model that
  keeps its values in float16's range says so (``float16_safe``) and skips the runtime's dtype probe.
- A sliding layer of Flint attends over a long pass in blocks of 256 queries, each over the keys its window reaches
  (the same softmax): 11.6 ms a layer over 4,096 tokens instead of 57 with the window as a mask over every key. A
  4,096-token prompt pass takes 6.9 s (10.9 before these changes).
- `python -m locals.models.pack MODEL` writes an expert-major copy of a streamed model's experts (21.4 GB for
  Flint, in ~/.cache/locals/experts, 34 s on the M2 Pro). Flint's experts are stacked per tensor in its files, nine
  slices of one expert in nine places; from the copy each is one read, scattered into its slot: decoding 10.7
  tokens per second instead of 7.3, a 4,096-token prompt pass 6.3 s instead of 6.7. Used when present.
- LOCALS_EXPERT_RAM_GB: experts Flint evicts from the GPU are kept in memory outside the GPU's share, purgeable
  (macOS takes it back before it would swap), and a later miss copies them back instead of reading the SSD.
- Experts are read the most urgent first, the parts of one expert at once, bypassing macOS's file cache (which
  held the same experts a second time); the slot taken is the least used one (uses counted less as they age), and
  more experts are read ahead the more surely they are guessed. At most four guessed reads run at once
  (LOCALS_EXPERT_GUESS_READS), so a read a layer waits for finds the SSD free: 139 ms a token instead of 156.
- Mixture-of-experts models decode several requests at once with each expert read and decoded once a step: the
  step's (token, expert) pairs grouped by expert, one threadgroup per used expert and block of its rows (MLX's
  affine 4-bit and mxfp4). On the M2 Pro with Mellum2.1-12B-A2.5B (64 experts, 8 a token) 16 requests decode at
  189 tokens per second together against 146 with mlx-lm's batch_generate (1.30x), 4 at 131 against 115; one
  request as before (86 against 85.5). LOCALS_MOE_KERNEL=0: MLX's kernels.
- A model can set its own prompt passes (HybridBatchEngine.prefill_tokens).

## 0.2.1 (2026-10-21)

- The installer installs the release from PyPI; run again, it updates to the newest release. `LOCALS_SOURCE` set to
  GitHub's archive installs the main branch.
- The README on PyPI: `uv tool install locals` and `uv tool upgrade locals`.
- `locals menubar` asks for `uv tool install 'locals[menubar]'` when rumps is missing.
- Tested on Python 3.14 too, which a plain `uv tool install locals` picks on a Mac without a newer Python.

## 0.2.0 (2026-10-21)

First public release.

- **Up to 3x faster with many requests at once.** GPU kernels written
  and tuned for the few requests at a time that agents send, float16 on M1 and
  M2, fused layers, and a prompt cache that keeps each agent's conversation:
  271 tokens/s with eight requests at once where mlx-lm gives 87 (an M2 Pro with
  16 GB), and no slower for one request. On an M5 Pro, where mlx-lm is much
  faster, 1.5x with eight to sixteen requests at once. What each part
  contributes is measured in docs/why-faster.
- **Many agents on one task.** `locals run` puts a team of agents on a task, each
  in its own copy of the project, with your tests as the check; the project is
  changed only when they pass.
- **One command for coding agents.** `locals code` opens OpenCode on a local
  model; `locals launch` does the same for Claude Code, OpenHands, Cline, Kilo
  Code, OpenClaw, Hermes Agent, Aider, editor extensions and apps that work with
  Ollama. `locals models` says which model to use for agents.
- **Switching from Ollama, LM Studio, mlx-lm or llama.cpp** in a command or two
  (README, "Coming from"): `locals launch ollama` answers apps set up for
  Ollama and the `ollama` command itself, Ollama's model names work, LM
  Studio's MLX models run where they are, and a GGUF file or an Ollama name
  Locals does not know gets the MLX version of the same model to use.
- **An OpenAI-, Anthropic- and Ollama-compatible server** on MLX, running in the
  background (`locals start`, `locals status`, `locals stop`) or in the menu bar.
  Anthropic's Messages API (`/v1/messages`) is what Claude Code and the Anthropic
  SDK speak; each step of a Claude Code session comes from the prompt cache.
- **Files appear as the agent writes them.** Tool calls stream while the model
  writes them, so an agent writing a file shows it line by line, instead of
  nothing for 10 to 30 seconds and then the whole file at once (Qwen3.5).
- **Tool calls and JSON that always parse**, in each model's own format: Qwen,
  Gemma 4, Llama, Mistral, LFM2.5, MiniCPM5 and Hermes-style models.
- **Models for every Mac.** A default that fits your memory, short names for
  tested models (`qwen`, `gemma`, `lfm`, `minicpm`), any MLX model from Hugging
  Face, and a clear message for models that do not fit or come in the GGUF
  format.
- **Safe defaults.** The server answers only apps on your Mac, and the
  commands of `locals run` and `locals solve` agents can write only inside their
  own copy of the project (and the temporary and cache folders).
- **Long conversations.** A request can be as long as fits in your Mac's
  memory next to the model, instead of a fixed 64k tokens: about 178,000 tokens
  for Qwen3.5-4B on a 16 GB Mac (`--max-context` still sets it). The next turn
  continues from where the previous reply began, so a follow-up question about
  a 138,000-token document starts in 2.4 seconds; it used to read the whole
  conversation again (eight minutes), and agents re-read their last tool result
  on every turn. Four multi-turn conversations at once now take 83% of their
  prompts from memory instead of 28%, and finish 36 turns in four minutes,
  against 23 before.
- **No pause while a long prompt is read.** When a new request with a long
  prompt arrives, the requests already answering keep streaming: their next
  tokens are computed in the same passes over the model as the new prompt,
  instead of waiting for all of it (12 seconds for 8,000 tokens on a 4B model).
  Short prompts are read before long ones: a new agent's first token comes in
  7 seconds instead of 41 while another agent's 30,000-token prompt is read.
- **Works with benchmark tools.** `ignore_eos` and
  `stream_options.include_usage` as vLLM and OpenAI handle them, so InferenceX,
  vLLM's and aiperf's clients measure Locals as they measure other servers.
- **Measures itself on a new Mac.** How prompts are read (MLX's 4-bit kernel or
  a dense one) is timed on each GPU instead of fixed for M2-class chips, and
  `locals tune --report` shows where Locals gains on this Mac: MLX's matrix
  kernels against the chosen ones for each number of requests at once. On
  Qwen3.5, tuning also times the whole model reading prompts with MLX's own
  recurrent-layer op (which has a version for the M5's neural accelerators)
  and keeps it where it is faster. Batches of 8 to 16 requests are computed
  at a larger, faster size where tuning finds one: on an M5 Pro, 12 requests
  at once ran 10-13% faster on a 4B model and 19-20% on a 9B. On the M5, a
  4-bit matrix kernel for its neural accelerators (`nax`) is among the kernels
  tuning chooses from: timed alone, one decode step's matrix products take
  1.6-3.4x less time than with MLX's kernels at 6 to 32 requests. It is
  experimental (whole decode steps are not measured yet); `LOCALS_NAX=0` turns
  it off.
- **Smoother commands.** `locals chat` and `locals generate` use the background
  server instead of loading a second copy of the model, so `--n` works with every
  model; `locals status`, `stop` and `--version` answer in a fraction of a second;
  `locals code --run` shows each step as it happens; `locals` alone shows what it
  can do; the installer no longer needs git.
- **A steadier team of agents.** `locals run` tells you at once when the check
  command cannot run (for example pytest is not installed), instead of failing
  after minutes; a check that already passes before the run guards the result
  instead of ending it after the first file; "fix parser.py" finds
  `src/.../parser.py`; agents that are stopped stop generating at once.
- **Fixed:** sixteen long requests arriving at once no longer run the GPU out
  of memory; a stream that waits for a long prompt no longer sends a line that
  strict clients failed to read; a second command no longer starts a second
  server while the first is still loading the model; OpenCode always runs on the
  local model, even when a project's own settings name another one; a mistyped
  model name is no longer saved as the default; a follow-up message in an agent
  session no longer reads the previous turn again (Qwen3.5 and Gemma 4); pressing
  Esc or Ctrl-C in an agent stops its reply on the server too; with MLX 0.32.4, a
  new user's first task on an M2 no longer fails with "Maximum threads per
  threadgroup"; two OpenCode agents started in the same second no longer fail
  with "database is locked"; an agent that repeats its last call is led to a
  different call (to another tool if it insists), instead of a plan in words or
  a question that ended its task; Aider no
  longer writes a new file named after a sentence of the model's reply;
  several Gemma 4 requests at once with prompts longer than 512 tokens no
  longer get empty or wrong replies; eight or more requests at once no longer
  get "Connection reset by peer" on macOS 27; a model that nearly fills the
  Mac's memory (a 27B on 24 GB) runs several requests at once instead of one at
  a time, without running out of memory with sixteen: each request's own state
  is counted, the room kept for reading a prompt is what reading one took, and
  MLX keeps fewer freed buffers when the model leaves little room;
  several long prompts joining the requests already answering no longer hold
  their memory twice for a moment (three 27k-token prompts took a 4B model on
  a 16 GB Mac past the GPU's memory limit); while a request waits for memory,
  the running ones' growth is still made room for; a long prompt stops being
  read as soon as its client leaves; on a Mac where less than 64k tokens fit
  next to a large model, a longer request gets a context error instead of
  running the GPU out of memory.
- **Fixed, settings and commands:** `locals disconnect` no longer deletes an
  OpenCode provider of your own named "mlx", nor Hermes settings for another
  local server (LM Studio, llama.cpp); OpenClaw configs with commas or colons
  inside strings are read; `locals tune` next to a running server no longer
  erases tuning results (it asks to stop the server first); a model loading in
  `locals serve` is not loaded a second time by another command; a fine-tune
  with the same name as the running model is loaded instead of ignored; a
  server restarted after an upgrade keeps its `--ollama-port`, `--max-context`
  and other options; `locals --port N chat` uses that port; `locals bench` no
  longer leaves a second engine loaded; `locals start` while `locals serve` is
  still fetching the model on the same port waits for it instead of loading
  the model a second time; an agent left running when another command
  switched the server to another model gets a message saying so, where its
  requests were answered by the other model; a kernel shape whose tuning
  stops locals twice is left on MLX's kernel instead of stopping every start.
- **Fixed, APIs:** web pages using the OpenAI SDK can call the server
  (`Authorization` passes the CORS check); request bodies sent in chunks are
  read; Anthropic `stop_sequences` report `stop_sequence`; an Anthropic
  assistant prefill is continued; a file of many identical lines written
  through a tool call is no longer cut; replies stream as they are written
  when no tools are offered (Llama and Mistral formats); errors on the
  Anthropic and Ollama APIs come in their own shapes; the stock engine keeps
  serving after an error; a tool parameter that is a nested model (`$ref`, as
  FastMCP and Pydantic write it) or an optional number arrives as an object or
  a number instead of its JSON text, and a parameter schema written as `true`
  no longer fails the request; a request with a large JSON schema no longer
  pauses every other reply while its grammar is made (31 s for one with
  `maxItems: 50`, now 0.1 s), and one too large to serve is refused in seconds
  with a clear message (it took 214 s with `maxLength: 6000`); a JSON schema
  with `allOf` (a base and an extension, as OpenAPI and zod write them) gets
  one object with all the properties, where the reply was two objects back to
  back and no JSON; an integer stays within its schema's `minimum` and
  `maximum` (a rating from 1 to 5 could come out as 7); a request with tools
  and a `response_format` at once (an OpenAI Agents SDK agent with an output
  type) can call the tools, where the model made their answers up as JSON;
  Qwen can write tool parameters named like `$select`, `größe` or `user id`;
  a JSON schema `pattern` is a search, as JSON Schema says (zod's
  `startsWith("https://")` allowed only `"https://"` itself), and a `format`
  without a regex (Pydantic's `Path`, `SecretStr`) is no longer a 400; LFM2
  calls to tools named like `resolve-library-id` (Context7) are read as calls
  instead of coming back as text.
- **Fixed, safety and steadiness:** `locals run` writes the files it changed
  together or none of them, keeps their line endings, bytes and permissions,
  and a stop during that write lets it finish and says which files changed;
  each `locals solve` attempt works in a clone of its own (an agent's
  `git stash pop` no longer reached your stash), and OpenCode and the checks
  there can write only inside the attempt's copy, also through a link out of
  the project; copies of the project go as soon as they are read, and a
  stopped `solve --plan`, `swarm` or `/n` turn leaves none behind and keeps
  what finished; a system-prompt snapshot whose folder was deleted under a
  running server no longer fails every request in flight; `locals stop`
  during kernel tuning no longer counts as a crash of that kernel; every
  malformed request gets a 4xx answer, and a client that stops reading no
  longer holds a server thread; a reply cut at `--max-tokens` is said, and
  `locals generate --schema` then exits 1; OpenClaw's and OpenCode's configs
  are found where those tools look; the installer uses Python 3.12, the
  version Locals is tested on; a request's own `top_p`, `top_k` and `seed`
  apply to it (they were ignored), so a seeded request alone gives the same
  reply; a server tuning kernels while idle waits while another Locals
  server works, instead of keeping timings taken on a busy GPU; after an MLX
  upgrade, the kernels this Mac timed are timed again; `solve --plan` no
  longer stops its last version when the others gave nothing.
