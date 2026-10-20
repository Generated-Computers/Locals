# Changelog

## 0.2.0 (unreleased)

First public release.

- **About three times faster with many requests at once.** GPU kernels written
  and tuned for the few requests at a time that agents send, float16 on M1 and
  M2, fused layers, and a prompt cache that keeps each agent's conversation:
  271 tokens/s with eight requests at once where mlx-lm gives 87, the same speed
  for one request, and the same answers. What each part contributes is measured
  in docs/why-faster.
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
- **Safe defaults.** The server answers only apps on your Mac, and agents'
  commands can write only inside their own copy of the project.
- **Long conversations.** A request can be as long as fits in your Mac's
  memory next to the model, instead of a fixed 64k tokens: about 178,000 tokens
  for Qwen3.5-4B on a 16 GB Mac (`--max-context` still sets it). The next turn
  continues from where the previous reply began, so a follow-up question about
  a 139,000-token document starts in 2.4 seconds; it used to read the whole
  conversation again (eight minutes), and agents re-read their last tool result
  on every turn. Four multi-turn conversations at once now take 83% of their
  prompts from memory instead of 28%, and finish 36 turns in four minutes
  instead of 23.
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
  at once ran 10-13% faster on a 4B model and 19-20% on a 9B.
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
  longer leaves a second engine loaded.
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
  back and no JSON.
