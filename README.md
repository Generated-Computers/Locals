<p align="center">
  <img src="https://raw.githubusercontent.com/Generated-Computers/Locals/8404f6b4b32b8070facd0290d157e3ed97777623/docs/assets/logo.png#gh-light-mode-only" alt="Generated Computers" width="80%">
  <img src="https://raw.githubusercontent.com/Generated-Computers/Locals/8404f6b4b32b8070facd0290d157e3ed97777623/docs/assets/logo-dark.png#gh-dark-mode-only" alt="Generated Computers" width="80%">
</p>

<h1 align="center">Locals</h1>

<p align="center">
  <b>Run many AI agents at once on your Mac, up to 3x faster.</b><br>
  A local server made for agents rather than plain chat: tool calls, long
  conversations and many agents working at the same time. For OpenCode,
  OpenHands, Cline, Aider, Claude Code and any OpenAI, Anthropic or Ollama app.
</p>

<p align="center">
  <a href="https://pypi.org/project/locals/"><img src="https://img.shields.io/pypi/v/locals.svg" alt="PyPI"></a>
  <a href="https://github.com/Generated-Computers/Locals/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="#getting-started"><img src="https://img.shields.io/badge/macOS-Apple%20silicon-lightgrey.svg" alt="macOS on Apple silicon"></a>
</p>

<p align="center">
  <a href="#getting-started"><b>Getting started</b></a> ·
  <a href="#works-with-your-agents"><b>Agents</b></a> ·
  <a href="https://github.com/Generated-Computers/Locals/blob/main/docs/benchmarks/README.md"><b>Benchmarks</b></a> ·
  <a href="#documentation"><b>Documentation</b></a> ·
  <a href="https://arxiv.org/abs/2610.09139"><b>Paper</b></a>
</p>

```bash
curl -LsSf https://raw.githubusercontent.com/Generated-Computers/Locals/main/install.sh | sh
```

Or, if you use [uv](https://docs.astral.sh/uv/), install it from [PyPI](https://pypi.org/project/locals/):

```bash
uv tool install --python 3.12 locals
```

Then, in a new Terminal window:

```bash
locals code ~/my-project
```

That is all. The first time, Locals downloads a model that suits your Mac
(about 3 GB), starts it in the background and opens the OpenCode assistant in
your project. Nothing leaves your computer.

## What it is

<img src="https://raw.githubusercontent.com/Generated-Computers/Locals/8404f6b4b32b8070facd0290d157e3ed97777623/docs/assets/emblem.png" alt="Generated Computers Co." width="35%" align="right">

Locals runs AI models on your own Mac for agents, not for chatting with a raw
model. An agent works differently from a person in a chat. It calls tools, and
every call has to be right. It sends a long prompt that grows a little every
turn. And it rarely works alone: coding assistants run helpers in parallel, and
teams of agents split one task between them. Locals is built for this work, from
the way it computes a batch of requests on your Mac's graphics chip to the way
it writes a tool call, so a team of agents gets its answers up to three times
faster than with the standard way of running models on a Mac.

It works with the coding agents you may already use, such as OpenCode, Cline,
Aider, OpenHands and Claude Code, and with any app or program that talks to
OpenAI, Anthropic or Ollama. Everything stays on your computer.

Locals is built on the paper
[Breaking the Space Barrier and its Application to Language Model Inference](https://arxiv.org/abs/2610.09139)
(Arip Asadulaev, 2026). A JSON format or a tool call can be described as a
graph, and checking a reply against it as finding a path in that graph. The
paper does this in less memory than Savitch's classical bound and turns it into
an inference engine: text the format already decides is written without running
the model, the checks run on the graphics chip, and a format needs about 3 MB of
memory instead of up to 1.5 GB. With it, sixteen tool-calling agents finish 2.5
times sooner than with mlx-lm and llguidance.

## Made for agents

- **Many agents at once.** Requests from every agent are computed together, with
  GPU kernels written for a few requests at a time, so eight agents working at
  once finish in about the time two would need one after the other.
- **Tool calls that always parse.** Replies are written in each model's own
  tool-call format under a grammar applied on the GPU, so an agent can call only
  the tools it offered, every argument arrives in the type its schema asks for,
  and a reply cut off by its length limit still closes its call. Requests for
  JSON follow their schema the same way.
- **Each turn picks up where the last one ended.** Locals keeps every
  conversation between turns, and the long system prompt an agent tool starts
  with (3,000 to 16,000 tokens) on disk across restarts, so an agent's next turn
  reads only what it added: the next turn of a 138,000-token conversation starts
  after 2.4 seconds.
- **Guards made for agents.** A small model can repeat itself or call the same
  tool forever; Locals notices and stops both. A memory guard keeps many agents
  with long prompts from ending in an out-of-memory error, and an agent that
  hangs up frees the server at once.
- **One command per agent tool.** `locals code` opens OpenCode in your project,
  and `locals launch` connects Hermes Agent, OpenClaw, Claude Code, Kilo Code,
  Cline, OpenHands and Aider.
- **A team of agents for bigger tasks.** `locals run` splits a task into parts,
  gives each part to several agents at once, each in its own copy of your
  project, and lets your tests decide.

## Results at a glance

<img align="right" width="48%" alt="Qwen3.5-4B on an M2 Pro with 16 GB. mlx-lm: 61, 82, 87, 87, 157 tokens per second with 1, 2, 4, 8 and 16 requests at once; Locals: 64, 116, 160, 271, 341 (1.0x, 1.4x, 1.8x, 3.1x, 2.2x)." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/requests-narrow.svg#gh-light-mode-only">
<img align="right" width="48%" alt="Qwen3.5-4B on an M2 Pro with 16 GB. mlx-lm: 61, 82, 87, 87, 157 tokens per second with 1, 2, 4, 8 and 16 requests at once; Locals: 64, 116, 160, 271, 341 (1.0x, 1.4x, 1.8x, 3.1x, 2.2x)." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/requests-narrow-dark.svg#gh-dark-mode-only">

mlx-lm, the standard way of running models on a Mac, answers several requests
together, but it gets no faster from four of them to eight: it writes 87 tokens
per second for four requests and the same 87 for eight. Locals keeps getting
faster as requests are added, up to three times mlx-lm's speed with eight at
once.

It is faster for a single agent too: when the reply repeats a file it is
rewriting, after a restart, and in long conversations. The table has these
cases, on a MacBook Pro with an M2 Pro and 16 GB and Qwen3.5-4B, the default
model.

<br clear="right">

| On a MacBook Pro with an M2 Pro and 16 GB, Qwen3.5-4B | Without Locals | With Locals | |
|---|---|---|---|
| 8 requests at once | 87 tokens/s (mlx-lm) | 271 tokens/s | 3.1x |
| 8 agents asking at once: time to the first word | 42 s (Ollama) | 1.6 s | 26x sooner |
| One agent rewriting a 100-line file | 43 tokens/s (mlx-lm) | 128 tokens/s | 3.0x |
| An agent's first request after a restart | 16 s (mlx-lm) | 1.7 s | 9.6x sooner |
| A follow-up question about a 48,000-token document | out of memory (mlx-lm) | 0.6 s | |
| 16 coding agents calling tools (Qwen3.5-2B, from the paper) | 75 s (mlx-lm) | 29 s | 2.5x |
| 40 programming problems from LiveCodeBench, 8 at a time (from the paper) | 20 min (mlx-lm) | 8 min | 2.5x |

<img alt="Qwen3.5-4B through the server, 1 to 16 requests at once. M2 Pro · 16 GB: mlx-lm [51, 79, 88, 75, 124], Locals [65, 116, 150, 244, 310]. M2 Max · 96 GB: mlx-lm [91, 121, 142, 152, 267], Locals [104, 177, 244, 403, 530]. M5 Pro · 24 GB: mlx-lm [86, 151, 232, 258, 333], Locals [90, 161, 253, 382, 508]." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/macs.svg#gh-light-mode-only" width="100%">
<img alt="Qwen3.5-4B through the server, 1 to 16 requests at once. M2 Pro · 16 GB: mlx-lm [51, 79, 88, 75, 124], Locals [65, 116, 150, 244, 310]. M2 Max · 96 GB: mlx-lm [91, 121, 142, 152, 267], Locals [104, 177, 244, 403, 530]. M5 Pro · 24 GB: mlx-lm [86, 151, 232, 258, 333], Locals [90, 161, 253, 382, 508]." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/macs-dark.svg#gh-dark-mode-only" width="100%">

Bigger models gain more on the M2 Max: 3.0x for Qwen3.5-9B and 3.1x for a 27B
model with eight requests at once. On the M5 Pro the gain is smaller, because
mlx-lm is much faster there: MLX has its own kernels for the M5's neural
accelerators, and these numbers were measured before Locals had its own. M1, M3
and M4 chips have not been measured yet: results from them are welcome as
[issues](https://github.com/Generated-Computers/Locals/issues), with the output
of `locals bench` and `locals tune --report`.

## Benchmarks

Every table, and how each was measured, is in
[Benchmarks](https://github.com/Generated-Computers/Locals/blob/main/docs/benchmarks/README.md). Unless a chart says otherwise, it was
measured on a MacBook Pro with an M2 Pro (19-core GPU) and 16 GB, with
Qwen3.5-4B (`mlx-community/Qwen3.5-4B-MLX-4bit`), MLX 0.32 and mlx-lm 0.32.

<img alt="M2 Pro · 16 GB: Qwen3.5-2B 3.1x (188 to 576 tokens/s), MiniCPM5-2B 3.0x (160 to 478 tokens/s), LFM2.5-2.6B 2.8x (144 to 404 tokens/s), Qwen3.5-4B 3.1x (87 to 271 tokens/s), Gemma 4 E4B 2.8x (80 to 224 tokens/s). M2 Max · 96 GB: Qwen3.5-4B 2.66x (152 to 403 tokens/s), Qwen3.5-9B 3.03x (72 to 220 tokens/s), Qwen3.8-27B 3.13x (25 to 78 tokens/s). M5 Pro · 24 GB: Qwen3.5-4B 1.48x (258 to 382 tokens/s), Qwen3.5-9B 1.58x (145 to 229 tokens/s)." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/models.svg#gh-light-mode-only" width="100%">
<img alt="M2 Pro · 16 GB: Qwen3.5-2B 3.1x (188 to 576 tokens/s), MiniCPM5-2B 3.0x (160 to 478 tokens/s), LFM2.5-2.6B 2.8x (144 to 404 tokens/s), Qwen3.5-4B 3.1x (87 to 271 tokens/s), Gemma 4 E4B 2.8x (80 to 224 tokens/s). M2 Max · 96 GB: Qwen3.5-4B 2.66x (152 to 403 tokens/s), Qwen3.5-9B 3.03x (72 to 220 tokens/s), Qwen3.8-27B 3.13x (25 to 78 tokens/s). M5 Pro · 24 GB: Qwen3.5-4B 1.48x (258 to 382 tokens/s), Qwen3.5-9B 1.58x (145 to 229 tokens/s)." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/models-dark.svg#gh-dark-mode-only" width="100%">

Models quantized to 4 bits in the usual way (in groups of 64) gain 2.3x to 3.3x
with 8 requests at once; other formats run at mlx-lm's speed
([below](#where-locals-is-not-faster)).

### Against other servers

<img alt="Qwen3.5-4B, 1,000-token prompts, 8 requests at once on an M2 Pro. Time to the first token: Locals 1.6 s, llama.cpp 2.1 s, Magnitude 16.1 s, Ollama 41.8 s. Tokens per second: Locals 110, llama.cpp 71, Magnitude 42, Ollama 34." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/servers.svg#gh-light-mode-only" width="100%">
<img alt="Qwen3.5-4B, 1,000-token prompts, 8 requests at once on an M2 Pro. Time to the first token: Locals 1.6 s, llama.cpp 2.1 s, Magnitude 16.1 s, Ollama 41.8 s. Tokens per second: Locals 110, llama.cpp 71, Magnitude 42, Ollama 34." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/servers-dark.svg#gh-dark-mode-only" width="100%">

Ollama 0.40.5, Magnitude 0.2.9 and llama.cpp `b11724`, one at a time on the GPU,
each with its own 4-bit build of the model, measured with SemiAnalysis's
InferenceX client (temperature 0, replies of natural length). The tables for
1,000- and 8,000-token prompts and for Gemma 4 are in
[Benchmarks](https://github.com/Generated-Computers/Locals/blob/main/docs/benchmarks/README.md#3-against-other-servers).

### Long prompts

**No pause while a long prompt is read.** When an agent sends an 8,000-token
prompt while others are getting answers, Locals reads it in the same passes as
their next tokens, so their replies keep streaming: the longest gap in them went
from 12 s to 3 s, with the same throughput, against reading the prompt on its
own. A 3,000-token request that arrives during a 30,000-token one gets its first
token after 7 s instead of 41.

### Where Locals is not faster

- **One request alone, with nothing to reuse**: about the same as mlx-lm (64
  against 61 tokens per second on the M2 Pro). It is faster alone when the reply
  repeats the prompt (file rewrites) or the prompt is cached.
- **Prompt-bound work**: reading a prompt is limited by the GPU's arithmetic.
  With 8,000-token prompts and 8 requests at once the server writes 57 tokens
  per second; most of the time goes into reading.
- **Formats outside the tuned kernels**: 4-bit affine quantization in groups of
  64 is covered; mxfp4 (a mixture-of-experts model, Mellum) ran at mlx-lm's
  speed.
- **The M5's neural accelerators**: MLX's own kernels already use them, so the
  gain there is 1.5x, not 3x (before Locals' own `nax` kernel, which is
  experimental).

## Works with your agents

<p align="center">
<img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/hermes.png" width="12.4%" alt="Hermes Agent" title="Hermes Agent"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/kilo.png" width="12.4%" alt="Kilo Code" title="Kilo Code"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/opencode.png" width="12.4%" alt="OpenCode" title="OpenCode"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/openhands.png" width="12.4%" alt="OpenHands" title="OpenHands"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/cline.png" width="12.4%" alt="Cline" title="Cline"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/aider.png" width="12.4%" alt="Aider" title="Aider"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/roo.png" width="12.4%" alt="Roo Code" title="Roo Code"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/openclaw.png" width="12.4%" alt="OpenClaw" title="OpenClaw">
</p>

Locals works with the coding agents and assistants people already use, on a
model running on your Mac. All of these but Roo Code were tested end to end with
Qwen3.5-4B on a 16 GB Mac (Cline and Kilo Code through their command-line
versions), on real tasks such as writing a program, running it and reporting
what it printed. One command starts Locals and connects the tool to it:

| Tool | Run |
|---|---|
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | `locals launch hermes` |
| [Kilo Code](https://github.com/Kilo-Org/kilocode) | `locals launch kilo` |
| [OpenCode](https://github.com/anomalyco/opencode) | `locals code ~/my-project` |
| [OpenHands](https://github.com/OpenHands/OpenHands) | `locals launch openhands` |
| [Cline](https://github.com/cline/cline) | `locals launch cline` |
| [Aider](https://github.com/Aider-AI/aider) | `locals launch aider` |
| [Roo Code](https://github.com/RooCodeInc/Roo-Code) | in VS Code, the settings from `locals launch editor` ([how](#use-it-in-apps-and-editors)) |
| [OpenClaw](https://github.com/openclaw/openclaw) | `locals launch openclaw` |
| [Claude Code](https://github.com/anthropics/claude-code) | `locals launch claude` (through Anthropic's Messages API, which Locals also answers) |

`locals launch list` shows every tool it can open. Locals keeps running in the
background after you close a tool: `locals status` shows what it is doing and
`locals stop` stops it.

<sub>The logos belong to their projects and are shown only to say that Locals
works with these tools.</sub>

## Getting started

You need a Mac with Apple silicon (M1 or newer) and macOS 14 or newer. A Mac
with 16 GB of memory gives the best results; with 8 GB, use one of the smaller
models.

Open the Terminal app and run:

```bash
curl -LsSf https://raw.githubusercontent.com/Generated-Computers/Locals/main/install.sh | sh
```

Then open a new Terminal window (the installer adds Locals to your shell's
PATH, which only new windows read) and start a coding assistant on one of your
projects:

```bash
locals code ~/my-project
```

After the first start, which downloads the model, Locals starts in seconds.

- **OpenCode** uses Locals as its default model from then on, so plain
  `opencode` uses it too; `locals disconnect opencode` gives OpenCode its own
  settings back.
- **To update** Locals, run the install command again, or `uv tool upgrade locals`.
- **To remove** it, run `locals stop`, `locals disconnect opencode` and
  `uv tool uninstall locals`.
- **With the uv package manager**, you can install from PyPI with
  `uv tool install --python 3.12 locals` instead.

### Coming from Ollama, LM Studio, mlx-lm or llama.cpp

Locals installs next to them and changes nothing of theirs: it has its own port
(8321) and its own folders. Only the memory is shared, so quit the other one
while Locals runs; two models at once rarely fit. Locals remembers the model and
port you start it with for later commands.

**Ollama.** Quit Ollama, then let Locals answer on Ollama's port. Apps set up
for Ollama (Open WebUI, Raycast, Continue) and the `ollama` command itself keep
working unchanged, whatever model name they ask for:

```bash
osascript -e 'quit app "Ollama"'      # or: brew services stop ollama
locals launch --model qwen3.5:4b ollama
```

Ollama's model names work in Locals, and for a model Locals does not know it
names the MLX version on Hugging Face to use instead. Ollama's own downloads
are in its own format (GGUF for most models), so a model is downloaded once
more, in MLX's. To
go back, run `locals stop` and start Ollama.

**LM Studio.** LM Studio's MLX models run in Locals where they are, with no
second download. Stop LM Studio's server and start Locals on its port, so apps
that use it keep working:

```bash
locals start --model ~/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit --port 1234
```

**mlx-lm.** The same models and the same download folder, so nothing downloads
again. `locals serve` takes the place of `mlx_lm.server`, with the same API:

```bash
locals serve --model mlx-community/Qwen3.5-4B-MLX-4bit --port 8080
```

**llama.cpp.** Locals runs MLX models, not GGUF files; give it one and it names
the MLX version of the same model. Start it on `llama-server`'s port to keep
your clients:

```bash
locals start --model qwen --port 8080
```

**Any other server with the OpenAI API** (vLLM, LocalAI): point your client at
`http://127.0.0.1:8321/v1`, or start Locals on the port the client already uses
with `--port`.

## Examples

**Work on a project with an assistant.** This opens OpenCode in your project, where
you can ask for changes in plain words and watch it read, edit and test your code:

```bash
locals code ~/my-project
```

**Give it one task and get the result.** The assistant does the task from start
to finish without opening a window, and prints what it did:

```bash
locals code ~/my-project --run "add a --verbose flag to cli.py and run the tests"
```

**Let a team of agents do a bigger task.** Several agents work on the parts of
the task at the same time, each in its own copy of your project, and your tests
decide when it is done. Your project is changed only when the tests pass:

```bash
locals run "write wordcount.py with a function count_words(text) and tests in test_wordcount.py" \
    --dir ~/my-project --check "python3 -m pytest -q"
```

**Ask a question about your code.** An agent reads the project and prints the
answer:

```bash
locals run "Which function parses dates, and which format does it expect?" --dir ~/my-project
```

**Chat with the model** in the Terminal, like a private chatbot:

```bash
locals chat
```

**Turn text into structured data.** Describe the fields you want in a JSON schema,
for example in a file called `person.json`:

```json
{"type": "object",
 "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "email": {"type": "string"},
                "status": {"enum": ["pending", "approved", "rejected"]}, "tags": {"type": "array", "items": {"type": "string"}}},
 "required": ["name", "age", "email", "status", "tags"]}
```

Then the answer always follows it, field by field:

```bash
locals generate --prompt "Ana Silva, 29, ana@example.com, application approved, tags: vip, early" \
    --schema person.json
```

```json
{"name":"Ana Silva","age":29,"email":"ana@example.com","status":"approved","tags":["vip","early"]}
```

## Use it in apps and editors

Start Locals once with the model you want; it keeps running in the background:

```bash
locals start --model qwen
```

**The OpenCode app.** Run `locals setup --model qwen` once, then choose Locals in
OpenCode's model list. This works in the desktop app, the editor extension and
the Terminal version alike.

**Kilo Code, Cline and Roo Code in VS Code**, and **Continue, Zed and JetBrains.**
Choose the provider called "OpenAI Compatible" and enter these values. Run
`locals launch editor` to print them for the model that is running.

| Setting | What to enter |
|---|---|
| Base URL | `http://127.0.0.1:8321/v1` |
| API key | any text, for example `local` |
| Model | the model's name from the table below, for example `locals/qwen3.5-4b` |
| Context window | the number `locals launch editor` prints, if the extension asks for it |

**Apps that work with Ollama**, such as Open WebUI or Raycast, need no settings:
run `locals launch ollama`.

## Models

Locals picks a model that fits your Mac's memory, so most people never need to
change it. To use another one, start Locals with its name, for example
`locals start --model lfm`. Locals runs one model at a time. The last two
columns are speeds in tokens per second (tps) with 8 requests at once, on the
M2 Pro with 16 GB above.

| Model | Start it with | Name in your editor | Size | mlx-lm | Locals |
|---|---|---|---|---|---|
| Qwen3.5-4B (the default) | `--model qwen` | `locals/qwen3.5-4b` | 2.8 GB | 87 tps | 271 tps |
| LFM2.5-2.6B | `--model lfm` | `locals/lfm2.5-2.6b` | 1.5 GB | 144 tps | 404 tps |
| MiniCPM5-2B | `--model minicpm` | `locals/minicpm5-2b` | 1.3 GB | 160 tps | 478 tps |
| Gemma 4 E4B | `--model gemma` | `locals/gemma-4-e4b-it` | 4.8 GB | 80 tps | 224 tps |

On a Mac with less than 12 GB of memory, the default is Qwen3.5-2B instead
(`--model qwen3.5-2b`; `locals/qwen3.5-2b` in your editor).

On a Mac with 24 GB or more, Recursive's Flint (40B, 2.5B active per token) runs
with `--model flint` (`locals/flint` in your editor; 17.8 GB). On an M4 Pro with
24 GB: 117 tps for one request and 305 with 8 at once, where mlx-lm gives 85 and
202. A 24 GB Mac gives the GPU 16 GB unless told otherwise; `locals start` says
how to raise that, until the next restart. For long work, `LOCALS_KV_BITS=8`
keeps Flint's keys and values in 8 bits: requests up to 67,584 tokens instead of
about 35,800, writing 4 to 13% slower.

Each model also has a thinking mode, in which it reasons before it answers: add
`-thinking` to the name, for example `locals/lfm2.5-2.6b-thinking`. The smaller
models are good choices for a Mac with 8 GB of memory, and other models made for
Apple silicon on [Hugging Face](https://huggingface.co/mlx-community) work too.

Conversations can be long: up to about 178,000 tokens with the default model
on a 16 GB Mac, a few hundred pages of text. On the M2 Pro, reading a
170,000-token document takes about 12 minutes the first time, and follow-up
questions about it start in 2 seconds. OpenCode is set up for 32,000 tokens,
and OpenClaw, Hermes Agent and Kilo Code for 64,000, which keeps each of their
steps quick.

## Using it from your own code

Locals understands the same requests as the OpenAI and Ollama services, so
programs written for them work with it unchanged. Start Locals with
`locals start`, then point your program at `http://127.0.0.1:8321/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8321/v1", api_key="local")
reply = client.chat.completions.create(
    model="locals",  # the model the server runs
    messages=[{"role": "user", "content": "Write a haiku about the sea."}],
)
print(reply.choices[0].message.content)
```

Send many requests at once, from threads or with an async client, and Locals
answers them together; that is where it is fastest. Streaming, answers in a
fixed JSON format, function calls, the Ollama library and plain `curl` all work
as well; the [examples](https://github.com/Generated-Computers/Locals/blob/main/examples/README.md) show each of them.

## Privacy and safety

Everything runs on your Mac. Locals only accepts requests from apps on your own
computer, so a website you visit cannot use it. When a team of agents works on a
task (`locals run`, `locals solve`), the commands they run can change files only
in their own copy of your project, and in the temporary and cache folders that
tools write to. `locals code` runs OpenCode in your project itself, as OpenCode
always runs.

## If something does not work

Run `locals status`, or open http://127.0.0.1:8321 in your browser, to see whether
Locals is running and what it is doing. On a chip other than the M2 Pro and
M2 Max, whose tuning ships with Locals, it tunes itself to your chip the first
time, in the background while it is idle: 10 to 20 minutes of idle time in all.
You can use it during that time, and it gets faster once it finishes.

Questions and problems are welcome as
[issues on GitHub](https://github.com/Generated-Computers/Locals/issues).

## Documentation

| Topic | Page |
|---|---|
| How Locals works: the server, the batch engine, the prompt cache, the team of agents behind `locals run`, and every setting | [How Locals works](https://github.com/Generated-Computers/Locals/blob/main/docs/README.md) |
| Every benchmark, and how it was measured | [Benchmarks](https://github.com/Generated-Computers/Locals/blob/main/docs/benchmarks/README.md) |
| Every table with its date, commit and command; the M2 Max | [Measurements](https://github.com/Generated-Computers/Locals/blob/main/docs/measurements/README.md), [M2 Max](https://github.com/Generated-Computers/Locals/blob/main/docs/measurements/m2-max.md) |
| Why Locals is faster than mlx-lm, part by part | [Why Locals is faster](https://github.com/Generated-Computers/Locals/blob/main/docs/why-faster/README.md) |
| Streaming, JSON, function calls, the Ollama library and `curl` | [Examples](https://github.com/Generated-Computers/Locals/blob/main/examples/README.md) |
| The tests, and the end-to-end suite with real agent tools | [Tests](https://github.com/Generated-Computers/Locals/blob/main/tests/README.md), [End to end](https://github.com/Generated-Computers/Locals/blob/main/tests/e2e/README.md) |
| Working on Locals itself | [CONTRIBUTING.md](https://github.com/Generated-Computers/Locals/blob/main/CONTRIBUTING.md) |

To work on Locals:

```bash
git clone https://github.com/Generated-Computers/Locals
cd Locals
uv sync --extra dev
uv run pytest tests -q -m "not model"
```

## Citation

If you use Locals in your research, please cite the paper:

```bibtex
@article{asadulaev2026breaking,
  title   = {Breaking the Space Barrier and its Application to Language Model Inference},
  author  = {Asadulaev, Arip},
  journal = {arXiv preprint arXiv:2610.09139},
  year    = {2026}
}
```

## License

Locals is released under the Apache License 2.0. It is made by
[Generated Computers](https://github.com/Generated-Computers).

## Star history

<a href="https://github.com/Generated-Computers/Locals/stargazers">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Generated-Computers/Locals/charts/stars-dark.svg">
    <img alt="Locals' GitHub stars from its first day" src="https://raw.githubusercontent.com/Generated-Computers/Locals/charts/stars.svg" width="480">
  </picture>
</a>
