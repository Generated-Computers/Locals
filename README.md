<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/logo-dark.png">
    <img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/logo.png" alt="Generated Computers" width="80%">
  </picture>
</p>

<h1 align="center">Locals</h1>

<p align="center">
  <b>Run many AI agents at once on your Mac, about three times faster.</b><br>
  OpenCode, OpenHands, Cline, Aider and any OpenAI or Ollama app, on one local
  server built for many agents working at the same time.
</p>

<p align="center">
  <a href="https://github.com/Generated-Computers/Locals/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="#getting-started"><img src="https://img.shields.io/badge/macOS-Apple%20silicon-lightgrey.svg" alt="macOS on Apple silicon"></a>
</p>

```bash
curl -LsSf https://raw.githubusercontent.com/Generated-Computers/Locals/main/install.sh | sh
locals code ~/my-project
```

That is all. The first time, Locals downloads a model that suits your Mac
(about 3 GB), starts it in the background and opens the OpenCode assistant in
your project. Nothing leaves your computer.

## What it is

<img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/emblem.png" alt="Generated Computers Co." width="35%" align="right">

Locals runs AI models on your own Mac and makes them fast when many requests
arrive at the same time. That is how today's AI agents work: a coding assistant
runs several helpers in parallel, and a team of agents can work on one task
together. Locals answers these requests together, and computes such a group of
requests far more efficiently on your Mac's graphics chip, so the whole team gets
its answers about three times faster than with the standard way of running
models on a Mac.

It works with the coding assistants you may already use, such as OpenCode,
Cline, Aider and OpenHands, and with any app or program that talks to OpenAI or
Ollama. Everything stays on your computer.

Locals is built on the paper
[Breaking the Space Barrier and its Application to Language Model Inference](https://arxiv.org/abs/2610.09139)
(Arip Asadulaev, 2026). The paper goes below the classical memory bound of
Savitch's theorem for finding paths in the kind of graph that describes a JSON
format or a tool call, and turns this into an inference engine: text the format
already decides is written without running the model, the checks run on the
graphics chip, and a format needs about 3 MB of memory instead of up to 1.5 GB.
The full results are in the paper, among them sixteen tool-calling agents
finishing 2.5 times sooner than with mlx-lm and llguidance.

## Why Locals

Coding assistants no longer send one request at a time. OpenCode, OpenHands and
Cline run helpers in parallel, and teams of agents split one task between them.
On a Mac, the standard way of running models, mlx-lm, answers such requests
together but barely gets faster when there are more of them: it writes 87 tokens
per second for four requests and the same 87 for eight, so every agent you add
slows the others down. Locals keeps getting faster as requests are added, so the
more agents you run, the more it gives you:

| On a MacBook Pro with an M2 Pro and 16 GB | mlx-lm | Locals | |
|---|---|---|---|
| 8 requests at once, Qwen3.5-4B | 87 tokens/s | 271 tokens/s | 3.1x |
| 16 coding agents calling tools, Qwen3.5-2B | 75 s | 29 s | 2.5x |
| 40 programming problems from LiveCodeBench, 8 at a time | 20 min | 8 min | 2.5x |
| One request alone | 61 tokens/s | 64 tokens/s | the same |

With 1, 2, 4, 8 and 16 requests at once, Locals writes 64, 116, 160, 271 and 341
tokens per second, and mlx-lm 61, 82, 87, 87 and 157. A token is roughly a word
piece.

- **Many agents in the time of a few.** Eight agents working at once finish in
  about the time two would need one after the other, and the answers are the
  same as mlx-lm's.
- **Each turn picks up where the last one ended.** Locals keeps every
  conversation between turns, and long system prompts on disk across restarts,
  so an agent's next turn reads only what it added. With four agents in long
  conversations, 83% of what they sent came from memory, and a follow-up
  question about a 138,000-token document started in 2.4 seconds instead of
  eight minutes of reading it again.
- **Tool calls that always parse.** Replies are written in the model's own
  tool-call format under a grammar, so only the tools an agent offers can be
  called, and a reply cut off by its length limit still closes its call. A small
  model can also repeat itself or call the same tool forever; Locals notices and
  stops both.
- **A team of agents for bigger tasks.** `locals run` splits a task into parts,
  gives each part to several agents at once, each in its own copy of your
  project, and lets your tests decide. On four tasks that each span several
  files, it passed all eight attempts in about three minutes each; one agent
  alone passed four of eight in about five minutes each.

**How.** mlx-lm's way of multiplying by the model's compressed weights gets almost
no faster per request for the handful of requests at a time that agents send.
Locals uses GPU code written and tuned for exactly these small groups, and on M1
and M2 chips computes in a number format the chip handles natively.
[Why Locals is faster](https://github.com/Generated-Computers/Locals/blob/main/docs/why-faster/README.md) measures each difference on its
own, and [Measurements](https://github.com/Generated-Computers/Locals/blob/main/docs/measurements/README.md) has every benchmark and how
to repeat it.

## Works with your agents

<p align="center">
<img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/hermes.png" width="12.4%" alt="Hermes Agent" title="Hermes Agent"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/kilo.png" width="12.4%" alt="Kilo Code" title="Kilo Code"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/opencode.png" width="12.4%" alt="OpenCode" title="OpenCode"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/openhands.png" width="12.4%" alt="OpenHands" title="OpenHands"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/cline.png" width="12.4%" alt="Cline" title="Cline"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/aider.png" width="12.4%" alt="Aider" title="Aider"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/roo.png" width="12.4%" alt="Roo Code" title="Roo Code"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/openclaw.png" width="12.4%" alt="OpenClaw" title="OpenClaw">
</p>

Locals works with the coding agents and assistants people already use, on a
model running on your Mac. Each of these was tested end to end with Qwen3.5-4B
on a 16 GB Mac, on real tasks such as writing a program, running it and
reporting what it printed. One command starts Locals and connects the tool to it:

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

`locals launch list` shows every tool it can open. Locals keeps running in the
background after you close a tool: `locals status` shows what it is doing and
`locals stop` stops it.

These tools rarely send one request at a time. They run subagents and helpers in
parallel, and several sessions can work at once; Locals answers all of them
together in one batch, which is where it is fastest. Most of them also start
every session with a long system prompt of 3,000 to 16,000 tokens. Locals reads it
once and keeps it, on disk too, so the next turns and the next sessions start
without reading it again, even after a restart.

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

Then start a coding assistant on one of your projects:

```bash
locals code ~/my-project
```

The first time, Locals downloads a model that suits your Mac (about 3 GB),
starts it in the background and opens the OpenCode assistant in your project.
From then on it starts in seconds. If you already use the uv package manager,
you can install with `uv tool install git+https://github.com/Generated-Computers/Locals`
instead.

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
columns are speeds in tokens per second (tps) with 8 requests at once.

| Model | Start it with | Name in your editor | Size | mlx-lm | Locals |
|---|---|---|---|---|---|
| Qwen3.5-4B (the default) | `--model qwen` | `locals/qwen3.5-4b` | 2.8 GB | 87 tps | 271 tps |
| LFM2.5-2.6B | `--model lfm` | `locals/lfm2.5-2.6b` | 1.6 GB | 144 tps | 404 tps |
| MiniCPM5-2B | `--model minicpm` | `locals/minicpm5-2b` | 1.4 GB | 160 tps | 478 tps |
| Gemma 4 E4B | `--model gemma` | `locals/gemma-4-e4b-it` | 4.8 GB | 80 tps | 224 tps |

Each model also has a thinking mode, in which it reasons before it answers: add
`-thinking` to the name, for example `locals/lfm2.5-2.6b-thinking`. The smaller
models are good choices for a Mac with 8 GB of memory, and other models made for
Apple silicon on [Hugging Face](https://huggingface.co/mlx-community) work too.

Conversations can be long: about 145,000 tokens with the default model on a
16 GB Mac, a few hundred pages of text. A long document takes a few minutes to
read the first time; follow-up questions about it start in seconds.

## Using it from your own code

Locals understands the same requests as the OpenAI and Ollama services, so
programs written for them work with it unchanged. Start Locals with
`locals start`, then point your program at `http://127.0.0.1:8321/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8321/v1", api_key="local")
reply = client.chat.completions.create(
    model="locals/qwen3.5-4b",
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
task, the commands they run can only change files in their own copy of your
project.

## If something does not work

Run `locals status`, or open http://127.0.0.1:8321 in your browser, to see whether
Locals is running and what it is doing. The first time Locals runs on a new Mac,
it spends a few minutes in the background tuning itself to your chip; you can
use it during that time, and it gets faster once it finishes.

Questions and problems are welcome as
[issues on GitHub](https://github.com/Generated-Computers/Locals/issues).

## Learn more

[How Locals works](https://github.com/Generated-Computers/Locals/blob/main/docs/README.md) describes the server, the batching engine,
the prompt cache, the team of agents behind `locals run`, and every setting.
[Measurements](https://github.com/Generated-Computers/Locals/blob/main/docs/measurements/README.md) has the benchmarks and how to repeat
them. [Tests](https://github.com/Generated-Computers/Locals/blob/main/tests/README.md) and [CONTRIBUTING.md](https://github.com/Generated-Computers/Locals/blob/main/CONTRIBUTING.md) are for
working on Locals itself:

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

Locals is released under the MIT license. It is made by
[Generated Computers](https://github.com/Generated-Computers).
