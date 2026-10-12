<p align="center">
  <img src="docs/assets/logo.png" alt="Generated Computers" width="480">
</p>

<h1 align="center">Locals</h1>

<p align="center">
  Run many AI agents at once on your Mac, about three times faster.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="#getting-started"><img src="https://img.shields.io/badge/macOS-Apple%20silicon-lightgrey.svg" alt="macOS on Apple silicon"></a>
</p>

## What it is

Locals runs AI models on your own Mac and makes them fast when many requests
arrive at the same time. That is exactly how today's AI agents work: a coding
assistant runs several helpers in parallel, and a team of agents can work on one
task together. Locals answers all of these requests together on your Mac's
graphics chip instead of one after another, so the whole team gets its answers
about three times faster than with the standard way of running models on a Mac.

It works with the coding assistants you may already use, such as OpenCode,
Cline, Aider and OpenHands, and with any app or program that talks to OpenAI or
Ollama. Everything stays on your computer.

## How much faster

We measured Locals against mlx-lm, the standard way to run models on Apple
silicon, with the same model (Qwen3.5-4B) on a MacBook Pro with an M2 Pro chip
and 16 GB of memory. Speed is the number of tokens, roughly word pieces, the
model writes per second for all requests together.

| Requests at the same time | mlx-lm | Locals | Faster by |
|---|---|---|---|
| 1 | 61 tokens/s | 64 tokens/s | about the same |
| 2 | 82 tokens/s | 116 tokens/s | 1.4x |
| 4 | 87 tokens/s | 160 tokens/s | 1.8x |
| 8 | 87 tokens/s | 271 tokens/s | 3.1x |
| 16 | 157 tokens/s | 341 tokens/s | 2.2x |

With one request Locals is as fast as mlx-lm. The more requests arrive together,
the bigger the difference: mlx-lm gains almost nothing between four and eight
requests, while Locals keeps adding speed. In practice, eight agents working at once finish in
about the time two agents would need one after the other.

The same holds on real work. Sixteen coding agents that call tools, running on
the smaller Qwen3.5-2B, finished in 29 seconds instead of 75, and forty programming problems solved eight at a time
took 8 minutes instead of 20. Each agent also starts its next step in under a
second, because Locals remembers the conversation it has already read instead of
reading it again. The answers are the same as mlx-lm's.

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

## A team of agents on one task

Because many agents cost little more than one, Locals can put a whole team on a
single task. Each agent works on its own part, in its own copy of your project,
and your tests check every change. Your project is only updated once the tests
pass.

```bash
locals run "write inventory.py, cli.py and tests for them" --dir ~/my-project
```

On tasks that span several files, a team of agents solved all eight attempts in
about three minutes each. A single agent working alone solved four of eight and
needed about five minutes for each.

## Use it in the Terminal

For each of these tools, one command starts Locals and connects the tool to it.

| If you use | Run |
|---|---|
| OpenCode | `locals code ~/my-project` |
| OpenHands | `locals launch openhands` |
| Cline or Kilo Code | `locals launch cline` or `locals launch kilo` |
| OpenClaw or Hermes Agent | `locals launch openclaw` or `locals launch hermes` |
| Aider | `locals launch aider` |

To see every supported tool, run `locals launch list`. Locals keeps running in
the background after you close a tool: `locals status` shows what it is doing
and `locals stop` stops it.

## Use it in the OpenCode app, VS Code and other editors

Locals works just as well in apps with windows and buttons. First start it with
the model you want. It keeps running in the background, so you only do this once:

```bash
locals start --model qwen
```

**The OpenCode app.** Run `locals setup --model qwen` once. It adds Locals to
OpenCode's settings, which the OpenCode desktop app, its editor extension and its
Terminal version all share. Then open OpenCode and choose Locals in the model
list. If you later switch to another model, run `locals setup` again with that
model's name.

**Kilo Code, Cline and Roo Code in VS Code.** Open the extension's settings and
choose the API provider called "OpenAI Compatible". Then fill in these four
fields:

| Setting | What to enter |
|---|---|
| Base URL | `http://127.0.0.1:8321/v1` |
| API key | any text, for example `local` |
| Model | the model's name from the table below, for example `locals/qwen3.5-4b` |
| Context window | `65536`, if the extension asks for it |

**Continue, Zed and JetBrains** have the same kind of setting, usually called an
OpenAI-compatible provider, and take the same four values. Run
`locals launch editor` to print them for the model that is running, together
with ready-made settings for Continue and Zed.

**Apps that work with Ollama**, such as Open WebUI or Raycast, need no settings:
run `locals launch ollama`, and Locals answers at the address where these apps
look for Ollama.

## Models

Locals picks a model that fits your Mac's memory, so most people never need to
change it. To use another one, start Locals with its name. Locals runs one model
at a time, and starting it with another name switches to that model:

```bash
locals start --model lfm
```

Every model below runs about three times faster with Locals when eight requests
arrive at once. The third column is the name to enter in an editor or app.

| Model | Start it with | Name in your editor | Download size | mlx-lm, 8 requests | Locals, 8 requests |
|---|---|---|---|---|---|
| Qwen3.5-4B (the default) | `--model qwen` | `locals/qwen3.5-4b` | 2.8 GB | 87 tokens/s | 271 tokens/s |
| LFM2.5-2.6B | `--model lfm` | `locals/lfm2.5-2.6b` | 1.6 GB | 144 tokens/s | 404 tokens/s |
| MiniCPM5-2B | `--model minicpm` | `locals/minicpm5-2b` | 1.4 GB | 160 tokens/s | 478 tokens/s |
| Gemma 4 E4B | `--model gemma` | `locals/gemma-4-e4b-it` | 4.8 GB | 80 tokens/s | 224 tokens/s |

Each model also has a thinking mode, in which it reasons before it answers. To use
it, add `-thinking` to the name in your editor, for example
`locals/lfm2.5-2.6b-thinking`. LFM2.5 works best this way.

The smaller models are good choices for a Mac with 8 GB of memory. You can also
use other models published for Apple silicon on
[Hugging Face](https://huggingface.co/mlx-community), for example
`--model mlx-community/Qwen3.5-2B-MLX-4bit`; its name in your editor is then
`locals/qwen3.5-2b`. If a model is too big for your Mac, Locals tells you before
downloading it. Models in the GGUF format, which Ollama and LM Studio use, do not
run here, but Locals will point you to a version of the same model that does.

## Using it from your own code

Locals understands the same requests as the OpenAI and Ollama services, so
programs written for them work with it unchanged, and many of them at once get
the same speed-up. Start Locals with `locals start`, then point your program at it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8321/v1", api_key="local")
reply = client.chat.completions.create(
    model="locals/qwen3.5-4b",
    messages=[{"role": "user", "content": "Write a haiku about the sea."}],
)
print(reply.choices[0].message.content)
```

The [examples](examples) folder has more, including structured answers in JSON
and tool calls.

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

## For developers

```bash
git clone https://github.com/Generated-Computers/Locals
cd Locals
uv sync --extra dev
uv run pytest tests -q -m "not model"
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for how the project is organized and tested.

## License

Locals is released under the MIT license. It is made by
[Generated Computers](https://github.com/Generated-Computers).
