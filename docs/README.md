# Technical documentation

How Locals works, for anyone who wants to understand it in depth or change it.
The [main README](../README.md) is the place to start using it; these pages
explain the machinery behind it, with the names of the files, functions, flags
and settings involved.

| Page | What it covers |
|---|---|
| [The package](../locals/README.md) | A map of the `locals` package: what each module does and how a request travels through them |
| [Server and APIs](server/README.md) | The HTTP server: endpoints, OpenAI and Ollama compatibility, request options, streaming, reply guards, context limits, security, command-line flags |
| [Batch engine](engine/README.md) | Continuous batching: the pipelined step loop, admission, prefill, mixed passes, the memory guard, sampling, constrained rows, cancellation and recovery |
| [Prompt cache](prompt-cache/README.md) | How conversations are kept between requests: checkpoints, the reply checkpoint, eviction, the disk cache, memory sizing |
| [Constrained decoding](decoding/README.md) | JSON schemas and regular expressions as token automata on the GPU, forced spans, budgets, tool-call formats of each model family, thinking modes, the library API |
| [Runtime and kernels](runtime/README.md) | Loading models, compute precision, fused layers, the Metal kernels for small batches, autotuning, model names and checks |
| [Command line](cli/README.md) | The `locals` commands, how the background server is managed, the integrations with coding tools and editors, files and environment variables |
| [Team of agents](agents/README.md) | `locals run`: planning, versions of each part, the check after every write, combining and repairing, the sandbox |
| [Measurements](measurements/README.md) | Every benchmark behind the numbers in the README, what else was measured, and how to repeat it |

The [tests](../tests/README.md) and [examples](../examples/README.md) folders
have their own pages.
