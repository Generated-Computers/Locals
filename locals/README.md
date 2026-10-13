# The `locals` package

This folder is the whole of Locals: the server, the batching engine, constrained
decoding, the GPU kernels, the command-line app and the team-of-agents harness.
This page maps it; the [technical documentation](../docs/README.md) explains each
part in depth.

## How a request travels

```
client (OpenCode, an editor, the openai library, curl)
  |  HTTP on 127.0.0.1:8321: OpenAI chat and text completions, or Ollama's /api
  v
server.py      checks the request, renders the chat template, picks the grammar for tool
               calls or a JSON schema, finds prompt-cache checkpoints, streams the reply back
  |
  v
service.py     BatchService: one model, one worker thread; every request is a Job
  |
  v
hybrid_batch.py  the batch engine: all running requests decode together, one token per row
               per step; new prompts are prefilled (in mixed passes while rows decode,
               mixed.py); finished rows go into the prompt cache
  |
  v
runtime.py, fuse.py, kernels.py, gdn_kernel.py ...  the model on the GPU, with fused
               layers and Metal kernels tuned for small batches
```

Agents talk to this server like to any OpenAI-compatible service. `locals run`
(harness.py) is itself a client: it starts many agents, each in its own copy of a
project, and they all share one batch and one prompt cache. More in
[Server and APIs](../docs/server/README.md), [Batch engine](../docs/engine/README.md)
and [Prompt cache](../docs/prompt-cache/README.md).

## Modules

### Serving

| Module | What it does |
|---|---|
| `server.py` | The OpenAI-compatible HTTP server over the batch service: endpoints, request preparation, streaming, reply guards, context limits ([details](../docs/server/README.md)) |
| `apis.py` | The Ollama API on the same engine: requests and streams translated to and from chat completions |
| `oai.py` | OpenAI chat messages to and from a model's native tool calling: message conversion, the reply grammar, reply parsing |
| `service.py` | `BatchService`: one model, one batch, many concurrent clients; our engine or stock mlx-lm behind the same interface |
| `hybrid_batch.py` | `HybridBatchEngine`: continuous batching for any mlx-lm model, recurrent and attention layers alike; admission, prefill, the memory guard, the prompt cache ([details](../docs/engine/README.md)) |
| `mixed.py` | Decode rows and a prompt chunk in one forward pass (Qwen3.5), so running rows keep decoding while a long prompt is read |
| `packed.py` | `Request` and the packed multi-request forward pass the batch engine builds on |
| `engine.py` | The single-stream generation loop with pluggable constraint controllers (`Controller`, `LocalsController`), and sampling |

### Constrained decoding

| Module | What it does |
|---|---|
| `grammar.py` | JSON Schema to a grammar of regular expressions (recursive transition network) |
| `regex.py` | Regular expression to a byte-level automaton (NFA, then DFA) |
| `vocab.py` | Each token's bytes and a byte trie of the vocabulary |
| `constraint.py` | A regex constraint compiled against a vocabulary: the token transition table |
| `forced.py` | Byte chains the grammar determines, turned into forced token spans |
| `budget.py` | Budget-aware masking: never a token after which no valid ending fits in `max_tokens` |
| `walk.py` | Constraints without tables: each step walks every token through the automaton on the GPU |
| `pda.py`, `pdactl.py` | Recursive schemas as a pushdown automaton, with the stack on the GPU |
| `lazy.py` | Constraint tables built only for the states that are visited |
| `formats.py` | The tool-call formats of each model family, and thinking styles |
| `api.py` | The library API: compile a grammar once, use it in any decode loop or in ours |
| `llg.py` | llguidance as an mlx-lm logits processor, the baseline the stock engine uses |
| `fields.py` | Block-parallel decoding of independent fields |

More in [Constrained decoding](../docs/decoding/README.md).

### Model runtime and GPU kernels

| Module | What it does |
|---|---|
| `runtime.py` | `Runtime`: load a model once, choose the compute precision, generate with or without a schema |
| `lm.py` | mlx-lm's functions with the same arguments, faster |
| `fuse.py` | Fused projections (q/k/v, gate/up, Qwen3.5's linear-attention inputs) as single matmuls |
| `kernels.py` | Routing of small-batch quantized matmuls to the fastest kernel, and the autotuner |
| `qmv_multi.py`, `qmv_frag.py`, `mma_kernel.py` | Metal kernels for 4-bit matmuls with few rows |
| `gdn_kernel.py` | One Metal kernel for a Gated DeltaNet decode step |
| `norm_kernel.py` | Fused residual add and RMS norm |
| `attention.py` | Ragged batch decode attention: one kernel launch for all rows |
| `defaults.py` | The tested models, their short names, the default for this Mac's memory, size checks before a download |
| `presets/` | Kernel tuning results shipped with the package (one preset, for the `applegpu_g14s` GPU architecture), so a first start on such a GPU skips most tuning |

More in [Runtime and kernels](../docs/runtime/README.md).

### Commands and integrations

| Module | What it does |
|---|---|
| `app.py` | The `locals` command: start, stop, status, code, launch, setup, run, generate, chat, and the background server ([details](../docs/cli/README.md)) |
| `cli.py` | The command-line entry point |
| `chat.py` | Interactive chat in the Terminal |
| `menubar.py` | Locals in the macOS menu bar |
| `agents.py` | `locals-agents`: many tool-using agents in parallel on one model |

### Team of agents

| Module | What it does |
|---|---|
| `harness.py` | `locals run`: plans a task, runs several agents per part in copies of the project, checks, combines and repairs ([details](../docs/agents/README.md)) |
| `plan.py` | Splitting one task into parts written at the same time |
| `solve.py` | Agents in copies of the project, the check after each write, the sandbox for their commands |

### Research decoders

These come from the work on exact decoding with certified decisions. The library
exposes them; the server does not use them by default.

| Module | What it does |
|---|---|
| `core.py` | Graph primitives from "Total Strongly-Few Directed Reachability" |
| `blocks.py` | Decision-bound decoding: independent blocks in parallel, certified exact |
| `regions.py` | Exact decisions in regions with few paths |
| `siblings.py` | Sibling-parallel decoding of recursive structures |
| `tree.py`, `treedec.py` | Shared-prefix (tree) caches and exact tree decoding of finite regions |
| `lookup.py` | Greedy decoding with prompt-lookup drafts verified in one pass |
