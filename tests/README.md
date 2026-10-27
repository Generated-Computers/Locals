# Tests

The test suite lives in `tests/`: 39 `test_*.py` files, a shared `conftest.py`, the workload definitions in `workloads.py`, and recorded agent requests in `tests/data/`. pytest is configured in `pyproject.toml` (`testpaths = ["tests"]`, and one registered marker, `model`).

Almost every test imports MLX, so the suite runs on Apple silicon only. Tests are split in two:

- **The fast suite** needs no model weights. At most it needs a tokenizer or a chat template from the local Hugging Face cache. It covers grammars, tokenizers, the server protocol, sampling and the agent tools.
- **The model suite** (tests marked `model`) loads real weights. It checks that every fast path decodes the same tokens as the plain path it replaces.

A third suite, `tests/e2e` (marked `e2e`), runs the `locals` command, the server and real agent tools as a user does. It is skipped unless `LOCALS_E2E=1` is set; see [tests/e2e/README.md](e2e/README.md).

## Running the tests

Install the package in editable mode with the `dev` extra, which adds `pytest>=8` and `build>=1.2`:

```bash
pip install -e ".[dev]"
```

Then run the suite you need:

```bash
python -m pytest tests -q -m "not model"   # fast suite (CONTRIBUTING.md: about 2 min)
python -m pytest tests -q                  # everything (CONTRIBUTING.md: ~6 minutes once the models are cached)
python -m pytest tests -q -m model         # the model suite only
LOCALS_TEST_MODEL=mlx-community/Qwen3.5-4B-MLX-4bit python -m pytest tests -q -m model
python -m pytest tests/test_regex.py -q    # one file
```

A test that needs a model or tokenizer you have not downloaded is skipped, not failed. A green run with many skips may therefore have tested little. Use `-rs` to see the skip reasons.

### Downloading what the tests need

These are the downloads given in CONTRIBUTING.md and used by CI:

```bash
# the fast suite: the Qwen2.5-1.5B tokenizer files only
python -c "from huggingface_hub import snapshot_download as d; d('mlx-community/Qwen2.5-1.5B-Instruct-4bit', allow_patterns=['*.json', '*.txt', 'tokenizer*'])"
# the model suite: the default LOCALS_TEST_MODEL
python -c "from huggingface_hub import snapshot_download as d; d('mlx-community/Qwen3.5-2B-MLX-4bit')"
```

Some model tests use other weights, named in the test files themselves:

```bash
# weights for test_engine_model.py, test_budget.py and test_server_attention.py
python -c "from huggingface_hub import snapshot_download as d; d('mlx-community/Qwen2.5-1.5B-Instruct-4bit')"
# test_budget.py::test_drip_batch_valid_under_tight_budget
python -c "from huggingface_hub import snapshot_download as d; d('mlx-community/Qwen3.5-0.8B-MLX-4bit')"
```

## The `model` marker

The marker is registered in `pyproject.toml` as "needs model weights cached locally". Most tests do not carry it themselves. `conftest.py` adds it at collection time: `pytest_collection_modifyitems` marks every test whose file is in `MODEL_TEST_FILES`, and every test that already has `model` among its keywords.

`MODEL_TEST_FILES` lists these 13 files:

`test_blocks.py`, `test_budget.py`, `test_engine_model.py`, `test_lm.py`, `test_lookup.py`, `test_mixed_prefill.py`, `test_pda.py`, `test_prefix_cache.py`, `test_q35_kernels.py`, `test_request_temperature.py`, `test_server_attention.py`, `test_server.py`, `test_walk_batch.py`.

A file that mixes both kinds marks single tests instead. `test_api.py` is mostly fast, but `test_api_generate_with_runtime_and_mlx_lm` carries `@pytest.mark.model`.

When you add tests:

- For a new file whose tests load weights, add its basename to `MODEL_TEST_FILES`.
- For one weight-loading test in an otherwise fast file, decorate it with `@pytest.mark.model`.

Because whole files are marked, some tests that need no weights run only with the model suite:

- `test_q35_kernels.py::test_kernels_match_mlx` uses random quantized layers.
- The first tests of `test_budget.py` and `test_pda.py` are tokenizer-only property tests.

There is also one exception the other way round. `test_table_compaction.py` loads `Qwen3.5-2B` weights through `Runtime.load`, but it is not in `MODEL_TEST_FILES` and has no marker. `-m "not model"` therefore runs it when that model is cached, and skips it otherwise (as in CI).

## Which models the tests use, and how they skip

Every fixture that needs a model first checks the local cache with `snapshot_download(repo, local_files_only=True)`. If that fails, it calls `pytest.skip("model not cached")` or `pytest.skip("tokenizer not cached")`. The check itself never downloads anything. `test_think_replay.py` does the same check with `AutoTokenizer.from_pretrained(repo, local_files_only=True)`.

Modules that need optional imports skip with `pytest.importorskip`:

- `mlx_lm` in most model and tokenizer files
- `outlines_core` in `test_json_safe.py`
- `llguidance` in one test of `test_agents.py`
- `mlx.core` and `mlx.nn` in `test_kernel_patching.py`

The check only asks whether the repository has a snapshot in the cache. It does not check which files that snapshot contains. A repository fetched for its tokenizer files alone (like the Qwen2.5-1.5B download above) passes the check. A model test that uses that repository will then go on to load the weights, so download the full repository before running those tests.

### Models

| Repository | Used as | Used by |
|---|---|---|
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | tokenizer (fast suite) | `test_api.py` (`TOKENIZER_MODEL`), `test_forced.py` |
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | weights | `test_engine_model.py`, `test_budget.py` (`MODEL`), `test_server_attention.py` (default of `LOCALS_TEST_ATTENTION_MODEL`) |
| `mlx-community/Qwen3.5-2B-MLX-4bit` | weights, default of `LOCALS_TEST_MODEL` | `test_api.py` (model test), `test_blocks.py`, `test_lm.py`, `test_lookup.py`, `test_mixed_prefill.py`, `test_pda.py`, `test_prefix_cache.py`, `test_q35_kernels.py`, `test_request_temperature.py`, `test_server.py`, `test_walk_batch.py`, `test_table_compaction.py` |
| `mlx-community/Qwen3.5-0.8B-MLX-4bit` | weights (`HYBRID`) | `test_budget.py::test_drip_batch_valid_under_tight_budget` |
| Other cached models | tokenizers and chat templates only; each case skips on its own | `test_agent_requests.py` (11 repositories in `MODELS`), `test_formats.py` (format detection list, Ministral, MiniCPM5), `test_chat.py` (`gemma-4-e4b-it-4bit`, `Qwen3.5-4B-MLX-4bit`), `test_think_replay.py` (`Qwen3.8-27B-4bit`, `Qwen3.5-4B-MLX-4bit`) |

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `LOCALS_TEST_MODEL` | `mlx-community/Qwen3.5-2B-MLX-4bit` | The model for the model tests listed above. Several of them assume a Qwen3.5 checkpoint: `test_q35_kernels.py` tests Qwen3.5 fusion and the GDN kernel, and `test_prefix_cache.py` hard-codes a Qwen turn with an empty think block. Point it at another Qwen3.5 size, as CONTRIBUTING.md does with the 4B. |
| `LOCALS_TEST_ATTENTION_MODEL` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | The plain-KV model for `test_server_attention.py` (Qwen2.5, Llama, Qwen3 or Mistral, per its docstring). |
| `LOCALS_DISK_CACHE` | set to `0` by `conftest.py` | Turns off the on-disk prompt cache. Outside tests, the default `1` makes `BatchService` keep system prompts under `~/.cache/locals/prompts` (or `LOCALS_DISK_CACHE_DIR`). |

## Protections in `conftest.py`

**No real user files (`_no_real_user_files`, autouse, every test).**

- It sets `LOCALS_DISK_CACHE=0`, so no test reads or writes the on-disk prompt cache.
- If `locals.app` imports, it points `app.LOCALS_DIR`, `app.OPENCODE_CONFIG` and `app.OPENCODE_BIN` into a temporary home directory.
- The docstring gives the reason: a test of `locals code` once rewrote `~/.config/opencode/opencode.json`.

Tests that touch other user-level state redirect it themselves:

- `test_background_tune.py` and `test_kernel_patching.py` monkeypatch `kernels._TUNE_PATH` to `tmp_path`.
- `test_app.py` sets `LOCALS_DIR` to `tmp_path`.

One test writes outside its temporary directory on purpose. `test_harness.py::test_agent_commands_write_only_in_their_copy` writes a file named `~/.locals-sandbox-test-<pid>` to prove that a command not confined by `sandbox-exec` can write there, then deletes it. It is skipped where `sandbox-exec` is missing.

**GPU memory between modules (`_free_gpu_memory_between_modules`, autouse, module scope).**

- After each module it runs `gc.collect()` and `mx.clear_cache()`.
- Each module loads its own model copies. Without this step, the docstring says, a 4B suite outgrows a 16 GB Mac's GPU limit, and later modules fail with Metal "Insufficient Memory".

**Fixture values released after each test (`pytest_runtest_teardown`, hookwrapper, `trylast`).**

- It clears `item.funcargs`.
- pytest otherwise keeps each test's fixture values until the session ends, which would keep every module's model alive.

`test_q35_kernels.py` applies the same rule inside one module: `_PAIR` holds one (stock, fused) model pair at a time. Its comment explains that two 4B pairs (4 x 2.6 GB) do not fit a 16 GB Mac's GPU.

## Shared workloads and test data

### `workloads.py`

The model tests share these prompts and schemas. They import them as `from workloads import ...`, which works because pytest puts `tests/` on `sys.path`. The users are `test_blocks.py`, `test_budget.py`, `test_engine_model.py`, `test_lookup.py`, `test_pda.py` and `test_walk_batch.py`.

| Name | What it is |
|---|---|
| `WORKLOADS` | Six extraction tasks, each with a JSON schema and a source text: `person`, `tool_call`, `product`, `event`, `classification`, `contact_card` |
| `make_prompt`, `prompt_text`, `compact_schema` | The extraction prompt: with the full schema, without it, or with a compact key/enum outline. `make_prompt` renders it through the chat template with thinking off. |
| `TREE`, `EXPR`, `THREAD`, `FS`, `CATS`, `JOBS` | Recursive schemas (`$ref` cycles) and a task for each |
| `PHRASINGS`, `recursive_prompt` | Three phrasings of a recursive-schema request |
| `valid`, `depth` | A small validator (`$ref`, `anyOf`, `enum`, object, array with `minItems`, scalar types) and a nesting-depth helper |

### `tests/data/`

These files hold real agent requests, recorded against a scripted OpenAI-compatible server. Only `test_agent_requests.py` reads them.

| File | Contents |
|---|---|
| `opencode_requests.json` | OpenCode 1.18.32, `opencode run`, paths scrubbed. Keys: `system`, `tools` (10), `request_options`, `title_request`, and `conversations`: `basic`, `text_and_call`, `two_calls`, `error`, `chain`, `followup`, `write`. |
| `agent_requests.json` | OpenClaw 2026.9.8 (11 tools; conversations `basic`, `two_calls`, `chain`, `error`, `first_turn`) and Hermes Agent (24 tools; conversations `basic`, `two_calls`, `chain`, `error`, `text_and_call`, plus a JSON-schema `title_request`). System prompts truncated; paths and host names scrubbed. |
| `framework_requests.json` | 23 conversations from pydantic-ai 2.54, smolagents 1.26, openai-agents 0.23, LlamaIndex 0.14, DSPy 3.4, the OpenAI SDK, LangChain, LiteLLM, Instructor and Aider 0.86. System prompts truncated. |

A conversation added under `conversations` is picked up by `_conversations()` and becomes one more parametrized case of `test_agent_conversation`.

## What each file checks

In the Suite column, "fast" means the file runs under `-m "not model"`. "Fast, cached tokenizers" means it also skips its cases for models that are not in the cache.

| File | Suite | What it checks |
|---|---|---|
| `test_agent_requests.py` | fast, cached tokenizers | The recorded OpenCode, OpenClaw, Hermes and framework requests go through `Server.prepare` with each cached model's chat template. The template must accept them, every user message and tool result must reach the prompt in order, the prompt must fit `max_context`, and the model's turn must stay open. Also covers title requests, `response_format` `{"type": "regex"}` and `guided_regex`, and that an unknown `response_format` type raises `ValueError`. No GPU: the service is a stand-in holding the tokenizer. |
| `test_agents.py` | fast | `locals.agents`: `ACTION_REGEX` accepts Qwen-native calls and rejects unknown tools, missing parameters and leading text. Covers `parse_action`, tools kept inside the agent's directory, and `_llg_grammar` naming special tokens by id for llguidance. |
| `test_api.py` | fast, plus one model test | The library API on the Qwen2.5 tokenizer: `Vocabulary`, `GrammarCompiler` (cached by source), `Matcher` (mask, `accept_token`, `jump_forward` spelling the next bytes, `fork`). Also regex and recursive (pushdown) grammars, the canonical boundary rule, tool-call grammars with `tool_choice` `"required"` and `"none"`, and `logits_processor` (masks like the matcher, restarts on a repeated prompt, `ValueError` on a disallowed token). |
| `test_apis.py` | fast | The Ollama API: `apis.ollama_to_chat`, `chat_to_ollama`, `OllamaStream` (streamed content and calls), and `think` levels mapped to `enable_thinking` and `reasoning_effort`. |
| `test_app.py` | fast | The `locals` command: OpenCode, OpenClaw and Hermes configs route model ids to the intended engine (`oai.engine_of`) with a context window of at least 64000. Also `locals code` setup and `--run`, short model names, keeping the user's OpenCode config, a commented config not being rewritten, restarting only our own idle background server, and `stop` never killing what a stale pid file names. |
| `test_background_tune.py` | fast | Untuned kernel shapes use MLX's kernel at once (`kernels._choose` returns `("mlx",)`). `tune_one` then tunes them one at a time, smallest batch first, into the tuning file. |
| `test_batch_filter.py` | fast | `hybrid_batch._filter` and `_append_rows` match mlx-lm's cache filtering and extension. `prefill_chunk` bounds the tokens of one forward pass by `PREFILL_TOKENS` and `PREFILL_SCORES`: 16 prompts of 1k tokens once ran a 16 GB Mac out of GPU memory. |
| `test_blocks.py` | model | Decision-bound decoding (`blocks.py`): the blocks partition the output, and the certified output is token-identical to sequential decoding in fewer passes. `regions.py` enumeration equals brute force, and the batched scorer equals single-row scoring. `lazy.py` rows equal the full table's columns. The walk kernel's rows equal the token table. |
| `test_budget.py` | model (whole file) | Budget-aware masking (`budget.py`). A simulated stopping rule with random and adversarial choices always ends in an accepting state within `max_tokens`, and `need(start)` is the true minimum. The pipelined, synchronous and drip batch engines stay valid under tight `max_tokens`, and the mask changes nothing when there is room. |
| `test_chain_cache.py` | fast | `hybrid_batch.chain_cache` drops stale copies of a conversation but keeps system-prompt entries and the just-finished request's checkpoint. The `DiskPromptCache` directory is mode `0o700`, and a file that does not load is forgotten. |
| `test_chat.py` | fast, cached tokenizers | `locals chat`'s `Conversation` builds the second turn in the model's own template (Gemma 4 and Qwen3.5) and reuses the cached history as a prefix. |
| `test_core.py` | fast | The graph primitives in `core.py` ("Total Strongly-Few Directed Reachability"): `st_connectivity_sf` against a brute-force reference on 40 random seeds, `root_budget`, `capped_traverse` on cycles, serial diamonds (2^r paths), rank and unrank of paths, rank-one updates, `through_counts`, `branch_kernel` bounds and `pair_budget`. |
| `test_engine_model.py` | model (Qwen2.5-1.5B) | Pipelined and synchronous decoding give identical tokens, constrained and free. Lossless forced spans give the same text as plain constrained decoding in fewer passes. Outputs match outlines-core's schema regex. Tree-KV branches at temperature 0 equal single greedy decoding. |
| `test_forced.py` | fast (Qwen2.5 tokenizer) | Token bytes round-trip, and the token table equals DFA stepping on 20000 random (state, token) pairs. Forced plans spell the forced bytes in the tokenizer's own split and stay valid. The stable-tail rule holds back `":"`, and the lossless planner holds back the last block. |
| `test_formats.py` | fast, cached tokenizers | `formats.py`: each format's reply grammar accepts what its models write and nothing else (hermes, llama3_json, mistral, mistral_v11, gemma4, MiniCPM5, LFM2.5). Replies parse back to OpenAI calls, automata stay under 1500 states, and streaming holds calls back. Also format detection from real chat templates, tool results reaching the next prompt, `parallel_tool_calls` false, and `reasoning_effort` mapping. |
| `test_harness.py` | fast | The `locals run` harness, with `harness._chat` monkeypatched to scripted replies. Covers parts, versions in copies, the check picking and blaming, `edit_file` whitespace forgiveness, stuck agents ending, a failing run leaving the project as it was, and `run_shell` confinement under `sandbox-exec`. |
| `test_json_safe.py` | fast | `grammar.json_safe_schema`: patterns match against the string as JSON writes it, and `{}` property schemas accept any value. Random walks through the DFA end in valid JSON. Impossible schemas raise `GrammarError`, and regex syntax the engine cannot honour raises `RegexError`. |
| `test_kernel_patching.py` | fast | Fusion and kernel routing on tiny quantized layers: mixed quantization does not fuse, a fused layer equals its parts and leaves the global random state alone, and routing changes only the model given. A cut-short tuning cache is set aside as `tune.json.bad`. Also covers LFM2-style layers without `mlp` and `head_rows` on an mxfp4 head. |
| `test_lfm3.py` | fast, plus cached-tokenizer cases | `locals/models/lfm3.py` on a small random LFM3 (no download): `register()` puts it where mlx-lm and transformers look, its partial RoPE is Liquid's (zero tail), a cache (convolution state, sliding-window ring, full keys) continues a prompt as one pass does, alone and in a left-padded batch, and `sanitize` takes Liquid's MLX names and the original ones. On LFM3's cached template: it reads as Qwen3.5's calls, two newlines apart and none after `</think>`, the grammar writes calls as the template renders them, requests think unless told not to, a reply with thinking off holds no think tags, and a thinking conversation stays a prefix of the next request (`preserve_thinking`). |
| `test_lm.py` | model | `locals.lm` as a drop-in: mlx-lm's `generate`, `stream_generate` and `batch_generate` run on a patched model. Greedy text matches stock mlx-lm in bfloat16 (first 40 characters), batch matches single requests (first 20), and `locals.patch is lm.patch`. |
| `test_lookup.py` | model | Prompt-lookup drafts: a rewound fused-GDN cache continues like sequential steps (same argmax, error within twice the kernel noise floor plus 0.05). Greedy lookup decoding is token-identical to greedy decoding on regex, pushdown and free prompts. |
| `test_mixed_prefill.py` | model | Prompts prefilled in mixed passes (`mixed.py`) give same-length replies with at least 90% equal tokens and the first 8 equal. The row that kept decoding keeps its first 16 tokens, and checkpoints inside mixed passes reach the prompt cache. |
| `test_model_size.py` | fast | `defaults.too_big` refuses weights that do not fit next to the GPU limit (the message names both sizes and `LOCALS_IGNORE_MEMORY`); unknown sizes and the override are not refused. Also `weights_gb`, `gguf_hint` and `short_name`. |
| `test_oai.py` | fast | `oai.py`: `not_containing` is exact on exhaustive and random strings. Covers the Qwen reply grammar, argument coercion, `ReplyStream` holding back partial tags, `convert_messages`, and the loop guards (`server._looping` for any period, line loops). Also per-request temperature, an engine error mid-stream as an SSE event, steered replies, `ignore_eos` with a usage chunk, the keepalive line, and the context limit from memory (`context_fit`, `_cap_context`). |
| `test_pda.py` | model (whole file) | Recursive schemas as a pushdown automaton: grammar shapes, and token tables equal to byte-level stepping on every reachable configuration (stacks up to depth 4, `MAX_POP`/`MAX_PUSH`). Real-vocabulary traces. With the model: pipelined equals sync, outputs nest deeper than the regex path's 3 levels, expression trees parse, and budget masking holds. |
| `test_plan.py` | fast | `locals solve --plan` with scripted agents: `plan.normalize` makes the plan safe (paths outside the project dropped), part prompts, parts written at once, search combining parts of samples, repairs, blame, `files_regex`, streamed `chat_files`, and the planner skipped when the task names its files. |
| `test_prefix_cache.py` | model | A follow-up served from the prompt cache gives the same tokens as a `BatchService` with effectively no cache (`cache_gb=0.001`). |
| `test_q35_kernels.py` | model (whole file) | Small-M kernels (`qmv_frag`, `qmv_multi`) agree with `mx.quantized_matmul` for M in 1, 2, 3, 5, 8, 11, 16 in bf16 and fp16. The fused, kernel-routed Qwen3.5 model matches stock logits for batches 1, 3, 8 within the stock noise floor; the GDN kernel matches the stock layer; long prefills (`PREFILL_M`) match. |
| `test_regex.py` | fast | `regex.compile_dfa` equals Python `re.fullmatch` on 12 patterns with 3000 random strings each (UTF-8 included). `utf8_sequences` is exact, and outlines-core schema regexes compile. |
| `test_reply_parsing.py` | fast | `parse_reply` edge cases: values holding an example call or `</think>`, string-typed values staying strings, and calls to tools not offered not counting as calls. Repeated data is not a loop. Ollama call ids are stable, and raw `generate` maps to a text completion. |
| `test_request_temperature.py` | model | In a `BatchService` sampling at `temp=1.0`, a `temp=0.0` request gives the same tokens alone and in a batch with `temp=1.5` rows. |
| `test_sampling.py` | fast | `engine.sample_logits`: temperature 0 is argmax, `top_k` and `top_p` keep the right sets, `NEG_INF` tokens are never sampled, and a row shorter than `top_k` still samples. |
| `test_server.py` | model | `Server` end to end: tool calls decode under the reply grammar with forced tokens, and the round trip is served from the prompt cache. Also `json_schema` response formats, thinking streamed as `reasoning_content`, a repeated call steered away, a moved context message hitting the checkpoint, `tool_choice` `"none"`, `ContextTooLong`, `ignore_eos` (also on the stock `mlx` engine), and a follow-up turn equal to a fresh read. |
| `test_server_attention.py` | model (Qwen2.5-1.5B) | On a plain-KV model, `_needs_checkpoints` is false, and a moved context message still hits the cache. |
| `test_server_guards.py` | fast | `_origin_ok` (with `LOCALS_ORIGINS`), `_host_ok` against DNS rebinding, `_template_switches` keeping only `reasoning_effort`, and `_unsupported` refusing `n>1` and `logprobs`. A real HTTP handler with a stub server returns 403 for a foreign Origin or Host and 400 for bodies that are not JSON objects. |
| `test_snapshots_bounded.py` | fast | `HybridBatchEngine._snapshot` keeps only the `max_snapshots` most recent shared-prefix snapshots. |
| `test_solve.py` | fast | `locals solve`: a git worktree workspace holds uncommitted edits, deletions and untracked files, and a copy workspace skips `node_modules`. Covers `manifest`, `changes` and `unified_diff`, `detect_check` and `run_check` with timeout, prompts, hedging, escalation, the check writer, and notes without the server's steering text. |
| `test_table_compaction.py` | fast marker, loads Qwen3.5-2B | The batch engine's combined mask table is rebuilt while a long row runs; the row's device state moves with it, and outputs still match their regexes. |
| `test_think_replay.py` | fast, cached tokenizers | A thinking reply replayed by tool-call id renders as the model wrote it: the Qwen3.8 second prompt extends the first, and Qwen3.5 rendering is unchanged. |
| `test_walk_batch.py` | model | The walk kernel in `HybridBatchEngine` (`WalkController`) is token-identical to token tables (`NoHealController`) on four workloads. A `Server` with `walk=True, typed_tools=True` returns arguments of the declared types. |

## How CI runs the tests

`.github/workflows/tests.yml` runs on every push to `main` and on every pull request. Its single job, `fast`, does the following:

- **Runner:** `macos-14`, Python 3.11, 3.12 and 3.13 (`fail-fast: false`, 30-minute timeout).
- **Install:** `pip install -e ".[dev]"`.
- **Tokenizer:** downloads only the tokenizer files of `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.
- **Tests:** `python -m pytest tests -q -m "not model"`.
- **Packaging check:** builds the wheel with `python -m build`, reinstalls it (`--no-deps --force-reinstall`) and prints `locals.__version__`.

The model suite does not run in CI. The workflow comment says it runs locally with `pytest -m model`. Any other cached model is absent in CI, so its cases skip there.

## House rule for new mechanisms

Every mechanism is tested against the plain path it replaces:

- forced spans against token-by-token decoding
- the GPU walk against the token table
- the pushdown against byte-level stepping
- certified blocks against sequential decoding
- budgets against a simulated stopping rule

Keep that discipline (CONTRIBUTING.md): **an exactness test first, a speed number second.** In practice, write a test that decodes with the new path and with the old one and asserts equal tokens. Examples are `test_pipelined_equals_sync`, `test_lookup_equals_greedy` and `test_batch_walk_matches_tables`. Where floating-point kernels make that impossible, assert equal greedy choices within a measured noise floor, as `test_q35_kernels.py` and `test_lookup.py` do. If the test loads weights, add the file to `MODEL_TEST_FILES`.
