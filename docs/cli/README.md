# Command line and integrations

Part of the [technical documentation](../README.md) of Locals. File and function names refer to the `locals` package.

This part of the package is the user-facing side. It is the `locals` command, the background server it manages, the configuration it writes for agent tools (OpenCode, OpenClaw, Hermes Agent, and Cline and Kilo Code in config files under `~/.locals`), the environment variables it sets for Aider and OpenHands, the settings it prints for editors, and the Ollama port it can serve, and the small standalone programs `locals generate`, `locals chat`, `locals agents` and `locals menubar`. The inference engine and the HTTP server are documented elsewhere. This section covers how the command line drives them.

## Files

| File | Role |
|---|---|
| `locals/app.py` | The `locals` command: argument parsing, every subcommand except the delegated ones, background-server management, tool configuration and launching, `swarm`, `bench`. |
| `locals/cli.py` | `locals generate`: one generation (optionally under a JSON schema or regex) from the command line, on the background server (in-process with `--local`). |
| `locals/chat.py` | `locals chat` / `locals-chat`: an interactive chat, on the background server (in-process with `--local`). |
| `locals/client.py` | The server side of `locals chat` and `locals generate`: starts or reuses the background server and talks to it. |
| `locals/agents.py` | `locals agents` / `locals-agents`: many tool-using agents on one in-process `BatchService`, with a built-in task suite. |
| `locals/menubar.py` | `locals menubar`: a macOS menu-bar app (needs the `menubar` extra). |
| `install.sh` | The one-line installer (uv, then `uv tool install`). |
| `pyproject.toml` | Package metadata, dependencies, console scripts. |

`locals run` and `locals solve` are parsed in `app.py` but implemented in `locals/harness.py` (`cmd_run`) and `locals/solve.py` (`cmd_solve`).

## Packaging and install

`pyproject.toml` declares the package `locals` version `0.2.0`, `requires-python = ">=3.11"` (the comment says mlx-lm 0.32 needs 3.11), and three console scripts:

| Script | Target |
|---|---|
| `locals` | `locals.app:main` |
| `locals-agents` | `locals.agents:main` |
| `locals-chat` | `locals.chat:main` |

Runtime dependencies are `mlx>=0.32,<0.34`, `mlx-lm>=0.32,<0.33`, `numpy>=1.26`, `huggingface_hub>=0.30`, `llguidance>=1.8` (for the stock `mlx/` engine), `outlines-core>=0.2,<0.3` and `rich>=13`. Two extras exist: `dev` (`pytest>=8`, `build>=1.2`) and `menubar` (`rumps>=0.4`). The shipped kernel-tuning presets `presets/*.json` are package data.

`install.sh` is a POSIX `sh` script. It does the following:

1. It refuses anything other than Darwin on `arm64` with macOS 14 or newer.
2. It uses `uv` from the `PATH` or from `~/.local/bin/uv`. If neither exists, it installs uv with `https://astral.sh/uv/install.sh`.
3. It runs `uv tool install --force "$SOURCE"`. `SOURCE` is `$LOCALS_SOURCE` when set, else `locals @ https://github.com/Generated-Computers/Locals/archive/refs/heads/main.tar.gz`, GitHub's archive of the main branch: unlike `git+https://...` it needs no git, which a Mac without the Command Line Tools lacks. `LOCALS_SOURCE` is anything `uv tool install` accepts. The script's header shows a package name (`LOCALS_SOURCE=locals`, for PyPI) or a wheel path, and a local folder also works.
4. If uv's bin directory was not on the original `PATH`, it runs `uv tool update-shell` and prints the `export PATH=...` line.
5. It prints `locals --version` and the next commands to run.

## How `locals` parses a command

`main(argv)` in `app.py` works in four steps:

1. **Platform check.** Unless the first argument is `--version`, `-h` or `--help`, `_platform_ok()` exits on anything that is not `darwin`/`arm64`.
2. **Delegated commands.** `generate`, `chat`, `agents` and `tune` have their own `argparse` parsers. `menubar` has none: `menubar.main(argv=None)` takes no options. `_delegate` maps the five names to their entry points. `main` skips any leading `--model`/`--port` options (both `--model X` and `--model=X` forms) and looks at the next word. If that word is a delegated command, `main` calls its `main` with the leading `--model` (never `--port`) followed by the command's own arguments, so the command's own options win. `menubar` gets the arguments that follow its name, without the leading `--model`. `menubar.main` ignores them.
3. **Saved defaults.** For every other command, `--model` and `--port` default to the values in `~/.locals/config.json` (`_load_saved`), and otherwise to `DEFAULT_MODEL` and `DEFAULT_PORT = 8321`. Commands save them with `remember(**kw)`, which merges non-`None` keys (`model`, `port`, and `opencode` from setup) and ignores I/O and JSON errors.
4. **Options before or after the subcommand.** `add_common` gives each subparser `--model`/`--port` with `default=argparse.SUPPRESS`. A value given before the subcommand is kept unless it is repeated after it.

`--version` prints `locals <__version__>` (`0.2.0` in `locals/__init__.py`).

## Model names (`--model`)

The `type=model_id` converter in `locals/defaults.py` resolves names as follows:

- Short names and aliases map to repositories through `MODELS` and `ALIASES`:

  | Short name | Aliases | Repository |
  |---|---|---|
  | `qwen3.5-4b` | `qwen`, `qwen3.5` | `mlx-community/Qwen3.5-4B-MLX-4bit` |
  | `qwen3.5-2b` | | `mlx-community/Qwen3.5-2B-MLX-4bit` |
  | `gemma-4-e4b` | `gemma`, `gemma4`, `gemma-4`, `gemma4-e4b`, `gemma-4-e4b-it` | `mlx-community/gemma-4-e4b-it-4bit` |
  | `lfm2.5-2.6b` | `lfm`, `lfm2.5` | `LiquidAI/LFM2.5-2.6B-MLX-4bit` |
  | `minicpm5-2b` | `minicpm`, `minicpm5` | `mlx-community/MiniCPM5-2B-mlx-4Bit` |

- An existing local path (a folder or a file, with `~` expanded) becomes its absolute path. Links are kept, not resolved., because the background server runs from another directory and later commands may run from other folders. A short name takes precedence over a folder with the same name.
- Anything else passes through unchanged as a Hugging Face id.

`DEFAULT_MODEL` is `default_model()`. It picks Qwen3.5-2B when the Mac reports between 0 and 12 GB of memory (`sysctl hw.memsize`), and Qwen3.5-4B otherwise.

`short_name(model)` is the model's name in the API. It is the last path component, lower-cased, with a trailing `-4bit` or `-mlx-4bit` removed. The server routes on the model id of a request:

- `locals/<short>` goes to Locals's engine.
- An id that starts with `mlx/`, `stock/` or `stock-` (any case) goes to stock mlx-lm. Every other id goes to Locals's engine, including `locals/<short>` and a plain Hugging Face id.
- An id containing `think` (any case) turns on the think block, and `--thinking` sets the default. A request's `chat_template_kwargs.enable_thinking` overrides both, and `reasoning_effort: "none"` turns thinking off.

OpenCode references these ids as `<provider>/<engine>/<short>` (`oc_model`), because OpenCode does not send the provider prefix to the server.

## Subcommands

| Command | Handler | What it does |
|---|---|---|
| `locals code [DIR]` | `cmd_code` | Sets up OpenCode if needed, downloads the model, starts or reuses the background server, then `exec`s OpenCode on `DIR`. With `--run TASK`, runs one task headless and prints the answer. |
| `locals launch [TOOL] [ARGS...]` | `cmd_launch` | Starts the server, then opens a tool on it (see the launch-target table below). With no tool, `list`: the tools on this Mac and whether each is connected. |
| `locals disconnect TOOL` | `cmd_disconnect` | Takes Locals out of OpenCode's, OpenClaw's or Hermes's own config (see "Connections" below). |
| `locals models` | `cmd_models` | The tested models: download size, memory needed, whether they fit this Mac, speed with 8 requests at once (measured here by `locals bench`, else on an M2 Pro). |
| `locals setup [opencode\|openclaw\|hermes]` | `cmd_setup` | Writes the tool's config. For `openclaw`/`hermes`, also downloads the model and starts the background server unless `--no-start` is given. |
| `locals serve` | `cmd_serve` | `os.execv` into `python -m locals.server` in the foreground. The server downloads the model itself. |
| `locals start` | `cmd_start` | Downloads the model, then starts the server in the background (`ensure_server`). |
| `locals stop` | `cmd_stop` | Stops the Locals server on `--port`, and only a Locals server. |
| `locals status` | `cmd_status` | Prints `/stats` as a table. |
| `locals run TASK` | `harness.cmd_run` | Locals's own multi-agent harness (plan, parallel versions in copies, check, repair). |
| `locals solve TASK` | `solve.cmd_solve` | OpenCode agents race in copies of the project. The first whose check passes wins. |
| `locals swarm` | `cmd_swarm` | Many OpenCode sessions at once on a task list, through one `opencode serve`. |
| `locals bench` | `cmd_bench` | The speed test on this Mac: Locals and stock mlx-lm with 1, 4 and 8 requests at once, through the HTTP API (`--agents`: tool-calling requests). |
| `locals generate` | `cli.main` | One generation on the background server. |
| `locals chat` | `chat.main` | Interactive chat on the background server. |
| `locals agents` | `agents.main` | In-process agent suite (no HTTP server). |
| `locals tune` | `cmd_tune` | The full one-time kernel autotuning: `Runtime.load(model, tune=mode)`. `--mode full` (default) or `quick`. |
| `locals menubar` | `menubar.main` | The menu-bar app. |

The description in `--help` is the module docstring of `app.py`. argparse adds the usage line and the list of subcommands with their `help=` strings, and that list includes `run`, `launch` and `menubar`.

## Server options (`add_server_opts`)

`serve`, `start`, `code`, `launch`, `run`, `solve` and `swarm` accept these options. The options that take a value default to `None` in `app.py`, which means the server's own default applies. `--thinking`, `--walk` and `--typed-tools` are `store_true` flags and are forwarded only when set. `server_cmd` forwards only the options that were given.

| Option | Server default (`locals/server.py`) | Meaning |
|---|---|---|
| `--engines` | `locals` | Engines the server may load. Add `mlx` for stock mlx-lm. |
| `--preload` | `locals` | Engines loaded at start (`none` for none). |
| `--slots` | `16` | Maximum batch size per engine. |
| `--temp` | `0.2` | Sampling temperature. |
| `--top-p` | `0.0` (off) | Nucleus sampling at temp > 0. |
| `--top-k` | `0` (off) | Top-k sampling at temp > 0. |
| `--max-tokens` | `4096` | Cap on completion tokens per request. |
| `--cache-gb` | sized from free GPU memory | Prompt cache of Locals's engine. |
| `--max-context` | what fits in GPU memory, at least 65536, at most the model's own | Prompt plus reply tokens per request. |
| `--thinking` | off | Think before every reply. |
| `--max-calls` | `6` | Tool calls kept per reply. |
| `--walk` | off | Constraints by the walk kernel (no token tables). |
| `--typed-tools` | off | Tool-call parameter names and types fixed by the grammar (use with `--walk`). |
| `--ollama` | off | Also listen on Ollama's port, `OLLAMA_PORT = 11434`. Forwarded as `--ollama-port`. |

`server_cmd` never passes `--host` or `--seed`, so a server started through `locals` always listens on `127.0.0.1` with seed 0. To use another interface, run `python -m locals.server --host ...` directly.

## The background server

### Starting it: `ensure_server` and `_ensure_server`

`ensure_server(args, wait_s=3600)` takes an exclusive `fcntl.flock` on `~/.locals/server-<port>.lock`. If another `locals` command (a second terminal, the menu bar) is already starting the server, this command prints that it is waiting, blocks, and then finds the server already running. Only one `ensure_server` caller at a time starts a model on a given port. `locals serve` (`os.execv`, no lock) and `python -m locals.server` do not take the lock. They rely on the server's own `_port_taken` check, which runs before loading. It then calls `_ensure_server`:

1. If `--engine mlx` is set and `--engines` is not, `args.engines` becomes `locals,mlx`, because a comparison needs both engines loaded.
2. **If something answers `/health` on the port**, it reads `/stats` (the running `model`) and `/health` (`build`, `engines`):
   - `stale` is true when the server reports a `build` different from `defaults.build_id()`. `build_id` is a SHA-1 over the names, sizes and modification times of `locals/*.py`, except `plan.py` and `solve.py`, which only the client runs. **Editing any other module in a checkout therefore makes a running background server stale.**
   - `same` is true when `/stats` reports no model, when the running model equals the requested one, or when it has the same short name.
   - If `same` and not `stale`, the server is reused. `_SERVER_OPTS` is `engines`, `slots`, `temp`, `top_p`, `top_k`, `max_tokens`, `cache_gb`, `max_calls`, `max_context`. If any of these other than `engines` was given, a note says they only apply on the next start. When a server is reused, `--preload`, `--thinking`, `--walk`, `--typed-tools` and `--ollama` are dropped without a note. If `mlx` was requested but the running server lacks it, the command exits and asks you to stop the server first.
   - Otherwise the command restarts the server only when it is ours and idle. `_background_pid` returns the pid saved in the pid file only if that pid is the one listening (`lsof`), and the server must have no `active` rows. A server that is not ours or is busy is used as-is when `same` (with a note that it runs older code). Otherwise the command exits and names `locals stop` or `--port`. A restart sends `SIGTERM` and waits up to 60 s (120 polls of 0.5 s) for the port to free.
3. If some other process holds the port (`_listener_pid` finds a process whose command line lacks `locals.server`), the command exits before loading anything.
   If a server this tool started is still loading (`_starting_pid`: the saved pid is a running `locals.server` that does not listen yet, for example after a command was stopped with Ctrl-C while it waited), the command waits for it and follows its log instead of starting a second copy of the model, then applies the checks above. `locals serve` refuses to start next to it, and `locals status` reports it as starting. Ctrl-C while waiting stops the waiting, not the server.
4. If `~/.locals/server.log` is larger than 50 MiB (`50 << 20`), it is moved to `server.log.1`.
5. It `Popen`s `server_cmd(args)`, which runs `python -m locals.server ...` with `sys.executable`. Output is appended to the log, the process gets `start_new_session=True`, and the working directory is the folder that contains the `locals` package. The child inherits the environment of this command. The pid is written to `~/.locals/server-<port>.pid`.
6. Once a second, until `wait_s`, it prints new log lines that start with `locals:`, `loading ` or `engine `, or that contain `tuning`. The docstring explains the reason: a first load tunes kernels for minutes, so there is no short timeout while the process is alive. If the process exits, the last 15 log lines written since the start are printed and the command exits. As soon as `/health` answers, it prints `server ready in Ns`.

### Stopping it: `cmd_stop`

`cmd_stop` never kills a process that is not a Locals server, because a pid file can outlive its process and the number can be reused:

- If something is listening on the port, it is stopped only if it answers `/health` with `status: ok` or its `ps` command line contains `locals.server`. Otherwise the command exits with "nothing stopped".
- If nothing is listening, the saved pid is stopped only if it is a Locals server. This is a server still loading its model.
- Pid files (`server-<port>.pid`, and the legacy `server.pid` from earlier 0.2.0 builds) that name the saved pid are removed.
- It sends `SIGTERM` and waits up to 10 s (40 polls of 0.25 s) for the port to free.

### `locals status` and the status page

`cmd_status` reads `/stats` and prints the following:

- model, uptime, request count and active rows;
- per engine: loaded, requests, prompt tokens, cached share, output tokens, forced tokens, tool calls and tok/s;
- per-engine prompt-cache size, entry count, trims and deferrals;
- GPU memory in use and at peak.

Its tok/s is `completion_tokens / decode_s`, and the header says so: output tokens over request wall time, prefill and queueing included.

The server also serves a status page at `GET /` (`STATUS_HTML` in `server.py`) that polls `/stats` every second and lists active rows. The menu bar's "Status page" item opens it, and `locals swarm` points to it.

### Model download: `fetch_model`

`fetch_model(model)` runs before the background start in `start`, `code`, `launch`, `setup openclaw|hermes`, `run` and `solve`, so there a download shows progress in the terminal instead of in the server log. `swarm` and `bench` call `ensure_server` without it. Their first download happens inside the server, and its progress goes to `server.log`. It refuses two kinds of model before downloading gigabytes:

- GGUF models (`defaults.gguf_hint` names MLX versions instead).
- Models whose weights plus `HEADROOM_GB = 2.0` exceed the GPU's recommended working set (`defaults.too_big`, skipped with `LOCALS_IGNORE_MEMORY=1`).

Local paths are used as-is. Hugging Face repositories are fetched with `snapshot_download` and these `allow_patterns`: `*.json`, `model*.safetensors`, `*.py`, `tokenizer.model`, `*.tiktoken`, `tiktoken.model`, `*.txt`, `*.jsonl`, `*.jinja`. A cache-only lookup comes first. Download failures are explained by `server._load_error`.

## Where files live

| Path | Written by | Contents |
|---|---|---|
| `~/.locals/config.json` | `remember` | Last `model`, `port`, `opencode` binary. These are the defaults of later commands. |
| `~/.locals/server-<port>.pid` | `_ensure_server` | Pid of the background server. `server.pid` is the older single-file name, still read. |
| `~/.locals/server-<port>.lock` | `ensure_server` | `flock` that serializes server starts per port. |
| `~/.locals/server.log`, `server.log.1` | `_ensure_server` | Output of background servers. One file for all ports. Rotated once past 50 MiB. |
| `~/.locals/backups/<tool>-<file>.before-locals` | `_back_up_once` | First copy of OpenClaw's `openclaw.json` / Hermes's `config.yaml` before Locals changed it. Never overwritten. The undo command is printed. |
| `~/.locals/openclaw-locals.json5` | `setup_openclaw` | The config fragment patched into OpenClaw. |
| `~/.locals/cline/` | `cmd_launch` | Cline config folder used with `--config`. `~/.cline` is left alone. |
| `~/.locals/kilo.json` | `cmd_launch` | Kilo Code config, passed by `KILO_CONFIG`. |
| `~/.locals/bench.json` | `cmd_bench` | The speeds `locals bench` measured, by model id, with the chip and the date. Read by `locals models`. |
| `~/.locals/solve/<project>-<stamp>/` | `solve.cmd_solve` | Runs of `locals solve`. |
| `~/.locals/runs/` | `harness.cmd_run` | `keep_dir` of `locals run`. |
| `~/.config/opencode/opencode.json` | `write_config`, `cmd_code` | OpenCode's config. A one-time copy is kept beside it as `opencode.json.before-locals`. |
| `~/.opencode/bin/opencode` | `install_opencode` | OpenCode, if Locals installed it. |
| `~/.local/bin/locals` | `cmd_setup` | Symlink created only for a development checkout when `locals` is not on the `PATH`. |
| `~/.cache/locals/` | engine and server | Kernel tuning (`tune_v6_<arch>.json`), system-prompt disk cache (`prompts/`), request dumps (`requests/`). See the environment variables below. |
| `./swarm_runs/<stamp>/` | `cmd_swarm` | `task_NN.json`, `summary.json`, `opencode-serve.log`. Location set with `--out`. |
| `./agent_runs/<stamp>-<engine>/` | `agents.run` | Agent sandboxes `agent_NN/`, `summary.json`, `transcripts.json`. Location set with `--out`. |

Writes to OpenCode's `opencode.json` (`write_config`, `cmd_code`) go through `_write_atomic`, which writes a temporary file in the same folder and then `os.replace`s it. OpenClaw's and Hermes's configs are changed by their own CLIs (`openclaw config patch`, `hermes config set`). Locals's own files (`config.json`, `kilo.json`, `openclaw-locals.json5`) are written with plain `write_text`. It follows a symlinked config to its target and keeps the file mode, so a crash never leaves half a config file.

## OpenCode

### Installing it

`opencode_bin()` looks on the `PATH`, then in `~/.opencode/bin/opencode`. If OpenCode is missing, or `--reinstall` is given, `install_opencode` asks `[Y/n]` and pipes OpenCode's official installer (`https://opencode.ai/install`, downloaded with a 60 s timeout) to `bash -s -- --no-modify-path --version <v>`. The default version is `OPENCODE_VERSION = "1.18.32"`, the version this integration was built and tested against, and `--opencode-version` overrides it. Without a TTY, the command only prints the install command and exits.

### The config it writes

`setup_opencode` chooses four values and calls `write_config(opencode_config(...))`:

- **Default agent:** `--agent`, else `default_agent(model)`. That function returns `orchestrator` when the chat format is `qwen_xml` (Qwen3.5) and `coder` otherwise. It also returns `orchestrator` when the tokenizer cannot be loaded or its format detected. The docstring gives the reason: Qwen3.5 delegated sensibly (11 of 13 real tasks), while the other models tested called the subagent tool over and over with junk arguments.
- **Context declared:** `--context`, else `min(32768, agent_context(model))`.
- **Engine:** `--engine` (`locals` or `mlx`), default `locals`.
- **Thinking variants:** `model_thinks(model)` is true when the chat template offers thinking (Qwen3, Qwen3.5, Gemma 4). It is also true when the tokenizer cannot be loaded, because the check falls back to `True`.

`opencode_config` produces the following:

- `provider.locals` is always written. `provider.mlx` is written only when the engine is `mlx`, because the comment says the stock engine is for comparisons and confuses everyday use. Each provider uses `npm: "@ai-sdk/openai-compatible"`, `baseURL: http://127.0.0.1:<port>/v1` and `apiKey: "local"`. Each lists `<engine>/<short>` and, with thinking, `<engine>/<short>-thinking`, with `tool_call: true`, `reasoning` set on the thinking variant, and `limit: {context, output}` (output default 4096).
- `model` and `small_model` are `<engine>/<engine>/<short>`, so Locals becomes OpenCode's default model.
- `default_agent` is the chosen agent, and `subagent_depth` is `1`.
- `agent` holds the team below, plus `build` and `plan` with `steps: AGENT_STEPS`.

`write_config` merges this into an existing file instead of replacing it:

- Top-level keys are replaced.
- `provider` and `agent` are merged per entry.
- Your own settings of OpenCode's `build` and `plan` agents are merged key by key.

It refuses a file that is not plain JSON (comments or trailing commas, which OpenCode's JSONC allows) or that is not a JSON object, because rewriting it would lose them.

`_opencode_ready(model, port)` decides whether `locals code` and `locals solve` need to run setup. It checks that OpenCode is installed, that the config's `model` ends with `/<short>`, and that `provider.locals.options.baseURL` matches the port.

### The agent team (`AGENTS`)

Permissions double as the tool list: a denied tool is not offered to the model. The comment says a 2B model does best with few tools and a short, procedural prompt. Every agent denies `skill`, `todowrite`, `webfetch`, `doom_loop`, `question` and `external_directory` (`_NO`).

| Agent | Mode | edit | bash | task (subagents) |
|---|---|---|---|---|
| `orchestrator` | primary | allow | allow | allow |
| `coder` | all | allow | allow | deny |
| `researcher` | all | deny | deny | deny |
| `tester` | all | deny | allow | deny |
| `reviewer` | all | deny | only `git diff*`, `git log*`, `git status*` | deny |

Every agent, plus `build` and `plan`, gets `steps = AGENT_STEPS = 40`. The comment explains the number: in Locals's OpenCode runs, every task where one agent turn took 30 or more steps failed (up to 218 steps), and no passing task took that many. After this many steps OpenCode asks the agent for a text answer.

### `locals code`

`cmd_code` runs these steps in order:

1. Checks that `DIR` exists.
2. Calls `fetch_model` first, so a model that cannot run (a typo, a GGUF, too big) is refused before OpenCode's config names it.
3. Runs setup if `_opencode_ready` is false, or if `--engine mlx` is given and the config has no `mlx` provider.
4. Calls `ensure_server`, then `remember` (a model is remembered only once its server started).
5. Writes `--agent` into the OpenCode config, so it persists for later launches. `--engine` is for this launch only.
6. Runs OpenCode with `OPENCODE_DISABLE_AUTOUPDATE=1` and `PWD=<dir>` (OpenCode reads its project directory from `$PWD`), and always with `-m <provider>/<engine>/<short>`: a model set in the project's own `opencode.json` or a global `.jsonc` would otherwise win over the one Locals wrote, and could send the project to a cloud model.

Without `--run`, it `execve`s `opencode <dir> -m ...` (`--logs` adds `--print-logs`). With `--run TASK`, it runs `opencode run --format json -m ... [-c] TASK` (`-c` comes from `--continue`) with stdin closed (`opencode run` reads a piped stdin into the task, and waited forever when Locals ran from a script). Each tool call is printed as one line as it happens (`_opencode_step`, for example `bash: python3 -m pytest -q`); then the last text part found by `_opencode_answer` (or the raw output tail) is printed, and the command exits with OpenCode's return code. Ctrl-C stops OpenCode.

### `locals swarm` and `OpenCodeServer`

`OpenCodeServer` starts one `opencode serve --port <free port>` and drives it over HTTP. Every agent is a session inside the one process, because separate `opencode run` processes contend for OpenCode's SQLite database.

- **Startup:** it waits up to 90 s for `/global/health`.
- **`run_task`:** creates a session, posts the task, and aborts it after `timeout` seconds with a timer. It returns the transcript.
- **`start_approver`:** headless mode. It listens on `/global/event` and answers `permission.asked` / `permission.v2.asked` with `once`. It answers `reject` when a property value is `external_directory` or `doom_loop`, and rejects all questions.

`cmd_swarm` takes tasks from `--tasks FILE` (a JSON list of strings or `{"task": ...}` objects, else one task per line with `#` comments skipped) or from `--task TEXT` repeated `--n` times (default 4). Other options:

- `--workspaces`: each task gets `DIR/swarm/task_NN`. Without it, the agents share `DIR` and a note warns that they may edit the same files.
- `--agent`: default `coder`.
- `--engine`: default `locals`.
- `--auto/--no-auto`: default on, starts the approver.
- `--timeout`: default 900 s per task.
- `--out`: default `swarm_runs`.

`swarm` requires OpenCode to be installed already ("run locals setup first"). While it runs, it prints a live line from `/stats`.

## Other agent tools

### Launch targets

`locals launch` first makes sure the tool exists. The tools in `INSTALLERS` are offered an install, and the ones in `TOOL_SITES` must already be installed. It then runs `fetch_model` and `ensure_server` with stdout redirected to stderr, so the tool's own output stays clean. Locals's own options go before the tool name. Everything after the name is passed to the tool (`argparse.REMAINDER`).

| Target | What Locals does |
|---|---|
| `list` (default) | `print_connections`: each tool, its connection state, and what to run (see "Connections" below). |
| `opencode` | Same as `locals code [DIR]`. `DIR` is the first argument after the name. |
| `aider` | `exec aider --model openai/locals/<short> --no-show-model-warnings ...` with `OPENAI_API_BASE=<base>/v1`, `OPENAI_API_KEY=locals`. |
| `cline` | `cline auth --config ~/.locals/cline -p openai -k locals -m locals/<short> -b <base>/v1`, then `exec cline --config ~/.locals/cline ...`. |
| `kilo` | Writes `~/.locals/kilo.json` (provider `locals`, model `locals/locals/<short>`, context `agent_context(model)`, output 4096), then `exec`s Kilo with `KILO_CONFIG` pointing to it. |
| `openhands` | `exec openhands --override-with-envs ...` with `LLM_MODEL=openai/locals/<short>`, `LLM_BASE_URL=<base>/v1`, `LLM_API_KEY=locals`, `OPENHANDS_SUPPRESS_BANNER=1`. |
| `openclaw`, `hermes` | Runs `setup_openclaw` / `setup_hermes` (below), then `exec`s the tool. |
| `ollama` | Makes the server also answer on port 11434. It exits if that port is held by something that is not Locals (Ollama itself, for example). It restarts an idle running server that lacks the Ollama port, and refuses while that server has active requests. It then prints the address and the model name `<short>:latest`. |
| `editor` | Prints the settings for an OpenAI-compatible provider: base URL, model `locals/<short>`, any API key, and the context window. The context comes from the running server's `/v1/models` (`context_length`), falling back to 65536. It also prints snippets for Continue, Cline/Roo/Kilo Code, Zed and JetBrains AI Assistant. |

For Aider, OpenHands, Cline and Kilo Code, Locals configures the launch only, through environment variables or a config file of its own. OpenCode, OpenClaw and Hermes get Locals written into their own config.

### Connections (`locals launch` with no tool, `locals disconnect`)

`connection(tool, port)` gives each tool one of these states, and `print_connections` lists them with the
server's address:

| State | When |
|---|---|
| `not installed` | The tool's command is not found (`opencode_bin`, `_installed`). The detail says how to install it. |
| `ready` | Aider, OpenHands, Cline and Kilo Code: Locals gives them its settings each time `locals launch` opens them, so there is nothing to keep connected. |
| `not connected` | OpenCode, OpenClaw or Hermes without a Locals provider in its config. |
| `connected` | The config's Locals provider points at this server's `<base>/v1`. The detail names the default model, or says when the tool's default is another model. |
| `needs repair` | The provider points at another address (another port), the config cannot be read (OpenCode's JSONC, a broken YAML), or the tool still uses Locals under its old name `stcon` (`LEGACY`). The detail names the command that fixes it (`locals code`, `locals launch <tool>`). |

`_tool_settings` reads each config directly: `opencode.json` (`provider.locals.options.baseURL`, `model`),
`$OPENCLAW_STATE_DIR/openclaw.json` (`models.providers.locals.baseUrl`, `agents.defaults.model.primary`) and
`$HERMES_HOME/config.yaml` (`model.base_url`, `model.default`). A setup that rewrites OpenCode's config
(`write_config`) also drops the old `stcon` provider, which the server no longer answers to.

`locals disconnect TOOL` (`cmd_disconnect`) takes Locals out again and leaves the rest of the config alone:

- **OpenCode** (`disconnect_opencode`): removes the `locals`, `mlx` and `stcon` providers and the agents
  Locals added, and puts `model`, `small_model`, `default_agent`, `subagent_depth`, `$schema` and the build
  and plan agents' step limit back as they were in `opencode.json.before-locals`, or removes them when
  OpenCode had none. A config Locals wrote from scratch ends up as `{}`.
- **OpenClaw** (`disconnect_openclaw`): `openclaw config unset models.providers.locals` (and `stcon`), then
  sets `agents.defaults.model.primary` back to the backup's value with `openclaw config set`, or unsets it.
- **Hermes** (`disconnect_hermes`): for each key `hermes_settings` sets, `hermes config set` with the
  backup's value, or `hermes config unset`.

### Finding and installing tools

`_tool_bin(name, env)` returns `$<NAME>_BIN` if it is set, else the tool from the `PATH`. `_installed` also looks in `uv tool dir --bin`, which may not be on the `PATH` yet. `_offer_install` asks `[Y/n]` and runs one of these:

| Tool | Command |
|---|---|
| `openhands` | `uv tool install --python 3.12 openhands` |
| `aider` | `uv tool install aider-chat` |
| `cline` | `npm install -g cline` |
| `kilo` | `npm install -g @kilocode/cli` |

Without a TTY, or without uv or npm, it prints what to run and exits.

### OpenClaw and Hermes Agent (`locals setup openclaw|hermes`)

Both tools speak OpenAI chat completions to a custom base URL. The config is written through the tool's own CLI when the tool is installed, so the format and its validation stay the tool's business. Otherwise the snippet is saved or printed.

`AGENT_CONTEXT = 65536` is the context declared to these tools. The comment says Hermes refuses models under 64,000 tokens, and 65536 is also the server's default minimum `--max-context`. `agent_context(model)` lowers it to the model's own `max_position_embeddings` when that is smaller.

- **OpenClaw** (`openclaw_patch`, `setup_openclaw`): writes `~/.locals/openclaw-locals.json5` containing:
  - `models.mode: "merge"`;
  - a `locals` provider with `baseUrl <base>/v1`, `apiKey "locals-local"`, `api "openai-completions"`, `timeoutSeconds 900`;
  - one model `locals/<short>` with zero cost, `contextWindow` and `maxTokens`;
  - `agents.defaults.model.primary: "locals/locals/<short>"`.

  It backs up `$OPENCLAW_STATE_DIR/openclaw.json` (default `~/.openclaw`) once, then runs `openclaw config patch --file ... --replace-path models.providers.locals.models`. The comment explains `--replace-path`: OpenClaw refuses a patch that drops entries, and a setup for another model must replace the list.
- **Hermes** (`hermes_settings`, `setup_hermes`): backs up `$HERMES_HOME/config.yaml` (default `~/.hermes`) once, then runs `hermes config set` for each of these keys:
  - `model.provider custom`
  - `model.base_url <base>/v1`
  - `model.default locals/<short>`
  - `model.api_key locals-local`
  - `model.context_length <ctx>`

  It warns when the context is under 64000.

If the tool is not installed, both functions print how to finish by hand and return `False`, and `cmd_setup` then starts nothing. Otherwise `cmd_setup` remembers the model and port. Unless `--no-start` is given, it downloads the model and starts the background server.

`locals setup` with no target (OpenCode) also creates the `~/.local/bin/locals` symlink for a development checkout, as listed in the files table.

## `locals bench`

`cmd_bench` measures the README's speed test on this Mac. It downloads the model if needed and starts or
reuses the background server; `main` sets the server's `--engines` to `locals,mlx` so it can load both.
For each engine in `--engines` (default `locals,mlx`) it runs `bench.sweep`: one short warm-up request,
then `--sizes` (default `1,4,8`) requests at once. Each request is one of sixteen short questions
(`bench.QUESTIONS`), streamed, with `--max-tokens` (default 256) tokens and `ignore_eos`, so every request
writes the same number of tokens.

`bench.measure` counts each request's tokens (from the usage chunk, else the streamed pieces) and the times
of its first and last token. The speed is all the tokens over the time from the first token of any request
to the last token of any, so reading the prompts at the start is left out, and a server that answers
requests one after another is measured as such (adding up each request's own rate would count it as fast
as one that runs them together). It works with any OpenAI-compatible server; for Ollama, thinking is
turned off with `"reasoning_effort": "none"`.

It prints a table with both engines and the ratio, and saves the speeds by model id in
`~/.locals/bench.json` with the chip, the memory and the date. `locals models` shows them.

`--agents` runs the older test instead (`_bench_agents`): `--n` (default 8) tool-calling requests at once,
`--rounds` (default 2) rounds, three tools (`BENCH_TOOLS`) and tasks cycled from `BENCH_TASKS`; it prints
output tokens over wall time per round, the median per engine and the `locals/mlx` ratio.

## `locals run` and `locals solve` options

These commands are parsed in `app.py`. Both also accept the server options above.

| `locals run TASK` | Default | |
|---|---|---|
| `--dir` | `.` | The project. Agents work in copies, and the project is written only when the check passes. |
| `--check` | the project's tests | When auto-detected, a ` -x` is removed so that every failure is reported. |
| `--parts` | 8 | Maximum number of parts. |
| `--workers` | 0 | `1` means one agent does the whole task, as a baseline. |
| `--rounds` | 2 | Repair rounds after the check. |
| `--steps` | 10 | Tool steps per agent. |
| `--race` | 2 | Whole-task agents racing the parts. |
| `--head` | 0 | Seconds the whole-task agents run alone first. |
| `--versions` | 3 | Agents per code part. |
| `--engine` | `locals` | |
| `--report` | none | JSON report path. |

| `locals solve TASK` | Default | |
|---|---|---|
| `--dir` | `.` | |
| `--check` | the project's tests | Without tests, an agent writes a check first. |
| `--start` | 1 | Agents at the start. |
| `--n` | 1 | Agents at once, at most. |
| `--burst` | 1 | Fresh agents per failure. |
| `--attempts` | 5 | |
| `--hedge` | 0 | Seconds without a result before one more agent starts. |
| `--fixes` | 1 | |
| `--agent` | `coder` | |
| `--timeout` | 600 | Seconds per attempt. |
| `--keep` | off | Keep the attempts' workspaces. |
| `--no-apply` | off | Save the winner as a diff instead of applying it. |
| `--plan`, `--plan-agents` | off | |
| `--parts` | 6 | With `--plan`. |
| `--samples` | 0 | With `--plan`: 3 versions for up to 4 parts, 2 for up to 8. |
| `--engine` | `locals` | |
| `--no-write-check` | off | |

## `locals generate` (`cli.py`)

This command runs on the background server: `client.connect` starts it if it is not running (as `locals code` does) or reuses it, with the model given by `--model`, else the model used last, else this Mac's default. One copy of the model stays in memory, and `--n` sends N requests that decode together in the server's batch, with or without a constraint. The prompt file and the schema are read and checked before anything starts. `--local`, `--dtype`, `--no-fuse` and `--no-forced` load the model in this process with `Runtime.load` instead (stop the server first: two copies of a model do not fit next to each other on a small Mac).

| Option | Default | |
|---|---|---|
| `--prompt` | required | `@file` reads the prompt from a file. |
| `--schema` | none | JSON schema file. |
| `--regex` | none | Regex constraint instead of a schema. |
| `--max-tokens` | 512 | |
| `--temp` | 0.0 | |
| `--seed` | 0 | |
| `--n` | 1 | More than 1 means N samples at once (temperature 0.8 when `--temp` is 0), printed one after another. |
| `--no-forced` | off | Disables forced-span compression. |
| `--no-fuse` | off | Keeps separate q/k/v and gate/up matmuls. |
| `--stats` | off | Speed line on stderr. |
| `--dtype` | `auto` | `auto` means float16 on M1/M2 GPUs. Implies `--local`. |
| `--local` | off | The model in this process instead of the server. |

## `locals chat` (`chat.py`)

This command runs on the background server (`client.chat`), started if needed, like `locals generate`. Its options are `--model` (default: the model used last), `--system`, `--temp 0.7`, `--max-tokens 1024`, `--think` (the `-thinking` model id; thoughts are printed dimmed), `--no-stats`, and `--local` (or `--dtype`) for the old in-process chat. Replies are streamed; the server's prompt cache keeps the conversation between turns, so a turn reads only what is new. The stats line gives the reply's tokens and speed, the time to the first token, and how much of the context came from the cache.

In-process (`--local`), `Conversation` keeps a ChatML conversation as one token stream whose KV cache persists, warms up with an 8-token generation, and takes `--top-p`.

| In-chat command | Effect |
|---|---|
| `/reset` | Starts a new conversation. |
| `/json <schema>` | Constrains the next reply only to a schema, given as a file path or inline JSON. |
| `/n <N>` | Generates N alternative replies to the next message at once (N requests in the server's batch; in-process, mlx-lm's `BatchGenerator`). The conversation continues from reply 1. |
| `/stats` | Toggles the speed line. |
| `/quit` | Exits. |

Ctrl-C stops the current reply (on the server: the connection closes and the server stops decoding it) and drops that turn.

## `locals agents` (`agents.py`)

This command runs many agents in threads, each with its own sandbox directory and conversation, against one in-process `BatchService`. No HTTP server is involved.

Each reply must be exactly one tool call, enforced by constrained decoding: `ACTION_REGEX` in Qwen3.5's XML format, or `fmt.calls(..., many=False, at_least_one=True)` for other formats (`ModelFormat`). The comment explains why: a 2B model otherwise writes the whole solution as prose.

The tools are `list_files`, `read_file`, `write_file`, `run_file`, `run_python` and `finish`:

- Paths are kept inside the sandbox. An absolute path is reduced to its file name.
- Code runs as `python -I` through `solve.run_shell` with a 10 s timeout, and output is truncated to 1500 characters.
- A step that exactly repeats the previous one gets a note in the tool result telling the model to change its approach.

Before the agents start, the common prefix of all first prompts (when longer than 32 tokens) is prefilled once with `svc.prime`.

| Option | Default | |
|---|---|---|
| `--engine` | `locals` | `stock` is mlx-lm plus llguidance. `both` runs each engine in a fresh subprocess and prints a comparison table. |
| `--agents` | 8 | |
| `--task` | none | Without it, the agents cycle through `SUITE`: 8 tasks with checks. |
| `--max-steps` | 8 | |
| `--max-tokens` | 700 | Per model call. |
| `--slots` | 16 | |
| `--temp` | 0.3 | |
| `--seed` | 0 | |
| `--out` | `agent_runs` | |
| `--no-exec` | off | |
| `--quiet` | off | Turns off the live `rich` dashboard. |
| `--summary-json` | none | `-` prints the summary to stdout. |

`run_python` and `run_file` execute model-written code as the current user. The module docstring states that this is not a security sandbox.

## `locals menubar` (`menubar.py`)

This command needs `rumps` (`uv tool install 'locals[menubar]'`). It reads the port from `~/.locals/config.json` and polls `/health` and `/stats` every 2 s. The title shows `locals ○` (stopped), `locals ◐` (starting or stopping) or `locals ●` (running), and the menu shows the model and the number of running requests.

The menu has three actions:

- **Start or stop the server:** runs `python -m locals.app start|stop` in a thread, the same commands as the terminal. Failures appear in an alert on the next refresh, because AppKit is not thread-safe.
- **Open OpenCode on a folder:** an AppleScript folder picker, then Terminal with `cd <folder> && locals code .`.
- **Status page:** opens the server's status page in the browser.

## Environment variables

### Read by Locals

Variables read by the server process must be set in the environment of the command that starts it. A background server inherits the environment of the `locals` command that started it, and a running server is reused, so run `locals stop` before changing them.

| Variable | Default | Effect | Where |
|---|---|---|---|
| `LOCALS_IGNORE_MEMORY` | unset | Skips the check that the weights plus 2 GB fit in GPU memory. | `defaults.py:74` |
| `LOCALS_TUNE` | `auto` (`Runtime`, `lm.patch`); `background` (`BatchService`, which both the server and `locals agents` use) | Kernel autotuning mode: `auto`, `background`, `quick`, `full` or `off`. `background` serves at once on MLX's kernels and tunes while idle. | `runtime.py:144`, `lm.py:64`, `service.py:280`, `kernels.py:848-861` |
| `LOCALS_TUNE_CACHE` | `~/.cache/locals/tune_v6_<arch>.json` | Path of the tuning results file. | `kernels.py:134` |
| `LOCALS_NO_PRESET` | unset | Hides the shipped tuning presets, so the first start behaves as on a GPU without them. | `kernels.py:148` |
| `LOCALS_MAX_TABLE_GB` | `3` | Largest token table a grammar may need. Above it, the request fails and the error suggests `--walk`. | `constraint.py:128` |
| `LOCALS_TABLE_CACHE_GB` | `0.5` | LRU limit on per-state constraint tables. | `engine.py:345` |
| `LOCALS_MAX_COMPILED` | `16` | Compiled constraints kept, least recently used first. | `runtime.py:162` |
| `LOCALS_COMPILED_GB` | `1.5` | Limit, in units of 1e9 bytes, on the tables of kept compiled constraints. | `runtime.py:245` |
| `LOCALS_MIX_CHUNK` | `1024` | Prompt tokens per mixed prefill and decode pass. `0` prefills a new prompt while the other rows wait. | `hybrid_batch.py:333` |
| `LOCALS_MAX_SNAPSHOTS` | `4` | Shared-prefix snapshots kept (at least 1). | `hybrid_batch.py:347` |
| `LOCALS_GROWTH_TRIM` | `1` | `0` turns off making room in the prompt cache every 32 decode steps for the rows' KV growth. | `hybrid_batch.py:181,1220` |
| `LOCALS_MEMLOG` | unset | Prints GPU memory around each admission. | `hybrid_batch.py:34` |
| `LOCALS_MIXLOG` | unset | Prints each mixed pass. | `hybrid_batch.py:35` |
| `LOCALS_PREFILL_LOG` | unset | Prints prefill and checkpoint lines. | `hybrid_batch.py:567,803,1176` |
| `LOCALS_REPLY_CHECKPOINT` | `1` | `0` turns off the "reply" prompt-cache checkpoint for follow-up turns. | `server.py:603` |
| `LOCALS_DISK_CACHE` | `1` | `0` turns off the system-prompt snapshots on disk, which survive server restarts. | `service.py:296` |
| `LOCALS_DISK_CACHE_DIR` | `~/.cache/locals/prompts` | Where those snapshots go. | `service.py:303` |
| `LOCALS_DISK_CACHE_GB` | `4` | Their size limit. | `service.py:305` |
| `LOCALS_ORIGINS` | empty | Extra browser origins allowed to call the server (comma-separated, `*` for any). | `server.py:86` |
| `LOCALS_HOSTS` | empty | Extra `Host` header names accepted, against DNS rebinding (comma-separated, `*`). | `server.py:110` |
| `LOCALS_DEBUG` | unset | Saves every request as received. | `server.py:295,378,1132` |
| `LOCALS_DEBUG_DIR` | `~/.cache/locals/requests` | Where those dumps go. The folder is mode 0700 and the files 0600. | `server.py:129` |
| `LOCALS_NO_SANDBOX` | unset | `solve.run_shell` does not confine commands with `sandbox-exec`. | `solve.py:242` |
| `OPENCLAW_BIN`, `HERMES_BIN`, `AIDER_BIN`, `CLINE_BIN`, `KILO_BIN`, `OPENHANDS_BIN` | `PATH` lookup | Path of that tool's command. | `app.py:330-331` (`_tool_bin`) |
| `OPENCLAW_STATE_DIR` | `~/.openclaw` | Where OpenClaw's `openclaw.json` is backed up from. | `app.py:354` |
| `HERMES_HOME` | `~/.hermes` | Where Hermes's `config.yaml` is backed up from. | `app.py:383` |
| `LOCALS_SOURCE` | `locals @ https://github.com/Generated-Computers/Locals/archive/refs/heads/main.tar.gz` | What `install.sh` installs. | `install.sh:37` |

### Set by Locals for the programs it starts

| Variable | Value | For |
|---|---|---|
| `OPENCODE_DISABLE_AUTOUPDATE`, `PWD` | `1`, the project folder | OpenCode in `code`, `swarm`, `solve` (`OpenCodeServer`) |
| `OPENAI_API_BASE`, `OPENAI_API_KEY` | `<base>/v1`, `locals` | Aider |
| `KILO_CONFIG` | `~/.locals/kilo.json` | Kilo Code |
| `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`, `OPENHANDS_SUPPRESS_BANNER` | `openai/locals/<short>`, `<base>/v1`, `locals`, `1` | OpenHands |
| `PYTHONDONTWRITEBYTECODE` | `1` | Commands run by `solve.run_shell` |

## Rough edges found while reading

- **Stale check in the agent suite.** The `reverse` task in `agents.SUITE` asks for `'locals agents'` reversed (`stnega slacol`) but checks for `"stnega nocts"`, left over from the rename. That task's check always fails.
- **Misleading status-page label.** The status page labels its last column "tok/s (decode)", but `decode_s` accumulates `t_done - t_submit` per request, which is wall time including queueing and prefill. `locals status` labels the same number correctly.
- **Delegated commands ignore saved defaults.** `generate`, `chat`, `agents` and `tune` do not read `~/.locals/config.json`. They use `DEFAULT_MODEL` unless `--model` is given. `menubar` ignores `--port` and uses the saved port.
- **Incomplete `--no-exec` help.** `locals agents --no-exec` is documented as "disable run_python", but it also disables `run_file`.
- **`/json` with `/n` in chat.** In `locals chat`, a pending `/json` schema is not applied to a `/n` turn and stays pending for the turn after.
- **Outdated `--help` text.** The `app.py` docstring at the top of `locals --help` does not describe `run`, `launch` or `menubar`. They appear only in argparse's generated list of subcommands. It also calls Qwen3.5-4B "the default", although below 12 GB of memory the default is Qwen3.5-2B.
- **Shared server log.** All ports share one `~/.locals/server.log`.

## Tests

`tests/test_app.py` covers this area without model weights, among other things:

- the OpenCode, OpenClaw and Hermes config shapes;
- keeping the user's OpenCode config, and refusing to rewrite a commented one;
- `stop` with no server, and `stop` with a stale pid file;
- restarting only an idle background server, and restarting one that runs older code;
- `--model` placed before a delegated command;
- an end-to-end `code --run` with stand-ins.

`tests/test_agents.py` covers the action regex and the sandbox. Tests marked `model` need weights cached locally; run `pytest -m 'not model'` for the fast suite.
