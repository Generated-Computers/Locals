# The end-to-end suite

These tests run Locals the way people use it: the `locals` command starts the server, real agent tools do real tasks on small projects, and a recording proxy between each tool and the server notes what the tool experiences. That covers how long it waits for the first token, the longest pause in a reply, errors inside streams, and hang-ups.

They take minutes per test, need a model on the GPU, and are skipped unless you ask for them:

```bash
LOCALS_E2E=1 python -m pytest tests/e2e -v
```

Each module starts its own server through `locals start`, in a HOME of its own, so one model is on the GPU at a time. Nothing in your home folder is read or written, with two exceptions: the Hugging Face model cache (models are not downloaded twice) and the kernel tuning file (read only). Every run writes `report.md` and `report.json` into the run folder: one row per agent run, plus notes from the scenarios.

## What it covers

| File | What a user does |
|---|---|
| `test_tools.py` | Each agent tool on real tasks, one at a time: OpenCode (`locals code --run`), Hermes, OpenClaw, Kilo Code, Cline, OpenHands, Aider and Claude Code (`locals launch TOOL`; Claude Code through Anthropic's Messages API, its settings in the test's HOME). The tasks are fixing a bug so the tests pass, renaming a method across files, adding a command with its test, and a question about a bigger code base. |
| `test_many_agents.py` | Several agents at once: two OpenCode terminals beside two other tools, then six OpenCode terminals in one HOME, measured against one alone. |
| `test_session.py` | A working session: five follow-up turns in one OpenCode session (`--continue`). Each follow-up's first request must come mostly from the prompt cache. |
| `test_interrupts.py` | Esc or Ctrl-C: streams cut off mid-reply, a hang-up while a 20k-token prompt is read, and Ctrl-C on a writing agent. The engine must stop those replies, and the next request must not wait for them. |
| `test_clients.py` | The OpenAI Python SDK (streams, tool calls, JSON schemas, errors, eight async streams), Ollama's HTTP API, and the Anthropic SDK (a tool-use round trip, streamed, from the prompt cache). |
| `test_lifecycle.py` | `locals status`, a restart (OpenCode's first request then comes from the prompt cache on disk), a port held by another program, malformed requests, a start with no network, and switching models with one model in memory. |
| `test_first_run.py` | A new user: an empty HOME, `install.sh` with a wheel built from this checkout, then the first `locals code PROJECT --run TASK` in a pseudo-terminal, answering yes when locals offers to install OpenCode. |

A tool must finish without a crash or a timeout, and every request it sends must be served without a server fault. Whether the model then does the task is reported rather than required one by one, because a small model fails some tasks on its own. `test_tasks_done` asks for at least half of them.

An agent that changes files outside its copy of the project fails the run, and the files are put back.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `LOCALS_E2E_MODEL` | `qwen` | the model |
| `LOCALS_E2E_HOME` | a new, empty HOME | a HOME where the agent tools are installed and set up (tools that need their own onboarding skip without one) |
| `LOCALS_E2E_PATH` | | folders with the tools' commands, put first on the PATH (colon separated) |
| `LOCALS_E2E_DIR` | a new temporary folder | where runs, logs and the report go |
| `LOCALS_E2E_PORT` | `8341` | the proxy's port; the server listens on the next one, and the new-user test on 8343 |
| `LOCALS_E2E_SERVER_ENV` | | extra server settings, e.g. `LOCALS_SOLO=0,LOCALS_MIX_CHUNK=512` |
| `LOCALS_E2E_SERVER_ARGS` | | extra options for `locals start`, e.g. `--temp 1.0 --top-p 0.95 --top-k 64` |
| `LOCALS_E2E_SWITCH` | `gemma` | the model the switching test switches to (empty: skipped) |
| `LOCALS_E2E_<TOOL>_BIN` | | a tool's command, if it is not on the PATH |
