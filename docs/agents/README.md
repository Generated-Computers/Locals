# `locals run`: the multi-agent harness

Part of the [technical documentation](../README.md) of Locals. File and function names refer to the `locals` package.

`locals run "TASK"` splits one coding task into parts and gives each part to several tool-using agents at once. After every file an agent writes, it runs the project's check. It then combines the best version of each part and sends fresh agents to repair the parts the check still blames. Each agent is an ordinary OpenAI-style chat request with tools, sent to the locals server, so all the agents decode as rows of one batch.

The control flow is Python in `locals/harness.py`: plan clean-up, scheduling, file ownership, the check, picking and write-back. The model does two things only: it writes the plan when the task names no files, and it does the parts.

```bash
locals run "write inventory.py, cli.py and test_inventory.py: ..." --dir PROJECT --check "python3 -m pytest -q"
```

## Where the code is

| File | Role in `locals run` |
|---|---|
| `locals/app.py` | The `run` sub-command and its flags, which call `harness.cmd_run`. Also `ensure_server`, `fetch_model`, `remember` and `LOCALS_DIR` (`~/.locals`). |
| `locals/harness.py` | `cmd_run` (command-line entry, status line, summary), `Run` (one run: plan, stages, pick, repairs, write-back), `Agent` (one tool loop), `TOOLS`, `SYSTEM`, and the prompt builders `worker_prompt` and `fixer_prompt`. |
| `locals/plan.py` | Planning: `make_plan`, `normalize`, `writable`, `cover`, `protect_tests`, `tests_after_code`, `asks_test_changes`, `project_digest`. Picking: `_Checks`, `_check_all`, `_search`, `_best`, `_score`, `_key`, `_blame`, `_merge`, `_put`. The rest of the file is `locals solve --plan` (`solve_with_plan`), which `locals run` does not use. |
| `locals/solve.py` | Shell and check helpers: `run_shell`, `write_sandbox`, `run_check`, `detect_check`, `manifest`, `SKIP_DIRS`. The rest is `locals solve` (`Race`, which runs OpenCode sessions), which `locals run` does not use. |
| `tests/test_harness.py` | Tests the harness with scripted model replies (`harness._chat` is monkeypatched), so no model is needed. |

## One run, step by step

`Run.go()` runs these steps in order:

1. **Staging copy.** Makes a staging copy of the project (`_copy("stage", self.project)`) and records the project's files with `manifest` (a sha1 for each file).
2. **Plan.** `Run.plan` produces a list of parts. Each part has `title`, `files`, `instructions` and `needs`.
3. **Racers.** Starts the whole-task racers `s0`, `s1`, ... (`_start_solo`). This happens only when there is a check, the plan has more than one part, `--race` is above 0 and `--workers` is not 1.
4. **Stage 1, code.** The code parts (empty `needs`) get `k` agents each (capped by `room`, see below), named `w{i}.{v}`. Then, with a check, `_pick` stages the best combination. Without a check, each part's first version is staged.
5. **Stage 2, tests.** The test parts get `k + 2` agents each, named `t{i}.{v}`, in copies that already hold the picked code. Then `_pick` runs over all parts.
6. **Find a check.** If there is still no check (no `--check`, and `cmd_run`'s `detect_check` found none in the project at the start), `detect_check` looks again in the staging copy, and `-x` is removed from what it finds (`report["check"]`). This way the tests the parts just wrote become the check for the repairs.
7. **Repairs.** Fresh agents named `f{r}.{i}.{v}` (`k + 1` per part) work on the parts the failing output blames. `_pick` runs after each round.
8. **Racer result.** If the parts did not pass, the run waits for the racers. A racer's passing files are staged only if they also pass in the staging copy (`_take_solo`).
9. **Write-back.** `_finish(ok)` writes the files the run changed into the project, or keeps them in `~/.locals/runs/`.

Whatever happens, every agent's copy and the staging copy are removed at the end.

## Flags

The `run` sub-command (`app.py`) passes its flags to `Run(...)` in `cmd_run`.

| Flag | Default | What it does |
|---|---|---|
| `task` | required | The task, in plain words. |
| `--dir` | `.` | The project. Agents work in copies of it, and the project is written only at the end. |
| `--check` | the project's tests | A command whose exit code 0 means the task is done. Without it, `detect_check(root)` is used, and the `-x` flag is removed from the detected command so that every failure shows and the fixers of all failing parts work at once. |
| `--parts` | 8 | The most parts a plan may have (`max_parts`). |
| `--versions` | 3 | Agents per code part (`Run.k`). Test parts get two more and repair parts one more (`_more`). |
| `--race` | 2 | Whole-task agents racing the parts (`Run.solos`). 0 turns them off. |
| `--head` | 0 | Seconds the racers run alone before the parts start. A comment says this helped neither Qwen3.5 nor Gemma 4. |
| `--workers` | 0 | Only `1` changes anything. It gives the single-agent baseline: one part owning every file of the plan (any file, `_AnyFile`, when the plan names none), one version, no racers, and `max_steps` multiplied by the number of planned parts. |
| `--rounds` | 2 | Repair rounds. Up to two more run while each round makes progress. |
| `--steps` | 10 | Tool steps per agent (`max_steps`). The help text says a fresh agent does better than a long session. Racers get `steps × parts`. |
| `--engine` | `locals` | `locals`, or `mlx` (stock mlx-lm, for comparison). The model id sent to the server is `{engine}/{short name}`. |
| `--report` | none | Writes the run's report as JSON to this path, also when the run crashes. |
| `--model`, `--port` | last used | Which server to use or start. The default model is Qwen3.5-4B, or Qwen3.5-2B on a Mac with less than 12 GB of memory (`default_model`). The default port is 8321. |

`run` also accepts the server options from `add_server_opts`, such as `--slots`, `--temp`, `--cache-gb` and `--max-context`. They apply only when `locals run` starts the server. If a server with the same model and the current build is already running, `_ensure_server` prints a note that `--slots`, `--temp`, `--top-p`, `--top-k`, `--max-tokens`, `--cache-gb`, `--max-calls` and `--max-context` apply when it next starts. Other server options get no note. If the running server has another model or older code, an idle background server started by locals is restarted with these options. Otherwise the running server is used as it is (older code) or the command exits (another model).

## Planning

`Run.plan()` calls `make_plan` on `project_digest(project)`, then passes the parts through `_owned` and `tests_after_code`.

**Parts from the task's file names.** `cover([], task, project)` makes one part for each code file the task names. The regex `_CODE_FILE` matches `.py`, `.js`, `.ts` and `.sh` names. `_file_name` drops names that are products or web addresses rather than files:

- a capitalized stem that is not in the project, such as `Node.js`;
- a path with a dot in a folder name, such as `github.com/x/y.py`, unless that path exists in the project.

If this gives any parts, the model is never asked for a plan. The comment says this saves 10 to 15 seconds and avoids the 4B planner chaining parts together and dropping files.

**Parts from the model.** Otherwise the model is asked for a plan:

- the reply must follow the JSON schema `PLAN_SCHEMA`: 1 to 8 parts, 1 to 6 files each, with fields `title`, `files`, `instructions` and `needs`;
- the system prompt is `PLANNER`;
- `max_tokens` is 1200;
- there are two attempts, at temperature 0.2 and then 0.7.

Four functions then make the plan safe to use:

- `normalize` cleans the file paths (no absolute paths, `~` or `..`). It keeps only `needs` that point at earlier parts, so the plan has no cycles. A part that shares a file with an earlier part is made to need it.
- `writable` limits what a part may write. When the task names code files, a part keeps only those and new test files. Otherwise it keeps new files and existing files the task mentions. An existing file the task does not name is treated as context, not work.
- `cover` gives its own part to any named code file the plan left out.
- `protect_tests` removes existing test files from the parts (see the next paragraph).

If neither attempt gives a usable plan, `make_plan` falls back to the task's file names. That is always empty here, because a task with usable file names never reaches the model, so it returns no parts. `Run.plan` then makes a single part, "the whole task", with no files. It does the same whenever `_owned` leaves no parts. Its agent may write any file (`_AnyFile`).

**Existing tests are the check.** By default, test files that already exist are removed from the parts (`protect_tests`, `_owned`). When there is a check, they also go into `Run.readonly`, and `write_file` and `edit_file` refuse them for every agent, even one that owns every file. A `run` command can still change them in the agent's own copy, and that can make the after-write check pass there and set `Run.solved`. Such a change never reaches the staging copy, because a version carries only part files, files written with the tools and named data files. The docstring gives the reason: an agent that rewrote the tests could make the check pass.

This does not apply when `asks_test_changes(task)` is true. It is true when:

- a verb acts on tests, as in "update test_api.py" or "add tests";
- the task asks for tests by name and is not about making tests pass;
- the task renames something, because the existing tests call the old name.

`_is_test` decides what counts as a test: a `.py` file whose name starts with `test`, or a `.py` file under a `tests/` folder.

**One owner per file.** `_owned` gives each file to the first part that names it. If two parts wrote one file at the same time, only the last write would survive.

**Code before tests.** `tests_after_code` overrides the plan's `needs`:

- a part with at least one non-test file needs nothing, so all code parts start together;
- a part made only of test files needs every code part.

The comments give two reasons. The 4B planner once chained six independent modules into five waves, so nothing ran side by side. And tests written before the code existed expected a different error from the one the code raised, so the repairs bent the code to fit the test.

## Agents

An `Agent` is one tool loop against `POST /v1/chat/completions` on the locals server. Each step sends the whole conversation with:

- `tools=TOOLS`
- `tool_choice="required"`
- `parallel_tool_calls=False`
- `max_tokens=3000`
- `temperature`, when the agent has one

One tool call runs per step.

### Tools

| Tool | Arguments | Behaviour |
|---|---|---|
| `list_files` | `path` (optional) | Lists the files under the folder, leaving out `SKIP_DIRS`, 300 at most. |
| `read_file` | `path` | Returns the file's text, cut at 6,000 characters. |
| `write_file` | `path`, `content` | Writes the whole file and adds a final newline. Refused for a file the agent does not own, a file in `readonly`, a folder, or a path outside its copy. |
| `edit_file` | `path`, `old`, `new` | Replaces `old`, which must occur exactly once (see below for what happens when it does not match). |
| `run` | `command` | Runs the command in bash in the agent's copy, sandboxed (see below), for 60 s at most. Returns `exit N` and the last 1,500 characters of output. On a timeout it returns a note and the last 1,000 characters. |
| `finish` | `summary` | Ends the agent. Refused while the check blames its part (see below). |

When `edit_file` finds no exact match, `_loose_replace` tries again. It first ignores whitespace around each line, then also ignores blank lines, and it re-indents `new` by how far `old` was off. If that also fails, the reply shows the closest lines in the file (`_closest`) so the agent can copy the old text exactly. The docstring explains why: small models copy indentation, trailing spaces and blank lines wrong, Gemma 4's edits missed this way again and again, and a third of all repair edits did.

Paths go through `_rel` and `_path`:

- `./` prefixes are removed;
- absolute paths, `..`, and symlinks that resolve outside the copy are refused;
- when an agent owns one file and leaves out the path of a write or edit, that file is used (Gemma 4 does this).

### Ending an agent

An agent ends when it calls `finish`, after `max_steps` steps, or when its `stop` event is set. The event is checked between steps, so a request already in flight is not cancelled.

`Agent.run` also ends an agent on these guards:

- the conversation grows past 90,000 characters;
- a reply comes back with no tool call. (The harness comment gives the server's loop guard as the cause, but the harness always sends `tool_choice="required"`. For such a request the server steers once and then returns the repeated call instead of ending the turn, so a reply without a call has some other cause, for example a reply cut short before its call is complete.) The first time, the agent gets the `NUDGE` message; the second time, it ends.
- the same two calls alternate three times.

Softer guards only add a note to the tool result: the same call three times, two calls repeating, and four tool calls in a row other than `write_file`/`edit_file`. A refused write or edit also resets the count, and an agent that may write any file (`_AnyFile`, an empty list) never gets this note.

`finish` is checked when the agent has a verifier (`part_verifier`). The check runs in the agent's copy. If it fails and its output blames the agent's part, `finish` is refused and the agent sees the last 2,000 characters of output. After two refusals the agent may finish unverified. `Agent.verified` is set when the check passes, or when it fails only in other parts.

## Copies

`Run._copy` makes each copy under the system temp folder (`tempfile.mkdtemp(prefix="locals-run-{tag}-")`):

- `.git` and cache folders are left out (`_COPY_SKIP`: `.git`, `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.tox`, `.DS_Store`);
- `.venv`, `venv`, `env` and `node_modules` at the top level are symlinked to the project's own instead of copied (`_LINK_DIRS`). The three env folder names are linked only when they contain `pyvenv.cfg`, since a source package may be called `env`.
- relative symlinks that would point outside the copy are re-pointed to their absolute target.

The staging copy is a copy of the project. Every agent's copy is a copy of the staging copy as it is when the agent starts. So the test writers in stage 2 and the fixers start from the code picked so far.

## The check after every write

After each successful `write_file` or `edit_file`, the check runs in that agent's copy, and its result is added to the tool result (`Run._after_write`). The docstring explains why: small models spent half their steps running the check and reading its output.

The agent gets one of three messages:

- "The project's check passes now. Call finish." The agent is marked verified and `Run.solved` is set, which stops every agent in the stage (but not the racers).
- "...still fails in your part:" followed by the last 1,500 characters of output, when `_blame` points at this part.
- "...nothing it reports is in your files."

Once an after-write check takes over 30 s, that agent gets no more after-write checks (`a.after_write = None`). This applies to racers too.

`run_check` runs the command through `run_shell` with a 300 s timeout. It is not sandboxed: `run_check` calls `run_shell` without `confine`. The code gives no reason. A run that ran no tests ("Ran 0 tests", "no tests ran") counts as a failure, because unittest exits 0 when it finds nothing.

`_blame(out, parts, cur)` decides which parts a failing output points at. A part is blamed when either:

- the output names one of the part's files as a whole word (`_names`). For example, `inventory.py` does not match inside `test_inventory.py`. For a test file, its module name also counts.
- the output calls (`name(`) a function or class that the part's current `.py` files define, for names longer than two characters.

## Versions, racers and how many run at once

**Versions.** `_versions` starts `k` agents for each part, each in its own copy. Each version gets its temperature from `_TEMPS = (None, 0.7, 0.9, 0.5, 1.0, 0.6, 0.8, 0.4)` by its number, so the versions differ. `None` means the server's own temperature: its `--temp`, 0.2 by default.

A version is the text of the part's files in the agent's copy. For a part with no files, it is the files the agent wrote. A version also includes any data file that the task names and that the agent's commands created or changed (`_outputs`). That covers `.json`, `.csv`, `.txt`, `.md`, `.yaml` and similar files up to 1 MiB. Without this, a part that ran its script in its copy, as the task asked, passed there and then failed in the project, where the output file never appeared.

Agents that wrote nothing leave no version. Identical versions are kept once.

**Group stop and the tail cut.** `_run_agents` runs the agents as threads:

- when one version of a part is verified, the other versions of that part stop;
- when only a quarter of the agents (at least one) are still working, the rest get half the time spent so far, at least 20 s, and then stop.

**Whole-task racers.** `_start_solo` starts `--race` agents that own every planned file, each in its own copy, with `max_steps × parts` steps. After every write, their check result comes from `_solo_word`.

The first racer whose copy passes saves its files as they passed (`won_files`) and sets `Run.won`. That stops the stage's agents (`_run_agents`) and makes `Run.go` skip the remaining stages and repairs. The other racers are stopped when `Run.go` reaches the end of the stages. The docstring gives the reason for racers: one agent alone did the easy tasks in a fifth of the time the parts took, and the parts did the hard ones it failed.

If the parts fail, the run waits until a racer passes or all racers stop. A winner's files are written into the staging copy, and they stay only if the check passes there too (`_take_solo`).

**How many at once.** `Run.at_once = 6` is the number of agents in flight that the run aims for, racers included. The limit comes from the server's prompt cache, which must hold each agent's conversation. The comment records that with Qwen3.5-4B on 16 GB and eight agents of 10k tokens, a third of the steps found nothing cached.

`_versions` gives each part `max(1, min(k, room // len(todo)))` agents, where `room` is 6 minus the racers still working. Without a check, `k` is 1, because there is nothing to pick by. Because of the floor of one, a stage with more parts than `room` runs one agent per part, which can be more than six in all.

Example with the defaults (`--versions 3`, `--race 2`), a plan with several parts, and both racers still working, so `room` is 4:

| Parts in the stage | Asked per part | Started per part |
|---|---|---|
| 1 code part | 3 | 3 |
| 2 code parts | 3 | 2 |
| 3 or more code parts | 3 | 1 |
| 1 test part | 5 | 4 |
| 1 part being repaired | 4 | 4 |

## Picking and combining

`Run._pick` chooses one version per part by running the check on combinations of versions:

1. **Whole versions first.** It checks version 0 of every part, then version 1 of every part, and so on. A part with fewer versions uses its last one. `_check_all` runs these a few at a time until one passes.
2. **Search.** If none passes, `_search` starts from the best combination checked so far. It tries every combination that swaps one part for another of its versions, merges the swaps that did better into one combination, and checks that. The best result becomes the next starting point. It stops when one passes, when nothing improves, or after 24 checks (`cap`). Mixing works because the parts are independent by design, so a part from one version fits with the parts from another.
3. **Choice.** `_best` takes a passing combination if there is one. Otherwise it takes the least bad by `_score`, which ranks failures in this order:
   - code that does not load (`SyntaxError`, `IndentationError`, `TabError`, `ImportError`, `ModuleNotFoundError`, `NameError`) is worst;
   - then the number of failed tests, read from the last unittest or pytest summary (no summary counts as worst);
   - then whether the failure is the same as before (`_key`: the failure lines with numbers removed).

`_Checks` checks each combination in a throwaway copy of the staging copy, with the combination written over it, four at a time. Checks that start before any check in this `_Checks` has finished get a 120 s timeout. `_pick` makes a new `_Checks` each time, so the timing starts over at every pick. Later checks get four times the median check time, at least 30 s, so a candidate that hangs costs only a few checks' time.

The chosen combination is written into the staging copy (`_stage`). In repair rounds, a combination is staged only if it passes or scores better than the current failure. Otherwise the staging copy stays as it was.

## Repair rounds

After the two stages, if the check fails, `Run.go` runs repair rounds:

- **Which parts.** `todo` is the parts `_blame` points at. If it points at none, every code part; if there are no code parts, every part.
- **Widening.** If the same parts fail the same way (`_key`) after a round on them, the parts whose modules they import are added. `_uses` finds these from `import` and `from ... import` lines. The added fixers are told the failure shows in code that uses theirs. The comment's example: an `Inventory` that never loaded its file failed only through `cli.py`.
- **Fixers.** Each part in `todo` gets `k + 1` fresh agents (`_more(1)`) with `fixer_prompt`.
- **Pick.** `_pick` runs with the current output as the baseline.

The fixer prompt contains:

- the shared prefix;
- the last 2,500 characters of the failing output;
- the text of every part's files from the plan (not the whole project), up to 5,000 characters in all (`_texts(root, files, limit=5000)`). A file that does not fit is named with its length.
- the lines of the output about this part's files and functions (`_mine`), with the traceback line before each and the exception after. The docstring explains why: a fixer given the whole output fixed another part's failure, the first one listed.

If only test parts are blamed, the fixers are told to take the code as right and fix the tests.

A round counts as progress (`_progress`) when fewer tests fail, when code now loads that did not, or when the failure count is the same but the failure changed. The loop runs `--rounds` rounds, plus up to two more while each round makes progress. It stops as soon as the check passes or a racer wins.

## The sandbox

Commands an agent runs with the `run` tool go through `run_shell(cmd, root, 60, confine=root)`. With `confine` set, `run_shell` wraps bash in macOS `sandbox-exec`, using the profile from `write_sandbox(root)`. The profile allows everything except writing files. Writes are allowed only under:

- the agent's copy;
- the temp folder (`tempfile.gettempdir()`), `/private/tmp` and `/private/var/tmp`;
- `~/Library/Caches` and `~/.cache`;
- `/dev/null`, `/dev/tty`, `/dev/zero` and `/dev/fd/*`.

The docstring records why: a command an agent ran with the absolute path the check named wrote into the folder of the check's script. When a write is refused, the output says "Operation not permitted", and the agent is told where it may write.

The sandbox is skipped when `LOCALS_NO_SANDBOX` is set to any non-empty value, or when `sandbox-exec` is not on the PATH. The check itself is never confined. The file tools do not use the sandbox: `_path` and file ownership confine them.

Every command, sandboxed or not, runs:

- with no terminal input (stdin is `/dev/null`), so a test that calls `input()` does not wait for the keyboard;
- with `PYTHONDONTWRITEBYTECODE=1`;
- in its own process group, which is killed when the command ends or times out, so background processes left in that group do not survive. A child that left the group (setsid, a daemon) may survive.

Process groups still running when locals exits are killed by an `atexit` hook.

## What is written to the project, and when

The project is touched once, by `Run._finish`, at the end of a run that completed:

- **Check passed, or the run started without a check** (`given_check` is None: no `--check`, and `cmd_run`'s `detect_check` found none in the project). In that case the files are written even when a check detected later from the tests the parts wrote (step 6) fails. Every file the run changed in the staging copy (`Run.touched`) is written into the project, with these safeguards:
  - A file that changed in the project since the run started (compared with the start `manifest`) is not overwritten. It is listed as `skipped`, and the run's version is kept as described below.
  - Files are written under temporary names (`.{name}.locals-{pid}`) and then renamed, so the project gets all of them or none, unless the process crashes between two renames.
  - An existing file keeps its permissions.
  - A path that resolves outside the project is refused.
- **Check failed, in a run that started with a check** (given or detected by `cmd_run`). Nothing is written. The files the run changed are copied to `~/.locals/runs/<project>-YYYYmmdd-HHMMSS/` (`_keep`, with `keep_dir=app.LOCALS_DIR / "runs"`).
- **Stopped with Ctrl-C, or crashed.** The work so far goes to the same folder and the project is not changed. `cmd_run` prints "stopped; the project was not changed".

`_finish` has no delete path, so the harness never deletes a file from the project. Only part files and named data files reach the staging copy. Scratch files that an agent's commands made in its copy are discarded with the copy.

## Prompts and the shared prefix

Each prompt is built so that the agents of one stage share a long common prefix:

- `SYSTEM` is one constant system prompt for every agent.
- `_shared_prompt` comes first in every prompt, in the same words for every agent of a stage. A racer's prompt lists one part, "the whole task", so its prefix differs after the task. It holds the project digest, the task and the list of parts. The digest (`project_digest(root, limit=6000)`) has every file's path and the text of the small files.
- `worker_prompt` then adds, for a part that builds on others, the text of their files (up to 9,000 characters, `_texts`). This text is the same for every test writer. After it come the agent's own part, its files, `TEST_HINT` for test parts, and the check command.
- `fixer_prompt` adds the failing output and the text of the parts' files, which are the same for every fixer in a round, before the lines specific to each part.

## Why this design suits a batch server

The locals server decodes requests that arrive together as rows of one batch, and it keeps a prompt cache. The harness is built around both:

- **Parallel agents are cheap.** The module docstring gives Qwen3.5-4B on an M2 Pro at 64 tokens/s for one stream and 271 tokens/s in all with eight streams. Versions, racers and fixers are rows of one batch, so N agents cost far less than N times one (module docstring). Each row still runs slower than a lone stream: 64 tokens/s alone, 271 in all with eight.
- **The common prompt is computed once.** Prompts start with the same text for every agent in a stage (see above), so the server prefills that prefix once. The rest of each conversation stays in the cache between steps, but only while the cache holds every agent's conversation (see `at_once`).
- **The cache sets how many agents run.** `at_once = 6` and the per-part version count come from what the cache can hold. The comments record two cases:
  - with eight agents of 10k tokens, a third of the steps found nothing cached;
  - with twelve agents of 5k tokens, every step prefilled its whole prompt again.
- **Agents stay short.** Agents are capped at `--steps` (10; racers `--steps` × parts) and at 90,000 characters. The check result arrives with each write instead of costing a step, and a stuck agent ends instead of looping, so the next round starts a fresh agent with a shorter prompt. Test and repair stages get more versions than code (`_more`) because they have fewer parts, so the batch has free rows.
- **The check decides, not the model.** With many cheap versions and a fixed rule for picking, a 4B model does not have to be right the first time.

## Output and report

While the run is going, `cmd_run` keeps one status line up to date. It shows:

- the seconds elapsed;
- each agent as `name:` followed by the first letter of its state and its step count;
- the server's active batch rows, from `/stats`.

Agent names follow a pattern: `s{v}` for racers, `w{i}.{v}` for code, `t{i}.{v}` for tests and `f{r}.{i}.{v}` for fixers.

At the end it prints:

- one line per agent;
- whether the check passed;
- the number of agents and the most that ran at once (`_most_at_once`);
- the racer that won, if any;
- the files written, skipped or kept.

When nothing was written, it prints the `finish` summary of an agent that owned every file or none. That is how a question such as "Which function parses dates?" gets its answer.

The exit status is 0 when the check passed or there was none, and 1 otherwise.

`--report PATH` writes `Run.report` as JSON. Its keys are:

- `parts`, `plan_s`, `workers_s`, `passed_before`, `passed`, `wall_s`;
- `checks`, with each check's result and the end of its output;
- `repairs`, with the parts and time of each round;
- `picks`, `won_by`, `changed`, `skipped`, `kept`, `tests_kept`, `check` (when it was detected after the stages, from the tests the parts wrote; not when `cmd_run` detected it at the start) and `most_at_once`;
- `agents`, with one entry per agent: `name`, `files`, `steps`, `tokens`, `wrote`, `summary`, `error`, `start_s`, `wall_s`, `verified` and `trace`. The trace lists each step's tool, its argument and the start of its result.

## Related: `locals solve`

`plan.py` and `solve.py` also hold `locals solve`, which works differently. Its `Race` runs OpenCode sessions in git worktrees or copies, and the first to pass the check wins. Its `--plan` mode writes each part in a single generation, in a `FILE` format constrained by a regular expression. `locals run` uses only the planning, picking, shell and check helpers listed in the table at the top.
