# Vision and unload measurements

Run on an M2 Max (32 GB), 2026-10-31, at commit `20bafe1`, with `mlx-community/Qwen3.5-4B-MLX-4bit` and
`mlx-community/Qwen3-VL-8B-Instruct-4bit`. mlx-vlm 0.7.11 (mlx 0.32.4, transformers 5.22.1), run through `uvx`.
Each command is run from this folder.

## Our answers against mlx-vlm's (`docs/server/README.md`, Images)

```
python make_pictures.py pics                                    # the six pictures and their questions
uvx --from mlx-vlm python -I run_mlxvlm.py <model> pics vlm_<m>.json      # mlx-vlm, its own prompt
uvx --from mlx-vlm python -I run_mlxvlm.py <model> pics vlm_<m>_nl.json nl  # with a newline after the image
<repo>/.venv/bin/python run_locals.py <model> pics loc_<m>.json [nodeepstack]  # ours, in process
<repo>/.venv/bin/python prompt_check.py <model> pics results/vlm_<m>.json      # our prompt against mlx-vlm's
python compare.py results                                       # ours against mlx-vlm's own prompt
```

Every question is asked twice; on every side both answers were the same. `run_locals.py nodeepstack` drops the
deepstack features the tower returns (it prints how many levels it saw: 3 for Qwen3-VL). Ours were run on
2026-10-31 at commit `929eb9f` with the image joined to the question without a newline (the change after it); the
mlx-vlm answers are the ones from commit `20bafe1`. `results/compare.txt` is the comparison and
`results/prompt_check.txt` the prompt check (the image's pad tokens counted as one).

Up to commit `929eb9f` our prompt had a newline after the image. Our answers from then are `loc_*_nl.json`, and
`results/compare_newline.txt` compares them with mlx-vlm given the same newline (`vlm_*_nl.json`).

## Unloading (`docs/server/README.md`, Unloading the model)

```
<repo>/.venv/bin/python unload_serve.py 8399 &                  # the server, its PID from $!
<repo>/.venv/bin/python unload_client.py 8399 <PID>
```

`unload_serve.py` starts `Server` and `serve` in Python, not `locals serve`, which saves the port and the model as
the next default. The output is `results/unload_run.txt`; `phys_footprint` is read with `footprint -p`.

## Prefill memory of a long prompt with one image (`docs/server/README.md`, Images)

```
python make_pictures.py pics
git archive 5b6a125 locals | tar -x -C before                   # the code before, beside the working tree's
PYTHONPATH=before <repo>/.venv/bin/python prefill_memory.py <model> pics/invoice_900x500.png mem before.json
<repo>/.venv/bin/python prefill_memory.py <model> pics/invoice_900x500.png mem after.json
```

`prefill_memory.py` prints which `locals` it imported. `mem` puts about 36,000 tokens of text before the picture;
`exact` asks the invoice question with `prefill_step` 256, so a chunk ends inside the image. Each process sends the
request twice, and both peaks were the same to the MB on every side. The output is `results/prefill_memory.txt`.

## `count_tokens` with images (`docs/server/README.md`, Images)

```
git archive c358179 locals | tar -x -C before                   # the code before the change
PYTHONPATH=before <repo>/.venv/bin/python unload_serve.py 8399 &  # from a folder outside the repo: the
<repo>/.venv/bin/python count_tokens_client.py 8399             # repo as the current folder wins over PYTHONPATH
PYTHONPATH=<repo> <repo>/.venv/bin/python unload_serve.py 8399 &  # after; each server stopped by its PID from $!
<repo>/.venv/bin/python count_tokens_client.py 8399
```

The client sends two 4032x3024 JPEGs (made in the script) to `count_tokens` seven times, then once to
`/v1/messages`, and prints both token counts and the median time. Each side was run twice, each by a fresh server.
The output is `results/count_tokens.txt`.

## Where an image request's time goes (`results/stage_cost.txt`, `results/mrope_cost.txt`)

```
<repo>/.venv/bin/python stage_cost.py <model>                   # vision.load, the tower, the prompt read, the request
<repo>/.venv/bin/python mrope_cost.py <model> mrope             # the prompt read with M-RoPE
<repo>/.venv/bin/python mrope_cost.py <model> stock             # with the layers' own rope: the floor for any M-RoPE speedup
```

Both send four requests of about 5,600 tokens (text, a 1280x800 and a 1024x1024 picture) and print the medians of
runs 2 to 4. Each line of the results is one process; each model and mode was run twice.

## A follow-up turn about the same images (`results/turn2_cost.txt`)

```
git archive 6b290df locals | tar -x -C before                   # the code before image prompts were cached
PYTHONPATH=before <repo>/.venv/bin/python turn2_cost.py <model>   # from a folder outside the repo (see above)
PYTHONPATH=<repo> <repo>/.venv/bin/python turn2_cost.py <model>
```

The same request as above is the first turn; the second adds its answer and a question. Each run empties the prompt
cache, asks the first turn, and times the second (one token). Each side was run twice per model.
