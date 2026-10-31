# Vision and unload measurements

Run on an M2 Max (32 GB), 2026-10-31, at commit `20bafe1`, with `mlx-community/Qwen3.5-4B-MLX-4bit` and
`mlx-community/Qwen3-VL-8B-Instruct-4bit`. mlx-vlm 0.7.11 (mlx 0.32.4, transformers 5.22.1), run through `uvx`.
Each command is run from this folder.

## Our answers against mlx-vlm's (`docs/server/README.md`, Images)

```
python make_pictures.py pics                                    # the six pictures and their questions
uvx --from mlx-vlm python -I run_mlxvlm.py <model> pics vlm_<m>.json      # mlx-vlm, its own prompt
uvx --from mlx-vlm python -I run_mlxvlm.py <model> pics vlm_<m>_nl.json nl  # with our newline after the image
<repo>/.venv/bin/python run_locals.py <model> pics loc_<m>.json [nodeepstack]  # ours, in process
python compare.py results                                       # ours against mlx-vlm's own prompt
```

Every question is asked twice; on every side both answers were the same. `run_locals.py nodeepstack` drops the
deepstack features the tower returns (it prints how many levels it saw: 3 for Qwen3-VL). The comparison with the
same prompt on both sides is `results/compare_same_prompt.txt`.

## Unloading (`docs/server/README.md`, Unloading the model)

```
<repo>/.venv/bin/python unload_serve.py 8399 &                  # the server, its PID from $!
<repo>/.venv/bin/python unload_client.py 8399 <PID>
```

`unload_serve.py` starts `Server` and `serve` in Python, not `locals serve`, which saves the port and the model as
the next default. The output is `results/unload_run.txt`; `phys_footprint` is read with `footprint -p`.
