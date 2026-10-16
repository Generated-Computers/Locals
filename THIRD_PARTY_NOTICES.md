# Third-party notices

locals builds on these projects; parts of its code follow theirs, as noted here.

## MLX (Apple)

https://github.com/ml-explore/mlx

- `locals/attention.py` adapts MLX's `sdpa_vector` Metal kernel.
- `locals/qmv_multi.py` follows the layout of MLX's `qmv_fast` kernel and uses
  MLX's nibble-mask technique.
- `locals/mma_kernel.py` includes MLX's Metal headers from the installed `mlx`
  package when it compiles its kernels (nothing of them is copied into this
  repository).

```
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## mlx-lm (Apple)

https://github.com/ml-explore/mlx-lm (MIT, Copyright © 2023 Apple Inc., the
license above)

- `locals/fuse.py` reimplements the forward passes of mlx-lm's Llama, Qwen2 and
  Qwen3.5 layers with fused projections; `locals/hybrid_batch.py` extends
  mlx-lm's `LRUPromptCache`.

## Recorded requests in `tests/data`

`tests/data/*.json` hold requests that OpenCode, OpenClaw, Hermes Agent and
the client libraries named in each file's `source` field sent to a scripted
server, used to test that locals answers them. The system prompts and tool
descriptions in them are those projects' text (some truncated), under their
own licenses.

## Logos of the agent tools in `docs/assets/agents`

The README shows these tools' icons to say that Locals works with them. Each
icon is taken from the tool's own repository, resized and set on a rounded tile;
the logos and names remain the property and trademarks of their owners.

| File | Tool | Taken from |
|---|---|---|
| `opencode.png` | OpenCode | https://github.com/anomalyco/opencode, `packages/desktop/icons/prod/icon.png` |
| `openhands.png` | OpenHands | https://github.com/OpenHands/OpenHands, `electron/build-resources/icon.png` |
| `cline.png` | Cline | https://github.com/cline/cline, `assets/icons/icon.png` |
| `kilo.png` | Kilo Code | Kilo Code's square logo (https://github.com/Kilo-Org/kilocode) |
| `roo.png` | Roo Code | https://github.com/RooCodeInc/Roo-Code, `src/assets/icons/icon.png` |
| `openclaw.png` | OpenClaw | https://github.com/openclaw/openclaw, `apps/linux/src-tauri/icons/icon.png` |
| `hermes.png` | Hermes Agent | https://github.com/NousResearch/hermes-agent, `apps/desktop/assets/icon.png` |
| `aider.png` | Aider | https://github.com/Aider-AI/aider, `aider/website/assets/icons/apple-touch-icon.png` |
