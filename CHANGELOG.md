# Changelog

## 0.2.0 (unreleased)

First public release.

- **About three times faster with many requests at once.** Continuous
  batching, kernels tuned for your GPU and a prompt cache: 271 tokens/s with
  eight requests at once where mlx-lm gives 87, the same speed for one request,
  and the same answers. Each agent's next turn starts in under a second.
- **Many agents on one task.** `locals run` puts a team of agents on a task, each
  in its own copy of the project, with your tests as the check; the project is
  changed only when they pass.
- **One command for coding agents.** `locals code` opens OpenCode on a local
  model; `locals launch` does the same for OpenHands, Cline, Kilo Code,
  OpenClaw, Hermes Agent, Aider, editor extensions and apps that work with Ollama.
- **An OpenAI- and Ollama-compatible server** on MLX, running in the background
  (`locals start`, `locals status`, `locals stop`) or in the menu bar.
- **Tool calls and JSON that always parse**, in each model's own format: Qwen,
  Gemma 4, Llama, Mistral, LFM2.5, MiniCPM5 and Hermes-style models.
- **Models for every Mac.** A default that fits your memory, short names for
  tested models (`qwen`, `gemma`, `lfm`, `minicpm`), any MLX model from Hugging
  Face, and a clear message for models that do not fit or come in the GGUF
  format.
- **Safe defaults.** The server answers only apps on your Mac, and agents'
  commands can write only inside their own copy of the project.
