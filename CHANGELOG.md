# Changelog

## 0.2.0 (unreleased)

First public release.

- **One command for coding agents.** `locals code` opens OpenCode on a local
  model; `locals launch` does the same for OpenHands, Cline, Kilo Code,
  OpenClaw, Hermes Agent, Aider, editor extensions and apps that work with Ollama.
- **An OpenAI- and Ollama-compatible server** on MLX, running in the background
  (`locals start`, `locals status`, `locals stop`) or in the menu bar.
- **Many requests at once.** Continuous batching, kernels tuned for your GPU and
  a prompt cache: about 3x the throughput of mlx-lm with eight requests at once,
  with the same answers.
- **Tool calls and JSON that always parse**, in each model's own format: Qwen,
  Gemma 4, Llama, Mistral, LFM2.5, MiniCPM5 and Hermes-style models.
- **Models for every Mac.** A default that fits your memory, short names for
  tested models (`qwen`, `gemma`, `lfm`, `minicpm`, `qwen3.8`), any MLX model
  from Hugging Face, and a clear message for models that do not fit or come in
  the GGUF format.
- **`locals run`**, a team of agents on one task: each in its own copy of the
  project, your tests as the check, the project changed only when they pass.
- **Safe defaults.** The server answers only apps on your Mac, and agents'
  commands can write only inside their own copy of the project.
