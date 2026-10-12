# Contributing

## Setup

```bash
git clone https://github.com/Generated-Computers/Locals
cd Locals
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Apple silicon is required (MLX). Tests that need a model or tokenizer skip
themselves when it is not in the local Hugging Face cache, so download them once:

```bash
python -c "from huggingface_hub import snapshot_download as d; d('mlx-community/Qwen2.5-1.5B-Instruct-4bit', allow_patterns=['*.json', '*.txt', 'tokenizer*'])"
python -c "from huggingface_hub import snapshot_download as d; d('mlx-community/Qwen3.5-2B-MLX-4bit')"   # model tests
```

The first is all the fast suite needs (its tokenizer); the second is for the model tests.

## Tests

```bash
python -m pytest tests -q -m "not model"   # fast: grammars, tokenizer, server protocol, sampling (about 2 min)
python -m pytest tests -q                  # everything, with the models (~6 minutes once the models are cached)
LOCALS_TEST_MODEL=mlx-community/Qwen3.5-4B-MLX-4bit python -m pytest tests -q -m model
```

Every mechanism is tested against the plain path it replaces: forced spans
against token-by-token decoding, the GPU walk against the token table, the
pushdown against byte-level stepping, certified blocks against sequential
decoding, budgets against a simulated stopping rule. Please keep that
discipline for new mechanisms: an exactness test first, a speed number second.

## Style

Plain Python, type hints where they help, docstrings that say what a function
guarantees. Comments explain *why*, the code says *what*. No new dependencies
without a reason that fits in one sentence.
