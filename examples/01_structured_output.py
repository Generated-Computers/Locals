"""Structured output with the locals engine.

    python examples/01_structured_output.py [model]

The output follows the JSON schema exactly. Text the schema forces (keys,
quotes, brackets) is written without running the model, so the output needs
fewer forward passes than free text, and the model's answer is unchanged.
"""

import json
import sys

import locals

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3.5-4B-MLX-4bit"

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"},
        "status": {"enum": ["pending", "approved", "rejected"]},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age", "email", "status", "tags"],
}

TEXT = ("Maria Gonzalez, 34, signed up with maria.g@example.com last week. Her application was "
        "approved. She is interested in hiking, photography and machine learning.")

rt = locals.load(MODEL)  # loads the model once; the first load of a new model shape tunes kernels
prompt = f"Extract the person from the text below as JSON following this schema:\n{json.dumps(SCHEMA)}\n\nText: {TEXT}"

res = rt.generate(prompt, schema=SCHEMA, max_tokens=200)
print(res.text)
record = json.loads(res.text)  # always parses: the output is in the schema's language
print(f"\n{len(res.tokens)} tokens in {res.passes} forward passes ({res.forced_tokens} written without a pass), "
      f"{res.tok_per_s:.0f} tok/s, finish={res.finish!r}")

# A regular expression works the same way.
res = rt.generate("Pick a colour and a fruit.", regex=r"(red|green|blue) (apple|pear|plum)", max_tokens=8)
print("\nregex:", res.text)

# Under a token limit the output stays valid: arrays close early, strings shorten.
res = rt.generate(prompt, schema=SCHEMA, max_tokens=28)
print("\n28-token limit:", res.text, "->", "valid JSON" if json.loads(res.text) else "")
