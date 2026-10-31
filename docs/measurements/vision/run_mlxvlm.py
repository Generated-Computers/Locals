"""mlx-vlm's answers. Usage: run_mlxvlm.py MODEL PICDIR OUT.json"""
import json, sys
from pathlib import Path
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model_id, pics, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
nl = len(sys.argv) > 4 and sys.argv[4] == "nl"  # the newline our server puts after the image
qs = json.loads((pics / "questions.json").read_text())
model, processor = load(model_id)
res = {}
for name, q in qs.items():
    prompt = apply_chat_template(processor, model.config, q, num_images=1)
    if nl:
        assert prompt.count("<|vision_end|>") == 1
        prompt = prompt.replace("<|vision_end|>", "<|vision_end|>\n")
    answers = [generate(model, processor, prompt, image=[str(pics / name)], max_tokens=60, temperature=0.0, verbose=False).text.strip()
               for _ in range(2)]
    res[name] = {"answers": answers, "prompt": prompt}
    print(name, answers, flush=True)
Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
