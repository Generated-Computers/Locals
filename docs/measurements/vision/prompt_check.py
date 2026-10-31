"""Our prompt text for each picture, against the prompt mlx-vlm recorded (image pads collapsed to one)."""
import base64, json, re, sys
from pathlib import Path
from locals import engine as eng
from locals.server import Server
model_id, pics, vlm = sys.argv[1], Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
seen = []
real = eng.Engine.generate_embedded
def wrapped(self, prompt, *a, **k):
    seen.append(list(prompt)); k["max_tokens"] = 1
    return real(self, prompt, *a, **k)
eng.Engine.generate_embedded = wrapped
srv = Server(model_id, engines=("locals",), preload=("locals",))
tok = srv._svc["locals"].tokenizer
qs = json.loads((pics / "questions.json").read_text())
same = 0
for name, q in qs.items():
    url = "data:image/png;base64," + base64.b64encode((pics / name).read_bytes()).decode()
    srv.complete({"model": "locals", "max_tokens": 1, "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": q}]}]})
    ours = re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>", tok.decode(seen[-1]))
    ok = ours == vlm[name]["prompt"]
    same += ok
    if not ok:
        print(name, "\n ours:", repr(ours), "\n vlm: ", repr(vlm[name]["prompt"]))
print(f"{model_id}: our prompt is mlx-vlm's on {same} of {len(qs)}")
srv.unload()
