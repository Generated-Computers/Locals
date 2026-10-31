"""Our server's answers, in process (Server.complete, as tests/test_vision_model.py).
Usage: run_locals.py MODEL PICDIR OUT.json [nodeepstack]"""
import base64, json, sys
from pathlib import Path
from locals import vision
from locals.server import Server

model_id, pics, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
nodeep = len(sys.argv) > 4 and sys.argv[4] == "nodeepstack"
calls = {"encode": 0, "deep_levels_seen": 0}
real = vision.Encoder.encode
def encode(self, pictures):
    f, deep = real(self, pictures)
    calls["encode"] += 1
    calls["deep_levels_seen"] = len(deep)
    return (f, ()) if nodeep else (f, deep)
vision.Encoder.encode = encode

qs = json.loads((pics / "questions.json").read_text())
srv = Server(model_id, engines=("locals",), preload=("locals",))
res = {}
for name, q in qs.items():
    url = "data:image/png;base64," + base64.b64encode((pics / name).read_bytes()).decode()
    req = {"model": "locals", "max_tokens": 60, "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": q}]}]}
    answers = [srv.complete(req)["choices"][0]["message"]["content"].strip() for _ in range(2)]
    res[name] = {"answers": answers}
    print(name, answers, flush=True)
print(f"encode calls {calls['encode']}, deepstack levels from the tower {calls['deep_levels_seen']}, "
      f"{'dropped' if nodeep else 'used'}", flush=True)
res["_deepstack"] = {"levels": calls["deep_levels_seen"], "dropped": nodeep, "encode_calls": calls["encode"]}
Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
srv.unload()
