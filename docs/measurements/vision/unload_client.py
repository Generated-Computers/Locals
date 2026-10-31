"""Two image requests, footprint, /unload, footprint, one image request timed (the model loaded again), footprint.
Twice, in the same server process. Usage: client.py PORT PID"""
import json, re, subprocess, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repo, for tests/
from tests.test_vision_model import _invoice

port, pid = int(sys.argv[1]), sys.argv[2]
base = f"http://127.0.0.1:{port}"

def post(path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def fp():
    out = subprocess.run(["footprint", "-p", pid], capture_output=True, text=True).stdout
    m = re.search(r"phys_footprint:\s*([\d.]+)\s*([KMG]B)", out)
    if not m:
        return "UNPARSED: " + out[:400]
    v, u = float(m.group(1)), m.group(2)
    return f"{v * {'KB': 1/1024, 'MB': 1, 'GB': 1024}[u]:,.0f} MB (raw {m.group(1)} {u})"

url = _invoice()
def ask(q):
    t = time.perf_counter()
    r = post("/v1/chat/completions", {"model": "locals", "max_tokens": 60, "temperature": 0, "messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": q}]}]})
    return time.perf_counter() - t, r["choices"][0]["message"]["content"]

for cycle in (1, 2):
    for q in ("What is the customer's name?", "What is the invoice number?"):
        s, a = ask(q)
        print(f"cycle {cycle}: image request {s:.2f} s -> {a!r}")
    print(f"cycle {cycle}: phys_footprint after two image requests: {fp()}")
    print(f"cycle {cycle}: /unload -> {post('/unload', {})}")
    time.sleep(1)
    print(f"cycle {cycle}: phys_footprint after unload: {fp()}")
    s, a = ask("What is the invoice number?")
    print(f"cycle {cycle}: next image request (loads the model again) {s:.2f} s -> {a!r}")
    print(f"cycle {cycle}: phys_footprint after it: {fp()}")
