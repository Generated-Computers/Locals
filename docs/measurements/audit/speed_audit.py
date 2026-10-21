"""A client-side check of tokens per second through a server's OpenAI API, trusting nothing the server says.

- Tokens are counted by the client: the streamed text is tokenized again with the model's own tokenizer. The
  server's usage.completion_tokens is recorded only to compare.
- The clock is the client's: from the moment the requests are sent to the last byte received (end to end), and
  from the first token to the last (decode, as `locals bench` reports it).
- Every request has a prompt no other request had (a nonce at its start), so no prompt cache can help; the same
  (size, index) gets the same prompt on every server, so replies can be compared.
- Temperature 0, the same max_tokens, essay prompts long enough that replies run to max_tokens.

    python speed_audit.py --base http://127.0.0.1:8080/v1 --model mlx-community/Qwen3.5-4B-MLX-4bit --label mlx-lm \\
        --out mlxlm.json
"""
import argparse
import hashlib
import http.client
import json
import statistics
import threading
import time
import urllib.parse

TOPICS = ["how refrigerators work", "the history of the printing press", "the causes of inflation", "the water cycle",
          "how vaccines train the immune system", "black holes", "compound interest", "the life of a honeybee",
          "how airplanes stay in the air", "why sleep matters", "recursion in programming", "the Roman aqueducts",
          "how batteries store energy", "the origins of chess", "plate tectonics", "how the internet routes packets",
          "the invention of paper", "how bridges carry loads", "photosynthesis", "the Silk Road", "how compilers work",
          "volcanoes", "the history of clocks", "how bees make honey", "ocean currents", "the first computers",
          "how muscles grow", "the moon landing", "how glass is made", "coral reefs", "the steam engine", "jazz"]


def prompt_for(size: int, i: int, seed: int) -> str:
    nonce = hashlib.sha1(f"{seed}-{size}-{i}".encode()).hexdigest()[:12]
    return (f"[request {nonce}] Write a detailed essay of at least 1000 words about {TOPICS[(i + size) % len(TOPICS)]}. "
            "Use several paragraphs and do not stop early.")


def one(base, body, rec, start):
    u = urllib.parse.urlsplit(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=1800)
    start.wait()
    rec["t_send"] = time.perf_counter()
    text, usage, finish, chunks = [], None, None, 0
    try:
        conn.request("POST", u.path.rstrip("/") + "/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = conn.getresponse()
        if r.status != 200:
            rec["error"] = f"HTTP {r.status}: {r.read()[:300]!r}"
            return
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices") or []:
                piece = (c.get("delta") or {}).get("content") or ""
                piece += (c.get("delta") or {}).get("reasoning_content") or (c.get("delta") or {}).get("reasoning") or ""
                if piece:
                    now = time.perf_counter()
                    rec.setdefault("t_first", now)
                    rec["t_last"] = now
                    text.append(piece)
                    chunks += 1
                finish = c.get("finish_reason") or finish
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        conn.close()
    rec.update(text="".join(text), usage=usage, finish=finish, chunks=chunks)


def run(base, model, size, max_tokens, seed, tok):
    recs = [{"i": i, "prompt": prompt_for(size, i, seed)} for i in range(size)]
    start = threading.Event()
    threads = []
    for rec in recs:
        body = {"model": model, "messages": [{"role": "user", "content": rec["prompt"]}], "max_tokens": max_tokens,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        threads.append(threading.Thread(target=one, args=(base, body, rec, start), daemon=True))
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()
    errors = [r["error"] for r in recs if "error" in r]
    if errors:
        raise SystemExit(f"size {size}: {errors[0]}")
    for r in recs:
        r["tokens"] = tok.encode(r["text"], add_special_tokens=False)
        r["n_client"] = len(r["tokens"])
        r["n_server"] = (r.get("usage") or {}).get("completion_tokens")
    t0 = min(r["t_send"] for r in recs)
    first = min(r["t_first"] for r in recs)
    last = max(r["t_last"] for r in recs)
    n_tok = sum(r["n_client"] for r in recs)
    srv = [r["n_server"] for r in recs if r["n_server"] is not None]
    return {"size": size, "tokens": n_tok, "e2e_s": last - t0, "e2e_tps": n_tok / (last - t0),
            "decode_tps": sum(r["n_client"] - 1 for r in recs) / (last - first),
            "ttft_median_s": statistics.median(r["t_first"] - r["t_send"] for r in recs),
            "ttft_max_s": max(r["t_first"] - r["t_send"] for r in recs),
            "server_tokens": sum(srv) if len(srv) == len(recs) else None,
            "finish": {f: sum(1 for r in recs if r["finish"] == f) for f in {r["finish"] for r in recs}},
            "think_tags": sum("<think>" in r["text"] or "</think>" in r["text"] for r in recs),
            "replies": [{"i": r["i"], "prompt": r["prompt"], "text": r["text"], "tokens": r["tokens"]} for r in recs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tokenizer", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--sizes", default="1,2,4,8,16")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    run(a.base, a.model, 2, 32, 999, tok)  # warm-up (prompts never used again)
    results = []
    for size in [int(x) for x in a.sizes.split(",")]:
        r = run(a.base, a.model, size, a.max_tokens, a.seed, tok)
        results.append(r)
        print(f"{a.label:8s} {size:2d} at once: end to end {r['e2e_tps']:6.1f} tok/s ({r['tokens']} tokens in {r['e2e_s']:.1f} s), "
              f"decode {r['decode_tps']:6.1f} tok/s, first token median {r['ttft_median_s']:.2f} s; "
              f"server counted {r['server_tokens']}; finish {r['finish']}; think tags {r['think_tags']}", flush=True)
    with open(a.out, "w") as f:
        json.dump({"label": a.label, "base": a.base, "model": a.model, "max_tokens": a.max_tokens, "results": results}, f)


if __name__ == "__main__":
    main()
