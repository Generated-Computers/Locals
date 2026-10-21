"""Two speed_audit.py result files side by side: speed per batch size, and whether the replies are the same text
(greedy decoding of the same model: the same tokens, until the two engines' arithmetic differs by a rounding
step), and whether either wrote degenerate text (one 8-token run repeated over and over).

    python compare.py mlxlm.json locals.json
"""
import json
import statistics
import sys


def repeated_share(tokens, n=8):
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def common_prefix(a, b):
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k


def main():
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    print(f"{'at once':>8} | {a['label']:>22} | {b['label']:>22} | ratio (end to end)")
    for ra, rb in zip(a["results"], b["results"]):
        print(f"{ra['size']:>8} | {ra['e2e_tps']:7.1f} e2e {ra['decode_tps']:7.1f} dec | {rb['e2e_tps']:7.1f} e2e "
              f"{rb['decode_tps']:7.1f} dec | {rb['e2e_tps'] / ra['e2e_tps']:.2f}x")
    same, prefixes, lengths, rep_a, rep_b = 0, [], [], [], []
    for ra, rb in zip(a["results"], b["results"]):
        for x, y in zip(ra["replies"], rb["replies"]):
            assert x["prompt"] == y["prompt"]
            same += x["tokens"] == y["tokens"]
            prefixes.append(common_prefix(x["tokens"], y["tokens"]))
            lengths.append((len(x["tokens"]), len(y["tokens"])))
            rep_a.append(repeated_share(x["tokens"]))
            rep_b.append(repeated_share(y["tokens"]))
    n = len(prefixes)
    print(f"\nreplies: {n}; identical token for token: {same}; common start median {statistics.median(prefixes)} "
          f"tokens (min {min(prefixes)}); lengths {a['label']} {sum(l[0] for l in lengths)}, {b['label']} "
          f"{sum(l[1] for l in lengths)} tokens")
    print(f"repeated 8-token runs, mean share: {a['label']} {statistics.mean(rep_a):.3f}, {b['label']} {statistics.mean(rep_b):.3f}"
          f" (max {max(rep_a):.3f} / {max(rep_b):.3f})")
    x, y = a["results"][-1]["replies"][0], b["results"][-1]["replies"][0]
    print(f"\nfirst reply at the largest size, {a['label']}:\n{x['text'][:400]}\n\n{b['label']}:\n{y['text'][:400]}")


if __name__ == "__main__":
    main()
