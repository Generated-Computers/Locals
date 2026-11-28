"""Adaptive adversary against the Kleene-prefix iteration and its relevance-pruned variant.

The pointer of gate k is chosen after the noise of gates < k is known: it goes to a zero-valued
position <= k-2 of largest depth.  Depths are computed with the recursions of Lemma 3.12(a) of
manuscript 001c:
   d(i) = 1 + d(j_i)                 if i is a reset,
        = max(d(i-1), 1 + d(j_i))    if i is neither a reset nor a NOT gate,
        = d(i-1)                     otherwise;
   D(i) = 1 + D(j_i)                 if i is a relevant reset,
        = min(D(i-1), 1 + D(j_i))    if i is an irrelevant reset,
        = max(D(i-1), 1 + D(j_i))    if i is relevant, neither a reset nor a NOT gate,
        = D(i-1)                     otherwise.
The number of rounds of an iteration is the largest depth.  Standard library only."""
import random, sys

def run(n, eps, target, rng):
    x = bytearray(n + 1); x[0] = 1                       # x_1 = 0 (gate 1 has pointer 0)
    d = [0] * (n + 1); D = [0] * (n + 1); d[1] = D[1] = 1
    depth = d if target == "d" else D
    best = 1                                             # deepest zero among positions <= k-2
    for k in range(2, n + 1):
        if k >= 3 and x[k - 2] == 0 and depth[k - 2] > depth[best]:
            best = k - 2
        j = rng.randrange(k) if rng.random() < eps else (best if k >= 3 else 0)
        xj, xp = x[j], x[k - 1]
        x[k] = 1 - (xp | xj)
        notgate = (j == k - 1)
        reset = (not notgate) and xj == 1
        relevant = (xp == 0)
        if reset: d[k] = 1 + d[j]
        elif not notgate: d[k] = max(d[k - 1], 1 + d[j])
        else: d[k] = d[k - 1]
        if reset and relevant: D[k] = 1 + D[j]
        elif reset: D[k] = min(D[k - 1], 1 + D[j])
        elif relevant and not notgate: D[k] = max(D[k - 1], 1 + D[j])
        else: D[k] = D[k - 1]
    return max(d), max(D)

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
print("%-6s %-9s %-22s %-22s %s" % ("eps", "n", "adversary vs KP: max d", "adversary vs pruned: max D", "bound (1-eps)eps n/(2+eps)"))
for eps in (0.5, 0.8, 0.9, 0.95):
    for n in (10**4, 10**5, 10**6):
        dk, _ = run(n, eps, "d", rng)
        _, Dp = run(n, eps, "D", rng)
        print("%-6.2f %-9d %-10d (%.4f n)   %-10d (%.4f n)      %.4f n" % (eps, n, dk, dk / n, Dp, Dp / n, (1 - eps) * eps / (2 + eps)))
# for comparison: an oblivious base at the same noise (pointers i-2), where 001c proves O(log n) above 3/4
def oblivious(n, eps, rng):
    x = bytearray(n + 1); x[0] = 1; d = [0] * (n + 1); d[1] = 1
    for k in range(2, n + 1):
        j = rng.randrange(k) if rng.random() < eps else max(0, k - 2)
        xj, xp = x[j], x[k - 1]; x[k] = 1 - (xp | xj)
        notgate = (j == k - 1); reset = (not notgate) and xj == 1
        d[k] = 1 + d[j] if reset else (max(d[k - 1], 1 + d[j]) if not notgate else d[k - 1])
    return max(d)
print("oblivious base j_i = i-2 at eps = 0.9:", [oblivious(n, 0.9, rng) for n in (10**4, 10**5, 10**6)])
