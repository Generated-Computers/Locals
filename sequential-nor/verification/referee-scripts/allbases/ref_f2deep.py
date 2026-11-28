#!/usr/bin/env python3
"""Referee: is eta improved by conditioning further below, WITHOUT an upper bound on the density of ones?
For every history up to gate k-1-m (m=2: (F2); m=4), the exact P[x_{k-1}=1 | history] against
  u_2 = a(1-r),   u_4 = a(1-r)(1+r a),
a = eps * (zeros among x_0..x_{k-1-m}) / (k-1),  r = 1 - eps * (ones among x_0..x_{k-1-m}) / (k-1)
(lower/upper bounds for the chance that a gate in between is resampled to a zero / is not resampled to a one)."""
import sys, os, itertools, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
from ref_check import tables, D_step, structured, random_base

def cond_x1(base, xs, k, eps, upto):
    """P[x_upto = 1 | x_0..x_{len(xs)-1}] exactly (values only; D not needed)."""
    m = len(xs) - 1
    if m == upto:
        return Fr(xs[upto])
    g = m + 1
    tot = Fr(0)
    for p, w in [(base[g], 1 - eps)] + [(p, eps / g) for p in range(g)]:
        xg = 1 if (xs[g-1] == 0 and xs[p] == 0) else 0
        tot += w * cond_x1(base, xs + (xg,), k, eps, upto)
    return tot

def run(n, bases, epss):
    worst = {2: None, 4: None}; cnt = {2: 0, 4: 0}
    for name, base in bases:
        for a, b in epss:
            eps = Fr(a, b)
            T, R, tot = tables(base, n, a, b, last_full=False)
            for m in (2, 4):
                for k in range(m + 2, n + 1):
                    h = k - 1 - m
                    seen = set()
                    for (xs, Ds) in T[h]:
                        if xs in seen: continue
                        seen.add(xs)
                        ones = sum(xs); zeros = h + 1 - ones
                        aa = eps * Fr(zeros, k - 1); rr = 1 - eps * Fr(ones, k - 1)
                        bound = aa * (1 - rr) if m == 2 else aa * (1 - rr) * (1 + rr * aa)
                        pr = cond_x1(base, xs, k, eps, k - 1)
                        cnt[m] += 1
                        d = bound - pr
                        if worst[m] is None or d > worst[m][0]:
                            worst[m] = (d, name, base[1:], str(eps), k, xs)
    for m in (2, 4):
        print("m=%d: %d (history, gate) pairs, max(bound - P[x_{k-1}=1|F_{k-1-m}]) = %+.3e %s" % (m, cnt[m], float(worst[m][0]), "ok" if worst[m][0] <= 0 else "VIOLATION %s" % (worst[m],)))

n = int(sys.argv[1])
rng = random.Random(3)
bases = list(structured(n).items()) + [("rnd", random_base(n, rng)) for _ in range(int(sys.argv[2]))]
run(n, bases, [(1, 2), (73, 100), (9, 10)])
