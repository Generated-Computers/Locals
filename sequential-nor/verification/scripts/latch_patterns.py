#!/usr/bin/env python3
"""Ideal noise levels of generalized latch chains (Remark "the level 4/9" of the note).

A chain is a periodic pattern of pairs (set check, reset check). A set check reads the rail of the
chain itself ('P') or position 1 ('0'); a reset check reads the other rail ('Q') or position 1.
In the ideal model (density of ones varpi(eps), independent reads, infinitely long chains) the
script finds, for every pattern of at most six pairs, the noise level at which the multiplier of
the difference of the two rails at the diagonal fixed point equals 1. The plain latch "PQ" has the
level 4/9, and no other pattern has a higher one.

Usage: python3 -I latch_patterns.py      (about 10 seconds)
"""
import math, itertools

def varpi(e): return ((2*e+1)-math.sqrt(4*e+1))/(2*e)

def stat(e, pat, x, y):
    """stationary probability that the state after a full period is 1, when the own rail has
    density x and the other rail density y"""
    A = e*varpi(e); c = 1-e
    a, b = 1.0, 0.0                       # state after the pairs so far: p -> a p + b
    for (ss, rs) in pat:
        s = A+c*(x if ss == 'P' else 0.0)
        r = A+c*(y if rs == 'Q' else 0.0)
        a, b = (1-r)*(1-s)*a, (1-r)*(s+(1-s)*b)
    return b/(1-a)

def mult(e, pat):
    p = 0.4
    for _ in range(400): p = stat(e, pat, p, p)
    h = 1e-6
    return (stat(e, pat, p+h, p-h)-stat(e, pat, p-h, p+h))/(2*h), p

def level(pat):
    lo, hi = 0.02, 0.749
    if mult(lo, pat)[0] <= 1: return 0.0
    if mult(hi, pat)[0] > 1: return hi
    for _ in range(40):
        mid = (lo+hi)/2
        if mult(mid, pat)[0] > 1: lo = mid
        else: hi = mid
    return lo

if __name__ == "__main__":
    res = []
    for L in range(1, 7):
        for combo in itertools.product(itertools.product(['P', '0'], ['Q', '0']), repeat=L):
            if not any(c[0] == 'P' for c in combo) or not any(c[1] == 'Q' for c in combo): continue
            res.append((level(list(combo)), combo))
    res.sort(reverse=True)
    print("%d patterns; the twelve highest levels:" % len(res))
    for t, combo in res[:12]:
        print("  %.5f  %s" % (t, " ".join(a+b for a, b in combo)))
    plain = [t for t, combo in res if all(c == ('P', 'Q') for c in combo)]
    others = [t for t, combo in res if not all(c == ('P', 'Q') for c in combo)]
    print("plain latch: %.6f (4/9 = %.6f); highest level of any other pattern: %.5f" % (max(plain), 4/9, max(others)))
    assert abs(max(plain)-4/9) < 1e-6 and max(others) < 4/9
