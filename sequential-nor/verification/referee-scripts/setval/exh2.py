#!/usr/bin/env python3
"""Referee, second script: finer comparisons for item 4.
 (1) three-valued (Kleene) iteration with exact short reads: is K3^(s) = {D'_R <= s}?
 (2) smallest array with K^(s) != {D'_R<=s} (set-valued strictly ahead of three-valued logic)
 (3) is the number of applications of G_R monotone in R?  (set-valued rounds are, see exh.py)
Usage: python3 -I exh2.py NMIN NMAX [RMAX]
"""
import sys
from itertools import product
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from exhlib import trace, status, depth_cases, setval, G

U = 2


def nor3(a, b):
    if a == 1 or b == 1:
        return 0
    if a == 0 and b == 0:
        return 1
    return U


def kleene(n, j, R):
    """three-valued iteration: long reads of unknown positions are U, short reads are the current values."""
    K = {0: 1}
    hist = [set(K)]
    while len(K) < n + 1:
        z = [U] * (n + 1)
        z[0] = 1
        for i in range(1, n + 1):
            p = j[i]
            r = z[p] if p >= i - R else K.get(p, U)
            z[i] = nor3(z[i - 1], r)
        K2 = {i: z[i] for i in range(n + 1) if z[i] != U}
        if len(K2) == len(K):
            return None
        K = K2
        hist.append(set(K))
    return hist


def main():
    nmin, nmax = int(sys.argv[1]), int(sys.argv[2])
    rmax = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    first = {}
    for n in range(nmin, nmax + 1):
        cnt = 0
        ne3 = {R: 0 for R in range(1, rmax + 1)}
        ahead = {R: 0 for R in range(1, rmax + 1)}
        nonmono = 0
        for tail in product(*[range(i) for i in range(1, n + 1)]):
            j = [None] + list(tail)
            cnt += 1
            x = trace(n, j)
            notg, reset, rel = status(n, j, x)
            prevapps = None
            for R in range(1, rmax + 1):
                short = lambda k: j[k] >= k - R
                Dprime = depth_cases(n, j, notg, reset, rel, lambda k, d: max(1, d) if short(k) else 1 + d)
                h3 = kleene(n, j, R)
                if h3 is None:
                    first.setdefault(('kleene stuck', R), (n, j[1:]))
                    continue
                bad = False
                for s in range(max(Dprime) + 1):
                    A = h3[min(s, len(h3) - 1)]
                    if A != {i for i in range(n + 1) if Dprime[i] <= s}:
                        bad = True
                if bad:
                    ne3[R] += 1
                    first.setdefault(("K3 != {D'_R<=s}", R), (n, j[1:]))
                hist, stop = setval(n, j, R, extra=0)
                a = False
                for s in range(len(hist)):
                    B = h3[min(s, len(h3) - 1)]
                    if not B <= hist[s][0]:
                        first.setdefault(('K3 not inside K', R), (n, j[1:], s))
                    if B != hist[s][0]:
                        a = True
                if a:
                    ahead[R] += 1
                    first.setdefault(('set-valued strictly ahead of three-valued', R), (n, j[1:]))
                y = [1] + [0] * n
                s = 0
                while True:
                    z = G(n, j, R, y)
                    if z == y:
                        break
                    y = z
                    s += 1
                apps = s + 1
                if prevapps is not None and apps > prevapps:
                    nonmono += 1
                    first.setdefault(('applications of G_R increase with R', R), (n, j[1:], prevapps, apps))
                prevapps = apps
        print("n=%d arrays=%d  K3!={D'<=s}: %s   set-valued strictly ahead: %s   G_R apps not monotone in R: %d"
              % (n, cnt, ne3, ahead, nonmono), flush=True)
    for k in sorted(first, key=str):
        print("  first instance of", k, ":", first[k])


main()
