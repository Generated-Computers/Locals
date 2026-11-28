#!/usr/bin/env python3
"""Referee: (F2) and (F3) pointwise, in instance form, for every history:
 (F2) P[x_{k-1}=1 | x_0..x_{k-3}] >= eps^2 (O_{k-3}/(k-2)) ((k-1-O_{k-3})/(k-1))            (k>=3)
 (F3) P[x_{k-1}=1 | x_0..x_{k-2}] >= eps (k-1-O_{k-2})/(k-1)   when x_{k-2}=0                (k>=2)
 (F1) P[gate k reads 0 | hist k-1] <= (1-eps)+eps(k-O_{k-1})/k ; P[kept or res. to a zero or to k-1] <= 1-eps(O_{k-1}-1)/k ;
      P[res. to a zero or to k-1] <= eps(k-O_{k-1}+1)/k."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
from ref_check import tables, structured, random_base
from ref_f2deep_lib import cond_x1
n = int(sys.argv[1]); rng = random.Random(4)
bases = list(structured(n).items()) + [("rnd", random_base(n, rng)) for _ in range(int(sys.argv[2]))]
w2 = w3 = w1 = None; c2 = c3 = c1 = 0; tight2 = tight3 = 0
for name, base in bases:
    for a, b in ((1, 2), (73, 100), (9, 10), (1, 1)):
        eps = Fr(a, b)
        T, R, tot = tables(base, n, a, b, last_full=False)
        for k in range(2, n + 1):
            if k >= 3:
                for xs in set(x for (x, d) in T[k-3]):
                    O = sum(xs); bd = eps * eps * Fr(O, k-2) * Fr(k-1-O, k-1)
                    d = bd - cond_x1(base, xs, eps, k-1); c2 += 1; tight2 += (d == 0)
                    if w2 is None or d > w2[0]: w2 = (d, name, base[1:], str(eps), k, xs)
            for xs in set(x for (x, d) in T[k-2]):
                if xs[k-2] == 0:
                    O = sum(xs); bd = eps * Fr(k-1-O, k-1)
                    d = bd - cond_x1(base, xs, eps, k-1); c3 += 1; tight3 += (d == 0)
                    if w3 is None or d > w3[0]: w3 = (d, name, base[1:], str(eps), k, xs)
            for xs in set(x for (x, d) in T[k-1]):
                O = sum(xs)
                opts = [(base[k], 1 - eps, 'K')] + [(p, eps / k, 'R') for p in range(k)]
                p_read0 = sum(w for p, w, kd in opts if xs[p] == 0)
                p_second = sum(w for p, w, kd in opts if kd == 'K' or xs[p] == 0 or p == k-1)
                p_third = sum(w for p, w, kd in opts if kd == 'R' and (xs[p] == 0 or p == k-1))
                for d in (p_read0 - ((1-eps) + eps*Fr(k-O, k)), p_second - (1 - eps*Fr(O-1, k)), p_third - eps*Fr(k-O+1, k)):
                    c1 += 1
                    if w1 is None or d > w1[0]: w1 = (d, name, base[1:], str(eps), k, xs)
print("(F2) pointwise: %d histories, %d tight, max(bound - prob) = %+.3e" % (c2, tight2, float(w2[0])), "ok" if w2[0] <= 0 else w2)
print("(F3) pointwise: %d histories, %d tight, max(bound - prob) = %+.3e" % (c3, tight3, float(w3[0])), "ok" if w3[0] <= 0 else w3)
print("(F1) three bounds pointwise: %d checks, max(prob - bound) = %+.3e" % (c1, float(w1[0])), "ok" if w1[0] <= 0 else w1)
