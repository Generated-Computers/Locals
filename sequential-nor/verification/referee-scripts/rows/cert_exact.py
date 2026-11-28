#!/usr/bin/env python3 -I
"""Referee check, written from Definition "the two-row chain" of the paper (not from rows_check.c).

Exact rational arithmetic.  Input: a dump file with one line "BOX EA EB DA DB" and four lines
"C y v0..v15" (hex doubles; state index = x' + 2 o' + 4 x'' + 8 o'', symbol index = b + 2 w).

Box operator: (T g)(s) = max over e1,e2 in [EA,EB], d1 in [DA,DB] and d2 = d2(outcome of the lower
read) in [DA,DB] of the expected value of g after one step.  The expectation is affine in each of
e1, d1, e2, d2(i) separately and the coefficients of the inner values are >= 0, so the maximum is
  max_{(e1,d1) corner} sum_i p_i(e1,d1) * max_{(e2,d2) corner} sum_k p_k(e2,d2) g(next(i,k)),
which is computed exactly here.  For a single eps in [EA,EB] and d1,d2 in a subinterval the operator
of the paper is <= this one (>= for the minimum).

Checks for all admissible words z_1..z_K y:
  (C1) T_{z_1}..T_{z_K}[T_y(phi+c_y) - c_{z_K}] <= 0,   (C2) T-_{z_1}..T-_{z_K} c_{z_K} >= 0,
and |c| <= 4, c = 0 on closed states.
mode "prune": leave a branch when the function is already <= 0 (>= 0) everywhere (sound: T monotone,
T0 = 0).  mode "full": no pruning, every word of length K+1 is evaluated to the end.
usage: cert_exact.py dumpfile theta K [prune|full]
"""
import sys
from fractions import Fraction as F
from itertools import product

def f(v, o, b, w):
    if v == 0 and b == 1: return w
    if v == 1 and b == 1: return o & w
    if v == 0 and b == 0: return o | w
    return o

def read(pair, sym):
    v, o = pair; b, w = sym
    return (1 - (v | b), f(v, o, b, w))

STATES = list(product((0, 1), repeat=4))          # (x', o', x'', o'')
SYMS = [(0, 0), (1, 0), (0, 1), (1, 1)]           # (b, w); index = b + 2 w
def sidx(s): return s[0] + 2 * s[1] + 4 * s[2] + 8 * s[3]
def yidx(y): return y[0] + 2 * y[1]

def nxt(s, y):
    """3x3 table of next states: lower outcome i, upper outcome k (0: true read, 1: closed 1, 2: closed 0)."""
    tab = []
    for i in range(3):
        low = read((s[0], s[1]), (y, (1, 0), (0, 0))[i])
        row = []
        for k in range(3):
            upp = read((s[2], s[3]), (low, (1, 0), (0, 0))[k])
            row.append(low + upp)
        tab.append(row)
    return tab
NX = {(y, s): nxt(s, y) for y in SYMS for s in STATES}

def mix(sel, E, D, gk, g1, g0):
    return sel((1 - e) * gk + e * (d * g1 + (1 - d) * g0) for e in E for d in D)

def T(sel, E, D, y, g):
    out = {}
    for s in STATES:
        tab = NX[(y, s)]
        A = [mix(sel, E, D, g[tab[i][0]], g[tab[i][1]], g[tab[i][2]]) for i in range(3)]
        out[s] = mix(sel, E, D, A[0], A[1], A[2])
    return out

def main():
    fn, theta, K = sys.argv[1], F(float(sys.argv[2])), int(sys.argv[3])
    mode = sys.argv[4] if len(sys.argv) > 4 else "prune"
    c = {}
    for line in open(fn):
        t = line.split()
        if t[0] == "BOX":
            EA, EB, DA, DB = [F(float.fromhex(x)) for x in t[1:5]]
        elif t[0] == "C":
            vals = [F(float.fromhex(x)) for x in t[2:18]]
            y = SYMS[int(t[1])]; assert yidx(y) == int(t[1])
            c[y] = {s: vals[sidx(s)] for s in STATES}
    E, D = (EA, EB), (DA, DB)
    print("box eps [%.17g, %.17g] d [%.17g, %.17g] theta %.17g K %d mode %s" % (EA, EB, DA, DB, theta, K, mode))
    assert 0 < EA <= EB <= 1 and 0 <= DA <= DB <= 1
    for y in SYMS:
        for s in STATES:
            assert abs(c[y][s]) <= 4
            if s[1] == 0 and s[3] == 0: assert c[y][s] == 0
    phi = {s: s[3] - theta * s[1] for s in STATES}
    stat = {"napp": 0, "words": 0, "worst": None}

    def walk(sel, sign, g, first, depth):
        # g = T_{z_j}..T_{z_K}[..] with `depth` operators applied; first = z_j
        ok_all = all(sign * v <= 0 for v in g.values())
        if depth == K or (mode == "prune" and ok_all):
            stat["words"] += 1
            w = max(sign * v for v in g.values())
            if stat["worst"] is None or w > stat["worst"]: stat["worst"] = w
            return ok_all
        for z in SYMS:
            if z[0] == 1 and first[0] == 1: continue
            stat["napp"] += 1
            if not walk(sel, sign, T(sel, E, D, z, g), z, depth + 1): return False
        return True

    ok = True
    for h in SYMS:
        for y in SYMS:
            if h[0] == 1 and y[0] == 1: continue
            g = T(max, E, D, y, {s: phi[s] + c[y][s] for s in STATES})
            g = {s: g[s] - c[h][s] for s in STATES}
            stat["napp"] += 2
            if not walk(max, 1, T(max, E, D, h, g), h, 1):
                ok = False; print("C1 FAILS at z_K=%s y=%s" % (h, y))
    print("C1 %s: %d operator applications, %d leaves, largest final value %.3e" %
          ("holds" if ok else "FAILS", stat["napp"], stat["words"], float(stat["worst"])))
    ok1 = ok; stat.update(napp=0, words=0, worst=None); ok = True
    for h in SYMS:
        stat["napp"] += 1
        if not walk(min, -1, T(min, E, D, h, c[h]), h, 1):
            ok = False; print("C2 FAILS at z_K=%s" % (h,))
    print("C2 %s: %d operator applications, %d leaves, least final value %.3e" %
          ("holds" if ok else "FAILS", stat["napp"], stat["words"], -float(stat["worst"])))
    print("RESULT", "AGREES (C1, C2, C3 hold exactly on the box)" if ok and ok1 else "DISAGREES")

main()
