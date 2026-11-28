#!/usr/bin/env python3 -I
"""Referee check, written from Definition "the three-row chain" of the paper (not from rows3_check.c).

Exact arithmetic.  Input: a dump with one line "BOX EA EB DA DB", four lines "C y v0..v63" (hex
doubles; state index = x1 + 2 o1 + 4 x2 + 8 o2 + 16 x3 + 32 o3, symbol index = b + 2 w).

Box operator: (T g)(s) = max over the noise level in [EA,EB] and the density in [DA,DB] of each of
the three reads, where the choice for row 2 may depend on the outcome of row 1 and the choice for
row 3 on the outcomes of rows 1 and 2.  The expectation is affine in each parameter and the weights
are >= 0, so
  T g = max_{corner} sum_{k1} p_{k1} max_{corner} sum_{k2} p_{k2} max_{corner} sum_{k3} p_{k3} g(next(k1,k2,k3)).
All doubles are dyadic rationals; they are turned into integers over a common power of two (exact,
asserted), and every value is an integer over a power of two.  A self test compares one application
with the same formula in fractions.Fraction and with plain expectations at points of the box.

Checks for all admissible words z_1..z_K y:
  (C1) T_{z_1}..T_{z_K}[T_y(phi+c_y) - c_{z_K}] <= 0,   (C2) T-_{z_1}..T-_{z_K} c_{z_K} >= 0,
and |c| <= 4, c = 0 on closed states; phi = o3 - theta*o2.
mode "prune": leave a branch when the function is already <= 0 (>= 0) everywhere (sound: T monotone,
T0 = 0).  mode "full": no pruning.
usage: cert_exact3.py dumpfile theta K [prune|full]
"""
import sys, random
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

STATES = [t[::-1] for t in product((0, 1), repeat=6)]   # (x1, o1, x2, o2, x3, o3), listed in the order of the index
SYMS = [(0, 0), (1, 0), (0, 1), (1, 1)]           # (b, w); index = b + 2 w
def sidx(s): return sum(s[i] << i for i in range(6))
def yidx(y): return y[0] + 2 * y[1]
IDX = {s: sidx(s) for s in STATES}
assert [IDX[s] for s in STATES] == list(range(64))

def nxt(s, y):
    """next state index for the outcomes (k1,k2,k3); 0: reads its target, 1: reads (1,0), 2: reads (0,0)"""
    tab = {}
    for k1 in range(3):
        p1 = read((s[0], s[1]), (y, (1, 0), (0, 0))[k1])
        for k2 in range(3):
            p2 = read((s[2], s[3]), (p1, (1, 0), (0, 0))[k2])
            for k3 in range(3):
                p3 = read((s[4], s[5]), (p2, (1, 0), (0, 0))[k3])
                tab[(k1, k2, k3)] = IDX[p1 + p2 + p3]
    return [[[tab[(a, b, c)] for c in range(3)] for b in range(3)] for a in range(3)]
NX = [[nxt(s, y) for s in STATES] for y in SYMS]   # NX[yidx][sidx][k1][k2][k3]

def T_int(sel, CO, y, g):
    """g: list of 64 integers; CO: the four corners as integer weight triples (true read, closed 1, closed 0)"""
    out = []
    for s in range(64):
        t = NX[y][s]; A = []
        for k1 in range(3):
            B = []
            for k2 in range(3):
                u = t[k1][k2]; a, b, c = g[u[0]], g[u[1]], g[u[2]]
                B.append(sel(p * a + q * b + r * c for p, q, r in CO))
            A.append(sel(p * B[0] + q * B[1] + r * B[2] for p, q, r in CO))
        out.append(sel(p * A[0] + q * A[1] + r * A[2] for p, q, r in CO))
    return out

def T_frac(sel, E, D, y, g):
    mix = lambda a, b, c: sel((1 - e) * a + e * (d * b + (1 - d) * c) for e in E for d in D)
    out = []
    for s in range(64):
        t = NX[y][s]
        out.append(mix(*[mix(*[mix(*[g[t[k1][k2][k3]] for k3 in range(3)]) for k2 in range(3)]) for k1 in range(3)]))
    return out

def E_point(e, d, y, g):
    pr = (1 - e, e * d, e * (1 - d)); out = []
    for s in range(64):
        t = NX[y][s]
        out.append(sum(pr[a] * pr[b] * pr[c] * g[t[a][b][c]] for a in range(3) for b in range(3) for c in range(3)))
    return out

def main():
    fn, theta, K = sys.argv[1], F(float(sys.argv[2])), int(sys.argv[3])
    mode = sys.argv[4] if len(sys.argv) > 4 else "prune"
    c = [None] * 4
    for line in open(fn):
        t = line.split()
        if not t: continue
        if t[0] == "BOX": EA, EB, DA, DB = [F(float.fromhex(x)) for x in t[1:5]]
        elif t[0] == "C":
            assert len(t) == 66
            c[int(t[1])] = [F(float.fromhex(x)) for x in t[2:66]]
    print("box eps [%.17g, %.17g] d [%.17g, %.17g] theta %.17g K %d mode %s" % (EA, EB, DA, DB, theta, K, mode))
    assert 0 < EA <= EB <= 1 and 0 <= DA <= DB <= 1
    for y in range(4):
        for s in STATES:
            assert abs(c[y][IDX[s]]) <= 4
            if s[1] == 0 and s[3] == 0 and s[5] == 0: assert c[y][IDX[s]] == 0
    print("C3 holds: |c| <= 4 (largest %.6f), c = 0 on the 8 closed states" % max(float(abs(v)) for cc in c for v in cc))
    phi = [s[5] - theta * s[3] for s in STATES]
    # common scales
    S = 0
    for v in (EA, EB, DA, DB):
        while (v * (1 << S)).denominator != 1: S += 1
    ONE = 1 << S
    CO = []
    for e in (EA, EB):
        for d in (DA, DB):
            ei, di = int(e * ONE), int(d * ONE)
            CO.append(((ONE - ei) * ONE, ei * di, ei * (ONE - di)))       # weights over 2^(2S)
    Q0 = 0
    for v in [theta] + [x for cc in c for x in cc]:
        while (v * (1 << Q0)).denominator != 1: Q0 += 1
    toint = lambda g, q: [int(v * (1 << q)) for v in g]
    STEP = 6 * S                                                           # one application of T: scale grows by 2^(6S)
    # self test
    rnd = random.Random(1)
    g0 = [phi[i] + c[2][i] for i in range(64)]
    for y in range(4):
        for sel in (max, min):
            a = T_int(sel, CO, y, toint(g0, Q0)); b = T_frac(sel, (EA, EB), (DA, DB), y, g0)
            assert all(F(a[i], 1 << (Q0 + STEP)) == b[i] for i in range(64))
            for _ in range(20):
                e = EA + (EB - EA) * F(rnd.randrange(0, 1001), 1000); d = DA + (DB - DA) * F(rnd.randrange(0, 1001), 1000)
                m = E_point(e, d, y, g0)
                assert all((m[i] <= b[i]) if sel is max else (m[i] >= b[i]) for i in range(64))
    print("self test passed (integers = fractions; plain expectations at points of the box lie inside)")

    stat = {"napp": 0, "words": 0, "worst": None}
    def walk(sel, sign, g, q, first, depth):
        ok_all = all(sign * v <= 0 for v in g)
        if depth == K or (mode == "prune" and ok_all):
            stat["words"] += 1
            w = F(max(sign * v for v in g), 1 << q)
            if stat["worst"] is None or w > stat["worst"]: stat["worst"] = w
            return ok_all
        for z in range(4):
            if SYMS[z][0] == 1 and SYMS[first][0] == 1: continue
            stat["napp"] += 1
            if not walk(sel, sign, T_int(sel, CO, z, g), q + STEP, z, depth + 1): return False
        return True

    ok1 = True
    for h in range(4):
        for y in range(4):
            if SYMS[h][0] == 1 and SYMS[y][0] == 1: continue
            g = T_int(max, CO, y, toint([phi[i] + c[y][i] for i in range(64)], Q0)); q = Q0 + STEP
            ch = toint(c[h], q)
            g = [g[i] - ch[i] for i in range(64)]
            stat["napp"] += 2
            if not walk(max, 1, T_int(max, CO, h, g), q + STEP, h, 1):
                ok1 = False; print("C1 FAILS at z_K=%d y=%d" % (h, y))
        print("  C1 z_K=%d done, %d applications so far" % (h, stat["napp"]), flush=True)
    print("C1 %s: %d operator applications, %d leaves, largest final value %.3e" %
          ("holds" if ok1 else "FAILS", stat["napp"], stat["words"], float(stat["worst"])))
    stat.update(napp=0, words=0, worst=None); ok2 = True
    for h in range(4):
        stat["napp"] += 1
        if not walk(min, -1, T_int(min, CO, h, toint(c[h], Q0)), Q0 + STEP, h, 1):
            ok2 = False; print("C2 FAILS at z_K=%d" % h)
    print("C2 %s: %d operator applications, %d leaves, least final value %.3e" %
          ("holds" if ok2 else "FAILS", stat["napp"], stat["words"], -float(stat["worst"])))
    print("RESULT", "AGREES (C1, C2, C3 hold exactly on the box)" if ok1 and ok2 else "DISAGREES")

main()
