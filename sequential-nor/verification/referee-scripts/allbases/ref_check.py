#!/usr/bin/env python3
"""Referee's independent check of Section "All bases" (inequalities twoa, twob, generic, reachtwo).

Exact rational arithmetic.  For a base of length n and noise eps=a/b, all noise outcomes
(kept / resampled to p, for every gate) are summed with their exact probabilities.

Part 0: D from three sources (simulated relevance-pruned iteration, anchor definition, four-case
        recursion) on all parent arrays; Lemma twotype (a),(b),(c).
Part B: instance-density form, G* = sure event.  Every step of the derivation of (reachtwo), and
        (twoa), (twob), (generic), with r_c, eps*beta, eta replaced by the conditional densities.
Part A: literal form with the events G*_k and constants c, beta, eta=eps^2 c(1-c).
"""
import sys, random, itertools, time
from fractions import Fraction as Fr

# ---------------------------------------------------------------- D, three ways
def trace(j):
    x = [1]
    for i in range(1, len(j)):
        x.append(1 if (x[i-1] == 0 and x[j[i]] == 0) else 0)
    return x

def D_iter(j):
    """rounds of the relevance-pruned iteration (Definition kp:pruned-def), simulated."""
    n = len(j) - 1
    known = {0: 1}
    D = [None] * (n + 1); D[0] = 0
    t = 0
    while len(known) < n + 1:
        t += 1
        assert t <= n + 2
        resolved = [False] * (n + 1); rreset = [False] * (n + 1)
        for k in range(1, n + 1):
            if j[k] == k - 1:
                resolved[k] = True
            elif j[k] in known:
                resolved[k] = True
                if known[j[k]] == 1:
                    rreset[k] = True
        new = {}
        for i in range(0, n + 1):
            r = i
            while r > 0 and not rreset[r]:
                r -= 1
            vr = 1 if r == 0 else 0
            ok = True
            for k in range(r + 1, i + 1):
                if (vr ^ ((k - 1 - r) & 1)) == 0 and not resolved[k]:
                    ok = False; break
            if ok:
                new[i] = vr ^ ((i - r) & 1)
        for i in known:
            assert i in new and new[i] == known[i]
        for i in new:
            if D[i] is None:
                D[i] = t
        known = new
    return D

def D_def(j, x):
    """D from the anchor definition (kp:eq-D)."""
    n = len(j) - 1
    D = [0] * (n + 1)
    rel = lambda k: x[k-1] == 0
    isnot = lambda k: j[k] == k - 1
    isreset = lambda k: (not isnot(k)) and x[j[k]] == 1
    for i in range(1, n + 1):
        rp = i
        while rp > 0 and not (isreset(rp) and rel(rp)):
            rp -= 1
        anchors = [r for r in range(max(rp, 1), i + 1) if isreset(r)]
        if rp == 0:
            anchors.append(0)
        best = None
        for r in anchors:
            vals = []
            if r >= 1:
                vals.append(D[j[r]])
            for k in range(r + 1, i + 1):
                if rel(k) and not isnot(k):
                    vals.append(D[j[k]])
            V = max(vals) if vals else 0
            best = V if best is None else min(best, V)
        D[i] = 1 + best
    return D

def D_step(k, p, xs, Ds):
    """D(k) from Lemma status; xs has x_0..x_k, Ds has D_0..D_{k-1}."""
    if k == 1:
        return 1
    notg = (p == k - 1)
    reset = (not notg) and xs[p] == 1
    rel = xs[k-1] == 0
    if reset and rel:
        return 1 + Ds[p]
    if reset:
        return min(Ds[k-1], 1 + Ds[p])
    if rel and not notg:
        return max(Ds[k-1], 1 + Ds[p])
    return Ds[k-1]

def part0(nmax):
    cnt = 0
    for n in range(2, nmax + 1):
        for tail in itertools.product(*[range(i) for i in range(1, n + 1)]):
            j = [None] + list(tail)
            x = trace(j)
            D1 = D_iter(j); D2 = D_def(j, x)
            D3 = [0]
            for k in range(1, n + 1):
                D3.append(D_step(k, j[k], x, D3))
            assert D1 == D2 == D3, (j, D1, D2, D3)
            D = D1
            for k in range(2, n + 1):
                p = j[k]; notg = (p == k - 1)
                assert D[k] <= k
                for s in range(0, n + 1):
                    if D[k] > s + 1 and x[k] == 1:      # (a)
                        assert x[k-1] == 0 and x[p] == 0
                        assert D[k-1] > s + 1 or ((not notg) and D[p] > s and x[p] == 0)
                    if D[k] > s + 1 and x[k] == 0:      # (b)
                        alt1 = x[k-1] == 0 and (not notg) and D[p] > s and x[p] == 1
                        alt2 = x[k-1] == 1 and D[k-1] > s + 1 and (notg or x[p] == 0 or (D[p] > s and x[p] == 1))
                        assert alt1 or alt2, (j, k, s)
            cnt += 1
    print("part 0: D (iteration = anchors = recursion) and Lemma twotype(a,b,c) hold on all %d parent arrays, 2<=n<=%d" % (cnt, nmax))

# ---------------------------------------------------------------- tables over all noise outcomes
def tables(base, n, a, b, last_full=True):
    """T[m]: {(xs,Ds): mass} for gates 1..m;  R[m]: {(xs, D_m, kind_m): mass};  tot[m].
    mass is an integer; probability = mass/tot[m]; eps=a/b.  Every outcome (kept, or resampled to
    p in 0..m-1) of every gate is used; outcomes with the same (x,D) history are merged."""
    T = [dict() for _ in range(n + 1)]
    R = [dict() for _ in range(n + 1)]
    tot = [1] * (n + 1)
    T[0][((1,), (0,))] = 1
    R[0][((1,), 0, '-')] = 1
    for m in range(1, n + 1):
        tot[m] = tot[m-1] * b * m
        cur = T[m]; red = R[m]
        for (xs, Ds), mass in T[m-1].items():
            opts = [('K', base[m], mass * (b - a) * m)] + [('R', p, mass * a) for p in range(m)]
            for kind, p, w in opts:
                if w == 0:
                    continue
                xm = 1 if (xs[m-1] == 0 and xs[p] == 0) else 0
                xs2 = xs + (xm,)
                dm = D_step(m, p, xs2, Ds)
                if last_full or m < n:
                    key = (xs2, Ds + (dm,))
                    cur[key] = cur.get(key, 0) + w
                k2 = (xs2, dm, kind)
                red[k2] = red.get(k2, 0) + w
    return T, R, tot

def tables_brute(base, n, a, b):
    """the same by plain enumeration of all (n+1)! outcomes (cross-check of the merging)."""
    T = [dict() for _ in range(n + 1)]
    def rec(m, xs, Ds, mass):
        T[m][(xs, Ds)] = T[m].get((xs, Ds), 0) + mass
        if m == n:
            return
        g = m + 1
        for kind, p, w in [('K', base[g], (b - a) * g)] + [('R', p, a) for p in range(g)]:
            if w == 0:
                continue
            j = [None] * (g + 1)
            xg = 1 if (xs[g-1] == 0 and xs[p] == 0) else 0
            xs2 = xs + (xg,)
            dg = D_step(g, p, xs2, Ds)
            rec(g, xs2, Ds + (dg,), mass * w)
    rec(0, (1,), (0,), 1)
    # masses at depth m must be multiplied by the total mass of the remaining gates: not needed,
    # T[m] here is already the marginal with total prod_{i<=m} b*i.
    return T

# ---------------------------------------------------------------- Part B
class Stats:
    def __init__(self):
        self.worst = {}   # name -> (max lhs-rhs, witness)
        self.count = {}
        self.eqfail = []
        self.nt = {}; self.nt_count = {}; self.tight = {}; self.ratio = {}
    def le(self, name, lhs, rhs, wit):
        d = lhs - rhs
        self.count[name] = self.count.get(name, 0) + 1
        if name not in self.worst or d > self.worst[name][0]:
            self.worst[name] = (d, wit)
        if lhs > 0:      # non-trivial cases: LHS positive
            self.nt_count[name] = self.nt_count.get(name, 0) + 1
            if name not in self.nt or d > self.nt[name][0]:
                self.nt[name] = (d, wit, lhs, rhs)
            if d == 0:
                self.tight[name] = self.tight.get(name, 0) + 1
            q = lhs / rhs if rhs > 0 else Fr(10 ** 9)
            if name not in self.ratio or q > self.ratio[name][0]:
                self.ratio[name] = (q, wit)
    def eq(self, name, lhs, rhs, wit):
        self.count[name] = self.count.get(name, 0) + 1
        if lhs != rhs:
            self.eqfail.append((name, wit, lhs, rhs))
    def report(self, title):
        print(title)
        for name in sorted(self.count):
            if name in self.worst:
                d, wit = self.worst[name]
                flag = "VIOLATION" if d > 0 else "ok"
                print("  %-40s checks %8d  max(LHS-RHS) = %+.3e %s" % (name, self.count[name], float(d), flag))
                if d > 0:
                    print("        violation witness:", wit)
                if name in self.nt:
                    d2, w2, l2, r2 = self.nt[name]
                    print("        with LHS>0: %8d checks, %d with equality; max(LHS-RHS) = %+.6e (LHS %.6f RHS %.6f) at %s" % (self.nt_count[name], self.tight.get(name, 0), float(d2), float(l2), float(r2), w2))
                    print("        max LHS/RHS = %.6f at %s" % (float(self.ratio[name][0]), self.ratio[name][1]))
            else:
                print("  %-40s checks %8d   (equalities, all hold unless listed)" % (name, self.count[name]))
        if self.eqfail:
            print("  EQUALITY FAILURES:", len(self.eqfail), self.eqfail[:3])

def E(Tm, totm, f):
    acc = Fr(0)
    for (xs, Ds), mass in Tm.items():
        v = f(xs, Ds)
        if v:
            acc += v * mass
    return acc / totm

def partB(base, n, a, b, st, tag):
    eps = Fr(a, b); ke = 1 - eps
    T, R, tot = tables(base, n, a, b)
    # I^v_s(k), G* sure
    I = {}
    for k in range(0, n + 1):
        for (xs, dk, kind), mass in R[k].items():
            for s in range(0, dk):
                I[(k, s, xs[k])] = I.get((k, s, xs[k]), 0) + mass
    def Iv(k, s, v):
        return Fr(I.get((k, s, v), 0), tot[k])
    def It(k, s):
        return Iv(k, s, 0) + Iv(k, s, 1)
    O = lambda xs, g: sum(xs[:g+1])
    for k in range(2, n + 1):
        j = base[k]; d = k - j
        for s in range(0, k):
            wit = (tag, tuple(base[1:]), str(eps), k, s)
            # ---- (twoa), (twob), (generic) in instance form
            R1 = lambda xs: ke + eps * Fr(k - O(xs, k-1), k)
            R2 = lambda xs: ke + eps * Fr(k - O(xs, k-1) + 1, k)
            t1a = E(T[k-1], tot[k-1], lambda xs, Ds: R1(xs) if (xs[k-1] == 0 and Ds[k-1] > s + 1) else 0)
            t1b = E(T[k-1], tot[k-1], lambda xs, Ds: R2(xs) if (xs[k-1] == 1 and Ds[k-1] > s + 1) else 0)
            if d >= 3:
                H = lambda xs: eps * eps * Fr(O(xs, k-3), k-2) * Fr(k - 1 - O(xs, k-3), k-1)
                Fa = E(T[k-3], tot[k-3], lambda xs, Ds: (1 - H(xs)) if (xs[j] == 0 and Ds[j] > s) else 0)
                Fb = E(T[k-3], tot[k-3], lambda xs, Ds: (1 - H(xs)) if (xs[j] == 1 and Ds[j] > s) else 0)
            elif d == 2:
                Fa = E(T[k-2], tot[k-2], lambda xs, Ds: (1 - eps * Fr(k - 1 - O(xs, k-2), k-1)) if (xs[k-2] == 0 and Ds[k-2] > s) else 0)
                Fb = Iv(k-2, s, 1)
            else:
                Fa = Fb = Fr(0)
            S0 = sum(Iv(p, s, 0) for p in range(0, k-1)); S1 = sum(Iv(p, s, 1) for p in range(0, k-1))
            st.le("twoa", Iv(k, s+1, 1), t1a + ke * Fa + eps / k * S0, wit)
            st.le("twob", Iv(k, s+1, 0), t1b + ke * Fb + 2 * eps / k * S1, wit)
            st.le("twob with eps/k (sharper)", Iv(k, s+1, 0), t1b + ke * Fb + eps / k * S1, wit)
            if d != 2:
                st.le("generic", It(k, s+1), t1a + t1b + ke * (Fa + Fb) + 2 * eps / k * (S0 + S1), wit)
            # ---- (reachtwo) chain
            if d == 2 and k >= 3:
                LHS = It(k, s+1)
                Lres = Lk1 = Lk2 = Lk3 = 0
                for (xs, dk, kind), mass in R[k].items():
                    if dk > s + 1:
                        if kind == 'R':
                            Lres += mass
                        elif xs[k-2] == 1:
                            Lk1 += mass
                        elif xs[k-1] == 1:
                            Lk2 += mass
                        else:
                            Lk3 += mass
                Lres, Lk1, Lk2, Lk3 = [Fr(v, tot[k]) for v in (Lres, Lk1, Lk2, Lk3)]
                assert Lres + Lk1 + Lk2 + Lk3 == LHS
                Tk1, t1 = T[k-1], tot[k-1]; Tk2, t2 = T[k-2], tot[k-2]
                I1 = Iv(k-2, s, 1)
                A1 = Iv(k-1, s+1, 1)
                Z = E(Tk1, t1, lambda xs, Ds: 1 if (xs[k-2] == 0 and xs[k-1] == 0 and Ds[k-1] > s+1) else 0)
                Y = E(Tk1, t1, lambda xs, Ds: 1 if (xs[k-2] == 1 and Ds[k-1] > s+1) else 0)
                assert A1 + Z + Y == It(k-1, s+1)
                B0 = E(Tk1, t1, lambda xs, Ds: 1 if (xs[k-2] == 0 and Ds[k-2] > s and xs[k-1] == 0) else 0)
                U0 = E(Tk1, t1, lambda xs, Ds: 1 if (xs[k-2] == 0 and xs[k-1] == 0 and (Ds[k-1] > s+1 or Ds[k-2] > s)) else 0)
                R3 = lambda xs: eps * Fr(k - O(xs, k-1) + 1, k)
                ER3 = E(Tk1, t1, lambda xs, Ds: R3(xs) if Ds[k-1] > s+1 else 0)
                EWA = E(Tk1, t1, lambda xs, Ds: (ke + R3(xs)) if (xs[k-1] == 1 and Ds[k-1] > s+1) else 0)
                EWZ = E(Tk1, t1, lambda xs, Ds: (ke + R3(xs)) if (xs[k-2] == 0 and xs[k-1] == 0 and Ds[k-1] > s+1) else 0)
                EVY = E(Tk1, t1, lambda xs, Ds: R3(xs) if (xs[k-2] == 1 and Ds[k-1] > s+1) else 0)
                SI = sum(It(p, s) for p in range(0, k-1))
                st.eq("R kept, x_{k-2}=1 (equality)", Lk1, ke * I1, wit)
                st.eq("R kept, x=0,1 (equality)", Lk2, ke * A1, wit)
                st.eq("R kept, x=0,0 (equality)", Lk3, ke * U0, wit)
                st.le("R kept, x=0,0 union", Lk3, ke * (Z + B0), wit)
                st.le("R resampled case", Lres, ER3 + eps / k * SI, wit)
                st.le("R display after F3", LHS, ke * I1 + ke * B0 + EWA + EWZ + EVY + eps / k * SI, wit)
                # F3 at gate k
                zk1 = lambda xs: Fr(k - 1 - O(xs, k-2), k-1)
                F3 = E(Tk2, t2, lambda xs, Ds: (1 - eps * zk1(xs)) if (xs[k-2] == 0 and Ds[k-2] > s) else 0)
                st.le("R (F3) for B0", B0, F3, wit)
                # gate k-1
                jj = base[k-1]; dd = k - 1 - jj
                W1 = lambda xs: ke + eps * Fr(k - O(xs, k-2), k)
                W0 = lambda xs: ke + eps * Fr(k - O(xs, k-2) + 1, k)
                V0 = lambda xs: eps * Fr(k - O(xs, k-2) + 1, k)
                R1p = lambda xs: ke + eps * Fr(k - 1 - O(xs, k-2), k-1)
                R2p = lambda xs: ke + eps * Fr(k - 1 - O(xs, k-2) + 1, k-1)
                ind = 1 if dd >= 2 else 0
                # tight forms (events of x_{k-2} kept)
                a1 = E(Tk2, t2, lambda xs, Ds: W1(xs) * R1p(xs) if (xs[k-2] == 0 and Ds[k-2] > s+1) else 0)
                a2 = E(Tk2, t2, lambda xs, Ds: W1(xs) if (ind and xs[k-2] == 0 and xs[jj] == 0 and Ds[jj] > s) else 0)
                a3 = E(Tk2, t2, lambda xs, Ds: W1(xs) * sum(1 for p in range(0, k-2) if xs[p] == 0 and Ds[p] > s) if xs[k-2] == 0 else 0)
                st.le("R gate k-1: I^1_{s+1}(k-1)", EWA, a1 + ke * a2 + eps / (k-1) * a3, wit)
                b2 = E(Tk2, t2, lambda xs, Ds: W0(xs) if (ind and xs[k-2] == 0 and xs[jj] == 1 and Ds[jj] > s) else 0)
                b3 = E(Tk2, t2, lambda xs, Ds: W0(xs) * sum(1 for p in range(0, k-2) if xs[p] == 1 and Ds[p] > s) if xs[k-2] == 0 else 0)
                st.le("R gate k-1: Z", EWZ, ke * b2 + eps / (k-1) * b3, wit)
                c1 = E(Tk2, t2, lambda xs, Ds: V0(xs) * R2p(xs) if (xs[k-2] == 1 and Ds[k-2] > s+1) else 0)
                c3 = E(Tk2, t2, lambda xs, Ds: V0(xs) * sum(1 for p in range(0, k-2) if xs[p] == 1 and Ds[p] > s) if xs[k-2] == 1 else 0)
                st.le("R gate k-1: Y", EVY, c1 + eps / (k-1) * c3, wit)
                # final form as in the paper: events of x_{k-2} dropped in the pointer terms, s+1 -> s
                f1 = E(Tk2, t2, lambda xs, Ds: W1(xs) * R1p(xs) if (xs[k-2] == 0 and Ds[k-2] > s) else 0)
                f2 = E(Tk2, t2, lambda xs, Ds: V0(xs) * R2p(xs) if (xs[k-2] == 1 and Ds[k-2] > s) else 0)
                f3 = E(Tk2, t2, lambda xs, Ds: (W1(xs) if xs[jj] == 0 else W0(xs)) if (ind and Ds[jj] > s) else 0)
                f4 = E(Tk2, t2, lambda xs, Ds: sum((W1(xs) if xs[p] == 0 else W0(xs) + V0(xs)) for p in range(0, k-2) if Ds[p] > s))
                RHS = ke * I1 + ke * F3 + f1 + f2 + ke * f3 + eps / k * SI + eps / (k-1) * f4
                st.le("R FINAL (reachtwo, instance)", LHS, RHS, wit)

# ---------------------------------------------------------------- Part A: literal form with G*
def k0_of(c):
    k = 3
    while not (Fr(1, 4) - Fr(1, 4 * (k-2) ** 2) >= c * (1 - c)):
        k += 1
    return k

GRID_C = [Fr(1, 20), Fr(1, 10), Fr(1, 6), Fr(1, 5), Fr(1, 4), Fr(3, 10), Fr(1, 3), Fr(2, 5)]
GRID_B = [Fr(1, 4), Fr(3, 10), Fr(2, 5), Fr(9, 20)]
GRID_G0 = [0, 1, 2, 3]

def partA(base, n, a, b, st, tag, grid_c=GRID_C, grid_b=GRID_B, grid_g0=GRID_G0, literal_only=False):
    eps = Fr(a, b); ke = 1 - eps
    T, R, tot = tables(base, n, a, b, last_full=False)
    del T
    # reduce R[k] to {(x_k, D_k, (minc_g0...), (minb_g0...)) : mass}
    red = []
    for k in range(0, n + 1):
        dct = {}
        for (xs, dk, kind), mass in R[k].items():
            mc = []; mb = []
            for g0 in grid_g0:
                vc = Fr(10); vb = Fr(10)
                o = 0
                for g in range(1, k + 1):
                    o += xs[g-1]                # o = O_{g-1}
                    if g > g0:
                        vc = min(vc, Fr(o - 1, g)); vb = min(vb, Fr(g - o, g))
                mc.append(vc); mb.append(vb)
            key = (xs[k], dk, tuple(mc), tuple(mb))
            dct[key] = dct.get(key, 0) + mass
        red.append(dct)
    for gi, g0 in enumerate(grid_g0):
        for c in grid_c:
            k0 = k0_of(c); k1 = max(g0, k0) + 4
            for beta in grid_b:
                rc = 1 - eps * c; eta = eps * eps * c * (1 - c)
                c0 = rc * rc + ke * (1 - eps * beta); c1 = eps * (1 - c) * rc + ke
                I = {}
                for k in range(0, n + 1):
                    for (v, dk, mc, mb), mass in red[k].items():
                        if mc[gi] >= c and mb[gi] >= beta:
                            for s in range(0, dk):
                                I[(k, s, v)] = I.get((k, s, v), 0) + mass
                Iv = lambda k, s, v: Fr(I.get((k, s, v), 0), tot[k])
                It = lambda k, s: Iv(k, s, 0) + Iv(k, s, 1)
                for k in range(2, n + 1):
                    j = base[k]; d = k - j
                    lit_ab = k >= k1; lit_r = k > k1
                    ok_ab = k >= max(g0 + 3, k0)        # where the proofs of (twoa),(twob) apply
                    ok_r = k >= max(g0 + 2, 3)          # where the proof of (reachtwo) applies
                    for s in range(0, k):
                        wit = (tag, tuple(base[1:]), str(eps), "g0=%d c=%s beta=%s" % (g0, c, beta), k, s)
                        S0 = sum(Iv(p, s, 0) for p in range(0, k-1)); S1 = sum(Iv(p, s, 1) for p in range(0, k-1))
                        gam = (1 - eta) if d >= 3 else 1
                        i2 = 1 if d >= 2 else 0
                        la, ra = Iv(k, s+1, 1), rc * Iv(k-1, s+1, 0) + ke * (1 - eta) * i2 * Iv(j, s, 0) + eps / k * S0
                        lb, rb = Iv(k, s+1, 0), rc * Iv(k-1, s+1, 1) + ke * gam * i2 * Iv(j, s, 1) + 2 * eps / k * S1
                        lg = It(k, s+1)
                        rg = rc * It(k-1, s+1) + ke * (1 - eta) * (1 if d >= 3 else 0) * It(j, s) + 2 * eps / k * (S0 + S1)
                        for cond, suffix in ((lit_ab, " [literal range k>=k1]"), (ok_ab and not literal_only, " [k>=max(g0+3,k0)]"), ((not ok_ab) and not literal_only, " [outside proof range]")):
                            if cond:
                                st.le("A twoa" + suffix, la, ra, wit)
                                st.le("A twob" + suffix, lb, rb, wit)
                                if d != 2:
                                    st.le("A generic" + suffix, lg, rg, wit)
                        if d == 2 and k >= 3:
                            jj = base[k-1]; dd = k - 1 - jj
                            rr = c0 * Iv(k-2, s, 0) + c1 * Iv(k-2, s, 1) + rc * ke * (1 if dd >= 2 else 0) * It(jj, s) \
                                + 3 * eps / (k-1) * (S0 + S1)
                            for cond, suffix in ((lit_r, " [literal range k>k1]"), (ok_r and not literal_only, " [k>=max(g0+2,3)]"), ((not ok_r) and not literal_only, " [outside proof range]")):
                                if cond:
                                    st.le("A reachtwo" + suffix, lg, rr, wit)

# ---------------------------------------------------------------- bases
def structured(n):
    out = {}
    out["reach2"] = [None] + [max(i - 2, 0) for i in range(1, n + 1)]
    out["reach3"] = [None] + [max(i - 3, 0) for i in range(1, n + 1)]
    out["alt23"] = [None] + [max(i - (2 if i % 2 == 0 else 3), 0) for i in range(1, n + 1)]
    out["alt32"] = [None] + [max(i - (3 if i % 2 == 0 else 2), 0) for i in range(1, n + 1)]
    out["not"] = [None] + [i - 1 for i in range(1, n + 1)]
    out["zero"] = [None] + [0] * n
    out["alt2not"] = [None] + [(max(i - 2, 0) if i % 2 == 0 else i - 1) for i in range(1, n + 1)]
    out["altnot2"] = [None] + [(max(i - 2, 0) if i % 2 == 1 else i - 1) for i in range(1, n + 1)]
    out["alt2zero"] = [None] + [(max(i - 2, 0) if i % 2 == 0 else 0) for i in range(1, n + 1)]
    out["altzero2"] = [None] + [(max(i - 2, 0) if i % 2 == 1 else 0) for i in range(1, n + 1)]
    out["2,2,3"] = [None] + [max(i - (3 if i % 3 == 0 else 2), 0) for i in range(1, n + 1)]
    out["2,not,3"] = [None] + [max(i - (2, 1, 3)[i % 3], 0) for i in range(1, n + 1)]
    out["reach4"] = [None] + [max(i - 4, 0) for i in range(1, n + 1)]
    return out

def random_base(n, rng):
    j = [None]
    for i in range(1, n + 1):
        u = rng.random()
        if u < 0.35:
            p = max(i - 2, 0)
        elif u < 0.6:
            p = max(i - 3, 0)
        elif u < 0.72:
            p = i - 1
        elif u < 0.84:
            p = 0
        else:
            p = rng.randrange(i)
        j.append(p)
    return j

def main():
    mode = sys.argv[1]
    if mode == "part0":
        part0(int(sys.argv[2]))
        return
    if mode == "crosscheck":
        n = int(sys.argv[2])
        rng = random.Random(5)
        for name, base in list(structured(n).items()) + [("rnd%d" % i, random_base(n, rng)) for i in range(5)]:
            for a, b in ((1, 2), (73, 100)):
                T, R, tot = tables(base, n, a, b)
                Tb = tables_brute(base, n, a, b)
                for m in range(n + 1):
                    assert T[m] == Tb[m], (name, m)
                    assert sum(T[m].values()) == tot[m]
        print("crosscheck: merged tables = brute-force enumeration of all (n+1)! outcomes, n=%d" % n)
        return
    n = int(sys.argv[2])
    which = sys.argv[3]            # all | structured | random:<count>:<seed>
    epss = [tuple(int(v) for v in e.split('/')) for e in sys.argv[4].split(',')]
    bases = []
    if which == "all":
        for tail in itertools.product(*[range(i) for i in range(1, n + 1)]):
            bases.append(("all", [None] + list(tail)))
    elif which == "structured":
        bases = list(structured(n).items())
    else:
        _, cnt, seed = which.split(':')
        rng = random.Random(int(seed))
        bases = [("rnd", random_base(n, rng)) for _ in range(int(cnt))]
    st = Stats()
    t0 = time.time()
    for name, base in bases:
        for a, b in epss:
            if mode == "B":
                partB(base, n, a, b, st, name)
            elif mode == "A":
                partA(base, n, a, b, st, name)
            elif mode == "Alit":
                partA(base, n, a, b, st, name, grid_c=[Fr(1, 20), Fr(1, 10), Fr(1, 5), Fr(1, 4)], grid_b=[Fr(1, 4), Fr(2, 5), Fr(9, 20)], grid_g0=[2, 3, 4], literal_only=True)
    st.report("mode %s  n=%d  bases=%s (%d)  eps=%s   [%.0f s]" % (mode, n, which, len(bases), sys.argv[4], time.time() - t0))

if __name__ == "__main__":
    main()
