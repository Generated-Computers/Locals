#!/usr/bin/env python3
"""Referee's exhaustive check of Definition def:setval / Proposition prop:setval.
Usage: python3 -I exh.py NMIN NMAX [RMAX]
Everything is implemented literally from the definitions (windows as tuples, sets of windows).
"""
import sys
from itertools import product


def nor(a, b):
    return 1 if (a == 0 and b == 0) else 0


def trace(n, j):
    x = [0] * (n + 1)
    x[0] = 1
    for i in range(1, n + 1):
        x[i] = nor(x[i - 1], x[j[i]])
    return x


def status(n, j, x):
    notg = [False] * (n + 1)
    reset = [False] * (n + 1)
    rel = [False] * (n + 1)
    for k in range(1, n + 1):
        notg[k] = (j[k] == k - 1)
        reset[k] = (not notg[k]) and x[j[k]] == 1
        rel[k] = (x[k - 1] == 0)
    return notg, reset, rel


def depth_paper(n, j, x, notg, reset, rel):
    """D by the paper's own definition (kp:eq-V, kp:eq-D): anchors."""
    D = [0] * (n + 1)
    for i in range(1, n + 1):
        # rho'(i): largest r<=i that is 0 or a relevant reset
        rp = 0
        for r in range(i, 0, -1):
            if reset[r] and rel[r]:
                rp = r
                break
        anchors = [r for r in range(max(rp, 1), i + 1) if reset[r]]
        if rp == 0:
            anchors.append(0)
        best = None
        for r in anchors:
            vals = []
            if r != 0:
                vals.append(D[j[r]])
            for k in range(r + 1, i + 1):
                if rel[k] and not notg[k]:
                    vals.append(D[j[k]])
            V = max(vals) if vals else 0
            if best is None or V < best:
                best = V
        D[i] = 1 + best
    return D


def depth_cases(n, j, notg, reset, rel, cost, d1=1):
    """four cases of lem:status; cost(k, Dp) replaces 1+D(p)."""
    D = [0] * (n + 1)
    if n >= 1:
        D[1] = d1
    for k in range(2, n + 1):
        c = cost(k, D[j[k]])
        if reset[k] and rel[k]:
            D[k] = c
        elif reset[k]:
            D[k] = min(D[k - 1], c)
        elif rel[k] and not notg[k]:
            D[k] = max(D[k - 1], c)
        else:
            D[k] = D[k - 1]
    return D


def phi(n, j, R, K, v):
    """One round, literally: returns (K', w, S)."""
    sigma0 = tuple([0] * (R - 1) + [1])
    S = [None] * (n + 1)
    S[0] = {sigma0}
    for i in range(1, n + 1):
        new = set()
        for sg in S[i - 1]:
            # sg stands for (z_{i-R},...,z_{i-1})
            if j[i] >= i - R:
                B = {sg[j[i] - (i - R)]}
            elif j[i] in K:
                B = {v[j[i]]}
            else:
                B = {0}
            a = sg[-1]
            for b in B:
                new.add(sg[1:] + (nor(a, b),))
        S[i] = new
    Kp = set()
    w = {}
    for i in range(n + 1):
        last = {sg[-1] for sg in S[i]}
        if len(last) == 1:
            Kp.add(i)
            w[i] = next(iter(last))
    return Kp, w, S


def setval(n, j, R, extra=2):
    """Returns list of (K^(s), v^(s)) for s=0..stop(+extra), and the stopping s."""
    full = set(range(n + 1))
    K, v = {0}, {0: 1}
    hist = [(K, v)]
    stop = None
    s = 0
    while True:
        if K == full and stop is None:
            stop = s
        if stop is not None and s >= stop + extra:
            break
        K2, v2, _ = phi(n, j, R, K, v)
        if stop is None and K2 == K and v2 == v:
            return hist, None  # stuck
        K, v = K2, v2
        hist.append((K, v))
        s += 1
    return hist, stop


def G(n, j, R, y):
    z = [0] * (n + 1)
    z[0] = 1
    for i in range(1, n + 1):
        r = z[j[i]] if j[i] >= i - R else y[j[i]]
        z[i] = nor(z[i - 1], r)
    return z


def main():
    nmin, nmax = int(sys.argv[1]), int(sys.argv[2])
    rmax = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    fails = {}
    first = {}

    def fail(tag, n, j, R, info=""):
        key = tag
        if key not in fails:
            fails[key] = (n, j[1:], R, info)
            print("FAIL %-28s n=%d j=%s R=%s %s" % (tag, n, j[1:], R, info), flush=True)

    for n in range(nmin, nmax + 1):
        cnt = 0
        st = {}  # statistics per R
        for R in range(1, rmax + 1):
            st[R] = dict(rounds_eq_DR1=0, rounds_eq_D=0, rounds_gt_Dp=0, apps_eq_r1=0, apps_gt_r=0,
                         maxrounds=0, rounds_lt_maxDprime=0, R1_K_ne_D=0, DR0=0, rounds_eq_ceil=0,
                         apps_lt_rounds=0, maxgap=0, rounds_eq_DR1_alt=0)
        for tail in product(*[range(i) for i in range(1, n + 1)]):
            j = [None] + list(tail)
            cnt += 1
            x = trace(n, j)
            notg, reset, rel = status(n, j, x)
            Dp = depth_paper(n, j, x, notg, reset, rel)
            D = depth_cases(n, j, notg, reset, rel, lambda k, d: 1 + d)
            if D != Dp:
                fail("lem:status vs definition of D", n, j, None, "%s %s" % (D, Dp))
            prevK = None
            for R in range(1, rmax + 1):
                short = lambda k: j[k] >= k - R
                DR = depth_cases(n, j, notg, reset, rel, lambda k, d: (0 if short(k) else 1) + d)
                DRalt = depth_cases(n, j, notg, reset, rel, lambda k, d: (0 if short(k) else 1) + d, d1=0)
                Dprime = depth_cases(n, j, notg, reset, rel, lambda k, d: max(1, d) if short(k) else 1 + d)
                if any(DR[i] > D[i] for i in range(n + 1)):
                    fail("(b) D_R<=D", n, j, R)
                if any(not (DR[i] <= Dprime[i] <= min(D[i], DR[i] + 1)) for i in range(n + 1)):
                    fail("extra: D_R<=D'_R<=min(D,D_R+1)", n, j, R)
                if any(DR[i] == 0 for i in range(1, n + 1)):
                    st[R]['DR0'] += 1
                hist, stop = setval(n, j, R)
                if stop is None:
                    fail("iteration stuck", n, j, R)
                    continue
                full = set(range(n + 1))
                # (a)
                for s, (K, v) in enumerate(hist):
                    if 0 not in K:
                        fail("0 not in K", n, j, R, "s=%d" % s)
                    if set(v.keys()) != K:
                        fail("v domain", n, j, R)
                    if any(v[p] != x[p] for p in K):
                        fail("(a) soundness", n, j, R, "s=%d" % s)
                    if s + 1 < len(hist) and not K <= hist[s + 1][0]:
                        fail("(a) monotone", n, j, R, "s=%d" % s)
                    if s >= stop and K != full:
                        fail("after stop not full", n, j, R)
                # (b) second claim, every s
                for s in range(len(hist) - 1):
                    Kn = hist[s + 1][0]
                    for i in range(n + 1):
                        if DR[i] <= s and i not in Kn:
                            fail("(b) D_R(i)<=s => i in K^(s+1)", n, j, R, "s=%d i=%d" % (s, i))
                        if DRalt[i] <= s and i not in Kn:
                            fail("extra: same with D_R(1)=0", n, j, R, "s=%d i=%d" % (s, i))
                for s in range(len(hist)):
                    Ks = hist[s][0]
                    for i in range(n + 1):
                        if D[i] <= s and i not in Ks:
                            fail("extra: D(i)<=s => i in K^(s)", n, j, R, "s=%d i=%d" % (s, i))
                        if Dprime[i] <= s and i not in Ks:
                            fail("extra: D'_R(i)<=s => i in K^(s)", n, j, R, "s=%d i=%d" % (s, i))
                    # thm:table(c) analogue
                    c = min(n, s * (R + 1))
                    if not set(range(c + 1)) <= Ks:
                        fail("extra: [0,s(R+1)] in K^(s)", n, j, R, "s=%d" % s)
                    if s < stop and any(DR[i] <= s and i not in Ks for i in range(n + 1)):
                        first.setdefault(('shift needed at s>=1' if s >= 1 else 'shift needed at s=0', R), (n, j[1:], s))
                    if R == 1 and Ks != {i for i in range(n + 1) if D[i] <= s}:
                        st[R]['R1_K_ne_D'] += 1
                mDR, mD, mDp = max(DR), max(D), max(Dprime)
                if stop > mDR + 1:
                    fail("(b) rounds<=maxD_R+1", n, j, R)
                if stop > mD:
                    fail("extra: rounds<=maxD", n, j, R, "rounds=%d maxD=%d" % (stop, mD))
                if stop > mDp:
                    fail("extra: rounds<=maxD'_R", n, j, R)
                if stop > -(-n // (R + 1)):
                    fail("extra: rounds<=ceil(n/(R+1))", n, j, R)
                st[R]['rounds_eq_DR1'] += (stop == mDR + 1)
                if stop == mDR + 1:
                    first.setdefault(('rounds = maxD_R+1', R), (n, j[1:], stop))
                if stop < mDp:
                    st[R]['rounds_lt_maxDprime'] += 1
                    first.setdefault(("rounds < maxD'_R", R), (n, j[1:], stop, mDp))
                st[R]['rounds_eq_DR1_alt'] += (stop == max(DRalt) + 1)
                st[R]['rounds_eq_D'] += (stop == mD)
                st[R]['rounds_eq_ceil'] += (stop == -(-n // (R + 1)))
                st[R]['maxrounds'] = max(st[R]['maxrounds'], stop)
                st[R]['maxgap'] = max(st[R]['maxgap'], mDR + 1 - stop)
                # (c)
                y = [1] + [0] * n
                ys = [y]
                for s in range(len(hist) + 2):
                    ys.append(G(n, j, R, ys[-1]))
                for s, (K, v) in enumerate(hist):
                    if any(ys[s][i] != x[i] for i in K):
                        fail("(c) y^(s)=x on K^(s)", n, j, R, "s=%d" % s)
                sj = 0
                while ys[sj + 1] != ys[sj]:
                    sj += 1
                    if sj + 1 >= len(ys):
                        ys.append(G(n, j, R, ys[-1]))
                if ys[sj] != x:
                    fail("thm:table(a)", n, j, R)
                apps = sj + 1
                if apps > stop + 1:
                    fail("(c) apps<=rounds+1", n, j, R, "apps=%d rounds=%d" % (apps, stop))
                st[R]['apps_eq_r1'] += (apps == stop + 1)
                if apps == stop + 1:
                    first.setdefault(('apps = rounds+1', R), (n, j[1:], stop))
                if apps < stop:
                    first.setdefault(('apps < rounds', R), (n, j[1:], apps, stop))
                st[R]['apps_lt_rounds'] += (apps < stop)
                # monotone in R
                if prevK is not None:
                    for s in range(max(len(prevK), len(hist))):
                        A = prevK[min(s, len(prevK) - 1)][0]
                        B = hist[min(s, len(hist) - 1)][0]
                        if not A <= B:
                            fail("extra: monotone in R", n, j, R, "s=%d" % s)
                prevK = hist
        print("n=%d arrays=%d" % (n, cnt))
        for R in range(1, rmax + 1):
            print("  R=%d %s" % (R, st[R]))
        sys.stdout.flush()
    for k in sorted(first, key=str):
        print("  first instance of", k, ":", first[k])
    print("failures:", len(fails))
    for k, v in fails.items():
        print("  first counterexample for [%s]: n=%d j=%s R=%s %s" % ((k,) + v))


main()
