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
                B = {0, 1}
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


