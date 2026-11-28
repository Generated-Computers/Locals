# Referee checks for Subsection "Three rows" (deterministic claims of Steps 2-3).
# Smoothed fixed-distance instances with small H; true rows, ideal rows 2 and 3,
# and the row "tilde" of comparison (B)/(C), built exactly as in the text.
# Usage: python3 -I seq_rows3_window.py [seeds] [H] [K] [n]
import random, sys

def NOR(a, b): return 1 if (a == 0 and b == 0) else 0
def f(v, o, b, w):
    if v == 0 and b == 1: return w
    if v == 1 and b == 1: return o & w
    if v == 0 and b == 0: return o | w
    return o
def read(pair, sym):
    return (NOR(pair[0], sym[0]), f(pair[0], pair[1], sym[0], sym[1]))

def depth(j, x):
    n = len(j) - 1
    D = [0] * (n + 1); D[1] = 1
    for i in range(2, n + 1):
        notg = (j[i] == i - 1); reset = (not notg) and x[j[i]] == 1; rel = (x[i - 1] == 0)
        if reset and rel: D[i] = 1 + D[j[i]]
        elif reset: D[i] = min(D[i - 1], 1 + D[j[i]])
        elif rel and not notg: D[i] = max(D[i - 1], 1 + D[j[i]])
        else: D[i] = D[i - 1]
    return D

def no_two_far(far, lo, hi):
    # no i with lo < i-1, i <= hi, both far
    for i in range(lo + 2, hi + 1):
        if far.get(i - 1) and far.get(i): return False
    return True

def run(seed, H, K, n, eps):
    R = random.Random(seed)
    pi = [0] * (n + 1); rho = [0] * (n + 1); j = [0] * (n + 1)
    for i in range(1, n + 1):
        pi[i] = 1 if R.random() < eps else 0
        rho[i] = R.randrange(i)
        j[i] = rho[i] if pi[i] else max(i - H, 0)
    x = [1] * (n + 1)
    for i in range(1, n + 1): x[i] = NOR(x[i - 1], x[j[i]])
    D = depth(j, x)
    cnt = {}
    def bump(k): cnt[k] = cnt.get(k, 0) + 1
    def fail(msg): raise AssertionError(msg)
    maxD = max(D)
    for trial in range(40):
        a = R.randrange(3 * H + K + 2, n)
        J = min(H - K, n - a)
        t = R.randrange(0, max(1, maxD - 2))
        I = lambda lev, p: 1 if D[p] > lev else 0
        g = lambda r, jj: a - (3 - r) * H + jj
        rng = range(-K + 1, J + 1)
        psi_s = 0.4
        beta = {(r, jj): (1 if R.random() < psi_s else 0) for r in (2, 3) for jj in rng}
        z = {jj: (x[g(0, jj)], I(t, g(0, jj))) for jj in rng}
        tr = {r: {jj: (x[g(r, jj)], I(t + r, g(r, jj))) for jj in range(-K, J + 1)} for r in (1, 2, 3)}
        near = {r: {jj: (pi[g(r, jj)] == 1 and rho[g(r, jj)] > g(1, jj)) for jj in rng} for r in (2, 3)}
        far = {r: {jj: (pi[g(r, jj)] == 1 and rho[g(r, jj)] <= g(1, jj) and x[rho[g(r, jj)]] == 1) for jj in rng} for r in (2, 3)}
        src = {r: {jj: (pi[g(r, jj)] == 1 and I(t + r - 1, rho[g(r, jj)]) == 1) for jj in rng} for r in (1, 2, 3)}
        # row 1 is the first half of a step, up to sources (inequality always)
        for jj in rng:
            G = g(1, jj)
            sym = z[jj] if pi[G] == 0 else (x[rho[G]], I(t, rho[G]))
            want = read(tr[1][jj - 1], sym)
            if want[0] != tr[1][jj][0] or tr[1][jj][1] > want[1]: fail("row1 bound")
            if want != tr[1][jj] and not src[1][jj]: fail("row1 equality off sources")
        def build(r, lower, start=(0, 0)):
            out = {-K: start}
            for jj in rng:
                G = g(r, jj)
                if pi[G] == 0: sym = lower[jj]
                elif rho[G] <= g(1, jj): sym = (x[rho[G]], I(t + r - 1, rho[G]))
                else: sym = (beta[(r, jj)], 0)
                out[jj] = read(out[jj - 1], sym)
            return out
        id2 = build(2, tr[1]); id3 = build(3, id2); til = build(3, tr[2])
        # true pairs change as in the definition at steps that are not near
        for r, lower in ((2, tr[1]), (3, tr[2])):
            for jj in rng:
                if near[r][jj]: continue
                G = g(r, jj)
                sym = lower[jj] if pi[G] == 0 else (x[rho[G]], I(t + r - 1, rho[G]))
                if read(tr[r][jj - 1], sym) != tr[r][jj]: fail("true row %d not as defined at a non-near step" % r)
        # start bound for Phi_0
        s0 = sum(tr[1][i][1] + (1 if (src[1][i] or src[2][i] or src[3][i]) else 0) for i in range(-K + 1, 1))
        if id2[0][1] + id3[0][1] > (K + 1) * s0: fail("Phi_0 bound")
        if id2[0][1] + id3[0][1] > 2 * s0: bump("Phi_0 bound with 2 fails")
        o1 = lambda e: tr[1][e][1]
        o2 = lambda e: tr[2][e][1]
        for jj in rng:
            # (A)
            dist = [c for c in rng if c <= jj and near[2][c]]
            cs = dist[-1] if dist else -K
            if id2[jj] != tr[2][jj]:
                bump("Delta")
                if not no_two_far(far[2], cs, jj): fail("(A) rows3delta")
            for c in range(-K, jj + 1):
                b = (id2[c - 1][1] if c > -K else 0) + sum(o1(e) + src[2][e] for e in range(max(c, 1 - K), jj + 1))
                if id2[jj][1] > b: fail("(A) status bound ideal")
                b = (tr[2][c - 1][1] if c > -K else tr[2][-K][1]) + sum(o1(e) + src[2][e] for e in range(max(c, 1 - K), jj + 1))
                if tr[2][jj][1] > b: fail("(A2) status bound true")
            # (B)
            dist = [c for c in rng if c <= jj and near[3][c]]
            cs = dist[-1] if dist else -K
            if til[jj] != tr[3][jj]:
                bump("B differ")
                if not no_two_far(far[3], cs, jj): fail("(B) regeneration")
                base = tr[3][cs - 1][1] if cs > -K else tr[3][-K][1]
                if tr[3][jj][1] > base + sum(o2(e) + src[3][e] for e in range(max(cs, 1 - K), jj + 1)): fail("(B) split")
            if tr[3][jj][1] > til[jj][1] + (1 if til[jj] != tr[3][jj] else 0): fail("(B) final")
            # (C)
            dist = [c for c in rng if c <= jj and pi[g(3, c)] == 0 and id2[c] != tr[2][c]]
            if til[jj] != id3[jj]:
                bump("C differ")
                if not dist: fail("(C) differ without a disturbed step")
                i = dist[-1]
                if not no_two_far(far[3], i, jj): fail("(C) two far ones")
                if til[jj][1] > til[i - 1][1] + sum(o2(l) + src[3][l] for l in range(i, jj + 1)): fail("(C) status")
            # alive, for tilde and for ideal 2
            for r, row, low in ((3, til, tr[2]), (2, id2, tr[1])):
                if row[jj][1] == 1:
                    ls = [l for l in rng if l <= jj and ((pi[g(r, l)] == 0 and low[l][1] == 1) or (pi[g(r, l)] == 1 and rho[g(r, l)] <= g(1, l) and I(t + r - 1, rho[g(r, l)]) == 1))]
                    if not ls: fail("alive: open without a read of status 1")
                    if any(far[r][e] for e in range(ls[-1] + 1, jj + 1)): fail("alive: far one after the last open read")
                    bump("alive tested")
    return cnt

if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 1200
    tot = {}
    for eps in (0.1, 0.3, 0.4, 0.455):
        for sd in range(seeds):
            c = run(1000 * sd + int(eps * 1000), H, K, n, eps)
            for k, v in c.items(): tot[k] = tot.get(k, 0) + v
    print("all assertions passed; counts:", tot)
