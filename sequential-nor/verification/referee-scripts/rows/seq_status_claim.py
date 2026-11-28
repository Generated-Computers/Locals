# Referee checks for Subsection "Counting positions":
#  (1) Lemma "status of a gate" on random parent arrays, with D obtained by
#      running the relevance-pruned iteration itself (not the recursion);
#  (2) the Claim of Step 1 on smoothed fixed-distance instances (stops = resets);
#  (3) Step 3: agreement of the true and the ideal upper pair after a
#      regeneration, and inequality (rowssplit).
import random, sys

def NOR(a, b): return 1 if (a == 0 and b == 0) else 0

def trace(j):
    n = len(j) - 1
    x = [1] * (n + 1)
    for i in range(1, n + 1):
        x[i] = NOR(x[i - 1], x[j[i]])
    return x

def depth_by_iteration(j):
    """Run Definition kp:pruned-def literally; return D(i) = first round known."""
    n = len(j) - 1
    known = [False] * (n + 1); v = [None] * (n + 1)
    known[0] = True; v[0] = 1
    D = [None] * (n + 1); D[0] = 0
    t = 0
    while not all(known):
        t += 1
        res = [False] * (n + 1); rr = [False] * (n + 1)
        for k in range(1, n + 1):
            res[k] = (j[k] == k - 1) or known[j[k]]
            rr[k] = res[k] and j[k] != k - 1 and v[j[k]] == 1
        nk = known[:]; nv = v[:]
        for i in range(1, n + 1):
            r = i
            while r > 0 and not rr[r]: r -= 1
            vr = 1 if r == 0 else 0
            ok = True
            for k in range(r + 1, i + 1):
                alpha = vr ^ ((k - 1 - r) & 1)
                if alpha == 0 and not res[k]: ok = False; break
            if ok:
                val = vr ^ ((i - r) & 1)
                if not known[i]:
                    nk[i] = True; nv[i] = val; D[i] = t
        known, v = nk, nv
        if t > n + 2: raise RuntimeError("no progress")
    return D, v

def depth_by_recursion(j, x):
    n = len(j) - 1
    D = [0] * (n + 1)
    if n >= 1: D[1] = 1
    for i in range(2, n + 1):
        notg = (j[i] == i - 1)
        reset = (not notg) and x[j[i]] == 1
        rel = (x[i - 1] == 0)
        if reset and rel: D[i] = 1 + D[j[i]]
        elif reset: D[i] = min(D[i - 1], 1 + D[j[i]])
        elif rel and not notg: D[i] = max(D[i - 1], 1 + D[j[i]])
        else: D[i] = D[i - 1]
    return D

def f(v, o, b, w):
    if v == 0 and b == 1: return w
    if v == 1 and b == 1: return o & w
    if v == 0 and b == 0: return o | w
    return o

def check_status(trials, nmax, rng):
    bad = 0
    for _ in range(trials):
        n = rng.randint(1, nmax)
        mode = rng.random()
        j = [0] * (n + 1)
        for i in range(1, n + 1):
            if mode < 0.3: j[i] = rng.randrange(i)
            elif mode < 0.6: j[i] = i - 1 if rng.random() < 0.5 else rng.randrange(i)
            else:
                H = 3
                j[i] = max(i - H, 0) if rng.random() < 0.6 else rng.randrange(i)
        x = trace(j)
        D, v = depth_by_iteration(j)
        assert v == x
        assert D == depth_by_recursion(j, x), (j, D)
        resets = [False] + [(j[k] != k - 1 and x[j[k]] == 1) for k in range(1, n + 1)]
        for i in range(1, n + 1):
            assert D[i] <= i
            p = j[i]
            for t in range(0, n + 1):
                I = lambda q, lev: 1 if D[q] > lev else 0
                lhs = I(i, t + 1)
                rhs = f(x[i - 1], I(i - 1, t + 1), x[p], I(p, t))
                if lhs > rhs: bad += 1; print("INEQ FAIL", j, i, t)
                if (p != i - 1 or I(p, t) == 0) and lhs != rhs:
                    bad += 1; print("EQ FAIL", j, i, t)
                if lhs == 1:
                    ok = False
                    k = i
                    while k >= 1:
                        if j[k] != k - 1 and I(j[k], t) == 1: ok = True; break
                        if resets[k]: break
                        k -= 1
                    if not ok: bad += 1; print("LAST CLAIM FAIL", j, i, t)
    return bad

def smoothed(n, H, eps, rng):
    j = [0] * (n + 1); pi = [0] * (n + 1)
    for i in range(1, n + 1):
        if rng.random() < eps:
            pi[i] = 1; j[i] = rng.randrange(i)
        else:
            j[i] = max(i - H, 0)
    return j, pi

def check_claim(trials, n, H, eps, rng):
    bad = 0; tested = 0
    for _ in range(trials):
        j, pi = smoothed(n, H, eps, rng)
        x = trace(j); D = depth_by_recursion(j, x)
        reset = [False] + [(j[k] != k - 1 and x[j[k]] == 1) for k in range(1, n + 1)]
        nxt = [n + 1] * (n + 2)          # first stop strictly after k
        for k in range(n - 1, -1, -1):
            nxt[k] = k + 1 if reset[k + 1] else nxt[k + 1]
        def N(k): return k if k > n else nxt[k] - 1
        maxD = max(D)
        for _ in range(400):
            i = rng.randint(1, n); m = rng.randint(1, 6)
            if D[i] <= m: continue
            t = rng.randint(0, D[i] - m - 1)      # I^{t+m}_i = 1
            tested += 1
            ok = False
            # (a)
            for p in range(1, i - m * H + 1):
                if D[p] > t:
                    E = p
                    for _q in range(m): E = N(E + H)
                    if i <= E: ok = True; break
            if not ok:
                for jj in range(0, m):
                    lev = t + m - jj - 1
                    for k in range(1, i - jj * H + 1):
                        if pi[k] == 1 and D[j[k]] > lev:
                            E = N(k)
                            for _q in range(jj): E = N(E + H)
                            if i <= E: ok = True; break
                    if ok: break
            if not ok: bad += 1; print("CLAIM FAIL", i, t, m)
    return bad, tested

def check_regen(trials, H, K, eps, rng, rows=12):
    """True vs ideal upper pair in windows; returns counts."""
    bad = 0; nwin = 0; diff_after = 0; split_bad = 0
    n = rows * H
    for _ in range(trials):
        j, pi = smoothed(n, H, eps, rng)
        x = trace(j); D = depth_by_recursion(j, x)
        O = [0] * (n + 1); c = 0
        for i in range(n + 1): c += x[i]; O[i] = c
        for _w in range(20):
            a = rng.randint(4 * H, n - 1)
            t = rng.randint(0, max(D[a] - 1, 0) + 1)
            s = a - H - K; J = min(H - K, n - a)
            psi_s = O[s] / (s + 1)
            I = lambda q, lev: 1 if D[q] > lev else 0
            y, oh = 0, 0
            lastnear = -K; regen_since = False
            nwin += 1
            for jj in range(-K + 1, J + 1):
                g2 = a + jj; g1 = a - H + jj
                near = False
                if pi[g2] == 0:
                    assert j[g2] == g1
                    rd = (x[g1], I(g1, t + 1))
                elif j[g2] <= g1:
                    rd = (x[j[g2]], I(j[g2], t + 1))
                else:
                    near = True
                    rd = (1 if rng.random() < psi_s else 0, 0)
                isregen = (not near) and y == 0 and x[g2 - 1] == 0 and x[j[g2]] == 1
                y, oh = NOR(y, rd[0]), f(y, oh, rd[0], rd[1])
                if near: lastnear = jj; regen_since = False
                if isregen: regen_since = True
                if regen_since:
                    if (y, oh) != (x[g2], I(g2, t + 2)):
                        bad += 1
                if jj >= 1:
                    # (rowssplit)
                    b = g2 if False else (a + lastnear - 1 if lastnear > -K else a - K)
                    rhs = I(b, t + 2)
                    for ii in range(max(lastnear, 1 - K), jj + 1):
                        rhs += I(a - H + ii, t + 1)
                        if pi[a + ii] == 1 and I(j[a + ii], t + 1) == 1: rhs += 1
                    if I(g2, t + 2) > rhs: split_bad += 1
    return bad, split_bad, nwin

if __name__ == "__main__":
    rng = random.Random(12345)
    print("status lemma failures:", check_status(4000, 14, rng))
    for eps in (0.2, 0.455, 0.7):
        print("claim eps", eps, check_claim(6, 400, 7, eps, rng))
        print("claim eps", eps, check_claim(6, 600, 25, eps, rng))
    for eps in (0.3, 0.455, 0.8):
        print("regen eps", eps, check_regen(40, 40, 8, eps, rng))
