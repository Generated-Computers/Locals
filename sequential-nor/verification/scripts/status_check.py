#!/usr/bin/env python3
"""Checks for the subsection "Above 4/9" of the section on block bases.

(1) Lemma "the recursion for D": for every parent array with n <= 7 and for random ones, the depth
    D computed from the definition with anchors (Section "The upper end") satisfies the four cases
    of the lemma, and its consequences (O1), (O2) hold for every round s.
(2) Proposition "the map of the open fractions": the closed form for the stationary law of the
    status chain agrees with the stationary law computed by iterating the 4-state chain, on a grid;
    the map is monotone on a grid; the fixed-point equations have the stated solution below 4/9
    and none above; the iterates from the top tend to 0 above 4/9.

Usage: python3 -I status_check.py        (a few seconds)
       python3 -I status_check.py iterates eps density f s1 s2 ...
           (the iterates quoted next to the measurements of block_unknown.c; for example
            python3 -I status_check.py iterates 0.46 0.2581 0.017 10 50 200 300)
"""
import itertools, math, random

def trace(j):
    x = [1]
    for i in range(1, len(j)): x.append(0 if (x[i-1] or x[j[i]]) else 1)
    return x

def depth(j):
    """D from the definition: anchors are the resets in [rho'(i), i], and 0 if rho'(i) = 0"""
    n = len(j)-1; x = trace(j); D = [0]*(n+1)
    rel = lambda k: x[k-1] == 0
    notg = lambda k: j[k] == k-1
    reset = lambda k: (not notg(k)) and x[j[k]] == 1
    for i in range(1, n+1):
        rho = 0
        for r in range(i, 0, -1):
            if reset(r) and rel(r): rho = r; break
        anchors = [r for r in range(max(rho, 1), i+1) if reset(r)]
        if rho == 0: anchors.append(0)
        best = None
        for r in anchors:
            vals = [D[j[k]] for k in range(r+1, i+1) if rel(k) and not notg(k)]
            if r >= 1: vals.append(D[j[r]])
            v = 1+(max(vals) if vals else 0)
            best = v if best is None else min(best, v)
        D[i] = best
    return x, D

def check_array(j):
    n = len(j)-1; x, D = depth(j)
    for k in range(2, n+1):
        t = j[k]; notg = (t == k-1); reset = (not notg) and x[t] == 1; rel = x[k-1] == 0
        if reset and rel: want = 1+D[t]
        elif reset: want = min(D[k-1], 1+D[t])
        elif rel and not notg: want = max(D[k-1], 1+D[t])
        else: want = D[k-1]
        assert D[k] == want, (j, k, D, want)
        for s in range(0, n+1):
            op = lambda p, r: D[p] > r
            if op(k, s+1) and not op(k-1, s+1):          # (O1)
                assert x[k-1] == 0 and not notg and op(t, s), (j, k, s)
            if op(k, s+1) and op(k-1, s+1):              # (O2)
                assert not (reset and not op(t, s)), (j, k, s)

def part1():
    cnt = 0
    for n in range(1, 8):
        for js in itertools.product(*[range(i) for i in range(1, n+1)]):
            check_array([0]+list(js)); cnt += 1
    random.seed(1)
    for _ in range(4000):
        n = random.randint(8, 40)
        check_array([0]+[random.randrange(i) for i in range(1, n+1)]); cnt += 1
    print("recursion for D, (O1), (O2): %d parent arrays, no failure" % cnt)

def varpi(e): return ((2*e+1)-math.sqrt(4*e+1))/(2*e)
def consts(e):
    A = e*varpi(e); c = 1-e; q = 0.6
    for _ in range(2000): q = 1-A-c*q/(1+q)
    return A, c, q, 1-q, q/(1+q)
def psi(q, sig, a, b):
    den = sig*(1+q)+q*(b-a)+a*b
    return (b+q*a-a*b)/((1+q)*den), (a+q*b+a*b)/((1+q)*den)
def stationary(q, sig, a, b):
    # states K1, K0, U1, U0; symbols: determined one, open one, determined zero, open zero
    p = {'d1': sig-a, 'o1': a, 'd0': q-b, 'o0': b}
    v = [0.25]*4
    for _ in range(3000):
        K1, K0, U1, U0 = v
        nK1 = K0*p['d0']
        nK0 = K1+K0*p['d1']+(U1+U0)*p['d1']
        nU1 = K0*p['o0']+U0*(p['d0']+p['o0'])
        nU0 = K0*p['o1']+U1*(1-p['d1'])+U0*p['o1']
        v = [nK1, nK0, nU1, nU0]
    return v[2], v[3], v[0]

def part2():
    worst = 0.0
    for e in (0.2, 0.4, 0.44, 0.46, 0.5, 0.6, 0.7):
        A, c, q, sig, pi = consts(e)
        assert abs(sig-(A+c*pi)) < 1e-12
        N = 12; prev = {}
        for i in range(N+1):
            for k in range(N+1):
                u1, u0 = pi*i/N, (1-pi)*k/N
                f = psi(q, sig, c*u1, c*u0); g = stationary(q, sig, c*u1, c*u0)
                worst = max(worst, abs(f[0]-g[0]), abs(f[1]-g[1]))
                assert abs(g[0]+g[2]-pi) < 1e-9          # P(value 1) = pi
                prev[(i, k)] = f
                if i > 0: assert f[0] >= prev[(i-1, k)][0]-1e-12 and f[1] >= prev[(i-1, k)][1]-1e-12
                if k > 0: assert f[0] >= prev[(i, k-1)][0]-1e-12 and f[1] >= prev[(i, k-1)][1]-1e-12
        lam = c/(1-q*q)
        # fixed point
        if lam > 1:
            Dl = sig*(lam-1); a = (-Dl+math.sqrt(Dl*Dl+4*Dl))/2; b = a+Dl
            f = psi(q, sig, a, b)
            assert abs(c*f[0]-a) < 1e-10 and abs(c*f[1]-b) < 1e-10
            assert a <= c*pi and b <= c*(1-pi)
        u = (pi, 1-pi)
        for _ in range(200000):
            u = psi(q, sig, c*u[0], c*u[1])
            if u[0]+u[1] < 1e-13: break
        print("eps=%.2f lambda=%.4f: iterates from the top tend to (%.6f, %.6f)%s" %
              (e, lam, u[0], u[1], "" if lam <= 1 else "  = fixed point (%.6f, %.6f)" % (a/c, b/c)))
        if lam < 1: assert u[0]+u[1] < 1e-12
        else: assert abs(c*u[0]-a) < 1e-6 and abs(c*u[1]-b) < 1e-6
    print("closed form against the iterated chain: largest difference %.2e" % worst)

def iterates(e, dens, f, rounds):
    """Iterates of the map from the top, for the comparison with block_unknown.c: the density of
    ones among all positions is dens instead of varpi(eps), and a read that is resampled into the
    blocks (probability eps*f) is counted like a read of the block below."""
    A = e*dens; c = 1-e; q = 0.6
    for _ in range(2000): q = 1-A-c*q/(1+q)
    sig = 1-q; pi = q/(1+q); ce = c+e*f
    u = (pi, 1-pi); out = []
    for s in range(1, max(rounds)+1):
        u = psi(q, sig, ce*u[0], ce*u[1])
        if s in rounds: out.append("%d: %.3f" % (s, u[0]+u[1]))
    print("eps=%.2f density=%.4f block fraction=%.3f: sum of the two open fractions after s rounds:  %s"
          % (e, dens, f, "   ".join(out)))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "iterates":
        iterates(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), [int(v) for v in sys.argv[5:]])
    else:
        part1(); part2(); print("ALL CHECKS PASSED")
