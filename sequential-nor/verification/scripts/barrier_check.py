"""Certificate for the first-moment barrier (Proposition "a base on which the chain count grows").

The base: zero gates (pointer 0) below n0, and in the active block gate q reads q-2 if q is even and
q-3 if q is odd, so every target is even.  S_q = (x_{q-2}, x_{q-1}, x_q) is the state, encoded as
4*x_{q-2} + 2*x_{q-1} + x_q.  A resampled gate reads a one with a probability p in [plo, phi]; every
transition below is the entrywise MINIMUM over that interval, so the computed vectors are lower bounds.

Checked in exact rational arithmetic (vectors rounded DOWN to multiples of 10^-40 after each step):
    sum_{m=0}^{M} v . Push_{q-m} . NR_{q-m+1} ... NR_q  >=  mu * v      (q even)
where Push_k = (free gates between the target k-d_k and k) . (gate k kept) and NR_g is "gate g is not
a reset".  Standard library only."""
from fractions import Fraction as F

SC = 10**40
down = lambda q: F((q.numerator * SC) // q.denominator, SC)          # largest multiple of 1e-40 <= q
VALID = [s for s in range(8) if not (s & 3) == 3 and not (s & 6) == 6]   # no two adjacent ones

def varpi_bounds(eps, steps=200):
    f = lambda w: w - eps * (1 - w) ** 2
    lo, hi = F(0), F(1, 2)
    for _ in range(steps):
        mid = F(round(((lo + hi) / 2) * 10**30), 10**30)
        if f(mid) < 0: lo = mid
        else: hi = mid
    return lo, hi

def dist(q): return 2 if q % 2 == 0 else 3

def gate(eps, plo, phi, q, kind):
    """8x8 matrix (dict of dicts) for gate q.  kind: 'NR' non-reset, 'FULL' any outcome, 'KEPT'."""
    d = dist(q); T = {s: {} for s in range(8)}
    for s in range(8):
        b0 = s & 1; v = (s >> (d - 1)) & 1
        add = lambda x, w: T[s].__setitem__(((s << 1) | x) & 7, T[s].get(((s << 1) | x) & 7, F(0)) + w)
        if kind == 'KEPT':
            add(1 - (b0 | v), 1 - eps)
        elif kind == 'NR':
            if v == 0: add(1 - b0, 1 - eps)                  # kept, reads a zero
            add(1 - b0, eps * (1 - phi))                     # resampled, not a reset
        else:  # FULL
            if b0 == 1: add(0, F(1))                         # x_q = 0 whatever happens
            else:
                add(1 - v, 1 - eps)                          # kept
                add(0, eps * plo); add(1, eps * (1 - phi))   # resampled (minimum over p)
    return T

def vecmat(v, T):
    out = [F(0)] * 8
    for s in range(8):
        if v[s]:
            for t, w in T[s].items(): out[t] += v[s] * w
    return [down(x) for x in out]

def operator(eps, plo, phi, v, M, q=10**6):
    """sum over window lengths m <= M of the contribution of chain gate k = q-m (q even)."""
    G = lambda g, kind: gate(eps, plo, phi, g, kind)
    total = [F(0)] * 8
    for m in range(M + 1):
        k = q - m; d = dist(k); tgt = k - d
        assert tgt % 2 == 0                                   # every target is even
        w = v[:]
        for g in range(tgt + 1, k): w = vecmat(w, G(g, 'FULL'))
        w = vecmat(w, G(k, 'KEPT'))
        for g in range(k + 1, q + 1): w = vecmat(w, G(g, 'NR'))
        total = [a + b for a, b in zip(total, w)]
    return total

def power(eps, plo, phi, M, its=60):
    v = [F(1) if s in VALID else F(0) for s in range(8)]
    fl = lambda x: [float(a) for a in x]
    vf = fl(v)
    # float power iteration for the eigenvector, then rationalise
    import math
    for _ in range(its):
        w = operator(eps, plo, phi, [F(round(a * 1e12), 10**12) for a in vf], M)
        wf = fl(w); s = sum(wf); vf = [a / s for a in wf]
    return [F(round(a * 1e12), 10**12) for a in vf]

def certify(eps, delta, mu, M=70):
    lo, hi = varpi_bounds(eps)
    plo, phi = lo - delta, hi + delta
    v = power(eps, plo, phi, M)
    Tv = operator(eps, plo, phi, v, M)
    ok = all(Tv[s] >= mu * v[s] for s in range(8))
    supp = [s for s in range(8) if v[s] > 0]
    ratio = min(float(Tv[s] / v[s]) for s in supp)
    print("eps=%s delta=%s varpi in [%.6f, %.6f]: support %s, min (Tv)_s/v_s = %.5f >= mu = %s : %s"
          % (eps, delta, float(lo), float(hi), ["{:03b}".format(t) for t in supp], ratio, mu, ok))
    assert ok and all(v[s] >= 0 for s in range(8)) and supp
    return v

if __name__ == "__main__":
    certify(F(7, 10), F(1, 2000), F(113, 100))
    certify(F(18, 25), F(1, 2000), F(101, 100))
    print("all certificates passed")
