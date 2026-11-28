# (a) Lemma spacing, (c) Theta^ord values.  Operators implemented directly from their definition.
import math, random, sys
from fractions import Fraction as F

def M(w, eps, c, beta):
    r = 1-eps*c
    return (max(a*w[1] + (1-a)*w[0] for a in (eps*beta, r)), w[0])
def phi(w, eps, c):
    return ((1-eps*c)*w[0], eps*(1-c)*w[1])
def B(d, y, eps, c, beta):
    w = phi(y, eps, c)
    for _ in range(d): w = M(w, eps, c, beta)
    return w
def s_of(y, eps, c, beta):
    r = 1-eps*c; e0 = eps*(1-c); g = eps*eps*beta*c
    return r*(1-g)*y[0] + g*e0*y[1]
def varpi(e): return ((2*e+1) - math.sqrt(4*e+1))/(2*e)
def theta_ord(D, eps, c, beta):
    r = 1-eps*c
    tail = 0.0 if D is None else r**D/(eps*c)
    return (1-eps)*(1/(eps*c + eps*eps*beta*c*(1-eps)) + tail)

def rand_adm(rnd, epsmax=F(3, 4), exact=True):
    # admissible pairs satisfy c < 1/2 and c + beta < 1 (from a nonempty G_g)
    while True:
        eps = F(rnd.randint(1, 75), 100) if exact else rnd.uniform(0.01, 0.75)
        c = F(rnd.randint(1, 49), 100) if exact else rnd.uniform(0.01, 0.4999)
        beta = F(rnd.randint(0, 99), 100) if exact else rnd.uniform(0.0, 0.999)
        if c + beta < 1 and eps <= epsmax: return eps, c, beta

if __name__ == "__main__":
    rnd = random.Random(2026)
    # (a) exact rationals: closed form, recursion, bounds, for d up to 60
    cnt = 0
    for trial in range(300):
        eps, c, beta = rand_adm(rnd)
        r = 1-eps*c; e0 = eps*(1-c); g = eps*eps*beta*c
        assert eps*beta <= r and r-e0 == 1-eps
        # random y with Delta >= 0
        y0 = F(rnd.randint(1, 50), 10); y1max = r*y0/e0
        y1 = y1max*F(rnd.randint(0, 20), 20)
        y = (y0, y1); Dl = r*y0 - e0*y1; assert Dl >= 0
        w = phi(y, eps, c)
        cs = [F(0), eps*beta]
        cur = w
        for d in range(1, 61):
            cur = M(cur, eps, c, beta)
            if d >= 2: cs.append(min(a*cs[d-2] + (1-a)*cs[d-1] for a in (eps*beta, r)))
            assert cur == (w[0]-cs[d]*Dl, w[0]-cs[d-1]*Dl), ("closed form", d)
            assert g <= cs[d] <= eps*beta
            if d == 1: assert cur == (r*y0 - eps*beta*Dl, r*y0)
            if d == 2: assert cs[2] == g
            if d >= 2: assert cur[0] <= s_of(y, eps, c, beta) and cur[1] <= s_of(y, eps, c, beta)
            assert cur == B(d, y, eps, c, beta) if d <= 6 else True
            cnt += 1
        assert s_of((1, 1), eps, c, beta) == r - g*(1-eps)
        # the condition used in Lemma sumspacing
        assert eps*c <= r*(1-g)
    print("(a) Lemma spacing: closed form, c_d recursion, gamma<=c_d<=eps*beta, B_1 formula, B_d<=s(y) for 2<=d<=60: exact, %d (params,y,d) cases ok" % cnt)
    # is Delta >= 0 necessary for B_d y <= s(y) 1 ?
    bad = 0
    for trial in range(2000):
        eps, c, beta = rand_adm(rnd, exact=False)
        y = (rnd.random(), rnd.random()*5)
        for d in (2, 3, 4, 7):
            z = B(d, y, eps, c, beta)
            if max(z) > s_of(y, eps, c, beta)*(1+1e-12): bad += 1
    print("    (without Delta>=0 the bound B_d y <= s(y) fails in %d of 8000 random cases; the lemma assumes Delta>=0)" % bad)
    # (c)
    for eps, Dd, bound in ((0.74, 16, 0.981), (0.737, 32, 0.985)):
        c = varpi(eps)-1e-4; v = theta_ord(Dd, eps, c, 0.4999)
        print("(c) Theta^ord_%d(%.3f; varpi-1e-4, 0.4999) = %.6f  claimed <= %.3f : %s" % (Dd, eps, v, bound, v <= bound))
    for Dd in (16, None, 8, 9):
        lo, hi = 0.6, 0.76
        f = lambda e: theta_ord(Dd, e, varpi(e), 0.5)
        if f(hi) > 1: print("    D=%s: Theta^ord > 1 on up to eps=0.76; value at 0.75: %.4f" % (Dd, f(0.75))); continue
        for _ in range(60):
            mid = (lo+hi)/2
            if f(mid) > 1: lo = mid
            else: hi = mid
        print("    D=%s: Theta^ord(c=varpi, beta=1/2) = 1 at eps = %.4f ; value at 0.75: %.4f" % (Dd, lo, f(0.75)))
