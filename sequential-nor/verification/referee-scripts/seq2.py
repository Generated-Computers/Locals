import math
from fractions import Fraction as F

def Lam(mu, e): return (1-e*mu)/(3-e-e*mu)
def U(l, e): return (1-e*l)/(2-e*l)
def seq(e, S, half=0.5):
    mu = [half]; l = [None]
    for s in range(S):
        l.append(Lam(mu[-1], e)); mu.append(U(l[-1], e))
    return l, mu
def limits(e):
    l, mu = seq(e, 400); return l[-1], mu[-1]
def varpi(e): return ((2*e+1) - math.sqrt(4*e+1))/(2*e)
def Tm(H, m, eps, c, beta):
    r = 1-eps*c; e0 = eps*(1-c)
    q = min(m, H-1)
    t0 = t1 = r**(m-q)
    for i in range(q-1, -1, -1):
        n1 = r*t0
        n0 = max(a*e0*t1 + (1-a)*r*t0 for a in (eps*beta, r))
        t0, t1 = n0, n1
    return t0, t1
def Theta(H, eps, c, beta):
    r = 1-eps*c
    s = sum(max(Tm(H, m, eps, c, beta)) for m in range(H-1)) + max(Tm(H, H-1, eps, c, beta))/(1-r)
    return (1-eps)*s
def Theta_brute(H, eps, c, beta):
    r = 1-eps*c; s = 0.0; m = 0
    while r**m > 1e-17: s += max(Tm(H, m, eps, c, beta)); m += 1
    return (1-eps)*s

if __name__ == "__main__":
    # sequences: monotone, ranges, l_1 = 1/3, on a grid of eps (exact rationals for a few)
    bad = 0
    for k in range(1, 1001):
        e = k/1000
        l, mu = seq(e, 60)
        if abs(l[1]-1/3) > 1e-15: bad += 1
        for s in range(1, 60):
            if l[s+1] < l[s]-1e-15 or mu[s] > mu[s-1]+1e-15: bad += 1
            if not (1/3-1e-15 <= l[s] <= 0.5 and 0 <= mu[s] <= 0.5): bad += 1
            if l[s] > mu[s] + 1e-15: bad += 1
    print("sequence violations on eps grid:", bad)
    for e in (F(1,20), F(7,10), F(1)):
        l, mu = seq(e, 8, F(1,2))
        assert l[1] == F(1,3) and all(l[s+1] >= l[s] for s in range(1, 8)) and all(mu[s+1] <= mu[s] for s in range(8))
    print("exact rational check of l_1=1/3 and monotonicity for eps=1/20,7/10,1: ok")
    for e in (0.69, 0.70, 0.73):
        l, mu = seq(e, 40); ls, ms = limits(e)
        print("eps=%.2f  l_40=%.12f mu_40=%.12f  l*=%.12f mu*=%.12f 1-mu*=%.6f  varpi=%.6f  (l_5=%.9f mu_5=%.9f)" % (e, l[40], mu[40], ls, ms, 1-ms, varpi(e), l[5], mu[5]))
    print("eps=1: limits", limits(1.0), (3-math.sqrt(5))/2)
    print("six claims:")
    for H, e, bound in ((3, 0.73, 0.964), (8, 0.70, 0.977), (16, 0.69, 0.995)):
        l, mu = seq(e, 40); c = l[40]-1e-4; beta = 1-mu[40]-1e-4
        a = Theta(H, e, c, beta); b = Theta_brute(H, e, c, beta)
        print("  H=%d eps=%.2f c=%.6f beta=%.6f eps*beta=%.4f r=%.4f  Theta=%.6f (brute %.6f) claimed <= %.3f : %s" % (H, e, c, beta, e*beta, 1-e*c, a, b, bound, a <= bound))
    for H, e, bound in ((3, 0.74, 0.985), (8, 0.72, 0.972), (16, 0.71, 0.994)):
        c = varpi(e)-1e-4; beta = 0.4999
        a = Theta(H, e, c, beta); b = Theta_brute(H, e, c, beta)
        print("  H=%d eps=%.2f c=%.6f beta=%.4f  Theta=%.6f (brute %.6f) claimed <= %.3f : %s" % (H, e, c, beta, a, b, bound, a <= bound))
    print("table 1: Theta_H(eps; l*, 1-mu*)")
    for H in (2, 3, 4, 6, 8, 16, 64):
        row = []
        for e in (0.69, 0.70, 0.71, 0.72, 0.73):
            ls, ms = limits(e); row.append(Theta(H, e, ls, 1-ms))
        lo, hi = 0.5, 0.9
        for _ in range(60):
            mid = (lo+hi)/2; ls, ms = limits(mid)
            if Theta(H, mid, ls, 1-ms) > 1: lo = mid
            else: hi = mid
        print("  %2d " % H + " ".join("%.4f" % v for v in row) + "  root %.4f" % lo)
    for H in (3, 8):
        ls, ms = limits(0.6); print("eps=0.6 Theta_%d(l*,1-mu*) = %.3f" % (H, Theta(H, 0.6, ls, 1-ms)))
    # monotonicity in H and T_m <= r^m for beta up to ~1-c (eps*beta <= r)
    bad = 0
    for e in (0.05, 0.3, 0.69, 0.73, 0.95):
        ls, ms = limits(e)
        for c in (0.05, ls-1e-4, 0.2):
            for beta in (0.0, 0.3, 1-ms-1e-4, 0.57, min(0.99, 1-c-1e-9)):
                r = 1-e*c
                assert e*beta <= r
                for H in range(2, 40):
                    for m in range(0, 70):
                        a = max(Tm(H, m, e, c, beta)); b = max(Tm(H+1, m, e, c, beta))
                        if b > a*(1+1e-12) or a > r**m*(1+1e-12): bad += 1
    print("violations of T_m^{H+1}<=T_m^H or T_m<=r^m with beta up to 1-c:", bad)
