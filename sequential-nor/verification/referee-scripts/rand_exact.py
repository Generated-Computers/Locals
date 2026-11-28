# randomized-parameter exact check of Lemma onestep (rational arithmetic), plus rigorous rational
# verification of the three numerical hypotheses of the theorem, plus numeric check of W_H h <= theta_H h.
import sys, random, math
from fractions import Fraction as F
sys.path.insert(0, sys.argv[1])
import exact

rnd = random.Random(12345)
worst = F(0); n = 0
for trial in range(150):
    H = rnd.choice((2, 3, 3, 4, 5, 6))
    eps = F(rnd.randint(5, 95), 100)
    g0 = rnd.choice((1, 2, 3, 4, 5))
    bmax = F(1, 2) - F(1, 2*g0)                 # g0 >= 1/(1-2 beta)  <=>  beta <= 1/2 - 1/(2 g0)
    beta = bmax*F(rnd.randint(0, 10), 10)
    c = F(rnd.randint(0, 45), 100)
    Pmax = g0 + rnd.randint(1, 4); mmax = rnd.randint(1, 7)
    exact.check_onegate(H, eps, c, beta, g0, 10)
    w = exact.check_onestep(H, eps, c, beta, g0, Pmax, mmax); n += 1
print("random exact onestep/onegate checks passed:", n)

# rigorous: rational c < varpi(eps) (tested by eps(1-c)^2 - c > 0), rational beta, exact Theta_H via geometric tail
def Theta_exact(H, eps, c, beta):
    r = 1-eps*c
    s = sum(max(exact.Tm(H, m, eps, c, beta)) for m in range(H-1)) + max(exact.Tm(H, H-1, eps, c, beta))/(1-r)
    return (1-eps)*s
for H, eps, c, bound in ((3, F(74,100), F(3309,10000), F(985,1000)), (8, F(72,100), F(32644,100000), F(971,1000)), (16, F(71,100), F(3241,10000), F(994,1000))):
    beta = F(4999, 10000)
    assert eps*(1-c)**2 - c > 0 and 0 < c < F(1,2)
    th = Theta_exact(H, eps, c, beta)
    print("H=%d eps=%s c=%s (<varpi, checked exactly) beta=%s: Theta_H = %.7f  <1: %s  <= %s: %s" % (H, eps, c, beta, float(th), th < 1, bound, th <= bound))

# kernel inequality (W_H h)(P) <= theta_H h(P), floats
def Tm(H, m, eps, c, beta):
    return max(exact.Tm(H, m, eps, c, beta))
for (H, eps, lam, g0) in ((3, 0.74, 400.0, 50), (8, 0.72, 300.0, 20), (4, 0.9, 5.0, 3)):
    c = ((2*eps+1) - math.sqrt(4*eps+1))/(2*eps) - 1e-4; beta = 0.4999; r = 1-eps*c
    Th = (1-eps)*(sum(Tm(H, m, eps, c, beta) for m in range(H-1)) + Tm(H, H-1, eps, c, beta)/(1-r))
    th = 1/((lam+1)*c) + Th
    T = [Tm(H, m, eps, c, beta) for m in range(0, 3001)]
    worst = 0.0
    for P in list(range(g0, g0+60)) + [500, 1000, 3000]:
        # work with h(q)/h(P) to avoid overflow
        logs = [lam*math.log(q+1) for q in range(P+1)]
        hP = logs[P]
        # prefix sums of h(q)/h(P) for q>=g0
        pre = [0.0]*(P+2)
        for q in range(P+1):
            pre[q+1] = pre[q] + (math.exp(logs[q]-hP) if q >= g0 else 0.0)
        tot = 0.0
        for k in range(g0+1, P+1):
            tot += r**(P-k)*eps/k*pre[k-1]           # sum_{q=g0}^{k-2}
            if k-H >= g0: tot += (1-eps)*T[P-k]*math.exp(logs[k-H]-hP)
        worst = max(worst, tot/th)
    print("kernel: H=%d eps=%.2f lambda=%g g0=%d Theta=%.4f theta=%.4f  max (W h)(P)/(theta h(P)) = %.4f" % (H, eps, lam, g0, Th, th, worst))
