#!/usr/bin/env python3
"""Certificate for the latch embedding at noise 2/5 (Corollary "noise 2/5" of the note).

Checks, in exact rational arithmetic with outward rounding to multiples of 10^-18, that the
parameters  eps = 2/5, theta = 1/2, h = 40, J = 75, eta = gamma = 1/500  are admissible in the
sense of Definition "admissible parameters": it produces numbers a_k <= a'_k (bounds for the
fraction of ones in a rail that should hold 1) and b'_k <= b_k (for a rail that should hold 0),
k = 0..J, that satisfy conditions (R), (G) and (V) of that definition.

Usage:  python3 -I latch_check.py            (the certificate at 2/5)
        python3 -I latch_check.py table      (floating point: fixed points of the ideal map)
"""
from fractions import Fraction as Fr
from math import isqrt, sqrt, log
import sys

SCALE = 10**18
def dn(x): return Fr((x.numerator*SCALE)//x.denominator, SCALE)
def up(x): return Fr(-((-x.numerator*SCALE)//x.denominator), SCALE)

def varpi_bounds(eps, digits=30):
    # varpi = ((2e+1) - sqrt(4e+1))/(2e); enclose the square root between rationals
    t = 4*eps+1; N = 10**digits
    r = isqrt(t.numerator*N*N*t.denominator)          # floor(sqrt(t)*N*den)
    lo, hi = Fr(r, N*t.denominator), Fr(r+1, N*t.denominator)
    assert lo*lo <= t <= hi*hi
    return ((2*eps+1)-hi)/(2*eps), ((2*eps+1)-lo)/(2*eps)

def F(s, r): return s*(1-r)/(s+r-s*r)

class Cert:
    def __init__(self, eps, theta, h, eta, gamma):
        self.eps, self.theta, self.h, self.eta, self.gamma = eps, theta, h, eta, gamma
        self.wlo, self.whi = varpi_bounds(eps)
        assert 0 < eta < self.wlo
    # read rates: a lower and an upper bound for the probability that a check reads a one,
    # when it points to a copy that holds a one with probability d
    def lm(self, d): return self.eps*(self.wlo-self.eta)+(1-self.eps)*d
    def lp(self, d): return self.eps*(self.whi+self.eta)+(1-self.eps)*d
    def lower(self, dset, dres):
        """lower bound, minus gamma, for a rail whose set checks see density >= dset and whose
        reset checks see density <= dres (chain started in state 0)"""
        s, r = self.lm(dset), self.lp(dres); tau = (1-s)*(1-r)
        return dn(F(s, r)*(1-tau**self.h)-self.gamma)
    def upper(self, dset, dres):
        """upper bound, plus gamma, for a rail whose set checks see density <= dset and whose
        reset checks see density >= dres (chain started in state 1)"""
        s, r = self.lp(dset), self.lm(dres); tau = (1-s)*(1-r); f = F(s, r)
        return up(f+(1-f)*tau**self.h+self.gamma)
    def restore(self, a, b, a2, b2):
        return self.lower(a, b), self.upper(b, a), self.upper(a2, b2), self.lower(b2, a2)
    def gate(self, a, b, a2, b2):
        """bounds (a_0, b_0) for the output rails of the gate stage, inputs in [a,a2] x [b2,b]"""
        th = self.theta
        mlo, mhi = (b2+a)/2, (b+a2)/2
        worlds = {"(0,0)": (th*a, b), "mixed": (mlo, th*mhi), "(1,1)": (a, th*b)}
        out = {k: (self.lower(ds, dr), self.upper(dr, ds)) for k, (ds, dr) in worlds.items()}
        return min(v[0] for v in out.values()), max(v[1] for v in out.values()), out

def certificate():
    eps, theta, h, J = Fr(2, 5), Fr(1, 2), 40, 75
    eta = gamma = Fr(1, 500)
    c = Cert(eps, theta, h, eta, gamma)
    print("eps = %s, theta = %s, h = %d, J = %d, eta = gamma = %s" % (eps, theta, h, J, eta))
    print("varpi(2/5) in [%.15f, %.15f]" % (c.wlo, c.whi))
    # the box claimed for the last stage
    box = dict(a=Fr(663, 1000), a2=Fr(717, 1000), b2=Fr(133, 1000), b=Fr(173, 1000))
    print("box for stage J: rails that should hold 1 in [%s, %s], rails that should hold 0 in [%s, %s]"
          % (box["a"], box["a2"], box["b2"], box["b"]))
    a0, b0, out = c.gate(box["a"], box["b"], box["a2"], box["b2"])
    for k, v in out.items():
        print("  gate stage, inputs %-6s: lower bound %.6f for the rail that should hold 1, upper bound %.6f for the other" % (k, v[0], v[1]))
    assert a0 > b0
    # the rounded values stated in the note; weaker than what (G) gives
    assert Fr(5269, 10000) <= a0 and Fr(3270, 10000) >= b0
    print("(G): least lower bound %.6f >= 0.5269, largest upper bound %.6f <= 0.3270" % (a0, b0))
    a0, b0 = Fr(5269, 10000), Fr(3270, 10000)
    a, b, a2, b2 = a0, b0, Fr(1), Fr(0)
    print("stage  0: a = %.6f  b = %.6f  a' = %.6f  b' = %.6f" % (a, b, a2, b2))
    seq = [(a, b, a2, b2)]
    for k in range(1, J+1):
        na, nb, na2, nb2 = c.restore(a, b, a2, b2)
        assert 0 <= nb2 <= nb <= 1 and 0 <= na <= na2 <= 1
        a, b, a2, b2 = na, nb, na2, nb2
        seq.append((a, b, a2, b2))
        if k % 5 == 0 or k == J:
            print("stage %2d: a = %.6f  b = %.6f  a' = %.6f  b' = %.6f" % (k, a, b, a2, b2))
    ok = a >= box["a"] and b <= box["b"] and a2 <= box["a2"] and b2 >= box["b2"]
    print("(R) and (G): the bounds after J stages lie inside the box:", ok)
    print("(V): a_J - b_J >= %.4f" % (box["a"]-box["b"]))
    assert ok and box["a"] > box["b"]
    # the monotonicity used: separation never decreases along the lower sequence after stage 0
    print("least separation a_k - b_k: %.6f (at stage %d)" % min((s[0]-s[1], i) for i, s in enumerate(seq)))
    # constants of the corollary
    from math import log, ceil
    M = ceil((1+log(12*(J+1)))/(2*float(gamma)**2))
    print("M = ceil((1+ln(12(J+1)))/(2 gamma^2)) = %d" % M)
    rho, tau, k = 1999, Fr(1, 2000), 5
    err = (Fr(1, 1+rho)+tau)/(1-eps)+eps**(2*k-1)/2
    print("density error with rho = %d, tau = %s, k = %d: %.6f <= eta: %s" % (rho, tau, k, err, err <= eta))
    assert err <= eta and Fr(1, 1+rho)+tau <= Fr(1, 4)
    print("(5/tau^2)^(2k+1) = %.3e" % float((5/tau**2)**(2*k+1)))
    print("CERTIFICATE VERIFIED")

def table():
    def varpi(e): return ((2*e+1)-sqrt(4*e+1))/(2*e)
    print(" eps     varpi    x*       y*       pi0      q        lambda   theta-range")
    for e in (0.10, 0.25, 0.30, 1/3, 0.35, 0.40, 0.42, 0.43, 0.44, 4/9-1e-6):
        w = 1-varpi(e); A = e*(1-w); B = e*w
        u = (1-sqrt(4*w-3))/(2*w); v = (1+sqrt(4*w-3))/(2*w)
        x, y = (1-A-u)/(1-e), (1-A-v)/(1-e)
        q = (-(1-B)+sqrt((1-B)**2+4*(1-A)))/2; p0 = (1-A-q)/(1-e)
        lam = (1-e)/(1-q*q)
        # check that (x, y) is a fixed point of the ideal map
        s, r = A+(1-e)*x, A+(1-e)*y
        assert abs(F(s, r)-x) < 1e-9 and abs(F(r, s)-y) < 1e-9 and abs(u*v-e) < 1e-12
        print("%.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f   (%.3f, %.3f)" % (e, 1-w, x, y, p0, q, lam, p0/x, min(1, 2*p0/(x+y))))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "table": table()
    else: certificate()
