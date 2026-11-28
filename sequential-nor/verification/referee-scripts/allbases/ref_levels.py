#!/usr/bin/env python3
"""Referee: analytic claims of Section "All bases" (own computation, rational brackets)."""
from fractions import Fraction as Fr
from math import isqrt

P = 10 ** 40

def varpi_bracket(eps):
    """rational lo < varpi(eps) < hi, varpi = (2e+1-sqrt(4e+1))/(2e)."""
    q = 4 * eps + 1
    s0 = isqrt(q.numerator * P * P // q.denominator)
    s_lo = Fr(s0 - 1, P); s_hi = Fr(s0 + 1, P)
    assert s_lo * s_lo < q < s_hi * s_hi
    lo = (2 * eps + 1 - s_hi) / (2 * eps); hi = (2 * eps + 1 - s_lo) / (2 * eps)
    # sanity: varpi - eps(1-varpi)^2 is increasing in varpi
    assert lo - eps * (1 - lo) ** 2 < 0 < hi - eps * (1 - hi) ** 2 and 0 < lo < hi < Fr(1, 2)
    return lo, hi

def chi(eps, w):
    return (1 - eps) * (1 - eps * eps * w * (1 - w)) - eps * w

def kappa2(eps, w):
    r = 1 - eps * w
    return r * r + (1 - eps) * (1 - eps / 2 + r)

def bracket(f, eps):
    """(lower, upper) bounds of f(eps, varpi(eps)) for f decreasing in varpi on (0,1/2)."""
    lo, hi = varpi_bracket(eps)
    return f(eps, hi), f(eps, lo)

for e in (Fr(7287, 10000), Fr(7288, 10000)):
    a, b = bracket(chi, e)
    print("chi(%s) in [%.12f, %.12f]" % (e, a, b))
a, b = bracket(chi, Fr(7287, 10000)); assert a > 0
a, b = bracket(chi, Fr(7288, 10000)); assert b < 0
a, b = bracket(kappa2, Fr(7287, 10000)); assert b < Fr(96, 100)
print("kappa_2(0.7287) <= %.10f < 0.96" % b)
print("=> 0.7287 < eps_1 < 0.7288 and kappa_2 < 0.96 on [0.7287,1] (kappa_2 decreasing): confirmed")

# monotonicity on a grid (sanity only; the argument in the text is analytic)
prev = None
for i in range(1, 1000):
    e = Fr(i, 1000)
    lo, hi = varpi_bracket(e)
    cur = (chi(e, lo), chi(e, hi), kappa2(e, lo), kappa2(e, hi), lo, hi)
    if prev is not None:
        assert cur[0] < prev[1] and cur[2] < prev[3] and cur[4] > prev[5]
    prev = cur
print("grid 0.001..0.999: chi, kappa_2 strictly decreasing, varpi increasing: ok")

def root(f, lo, hi, it=60):
    """bisection for the root of eps -> f(eps, varpi(eps)), f decreasing; float varpi."""
    from math import sqrt
    def g(e):
        w = (2 * e + 1 - sqrt(4 * e + 1)) / (2 * e)
        return f(e, w)
    assert g(lo) > 0 > g(hi)
    for _ in range(it):
        m = (lo + hi) / 2
        if g(m) > 0:
            lo = m
        else:
            hi = m
    return lo

print("eps_1 = root of chi                         = %.6f" % root(chi, 0.5, 0.9))
print("root of kappa_2 = 1                          = %.6f" % root(lambda e, w: kappa2(e, w) - 1, 0.5, 0.9))
# Remark (ii): (r+(1-eps)(1-eta_2))(r+1-eps) = 1
def rem2(e, w):
    r = 1 - e * w; eta2 = e * e * w * (1 - w)
    return (r + (1 - e) * (1 - eta2)) * (r + 1 - e) - 1
print("Remark (ii): root of (r+(1-e)(1-eta2))(r+1-e)=1 = %.6f" % root(rem2, 0.5, 0.9))
e = Fr(7394, 10000); lo, hi = varpi_bracket(e); print("   value at 0.7394: [%.3e, %.3e];" % (rem2(e, hi), rem2(e, lo)), end=" ")
e = Fr(7393, 10000); lo, hi = varpi_bracket(e); print("at 0.7393: [%.3e, %.3e]" % (rem2(e, hi), rem2(e, lo)))
print("3/4 level: root of (1-eps) = eps*varpi         = %.6f" % root(lambda e, w: (1 - e) - e * w, 0.5, 0.9))

# Remark (iii): conditioning m gates below.  a(t)=eps(1-t) <= P[gate reads 0 | x_{i-1}=0] <= r(t)=1-eps t.
# u_m = inf P[x_{k-1}=1 | F_{k-1-m}], v_m = sup:  u_m >= a(1-v_{m-1}), v_m <= r(1-u_{m-1}), u_0=0, v_0=1.
def u_m(e, t, m):
    a = e * (1 - t); r = 1 - e * t
    u, v = 0.0, 1.0
    for _ in range(m):
        u, v = a * (1 - v), r * (1 - u)
    return u
def u_inf(e, t):
    a = e * (1 - t); r = 1 - e * t
    return a * (1 - r) / (1 - a * r)
print()
print("Remark (iii).  eta_m(t) = lower bound for P[x_{k-1}=1 | F_{k-1-m}] at density of ones t")
e = 0.72
from math import sqrt
w = (2 * e + 1 - sqrt(4 * e + 1)) / (2 * e)
for m in (2, 3, 4, 5, 6, 8, 12, 40):
    print("  eps=0.72  m=%2d   eta_m(varpi)=%.5f   eta_m(1/2)=%.5f   min over t in [varpi,1/2] = %.5f" % (
        m, u_m(e, w, m), u_m(e, 0.5, m), min(u_m(e, w + (0.5 - w) * i / 2000, m) for i in range(2001))))
def lvl(etaf):
    return root(lambda e, w: (1 - e) * (1 - etaf(e, w)) - e * w, 0.5, 0.9)
print("  level of theta_3<1 with eta = eta_2(varpi) [the paper]                         : %.5f" % lvl(lambda e, w: u_m(e, w, 2)))
for m in (3, 4, 5, 6, 8, 12):
    print("  m=%2d: eta=eta_m(varpi) (density of ones = varpi): %.5f ;  eta=min_{t in [varpi,1/2]} eta_m(t) (every base): %.5f" % (
        m, lvl(lambda e, w: u_m(e, w, m)), lvl(lambda e, w: min(u_m(e, w + (0.5 - w) * i / 400, m) for i in range(401)))))
print("  m=inf: density varpi: %.5f ;  every base (min over t): %.5f" % (
    lvl(u_inf), lvl(lambda e, w: min(u_inf(e, w + (0.5 - w) * i / 400) for i in range(401)))))
