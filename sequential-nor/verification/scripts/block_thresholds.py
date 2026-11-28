"""Numerical facts quoted in Section 10 (block bases).  Standard library only.

q(eps) solves  q = 1 - (1-eps) q/(1+q) - eps*varpi(eps);  d_eps = q/(1+q).
Checked: the golden-ratio identities for d* = (3-sqrt 5)/2; q(eps) - eps changes sign exactly once,
at the root eps_K of eps^3 + 4 eps^2 + 3 eps - 4; q(4/9) = 2/3 and varpi(4/9) = 1/4 exactly;
q(eps)^2 - eps is positive below 4/9 and negative above, on a grid of step 1e-4."""
from fractions import Fraction as F
import math

def varpi(e): return ((2 * e + 1) - math.sqrt(4 * e + 1)) / (2 * e)

def q_of(e):
    w = varpi(e); f = lambda q: 1 - (1 - e) * q / (1 + q) - e * w - q      # decreasing in q
    lo, hi = 0.0, 1.0
    assert f(lo) > 0 > f(hi)
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(mid) > 0: lo = mid
        else: hi = mid
    return (lo + hi) / 2

ds = (3 - math.sqrt(5)) / 2
assert abs(ds - (1 - ds) ** 2) < 1e-15 and abs(ds * ds - 3 * ds + 1) < 1e-15
assert abs(1 / (2 - ds) - (1 - ds)) < 1e-15                      # zeta -> 1 - d*
assert abs((1 - ds) / (2 - ds) - ds) < 1e-15                     # h(d*) = d*
print("d* = %.6f, 1 - d* = %.6f" % (ds, 1 - ds))

# exact values at 4/9
e = F(4, 9); w = F(1, 4); q = F(2, 3)
assert w == e * (1 - w) ** 2 and q == 1 - (1 - e) * q / (1 + q) - e * w and q * q == e
assert q / (1 + q) == F(2, 5)
print("eps = 4/9: varpi = 1/4, q = 2/3, block density 2/5, q^2 = eps")

# the root of the cubic and the sign of q(eps) - eps
cub = lambda x: x ** 3 + 4 * x ** 2 + 3 * x - 4
lo, hi = 0.0, 1.0
for _ in range(80):
    mid = (lo + hi) / 2
    if cub(mid) < 0: lo = mid
    else: hi = mid
eK = (lo + hi) / 2
assert abs(q_of(eK) - eK) < 1e-12
print("eps_K = %.6f, q(eps_K) = %.6f, varpi = %.6f, block density = %.6f" % (eK, q_of(eK), varpi(eK), q_of(eK) / (1 + q_of(eK))))
for k in range(1, 10000):
    e = k / 10000
    qq = q_of(e)
    assert (qq > e) == (e < eK), e
    if abs(e - 4 / 9) > 1e-4: assert (qq * qq > e) == (e < 4 / 9), e
print("q(eps) > eps exactly for eps < eps_K, and q(eps)^2 > eps exactly for eps < 4/9, on the grid")
# growth rates quoted in the proofs
for e in (0.30, 0.45, 0.55, 0.66):
    print("eps %.2f: no zero gates: (1-eps)/d* = %.3f, (1-eps)/(1-d*) = %.3f;  over zero gates: (1-eps)/(1-q) = %.3f, (1-eps)/(1-q^2) = %.3f"
          % (e, (1 - e) / ds, (1 - e) / (1 - ds), (1 - e) / (1 - q_of(e)), (1 - e) / (1 - q_of(e) ** 2)))
print("all checks passed")
