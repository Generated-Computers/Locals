"""Asserts the numerical claims of the refined NOR-tree embedding (noise 23/100) of
seqnor-logspace-and-gap.tex: robust zero, pair leaves, double top, heavy zero padding and the
two-sided density lemma.  The recursion is evaluated in exact rational arithmetic and rounded UP
to 40 digits after every step, so the printed values are rigorous upper bounds.
Standard library only."""
from fractions import Fraction as F
import math

SC = 10**40
up = lambda q: F(-((-q.numerator * SC) // q.denominator), SC)       # least multiple of 1e-40 >= q
ln2 = math.log(2)
D = lambda p, q: p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

def varpi_bounds(eps, steps=200):
    """Rational lo < varpi(eps) < hi, where varpi is the root in (0, 1/2) of w = eps (1 - w)^2."""
    f = lambda w: w - eps * (1 - w) ** 2                               # increasing on (0, 1/2)
    lo, hi = F(0), F(1, 2)
    assert f(lo) < 0 < f(hi)
    for _ in range(steps):
        mid = (lo + hi) / 2
        mid = F(round(mid * 10**30), 10**30)
        if f(mid) < 0: lo = mid
        else: hi = mid
    return lo, hi

def recursion(A, B, d, w0, w1):
    """Upper bounds on (Q0, Q1) for the refined design with flag probabilities A, B."""
    zf = up((1 - (1 - B) * (1 - A)) * B)                                  # robust zero fails
    l0 = up(1 - (1 - w0) ** 2 * (1 - zf) * (1 - A) ** 2)                  # pair leaf, input 0
    l1 = up((1 - (1 - w1) * (1 - B)) * (1 - (1 - w1) * (1 - B) * (1 - A)))  # pair leaf, input 1
    OR = lambda p: up(1 - (1 - p) ** 2 * (1 - A))                         # node intended 1
    AND = lambda p: up(p * (1 - (1 - p) * (1 - B)))                       # node intended 0
    p1, p0 = l1, l0
    for h in range(1, d + 1):
        p1 = OR(p1) if h % 2 else AND(p1)
        p0 = AND(p0) if h % 2 else OR(p0)
    r1, r0 = p1, p0
    Q1 = up(1 - (1 - r0) ** 2 * (1 - A) ** 2)                             # cell (0,0)
    Q0a = up(1 - (1 - r1) * (1 - B * (1 - (1 - B) * (1 - A))))            # cells (1,0), (1,1)
    Q0b = up(1 - (1 - r1) * (1 - A))                                      # cell (0,1)
    return max(Q0a, Q0b), Q1

# --- density constants ------------------------------------------------------------------------
rho, tau, E = 999, F(1, 2000), F(1, 400)
eta = F(1, rho + 1) + tau
for e in (F(1, 4), F(23, 100), F(1, 10), F(1, 100)):                   # the bound is increasing in eps
    assert eta / (1 - e) + e ** 5 / 2 <= E
print("density error: eta/(1-eps) + eps^5/2 <= %.6f <= E = 0.0025 for eps <= 1/4" % float(eta / F(3, 4) + F(1, 4) ** 5 / 2))
# stage bound Q_k <= (5/tau^2)^(k+1) W, checked by iterating the definition with W = 3 and c = 0
# (the worst case of the inequalities used): Q' = ceil(2(Q+2+c)/tau) + ceil(4 (K+2) ln(n+1) / tau^2)
for W in (3, 10, 10**6):
    Q = math.ceil(1 / (2 * tau))
    for k in range(6):
        Qn = math.ceil(2 * (Q + W) / tau) + math.ceil(4 * W / tau ** 2)
        assert Qn <= (5 / tau ** 2) ** (k + 2) * W
        Q = Qn
assert (5 / tau ** 2) ** 7 <= F(13, 10) * 10**51
print("pre must be at least (5/tau^2)^7 W <= 1.3e51 W")

# --- the recursion at noise 23/100 ---------------------------------------------------------------
eps = F(23, 100)
vlo, vhi = varpi_bounds(eps)
A = eps * (vhi + E)
B = eps * (1 - vlo + E)
print("varpi(0.23) in [%.9f, %.9f];  A <= %.6f, B <= %.6f" % (float(vlo), float(vhi), float(A), float(B)))
d, w0, w1 = 25, F(7, 40), F(9, 40)
Q0, Q1 = recursion(A, B, d, w0, w1)
assert Q0 <= F(1161, 10000) and Q1 <= F(1563, 10000), (float(Q0), float(Q1))
kap = 0.0148
assert D(0.175, 0.1161) >= kap and D(0.225, 0.1563) >= kap
assert kap * 47 / ln2 >= 1
print("noise 23/100, d = 25, tolerances 7/40, 9/40:  Q0 <= %.6f  Q1 <= %.6f  kappa >= %.4f" % (float(Q0), float(Q1), kap))

# A(eps) = eps (varpi(eps) + E) and B(eps) = eps (1 - varpi(eps) + E) are increasing on (0, 23/100]:
grid = [F(k, 400) for k in range(1, 93)]                               # 0.0025 .. 0.23
prevA = prevB = F(0)
for e in grid:
    lo, hi = varpi_bounds(e, steps=80)
    a_hi, b_hi = e * (hi + E), e * (1 - lo + E)
    a_lo, b_lo = e * (lo + E), e * (1 - hi + E)
    assert a_lo >= prevA - F(1, 10**20) and b_lo >= prevB - F(1, 10**20)
    prevA, prevB = a_hi, b_hi
    assert a_hi <= A + F(1, 10**20) and b_hi <= B + F(1, 10**20)
print("A(eps) and B(eps) increase on the grid up to 0.23 and stay below their values at 0.23")
# the recursion at smaller noise (spot check of monotonicity)
for e in (F(1, 20), F(1, 10), F(3, 20), F(1, 5)):
    lo, hi = varpi_bounds(e)
    q0, q1 = recursion(e * (hi + E), e * (1 - lo + E), d, w0, w1)
    assert q0 <= F(1161, 10000) and q1 <= F(1563, 10000)

# --- the recursion at noise 1/4 (the statement of the note) -------------------------------------
eps4 = F(1, 4)
vlo4, vhi4 = varpi_bounds(eps4)
A4 = eps4 * (vhi4 + E)
B4 = eps4 * (1 - vlo4 + E)
print("varpi(1/4) in [%.10f, %.10f];  A <= %.7f, B <= %.7f" % (float(vlo4), float(vhi4), float(A4), float(B4)))
assert F(17157287, 10**8) <= vlo4 and vhi4 <= F(17157288, 10**8)
assert A4 <= F(43519, 10**6) and B4 <= F(207732, 10**6)
d4, v0, v1 = 51, F(4, 25), F(91, 500)
# the recursion is evaluated with the ROUNDED values printed in the note, which dominate A4, B4
A4r, B4r = F(43519, 10**6), F(207732, 10**6)
P0, P1 = recursion(A4r, B4r, d4, v0, v1)
assert P0 <= F(13848, 10**5) and P1 <= F(15904, 10**5), (float(P0), float(P1))
kap4 = 0.00185
assert D(0.16, 0.13848) >= kap4 and D(0.182, 0.15904) >= kap4
assert kap4 * 375 / ln2 >= 1
print("noise 1/4, d = 51, tolerances 4/25, 91/500, A <= 0.043519, B <= 0.207732:  Q0 <= %.7f  Q1 <= %.7f  kappa >= %.5f" % (float(P0), float(P1), kap4))
print("   D-values %.6f %.6f" % (D(0.16, 0.13848), D(0.182, 0.15904)))
grid = [F(k, 400) for k in range(1, 101)]                              # 0.0025 .. 0.25
prevA = prevB = F(0)
for e in grid:
    lo, hi = varpi_bounds(e, steps=80)
    assert e * (lo + E) >= prevA - F(1, 10**20) and e * (1 - hi + E) >= prevB - F(1, 10**20)
    prevA, prevB = e * (hi + E), e * (1 - lo + E)
    assert prevA <= A4 + F(1, 10**20) and prevB <= B4 + F(1, 10**20)
for e in (F(1, 20), F(1, 10), F(3, 20), F(1, 5), F(23, 100)):
    lo, hi = varpi_bounds(e)
    q0, q1 = recursion(e * (hi + E), e * (1 - lo + E), d4, v0, v1)
    assert q0 <= F(13848, 10**5) and q1 <= F(15904, 10**5)
print("A(eps), B(eps) increase on the grid up to 1/4; recursion at five smaller noise levels below the bounds")
size4 = 2 * (7 * 2**d4 - 1) + 3
assert size4 == 7 * 2**(d4 + 1) + 1 == 31525197391593473
# Lemma (parameters) for d = 51: the inequalities used, at the extreme cases a = 1, s = 2
c4 = size4
for a_ in (1, 2, 5, 50):
    for sm in (1500, 10**4, 10**9):
        lnZ = 68 * math.log(10) + 2 * math.log(a_ + 2) + math.log(sm)
        assert (a_ + 3) * (lnZ + 1) <= (a_ + 2) ** 2 * sm
        pre_bound = 1.3e51 * (c4 + 3 + (a_ + 2) ** 2 * sm) + 1
        assert pre_bound <= 4.2e67 * (a_ + 2) ** 2 * sm
        assert pre_bound + 1 + 1000 * c4 * sm <= 1e68 * (a_ + 2) ** 2 * sm
    s_ = 1e76 * (a_ + 2) ** 4
    assert math.log2(s_) <= 253 * (a_ + 2)
    assert s_ / math.log2(s_) >= 7.5e70 * (a_ + 2) ** 3
print("parameter inequalities for d = 51 hold on the sample")

# --- sizes -----------------------------------------------------------------------------------------
size = lambda h: 6 if h == 0 else 2 * size(h - 1) + 1
assert all(size(h) == 7 * 2**h - 1 for h in range(30))
c = 2 * size(d) + 3
assert c == 7 * 2**(d + 1) + 1 == 469762049
print("cell size c = %d" % c)
print("all assertions passed")
