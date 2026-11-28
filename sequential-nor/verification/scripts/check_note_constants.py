"""Asserts the numerical inequalities used in seqnor-logspace-and-gap.tex.
Standard library only.  Run:  python3 check_note_constants.py"""
import itertools, math
from fractions import Fraction as F

ln2 = math.log(2)
D = lambda p, q: p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

# --- Section 3: constants of Lemma (density), Proposition (embedding), Lemma (parameters)
assert 0.0111 * 65 / ln2 > 1
assert D(1 / 6, 0.1161) >= 0.0111
assert D(1 / 30, 1 / 64) >= 0.0077
assert 0.0077 * 130 >= 1
assert 92 / ln2 > 130
assert 3**10 * 4 + (3**10 - 1) // 2 * 6 == 413340
assert 2236 + 184 + 92 + 502 + 413340 == 416354 <= 10**7
assert 92 * 24.3 <= 2236 and math.log2(1e7) + 1 <= 24.3

# recursion of (P3) at eps = 1/64, exact rationals
phi = lambda q: 3 * q * q - 2 * q**3
e = F(1, 64)
q = 1 - (1 - F(2931, 1000) * e) * F(25, 36)
for _ in range(10):
    q = 1 - (1 - F(486, 100) * e) * (1 - phi(q))
assert q <= F(1161, 10000), float(q)

# gadgets N and M' on all inputs (x_0 = 1, x_1 = 0)
NOR = lambda a, b: 1 - (a | b)
for a, b, c, prev in itertools.product((0, 1), repeat=4):
    n0 = NOR(prev, 1); n1 = NOR(n0, a); n2 = NOR(n1, 0); n3 = NOR(n2, b)      # N(a,b) = (0,a,1,b)
    assert n3 == NOR(a, b) and n0 + n1 + n2 + n3 >= 1
for a, b, c in itertools.product((0, 1), repeat=3):                            # M'(a,b,c), c = x_{k-1}
    y0 = NOR(c, a); y1 = NOR(y0, a); y2 = NOR(y1, c); y3 = NOR(y2, y1); y4 = NOR(y3, b); y5 = NOR(y4, y0)
    assert y5 == int(a + b + c >= 2) and y0 + y1 + y2 + y3 + y4 + y5 >= 2

# Lemma (parameters) on a grid
def comp(a, s):
    x = 65 * (a + 1) * math.log2(s)
    m = 1
    while m < x:
        m *= 2
    Z = 10**7 * (a + 2) ** 2 * s * m
    b = (Z - 1).bit_length()                    # ceil(log2 Z)
    P = 23 * (a + 2) * b + 125
    return m, Z, P, 1 + 4 * P + 413340 * s * m
for a in (1, 2, 6, 10, 50, 1000, 10**6):
    for s in (2, 3, 10, 1000, 10**6, 10**12, 10**30, 10**100):
        m, Z, P, n = comp(a, s)
        assert n + 1 <= Z                                                   # (i)
        assert 2 + 4 * P > 130 * (a + 2) * math.log(n + 1) + 500             # (ii)
        assert m < 130 * (a + 1) * math.log2(s)                              # (iii), first part
        if s >= 10**11 * (a + 2) ** 4:
            assert n + 1 <= s * s                                            # (iii), second part
    sa = 10**11 * (a + 2) ** 4
    assert math.log2(sa) <= 14.4 * (a + 2)
    assert comp(a, sa)[3] + 1 <= sa * sa

# proof of Theorem (hardness): 2*delta + 3*nu <= n^-K / 2 for n >= 3
for K in range(1, 12):
    for n in (3, 4, 10, 10**3, 10**9):
        assert 7 * F(1, (n + 1) ** (K + 2)) <= F(1, 2 * n**K)
# Corollary (a): e^{-2(n+1)} <= 2^{-n}/(n+1)
for n in range(1, 200):
    assert math.exp(-2 * (n + 1)) <= 2.0**-n / (n + 1)

# --- Section 2: constants of the Remark for eps = 1
eps, eta, lam = 1.0, 0.126, 13.1
c = eps / (2 + eps) - eta
theta = (eps / (lam + 1) + 1 - eps) / (eps * c)
vartheta = theta / (eps * c)
Theta = max(1.0, vartheta)
a_ = (lam + 2) / math.log(1 / theta)
kappa = (2 / (0.9 * eta * (2 + eps))) ** 2
assert abs(c - 0.2073) < 1e-3 and abs(theta - 0.342) < 1e-3 and abs(vartheta - 1.650) < 1e-3
assert theta < 1 and kappa <= 34.6
g = 3
while (math.log(g) + 4) / (2 * g) > eta / 10:
    g += 1
assert g == 397
for k in range(1, 400):                       # n + 1 = 2^k
    n = 2**k - 1
    if n < 2:
        continue
    L = math.ceil(((lam + 1) * math.log(n + 1) + math.log(n)) / math.log(1 / theta))
    log2M = math.log2(n) + (lam + 1) * k + math.log2(L) + L * math.log2(Theta)
    B0 = 2 * log2M + 2 * k + L
    assert B0 <= 56.1 * k + 2 * math.log2(k) + 10, (k, B0)
    C1 = 2 * (lam + 3) + a_ * (3 + 2 * math.log2(Theta)); C2 = 3 + 2 * math.log2(Theta)
    assert B0 <= C1 * k + C2
print("all checks passed")
