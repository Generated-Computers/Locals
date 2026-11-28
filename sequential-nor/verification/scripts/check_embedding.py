"""Asserts the numerical claims of Section 4 (the NOR-tree embedding) of seqnor-logspace-and-gap.tex.
The recursion is evaluated in exact rational arithmetic and rounded UP to 40 digits after every step,
so the printed values are rigorous upper bounds.  Standard library only."""
from fractions import Fraction as F
import math

SC = 10**40
up = lambda q: F(-((-q.numerator * SC) // q.denominator), SC)      # least multiple of 1e-40 that is >= q
ln2 = math.log(2)
D = lambda p, q: p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

def recursion(eps, abar, d, w0, w1):
    A, B = eps * abar, eps
    OR = lambda p: up(1 - (1 - p) ** 2 * (1 - A))          # node intended 1, children intended 0
    AND = lambda p: up(p * (1 - (1 - p) * (1 - B)))         # node intended 0, children intended 1
    p = up(1 - (1 - w1) * (1 - B))                          # leaf over a bundle of value 1
    for h in range(1, d + 1):
        p = OR(p) if h % 2 else AND(p)
    r1 = p
    p = up(1 - (1 - w0) * (1 - B) * (1 - A))                # leaf over a bundle of value 0
    for h in range(1, d + 1):
        p = AND(p) if h % 2 else OR(p)
    r0 = p
    Q1 = up(1 - (1 - r0) ** 2 * (1 - A))                    # top gate, inputs (0,0)
    Q0 = up(1 - (1 - r1) * (1 - B))                         # top gate, inputs (1,0); dominates (1,1) and (0,1)
    assert AND(r1) <= Q0 and r1 <= Q0
    return Q0, Q1

# block sizes
size = lambda h: 2 if h == 0 else 2 * size(h - 1) + 1
assert all(size(h) == 3 * 2**h - 1 for h in range(0, 22))
assert 2 * size(15) + 1 == 196607 and 2 * size(19) + 1 == 3145727

# Corollary (noise 1/10): no padding, abar = 0.501, d = 15, tolerances 3/10 and 7/50
Q0, Q1 = recursion(F(1, 10), F(501, 1000), 15, F(3, 10), F(7, 50))
assert Q0 <= F(2311, 10000) and Q1 <= F(900, 10000), (float(Q0), float(Q1))
assert D(0.3, 0.2311) >= 0.0125 and D(0.14, 0.09) >= 0.0125
assert 0.0125 * 56 / ln2 >= 1
print("noise 1/10 : Q0 <= %.6f  Q1 <= %.6f" % (float(Q0), float(Q1)))

# Corollary (noise 3/20): zero padding, abar = 0.17, d = 19, tolerances 3/10 and 1/10
Q0, Q1 = recursion(F(3, 20), F(17, 100), 19, F(3, 10), F(1, 10))
assert Q0 <= F(2099, 10000) and Q1 <= F(493, 10000), (float(Q0), float(Q1))
assert D(0.3, 0.2099) >= 0.0214 and D(0.1, 0.0493) >= 0.0214
assert 0.0214 * 33 / ln2 >= 1
print("noise 3/20 : Q0 <= %.6f  Q1 <= %.6f" % (float(Q0), float(Q1)))

# the bounds are monotone in the noise: spot check at smaller noise
for e in (F(1, 100), F(1, 20), F(2, 25)):
    q0, q1 = recursion(e, F(501, 1000), 15, F(3, 10), F(7, 50)); assert q0 <= F(2311, 10000) and q1 <= F(9, 100)
for e in (F(1, 100), F(1, 10), F(1, 8)):
    q0, q1 = recursion(e, F(17, 100), 19, F(3, 10), F(1, 10)); assert q0 <= F(2099, 10000) and q1 <= F(493, 10000)

# Lemma (density under zero padding)
assert D(0.1648, 0.15) >= 0.000836 and 1200 * 0.000836 >= 1
assert 0.1648 + 0.0052 <= 0.17
for g in (10**4, 10**5, 10**9):
    assert 1 + g / 200 + g / (99 * 102) <= 0.0052 * g
assert 501 / 1000 >= (500 + 1) / (2 * 500)

# Lemma (parameters of the padded compiler)
assert 100 * 3145727 <= 3.15e8 and 24864 + 1680 + 840 + 10001 == 37385 and 3.15e8 + 37385 <= 4e8
assert 840 * 29.6 <= 24864 and math.log2(4e8) + 1 <= 29.6 and 840 / ln2 >= 1200
def comp(a, s):
    x = 33 * (a + 1) * math.log2(s); m = 1
    while m < x: m *= 2
    Z = 4 * 10**8 * (a + 2) ** 2 * s * m
    b = (Z - 1).bit_length()
    pre = 840 * (a + 2) * b + 10**4
    return m, Z, pre, pre + 100 * 3145727 * s * m
for a in (1, 2, 6, 10, 50, 1000, 10**6):
    for s in (2, 3, 10, 1000, 10**6, 10**12, 10**30, 10**100):
        m, Z, pre, n = comp(a, s)
        assert n + 1 <= Z
        assert pre >= 1200 * (a + 2) * math.log(n + 1) + 10**4
        assert m < 66 * (a + 1) * math.log2(s)
        if s >= 10**13 * (a + 2) ** 4: assert n + 1 <= s * s
    sa = 10**13 * (a + 2) ** 4
    assert math.log2(sa) <= 16.6 * (a + 2) and comp(a, sa)[3] + 1 <= sa * sa
print("all checks passed")
