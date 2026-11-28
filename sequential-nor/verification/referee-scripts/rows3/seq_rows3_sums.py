# Referee check: the multiple geometric sums after (rows3C) with (rows3A2) inserted.
# Brute force of the coefficient of every u and q, against the constants claimed
# (C_2 for near and source terms, C_2*kappa^{max(e,1)/2} for the start terms).
# The factors 2 eps/m_0 and eps are left out on both sides.
import sys
def check(kap, r1, K, J):
    Cg = 1 / (kap * (1 - kap) ** 2); C2 = Cg ** 3 / (1 - r1) ** 2
    nu2 = {}; nu1 = {}; st2 = 0.0; st1 = {}; sq = {}
    add = lambda d, k, v: d.__setitem__(k, d.get(k, 0.0) + v)
    for j in range(1, J + 1):
        for i in range(-K + 1, j + 1):
            for l in range(-K + 1, j + 1):
                W = kap ** (j - i - 1) * (r1 ** (i - 1 - l) if l < i else 1.0)
                for c in range(-K + 1, i + 1):
                    w = W * kap ** (i - c - 1)
                    add(nu2, min(c - 1, l), w)
                    for e in range(c, l + 1): add(nu1, e, w)
                w = W * kap ** (i + K - 1)
                st2 += w
                for e in range(1 - K, l + 1): add(st1, e, w)
                for e in range(1 - K, l + 1): add(sq, e, W * kap ** max(i - e - 1, 0))
    worst = lambda d: max(d.values())
    r_st1 = max(v / kap ** (max(e, 1) / 2) for e, v in st1.items())
    print("kappa=%.2f r1=%.2f  C_2=%.3g  near u2 %.3g  near u1 %.3g  start u2 %.3g  start u1/kappa^(max(e,1)/2) %.3g  source %.3g"
          % (kap, r1, C2, worst(nu2), worst(nu1), st2, r_st1, worst(sq)))
    assert max(worst(nu2), worst(nu1), st2, r_st1, worst(sq)) <= C2
for kap, r1, J in ((0.5, 0.5, 40), (0.7, 0.6, 60), (0.8, 0.8, 70)):
    check(kap, r1, 8, J)
print("ok")
