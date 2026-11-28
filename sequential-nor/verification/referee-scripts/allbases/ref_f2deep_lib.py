from fractions import Fraction as Fr
def cond_x1(base, xs, eps, upto):
    """P[x_upto = 1 | x_0..x_{len(xs)-1}] exactly."""
    m = len(xs) - 1
    if m == upto:
        return Fr(xs[upto])
    g = m + 1
    tot = Fr(0)
    for p, w in [(base[g], 1 - eps)] + [(p, eps / g) for p in range(g)]:
        xg = 1 if (xs[g-1] == 0 and xs[p] == 0) else 0
        tot += w * cond_x1(base, xs + (xg,), eps, upto)
    return tot
