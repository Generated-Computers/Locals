# Ordered bases (non-NOT pointers increase with the gate).  Checks for the note:
#  1. the closed-form factor Theta_ord_D = (1-eps) * ( 1/(eps*c + gamma*(1-eps)) + r^D/(eps*c) ),
#     gamma = eps^2*beta*c, and the noise level where it equals 1;
#  2. the lemma behind it: for every spacing sequence d_1, d_2, ... >= 1,
#     sum_u || B_{d_u} ... B_{d_1} 1 ||_inf <= 1/(1 - s_0), with the exact operators;
#  3. a simulation of three ordered bases.
# Standard library only.
import math, random, itertools
def varpi(e): return ((2*e+1) - math.sqrt(4*e+1))/(2*e)
def consts(e, c, beta):
    r = 1-e*c; e0 = e*(1-c); alo = e*beta; gamma = e*e*beta*c
    return r, e0, alo, gamma
def theta_ord(e, D, c=None, beta=0.5):
    if c is None: c = varpi(e)
    r, e0, alo, gamma = consts(e, c, beta)
    tail = 0.0 if D is None else r**D/(e*c)
    return (1-e)*(1.0/(e*c + gamma*(1-e)) + tail)
def M(w, alo, ahi): return (max(alo*w[1]+(1-alo)*w[0], ahi*w[1]+(1-ahi)*w[0]), w[0])
def B(y, d, r, e0, alo):
    w = (r*y[0], e0*y[1])
    for _ in range(d): w = M(w, alo, r)
    return w
def series(ds, e, c, beta, extra=200):
    r, e0, alo, gamma = consts(e, c, beta); y = (1.0, 1.0); tot = 1.0
    for d in ds:
        y = B(y, d, r, e0, alo); tot += max(y)
    return tot, y   # partial sum and last state
if __name__ == "__main__":
    print("Theta_ord_D with c = varpi(eps), beta = 1/2")
    for D in (8, 12, 16, 24, 32, None):
        lo, hi = 0.5, 0.75
        for _ in range(60):
            mid = (lo+hi)/2
            if theta_ord(mid, D) < 1: hi = mid
            else: lo = mid
        print("  D=%-4s equals 1 at %.4f   values at 0.735: %.4f  0.74: %.4f  0.745: %.4f" %
              ("inf" if D is None else D, hi, theta_ord(0.735, D), theta_ord(0.74, D), theta_ord(0.745, D)))
    for (D, e, stated) in ((16, 0.74, 0.981), (32, 0.737, 0.986)):
        val = theta_ord(e, D, c=varpi(e)-1e-4, beta=0.4999)
        print("claim D>=%d at eps=%.3f: %.6f (stated bound %.3f)" % (D, e, val, stated)); assert val <= stated
    # hypothesis of the sum lemma for eps <= 3/4 and every admissible pair (c < 1/2, beta + c < 1)
    rng = random.Random(1)
    for _ in range(200000):
        e = rng.uniform(0.01, 0.75); c = rng.uniform(1e-3, 0.5); beta = rng.uniform(0, 1-c)
        r, e0, alo, gamma = consts(e, c, beta); assert e*c <= r*(1-gamma)
    # the sum lemma against the exact operators: exhaustive short prefixes, then random long sequences
    worst = 0.0
    for (e, c, beta) in ((0.74, varpi(0.74), 0.5), (0.70, varpi(0.70), 0.5), (0.5, 0.3, 0.55), (0.3, 0.2, 0.7), (0.74, 0.35, 0.57), (0.1, 0.05, 0.9)):
        r, e0, alo, gamma = consts(e, c, beta); bound = 1.0/(e*c + gamma*(1-e))
        for L in range(1, 9):
            for ds in itertools.product((1, 2, 3, 4, 9), repeat=L):
                tot, y = series(ds, e, c, beta)
                # partial sums must stay below the bound
                assert tot <= bound*(1+1e-12), (e, c, beta, ds, tot, bound)
                worst = max(worst, tot/bound)
        for _ in range(3000):
            ds = [rng.choice((1, 1, 2, 2, 3, 5, 40)) for _ in range(300)]
            tot, y = series(ds, e, c, beta); assert tot <= bound*(1+1e-12); worst = max(worst, tot/bound)
        const2, _ = series([2]*400, e, c, beta)
        print("eps=%.2f c=%.3f beta=%.2f: bound 1/(1-s0) = %.4f, constant spacing 2 gives %.4f, spacing 1 gives %.4f"
              % (e, c, beta, bound, const2, series([1]*400, e, c, beta)[0]))
    print("largest partial sum / bound over all sequences tried: %.4f" % worst)
    # simulation: mean number of active gates k <= P that keep their pointer with no reset in (k,P]
    def base_even(n, H):            # even gates read H below, odd gates are NOT gates
        return [0]+[(i-H if (i % 2 == 0 and i > H) else i-1) for i in range(1, n+1)]
    def base_half(n):               # j_{2t} = t, j_{2t+1} = 2t
        return [0]+[(i//2 if i % 2 == 0 else i-1) for i in range(1, n+1)]
    def base_random(n, D, seed):    # pointers increase by 1, 2 or 3 per active gate, distance >= D
        rg = random.Random(seed); j = [0]*(n+1); last = 0
        for i in range(1, n+1):
            cand = last + rg.choice((1, 2, 3))
            if rg.random() < 0.7 and cand <= i-D: j[i] = cand; last = cand
            else: j[i] = i-1
        return j
    def measure(j, eps, seed):
        n = len(j)-1; rg = random.Random(seed); x = [1]*(n+1); keep = [False]*(n+1); reset = [False]*(n+1); ones = 1
        for i in range(1, n+1):
            if rg.random() < eps: t = rg.randrange(i)
            else: t = j[i]; keep[i] = (j[i] != i-1 and j[i] >= 1)
            x[i] = 1 - (x[i-1] | x[t]); reset[i] = (t != i-1 and x[t] == 1); ones += x[i]
        nxt = n+1; tot = 0
        for k in range(n, n//2, -1):
            if keep[k]: tot += nxt - k
            if reset[k]: nxt = k
        return tot/(n - n//2), ones/(n+1)
    n = 600000
    for name, j, D in (("even gates read 32 below", base_even(n, 32), 32), ("halving base", base_half(n), 32), ("random ordered, D=32", base_random(n, 32, 7), 32)):
        for e in (0.60, 0.74):
            mean, dens = measure(j, e, 11)
            print("%-26s eps=%.2f: density %.3f, measured count %.3f, Theta_ord_32 = %.3f, plain bound %.3f"
                  % (name, e, dens, mean, theta_ord(e, D), (1-e)/(e*varpi(e))))
            assert mean <= theta_ord(e, D)
