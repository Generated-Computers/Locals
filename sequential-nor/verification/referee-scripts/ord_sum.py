# (b) Lemma sumspacing: search for a violating distance sequence.
#  1. value iteration from above for V(y) = sup over sequences d_u in 1..DMAX of sum_u ||y_u||  (rho_u = 1 is the worst
#     case by monotonicity; V is positively homogeneous and convex, so linear interpolation in the ratio keeps an upper bound);
#  2. exact rational exhaustive search over short sequences + random long sequences with random factors rho_u (incl. 0);
#  3. hill climbing on long sequences.
import sys, math, random, itertools
from fractions import Fraction as F
sys.path.insert(0, sys.argv[1])
from ord_ops import M, phi, B, s_of, varpi, rand_adm

def norm(y): return max(y)

def value_iteration(eps, c, beta, DMAX=12, G=1500, tol=1e-13):
    r = 1-eps*c
    tmax = 1.0/(1-eps*beta) + 0.05           # ratios y(1)/y(0) after any B_d stay below 1/(1-eps*beta)
    n1 = max(2, int((G-1)/tmax)); G = int(math.ceil(tmax*n1)) + 1; tmax = (G-1)/n1   # theta = 1 is a grid node (kink of the norm)
    grid = [i/n1 for i in range(G)]
    trans = []                                # for each grid point: list of (a_d, index position) for y=(1,theta)
    for th in grid:
        row = []
        w = phi((1.0, th), eps, c)
        for d in range(1, DMAX+1):
            w = M(w, eps, c, beta)
            a = w[0]; t2 = w[1]/w[0]
            assert t2 <= tmax + 1e-12
            pos = t2/tmax*(G-1); i = min(int(pos), G-2); row.append((a, i, pos-i))
        trans.append(row)
    V = [max(1.0, th)/(1-r) for th in grid]   # upper bound: each B_d contracts the norm by r
    it = 0
    while True:
        it += 1
        newV = [0.0]*G; diff = 0.0
        for gi in range(G):
            best = 0.0
            for (a, i, f) in trans[gi]:
                v = a*(V[i]*(1-f) + V[i+1]*f)
                if v > best: best = v
            nv = max(1.0, grid[gi]) + best
            newV[gi] = nv
            dd = abs(nv - V[gi])
            if dd > diff: diff = dd
        V = newV
        if diff < tol*V[0] or it > 6000: break
    # value at theta = 1 and greedy policy rolled out exactly
    def Vat(th):
        pos = th/tmax*(G-1); i = min(int(pos), G-2); f = pos-i
        return V[i]*(1-f) + V[i+1]*f
    y = (1.0, 1.0); tot = 1.0; pol = []
    for u in range(400):
        best = None
        for d in range(1, DMAX+1):
            z = B(d, y, eps, c, beta); val = z[0]*Vat(z[1]/z[0])
            if best is None or val > best[0]: best = (val, d, z)
        y = best[2]; tot += norm(y); pol.append(best[1])
    return Vat(1.0), tot, pol[:14], it

if __name__ == "__main__":
    rnd = random.Random(99)
    print("1. value iteration (upper bound on the sup over all sequences with d in 1..12) vs 1/(1-s_0):")
    sets = [(0.74, varpi(0.74), 0.5), (0.70, varpi(0.70), 0.5), (0.65, varpi(0.65), 0.5), (0.60, varpi(0.60), 0.5), (0.75, 1/3, 0.5),
            (0.74, 0.35, 0.57), (0.5, 0.3, 0.55), (0.3, 0.2, 0.7), (0.75, 0.49, 0.5), (0.75, 0.2, 0.79), (0.2, 0.45, 0.1), (0.6, 0.1, 0.0), (0.75, 0.4, 0.59)]
    for _ in range(12):
        e, c, b = rand_adm(rnd, exact=False)
        if e*c > 0.04: sets.append((e, c, b))
    worst = 0.0
    for (e, c, b) in sets:
        r = 1-e*c; g = e*e*b*c; bound = 1/(e*c + g*(1-e))
        v, roll, pol, it = value_iteration(e, c, b, G=1200 if len(sets) > 13 else 1500)
        y = (1.0, 1.0); t2 = 1.0
        for u in range(3000):
            y = B(2, y, e, c, b); t2 += norm(y)
        worst = max(worst, v/bound)
        print("  eps=%.3f c=%.4f beta=%.3f: bound %.6f | sup<= %.6f (ratio %.5f) | greedy rollout %.6f, policy %s | all-2: %.6f (%.2f%% below bound)" % (e, c, b, bound, v, v/bound, roll, "".join(str(d) if d < 10 else "+" for d in pol), t2, 100*(1-t2/bound)))
        if v > bound*(1+1e-7):
            # the interpolated value is only an upper bound; refine the grid and watch it come down
            vs = [value_iteration(e, c, b, G=GG)[0] for GG in (4000, 12000)]
            print("     grid refinement: sup<= %.7f (G=4000), %.7f (G=12000); bound %.7f; rollout %.7f" % (vs[0], vs[1], bound, roll))
            assert vs[1] <= bound*(1+2e-5) and vs[1] <= vs[0] <= v, "VIOLATION?"
    print("  largest sup/bound: %.6f" % worst)

    print("2. exact rationals: exhaustive sequences over {1,2,3,4,7}^L, L<=7, every partial sum, random admissible params, random rho")
    cnt = 0; worst = F(0)
    for trial in range(25):
        e, c, b = rand_adm(rnd)
        r = 1-e*c; g = e*e*b*c; bound = 1/(e*c + g*(1-e))
        rhos = [F(1)]*7 if trial % 2 == 0 else [F(rnd.choice((0, 1, 2, 3, 4, 4, 4)), 4) for _ in range(7)]
        def dfs(y, tot, depth):
            global cnt, worst
            cnt += 1
            assert tot <= bound, ("VIOLATION", e, c, b, depth, float(tot), float(bound))
            if tot/bound > worst: worst = tot/bound
            if depth == 7: return
            for d in (1, 2, 3, 4, 7):
                z = B(d, y, e, c, b); z = (rhos[depth]*z[0], rhos[depth]*z[1])
                dfs(z, tot + norm(z), depth+1)
        dfs((F(1), F(1)), F(1), 0)
    print("   %d partial sums checked exactly, max partial/bound %.5f" % (cnt, float(worst)))

    print("3. random long sequences and hill climbing (floats), random admissible params with eps<=3/4:")
    worst = 0.0; nseq = 0
    for trial in range(400):
        e, c, b = rand_adm(rnd, exact=False)
        r = 1-e*c; g = e*e*b*c; bound = 1/(e*c + g*(1-e))
        L = min(4000, int(40/(e*c)) + 20)
        def total(ds, rho=None):
            y = (1.0, 1.0); tot = 1.0
            for u, d in enumerate(ds):
                y = B(d, y, e, c, b)
                if rho is not None: y = (rho[u]*y[0], rho[u]*y[1])
                tot += norm(y)
            return tot
        cands = [[2]*L, [1]*L, [3]*L, [1, 2]*(L//2), [2, 1, 1]*(L//3), [2, 2, 1]*(L//3), [2, 3]*(L//2), [1]*5+[2]*(L-5), [2]*5+[1]*(L-5)]
        for ds in cands:
            t = total(ds); nseq += 1; worst = max(worst, t/bound); assert t <= bound*(1+1e-9), ("VIOLATION", e, c, b, ds[:10])
        if trial < 60:
            Ls = min(L, 60)
            ds = [rnd.choice((1, 2, 2, 3, 4, 12)) for _ in range(Ls)]; cur = total(ds + [2]*200)
            for step in range(300):
                i = rnd.randrange(Ls); old = ds[i]; ds[i] = rnd.randint(1, 12)
                t = total(ds + [2]*200)
                if t >= cur: cur = t
                else: ds[i] = old
            nseq += 300; worst = max(worst, cur/bound); assert cur <= bound*(1+1e-9), ("VIOLATION", e, c, b, ds)
            rho = [rnd.choice((0.0, 0.3, 0.9, 1.0, 1.0, 1.0)) if rnd.random() < 0.2 else 1.0 for _ in range(Ls+200)]
            assert total(ds + [2]*200, rho) <= cur*(1+1e-12)
    print("   %d sequences, no violation, max sum/bound %.5f" % (nseq, worst))
