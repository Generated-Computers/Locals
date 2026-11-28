# Task 4 (a),(b): facts (b),(c) of the rewritten Lemma sumspacing, and the group version of the lemma:
#   y_1 = 1, y_{u+1} = rho_u r^{mu_u-1} B_{d_u} y_u   (the lemma's recursion)    and
#   z_1 = 1, z_{u+1} = M^{d_u}(phi^{mu_u} z_u)        (the exact recursion it dominates),
#   claim  sum_u S_{mu_u} ||y_u|| <= 1/(1-s_0).
import sys, math, random
from fractions import Fraction as F
sys.path.insert(0, sys.argv[1])
from ord_ops import M, phi, B, s_of, varpi

def rand_adm2(rnd, exact=True, epsmax=0.75, betamax=2/3):
    while True:
        if exact:
            eps = F(rnd.randint(1, 75), 100); c = F(rnd.randint(1, 49), 100); beta = F(rnd.randint(0, 66), 100)
        else:
            eps = rnd.uniform(0.01, epsmax); c = rnd.uniform(0.005, 0.4999); beta = rnd.uniform(0.0, betamax)
        if c + beta < 1: return eps, c, beta
def S(mu, e0): return sum(e0**a for a in range(mu))
def Bmu(d, mu, y, eps, c, beta):
    r = 1-eps*c; e0 = eps*(1-c)
    w = (r**mu*y[0], e0**mu*y[1])
    for _ in range(d): w = M(w, eps, c, beta)
    return w

if __name__ == "__main__":
    rnd = random.Random(31337)
    # ---- (a) facts (b) and (c), exact rationals
    nb = nc = 0
    for trial in range(3000):
        eps, c, beta = rand_adm2(rnd)
        r = 1-eps*c; e0 = eps*(1-c); g = eps*eps*beta*c; s0 = r - g*(1-eps)
        assert 1-s0 == eps*c*(1+eps*beta*(1-eps))
        rho1 = r/(r-eps*beta*(1-eps))
        assert (1-g)*(1-eps*beta) - e0*eps*beta >= 0
        for q in range(0, 11):
            ratio = 1 + (rho1-1)*F(q, 10); y = (F(1), ratio)          # y in the cone
            assert max(y) == y[1]
            Phi = max(y) + s_of(y, eps, c, beta)/(1-s0)
            assert Phi >= max(y)/c, ("fact (b)", eps, c, beta, q); nb += 1
        for mu in range(1, 40):
            assert c*(S(mu, e0) - r**(mu-1)) <= 1 - r**(mu-1), ("fact (c)", eps, c, beta, mu); nc += 1
            f1 = 1-(1-c)*r**mu - c*S(mu+1, e0); f0 = 1-(1-c)*r**(mu-1) - c*S(mu, e0)
            assert f1 - f0 == c*e0*(r**(mu-1) - e0**(mu-1))
    print("(a) fact (b): %d (params, cone point) cases ok; fact (c) and its difference identity: %d cases ok; identity 1-s_0 = eps c (1+eps beta (1-eps)) ok" % (nb, nc))
    # are the assumptions eps<=3/4, beta<=2/3 needed for (b)?
    bad = 0; ex = None
    for trial in range(200000):
        eps = rnd.uniform(0.01, 1.0); c = rnd.uniform(0.005, 0.4999); beta = rnd.uniform(0, 1-c)
        e0 = eps*(1-c); g = eps*eps*beta*c
        if (1-g)*(1-eps*beta) - e0*eps*beta < 0:
            bad += 1
            if eps <= 0.75 and (ex is None or beta < ex[2]): ex = (eps, c, beta)
    print("    without beta<=2/3 / eps<=3/4 the reduced inequality (1-gamma)(1-eps beta) >= e_0 eps beta fails in %d of 200000 draws with c<1/2, c+beta<1, eps<=1; smallest beta among failures with eps<=3/4: %s" % (bad, ex))

    # ---- (b) exhaustive, exact rationals: sequences of (d, mu), every partial sum, both recursions, random rho
    cnt = 0; worst = F(0); worst_exact = F(0)
    acts = [(d, mu) for d in (1, 2, 3, 6) for mu in (1, 2, 3, 7)]
    for trial in range(8):
        eps, c, beta = rand_adm2(rnd)
        if trial == 0: eps, c, beta = F(74, 100), F(33, 100), F(1, 2)
        r = 1-eps*c; e0 = eps*(1-c); g = eps*eps*beta*c; bound = 1/(eps*c + g*(1-eps))
        Sinf = 1/(1-e0)
        Lmax = 4
        rhos = [F(1)]*Lmax if trial % 2 == 0 else [F(rnd.choice((0, 2, 3, 4, 4)), 4) for _ in range(Lmax)]
        def dfs(y, z, tot, totz, depth):
            global cnt, worst, worst_exact
            # ending here with an arbitrarily large last group: S_mu <= 1/(1-e0)
            cnt += 1
            fin = tot + Sinf*max(y); finz = totz + Sinf*max(z)
            assert fin <= bound, ("VIOLATION (lemma recursion)", eps, c, beta, depth, float(fin), float(bound))
            assert finz <= fin
            if fin/bound > worst: worst = fin/bound
            if finz/bound > worst_exact: worst_exact = finz/bound
            if depth == Lmax: return
            for (d, mu) in acts:
                y2 = B(d, y, eps, c, beta); f = rhos[depth]*r**(mu-1); y2 = (f*y2[0], f*y2[1])
                z2 = Bmu(d, mu, z, eps, c, beta); z2 = (rhos[depth]*z2[0], rhos[depth]*z2[1])
                assert z2[0] <= y2[0] and z2[1] <= y2[1]
                Sm = S(mu, e0)
                dfs(y2, z2, tot + Sm*max(y), totz + Sm*max(z), depth+1)
        one = (F(1), F(1)); dfs(one, one, F(0), F(0), 0)
    print("(b) exhaustive exact: %d sequence prefixes (each closed with an infinite last group), no violation; max sum/bound %.5f (lemma recursion), %.5f (exact recursion)" % (cnt, float(worst), float(worst_exact)))

    # ---- (b) search: value iteration from above for the lemma recursion with actions (d, mu), plus terminal 'infinite group'
    def value_iteration(eps, c, beta, DS=(1, 2, 3, 4, 5, 8, 12), MUS=(1, 2, 3, 4, 6, 10, 20), n1=500, tol=1e-12, exact_rec=False):
        r = 1-eps*c; e0 = eps*(1-c)
        tmax = 1.0/(1-eps*beta) + 0.05
        if exact_rec: tmax = max(tmax, 1.2)
        G = int(math.ceil(tmax*n1)) + 1; tmax = (G-1)/n1
        grid = [i/n1 for i in range(G)]
        trans = []
        for th in grid:
            row = []; nrm = max(1.0, th)
            for mu in MUS:
                Sm = sum(e0**a for a in range(mu))
                for d in DS:
                    if exact_rec: z = Bmu(d, mu, (1.0, th), eps, c, beta)
                    else:
                        z = B(d, (1.0, th), eps, c, beta); f = r**(mu-1); z = (f*z[0], f*z[1])
                    t2 = z[1]/z[0]; assert t2 <= tmax + 1e-12, (t2, tmax)
                    pos = t2*n1; i = min(int(pos), G-2); row.append((Sm*nrm, z[0], i, pos-i, d, mu))
            trans.append((nrm/(1-e0), row))
        V = [max(1.0, th)/((1-e0)*(1-r)) for th in grid]
        it = 0
        while True:
            it += 1; newV = [0.0]*G; diff = 0.0
            for gi in range(G):
                term, row = trans[gi]; best = term
                for (cur, a, i, f, d, mu) in row:
                    v = cur + a*(V[i]*(1-f) + V[i+1]*f)
                    if v > best: best = v
                newV[gi] = best; dd = abs(best-V[gi])
                if dd > diff: diff = dd
            V = newV
            if diff < tol*V[n1] or it > 5000: break
        # greedy policy from y = 1
        y = (1.0, 1.0); scale = 1.0; pol = []
        for stepn in range(12):
            th = y[1]/y[0]; pos = th*n1; gi = min(int(round(pos)), G-1)
            term, row = trans[gi]; best = (term, None)
            for (cur, a, i, f, d, mu) in row:
                v = cur + a*(V[i]*(1-f) + V[i+1]*f)
                if v > best[0]: best = (v, (d, mu))
            if best[1] is None: pol.append("end"); break
            pol.append(best[1]); d, mu = best[1]
            if exact_rec: y = Bmu(d, mu, y, eps, c, beta)
            else:
                z = B(d, y, eps, c, beta); f = r**(mu-1); y = (f*z[0], f*z[1])
        return V[n1], pol
    print("(b) value iteration, upper bound on sup over sequences of (d,mu) [d<=12, mu<=20, or an infinite last group]:")
    sets = [(0.74, varpi(0.74), 0.5), (0.70, varpi(0.70), 0.5), (0.75, 1/3, 0.5), (0.74, 0.35, 0.57), (0.75, 0.33, 0.6666), (0.75, 0.49, 0.5), (0.3, 0.2, 0.6), (0.5, 0.05, 0.66), (0.75, 0.05, 0.6666), (0.2, 0.45, 0.1)]
    for _ in range(8):
        e, c, b = rand_adm2(rnd, exact=False)
        if e*c > 0.04: sets.append((e, c, b))
    worst = 0.0
    for (e, c, b) in sets:
        g = e*e*b*c; bound = 1/(e*c + g*(1-e))
        v, pol = value_iteration(e, c, b)
        vz, polz = value_iteration(e, c, b, exact_rec=True)
        worst = max(worst, v/bound)
        print("   eps=%.3f c=%.4f beta=%.4f: bound %.6f | sup<= %.6f (ratio %.5f), policy %s | exact recursion sup<= %.6f (ratio %.5f)" % (e, c, b, bound, v, v/bound, pol[:5], vz, vz/bound))
        assert v <= bound*(1+2e-5) and vz <= v*(1+1e-9), "VIOLATION?"
    print("   largest sup/bound: %.6f" % worst)
    # random long sequences, floats, many parameter sets
    nseq = 0; worst = 0.0
    for trial in range(1500):
        e, c, b = rand_adm2(rnd, exact=False); r = 1-e*c; e0 = e*(1-c); g = e*e*b*c; bound = 1/(e*c + g*(1-e))
        L = min(1500, int(30/(e*c))+10)
        for rep in range(4):
            y = (1.0, 1.0); tot = 0.0
            style = rnd.choice(("blocks", "mixed", "two", "ones"))
            for u in range(L):
                if style == "blocks": d, mu = rnd.choice((1, 2)), rnd.choice((1, 2, 5, 30))
                elif style == "mixed": d, mu = rnd.randint(1, 6), rnd.randint(1, 4)
                elif style == "two": d, mu = 2, rnd.choice((1, 1, 1, 2))
                else: d, mu = 1, rnd.choice((1, 2, 3))
                tot += sum(e0**a for a in range(mu))*max(y)
                z = B(d, y, e, c, b); f = r**(mu-1); y = (f*z[0], f*z[1])
            tot += max(y)/(1-e0); nseq += 1; worst = max(worst, tot/bound)
            assert tot <= bound*(1+1e-9), ("VIOLATION", e, c, b, style)
    print("   %d random long sequences over random admissible parameters: no violation, max sum/bound %.5f" % (nseq, worst))
