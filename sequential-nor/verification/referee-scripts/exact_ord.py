# (d) exact rational check of Lemma ordstep (and of the typed product bound with it, length-2 skeletons) on small
#     (D,N)-ordered bases, by DP over trace prefixes; plus Lemma ordsum (deterministic) on random ordered bases.
import sys, random, math
from fractions import Fraction as F
sys.path.insert(0, sys.argv[1])
from ord_ops import M, phi, B, rand_adm, theta_ord

def prefixes(P):
    out = [(1,)]
    for g in range(1, P+1):
        nxt = []
        for x in out:
            nxt.append(x+(0,))
            if x[-1] == 0 and g >= 2: nxt.append(x+(1,))
        out = nxt
    return out
def active(j, i): return j[i] != i-1 and j[i] >= 1
def ordered_params(j, N):
    """largest D such that j is (D,N)-ordered, or None"""
    n = len(j)-1; Dm = None; act = [i for i in range(1, n+1) if active(j, i)]
    for g in act:
        if g <= N: continue
        if any(j[k] >= j[g] for k in act if k < g): return None
        Dm = g-j[g] if Dm is None else min(Dm, g-j[g])
    return Dm
def inG(x, g, c, beta):
    ones = sum(x); return F(ones-1, g) >= c and F(g-ones, g) >= beta
def step(x, g, j, eps, cons):
    ones = sum(x); o2 = ones - x[g-1]; out = [0, 0]
    jb = j[g]; rs_b = (jb != g-1) and x[jb] == 1
    if cons is None or cons == 'pi0' or (cons == 'noreset' and not rs_b):
        out[0 if rs_b else 1-x[g-1]] += 1-eps
    if cons is None:
        out[0] += eps*F(o2, g); out[1-x[g-1]] += eps*F(g-o2, g)
    elif cons == 'noreset':
        out[1-x[g-1]] += eps*F(g-o2, g)
    elif isinstance(cons, tuple):
        p = cons[1]; rs = (p != g-1) and x[p] == 1
        out[0 if rs else 1-x[g-1]] += eps/g
    return out
def prob(x0, gates, j, eps, c, beta, g0):
    cur = {x0: F(1)}
    for g, cons in gates:
        nxt = {}
        for x, p in cur.items():
            if g > g0 and not inG(x, g, c, beta): continue
            o = step(x, g, j, eps, cons)
            for v in (0, 1):
                if o[v]: nxt[x+(v,)] = nxt.get(x+(v,), 0) + p*o[v]
        cur = nxt
    return sum(cur.values())
def ordstep_bound(j, D, k, m, eps, c, beta):
    """(1-eps) r^{m-u} B_{d_1}...B_{d_u} 1  as a 2-vector, and the d_i"""
    r = 1-eps*c
    gs = [g for g in range(k+1, k+min(m, D-1)+1) if active(j, g)]
    t = [j[k]] + [j[g] for g in gs]; ds = [t[i]-t[i-1] for i in range(1, len(t))]
    assert all(d >= 1 for d in ds) and (not gs or t[-1] < k)
    y = (F(1), F(1)) if isinstance(eps, F) else (1.0, 1.0)
    for d in reversed(ds): y = B(d, y, eps, c, beta)
    f = (1-eps)*r**(m-len(gs))
    return (f*y[0], f*y[1]), ds

def bases(n, rnd):
    out = []
    out.append(("even gates read 3 below, odd gates NOT", [0]+[(i-3 if (i % 2 == 0 and i > 3) else i-1) for i in range(1, n+1)], 3))
    out.append(("j_2t=t, j_2t+1=2t", [0]+[(i//2 if i % 2 == 0 else i-1) for i in range(1, n+1)], 4))
    out.append(("j_2t=t, j_2t+1=2t", [0]+[(i//2 if i % 2 == 0 else i-1) for i in range(1, n+1)], 6))
    j = [0]+[i-1 for i in range(1, n+1)]
    for q, g in enumerate(range(8, min(14, n+1))): j[g] = 1+2*q
    out.append(("pointers increasing by 2 (gates 8..13 read 1,3,..,11)", j, 0))
    out.append(("j^3", [0]+[max(i-3, 0) for i in range(1, n+1)], 3))
    for t in range(6):
        Dd = rnd.choice((2, 2, 3, 4)); j = [0]*(n+1); last = 0
        for i in range(1, n+1):
            u = rnd.random(); cand = last + rnd.choice((1, 1, 2, 3))
            if u < 0.6 and cand <= i-Dd and i >= 2: j[i] = cand; last = cand
            elif u < 0.75 and i >= 2: j[i] = 0            # non-NOT gate reading x_0: not active
            else: j[i] = i-1
        out.append(("random ordered #%d" % t, j, 0))
    return out

if __name__ == "__main__":
    rnd = random.Random(4242); n = int(sys.argv[2])
    tot = 0
    for name, j, N in bases(n, rnd):
        Dmax = ordered_params(j, N); assert Dmax is not None and Dmax >= 2, (name, Dmax)
        acts = [i for i in range(1, n+1) if active(j, i)]
        worst = F(0); cases = 0; pos = 0; shapes = set()
        for trial in range(4):
            eps, c, beta = rand_adm(rnd)
            if trial == 0: eps, c, beta = F(74, 100), F(1, 4), F(2, 5)
            g0 = max(N, 1) + rnd.choice((0, 0, 1))
            for Dd in sorted({Dmax, 2}):
                for k in acts:
                    P = j[k]
                    if P < g0: continue
                    for m in range(0, n-k+1):
                        bnd, ds = ordstep_bound(j, Dd, k, m, eps, c, beta); shapes.add(tuple(ds))
                        gates = [(g, None) for g in range(P+1, k)] + [(k, 'pi0')] + [(g, 'noreset') for g in range(k+1, k+m+1)]
                        for x in prefixes(P):
                            pr = prob(x, gates, j, eps, c, beta, g0); cases += 1
                            assert pr <= bnd[x[-1]], ("ordstep FAILS", name, eps, c, beta, g0, Dd, k, m, x, float(pr), float(bnd[x[-1]]))
                            if pr > 0 and m >= 1: pos += 1; worst = max(worst, pr/bnd[x[-1]])
        tot += cases
        print("ordstep ok: %-55s N=%d D=%d active gates %d, distance patterns %d, cases %d (positive, m>=1: %d), max prob/bound %.4f" % (name, N, Dmax, len(acts), len(shapes), cases, pos, float(worst)), flush=True)
    print("total exact ordstep cases:", tot)

    # typed product bound with ordstep, skeletons of length 2, conditional on the prefix up to P_2
    for name, j, N in bases(12, random.Random(7))[:4] + bases(12, random.Random(8))[5:8]:
        Dmax = ordered_params(j, N); nn = 12
        if Dmax is None or Dmax < 2: continue
        eps, c, beta = F(7, 10), F(1, 5), F(2, 5); g0 = max(N, 2); r = 1-eps*c; cnt = 0
        def steps_from(Pprev):
            for k in range(g0+2, Pprev+1):
                for q in range(g0, k-1): yield (k, q, 'res')
                if active(j, k) and j[k] >= g0: yield (k, j[k], 'base')
        for P0 in range(g0, nn+1):
            for s1 in steps_from(P0):
                for s2 in steps_from(s1[1]):
                    cons = {}; bound = F(1); Pprev = P0
                    for (k, q, ty) in (s1, s2):
                        cons[k] = 'pi0' if ty == 'base' else ('res', q)
                        for g in range(k+1, Pprev+1): cons[g] = 'noreset'
                        m = Pprev-k
                        bound *= max(ordstep_bound(j, Dmax, k, m, eps, c, beta)[0]) if ty == 'base' else r**m*eps/k
                        Pprev = q
                    gates = [(g, cons.get(g)) for g in range(s2[1]+1, P0+1)]
                    for x in prefixes(s2[1]):
                        pr = prob(x, gates, j, eps, c, beta, g0); cnt += 1
                        assert pr <= bound, ("typed/ord FAILS", name, P0, s1, s2, x)
        print("typed product with ordstep, length-2 skeletons ok: %-50s cases %d" % (name, cnt), flush=True)

    # Lemma ordsum (deterministic): sum_k w_{k,P0-k} <= Theta^ord_D on random ordered bases, floats
    rnd = random.Random(5); worst = 0.0; nchk = 0
    for trial in range(300):
        nn = 400; Dd = rnd.choice((2, 3, 5, 8, 12, 20)); j = [0]*(nn+1); last = 0
        pa = rnd.choice((0.2, 0.5, 0.8, 1.0)); steps = rnd.choice(((1,), (1, 2), (1, 2, 3, 6), (2,), (1, 1, 1, 5)))
        for i in range(1, nn+1):
            cand = last + rnd.choice(steps)
            if rnd.random() < pa and cand <= i-Dd and i >= 2: j[i] = cand; last = cand
            elif rnd.random() < 0.1 and i >= 2: j[i] = 0
            else: j[i] = i-1
        assert ordered_params(j, 0) is None or ordered_params(j, 0) >= Dd
        if ordered_params(j, 0) is None: continue
        eps, c, beta = rand_adm(rnd, exact=False); g0 = rnd.choice((1, 3, 10)); th = theta_ord(Dd, eps, c, beta)
        for P0 in rnd.sample(range(g0+2, nn+1), 25):
            ssum = 0.0
            for k in range(1, P0+1):
                if active(j, k) and j[k] >= g0:
                    ssum += max(ordstep_bound(j, Dd, k, P0-k, eps, c, beta)[0])
            nchk += 1; worst = max(worst, ssum/th)
            assert ssum <= th*(1+1e-12), ("ordsum FAILS", Dd, eps, c, beta, P0)
    print("Lemma ordsum on random ordered bases: %d windows, max sum/Theta^ord_D = %.4f" % (nchk, worst))
