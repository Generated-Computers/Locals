# Task 4 (c): exact rational check of the rewritten Lemma ordstep (ARBITRARY base, event C° with pi_g = 1 on window gates
# reading P), of the typed product bound with E°_sigma for length-2 skeletons, and of Lemma ordsum on ordered bases with ties.
import sys, random, math
from fractions import Fraction as F
sys.path.insert(0, sys.argv[1])
from ord_ops import M, phi, B, theta_ord
from grp_sum import rand_adm2

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
def ordered_D(j, N):
    n = len(j)-1; Dm = None; act = [i for i in range(1, n+1) if active(j, i)]
    for g in act:
        if g <= N: continue
        if any(j[k] > j[g] for k in act if k < g): return None
        Dm = g-j[g] if Dm is None else min(Dm, g-j[g])
    return Dm
def inG(x, g, c, beta):
    ones = sum(x); return F(ones-1, g) >= c and F(g-ones, g) >= beta
def step(x, g, j, eps, cons):
    # cons: None | 'pi0' | 'noreset' | 'res_noreset' (pi=1 and not a reset) | ('res', q)
    ones = sum(x); o2 = ones - x[g-1]; out = [0, 0]
    jb = j[g]; rs_b = (jb != g-1) and x[jb] == 1
    if cons is None or cons == 'pi0' or (cons == 'noreset' and not rs_b):
        out[0 if rs_b else 1-x[g-1]] += 1-eps
    if cons is None:
        out[0] += eps*F(o2, g); out[1-x[g-1]] += eps*F(g-o2, g)
    elif cons in ('noreset', 'res_noreset'):
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
def ordstep_bound(j, k, m, eps, c, beta):
    """(1-eps) e0^a r^{m-a-u} B_{d_1}..B_{d_u} 1 (2-vector), a, u, ds  -- as in the rewritten lemma (no D)"""
    r = 1-eps*c; e0 = eps*(1-c); P = j[k]
    win = range(k+1, k+m+1)
    a = sum(1 for g in win if j[g] == P)
    ts = sorted({j[g] for g in win if active(j, g) and P < j[g] < k})
    t = [P] + ts; ds = [t[i]-t[i-1] for i in range(1, len(t))]
    y = (F(1), F(1)) if isinstance(eps, F) else (1.0, 1.0)
    for d in reversed(ds): y = B(d, y, eps, c, beta)
    assert m-a-len(ts) >= 0
    f = (1-eps)*e0**a*r**(m-a-len(ts))
    return (f*y[0], f*y[1]), a, len(ts), ds
def cgates(j, k, m, top=True):
    P = j[k]
    out = [(g, None) for g in range(P+1, k)] + [(k, 'pi0')]
    for g in range(k+1, k+m+1):
        assert not (j[g] == P and j[g] == g-1)             # a window gate reading P is never a NOT gate
        out.append((g, 'res_noreset' if (top and j[g] == P) else 'noreset'))
    return out

def bases(n, rnd):
    out = []
    out.append(("floor(i/2)", [0]+[i//2 for i in range(1, n+1)], 4))
    j = [0]+[i-1 for i in range(1, n+1)]
    for (lo, hi, p) in ((6, 8, 2), (9, 12, 5), (13, n, 9)):
        for g in range(lo, min(hi, n)+1): j[g] = p
    out.append(("blocks: gates 6-8 read 2, 9-12 read 5, 13.. read 9", j, 0))
    j = [0]+[i-1 for i in range(1, n+1)]
    for g in range(5, n+1): j[g] = 3 if g % 2 == 1 else g-1
    out.append(("odd gates >=5 all read 3, NOT gates between", j, 0))
    for t in range(4):       # nondecreasing with ties, some zero-pointer gates
        j = [0]*(n+1); last = 1
        for i in range(1, n+1):
            u = rnd.random(); cand = last + rnd.choice((0, 0, 1, 2))
            if u < 0.65 and cand <= i-2 and i >= 3: j[i] = cand; last = cand
            elif u < 0.75 and i >= 2: j[i] = 0
            else: j[i] = i-1
        out.append(("random nondecreasing with ties #%d" % t, j, 0))
    for t in range(5):       # arbitrary pointers
        j = [0]+[rnd.randrange(i) for i in range(1, n+1)]
        out.append(("arbitrary random pointers #%d" % t, j, None))
    out.append(("decreasing pointers", [0, 0]+[max(n-2*i, 0) if n-2*i <= i-2 else i-1 for i in range(2, n+1)], None))
    return out

if __name__ == "__main__":
    rnd = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 808); n = int(sys.argv[2]); tot = 0
    for name, j, N in bases(n, rnd):
        acts = [i for i in range(1, n+1) if active(j, i)]
        worst = F(0); cases = 0; pos = 0; maxa = 0; maxu = 0; dup = 0; fail_plain = 0
        for trial in range(3):
            eps, c, beta = rand_adm2(rnd)
            if trial == 0: eps, c, beta = F(74, 100), F(1, 4), F(2, 5)
            g0 = (N or 0) + rnd.choice((1, 1, 2))
            for k in acts:
                P = j[k]
                if P < g0: continue
                for m in range(0, n-k+1):
                    bnd, a, u, ds = ordstep_bound(j, k, m, eps, c, beta); maxa = max(maxa, a); maxu = max(maxu, u)
                    gates = cgates(j, k, m)
                    for x in prefixes(P):
                        pr = prob(x, gates, j, eps, c, beta, g0); cases += 1
                        assert pr <= bnd[x[-1]], ("ordstep FAILS", name, eps, c, beta, g0, k, m, x, float(pr), float(bnd[x[-1]]))
                        if pr > 0 and m >= 1: pos += 1; worst = max(worst, pr/bnd[x[-1]])
                        if a > 0:
                            # without the requirement pi_g = 1 the bound with e0^a is not claimed; count how often it would fail
                            pr2 = prob(x, cgates(j, k, m, top=False), j, eps, c, beta, g0)
                            if pr2 > bnd[x[-1]]: fail_plain += 1
        tot += cases
        print("ordstep ok: %-52s cases %6d (positive, m>=1: %5d) max a=%d max u=%d  max prob/bound %.4f | plain event C (no pi_g=1) exceeds the bound in %d cases" % (name, cases, pos, maxa, maxu, float(worst), fail_plain), flush=True)
    print("total exact ordstep cases:", tot)
    if len(sys.argv) > 3: sys.exit(0)

    # typed product bound with E°_sigma, skeletons of length 2 (conditional on the prefix up to P_2)
    for name, j, N in bases(12, random.Random(9))[:5] + bases(12, random.Random(10))[7:9]:
        nn = 12; eps, c, beta = F(7, 10), F(1, 5), F(2, 5); g0 = max(N or 0, 2); r = 1-eps*c; cnt = 0
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
                        for g in range(k+1, Pprev+1):
                            cons[g] = 'res_noreset' if (ty == 'base' and j[g] == q) else 'noreset'
                        m = Pprev-k
                        bound *= max(ordstep_bound(j, k, m, eps, c, beta)[0]) if ty == 'base' else r**m*eps/k
                        Pprev = q
                    gates = [(g, cons.get(g)) for g in range(s2[1]+1, P0+1)]
                    for x in prefixes(s2[1]):
                        pr = prob(x, gates, j, eps, c, beta, g0); cnt += 1
                        assert pr <= bound, ("typed/E° FAILS", name, P0, s1, s2, x)
        print("typed product for E°_sigma, length-2 skeletons ok: %-52s cases %d" % (name, cnt), flush=True)

    # Lemma ordsum on random (D,0)-ordered bases with ties, floats
    rnd = random.Random(6); worst = 0.0; nchk = 0
    for trial in range(400):
        nn = 300; Dd = rnd.choice((2, 3, 5, 8, 12)); j = [0]*(nn+1); last = 1
        pa = rnd.choice((0.3, 0.7, 1.0)); steps = rnd.choice(((0, 1), (0, 0, 0, 1), (0, 1, 2), (0, 0, 2, 5), (1,), (0, 0, 0, 0, 0, 0, 3)))
        for i in range(1, nn+1):
            cand = last + rnd.choice(steps)
            if rnd.random() < pa and cand <= i-Dd and i >= 2: j[i] = cand; last = cand
            elif rnd.random() < 0.1 and i >= 2: j[i] = 0
            else: j[i] = i-1
        Dm = ordered_D(j, 0); assert Dm is None or Dm >= Dd
        if Dm is None: continue
        eps, c, beta = rand_adm2(rnd, exact=False); g0 = rnd.choice((1, 3, 10)); th = theta_ord(Dd, eps, c, beta)
        for P0 in rnd.sample(range(g0+2, nn+1), 20):
            ssum = 0.0
            for k in range(1, P0+1):
                if active(j, k) and j[k] >= g0: ssum += max(ordstep_bound(j, k, P0-k, eps, c, beta)[0])
            nchk += 1; worst = max(worst, ssum/th)
            assert ssum <= th*(1+1e-12), ("ordsum FAILS", Dd, eps, c, beta, P0)
    print("Lemma ordsum on random ordered bases with ties: %d windows, max sum/Theta^ord_D = %.4f" % (nchk, worst))
