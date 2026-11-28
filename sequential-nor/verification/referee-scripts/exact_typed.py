# Exact check of Lemma typed in its conditional form:
#   P[E*_sigma  and  G_g for P_l < g <= P_0 | trace prefix up to P_l] <= product bound,
# for every typed skeleton with P_0 <= n, length <= lmax, and every reachable prefix. Rational arithmetic.
import sys
from fractions import Fraction as F

def base(H, i): return max(i-H, 0)

def Tm(H, m, eps, c, beta):
    r = 1-eps*c; e0 = eps*(1-c)
    q = min(m, H-1)
    t0 = t1 = r**(m-q)
    for i in range(q-1, -1, -1):
        n1 = r*t0
        n0 = max(a*e0*t1 + (1-a)*r*t0 for a in (eps*beta, r))
        t0, t1 = n0, n1
    return t0, t1

def prefixes(P):
    out = [(1,)]
    for g in range(1, P+1):
        nxt = []
        for x in out:
            nxt.append(x+(0,))
            if x[-1] == 0 and g >= 2: nxt.append(x+(1,))
        out = nxt
    return out

def step(x, ones, g, H, eps, cons):
    # x: prefix of length g, ones = number of ones in x. returns list of (x_g, prob)
    o2 = ones - x[g-1]                      # ones among positions <= g-2
    out = [0, 0]
    jb = base(H, g)
    rs_b = (jb != g-1) and x[jb] == 1
    vb = 0 if rs_b else 1-x[g-1]
    if cons is None or cons == 'pi0' or (cons == 'noreset' and not rs_b):
        out[vb] += 1-eps
    if cons is None:
        out[0] += eps*F(o2, g); out[1-x[g-1]] += eps*F(g-o2, g)
    elif cons == 'noreset':
        out[1-x[g-1]] += eps*F(g-o2, g)
    elif isinstance(cons, tuple):
        p = cons[1]
        rs = (p != g-1) and x[p] == 1
        out[0 if rs else 1-x[g-1]] += eps/g
    return out

def prob(x0, gates, H, eps, c, g0):
    cur = {x0: F(1)}
    for g, cons in gates:
        nxt = {}
        for x, p in cur.items():
            ones = sum(x)
            if g > g0 and not (F(ones-1, g) >= c): continue
            o = step(x, ones, g, H, eps, cons)
            for v in (0, 1):
                if o[v]:
                    y = x+(v,); nxt[y] = nxt.get(y, 0) + p*o[v]
        cur = nxt
    return sum(cur.values())

def skeletons(P0, l, g0, H):
    yield []
    if l == 0: return
    for k in range(g0+2, P0+1):
        opts = [(q, 'res') for q in range(g0, k-1)]
        if k-H >= g0: opts.append((k-H, 'base'))
        for q, ty in opts:
            for rest in skeletons(q, l-1, g0, H):
                yield [(k, q, ty)] + rest

def run(H, eps, c, beta, g0, n, lmax):
    r = 1-eps*c; worst = F(0); cnt = 0; wsk = None
    for P0 in range(g0, n+1):
        for steps in skeletons(P0, lmax, g0, H):
            if not steps: continue
            cons = {}; bound = F(1); Pprev = P0
            for (k, q, ty) in steps:
                cons[k] = 'pi0' if ty == 'base' else ('res', q)
                for g in range(k+1, Pprev+1): cons[g] = 'noreset'
                m = Pprev-k
                bound *= (1-eps)*max(Tm(H, m, eps, c, beta)) if ty == 'base' else r**m*eps/k
                Pprev = q
            Pl = steps[-1][1]
            gates = [(g, cons.get(g)) for g in range(Pl+1, P0+1)]
            for x in prefixes(Pl):
                pr = prob(x, gates, H, eps, c, g0)
                cnt += 1
                assert pr <= bound, ("typed FAILS", H, eps, c, beta, P0, steps, x, float(pr), float(bound))
                if len(steps) >= 2 and pr/bound > worst: worst = pr/bound; wsk = (P0, steps, x)
    return worst, cnt, wsk

if __name__ == "__main__":
    n = int(sys.argv[1]); lmax = int(sys.argv[2])
    for H in (2, 3, 4):
        for eps in (F(1,2), F(74,100)):
            for (g0, beta) in ((2, F(1,4)), (3, F(1,3))):
                for c in (F(0), F(1,10), F(1,4)):
                    w, cnt, wsk = run(H, eps, c, beta, g0, n, lmax)
                    print("typed ok H=%d eps=%s g0=%d beta=%s c=%s cases=%d max ratio (l>=2)=%.4f at %s" % (H, eps, g0, beta, c, cnt, float(w), wsk), flush=True)
