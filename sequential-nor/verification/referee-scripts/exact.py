# Exact check (rational arithmetic) of Lemma onegate, Lemma onestep and Lemma typed on small instances,
# by dynamic programming over trace prefixes. Everything constrained (resets, G_g) is a function of
# the trace prefix and the current gate's noise, so the DP is exact.
import itertools, sys
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
    # all reachable traces x_0..x_P : x_0=1, x_1=0, no adjacent ones
    out = [(1,)]
    for g in range(1, P+1):
        nxt = []
        for x in out:
            nxt.append(x+(0,))
            if x[-1] == 0 and g >= 2: nxt.append(x+(1,))
        out = nxt
    # note: x_g=0 after x_{g-1}=0 needs a reset, i.e. a one at <= g-2: x_0=1 works for g>=2
    return out

def step(x, g, H, eps, cons):
    """distribution of x_g given prefix x (length g), restricted to the constraint cons on gate g.
    cons: None | 'pi0' | ('res', q) | 'noreset'.  returns dict x_g -> prob"""
    out = {}
    def add(v, p):
        out[v] = out.get(v, 0) + p
    # pi = 0
    jb = base(H, g)
    def outcome(p):
        isnot = (p == g-1)
        reset = (not isnot) and x[p] == 1
        return (0 if reset else 1-x[g-1]), reset
    v, rs = outcome(jb)
    if cons is None or cons == 'pi0' or (cons == 'noreset' and not rs):
        add(v, 1-eps)
    if cons != 'pi0':
        for p in range(g):
            v, rs = outcome(p)
            if cons is None or (cons == 'noreset' and not rs) or (isinstance(cons, tuple) and cons[1] == p):
                add(v, eps/g)
    return out

def prob(x0, gates, H, eps, c, g0):
    """P[ all constraints and all G_g (g in gates, g>g0) | prefix x0 ];  gates: list of (g, cons) consecutive from len(x0)"""
    cur = {x0: F(1)}
    for g, cons in gates:
        nxt = {}
        for x, p in cur.items():
            assert len(x) == g
            if g > g0 and not (F(sum(x)-1, g) >= c):   # G_g = {(O_{g-1}-1)/g >= c}
                continue
            for v, pv in step(x, g, H, eps, cons).items():
                y = x+(v,)
                nxt[y] = nxt.get(y, 0) + p*pv
        cur = nxt
    return sum(cur.values())

def check_onegate(H, eps, c, beta, g0, gmax):
    r = 1-eps*c; e0 = eps*(1-c); worst = []
    for g in range(g0+1, gmax+1):
        for x in prefixes(g-1):
            if not (F(sum(x)-1, g) >= c): continue
            # (i)
            jb = base(H, g)
            p_res_notreset = sum(eps/g for p in range(g) if not (p != g-1 and x[p] == 1))
            assert p_res_notreset <= e0, ("onegate(i)", H, eps, c, g, x)
            d = step(x, g, H, eps, None)
            p1 = d.get(1, 0)
            if x[-1] == 1: assert p1 == 0
            else:
                assert eps*beta <= p1 <= r, ("onegate(ii)", H, eps, c, beta, g, x, p1)
    return True

def check_onestep(H, eps, c, beta, g0, Pmax, mmax):
    worst = F(0)
    for P in range(g0, Pmax+1):
        k = P+H
        for m in range(0, mmax+1):
            t0, t1 = Tm(H, m, eps, c, beta)
            gates = [(g, None) for g in range(P+1, k)] + [(k, 'pi0')] + [(g, 'noreset') for g in range(k+1, k+m+1)]
            for x in prefixes(P):
                pr = prob(x, gates, H, eps, c, g0)
                bound_v = (1-eps)*(t0 if x[-1] == 0 else t1)
                assert pr <= bound_v, ("onestep", H, eps, c, beta, P, m, x, float(pr), float(bound_v))
                if bound_v > 0: worst = max(worst, pr/bound_v)
    return worst

def check_typed(H, eps, c, beta, g0, n, lmax):
    r = 1-eps*c; worst = F(0); cnt = 0
    def rec(Pprev, l):
        # yield lists of steps (k, P, type)
        yield []
        if l == 0: return
        for k in range(g0+2, Pprev+1):
            opts = [(q, 'res') for q in range(g0, k-1)]
            if k-H >= g0: opts.append((k-H, 'base'))
            for q, ty in opts:
                for rest in rec(q, l-1):
                    yield [(k, q, ty)] + rest
    for P0 in range(g0, n+1):
        for steps in rec(P0, lmax):
            if not steps: continue
            cons = {}
            bound = F(1); Pprev = P0
            for (k, q, ty) in steps:
                cons[k] = 'pi0' if ty == 'base' else ('res', q)
                for g in range(k+1, Pprev+1): cons[g] = 'noreset'
                m = Pprev-k
                bound *= (1-eps)*max(Tm(H, m, eps, c, beta)) if ty == 'base' else r**m*eps/k
                Pprev = q
            Pl = steps[-1][1]
            # exact probability, averaging over the law of the prefix up to P_l: start from x_0
            gates = [(g, cons.get(g)) for g in range(1, P0+1)]
            pr = prob((1,), gates, H, eps, c, g0)
            cnt += 1
            assert pr <= bound, ("typed", H, eps, c, beta, P0, steps, float(pr), float(bound))
            if bound > 0: worst = max(worst, pr/bound)
    return worst, cnt

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "step":
        for H in (2, 3, 4, 5):
            for eps in (F(3,10), F(6,10), F(74,100), F(9,10)):
                for (g0, beta) in ((1, F(0)), (2, F(1,4)), (4, F(3,8))):
                    for c in (F(1,20), F(1,5), F(33,100)):
                        check_onegate(H, eps, c, beta, g0, 11)
                        w = check_onestep(H, eps, c, beta, g0, 7 if g0 < 4 else 8, 6)
                        print("onestep ok H=%d eps=%s g0=%d beta=%s c=%s  max ratio prob/bound=%.4f" % (H, eps, g0, beta, c, float(w)), flush=True)
    else:
        for H in (2, 3, 4):
            for eps in (F(6,10), F(74,100)):
                for (g0, beta) in ((1, F(0)), (2, F(1,4))):
                    for c in (F(1,20), F(1,5)):
                        w, cnt = check_typed(H, eps, c, beta, g0, 11, 2)
                        print("typed ok H=%d eps=%s g0=%d beta=%s c=%s  skeletons=%d max ratio=%.4f" % (H, eps, g0, beta, c, cnt, float(w)), flush=True)
