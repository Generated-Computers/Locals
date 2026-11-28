# Exact (rational) re-check of Lemma onegate and Lemma onestep with the NEW events
#   G_g = {(O_{g-1}-1)/g >= c  and  (g-O_{g-1})/g >= beta},
# for arbitrary c>0, beta>=0 (beta may exceed 1/2), any g0>=1, H>=2.  No relation between g0 and beta is assumed.
import sys, random
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
def inG(x, g, c, beta):
    ones = sum(x)
    return F(ones-1, g) >= c and F(g-ones, g) >= beta
def step(x, g, H, eps, cons):
    ones = sum(x); o2 = ones - x[g-1]; out = [0, 0]
    jb = base(H, g); rs_b = (jb != g-1) and x[jb] == 1
    if cons is None or cons == 'pi0' or (cons == 'noreset' and not rs_b):
        out[0 if rs_b else 1-x[g-1]] += 1-eps
    if cons is None:
        out[0] += eps*F(o2, g); out[1-x[g-1]] += eps*F(g-o2, g)
    elif cons == 'noreset':
        out[1-x[g-1]] += eps*F(g-o2, g)
    return out
def prob(x0, gates, H, eps, c, beta, g0):
    cur = {x0: F(1)}
    for g, cons in gates:
        nxt = {}
        for x, p in cur.items():
            if g > g0 and not inG(x, g, c, beta): continue
            o = step(x, g, H, eps, cons)
            for v in (0, 1):
                if o[v]: nxt[x+(v,)] = nxt.get(x+(v,), 0) + p*o[v]
        cur = nxt
    return sum(cur.values())

rnd = random.Random(777); nontriv = 0; tot = 0; worst = F(0); viol_interval = 0
for trial in range(220):
    H = rnd.choice((2, 2, 3, 3, 4, 5, 6))
    eps = F(rnd.randint(3, 97), 100)
    g0 = rnd.choice((1, 2, 3, 5))
    c = F(rnd.randint(1, 40), 100)
    beta = F(rnd.randint(0, 75), 100)
    r = 1-eps*c; e0 = eps*(1-c)
    if eps*beta > r: viol_interval += 1
    # onegate on G_g
    for g in range(g0+1, 12):
        for x in prefixes(g-1):
            if not inG(x, g, c, beta): continue
            assert c + beta < 1                       # nonempty G_g forces c+beta <= (g-1)/g
            o2 = sum(x)-x[-1]
            assert eps*F(g-o2, g) <= e0               # (i): P[pi=1, not a reset]
            p1 = step(x, g, H, eps, None)[1]
            if x[-1] == 1: assert p1 == 0
            else: assert eps*beta <= p1 <= r
    Pmax = g0 + rnd.randint(1, 4); mmax = rnd.randint(1, 7)
    for P in range(g0, Pmax+1):
        k = P+H
        for m in range(0, mmax+1):
            t = Tm(H, m, eps, c, beta)
            gates = [(g, None) for g in range(P+1, k)] + [(k, 'pi0')] + [(g, 'noreset') for g in range(k+1, k+m+1)]
            for x in prefixes(P):
                pr = prob(x, gates, H, eps, c, beta, g0); tot += 1
                b = (1-eps)*t[x[-1]]
                assert pr <= b, ("onestep FAILS", H, eps, c, beta, g0, P, m, x, float(pr), float(b))
                if pr > 0 and m >= 1:
                    nontriv += 1; worst = max(worst, pr/b)
print("new G_g: onegate/onestep exact checks passed; cases %d, with positive probability and m>=1: %d, max prob/bound there %.4f; parameter draws with eps*beta > r (G_g then empty): %d" % (tot, nontriv, float(worst), viol_interval))
