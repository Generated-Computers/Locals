# Ties: several gates of a window may read the same position.  Checks for the generalised lemmas.
#  (c)  Phi(y) >= ||y||/c on the cone, i.e. (1-gamma)(1-eps*beta) >= e0*eps*beta
#  (d)  c*(S_mu - r^(mu-1)) <= 1 - r^(mu-1),  S_mu = 1 + e0 + ... + e0^(mu-1)
#  sum lemma with groups: y_1 = 1, y_{u+1} = M^d[phi^mu y_u] (exact), sum_u S_mu ||y_u|| <= 1/(1-s0)
#  simulation of two bases with ties, counting canonical anchors only
import math, random, itertools
def varpi(e): return ((2*e+1) - math.sqrt(4*e+1))/(2*e)
def consts(e, c, beta):
    r = 1-e*c; e0 = e*(1-c); alo = e*beta; gamma = e*e*beta*c
    return r, e0, alo, gamma
def M(w, alo, ahi): return (max(alo*w[1]+(1-alo)*w[0], ahi*w[1]+(1-ahi)*w[0]), w[0])
def step(y, d, mu, r, e0, alo):
    w = (r**mu*y[0], e0**mu*y[1])
    for _ in range(d): w = M(w, alo, r)
    return w
def total(seq, e, c, beta):
    r, e0, alo, gamma = consts(e, c, beta); y = (1.0, 1.0); tot = 0.0
    for (d, mu) in seq:
        tot += (1-e0**mu)/(1-e0)*max(y); y = step(y, d, mu, r, e0, alo)
    return tot
rng = random.Random(5)
for _ in range(300000):
    e = rng.uniform(0.01, 0.75); c = rng.uniform(1e-3, 0.5); beta = rng.uniform(0, min(2/3, 1-c))
    r, e0, alo, gamma = consts(e, c, beta)
    assert (1-gamma)*(1-e*beta) >= e0*e*beta
    mu = rng.randint(1, 40); S = (1-e0**mu)/(1-e0)
    assert c*(S - r**(mu-1)) <= 1 - r**(mu-1) + 1e-12
print("inequalities (c) and (d): ok on 300000 random parameter sets")
worst = 0.0
for (e, c, beta) in ((0.74, varpi(0.74), 0.5), (0.70, varpi(0.70), 0.5), (0.5, 0.3, 0.55), (0.3, 0.2, 0.66), (0.74, 0.35, 0.57), (0.1, 0.05, 0.6)):
    r, e0, alo, gamma = consts(e, c, beta); bound = 1.0/(e*c + gamma*(1-e))
    alphabet = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (2, 5), (7, 3)]
    for L in range(1, 7):
        for seq in itertools.product(alphabet, repeat=L):
            t = total(seq, e, c, beta); assert t <= bound*(1+1e-12), (e, c, beta, seq, t, bound); worst = max(worst, t/bound)
    for _ in range(4000):
        seq = [(rng.choice((1, 1, 2, 2, 3, 30)), rng.choice((1, 1, 1, 2, 3, 20))) for _ in range(200)]
        t = total(seq, e, c, beta); assert t <= bound*(1+1e-12); worst = max(worst, t/bound)
    print("eps=%.2f c=%.3f beta=%.2f: bound %.4f; all groups of size 2 at distance 2: %.4f; one group of size 1000: %.4f"
          % (e, c, beta, bound, total([(2, 2)]*300, e, c, beta), total([(2, 1000)], e, c, beta)))
print("largest sum / bound over all sequences tried: %.4f" % worst)
# simulation: canonical anchors only
def base_floor_half(n): return [0]+[(i//2 if i//2 != i-1 else i-1) for i in range(1, n+1)]
def base_blocks(n, ell):          # every gate of a block reads the middle of the block below
    return [0]+[(max(((i-1)//ell-1)*ell + ell//2, 0) if i > ell else i-1) for i in range(1, n+1)]
def measure(j, eps, seed):
    n = len(j)-1; rg = random.Random(seed); x = [1]*(n+1); keep = [False]*(n+1); reset = [False]*(n+1)
    for i in range(1, n+1):
        if rg.random() < eps: t = rg.randrange(i)
        else: t = j[i]; keep[i] = (j[i] != i-1 and j[i] >= 1)
        x[i] = 1 - (x[i-1] | x[t]); reset[i] = (t != i-1 and x[t] == 1)
    # for each P0: count gates k <= P0 that keep their pointer, with no reset in (k,P0] and no gate in
    # (k,P0] keeping the same pointer.  Scan down from P0 for a sample of P0.
    tot_all = tot_can = 0; cnt = 0
    for P0 in range(n, n//2, -97):
        seen = set(); k = P0
        while k >= 1:
            if keep[k]:
                tot_all += 1
                if j[k] not in seen: tot_can += 1; seen.add(j[k])
            if reset[k]: break
            k -= 1
        cnt += 1
    return tot_all/cnt, tot_can/cnt
n = 600000
for name, j in (("j_i = floor(i/2)", base_floor_half(n)), ("blocks of 64 reading one position", base_blocks(n, 64))):
    for e in (0.60, 0.74):
        a, cmean = measure(j, e, 3)
        th = (1-e)*(1.0/(e*varpi(e) + e*e*0.5*varpi(e)*(1-e)) + (1-e*varpi(e))**32/(e*varpi(e)))
        print("%-34s eps=%.2f: all anchors %.3f, canonical anchors %.3f, Theta_ord_32 = %.3f" % (name, e, a, cmean, th))
        assert cmean <= th
