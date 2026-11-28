# Per-m test of the one-step lemma: P[pi_k=0, no reset in (k,k+m] | x_{k-H}=v] <= (1-eps) t_0^{(m)}(v).
import random, sys
sys.path.insert(0, '.')
from fd_bound import varpi
def tvals(e, H, m, c, beta):
    r = 1-e*c; e0 = e*(1-c); q = min(m, H-1)
    T0 = T1 = r**(m-q)
    for i in range(q-1, -1, -1):
        n1 = r*T0
        T0, T1 = max(a*e0*T1 + (1-a)*r*T0 for a in (e*beta, r)), n1
    return T0, T1
def run(n, H, eps, seed, M=10):
    rng = random.Random(seed)
    x = [1]*(n+1); base = [False]*(n+1); reset = [False]*(n+1)
    for i in range(1, n+1):
        if rng.random() < eps: j = rng.randrange(i)
        else: j = max(i-H, 0); base[i] = True
        x[i] = 1 - (x[i-1] | x[j]); reset[i] = (j != i-1 and x[j] == 1)
    nxt = n+1; cnt = [[0]*M for _ in (0,1)]; tot = [0, 0]
    for k in range(n-M, n//4, -1):
        v = x[k-H]; tot[v] += 1
        if base[k]:
            for m in range(min(M, nxt-k)): cnt[v][m] += 1
        if reset[k]: nxt = k
    worst = 0.0
    for m in range(M):
        b = tvals(eps, H, m, varpi(eps), 0.5)
        for v in (0, 1):
            emp = cnt[v][m]/tot[v]; bound = (1-eps)*b[v]
            if m == 0: assert abs(emp/bound-1) < 0.01   # equality: the gate keeps its pointer
            else: worst = max(worst, emp/bound)
            if m in (0, 1, 2, 4, 8) : print("  H=%d eps=%.2f m=%d v=%d  measured %.4f  bound %.4f" % (H, eps, m, v, emp, bound))
    return worst
for H, eps in ((3, 0.74), (8, 0.74), (8, 0.60), (16, 0.50)):
    w = run(1500000, H, eps, 11); print("H=%d eps=%.2f: largest measured/bound over 1<=m<10: %.3f" % (H, eps, w)); assert w < 1.0
