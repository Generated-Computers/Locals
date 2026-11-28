# Monte Carlo test of Lemma onestep on S_eps(j^H):  P[pi_k=0, no reset in (k,k+m] | F_P]  vs  (1-eps) t_0(x_P),
# P = k-H, with c := measured lower bound of (O_{g-1}-1)/g over the tested range and beta = 1/2 - 1/(2 g_start).
import sys, random, math
from array import array

def tvals(H, m, eps, c, beta):
    r = 1-eps*c; e0 = eps*(1-c)
    q = min(m, H-1)
    t0 = t1 = r**(m-q)
    for i in range(q-1, -1, -1):
        n1 = r*t0
        n0 = max(a*e0*t1 + (1-a)*r*t0 for a in (eps*beta, r))
        t0, t1 = n0, n1
    return t0, t1

def varpi(eps): return ((2*eps+1) - math.sqrt(4*eps+1))/(2*eps)

def run(H, eps, n, seed, gstart=20000, mmax=12):
    rnd = random.Random(seed)
    x = bytearray(n+1); pi = bytearray(n+1); rs = bytearray(n+1)
    x[0] = 1; ones = 1; cmin = 1.0
    rr = rnd.random; ri = rnd.randrange
    for g in range(1, n+1):
        if g > gstart:
            d = (ones-1)/g
            if d < cmin: cmin = d
        if rr() < eps:
            pi[g] = 1; p = ri(g)
        else:
            p = g-H if g >= H else 0
        if p != g-1 and x[p]:
            rs[g] = 1                      # reset: x_g = 0
        elif not x[g-1]:
            x[g] = 1; ones += 1
    dens = ones/(n+1)
    nr = array('i', [0])*(n+2); nxt = n+1+10**6   # next reset strictly above i
    for i in range(n, -1, -1):
        nr[i] = nxt
        if rs[i]: nxt = i
    beta = 0.5 - 0.5/gstart
    # stratify on x_P and on a finer F_P-measurable key: (x_{P-1}, x_P, reset at P, pi_P)
    cnt = {}; hit = {}
    cntv = [0, 0]; hitv = [[0]*(mmax+1) for _ in range(2)]
    tail = [0.0, 0.0]   # mean number of m>=0 with the event (unbounded m), by v
    for k in range(gstart+H, n-mmax-2000):
        P = k-H; v = x[P]
        key = (x[P-1], v, rs[P], pi[P])
        cntv[v] += 1; cnt[key] = cnt.get(key, 0)+1
        if pi[k] == 0:
            run_len = nr[k]-k-1          # gates k+1..k+run_len are not resets
            tail[v] += min(run_len, n-k)+1
            h = hit.setdefault(key, [0]*(mmax+1))
            top = min(run_len, mmax)
            for m in range(top+1):
                hitv[v][m] += 1; h[m] += 1
    print("H=%d eps=%.2f n=%d density=%.4f  varpi=%.4f  c_meas(min over g>%d)=%.4f" % (H, eps, n, dens, varpi(eps), gstart, cmin))
    worst = 0.0
    for m in range(mmax+1):
        t = tvals(H, m, eps, cmin, beta)
        line = "  m=%2d" % m
        for v in (0, 1):
            f = hitv[v][m]/cntv[v]; b = (1-eps)*t[v]
            se = math.sqrt(max(f*(1-f), 1e-12)/cntv[v])
            line += "   v=%d meas=%.5f bound=%.5f ratio=%.3f z=%.1f" % (v, f, b, f/b, (f-b)/se)
            if m >= 1: worst = max(worst, f/b)
        print(line)
    # finer strata
    wkey = (0.0, None, None)
    for key, c in sorted(cnt.items()):
        if c < 20000: continue
        h = hit.get(key, [0]*(mmax+1))
        for m in range(1, mmax+1):
            t = tvals(H, m, eps, cmin, beta)
            f = h[m]/c; b = (1-eps)*t[key[1]]
            se = math.sqrt(max(f*(1-f), 1e-12)/c)
            if f/b > wkey[0]: wkey = (f/b, key, m, (f-b)/se, c)
    print("  worst ratio over m>=1, coarse:", round(worst, 4), " over strata (x_{P-1},x_P,reset_P,pi_P):", wkey)
    for v in (0, 1):
        r = 1-eps*varpi(eps)
        s = 0.0; m = 0
        while r**m > 1e-16:
            s += tvals(H, m, eps, varpi(eps), 0.5)[v]; m += 1
        print("  mean #(k keeps base pointer, no reset above up to P) given x_{k-H}=%d: measured %.3f   (1-eps)*sum_m t_0(%d) with c=varpi: %.3f" % (v, tail[v]/cntv[v], v, (1-eps)*s))

if __name__ == "__main__":
    H = int(sys.argv[1]); eps = float(sys.argv[2]); n = int(sys.argv[3]); seed = int(sys.argv[4])
    run(H, eps, n, seed)
