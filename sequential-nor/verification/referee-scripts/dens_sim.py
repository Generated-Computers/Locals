# (b) simulation of S_eps(j^H): prefix densities psi_g vs [l*, mu*], and the interval inequalities
#   X >= Lambda(mu) m - Delta(m),  X <= U(l) m + Delta(m),  |sum (x_i - a_i)| <= sqrt(2 m ln(n+1)),
# with mu = max, l = min of psi_{i-1} over i in (a,b]  (the tightest admissible choice), on sampled intervals.
import sys, random, math
from array import array

def Lam(mu, e): return (1-e*mu)/(3-e-e*mu)
def U(l, e): return (1-e*l)/(2-e*l)
def limits(e):
    mu = 0.5
    for s in range(400):
        l = Lam(mu, e); mu = U(l, e)
    return l, mu

def run(H, eps, n, seed):
    rnd = random.Random(seed); rr = rnd.random; ri = rnd.randrange
    x = bytearray(n+1); x[0] = 1
    O = array('i', [0])*(n+1); O[0] = 1
    SA = array('d', [0.0])*(n+1)       # SA[g] = sum_{i<=g} a_i  (a_i from the exact formula, i >= H; for i < H computed directly)
    ones = 1; sa = 0.0
    for g in range(1, n+1):
        z = 1-x[g-1]
        jb = g-H if g >= H else 0
        if z:
            a = eps*(1-ones/g) + (1-eps)*(1-x[jb])
            sa += a
        p = ri(g) if rr() < eps else jb
        if z and not x[p]:
            x[g] = 1; ones += 1
        O[g] = ones; SA[g] = sa
    ls, ms = limits(eps)
    L = math.log(n+1)
    out = "H=%2d eps=%.2f n=%d: l*=%.4f mu*=%.4f psi_n=%.4f" % (H, eps, n, ls, ms, ones/(n+1))
    for g1 in (1000, 20000, 200000):
        lo = 1.0; hi = 0.0
        for g in range(g1, n+1):
            v = O[g]/(g+1)
            if v < lo: lo = v
            if v > hi: hi = v
        out += " | g>=%d: psi in [%.4f, %.4f]%s" % (g1, lo, hi, "" if (ls <= lo and hi <= ms) else " OUTSIDE")
    print(out)
    # interval checks
    worst = {"low": 1e18, "up": 1e18, "mart": 1e18}; nchk = 0; viol = 0
    def sweep(a, bmax, step=1):
        nonlocal nchk, viol
        mu = 0.0; l = 1.0; Oa = O[a]; Sa = SA[a]
        for b in range(a+1, bmax+1):
            v = O[b-1]/b                      # psi_{b-1}
            if v > mu: mu = v
            if v < l: l = v
            if step > 1 and (b-a) % step: continue
            m = b-a; X = O[b]-Oa; S = math.sqrt(2*m*L); Dl = H+1+S
            s1 = X - (Lam(mu, eps)*m - Dl); s2 = (U(l, eps)*m + Dl) - X; s3 = S - abs(X - (SA[b]-Sa))
            if s1 < worst["low"]: worst["low"] = s1
            if s2 < worst["up"]: worst["up"] = s2
            if s3 < worst["mart"]: worst["mart"] = s3
            # normalised slack of the lower bound, to see how tight it gets
            nchk += 1
            if s1 < 0 or s2 < 0 or s3 < 0: viol += 1
    sweep(H, min(n, 20000))                       # every interval (H, b]
    for a in (H, H+1, H+2, 2*H+7, 1000, 100000, n//2):
        sweep(a, n, step=97)                      # long intervals
    r2 = random.Random(seed+1)
    for _ in range(400):
        a = r2.randrange(H, n-3001); sweep(a, a+3000)   # short and medium intervals everywhere
    # tightness of the lower bound without the Delta allowance on long intervals: X/m - Lambda(mu)
    a = n//2; m = n-a; mu = max(O[p]/(p+1) for p in range(a, n)); l = min(O[p]/(p+1) for p in range(a, n))
    X = O[n]-O[a]
    print("   intervals checked: %d, violations: %d, min slack lower %.2f upper %.2f martingale %.2f | (n/2,n]: X/m=%.4f  Lambda(mu)=%.4f  U(l)=%.4f" % (nchk, viol, worst["low"], worst["up"], worst["mart"], X/m, Lam(mu, eps), U(l, eps)))

if __name__ == "__main__":
    n = int(sys.argv[1])
    cfgs = [(int(a), float(b)) for a, b in (s.split(":") for s in sys.argv[2:])]
    for k, (H, eps) in enumerate(cfgs):
        run(H, eps, n, 1000+k)
