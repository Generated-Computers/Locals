# Task 5 (a): Lemma typebit in exact rational arithmetic, plus the two example values and the 2x2 recursion of Remark below(iii).
import sys, random, math
from fractions import Fraction as F

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
    ones = sum(x); return F(ones-1, g) >= c and F(g-ones, g) >= beta
def gate_dist(x, g, jb, eps, only_pi0=False):
    """law of x_g given the prefix x (length g); if only_pi0, restricted to pi_g = 0"""
    out = [F(0), F(0)]
    out[1-(x[g-1] | x[jb])] += 1-eps
    if not only_pi0:
        ones = sum(x)
        if x[g-1] == 1: out[0] += eps
        else:
            out[1] += eps*F(g-ones, g); out[0] += eps*F(ones, g)
    return out
def rand_adm2(rnd):
    while True:
        eps = F(rnd.randint(1, 75), 100); c = F(rnd.randint(1, 49), 100); beta = F(rnd.randint(0, 66), 100)
        if c + beta < 1: return eps, c, beta
def sharp(Psi, eps, c, beta):
    r = 1-eps*c; g = eps*eps*beta*c
    return (max(r*Psi[0] + (1-r)*Psi[1], (1-g)*Psi[1] + g*Psi[0]), Psi[0])

if __name__ == "__main__":
    rnd = random.Random(555); n = 13; cases = 0; tight = F(0); pos = 0
    bases = [("j^3", [0]+[max(i-3, 0) for i in range(1, n+1)]), ("j^5", [0]+[max(i-5, 0) for i in range(1, n+1)]),
             ("floor(i/2)", [0]+[i//2 for i in range(1, n+1)])]
    for t in range(12): bases.append(("random pointers #%d" % t, [0]+[rnd.randrange(i) for i in range(1, n+1)]))
    for name, j in bases:
        for trial in range(6):
            eps, c, beta = rand_adm2(rnd)
            if trial == 0: eps, c, beta = F(74, 100), F(1, 4), F(2, 5)
            r = 1-eps*c; e0 = eps*(1-c); g0 = rnd.choice((1, 1, 2, 3))
            Psis = [(F(1), F(0)), (F(0), F(1)), (e0, r), (r, e0), (F(1), F(1)), (F(rnd.randint(0, 9), 3), F(rnd.randint(0, 9), 3)),
                    (r*r*e0, e0*e0*r), (e0**3, r**3)]
            for kp in range(4, n+1):
                Pp = j[kp]
                if Pp < g0 or kp-Pp < 3: continue
                for x in prefixes(kp-3):
                    # distribution of the prefix up to k'-1 with the events G_{k'-2}, G_{k'-1}
                    cur = {x: F(1)}
                    for g in (kp-2, kp-1):
                        nxt = {}
                        for xx, p in cur.items():
                            assert g > g0
                            if not inG(xx, g, c, beta): continue
                            o = gate_dist(xx, g, j[g], eps)
                            for v in (0, 1):
                                if o[v]: nxt[xx+(v,)] = nxt.get(xx+(v,), 0) + p*o[v]
                        cur = nxt
                    for Psi in Psis:
                        val = F(0)
                        for xx, p in cur.items():
                            o = gate_dist(xx, kp, Pp, eps, only_pi0=True)
                            val += p*(o[0]*Psi[0] + o[1]*Psi[1])
                        bnd = (1-eps)*sharp(Psi, eps, c, beta)[x[Pp]]
                        cases += 1
                        assert val <= bnd, ("typebit FAILS", name, eps, c, beta, g0, kp, x, Psi, float(val), float(bnd))
                        if val > 0: pos += 1; tight = max(tight, val/bnd)
    print("Lemma typebit: %d exact cases (%d with positive left side), no violation; max lhs/rhs = %.4f" % (cases, pos, float(tight)))
    # the two example values, exact, on random admissible parameters
    for trial in range(2000):
        eps, c, beta = rand_adm2(rnd); r = 1-eps*c; e0 = eps*(1-c); g = eps*eps*beta*c; s0 = r-g*(1-eps)
        assert sharp((e0, r), eps, c, beta)[0] == s0
        assert sharp((r, e0), eps, c, beta)[0] == r - eps*c*(1-eps) <= s0
        # both entries of M^2 w are bounded by Psi^sharp(0), for w(b) = Psi(1-b), directly from the definition of M
        for Psi in ((e0, r), (r, e0), (F(rnd.randint(0, 9)), F(rnd.randint(0, 9)))):
            w = (Psi[1], Psi[0])
            def Mop(w): return (max(a*w[1] + (1-a)*w[0] for a in (eps*beta, r)), w[0])
            m2 = Mop(Mop(w)); assert max(m2) <= sharp(Psi, eps, c, beta)[0]
    print("examples: Psi=(e0,r) gives Psi#(0)=s_0; Psi=(r,e0) gives r-eps c(1-eps) <= s_0; M^2 w <= Psi#(0): exact on 2000 random admissible parameter sets")

    # Remark below(iii): growth factor of  Y'(0) = (1-eps) sum_m r^m Psi_m#(0),  Y'(1) = (1-eps) sum_m e0^m Y(m mod 2),  Psi_m(c) = Y(c xor m)
    def varpi(e): return ((2*e+1) - math.sqrt(4*e+1))/(2*e)
    def growth(e, c=None, beta=0.5, iters=4000):
        if c is None: c = varpi(e)
        r = 1-e*c; e0 = e*(1-c); g = e*e*beta*c
        Y = (1.0, 1.0); lam = None
        for it in range(iters):
            A = max(r*Y[0] + (1-r)*Y[1], (1-g)*Y[1] + g*Y[0])       # m even: Psi = (Y0, Y1)
            Bv = max(r*Y[1] + (1-r)*Y[0], (1-g)*Y[0] + g*Y[1])      # m odd:  Psi = (Y1, Y0)
            Y0 = (1-e)*(A + r*Bv)/(1-r*r)
            Y1 = (1-e)*(Y[0] + e0*Y[1])/(1-e0*e0)
            lam = max(Y0, Y1)/max(Y); Y = (Y0/max(Y0, Y1), Y1/max(Y0, Y1))
        # also by brute-force summation over m for one step, as a check of the closed sums
        tot0 = sum(r**m*sharp((Y[m % 2], Y[1-(m % 2)]), e, c, beta)[0] for m in range(400))*(1-e)
        tot1 = sum(e0**m*Y[m % 2] for m in range(400))*(1-e)
        assert abs(max(tot0, tot1)/max(Y) - lam) < 1e-9
        return lam, Y
    for e in (0.72, 0.73, 0.74, 0.75):
        lam, Y = growth(e); print("recursion growth factor at eps=%.2f: %.4f   (eigenvector %.4f, %.4f);   001c factor %.4f" % (e, lam, Y[0], Y[1], (1-e)/(e*varpi(e))))
    lo, hi = 0.70, 0.75
    for _ in range(50):
        mid = (lo+hi)/2
        if growth(mid, iters=1500)[0] > 1: lo = mid
        else: hi = mid
    print("growth factor equals 1 at eps = %.4f" % lo)
