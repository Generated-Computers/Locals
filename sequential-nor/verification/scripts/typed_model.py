# Exploration (not a proof): worst-case growth of the chain count in a conservative two-level model
# in which the weight passed from one chain step to the next is a function of the value v at the
# current target, and one window gate per step (the gate just above the step's gate) is credited:
#   zone G1 / G2: it reads 1 / at least 2 above the step's target (fresh, same level)
#   zone T      : it reads the target itself
#   zone B(o)   : it reads o below the target; resolved at the next step:
#                 inside the next window (value fixed by parity) or in the next gap (fresh)
# The base designer picks a zone for every window length m' separately; value iteration finds the worst.
import math, sys
def varpi(e): return ((2*e+1) - math.sqrt(4*e+1))/(2*e)
def run(e, c=None, beta=0.5, allow=("G1","G2","T","B"), O=8, MM=28, NG=96, iters=40, verbose=False):
    if c is None: c = varpi(e)
    r = 1-e*c; e0 = e*(1-c); gam = e*e*beta*c; s0 = r-gam*(1-e)
    phi = (r, e0)
    def mix(vb0, vb1):            # max over p in [gam, r] of p*vb1 + (1-p)*vb0
        return r*vb1+(1-r)*vb0 if vb1 >= vb0 else (1-gam)*vb0+gam*vb1
    ts = [i*(math.pi/2)/(NG-1) for i in range(NG)]
    pends = [None]+list(range(1, O+1))
    # W[pi][gi] = value at direction ts[gi] with pending state pends[pi]
    W = [[max(math.cos(t), math.sin(t)) for t in ts] for _ in pends]
    def val(Wt, y, pend):
        n = math.hypot(y[0], y[1])
        if n == 0: return 0.0
        t = math.atan2(y[1], y[0]); p = t/(math.pi/2)*(NG-1); i = min(int(p), NG-2); f = p-i
        row = Wt[0 if pend is None else pend]
        return n*((1-f)*row[i]+f*row[i+1])
    zones = []
    if "G1" in allow: zones.append(("G1", r-e*beta*(1-e), r, None))
    if "G2" in allow: zones.append(("G2", s0, s0, None))
    if "T" in allow: zones.append(("T", e0, e0, None))
    if "B" in allow: zones += [("B%d" % o, 1.0, 1.0, o) for o in range(1, O+1)]
    else: zones.append(("none", r, r, None))        # uncredited gate
    rate = None
    for it in range(iters):
        newW = []
        for pend in pends:
            row = []
            for t in ts:
                Y = (math.cos(t), math.sin(t)); tot = 0.0
                for m in range(0, MM+1):
                    p = m & 1
                    if pend is None: kind = "none"
                    elif pend <= m: kind = "W"
                    else: kind = "L"; ee = pend-m
                    def pf(cv):
                        if kind == "W": return phi[cv ^ ((m-pend) & 1)]
                        return 1.0
                    Psi = (Y[0 ^ p]*pf(0), Y[1 ^ p]*pf(1))          # Psi[c], c = value at the step's gate
                    if kind == "L" and ee == 1: term0 = mix(Psi[1]*r, Psi[0]*e0)       # (value at b=0, value at b=1); b=0 -> c=1, b=1 -> c=0
                    elif kind == "L": term0 = min(s0*max(Psi), r*mix(Psi[1], Psi[0]))
                    else: term0 = mix(Psi[1], Psi[0])
                    term1 = Psi[0]*(s0 if kind == "L" else 1.0)
                    if m == 0:
                        best = val(W, ((1-e)*term0, (1-e)*term1), None)
                    else:
                        pw = (1-e)*r**(m-1); best = 0.0
                        for (nm, f0, f1, o2) in zones:
                            v = val(W, (pw*f0*term0, pw*f1*term1), o2)
                            if v > best: best = v
                    tot += best
                row.append(tot)
            newW.append(row)
        mid = (NG-1)//2
        rate = newW[0][mid]/W[0][mid]
        # renormalise
        sc = newW[0][mid]
        W = [[x/sc for x in row] for row in newW]
        if verbose and it % 5 == 4: print("  it %d rate %.5f" % (it+1, rate)); sys.stdout.flush()
    return rate
def covered_all(e, c=None, beta=0.5):
    # illustration: every window gate reads below the target and lands in the next window with the
    # parity that suits value 0; every gate is credited.  Growth factor of the 2x2 recursion
    #   Y'(0) = (1-e) sum_m r^m  * [ m even: r Y(0)+(1-r) Y(1) ;  m odd: (1-g) Y(0)+g Y(1) ]
    #   Y'(1) = (1-e) sum_m e0^m * Y(m mod 2)
    if c is None: c = varpi(e)
    r = 1-e*c; e0 = e*(1-c); g = e*e*beta*c
    a00 = (1-e)/(1-r*r)*(r + r*(1-g)); a01 = (1-e)/(1-r*r)*((1-r) + r*g)
    a10 = (1-e)/(1-e0*e0); a11 = (1-e)/(1-e0*e0)*e0
    tr = a00+a11; det = a00*a11-a01*a10; lam = (tr+math.sqrt(tr*tr-4*det))/2
    assert (lam-a11)/a10 >= 1          # the eigenvector has Y(0) >= Y(1), as the formula for Y'(0) assumes
    return lam
if __name__ == "__main__":
    for ee in (0.72, 0.73, 0.74, 0.75):
        print("all reads covered, every gate credited, eps=%.2f: %.4f   (plain %.4f)" % (ee, covered_all(ee), (1-ee)/(ee*varpi(ee))))
    e = 0.74
    print("plain factor at 0.74: %.4f" % ((1-e)/(e*varpi(e))))
    print("no credit at all    : %.4f" % run(e, allow=()))
    print("only G1, G2, T      : %.4f  (the base can always choose 'below', uncredited)" % run(e, allow=("G1","G2","T")))
    print("full model          : %.4f" % run(e, verbose=True))
