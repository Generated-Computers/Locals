# (e) simulation on ordered bases: mean over P_0 of  #{active k <= P_0 : pi_k = 0, no reset in (k, P_0]}
#     against Theta^ord_D with the measured density constants (c = min (O_{g-1}-1)/g, beta = min (g-O_{g-1})/g over g > gstart).
import sys, random, math
from array import array

def theta_ord(D, eps, c, beta):
    r = 1-eps*c
    return (1-eps)*(1/(eps*c + eps*eps*beta*c*(1-eps)) + r**D/(eps*c))
def varpi(e): return ((2*e+1) - math.sqrt(4*e+1))/(2*e)

def make_base(kind, n, D, seed):
    rg = random.Random(seed); j = array('i', [0])*(n+1)
    if kind == "half":
        for i in range(1, n+1): j[i] = i//2 if i % 2 == 0 else i-1
    elif kind == "even":           # even gates read D below (all pointer increments are 2), odd gates are NOT gates
        for i in range(1, n+1): j[i] = i-D if (i % 2 == 0 and i > D) else i-1
    elif kind == "random":         # pointer increments 1,2,3 at random, distance kept in [D, 3D], 70% of gates active
        last = 0
        for i in range(1, n+1):
            cand = last + rg.choice((1, 2, 3))
            if i-cand > 3*D: cand = i-3*D
            if rg.random() < 0.7 and cand <= i-D and cand > last: j[i] = cand; last = cand
            else: j[i] = i-1
    elif kind == "sparse2":        # every third gate active, pointer increments 3 (distance constant D), others NOT
        for i in range(1, n+1): j[i] = i-D if (i % 3 == 0 and i > D) else i-1
    elif kind == "fixed":
        for i in range(1, n+1): j[i] = max(i-D, 0)
    return j

def run(kind, D, eps, n, seed, gstart=20000):
    j = make_base(kind, n, D, seed)
    # verify ordering above gstart (and globally for the pointer order)
    last = 0; mind = 10**9
    for i in range(2, n+1):
        if j[i] != i-1 and j[i] >= 1:
            assert j[i] > last, "pointers not increasing"; last = j[i]
            if i > gstart: mind = min(mind, i-j[i])
    Deff = min(mind, 10**6)
    rnd = random.Random(seed+17); rr = rnd.random; ri = rnd.randrange
    x = bytearray(n+1); keep = bytearray(n+1); rs = bytearray(n+1); x[0] = 1; ones = 1; cmin = 1.0; bmin = 1.0
    for g in range(1, n+1):
        if g > gstart:
            d = (ones-1)/g
            if d < cmin: cmin = d
            d = (g-ones)/g
            if d < bmin: bmin = d
        if rr() < eps: p = ri(g)
        else:
            p = j[g]
            if p != g-1 and p >= 1: keep[g] = 1
        if p != g-1 and x[p]: rs[g] = 1
        elif not x[g-1]: x[g] = 1; ones += 1
    # count[P0] summed: active kept gate k contributes to P0 = k .. nextreset(k)-1 ; split by parity of P0 and by x_{j_k}
    nxt = n+1; tot = 0; totpar = [0, 0]; byv = [0, 0]; nk = [0, 0]
    lo = gstart+10; hi = n-5000
    for k in range(n, lo-1, -1):
        if keep[k] and j[k] >= gstart and k <= hi:
            L = min(nxt, hi+1)-k             # P0 in [k, min(nxt-1, hi)]
            if L > 0:
                tot += L; v = x[j[k]]; byv[v] += L
                e = (L+1)//2; o = L//2       # P0 = k, k+1, ...: parity of k first
                totpar[k & 1] += e; totpar[1-(k & 1)] += o
        if rs[k]: nxt = k
    for k in range(lo, hi+1):
        if j[k] != k-1 and j[k] >= 1: nk[x[j[k]]] += 1
    npos = hi-lo+1
    th = theta_ord(Deff, eps, cmin, bmin)
    print("%-8s D=%-3d eps=%.3f n=%d: density %.4f c_meas %.4f beta_meas %.4f | mean count %.4f (P0 even %.4f, odd %.4f) | Theta^ord_D(measured c,beta) = %.4f | with varpi,1/2: %.4f | 001c factor (1-eps)/(eps c_meas) = %.4f" % (
        kind, Deff, eps, n, ones/(n+1), cmin, bmin, tot/npos, totpar[0]/(npos/2), totpar[1]/(npos/2), th, theta_ord(Deff, eps, varpi(eps), 0.5), (1-eps)/(eps*cmin)))
    assert tot/npos <= th and max(totpar)/(npos/2) <= th, "MEASURED SUM ABOVE Theta^ord"

if __name__ == "__main__":
    n = int(sys.argv[1])
    for a in sys.argv[2:]:
        kind, D, eps, seed = a.split(":"); run(kind, int(D), float(eps), n, int(seed))
