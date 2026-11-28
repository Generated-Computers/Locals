# Task 4 (e): simulation on bases with ties.  For every P_0 count the top-reading anchors
#   { active k <= P_0, j_k >= gstart : pi_k = 0, no reset in (k,P_0], no g in (k,P_0] with pi_g = 0 and j_g = j_k }
# and compare the mean with Theta^ord_D computed with the measured density constants.
import sys, random, math
from array import array

def theta_ord(D, eps, c, beta):
    r = 1-eps*c
    return (1-eps)*(1/(eps*c + eps*eps*beta*c*(1-eps)) + r**D/(eps*c))

def make_base(kind, n, D, seed):
    rg = random.Random(seed); j = array('i', [0])*(n+1)
    if kind == "half":                       # floor(i/2): two gates per position
        for i in range(1, n+1): j[i] = i//2
    elif kind == "blocks":                   # blocks of B gates all reading one position D below the block start
        Bz = 6
        for i in range(1, n+1):
            start = (i//Bz)*Bz
            j[i] = start-D if start-D >= 1 else i-1
    elif kind == "blocksnot":                # the same, but every second gate of a block is a NOT gate
        Bz = 8
        for i in range(1, n+1):
            start = (i//Bz)*Bz
            j[i] = start-D if (start-D >= 1 and i % 2 == 0) else i-1
    elif kind == "randties":                 # nondecreasing pointers, increments 0 (often),1,2,3; distance in [D, 3D]
        last = 1
        for i in range(1, n+1):
            cand = last + rg.choice((0, 0, 0, 1, 2, 3))
            if i-cand > 3*D: cand = i-3*D
            if rg.random() < 0.8 and cand <= i-D and cand >= last: j[i] = cand; last = cand
            else: j[i] = i-1
    elif kind == "bigblocks":                # long blocks: 40 gates reading one position
        Bz = 40
        for i in range(1, n+1):
            start = (i//Bz)*Bz
            j[i] = start-D if start-D >= 1 else i-1
    return j

def run(kind, D, eps, n, seed, gstart=20000):
    j = make_base(kind, n, D, seed)
    last = 0; mind = 10**9
    for i in range(2, n+1):
        if j[i] != i-1 and j[i] >= 1:
            assert j[i] >= last, "pointers decrease"; last = j[i]
            if i > gstart: mind = min(mind, i-j[i])
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
    nxt = n+1; tot_top = 0; tot_all = 0
    lastkept = array('i', [0])*(n+1)          # lastkept[p] = lowest kept gate seen so far (from the top) with pointer p
    lo = gstart+10; hi = n-5000
    for k in range(n, lo-1, -1):
        if keep[k] and j[k] >= gstart and k <= hi:
            stop_all = min(nxt, hi+1)
            L = stop_all-k
            if L > 0: tot_all += L
            same = lastkept[j[k]]
            stop_top = min(stop_all, same) if same else stop_all
            L = stop_top-k
            if L > 0: tot_top += L
        if keep[k]: lastkept[j[k]] = k
        if rs[k]: nxt = k
    npos = hi-lo+1
    th = theta_ord(mind, eps, cmin, bmin)
    print("%-9s D=%-3d eps=%.3f n=%d: density %.4f c_meas %.4f beta_meas %.4f | top-reading anchors per window %.4f | all anchors %.4f | Theta^ord_D(measured) = %.4f | 001c factor %.4f" % (
        kind, mind, eps, n, ones/(n+1), cmin, bmin, tot_top/npos, tot_all/npos, th, (1-eps)/(eps*cmin)))
    assert bmin <= 2/3 and tot_top/npos <= th, "MEASURED ABOVE Theta^ord"

if __name__ == "__main__":
    n = int(sys.argv[1])
    for a in sys.argv[2:]:
        kind, D, eps, seed = a.split(":"); run(kind, int(D), float(eps), n, int(seed))
