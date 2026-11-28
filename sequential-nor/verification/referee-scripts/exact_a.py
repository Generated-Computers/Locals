# (c) exact check, by enumeration over the noise of gate i for every reachable trace prefix, of
#     a_i = P[x_i=1 | F_{i-1}] = z_{i-1}( eps(1-psi_{i-1}) + (1-eps)(1-x_{i-H}) )   for i >= H,
# the two bounds derived from it, and the deterministic part of the interval inequalities
# (the counting of zeros among a..b-1 and of ones among a+1-H..b-H) for all H <= a < b <= n.
import sys
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

def a_enum(x, i, H, eps):
    # x = (x_0..x_{i-1}); enumerate pi_i, rho_i
    p1 = F(0)
    jb = max(i-H, 0)
    if (1 - (x[i-1] | x[jb])) == 1: p1 += 1-eps
    for p in range(i):
        if (1 - (x[i-1] | x[p])) == 1: p1 += eps/i
    return p1

def reachable_by_full_enumeration(n, H):
    # all traces produced by all pointer arrays (support of the smoothing for 0<eps<1)
    cur = {(1,)}
    for i in range(1, n+1):
        nxt = set()
        for x in cur:
            for p in range(i):
                nxt.add(x + (1 - (x[i-1] | x[p]),))
        cur = nxt
    return cur

n = int(sys.argv[1])
assert reachable_by_full_enumeration(9, 3) == set(prefixes(9))
for H in (1, 2, 3, 4, 5):
    for eps in (F(1,20), F(7,10), F(1)):
        mism_lt_H = 0; checked = 0; detchecks = 0
        for x in prefixes(n):
            # conditional means along this trace
            a = [None]*(n+1)
            for i in range(1, n+1):
                pre = x[:i]
                ae = a_enum(pre, i, H, eps)
                z = 1-pre[i-1]; psi = F(sum(pre), i)
                if i >= H:
                    form = z*(eps*(1-psi) + (1-eps)*(1-pre[i-H]))
                    assert ae == form, ("formula fails", H, eps, i, pre)
                    lo = eps*z*(1-psi) + (1-eps)*(z - pre[i-H]); hi = z*(1-eps*psi)
                    assert lo <= ae <= hi
                    checked += 1
                else:
                    form = z*(eps*(1-psi) + (1-eps)*(1-pre[0]))   # what the formula would say with pointer 0
                    if ae != z*(eps*(1-psi) + (1-eps)*(1-(pre[i-H] if i-H >= 0 else 0))): mism_lt_H += 1
                    assert ae <= z*(1-eps*psi)                      # upper bound holds for every i
                a[i] = ae
            O = [0]*(n+1); s = 0
            for g in range(n+1): s += x[g]; O[g] = s
            psi = [F(O[g], g+1) for g in range(n+1)]
            for aa in range(H, n):
                for b in range(aa+1, n+1):
                    m = b-aa; X = O[b]-O[aa]
                    zeros = sum(1-x[p] for p in range(aa, b)); ones_sh = sum(x[p] for p in range(aa+1-H, b-H+1))
                    assert m-X-1 <= zeros <= m-X+1 and ones_sh <= X+H
                    mu = max(psi[i-1] for i in range(aa+1, b+1)); l = min(psi[i-1] for i in range(aa+1, b+1))
                    SA = sum(a[i] for i in range(aa+1, b+1))
                    assert SA >= eps*(1-mu)*(m-X-1) + (1-eps)*((m-X-1)-(X+H))
                    assert SA <= (1-eps*l)*(m-X+1)
                    # algebra: if X >= SA - S then (3-eps-eps mu) X >= (1-eps mu) m - (H+1) - S; test with S := SA - X (can be negative)
                    S = SA - X
                    assert (3-eps-eps*mu)*X >= (1-eps*mu)*m - (H+1) - S
                    assert (2-eps*l)*X <= (1-eps*l)*(m+1) + (-S)
                    detchecks += 1
        print("H=%d eps=%s n=%d: formula exact for i>=H at %d (prefix,i) pairs; interval counting/algebra ok on %d (trace,a,b); #(i<H) where a_i differs from the i>=H formula read with x_{i-H}:=0: %d" % (H, eps, n, checked, detchecks, mism_lt_H))
