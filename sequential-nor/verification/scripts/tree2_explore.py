# Exploration for the refined NOR-tree embedding (robust zero, pair leaves, double top) with
# two-sided density bounds: A = eps*(varpi+delta) (misread of a 0), B = eps*(1-varpi+delta) (of a 1).
import math
def varpi(e): return ((2*e+1) - math.sqrt(4*e+1))/(2*e)
def cell(A, B, w0, w1, d, pair=True, robust=True, double=True):
    zf = (1-(1-B)*(1-A))*B if robust else B
    if pair:
        l0 = 1-(1-w0)**2*(1-zf)*(1-A)**2
        l1 = (1-(1-w1)*(1-B))*(1-(1-w1)*(1-B)*(1-A))
    else:
        l0 = 1-(1-w0)*(1-zf)*(1-A)
        l1 = 1-(1-w1)*(1-B)
    OR = lambda p: 1-(1-p)**2*(1-A)
    AND = lambda p: p*(1-(1-p)*(1-B))
    p1, p0 = l1, l0
    for h in range(1, d+1):
        p1 = OR(p1) if h % 2 == 1 else AND(p1)
        p0 = AND(p0) if h % 2 == 1 else OR(p0)
    r1, r0 = p1, p0
    if double:
        Q1 = 1-(1-r0)**2*(1-A)**2
        Q0 = max(1-(1-r1)*(1-B*(1-(1-B)*(1-A))), 1-(1-r1)*(1-A))
    else:
        Q1 = 1-(1-r0)**2*(1-A)
        Q0 = 1-(1-r1)*(1-B)
    return Q0, Q1
def feasible(e, d, margin, delta, **kw):
    A = e*(varpi(e)+delta); B = e*(1-varpi(e)+delta); t0 = t1 = 1e-9
    for _ in range(4000):
        Q0, Q1 = cell(A, B, t0, t1, d, **kw)
        n0, n1 = margin*Q0, margin*Q1
        if n0 >= 0.49 or n1 >= 0.49: return False
        if abs(n0-t0) < 1e-13 and abs(n1-t1) < 1e-13: return True
        t0, t1 = n0, n1
    return True
def maxeps(d, margin, delta, **kw):
    lo, hi = 0.0, 0.5
    for _ in range(40):
        e = (lo+hi)/2
        if feasible(e, d, margin, delta, **kw): lo = e
        else: hi = e
    return lo
if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    for delta in (0.012, 0.004):
        for margin in (1.05, 1.2):
            for name, kw in (("plain", dict(pair=False, robust=False, double=False)),
                             ("robust zero", dict(pair=False, robust=True, double=False)),
                             ("all three", dict(pair=True, robust=True, double=True))):
                print("delta=%.3f margin=%.2f %-12s " % (delta, margin, name) +
                      "  ".join("d=%d: %.4f" % (d, maxeps(d, margin, delta, **kw)) for d in (15, 21, 25, 31, 41)))

# ---------------------------------------------------------------------------------------------
# The numbers quoted in Remarks 37 and 50 of the note (run: python3 -I tree2_explore.py note).
def cell_nodes(A, B, w0, w1, d, nodes="single", refined=True):
    """Refined cells (refined=True) or the cells of Section 4 (refined=False);
    nodes: 'single', 'triple' (three gates at every node), 'triple0' (only at nodes whose
    children should hold 1, i.e. nodes that should output 0)."""
    if refined:
        zf = (1-(1-B)*(1-A))*B
        l0 = 1-(1-w0)**2*(1-zf)*(1-A)**2
        l1 = (1-(1-w1)*(1-B))*(1-(1-w1)*(1-B)*(1-A))
    else:
        l0 = 1-(1-w0)*(1-B)*(1-A)
        l1 = 1-(1-w1)*(1-B)
    OR1 = lambda p: 1-(1-p)**2*(1-A)
    AND1 = lambda p: p*(1-(1-p)*(1-B))
    OR3 = lambda p: 1-(1-p)**2*(1-A)**2
    AND3 = lambda p: p*(1-(1-p)*(1-A)) + (1-p)*B*(1-(1-p*B)*(1-A))
    OR = OR3 if nodes == "triple" else OR1
    AND = AND3 if nodes in ("triple", "triple0") else AND1
    p1, p0 = l1, l0
    for h in range(1, d+1):
        p1 = OR(p1) if h % 2 == 1 else AND(p1)
        p0 = AND(p0) if h % 2 == 1 else OR(p0)
    r1, r0 = p1, p0
    if refined:
        Q1 = 1-(1-r0)**2*(1-A)**2
        Q0 = max(1-(1-r1)*(1-B*(1-(1-B)*(1-A))), 1-(1-r1)*(1-A))
    else:
        Q1 = 1-(1-r0)**2*(1-A)
        Q0 = 1-(1-r1)*(1-B)
    return Q0, Q1
def threshold(d, margin, flags, **kw):
    """flags(e) -> (A, B).  Largest eps at which iterating t <- margin*Q(t) converges below 0.49."""
    def ok(e):
        A, B = flags(e); t0 = t1 = 1e-9
        for _ in range(6000):
            Q0, Q1 = cell_nodes(A, B, t0, t1, d, **kw)
            n0, n1 = margin*Q0, margin*Q1
            if n0 >= 0.49 or n1 >= 0.49: return False
            if abs(n0-t0) < 1e-14 and abs(n1-t1) < 1e-14: return True
            t0, t1 = n0, n1
        return True
    lo, hi = 0.0, 0.6
    for _ in range(40):
        mid = (lo+hi)/2
        if ok(mid): lo = mid
        else: hi = mid
    return lo
if __name__ == "__main__" and len(__import__('sys').argv) > 1 and __import__('sys').argv[1] == "note":
    proof = lambda e: (e*(varpi(e)+1/400), e*(1-varpi(e)+1/400))      # the flags of Section 5
    exact = lambda e: (e*varpi(e), e*(1-varpi(e)))                     # exact misread probabilities
    old = lambda e: (e*0.17, e)                                        # the flags of Section 4
    print("Remark 50(ii): 20% margin, density error 1/400:", ", ".join("d=%d %.4f" % (d, threshold(d, 1.2, proof)) for d in (21, 25, 31)))
    print("Remark 50(ii): no margin, density error 1/400:   d=41 %.4f" % threshold(41, 1.0001, proof))
    print("Remark 50(ii): exact probabilities, no margin:  ", ", ".join("d=%d %.4f" % (d, threshold(d, 1.0001, exact)) for d in (31, 61)))
    print("Remark 50(ii): triple gates only at nodes that should hold 0 (exact, no margin, d=61): %.4f" % threshold(61, 1.0001, exact, nodes="triple0"))
    print("Remark 50(ii): triple gates at every node (exact, no margin):", ", ".join("d=%d %.4f" % (d, threshold(d, 1.0001, exact, nodes="triple")) for d in (31, 61)))
    print("Remark 50(i): d=25, 20%% margin: Section 4 cells and flags %.4f; Section 4 cells, Section 5 flags %.4f; refined cells %.4f"
          % (threshold(25, 1.2, old, refined=False), threshold(25, 1.2, proof, refined=False), threshold(25, 1.2, proof)))
    print("Remark 37: Section 4 cells, exact probabilities, no margin:", ", ".join("d=%d %.4f" % (d, threshold(d, 1.0001, exact, refined=False)) for d in (31, 61, 101)))
