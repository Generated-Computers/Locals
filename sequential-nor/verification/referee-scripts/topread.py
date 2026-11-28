# Task 4 (d): the top-reading replacement, on exhaustively enumerated small instances (every base, every noise),
# and on random larger ones: for EVERY dependency chain, replacing k_s (at steps with pi_{k_s}=0) by the largest
# k in [k_s, P_{s-1}] with pi_k=0 and j_k=P_s gives a valid dependency chain with the same positions that is top-reading.
import sys, itertools, random

def analyse(n, j, pi, rho):
    jt = [0]*(n+1); x = [1]*(n+1); rs = [False]*(n+1)
    for i in range(1, n+1):
        jt[i] = rho[i] if pi[i] else j[i]
        x[i] = 1-(x[i-1] | x[jt[i]])
        rs[i] = (jt[i] != i-1 and x[jt[i]] == 1)
    return jt, x, rs

def valid_step(jt, rs, k, Pprev, Pnext):
    if not (1 <= k <= Pprev): return False
    if jt[k] == k-1 or jt[k] != Pnext: return False
    return not any(rs[g] for g in range(k+1, Pprev+1))

def check_instance(n, j, pi, rho, stats):
    jt, x, rs = analyse(n, j, pi, rho)
    # nores[k][P] : no reset in (k, P]
    def chains(Pprev, acc):
        yield acc
        for k in range(Pprev, 0, -1):
            if jt[k] != k-1:
                yield from chains(jt[k], acc + [(k, Pprev, jt[k])])
            if rs[k]: break                  # gates below a reset in (.,Pprev] cannot be used: the reset k itself may be used, lower ones not
    for P0 in range(1, n+1):
        for ch in chains(P0, []):
            if not ch: continue
            stats[0] += 1
            changed = False
            for (k, Pprev, Pnext) in ch:
                assert valid_step(jt, rs, k, Pprev, Pnext)
                k2 = k
                if pi[k] == 0:
                    k2 = max(g for g in range(k, Pprev+1) if pi[g] == 0 and j[g] == Pnext)
                    if k2 != k: changed = True
                    # top-reading at the new gate
                    assert not any(pi[g] == 0 and j[g] == Pnext for g in range(k2+1, Pprev+1))
                assert valid_step(jt, rs, k2, Pprev, Pnext), ("replacement invalid", n, j, pi, rho, ch)
            if changed: stats[1] += 1

def all_instances(n):
    ranges = [range(i) for i in range(1, n+1)]
    for jtail in itertools.product(*ranges):
        j = [0]+list(jtail)
        # noise: per gate either pi=0 or (pi=1, rho)
        opts = [[(0, 0)] + [(1, p) for p in range(i)] for i in range(1, n+1)]
        for nz in itertools.product(*opts):
            pi = [0]+[a for a, b in nz]; rho = [0]+[b for a, b in nz]
            yield j, pi, rho

if __name__ == "__main__":
    nmax = int(sys.argv[1]); stats = [0, 0]; ninst = 0
    for n in range(2, nmax+1):
        for j, pi, rho in all_instances(n):
            check_instance(n, j, pi, rho, stats); ninst += 1
        print("n=%d exhaustive (all bases x all noise): instances so far %d, chains checked %d, chains changed by the replacement %d" % (n, ninst, stats[0], stats[1]), flush=True)
    rnd = random.Random(5); stats = [0, 0]
    for trial in range(40000):
        n = rnd.randint(7, 14)
        kind = rnd.random()
        if kind < 0.4: j = [0]+[rnd.randrange(i) for i in range(1, n+1)]
        elif kind < 0.7: j = [0]+[i//2 for i in range(1, n+1)]
        else:
            p = rnd.randint(1, 3); j = [0]+[(p if i >= p+2 and rnd.random() < 0.7 else i-1) for i in range(1, n+1)]
        eps = rnd.choice((0.2, 0.5, 0.8))
        pi = [0]+[1 if rnd.random() < eps else 0 for i in range(1, n+1)]; rho = [0]+[rnd.randrange(i) for i in range(1, n+1)]
        check_instance(n, j, pi, rho, stats)
    print("random instances n=7..14 (arbitrary, floor(i/2), block bases): chains checked %d, changed %d: all replacements valid, same positions, top-reading" % (stats[0], stats[1]))
