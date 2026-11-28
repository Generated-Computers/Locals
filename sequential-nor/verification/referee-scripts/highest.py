# Task 5 (b): Lemma highest, exhaustively over all parent arrays of length n, every g_0 in 1..3, every P_0 and every l.
#  - monotonicity: q < t, no reset in (q,t]  =>  delta(t) >= delta(q)
#  - the chain built by the lemma's rule exists, is an admissible dependency chain of length l, and has both properties
#  - independently of the construction: among ALL admissible chains of length l from P_0, at least one has both properties
import sys, itertools

def check_array(n, j, g0s, stats, brute):
    x = [1]*(n+1); rs = [False]*(n+1)
    for i in range(1, n+1):
        x[i] = 1-(x[i-1] | x[j[i]]); rs[i] = (j[i] != i-1 and x[j[i]] == 1)
    # nores[k][P]: no reset in (k, P]
    last_reset_le = [0]*(n+1); lr = 0
    for i in range(0, n+1):
        if i >= 1 and rs[i]: lr = i
        last_reset_le[i] = lr                 # largest reset <= i (0 if none)
    def nores(a, b): return last_reset_le[b] <= a       # no reset in (a, b]
    for g0 in g0s:
        delta = [-1]*(n+1)
        for q in range(g0, n+1):
            best = 0
            for k in range(q, 0, -1):
                if not nores(k, q): break
                p = j[k]
                if p <= k-2 and p >= g0: best = max(best, 1+delta[p])
            delta[q] = best
        # monotonicity along reset-free stretches
        for q in range(g0, n+1):
            for t in range(q+1, n+1):
                if nores(q, t):
                    stats['mono'] += 1
                    assert delta[t] >= delta[q], ("monotonicity fails", j, g0, q, t)
        def propA(Pprev, Ps):
            for k in range(Pprev, 0, -1):
                if not nores(k, Pprev): break
                t = j[k]
                if Ps < t <= k-2 and nores(Ps, t): return False
            return True
        def propB(Pprev, Ps, ks):
            for k in range(Pprev, ks, -1):
                if not nores(k, Pprev): break
                if j[k] == Ps: return False
            return True
        def valid(Pprev, k, Ps):
            return 1 <= k <= Pprev and j[k] == Ps and Ps <= k-2 and Ps >= g0 and nores(k, Pprev)
        for P0 in range(g0, n+1):
            for l in range(1, delta[P0]+1):
                stats['pairs'] += 1
                # the lemma's construction
                P = P0; chain = []
                for s in range(1, l+1):
                    assert delta[P] >= l-s+1
                    cands = []
                    for k in range(P, 0, -1):
                        if not nores(k, P): break
                        q = j[k]
                        if g0 <= q <= k-2 and delta[q] >= l-s: cands.append((q, k))
                    assert cands, ("candidate set empty", j, g0, P0, l, s)
                    q, k = max(cands)
                    assert valid(P, k, q)
                    assert propA(P, q), ("property A fails", j, g0, P0, l, s, k, q)
                    assert propB(P, q, k), ("property B fails", j, g0, P0, l, s, k, q)
                    chain.append((k, q)); P = q
                if brute:
                    # all admissible chains of length l from P0: how many have both properties?
                    def rec(Pprev, left):
                        if left == 0: yield True; return
                        for k in range(Pprev, 0, -1):
                            if not nores(k, Pprev): break
                            q = j[k]
                            if g0 <= q <= k-2:
                                ok = propA(Pprev, q) and propB(Pprev, q, k)
                                for sub in rec(q, left-1): yield ok and sub
                    res = list(rec(P0, l)); assert res
                    stats['chains'] += len(res); stats['good'] += sum(res)
                    assert any(res), ("no chain with both properties", j, g0, P0, l)

if __name__ == "__main__":
    nmax = int(sys.argv[1]); stats = {'mono': 0, 'pairs': 0, 'chains': 0, 'good': 0}; narr = 0
    for n in range(3, nmax+1):
        for jt in itertools.product(*[range(i) for i in range(1, n+1)]):
            check_array(n, [0]+list(jt), (1, 2, 3), stats, brute=(n <= 8)); narr += 1
        print("n=%d: arrays so far %d, (g0,P0,l) cases %d, monotonicity checks %d, admissible chains enumerated %d of which %d have both properties" % (n, narr, stats['pairs'], stats['mono'], stats['chains'], stats['good']), flush=True)
