# greedy_steps_check.py: checks Lemma "greedy steps and the pruned depth" on random parent arrays.
# For a parent array j it computes the trace, the pruned depth D by the recursion of the switch
# lemma, the steps s(i) of the parallel greedy algorithm by simulating it on the graph with the
# edges {i-1,i} and {j_i,i}, and checks that the greedy set is {i: x_i = 1}, the recursion for s,
# and D(i) <= 2 s(i).      usage: python3 -I greedy_steps_check.py [trials]
import random, sys
def run(n, rnd, mode):
    j = [0]*(n+1)
    for i in range(1, n+1):
        if mode == 0: j[i] = rnd.randrange(i)
        elif mode == 1: j[i] = max(i-3, 0) if rnd.random() < 0.6 else rnd.randrange(i)
        else: j[i] = i-1 if rnd.random() < 0.3 else rnd.randrange(i)
    x = [1]+[0]*n; D = [0]*(n+1)
    for i in range(1, n+1):
        a, b, p = x[i-1], x[j[i]], j[i]
        x[i] = 0 if (a or b) else 1
        if i == 1: D[i] = 1
        elif p == i-1: D[i] = D[i-1]
        elif not a and b: D[i] = 1+D[p]
        elif a and b: D[i] = min(D[i-1], 1+D[p])
        elif not a and not b: D[i] = max(D[i-1], 1+D[p])
        else: D[i] = D[i-1]
    nb = [set() for _ in range(n+1)]
    for i in range(1, n+1):
        nb[i].add(i-1); nb[i-1].add(i); nb[i].add(j[i]); nb[j[i]].add(i)
    alive = set(range(n+1)); s = [0]*(n+1); inset = [0]*(n+1); step = 0
    while alive:
        step += 1
        roots = [v for v in alive if not any(u < v and u in alive for u in nb[v])]
        gone = set(roots)
        for v in roots:
            inset[v] = 1
            gone |= {u for u in nb[v] if u in alive}
        for v in gone: s[v] = step
        alive -= gone
    assert inset == x
    for i in range(1, n+1):
        N = {i-1, j[i]}
        if x[i]: assert s[i] == 1+max(s[u] for u in N)
        else: assert s[i] == min(s[u] for u in N if x[u])
        assert D[i] <= 2*s[i]
    return max(D), max(s)
rnd = random.Random(1); T = int(sys.argv[1]) if len(sys.argv) > 1 else 3000; worst = 0
for t in range(T):
    d, st = run(rnd.randrange(2, 60), rnd, t % 3); worst = max(worst, d/st)
print("ok:", T, "arrays; largest max D / max s =", round(worst, 3))
