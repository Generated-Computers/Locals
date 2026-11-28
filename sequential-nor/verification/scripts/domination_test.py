"""Pathwise test of the domination step for the NOR-tree embedding (my own code).
For every node of every cell: if its output differs from the intended value, the event D(node) must hold,
where D is built from the weakest events the flags are guaranteed to contain:
   B-type := misread at the gate;   A-type := misread at the gate AND the listed pointer holds a 0."""
import random, sys
rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)

def run(s, d, m, eps, pre):
    # random NOR circuit; inputs: gate index >= 1, 0 = constant c0 (value 0), -1 = constant c1 (value 1)
    U, V, val = [], [], {0: 0, -1: 1}
    for t in range(1, s + 1):
        pick = lambda: rng.choice([0, -1]) if (t == 1 or rng.random() < 0.2) else rng.randrange(1, t)
        a, b = pick(), pick()
        if t > 1 and rng.random() < 0.5: a = t - 1
        U.append(a); V.append(b); val[t] = 1 - (val[a] | val[b])
    base = [0, 0] + [0] * (pre - 1)                    # position 0, gate 1, preamble of pointer-0 gates
    out = {}
    def copy_pos(w): return 1 if w == 0 else 0 if w == -1 else out[(w, rng.randrange(m))]
    def build(h, w):
        if h == 0:
            base.append(0); z = len(base) - 1
            cp = copy_pos(w); base.append(cp); l = len(base) - 1
            return {'k': 'leaf', 'z': z, 'l': l, 'cp': cp, 'b': val[w], 'out': l, 'want': 1 - val[w]}
        L = build(h - 1, w); R = build(h - 1, w)
        base.append(L['out']); g = len(base) - 1
        return {'k': 'node', 'L': L, 'R': R, 'g': g, 'c': L['want'], 'out': g, 'want': 1 - L['want']}
    cells = []
    for t in range(1, s + 1):
        for l in range(m):
            Tu = build(d, U[t - 1]); Tv = build(d, V[t - 1])
            base.append(Tu['out']); g = len(base) - 1
            assert Tu['want'] == val[U[t - 1]] and Tv['want'] == val[V[t - 1]]
            out[(t, l)] = g
            cells.append({'k': 'top', 'L': Tu, 'R': Tv, 'g': g, 'bu': Tu['want'], 'bv': Tv['want'], 'out': g, 'want': val[t]})
    n = len(base) - 1
    ptr = base[:]
    for i in range(1, n + 1):
        if rng.random() < eps: ptr[i] = rng.randrange(i)
    x = [1] + [0] * n
    for i in range(1, n + 1): x[i] = 1 - (x[i - 1] | x[ptr[i]])
    mis = [False] * (n + 1)
    for i in range(1, n + 1): mis[i] = x[ptr[i]] != x[base[i]]
    A = lambda g: mis[g] and x[base[g]] == 0
    B = lambda g: mis[g]
    stats = {'nodes': 0, 'wrong': 0}
    def Dev(nd):
        if nd['k'] == 'leaf':
            bad = x[nd['cp']] != nd['b']
            Dv = (bad or B(nd['z']) or A(nd['l'])) if nd['b'] == 0 else (bad or B(nd['l']))
        elif nd['k'] == 'node':
            D1, D2 = Dev(nd['L']), Dev(nd['R'])
            Dv = (D1 or D2 or A(nd['g'])) if nd['c'] == 0 else (D2 and (D1 or B(nd['g'])))
        else:
            Du, Dw = Dev(nd['L']), Dev(nd['R']); g = nd['g']
            Dv = {(0, 0): Du or Dw or A(g), (1, 1): Dw and (Du or B(g)), (1, 0): Du or B(g), (0, 1): Dw}[(nd['bu'], nd['bv'])]
        wrong = x[nd['out']] != nd['want']
        stats['nodes'] += 1; stats['wrong'] += wrong
        assert (not wrong) or Dv, ("domination violated", nd['k'])
        return Dv
    for c in cells: Dev(c)
    if eps == 0: assert stats['wrong'] == 0
    return stats

tot = {'nodes': 0, 'wrong': 0}
for trial in range(1500):
    s = rng.randrange(2, 9); d = rng.choice([1, 3]); m = rng.randrange(1, 6)
    eps = rng.choice([0.0, 0.05, 0.15, 0.3, 0.5, 1.0])
    st = run(s, d, m, eps, pre=rng.randrange(1, 4))
    for k in tot: tot[k] += st[k]
print("nodes checked:", tot['nodes'], " wrong nodes:", tot['wrong'], " domination violations: 0")
