import itertools
import random

import pytest

from locals.core import (
    Status,
    brute_force_counts,
    branch_kernel,
    capped_traverse,
    pair_budget,
    path_count_matrix,
    path_mass,
    rank_one_delete,
    rank_one_insert,
    rank_path,
    reaches_via_intervals,
    root_budget,
    serial_diamonds,
    st_connectivity_sf,
    st_connectivity_sf_capped,
    through_counts,
    unfolding_occurrences,
    unrank_path,
    universal_source_lift,
)


def random_graph(n, p, rng, acyclic=True, multi=False):
    out = [[] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if acyclic and v <= u:
                continue
            if rng.random() < p:
                out[u].append(v)
                if multi and rng.random() < 0.2:
                    out[u].append(v)
    return out


def true_counts(out):
    """Exact N(a, v): DP walk counts, infinite when a cycle lies between a and v."""
    n = len(out)
    reach = [[False] * n for _ in range(n)]
    for a in range(n):
        reach[a][a] = True
        stack = [a]
        while stack:
            u = stack.pop()
            for v in out[u]:
                if not reach[a][v]:
                    reach[a][v] = True
                    stack.append(v)
    on_cycle = [any(reach[w][v] for w in out[v]) for v in range(n)]
    C = [[0] * n for _ in range(n)]
    for a in range(n):
        cur = [0] * n
        cur[a] = 1
        tot = cur[:]
        for _ in range(n):
            nxt = [0] * n
            for u in range(n):
                if cur[u]:
                    for v in out[u]:
                        nxt[v] += cur[u]
            cur = nxt
            tot = [x + y for x, y in zip(tot, cur)]
        for v in range(n):
            inf = any(on_cycle[c] and reach[a][c] and reach[c][v] for c in range(n))
            C[a][v] = float("inf") if inf else tot[v]
    return C


def reference_st_connectivity_sf(out, s, t, k):
    C = true_counts(out)
    n = len(out)
    return all(C[u][v] <= k for u in range(n) for v in range(n)) and C[s][t] >= 1


def test_cycle_tightness_remark_4_4():
    for B in range(1, 8):
        cycle = [[(i + 1) % B] for i in range(B)]
        r = root_budget(cycle, 0, [1] * B, [0])
        assert not r.accept
        # Depth B-1 has each endpoint once; node B+1 closes the tour.
        t = capped_traverse(cycle, 0, B)
        assert t.status is Status.GLOBAL_OVERFLOW and t.total == B + 1


def test_serial_diamonds_count():
    for r in range(1, 9):
        out, s, t = serial_diamonds(r)
        C = path_count_matrix(out)
        assert C[s][t] == 2**r


@pytest.mark.parametrize("seed", range(40))
def test_st_connectivity_sf_matches_reference(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 6)
    out = random_graph(n, rng.random() * 0.7, rng, acyclic=rng.random() < 0.7, multi=rng.random() < 0.3)
    for k in range(0, 5):
        s, t = rng.randrange(n), rng.randrange(n)
        want = reference_st_connectivity_sf(out, s, t, k)
        assert st_connectivity_sf(out, s, t, k) == want
        assert st_connectivity_sf_capped(out, s, t, k) == want


@pytest.mark.parametrize("seed", range(30))
def test_root_budget_counts_and_faithful_recount(seed):
    rng = random.Random(100 + seed)
    n = rng.randint(2, 7)
    out = random_graph(n, 0.4, rng, acyclic=rng.random() < 0.6, multi=True)
    kappa = [rng.randint(0, 4) for _ in range(n)]
    acc = [rng.randrange(n)]
    fast = root_budget(out, 0, kappa, acc)
    slow = root_budget(out, 0, kappa, acc, faithful_recount=True)
    assert fast.accept == slow.accept
    C = true_counts(out)
    want = all(C[0][v] <= kappa[v] for v in range(n)) and C[0][acc[0]] >= 1 and sum(kappa) > 0
    assert fast.accept == want
    if fast.accept:
        for v in range(n):
            assert fast.counts.get(v, 0) == C[0][v]
    if fast.witness is not None and fast.reason != "no accepting vertex reached":
        assert C[0][fast.witness] > kappa[fast.witness]


def test_universal_source_bijection():
    rng = random.Random(7)
    out = random_graph(5, 0.5, rng, multi=True)
    lifted, rho = universal_source_lift(out)
    n = len(out)
    C = path_count_matrix(out)
    counts = brute_force_counts(lifted, rho, n + 2)
    for a in range(n):
        for v in range(n):
            assert counts.get(a * n + v, 0) == C[a][v]


@pytest.mark.parametrize("seed", range(20))
def test_rank_unrank_and_interval_mask(seed):
    rng = random.Random(200 + seed)
    n = rng.randint(1, 7)
    out = random_graph(n, 0.45, rng, multi=rng.random() < 0.5)
    mass = path_mass(out)
    C = path_count_matrix(out)
    for v in range(n):
        assert mass[v] == sum(C[v])
        seen = set()
        for r in range(1, mass[v] + 1):
            end, path = unrank_path(out, mass, v, r)
            assert rank_path(out, mass, v, path) == r
            seen.add(tuple(path))
        assert len(seen) == mass[v]
    # DAG mask rule over unfolding occurrences, rooted at a fresh universal root.
    root = n
    rooted = [list(e) for e in out] + [list(range(n))]
    rmass = path_mass(rooted)
    occ = unfolding_occurrences(rooted, root, rmass)
    for key, q in itertools.product(range(n), repeat=2):
        assert reaches_via_intervals(occ, rmass, key, q) == (C[key][q] >= 1)
    total_occ = sum(len(v) for v in occ.values())
    assert total_occ == rmass[root]


@pytest.mark.parametrize("seed", range(25))
def test_rank_one_updates(seed):
    rng = random.Random(300 + seed)
    n = rng.randint(2, 7)
    out = random_graph(n, 0.3, rng)
    C = path_count_matrix(out)
    x, y = rng.randrange(n), rng.randrange(n)
    ok, C2 = rank_one_insert(C, x, y)
    new = [list(e) for e in out]
    new[x].append(y)
    try:
        want = path_count_matrix(new)
    except ValueError:
        want = None
    if want is None:
        assert not ok
    else:
        assert ok and C2 == want
        assert rank_one_delete(C2, x, y) == C


def test_through_counts():
    out, s, t = serial_diamonds(3)
    C = path_count_matrix(out)
    tc = through_counts(C, s, t)
    assert tc[s] == tc[t] == 8
    assert tc[1] == 4  # first-diamond branch vertex lies on half the paths


@pytest.mark.parametrize("seed", range(30))
def test_branch_kernel_bounds(seed):
    rng = random.Random(400 + seed)
    n = rng.randint(2, 9)
    out = random_graph(n, 0.35, rng)
    # Subdivide some edges to create unary chains.
    extra = []
    for u in range(n):
        new = []
        for v in out[u]:
            if rng.random() < 0.5:
                w = n + len(extra)
                extra.append([v])
                new.append(w)
            else:
                new.append(v)
        out[u] = new
    out += extra
    s, t = 0, n - 1
    C = path_count_matrix(out)
    r = C[s][t]
    if r == 0 or s == t:
        return
    verts, edges = branch_kernel(out, s, t)
    # Kernel preserves the number of terminal paths.
    kout = {v: [] for v in verts}
    for e in edges:
        kout[e.tail].append(e.head)
    idx = {v: i for i, v in enumerate(verts)}
    kadj = [[idx[h] for h in kout[v]] for v in verts]
    assert path_count_matrix(kadj)[idx[s]][idx[t]] == r
    c = len(edges) - len(verts) + 1
    assert len(verts) <= 2 * c + 2 <= 2 * r + 2
    assert len(edges) <= 3 * c + 1
    assert c <= r - 1


def test_pair_budget_nonuniform():
    out, s, t = serial_diamonds(2)
    n = len(out)
    C = path_count_matrix(out)
    kappa = [[C[a][v] for v in range(n)] for a in range(n)]
    assert pair_budget(out, kappa, s, t).accept
    kappa[s][t] -= 1
    assert not pair_budget(out, kappa, s, t).accept
