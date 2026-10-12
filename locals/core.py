"""Graph primitives from "Total Strongly-Few Directed Reachability".

Each function names the result it implements. A graph is a directed multigraph
given as adjacency lists of edge occurrences: ``out[v]`` lists the heads of the
edges leaving ``v``. A head may repeat; parallel occurrences are distinct walks
(Def. 2.1). Walks may repeat vertices and the empty walk is counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

Adj = Sequence[Sequence[int]]


class Status(Enum):
    OK = 0
    GLOBAL_OVERFLOW = 1
    TARGET_OVERFLOW = 2


@dataclass
class Traversal:
    status: Status
    total: int  # unfolding nodes entered, including the empty root walk
    target_count: int
    max_stack: int
    counts: Optional[Dict[int, int]] = None  # endpoint counts, when requested
    witness: Optional[int] = None  # an over-budget endpoint (Cor. 4.3)


def capped_traverse(
    out: Adj,
    root: int,
    budget: int,
    target: Optional[int] = None,
    target_budget: int = 0,
    collect_counts: bool = False,
) -> Traversal:
    """FULL-TRAVERSE (Listing 3, Thm 4.5).

    Depth-first traversal of the complete, untruncated walk unfolding from
    ``root`` with an explicit frame stack. It stops on entering unfolding node
    ``budget + 1``, so it terminates within ``budget + 1`` node entries on every
    input, including cyclic ones. With ``collect_counts`` the traversal also
    accumulates every endpoint count (Lemma 3.2), which replaces the per-target
    recount passes once the global pass has completed.
    """
    counts: Dict[int, int] = {root: 1} if collect_counts else {}
    total = 1
    tc = 1 if target == root else 0
    if total > budget:
        return Traversal(Status.GLOBAL_OVERFLOW, total, tc, 0, counts or None, root)
    if target is not None and tc > target_budget:
        return Traversal(Status.TARGET_OVERFLOW, total, tc, 0, counts or None, target)
    stack: List[Tuple[int, int]] = []
    cur, nxt = root, 0
    max_stack = 0
    while True:
        edges = out[cur]
        if nxt >= len(edges):
            if not stack:
                return Traversal(Status.OK, total, tc, max_stack, counts or None)
            cur, nxt = stack.pop()
            continue
        head = edges[nxt]
        stack.append((cur, nxt + 1))
        if len(stack) > max_stack:
            max_stack = len(stack)
        cur, nxt = head, 0
        total += 1
        if collect_counts:
            counts[cur] = counts.get(cur, 0) + 1
        if total == budget + 1:
            return Traversal(Status.GLOBAL_OVERFLOW, total, tc, max_stack, counts or None, cur)
        if target is not None and cur == target:
            tc += 1
            if tc == target_budget + 1:
                return Traversal(Status.TARGET_OVERFLOW, total, tc, max_stack, counts or None, cur)


@dataclass
class BudgetResult:
    accept: bool
    reason: str
    counts: Optional[Dict[int, int]] = None
    witness: Optional[int] = None
    node_entries: int = 0


def root_budget(
    out: Adj,
    root: int,
    kappa: Sequence[int],
    accepting: Sequence[int],
    faithful_recount: bool = False,
) -> BudgetResult:
    """RootBudget decider (Def. 4.1, Thm 4.2, Thm 4.5).

    Accepts iff N(root, q) <= kappa[q] for every q and some accepting vertex is
    reachable. One capped global pass at B = sum(kappa); when it completes, the
    unfolding is finite with at most B nodes and every count is exact. With
    ``faithful_recount`` the per-vertex recounts of the proof are run literally.
    """
    B = sum(kappa)
    if B == 0:
        return BudgetResult(False, "B=0")
    g = capped_traverse(out, root, B, collect_counts=True)
    if g.status is not Status.OK:
        # Pigeonhole: more than B = sum(kappa) nodes, so some endpoint is over.
        counts = g.counts or {}
        w = next(v for v, c in counts.items() if c > kappa[v])
        return BudgetResult(False, "global overflow", counts, w, g.total)
    counts = g.counts or {}
    entries = g.total
    if faithful_recount:
        counts = {}
        for q in range(len(out)):
            r = capped_traverse(out, root, B, target=q, target_budget=kappa[q])
            entries += r.total
            if r.status is not Status.OK:
                return BudgetResult(False, "target overflow", None, q, entries)
            if r.target_count:
                counts[q] = r.target_count
    for v, c in counts.items():
        if c > kappa[v]:
            return BudgetResult(False, "target overflow", counts, v, entries)
    if not any(counts.get(f, 0) >= 1 for f in accepting):
        return BudgetResult(False, "no accepting vertex reached", counts, None, entries)
    return BudgetResult(True, "ok", counts, None, entries)


def universal_source_lift(out: Adj) -> Tuple[List[List[int]], int]:
    """Construction 4.6. Vertex (a, v) is ``a * n + v``; the new root is n*n.

    N_U(rho, (a, v)) = N_G(a, v) (Lemma 4.7), so all-pairs budgets become one
    rooted row.
    """
    n = len(out)
    rho = n * n
    lifted: List[List[int]] = [[] for _ in range(n * n + 1)]
    for a in range(n):
        base = a * n
        for u in range(n):
            lifted[base + u] = [base + v for v in out[u]]
    lifted[rho] = [a * n + a for a in range(n)]
    return lifted, rho


def pair_budget(out: Adj, kappa: Sequence[Sequence[int]], s: int, t: int) -> BudgetResult:
    """PairBudget via the universal-source lift (Def. 4.8, Thm 4.9)."""
    n = len(out)
    lifted, rho = universal_source_lift(out)
    flat = [kappa[a][v] for a in range(n) for v in range(n)] + [1]
    return root_budget(lifted, rho, flat, [s * n + t])


def locals_sf(out: Adj, s: int, t: int, k: int) -> bool:
    """Membership in LOCALSsf (Def. 2.3) via the lift, total on every input."""
    if k <= 0 or len(out) == 0:
        return False
    n = len(out)
    return pair_budget(out, [[k] * n for _ in range(n)], s, t).accept


def locals_sf_capped(out: Adj, s: int, t: int, k: int) -> bool:
    """DECIDE-LOCALS-sf (Listings 1 and 2, Thm 3.8): depth cap nk, node cap nk."""
    n = len(out)
    if k <= 0 or n == 0:
        return False
    cap = n * k
    reached = False
    for a in range(n):
        counts = _depth_capped_counts(out, a, cap, cap)
        if counts is None:
            return False
        if any(c > k for c in counts.values()):
            return False
        if a == s and counts.get(t, 0) >= 1:
            reached = True
    return reached


def _depth_capped_counts(out: Adj, a: int, depth_cap: int, node_cap: int) -> Optional[Dict[int, int]]:
    counts = {a: 1}
    total = 1
    stack: List[Tuple[int, int, int]] = []
    cur, nxt, depth = a, 0, 0
    while True:
        if depth == depth_cap or nxt >= len(out[cur]):
            if not stack:
                return counts
            cur, nxt, depth = stack.pop()
            continue
        head = out[cur][nxt]
        stack.append((cur, nxt + 1, depth))
        cur, nxt, depth = head, 0, depth + 1
        total += 1
        if total == node_cap + 1:
            return None
        counts[cur] = counts.get(cur, 0) + 1


def brute_force_counts(out: Adj, a: int, max_len: int) -> Dict[int, int]:
    """Reference N^{<=max_len}(a, .) by explicit walk enumeration."""
    counts: Dict[int, int] = {}

    def rec(v: int, d: int) -> None:
        counts[v] = counts.get(v, 0) + 1
        if d < max_len:
            for w in out[v]:
                rec(w, d + 1)

    rec(a, 0)
    return counts


def topological_order(out: Adj) -> Optional[List[int]]:
    n = len(out)
    indeg = [0] * n
    for u in range(n):
        for v in out[u]:
            indeg[v] += 1
    order = [v for v in range(n) if indeg[v] == 0]
    i = 0
    while i < len(order):
        u = order[i]
        i += 1
        for v in out[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                order.append(v)
    return order if len(order) == n else None


def path_count_matrix(out: Adj) -> List[List[int]]:
    """C = I + AC = (I - A)^{-1} on a DAG (Thm A3.1); raises on a cycle."""
    order = topological_order(out)
    if order is None:
        raise ValueError("graph has a directed cycle")
    n = len(out)
    C = [[0] * n for _ in range(n)]
    for u in reversed(order):
        row = C[u]
        row[u] = 1
        for v in out[u]:
            rv = C[v]
            for x in range(n):
                if rv[x]:
                    row[x] += rv[x]
    return C


def path_mass(out: Adj) -> List[int]:
    """M(v) = 1 + sum_{v->u} M(u) (Thm A25.1); raises on a cycle."""
    order = topological_order(out)
    if order is None:
        raise ValueError("graph has a directed cycle")
    m = [0] * len(out)
    for v in reversed(order):
        m[v] = 1 + sum(m[u] for u in out[v])
    return m


def rank_path(out: Adj, mass: Sequence[int], v: int, edge_indices: Sequence[int]) -> int:
    """rank_v (Thm A27.1): the path from v choosing these edge indices."""
    r = 1
    for i in edge_indices:
        r += 1 + sum(mass[u] for u in out[v][:i])
        v = out[v][i]
    return r


def unrank_path(out: Adj, mass: Sequence[int], v: int, r: int) -> Tuple[int, List[int]]:
    """Stackless unranking (Listing 4). Returns (endpoint, edge indices)."""
    if not 1 <= r <= mass[v]:
        raise ValueError("rank out of range")
    path: List[int] = []
    while r != 1:
        r -= 1
        for i, u in enumerate(out[v]):
            if r > mass[u]:
                r -= mass[u]
            else:
                path.append(i)
                v = u
                break
    return v, path


def unfolding_occurrences(out: Adj, root: int, mass: Optional[Sequence[int]] = None) -> Dict[int, List[int]]:
    """Preorder ranks of every occurrence of every vertex in the unfolding T_root.

    The occurrence with rank r ending at v has subtree interval [r, r + M(v)),
    because rank blocks are consecutive of size M (Thm A27.1).
    """
    mass = mass or path_mass(out)
    occ: Dict[int, List[int]] = {}
    stack = [(root, 1)]
    while stack:
        v, r = stack.pop()
        occ.setdefault(v, []).append(r)
        nxt = r + 1
        children = []
        for u in out[v]:
            children.append((u, nxt))
            nxt += mass[u]
        stack.extend(reversed(children))
    for v in occ:
        occ[v].sort()
    return occ


def reaches_via_intervals(occ: Dict[int, List[int]], mass: Sequence[int], key: int, query: int) -> bool:
    """key ~> query iff some occurrence of query lies in the subtree interval of
    some occurrence of key in the unfolding (the DAG mask rule)."""
    for rk in occ.get(key, ()):
        hi = rk + mass[key]
        for rq in occ.get(query, ()):
            if rk <= rq < hi:
                return True
    return False


def rank_one_insert(C: List[List[int]], x: int, y: int, k: Optional[int] = None) -> Tuple[bool, List[List[int]]]:
    """Thm A13.1: inserting x -> y gives C' = C + C[:, x] C[y, :].

    Rejects when C[y][x] > 0 (a cycle) or when a new count exceeds k.
    """
    n = len(C)
    if C[y][x] > 0:
        return False, C
    col = [C[u][x] for u in range(n)]
    row = C[y]
    new = [[C[u][v] + col[u] * row[v] for v in range(n)] for u in range(n)]
    if k is not None and any(val > k for r in new for val in r):
        return False, C
    return True, new


def rank_one_delete(C: List[List[int]], x: int, y: int) -> List[List[int]]:
    """Section A13.3: deleting one occurrence of x -> y."""
    n = len(C)
    col = [C[u][x] for u in range(n)]
    row = C[y]
    return [[C[u][v] - col[u] * row[v] for v in range(n)] for u in range(n)]


def through_counts(C: Sequence[Sequence[int]], s: int, t: int) -> List[int]:
    """Section A13.1: number of s-t paths through each vertex, C_sv * C_vt."""
    return [C[s][v] * C[v][t] for v in range(len(C))]


@dataclass
class KernelEdge:
    tail: int
    head: int
    chain: List[int] = field(default_factory=list)  # suppressed interior vertices


def live_interval(out: Adj, s: int, t: int) -> Tuple[List[bool], List[bool]]:
    """Forward cone of s and backward cone of t."""
    n = len(out)
    fwd = [False] * n
    fwd[s] = True
    stack = [s]
    while stack:
        u = stack.pop()
        for v in out[u]:
            if not fwd[v]:
                fwd[v] = True
                stack.append(v)
    rev: List[List[int]] = [[] for _ in range(n)]
    for u in range(n):
        for v in out[u]:
            rev[v].append(u)
    bwd = [False] * n
    bwd[t] = True
    stack = [t]
    while stack:
        u = stack.pop()
        for v in rev[u]:
            if not bwd[v]:
                bwd[v] = True
                stack.append(v)
    return fwd, bwd


def branch_kernel(out: Adj, s: int, t: int) -> Tuple[List[int], List[KernelEdge]]:
    """Thm A10.1: suppress internal 1-in/1-out vertices of the live s-t DAG.

    Returns the retained vertices and kernel edges; each kernel edge carries
    the chain of suppressed vertices it replaces.
    """
    fwd, bwd = live_interval(out, s, t)
    live = [f and b for f, b in zip(fwd, bwd)]
    n = len(out)
    lout = [[v for v in out[u] if live[v]] if live[u] else [] for u in range(n)]
    indeg = [0] * n
    for u in range(n):
        for v in lout[u]:
            indeg[v] += 1
    retained = [
        v for v in range(n)
        if live[v] and (v in (s, t) or indeg[v] != 1 or len(lout[v]) != 1)
    ]
    keep = set(retained)
    edges: List[KernelEdge] = []
    for u in retained:
        for v in lout[u]:
            chain: List[int] = []
            while v not in keep:
                chain.append(v)
                v = lout[v][0]
            edges.append(KernelEdge(u, v, chain))
    return retained, edges


def serial_diamonds(r: int) -> Tuple[List[List[int]], int, int]:
    """D_r of Section A6.1: 2^r paths from z_0 to z_r."""
    out: List[List[int]] = [[] for _ in range(3 * r + 1)]
    for i in range(r):
        z0, a, b, z1 = 3 * i, 3 * i + 1, 3 * i + 2, 3 * i + 3
        out[z0] += [a, b]
        out[a].append(z1)
        out[b].append(z1)
    return out, 0, 3 * r
