"""Independent generic binary-clause preprocessing for a stronger CUDD control.

Only redundant literal equivalences are inferred. No original variable is
eliminated, no query is observed, and no SPECTRA index/quotient is consumed.
All original clauses still enter the BDD. This is established 2-SAT reasoning,
not a novel algorithm or an independently produced benchmark result.
"""
from __future__ import annotations

from typing import Sequence


def binary_equivalences(nvars: int, clauses: Sequence[Sequence[int]]) -> tuple[bool, tuple[tuple[int, int], ...]]:
    """Return contradiction and signed equalities entailed by a VALIDATED CNF.

    Nonbinary clauses are ignored for inference, not deleted from the formula.
    Literal +v is node 2*(v-1); -v is its xor-1 complement. Iterative Kosaraju
    avoids Python recursion limits. A unit supplies one implication. An empty
    clause is already a contradiction. Call freeze_cnf before this function.
    """
    graph = [[] for _ in range(2 * nvars)]
    reverse = [[] for _ in graph]

    def node(literal: int) -> int:
        return 2 * (abs(literal) - 1) + int(literal < 0)

    def arc(left: int, right: int) -> None:
        graph[left].append(right)
        reverse[right].append(left)

    for clause in clauses:
        if not clause:
            return True, ()
        if len(clause) == 1:
            x = node(clause[0]); arc(x ^ 1, x)
        elif len(clause) == 2:
            x, y = map(node, clause)
            arc(x ^ 1, y); arc(y ^ 1, x)
    seen = bytearray(2 * nvars)
    finish = []
    for root in range(2 * nvars):
        if seen[root]:
            continue
        seen[root] = 1
        stack = [(root, 0)]
        while stack:
            vertex, offset = stack[-1]
            if offset == len(graph[vertex]):
                finish.append(vertex); stack.pop()
                continue
            neighbor = graph[vertex][offset]
            stack[-1] = (vertex, offset + 1)
            if not seen[neighbor]:
                seen[neighbor] = 1; stack.append((neighbor, 0))
    component = [-1] * (2 * nvars)
    groups = []
    for root in reversed(finish):
        if component[root] != -1:
            continue
        group = []
        index = len(groups)
        component[root] = index
        stack = [root]
        while stack:
            vertex = stack.pop(); group.append(vertex)
            for neighbor in reverse[vertex]:
                if component[neighbor] == -1:
                    component[neighbor] = index; stack.append(neighbor)
        groups.append(group)
    if any(component[2*v] == component[2*v+1] for v in range(nvars)):
        return True, ()
    equalities = set()
    for group in groups:
        representative = min(group)
        a = representative // 2 + 1
        for other in group:
            if other == representative:
                continue
            b = other // 2 + 1
            sign = 1 if (representative & 1) == (other & 1) else -1
            # min(group) ensures a < b; the complementary component emits the
            # same canonical equality and the set removes that exact duplicate.
            equalities.add((a, sign * b))
    return False, tuple(sorted(equalities, key=lambda pair: (pair[0], abs(pair[1]), pair[1])))


def equivalence_order(permutation: Sequence[int], pairs: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    """Stably group equivalent variables without eliminating original variables.

    Both arguments come from validated CNF preprocessing. Group order and member
    order inherit the query-blind base permutation; signed equalities only affect
    grouping, not the Boolean interpretation or the returned model labels.
    """
    parent = list(range(len(permutation) + 1))

    def root(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for left, right in pairs:
        a, b = root(left), root(abs(right))
        if a != b:
            parent[b] = a
    groups: dict[int, list[int]] = {}
    for v in permutation:
        groups.setdefault(root(v), []).append(v)
    return tuple(v for group in groups.values() for v in group)
