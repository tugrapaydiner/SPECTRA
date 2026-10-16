"""Prepared ORIGINAL-input replay, independent of candidate indices and answers.

The portable reference remains proof.verify_answer. This observer retains only
validated original domains/edges. Small graphs use exact adjacency bitsets;
large graphs keep original edge pairs rather than allocating an n-squared bank.
"""
from __future__ import annotations

import itertools

from .proof import decode_proof


class OriginalReplay:
    def __init__(self, problem):
        n, k = problem.n, problem.palette
        if type(n) is not int or not 0 <= n <= 100_000 or type(k) is not int or not 1 <= k <= 64:
            raise ValueError('invalid original geometry')
        masks = tuple(problem.masks)
        if len(masks) != n or any(type(d) is not int or d <= 0 or d >> k or d.bit_count() > 2 for d in masks):
            raise ValueError('invalid original domains')
        edges = set()
        for row in problem.edges:
            u, v = row
            if any(type(x) is not int or not 0 <= x < n for x in (u, v)) or u == v:
                raise ValueError('invalid original edge')
            edges.add((min(u, v), max(u, v)))
        self.n, self.palette, self.masks = n, k, masks
        self.choices = tuple(tuple(c for c in range(k) if d >> c & 1) for d in masks)
        self.edges = tuple(sorted(edges))
        self.adjacency = None
        self.edge_set = None
        if n <= 4096:
            adjacency = [0]*n
            for u, v in self.edges:
                adjacency[u] |= 1 << v
                adjacency[v] |= 1 << u
            self.adjacency = tuple(adjacency)
        else:
            self.edge_set = frozenset(edges)
        self._implication = self._reverse = None

    def _pins(self, query):
        allowed = {}
        for row in query:
            vertex, mask = row
            if (type(vertex) is not int or not 0 <= vertex < self.n or
                    type(mask) is not int or not 0 <= mask < 1 << 64):
                raise ValueError('invalid original request')
            allowed[vertex] = allowed.get(vertex, self.masks[vertex]) & mask
        return allowed

    def _color(self, literal):
        row = self.choices[literal//2]
        return row[literal%2] if literal%2 < len(row) else None

    def _hop(self, a, b):
        u, v = a//2, b//2
        if u == v:
            return a == 2*u+1 and b == 2*u and len(self.choices[u]) == 1
        exists = ((self.adjacency[u] >> v) & 1 if self.adjacency is not None
                  else (min(u, v), max(u, v)) in self.edge_set)
        return bool(exists and self._color(a) is not None and
                    self._color(b ^ 1) is not None and self._color(a) == self._color(b ^ 1))

    def verify(self, query, status, labels, proof):
        try:
            allowed = self._pins(query)
            if type(labels) is not bytes or type(proof) is not bytes:
                return False
            if status == 'SAT_VERIFIED':
                if proof or len(labels) != self.n:
                    return False
                classes = [0]*self.palette if self.adjacency is not None else None
                for v, c in enumerate(labels):
                    if c >= self.palette or not ((allowed.get(v, self.masks[v]) >> c) & 1):
                        return False
                    if classes is not None:
                        classes[c] |= 1 << v
                if classes is not None:
                    return all(not (self.adjacency[v] & classes[c]) for v, c in enumerate(labels))
                return all(labels[u] != labels[v] for u, v in self.edges)
            if status != 'UNSAT_VERIFIED' or labels:
                return False
            kind, a, b = decode_proof(proof, self.n)
            if kind == 1:
                return a[0] < self.n and allowed.get(a[0], self.masks[a[0]]) == 0
            if (any(not 0 <= x < 2*self.n for x in a+b) or
                    len(set(a)) != len(a) or len(set(b)) != len(b)):
                return False
            if not all(self._hop(x, y) for path in (a, b) for x, y in zip(path, path[1:])):
                return False
            if kind == 3:
                return a[0] == b[-1] and a[-1] == b[0] and a[-1] == (a[0] ^ 1)
            return (a[-1] == (b[-1] ^ 1) and
                    all(self._color(v) is not None and
                        allowed.get(v//2, self.masks[v//2]) == 1 << self._color(v)
                        for v in (a[0], b[0])))
        except (ValueError, TypeError, IndexError, OverflowError):
            return False

    def feasible(self, query):
        """Separate exact SCC replay on an independently encoded original relation.

        Singleton choices are duplicated here, unlike the candidate's virtual
        unavailable literal/unit encoding. No candidate SCC/root/proof is used.
        Static adjacency is retained; current pin arcs and SCCs are freshly built.
        """
        allowed = self._pins(query)
        choices = tuple(row if len(row) == 2 else (row[0], row[0]) for row in self.choices)
        if self._implication is None:
            graph, reverse = [[] for _ in range(2*self.n)], [[] for _ in range(2*self.n)]
            for u, v in self.edges:
                for a in range(2):
                    for b in range(2):
                        if choices[u][a] == choices[v][b]:
                            x, y = 2*u+a, 2*v+1-b
                            graph[x].append(y); reverse[y].append(x)
                            x, y = 2*v+b, 2*u+1-a
                            graph[x].append(y); reverse[y].append(x)
            self._implication, self._reverse = tuple(map(tuple, graph)), tuple(map(tuple, reverse))
        extra, back = {}, {}
        for v, domain in allowed.items():
            for a in range(2):
                if not ((domain >> choices[v][a]) & 1):
                    x, y = 2*v+a, 2*v+1-a
                    extra.setdefault(x, []).append(y); back.setdefault(y, []).append(x)
        seen, order = set(), []
        for root in range(2*self.n):
            if root in seen:
                continue
            seen.add(root)
            stack = [(root, itertools.chain(self._implication[root], extra.get(root, ())))]
            while stack:
                u, neighbors = stack[-1]
                v = next(neighbors, None)
                if v is None:
                    order.append(u); stack.pop()
                elif v not in seen:
                    seen.add(v)
                    stack.append((v, itertools.chain(self._implication[v], extra.get(v, ()))))
        component = [-1]*(2*self.n)
        for root in reversed(order):
            if component[root] != -1:
                continue
            component[root] = root
            stack = [root]
            while stack:
                u = stack.pop()
                for v in itertools.chain(self._reverse[u], back.get(u, ())):
                    if component[v] == -1:
                        component[v] = root; stack.append(v)
        return all(component[2*v] != component[2*v+1] for v in range(self.n))
