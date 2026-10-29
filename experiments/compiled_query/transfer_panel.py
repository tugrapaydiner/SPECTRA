"""Deterministic structural fixtures shared by current exact-query benchmarks.

The original frozen transfer runner and comparator build live in Git history.
Generator functions and seed definitions retain their original implementation.
"""
from __future__ import annotations

import hashlib
import math
import random
import statistics

from .binary_list import BinaryListProblem
from .cases import Case

FAMILIES = ('random_tree', 'percolated_grid', 'geometric', 'preferential',
            'rewired_ring', 'communities', 'percolated_cube', 'random_bipartite')


def seed_for(label: str) -> int:
    return int.from_bytes(hashlib.sha256(('SPECTRA-20261015-v1/' + label).encode()).digest()[:8], 'big')


def topology(family: str, n: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    if family not in FAMILIES or n < 8 or n & (n-1):
        raise ValueError('known family and power-of-two n >= 8 required')
    edges: set[tuple[int, int]] = set()
    def add(u, v):
        if u != v:
            edges.add((min(u, v), max(u, v)))
    if family == 'random_tree':
        for v in range(1, n):
            add(v, rng.randrange(v))
    elif family == 'percolated_grid':
        width = 1 << ((n.bit_length()-1)//2)
        for v in range(n):
            if v % width and rng.random() < .88:
                add(v, v-1)
            if v >= width and rng.random() < .88:
                add(v, v-width)
            if v >= width and v % width and rng.random() < .08:
                add(v, v-width-1)
    elif family == 'geometric':
        points = [(rng.random(), rng.random()) for _ in range(n)]
        radius2 = 6 / (math.pi*n)
        for u in range(n):
            for v in range(u):
                if sum((a-b)**2 for a, b in zip(points[u], points[v])) <= radius2:
                    add(u, v)
    elif family == 'preferential':
        edges.update(((0, 1), (0, 2), (1, 2)))
        urn = [0, 0, 1, 1, 2, 2]
        for v in range(3, n):
            targets = set()
            while len(targets) < 2:
                targets.add(rng.choice(urn))
            for u in sorted(targets):
                add(u, v)
                urn.extend((u, v))
    elif family == 'rewired_ring':
        for v in range(n):
            for distance in (1, 2):
                u = rng.randrange(n) if rng.random() < .15 else (v+distance) % n
                add(v, u)
    elif family == 'communities':
        for u in range(n):
            for v in range(u):
                probability = .07 if u//(n//4) == v//(n//4) else .005
                if rng.random() < probability:
                    add(u, v)
    elif family == 'percolated_cube':
        for v in range(n):
            for bit in range(n.bit_length()-1):
                u = v ^ (1 << bit)
                if u > v and rng.random() < .88:
                    add(u, v)
    else:
        for u in range(n//2):
            for v in range(n//2, n):
                if rng.random() < 8/n:
                    add(u, v)
    permutation = list(range(n))
    rng.shuffle(permutation)
    return tuple(sorted((min(permutation[u], permutation[v]), max(permutation[u], permutation[v]))
                        for u, v in edges))


def make_case(family: str, replica: int, *, count: int = 1024, n: int | None = None) -> Case:
    """No solver calls, filtering, repair, rejection, or answer-dependent loops."""
    if type(replica) is not int or not 0 <= replica < 6 or type(count) is not int or not 1 <= count <= 1024:
        raise ValueError('replica/count outside the fixed panel')
    n = n if n is not None else (128 if replica < 3 else 512)
    rng = random.Random(seed_for(f'{family}/{replica}'))
    edges = topology(family, n, rng)
    regime = replica % 3
    k = 2 if regime == 0 else 8
    masks = []
    for v in range(n):
        if regime == 0:
            masks.append(3)
        else:
            colors = rng.sample(range(k), 1 if regime == 2 and v % 7 == 0 else 2)
            masks.append(sum(1 << c for c in colors))
    p = BinaryListProblem.create(n, k, edges, masks)
    queries = []
    for i in range(count):
        if (i+1) % 29 == 0:
            queries.append(())
            continue
        width = (1, 2, 4, 8)[i % 4]
        row = []
        for v in rng.sample(range(n), min(width, n)):
            colors = [c for c in range(k) if masks[v] >> c & 1]
            allowed = 1 << rng.choice(colors)
            if (i+1) % 11 == 0:
                allowed = masks[v]
            if (i+1) % 23 == 0:
                allowed |= 1 << 63  # irrelevant bits cannot add original colours
            row.append((v, allowed))
        if (i+1) % 17 == 0:
            row.append((row[0][0], masks[row[0][0]] ^ (row[0][1] & masks[row[0][0]])))
        if (i+1) % 31 == 0:
            row.append((row[0][0], 0))
        queries.append(tuple(row))
    name = f'{family}-r{replica}-n{n}-d{regime}'
    return Case(name, name, 'prospective synthetic structural panel v1; no selection by outcomes', p, tuple(queries))


def gmean(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


if __name__ == "__main__":
    raise SystemExit(
        "The frozen transfer runner is archived; use its complete source at "
        "https://github.com/tugrapaydiner/SPECTRA/tree/39e9ee0b89f1261106231ee94546758c510291ba"
    )
