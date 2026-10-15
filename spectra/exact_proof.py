"""Public standard-library SBP1 verifier, independent of the native solver.

No compiler imports, native index, saved feasibility result, SCC certificate or
answer bank. Literal 2*v denotes the lower available colour; 2*v+1 the higher
(or the unavailable alternative for a singleton). All integers are little endian.
"""
from __future__ import annotations

import struct

HEADER = struct.Struct('<4sIII')


def decode_proof(payload: bytes, n: int) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    if type(n) is not int or not 0 <= n <= 100_000:
        raise ValueError('invalid original variable count')
    if type(payload) is not bytes or not HEADER.size <= len(payload) <= 20 + 16*n:
        raise ValueError('invalid bounded proof buffer')
    magic, kind, left, right = HEADER.unpack_from(payload)
    if magic != b'SBP1' or len(payload) != HEADER.size + 4*(left+right):
        raise ValueError('proof header/length differs')
    if kind == 1:
        if left != 1 or right != 0:
            raise ValueError('empty-domain proof needs one original vertex')
    elif kind in (2, 3):
        if not 1 <= left <= 2*n or not 1 <= right <= 2*n:
            raise ValueError('path size outside original literal count')
    else:
        raise ValueError('unsupported proof kind')
    values = tuple(value[0] for value in struct.iter_unpack('<I', payload[HEADER.size:]))
    return kind, values[:left], values[left:]


def verify_answer(problem, query, status: str, labels: bytes, certificate: bytes) -> bool:
    """Reject invalid input/evidence; check truth only from original constraints.

    `visited`/timing counters are not evidence and are intentionally not inputs.
    A valid proof can remain valid under stronger restrictions; mere difference
    from a previous byte string does not make a proof fresh or stale.
    """
    try:
        n, k = problem.n, problem.palette
        if type(n) is not int or not 0 <= n <= 100_000 or type(k) is not int or not 1 <= k <= 64:
            return False
        masks = tuple(problem.masks)
        if len(masks) != n or any(type(d) is not int or d <= 0 or d >> k or d.bit_count() > 2 for d in masks):
            return False
        choices = tuple(tuple(c for c in range(k) if d & (1 << c)) for d in masks)
        edges = set()
        for row in problem.edges:
            u, v = row
            if any(type(x) is not int or not 0 <= x < n for x in (u, v)) or u == v:
                return False
            edges.add((min(u, v), max(u, v)))
        allowed = list(masks)
        for row in query:
            vertex, domain = row
            if (type(vertex) is not int or not 0 <= vertex < n or
                    type(domain) is not int or not 0 <= domain < 1 << 64):
                return False
            allowed[vertex] &= domain
        if type(labels) is not bytes or type(certificate) is not bytes:
            return False
        if status == 'SAT_VERIFIED':
            return (not certificate and len(labels) == n
                    and all(c < k and (allowed[v] >> c) & 1 for v, c in enumerate(labels))
                    and all(labels[u] != labels[v] for u, v in edges))
        if status != 'UNSAT_VERIFIED' or labels:
            return False
        kind, a, b = decode_proof(certificate, n)
        if kind == 1:
            return a[0] < n and allowed[a[0]] == 0
        if (any(not 0 <= x < 2*n for x in a+b) or
                len(set(a)) != len(a) or len(set(b)) != len(b)):
            return False

        def color(literal):
            row = choices[literal // 2]
            return row[literal % 2] if literal % 2 < len(row) else None

        for path in (a, b):
            for before, after in zip(path, path[1:]):
                u, v = before // 2, after // 2
                if u == v:
                    if before != 2*u+1 or after != 2*u or len(choices[u]) != 1:
                        return False
                elif ((min(u, v), max(u, v)) not in edges or color(before) is None
                      or color(after ^ 1) is None or color(before) != color(after ^ 1)):
                    return False
        if kind == 3:
            return a[0] == b[-1] and a[-1] == b[0] and a[-1] == (a[0] ^ 1)
        return (a[-1] == (b[-1] ^ 1) and
                all(color(root) is not None and allowed[root//2] == 1 << color(root)
                    for root in (a[0], b[0])))
    except (ValueError, TypeError, AttributeError, OverflowError, struct.error):
        return False
