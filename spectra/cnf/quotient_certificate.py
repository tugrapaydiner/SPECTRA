"""Independent, standard-library-only verifier for exact quotient certificates.

The verifier rebuilds the ORIGINAL binary implication graph without union-find or
native compiled indexes. It checks that identified choices have the same strongly
connected component, and reconstructs every quotient palette and forbidden pair.
This proves preservation of the whole solution relation, including restrictions
on any original vertex. It is not a proof that every search algorithm is correct,
a machine-checked proof of Python/compiler/hardware, or an UNSAT proof for the
multi-choice part. No native code, pickle, or certificate-supplied code is executed.
"""
from __future__ import annotations
from collections import defaultdict
from itertools import accumulate


class InvalidCertificate(ValueError):
    """Original inputs or supplied integer-only evidence violate the contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidCertificate(message)


def _integer(x, lo, hi, name):
    require(type(x) is int and lo <= x <= hi, name)
    return x


def _components(adjacency, reverse):
    """Iterative Kosaraju on the independently constructed original graph."""
    size = len(adjacency)
    seen = bytearray(size)
    order = []
    for root in range(size):
        if seen[root]:
            continue
        seen[root] = 1
        stack = [(root, iter(adjacency[root]))]
        while stack:
            vertex, children = stack[-1]
            child = next(children, None)
            if child is None:
                order.append(vertex)
                stack.pop()
            elif not seen[child]:
                seen[child] = 1
                stack.append((child, iter(adjacency[child])))
    components = [-1] * size
    next_component = 0
    for root in reversed(order):
        if components[root] != -1:
            continue
        stack = [root]
        components[root] = next_component
        while stack:
            vertex = stack.pop()
            for child in reverse[vertex]:
                if components[child] == -1:
                    components[child] = next_component
                    stack.append(child)
        next_component += 1
    return components


def verify_quotient(n: int, k: int, edges: tuple, masks: tuple,
                    certificate: dict) -> dict:
    """Reject altered mapping or omitted constraints; return a scoped audit receipt.

    For obvious original contradictions, the native index can have a partial map.
    An independent contradiction is then enough to verify the impossible flag;
    the meaningless mapping of an impossible problem is explicitly NOT certified.
    """
    _integer(n, 0, 100000, 'invalid n')
    _integer(k, 0, 64, 'invalid k')
    require(type(edges) is tuple and len(edges) <= 2000000, 'invalid edge bank')
    require(type(masks) is tuple and len(masks) in (0, n), 'invalid mask bank')
    full = (1 << k) - 1
    original = masks or (full,) * n
    for d in original:
        _integer(d, 0, full, 'invalid original domain')
    for edge in edges:
        require(type(edge) is tuple and len(edge) == 2, 'invalid original edge')
        for v in edge:
            _integer(v, 0, n - 1, 'invalid original endpoint')
    fields = {'schema', 'n', 'k', 'impossible', 'node', 'wide', 'colour0', 'colour1',
              'palettes', 'initial', 'start', 'offsets', 'arcs'}
    require(type(certificate) is dict and set(certificate) == fields,
            'unexpected certificate fields')
    c = certificate
    require(type(c['schema']) is str and c['schema'] == 'spectra.quotient.certificate.v1', 'unsupported schema')
    require(type(c['n']) is int and type(c['k']) is int and c['n'] == n and c['k'] == k,
            'certificate geometry differs')
    require(type(c['impossible']) is bool, 'invalid impossible flag')

    binary = [v for v in range(n) if 0 < original[v].bit_count() <= 2]
    binary_id = {v: i for i, v in enumerate(binary)}
    literals = {}
    for v in binary:
        colours = [d for d in range(k) if original[v] >> d & 1]
        literals[v] = {colour: 2 * binary_id[v] + j for j, colour in enumerate(colours)}
    adj = [[] for _ in range(2 * len(binary))]
    rev = [[] for _ in adj]

    def add(a, b):
        adj[a].append(b)
        rev[b].append(a)

    for v in binary:
        if original[v].bit_count() == 1:
            a = 2 * binary_id[v]
            add(a ^ 1, a)
    for a, b in edges:
        if a not in binary_id or b not in binary_id:
            continue
        shared = original[a] & original[b]
        while shared:
            colour = (shared & -shared).bit_length() - 1
            shared &= shared - 1
            la, lb = literals[a][colour], literals[b][colour]
            add(la, lb ^ 1)
            add(lb, la ^ 1)
    components = _components(adj, rev)
    original_contradiction = (any(not d for d in original)
                              or any(a == b for a, b in edges)
                              or any(components[2 * i] == components[2 * i + 1]
                                     for i in range(len(binary))))
    if original_contradiction and c['impossible']:
        return {'schema': 'spectra.quotient.audit.v1', 'valid': True,
                'original_contradiction_verified': True,
                'mapping_verified': False, 'native_code_executed': False}

    for field in ('node', 'wide', 'colour0', 'colour1'):
        require(type(c[field]) is list and len(c[field]) == n, 'bad ' + field)
    require(type(c['palettes']) is list and len(c['palettes']) <= n, 'bad palettes')
    qn = len(c['palettes'])
    owner = defaultdict(list)
    for v in range(n):
        q = _integer(c['node'][v], 0, qn - 1, 'bad quotient node')
        _integer(c['wide'][v], 0, 1, 'bad wide flag')
        _integer(c['colour0'][v], 0, 255, 'bad colour0')
        _integer(c['colour1'][v], 0, 255, 'bad colour1')
        owner[q].append(v)
    require(set(owner) == set(range(qn)), 'unowned quotient node')
    for d in c['palettes']:
        _integer(d, 0, (1 << 64) - 1, 'bad quotient palette')
    expected_initial = list(c['palettes'])
    forward = {}
    for q, originals in owner.items():
        wide_flags = {c['wide'][v] for v in originals}
        require(len(wide_flags) == 1, 'mixed wide/binary quotient owner')
        if wide_flags == {1}:
            require(len(originals) == 1, 'wide variables cannot be aliased')
            v = originals[0]
            require(c['palettes'][q] == original[v], 'wide palette changed')
            require(c['colour0'][v] == c['colour1'][v] == 255, 'wide lift map differs')
            forward[v] = {colour: colour for colour in range(k) if original[v] >> colour & 1}
        else:
            require(c['palettes'][q] == 3, 'binary palette differs')
            references = [None, None]
            for v in originals:
                require(v in binary_id, 'nonbinary variable aliased')
                lift = [c['colour0'][v], c['colour1'][v]]
                require(len(set(lift)) == 2, 'noninvertible binary lift')
                allowed_colours = set(literals[v])
                require({x for x in lift if x != 255} == allowed_colours,
                        'binary original colour set changed')
                forward[v] = {}
                for side, colour in enumerate(lift):
                    if colour == 255:
                        require(original[v].bit_count() == 1, 'missing binary choice')
                        lit = 2 * binary_id[v] + 1
                        expected_initial[q] &= ~(1 << side)
                    else:
                        lit = literals[v][colour]
                        forward[v][colour] = side
                    component = components[lit]
                    if references[side] is None:
                        references[side] = component
                    require(references[side] == component,
                            'identified choices are not implication-equivalent')
    require(type(c['initial']) is list, 'bad initial bank')
    require(len(c['initial']) == qn, 'bad initial size')
    for d in c['initial']:
        _integer(d, 0, (1 << 64) - 1, 'bad initial mask')

    atom = {}
    start = [0]
    for q, palette in enumerate(c['palettes']):
        for colour in range(64):
            if palette >> colour & 1:
                atom[q, colour] = len(atom)
        start.append(len(atom))
    require(type(c['start']) is list and c['start'] == start, 'atom offsets differ')
    rewritten = defaultdict(int)
    for a, b in edges:
        shared = original[a] & original[b]
        while shared:
            colour = (shared & -shared).bit_length() - 1
            shared &= shared - 1
            qa, qb = c['node'][a], c['node'][b]
            va, vb = forward[a][colour], forward[b][colour]
            if qa == qb:
                if va == vb:
                    expected_initial[qa] &= ~(1 << va)
            else:
                rewritten[atom[qa, va], qb] |= 1 << vb
                rewritten[atom[qb, vb], qa] |= 1 << va
    require(c['initial'] == expected_initial, 'initial constraints differ')
    expected_arcs = []
    counts = [0] * len(atom)
    for (source, target), forbidden in sorted(rewritten.items()):
        expected_arcs.append([target, forbidden])
        counts[source] += 1
    offsets = [0, *accumulate(counts)]
    require(type(c['arcs']) is list and c['arcs'] == expected_arcs,
            'rewritten original constraints differ')
    require(type(c['offsets']) is list and c['offsets'] == offsets, 'arc offsets differ')
    # Reject bools masquerading as integer 0/1 in transport arrays.
    for field in ('start', 'offsets'):
        require(all(type(x) is int for x in c[field]), 'noninteger ' + field)
    require(all(type(pair) is list and len(pair) == 2 and
                all(type(x) is int for x in pair) for pair in c['arcs']), 'bad arc encoding')
    require(c['impossible'] == any(not d for d in expected_initial),
            'unsupported impossible flag')
    return {'schema': 'spectra.quotient.audit.v1', 'valid': True,
            'original_contradiction_verified': original_contradiction or c['impossible'], 'mapping_verified': True,
            'vertices': n, 'quotient_vertices': qn, 'binary_vertices': len(binary),
            'original_edges': len(edges), 'quotient_arcs': len(expected_arcs),
            'native_code_executed': False}
