"""Translate a sum of parity constraints into a bounded original-CNF LRAT proof."""
from __future__ import annotations

from collections import defaultdict, deque
from itertools import chain
from shutil import copyfileobj
from tempfile import SpooledTemporaryFile

from benchmarks.sat.lrat import LRATWriter, ProofLimit
from benchmarks.sat.parity import conflicting_product_rows


class _Extensions:
    def __init__(self, writer, nvars):
        self.writer = writer
        self.variable = nvars

    def record(self, clause, hints=()):
        return self.writer.add(clause, hints), clause

    def gate(self, a, b, *, conjunction):
        self.variable += 1
        v = self.variable
        if conjunction:
            first = self.record((v, -a, -b))
            hints = (-first[0],)
            return v, (first, self.record((-v, a), hints), self.record((-v, b), hints))
        first = self.record((v, a, -b))
        second = self.record((v, -a, b))
        hints = (-first[0], -second[0])
        return v, (first, second, self.record((-v, a, b), hints), self.record((-v, -a, -b), hints))


def _unit_hints(clause, premises):
    assigned = {-v for v in clause}
    hints = []
    while True:
        later = []
        changed = False
        for number, row in premises:
            unit = 0
            for literal in row:
                if literal in assigned:
                    break
                if -literal not in assigned:
                    if unit:
                        later.append((number, row))
                        break
                    unit = literal
            else:
                hints.append(number)
                if not unit:
                    return hints
                assigned.add(unit)
                changed = True
        if not changed:
            return None
        premises = later


def _derive(proof, clause, premises, splits):
    hints = _unit_hints(clause, premises)
    if hints is not None:
        return proof.record(clause, hints)
    for index, pivot in enumerate(splits):
        if pivot not in clause and -pivot not in clause:
            positive = _derive(proof, (*clause, pivot), premises, splits[index + 1:])
            negative = _derive(proof, (*clause, -pivot), premises, splits[index + 1:])
            return proof.record(clause, (positive[0], negative[0]))
    raise ValueError("parity clause has no local unit-propagation derivation")


def _patterns(variables, odd):
    for mask in range(1 << len(variables)):
        if mask.bit_count() % 2 != odd:
            yield mask, tuple(-v if mask & (1 << i) else v for i, v in enumerate(variables))


def _table(proof, variables, odd, premises, splits=()):
    return tuple(_derive(proof, clause, premises, splits) for _, clause in _patterns(sorted(variables), odd))


def _product_nodes(proof, original, rows):
    terms = sorted({term for row in rows for term in row.terms if term != (0, 0)})
    if len(terms) > 8192:
        raise ProofLimit
    symbols, definitions = {}, {}
    for a, b in terms:
        if a and b:
            symbols[a, b], definitions[a, b] = proof.gate(a, b, conjunction=True)
        else:
            symbols[a, b] = a or b
    nodes = {}
    for index, row in enumerate(rows):
        variables = {symbols[term] for term in row.terms if term != (0, 0)}
        odd = int((0, 0) in row.terms)
        source = [(i, original[i - 1]) for i in row.clauses]
        for term in row.terms:
            source.extend(definitions.get(term, ()))
        nodes[index] = variables, odd, _table(proof, variables, odd, source, row.splits)
    return nodes


def _contract(proof, nodes):
    incident = defaultdict(set)
    for index, (variables, _, _) in nodes.items():
        for v in variables:
            incident[v].add(index)
    queue = deque(i for i, (variables, _, _) in nodes.items() if len(variables) <= 2)
    while queue:
        index = queue.popleft()
        if index not in nodes:
            continue
        variables, odd, fact = nodes.pop(index)
        if not variables:
            if odd:
                return {index: (variables, odd, fact)}
            continue
        for v in variables:
            incident[v].discard(index)
        pivot, *rest = sorted(variables, key=lambda v: (len(incident[v]), v))
        for other in sorted(incident.pop(pivot, ())):
            before, side, table = nodes[other]
            after = before ^ variables
            nodes[other] = after, side ^ odd, _table(proof, after, side ^ odd, table + fact, rest)
            for v in rest:
                if v in after:
                    incident[v].add(other)
                else:
                    incident[v].discard(other)
            if len(after) <= 2:
                queue.append(other)
    return nodes


def _pair_occurrences(proof, nodes):
    occurrences = defaultdict(list)
    for index, (variables, _, _) in sorted(nodes.items()):
        for v in variables:
            occurrences[v].append(index)
    for v, indices in sorted(occurrences.items()):
        if len(indices) % 2:
            raise ValueError("parity sum contains an uncancelled variable")
        for start in range(2, len(indices), 2):
            proof.variable += 1
            alias = proof.variable
            first = proof.record((alias, -v))
            definitions = (first, proof.record((-alias, v), (-first[0],)))
            for index in indices[start:start + 2]:
                variables, odd, table = nodes[index]
                after = (variables - {v}) | {alias}
                nodes[index] = after, odd, _table(proof, after, odd, table + definitions)
    edges = defaultdict(list)
    for index, (variables, _, _) in nodes.items():
        for v in sorted(variables):  # Use equation order to break ties between shortest routes.
            edges[v].append(index)
    return edges


def _cycle_routes(nodes, edges):
    adjacent = defaultdict(dict)
    for edge, (a, b) in edges.items():
        adjacent[a][edge], adjacent[b][edge] = b, a
    seen, tree = set(), set()
    for start in sorted(nodes, key=lambda i: (-len(adjacent[i]), i)):
        if start not in seen:
            queue = [start]
            seen.add(start)
            for node in queue:
                for edge, other in adjacent[node].items():
                    if other not in seen:
                        seen.add(other)
                        queue.append(other)
                        tree.add(edge)

    visits = 0
    for edge in sorted(edges.keys() - tree, reverse=True):
        start, end = edges[edge]
        del adjacent[start][edge], adjacent[end][edge]
        parents = {start: None}
        queue = deque([start])
        while end not in parents:
            node = queue.popleft()
            visits += len(adjacent[node])
            if visits > 4_000_000:
                raise ProofLimit
            for link, other in adjacent[node].items():
                if other not in parents:
                    parents[other] = node, link
                    queue.append(other)
        route = []
        node = end
        while node != start:
            previous, link = parents[node]
            route.append((previous, link, node))
            node = previous
        yield edge, route[::-1]


def _flip(proof, node, pivot, endpoint, changes):
    before, odd, previous = node
    after = set(before)
    if endpoint:
        after.remove(pivot)
    replacements = {}
    for old, new, definitions in changes:
        after.remove(old)
        after.add(new)
        replacements[old] = new

    positions = {v: i for i, v in enumerate(sorted(after))}
    old_variables = sorted(before)
    toggled = sum(1 << i for i, v in enumerate(old_variables) if v == pivot or v in replacements)
    sources = [(i, positions[replacements.get(v, v)]) for i, v in enumerate(old_variables) if v != pivot]
    records = []
    for mask, clause in _patterns(sorted(after), odd):
        base = sum(1 << i for i, bit in sources if mask & (1 << bit))
        positive, negative = [], []
        for _, new, definitions in changes:
            # Fixing the pivot makes each XOR gate propagate the old edge from the new edge.
            one = bool(mask & (1 << positions[new]))
            positive.append(definitions[2 if one else 1][0])
            negative.append(definitions[3 if one else 0][0])
        # Each consecutive pair of assignments has exactly one row of the wrong parity.
        positive.append(previous[base >> 1][0])
        negative.append(previous[(base ^ toggled) >> 1][0])
        yes = proof.record((*clause, pivot), positive)
        no = proof.record((*clause, -pivot), negative)
        records.append(proof.record(clause, (yes[0], no[0])))
    return after, odd, tuple(records)


def _finish(proof, nodes):
    nodes = _contract(proof, nodes)
    edges = _pair_occurrences(proof, nodes)
    labels = {edge: edge for edge in edges}
    for edge, route in _cycle_routes(nodes, edges):
        pivot = labels[edge]
        changes = defaultdict(list)
        for start, link, end in route:
            old = labels[link]
            new, definitions = proof.gate(old, pivot, conjunction=False)
            labels[link] = new
            changes[start].append((old, new, definitions))
            changes[end].append((old, new, definitions))
        endpoints = route[0][0], route[-1][2]
        for index, substitutions in changes.items():
            nodes[index] = _flip(proof, nodes[index], pivot, index in endpoints, substitutions)
    hints = _unit_hints((), chain.from_iterable(table for _, _, table in nodes.values()))
    if hints is None:
        raise ValueError("final parity forest has no unit contradiction")
    proof.record((), hints)


def write_coupled_parity_proof(admitted, path, max_rows, max_literals, max_bytes):
    """Return False for unsupported inputs and exhausted bounds; expose only complete proofs."""
    if admitted.receipt.dimacs_bytes > 1 << 20 or not min(max_rows, max_literals, max_bytes):
        return False
    rows = conflicting_product_rows(admitted.nvars, admitted.clauses)
    if not rows:
        return False
    with SpooledTemporaryFile(max_size=1 << 20, dir=path.parent) as stream:
        writer = LRATWriter(stream, len(admitted.clauses), min(max_rows, 250_000),
                            min(max_literals, 1_500_000), min(max_bytes, 8 << 20))
        proof = _Extensions(writer, admitted.nvars)
        try:
            _finish(proof, _product_nodes(proof, admitted.clauses, rows))
        except ProofLimit:
            return False
        stream.seek(0)
        with path.open("xb") as destination:
            copyfileobj(stream, destination, 65536)
    return True
