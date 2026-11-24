"""Bounded LRAT refutations of binary-encoded pigeonhole subformulas."""
from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from itertools import combinations
from pathlib import Path

from benchmarks.sat.competition_input import admit_and_load_cnf
from benchmarks.sat.lrat import LRATWriter, ProofLimit


def _find_core(nvars, clauses):
    if nvars > 4096 or len(clauses) > 20_000:
        return None
    occurrences = defaultdict(list)
    literal_count = 0
    for index, clause in enumerate(clauses):
        variables = [abs(literal) for literal in clause]
        literal_count += len(clause)
        if (not clause or len(clause) > 12 or literal_count > 200_000
                or len(set(variables)) != len(variables)):
            return None
        for variable in variables:
            occurrences[variable].append(index)

    # Bits in the same block occur together in every clause that mentions them.
    by_occurrences = defaultdict(list)
    for variable in sorted(occurrences):
        by_occurrences[tuple(occurrences[variable])].append(variable)
    blocks = list(by_occurrences.values())
    if not 3 <= len(blocks) <= 128 or any(len(block) > 6 for block in blocks):
        return None
    position = {v: (g, bit) for g, block in enumerate(blocks) for bit, v in enumerate(block)}
    domains = [set(range(1 << len(block))) for block in blocks]
    forbidden = defaultdict(set)
    for clause in clauses:
        values = {}
        for literal in clause:
            group, bit = position[abs(literal)]
            values[group] = values.get(group, 0) | ((literal < 0) << bit)
        if len(values) == 1:
            (group, value), = values.items()
            domains[group].discard(value)
        elif len(values) == 2:
            left, right = sorted(values)
            forbidden[left, right].add((values[left], values[right]))
        else:
            return None

    def incompatible(left, a, right, b):
        if left > right:
            left, a, right, b = right, b, left, a
        return (a, b) in forbidden.get((left, right), ())

    remaining = 1_000_000
    for anchor, domain in enumerate(domains):
        size = len(domain)
        if not 2 <= size <= 24 or size >= len(blocks):
            continue
        slots = sorted(domain)
        aligned = [(anchor, slots)]
        for group, values in enumerate(domains):
            if group == anchor or len(values) != size:
                continue
            images = []
            for a in slots:
                remaining -= size
                if remaining < 0:
                    return None
                partners = [b for b in values if incompatible(anchor, a, group, b)]
                if len(partners) != 1:
                    break
                images.append(partners[0])
            if len(set(images)) != size:
                continue
            remaining -= size * len(aligned)
            if remaining < 0:
                return None
            if all(incompatible(other, a, group, b)
                   for other, prior in aligned for a, b in zip(prior, images)):
                aligned.append((group, images))
                if len(aligned) == size + 1:
                    return [(blocks[g], choices) for g, choices in aligned]
    return None


def _pattern(bits, value):
    return [v if value & (1 << bit) else -v for bit, v in enumerate(bits)]


def _write_lrat(writer, nvars, clauses, core):
    source = {tuple(sorted(clause)): i for i, clause in enumerate(clauses, 1)}
    add = writer.add

    fresh = nvars
    cells, covers, bit_hints, exclusions = [], [], [], []
    for bits, slots in core:
        row, implications, patterns, branches = [], [], [], {}
        for value in slots:
            fresh += 1
            row.append(fresh)
            pattern = _pattern(bits, value)
            excluded = [-v for v in pattern]
            definition = add([fresh, *excluded])
            branches[value] = definition
            implications.append([add([-fresh, v], [-definition]) for v in pattern])
            patterns.append(excluded)
        for value in range(1 << len(bits)):
            if value not in branches:
                branches[value] = source[tuple(sorted(-v for v in _pattern(bits, value)))]
        # Resolve the two possible settings of each bit to prove the row covers its domain.
        for depth in reversed(range(len(bits))):
            branches = {prefix: add(row + [-v for v in _pattern(bits[:depth], prefix)],
                                    [branches[prefix], branches[prefix | (1 << depth)]])
                        for prefix in range(1 << depth)}
        cells.append(row)
        covers.append(branches[0])
        bit_hints.append(implications)
        exclusions.append(patterns)

    mutex = {}
    for p, q in combinations(range(len(cells)), 2):
        for h, (a, b) in enumerate(zip(cells[p], cells[q])):
            original = source[tuple(sorted(exclusions[p][h] + exclusions[q][h]))]
            mutex[p, q, h] = add([-a, -b], bit_hints[p][h] + bit_hints[q][h] + [original])

    while len(cells[0]) > 1:
        last_p, last_h = len(cells) - 1, len(cells[0]) - 1
        next_cells, next_covers, reverse = [], [], []
        for p, pigeon in enumerate(cells[:-1]):
            row, keep, move, backward = [], [], [], []
            for a, b in zip(pigeon[:-1], cells[last_p][:-1]):
                # Cook's reduction gives the last pigeon's hole to the occupant of the last hole.
                fresh += 1
                row.append(fresh)
                kept = add([fresh, -a])
                moved = add([fresh, -pigeon[last_h], -b])
                keep.append(kept)
                move.append(moved)
                backward.append((add([-fresh, a, pigeon[last_h]], [-kept, -moved]),
                                 add([-fresh, a, b], [-kept, -moved])))
            hints = keep + [covers[p]] + move + [covers[last_p], mutex[p, last_p, last_h]]
            next_covers.append(add(row, hints))
            next_cells.append(row)
            reverse.append(backward)
        next_mutex = {}
        for p, q in combinations(range(len(next_cells)), 2):
            for h, (left, right) in enumerate(zip(next_cells[p], next_cells[q])):
                bridge = add([-left, -cells[q][h]],
                             [mutex[p, q, h], mutex[q, last_p, h], reverse[p][h][1]])
                next_mutex[p, q, h] = add([-left, -right], [
                    bridge, *reverse[q][h], mutex[p, q, last_h], mutex[p, last_p, h], reverse[p][h][0]])
        cells, covers, mutex = next_cells, next_covers, next_mutex
    add([], [covers[0], covers[1], mutex[0, 1, 0]])


def write_pigeonhole_proof(cnf: Path, proof: Path, max_bytes: int) -> bool:
    """Write a complete ASCII LRAT proof for an admitted core that fits the budget."""
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("max_proof_bytes must be a nonnegative integer")
    if max_bytes == 0:
        return False
    try:
        admitted = admit_and_load_cnf(cnf, max_source_bytes=1 << 20, max_dimacs_bytes=1 << 20,
                                      max_variable_bitmap_bytes=513)
    except MemoryError:
        return False
    return write_admitted_pigeonhole_proof(admitted, proof, 100_000, 1_000_000, max_bytes)


def write_admitted_pigeonhole_proof(admitted, proof, max_rows, max_literals, max_bytes):
    if admitted.receipt.dimacs_bytes > 1 << 20 or not max_bytes or not max_rows:
        return False
    core = _find_core(admitted.nvars, admitted.clauses)
    if core is None:
        return False
    with BytesIO() as buffer:
        writer = LRATWriter(buffer, len(admitted.clauses), min(max_rows, 100_000),
                            min(max_literals, 1_000_000), min(max_bytes, 8 << 20))
        try:
            _write_lrat(writer, admitted.nvars, admitted.clauses, core)
        except ProofLimit:
            return False
        with buffer.getbuffer() as view, proof.open("xb") as destination:
            destination.write(view)
    return True

