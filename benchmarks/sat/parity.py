"""Find contradictions between two affine GF(2) systems and their binary clauses."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import NamedTuple


class _Equation(NamedTuple):
    variables: tuple[int, ...]
    odd: int
    clauses: tuple[int, ...]


class _Space(NamedTuple):
    basis: list[int]
    forms: dict[int, int]
    reasons: dict[int, int]


class ProductRow(NamedTuple):
    terms: tuple[tuple[int, int], ...]
    clauses: tuple[int, ...]
    splits: tuple[int, ...]


def _bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _equations(clauses):
    groups = defaultdict(dict)
    for number, row in enumerate(clauses, 1):
        if 1 <= len(row) <= 4 and len(set(map(abs, row))) == len(row):
            signed = tuple(sorted(row, key=abs))
            groups[tuple(map(abs, signed))].setdefault(signed, number)

    equations = []
    for variables, rows in sorted(groups.items()):
        if len(variables) < 2 or len(rows) < 2:
            continue
        position = {v: 1 << i for i, v in enumerate(variables)}
        full = (1 << len(variables)) - 1
        forbidden = {}
        for width in range(1, len(variables) + 1):
            for subset in combinations(variables, width):
                free = full ^ sum(position[v] for v in subset)
                for row, number in groups.get(subset, {}).items():
                    pattern = sum(position[-v] for v in row if v < 0)
                    rest = free
                    while True:
                        forbidden.setdefault(pattern | rest, number)
                        if not rest:
                            break
                        rest = (rest - 1) & free
        sides = {m.bit_count() % 2 for m in range(full + 1) if m not in forbidden}
        if len(sides) == 1:
            odd = sides.pop()
            reasons = dict.fromkeys(forbidden[m] for m in range(full + 1) if m.bit_count() % 2 != odd)
            equations.append(_Equation(variables, odd, tuple(reasons)))
    return equations


def _components(equations):
    incident = defaultdict(list)
    for index, equation in enumerate(equations):
        for v in equation.variables:
            incident[v].append(index)
    unseen = set(range(len(equations)))
    components = []
    while unseen and len(components) < 2:
        seed = min(unseen)
        unseen.remove(seed)
        queue = [seed]
        for index in queue:
            for v in equations[index].variables:
                for other in incident.pop(v, ()):
                    if other in unseen:
                        unseen.remove(other)
                        queue.append(other)
        components.append(sorted(queue))
    return components if not unseen else []


def _affine_space(equations, component):
    pivots = {}
    variables = set()
    for index in component:
        equation = equations[index]
        variables.update(equation.variables)
        row = sum(1 << v for v in equation.variables) | equation.odd
        reason = 1 << index
        while row > 1:
            pivot = row.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = row, reason
                break
            other, source = pivots[pivot]
            row ^= other
            reason ^= source
        if row == 1:
            return None

    basis, forms, reasons = [0], {0: 1}, {0: 0}  # Coordinate zero represents the constant one.
    for v in sorted(variables):
        if v not in pivots:
            forms[v], reasons[v] = 1 << len(basis), 0
            basis.append(v)
        else:
            row, reason = pivots[v]
            value = row & 1
            for other in _bits(row ^ (1 << v) ^ value):
                value ^= forms[other]
                reason ^= reasons[other]
            forms[v], reasons[v] = value, reason
    return _Space(basis, forms, reasons)


def _conflicting_pairs(clauses, left, right):
    width = len(left.basis)
    if width * len(right.basis) > 65_536:
        return None
    pairs = []
    for number, row in enumerate(clauses, 1):
        if len(row) == 2:
            a, b = row
            if abs(a) in right.forms:
                a, b = b, a
            if abs(a) in left.forms and abs(b) in right.forms:
                pairs.append((number, a, b))
    if len(pairs) > 4096:
        return None
    pivots = {}
    steps = 0
    for index, (_, a, b) in enumerate(pairs):
        x, y = left.forms[abs(a)] ^ (a > 0), right.forms[abs(b)] ^ (b > 0)
        product = 0  # The indicators that falsify this binary clause cannot both be one.
        for bit in _bits(y):
            product ^= x << (bit * width)
        reason = 1 << index
        while product > 1:
            pivot = product.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = product, reason
                break
            other, source = pivots[pivot]
            product ^= other
            reason ^= source
            steps += 1
            if steps > 100_000:
                return None
        if product == 1:
            return [pairs[j] for j in _bits(reason)]
    return None


def _multipliers(left, right, terms):
    partial, left_reasons, right_reasons = defaultdict(int), defaultdict(int), defaultdict(int)
    for a, b in terms:
        left_reasons[b] ^= left.reasons[a]
        partial[b] ^= left.forms[a]
    for b, value in partial.items():
        for bit in _bits(value):
            right_reasons[left.basis[bit]] ^= right.reasons[b]
    return left_reasons, right_reasons


def _linearize(equations, components, spaces, pairs):
    rows = []
    remainder = set()
    for number, a, b in pairs:
        terms = {(abs(a), abs(b))}
        if a > 0:
            terms.add((0, abs(b)))
        if b > 0:
            terms.add((abs(a), 0))
        if a > 0 and b > 0:
            terms.add((0, 0))
        remainder ^= terms
        rows.append(ProductRow(tuple(sorted(terms)), (number,), (abs(a), abs(b))))

    complements = []
    for component in components:
        odd_variables = set()
        constant = 0
        for index in component:
            odd_variables.symmetric_difference_update(equations[index].variables)
            constant ^= equations[index].odd
        # A zero sum of the component's rows allows the shorter complementary derivation.
        complements.append(sum(1 << i for i in component) if not odd_variables and not constant else 0)

    def row_count(factors):
        return sum(min(mask.bit_count(), (mask ^ complement).bit_count())
                   for group, complement in zip(factors, complements) for mask in group.values())

    left, right = spaces
    forward = _multipliers(left, right, remainder)
    reverse = _multipliers(right, left, ((b, a) for a, b in remainder))[::-1]
    selected_factors = min((forward, reverse), key=row_count)
    for side, (factors, complement) in enumerate(zip(selected_factors, complements)):
        for factor, selected in sorted(factors.items()):
            if (selected ^ complement).bit_count() < selected.bit_count():
                selected ^= complement
            if len(rows) + selected.bit_count() > 8192:
                return None
            for index in _bits(selected):
                equation = equations[index]
                variables = (*equation.variables, 0) if equation.odd else equation.variables
                terms = ((factor, v) if side else (v, factor) for v in variables)
                rows.append(ProductRow(tuple(sorted(terms)), equation.clauses, (factor,) if factor else ()))
    return rows


def conflicting_product_rows(nvars, clauses):
    """Return original-clause-supported product equations whose XOR is the constant one."""
    if nvars > 2048 or len(clauses) > 20_000 or sum(map(len, clauses)) > 160_000:
        return None
    equations = _equations(clauses)
    if not 2 <= len(equations) <= 2048:
        return None
    components = _components(equations)
    if len(components) != 2:
        return None
    spaces = [_affine_space(equations, component) for component in components]
    if any(space is None for space in spaces):
        return None
    pairs = _conflicting_pairs(clauses, *spaces)
    return _linearize(equations, components, spaces, pairs) if pairs else None
