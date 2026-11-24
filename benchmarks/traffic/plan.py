"""Deterministic traffic-aware colour-plan construction."""
from __future__ import annotations

from decimal import Decimal
from typing import Sequence


def verify_coloring_edges(edges: Sequence[tuple[int, int]], labels: Sequence[int]) -> bool:
    return all(
        0 <= left < len(labels) and 0 <= right < len(labels)
        and labels[left] != labels[right]
        for left, right in edges
    )


def color_graph(adjacency: Sequence[frozenset[int]],
                weights: Sequence[Decimal] | None = None) -> tuple[int, ...]:
    vertices = len(adjacency)
    if weights is None:
        weights = (Decimal(0),) * vertices
    if len(weights) != vertices:
        raise ValueError("weight inventory differs")
    colors = [-1] * vertices
    saturation = [set() for _ in range(vertices)]
    degree = [len(row) for row in adjacency]
    for _ in range(vertices):
        candidates = [v for v in range(vertices) if colors[v] < 0]
        vertex = max(
            candidates,
            key=lambda v: (len(saturation[v]), weights[v], degree[v], -v),
        )
        color = 0
        while color in saturation[vertex]:
            color += 1
        colors[vertex] = color
        for neighbor in adjacency[vertex]:
            if colors[neighbor] < 0:
                saturation[neighbor].add(color)
    result = tuple(colors)
    edges = tuple(
        (left, right)
        for left in range(vertices)
        for right in adjacency[left]
        if left < right
    )
    if not verify_coloring_edges(edges, result):
        raise AssertionError("coloring construction failed")
    return result


def _maximum_weight_assignment(matrix: Sequence[Sequence[Decimal]]) -> tuple[int, ...]:
    """Deterministic square Hungarian algorithm with exact Decimal arithmetic."""
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("assignment matrix must be nonempty and square")
    largest = max(max(row) for row in matrix)
    costs = [[largest - value for value in row] for row in matrix]
    u = [Decimal(0)] * (size + 1)
    v = [Decimal(0)] * (size + 1)
    matching = [0] * (size + 1)
    previous = [0] * (size + 1)
    infinity = Decimal("Infinity")
    for row in range(1, size + 1):
        matching[0] = row
        minimum = [infinity] * (size + 1)
        used = [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            active_row = matching[column]
            delta = infinity
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[active_row - 1][candidate - 1]
                    - u[active_row] - v[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            if not delta.is_finite():
                raise AssertionError("assignment has no augmenting path")
            for candidate in range(size + 1):
                if used[candidate]:
                    u[matching[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matching[column] == 0:
                break
        while True:
            prior = previous[column]
            matching[column] = matching[prior]
            column = prior
            if column == 0:
                break
    assignment = [-1] * size
    for column in range(1, size + 1):
        assignment[matching[column] - 1] = column - 1
    if sorted(assignment) != list(range(size)):
        raise AssertionError("assignment is not a permutation")
    return tuple(assignment)


def align_target_plan(plan_a: Sequence[int], plan_b: Sequence[int],
                      traffic_weights: Sequence[Decimal]) -> tuple[int, ...]:
    """Align arbitrary colour names while minimizing measured traffic migration."""
    if len(plan_a) != len(plan_b) or len(plan_a) != len(traffic_weights) or not plan_a:
        raise ValueError("plan geometry differs")
    palette = max(max(plan_a), max(plan_b)) + 1
    agreement = [[Decimal(0) for _ in range(palette)] for _ in range(palette)]
    for reference, target, weight in zip(plan_a, plan_b, traffic_weights):
        if (type(reference) is not int or type(target) is not int
                or not 0 <= reference < palette or not 0 <= target < palette):
            raise ValueError("plan colour outside palette")
        if type(weight) is not Decimal or not weight.is_finite() or weight < 0:
            raise ValueError("traffic weight must be a finite nonnegative Decimal")
        agreement[target][reference] += weight
    mapping = _maximum_weight_assignment(agreement)
    return tuple(mapping[color] for color in plan_b)
