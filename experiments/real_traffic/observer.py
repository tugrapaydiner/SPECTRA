"""Independent exact observer for the binary-list query contract."""
from __future__ import annotations

from typing import Sequence


def verify_coloring_edges(edges: Sequence[tuple[int, int]], labels: Sequence[int]) -> bool:
    return all(
        0 <= left < len(labels) and 0 <= right < len(labels)
        and labels[left] != labels[right]
        for left, right in edges
    )


def verify_labels(edges: Sequence[tuple[int, int]], masks: Sequence[int],
                  labels: Sequence[int],
                  restrictions: Sequence[tuple[int, int]] = ()) -> bool:
    if len(masks) != len(labels):
        return False
    if any(color < 0 or not (masks[v] >> color & 1)
           for v, color in enumerate(labels)):
        return False
    if not verify_coloring_edges(edges, labels):
        return False
    return all(
        0 <= vertex < len(labels) and allowed >> labels[vertex] & 1
        for vertex, allowed in restrictions
    )


def masks_from_plans(plan_a: Sequence[int], plan_b: Sequence[int]) -> tuple[int, ...]:
    if len(plan_a) != len(plan_b):
        raise ValueError("plan geometry differs")
    return tuple((1 << left) | (1 << right) for left, right in zip(plan_a, plan_b))


def _literal_for_color(vertex: int, color: int, mask: int) -> int:
    choices = [bit for bit in range(mask.bit_length()) if mask >> bit & 1]
    if not choices or len(choices) > 2 or color not in choices:
        raise ValueError("color outside binary list")
    if len(choices) == 1:
        return vertex + 1
    return -(vertex + 1) if color == choices[0] else vertex + 1


def binary_clauses(edges: Sequence[tuple[int, int]],
                   masks: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    clauses: list[tuple[int, ...]] = []
    for vertex, mask in enumerate(masks):
        if mask.bit_count() == 1:
            clauses.append((vertex + 1,))
    for left, right in edges:
        shared = masks[left] & masks[right]
        while shared:
            bit = shared & -shared
            shared ^= bit
            color = bit.bit_length() - 1
            clauses.append((
                -_literal_for_color(left, color, masks[left]),
                -_literal_for_color(right, color, masks[right]),
            ))
    return tuple(clauses)


def restrictions_to_units(restrictions: Sequence[tuple[int, int]],
                          masks: Sequence[int]) -> tuple[int, ...]:
    units: list[int] = []
    for vertex, allowed in restrictions:
        if type(vertex) is not int or not 0 <= vertex < len(masks):
            raise ValueError("restriction vertex outside graph")
        available = allowed & masks[vertex]
        if available.bit_count() != 1:
            raise ValueError("restriction must select one available color")
        units.append(_literal_for_color(
            vertex, available.bit_length() - 1, masks[vertex]
        ))
    return tuple(units)


def solve_binary(edges: Sequence[tuple[int, int]], masks: Sequence[int],
                 restrictions: Sequence[tuple[int, int]] = ()) -> tuple[int, ...] | None:
    """Solve the exact binary-list instance with an independent iterative 2-SAT pass."""
    vertices = len(masks)
    if any(mask <= 0 or mask.bit_count() > 2 for mask in masks):
        raise ValueError("binary solver requires one or two colors per vertex")
    clauses = list(binary_clauses(edges, masks))
    clauses.extend((unit,) for unit in restrictions_to_units(restrictions, masks))
    size = 2 * vertices
    adjacency = [[] for _ in range(size)]
    reverse = [[] for _ in range(size)]

    def node(literal: int) -> int:
        variable = abs(literal) - 1
        return 2 * variable + (1 if literal > 0 else 0)

    def add(source: int, target: int) -> None:
        adjacency[source].append(target)
        reverse[target].append(source)

    for clause in clauses:
        if len(clause) == 1:
            literal = node(clause[0])
            add(literal ^ 1, literal)
        elif len(clause) == 2:
            left, right = node(clause[0]), node(clause[1])
            add(left ^ 1, right)
            add(right ^ 1, left)
        else:
            raise ValueError("not a 2-CNF clause")

    seen = bytearray(size)
    order: list[int] = []
    for root in range(size):
        if seen[root]:
            continue
        seen[root] = 1
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            vertex, offset = stack[-1]
            if offset == len(adjacency[vertex]):
                order.append(vertex)
                stack.pop()
                continue
            child = adjacency[vertex][offset]
            stack[-1] = (vertex, offset + 1)
            if not seen[child]:
                seen[child] = 1
                stack.append((child, 0))

    component = [-1] * size
    next_component = 0
    for root in reversed(order):
        if component[root] != -1:
            continue
        component[root] = next_component
        stack = [root]
        while stack:
            vertex = stack.pop()
            for child in reverse[vertex]:
                if component[child] == -1:
                    component[child] = next_component
                    stack.append(child)
        next_component += 1
    if any(component[2 * v] == component[2 * v + 1] for v in range(vertices)):
        return None

    truth = [component[2 * v + 1] > component[2 * v] for v in range(vertices)]

    def labels_from(bits: Sequence[bool]) -> tuple[int, ...]:
        labels = []
        for vertex, mask in enumerate(masks):
            choices = [color for color in range(mask.bit_length()) if mask >> color & 1]
            labels.append(choices[0] if len(choices) == 1 else choices[int(bits[vertex])])
        return tuple(labels)

    labels = labels_from(truth)
    if not verify_labels(edges, masks, labels, restrictions):
        labels = labels_from([not bit for bit in truth])
    if not verify_labels(edges, masks, labels, restrictions):
        raise AssertionError("2-SAT assignment reconstruction failed")
    return labels
