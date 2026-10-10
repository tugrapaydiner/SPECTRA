"""Small, independently checkable contradiction proofs for binary-list queries."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

from experiments.real_traffic.observer import (
    binary_clauses,
    restrictions_to_units,
    solve_binary,
)


class InvalidContradiction(ValueError):
    """A claimed implication contradiction does not follow from the inputs."""


@dataclass(frozen=True)
class ExactOutcome:
    status: str
    labels: tuple[int, ...] | None
    contradiction: dict | None


def _node(literal: int, variables: int) -> int:
    if type(literal) is not int or literal == 0 or abs(literal) > variables:
        raise ValueError("literal outside variable inventory")
    variable = abs(literal) - 1
    return 2 * variable + (1 if literal > 0 else 0)


def _literal(node: int) -> int:
    variable = node // 2 + 1
    return variable if node & 1 else -variable


def _graph_from_clauses(variables: int, clauses: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    adjacency = [set() for _ in range(2 * variables)]
    for clause in clauses:
        if len(clause) == 1:
            literal = _node(clause[0], variables)
            adjacency[literal ^ 1].add(literal)
        elif len(clause) == 2:
            left = _node(clause[0], variables)
            right = _node(clause[1], variables)
            adjacency[left ^ 1].add(right)
            adjacency[right ^ 1].add(left)
        else:
            raise ValueError("only unit and binary clauses are supported")
    return tuple(tuple(sorted(row)) for row in adjacency)


def _components(adjacency: Sequence[Sequence[int]]) -> tuple[int, ...]:
    reverse = [[] for _ in adjacency]
    for source, targets in enumerate(adjacency):
        for target in targets:
            reverse[target].append(source)
    seen = bytearray(len(adjacency))
    order: list[int] = []
    for root in range(len(adjacency)):
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
    component = [-1] * len(adjacency)
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
    return tuple(component)


def _path(adjacency: Sequence[Sequence[int]], start: int, target: int,
          component: Sequence[int]) -> tuple[int, ...]:
    wanted = component[start]
    previous = {start: -1}
    queue = deque([start])
    while queue and target not in previous:
        vertex = queue.popleft()
        for child in adjacency[vertex]:
            if component[child] == wanted and child not in previous:
                previous[child] = vertex
                queue.append(child)
    if target not in previous:
        raise AssertionError("contradictory SCC has no internal path")
    result = []
    cursor = target
    while cursor != -1:
        result.append(cursor)
        cursor = previous[cursor]
    return tuple(reversed(result))


def solve_with_proof(edges: Sequence[tuple[int, int]], masks: Sequence[int],
                     restrictions: Sequence[tuple[int, int]] = ()) -> ExactOutcome:
    """Return a checked witness or two implication paths proving contradiction."""
    variables = len(masks)
    clauses = [*binary_clauses(edges, masks)]
    clauses.extend((literal,) for literal in restrictions_to_units(restrictions, masks))
    adjacency = _graph_from_clauses(variables, clauses)
    component = _components(adjacency)
    conflict = next(
        (variable for variable in range(variables)
         if component[2 * variable] == component[2 * variable + 1]),
        None,
    )
    if conflict is None:
        labels = solve_binary(edges, masks, restrictions)
        if labels is None:
            raise AssertionError("independent solver disagrees with SCC result")
        return ExactOutcome("SAT", labels, None)
    negative = 2 * conflict
    positive = negative + 1
    certificate = {
        "schema": "spectra.real_traffic.contradiction.v1",
        "variable": conflict,
        "positive_to_negative": [
            _literal(node) for node in _path(adjacency, positive, negative, component)
        ],
        "negative_to_positive": [
            _literal(node) for node in _path(adjacency, negative, positive, component)
        ],
    }
    verify_contradiction(edges, masks, restrictions, certificate)
    return ExactOutcome("UNSAT", None, certificate)


def _literal_for_colour(vertex: int, colour: int, mask: int) -> int:
    choices = [bit for bit in range(mask.bit_length()) if mask >> bit & 1]
    if not choices or len(choices) > 2 or colour not in choices:
        raise InvalidContradiction("colour outside binary list")
    if len(choices) == 1:
        return vertex + 1
    return -(vertex + 1) if colour == choices[0] else vertex + 1


def _original_implications(edges: Sequence[tuple[int, int]], masks: Sequence[int],
                           restrictions: Sequence[tuple[int, int]]) -> set[tuple[int, int]]:
    variables = len(masks)
    implications: set[tuple[int, int]] = set()

    def add_unit(literal: int) -> None:
        node = _node(literal, variables)
        implications.add((node ^ 1, node))

    for vertex, mask in enumerate(masks):
        if type(mask) is not int or mask <= 0 or mask.bit_count() > 2:
            raise InvalidContradiction("invalid binary list")
        if mask.bit_count() == 1:
            add_unit(vertex + 1)
    for edge in edges:
        if (type(edge) is not tuple or len(edge) != 2
                or any(type(v) is not int or not 0 <= v < variables for v in edge)):
            raise InvalidContradiction("invalid edge")
        left_vertex, right_vertex = edge
        shared = masks[left_vertex] & masks[right_vertex]
        while shared:
            bit = shared & -shared
            shared ^= bit
            colour = bit.bit_length() - 1
            left_literal = _literal_for_colour(left_vertex, colour, masks[left_vertex])
            right_literal = _literal_for_colour(right_vertex, colour, masks[right_vertex])
            left = _node(left_literal, variables)
            right = _node(right_literal, variables)
            implications.add((left, right ^ 1))
            implications.add((right, left ^ 1))
    seen_vertices: set[int] = set()
    for restriction in restrictions:
        if type(restriction) is not tuple or len(restriction) != 2:
            raise InvalidContradiction("invalid restriction")
        vertex, allowed = restriction
        if type(vertex) is not int or not 0 <= vertex < variables or vertex in seen_vertices:
            raise InvalidContradiction("invalid or duplicate restriction vertex")
        if type(allowed) is not int:
            raise InvalidContradiction("invalid restriction mask")
        available = allowed & masks[vertex]
        if available.bit_count() != 1:
            raise InvalidContradiction("restriction does not select one available colour")
        seen_vertices.add(vertex)
        add_unit(_literal_for_colour(vertex, available.bit_length() - 1, masks[vertex]))
    return implications


def verify_contradiction(edges: Sequence[tuple[int, int]], masks: Sequence[int],
                         restrictions: Sequence[tuple[int, int]], certificate: dict) -> dict:
    """Check two opposite implication paths directly from original inputs."""
    fields = {"schema", "variable", "positive_to_negative", "negative_to_positive"}
    if type(certificate) is not dict or set(certificate) != fields:
        raise InvalidContradiction("unexpected certificate fields")
    if certificate["schema"] != "spectra.real_traffic.contradiction.v1":
        raise InvalidContradiction("unsupported certificate schema")
    variables = len(masks)
    variable = certificate["variable"]
    if type(variable) is not int or not 0 <= variable < variables:
        raise InvalidContradiction("certificate variable outside inventory")
    implications = _original_implications(edges, masks, restrictions)
    positive = variable + 1
    negative = -positive

    def check_path(name: str, start: int, target: int) -> int:
        path = certificate[name]
        if (type(path) is not list or len(path) < 2
                or path[0] != start or path[-1] != target
                or any(type(literal) is not int or literal == 0
                       or abs(literal) > variables for literal in path)):
            raise InvalidContradiction(f"bad {name} path")
        edges_checked = 0
        for left, right in zip(path, path[1:]):
            if (_node(left, variables), _node(right, variables)) not in implications:
                raise InvalidContradiction(f"missing implication in {name}")
            edges_checked += 1
        return edges_checked

    first = check_path("positive_to_negative", positive, negative)
    second = check_path("negative_to_positive", negative, positive)
    return {
        "schema": "spectra.real_traffic.contradiction.audit.v1",
        "valid": True,
        "variable": variable,
        "path_edges": first + second,
        "native_code_executed": False,
    }
