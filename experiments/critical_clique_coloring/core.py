"""Certified critical-clique quotient for exact vertex coloring.

This module is solver-independent.  It finds maximal true-twin classes (equal
closed neighborhoods), audits the resulting clique modules against the full
original graph, and reconstructs ordinary vertex colorings from weighted
quotient colorings.

A quotient vertex C has demand |C|.  A k-coloring of the original graph is
therefore equivalent to choosing exactly |C| colors for every quotient vertex,
with disjoint chosen-color sets on quotient edges.  The equivalence preserves
explicit color pins on original vertices when all pinned colors inside a class
are distinct.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Graph:
    name: str
    path: Path
    n: int
    edges: tuple[tuple[int, int], ...]
    adjacency: tuple[frozenset[int], ...]
    sha256: str

    @property
    def m(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class CriticalCliqueQuotient:
    classes: tuple[tuple[int, ...], ...]
    class_of: tuple[int, ...]
    weights: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    adjacency: tuple[frozenset[int], ...]
    certificate_sha256: str

    @property
    def n(self) -> int:
        return len(self.classes)

    @property
    def m(self) -> int:
        return len(self.edges)

    @property
    def vertices_removed(self) -> int:
        return len(self.class_of) - len(self.classes)


class GraphFormatError(ValueError):
    """Raised when an input graph violates the strict DIMACS contract."""


def parse_dimacs_col(path: str | Path, *, name: str | None = None) -> Graph:
    """Parse a simple undirected DIMACS ``.col`` graph strictly and deterministically."""
    source = Path(path).resolve()
    raw = source.read_bytes()
    text = raw.decode("utf-8", errors="strict")
    declared_n: int | None = None
    declared_m: int | None = None
    edge_set: set[tuple[int, int]] = set()

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line[0] in {"c", "#", "%"}:
            continue
        fields = line.split()
        tag = fields[0].lower()
        if tag == "p":
            if declared_n is not None:
                raise GraphFormatError(f"duplicate problem line at {lineno}")
            if len(fields) != 4 or fields[1].lower() not in {"edge", "edges", "col"}:
                raise GraphFormatError(f"unsupported problem line at {lineno}: {line!r}")
            declared_n, declared_m = int(fields[2]), int(fields[3])
            if declared_n < 0 or declared_m < 0:
                raise GraphFormatError("negative graph dimensions")
            continue
        if tag not in {"e", "a"} or len(fields) != 3:
            raise GraphFormatError(f"unsupported record at {lineno}: {line!r}")
        if declared_n is None:
            raise GraphFormatError(f"edge before problem line at {lineno}")
        u, v = int(fields[1]) - 1, int(fields[2]) - 1
        if not (0 <= u < declared_n and 0 <= v < declared_n):
            raise GraphFormatError(f"endpoint outside 1..{declared_n} at {lineno}")
        if u == v:
            raise GraphFormatError(f"self-loop at {lineno}")
        edge = (u, v) if u < v else (v, u)
        if edge in edge_set:
            raise GraphFormatError(f"duplicate edge {edge[0] + 1} {edge[1] + 1}")
        edge_set.add(edge)

    if declared_n is None or declared_m is None:
        raise GraphFormatError("missing problem line")
    if len(edge_set) != declared_m:
        raise GraphFormatError(f"declared {declared_m} edges but parsed {len(edge_set)}")

    adjacency = [set() for _ in range(declared_n)]
    for u, v in sorted(edge_set):
        adjacency[u].add(v)
        adjacency[v].add(u)
    return Graph(
        name=name or source.name,
        path=source,
        n=declared_n,
        edges=tuple(sorted(edge_set)),
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        sha256=sha256(raw).hexdigest(),
    )


def verify_clique(graph: Graph, vertices: Sequence[int], *, expected_size: int | None = None) -> None:
    clique = tuple(vertices)
    if expected_size is not None and len(clique) != expected_size:
        raise AssertionError(f"expected clique size {expected_size}, received {len(clique)}")
    if len(set(clique)) != len(clique):
        raise AssertionError("clique contains duplicate vertices")
    if any(v < 0 or v >= graph.n for v in clique):
        raise AssertionError("clique vertex outside graph")
    for index, u in enumerate(clique):
        for v in clique[index + 1 :]:
            if v not in graph.adjacency[u]:
                raise AssertionError(f"non-edge inside claimed clique: {u + 1}, {v + 1}")


def build_critical_clique_quotient(graph: Graph) -> CriticalCliqueQuotient:
    """Build and fully audit the maximal true-twin (critical-clique) quotient."""
    by_closed_neighborhood: dict[tuple[int, ...], list[int]] = {}
    for vertex in range(graph.n):
        signature = tuple(sorted((*graph.adjacency[vertex], vertex)))
        by_closed_neighborhood.setdefault(signature, []).append(vertex)

    classes = tuple(
        sorted(
            (tuple(sorted(vertices)) for vertices in by_closed_neighborhood.values()),
            key=lambda vertices: vertices[0],
        )
    )
    class_of = [-1] * graph.n
    for class_id, members in enumerate(classes):
        for vertex in members:
            if class_of[vertex] != -1:
                raise AssertionError("critical-clique partition overlaps")
            class_of[vertex] = class_id
    if any(class_id < 0 for class_id in class_of):
        raise AssertionError("critical-clique partition does not cover all vertices")

    quotient_edges: set[tuple[int, int]] = set()
    for u, v in graph.edges:
        left, right = class_of[u], class_of[v]
        if left != right:
            quotient_edges.add((left, right) if left < right else (right, left))

    quotient_adjacency = [set() for _ in classes]
    for left, right in quotient_edges:
        quotient_adjacency[left].add(right)
        quotient_adjacency[right].add(left)

    # O(n+m) complete module audit.  Every class must be a clique, and each
    # original member must see either every or no vertex of each outside class.
    for class_id, members in enumerate(classes):
        expected_external = frozenset(quotient_adjacency[class_id])
        member_set = set(members)
        for vertex in members:
            internal = graph.adjacency[vertex] & member_set
            if internal != member_set - {vertex}:
                raise AssertionError(f"class {class_id} is not a clique")
            counts: dict[int, int] = {}
            for neighbor in graph.adjacency[vertex]:
                other = class_of[neighbor]
                if other != class_id:
                    counts[other] = counts.get(other, 0) + 1
            observed = set()
            for other, count in counts.items():
                if count != len(classes[other]):
                    raise AssertionError(
                        f"nonuniform adjacency from vertex {vertex} to class {other}: "
                        f"{count}/{len(classes[other])}"
                    )
                observed.add(other)
            if frozenset(observed) != expected_external:
                raise AssertionError(f"members of class {class_id} have different external neighborhoods")

    certificate_material = [
        "spectra.critical_clique.v1",
        graph.sha256,
        str(graph.n),
        str(graph.m),
        ";".join(",".join(map(str, members)) for members in classes),
        ";".join(f"{u},{v}" for u, v in sorted(quotient_edges)),
    ]
    certificate = sha256("\n".join(certificate_material).encode("ascii")).hexdigest()
    return CriticalCliqueQuotient(
        classes=classes,
        class_of=tuple(class_of),
        weights=tuple(map(len, classes)),
        edges=tuple(sorted(quotient_edges)),
        adjacency=tuple(frozenset(neighbors) for neighbors in quotient_adjacency),
        certificate_sha256=certificate,
    )


def selected_colors_from_coloring(
    coloring: Sequence[int], quotient: CriticalCliqueQuotient
) -> tuple[frozenset[int], ...]:
    if len(coloring) != len(quotient.class_of):
        raise AssertionError("coloring length differs from original graph")
    return tuple(frozenset(coloring[v] for v in members) for members in quotient.classes)


def verify_weighted_quotient_coloring(
    quotient: CriticalCliqueQuotient,
    selected_colors: Sequence[Iterable[int]],
    color_count: int,
) -> tuple[frozenset[int], ...]:
    if len(selected_colors) != quotient.n:
        raise AssertionError("weighted coloring length differs from quotient")
    normalized = tuple(frozenset(colors) for colors in selected_colors)
    palette = set(range(color_count))
    for class_id, colors in enumerate(normalized):
        if len(colors) != quotient.weights[class_id]:
            raise AssertionError(
                f"class {class_id} selects {len(colors)} colors but has demand {quotient.weights[class_id]}"
            )
        if not colors <= palette:
            raise AssertionError(f"class {class_id} uses a color outside 0..{color_count - 1}")
    for left, right in quotient.edges:
        overlap = normalized[left] & normalized[right]
        if overlap:
            raise AssertionError(f"quotient edge {left},{right} shares colors {sorted(overlap)}")
    return normalized


def reconstruct_coloring(
    graph: Graph,
    quotient: CriticalCliqueQuotient,
    selected_colors: Sequence[Iterable[int]],
    color_count: int,
    *,
    pinned_vertex_colors: Mapping[int, int] | None = None,
) -> tuple[int, ...]:
    """Lift a weighted quotient model to a fully checked original coloring."""
    normalized = verify_weighted_quotient_coloring(quotient, selected_colors, color_count)
    pins = dict(pinned_vertex_colors or {})
    if any(v < 0 or v >= graph.n or c < 0 or c >= color_count for v, c in pins.items()):
        raise AssertionError("pin outside graph or palette")

    coloring = [-1] * graph.n
    for class_id, members in enumerate(quotient.classes):
        selected = set(normalized[class_id])
        used = set()
        for vertex in members:
            if vertex in pins:
                color = pins[vertex]
                if color not in selected:
                    raise AssertionError(f"pinned color {color} absent from class {class_id}")
                if color in used:
                    raise AssertionError(f"duplicate pinned color {color} inside class {class_id}")
                coloring[vertex] = color
                used.add(color)
        remaining_vertices = [vertex for vertex in members if coloring[vertex] < 0]
        remaining_colors = sorted(selected - used)
        if len(remaining_vertices) != len(remaining_colors):
            raise AssertionError("pins and selected colors do not admit a bijection")
        for vertex, color in zip(remaining_vertices, remaining_colors, strict=True):
            coloring[vertex] = color

    verify_coloring(graph, coloring, color_count, pinned_vertex_colors=pins)
    return tuple(coloring)


def verify_coloring(
    graph: Graph,
    coloring: Sequence[int],
    color_count: int,
    *,
    pinned_vertex_colors: Mapping[int, int] | None = None,
    allowed_colors: Sequence[Iterable[int]] | None = None,
) -> None:
    if len(coloring) != graph.n:
        raise AssertionError("coloring length differs from graph")
    if any(type(color) is not int or color < 0 or color >= color_count for color in coloring):
        raise AssertionError("coloring contains an invalid color")
    for u, v in graph.edges:
        if coloring[u] == coloring[v]:
            raise AssertionError(f"edge conflict at {u + 1},{v + 1} on color {coloring[u]}")
    if allowed_colors is not None:
        if len(allowed_colors) != graph.n:
            raise AssertionError("list count differs from graph")
        for vertex, allowed in enumerate(allowed_colors):
            if coloring[vertex] not in set(allowed):
                raise AssertionError(f"list violation at vertex {vertex + 1}")
    for vertex, color in (pinned_vertex_colors or {}).items():
        if coloring[vertex] != color:
            raise AssertionError(f"pin violation at vertex {vertex + 1}")


def brute_force_colorable(
    graph: Graph,
    color_count: int,
    *,
    pinned_vertex_colors: Mapping[int, int] | None = None,
) -> tuple[int, ...] | None:
    """Small-instance reference used only by public exhaustive tests."""
    pins = dict(pinned_vertex_colors or {})
    order = sorted(range(graph.n), key=lambda v: (-len(graph.adjacency[v]), v))
    coloring = [-1] * graph.n

    def search(position: int) -> bool:
        if position == len(order):
            return True
        vertex = order[position]
        choices = (pins[vertex],) if vertex in pins else range(color_count)
        for color in choices:
            if all(coloring[neighbor] != color for neighbor in graph.adjacency[vertex]):
                coloring[vertex] = color
                if search(position + 1):
                    return True
                coloring[vertex] = -1
        return False

    return tuple(coloring) if search(0) else None


def brute_force_weighted_colorable(
    quotient: CriticalCliqueQuotient,
    color_count: int,
    *,
    class_required_colors: Mapping[int, Iterable[int]] | None = None,
) -> tuple[frozenset[int], ...] | None:
    """Small-instance weighted reference used only by public tests."""
    required = {class_id: frozenset(colors) for class_id, colors in (class_required_colors or {}).items()}
    options: list[tuple[frozenset[int], ...]] = []
    palette = range(color_count)
    for class_id, weight in enumerate(quotient.weights):
        need = required.get(class_id, frozenset())
        if len(need) > weight:
            return None
        choices = tuple(
            frozenset(colors)
            for colors in combinations(palette, weight)
            if need <= set(colors)
        )
        if not choices:
            return None
        options.append(choices)

    assignment: list[frozenset[int] | None] = [None] * quotient.n
    order = sorted(range(quotient.n), key=lambda q: (-len(quotient.adjacency[q]), -quotient.weights[q], q))

    def search(position: int) -> bool:
        if position == len(order):
            return True
        class_id = order[position]
        for colors in options[class_id]:
            if all(
                assignment[neighbor] is None or not (colors & assignment[neighbor])
                for neighbor in quotient.adjacency[class_id]
            ):
                assignment[class_id] = colors
                if search(position + 1):
                    return True
                assignment[class_id] = None
        return False

    if not search(0):
        return None
    return tuple(colors if colors is not None else frozenset() for colors in assignment)
