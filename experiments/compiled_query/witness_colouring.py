"""Deterministic witness colourings for binary-list transfer workloads.

The routines in this module use graph structure only. They never call a SAT
solver, inspect request answers, or consult benchmark timings.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
from pathlib import Path
import struct

from experiments.critical_clique_coloring.core import (
    Graph,
    verify_coloring,
)
from experiments.wap_support.workload import deterministic_dsatur

SCHEMA = "spectra.witness_topology.v1"
DEFAULT_RESTARTS = 256
DEFAULT_MAX_WITNESS_COLOURS = 63


@dataclass(frozen=True)
class WitnessTopology:
    graph: Graph
    colouring: tuple[int, ...]
    receipt: dict


def colour_count(colouring) -> int:
    return max(colouring, default=-1) + 1


def _verify(graph: Graph, colouring) -> tuple[int, ...]:
    result = tuple(colouring)
    verify_coloring(graph, result, colour_count(result))
    return result


def smallest_last_colouring(graph: Graph) -> tuple[int, ...]:
    """Greedily colour the reverse minimum-degree elimination order."""
    degree = [len(row) for row in graph.adjacency]
    alive = [True] * graph.n
    queue = [(degree[vertex], vertex) for vertex in range(graph.n)]
    heapq.heapify(queue)
    removed: list[int] = []
    while queue:
        observed, vertex = heapq.heappop(queue)
        if not alive[vertex] or observed != degree[vertex]:
            continue
        alive[vertex] = False
        removed.append(vertex)
        for neighbour in graph.adjacency[vertex]:
            if alive[neighbour]:
                degree[neighbour] -= 1
                heapq.heappush(queue, (degree[neighbour], neighbour))

    colours = [-1] * graph.n
    for vertex in reversed(removed):
        used = {
            colours[neighbour]
            for neighbour in graph.adjacency[vertex]
            if colours[neighbour] >= 0
        }
        colour = 0
        while colour in used:
            colour += 1
        colours[vertex] = colour
    return _verify(graph, colours)


def _tie_rank(graph: Graph, trial: int, vertex: int) -> int:
    payload = (
        b"SPECTRA witness DSATUR v1\0"
        + bytes.fromhex(graph.sha256)
        + struct.pack("<II", trial, vertex)
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def seeded_dsatur(graph: Graph, trial: int) -> tuple[int, ...]:
    """DSATUR with a cryptographic, version-stable structural tie order."""
    if type(trial) is not int or not 0 <= trial < 1 << 32:
        raise ValueError("trial must be an unsigned 32-bit integer")
    colours = [-1] * graph.n
    saturation = [set() for _ in range(graph.n)]
    degree = [len(row) for row in graph.adjacency]
    version = [0] * graph.n
    ranks = [_tie_rank(graph, trial, vertex) for vertex in range(graph.n)]
    queue = [
        (0, -degree[vertex], ranks[vertex], vertex, 0)
        for vertex in range(graph.n)
    ]
    heapq.heapify(queue)

    for _ in range(graph.n):
        while True:
            negative_saturation, negative_degree, _rank, vertex, observed = (
                heapq.heappop(queue)
            )
            if (
                colours[vertex] < 0
                and observed == version[vertex]
                and -negative_saturation == len(saturation[vertex])
                and -negative_degree == degree[vertex]
            ):
                break
        colour = 0
        while colour in saturation[vertex]:
            colour += 1
        colours[vertex] = colour
        for neighbour in graph.adjacency[vertex]:
            if colours[neighbour] < 0 and colour not in saturation[neighbour]:
                saturation[neighbour].add(colour)
                version[neighbour] += 1
                heapq.heappush(
                    queue,
                    (
                        -len(saturation[neighbour]),
                        -degree[neighbour],
                        ranks[neighbour],
                        neighbour,
                        version[neighbour],
                    ),
                )
    return _verify(graph, colours)


def best_witness_colouring(
    graph: Graph,
    *,
    restarts: int = DEFAULT_RESTARTS,
) -> tuple[tuple[int, ...], dict]:
    """Return the best deterministic colouring from a fixed structural portfolio."""
    if type(restarts) is not int or not 0 <= restarts <= 4096:
        raise ValueError("restarts must be an integer in [0, 4096]")

    candidates: list[tuple[str, int | None, tuple[int, ...]]] = [
        ("deterministic-dsatur", None, _verify(graph, deterministic_dsatur(graph))),
        ("smallest-last", None, smallest_last_colouring(graph)),
    ]
    for trial in range(restarts):
        candidates.append(("seeded-dsatur", trial, seeded_dsatur(graph, trial)))

    method, trial, colouring = min(
        candidates,
        key=lambda item: (
            colour_count(item[2]),
            item[2],
            item[0],
            -1 if item[1] is None else item[1],
        ),
    )
    receipt = {
        "method": method,
        "trial": trial,
        "restarts": restarts,
        "colour_count": colour_count(colouring),
        "colouring_sha256": hashlib.sha256(bytes(colouring)).hexdigest(),
    }
    return colouring, receipt


def degree_capped_subgraph(graph: Graph, *, degree_cap: int) -> Graph:
    """Retain a deterministic maximal edge subset with bounded degree."""
    if type(degree_cap) is not int or not 0 <= degree_cap <= 1_000_000:
        raise ValueError("degree_cap must be a bounded nonnegative integer")
    prefix = (
        b"SPECTRA degree-capped topology v1\0"
        + bytes.fromhex(graph.sha256)
        + struct.pack("<II", graph.n, degree_cap)
    )

    def edge_key(edge: tuple[int, int]):
        left, right = edge
        return (
            hashlib.sha256(prefix + struct.pack("<II", left, right)).digest(),
            left,
            right,
        )

    degree = [0] * graph.n
    retained: list[tuple[int, int]] = []
    for left, right in sorted(graph.edges, key=edge_key):
        if degree[left] < degree_cap and degree[right] < degree_cap:
            retained.append((left, right))
            degree[left] += 1
            degree[right] += 1

    edges = tuple(sorted(retained))
    adjacency = [set() for _ in range(graph.n)]
    digest = hashlib.sha256(prefix)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
        digest.update(struct.pack("<II", left, right))
    return Graph(
        name=f"{graph.name}#degree-{degree_cap}",
        path=Path(graph.path),
        n=graph.n,
        edges=edges,
        adjacency=tuple(frozenset(row) for row in adjacency),
        sha256=digest.hexdigest(),
    )


def prepare_witness_topology(
    graph: Graph,
    *,
    max_witness_colours: int = DEFAULT_MAX_WITNESS_COLOURS,
    restarts: int = DEFAULT_RESTARTS,
) -> WitnessTopology:
    """Keep the full graph when admissible, otherwise use a bounded-degree subset."""
    if (
        type(max_witness_colours) is not int
        or not 1 <= max_witness_colours <= 63
    ):
        raise ValueError("max_witness_colours must be an integer in [1, 63]")

    full_colouring, full_receipt = best_witness_colouring(
        graph, restarts=restarts
    )
    if colour_count(full_colouring) <= max_witness_colours:
        prepared = graph
        colouring = full_colouring
        selected = full_receipt
        mode = "full"
        degree_cap = None
    else:
        degree_cap = max_witness_colours - 1
        prepared = degree_capped_subgraph(graph, degree_cap=degree_cap)
        colouring, selected = best_witness_colouring(
            prepared, restarts=restarts
        )
        if colour_count(colouring) > max_witness_colours:
            raise AssertionError(
                "bounded-degree topology exceeded its greedy colour guarantee"
            )
        mode = "degree-capped"

    maximum_degree = max(
        (len(row) for row in prepared.adjacency),
        default=0,
    )
    receipt = {
        "schema": SCHEMA,
        "topology_mode": mode,
        "original_graph_sha256": graph.sha256,
        "prepared_topology_sha256": prepared.sha256,
        "vertices": graph.n,
        "original_edges": len(graph.edges),
        "retained_edges": len(prepared.edges),
        "dropped_edges": len(graph.edges) - len(prepared.edges),
        "degree_cap": degree_cap,
        "maximum_retained_degree": maximum_degree,
        "full_graph_best_colours": full_receipt["colour_count"],
        "full_graph_best_method": full_receipt["method"],
        "full_graph_best_trial": full_receipt["trial"],
        "selected_method": selected["method"],
        "selected_trial": selected["trial"],
        "restarts": restarts,
        "witness_colours": colour_count(colouring),
        "witness_sha256": hashlib.sha256(bytes(colouring)).hexdigest(),
    }
    _verify(prepared, colouring)
    if maximum_degree > max_witness_colours - 1 and mode == "degree-capped":
        raise AssertionError("degree-capped topology exceeds its declared cap")
    return WitnessTopology(prepared, colouring, receipt)
