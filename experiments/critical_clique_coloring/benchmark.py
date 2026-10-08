#!/usr/bin/env python3
"""Development benchmark for a certified critical-clique coloring compiler.

Primary arms use the same pinned CaDiCaL backend and the same complete maximum-
clique symmetry fixing:

* direct_assignment: classical one-hot assignment encoding on the original graph;
* direct_pop: partial-order encoding on the original graph (strong compact control);
* critical_clique_assignment: exact weighted coloring of the audited true-twin quotient.

Every measured run starts from graph bytes, includes parsing, quotient detection
where applicable, formula construction, solver construction, solve, full model
materialization, lifting, original-graph verification, and solver disposal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from core import (
    CriticalCliqueQuotient,
    Graph,
    build_critical_clique_quotient,
    parse_dimacs_col,
    reconstruct_coloring,
    verify_clique,
    verify_coloring,
)

SCHEMA = "spectra.critical_clique_coloring.benchmark.v1"
DEFAULT_SOLVER = "cadical195"
ARMS = ("direct_assignment", "direct_pop", "critical_clique_assignment")
BUFFER_SIZE = 32768


@dataclass
class RunReceipt:
    schema: str
    arm: str
    repeat: int
    graph_name: str
    graph_sha256: str
    solver: str
    status: str
    color_count: int
    parse_ns: int
    quotient_ns: int
    formula_and_solver_load_ns: int
    solve_ns: int
    materialize_and_verify_ns: int
    dispose_ns: int
    total_ns: int
    clauses: int
    variables: int
    auxiliary_variables: int
    original_vertices: int
    original_edges: int
    quotient_vertices: int | None
    quotient_edges: int | None
    quotient_certificate_sha256: str | None
    witness_sha256: str | None
    solver_stats: dict[str, int | float]
    peak_rss_kib: int


class FormulaEmitter:
    def __init__(self, solver, *, top_variable: int, buffer_size: int = BUFFER_SIZE):
        self.solver = solver
        self.top = top_variable
        self.clauses = 0
        self._buffer: list[list[int]] = []
        self._buffer_size = buffer_size

    def new_var(self) -> int:
        self.top += 1
        return self.top

    def add_clause(self, clause: Iterable[int]) -> None:
        normalized = list(clause)
        if not normalized:
            # PySAT accepts the empty clause and marks the formula inconsistent.
            self._buffer.append([])
        else:
            self._buffer.append(normalized)
        self.clauses += 1
        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def add_formula(self, clauses: Iterable[Sequence[int]]) -> None:
        for clause in clauses:
            self.add_clause(clause)

    def flush(self) -> None:
        if self._buffer:
            self.solver.append_formula(self._buffer, no_return=True)
            self._buffer.clear()


def _load_pysat():
    from pysat.card import CardEnc, EncType  # type: ignore
    from pysat.formula import IDPool  # type: ignore
    from pysat.solvers import Solver  # type: ignore

    return CardEnc, EncType, IDPool, Solver


def _assignment_var(entity: int, color: int, color_count: int) -> int:
    return entity * color_count + color + 1


def _threshold_var(vertex: int, threshold: int, color_count: int) -> int:
    # threshold is in 0..k-2 and the literal means color(vertex) > threshold.
    return vertex * (color_count - 1) + threshold + 1


def _add_exactly(
    emitter: FormulaEmitter,
    literals: Sequence[int],
    bound: int,
    *,
    vpool,
    CardEnc,
    EncType,
) -> None:
    if bound < 0 or bound > len(literals):
        emitter.add_clause(())
        return
    encoded = CardEnc.equals(lits=list(literals), bound=bound, vpool=vpool, encoding=EncType.seqcounter)
    emitter.add_formula(encoded.clauses)
    emitter.top = max(emitter.top, vpool.top)


def _add_at_most_one(emitter: FormulaEmitter, literals: Sequence[int], *, vpool, CardEnc, EncType) -> None:
    if len(literals) <= 1:
        return
    encoded = CardEnc.atmost(lits=list(literals), bound=1, vpool=vpool, encoding=EncType.seqcounter)
    emitter.add_formula(encoded.clauses)
    emitter.top = max(emitter.top, vpool.top)


def _add_direct_assignment(
    graph: Graph,
    color_count: int,
    pins: Mapping[int, int],
    emitter: FormulaEmitter,
    *,
    vpool,
    CardEnc,
    EncType,
) -> None:
    for vertex in range(graph.n):
        literals = [_assignment_var(vertex, color, color_count) for color in range(color_count)]
        emitter.add_clause(literals)
        _add_at_most_one(emitter, literals, vpool=vpool, CardEnc=CardEnc, EncType=EncType)
    for u, v in graph.edges:
        for color in range(color_count):
            emitter.add_clause((-_assignment_var(u, color, color_count), -_assignment_var(v, color, color_count)))
    for vertex, color in sorted(pins.items()):
        emitter.add_clause((_assignment_var(vertex, color, color_count),))


def _add_direct_pop(graph: Graph, color_count: int, pins: Mapping[int, int], emitter: FormulaEmitter) -> None:
    if color_count < 1:
        raise ValueError("color_count must be positive")
    if color_count == 1:
        for _edge in graph.edges:
            emitter.add_clause(())
        return
    for vertex in range(graph.n):
        for threshold in range(1, color_count - 1):
            emitter.add_clause(
                (-_threshold_var(vertex, threshold, color_count), _threshold_var(vertex, threshold - 1, color_count))
            )
    for u, v in graph.edges:
        # Both endpoints having color 0 is forbidden.
        emitter.add_clause((_threshold_var(u, 0, color_count), _threshold_var(v, 0, color_count)))
        # Both endpoints having an interior color is forbidden.
        for color in range(1, color_count - 1):
            emitter.add_clause(
                (
                    -_threshold_var(u, color - 1, color_count),
                    _threshold_var(u, color, color_count),
                    -_threshold_var(v, color - 1, color_count),
                    _threshold_var(v, color, color_count),
                )
            )
        # Both endpoints having color k-1 is forbidden.
        emitter.add_clause(
            (-_threshold_var(u, color_count - 2, color_count), -_threshold_var(v, color_count - 2, color_count))
        )
    for vertex, color in sorted(pins.items()):
        for threshold in range(color_count - 1):
            literal = _threshold_var(vertex, threshold, color_count)
            emitter.add_clause((literal if threshold < color else -literal,))


def _add_critical_clique_assignment(
    quotient: CriticalCliqueQuotient,
    color_count: int,
    pins: Mapping[int, int],
    emitter: FormulaEmitter,
    *,
    vpool,
    CardEnc,
    EncType,
) -> None:
    for class_id, weight in enumerate(quotient.weights):
        literals = [_assignment_var(class_id, color, color_count) for color in range(color_count)]
        _add_exactly(emitter, literals, weight, vpool=vpool, CardEnc=CardEnc, EncType=EncType)
    for left, right in quotient.edges:
        for color in range(color_count):
            emitter.add_clause(
                (-_assignment_var(left, color, color_count), -_assignment_var(right, color, color_count))
            )
    for vertex, color in sorted(pins.items()):
        emitter.add_clause((_assignment_var(quotient.class_of[vertex], color, color_count),))


def _solve_with_timeout(solver, timeout_seconds: float) -> bool | None:
    timer = threading.Timer(timeout_seconds, solver.interrupt)
    timer.daemon = True
    timer.start()
    try:
        return solver.solve_limited(expect_interrupt=True)
    finally:
        timer.cancel()
        try:
            solver.clear_interrupt()
        except Exception:
            pass


def _decode_direct_assignment(model: Sequence[int], graph: Graph, color_count: int) -> tuple[int, ...]:
    positive = {literal for literal in model if literal > 0}
    coloring = []
    for vertex in range(graph.n):
        colors = [
            color
            for color in range(color_count)
            if _assignment_var(vertex, color, color_count) in positive
        ]
        if len(colors) != 1:
            raise AssertionError(f"direct assignment model gives vertex {vertex + 1} colors {colors}")
        coloring.append(colors[0])
    return tuple(coloring)


def _decode_direct_pop(model: Sequence[int], graph: Graph, color_count: int) -> tuple[int, ...]:
    if color_count == 1:
        return tuple(0 for _ in range(graph.n))
    positive = {literal for literal in model if literal > 0}
    coloring = []
    for vertex in range(graph.n):
        values = [
            _threshold_var(vertex, threshold, color_count) in positive
            for threshold in range(color_count - 1)
        ]
        if any(values[index] and not values[index - 1] for index in range(1, len(values))):
            raise AssertionError(f"nonmonotone partial-order model at vertex {vertex + 1}")
        coloring.append(sum(values))
    return tuple(coloring)


def _decode_critical_clique_assignment(
    model: Sequence[int],
    graph: Graph,
    quotient: CriticalCliqueQuotient,
    color_count: int,
    pins: Mapping[int, int],
) -> tuple[int, ...]:
    positive = {literal for literal in model if literal > 0}
    selected = []
    for class_id in range(quotient.n):
        colors = frozenset(
            color
            for color in range(color_count)
            if _assignment_var(class_id, color, color_count) in positive
        )
        selected.append(colors)
    return reconstruct_coloring(
        graph,
        quotient,
        selected,
        color_count,
        pinned_vertex_colors=pins,
    )


def run_arm(
    graph_path: Path,
    graph_name: str,
    color_count: int,
    clique: Sequence[int],
    arm: str,
    repeat: int,
    *,
    solver_name: str,
    timeout_seconds: float,
) -> RunReceipt:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    CardEnc, EncType, IDPool, Solver = _load_pysat()
    total_start = time.perf_counter_ns()

    parse_start = time.perf_counter_ns()
    graph = parse_dimacs_col(graph_path, name=graph_name)
    parse_ns = time.perf_counter_ns() - parse_start
    verify_clique(graph, clique, expected_size=color_count)
    pins = {vertex: color for color, vertex in enumerate(clique)}

    quotient: CriticalCliqueQuotient | None = None
    quotient_start = time.perf_counter_ns()
    if arm == "critical_clique_assignment":
        quotient = build_critical_clique_quotient(graph)
        # A maximum clique must contain every member of each true-twin class it intersects.
        clique_set = set(clique)
        for vertex in clique:
            members = set(quotient.classes[quotient.class_of[vertex]])
            if not members <= clique_set:
                raise AssertionError("pinned maximum clique cuts through a critical clique")
    quotient_ns = time.perf_counter_ns() - quotient_start

    base_variables = (
        graph.n * color_count
        if arm == "direct_assignment"
        else graph.n * (color_count - 1)
        if arm == "direct_pop"
        else quotient.n * color_count  # type: ignore[union-attr]
    )
    formula_start = time.perf_counter_ns()
    solver = Solver(name=solver_name, use_timer=True)
    emitter = FormulaEmitter(solver, top_variable=base_variables)
    vpool = IDPool(start_from=base_variables + 1)
    if arm == "direct_assignment":
        _add_direct_assignment(
            graph,
            color_count,
            pins,
            emitter,
            vpool=vpool,
            CardEnc=CardEnc,
            EncType=EncType,
        )
    elif arm == "direct_pop":
        _add_direct_pop(graph, color_count, pins, emitter)
    else:
        assert quotient is not None
        _add_critical_clique_assignment(
            quotient,
            color_count,
            pins,
            emitter,
            vpool=vpool,
            CardEnc=CardEnc,
            EncType=EncType,
        )
    emitter.flush()
    formula_and_solver_load_ns = time.perf_counter_ns() - formula_start

    solve_start = time.perf_counter_ns()
    sat = _solve_with_timeout(solver, timeout_seconds)
    solve_ns = time.perf_counter_ns() - solve_start
    status = "SAT" if sat is True else "UNSAT" if sat is False else "TIMEOUT"

    witness_sha256: str | None = None
    materialize_start = time.perf_counter_ns()
    if sat is True:
        model = solver.get_model()
        if model is None:
            raise AssertionError("solver reported SAT without a model")
        if arm == "direct_assignment":
            coloring = _decode_direct_assignment(model, graph, color_count)
        elif arm == "direct_pop":
            coloring = _decode_direct_pop(model, graph, color_count)
        else:
            assert quotient is not None
            coloring = _decode_critical_clique_assignment(model, graph, quotient, color_count, pins)
        verify_coloring(graph, coloring, color_count, pinned_vertex_colors=pins)
        witness_bytes = b"".join(int(color).to_bytes(2, "little", signed=False) for color in coloring)
        witness_sha256 = hashlib.sha256(witness_bytes).hexdigest()
    materialize_and_verify_ns = time.perf_counter_ns() - materialize_start

    try:
        stats = dict(solver.accum_stats())
    except Exception:
        stats = {}
    dispose_start = time.perf_counter_ns()
    solver.delete()
    dispose_ns = time.perf_counter_ns() - dispose_start
    total_ns = time.perf_counter_ns() - total_start

    return RunReceipt(
        schema=SCHEMA,
        arm=arm,
        repeat=repeat,
        graph_name=graph.name,
        graph_sha256=graph.sha256,
        solver=solver_name,
        status=status,
        color_count=color_count,
        parse_ns=parse_ns,
        quotient_ns=quotient_ns,
        formula_and_solver_load_ns=formula_and_solver_load_ns,
        solve_ns=solve_ns,
        materialize_and_verify_ns=materialize_and_verify_ns,
        dispose_ns=dispose_ns,
        total_ns=total_ns,
        clauses=emitter.clauses,
        variables=emitter.top,
        auxiliary_variables=max(0, emitter.top - base_variables),
        original_vertices=graph.n,
        original_edges=graph.m,
        quotient_vertices=quotient.n if quotient is not None else None,
        quotient_edges=quotient.m if quotient is not None else None,
        quotient_certificate_sha256=quotient.certificate_sha256 if quotient is not None else None,
        witness_sha256=witness_sha256,
        solver_stats={str(key): value for key, value in stats.items()},
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


def _median_ns(rows: Sequence[RunReceipt], arm: str) -> float:
    values = [row.total_ns for row in rows if row.arm == arm and row.status == "SAT"]
    return statistics.median(values) if values else math.inf


def summarize(rows: Sequence[RunReceipt]) -> dict:
    medians = {arm: _median_ns(rows, arm) for arm in ARMS}
    candidate = medians["critical_clique_assignment"]
    ratios = {
        control: candidate / medians[control] if medians[control] not in {0, math.inf} else math.inf
        for control in ("direct_assignment", "direct_pop")
    }
    return {
        "schema": "spectra.critical_clique_coloring.summary.v1",
        "all_sat_and_verified": all(row.status == "SAT" and row.witness_sha256 for row in rows),
        "median_total_ns": medians,
        "candidate_ratios": ratios,
        "best_control": min(("direct_assignment", "direct_pop"), key=lambda arm: medians[arm]),
        "candidate_vs_best_control": candidate / min(medians["direct_assignment"], medians["direct_pop"]),
        "rows": len(rows),
    }


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--solver", default=DEFAULT_SOLVER)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    metadata_bytes = args.metadata.read_bytes()
    metadata = json.loads(metadata_bytes)
    instance = metadata["instances"][args.name]
    color_count = int(instance["chromatic_number"])
    clique = tuple(int(vertex) - 1 for vertex in instance["maximum_clique_1based"])

    rows: list[RunReceipt] = []
    arms = tuple(args.arms)
    for repeat in range(args.repeats):
        order = arms if repeat % 2 == 0 else tuple(reversed(arms))
        for arm in order:
            receipt = run_arm(
                args.graph,
                args.name,
                color_count,
                clique,
                arm,
                repeat,
                solver_name=args.solver,
                timeout_seconds=args.timeout_seconds,
            )
            rows.append(receipt)
            print(json.dumps(asdict(receipt), sort_keys=True), flush=True)

    result = {
        "schema": "spectra.critical_clique_coloring.result.v1",
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "core_source_sha256": hashlib.sha256((Path(__file__).parent / "core.py").read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "environment": _environment(),
        "summary": summarize(rows),
        "receipts": [asdict(row) for row in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True), flush=True)
    return 0 if result["summary"]["all_sat_and_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
