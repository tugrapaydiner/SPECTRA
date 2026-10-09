"""Complete-session solver arms for the WAP support-query study."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time
from typing import Sequence

from spectra.cnf.quotient_query import QuotientRuntime
from experiments.wap_support.workload import (
    WorkloadCase, compact_binary_clauses, model_to_labels,
    restrictions_to_assumptions,
)

ARMS = ("scc", "hybrid", "parity", "none", "minicard", "cadical195", "cache_minicard")
MAX_WORK = 10_000_000
MAX_BYTES = 512 * 1024 * 1024
CACHE_SIZE = 16


@dataclass(frozen=True)
class SessionResult:
    schema: str
    arm: str
    case_sha256: str
    order_name: str
    query_count: int
    status: str
    setup_ns: int
    query_ns: int
    dispose_ns: int
    session_ns: int
    per_query_ns: tuple[int, ...]
    output_sha256: str
    output_bytes: bytes
    unique_outputs: int
    cache_hits: int
    solver_stats: dict
    candidate_info: dict

    def metadata(self) -> dict:
        result = asdict(self)
        result.pop("output_bytes")
        result["per_query_ns"] = list(self.per_query_ns)
        return result


def order_indices(count: int, order_name: str, seed: int) -> tuple[int, ...]:
    if order_name == "forward":
        return tuple(range(count))
    if order_name == "reverse":
        return tuple(reversed(range(count)))
    if order_name == "shuffle":
        import random
        values = list(range(count))
        random.Random(seed).shuffle(values)
        return tuple(values)
    raise ValueError("unknown query order")


def _cache_eligible(labels: bytes, query: Sequence[Sequence[int]]) -> bool:
    return all((allowed >> labels[vertex]) & 1 for vertex, allowed in query)


def _run_session(case: WorkloadCase, runtime: QuotientRuntime, arm: str,
                 *, order_name: str, order_seed: int, validate_case: bool) -> SessionResult:
    if arm not in ARMS:
        raise ValueError("unknown arm")
    if type(validate_case) is not bool:
        raise TypeError("validate_case must be bool")
    if validate_case:
        case.validate()
    order = order_indices(case.query_count, order_name, order_seed)
    session_start = time.perf_counter_ns()
    setup_start = session_start
    outputs: list[bytes] = []
    query_times: list[int] = []
    cache_hits = 0
    stats: dict = {}
    candidate_info: dict = {}

    if arm in ("scc", "hybrid", "parity", "none"):
        prepared = runtime.prepare(case.n, case.palette, case.edges, masks=case.masks,
                                   mode=arm, max_build_bytes=MAX_BYTES)
        candidate_info = prepared.info
        setup_ns = time.perf_counter_ns() - setup_start
        for index in order:
            query = case.queries[index]
            start = time.perf_counter_ns()
            answer = prepared.solve(query, max_work=MAX_WORK,
                                    max_state_bytes=MAX_BYTES, core_first=True)
            elapsed = time.perf_counter_ns() - start
            if answer.status != "SAT_VERIFIED":
                raise AssertionError(f"{arm} failed accepted query {index}: {answer.reason}")
            outputs.append(answer.labels)
            query_times.append(elapsed)
        query_ns = sum(query_times)
        dispose_start = time.perf_counter_ns()
        prepared.close()
        dispose_ns = time.perf_counter_ns() - dispose_start
    else:
        from pysat.solvers import Solver
        checker = runtime.checker(case.n, case.palette, case.edges, case.masks, MAX_BYTES)
        clauses = compact_binary_clauses(_case_graph(case), case.masks)
        solver_name = "minicard" if arm in ("minicard", "cache_minicard") else "cadical195"
        solver = Solver(name=solver_name, bootstrap_with=clauses, use_timer=True)
        cache: list[bytes] = []
        setup_ns = time.perf_counter_ns() - setup_start
        try:
            for index in order:
                query = case.queries[index]
                start = time.perf_counter_ns()
                labels: bytes | None = None
                if arm == "cache_minicard":
                    for position in range(len(cache) - 1, -1, -1):
                        candidate = cache[position]
                        if _cache_eligible(candidate, query):
                            labels = candidate
                            cache.append(cache.pop(position))
                            cache_hits += 1
                            break
                if labels is None:
                    assumptions = restrictions_to_assumptions(query, case.masks)
                    if not solver.solve(assumptions=assumptions):
                        raise AssertionError(f"{arm} reported UNSAT for accepted query {index}")
                    model = solver.get_model()
                    if model is None:
                        raise AssertionError("SAT control returned no model")
                    labels = model_to_labels(model, case.masks)
                    if arm == "cache_minicard":
                        cache.append(labels)
                        del cache[:-CACHE_SIZE]
                if not checker.check(labels, query):
                    raise AssertionError(f"{arm} output fails original checker")
                outputs.append(labels)
                query_times.append(time.perf_counter_ns() - start)
            try:
                stats = {str(key): value for key, value in solver.accum_stats().items()}
            except Exception:
                stats = {}
            query_ns = sum(query_times)
        finally:
            dispose_start = time.perf_counter_ns()
            solver.delete()
            checker.close()
            dispose_ns = time.perf_counter_ns() - dispose_start

    session_ns = time.perf_counter_ns() - session_start
    output = b"".join(outputs)
    if len(output) != case.query_count * case.n:
        raise AssertionError("complete output bank has wrong size")
    return SessionResult(
        "spectra.wap_support.session.v1", arm, case.case_sha256, order_name,
        case.query_count, "COMPLETE", setup_ns, query_ns, dispose_ns, session_ns,
        tuple(query_times), hashlib.sha256(output).hexdigest(), output,
        len(set(outputs)), cache_hits, stats, candidate_info,
    )


def run_session(case: WorkloadCase, runtime: QuotientRuntime, arm: str,
                *, order_name: str, order_seed: int) -> SessionResult:
    """Canonical research endpoint: repeat the complete structural case audit."""
    return _run_session(case, runtime, arm, order_name=order_name,
                        order_seed=order_seed, validate_case=True)


def run_prevalidated_session(case: WorkloadCase, runtime: QuotientRuntime, arm: str,
                             *, order_name: str, order_seed: int) -> SessionResult:
    """Service endpoint for a hash-bound case validated before binary packing.

    This skips only ``WorkloadCase.validate``.  Fresh solver/index construction,
    every query, complete witness materialisation, original-input checking,
    diagnostics, and disposal remain identical to :func:`run_session`.
    """
    return _run_session(case, runtime, arm, order_name=order_name,
                        order_seed=order_seed, validate_case=False)


def _case_graph(case: WorkloadCase):
    """Minimal immutable graph facade needed by the compact encoder."""
    from experiments.critical_clique_coloring.core import Graph
    adjacency = [set() for _ in range(case.n)]
    for left, right in case.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return Graph(case.graph_name, __import__("pathlib").Path(case.graph_name), case.n,
                 case.edges, tuple(frozenset(x) for x in adjacency), case.graph_sha256)
