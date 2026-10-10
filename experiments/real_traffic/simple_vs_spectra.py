"""Adverse same-host comparison between SPECTRA and a cheap exact traversal.

This is a secondary non-promotion audit. It discovers the current prepared SPECTRA
runtime from a short explicit module list, records the selected API, and refuses to
continue unless a sample query agrees with the independently retained case.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import importlib
import inspect
import json
from pathlib import Path
import random
import statistics
import tempfile
import time
from typing import Any, Callable, Sequence

from experiments.real_traffic import TrafficCase, verify_contradiction, verify_labels
from experiments.real_traffic.simple_traversal import (
    SimpleTraversalRuntime,
    build_simple_traversal,
)

MAX_BYTES = 512 * 1024 * 1024
MAX_WORK = 100_000_000
MODULES = (
    "experiments.real_traffic.compiled_support",
    "experiments.real_traffic.quotient_session",
    "spectra.cnf.quotient_query",
)


def mapped_call(function: Callable, context: dict[str, Any]) -> Any:
    signature = inspect.signature(function)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    aliases = {
        "n": "vertices",
        "nvars": "vertices",
        "number_of_vertices": "vertices",
        "k": "palette",
        "colors": "palette",
        "colours": "palette",
        "restrictions": "query",
        "assumptions": "query",
        "query_bank": "queries",
        "batch": "queries",
        "max_build_bytes": "max_bytes",
        "max_state_bytes": "max_bytes",
        "state_bytes": "max_bytes",
        "work": "max_work",
        "work_limit": "max_work",
    }
    for name, parameter in signature.parameters.items():
        key = aliases.get(name, name)
        if key in context:
            value = context[key]
        elif parameter.default is not inspect._empty:
            continue
        else:
            raise TypeError(f"unmapped required parameter {name} in {signature}")
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        elif parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            kwargs[name] = value
        else:
            raise TypeError(
                f"unsupported variadic parameter {name} in {signature}"
            )
    return function(*args, **kwargs)


def status_and_labels(value: Any) -> tuple[str, bytes, str | None]:
    if isinstance(value, dict):
        status = value.get("status")
        labels = value.get("labels", value.get("witness", b""))
        reason = value.get("reason")
    else:
        status = getattr(value, "status", None)
        labels = getattr(value, "labels", getattr(value, "witness", b""))
        reason = getattr(value, "reason", None)
        if status is None and isinstance(value, tuple) and len(value) >= 2:
            labels, status = value[0], value[1]
    status_text = str(status).upper()
    reason_text = None if reason is None else str(reason)
    if "UNSAT" in status_text:
        normalized = "UNSAT"
    elif "SAT" in status_text and "UNSAT" not in status_text:
        normalized = "SAT"
    elif reason_text in {
        "exhausted",
        "base_contradiction",
        "restriction_conflict",
        "kernel_contradiction",
        "contradiction",
    }:
        normalized = "UNSAT"
    else:
        normalized = "UNKNOWN"
    if labels is None:
        payload = b""
    elif isinstance(labels, bytes):
        payload = labels
    else:
        payload = bytes(labels)
    return normalized, payload, reason_text


def extract_results(value: Any, count: int) -> Sequence[Any] | None:
    if isinstance(value, dict):
        for key in ("results", "queries", "answers", "rows"):
            candidate = value.get(key)
            if isinstance(candidate, (list, tuple)) and len(candidate) == count:
                return candidate
    for key in ("results", "answers", "rows"):
        candidate = getattr(value, key, None)
        if isinstance(candidate, (list, tuple)) and len(candidate) == count:
            return candidate
    if isinstance(value, (list, tuple)) and len(value) == count:
        return value
    return None


@dataclass(frozen=True)
class CandidateSpec:
    module: str
    builder: str
    runtime: str
    prepare: str
    solve: str
    batch: bool
    library: str


class Candidate:
    def __init__(self, spec: CandidateSpec, runtime: Any):
        self.spec = spec
        self.runtime = runtime

    def context(self, case: TrafficCase, *, query=(), queries=()) -> dict[str, Any]:
        return {
            "vertices": case.vertices,
            "palette": case.palette,
            "edges": case.edges,
            "masks": case.masks,
            "case": case,
            "mode": "scc",
            "max_bytes": MAX_BYTES,
            "max_work": MAX_WORK,
            "query": query,
            "queries": queries,
        }

    def prepare(self, case: TrafficCase) -> Any:
        return mapped_call(
            getattr(self.runtime, self.spec.prepare),
            self.context(case),
        )

    def run(
        self,
        prepared: Any,
        case: TrafficCase,
        ordered_queries: tuple[tuple[tuple[int, int], ...], ...],
    ) -> list[Any]:
        method = getattr(prepared, self.spec.solve)
        if self.spec.batch:
            value = mapped_call(
                method,
                self.context(case, queries=ordered_queries),
            )
            results = extract_results(value, len(ordered_queries))
            if results is None:
                raise AssertionError(
                    "batch API did not return one result per query"
                )
            return list(results)
        return [
            mapped_call(method, self.context(case, query=query))
            for query in ordered_queries
        ]


def discover_candidate(case: TrafficCase, build_root: Path) -> Candidate:
    failures: list[str] = []
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            failures.append(f"{module_name}: import: {error!r}")
            continue
        builders = sorted(
            (name, value)
            for name, value in vars(module).items()
            if inspect.isfunction(value) and name.startswith("build_")
        )
        runtimes = sorted(
            (name, value)
            for name, value in vars(module).items()
            if inspect.isclass(value) and name.endswith("Runtime")
        )
        for builder_name, builder in builders:
            directory = build_root / (
                module_name.rsplit(".", 1)[-1] + "-" + builder_name
            )
            directory.mkdir(parents=True, exist_ok=False)
            try:
                library = mapped_call(
                    builder,
                    {
                        "directory": directory,
                        "destination": directory,
                        "path": directory,
                        "out": directory,
                    },
                )
            except Exception as error:
                failures.append(
                    f"{module_name}.{builder_name}: build: {error!r}"
                )
                continue
            for runtime_name, runtime_class in runtimes:
                try:
                    runtime = runtime_class(library)
                except Exception as error:
                    failures.append(
                        f"{module_name}.{builder_name}/{runtime_name}: "
                        f"runtime: {error!r}"
                    )
                    continue
                for prepare_name in ("prepare", "compile", "create"):
                    if not hasattr(runtime, prepare_name):
                        continue
                    try:
                        temporary_spec = CandidateSpec(
                            module_name,
                            builder_name,
                            runtime_name,
                            prepare_name,
                            "",
                            False,
                            str(library),
                        )
                        candidate = Candidate(temporary_spec, runtime)
                        prepared = candidate.prepare(case)
                    except Exception as error:
                        failures.append(
                            f"{module_name}.{runtime_name}.{prepare_name}: "
                            f"{error!r}"
                        )
                        continue
                    try:
                        for solve_name in (
                            "solve_many",
                            "solve_batch",
                            "run_many",
                            "query_many",
                            "session",
                            "solve",
                            "query",
                            "run",
                        ):
                            if not hasattr(prepared, solve_name):
                                continue
                            batch = solve_name in {
                                "solve_many",
                                "solve_batch",
                                "run_many",
                                "query_many",
                                "session",
                            }
                            spec = CandidateSpec(
                                module_name,
                                builder_name,
                                runtime_name,
                                prepare_name,
                                solve_name,
                                batch,
                                str(library),
                            )
                            selected = Candidate(spec, runtime)
                            try:
                                results = selected.run(
                                    prepared, case, (case.queries[0],)
                                )
                                status, labels, _ = status_and_labels(results[0])
                                expected = case.statuses[0]
                                if status != expected:
                                    raise AssertionError(
                                        f"sample status {status} differs "
                                        f"from {expected}"
                                    )
                                if status == "SAT" and not verify_labels(
                                    case.edges,
                                    case.masks,
                                    labels,
                                    case.queries[0],
                                ):
                                    raise AssertionError(
                                        "sample witness is invalid"
                                    )
                                return selected
                            except Exception as error:
                                failures.append(
                                    f"{module_name}.{runtime_name}."
                                    f"{solve_name}: {error!r}"
                                )
                    finally:
                        if hasattr(prepared, "close"):
                            prepared.close()
    raise RuntimeError(
        "No current SPECTRA runtime passed the sample query:\n"
        + "\n".join(failures[-30:])
    )


def order_indices(case: TrafficCase, order: str) -> list[int]:
    indices = list(range(len(case.queries)))
    if order == "reverse":
        indices.reverse()
    elif order == "shuffle":
        random.Random(int(case.case_sha256[:16], 16)).shuffle(indices)
    elif order != "chronological":
        raise ValueError(order)
    return indices


def check_results(
    case: TrafficCase, indices: Sequence[int], values: Sequence[Any]
) -> None:
    if len(values) != len(indices):
        raise AssertionError("result inventory differs")
    for index, value in zip(indices, values):
        status, labels, _ = status_and_labels(value)
        expected = case.statuses[index]
        if status != expected:
            raise AssertionError(
                f"status differs at query {index}: {status} vs {expected}"
            )
        query = case.queries[index]
        if status == "SAT":
            if not verify_labels(case.edges, case.masks, labels, query):
                raise AssertionError(f"invalid witness at query {index}")
        else:
            certificate = case.contradictions[index]
            if certificate is None:
                raise AssertionError(
                    f"missing contradiction at query {index}"
                )
            verify_contradiction(
                case.edges, case.masks, query, certificate
            )


def run_candidate(
    candidate: Candidate, case: TrafficCase, indices: Sequence[int]
) -> dict[str, Any]:
    queries = tuple(case.queries[index] for index in indices)
    started = time.perf_counter_ns()
    prepared = candidate.prepare(case)
    setup_ns = time.perf_counter_ns() - started
    try:
        query_started = time.perf_counter_ns()
        values = candidate.run(prepared, case, queries)
        check_results(case, indices, values)
        query_ns = time.perf_counter_ns() - query_started
    finally:
        dispose_started = time.perf_counter_ns()
        if hasattr(prepared, "close"):
            prepared.close()
        dispose_ns = time.perf_counter_ns() - dispose_started
    return {
        "setup_ns": setup_ns,
        "query_ns": query_ns,
        "dispose_ns": dispose_ns,
        "session_ns": setup_ns + query_ns + dispose_ns,
    }


def run_simple(
    runtime: SimpleTraversalRuntime,
    case: TrafficCase,
    indices: Sequence[int],
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    prepared = runtime.prepare(
        case.edges, case.masks, max_bytes=MAX_BYTES
    )
    setup_ns = time.perf_counter_ns() - started
    try:
        query_started = time.perf_counter_ns()
        results = []
        for index in indices:
            result = prepared.solve(case.queries[index])
            results.append({
                "status": result.status,
                "labels": result.labels,
            })
        check_results(case, indices, results)
        query_ns = time.perf_counter_ns() - query_started
    finally:
        dispose_started = time.perf_counter_ns()
        prepared.close()
        dispose_ns = time.perf_counter_ns() - dispose_started
    return {
        "setup_ns": setup_ns,
        "query_ns": query_ns,
        "dispose_ns": dispose_ns,
        "session_ns": setup_ns + query_ns + dispose_ns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    case = TrafficCase.read(args.case)
    args.out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(
            prefix="spectra-simple-audit-") as temporary:
        build_root = Path(temporary)
        candidate = discover_candidate(case, build_root)
        simple_library = build_simple_traversal(build_root / "simple")
        simple = SimpleTraversalRuntime(simple_library)
        rows = []
        schedule = [
            (round_index, order)
            for round_index in range(args.rounds)
            for order in ("chronological", "reverse", "shuffle")
        ]
        random.Random(20261010).shuffle(schedule)
        for round_index, order in schedule:
            indices = order_indices(case, order)
            for arm in ("spectra", "simple"):
                gc.collect()
                result = (
                    run_candidate(candidate, case, indices)
                    if arm == "spectra"
                    else run_simple(simple, case, indices)
                )
                rows.append({
                    "round": round_index,
                    "order": order,
                    "arm": arm,
                    **result,
                })
    grouped = {}
    for arm in ("spectra", "simple"):
        values = [
            row["session_ns"] for row in rows if row["arm"] == arm
        ]
        grouped[arm] = {
            "mean_ns": statistics.fmean(values),
            "median_ns": statistics.median(values),
            "minimum_ns": min(values),
            "maximum_ns": max(values),
        }
    report = {
        "schema": "spectra.real_traffic.simple_comparison.v1",
        "case_sha256": case.case_sha256,
        "queries": len(case.queries),
        "candidate": asdict(candidate.spec),
        "rows": rows,
        "summary": grouped,
        "spectra_over_simple": (
            grouped["spectra"]["mean_ns"]
            / grouped["simple"]["mean_ns"]
        ),
        "decision": (
            "substantial"
            if grouped["spectra"]["mean_ns"]
            <= 0.70 * grouped["simple"]["mean_ns"]
            else "not_substantial"
        ),
    }
    (args.out / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.out / "decision.txt").write_text(
        report["decision"] + "\n"
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print("SPECTRA/simple:", report["spectra_over_simple"])


if __name__ == "__main__":
    main()
