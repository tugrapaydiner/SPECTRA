"""Run a fixed exposed SAT-only panel with a matched CaDiCaL backend.

The four instances were already public in the SAT Competition 2026 Main results.
They are development diagnostics, not prospective confirmation. Selection was
frozen from the winning solver's public score table before this harness downloaded
or inspected the formulas. Every candidate answer must be SAT and is checked on
the exact admitted original clauses.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter_ns
from typing import Iterable
from urllib.request import Request, urlopen

from experiments.general_competition.adaptive_backend import (
    POLICY,
    RESULT_CLASSIFICATIONS,
    choose_frontend,
    classify_comparison,
    compare_adaptive_backend,
)
from experiments.general_competition.competition_input import admit_and_load_cnf
from experiments.general_competition.matched_backend import (
    ModelValidationError,
    prepare_matched_formula,
)
from experiments.general_competition.native_binary_core import (
    NativeBinaryCore,
    build_native_binary_core,
)

SCHEMA = "spectra.general_exposed_panel.v4"
BASE_URL = "https://benchmark-database.de/file/"
MAX_DOWNLOAD_BYTES = 512 << 20


@dataclass(frozen=True)
class PanelCase:
    instance_id: str
    reference_solver: str
    reference_seconds: str
    expected: str = "sat"


PANEL = (
    PanelCase("5de866bc7ff4dfde3cd682b5aca75894", "anders_satsuma-iter-kissat[main]", "2.12"),
    PanelCase("6cd46641ebd4c840b8f116d14ca17dfd", "anders_satsuma-iter-kissat[main]", "2.14"),
    PanelCase("af8248aa34c4803bfc0add954cf2e620", "anders_satsuma-iter-kissat[main]", "2.45"),
    PanelCase("ee44c3f464238366729bf2cb16eb412a", "anders_satsuma-iter-kissat[main]", "4.22"),
)


def _download(case: PanelCase, directory: Path) -> Path:
    if len(case.instance_id) != 32 or any(
        character not in "0123456789abcdef" for character in case.instance_id
    ):
        raise ValueError("panel instance id must be a lowercase 32-hex GBD id")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{case.instance_id}.cnf.xz"
    if destination.exists():
        return destination
    request = Request(
        BASE_URL + case.instance_id,
        headers={"User-Agent": "SPECTRA exposed SAT Competition development panel"},
    )
    total = 0
    try:
        with urlopen(request, timeout=180) as response, destination.open("xb") as output:
            while chunk := response.read(1 << 20):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise MemoryError("exposed panel source exceeds download limit")
                output.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if total == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("exposed panel download is empty")
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_samples(samples: list[dict]) -> dict:
    """Keep every cell while separating algorithm work from fallback variation."""
    counts = Counter(sample["result_classification"] for sample in samples)
    if set(counts) - set(RESULT_CLASSIFICATIONS):
        raise ValueError("sample has an unknown result classification")
    timed_classes = {
        "SPECTRA_NATIVE_WIN", "SPECTRA_FRONTEND_WIN", "UPSTREAM_FALLBACK_PARITY", "LOSS"
    }
    timed = [sample for sample in samples if sample["result_classification"] in timed_classes]
    if any(classify_comparison(sample) != sample["result_classification"] for sample in timed):
        raise ValueError("sample classification disagrees with its checked comparison evidence")
    frontend = [sample for sample in timed if sample["adaptive_decision"] == "reduce"]
    fallback = [sample for sample in timed if sample["result_classification"] == "UPSTREAM_FALLBACK_PARITY"]

    def ratios(rows: list[dict]) -> tuple[float | None, float | None]:
        values = [sample["hybrid_over_plain"] for sample in rows]
        if not values:
            return None, None
        return median(values), math.exp(math.fsum(math.log(value) for value in values) / len(values))

    overall_median, overall_geomean = ratios(timed)
    frontend_median, frontend_geomean = ratios(frontend)
    fallback_median, fallback_geomean = ratios(fallback)
    return {
        "classification_counts": {name: counts[name] for name in RESULT_CLASSIFICATIONS},
        "spectra_native_wins": counts["SPECTRA_NATIVE_WIN"],
        "spectra_frontend_wins": counts["SPECTRA_FRONTEND_WIN"],
        "upstream_fallback_parity": counts["UPSTREAM_FALLBACK_PARITY"],
        "cells": len(samples),
        "timed_cells": len(timed),
        "non_timing_cells": len(samples) - len(timed),
        "frontend_cells": len(frontend),
        "median_hybrid_over_plain": overall_median,
        "geometric_mean_hybrid_over_plain": overall_geomean,
        "frontend_median_hybrid_over_plain": frontend_median,
        "frontend_geometric_mean_hybrid_over_plain": frontend_geomean,
        "fallback_median_hybrid_over_plain": fallback_median,
        "fallback_geometric_mean_hybrid_over_plain": fallback_geomean,
        "decisions": dict(sorted(Counter(
            sample.get("adaptive_decision") or "not_run" for sample in samples
        ).items())),
    }


def _failure_sample(
    error: Exception,
    *,
    phase: str,
    repeat: int,
    decision: str | None = None,
    elapsed_ns: int | None = None,
    attempted: bool = False,
) -> dict:
    if isinstance(error, ModelValidationError):
        classification = "INVALID_MODEL"
    elif isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        classification = "TIMEOUT"
    elif isinstance(error, (MemoryError, OSError, subprocess.CalledProcessError)):
        classification = "RESOURCE_FAILURE"
    else:
        classification = "UNKNOWN"
    record = {
        "schema": "spectra.general_panel_failure.v1",
        "result_classification": classification,
        "adaptive_policy": POLICY,
        "adaptive_decision": decision,
        "accepted_for_timing": False,
        "attempted": attempted,
        "repeat": repeat,
        "execution_order": "plain-first" if repeat % 2 == 0 else "hybrid-first",
        "phase": phase,
        "error_type": type(error).__name__,
        "error": str(error),
        "elapsed_until_failure_ns": elapsed_ns,
    }
    if isinstance(error, ModelValidationError):
        record["rejected_evidence"] = error.evidence
    return record


def run_panel(
    directory: str | Path,
    *,
    repeats: int = 3,
    solver_name: str = "cadical195",
) -> dict:
    if type(repeats) is not int or repeats < 1 or repeats > 9:
        raise ValueError("repeats must be an exact integer in [1, 9]")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    native_directory = root / "native-binary-core"
    native_library = native_directory / "libspectra_general_core.so"
    initialization_error = None
    try:
        if not native_library.is_file():
            native_library = build_native_binary_core(native_directory)
        native_core = NativeBinaryCore(native_library)
    except Exception as error:
        initialization_error = error
    cases = []
    correctness_failure = False

    for case in PANEL:
        record = {
            "instance_id": case.instance_id,
            "source": BASE_URL + case.instance_id,
            "expected": case.expected,
            "reference_solver": case.reference_solver,
            "reference_seconds": case.reference_seconds,
        }
        samples = []
        if initialization_error is not None or correctness_failure:
            error = initialization_error or RuntimeError("not run after an earlier correctness failure")
            phase = "native_initialization" if initialization_error is not None else "blocked_by_correctness_failure"
            samples = [_failure_sample(error, phase=phase, repeat=repeat) for repeat in range(repeats)]
        else:
            phase = "acquisition"
            try:
                source = _download(case, root / "instances")
                record.update({"source_bytes": source.stat().st_size, "source_sha256": _sha256(source)})
                phase = "admission"
                admitted = admit_and_load_cnf(
                    source,
                    max_source_bytes=MAX_DOWNLOAD_BYTES,
                    max_dimacs_bytes=2 << 30,
                )
                record.update({
                    "dimacs_sha256": admitted.receipt.dimacs_sha256,
                    "semantic_sha256": admitted.receipt.semantic_sha256,
                    "variables": admitted.receipt.declared_variables,
                    "declared_clauses": admitted.receipt.declared_clauses,
                    "problem_class": admitted.receipt.problem_class,
                })
                phase = "preparation"
                prepared_started = perf_counter_ns()
                prepared = prepare_matched_formula(admitted.nvars, admitted.clauses)
                canonicalization_ns = perf_counter_ns() - prepared_started
                packing_started = perf_counter_ns()
                native_prepared = native_core.prepare_canonical(prepared.nvars, prepared.clauses)
                record.update({
                    "canonical_clauses": len(prepared.clauses),
                    "empty_clauses": prepared.empty_clauses,
                    "unit_clauses": prepared.unit_clauses,
                    "binary_clauses": prepared.binary_clauses,
                    "long_clauses": prepared.long_clauses,
                    "untimed_canonicalization_ns": canonicalization_ns,
                    "untimed_native_packing_ns": perf_counter_ns() - packing_started,
                })
            except Exception as error:
                samples = [_failure_sample(error, phase=phase, repeat=repeat) for repeat in range(repeats)]
            else:
                decision, _ = choose_frontend(prepared)
                for repeat in range(repeats):
                    order = "plain-first" if repeat % 2 == 0 else "hybrid-first"
                    if correctness_failure:
                        sample = _failure_sample(
                            RuntimeError("not run after an earlier correctness failure"),
                            phase="blocked_by_correctness_failure", repeat=repeat, decision=decision,
                        )
                    else:
                        started = perf_counter_ns()
                        try:
                            comparison = compare_adaptive_backend(
                                prepared,
                                solver_name=solver_name,
                                order=order,
                                native_core=native_core,
                                native_prepared=native_prepared,
                            )
                            sample = comparison.record()
                            sample.update({"repeat": repeat, "execution_order": order, "attempted": True})
                        except Exception as error:
                            sample = _failure_sample(
                                error, phase="comparison", repeat=repeat, decision=decision,
                                elapsed_ns=perf_counter_ns() - started, attempted=True,
                            )
                        if sample["result_classification"] in ("INVALID_MODEL", "INVALID_PROOF"):
                            correctness_failure = True
                    samples.append(sample)
        record.update({"samples": samples, **summarize_samples(samples)})
        cases.append(record)

    all_samples = [sample for case in cases for sample in case["samples"]]
    summary = summarize_samples(all_samples)
    return {
        "schema": SCHEMA,
        "status": (
            "FAILED_CORRECTNESS" if correctness_failure
            else "INCOMPLETE" if summary["non_timing_cells"] else "MEASURED"
        ),
        "solver": solver_name,
        "frontend": POLICY,
        "repeats": repeats,
        "selection": (
            "four exposed 2026 Main SAT instances with winning-solver public runtimes "
            "between 2.12 and 4.22 seconds; no formula replaced after acquisition"
        ),
        "metric": "post-admission fresh-construction solver stage",
        "preparation_scope": (
            "admission, one canonicalization, immutable ABI packing, and native builds "
            "are retained diagnostics outside the metric; every sample rebuilds the "
            "selected candidate path and the matched CaDiCaL control from scratch"
        ),
        "win_accounting": (
            "only checked distinct SPECTRA work can be an algorithm win; plain-route "
            "cells are UPSTREAM_FALLBACK_PARITY at every timing ratio; equality is "
            "a non-winning LOSS for the reducer; unverified UNSAT is UNKNOWN; "
            "all-cell ratios are diagnostics, not estimates of frontend speedup"
        ),
        **summary,
        "cases": cases,
        "claim_boundary": (
            "exposed SAT-only development result; same CaDiCaL fallback; not PAR-2, "
            "not standalone native SPECTRA-G, not a hidden holdout, not Main-track "
            "proof compliance, and not general leadership"
        ),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--solver", default="cadical195")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_panel(args.directory, repeats=args.repeats, solver_name=args.solver)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(text, end="")
    if report["status"] != "MEASURED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
