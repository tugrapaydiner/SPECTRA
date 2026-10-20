"""Run a fixed exposed SAT-only panel with a matched CaDiCaL backend.

The four instances were already public in the SAT Competition 2026 Main results.
They are development diagnostics, not prospective confirmation. Selection was
frozen from the winning solver's public score table before this harness downloaded
or inspected the formulas. Every candidate answer must be SAT and is checked on
the exact admitted original clauses.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Iterable
from urllib.request import Request, urlopen

from experiments.general_competition.competition_input import admit_and_load_cnf
from experiments.general_competition.matched_backend import compare_same_backend

SCHEMA = "spectra.general_exposed_panel.v1"
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
    cases = []

    for case in PANEL:
        source = _download(case, root / "instances")
        admitted = admit_and_load_cnf(
            source,
            max_source_bytes=MAX_DOWNLOAD_BYTES,
            max_dimacs_bytes=2 << 30,
        )
        samples = []
        for repeat in range(repeats):
            order = "plain-first" if repeat % 2 == 0 else "hybrid-first"
            comparison = compare_same_backend(
                admitted.nvars,
                admitted.clauses,
                solver_name=solver_name,
                order=order,
            )
            if not comparison.accepted_for_timing:
                raise RuntimeError("SAT-only exposed panel produced an unaccepted answer")
            samples.append(comparison.record())

        ratios = [sample["hybrid_over_plain"] for sample in samples]
        case_record = {
            "instance_id": case.instance_id,
            "source": BASE_URL + case.instance_id,
            "source_bytes": admitted.receipt.source_bytes,
            "source_sha256": _sha256(source),
            "dimacs_sha256": admitted.receipt.dimacs_sha256,
            "semantic_sha256": admitted.receipt.semantic_sha256,
            "variables": admitted.receipt.declared_variables,
            "declared_clauses": admitted.receipt.declared_clauses,
            "problem_class": admitted.receipt.problem_class,
            "expected": case.expected,
            "reference_solver": case.reference_solver,
            "reference_seconds": case.reference_seconds,
            "samples": samples,
            "median_hybrid_over_plain": median(ratios),
            "hybrid_wins": sum(ratio < 1.0 for ratio in ratios),
        }
        cases.append(case_record)

    all_ratios = [
        sample["hybrid_over_plain"]
        for case in cases
        for sample in case["samples"]
    ]
    return {
        "schema": SCHEMA,
        "status": "MEASURED",
        "solver": solver_name,
        "repeats": repeats,
        "selection": (
            "four exposed 2026 Main SAT instances with winning-solver public runtimes "
            "between 2.12 and 4.22 seconds; no formula replaced after acquisition"
        ),
        "metric": "post-admission fresh-construction solver stage",
        "median_hybrid_over_plain": median(all_ratios),
        "hybrid_wins": sum(ratio < 1.0 for ratio in all_ratios),
        "cells": len(all_ratios),
        "cases": cases,
        "claim_boundary": (
            "exposed SAT-only development result; same CaDiCaL backend; not PAR-2, "
            "not a hidden holdout, not Main-track proof compliance, and not general leadership"
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


if __name__ == "__main__":
    main()
