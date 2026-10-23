"""Open and measure the frozen mixed direct 2-SAT holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .cases import load_case
from .direct_twosat_benchmark import run as run_challenge
from .direct_twosat_holdout import (
    EXPECTED,
    GRAPH_INVENTORY,
    open_holdout,
)

COUNTS = (1, 8, 64, 1024)
REPEATS = 30
CLASSIFICATION = (
    "prospectively frozen mixed SAT/UNSAT direct-2SAT holdout; four unopened "
    "DIMACS topology families; synthetic witness/trap restrictions; no general "
    "SAT, graph-colouring, operator or production claim"
)


def _gmean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def run(
    graph_directory: Path,
    spectra_library: Path,
    direct_library: Path,
    destination: Path,
) -> dict:
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    inventory = open_holdout(
        graph_directory,
        destination / "opened-cases",
    )
    cases = []
    for name in GRAPH_INVENTORY:
        case, observed = load_case(
            destination / "opened-cases" / name / "case.json"
        )
        if observed != inventory["graphs"][name]["case_sha256"]:
            raise AssertionError(f"{name}: case hash differs after opening")
        cases.append(case)

    benchmark = run_challenge(
        Path(spectra_library).resolve(strict=True),
        Path(direct_library).resolve(strict=True),
        destination / "benchmark",
        repeats=REPEATS,
        cases=tuple(cases),
        counts=COUNTS,
        classification=CLASSIFICATION,
    )

    correctness_failures = []
    for record in benchmark["records"]:
        count = record["count"]
        expected_sat = sum(EXPECTED[:count])
        expected_unsat = count - expected_sat
        if (record["sat"], record["unsat"]) != (
            expected_sat,
            expected_unsat,
        ):
            correctness_failures.append({
                "case": record["case"],
                "count": count,
                "observed": [record["sat"], record["unsat"]],
                "expected": [expected_sat, expected_unsat],
            })

    by_count = {}
    for count in COUNTS:
        rows = [row for row in benchmark["records"] if row["count"] == count]
        ratios = [row["auto_over_best_direct"] for row in rows]
        by_count[str(count)] = {
            "ratios": {row["case"]: row["auto_over_best_direct"] for row in rows},
            "geomean": _gmean(ratios),
            "maximum": max(ratios),
            "auto_wins": sum(ratio < 1 for ratio in ratios),
            "direct_wins_or_ties": sum(ratio >= 1 for ratio in ratios),
        }

    gates = {
        "correctness": not correctness_failures,
        "q64_geomean_le_0_90": by_count["64"]["geomean"] <= 0.90,
        "q64_every_graph_le_1_05": by_count["64"]["maximum"] <= 1.05,
        "q1024_geomean_le_0_75": by_count["1024"]["geomean"] <= 0.75,
        "q1024_every_graph_le_0_90": by_count["1024"]["maximum"] <= 0.90,
    }
    verdict = {
        "schema": "spectra.direct_twosat_holdout_result.v1",
        "classification": CLASSIFICATION,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "correctness_failures": correctness_failures,
        "request_counts": by_count,
        "source_receipts": inventory,
        "benchmark_schema": benchmark["schema"],
        "platform": benchmark["platform"],
        "machine": benchmark["machine"],
        "repeats": benchmark["repeats"],
        "benchmark_results_sha256": hashlib.sha256(
            (destination / "benchmark" / "results.json").read_bytes()
        ).hexdigest(),
    }
    (destination / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return verdict


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--spectra-library", type=Path, required=True)
    parser.add_argument("--direct-library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        args.graph_dir,
        args.spectra_library,
        args.direct_library,
        args.out,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
