"""Evaluate the frozen 17 October 2026 tile64 external-competition audit.

This consumes the existing failure-aware compiled-query audit.  It does not
recompute timings, select cases, or relax thresholds after measurement.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

CANDIDATE = "exact-bank-scc"
COUNT = 1024
EXTERNAL_LIMIT = 0.50
EXTERNAL_GRAPH_LIMIT = 0.67
RAW_LIMIT = 0.75
RAW_GRAPH_LIMIT = 0.85


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict or value.get("schema") != "spectra.compiled_query.audit.v1":
        raise ValueError("unexpected compiled-query audit schema")
    return value


def _comparison(audit: dict[str, Any], baseline: str) -> dict[str, Any]:
    rows = [
        row for row in audit.get("comparisons", ())
        if row.get("candidate") == CANDIDATE
        and row.get("baseline") == baseline
        and row.get("count") == COUNT
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one {CANDIDATE}/{baseline} comparison, got {len(rows)}")
    return rows[0]


def _finite_ratio(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {name}")
    return float(value)


def evaluate(audit: dict[str, Any], *, label: str, expect_cudd: bool) -> dict[str, Any]:
    failures: list[str] = []
    checks: dict[str, Any] = {}

    scheduled = audit.get("scheduled_jobs")
    complete = audit.get("all_scheduled_decisions_complete") is True
    checks["all_scheduled_jobs_complete"] = complete
    if not complete:
        failures.append("one or more scheduled jobs were not COMPLETE")

    if type(scheduled) is not int or scheduled <= 0:
        failures.append("invalid scheduled job count")
        expected_decisions = None
    else:
        expected_decisions = scheduled * COUNT

    decisions = audit.get("audited_decisions")
    if type(decisions) is not dict:
        failures.append("missing audited decision inventory")
        sat_decisions = None
    else:
        sat_decisions = decisions.get("SAT_VERIFIED", 0)
        non_sat = {key: value for key, value in decisions.items()
                   if key != "SAT_VERIFIED" and value}
        if non_sat:
            failures.append("the fixed all-SAT audit produced non-SAT decisions: " + repr(non_sat))
    checks["expected_audited_decisions"] = expected_decisions
    checks["observed_sat_decisions"] = sat_decisions
    if expected_decisions is not None and sat_decisions != expected_decisions:
        failures.append(
            f"audited SAT decisions differ: observed={sat_decisions}, expected={expected_decisions}"
        )
    if audit.get("original_non_sat_requests_checked") != 0:
        failures.append("the fixed all-SAT workload unexpectedly required non-SAT replay")
    if audit.get("outside_replication") is not False:
        failures.append("audit classification no longer states project-owned development evidence")

    baselines = ["exact-bank-raw", "minicard", "cadical195"]
    if expect_cudd:
        baselines.append("cudd-reduced-natural")

    comparisons: dict[str, Any] = {}
    for baseline in baselines:
        try:
            row = _comparison(audit, baseline)
            if row.get("eligible") is not True:
                failures.append(f"{baseline}: comparison is incomplete or excluded")
                comparisons[baseline] = {"eligible": False, "raw": row}
                continue
            if row.get("graph_clusters") != 2:
                failures.append(
                    f"{baseline}: expected two independent graph clusters, got {row.get('graph_clusters')}"
                )
            ratio = _finite_ratio(row.get("geomean_complete_ratio"), baseline + " geomean ratio")
            cluster_raw = row.get("cluster_ratios")
            if type(cluster_raw) is not dict or len(cluster_raw) != 2:
                raise ValueError(f"{baseline}: invalid cluster-ratio inventory")
            cluster_ratios = {
                str(name): _finite_ratio(value, f"{baseline}/{name} ratio")
                for name, value in cluster_raw.items()
            }
            if baseline == "exact-bank-raw":
                mean_limit, graph_limit = RAW_LIMIT, RAW_GRAPH_LIMIT
            else:
                mean_limit, graph_limit = EXTERNAL_LIMIT, EXTERNAL_GRAPH_LIMIT
            mean_pass = ratio <= mean_limit
            graph_pass = max(cluster_ratios.values()) <= graph_limit
            if not mean_pass:
                failures.append(
                    f"{baseline}: geomean ratio {ratio:.6f} exceeds frozen {mean_limit:.2f} limit"
                )
            if not graph_pass:
                failures.append(
                    f"{baseline}: worst graph ratio {max(cluster_ratios.values()):.6f} "
                    f"exceeds frozen {graph_limit:.2f} limit"
                )
            comparisons[baseline] = {
                "eligible": True,
                "geomean_complete_ratio": ratio,
                "cluster_ratios": cluster_ratios,
                "geomean_limit": mean_limit,
                "per_graph_limit": graph_limit,
                "geomean_pass": mean_pass,
                "per_graph_pass": graph_pass,
                "matching_certificate_contract": row.get("matching_certificate_contract"),
            }
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"{baseline}: malformed comparison: {exc}")
            comparisons[baseline] = {"eligible": False, "error": str(exc)}

    return {
        "schema": "spectra.tile64_external_competition_gate.v1",
        "classification": "exposed development audit; not a holdout or outside replication",
        "label": label,
        "candidate": CANDIDATE,
        "query_count": COUNT,
        "expect_cudd": expect_cudd,
        "thresholds": {
            "external_geomean_max": EXTERNAL_LIMIT,
            "external_per_graph_max": EXTERNAL_GRAPH_LIMIT,
            "raw_geomean_max": RAW_LIMIT,
            "raw_per_graph_max": RAW_GRAPH_LIMIT,
        },
        "checks": checks,
        "comparisons": comparisons,
        "failures": failures,
        "pass": not failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expect-cudd", action="store_true")
    args = parser.parse_args(argv)
    result = evaluate(_load(args.audit), label=args.label, expect_cudd=args.expect_cudd)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
