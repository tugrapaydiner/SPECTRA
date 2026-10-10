"""Build and independently recheck a trace-driven repeated-support case."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.real_traffic import TrafficCase, build_case


def checked_source(root: Path, name: str, expected: dict[str, dict]) -> bytes:
    payload = (root / name).read_bytes()
    record = expected.get(name)
    if record is None:
        raise ValueError(f"missing source receipt: {name}")
    if len(payload) != record["bytes"]:
        raise ValueError(f"source size differs: {name}")
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(f"source digest differs: {name}")
    return payload


def prepare(source: str | Path, destination: str | Path, *, week: str) -> dict:
    source_root = Path(source)
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source_root / "sources.json").read_text())
    if manifest.get("schema") != "spectra.real_traffic.sources.v1" or manifest.get("week") != week:
        raise ValueError("source manifest differs")
    expected = {record["name"]: record for record in manifest["sources"]}
    case = build_case(
        week_name=week,
        routing_bytes=checked_source(source_root, "A", expected),
        demands_bytes=checked_source(source_root, "demands", expected),
        links_bytes=checked_source(source_root, "links", expected),
        traffic_bytes=checked_source(source_root, f"{week}.gz", expected),
    )
    case_path = out / "case.json"
    case.write(case_path)
    # Re-open from bytes and recompute every SAT/UNSAT outcome before reporting.
    replayed = TrafficCase.read(case_path)
    if replayed != case:
        raise AssertionError("case round trip differs")
    changed = sum(left != right for left, right in zip(case.plan_a, case.plan_b))
    proof_edges = sum(
        len(proof["positive_to_negative"]) + len(proof["negative_to_positive"]) - 2
        for proof in case.contradictions if proof is not None
    )
    summary = {
        "schema": "spectra.real_traffic.summary.v1",
        "status": "DEVELOPMENT_DATA_READY",
        "week": week,
        "vertices": case.vertices,
        "edges": len(case.edges),
        "palette": case.palette,
        "changed_routes": changed,
        "queries": len(case.queries),
        "sat_queries": case.sat_queries,
        "unsat_queries": case.unsat_queries,
        "unsat_contradictions": sum(
            proof is not None for proof in case.contradictions
        ),
        "unsat_proof_edges": proof_edges,
        "unique_sat_witnesses": case.unique_witnesses,
        "duplicate_proposals": case.duplicate_queries,
        "proposal_attempts": case.proposal_attempts,
        "case_sha256": case.case_sha256,
        "case_bytes": case_path.stat().st_size,
        "boundary": (
            "The topology, fixed routes, and traffic values are measured Abilene data. "
            "The two wavelength plans and the protection/migration policy are disclosed "
            "research constructions, not operator-deployed configurations."
        ),
        "scientific_decision": (
            "This prepares development data only. It is not a performance result, "
            "confirmation result, deployment claim, or flagship decision."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "sources.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--week", default="X01")
    args = parser.parse_args()
    prepare(args.source, args.out, week=args.week)


if __name__ == "__main__":
    main()
