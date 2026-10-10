"""Build a fixed-plan, multiweek trace service case."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.real_traffic.case import TrafficCase
from experiments.real_traffic.series import build_series_case


def checked(root: Path, name: str, receipts: dict) -> bytes:
    payload = (root / name).read_bytes()
    record = receipts.get(name)
    if record is None or record["bytes"] != len(payload):
        raise ValueError("source receipt differs: " + name)
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError("source digest differs: " + name)
    return payload


def prepare_series(source, destination, weeks):
    source_root = Path(source)
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source_root / "sources.json").read_text())
    if (manifest.get("schema") != "spectra.real_traffic.series_sources.v1"
            or manifest.get("weeks") != list(weeks)):
        raise ValueError("series source manifest differs")
    receipts = {record["name"]: record for record in manifest["sources"]}
    case = build_series_case(
        weeks=weeks,
        routing_bytes=checked(source_root, "A", receipts),
        demands_bytes=checked(source_root, "demands", receipts),
        links_bytes=checked(source_root, "links", receipts),
        traffic_bytes={week: checked(source_root, week + ".gz", receipts)
                       for week in weeks},
    )
    path = output / "case.json"
    case.write(path)
    if TrafficCase.read(path) != case:
        raise AssertionError("multiweek case round trip differs")
    summary = {
        "schema": "spectra.real_traffic.series_summary.v1",
        "decision": "DEVELOPMENT_ONLY",
        "weeks": list(weeks),
        "vertices": case.vertices,
        "edges": len(case.edges),
        "palette": case.palette,
        "queries": len(case.queries),
        "sat_queries": case.sat_queries,
        "unsat_queries": case.unsat_queries,
        "duplicate_proposals_excluded": case.duplicate_queries,
        "unique_sat_witnesses": case.unique_witnesses,
        "case_sha256": case.case_sha256,
        "case_bytes": path.stat().st_size,
        "boundary": (
            "The first development week fixes the two plans. Every later week only "
            "selects original-address questions from measured traffic. Repeated exact "
            "questions are counted and excluded so the study measures distinct support "
            "requests rather than a trivial memoization workload."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "sources.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weeks", nargs="+", required=True)
    args = parser.parse_args()
    prepare_series(args.source, args.out, tuple(args.weeks))


if __name__ == "__main__":
    main()
