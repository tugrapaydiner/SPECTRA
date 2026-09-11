#!/usr/bin/env python3
"""Verify the byte-identical original CNF pilot stored durably in Git.

This does not run native solvers or collect new timings. Every original payload,
source hash, input, witness, work record and summary passes the original verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import sat_workload_gate as pilot


def verify(directory: Path) -> dict:
    manifest = pilot.read_json(directory/"manifest.json")
    if (type(manifest) is not dict or manifest.get("schema") != "spectra.sat_retained.v1"
            or manifest.get("archive") != "pilot.zip"
            or type(manifest.get("bytes")) is not int or not 0 < manifest["bytes"] <= 50_000_000):
        raise ValueError("unsupported retained SAT format")
    path = directory/"pilot.zip"
    if not path.is_file() or path.is_symlink() or path.stat().st_size != manifest["bytes"]:
        raise ValueError("missing, oversized or nonregular evidence archive")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("sha256"):
        raise ValueError("evidence archive digest mismatch")
    result = pilot.verify(path)
    with zipfile.ZipFile(path) as archive:
        # The original verifier has already checked the exact inventory, member
        # sizes/digests, canonical records, input regeneration and all witnesses.
        source = json.loads(archive.read("source.json"))
        if (source.get("git_commit") != manifest.get("execution_commit")
                or source.get("git_tree") != manifest.get("execution_tree")):
            raise ValueError("retained execution source mismatch")
        protocol = json.loads(archive.read("protocol.json"))
        rows = [json.loads(line) for line in archive.read("rows.jsonl").splitlines()]
    result["conflict_budget_diagnostic"] = {
        solver: {"requested": protocol["conflict_budget"],
            "maximum_observed": max((r["solver_stats"]["conflicts"] for r in rows
                if r["solver"] == solver and "solver_stats" in r), default=None),
            "observations_above_request": sum(r["solver_stats"]["conflicts"] > protocol["conflict_budget"]
                for r in rows if r["solver"] == solver and "solver_stats" in r)}
        for solver in protocol["solvers"]}
    result["storage"] = "byte-identical original seven-payload ZIP"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT/"results/sat_workload")
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
