#!/usr/bin/env python3
"""Read-only, CPU-only audit of SPECTRA's retained M14/M15 evidence.

Python 3.10+, standard library only. Does not import SPECTRA, load PyTorch,
execute checkpoint pickle, train a model, or access the network.

Usage (three archives downloaded from the GitHub Actions artifact IDs below):
  python SPECTRA_independent_audit.py --evidence-dir . --out audit_findings.json

Exit codes: 0 = no overlap found; 1 = verified evidence contains a cross-ancestor
holdout overlap; 2 = invalid/missing evidence. Exit 1 is EXPECTED for this snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SNAPSHOT = "1e29cb10662cb83ba9e28e4d30164fc373a547a5"
SPECS = {
    "m14_original": {
        "filename": "spectra_m14_evidence.zip", "artifact_id": 10035946017,
        "sha256": "0306f64efb48c264aa7b387ff6b449def011fb64ad02aa74eb03ee3b4e58e314",
    },
    "m14_source_for_m15": {
        "filename": "spectra_m14_source_evidence.zip", "artifact_id": 10077794916,
        "sha256": "5729932600743b2cef19d9e0b9baf112ec75b627552af4d30a089266994a2f37",
    },
    "m15": {
        "filename": "spectra_m15_evidence.zip", "artifact_id": 10080887785,
        "sha256": "c49e5c06e0f2822088cb9ce807fcc6bc37da6236962953ccc8c594be01c55b2e",
    },
}
DEV = "experiment/manifests/train_validation_development.json"
ROWS = "experiment/confirmation_rows.jsonl"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quantile(values: Iterable[float], q: float) -> float | None:
    """Linear interpolation, equivalent to the usual NumPy default percentile."""
    a = sorted(float(v) for v in values)
    if not a:
        return None
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0,1]")
    x = (len(a) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return a[lo] + (x - lo) * (a[hi] - a[lo])


def read_json(z: zipfile.ZipFile, name: str) -> Any:
    return json.loads(z.read(name))


def read_rows(z: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
    with z.open(name) as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_archive(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    digest = file_sha256(path)
    if digest != spec["sha256"]:
        raise ValueError(f"Unexpected archive SHA256: {path}: {digest}")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate ZIP member names: {path}")
        if sum(i.file_size for i in z.infolist()) > 1024**3:
            raise ValueError("Unexpected uncompressed size >1 GiB")
        verified = 0
        omitted_source_members = []
        for line in z.read("evidence_sha256.txt").decode().splitlines():
            if not line.strip():
                continue
            expected, recorded = line.split(None, 1)
            # Actions strips the workflow's .m14a5/ or .m15/ root on upload.
            name = recorded.strip()
            if name.startswith((".m14a5/", ".m15/")):
                name = name.split("/", 1)[1]
            if name not in names:
                if spec["artifact_id"] == 10080887785 and name.startswith(
                    "experiment/accepted_m14_artifact/"
                ):
                    omitted_source_members.append(name)
                    continue
                raise ValueError(f"Unexpected missing hashed member: {name}")
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError(f"Member hash mismatch: {name}")
            verified += 1
        ci_lines = z.read("pytest_fast.txt").decode(errors="replace").splitlines()
        ci_summary = next((s.strip("= ") for s in reversed(ci_lines) if "passed" in s), None)
        env = read_json(z, "experiment/environment.json")
    return {
        "artifact_id": spec["artifact_id"], "filename": path.name,
        "archive_sha256": digest, "archive_sha256_matches": True,
        "present_embedded_hashes_verified": verified,
        "intentionally_omitted_source_members": omitted_source_members,
        "unexpected_missing_or_mismatched_members": [],
        "retained_ci_summary_not_a_new_test_run": ci_summary,
        "retained_environment": env,
    }


def entries(manifest: dict[str, Any]):
    for split, data in manifest["splits"].items():
        for index, row in enumerate(data["examples"]):
            yield split, index, row


def cross_overlap(ancestor: dict[str, Any], descendant: dict[str, Any]):
    by_fp = defaultdict(list)
    for split, index, row in entries(ancestor):
        by_fp[row["fingerprint"]].append({
            "split": split, "index_zero_based": index, "example_id": row["id"]
        })
    hits = []
    for split, index, row in entries(descendant):
        for previous in by_fp.get(row["fingerprint"], []):
            hits.append({
                "fingerprint": row["fingerprint"], "ancestor": previous,
                "descendant": {"split": split, "index_zero_based": index,
                               "example_id": row["id"]},
            })
    return hits


def summarize(rows: list[dict[str, Any]], seed_key: str) -> dict[str, Any]:
    by_config = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["config_id"], row.get(seed_key), row["example_id"])
        if key in seen:
            raise ValueError(f"Duplicate model-example row: {key}")
        seen.add(key)
        if row["semantic_success"] not in (0, 1):
            raise ValueError("Semantic success must be binary")
        by_config[row["config_id"]].append(row)
    result = {}
    for config, rr in sorted(by_config.items()):
        times = [float(r["latency_ms"]) for r in rr if r.get("latency_ms") is not None]
        if any(not math.isfinite(t) or t < 0 for t in times):
            raise ValueError("Invalid latency")
        successes = sum(int(r["semantic_success"]) for r in rr)
        result[config] = {
            "model_example_rows": len(rr), "successes": successes,
            "success_rate": successes / len(rr),
            "unique_example_ids": len({r["example_id"] for r in rr}),
            "training_seeds": sorted({r[seed_key] for r in rr if r.get(seed_key) is not None}),
            "timed_model_example_rows": len(times),
            "mean_latency_ms": statistics.mean(times) if times else None,
            "median_latency_ms": statistics.median(times) if times else None,
            "p95_latency_ms": quantile(times, .95),
        }
    return result


def paired_regressions(rows: list[dict[str, Any]], excluded_ids: set[str]):
    rr = [r for r in rows if r["example_id"] not in excluded_ids]
    base = {(r["core_seed"], r["example_id"]): int(r["semantic_success"])
            for r in rr if r["config_id"] == "semantic_exit_k4"}
    results = summarize(rr, "core_seed")
    for config in results:
        candidates = [r for r in rr if r["config_id"] == config]
        keys = {(r["core_seed"], r["example_id"]) for r in candidates}
        if keys != set(base):
            raise ValueError(f"Unmatched candidate/base rows for {config}")
        regressions = sum(base[(r["core_seed"], r["example_id"])] == 1
                          and r["semantic_success"] == 0 for r in candidates)
        improvements = sum(base[(r["core_seed"], r["example_id"])] == 0
                           and r["semantic_success"] == 1 for r in candidates)
        results[config]["regressions_against_semantic_exit"] = regressions
        results[config]["improvements_against_semantic_exit"] = improvements
    primary = results["learned_strong_d4_int8"]["regressions_against_semantic_exit"]
    prevention = {}
    for key in ("learned_strong_guard_d4_int8", "learned_oracle_d4_int8", "learned_strong_d4_fp32"):
        n = results[key]["regressions_against_semantic_exit"]
        prevention[key] = (primary - n) / primary if primary else None
    return {"excluded_example_ids": sorted(excluded_ids), "operating_points": results,
            "regression_reduction_fractions": prevention}


def count_4x4_solutions() -> int:
    """Exhaustively enumerate row permutations; no ML or repository code involved."""
    permutations = list(itertools.permutations(range(1, 5)))
    count = 0
    for rows in itertools.product(permutations, repeat=4):
        if any(len({rows[r][c] for r in range(4)}) != 4 for c in range(4)):
            continue
        if any(len({rows[r][c] for r in range(br, br+2) for c in range(bc, bc+2)}) != 4
               for br in (0, 2) for bc in (0, 2)):
            continue
        count += 1
    return count


def evaluated_pool_diagnosis(z: zipfile.ZipFile, rows: list[dict[str, Any]], excluded: set[str]):
    """Separate retained evaluated-pool coverage from final selection success.

    This uses recorded semantic labels. It does not rerun their task validator,
    and an oracle selection is diagnostic: checker cost is not free deployment.
    """
    config = "learned_strong_d4_int8"
    primary = {(r["core_seed"], r["example_id"]): r for r in rows
               if r["config_id"] == config and r["example_id"] not in excluded}
    pools = defaultdict(list)
    with z.open("experiment/confirmation_leaf_rows.jsonl") as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["core_seed"], r["example_id"])
            if r["config_id"] == config and key in primary:
                pools[key].append(r)
    if set(pools) != set(primary):
        raise ValueError("Incomplete evaluated-pool evidence")
    successes = coverage = missed_valid = 0
    for key, result in primary.items():
        leaf = pools[key]
        valid_seen = any(bool(r["semantic_valid"]) for r in leaf)
        solved = bool(result["semantic_success"])
        selected = [r for r in leaf if r["selected"]]
        if not selected or any(bool(r["semantic_valid"]) != solved for r in selected):
            raise ValueError("Selected leaf and returned-answer labels disagree")
        if solved and not valid_seen:
            raise ValueError("Success has no valid evaluated candidate")
        coverage += valid_seen
        successes += solved
        missed_valid += valid_seen and not solved
    n = len(primary)
    return {
        "config_id": config, "excluded_example_ids": sorted(excluded),
        "model_example_rows": n, "returned_valid": successes,
        "evaluated_pool_contains_valid": coverage,
        "returned_invalid_despite_valid_evaluated_candidate": missed_valid,
        "no_valid_evaluated_candidate": n - coverage,
        "coverage_rate": coverage / n,
        "selection_success_conditional_on_coverage": successes / coverage,
        "fraction_of_returned_failures_with_valid_evaluated_candidate": missed_valid / (n-successes),
        "scope": "Post-hoc decomposition of stored labels; additional checking has an unmeasured cost.",
    }


def self_test():
    assert quantile([0, 10], .95) == 9.5
    assert quantile([], .5) is None
    a = {"splits": {"train": {"examples": [{"fingerprint": "same", "id": "old"}]}}}
    b = {"splits": {"test": {"examples": [{"fingerprint": "same", "id": "new"}]}}}
    hit = cross_overlap(a, b)
    assert len(hit) == 1 and hit[0]["ancestor"]["example_id"] == "old"
    assert hit[0]["descendant"]["example_id"] == "new"
    rows = [
        {"config_id": "semantic_exit_k4", "core_seed": 1, "example_id": "x", "semantic_success": 1, "latency_ms": 1},
        {"config_id": "learned_strong_d4_int8", "core_seed": 1, "example_id": "x", "semantic_success": 0, "latency_ms": 2},
    ]
    assert summarize(rows, "core_seed")["learned_strong_d4_int8"]["successes"] == 0
    try:
        summarize(rows + rows, "core_seed")
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate rows were not rejected")
    print("Audit helper self-tests passed (not SPECTRA's test suite).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence-dir", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("SPECTRA_audit_findings.json"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    report: dict[str, Any] = {
        "audit_date": "2026-09-10", "reviewed_main_commit": SNAPSHOT,
        "scope": "Read-only source review plus independent retained-evidence recomputation",
        "limitations": [
            "No full training, model inference, native compilation, or SPECTRA pytest rerun.",
            "Stored semantic-success labels are reaggregated, not independently checked against predictions.",
            "Bootstrap confidence intervals and original paired-median gate are not recomputed here.",
            "Fingerprint checks are exact-content checks, not Sudoku symmetry-class or distribution-shift audits.",
            "M14 original and later M14 source run are distinct; their timings are not pooled.",
            "M15 sensitivity analysis is post hoc, not a replacement preregistered confirmation.",
        ],
        "archives": {},
    }
    archives = {}
    try:
        for key, spec in SPECS.items():
            path = args.evidence_dir / spec["filename"]
            report["archives"][key] = verify_archive(path, spec)
            archives[key] = zipfile.ZipFile(path)
        original = archives["m14_original"]
        source = archives["m14_source_for_m15"]
        m15 = archives["m15"]
        report["m14_original_confirmation"] = summarize(read_rows(original, ROWS), "seed")
        report["m14_source_run_confirmation"] = summarize(read_rows(source, ROWS), "seed")
        report["m14_training_manifest_byte_identity_across_original_and_source"] = original.read(DEV) == source.read(DEV)
        source_manifest = read_json(source, DEV)
        conf15 = read_json(m15, "experiment/manifests/confirmation.json")
        overlap = cross_overlap(source_manifest, conf15)
        report["m15_confirmation_vs_m14_source_hierarchy"] = overlap
        report["m15_development_vs_m14_source_hierarchy"] = cross_overlap(source_manifest, read_json(m15, DEV))
        report["m15_shift_vs_m14_source_hierarchy"] = cross_overlap(
            source_manifest, read_json(m15, "experiment/manifests/shift_low_clue.json"))
        report["m15_within_milestone_confirmation_overlap"] = cross_overlap(read_json(m15, DEV), conf15)
        rows15 = read_rows(m15, ROWS)
        excluded = {h["descendant"]["example_id"] for h in overlap}
        report["m15_original_confirmation"] = paired_regressions(rows15, set())
        report["m15_posthoc_ancestor_overlap_exclusion_sensitivity"] = paired_regressions(rows15, excluded)
        report["m15_evaluated_pool_diagnosis"] = evaluated_pool_diagnosis(m15, rows15, set())
        report["m15_evaluated_pool_diagnosis_excluding_ancestor_overlap"] = evaluated_pool_diagnosis(m15, rows15, excluded)
        count = count_4x4_solutions()
        report["independently_enumerated_4x4_solution_universe"] = {
            "valid_completed_boards": count, "uint8_raw_cell_storage_bytes": count * 16,
            "scope": "Complete 4x4 solution boards only, not all clue masks, not a latency measurement.",
        }
        report["verdict"] = "CROSS_ANCESTOR_CONFIRMATION_OVERLAP_DETECTED" if overlap else "NO_EXACT_OVERLAP_DETECTED"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(report["verdict"])
        print(f"Findings written to {args.out}")
        return 1 if overlap else 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 2
    finally:
        for z in archives.values():
            z.close()


if __name__ == "__main__":
    raise SystemExit(main())
