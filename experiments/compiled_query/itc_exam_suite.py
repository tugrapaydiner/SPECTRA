"""Prospective multi-instance ITC-2007 exam-repair transfer suite.

The structural-v10 selector is developed from the exposed synthetic panel,
the retained failed-v3 confirmation, and the published ``set9`` transfer.  ``set10`` and ``set12`` are fixed holdout
instances: their execution results must not be used to retune the policy.  This
suite remains a bounded conflict-core repair study, not a full timetabling claim.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import shutil
import statistics
from typing import Iterable

from experiments.compiled_query import itc_exam_transfer as transfer

SCHEMA = "spectra.itc2007.exam_repair_suite.v1"
DEFAULT_HOLDOUT_MAX_RATIO = 1.05


@dataclass(frozen=True)
class SuiteCase:
    name: str
    role: str
    instance_path: str
    solution_path: str
    expected_mode: str | None = None

    def validate(self) -> "SuiteCase":
        if (type(self.name) is not str or not self.name
                or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-" for ch in self.name)
                or self.name.startswith("-") or self.name.endswith("-")):
            raise ValueError("suite case name must be lowercase kebab-case")
        if self.role not in ("development", "holdout", "replication"):
            raise ValueError("suite case role differs")
        if self.expected_mode not in (None, "raw", "scc"):
            raise ValueError("suite expected mode differs")
        for label, value in (("instance", self.instance_path), ("solution", self.solution_path)):
            if type(value) is not str or not value:
                raise ValueError(f"suite case {label} path must be nonempty")
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"suite case {label} path must stay under upstream root")
        return self


DEFAULT_CASES = (
    SuiteCase(
        "set9-development", "development",
        "examtt-instances/Instances/set9.exam",
        "InitMethods/SolStore/set9_OBSIFeas_0.sol", "scc",
    ),
    SuiteCase(
        "set10-holdout", "holdout",
        "examtt-instances/Instances/set10.exam",
        "InitMethods/SolStore/set10_OBSIFeas_0.sol", "scc",
    ),
    SuiteCase(
        "set12-holdout", "holdout",
        "examtt-instances/Instances/set12.exam",
        "InitMethods/SolStore/set12_OBSIFeas_0.sol", "raw",
    ),
)


def _cases(cases: Iterable[SuiteCase]) -> tuple[SuiteCase, ...]:
    checked = tuple(case.validate() for case in cases)
    if not checked or len({case.name for case in checked}) != len(checked):
        raise ValueError("suite requires unique nonempty cases")
    if not any(case.role == "holdout" for case in checked):
        raise ValueError("suite requires at least one holdout")
    return checked


def _ratio(value: float | None) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not 1.0 <= float(value) <= 10.0:
        raise ValueError("holdout max auto/raw ratio must be in [1, 10] or null")
    return float(value)


def _manifest_path(root: Path) -> Path:
    return root / "suite-manifest.json"


def freeze(root: Path, upstream_root: Path, *, cases: Iterable[SuiteCase] = DEFAULT_CASES,
           query_count: int = 1024,
           holdout_max_auto_raw_ratio: float | None = DEFAULT_HOLDOUT_MAX_RATIO,
           source_commit: str | None = None) -> dict:
    root = root.resolve()
    upstream_root = upstream_root.resolve()
    if root.exists():
        raise FileExistsError(root)
    checked = _cases(cases)
    threshold = _ratio(holdout_max_auto_raw_ratio)
    license_path = upstream_root / "LICENSE"
    if not license_path.is_file():
        raise FileNotFoundError(license_path)
    staging = root.with_name(root.name + ".partial")
    if staging.exists():
        raise FileExistsError(staging)
    (staging / "cases").mkdir(parents=True, exist_ok=False)
    try:
        case_records = []
        for case in checked:
            case_root = staging / "cases" / case.name
            manifest = transfer.freeze(
                case_root,
                upstream_root / case.instance_path,
                upstream_root / case.solution_path,
                query_count=query_count,
                license_path=license_path,
                source_commit=source_commit,
                dataset_instance_path=case.instance_path,
                dataset_solution_path=case.solution_path,
                evaluation_role=case.role,
            )
            case_records.append({
                "name": case.name,
                "role": case.role,
                "instance_path": case.instance_path,
                "solution_path": case.solution_path,
                "expected_mode": case.expected_mode,
                "manifest_sha256": (case_root / "manifest.sha256").read_text().strip(),
                "instance_sha256": manifest["instance_sha256"],
                "solution_sha256": manifest["solution_sha256"],
            })
        manifest = {
            "schema": SCHEMA,
            "classification": (
                "prospective multi-instance bounded exam-repair transfer; set9 is "
                "development and set10/set12 are fixed execution holdouts"
            ),
            "source_commit": case_records and json.loads(
                (staging / "cases" / case_records[0]["name"] / "manifest.json").read_text()
            )["source_commit"],
            "dataset_repository": transfer.DATASET_REPOSITORY,
            "dataset_commit": transfer.DATASET_COMMIT,
            "query_count_per_case": query_count,
            "selector_policy": "structural-v10",
            "selector_development_sources": [
                "exposed synthetic graph panel",
                "retained failed-v3 confirmation and v4 density sweep",
                "set9 development transfer",
            ],
            "holdout_policy": (
                "set10 and set12 results may falsify the frozen selector but may not "
                "change it inside this suite"
            ),
            "holdout_max_auto_raw_ratio": threshold,
            "gate_metric": "median of paired per-repetition auto/raw elapsed ratios",
            "cases": case_records,
        }
        payload = transfer.canonical_json(manifest)
        transfer.write_once(_manifest_path(staging), payload)
        transfer.write_once(
            staging / "suite-manifest.sha256",
            (transfer.digest(payload) + "\n").encode(),
        )
        staging.rename(root)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_manifest(root: Path) -> dict:
    root = root.resolve()
    expected = (root / "suite-manifest.sha256").read_text().strip()
    manifest = json.loads(transfer.read_bound(_manifest_path(root), expected))
    if (manifest.get("schema") != SCHEMA
            or manifest.get("dataset_repository") != transfer.DATASET_REPOSITORY
            or manifest.get("dataset_commit") != transfer.DATASET_COMMIT
            or manifest.get("selector_policy") not in ("structural-v2", "structural-v3", "structural-v4", "structural-v5", "structural-v6", "structural-v7", "structural-v8", "structural-v10")
            or type(manifest.get("query_count_per_case")) is not int
            or manifest["query_count_per_case"] < 1):
        raise ValueError("suite manifest identity differs")
    threshold = _ratio(manifest.get("holdout_max_auto_raw_ratio"))
    records = manifest.get("cases")
    if type(records) is not list:
        raise ValueError("suite case inventory differs")
    checked = _cases(SuiteCase(
        item.get("name"), item.get("role"),
        item.get("instance_path"), item.get("solution_path"),
        item.get("expected_mode"),
    ) for item in records if type(item) is dict)
    if len(checked) != len(records):
        raise ValueError("suite case inventory differs")
    for case, item in zip(checked, records):
        case_root = root / "cases" / case.name
        if ((case_root / "manifest.sha256").read_text().strip()
                != item.get("manifest_sha256")):
            raise ValueError(f"suite case {case.name} manifest hash differs")
        child = transfer.load_manifest(case_root)
        if (child.get("dataset_instance_path") != case.instance_path
                or child.get("dataset_solution_path") != case.solution_path
                or child.get("evaluation_role") != case.role
                or child.get("query_count") != manifest["query_count_per_case"]
                or child.get("instance_sha256") != item.get("instance_sha256")
                or child.get("solution_sha256") != item.get("solution_sha256")):
            raise ValueError(f"suite case {case.name} identity differs")
    manifest["holdout_max_auto_raw_ratio"] = threshold
    return manifest


def run(root: Path) -> dict:
    root = root.resolve()
    manifest = load_manifest(root)
    if (root / "suite-run.json").exists():
        raise FileExistsError("suite execution already exists")
    records = []
    for item in manifest["cases"]:
        case_root = root / "cases" / item["name"]
        record = transfer.run(case_root)
        records.append({
            "name": item["name"],
            "run_sha256": (case_root / "run.sha256").read_text().strip(),
            "execution_order": record["execution_order"],
        })
    result = {
        "schema": SCHEMA + ".run",
        "suite_manifest_sha256": (root / "suite-manifest.sha256").read_text().strip(),
        "cases": records,
    }
    payload = transfer.canonical_json(result)
    transfer.write_once(root / "suite-run.json", payload)
    transfer.write_once(root / "suite-run.sha256", (transfer.digest(payload) + "\n").encode())
    return result


def audit(root: Path) -> dict:
    root = root.resolve()
    manifest = load_manifest(root)
    run_expected = (root / "suite-run.sha256").read_text().strip()
    run_record = json.loads(transfer.read_bound(root / "suite-run.json", run_expected))
    if (run_record.get("schema") != SCHEMA + ".run"
            or run_record.get("suite_manifest_sha256")
            != (root / "suite-manifest.sha256").read_text().strip()):
        raise ValueError("suite run identity differs")
    run_cases = run_record.get("cases")
    if type(run_cases) is not list or len(run_cases) != len(manifest["cases"]):
        raise ValueError("suite run case inventory differs")

    threshold = manifest["holdout_max_auto_raw_ratio"]
    summaries = []
    selector_failures = []
    holdout_failures = []
    development_failures = []
    for item, run_item in zip(manifest["cases"], run_cases):
        if type(run_item) is not dict or run_item.get("name") != item["name"]:
            raise ValueError("suite run case order differs")
        case_root = root / "cases" / item["name"]
        if (case_root / "run.sha256").read_text().strip() != run_item.get("run_sha256"):
            raise ValueError(f"suite case {item['name']} run hash differs")
        result = transfer.audit(case_root)
        child_run = json.loads(transfer.read_bound(
            case_root / "run.json", (case_root / "run.sha256").read_text().strip()
        ))
        auto_elapsed = child_run["modes"]["auto"]["elapsed_ns"]
        raw_elapsed = child_run["modes"]["raw"]["elapsed_ns"]
        scc_elapsed = child_run["modes"]["scc"]["elapsed_ns"]
        paired_median = statistics.median(
            auto / raw for auto, raw in zip(auto_elapsed, raw_elapsed)
        )
        auto_scc_paired = statistics.median(
            auto / scc for auto, scc in zip(auto_elapsed, scc_elapsed)
        )
        scc_raw_paired = statistics.median(
            scc / raw for scc, raw in zip(scc_elapsed, raw_elapsed)
        )
        selected = result["modes"]["auto"]["selected_mode"]
        reason = result["modes"]["auto"]["auto_policy_reason"]
        expected_mode = item.get("expected_mode")
        gate_pass = True
        if expected_mode == "scc":
            gate_pass = (
                selected == "scc"
                and auto_scc_paired <= 1.03
                and scc_raw_paired <= 0.98
            )
        elif expected_mode == "raw":
            gate_pass = selected == "raw" and paired_median <= (threshold or 1.05)
        elif item["role"] == "holdout" and threshold is not None:
            gate_pass = paired_median <= threshold
        if not gate_pass:
            selector_failures.append(item["name"])
            if item["role"] == "holdout":
                holdout_failures.append(item["name"])
            else:
                development_failures.append(item["name"])
        summaries.append({
            "name": item["name"],
            "role": item["role"],
            "functional_complete": result["functional_complete"],
            "baseline_valid_full_itc_hard_constraints": result[
                "baseline_valid_full_itc_hard_constraints"
            ],
            "instance_sha256": result["instance_sha256"],
            "solution_sha256": result["solution_sha256"],
            "model": result["model"],
            "queries": result["queries"],
            "sat": result["modes"]["auto"]["sat"],
            "unsat": result["modes"]["auto"]["unsat"],
            "proof_kinds": result["modes"]["auto"]["proof_kinds"],
            "expected_mode": expected_mode,
            "auto_selected_mode": selected,
            "auto_policy_reason": reason,
            "auto_vs_raw_median_ratio": result["auto_vs_raw_median_ratio"],
            "auto_vs_raw_paired_median_ratio": paired_median,
            "auto_vs_scc_paired_median_ratio": auto_scc_paired,
            "scc_vs_raw_paired_median_ratio": scc_raw_paired,
            "selector_guard_pass": gate_pass,
            "holdout_gate_pass": (
                gate_pass if item["role"] == "holdout" else None
            ),
            "audit_sha256": (case_root / "audit.sha256").read_text().strip(),
        })
    functional = all(
        item["functional_complete"] and item["baseline_valid_full_itc_hard_constraints"]
        for item in summaries
    )
    result = {
        "schema": SCHEMA + ".audit",
        "functional_complete": functional,
        "source_commit": manifest["source_commit"],
        "dataset_commit": manifest["dataset_commit"],
        "selector_policy": manifest["selector_policy"],
        "query_count_per_case": manifest["query_count_per_case"],
        "holdout_max_auto_raw_ratio": threshold,
        "holdout_gate_pass": not holdout_failures,
        "selector_guard_pass": not selector_failures,
        "selector_failures": selector_failures,
        "holdout_failures": holdout_failures,
        "development_failures": development_failures,
        "cases": summaries,
        "interpretation": [
            "set9 is development evidence; set10 and set12 are prospective execution holdouts",
            "the selector guard covers every case; holdout failures are reported separately",
            "the gate limits auto-policy overhead relative to the explicit raw control",
            "all outputs remain complete proof-carrying conflict-core repair answers",
            "this is not full ITC-2007 timetable feasibility or optimization",
            "project-owned CI is not outside-team replication or adoption",
        ],
    }
    payload = transfer.canonical_json(result)
    path = root / "suite-audit.json"
    if path.exists():
        expected = (root / "suite-audit.sha256").read_text().strip()
        if transfer.read_bound(path, expected) != payload:
            raise ValueError("existing suite audit differs from independent replay")
    else:
        transfer.write_once(path, payload)
        transfer.write_once(
            root / "suite-audit.sha256", (transfer.digest(payload) + "\n").encode()
        )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument(
        "--holdout-max-auto-raw-ratio", type=float,
        default=DEFAULT_HOLDOUT_MAX_RATIO,
    )
    args = parser.parse_args(argv)
    if args.command == "freeze":
        if args.upstream_root is None:
            parser.error("freeze requires --upstream-root")
        value = freeze(
            args.root, args.upstream_root, query_count=args.queries,
            holdout_max_auto_raw_ratio=args.holdout_max_auto_raw_ratio,
        )
    elif args.command == "run":
        value = run(args.root)
    else:
        value = audit(args.root)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
