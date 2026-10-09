"""Strict audit and graph-clustered analysis for packed WAP service sessions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import zlib

from experiments.wap_service_format.format import (
    load_packed_case, read_manifest, receipt_by_name,
)
from experiments.wap_service_format.run import (
    ADDRESS_SPACE_BYTES, ARMS, ORDERS, schedule,
)
from experiments.wap_support.controls import order_indices
from experiments.wap_support.workload import WorkloadCase
from spectra.cnf.quotient_query import QuotientRuntime

PRIMARY = "scc"
CONTROL = "minicard"
MEAN_UPPER_GATE = 0.50
P95_UPPER_GATE = 1.10
EXPECTED_PYSAT = "1.9.dev15"


def require(ok: object, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, probability):
    values = sorted(values)
    require(values, "empty quantile")
    position = (len(values) - 1) * probability
    low = int(position); high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def validate_file_manifest(root: Path) -> dict:
    manifest_path = root / "MANIFEST.json"
    require(manifest_path.is_file(), "missing evidence manifest")
    manifest = json.loads(manifest_path.read_text())
    require(type(manifest) is dict, "evidence manifest must be an object")
    for name, digest in manifest.items():
        require(type(name) is str and type(digest) is str and len(digest) == 64,
                "invalid evidence manifest entry")
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file(),
                "evidence manifest path differs")
        require(sha256(path) == digest, "evidence hash mismatch: " + name)
    actual = {str(path.relative_to(root)) for path in root.rglob("*")
              if path.is_file() and path.name != "MANIFEST.json"}
    require(set(manifest) == actual, "evidence file inventory differs")
    return manifest


def bounded_decompress(compressed: bytes, expected: int) -> bytes:
    decoder = zlib.decompressobj()
    output = decoder.decompress(compressed, expected + 1)
    require(len(output) <= expected and decoder.eof, "compressed output exceeds or truncates contract")
    require(not decoder.unconsumed_tail and not decoder.unused_data,
            "compressed output has trailing bytes")
    output += decoder.flush()
    require(len(output) == expected, "output length differs")
    return output


def load_cases(packed_root: Path, canonical_root: Path | None):
    manifest = read_manifest(packed_root / "MANIFEST.json")
    receipts = receipt_by_name(manifest)
    cases = {}
    for name, receipt in sorted(receipts.items()):
        path = packed_root / name
        require(path.is_file() and sha256(path) == receipt.packed_file_sha256,
                "packed artifact hash differs: " + name)
        loaded = load_packed_case(path, expected_sha256=receipt.packed_file_sha256)
        case = loaded.case
        require(case.case_sha256 == receipt.case_sha256 and case.n == receipt.n
                and len(case.edges) == receipt.edges
                and case.query_count == receipt.query_count
                and case.query_width == receipt.query_width,
                "packed receipt geometry differs: " + name)
        if canonical_root is not None:
            source = canonical_root / receipt.source_name
            require(source.is_file() and sha256(source) == receipt.source_file_sha256,
                    "canonical source hash differs: " + name)
            canonical = WorkloadCase.from_json(source)
            require(canonical == case, "packed case differs from canonical validated case: " + name)
        cases[name] = case
    require(cases, "empty packed case panel")
    return manifest, receipts, cases


def validate(packed_root: Path, canonical_root: Path | None, evidence: Path,
             runtime_path: Path):
    manifest, receipts, cases = load_cases(packed_root, canonical_root)
    validate_file_manifest(evidence)
    jobs = schedule(manifest)
    require(json.loads((evidence / "schedule.json").read_text()) == [list(x) for x in jobs],
            "packed schedule differs")
    rows = json.loads((evidence / "sessions.json").read_text())
    require(type(rows) is list and len(rows) == len(jobs), "packed session inventory incomplete")
    environment = json.loads((evidence / "environment.json").read_text())
    require(environment.get("address_space_limit_bytes") == ADDRESS_SPACE_BYTES,
            "address-space envelope differs")
    require(environment.get("runtime_library_sha256") == sha256(runtime_path),
            "runtime library identity differs")
    require(environment.get("packed_manifest_sha256") == sha256(packed_root / "MANIFEST.json"),
            "packed manifest identity differs")
    require(environment.get("arms") == list(ARMS) and environment.get("orders") == list(ORDERS),
            "solver or order inventory differs")
    require(environment.get("python_sat_version") == EXPECTED_PYSAT,
            "PySAT version differs")

    runtime = QuotientRuntime(runtime_path)
    checkers = {name: runtime.checker(case.n, case.palette, case.edges, case.masks,
                                      512 * 1024 * 1024)
                for name, case in cases.items()}
    try:
        seen = set()
        for row, job in zip(rows, jobs, strict=True):
            packed_name, arm, order, seed = job
            require((row.get("packed_name"), row.get("arm"), row.get("order"),
                     row.get("order_seed")) == job, "session order differs")
            session_id = row.get("session_id")
            require(type(session_id) is str and session_id not in seen, "duplicate session id")
            seen.add(session_id)
            folder = evidence / "sessions" / session_id
            require(folder.is_dir(), "missing session folder")
            metadata_path = folder / "session.json"
            metadata = json.loads(metadata_path.read_text())
            require(json.loads((folder / "outer.json").read_text()) == row,
                    "outer receipt differs")
            require(sha256(metadata_path) == row.get("session_metadata_sha256"),
                    "session metadata hash differs")
            case = cases[packed_name]
            receipt = receipts[packed_name]
            require(metadata.get("schema") == "spectra.wap_service_format.session.v1"
                    and metadata.get("status") == "COMPLETE"
                    and metadata.get("arm") == arm
                    and metadata.get("order_name") == order
                    and metadata.get("case_sha256") == case.case_sha256,
                    "session semantic identity differs")
            require(metadata.get("packed_file_sha256") == receipt.packed_file_sha256
                    and metadata.get("packed_payload_sha256") == receipt.payload_sha256
                    and metadata.get("packed_bytes") == receipt.packed_bytes
                    and metadata.get("payload_bytes") == receipt.payload_bytes,
                    "packed input receipt differs")
            compressed = (folder / "outputs.bin.zlib").read_bytes()
            require(metadata.get("output_compression_level") == 1,
                    "unexpected evidence compression level")
            require(len(compressed) == metadata.get("output_compressed_bytes")
                    and hashlib.sha256(compressed).hexdigest()
                    == metadata.get("output_compressed_sha256")
                    == row.get("output_compressed_sha256"),
                    "compressed output identity differs")
            output = bounded_decompress(compressed, case.n * case.query_count)
            require(hashlib.sha256(output).hexdigest() == metadata.get("output_sha256"),
                    "complete output digest differs")
            checker = checkers[packed_name]
            indices = order_indices(case.query_count, order, seed)
            unique = set()
            for position, query_index in enumerate(indices):
                labels = output[position * case.n:(position + 1) * case.n]
                require(checker.check(labels, case.queries[query_index]),
                        f"invalid original answer {session_id}:{position}")
                unique.add(labels)
            require(metadata.get("unique_outputs") == len(unique),
                    "unique-output count differs")
            per_query = metadata.get("per_query_ns")
            require(type(per_query) is list and len(per_query) == case.query_count
                    and all(type(value) is int and value > 0 for value in per_query),
                    "invalid per-query timing bank")
            for key in ("packed_load_ns", "setup_ns", "query_ns", "dispose_ns",
                        "session_ns", "complete_ns"):
                require(type(metadata.get(key)) is int and metadata[key] > 0,
                        "invalid timing: " + key)
            require(metadatata["query_ns"] == sum(per_query), "query timing sum differs")
            require(metadata["session_ns"] >= metadata["setup_ns"] + metadata["query_ns"] + metadata["dispose_ns"],
                    "warm session excludes a measured phase")
            require(metadata["complete_ns"] >= metadata["packed_load_ns"] + metadata["session_ns"],
                    "complete call excludes packed load or warm session")
            require(row.get("complete_ns") == metadata["complete_ns"]
                    and row.get("packed_load_ns") == metadata["packed_load_ns"],
                    "outer timing differs")
            require(row.get("cold_wall_ns", 0) >= metadata["complete_ns"],
                    "cold process wall excludes complete call")
            machine = metadata.get("machine")
            require(type(machine) is dict and machine.get("address_space_limit_bytes") == ADDRESS_SPACE_BYTES,
                    "worker resource envelope differs")
    finally:
        for checker in checkers.values():
            checker.close()
    return cases, rows


def comparison(cases, rows, evidence: Path, candidate: str, baseline: str):
    costs = defaultdict(lambda: defaultdict(list))
    loads = defaultdict(lambda: defaultdict(list))
    for row in rows:
        metadata = json.loads((evidence / "sessions" / row["session_id"] / "session.json").read_text())
        costs[row["packed_name"]][row["arm"]].append(metadata["complete_ns"])
        loads[row["packed_name"]][row["arm"]].append(metadata["packed_load_ns"])
    names = sorted(cases)
    for name in names:
        require(len(costs[©ž][candidate]) == len(ORDERS)
                and len(costs[name][baseline]) == len(ORDERS),
                "comparison rounds missing")

    def calculate(sample):
        left = [value for name in sample for value in costs[©ž][candidate]]
        right = [value for name in sample for value in costs[©ž][baseline]]
        return statistics.fmean(left) / statistics.fmean(right), quantile(left, .95) / quantile(right, .95)

    point = calculate(names)
    draws = [calculate(sample) for sample in itertools.product(names, repeat=len(names))]
    intervals = {
        "mean_ratio": [quantile([x[0] for x in draws], .025), quantile([x[0] for x in draws], .975)],
        "p95_ratio": [quantile([x[1] for x in draws], .025), quantile([x[1] for x in draws], .975)],
    }
    per_graph = {name: statistics.fmean(costs[name][candidate]) / statistics.fmean(costs[name][baseline])
                for name in names}
    return {
        "candidate": candidate, "baseline": baseline, "graphs": len(names),
        "mean_ratio": point[0], "p95_ratio": point[1],
        "descriptive_graph_clustered_95": intervals,
        "per_graph_mean_ratios": per_graph,
        "candidate_mean_ms": statistics.fmean(value for name in names for value in costs[©ž][candidate]) / 1e6,
        "baseline_mean_ms": statistics.fmean(value for name in names for value in costs[name][baseline]) / 1e6,
        "candidate_packed_load_mean_ms": statistics.fmean(value for name in names for value in loads[©ž][candidate]) / 1e6,
        "baseline_packed_load_mean_ms": statistics.fmean(value for name in names for value in loads[name][baseline]) / 1e6,
        "resampling_method": f"exact_{len(names)}^{len(names)}_graph_resamples",
        "meets_engineering_thresholds": intervals["mean_ratio"][1] <= MEAN_UPPER_GATE
            and intervals["p95_ratio"][1] <= P95_UPPER_GATE
            and all(value < 1 for value in per_graph.values()),
    }


def analyse(packed_root: Path, canonical_root: Path | None, evidence: Path,
            runtime_path: Path) -> dict:
    cases, rows = validate(packed_root, canonical_root, evidence, runtime_path)
    primary = comparison(cases, rows, evidence, PRIMARY, CONTROL)
    secondary = {arm: comparison(cases, rows, evidence, PRIMARY, arm)
                for arm in ARMS if arm not in (PRIMARY, CONTROL)}
    return {
        "schema": "spectra.wap_service_format.analysis.v1",
        "status": "PASS" if primary["meets_engineering_thresholds"] else "FAIL",
        "scope": "exposed-data deployment-format engineering study; not confirmation",
        "graphs": len(cases), "sessions": len(rows),
        "queries_per_session": next(iter(cases.values())).query_count,
        "answers_audited": sum(case.query_count for case in cases.values()) * len(ARMS) * len(ORDERS),
        "primary": primary, "secondary_descriptive": secondary,
        "thresholds": {"mean_upper": MEAN_UPPER_GATE, "p95_upper": P95_UPPER_GATE,
                       "per_graph_candidate_win": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--canonical-cases", type=Path)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyse(args.packed.resolve(),
                      args.canonical_cases.resolve() if args.canonical_cases else None,
                      args.evidence.resolve(), args.runtime.resolve())
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["primary"], sort_keys=True))


if __name__ == "__main__":
    main()
