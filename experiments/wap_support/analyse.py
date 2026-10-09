"""Strict evidence audit and graph-clustered gate for WAP support queries."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import zlib

from experiments.wap_support.controls import ARMS
from experiments.wap_support.controls import order_indices
from experiments.wap_support.run import ORDERS, schedule
from experiments.wap_support.workload import WorkloadCase, build_case
from spectra.cnf.quotient_query import QuotientRuntime

PRIMARY = "scc"
CONTROL = "minicard"
MEAN_UPPER_GATE = 0.50
P95_UPPER_GATE = 1.10


def require(ok, message):
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


def validate_manifest(root: Path) -> dict:
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for name, digest in manifest.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), "manifest escapes evidence root")
        require(path.is_file() and sha256(path) == digest, "evidence hash mismatch: " + name)
    actual = {str(path.relative_to(root)) for path in root.rglob("*")
              if path.is_file() and path.name != "MANIFEST.json"}
    require(set(manifest) == actual, "manifest file inventory differs")
    return manifest


def rebuild_cases(case_root: Path, inventory_path: Path) -> None:
    inventory = json.loads(inventory_path.read_text())
    for case_path in sorted((case_root / "cases").glob("*.json")):
        name = case_path.name.removesuffix(".json")
        expected = WorkloadCase.from_json(case_path)
        rebuilt, _ = build_case(case_root / "graphs" / name, graph_name=name,
                                expected_git_blob_sha1=inventory["graphs"][name]["git_blob_sha1"])
        require(rebuilt.case_sha256 == expected.case_sha256,
                "case reconstruction differs: " + name)


def validate(case_root: Path, evidence: Path, runtime_path: Path,
             inventory_path: Path, *, reconstruct: bool) -> tuple[dict, list[dict]]:
    validate_manifest(case_root); validate_manifest(evidence)
    if reconstruct:
        rebuild_cases(case_root, inventory_path)
    cases = {path.name: WorkloadCase.from_json(path)
             for path in sorted((case_root / "cases").glob("*.json"))}
    expected_schedule = schedule(list(sorted((case_root / "cases").glob("*.json"))))
    require(json.loads((evidence / "schedule.json").read_text()) == [list(x) for x in expected_schedule],
            "schedule differs")
    rows = json.loads((evidence / "sessions.json").read_text())
    require(len(rows) == len(expected_schedule), "session inventory incomplete")
    runtime = QuotientRuntime(runtime_path)
    checkers = {name: runtime.checker(case.n, case.palette, case.edges, case.masks,
                                      512 * 1024 * 1024)
                for name, case in cases.items()}
    try:
        seen = set()
        for row, job in zip(rows, expected_schedule, strict=True):
            case_name, arm, order, seed = job
            require((row["case_file"], row["arm"], row["order"], row["order_seed"])
                    == (case_name, arm, order, seed), "session order differs")
            session_id = row["session_id"]
            require(session_id not in seen, "duplicate session id"); seen.add(session_id)
            folder = evidence / "sessions" / session_id
            metadata = json.loads((folder / "session.json").read_text())
            require(sha256(folder / "session.json") == row["session_metadata_sha256"],
                    "session metadata hash differs")
            compressed = (folder / metadata["output_file"]).read_bytes()
            require(hashlib.sha256(compressed).hexdigest() == metadata["output_compressed_sha256"],
                    "compressed output hash differs")
            output = zlib.decompress(compressed)
            require(hashlib.sha256(output).hexdigest() == metadata["output_sha256"],
                    "output digest differs")
            case = cases[case_name]
            require(len(output) == case.n * case.query_count, "output bank length differs")
            indices = order_indices(case.query_count, order, seed)
            checker = checkers[case_name]
            for position, query_index in enumerate(indices):
                labels = output[position * case.n:(position + 1) * case.n]
                require(checker.check(labels, case.queries[query_index]),
                        f"invalid output {session_id}:{position}")
            for key in ("setup_ns", "query_ns", "dispose_ns", "session_ns", "complete_ns"):
                require(type(metadata[key]) is int and metadata[key] > 0, "invalid timing " + key)
            require(metadata["complete_ns"] >= metadata["session_ns"], "complete time excludes session")
            require(metadata["status"] == "COMPLETE" and metadata["query_count"] == case.query_count,
                    "incomplete session")
    finally:
        for checker in checkers.values(): checker.close()
    return cases, rows


def comparison(cases: dict, rows: list[dict], evidence: Path,
               candidate: str, baseline: str) -> dict:
    costs = defaultdict(dict)
    for row in rows:
        metadata = json.loads((evidence / "sessions" / row["session_id"] / "session.json").read_text())
        costs[row["case_file"]].setdefault(row["arm"], []).append(metadata["complete_ns"])
    names = sorted(cases)
    for name in names:
        require(len(costs[name][candidate]) == len(ORDERS), "candidate rounds missing")
        require(len(costs[name][baseline]) == len(ORDERS), "baseline rounds missing")
    def calculate(sample):
        left = [x for name in sample for x in costs[name][candidate]]
        right = [x for name in sample for x in costs[name][baseline]]
        return statistics.fmean(left) / statistics.fmean(right), quantile(left,.95) / quantile(right,.95)
    point = calculate(names)
    # Exact graph-clustered bootstrap for five holdout graphs; deterministic Monte
    # Carlo with the same seed is used for a differently sized development panel.
    if len(names) <= 6:
        samples = itertools.product(names, repeat=len(names))
    else:
        import random
        rng = random.Random(739291)
        samples = ([rng.choice(names) for _ in names] for _ in range(10000))
    draws = [calculate(sample) for sample in samples]
    intervals = {
        "mean_ratio": [quantile([d[0] for d in draws], .025), quantile([d[0] for d in draws], .975)],
        "p95_ratio": [quantile([d[1] for d in draws], .025), quantile([d[1] for d in draws], .975)],
    }
    per_graph = {name: statistics.fmean(costs[name][candidate]) / statistics.fmean(costs[name][baseline])
                 for name in names}
    return {
        "candidate": candidate, "baseline": baseline, "graphs": len(names),
        "mean_ratio": point[0], "p95_ratio": point[1],
        "descriptive_graph_clustered_95": intervals,
        "per_graph_mean_ratios": per_graph,
        "gate": (intervals["mean_ratio"][1] <= MEAN_UPPER_GATE
                 and intervals["p95_ratio"][1] <= P95_UPPER_GATE
                 and all(value < 1 for value in per_graph.values())),
    }


def analyse(case_root: Path, evidence: Path, runtime_path: Path,
            inventory_path: Path, *, reconstruct: bool) -> dict:
    cases, rows = validate(case_root, evidence, runtime_path, inventory_path,
                           reconstruct=reconstruct)
    primary = comparison(cases, rows, evidence, PRIMARY, CONTROL)
    secondary = {arm: comparison(cases, rows, evidence, PRIMARY, arm)
                 for arm in ARMS if arm != PRIMARY}
    return {
        "schema": "spectra.wap_support.analysis.v1",
        "graphs": len(cases), "sessions": len(rows), "queries_per_session": 1024,
        "primary": primary, "secondary": secondary,
        "thresholds": {"mean_upper": MEAN_UPPER_GATE, "p95_upper": P95_UPPER_GATE},
        "verdict": "PASS" if primary["gate"] else "FAIL",
        "scope": "synthetic list/query workload on public WAP optical conflict topologies",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",type=Path,required=True)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--inventory",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--reconstruct-cases",action="store_true")
    args=parser.parse_args()
    result=analyse(args.cases.resolve(),args.evidence.resolve(),args.runtime.resolve(),
                   args.inventory.resolve(),reconstruct=args.reconstruct_cases)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result["primary"],sort_keys=True))

if __name__=="__main__":main()
