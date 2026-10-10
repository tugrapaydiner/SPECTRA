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

from experiments.wap_support.controls import ARMS, order_indices
from experiments.wap_support.run import (
    ADDRESS_SPACE_BYTES,
    ORDERS,
    schedule,
)
from experiments.wap_support.workload import (
    QUERY_COUNT,
    WorkloadCase,
    build_case,
)
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


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def quantile(values, probability):
    values = sorted(values)
    require(values, "empty quantile")
    require(0 <= probability <= 1, "invalid quantile probability")
    position = (len(values) - 1) * probability
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def validate_manifest(root: Path) -> dict:
    require(root.is_dir(), "evidence root is not a directory")
    manifest_path = root / "MANIFEST.json"
    require(manifest_path.is_file(), "missing evidence manifest")
    manifest = json.loads(manifest_path.read_text())
    require(type(manifest) is dict, "manifest must be an object")
    for name, digest in manifest.items():
        require(type(name) is str and type(digest) is str and len(digest) == 64,
                "invalid manifest entry")
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), "manifest escapes evidence root")
        require(path.is_file() and sha256(path) == digest,
                "evidence hash mismatch: " + name)
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    require(set(manifest) == actual, "manifest file inventory differs")
    return manifest


def bounded_decompress(compressed: bytes, expected_bytes: int) -> bytes:
    require(type(expected_bytes) is int and expected_bytes >= 0,
            "invalid expected output size")
    decompressor = zlib.decompressobj()
    output = decompressor.decompress(compressed, expected_bytes + 1)
    require(len(output) <= expected_bytes, "compressed output expands beyond contract")
    require(decompressor.eof, "compressed output is truncated or exceeds contract")
    require(not decompressor.unconsumed_tail and not decompressor.unused_data,
            "compressed output has trailing or unconsumed bytes")
    output += decompressor.flush()
    require(len(output) == expected_bytes, "decompressed output length differs")
    return output


def validate_case_package(case_root: Path, inventory_path: Path,
                          expected_roles: set[str] | None) -> dict[str, WorkloadCase]:
    validate_manifest(case_root)
    inventory = json.loads(inventory_path.read_text())
    require(inventory.get("schema") == "spectra.wap.upstream_inventory.v2",
            "unexpected upstream inventory schema")
    require(type(inventory.get("graphs")) is dict, "invalid upstream graph inventory")
    case_inventory = json.loads((case_root / "case_inventory.json").read_text())
    require(case_inventory.get("schema") == "spectra.wap_support.case_inventory.v1",
            "unexpected case inventory schema")
    require(case_inventory.get("upstream_repository") == inventory.get("repository")
            and case_inventory.get("upstream_commit") == inventory.get("commit"),
            "case package upstream identity differs")
    require(case_inventory.get("inventory_sha256") == sha256(inventory_path),
            "case package inventory digest differs")
    roles = case_inventory.get("roles")
    require(type(roles) is list and all(type(role) is str for role in roles),
            "invalid case roles")
    role_set = set(roles)
    if expected_roles is not None:
        require(role_set == expected_roles, "case package roles differ from protocol")
    expected_names = sorted(
        name for name, record in inventory["graphs"].items()
        if record.get("role") in role_set
    )
    receipts = case_inventory.get("cases")
    require(type(receipts) is list and len(receipts) == len(expected_names),
            "case receipt inventory differs")
    by_name = {}
    for receipt in receipts:
        require(type(receipt) is dict and type(receipt.get("name")) is str,
                "invalid case receipt")
        name = receipt["name"]
        require(name not in by_name, "duplicate case receipt")
        by_name[name] = receipt
    require(sorted(by_name) == expected_names, "selected graph inventory differs")
    actual_graphs = sorted(path.name for path in (case_root / "graphs").glob("*.col"))
    actual_cases = sorted(path.name.removesuffix(".json")
                          for path in (case_root / "cases").glob("*.json"))
    actual_generators = sorted(path.name.removesuffix(".json")
                               for path in (case_root / "generators").glob("*.json"))
    require(actual_graphs == actual_cases == actual_generators == expected_names,
            "case package file roles differ")

    cases: dict[str, WorkloadCase] = {}
    for name in expected_names:
        record = inventory["graphs"][name]
        receipt = by_name[name]
        require(receipt.get("role") == record.get("role"), "graph role differs: " + name)
        graph_path = case_root / "graphs" / name
        case_path = case_root / "cases" / f"{name}.json"
        generator_path = case_root / "generators" / f"{name}.json"
        graph_bytes = graph_path.read_bytes()
        require(len(graph_bytes) == record.get("bytes"), "graph byte length differs: " + name)
        require(git_blob_sha1(graph_bytes) == record.get("git_blob_sha1"),
                "graph Git blob differs: " + name)
        case = WorkloadCase.from_json(case_path)
        require(case.graph_name == name and case.graph_sha256 == sha256(graph_path)
                and case.graph_git_blob_sha1 == record["git_blob_sha1"],
                "case graph identity differs: " + name)
        require(receipt.get("case_sha256") == case.case_sha256
                and receipt.get("case_file_sha256") == sha256(case_path)
                and receipt.get("generator_file_sha256") == sha256(generator_path)
                and receipt.get("graph_sha256") == sha256(graph_path)
                and receipt.get("graph_git_blob_sha1") == record["git_blob_sha1"],
                "case receipt digest differs: " + name)
        require(type(receipt.get("construction_ns")) is int
                and receipt["construction_ns"] > 0,
                "invalid construction time: " + name)
        generator = json.loads(generator_path.read_text())
        require(generator.get("schema") == "spectra.wap_support.generator_receipt.v1"
                and generator.get("case_sha256") == case.case_sha256,
                "generator identity differs: " + name)
        require(generator.get("graph_role") == record["role"]
                and generator.get("graph_bytes") == len(graph_bytes)
                and generator.get("graph_sha256") == sha256(graph_path)
                and generator.get("construction_ns") == receipt["construction_ns"],
                "generator graph receipt differs: " + name)
        attempts = generator.get("attempts")
        require(type(attempts) is list and len(attempts) == case.generator_attempts,
                "generator attempt inventory differs: " + name)
        accepted = [entry for entry in attempts
                    if not entry.get("duplicate") and entry.get("sat") is True]
        rejected = [entry for entry in attempts
                    if not entry.get("duplicate") and entry.get("sat") is False]
        require(len(accepted) == case.query_count == generator.get("accepted")
                and len(rejected) == case.generator_rejections == generator.get("rejected"),
                "generator acceptance counters differ: " + name)
        require([entry.get("query_sha256") for entry in accepted]
                == [canonical_sha256(query) for query in case.queries],
                "accepted query order differs: " + name)
        model_hashes = {entry.get("model_sha256") for entry in accepted}
        require(None not in model_hashes
                and len(model_hashes) == case.generator_unique_model_hashes
                == generator.get("unique_oracle_models"),
                "generator model diversity differs: " + name)
        cases[f"{name}.json"] = case
    return cases


def rebuild_cases(case_root: Path, inventory_path: Path) -> None:
    inventory = json.loads(inventory_path.read_text())
    for case_path in sorted((case_root / "cases").glob("*.json")):
        name = case_path.name.removesuffix(".json")
        expected = WorkloadCase.from_json(case_path)
        rebuilt, generator = build_case(
            case_root / "graphs" / name,
            graph_name=name,
            expected_git_blob_sha1=inventory["graphs"][name]["git_blob_sha1"],
        )
        require(rebuilt.case_sha256 == expected.case_sha256,
                "case reconstruction differs: " + name)
        require(generator["case_sha256"] == expected.case_sha256,
                "generator reconstruction differs: " + name)


def validate(case_root: Path, evidence: Path, runtime_path: Path,
             inventory_path: Path, *, reconstruct: bool,
             expected_roles: set[str] | None = None) -> tuple[dict, list[dict]]:
    cases = validate_case_package(case_root, inventory_path, expected_roles)
    validate_manifest(evidence)
    if reconstruct:
        rebuild_cases(case_root, inventory_path)
    expected_schedule = schedule(sorted((case_root / "cases").glob("*.json")))
    require(json.loads((evidence / "schedule.json").read_text())
            == [list(item) for item in expected_schedule], "schedule differs")
    rows = json.loads((evidence / "sessions.json").read_text())
    require(type(rows) is list and len(rows) == len(expected_schedule),
            "session inventory incomplete")
    environment = json.loads((evidence / "environment.json").read_text())
    require(environment.get("address_space_limit_bytes") == ADDRESS_SPACE_BYTES,
            "address-space envelope differs")
    require(type(environment.get("affinity")) is list and environment["affinity"],
            "invalid affinity receipt")
    if environment.get("runtime_library_sha256") is not None:
        require(environment["runtime_library_sha256"] == sha256(runtime_path),
                "runtime library identity differs")
    if environment.get("python_sat_version") is not None:
        require(environment["python_sat_version"] == EXPECTED_PYSAT,
                "PySAT version differs")

    runtime = QuotientRuntime(runtime_path)
    checkers = {
        name: runtime.checker(case.n, case.palette, case.edges, case.masks,
                              512 * 1024 * 1024)
        for name, case in cases.items()
    }
    try:
        seen = set()
        for row, job in zip(rows, expected_schedule, strict=True):
            require(type(row) is dict, "invalid session row")
            case_name, arm, order, seed = job
            require((row.get("case_file"), row.get("arm"), row.get("order"),
                     row.get("order_seed")) == (case_name, arm, order, seed),
                    "session order differs")
            session_id = row.get("session_id")
            require(type(session_id) is str and session_id not in seen,
                    "duplicate or invalid session id")
            seen.add(session_id)
            folder = (evidence / "sessions" / session_id).resolve()
            require(folder.is_relative_to((evidence / "sessions").resolve())
                    and folder.is_dir(), "session folder escapes evidence")
            metadata_path = folder / "session.json"
            outer_path = folder / "outer.json"
            metadata = json.loads(metadata_path.read_text())
            outer = json.loads(outer_path.read_text())
            require(outer == row, "outer session receipt differs")
            require(sha256(metadata_path) == row.get("session_metadata_sha256"),
                    "session metadata hash differs")
            case = cases[case_name]
            require(metadata.get("schema") == "spectra.wap_support.session.v1"
                    and metadata.get("arm") == arm
                    and metadata.get("case_sha256") == case.case_sha256
                    and metadata.get("order_name") == order,
                    "session semantic identity differs")
            require(metadata.get("status") == "COMPLETE"
                    and metadata.get("query_count") == case.query_count == QUERY_COUNT,
                    "incomplete session")
            require(metadata.get("output_file") == "outputs.bin.zlib",
                    "unexpected output path")
            output_path = (folder / metadata["output_file"]).resolve()
            require(output_path.is_relative_to(folder), "output path escapes session")
            compressed = output_path.read_bytes()
            require(len(compressed) == metadata.get("output_compressed_bytes"),
                    "compressed output size differs")
            require(hashlib.sha256(compressed).hexdigest()
                    == metadata.get("output_compressed_sha256")
                    == row.get("output_compressed_sha256"),
                    "compressed output hash differs")
            output = bounded_decompress(compressed, case.n * case.query_count)
            require(hashlib.sha256(output).hexdigest() == metadata.get("output_sha256"),
                    "output digest differs")
            indices = order_indices(case.query_count, order, seed)
            checker = checkers[case_name]
            unique_outputs = set()
            for position, query_index in enumerate(indices):
                labels = output[position * case.n:(position + 1) * case.n]
                require(checker.check(labels, case.queries[query_index]),
                        f"invalid output {session_id}:{position}")
                unique_outputs.add(labels)
            require(metadata.get("unique_outputs") == len(unique_outputs),
                    "unique-output counter differs")
            per_query = metadata.get("per_query_ns")
            require(type(per_query) is list and len(per_query) == case.query_count
                    and all(type(value) is int and value > 0 for value in per_query),
                    "invalid per-query timing bank")
            for key in ("setup_ns", "query_ns", "dispose_ns", "session_ns", "complete_ns"):
                require(type(metadata.get(key)) is int and metadata[key] > 0,
                        "invalid timing " + key)
            require(metadata["query_ns"] == sum(per_query), "query timing sum differs")
            require(metadata["session_ns"] >= (metadata["setup_ns"]
                    + metadata["query_ns"] + metadata["dispose_ns"]),
                    "session time excludes measured phases")
            require(metadata["complete_ns"] >= metadata["session_ns"]
                    and row.get("complete_ns") == metadata["complete_ns"],
                    "complete time excludes session or differs")
            require(type(row.get("cold_wall_ns")) is int
                    and row["cold_wall_ns"] >= metadata["complete_ns"],
                    "cold wall time excludes complete call")
            require(type(metadata.get("cache_hits")) is int
                    and 0 <= metadata["cache_hits"] <= case.query_count,
                    "invalid cache-hit counter")
            machine = metadata.get("machine")
            require(type(machine) is dict
                    and machine.get("address_space_limit_bytes") == ADDRESS_SPACE_BYTES
                    and machine.get("threads", 0) >= 1,
                    "session machine envelope differs")
    finally:
        for checker in checkers.values():
            checker.close()
    return cases, rows


def comparison(cases: dict, rows: list[dict], evidence: Path,
               candidate: str, baseline: str) -> dict:
    require(candidate in ARMS and baseline in ARMS and candidate != baseline,
            "invalid comparison arms")
    costs = defaultdict(dict)
    for row in rows:
        metadata = json.loads(
            (evidence / "sessions" / row["session_id"] / "session.json").read_text()
        )
        costs[row["case_file"]].setdefault(row["arm"], []).append(metadata["complete_ns"])
    names = sorted(cases)
    require(names, "empty graph panel")
    for name in names:
        require(len(costs[name][candidate]) == len(ORDERS), "candidate rounds missing")
        require(len(costs[name][baseline]) == len(ORDERS), "baseline rounds missing")

    def calculate(sample):
        left = [value for name in sample for value in costs[name][candidate]]
        right = [value for name in sample for value in costs[name][baseline]]
        return (statistics.fmean(left) / statistics.fmean(right),
                quantile(left, .95) / quantile(right, .95))

    point = calculate(names)
    if len(names) <= 6:
        samples = itertools.product(names, repeat=len(names))
        method = f"exact_{len(names)}^{len(names)}_graph_resamples"
    else:
        import random
        rng = random.Random(739291)
        samples = ([rng.choice(names) for _ in names] for _ in range(10000))
        method = "fixed_seed_10000_graph_resamples"
    draws = [calculate(sample) for sample in samples]
    intervals = {
        "mean_ratio": [quantile([draw[0] for draw in draws], .025),
                       quantile([draw[0] for draw in draws], .975)],
        "p95_ratio": [quantile([draw[1] for draw in draws], .025),
                      quantile([draw[1] for draw in draws], .975)],
    }
    per_graph = {
        name: statistics.fmean(costs[name][candidate])
        / statistics.fmean(costs[name][baseline])
        for name in names
    }
    thresholds_pass = (
        intervals["mean_ratio"][1] <= MEAN_UPPER_GATE
        and intervals["p95_ratio"][1] <= P95_UPPER_GATE
        and all(value < 1 for value in per_graph.values())
    )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "graphs": len(names),
        "mean_ratio": point[0],
        "p95_ratio": point[1],
        "descriptive_graph_clustered_95": intervals,
        "per_graph_mean_ratios": per_graph,
        "resampling_method": method,
        "meets_primary_numeric_thresholds": thresholds_pass,
    }


def analyse(case_root: Path, evidence: Path, runtime_path: Path,
            inventory_path: Path, *, reconstruct: bool,
            expected_roles: set[str] | None = None) -> dict:
    cases, rows = validate(
        case_root,
        evidence,
        runtime_path,
        inventory_path,
        reconstruct=reconstruct,
        expected_roles=expected_roles,
    )
    primary = comparison(cases, rows, evidence, PRIMARY, CONTROL)
    secondary = {
        arm: comparison(cases, rows, evidence, PRIMARY, arm)
        for arm in ARMS
        if arm not in (PRIMARY, CONTROL)
    }
    primary_gate = primary["meets_primary_numeric_thresholds"]
    return {
        "schema": "spectra.wap_support.analysis.v2",
        "graphs": len(cases),
        "sessions": len(rows),
        "queries_per_session": QUERY_COUNT,
        "answers_audited": len(rows) * QUERY_COUNT,
        "primary": primary,
        "secondary_descriptive": secondary,
        "thresholds": {"mean_upper": MEAN_UPPER_GATE,
                       "p95_upper": P95_UPPER_GATE,
                       "per_graph_candidate_win": True},
        "verdict": "PASS" if primary_gate else "FAIL",
        "scope": "synthetic list/query workload on public WAP optical conflict topologies",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reconstruct-cases", action="store_true")
    parser.add_argument("--expected-role", action="append", default=[])
    args = parser.parse_args()
    result = analyse(
        args.cases.resolve(),
        args.evidence.resolve(),
        args.runtime.resolve(),
        args.inventory.resolve(),
        reconstruct=args.reconstruct_cases,
        expected_roles=set(args.expected_role) if args.expected_role else None,
    )
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["primary"], sort_keys=True))


if __name__ == "__main__":
    main()
