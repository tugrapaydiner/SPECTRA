"""Verify the prospective WAP holdout freeze before any holdout acquisition.

This utility is intentionally standard-library only.  It binds source bytes,
protocol constants, upstream graph identities, the public freeze commit, and the
single marker commit that authorizes one holdout opening.  It does not download
or inspect graph bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
FREEZE_REL = "experiments/wap_support/FREEZE.json"
MARKER_REL = "experiments/wap_support/HOLDOUT_OPEN.json"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def strict_json_bytes(payload: bytes, label: str) -> dict:
    def pairs(values):
        result = {}
        for key, value in values:
            require(type(key) is str and key not in result,
                    f"duplicate or non-string key in {label}: {key!r}")
            result[key] = value
        return result

    value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs)
    require(type(value) is dict, label + " must be a JSON object")
    return value


def strict_json(path: Path, label: str) -> dict:
    require(path.is_file(), "missing " + label)
    return strict_json_bytes(path.read_bytes(), label)


def git(*arguments: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, input=input_bytes,
        capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError(
            "git command failed: " + " ".join(arguments) + "\n"
            + result.stderr.decode("utf-8", errors="replace")
        )
    return result.stdout


def commit_id(value: object, label: str) -> str:
    require(type(value) is str and HEX40.fullmatch(value), "invalid " + label)
    resolved = git("rev-parse", value + "^{commit}").decode().strip()
    require(resolved == value, label + " does not identify the exact commit")
    return value


def safe_path(value: object) -> str:
    require(type(value) is str and value, "invalid frozen source path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts
            and str(path) == value and "\\" not in value,
            "unsafe frozen source path: " + value)
    return value


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_bytes(commit: str, path: str) -> bytes:
    return git("show", f"{commit}:{path}")


def verify_source_inventory(freeze: dict) -> dict:
    source_commit = commit_id(freeze.get("source_commit"), "source_commit")
    sources = freeze.get("sources")
    require(type(sources) is dict and sources, "empty frozen source inventory")
    observed = {}
    for raw_path, digest in sorted(sources.items()):
        path = safe_path(raw_path)
        require(type(digest) is str and HEX64.fullmatch(digest),
                "invalid source digest: " + path)
        committed = source_bytes(source_commit, path)
        require(sha256(committed) == digest,
                "source commit digest differs: " + path)
        working = ROOT / path
        require(working.is_file() and sha256(working.read_bytes()) == digest,
                "checked-out source differs: " + path)
        observed[path] = digest
    return {"source_commit": source_commit, "sources": observed}


def verify_constants(freeze: dict) -> None:
    from experiments.wap_support import analyse, controls, run, workload

    expected = {
        "query_count": workload.QUERY_COUNT,
        "query_width": workload.QUERY_WIDTH,
        "max_generator_attempts": workload.MAX_GENERATOR_ATTEMPTS,
        "min_unique_oracle_models": workload.MIN_UNIQUE_ORACLE_MODELS,
        "master_seed": workload.MASTER_SEED,
        "arms": list(controls.ARMS),
        "orders": list(run.ORDERS),
        "schedule_seed": run.SCHEDULE_SEED,
        "max_work": controls.MAX_WORK,
        "max_native_bytes": controls.MAX_BYTES,
        "model_cache_size": controls.CACHE_SIZE,
        "address_space_bytes": run.ADDRESS_SPACE_BYTES,
        "session_deadline_seconds": run.SESSION_DEADLINE_SECONDS,
        "primary_candidate": analyse.PRIMARY,
        "primary_control": analyse.CONTROL,
        "mean_ratio_upper_gate": analyse.MEAN_UPPER_GATE,
        "p95_ratio_upper_gate": analyse.P95_UPPER_GATE,
        "python_sat": analyse.EXPECTED_PYSAT,
    }
    require(freeze.get("constants") == expected,
            "frozen constants differ from executable protocol")
    require(sys.version_info[:2] == (3, 13), "holdout requires Python 3.13")


def verify_upstream(freeze: dict) -> None:
    inventory_path = ROOT / "experiments/quotient_application/WAP_UPSTREAM_INVENTORY.json"
    inventory = strict_json(inventory_path, "upstream inventory")
    require(inventory.get("schema") == "spectra.wap.upstream_inventory.v2",
            "unexpected upstream inventory schema")
    holdouts = {
        name: {"bytes": record["bytes"], "git_blob_sha1": record["git_blob_sha1"]}
        for name, record in sorted(inventory["graphs"].items())
        if record.get("role") == "prospective_holdout"
    }
    require(len(holdouts) == 5, "holdout inventory must contain exactly five graphs")
    expected = {
        "repository": inventory["repository"],
        "commit": inventory["commit"],
        "role": "prospective_holdout",
        "graphs": holdouts,
    }
    require(freeze.get("upstream") == expected,
            "frozen upstream holdout inventory differs")


def changed_paths(left: str, right: str) -> list[str]:
    raw = git("diff", "--name-only", "--no-renames", left + ".." + right)
    return [line for line in raw.decode().splitlines() if line]


def verify_marker(freeze_path: Path, freeze: dict, marker_path: Path) -> dict:
    marker = strict_json(marker_path, "holdout marker")
    require(set(marker) == {"schema", "freeze_commit", "freeze_sha256", "opened_utc"},
            "unexpected marker fields")
    require(marker.get("schema") == "spectra.wap_support.holdout_open.v1",
            "unexpected marker schema")
    freeze_commit = commit_id(marker.get("freeze_commit"), "freeze_commit")
    freeze_bytes = freeze_path.read_bytes()
    require(type(marker.get("freeze_sha256")) is str
            and marker["freeze_sha256"] == sha256(freeze_bytes),
            "marker freeze digest differs")
    require(type(marker.get("opened_utc")) is str
            and re.fullmatch(r"20\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ", marker["opened_utc"]),
            "invalid marker timestamp")
    committed_freeze = source_bytes(freeze_commit, FREEZE_REL)
    require(committed_freeze == freeze_bytes,
            "freeze bytes differ from public freeze commit")
    source_commit = freeze["source_commit"]
    parent = git("rev-parse", freeze_commit + "^").decode().strip()
    require(parent == source_commit, "freeze commit is not directly above source commit")
    require(changed_paths(source_commit, freeze_commit) == [FREEZE_REL],
            "freeze commit changed files other than FREEZE.json")
    head = git("rev-parse", "HEAD").decode().strip()
    require(changed_paths(freeze_commit, head) == [MARKER_REL],
            "post-freeze history changed files other than the opening marker")
    require(not git("status", "--porcelain"), "working tree is not clean")
    return {"freeze_commit": freeze_commit, "head_commit": head, "marker": marker}


def verify(freeze_path: Path, marker_path: Path | None) -> dict:
    freeze = strict_json(freeze_path, "freeze")
    require(set(freeze) == {
        "schema", "source_commit", "sources", "upstream", "constants",
        "statistics", "development_receipts", "claims", "stop_rule",
    }, "unexpected freeze fields")
    require(freeze.get("schema") == "spectra.wap_support.freeze.v1",
            "unexpected freeze schema")
    source = verify_source_inventory(freeze)
    verify_constants(freeze)
    verify_upstream(freeze)
    statistics = freeze.get("statistics")
    require(statistics == {
        "independent_graph_clusters": 5,
        "exact_graph_resamples": 3125,
        "orders_per_graph": 3,
        "answers_expected": 107520,
    }, "frozen statistical contract differs")
    claims = freeze.get("claims")
    require(type(claims) is dict
            and claims.get("scope") == "synthetic list/query workload on public WAP optical conflict topologies"
            and claims.get("not_claimed") == [
                "ordinary chromatic-number superiority",
                "deployed optical-network traces",
                "general SAT superiority",
                "learned reasoning",
                "algorithmic novelty",
                "independent-team replication",
                "external adoption",
            ], "claim boundary differs")
    require(freeze.get("stop_rule") == {
        "no_post_opening_tuning": True,
        "failure_is_preserved": True,
        "changes_require_new_unopened_data": True,
    }, "stop rule differs")
    result = {
        "schema": "spectra.wap_support.freeze_verification.v1",
        "freeze_sha256": sha256(freeze_path.read_bytes()),
        **source,
        "constants_verified": True,
        "upstream_verified_without_graph_access": True,
    }
    if marker_path is not None:
        result.update(verify_marker(freeze_path, freeze, marker_path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.freeze.resolve(), args.marker.resolve() if args.marker else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
