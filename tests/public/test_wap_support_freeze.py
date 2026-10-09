"""Unit contracts for the standard-library prospective freeze verifier."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.wap_support import analyse, controls, run, workload
from experiments.wap_support.verify_freeze import (
    safe_path, strict_json_bytes, verify_constants, verify_upstream,
)


def constants():
    return {
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


def upstream():
    path = Path("experiments/quotient_application/WAP_UPSTREAM_INVENTORY.json")
    inventory = json.loads(path.read_text())
    return {
        "repository": inventory["repository"],
        "commit": inventory["commit"],
        "role": "prospective_holdout",
        "graphs": {
            name: {"bytes": record["bytes"], "git_blob_sha1": record["git_blob_sha1"]}
            for name, record in sorted(inventory["graphs"].items())
            if record["role"] == "prospective_holdout"
        },
    }


def test_strict_json_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate"):
        strict_json_bytes(b'{"x":1,"x":2}', "test")


@pytest.mark.parametrize("path", ["/absolute", "../escape", "a/../b", "a\\b", ""])
def test_frozen_paths_must_be_safe_repository_paths(path):
    with pytest.raises(ValueError):
        safe_path(path)


def test_executable_constants_are_exactly_bound():
    verify_constants({"constants": constants()})
    changed = constants(); changed["query_count"] += 1
    with pytest.raises(ValueError, match="constants differ"):
        verify_constants({"constants": changed})


def test_upstream_holdout_inventory_is_exactly_bound():
    verify_upstream({"upstream": upstream()})
    changed = upstream(); changed["graphs"] = dict(changed["graphs"])
    first = next(iter(changed["graphs"]))
    changed["graphs"][first] = dict(changed["graphs"][first])
    changed["graphs"][first]["bytes"] += 1
    with pytest.raises(ValueError, match="inventory differs"):
        verify_upstream({"upstream": changed})
