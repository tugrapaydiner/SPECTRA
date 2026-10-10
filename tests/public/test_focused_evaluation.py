"""Study contracts use tiny fixtures, never the evaluation seed range."""
import copy
import itertools
import json

import pytest

from data.cnf import CNF
from experiments.focused_evaluation import analyse, study


def fixture():
    p = CNF(2, ((1, 2), (-1, 2)))
    case = {"id": "tiny:2:0", "sha256": p.sha256(), "formula": p.record()}
    rows = []
    for seed, arm, round_id in itertools.product(study.SEARCH_SEEDS, study.ARMS, range(study.ROUNDS)):
        result = (study.call(case, arm, seed) if arm != "glucose4" else
                  {"status": "SAT_VERIFIED", "witness": [False, True],
                   "stats": {"conflicts": 0}, "conflict_budget_requested": study.CONFLICTS})
        rows.append({"id": case["id"], "sha256": case["sha256"], "seed": seed,
                     "arm": arm, "round": round_id, "wall_ns": 100, "cpu_ns": 100,
                     "result": result})
    return [case], rows


def test_complete_rows_accept_and_timing_rounds_are_not_new_inputs():
    cases, rows = fixture()
    analyse.validate_rows(cases, rows)
    assert analyse.metrics(rows)["indexed"]["pairs"] == 2


@pytest.mark.parametrize("damage", ["missing", "duplicate", "formula", "seed", "witness", "budget", "trajectory", "timing"])
def test_damaged_observations_are_rejected(damage):
    cases, rows = fixture()
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif damage == "formula":
        rows[0]["sha256"] = "0"*64
    elif damage == "seed":
        rows[0]["result"]["seed"] = 123
    elif damage == "witness":
        rows[0]["result"]["witness"] = [False, False]
    elif damage == "budget":
        rows[0]["result"]["flips"] = 2049
    elif damage == "trajectory":
        rows[0]["result"]["path_sha256"] = "0"*64
    elif damage == "timing":
        rows[0]["cpu_ns"] = -1
    with pytest.raises(ValueError):
        analyse.validate_rows(cases, rows)


def test_empty_source_inventory_rejected():
    with pytest.raises(ValueError, match="incomplete source freeze"):
        study.check_freeze({"sources": {}})


def test_empty_artifact_inventory_rejected(tmp_path):
    for name in analyse.ARTIFACTS:
        (tmp_path/name).write_text("{}")
    (tmp_path/"MANIFEST.json").write_text(json.dumps({"status": "COMPLETE", "files": {}}))
    with pytest.raises(ValueError, match="incomplete artifact bindings"):
        analyse.load(tmp_path)


def test_generator_smoke_is_explicitly_outside_evaluation_range():
    spec = {"id": "smoke", "family": "planted3", "nvars": 12, "index": 0,
            "generator_seed": 202610059999}
    case = study.generate(spec)
    assert case == study.generate(spec)
    assert len(case["formula"]["clauses"]) == 50
    for arm in study.ARMS[:-1]:
        study.validate_result(case, arm, study.call(case, arm, 17))
