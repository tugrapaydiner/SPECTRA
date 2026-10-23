"""Selection replay and all-or-nothing acquisition evidence contracts."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import random
import sqlite3
from urllib.error import URLError

import pytest

from experiments.general_competition import exposed_inventory as inventory


def row(number: int, *, family="family", expected="sat", year=2023, **values):
    identity = f"{number:032x}"
    result = {"id": identity, "family": family, "expected": expected, "year": year,
              "isohash": f"iso-{number}", "isohash2": f"iso2-{number}",
              "source_url": inventory.BASE_URL + identity, "metadata_bytes": "10",
              "variables": "2", "clauses": "1"}
    return {**result, **values}


def receipt(tmp_path, instances):
    path = tmp_path / "selection.json"
    inventory._write_new(path, {"schema": inventory.SCHEMA,
                              "selected_count": len(instances), "instances": instances,
                              "claim_boundary": "exposed development only"})
    return path


def quotas(sat=0, unsat=0, unknown=0):
    return {"sat": sat, "unsat": unsat, "unknown": unknown}


def test_selection_and_exclusions_survive_input_and_json_key_reordering():
    records = [row(year * 1000 + verdict_index * 100 + index,
                   family=f"family-{index % 3}", expected=expected, year=year)
               for year in (2023, 2024)
               for verdict_index, expected in enumerate(("sat", "unsat", "unknown"))
               for index in range(9)]
    declared = quotas(4, 3, 2)
    chosen, excluded = inventory.select_records(records, years=(2023, 2024), quotas=declared)
    random.Random(29).shuffle(records)
    # Selection receipts use sorted JSON keys; this must not change strata order.
    serialized = json.loads(json.dumps(declared, sort_keys=True))
    assert list(serialized) == ["sat", "unknown", "unsat"]
    assert inventory.select_records(records, years=(2023, 2024), quotas=serialized) == (chosen, excluded)
    assert len(chosen) + len(excluded) == len(records)
    assert Counter((item["year"], item["expected"]) for item in chosen) == {
        (year, expected): count for year in (2023, 2024) for expected, count in declared.items()
    }


def test_family_round_robin_visits_each_family_before_repeating():
    records = [row(i, family="large") for i in range(1, 11)]
    records += [row(11, family="small-a"), row(12, family="small-b")]
    chosen, _ = inventory.select_records(records, years=(2023,), quotas=quotas(sat=4))
    assert len({item["family"] for item in chosen[:3]}) == 3
    assert chosen[3]["family"] == "large"


def test_solver_results_names_and_timings_cannot_change_selection():
    records = [row(i, family=f"family-{i % 3}") for i in range(1, 20)]
    original, _ = inventory.select_records(records, years=(2023,), quotas=quotas(sat=7))
    changed = [{**item, "filename": f"winner-{20-i}.cnf", "solver": "SPECTRA",
                "runtime": 10000 if i % 2 else 0.00001, "winner": bool(i % 2)}
               for i, item in enumerate(records)]
    replay, _ = inventory.select_records(changed, years=(2023,), quotas=quotas(sat=7))
    assert [item["id"] for item in original] == [item["id"] for item in replay]


def test_exact_and_both_isomorphism_hashes_deduplicate_across_years():
    records = [row(1), row(2), row(1, year=2024),
               row(3, year=2024, isohash="iso-1"),
               row(4, year=2024, isohash2="iso2-2"), row(5, year=2024), row(6, year=2024)]
    chosen, excluded = inventory.select_records(records, years=(2023, 2024), quotas=quotas(sat=2))
    assert {item["id"] for item in chosen} == {f"{i:032x}" for i in (1, 2, 5, 6)}
    assert len(excluded) == 3
    assert all(item["exclusion"].startswith("duplicate:") for item in excluded)
    for key in ("id", "isohash", "isohash2"):
        assert len({item[key] for item in chosen}) == 4


def test_absent_isomorphism_metadata_does_not_collapse_unrelated_formulas():
    records = [row(i, isohash=empty, isohash2=empty)
               for i, empty in enumerate((None, "", "empty", "unknown", "None"), 1)]
    chosen, excluded = inventory.select_records(records, years=(2023,), quotas=quotas(sat=5))
    assert len(chosen) == 5 and excluded == []


def test_deduplication_cannot_relax_a_declared_stratum_quota():
    with pytest.raises(ValueError, match="insufficient distinct 2024/sat"):
        inventory.select_records([row(1), row(1, year=2024)], years=(2023, 2024), quotas=quotas(sat=1))


@pytest.mark.parametrize("bad_years", [(), (2023, 2023), (True,), (2023.0,)])
def test_year_validation(bad_years):
    with pytest.raises(ValueError, match="years"):
        inventory.select_records([row(1)], years=bad_years, quotas=quotas(sat=1))


@pytest.mark.parametrize("bad_quotas", [quotas(), quotas(sat=-1), quotas(sat=True),
                                        quotas(sat=1.0), {"sat": 1}])
def test_quota_validation(bad_quotas):
    with pytest.raises(ValueError, match="quotas"):
        inventory.select_records([row(1)], years=(2023,), quotas=bad_quotas)


def test_duplicate_metadata_memberships_are_rejected():
    with pytest.raises(ValueError, match="duplicate year/GBD"):
        inventory.select_records([row(1), row(1)], years=(2023,), quotas=quotas(sat=1))


def databases(tmp_path, records):
    meta, base = tmp_path / "meta.db", tmp_path / "base.db"
    with sqlite3.connect(meta) as connection:
        connection.execute("CREATE TABLE track (hash TEXT, value TEXT)")
        connection.execute("CREATE TABLE features (hash TEXT, family TEXT, result TEXT, isohash TEXT, isohash2 TEXT)")
        for item in records:
            connection.execute("INSERT INTO track VALUES (?,?)", (item["id"], f"main_{item['year']}"))
            connection.execute("INSERT INTO features VALUES (?,?,?,?,?)",
                               (item["id"], item["family"], item["expected"], item["isohash"], item["isohash2"]))
    with sqlite3.connect(base) as connection:
        connection.execute("CREATE TABLE features (hash TEXT, bytes TEXT, variables TEXT, clauses TEXT)")
        for item in records:
            connection.execute("INSERT INTO features VALUES (?,?,?,?)",
                               (item["id"], item["metadata_bytes"], item["variables"], item["clauses"]))
    return meta, base


def test_database_selection_retains_every_size_and_metadata_exclusion(tmp_path):
    records = [row(1), row(2, expected="unsat"), row(3, expected="unknown"),
               row(4, metadata_bytes="11"), row(5, metadata_bytes="empty"),
               row(6, family="unknown"), row(7, expected="empty"), row(8)]
    meta, base = databases(tmp_path, records)
    hashes = (inventory.sha256(meta), inventory.sha256(base))
    out = tmp_path / "selection.json"
    result = inventory.select(meta, base, out, years=(2023,), quotas=quotas(1, 1, 1), max_metadata_bytes=10)
    assert result["selected_count"] == 3
    assert len(result["exclusions"]) == 5
    assert Counter(item["exclusion"] for item in result["exclusions"]) == {
        "predeclared_metadata_size_bound": 2, "missing_stratum_metadata": 2,
        "predeclared_stratum_quota": 1,
    }
    all_ids = [item["id"] for item in result["instances"] + result["exclusions"]]
    assert Counter(all_ids) == Counter(item["id"] for item in records)
    assert hashes == (result["metadata"]["meta_sha256"], result["metadata"]["base_sha256"])
    assert hashes == (inventory.sha256(meta), inventory.sha256(base))
    frozen_bytes = out.read_bytes()
    with pytest.raises(FileExistsError):
        inventory.select(meta, base, out, years=(2023,), quotas=quotas(1, 1, 1), max_metadata_bytes=10)
    assert out.read_bytes() == frozen_bytes


def test_acquisition_emits_complete_inventory_with_source_and_selection_hashes(tmp_path, monkeypatch):
    instances = [row(1), row(2, expected="unsat"), row(3, expected="unknown")]
    selected = receipt(tmp_path, instances)
    requested = []

    def fetch(request, timeout):
        requested.append((request.full_url, timeout))
        return io.BytesIO(request.full_url.encode())

    monkeypatch.setattr(inventory, "urlopen", fetch)
    destination = tmp_path / "download"
    result = inventory.acquire(selected, destination, timeout=7)
    emitted = json.loads((destination / "inventory.json").read_text())
    assert result["status"] == "ACQUIRED" and result["replacement_instances"] == 0
    assert result["selected_count"] == result["acquired_count"] == 3
    assert result["failures"] == []
    assert result["selection_sha256"] == emitted["selection_sha256"] == inventory.sha256(selected)
    assert requested == [(item["source_url"], 7) for item in instances]
    assert [item["expected"] for item in emitted["instances"]] == ["SAT", "UNSAT", "UNKNOWN"]
    for item in emitted["instances"]:
        assert item["sha256"] == inventory.sha256(destination / item["path"])
        assert item["source_bytes"] == len(item["source_url"].encode())


class Chunks:
    def __init__(self, values):
        self.values = iter(values)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        value = next(self.values, b"")
        if isinstance(value, Exception):
            raise value
        return value


@pytest.mark.parametrize("failure", ["network", "oversize", "empty"])
def test_acquisition_keeps_failures_and_partial_bytes_without_replacement(tmp_path, monkeypatch, failure):
    instances = [row(1), row(2), row(3)]
    selected = receipt(tmp_path, instances)
    calls = []

    def fetch(request, timeout):
        calls.append(request.full_url)
        if request.full_url == instances[1]["source_url"]:
            return Chunks({"network": [b"abc", URLError("connection lost")],
                           "oversize": [b"abc", b"def"], "empty": []}[failure])
        return io.BytesIO(b"ok")

    monkeypatch.setattr(inventory, "urlopen", fetch)
    destination = tmp_path / "download"
    result = inventory.acquire(selected, destination, max_source_bytes=4)
    assert result["status"] == "ACQUISITION_FAILED"
    assert result["selected_count"] == 3 and result["acquired_count"] == 2
    assert result["replacement_instances"] == 0
    assert calls == [item["source_url"] for item in instances]
    assert not (destination / "inventory.json").exists()
    partial = json.loads((destination / "acquired-partial.json").read_text())
    assert [item["id"] for item in partial["instances"]] == [instances[i]["id"] for i in (0, 2)]
    failed = result["failures"]
    assert len(failed) == 1 and failed[0]["id"] == instances[1]["id"]
    expected_bytes = b"" if failure == "empty" else b"abc"
    assert failed[0]["retained_partial_bytes"] == len(expected_bytes)
    assert failed[0]["partial_sha256"] == hashlib.sha256(expected_bytes).hexdigest()
    assert (destination / "instances" / (instances[1]["id"] + ".cnf.xz")).read_bytes() == expected_bytes
    assert json.loads((destination / "acquisition.json").read_text()) == result


def test_acquisition_digest_binds_to_original_selection_bytes(tmp_path, monkeypatch):
    selected = receipt(tmp_path, [row(1)])
    digest_before = inventory.sha256(selected)

    def fetch(request, timeout):
        selected.write_text("{}\n", encoding="utf-8")
        return io.BytesIO(b"original selected source")

    monkeypatch.setattr(inventory, "urlopen", fetch)
    destination = tmp_path / "download"
    result = inventory.acquire(selected, destination)
    assert inventory.sha256(selected) != digest_before
    assert result["selection_sha256"] == digest_before
    assert json.loads((destination / "inventory.json").read_text())["selection_sha256"] == digest_before


@pytest.mark.parametrize("case", ["count", "bool_count", "empty", "duplicate", "bad_id", "escape",
                                  "foreign_url", "wrong_url_id", "family", "expected", "missing_boundary"])
def test_invalid_selection_is_rejected_before_downloading(tmp_path, monkeypatch, case):
    selected = receipt(tmp_path, [row(1), row(2)])
    data = json.loads(selected.read_text())
    if case == "count":
        data["selected_count"] = 1
    elif case == "bool_count":
        data["instances"] = [row(1)]
        data["selected_count"] = True
    elif case == "empty":
        data["instances"] = []
        data["selected_count"] = 0
    elif case == "duplicate":
        data["instances"][1] = deepcopy(data["instances"][0])
    elif case in ("bad_id", "escape"):
        data["instances"][1]["id"] = "x" * 32 if case == "bad_id" else "../outside"
    elif case == "foreign_url":
        data["instances"][1]["source_url"] = "https://example.invalid/" + row(2)["id"]
    elif case == "wrong_url_id":
        data["instances"][1]["source_url"] = inventory.BASE_URL + row(3)["id"]
    elif case == "family":
        data["instances"][1]["family"] = None
    elif case == "expected":
        data["instances"][1]["expected"] = "unexpected"
    else:
        del data["claim_boundary"]
    selected.write_text(json.dumps(data), encoding="utf-8")
    calls = []
    monkeypatch.setattr(inventory, "urlopen", lambda *args, **kwargs: calls.append(args))
    destination = tmp_path / "download"
    with pytest.raises(ValueError):
        inventory.acquire(selected, destination)
    assert calls == [] and not destination.exists()


def test_existing_acquisition_directory_is_never_overwritten(tmp_path, monkeypatch):
    selected = receipt(tmp_path, [row(1)])
    destination = tmp_path / "download"
    destination.mkdir()
    marker = destination / "retained.txt"
    marker.write_bytes(b"previous failure")
    calls = []
    monkeypatch.setattr(inventory, "urlopen", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(FileExistsError):
        inventory.acquire(selected, destination)
    assert calls == [] and marker.read_bytes() == b"previous failure"
