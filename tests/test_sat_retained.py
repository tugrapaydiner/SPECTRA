"""Actual retained development records are separate from synthetic gate fixtures."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from scripts.verify_sat_admission import verify

ROOT = Path(__file__).resolve().parents[1]/"results/sat_workload"


def test_actual_retained_pilot_payloads_and_witnesses():
    report = verify(ROOT)
    assert report["integrity"] == "PASS"
    assert report["instances"] == 48 and report["observations"] == 288
    assert report["sat_witnesses_independently_checked"] == 189
    assert report["tier_gates"] == ["NO_FEASIBLE_TIER"]*6
    assert report["conflict_budget_diagnostic"] == {
        "cadical195": {"requested": 2000, "maximum_observed": 2004, "observations_above_request": 21},
        "glucose4": {"requested": 2000, "maximum_observed": 37797, "observations_above_request": 33}}


@pytest.mark.parametrize("mutation", ["missing", "bytes", "path", "digest", "source", "schema", "false_witness"])
def test_retained_evidence_rejects_corruption(tmp_path, mutation):
    for name in ("pilot.zip", "manifest.json"):
        shutil.copy(ROOT/name, tmp_path/name)
    path = tmp_path/"manifest.json"
    manifest = json.loads(path.read_text())
    archive_path = tmp_path/"pilot.zip"
    if mutation == "missing":
        archive_path.unlink()
    elif mutation == "bytes":
        raw = bytearray(archive_path.read_bytes()); raw[-1] ^= 1; archive_path.write_bytes(raw)
    elif mutation == "path":
        manifest["archive"] = "../outside"
    elif mutation == "digest":
        manifest["sha256"] = "0"*64
    elif mutation == "source":
        manifest["execution_commit"] = "0"*40
    elif mutation == "schema":
        manifest["schema"] = "other"
    else:
        # Re-seal both outer and member digests: semantic verification must still
        # reject a falsified witness, rather than merely trusting hash integrity.
        with zipfile.ZipFile(archive_path) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        rows = [json.loads(line) for line in members["rows.jsonl"].splitlines()]
        row = next(r for r in rows if r["status"] == "SAT_VERIFIED")
        case = next(c for c in json.loads(members["cases.json"]) if c["case_id"] == row["case_id"])
        for literal in case["formula"]["clauses"][0]:
            row["witness"][abs(literal)-1] = literal < 0
        members["rows.jsonl"] = "".join(json.dumps(r, sort_keys=True)+"\n" for r in rows).encode()
        hashes = json.loads(members["sha256.json"])
        hashes["rows.jsonl"] = hashlib.sha256(members["rows.jsonl"]).hexdigest()
        members["sha256.json"] = (json.dumps(hashes, indent=2, sort_keys=True)+"\n").encode()
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, raw in members.items():
                archive.writestr(name, raw)
        manifest["bytes"] = archive_path.stat().st_size
        manifest["sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify(tmp_path)
