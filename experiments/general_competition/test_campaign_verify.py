"""Adversarial replay checks using tiny synthetic campaigns, never benchmarks."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from . import campaign_verify as verifier
from .campaign import digest, run_campaign, score_campaign
from .campaign_verify import TrustedProofChecker, VerificationError, verify_campaign


pytestmark = pytest.mark.skipif(os.name not in ("posix", "nt"), reason="campaign process supervision unavailable")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text())


def seal(root: Path) -> None:
    write_json(root / "evidence-sha256.json", {
        path.relative_to(root).as_posix(): digest(path)
        for path in root.rglob("*") if path.is_file() and path.name != "evidence-sha256.json"})


def cell_records(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "cells.jsonl").read_text().splitlines()]


def write_cells(root: Path, cells: list[dict]) -> None:
    (root / "cells.jsonl").write_text("".join(json.dumps(cell, sort_keys=True) + "\n" for cell in cells))
    for cell in cells:
        write_json(root / cell["directory"] / "cell.json", cell)


def fake_solver(ident: str, code: str, **extra) -> dict:
    return {"id": ident, "track": "upstream-baseline", "source_commit": "a" * 40,
            "upstream_engines": ["synthetic-test-only"],
            "command": [sys.executable, "-c", code, "{cnf}", "{proof}"], **extra}


def make_campaign(base: Path, formulas: list[tuple[str, str, str]], solvers: list[dict], *, compressed: bool = False) -> Path:
    base.mkdir()
    instances = []
    for index, (ident, cnf, expected) in enumerate(formulas):
        source = base / (ident + ".cnf")
        source.write_text(cnf, encoding="ascii", newline="\n")
        entry = {"id": ident, "family": "family-" + str(index), "path": str(source), "sha256": digest(source),
                 "expected": expected, "source_url": "https://example.invalid/synthetic-test-only"}
        if ident == "missing":
            source.unlink()
        instances.append(entry)
    inventory, field = base / "inventory.json", base / "solvers.json"
    write_json(inventory, {"schema": "spectra.general_inventory.v1", "instances": instances})
    write_json(field, {"schema": "spectra.general_solver_field.v1", "solvers": solvers})
    output = base / "evidence"
    run_campaign(argparse.Namespace(inventory=inventory, solvers=field, out=output,
                 checker=base / "absent-checker", timeout=5.0, checker_timeout=5.0,
                 memory_mib=256, output_mib=16, dimacs_mib=16, compress_evidence=compressed))
    return output


@pytest.fixture(scope="module")
def plain_base(tmp_path_factory):
    return make_campaign(tmp_path_factory.mktemp("replay-fixtures") / "plain", [
        ("positive", "p cnf 1 1\n1 0\n", "SAT"),
        ("missing", "p cnf 1 1\n1 0\n", "SAT"),
    ], [fake_solver("valid", "print('s SATISFIABLE'); print('v 1 0'); raise SystemExit(10)"),
        fake_solver("wrong", "print('s SATISFIABLE'); print('v -1 0'); raise SystemExit(10)"),
        fake_solver("unknown", "print('s UNKNOWN')")])


@pytest.fixture
def archive(tmp_path, plain_base):
    return Path(shutil.copytree(plain_base, tmp_path / "archive"))


@pytest.fixture(scope="module")
def compressed_base(tmp_path_factory):
    return make_campaign(tmp_path_factory.mktemp("replay-compressed") / "compressed",
        [("large", "c" + " padding" * 10000 + "\np cnf 20000 1\n1 0\n", "SAT")],
        [fake_solver("valid", "print('c' + ' log' * 20000); print('s SATISFIABLE'); print('v 1 0'); raise SystemExit(10)")],
        compressed=True)


@pytest.fixture(scope="module")
def proof_base(tmp_path_factory):
    base = tmp_path_factory.mktemp("replay-proof")
    marker = base / "archived-command-must-not-run"
    code = "from pathlib import Path; import sys; Path(sys.argv[2]).write_bytes(b'0\\n'); print('s UNSATISFIABLE'); raise SystemExit(20)"
    checker_code = f"from pathlib import Path; Path({str(marker)!r}).write_text('ran'); print('s VERIFIED')"
    output = make_campaign(base / "proof", [("empty", "p cnf 0 1\n0\n", "UNSAT")],
                           [fake_solver("proof", code, proof_checker={"command": [sys.executable, "-c", checker_code, "{cnf}", "{proof}"],
                                                                     "success_line": "s VERIFIED"})])
    marker.unlink()
    return output, marker


def test_plain_archive_is_portable_read_only_and_never_executes_commands(archive, monkeypatch):
    before = {path.relative_to(archive).as_posix(): digest(path) for path in archive.rglob("*") if path.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError("default replay must not launch processes")
    monkeypatch.setattr(verifier, "run_process", forbidden)
    result = verify_campaign(archive, expected_manifest_sha256=digest(archive / "evidence-sha256.json"))
    assert result["status"] == "CONSISTENT_EXPOSED_ARCHIVE"
    assert result["externally_pinned"] and result["cells_verified"] == 6
    assert result["input_checks"]["reparsed_retained_inputs"] == 1
    assert result["fresh_answer_checks"]["verified_models_not_rechecked"] == 1
    assert not result["hidden_evidence_verified"] and not result["independent_replication_verified"]
    assert not result["source_to_binary_rebuild_link_verified"]
    assert before == {path.relative_to(archive).as_posix(): digest(path) for path in archive.rglob("*") if path.is_file()}
    # The retained invalid-model solver remains disqualified in every family.
    summary = read_json(archive / "summary.json")
    assert all(next(row for row in table["standings"] if row["solver_id"] == "wrong")["disqualified"]
               for table in [summary, *summary["families"].values()])


@pytest.mark.parametrize("change", ["mutation", "missing", "extra"])
def test_exact_file_inventory_rejects_mutated_missing_or_extra_bytes(archive, change):
    path = archive / "instances/00000/solver-000/solve/stdout.txt"
    if change == "mutation":
        path.write_text("s SATISFIABLE\nv -1 0\n")
    elif change == "missing":
        path.unlink()
    else:
        (archive / "unexpected.txt").write_text("extra")
    with pytest.raises(VerificationError, match="file (hash|inventory) differs"):
        verify_campaign(archive)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "instances/../alias", "a//b", "a\\b", "a/./b", "C:/drive"])
def test_hash_manifest_rejects_path_traversal_and_noncanonical_names(archive, name):
    hashes = read_json(archive / "evidence-sha256.json")
    hashes[name] = "a" * 64
    write_json(archive / "evidence-sha256.json", hashes)
    with pytest.raises(VerificationError, match="evidence path"):
        verify_campaign(archive)


@pytest.mark.parametrize("kind", ["file", "directory", "root"])
def test_symlinks_are_never_followed(archive, tmp_path, kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("not evidence")
    try:
        if kind == "root":
            link = tmp_path / "linked-archive"
            link.symlink_to(archive, target_is_directory=True)
            target = link
        else:
            (archive / "link").symlink_to(outside if kind == "directory" else outside / "secret", target_is_directory=kind == "directory")
            target = archive
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise
    with pytest.raises(VerificationError, match="symlink"):
        verify_campaign(target)


@pytest.mark.parametrize("payload", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e10000}'])
def test_strict_metadata_rejects_duplicate_keys_and_nonfinite_numbers(archive, payload):
    (archive / "campaign.json").write_text(payload)
    seal(archive)
    with pytest.raises(VerificationError, match="duplicate JSON|non-finite"):
        verify_campaign(archive)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "unknown_solver"])
def test_resealed_archive_cannot_change_declared_cell_rectangle(archive, change):
    cells = cell_records(archive)
    if change == "missing":
        cells.pop()
    elif change == "extra":
        cells.append(cells[0])
    elif change == "duplicate":
        cells[-1] = cells[0]
    else:
        cells[-1]["solver_id"] = "undeclared"
    write_cells(archive, cells)
    seal(archive)
    with pytest.raises(VerificationError, match="(declared|extra|duplicate) cells|rectangle"):
        verify_campaign(archive)


@pytest.mark.parametrize("change", ["rank", "score", "csv", "family_disqualification"])
def test_scores_and_disqualifications_are_recomputed_after_resealing(archive, change):
    summary = read_json(archive / "summary.json")
    if change == "rank":
        summary["standings"][0]["rank"] = 99
    elif change == "score":
        summary["standings"][0]["overall_par2"] = "0"
    elif change == "family_disqualification":
        for row in summary["families"]["family-1"]["standings"]:
            if row["solver_id"] == "wrong":
                row["disqualified"] = False
    else:
        (archive / "scores.csv").write_bytes((archive / "scores.csv").read_bytes().replace(b"sat-verified", b"unknown", 1))
    write_json(archive / "summary.json", summary)
    seal(archive)
    with pytest.raises(VerificationError, match="scores.csv|recomputed campaign summary"):
        verify_campaign(archive)


@pytest.mark.parametrize("change", ["runtime", "answer", "checker_timeout", "admission_timeout", "model_missing", "output_limit"])
def test_resealed_process_and_outcome_contradictions_are_rejected(archive, change):
    cells = cell_records(archive)
    cell = next(item for item in cells if item["status"] == "SAT_VERIFIED")
    directory = archive / cell["directory"]
    if change == "runtime":
        cell["elapsed_seconds"] = 0.000001
    elif change == "answer":
        cell["reported_answer"] = "UNSAT"
    elif change == "checker_timeout":
        process = read_json(directory / "check/process.json")
        process["elapsed_seconds"] = 6.0
        cell["checking_seconds"] = 6.0
        write_json(directory / "check/process.json", process)
    elif change == "output_limit":
        path = directory / "solve/process.json"
        process = read_json(path)
        process["output_limit_exceeded"] = True
        write_json(path, process)
    elif change == "admission_timeout":
        path = archive / "instances/00000/admission-process/process.json"
        process = read_json(path)
        process["elapsed_seconds"] = 6.0
        write_json(path, process)
        admissions = read_json(archive / "admissions.json")
        admissions["admissions"][0]["process"] = process
        write_json(archive / "admissions.json", admissions)
    else:
        (directory / "complete.model").unlink()
    write_cells(archive, cells)
    seal(archive)
    with pytest.raises(VerificationError, match="differs|lacks complete model"):
        verify_campaign(archive)


def test_fresh_checks_reject_temporary_storage_inside_evidence(archive, monkeypatch):
    monkeypatch.setattr(verifier.tempfile, "gettempdir", lambda: str(archive))
    before = sorted(path.relative_to(archive).as_posix() for path in archive.rglob("*"))
    with pytest.raises(VerificationError, match="temporary directory is inside evidence"):
        verify_campaign(archive, recheck_models=True)
    assert sorted(path.relative_to(archive).as_posix() for path in archive.rglob("*")) == before


def test_copied_manifest_hash_binding_survives_resealing(archive):
    inventory = read_json(archive / "inventory.json")
    inventory["instances"][0]["family"] = "new-family"
    write_json(archive / "inventory.json", inventory)
    seal(archive)
    with pytest.raises(VerificationError, match="copied inventory.json binding"):
        verify_campaign(archive)


def test_resealed_solver_manifest_must_match_the_recorded_invocation(archive):
    field = read_json(archive / "solvers.json")
    field["solvers"][0]["command"][2] = "print('different solver')"
    write_json(archive / "solvers.json", field)
    config = read_json(archive / "campaign.json")
    config.update(solver_metadata=field["solvers"], solver_field_sha256=digest(archive / "solvers.json"))
    write_json(archive / "campaign.json", config)
    seal(archive)
    with pytest.raises(VerificationError, match="recorded solver argv"):
        verify_campaign(archive)


def test_recorded_checker_must_have_checked_the_retained_original_path(archive):
    path = archive / "instances/00000/solver-000/check/process.json"
    process = read_json(path)
    index = process["command"].index("--cnf") + 1
    process["command"][index] = "/some/other/formula.cnf"
    write_json(path, process)
    seal(archive)
    with pytest.raises(VerificationError, match="recorded checker argv"):
        verify_campaign(archive)


@pytest.mark.parametrize("directory", ["instances/00000/solver-999", "instances/99999"])
def test_even_empty_extra_grid_directories_are_rejected(archive, directory):
    (archive / directory).mkdir()
    with pytest.raises(VerificationError, match="(cell|instance) directory rectangle"):
        verify_campaign(archive)


def test_external_hash_pin_detects_fully_resealed_archive(archive):
    pin = digest(archive / "evidence-sha256.json")
    config = read_json(archive / "campaign.json")
    config["cpu_model"] = "different host description"
    write_json(archive / "campaign.json", config)
    seal(archive)
    with pytest.raises(VerificationError, match="external evidence manifest hash"):
        verify_campaign(archive, expected_manifest_sha256=pin)


def test_original_cnf_and_admission_counts_are_reparsed(archive):
    path = archive / "instances/00000/admission.json"
    receipt = read_json(path)
    receipt["declared_clauses"] = 5
    write_json(path, receipt)
    seal(archive)
    with pytest.raises(VerificationError, match="admission receipt .* declared_clauses"):
        verify_campaign(archive)


def test_recorded_original_input_mutation_never_receives_a_ranking(archive):
    path = archive / "artifact-integrity.json"
    record = read_json(path)
    record["original_input_mutations"] = [{"instance_id": "positive", "solver_id": "wrong"}]
    write_json(path, record)
    seal(archive)
    with pytest.raises(VerificationError, match="original input mutations prevent"):
        verify_campaign(archive)


def replace_policy_fixture(archive: Path, policy: str | None, expected: str = "UNKNOWN", *, legacy_conflict: bool = False) -> dict:
    """Make a small archived-policy fixture without rerunning timed processes."""
    inventory = read_json(archive / "inventory.json")
    inventory["instances"][0]["expected"] = expected
    write_json(archive / "inventory.json", inventory)
    config = read_json(archive / "campaign.json")
    config["inventory_sha256"] = digest(archive / "inventory.json")
    if policy is None:
        config.pop("scoring_policy", None)
    else:
        config["scoring_policy"] = policy
    write_json(archive / "campaign.json", config)
    cells = cell_records(archive)
    for cell in cells:
        if cell["instance_id"] == "positive":
            cell["expected"] = expected
            if legacy_conflict and cell["status"] == "SAT_VERIFIED":
                cell["status"] = "INVALID_VERDICT_CONFLICT"
    write_cells(archive, cells)
    summary = score_campaign(cells, inventory, read_json(archive / "solvers.json"), config["timeout_seconds"],
                             scoring_policy=policy or "legacy-v1")
    summary.update(evidence_class="EXPOSED_DEVELOPMENT", declared_field_complete=True,
                   complete_roadmap_field_verified=False, exact_semantic_duplicate_pairs=[], claim=verifier.CLAIM)
    write_json(archive / "summary.json", summary)
    (archive / "scores.csv").write_bytes(verifier._csv(cells, config["timeout_seconds"]))
    seal(archive)
    return summary


@pytest.mark.parametrize("policy", [None, "legacy-v1", "declared-inventory-v2"])
def test_scoring_policy_dispatch_preserves_legacy_inference_and_declared_strata(archive, policy):
    summary = replace_policy_fixture(archive, policy)
    result = verify_campaign(archive)
    assert result["scoring_policy"] == (policy or "legacy-v1")
    assert summary["unknown_instances"] == (1 if policy == "declared-inventory-v2" else 0)
    valid = next(row for row in summary["standings"] if row["solver_id"] == "valid")
    assert valid["sat_solved"] == (0 if policy == "declared-inventory-v2" else 1)
    if policy == "declared-inventory-v2":
        assert summary["verified_answer_counts"]["valid"]["metadata_unknown"] == 1


def test_verifier_does_not_silently_reinterpret_legacy_policy(archive):
    replace_policy_fixture(archive, None)
    config = read_json(archive / "campaign.json")
    config["scoring_policy"] = "declared-inventory-v2"
    write_json(archive / "campaign.json", config)
    seal(archive)
    with pytest.raises(VerificationError, match="recomputed campaign summary"):
        verify_campaign(archive)


def test_legacy_metadata_conflict_is_replayable_but_v2_requires_adjudication(archive):
    summary = replace_policy_fixture(archive, None, expected="UNSAT", legacy_conflict=True)
    assert verify_campaign(archive)["scoring_policy"] == "legacy-v1"
    assert next(row for row in summary["standings"] if row["solver_id"] == "valid")["disqualified"]
    config = read_json(archive / "campaign.json")
    config["scoring_policy"] = "declared-inventory-v2"
    write_json(archive / "campaign.json", config)
    seal(archive)
    with pytest.raises(VerificationError, match="metadata verdict conflict requires adjudication"):
        verify_campaign(archive)


def test_fresh_models_check_the_original_formula_and_complete_witness(archive):
    pin = digest(archive / "evidence-sha256.json")
    result = verify_campaign(archive, recheck_models=True, checker_timeout=5, checker_memory_mib=256, checker_output_mib=16)
    assert result["fresh_answer_checks"]["models_rechecked"] == 1
    assert digest(archive / "evidence-sha256.json") == pin
    cells = cell_records(archive)
    cell = next(item for item in cells if item["status"] == "SAT_VERIFIED")
    directory = archive / cell["directory"]
    (directory / "solve/stdout.txt").write_text("s SATISFIABLE\nv -1 0\n")
    record = read_json(directory / "solve/process.json")
    record["stdout_sha256"] = digest(directory / "solve/stdout.txt")
    write_json(directory / "solve/process.json", record)
    seal(archive)
    # A self-consistent forged recorded receipt is not an independent answer check.
    assert verify_campaign(archive)["fresh_answer_checks"]["models_rechecked"] == 0
    with pytest.raises(VerificationError, match="fresh model check failed"):
        verify_campaign(archive, recheck_models=True, checker_timeout=5, checker_memory_mib=256, checker_output_mib=16)


def test_archived_checker_commands_are_not_trusted(proof_base):
    archive, marker = proof_base
    result = verify_campaign(archive)
    assert result["fresh_answer_checks"]["verified_proofs_not_rechecked"] == 1
    assert not marker.exists()
    with pytest.raises(VerificationError, match="explicitly trusted proof checker"):
        verify_campaign(archive, require_all_proofs=True)
    assert not marker.exists()


@pytest.mark.parametrize("policy,line,status", [
    ("legacy-v1", "s MEMORY ALLOCATION ERROR", "INVALID_PROOF"),
    ("declared-inventory-v2", "s MEMORY ALLOCATION ERROR", "CHECKER_RESOURCE_FAILURE"),
    ("declared-inventory-v2", "s NOT VERIFIED", "INVALID_PROOF"),
])
def test_checker_resource_reclassification_is_explicitly_policy_scoped(tmp_path, proof_base, policy, line, status):
    archive = Path(shutil.copytree(proof_base[0], tmp_path / "proof-archive"))
    config = read_json(archive / "campaign.json")
    config["scoring_policy"] = policy
    write_json(archive / "campaign.json", config)
    cells = cell_records(archive)
    cell = cells[0]
    check = archive / cell["directory"] / "check"
    (check / "stdout.txt").write_text(line + "\n")
    process = read_json(check / "process.json")
    process["stdout_sha256"] = digest(check / "stdout.txt")
    write_json(check / "process.json", process)
    cell["status"] = status
    if status == "CHECKER_RESOURCE_FAILURE":
        cell["checker_resource_diagnostic"] = line
    write_cells(archive, cells)
    summary = score_campaign(cells, read_json(archive / "inventory.json"), read_json(archive / "solvers.json"),
                             config["timeout_seconds"], scoring_policy=policy)
    summary.update(evidence_class="EXPOSED_DEVELOPMENT", declared_field_complete=True,
                   complete_roadmap_field_verified=False, exact_semantic_duplicate_pairs=[], claim=verifier.CLAIM)
    write_json(archive / "summary.json", summary)
    (archive / "scores.csv").write_bytes(verifier._csv(cells, config["timeout_seconds"]))
    seal(archive)
    assert verify_campaign(archive)["scoring_policy"] == policy
    assert summary["standings"][0]["disqualified"] == (status == "INVALID_PROOF")


@pytest.mark.parametrize("code,passes", [("print('s VERIFIED')", True), ("print('s VERIFIED'); raise SystemExit(1)", False),
                                          ("print('not VERIFIED')", False),
                                          ("Path(sys.argv[1]).write_text('mutated CNF'); print('s VERIFIED')", False),
                                          ("Path(sys.argv[2]).write_text('mutated proof'); print('s VERIFIED')", False)])
def test_fresh_proofs_require_explicit_checker_exit_and_success_token(tmp_path, proof_base, code, passes):
    archive, marker = proof_base
    # The caller explicitly supplies the Python executable and checker program.
    code = ("from pathlib import Path; import sys; "
            "assert Path(sys.argv[1]).read_bytes() == b'p cnf 0 1\\n0\\n'; "
            "assert Path(sys.argv[2]).read_bytes() == b'0\\n'; " + code)
    checker = Path(sys.executable)
    pin = digest(archive / "evidence-sha256.json")
    kwargs = {"proof_checkers": {"proof": TrustedProofChecker(
                  checker, arguments=("-c", code, "{cnf}", "{proof}"), expected_sha256=digest(checker))},
              "require_all_proofs": True, "checker_timeout": 5, "checker_memory_mib": 256, "checker_output_mib": 16}
    if passes:
        result = verify_campaign(archive, **kwargs)
        assert result["fresh_answer_checks"]["proofs_rechecked"] == 1
    else:
        with pytest.raises(VerificationError, match="fresh proof check failed|fresh checker mutated its input"):
            verify_campaign(archive, **kwargs)
    assert digest(archive / "evidence-sha256.json") == pin and not marker.exists()


def test_compressed_archive_decodes_identities_and_rechecks_complete_model(compressed_base):
    pin = digest(compressed_base / "evidence-sha256.json")
    result = verify_campaign(compressed_base, recheck_models=True, checker_timeout=5, checker_memory_mib=256, checker_output_mib=16)
    assert result["compressed_files_decoded_and_verified"] == 3
    assert result["fresh_answer_checks"]["models_rechecked"] == 1
    assert digest(compressed_base / "evidence-sha256.json") == pin


@pytest.mark.parametrize("change", ["decoded_hash", "decoded_size", "duplicate", "traversal", "journal", "trailing_xz", "invalid_xz"])
def test_resealed_compression_receipts_cannot_hide_corruption(tmp_path, compressed_base, change):
    archive = Path(shutil.copytree(compressed_base, tmp_path / "compressed"))
    receipt_path = archive / "instances/00000/storage.json"
    receipt = read_json(receipt_path)
    entry = receipt["entries"][0]
    if change == "decoded_hash":
        entry["uncompressed_sha256"] = "0" * 64
    elif change == "decoded_size":
        entry["uncompressed_bytes"] -= 1
    elif change == "duplicate":
        receipt["entries"].append(entry.copy())
    elif change == "traversal":
        entry["logical_path"] = "../input.cnf"
    elif change == "trailing_xz":
        path = archive / entry["stored_path"]
        with path.open("ab") as stream:
            stream.write(b"trailing garbage")
        entry["stored_bytes"] = path.stat().st_size
        entry["stored_sha256"] = digest(path)
    elif change == "invalid_xz":
        path = archive / entry["stored_path"]
        path.write_bytes(b"invalid compressed payload")
        entry["stored_bytes"] = path.stat().st_size
        entry["stored_sha256"] = digest(path)
    write_json(receipt_path, receipt)
    if change != "journal":
        (receipt_path.parent / "storage.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in receipt["entries"]))
    else:
        (receipt_path.parent / "storage.jsonl").write_text("")
    seal(archive)
    with pytest.raises(VerificationError, match="(decoded|compressed|ambiguous|evidence path|journal|trailing)"):
        verify_campaign(archive)


def test_explicit_original_source_mapping_checks_hash_and_decompressed_identity(tmp_path, archive):
    raw = archive / "instances/00000/input.cnf"
    source = tmp_path / "source.cnf.gz"
    source.write_bytes(gzip.compress(raw.read_bytes(), mtime=0))
    inventory = read_json(archive / "inventory.json")
    inventory["instances"][0].update(path="not-accessed/source.cnf.gz", sha256=digest(source))
    write_json(archive / "inventory.json", inventory)
    config = read_json(archive / "campaign.json")
    config["inventory_sha256"] = digest(archive / "inventory.json")
    write_json(archive / "campaign.json", config)
    receipt_path = raw.with_name("admission.json")
    receipt = read_json(receipt_path)
    receipt.update(original_source="not-accessed/source.cnf.gz", original_source_sha256=digest(source), original_source_bytes=source.stat().st_size)
    write_json(receipt_path, receipt)
    cells = cell_records(archive)
    for cell in cells:
        if cell["instance_id"] == "positive":
            for key in ("input_integrity_after_solve", "input_integrity_after_check"):
                cell[key]["source_sha256"] = digest(source)
    write_cells(archive, cells)
    seal(archive)
    assert verify_campaign(archive)["input_checks"]["compressed_sources_not_available"] == 1
    result = verify_campaign(archive, original_sources={"positive": source})
    assert result["input_checks"]["explicit_original_sources_rechecked"] == 1
    source.write_bytes(b"changed")
    with pytest.raises(VerificationError, match="explicit original source hash"):
        verify_campaign(archive, original_sources={"positive": source})


def test_cli_emits_rejection_and_nonzero_exit_for_incomplete_campaign(archive, capsys):
    cells = cell_records(archive)
    write_cells(archive, cells[:-1])
    seal(archive)
    assert verifier.main([str(archive)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "REJECTED" and "No ranking" in result["claim"]
