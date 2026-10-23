"""Adversarial contracts for exposed campaign evidence, not performance claims."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import sys

import pytest

from .campaign import (
    _status, checker_resource_diagnostic, check_model, digest, load_inventory, load_solvers,
    run_campaign, run_process, score_campaign, stage_input,
)


def store(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def instance(path, ident="i", family="logic", expected="SAT"):
    return {"id": ident, "family": family, "path": str(path), "sha256": digest(path),
            "expected": expected, "source_url": "https://example.invalid/test-only"}


def solver(ident="baseline", code="print('s SATISFIABLE'); print('v 1 0'); raise SystemExit(10)", **updates):
    item = {"id": ident, "track": "upstream-baseline", "source_commit": "a" * 40,
            "upstream_engines": ["test-only-fake-solver"],
            "command": [sys.executable, "-c", code, "{cnf}", "{proof}"]}
    item.update(updates)
    return item


def manifests(tmp_path, instances, solvers):
    return (store(tmp_path / "inventory.json", {"schema": "spectra.general_inventory.v1", "instances": instances}),
            store(tmp_path / "solvers.json", {"schema": "spectra.general_solver_field.v1", "solvers": solvers}))


def arguments(tmp_path, inventory, field, checker=None):
    return argparse.Namespace(inventory=inventory, solvers=field, out=tmp_path / "evidence",
                              checker=checker or tmp_path / "missing-checker", timeout=2.0,
                              checker_timeout=2.0, memory_mib=512, output_mib=16, dimacs_mib=16)


def cell(solver_id="a", instance_id="i", family="logic", expected="SAT", status="SAT_VERIFIED", elapsed=1.0):
    return {"solver_id": solver_id, "instance_id": instance_id, "family": family,
            "expected": expected, "status": status, "elapsed_seconds": elapsed}


@pytest.mark.parametrize("extension,compress", [("cnf", lambda x: x), ("cnf.gz", gzip.compress), ("cnf.xz", lzma.compress)])
def test_staging_preserves_exact_decompressed_bytes_and_identity(tmp_path, extension, compress):
    original = b"c original whitespace\np cnf 2 2\n 1  -2 0\n2 0\n"
    source = tmp_path / ("source." + extension)
    source.write_bytes(compress(original))
    target = tmp_path / "input.cnf"
    stage_input(source, digest(source), target, 1024)
    assert target.read_bytes() == original
    receipt = json.loads((tmp_path / "admission.json").read_text())
    assert receipt["original_source_sha256"] == digest(source)
    assert receipt["dimacs_sha256"] == hashlib.sha256(original).hexdigest()


def test_staging_rejects_mutated_source_and_decompression_overflow(tmp_path):
    source = tmp_path / "source.gz"
    source.write_bytes(gzip.compress(b"p cnf 1 1\n1 0\n"))
    with pytest.raises(ValueError, match="SHA-256"):
        stage_input(source, "0" * 64, tmp_path / "out", 100)
    with pytest.raises(MemoryError, match="byte limit"):
        stage_input(source, digest(source), tmp_path / "out", 4)


@pytest.mark.parametrize("witness", ["s SATISFIABLE\nv -1 0\n", "v 1 -1 0\n", "v 2 0\n", "v 1\n", "v 1 0 1\n", "v bogus 0\n"])
def test_model_checker_rejects_wrong_contradictory_out_of_range_and_malformed(tmp_path, witness):
    cnf, out = tmp_path / "input.cnf", tmp_path / "stdout.txt"
    cnf.write_text("p cnf 1 1\n1 0\n")
    out.write_text(witness)
    with pytest.raises(ValueError):
        check_model(cnf, out, tmp_path / "complete.model")
    assert not (tmp_path / "complete.model").exists()


def test_model_checker_independently_completes_absent_variables_and_retains_full_model(tmp_path):
    cnf, out = tmp_path / "input.cnf", tmp_path / "stdout.txt"
    cnf.write_text("p cnf 3 2\n1 2 0\n-3 0\n")
    out.write_text("s SATISFIABLE\nv 1 0\n")
    check_model(cnf, out, tmp_path / "complete.model")
    assert "v 1 -2 -3\n" in (tmp_path / "complete.model").read_text()


def test_empty_clause_never_accepts_model(tmp_path):
    cnf, out = tmp_path / "input.cnf", tmp_path / "stdout.txt"
    cnf.write_text("p cnf 0 1\n0\n")
    out.write_text("v 0\n")
    with pytest.raises(ValueError, match="fails original clause"):
        check_model(cnf, out, tmp_path / "complete.model")


def test_manifest_requires_pinned_sources_and_disclosed_track(tmp_path):
    field = tmp_path / "field.json"
    good = solver()
    good.pop("source_commit")
    store(field, {"schema": "spectra.general_solver_field.v1", "solvers": [good]})
    with pytest.raises(ValueError, match="source_commit or"):
        load_solvers(field)
    good["source_archive_sha256"] = "b" * 64
    store(field, {"schema": "spectra.general_solver_field.v1", "solvers": [good]})
    assert load_solvers(field)["solvers"][0]["source_archive_sha256"] == "b" * 64
    good["track"] = "standalone-native"
    store(field, {"schema": "spectra.general_solver_field.v1", "solvers": [good]})
    with pytest.raises(ValueError, match="must not include upstream"):
        load_solvers(field)


def test_manifest_duplicate_ids_cannot_silently_replace_cells(tmp_path):
    cnf = tmp_path / "input.cnf"
    cnf.write_text("p cnf 0 0\n")
    path, _ = manifests(tmp_path, [instance(cnf), instance(cnf)], [solver()])
    with pytest.raises(ValueError, match="duplicate"):
        load_inventory(path)


def test_score_rejects_missing_extra_and_duplicate_declared_cells():
    inventory = {"instances": [{"id": "i", "family": "logic"}]}
    field = {"solvers": [{"id": "a"}, {"id": "b"}]}
    for cells in ([cell()], [cell(), cell()], [cell(), cell("b"), cell("c")]):
        with pytest.raises(ValueError, match="missing declared cells"):
            score_campaign(cells, inventory, field, 10)


def test_score_penalizes_invalid_answers_and_retains_subtrack_and_family_losses():
    inventory = {"instances": [{"id": "sat", "family": "easy"}, {"id": "unsat", "family": "hard"}]}
    field = {"solvers": [{"id": "a"}, {"id": "b"}]}
    cells = [cell("a", "sat", "easy", "SAT", "SAT_VERIFIED", 1),
             cell("a", "unsat", "hard", "UNSAT", "INVALID_PROOF", 0.1),
             cell("b", "sat", "easy", "SAT", "TIMEOUT", 10),
             cell("b", "unsat", "hard", "UNSAT", "UNSAT_VERIFIED", 2)]
    result = score_campaign(cells, inventory, field, 10)
    a = next(row for row in result["standings"] if row["solver_id"] == "a")
    assert a["disqualified"] and a["solved"] == 1 and a["unsat_par2"] == "20"
    assert result["standings"][0]["solver_id"] == "b"
    assert set(result["families"]) == {"easy", "hard"}
    assert result["resource_summary"]["a"]["statuses"]["INVALID_PROOF"] == 1


@pytest.mark.skipif(os.name != "posix", reason="POSIX process limits")
def test_timeout_and_spawn_failure_keep_raw_streams_and_process_receipts(tmp_path):
    result = run_process([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(5)"],
                         tmp_path / "timeout", timeout=0.2, memory_mib=128, output_mib=1)
    assert result["timed_out"] and result["returncode"] < 0
    assert "started" in (tmp_path / "timeout" / "stdout.txt").read_text()
    assert (tmp_path / "timeout" / "process.json").is_file()
    assert result["peak_rss_kib"] > 0
    result = run_process([str(tmp_path / "no-executable")], tmp_path / "missing", timeout=1, memory_mib=128, output_mib=1)
    assert result["spawn_error"] and (tmp_path / "missing" / "stderr.txt").exists()


def test_answer_cannot_outrun_timeout_or_wrong_exit_code():
    process = {"timed_out": False, "elapsed_seconds": 1.5, "spawn_error": None, "returncode": 10}
    assert _status(process, "SAT", 1) == "TIMEOUT"
    process["elapsed_seconds"] = 0.1
    assert _status(process, "SAT", 1) == "SAT"
    process["returncode"] = 20
    assert _status(process, "SAT", 1) == "INVALID_OUTPUT"


@pytest.mark.skipif(os.name != "posix", reason="POSIX campaign execution")
def test_campaign_retains_checked_sat_invalid_model_unknown_and_admission_failure(tmp_path):
    cnf, missing = tmp_path / "input.cnf", tmp_path / "missing.cnf"
    cnf.write_text("p cnf 1 1\n1 0\n")
    missing_entry = instance(cnf, "missing")
    missing_entry["path"] = str(missing)
    inventory, field = manifests(tmp_path, [instance(cnf), missing_entry], [
        solver("valid"), solver("invalid", "print('s SATISFIABLE'); print('v -1 0'); raise SystemExit(10)"),
        solver("unknown", "print('s UNKNOWN')")])
    result = run_campaign(arguments(tmp_path, inventory, field))
    assert result["instance_count"] == 2 and result["solver_count"] == 3
    assert result["complete_roadmap_field_verified"] is False
    assert result["evidence_class"] == "EXPOSED_DEVELOPMENT"
    cells = [json.loads(line) for line in (tmp_path / "evidence" / "cells.jsonl").read_text().splitlines()]
    assert len(cells) == 6
    assert CounterFor(cells) == {"SAT_VERIFIED": 1, "INVALID_MODEL": 1, "UNKNOWN": 1, "ADMISSION_FAILURE": 3}
    hashes = json.loads((tmp_path / "evidence" / "evidence-sha256.json").read_text())
    for filename, expected in hashes.items():
        assert digest(tmp_path / "evidence" / filename) == expected
    with pytest.raises(FileExistsError):
        run_campaign(arguments(tmp_path, inventory, field))


def CounterFor(cells):
    from collections import Counter
    return dict(Counter(row["status"] for row in cells))


@pytest.mark.skipif(os.name != "posix", reason="POSIX campaign execution")
@pytest.mark.parametrize("kind,expected", [("unsupported", "UNAVAILABLE_CHECKER"),
    ("reject", "INVALID_PROOF"), ("wrong-exit", "INVALID_PROOF"),
    ("misleading-token", "INVALID_PROOF"), ("accepted", "UNSAT_VERIFIED")])
def test_proof_status_requires_supported_checker_exact_token_and_exit_code(tmp_path, kind, expected):
    cnf = tmp_path / "input.cnf"
    cnf.write_text("p cnf 1 2\n1 0\n-1 0\n")
    code = "import pathlib,sys; pathlib.Path(sys.argv[2]).write_text('0\\n'); print('s UNSATISFIABLE'); raise SystemExit(20)"
    checker_code = {"reject": "print('s NOT VERIFIED'); raise SystemExit(1)",
                    "wrong-exit": "print('s VERIFIED'); raise SystemExit(1)",
                    "misleading-token": "print('warning: s VERIFIED is absent')",
                    "accepted": "print('s VERIFIED')"}.get(kind, "print('s VERIFIED')")
    entry = solver("proof", code, proof_format="DSR" if kind == "unsupported" else "DRAT")
    if kind != "unsupported":
        entry["proof_checker"] = {"command": [sys.executable, "-c", checker_code, "{cnf}", "{proof}"], "success_line": "s VERIFIED"}
    inventory, field = manifests(tmp_path, [instance(cnf, expected="UNSAT")], [entry])
    result = run_campaign(arguments(tmp_path, inventory, field))
    assert result["resource_summary"]["proof"]["statuses"] == {expected: 1}
    assert result["standings"][0]["disqualified"] == (expected == "INVALID_PROOF")


@pytest.mark.skipif(os.name != "posix", reason="POSIX campaign execution")
def test_missing_checker_does_not_falsely_disqualify_valid_proof(tmp_path):
    cnf = tmp_path / "input.cnf"
    cnf.write_text("p cnf 1 2\n1 0\n-1 0\n")
    code = "import pathlib,sys; pathlib.Path(sys.argv[2]).write_text('0\\n'); print('s UNSATISFIABLE'); raise SystemExit(20)"
    inventory, field = manifests(tmp_path, [instance(cnf, expected="UNSAT")], [solver("proof", code)])
    result = run_campaign(arguments(tmp_path, inventory, field))
    assert result["resource_summary"]["proof"]["statuses"] == {"CHECKER_RESOURCE_FAILURE": 1}
    assert not result["standings"][0]["disqualified"]


@pytest.mark.skipif(os.name != "posix" or shutil.which("g++") is None, reason="native build")
def test_native_worker_emits_independently_checkable_complete_model_and_budget_unknown(tmp_path):
    from .native_cdcl import build_native_cdcl
    library = build_native_cdcl(tmp_path / "build")
    cnf = tmp_path / "input.cnf"
    cnf.write_text("p cnf 3 2\n1 2 3 0\n-1 2 0\n")
    command = [sys.executable, "-m", "experiments.general_competition.campaign", "native",
               "--library", str(library), "--cnf", "{cnf}", "--proof", "{proof}", "--stats", "{stats}"]
    native = solver("native", command=command, track="standalone-native", upstream_engines=[], artifact_paths=[str(library)])
    inventory, field = manifests(tmp_path, [instance(cnf)], [native])
    result = run_campaign(arguments(tmp_path, inventory, field))
    assert result["resource_summary"]["native"]["statuses"] == {"SAT_VERIFIED": 1}
    stats = json.loads((tmp_path / "evidence" / "instances" / "00000" / "solver-000" / "stats.json").read_text())
    assert stats["independent_checking_included"] is False
    assert stats["proof_generation_included_in_solve"] is True
    assert "decisions" in stats["stats"]
    config = json.loads((tmp_path / "evidence" / "campaign.json").read_text())
    assert config["artifact_sha256"]["native"][str(library)] == digest(library)


def test_score_rejects_cell_family_verdict_and_runtime_mutation():
    inventory = {"instances": [{"id": "i", "family": "logic", "expected": "SAT"}]}
    field = {"solvers": [{"id": "a"}]}
    for mutated in (cell(family="removed-hard-family"), cell(expected="UNSAT"),
                    cell(elapsed=float("nan")), cell(elapsed=11)):
        with pytest.raises(ValueError):
            score_campaign([mutated], inventory, field, 10)


def test_wrong_answer_disqualifies_solver_in_other_families_and_subtracks():
    inventory = {"instances": [{"id": "sat", "family": "easy"}, {"id": "unsat", "family": "hard"}]}
    field = {"solvers": [{"id": "a"}, {"id": "b"}]}
    cells = [cell("a", "sat", "easy", "SAT", "SAT_VERIFIED", 1),
             cell("a", "unsat", "hard", "UNSAT", "INVALID_PROOF", 0.1),
             cell("b", "sat", "easy", "SAT", "SAT_VERIFIED", 2),
             cell("b", "unsat", "hard", "UNSAT", "UNSAT_VERIFIED", 2)]
    result = score_campaign(cells, inventory, field, 10)
    assert result["families"]["easy"]["standings"][0]["solver_id"] == "b"
    assert result["standings"][0]["sat_rank"] == 1
    assert result["standings"][1]["sat_rank"] == 2
    assert result["families"]["easy"]["standings"][1]["disqualified"]


@pytest.mark.skipif(os.name != "posix" or shutil.which("g++") is None, reason="native build")
def test_complete_campaign_native_unsat_passes_real_external_drat_trim(tmp_path):
    checker_path = os.environ.get("DRAT_TRIM")
    if not checker_path:
        pytest.skip("set DRAT_TRIM to the pinned external checker")
    from .native_cdcl import build_native_cdcl
    library = build_native_cdcl(tmp_path / "build")
    cnf = tmp_path / "input.cnf"
    cnf.write_text("p cnf 2 4\n1 2 0\n-1 2 0\n1 -2 0\n-1 -2 0\n")
    command = [sys.executable, "-m", "experiments.general_competition.campaign", "native",
               "--library", str(library), "--cnf", "{cnf}", "--proof", "{proof}", "--stats", "{stats}"]
    native = solver("native", command=command, track="standalone-native", upstream_engines=[])
    inventory, field = manifests(tmp_path, [instance(cnf, expected="UNSAT")], [native])
    result = run_campaign(arguments(tmp_path, inventory, field, Path(checker_path)))
    assert result["resource_summary"]["native"]["statuses"] == {"UNSAT_VERIFIED": 1}
    assert result["standings"][0]["unsat_solved"] == 1
    cells = [json.loads(line) for line in (tmp_path / "evidence" / "cells.jsonl").read_text().splitlines()]
    assert cells[0]["proof_bytes"] > 0 and cells[0]["checking_seconds"] > 0


def test_model_checker_handles_admitted_comment_syntax(tmp_path):
    cnf, out = tmp_path / "input.cnf", tmp_path / "stdout.txt"
    cnf.write_text("comment without a space after c\np cnf 1 1\n1 0\n")
    out.write_text("v 1 0\n")
    check_model(cnf, out, tmp_path / "complete.model")
    assert (tmp_path / "complete.model").is_file()


@pytest.mark.skipif(os.name != "posix", reason="POSIX campaign execution")
@pytest.mark.parametrize("late_phase,expected", [("check", "CHECKER_TIMEOUT"), ("admission-process", "ADMISSION_FAILURE")])
def test_late_successful_checker_or_admission_cannot_escape_deadline(tmp_path, monkeypatch, late_phase, expected):
    from . import campaign
    cnf = tmp_path / "input.cnf"
    cnf.write_text("p cnf 1 1\n1 0\n")
    inventory, field = manifests(tmp_path, [instance(cnf)], [solver()])
    real = campaign.run_process

    def late(command, destination, **limits):
        record = real(command, destination, **limits)
        if destination.name == late_phase:
            assert record["returncode"] == 0
            record["timed_out"] = False
            record["elapsed_seconds"] = limits["timeout"] + 0.01
        return record

    monkeypatch.setattr(campaign, "run_process", late)
    result = run_campaign(arguments(tmp_path, inventory, field))
    assert result["resource_summary"]["baseline"]["statuses"] == {expected: 1}
    assert result["standings"][0]["solved"] == 0


@pytest.mark.skipif(os.name != "posix", reason="POSIX campaign execution")
def test_solver_cannot_change_original_cnf_or_contaminate_next_solver(tmp_path):
    cnf = tmp_path / "input.cnf"
    cnf.write_text("p cnf 1 1\n-1 0\n")
    malicious = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('p cnf 1 1\\n1 0\\n'); print('s SATISFIABLE'); print('v 1 0'); raise SystemExit(10)"
    correct = "print('s SATISFIABLE'); print('v -1 0'); raise SystemExit(10)"
    inventory, field = manifests(tmp_path, [instance(cnf)], [solver("mutator", malicious), solver("honest", correct)])
    result = run_campaign(arguments(tmp_path, inventory, field))
    assert result["resource_summary"]["mutator"]["statuses"] == {"INVALID_INPUT_MUTATION": 1}
    assert result["resource_summary"]["honest"]["statuses"] == {"SAT_VERIFIED": 1}
    assert result["standings"][0]["solver_id"] == "honest"
    root = tmp_path / "evidence" / "instances" / "00000"
    assert (root / "input.cnf").read_text() == cnf.read_text()
    assert (root / "solver-000" / "input.cnf").read_text() == "p cnf 1 1\n1 0\n"
    assert not (root / "solver-001" / "input.cnf").exists()
    assert not (root / "solver-000" / "complete.model").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX campaign execution")
def test_declared_artifact_hash_mismatch_cannot_run_solver(tmp_path):
    cnf, artifact = tmp_path / "input.cnf", tmp_path / "artifact"
    cnf.write_text("p cnf 1 1\n1 0\n")
    artifact.write_text("unexpected build")
    entry = solver(expected_artifact_sha256={str(artifact): "b" * 64})
    inventory, field = manifests(tmp_path, [instance(cnf)], [entry])
    result = run_campaign(arguments(tmp_path, inventory, field))
    assert result["resource_summary"]["baseline"]["statuses"] == {"SOURCE_IDENTITY_FAILURE": 1}
    assert not (tmp_path / "evidence" / "instances" / "00000" / "solver-000" / "solve").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX campaign execution")
def test_artifact_change_during_campaign_invalidates_ranking_and_keeps_raw_cells(tmp_path):
    cnf, artifact = tmp_path / "input.cnf", tmp_path / "artifact"
    cnf.write_text("p cnf 1 1\n1 0\n")
    artifact.write_text("original build")
    code = f"import pathlib; pathlib.Path({str(artifact)!r}).write_text('changed build'); print('s SATISFIABLE'); print('v 1 0'); raise SystemExit(10)"
    entry = solver(code=code, expected_artifact_sha256={str(artifact): digest(artifact)})
    inventory, field = manifests(tmp_path, [instance(cnf)], [entry])
    result = run_campaign(arguments(tmp_path, inventory, field))
    assert result["status"] == "INVALID_CAMPAIGN"
    assert result["cells_retained"] == 1
    assert (tmp_path / "evidence" / "cells.jsonl").is_file()
    assert (tmp_path / "evidence" / "evidence-sha256.json").is_file()


@pytest.mark.skipif(os.name != "posix", reason="POSIX campaign execution")
def test_solver_cells_have_private_working_directories(tmp_path):
    cnf = tmp_path / "input.cnf"
    cnf.write_text("p cnf 1 1\n1 0\n")
    code = "import pathlib; assert not pathlib.Path('solver-cache').exists(); pathlib.Path('solver-cache').write_text('private'); print('s SATISFIABLE'); print('v 1 0'); raise SystemExit(10)"
    inventory, field = manifests(tmp_path, [instance(cnf)], [solver("first", code), solver("second", code)])
    result = run_campaign(arguments(tmp_path, inventory, field))
    assert all(row["solved"] == 1 for row in result["standings"])


def test_score_rejects_relabeling_assisted_entry_as_standalone():
    inventory = {"instances": [{"id": "i", "family": "logic", "expected": "SAT"}]}
    field = {"solvers": [{"id": "a", "track": "assisted-portfolio"}]}
    entry = {**cell(), "track": "standalone-native"}
    with pytest.raises(ValueError, match="track differs"):
        score_campaign([entry], inventory, field, 10)


@pytest.mark.skipif(os.name != "posix", reason="POSIX campaign execution")
def test_corrupting_shared_original_invalidates_whole_ranking(tmp_path):
    cnf = tmp_path / "input.cnf"
    cnf.write_text("p cnf 1 1\n-1 0\n")
    code = "import pathlib,sys; pathlib.Path(sys.argv[1]).parent.parent.joinpath('input.cnf').write_text('p cnf 1 1\\n1 0\\n'); print('s SATISFIABLE'); print('v 1 0'); raise SystemExit(10)"
    inventory, field = manifests(tmp_path, [instance(cnf)], [solver("mutator", code), solver("innocent")])
    result = run_campaign(arguments(tmp_path, inventory, field))
    assert result["status"] == "INVALID_CAMPAIGN"
    assert "original admitted CNF" in result["error"]
    assert result["cells_retained"] == 2


def test_alternative_solver_exit_codes_require_explicit_declaration():
    process = {"timed_out": False, "elapsed_seconds": 0.1, "spawn_error": None, "returncode": 0}
    assert _status(process, "SAT", 1) == "INVALID_OUTPUT"
    assert _status(process, "UNSAT", 1) == "INVALID_OUTPUT"
    declaration = {"SAT": [0, 10], "UNSAT": [0, 20]}
    assert _status(process, "SAT", 1, declaration) == "SAT"
    assert _status(process, "UNSAT", 1, declaration) == "UNSAT"


def test_compression_roundtrip_preserves_logical_identity_and_journal(tmp_path):
    from .campaign import compress_evidence_file
    data = b"1 -2 3 0\n" * 10000
    source = tmp_path / "proof.drat"
    source.write_bytes(data)
    entry = compress_evidence_file(source, tmp_path, tmp_path / "storage.jsonl")
    assert not source.exists()
    stored = tmp_path / entry["stored_path"]
    assert lzma.decompress(stored.read_bytes()) == data
    assert entry["logical_path"] == "proof.drat"
    assert entry["uncompressed_sha256"] == hashlib.sha256(data).hexdigest()
    assert entry["uncompressed_bytes"] == len(data)
    assert entry["stored_sha256"] == digest(stored)
    assert entry["stored_bytes"] == stored.stat().st_size
    assert entry["roundtrip_verified"] is True
    assert json.loads((tmp_path / "storage.jsonl").read_text()) == entry


def test_compression_corruption_keeps_original_and_publishes_no_archive(tmp_path, monkeypatch):
    from . import campaign
    source = tmp_path / "proof.drat"
    source.write_bytes(b"0\n" * 40000)
    original = source.read_bytes()
    monkeypatch.setattr(campaign, "_decoded_identity", lambda _: (len(original), "0" * 64))
    with pytest.raises(ValueError, match="round-trip"):
        campaign.compress_evidence_file(source, tmp_path, tmp_path / "storage.jsonl")
    assert source.read_bytes() == original
    assert not (tmp_path / "proof.drat.xz").exists()
    assert not (tmp_path / "storage.jsonl").exists()
    assert not list(tmp_path.glob(".compress-*"))


def test_compression_never_overwrites_existing_archive_or_follows_symlink(tmp_path):
    from .campaign import compress_evidence_file
    source = tmp_path / "proof.drat"
    source.write_text("original")
    archive = tmp_path / "proof.drat.xz"
    archive.write_text("existing")
    with pytest.raises(FileExistsError):
        compress_evidence_file(source, tmp_path, tmp_path / "storage.jsonl")
    assert source.read_text() == "original" and archive.read_text() == "existing"
    if os.name == "posix":
        link = tmp_path / "alias"
        link.symlink_to(source)
        with pytest.raises(ValueError, match="non-symlink"):
            compress_evidence_file(link, tmp_path, tmp_path / "storage.jsonl")
        assert source.read_text() == "original"


def test_compression_journal_failure_keeps_original_and_verified_archive(tmp_path):
    from .campaign import compress_evidence_file
    source = tmp_path / "proof.drat"
    data = b"0\n" * 40000
    source.write_bytes(data)
    journal = tmp_path / "storage.jsonl"
    journal.mkdir()
    with pytest.raises(OSError):
        compress_evidence_file(source, tmp_path, journal)
    assert source.read_bytes() == data
    assert lzma.decompress((tmp_path / "proof.drat.xz").read_bytes()) == data


@pytest.mark.skipif(os.name != "posix", reason="POSIX campaign execution")
def test_campaign_compression_retains_checked_result_and_hashes_all_stored_files(tmp_path):
    cnf = tmp_path / "input.cnf"
    cnf.write_text("c " + "large retained comment " * 5000 + "\np cnf 1 1\n1 0\n")
    code = "import pathlib,sys; pathlib.Path(sys.argv[2]).write_text('1 -2 0\\n'*10000); pathlib.Path('derived.lsr').write_text('0\\n'*40000); print('c '+ 'large log '*10000); print('s SATISFIABLE'); print('v 1 0'); raise SystemExit(10)"
    inventory, field = manifests(tmp_path, [instance(cnf)], [solver(code=code)])
    args = arguments(tmp_path, inventory, field)
    args.compress_evidence = True
    result = run_campaign(args)
    assert result["resource_summary"]["baseline"]["statuses"] == {"SAT_VERIFIED": 1}
    root = args.out
    storage = []
    for path in root.rglob("storage.json"):
        receipt = json.loads(path.read_text())
        assert receipt["schema"] == "spectra.general_evidence_storage.v1"
        storage.extend(receipt["entries"])
    logical = {entry["logical_path"] for entry in storage}
    assert "instances/00000/input.cnf" in logical
    assert "instances/00000/solver-000/proof.drat" in logical
    assert "instances/00000/solver-000/derived.lsr" in logical
    assert "instances/00000/solver-000/solve/stdout.txt" in logical
    for entry in storage:
        data = lzma.decompress((root / entry["stored_path"]).read_bytes())
        assert len(data) == entry["uncompressed_bytes"]
        assert hashlib.sha256(data).hexdigest() == entry["uncompressed_sha256"]
    hashes = json.loads((root / "evidence-sha256.json").read_text())
    assert set(hashes) == {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()} - {"evidence-sha256.json"}
    assert all(digest(root / path) == expected for path, expected in hashes.items())


@pytest.mark.skipif(os.name != "posix", reason="POSIX campaign execution")
def test_campaign_storage_failure_retains_cell_and_refuses_partial_ranking(tmp_path, monkeypatch):
    from . import campaign
    cnf = tmp_path / "input.cnf"
    cnf.write_text("p cnf 1 1\n1 0\n")
    inventory, field = manifests(tmp_path, [instance(cnf)], [solver()])
    args = arguments(tmp_path, inventory, field)
    args.compress_evidence = True

    def fail(*_args, **_kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(campaign, "compress_evidence_scope", fail)
    result = run_campaign(args)
    assert result["status"] == "ABORTED_STORAGE_FAILURE"
    assert result["cells_retained"] == 1
    assert result["declared_field_complete"] is False
    assert "standings" not in result
    assert (args.out / "cells.jsonl").is_file()
    assert (args.out / "evidence-sha256.json").is_file()


def test_declared_policy_preserves_unknown_strata_and_subtrack_tie_breaks():
    inventory = {"instances": [
        {"id": "sat", "family": "sat-family", "expected": "SAT"},
        {"id": "unknown-sat", "family": "sat-family", "expected": "UNKNOWN"},
        {"id": "unsat", "family": "unsat-family", "expected": "UNSAT"},
        {"id": "unknown-unsat", "family": "unsat-family", "expected": "UNKNOWN"},
    ]}
    field = {"solvers": [{"id": "a"}, {"id": "z"}]}
    cells = []
    for ident in ("a", "z"):
        for row in inventory["instances"]:
            answer = "UNSAT" if "unsat" in row["id"] else "SAT"
            is_unknown = row["expected"] == "UNKNOWN"
            status = "TIMEOUT" if ident == "a" and is_unknown else answer + "_VERIFIED"
            runtime = 5.0 if status == "TIMEOUT" else 0.2 if is_unknown else 2.0
            cells.append(cell(ident, row["id"], row["family"], row["expected"], status, runtime))
    legacy = score_campaign(cells, inventory, field, 5)
    assert legacy == score_campaign(cells, inventory, field, 5, scoring_policy="legacy-v1")
    assert "scoring_policy" not in legacy and "verified_answer_counts" not in legacy
    assert legacy["unknown_instances"] == 0
    declared = score_campaign(cells, inventory, field, 5, scoring_policy="declared-inventory-v2")
    assert (declared["sat_instances"], declared["unsat_instances"], declared["unknown_instances"]) == (1, 1, 2)
    old = {row["solver_id"]: row for row in legacy["standings"]}
    current = {row["solver_id"]: row for row in declared["standings"]}
    for ident, row in current.items():
        assert row["overall_par2"] == old[ident]["overall_par2"] and row["solved"] == old[ident]["solved"]
        assert row["sat_par2"] == row["unsat_par2"] == "2.0"
        assert row["sat_solved"] == row["unsat_solved"] == 1
    # Unknown solves must not break a tie inside either predeclared subtrack.
    assert current["a"]["sat_rank"] == current["a"]["unsat_rank"] == 1
    assert old["z"]["sat_rank"] == old["z"]["unsat_rank"] == 1
    assert declared["verified_answer_counts"]["z"] == {"SAT": 2, "UNSAT": 2, "metadata_unknown": 2}
    assert all(table["unknown_instances"] == 1 for table in declared["families"].values())


@pytest.mark.parametrize("status,receipt", [
    ("SAT_VERIFIED", False), ("SAT_VERIFIED", True), ("INVALID_VERDICT_CONFLICT", False),
])
def test_declared_scoring_refuses_metadata_conflict_instead_of_disqualifying_solver(status, receipt):
    inventory = {"instances": [{"id": "i", "family": "logic", "expected": "UNSAT"}]}
    field = {"solvers": [{"id": "a"}]}
    result = cell(expected="UNSAT", status=status)
    if receipt:
        result["metadata_conflict"] = {"expected": "UNSAT", "verified_answer": "SAT", "disposition": "NEEDS_ADJUDICATION"}
    with pytest.raises(ValueError, match="metadata verdict conflict requires adjudication"):
        score_campaign([result], inventory, field, 5, scoring_policy="declared-inventory-v2")
    if status == "INVALID_VERDICT_CONFLICT":
        legacy = score_campaign([result], inventory, field, 5)
        assert legacy["standings"][0]["disqualified"]


def test_campaign_scoring_policy_cannot_be_silently_misspelled():
    with pytest.raises(ValueError, match="scoring_policy"):
        score_campaign([], {"instances": []}, {"solvers": []}, 5, scoring_policy="declared-v2")


@pytest.mark.skipif(os.name != "posix", reason="POSIX campaign execution")
@pytest.mark.parametrize("answer", ["SAT", "UNSAT"])
def test_new_campaign_retains_verified_metadata_conflict_and_refuses_ranking(tmp_path, answer):
    cnf = tmp_path / "input.cnf"
    cnf.write_text("p cnf 1 1\n1 0\n" if answer == "SAT" else "p cnf 1 2\n1 0\n-1 0\n")
    entry = solver()
    if answer == "UNSAT":
        entry = solver(code="import pathlib,sys; pathlib.Path(sys.argv[2]).write_text('0\\n'); print('s UNSATISFIABLE'); raise SystemExit(20)")
        entry["proof_checker"] = {"command": [sys.executable, "-c", "print('s VERIFIED')", "{cnf}", "{proof}"],
                                  "success_line": "s VERIFIED"}
    expected = "UNSAT" if answer == "SAT" else "SAT"
    inventory, field = manifests(tmp_path, [instance(cnf, expected=expected)], [entry])
    args = arguments(tmp_path, inventory, field)
    summary = run_campaign(args)
    assert summary["status"] == "INVALID_CAMPAIGN" and "standings" not in summary
    assert "requires adjudication" in summary["error"]
    retained = json.loads((args.out / "cells.jsonl").read_text())
    assert retained["status"] == answer + "_VERIFIED"
    assert retained["metadata_conflict"] == {"expected": expected, "verified_answer": answer, "disposition": "NEEDS_ADJUDICATION"}
    assert json.loads((args.out / "campaign.json").read_text())["scoring_policy"] == "declared-inventory-v2"
    assert (args.out / "evidence-sha256.json").is_file()


@pytest.mark.parametrize("proof_format,code,stdout,stderr,expected", [
    ("DRAT", 0, "c MEMOUT: reallocation of proof list failed", "", "CHECKER_RESOURCE_FAILURE"),
    ("DRUP", 1, "s MEMORY ALLOCATION ERROR", "", "CHECKER_RESOURCE_FAILURE"),
    ("DSR", 255, "", "Ran out of memory on xcalloc()", "CHECKER_RESOURCE_FAILURE"),
    ("DSR", 1, "", "Ran out of memory on xcalloc()", "INVALID_PROOF"),
    ("DSR", 255, "", "c Error: Unable to open the file missing-proof.dsr", "INVALID_PROOF"),
    ("DRAT", 1, "s NOT VERIFIED", "", "INVALID_PROOF"),
    ("DRAT", 0, "warning: c MEMOUT: reallocation of proof list failed", "", "INVALID_PROOF"),
    ("PB3", 1, "", "There is no contradicting constraint in the database.", "INVALID_PROOF"),
])
@pytest.mark.skipif(os.name != "posix", reason="POSIX campaign execution")
def test_checker_resource_diagnostics_do_not_become_invalid_proofs(tmp_path, proof_format, code, stdout, stderr, expected):
    cnf = tmp_path / "input.cnf"
    cnf.write_text("p cnf 1 2\n1 0\n-1 0\n")
    entry = solver("proof", "import pathlib,sys; pathlib.Path(sys.argv[2]).write_text('0\\n'); print('s UNSATISFIABLE'); raise SystemExit(20)",
                   proof_format=proof_format)
    checker_code = f"import sys; print({stdout!r}); print({stderr!r}, file=sys.stderr); raise SystemExit({code})"
    entry["proof_checker"] = {"command": [sys.executable, "-c", checker_code, "{cnf}", "{proof}"], "success_line": "s VERIFIED"}
    inventory, field = manifests(tmp_path, [instance(cnf, expected="UNSAT")], [entry])
    args = arguments(tmp_path, inventory, field)
    summary = run_campaign(args)
    retained = json.loads((args.out / "cells.jsonl").read_text())
    assert retained["status"] == expected
    assert summary["standings"][0]["solved"] == 0
    assert summary["standings"][0]["disqualified"] == (expected == "INVALID_PROOF")
    assert ("checker_resource_diagnostic" in retained) == (expected == "CHECKER_RESOURCE_FAILURE")


def test_dsr_io_failure_only_recognizes_explicit_lsr_output():
    process = {"returncode": 255, "command": ["/trusted/dsr-trim", "/input.cnf", "/proof.dsr", "/derived.lsr"]}
    output_error = "c Error: Unable to open the file /derived.lsr"
    assert checker_resource_diagnostic("DSR", process, [], [output_error]) == output_error
    assert checker_resource_diagnostic("DSR", process, [], ["c Error: Unable to open the file /proof.dsr"]) is None
    assert checker_resource_diagnostic("DSR", {**process, "returncode": 1}, [], [output_error]) is None
    assert checker_resource_diagnostic("DRAT", process, [], [output_error]) is None


@pytest.mark.parametrize("requested,expected", [
    (None, "declared-inventory-v2"), ("declared-inventory-v2", "declared-inventory-v2"),
    ("legacy-v1", "legacy-v1"),
])
def test_scoring_policy_cli_default_and_explicit_choices(tmp_path, monkeypatch, requested, expected):
    from . import campaign
    observed = []

    def capture(args):
        observed.append(args.scoring_policy)
        return {"status": "CLI_ONLY_TEST"}

    monkeypatch.setattr(campaign, "run_campaign", capture)
    command = ["run", "--inventory", str(tmp_path / "inventory.json"),
               "--solvers", str(tmp_path / "solvers.json"), "--out", str(tmp_path / "out"),
               "--checker", str(tmp_path / "checker")]
    if requested is not None:
        command += ["--scoring-policy", requested]
    assert campaign.main(command) == 0 and observed == [expected]
    with pytest.raises(SystemExit) as error:
        campaign.main(command + ["--scoring-policy", "unrecognized"])
    assert error.value.code == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX campaign execution")
def test_explicit_legacy_run_preserves_historical_metadata_conflict_status(tmp_path):
    cnf = tmp_path / "input.cnf"
    cnf.write_text("p cnf 1 1\n1 0\n")
    inventory, field = manifests(tmp_path, [instance(cnf, expected="UNSAT")], [solver()])
    args = arguments(tmp_path, inventory, field)
    args.scoring_policy = "legacy-v1"
    result = run_campaign(args)
    retained = json.loads((args.out / "cells.jsonl").read_text())
    assert retained["status"] == "INVALID_VERDICT_CONFLICT" and "metadata_conflict" not in retained
    assert result["standings"][0]["disqualified"] and "scoring_policy" not in result
    assert json.loads((args.out / "campaign.json").read_text())["scoring_policy"] == "legacy-v1"


@pytest.fixture(scope="module")
def preprocessed_native_libraries(tmp_path_factory):
    if sys.platform != "linux" or shutil.which("g++") is None:
        pytest.skip("native binary frontend currently requires Linux and g++")
    from .native_binary_core import build_native_binary_core
    from .native_cdcl import build_native_cdcl
    root = tmp_path_factory.mktemp("preprocessed-native-worker")
    return build_native_cdcl(root / "backend"), build_native_binary_core(root / "frontend")


def _preprocessed_worker_args(tmp_path, libraries, nvars, rows, **limits):
    from .proof_backend import write_dimacs
    return argparse.Namespace(
        cnf=write_dimacs(tmp_path / "input.cnf", nvars, rows),
        library=libraries[0], binary_library=libraries[1],
        proof=tmp_path / "proof.drat", stats=tmp_path / "stats.json",
        **{"max_conflicts": 10000, "max_learned_literals": 100000,
           "max_proof_clauses": 100000, "max_proof_literals": 1000000,
           "max_proof_bytes": 10000000, **limits})


def _preprocessed_unsat_rows():
    # The quotient creates units; the residual alone is not a proof of the
    # original. Both the substitution prefix and lifted backend are required.
    return ((-1, 2), (1, -2), (1, 2), (-3, 4), (3, -4), (3, 4),
            (-1, -3, 5, 6), (-1, -3, 5, -6),
            (-1, -3, -5, 6), (-1, -3, -5, -6))


def test_preprocessed_native_campaign_checks_original_models_and_proofs(
    tmp_path, preprocessed_native_libraries,
):
    checker = os.environ.get("DRAT_TRIM")
    if not checker:
        pytest.skip("set DRAT_TRIM to the pinned external checker")
    from .proof_backend import write_dimacs
    fixtures = (
        ("empty-sat", 0, (), "SAT"),
        ("signed-model", 3, ((1, 2), (-1, -2), (1,)), "SAT"),
        ("residual-model", 6, ((-1, 2), (1, -2), (3, 4, 5)), "SAT"),
        ("empty-unsat", 0, ((),), "UNSAT"),
        ("binary-conflict", 2, ((1, 2), (1, -2), (-1, 2), (-1, -2)), "UNSAT"),
        ("quotient-conflict", 5, ((-1, 2), (1, -2), (1, 2), (-3, 4),
                                (3, -4), (3, 4), (-5,), (-1, -3, 5)), "UNSAT"),
        ("backend-proof", 6, _preprocessed_unsat_rows(), "UNSAT"),
    )
    entries = [instance(write_dimacs(tmp_path / (name + ".cnf"), nvars, rows),
                        ident=name, expected=expected)
               for name, nvars, rows, expected in fixtures]
    backend, frontend = preprocessed_native_libraries
    command = [sys.executable, "-m", "experiments.general_competition.campaign", "native",
               "--library", str(backend), "--binary-library", str(frontend),
               "--cnf", "{cnf}", "--proof", "{proof}", "--stats", "{stats}"]
    # This pinned DRAT-trim returns 1 with "s VERIFIED" when the original
    # already contains an empty clause. Declare its contract explicitly; the
    # runner's default exit-0 requirement and historical scores stay intact.
    entry = solver("native", command=command, track="standalone-native", upstream_engines=[],
                   artifact_paths=[str(backend), str(frontend)], proof_checker={
                       "command": [str(Path(checker).resolve()), "{cnf}", "{proof}"],
                       "success_line": "s VERIFIED", "accepted_exit_codes": [0, 1]})
    inventory, field = manifests(tmp_path, entries, [entry])
    result = run_campaign(arguments(tmp_path, inventory, field, Path(checker)))
    assert result["resource_summary"]["native"]["statuses"] == {
        "SAT_VERIFIED": 3, "UNSAT_VERIFIED": 4}
    stats = []
    for index in range(len(fixtures)):
        folder = tmp_path / "evidence/instances" / f"{index:05d}" / "solver-000"
        value = json.loads((folder / "stats.json").read_text())
        stats.append(value)
        assert value["frontend"] == "native-binary-core-v1"
        assert value["independent_checking_included"] is False
        assert value["proof_generation_included_in_solve"] is False
        assert value["worker_seconds"] >= sum(value[key] for key in (
            "preprocessing_seconds", "native_solve_seconds", "proof_composition_seconds",
            "output_serialization_seconds"))
        assert value["frontend_library_sha256"] == digest(frontend)
        assert value["proof_budget_reason"] is None
        composition = value["proof_composition"]
        if composition is not None:
            proof = folder / "proof.drat"
            assert proof.stat().st_size == composition["proof_bytes"]
            assert len(proof.read_text().splitlines()) == sum(composition[key] for key in (
                "prefix_clauses", "backend_clauses", "frontend_terminal_clauses"))
    assert [value["backend_called"] for value in stats] == [False, False, True, False, False, False, True]
    assert all(value["stats"]["conflicts"] == 0 for value in (stats[0], stats[1], *stats[3:6]))
    composed = stats[-1]["proof_composition"]
    assert composed["prefix_clauses"] == 10 and composed["prefix_literals"] == 26
    assert composed["backend_clauses"] == stats[-1]["stats"]["learned_clauses"] + 1
    assert composed["backend_literals"] == stats[-1]["stats"]["learned_literals"]
    assert composed["frontend_terminal_clauses"] == 0


@pytest.mark.parametrize("limit,reason", (
    ("max_proof_clauses", "proof_clause_budget"),
    ("max_proof_literals", "proof_literal_budget"),
    ("max_proof_bytes", "proof_byte_budget"),
))
def test_preprocessed_worker_proof_budget_emits_unknown_without_partial_proof(
    tmp_path, preprocessed_native_libraries, capsys, limit, reason,
):
    from .campaign import native_worker
    args = _preprocessed_worker_args(tmp_path, preprocessed_native_libraries, 6,
                                     _preprocessed_unsat_rows(), **{limit: 0})
    assert native_worker(args) == 0
    assert capsys.readouterr().out == "s UNKNOWN\n"
    assert not args.proof.exists()
    stats = json.loads(args.stats.read_text())
    assert stats["status_code"] == 0 and stats["proof_budget_reason"] == reason
    assert stats["backend_status_code"] == 20
    assert stats["proof_composition"] is None and stats["backend_called"] is True


@pytest.mark.parametrize("limit", ("max_conflicts", "max_learned_literals",
                                   "max_proof_clauses", "max_proof_literals", "max_proof_bytes"))
def test_preprocessed_worker_validates_limits_before_terminal_frontend(
    tmp_path, preprocessed_native_libraries, limit, capsys,
):
    from .campaign import native_worker
    args = _preprocessed_worker_args(tmp_path, preprocessed_native_libraries, 0, ((),), **{limit: -1})
    with pytest.raises(ValueError):
        native_worker(args)
    assert not capsys.readouterr().out
    assert not args.stats.exists() and not args.proof.exists()


def test_preprocessed_worker_backend_unknown_is_not_upgraded(
    tmp_path, preprocessed_native_libraries, capsys,
):
    from .campaign import native_worker
    args = _preprocessed_worker_args(tmp_path, preprocessed_native_libraries, 6,
                                     _preprocessed_unsat_rows(), max_conflicts=0)
    assert native_worker(args) == 0
    assert capsys.readouterr().out == "s UNKNOWN\n"
    stats = json.loads(args.stats.read_text())
    assert stats["status_code"] == 1 and stats["stats"]["conflicts"] == 1
    assert stats["proof_budget_reason"] is None and stats["proof_composition"] is None
    assert not args.proof.exists()


def test_preprocessed_worker_frontend_proof_budget_keeps_backend_idle(
    tmp_path, preprocessed_native_libraries, capsys,
):
    from .campaign import native_worker
    args = _preprocessed_worker_args(tmp_path, preprocessed_native_libraries, 0, ((),),
                                     max_proof_bytes=1)
    assert native_worker(args) == 0
    assert capsys.readouterr().out == "s UNKNOWN\n"
    stats = json.loads(args.stats.read_text())
    assert stats["proof_budget_reason"] == "proof_byte_budget"
    assert stats["backend_called"] is False and stats["backend_status_code"] is None
    assert stats["native_solve_seconds"] == 0 and not any(stats["stats"].values())
    assert stats["proof_composition"] is None and not args.proof.exists()


def test_preprocessed_worker_malformed_composition_is_not_a_budget_unknown(
    tmp_path, preprocessed_native_libraries, monkeypatch, capsys,
):
    from . import native_cdcl
    from .campaign import native_worker
    def malformed(*args, **kwargs):
        raise ValueError("malformed original-CNF evidence")
    monkeypatch.setattr(native_cdcl, "_compose_native_original_rup", malformed)
    args = _preprocessed_worker_args(tmp_path, preprocessed_native_libraries, 6,
                                     _preprocessed_unsat_rows(), max_proof_bytes=0)
    with pytest.raises(ValueError, match="malformed original-CNF evidence"):
        native_worker(args)
    assert not capsys.readouterr().out
    assert not args.proof.exists() and not args.stats.exists()
