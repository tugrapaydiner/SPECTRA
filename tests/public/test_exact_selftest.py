"""Release-facing exact self-test and no-native audit contracts.

This suite checks evidence structure and semantic coverage. It is not a benchmark
and does not upgrade project-owned CI into independent scientific replication.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from spectra.evidence import capture, verify
from spectra.exact_selftest import AUDIT_SCHEMA, SCHEMA, verify_exact_selftest


REPOSITORY = Path(__file__).resolve().parents[2]


def _compiler_available() -> bool:
    return bool(shutil.which("cl" if sys.platform == "win32" else "g++"))


def _run(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "spectra", *(str(value) for value in arguments)],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )


def _refresh_manifest(root: Path) -> None:
    path = root / "SHA256.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    manifest = capture(root, old["artifacts"])
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_audit_refuses_an_already_loaded_native_module(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "_spectra_exact_binary", object())
    with pytest.raises(RuntimeError, match="fresh process"):
        verify_exact_selftest(tmp_path)


@pytest.mark.parametrize("reader", ["audit", "compare"])
@pytest.mark.parametrize("raw", [
    b'{"schema":"wrong","schema":"right"}',
    b'{"elapsed_ns":NaN}',
    b'{"elapsed_ns":1e999}',
    b'{"nested":' + b'[' * 10000 + b']' * 10000 + b'}',
], ids=["duplicate-key", "nonfinite", "overflowing-float", "deep-nesting"])
def test_exact_artifact_json_rejects_ambiguous_or_unbounded_values(tmp_path, reader, raw):
    from spectra.exact_compare import _load_report
    from spectra.exact_selftest import _load_json

    path = tmp_path / "selftest.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        if reader == "audit":
            _load_json(path, max_bytes=1 << 20)
        else:
            _load_report(tmp_path, max_bytes=1 << 20)


def test_exact_audit_binary_limit_is_not_an_allocation_request(tmp_path):
    from spectra.exact_selftest import _read_bounded

    path = tmp_path / "packet.sbc1"
    payload = b"a small retained artifact"
    path.write_bytes(payload)
    assert _read_bounded(path, max_bytes=sys.maxsize - 1) == payload
    assert _read_bounded(path, max_bytes=len(payload)) == payload
    with pytest.raises(ValueError, match="exceeds"):
        _read_bounded(path, max_bytes=len(payload) - 1)


@pytest.mark.skipif(
    sys.platform not in ("linux", "win32") or not _compiler_available(),
    reason="supported native compiler is unavailable",
)
def test_cli_exact_selftest_and_fresh_no_native_audit(tmp_path):
    root = tmp_path / "exact-selftest"
    generated = _run("exact", "selftest", root)
    assert generated.returncode == 0, generated.stderr
    emitted = json.loads(generated.stdout)
    saved = json.loads((root / "selftest.json").read_text(encoding="utf-8"))
    assert emitted == saved

    assert saved["schema"] == SCHEMA
    assert saved["status"] == "PASS"
    assert saved["exact_api_version"] == 1
    assert saved["native_abi"] == 3
    assert saved["checks"] == {
        "independent_python_replay": True,
        "native_original_input_observer": True,
        "proof_kinds": [1, 2, 3],
        "receipt_bound_native_load": True,
        "tampered_output_rejected": True,
    }

    cases = {case["name"]: case for case in saved["cases"]}
    assert cases["mixed-current-query"]["proof_kinds"] == [None, None, 2, 1]
    assert cases["mixed-current-query"]["selected_mode"] == "raw"
    assert cases["mixed-current-query"]["auto_policy_reason"] == "short_weak_compression"
    assert (
        cases["mixed-current-query"]["sat"],
        cases["mixed-current-query"]["unsat"],
    ) == (2, 2)
    assert cases["strong-compression"]["proof_kinds"] == [None] * 8
    assert cases["strong-compression"]["selected_mode"] == "scc"
    assert (
        cases["strong-compression"]["auto_policy_reason"]
        == "scc_strong_compression"
    )
    assert (
        cases["strong-compression"]["sat"],
        cases["strong-compression"]["unsat"],
    ) == (8, 0)
    assert cases["base-static-contradiction"]["proof_kinds"] == [3]
    assert cases["base-static-contradiction"]["selected_mode"] == "raw"
    assert cases["base-static-contradiction"]["auto_policy_reason"] == "base_infeasible"
    assert (
        cases["base-static-contradiction"]["sat"],
        cases["base-static-contradiction"]["unsat"],
    ) == (0, 1)

    source = root / saved["build"]["source"]
    receipt = root / saved["build"]["receipt"]
    library = root / saved["build"]["library"]
    assert source.is_file() and receipt.is_file() and library.is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == saved["build"]["source_sha256"]
    assert hashlib.sha256(library.read_bytes()).hexdigest() == saved["build"]["library_sha256"]

    manifest = json.loads((root / "SHA256.json").read_text(encoding="utf-8"))
    assert verify(root, manifest) == {
        "verified": True,
        "artifacts": 10,
        "format": "spectra.evidence_freeze.v1",
    }
    assert {
        "selftest.json",
        saved["build"]["source"],
        saved["build"]["receipt"],
        saved["build"]["library"],
        *(case["packet"] for case in saved["cases"]),
        *(case["output"] for case in saved["cases"]),
    } == set(manifest["artifacts"])

    # The audit contract is intentionally process-bound. Running it through a
    # fresh interpreter proves that it does not inherit the generator's loaded
    # extension or native state; an in-process call would invalidate that claim.
    audited = _run("exact", "audit", root)
    assert audited.returncode == 0, audited.stderr
    audit = json.loads(audited.stdout)
    assert audit == {
        "schema": AUDIT_SCHEMA,
        "verified": True,
        "selftest_schema": SCHEMA,
        "origin_spectra_version": saved["spectra_version"],
        "auditor_spectra_version": saved["spectra_version"],
        "exact_api_version": 1,
        "native_abi": 3,
        "cases": 3,
        "queries": 13,
        "sat": 10,
        "unsat": 3,
        "proof_kinds": [1, 2, 3],
        "selected_modes": {
            "base-static-contradiction": "raw",
            "mixed-current-query": "raw",
            "strong-compression": "scc",
        },
        "selector_reasons": {
            "base-static-contradiction": "base_infeasible",
            "mixed-current-query": "short_weak_compression",
            "strong-compression": "scc_strong_compression",
        },
        "tamper_checks": 3,
        "manifest_artifacts": 10,
        "source_sha256": saved["build"]["source_sha256"],
        "library_sha256": saved["build"]["library_sha256"],
        "native_execution": False,
        "native_module_loaded": False,
    }

    duplicate = _run("exact", "selftest", root)
    assert duplicate.returncode == 2
    assert "exist" in duplicate.stderr.lower()

    bounded = _run("exact", "audit", root, "--max-bytes", "1")
    assert bounded.returncode == 2
    assert "exceeds max-bytes" in bounded.stderr.lower()

    for unbound in (root / "unbound.txt", root / "native" / "SHA256.json"):
        unbound.write_text("not frozen\n", encoding="utf-8")
        extra = _run("exact", "audit", root)
        assert extra.returncode == 2
        assert "unbound or missing" in extra.stderr.lower()
        unbound.unlink()

    huge_cap = _run("exact", "audit", root, "--max-bytes", sys.maxsize - 1)
    assert huge_cap.returncode == 0, huge_cap.stderr
    assert json.loads(huge_cap.stdout) == audit

    manifest_path = root / "SHA256.json"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        '{"format":"wrong",' + original_manifest[1:], encoding="utf-8",
    )
    ambiguous = _run("exact", "audit", root)
    assert ambiguous.returncode == 2
    assert "duplicate JSON key" in ambiguous.stderr
    manifest_path.write_text(original_manifest, encoding="utf-8")

    report_path = root / "selftest.json"
    original_report = report_path.read_text(encoding="utf-8")
    forged = json.loads(original_report)
    forged["platform"] = "forged-platform"
    report_path.write_text(
        json.dumps(forged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_manifest(root)
    environment_mismatch = _run("exact", "audit", root)
    assert environment_mismatch.returncode == 2
    assert "reported platform differs" in environment_mismatch.stderr.lower()
    report_path.write_text(original_report, encoding="utf-8")
    _refresh_manifest(root)

    output = root / cases["mixed-current-query"]["output"]
    payload = output.read_bytes()
    output.write_bytes(bytes((payload[0] ^ 1,)) + payload[1:])
    tampered = _run("exact", "audit", root)
    assert tampered.returncode == 2
    assert "changed" in tampered.stderr.lower()
