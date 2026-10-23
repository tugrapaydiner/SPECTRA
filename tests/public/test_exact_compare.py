"""Cross-platform exact self-test comparison contracts.

The comparison audits every bundle in a fresh process and requires identical
source, deterministic packets, certified outputs, proofs and selector receipts.
Platform, Python, compiler and native-library identities may differ.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from spectra.evidence import capture
from spectra.exact_compare import COMPARE_SCHEMA


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


@pytest.mark.skipif(
    sys.platform not in ("linux", "win32") or not _compiler_available(),
    reason="supported native compiler is unavailable",
)
def test_cli_exact_compare_accepts_environment_changes_but_rejects_semantic_drift(tmp_path):
    first = tmp_path / "first"
    generated = _run("exact", "selftest", first)
    assert generated.returncode == 0, generated.stderr
    first_report = json.loads((first / "selftest.json").read_text(encoding="utf-8"))

    second = tmp_path / "second"
    shutil.copytree(first, second)
    second_report_path = second / "selftest.json"
    second_report = json.loads(second_report_path.read_text(encoding="utf-8"))
    second_report["platform"] = "different-test-platform"
    second_report["machine"] = "different-test-machine"
    second_report_path.write_text(
        json.dumps(second_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    second_receipt_path = second / second_report["build"]["receipt"]
    second_receipt = json.loads(
        second_receipt_path.read_text(encoding="utf-8")
    )
    second_receipt["platform"] = "different-test-platform"
    second_receipt_path.write_text(
        json.dumps(second_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_manifest(second)

    compared = _run("exact", "compare", first, second)
    assert compared.returncode == 0, compared.stderr
    result = json.loads(compared.stdout)
    assert result["schema"] == COMPARE_SCHEMA
    assert result["verified"] is True
    assert result["bundles"] == 2
    assert result["cases"] == 3
    assert (result["queries"], result["sat"], result["unsat"]) == (13, 10, 3)
    assert result["proof_kinds"] == [1, 2, 3]
    assert result["selected_modes"] == {
        "base-static-contradiction": "raw",
        "mixed-current-query": "raw",
        "strong-compression": "scc",
    }
    assert result["selector_reasons"] == {
        "base-static-contradiction": "base_infeasible",
        "mixed-current-query": "short_weak_compression",
        "strong-compression": "scc_strong_compression",
    }
    assert result["source_sha256"] == first_report["build"]["source_sha256"]
    assert result["distinct_library_hashes"] == 1
    assert result["native_execution"] is False
    assert result["native_module_loaded"] is False
    assert len(result["semantic_fingerprint"]) == 64
    assert all(
        character in "0123456789abcdef"
        for character in result["semantic_fingerprint"]
    )
    assert result["packet_hashes"] == {
        case["name"]: case["packet_sha256"] for case in first_report["cases"]
    }
    assert result["output_hashes"] == {
        case["name"]: case["output_sha256"] for case in first_report["cases"]
    }
    assert result["environments"][1]["platform"] == "different-test-platform"
    assert result["environments"][1]["machine"] == "different-test-machine"

    single = _run("exact", "compare", first)
    assert single.returncode == 2
    assert "at least two" in single.stderr.lower()

    second_report = json.loads(second_report_path.read_text(encoding="utf-8"))
    second_report["spectra_version"] = "9.9.9"
    second_report_path.write_text(
        json.dumps(second_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_manifest(second)
    mismatch = _run("exact", "compare", first, second)
    assert mismatch.returncode == 2
    assert "different semantic identity" in mismatch.stderr.lower()
