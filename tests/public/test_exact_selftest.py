"""Release-facing exact self-test contracts.

This suite checks evidence structure and semantic coverage. It is not a benchmark
and does not upgrade project-owned CI into independent scientific replication.
"""
from __future__ import annotations

import json
import shutil
import sys

import pytest

from spectra.cli import main as cli_main
from spectra.evidence import verify
from spectra.exact_selftest import SCHEMA


def _compiler_available() -> bool:
    return bool(shutil.which("cl" if sys.platform == "win32" else "g++"))


@pytest.mark.skipif(
    sys.platform not in ("linux", "win32") or not _compiler_available(),
    reason="supported native compiler is unavailable",
)
def test_cli_exact_selftest_builds_and_freezes_all_proof_paths(tmp_path, capsys):
    root = tmp_path / "exact-selftest"
    assert cli_main(["exact", "selftest", str(root)]) == 0
    emitted = json.loads(capsys.readouterr().out)
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
    assert (cases["mixed-current-query"]["sat"],
            cases["mixed-current-query"]["unsat"]) == (2, 2)
    assert cases["base-static-contradiction"]["proof_kinds"] == [3]
    assert (cases["base-static-contradiction"]["sat"],
            cases["base-static-contradiction"]["unsat"]) == (0, 1)

    manifest = json.loads((root / "SHA256.json").read_text(encoding="utf-8"))
    assert verify(root, manifest) == {
        "verified": True,
        "artifacts": len(manifest["artifacts"]),
        "format": "spectra.evidence_freeze.v1",
    }
    assert {
        "selftest.json",
        saved["build"]["receipt"],
        saved["build"]["library"],
        *(case["packet"] for case in saved["cases"]),
        *(case["output"] for case in saved["cases"]),
    }.issubset(manifest["artifacts"])

    assert cli_main(["exact", "selftest", str(root)]) == 2
    assert "exist" in capsys.readouterr().err.lower()
