"""Exercise release-facing exact self-test and audit from an installed wheel.

Run with the installation venv's Python and ``-I`` so repository packages cannot
satisfy imports. Native compilation occurs in a disposable child process; a
separate fresh child audits the retained evidence without loading native code.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help=(
            "retain the complete self-test bundle at this new path; when omitted "
            "the bundle is checked in a disposable directory"
        ),
    )
    args = parser.parse_args(argv)
    output = args.out.resolve() if args.out is not None else None
    retained_evidence = (
        args.evidence_dir.resolve() if args.evidence_dir is not None else None
    )
    if retained_evidence is not None:
        require(
            not retained_evidence.exists(),
            "evidence directory already exists: " + str(retained_evidence),
        )
        retained_evidence.parent.mkdir(parents=True, exist_ok=True)

    previous = Path.cwd()
    with tempfile.TemporaryDirectory(
            prefix="spectra-exact-installed-",
            ignore_cleanup_errors=(os.name == "nt")) as temporary:
        root = Path(temporary)
        try:
            os.chdir(root)
            import spectra.evidence as evidence
            import spectra.exact as exact
            import spectra.exact_selftest as exact_selftest

            module_path = Path(exact.__file__).resolve()
            selftest_module_path = Path(exact_selftest.__file__).resolve()
            source_path = exact.SOURCE.resolve()
            prefix = Path(sys.prefix).resolve()
            require(module_path.is_relative_to(prefix),
                    "exact API must come from installed wheel")
            require(selftest_module_path.is_relative_to(prefix),
                    "exact self-test must come from installed wheel")
            require(source_path.is_relative_to(prefix) and source_path.is_file(),
                    "installed wheel must include exact native source")
            for dependency in ("torch", "numpy", "pytest", "pysat"):
                require(importlib.util.find_spec(dependency) is None,
                        "unexpected optional dependency: " + dependency)
            require(importlib.util.find_spec("experiments") is None,
                    "installed contract must not import repository experiments")
            require("_spectra_exact_binary" not in sys.modules,
                    "parent process unexpectedly loaded the native runtime")

            def installed_command(*values):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-m",
                        "spectra",
                        *(str(value) for value in values),
                    ],
                    cwd=root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                require(
                    completed.returncode == 0,
                    "installed command failed:\n"
                    + completed.stdout + "\n" + completed.stderr,
                )
                return json.loads(completed.stdout)

            evidence_root = retained_evidence or (root / "selftest")
            selftest = installed_command("exact", "selftest", evidence_root)
            require(selftest["schema"] == exact_selftest.SCHEMA,
                    "installed self-test schema differs")
            require(selftest["status"] == "PASS",
                    "installed exact self-test failed")
            require(selftest["checks"]["proof_kinds"] == [1, 2, 3],
                    "installed self-test proof coverage differs")
            require(
                [(case["sat"], case["unsat"]) for case in selftest["cases"]]
                == [(2, 2), (8, 0), (0, 1)],
                "installed self-test case outcomes differ",
            )

            audit = installed_command("exact", "audit", evidence_root)
            require(audit["schema"] == exact_selftest.AUDIT_SCHEMA,
                    "installed audit schema differs")
            require(audit["verified"] and audit["native_execution"] is False,
                    "installed no-native audit failed")
            require(audit["native_module_loaded"] is False,
                    "audit process loaded the native module")
            require(audit["proof_kinds"] == [1, 2, 3],
                    "installed audit proof coverage differs")
            require(set(audit["selected_modes"].values()) == {"raw", "scc"},
                    "installed audit did not cover both selector modes")
            require((audit["cases"], audit["queries"], audit["sat"], audit["unsat"])
                    == (3, 13, 10, 3),
                    "installed audit counts differ")

            manifest = json.loads(
                (evidence_root / "SHA256.json").read_text(encoding="utf-8"))
            manifest_check = evidence.verify(evidence_root, manifest)
            require(
                manifest_check == {
                    "verified": True,
                    "artifacts": 10,
                    "format": "spectra.evidence_freeze.v1",
                },
                "installed self-test manifest verification differs",
            )

            retained_source = evidence_root / selftest["build"]["source"]
            retained_library = evidence_root / selftest["build"]["library"]
            receipt = json.loads(
                (evidence_root / selftest["build"]["receipt"]).read_text(
                    encoding="utf-8"))
            require(retained_source.read_bytes() == source_path.read_bytes(),
                    "retained source differs from installed source")
            require(
                hashlib.sha256(retained_source.read_bytes()).hexdigest()
                == receipt["source_sha256"] == audit["source_sha256"],
                "retained source identity differs",
            )
            require(
                hashlib.sha256(retained_library.read_bytes()).hexdigest()
                == receipt["library_sha256"] == audit["library_sha256"],
                "retained native library identity differs",
            )
            require(receipt["exact_api_version"] == exact.EXACT_API_VERSION == 1,
                    "exact API version differs")
            require(receipt["native_abi"] == exact.EXACT_NATIVE_ABI == 3,
                    "native ABI differs")
            require("_spectra_exact_binary" not in sys.modules,
                    "parent process loaded native code during child checks")

            report = {
                "status": "PASS",
                "scope": (
                    "actual installed wheel; fresh isolated self-test process; "
                    "separate fresh no-native audit process"
                ),
                "module": str(module_path),
                "selftest_module": str(selftest_module_path),
                "source": str(source_path),
                "selftest_schema": selftest["schema"],
                "audit_schema": audit["schema"],
                "exact_api_version": exact.EXACT_API_VERSION,
                "native_abi": exact.EXACT_NATIVE_ABI,
                "cases": audit["cases"],
                "queries": audit["queries"],
                "sat": audit["sat"],
                "unsat": audit["unsat"],
                "proof_kinds": audit["proof_kinds"],
                "optional_dependencies": False,
                "manifest_verified": True,
                "manifest_artifacts": manifest_check["artifacts"],
                "retained_source": True,
                "evidence_retained": retained_evidence is not None,
                "audit_process_isolated": True,
                "audit_native_execution": audit["native_execution"],
                "audit_native_module_loaded": audit["native_module_loaded"],
                "receipt_source_sha256": receipt["source_sha256"],
                "receipt_library_sha256": receipt["library_sha256"],
            }
            text = json.dumps(report, indent=2, sort_keys=True) + "\n"
            if output is not None:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(text, encoding="utf-8")
            print(text, end="")
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    main()
