"""Exercise the release-facing exact self-test from an installed wheel.

Run with the installation venv's Python and ``-I`` so repository packages cannot
satisfy imports. Native compilation is explicit and uses the host compiler.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    output = args.out.resolve() if args.out is not None else None
    previous = Path.cwd()
    # A loaded extension module remains locked until process exit on Windows.
    # The temporary build is disposable, so defer that platform's final removal
    # to the runner/OS rather than converting a completed API check into failure.
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

            evidence_root = root / "selftest"
            selftest = exact_selftest.run_exact_selftest(evidence_root)
            require(selftest["schema"] == exact_selftest.SCHEMA,
                    "installed self-test schema differs")
            require(selftest["status"] == "PASS",
                    "installed exact self-test failed")
            require(selftest["checks"]["proof_kinds"] == [1, 2, 3],
                    "installed self-test proof coverage differs")
            require(
                [(case["sat"], case["unsat"]) for case in selftest["cases"]]
                == [(2, 2), (0, 1)],
                "installed self-test case outcomes differ",
            )
            manifest = json.loads(
                (evidence_root / "SHA256.json").read_text(encoding="utf-8"))
            manifest_check = evidence.verify(evidence_root, manifest)
            require(manifest_check["verified"],
                    "installed self-test manifest verification failed")

            receipt = json.loads(
                (evidence_root / selftest["build"]["receipt"]).read_text(
                    encoding="utf-8"))
            require(receipt["exact_api_version"] == exact.EXACT_API_VERSION == 1,
                    "exact API version differs")
            require(receipt["native_abi"] == exact.EXACT_NATIVE_ABI == 3,
                    "native ABI differs")
            report = {
                "status": "PASS",
                "scope": (
                    "actual installed wheel; isolated Python; explicit native "
                    "build and semantic self-test"
                ),
                "module": str(module_path),
                "selftest_module": str(selftest_module_path),
                "source": str(source_path),
                "selftest_schema": selftest["schema"],
                "exact_api_version": exact.EXACT_API_VERSION,
                "native_abi": exact.EXACT_NATIVE_ABI,
                "cases": len(selftest["cases"]),
                "proof_kinds": selftest["checks"]["proof_kinds"],
                "optional_dependencies": False,
                "manifest_verified": True,
                "manifest_artifacts": manifest_check["artifacts"],
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
