"""Offline comparison of independently generated exact self-test bundles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from . import __version__
from .exact import DEFAULT_BYTES, EXACT_API_VERSION, EXACT_NATIVE_ABI
from .exact_selftest import SCHEMA as SELFTEST_SCHEMA, verify_exact_selftest

COMPARE_SCHEMA = "spectra.exact_selftest_compare.v1"
_JSON_MAX_BYTES = 16 << 20


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_max_bytes(max_bytes: int) -> None:
    if type(max_bytes) is not int or not 1 <= max_bytes <= sys.maxsize - 1:
        raise ValueError("max-bytes must be a positive bounded integer")


def _load_report(root: Path, max_bytes: int) -> dict:
    path = root / "selftest.json"
    with path.open("rb") as stream:
        data = stream.read(min(max_bytes, _JSON_MAX_BYTES) + 1)
    if len(data) > min(max_bytes, _JSON_MAX_BYTES):
        raise ValueError(f"{path} exceeds max-bytes")
    value = json.loads(data)
    if type(value) is not dict:
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _identity(report: dict) -> dict:
    cases = []
    for row in report["cases"]:
        cases.append({
            "name": row["name"],
            "vertices": row["vertices"],
            "edges": row["edges"],
            "queries": row["queries"],
            "sat": row["sat"],
            "unsat": row["unsat"],
            "proof_kinds": row["proof_kinds"],
            "selected_mode": row["selected_mode"],
            "auto_policy": row["auto_policy"],
            "auto_policy_reason": row["auto_policy_reason"],
            "packet_bytes": row["packet_bytes"],
            "packet_sha256": row["packet_sha256"],
            "output_bytes": row["output_bytes"],
            "output_sha256": row["output_sha256"],
        })
    return {
        "selftest_schema": report["schema"],
        "origin_spectra_version": report["spectra_version"],
        "exact_api_version": report["exact_api_version"],
        "native_abi": report["native_abi"],
        "source_sha256": report["build"]["source_sha256"],
        "cases": cases,
        "proof_kinds": report["checks"]["proof_kinds"],
    }


def compare_exact_selftests(
    directories,
    *,
    max_bytes: int = DEFAULT_BYTES,
) -> dict:
    """Audit and compare two or more bundles without loading native code.

    Platform, Python, compiler and native-library identities may differ. The
    retained source, deterministic packets, certified outputs, proof coverage and
    frozen selector receipts must match exactly.
    """
    _validate_max_bytes(max_bytes)
    _require(
        "_spectra_exact_binary" not in sys.modules,
        "exact compare must run in a fresh process without the native module loaded",
    )
    _require(
        not isinstance(directories, (str, bytes, Path)),
        "exact compare directories must be an iterable of paths",
    )
    supplied = tuple(Path(directory) for directory in directories)
    roots = tuple(path.resolve(strict=True) for path in supplied)
    _require(len(roots) >= 2, "exact compare needs at least two self-test directories")
    _require(len(set(roots)) == len(roots), "exact compare directories must be distinct")

    audits = []
    reports = []
    environments = []
    for supplied_root, root in zip(supplied, roots):
        audit = verify_exact_selftest(supplied_root, max_bytes=max_bytes)
        report = _load_report(root, max_bytes)
        audits.append(audit)
        reports.append(report)
        environments.append({
            "root": str(root),
            "platform": report["platform"],
            "machine": report["machine"],
            "python": report["python"],
            "compiler": report["build"]["compiler"],
            "library_sha256": report["build"]["library_sha256"],
        })

    reference = _identity(reports[0])
    _require(reference["selftest_schema"] == SELFTEST_SCHEMA, "self-test schema differs")
    _require(reference["exact_api_version"] == EXACT_API_VERSION, "exact API differs")
    _require(reference["native_abi"] == EXACT_NATIVE_ABI, "native ABI differs")
    for index, report in enumerate(reports[1:], start=2):
        _require(
            _identity(report) == reference,
            f"self-test bundle {index} has a different semantic identity",
        )

    fingerprint = hashlib.sha256(
        json.dumps(
            reference,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    total_queries = sum(case["queries"] for case in reference["cases"])
    total_sat = sum(case["sat"] for case in reference["cases"])
    total_unsat = sum(case["unsat"] for case in reference["cases"])
    _require(
        "_spectra_exact_binary" not in sys.modules,
        "exact compare loaded the native module",
    )
    return {
        "schema": COMPARE_SCHEMA,
        "verified": True,
        "bundles": len(roots),
        "semantic_fingerprint": fingerprint,
        "selftest_schema": reference["selftest_schema"],
        "origin_spectra_version": reference["origin_spectra_version"],
        "auditor_spectra_version": __version__,
        "exact_api_version": reference["exact_api_version"],
        "native_abi": reference["native_abi"],
        "source_sha256": reference["source_sha256"],
        "cases": len(reference["cases"]),
        "queries": total_queries,
        "sat": total_sat,
        "unsat": total_unsat,
        "proof_kinds": reference["proof_kinds"],
        "packet_hashes": {
            case["name"]: case["packet_sha256"] for case in reference["cases"]
        },
        "output_hashes": {
            case["name"]: case["output_sha256"] for case in reference["cases"]
        },
        "distinct_library_hashes": len({
            audit["library_sha256"] for audit in audits
        }),
        "environments": environments,
        "native_execution": False,
        "native_module_loaded": False,
    }
