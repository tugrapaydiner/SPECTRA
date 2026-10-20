from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def write(name: str, value: str) -> None:
    (ROOT / name).write_text(value, encoding="utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def patch_policy_model() -> None:
    name = "experiments/compiled_query/auto_selector_benchmark.py"
    text = read(name)
    text = replace_once(
        text,
        "    if count >= 64 and components == 2 * vertices and edges >= 4 * vertices:\n",
        "    if (vertices > 0 and count >= 64 and components == 2 * vertices\n"
        "            and edges >= 4 * vertices):\n",
        "nonempty dense benchmark policy",
    )
    text = replace_once(
        text,
        "    if count >= 8 and 2 * components <= vertices:\n",
        "    if vertices > 0 and count >= 8 and 2 * components <= vertices:\n",
        "nonempty compressed benchmark policy",
    )
    write(name, text)


def patch_manifest_boundaries() -> None:
    name = "spectra/exact_selftest.py"
    text = read(name)
    marker = '        if path.is_file() and path.name != "SHA256.json"\n'
    if text.count(marker) != 2:
        raise AssertionError("expected generation and audit manifest checks")
    text = text.replace(
        marker,
        '        if path.is_file() and path != root / "SHA256.json"\n',
        1,
    )
    text = replace_once(
        text,
        marker,
        "        if path.is_file() and path != manifest_path\n",
        "audit manifest inventory boundary",
    )
    write(name, text)

    name = "experiments/compiled_query/twosat_development_benchmark.py"
    text = read(name)
    text = replace_once(
        text,
        '        if path.is_file() and path.name != "SHA256.json":\n',
        '        if path.is_file() and path != root / "SHA256.json":\n',
        "development manifest inventory boundary",
    )
    write(name, text)


def patch_public_tests() -> None:
    name = "tests/public/test_exact_selftest.py"
    text = read(name)
    text = replace_once(
        text,
        '    unbound.unlink()\n\n    report_path = root / "selftest.json"\n',
        '    unbound.unlink()\n\n'
        '    nested_manifest = root / "cases" / "SHA256.json"\n'
        '    nested_manifest.write_text("not the root manifest\\n", encoding="utf-8")\n'
        '    nested = _run("exact", "audit", root)\n'
        '    assert nested.returncode == 2\n'
        '    assert "unbound or missing" in nested.stderr.lower()\n'
        '    nested_manifest.unlink()\n\n'
        '    report_path = root / "selftest.json"\n',
        "nested manifest rejection test",
    )
    marker = "@pytest.mark.skipif(\n"
    new_test = '''def test_installation_checker_keeps_reports_outside_the_frozen_bundle(tmp_path):
    evidence = tmp_path / "retained-bundle"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_exact_installation.py",
            "--out",
            str(evidence / "report.json"),
            "--evidence-dir",
            str(evidence),
        ],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "outside the retained evidence directory" in (
        result.stdout + result.stderr
    ).lower()
    assert not evidence.exists()


'''
    text = replace_once(
        text,
        marker,
        new_test + marker,
        "installed evidence contamination test",
    )
    write(name, text)


def patch_installation_checker() -> None:
    name = "scripts/check_exact_installation.py"
    text = read(name)
    anchor = '''    retained_evidence = (
        args.evidence_dir.resolve() if args.evidence_dir is not None else None
    )
    if retained_evidence is not None:
'''
    replacement = '''    retained_evidence = (
        args.evidence_dir.resolve() if args.evidence_dir is not None else None
    )
    if output is not None and retained_evidence is not None:
        try:
            output.relative_to(retained_evidence)
        except ValueError:
            pass
        else:
            raise RuntimeError(
                "output report must be outside the retained evidence directory"
            )
    if retained_evidence is not None:
'''
    text = replace_once(
        text,
        anchor,
        replacement,
        "installed evidence path boundary",
    )
    write(name, text)


def patch_edgeless_receipts() -> None:
    for name in (
        "spectra/_native/exact_binary.cpp",
        "experiments/compiled_query/exact_native.cpp",
    ):
        text = read(name)
        text = replace_once(
            text,
            '            "retained_proof_arcs",static_cast<unsigned long long>(fast_arcs),\n'
            '            "condensation_arcs",static_cast<unsigned long long>(fast_arcs),\n',
            '            "retained_proof_arcs",static_cast<unsigned long long>(0),\n'
            '            "condensation_arcs",static_cast<unsigned long long>(0),\n',
            f"edgeless metadata in {name}",
        )
        write(name, text)

    name = "tests/public/test_exact_edgeless.py"
    text = read(name)
    text = replace_once(
        text,
        "        assert result.info['bank_kernel']=='edgeless'\n"
        "        assert result.info['selected_mode']=='scc'\n",
        "        assert result.info['bank_kernel']=='edgeless'\n"
        "        assert result.info['retained_proof_arcs']==0\n"
        "        assert result.info['condensation_arcs']==0\n"
        "        assert result.info['retained_array_bytes']==0\n"
        "        assert result.info['native_observer_array_bytes']==0\n"
        "        assert result.info['original_adjacency_retained'] is False\n"
        "        assert result.info['selected_mode']=='scc'\n",
        "edgeless retained-state contract",
    )
    write(name, text)


def patch_policy_tests_and_docs() -> None:
    name = "experiments/compiled_query/test_exact.py"
    text = read(name)
    addition = '''

def test_benchmark_policy_model_does_not_infer_compression_from_an_empty_graph():
    from .auto_selector_benchmark import expected_policy

    assert expected_policy(
        base=True, pins=0, count=8, vertices=0, edges=0, components=0
    ) == ("raw", "short_weak_compression")
    assert expected_policy(
        base=True, pins=0, count=64, vertices=0, edges=0, components=0
    ) == ("scc", "scc_long_bank")
'''
    if "def test_benchmark_policy_model_does_not_infer_compression_from_an_empty_graph" in text:
        raise AssertionError("benchmark policy regression test already exists")
    write(name, text.rstrip() + "\n" + addition)

    policy_doc = '''        The default ``auto`` mode uses frozen structural policy v11. Edgeless
        graphs use a direct-domain kernel while retaining the same structural mode
        and reason receipts. Banks of at most eight requests on graphs above 512
        vertices take the exact raw path directly, and fixed contradictions stay
        raw. Dense high-pin banks shorter than 64 requests stay raw. Below 96
        vertices, dense high-pin banks stay raw only when SCC compression is
        present. Nonempty dense uncompressed graphs with at least 64 requests and
        at least four edges per vertex stay raw. Other feasible banks of at least
        64 requests use SCC; nonempty banks from eight requests use SCC earlier
        when SCCs compress the literal graph by at least fourfold. Selection uses
        immutable graph/request geometry only, never answers or measured timings.
        ``info['selected_mode']`` and ``info['auto_policy_reason']`` expose the
        decision. Explicit modes keep their meaning; the automatic policy is a
        frozen structural dispatcher.
'''
    old_end = (
        "        their meaning; the automatic policy is a frozen structural dispatcher.\n"
    )
    for name in ("spectra/exact.py", "experiments/compiled_query/exact.py"):
        text = read(name)
        start = text.index(
            "        The default ``auto`` mode uses frozen structural policy v10."
        )
        end = text.index(old_end, start) + len(old_end)
        write(name, text[:start] + policy_doc + text[end:])


def patch_status_docs() -> None:
    name = "docs/PART3_RELEASE_REPLICATION_READINESS.md"
    text = read(name)
    text = replace_once(
        text,
        "**Branch:** `part3/release-replication-readiness`  \n",
        "**Branch:** `finish-part3`\n",
        "Part 3 branch name",
    )
    old_goal = '''Part 2 completed the frozen `structural-v10` automatic-selector milestone. Part 3
does not retune that selector. Its controllable goal is to turn the validated
source tree into a release-facing, independently auditable exact service with one
obvious smoke-test path, portable receipts, clean installed-wheel checks, and an
explicit boundary between engineering verification and scientific replication.
'''
    new_goal = '''Part 2 completed the frozen `structural-v10` automatic-selector milestone. Part 3
preserves that evidence while developing the `structural-v11` candidate, including
the edgeless and direct-conflict paths. Because the native source and selector
receipts changed, every release and cross-platform gate must be rerun on one final
Part 3 identity. The broader goal is a release-facing, independently auditable exact
service with portable receipts, clean installed-wheel checks, and an explicit
boundary between engineering verification and scientific replication.
'''
    text = replace_once(text, old_goal, new_goal, "Part 3 goal")
    start = text.index("## Reference cross-platform result\n")
    end = text.index("## Frozen acceptance gates for this track\n", start)
    current = '''## Current candidate status

The four-platform `v2` bundle recorded under pull request #73 belongs to the
earlier pre-`structural-v11` source. Its source hash and semantic fingerprint are
preserved in workflow history, but they are historical evidence only and must not
be presented as identities for the current candidate.

The `finish-part3` branch remains a draft. A current reference result exists only
after one exact final source identity passes the complete Linux 3.11, Linux 3.13,
native ARM64, Windows/MSVC, UBSan, ASan+UBSan, installed-wheel, no-native audit,
cross-bundle comparison, repository-audit and package checks. The new source hash,
bundle fingerprints and workflow identities will be recorded after that matrix
passes; old hashes are never copied forward.

'''
    write(name, text[:start] + current + text[end:])

    name = "docs/EXACT_REPLICATION.md"
    text = read(name)
    text = replace_once(
        text,
        "A successful run creates exactly ten frozen files beneath that directory:\n",
        "A successful run creates ten manifest-bound artifacts plus `SHA256.json`:\n",
        "replication artifact count",
    )
    write(name, text)

    name = "experiments/compiled_query/TWOSAT_CHALLENGE_PROTOCOL.md"
    text = read(name)
    text = replace_once(
        text,
        "**Branch:** `part3/release-replication-readiness`\n",
        "**Branch:** `finish-part3`\n",
        "2-SAT protocol branch name",
    )
    write(name, text)

    name = "CHANGELOG.md"
    text = read(name)
    text = replace_once(
        text,
        "- Retain complete installed-wheel bundles from Linux 3.11, Linux 3.13, ARM64 and\n"
        "  Windows, then compare them in one fresh no-native CI job and publish the semantic\n"
        "  comparison receipt as a separate artifact.\n",
        "- Prepare CI to retain complete installed-wheel bundles from Linux 3.11, Linux\n"
        "  3.13, ARM64 and Windows, compare them in a fresh no-native job, and publish a\n"
        "  semantic receipt only after the final candidate matrix passes.\n",
        "unreleased cross-platform wording",
    )
    write(name, text)


def main() -> None:
    patch_policy_model()
    patch_manifest_boundaries()
    patch_public_tests()
    patch_installation_checker()
    patch_edgeless_receipts()
    patch_policy_tests_and_docs()
    patch_status_docs()
    Path(__file__).unlink()


if __name__ == "__main__":
    main()
