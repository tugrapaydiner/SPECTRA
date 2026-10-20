from __future__ import annotations

from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def patch_selector_docstrings() -> None:
    old = """        The default ``auto`` mode uses frozen structural policy v10. Banks of
        at most eight requests on graphs above 512 vertices take the exact raw
        path directly, and fixed contradictions stay raw. Dense high-pin banks
        shorter than 64 requests stay raw. Below 96 vertices, dense high-pin
        banks stay raw only when SCC compression is present; uncompressed long
        banks may use SCC. At exactly 64 requests, dense uncompressed graphs
        whose edge count is at least four times the vertex count stay raw.
        Other feasible banks of at least 64 requests use SCC; banks from eight
        requests use SCC earlier when SCCs compress the literal graph by at least
        fourfold. Selection uses immutable graph/request geometry only, never
        answers or measured timings. ``info['selected_mode']`` and
        ``info['auto_policy_reason']`` expose the decision. Explicit modes keep
        their meaning; the automatic policy is a frozen structural dispatcher.
"""
    new = """        The default ``auto`` mode uses frozen structural policy v11. Banks of
        at most eight requests on graphs above 512 vertices take the exact raw
        path directly, and fixed contradictions stay raw. Dense high-pin banks
        shorter than 64 requests stay raw. Below 96 vertices, dense high-pin
        banks stay raw when SCC compression is present. Long banks of at least
        64 requests also stay raw when SCC construction leaves all ``2*n``
        literals separate and the original graph has at least four edges per
        vertex. Other feasible long banks use SCC; banks from eight requests use
        SCC earlier when SCCs compress the literal graph by at least fourfold.
        Selection uses immutable graph/request geometry, bank length, and pin
        counts only—never answers or measured timings. For edgeless problems,
        ``auto`` may then use the separately replayed direct-domain bank kernel.
        ``info['selected_mode']``, ``info['auto_policy_reason']``, and
        ``info['bank_kernel']`` expose the decision. Explicit modes keep their
        meaning; the automatic policy is a frozen structural dispatcher.
"""
    for path in ("spectra/exact.py", "experiments/compiled_query/exact.py"):
        text = replace_once(read(path), old, new, label=f"{path} selector docstring")
        write(path, text)


def patch_selector_tests() -> None:
    public_path = "tests/public/test_auto_selector_v11.py"
    text = read(public_path)
    text = replace_once(
        text,
        'tmp_path_factory.mktemp("selector-v10-runtime")',
        'tmp_path_factory.mktemp("selector-v11-runtime")',
        label="selector fixture identity",
    )
    anchor = "\n\ndef solve(runtime, problem, rows):\n"
    check = dedent(
        """

        def test_public_docstring_names_the_frozen_v11_policy():
            doc = ExactRuntime.solve_session_packet.__doc__ or ""
            assert "structural policy v11" in doc
            assert "structural policy v10" not in doc
            assert "at least 64 requests" in doc
            assert "bank_kernel" in doc
        """
    )
    text = replace_once(text, anchor, check + anchor, label="public selector identity test")
    write(public_path, text)

    research_path = "experiments/compiled_query/test_exact.py"
    text = read(research_path)
    marker = "def test_selector_policy_identity_matches_the_public_api():"
    if marker in text:
        raise AssertionError("research selector identity test already exists")
    text += dedent(
        """


        def test_selector_policy_identity_matches_the_public_api():
            from spectra.exact import ExactRuntime as PublicExactRuntime

            for runtime in (PublicExactRuntime, ExactRuntime):
                doc = runtime.solve_session_packet.__doc__ or ""
                assert "structural policy v11" in doc
                assert "structural policy v10" not in doc
                assert "at least 64 requests" in doc
                assert "bank_kernel" in doc
        """
    )
    write(research_path, text)


def patch_exact_api_doc() -> None:
    path = "docs/EXACT_API.md"
    text = read(path)
    old = """The default `auto` mode uses only graph structure and request-bank length. It does
not inspect measured timings or cached answers. The result metadata names the selected engine. For edgeless graphs, `auto` uses a separately replayed direct-domain kernel and reports `bank_kernel=\"edgeless\"`; explicit modes retain their original implementations for controlled comparisons. Callers can still force `raw`, `scc`, `sparse`, or `route`.
"""
    new = """The default `auto` mode is the frozen `structural-v11` dispatcher. It uses only
immutable graph/SCC geometry, bank length, and pin counts; it never inspects
answers, measured timings, or cached results. Long banks normally use SCC/tile-64,
but remain on raw when SCC construction leaves every literal separate and the
original graph is dense enough that the quotient adds work. Short, contradictory,
and dense proof-heavy shapes retain their frozen raw rules. The result metadata
reports `auto_policy`, `selected_mode`, and `auto_policy_reason`.

For edgeless graphs, `auto` can execute a separately replayed direct-domain kernel
and reports `bank_kernel=\"edgeless\"`. Current restrictions that directly force an
original implication conflict can emit a minimal portable kind-2 proof without a
full closure replay. These are implementation paths under the same checked answer
contract; callers can still force `raw`, `scc`, `sparse`, or `route` for controlled
comparisons.
"""
    text = replace_once(text, old, new, label="EXACT_API auto policy")
    old = """The current performance evidence uses exposed synthetic development graphs and is
not an independent application benchmark. The installed API does not inherit the
frozen WAP result, create a general SAT claim, or establish algorithmic novelty.
"""
    new = """The current exact-service evidence includes a prospectively frozen mixed direct
2-SAT holdout and disclosed development comparisons against the strongest of three
project-authored direct engines and the official CaDiCaL 3.0.1 core. Those results
are narrow repeated binary-list service comparisons, not ordinary SAT Competition
or graph-colouring benchmarks. The installed API does not inherit the frozen WAP
claim, create general SAT leadership, establish algorithmic novelty, or substitute
for outside-team replication and operator-authored workloads.
"""
    text = replace_once(text, old, new, label="EXACT_API evidence boundary")
    write(path, text)


def write_part3_status() -> None:
    write(
        "docs/PART3_RELEASE_REPLICATION_READINESS.md",
        dedent(
            """\
            # Part 3 — exact-service competition, release and replication readiness

            **Started:** 2026-10-19  
            **Working branch:** `finish-part3`  
            **Base:** merged and tagged Part 2 commit `527590406c2c624a6aa050e17c6f45b0f000ac5b`  
            **Review:** draft pull request #76

            ## Status

            Part 3 is **in progress**. It has not been merged, tagged as complete, or
            published as a package release. The branch combines release-facing evidence
            tooling with the current structural-v11 exact runtime and the retained
            competition studies. Passing project-owned CI is necessary engineering
            evidence, not outside replication or a broad state-of-the-art claim.

            Earlier Part 3 branches were archived and consolidated into this one working
            branch. A later cleanup accidentally removed the newest competition files;
            that source was restored before the permanent matrix was pointed at it. The
            draft PR records the recovery commits and exact current head.

            ## Release-facing commands

            ```bash
            spectra exact selftest exact-selftest
            spectra exact audit exact-selftest
            spectra exact compare first-bundle second-bundle
            ```

            `selftest` explicitly builds the bundled native runtime, exercises raw and SCC
            automatic paths, SAT, and all three portable `SBP1` contradiction kinds,
            independently replays every `SBC1 -> SCQ2` bank, rejects tampering, retains the
            exact C++ source, and writes a portable manifest.

            `audit` runs in a fresh process without executing retained native code. It
            verifies the frozen inventory, build receipt, source and binary hashes,
            packets, complete labels, contradiction proofs, selector receipts, and
            mutation checks. `compare` audits multiple bundles and permits only declared
            environment differences while requiring one shared semantic result.

            A successful self-test directory contains ten manifest-bound files plus
            `SHA256.json`: the semantic receipt, retained native source, build receipt,
            platform library, three request packets, and three certified output banks.
            Failed runs leave partial evidence and must be rerun into a new directory.

            ## Current runtime identity

            The one-call public dispatcher is `structural-v11`. Its decision uses only
            immutable graph/SCC geometry, bank length, and pin counts. It keeps long dense
            uncompressed banks on raw when a quotient would remove no literals, otherwise
            uses the frozen raw/SCC rules inherited from Part 2. Edgeless problems may use
            a separately replayed direct-domain kernel. Direct forced conflicts can emit a
            minimal original-literal proof without replaying a complete closure.

            Explicit `raw`, `scc`, `sparse`, and `route` modes remain available as
            controls. Public and research native C++ sources must remain byte-identical.

            ## Retained competition evidence

            ### Prospective direct 2-SAT holdout

            Four graph identities, a deterministic answer-free bank with exactly 512 SAT
            and 512 UNSAT rows, six measured arms, schedules, and numerical gates were
            frozen before the graph bytes were acquired. SPECTRA `auto` passed on x86-64
            and native ARM64 and won all 32 measured architecture/graph/bank cells. At
            1,024 requests it was about 2.19× faster on x86-64 and 2.50× faster on ARM64
            than the fastest median of the independently implemented closure, scalar-DAG,
            and tile-64 challenger engines.

            The original structural-v10 holdout also exposed a negative result: explicit
            raw was faster than public auto on every 1,024-row graph. That opened evidence
            motivated structural-v11, so the later replay is development evidence rather
            than a second prospective confirmation.

            ### Structural-v11 replay

            On the already opened mixed banks, structural-v11 retained complete
            proof-carrying output and won all 32 cells. Public auto was about 4.45× faster
            on x86-64 and 4.33× faster on ARM64 than the fastest project-authored direct
            engine at 1,024 requests. The replay validates the implementation opportunity;
            it does not reset the holdout boundary.

            ### Official CaDiCaL 3.0.1 comparison

            The official unmodified CaDiCaL 3.0.1 core was pinned by commit and source
            manifest. Under the disclosed matched first-use proof-carrying contract,
            SPECTRA won all 32 measured cells. Geometric-mean speedups ranged from about
            9.26× at one request to 25.82× at 1,024 requests depending on architecture.

            CaDiCaL is a general SAT solver and SPECTRA is specialized for fixed
            one/two-colour graph domains. The adapter, proof translation, and benchmark
            harness are project-authored, and the workload was already open. This result
            therefore does not establish general SAT leadership or outside implementation
            superiority.

            ## Fresh-transfer result: retained failure

            The six-family transfer protocol was frozen before opening a new graph set.
            Graph-free validation passed, but the first candidate, `DSJC500.5.col`,
            exceeded the predeclared 63-colour deterministic-DSATUR admission limit. The
            correct verdict is `FAIL_ADMISSION` on both architectures:

            - no solver arm was timed;
            - no performance cell was produced;
            - no graph was replaced;
            - no threshold was relaxed; and
            - all six acquired candidates are now exposed development inputs.

            A stronger deterministic witness-colouring procedure may be developed on that
            exposed suite, but a future prospective transfer must use a genuinely new
            inventory frozen only after the replacement generator passes its complete
            graph-free tests.

            ## Merge and release gates

            Part 3 may be considered merge-ready only when one exact final source identity
            passes all of the following:

            1. active-tree, release, packaging, and workspace audits;
            2. public structural-v11, edgeless, packet, proof, self-test, audit, compare,
               exam-repair, and ITC contracts;
            3. isolated installed-wheel execution on Linux Python 3.11/3.13, native ARM64,
               and Windows/MSVC;
            4. fresh-process no-native auditing and cross-platform bundle comparison;
            5. UBSan and ASan+UBSan for the exact runtime and retained challenger adapters;
            6. direct 2-SAT and official CaDiCaL contracts on x86-64 and ARM64;
            7. byte equality between public and research native sources; and
            8. preservation of every failed admission, negative cell, and claim boundary.

            A package release is a separate decision. The published version remains
            `0.7.1` until an intentional release commit updates both version fields and
            final wheel/sdist assets are downloaded and verified.

            ## Gates that project-owned CI cannot award

            Part 3 remains scientifically incomplete without attributable external work:

            - outside-team reproduction with the complete raw bundle;
            - an independently written adapter or challenger;
            - operator-authored restrictions and acceptance metrics;
            - maintained use outside this repository;
            - expert closest-work review; and
            - production security, reliability, and deployment evidence.

            These open gates are not reasons to weaken the engineering contract, and they
            cannot be converted into completion by adding more project-owned test runs.
            """
        ),
    )


def patch_current_status() -> None:
    path = "docs/STATUS.md"
    text = read(path)
    marker = "## Flagship research result\n"
    section = dedent(
        """\
        ## Part 3 draft branch

        Draft pull request #76 on `finish-part3` contains the unreleased exact-service
        competition and release-readiness work. It is not part of `main` or the published
        v0.7.1 package.

        The draft currently includes structural-v11, the edgeless direct-domain kernel,
        release-facing self-test/audit/compare commands, a separately implemented direct
        2-SAT challenger, and an adapter around the official CaDiCaL 3.0.1 core. Retained
        evidence includes:

        - a prospectively frozen mixed direct-2-SAT holdout that passed on x86-64 and
          ARM64, with public `auto` winning all 32 cells and full-bank gains of about
          2.19× and 2.50× over the fastest of three direct engines;
        - an opened structural-v11 replay with about 4.45× and 4.33× full-bank gains over
          the same project-authored challenger; and
        - a disclosed official-CaDiCaL comparison with all 32 cells won and geometric
          gains from roughly 9.26× to 25.82× depending on bank size and architecture.

        These are narrow repeated binary-list exact-service results. They do not establish
        general SAT or graph-colouring leadership, asymptotic novelty, outside-team
        replication, operator value, or production readiness.

        The newest six-family prospective transfer is a retained `FAIL_ADMISSION`, not a
        speed result: `DSJC500.5.col` exceeded the frozen deterministic-DSATUR 63-colour
        limit before any solver was timed. All six candidates are now exposed and cannot
        be reused as unopened confirmation. Part 3 remains in progress while the stronger
        generator, a genuinely new transfer inventory, external replication, and release
        gates remain open.

        """
    )
    if "## Part 3 draft branch" in text:
        raise AssertionError("Part 3 status section already exists")
    text = replace_once(text, marker, section + marker, label="STATUS Part 3 section")
    write(path, text)


def patch_changelog() -> None:
    path = "CHANGELOG.md"
    text = read(path)
    anchor = """- Retain complete installed-wheel bundles from Linux 3.11, Linux 3.13, ARM64 and
  Windows, then compare them in one fresh no-native CI job and publish the semantic
  comparison receipt as a separate artifact.
"""
    additions = anchor + """- Advance the one-call dispatcher to `structural-v11`, retain dense long
  uncompressed banks on raw, and add independently replayed edgeless and direct
  forced-conflict paths without changing the proof-carrying answer contract.
- Restore the prospective mixed direct-2-SAT holdout, the official CaDiCaL 3.0.1
  matched comparison, and their complete claim boundaries after a consolidation
  cleanup accidentally removed those active source files.
- Retain the six-family fresh-transfer opening as `FAIL_ADMISSION`: no solver was
  timed after the first graph exceeded the frozen 63-colour witness limit, no graph
  was replaced, and the exposed candidates cannot be reused prospectively.
"""
    text = replace_once(text, anchor, additions, label="CHANGELOG Part 3 evidence")
    text = replace_once(
        text,
        "fail-closed active-tree budget (14 MiB, 950 files, 2 MiB per file)",
        "fail-closed active-tree budget (14 MiB, 1,000 files, 2 MiB per file)",
        label="CHANGELOG active-tree ceiling",
    )
    write(path, text)


def patch_exact_workflow() -> None:
    path = ".github/workflows/exact-public-api.yml"
    text = read(path)
    text = replace_once(
        text,
        "  cancel-in-progress: false\n",
        "  cancel-in-progress: true\n",
        label="exact workflow concurrency",
    )
    text = replace_once(
        text,
        "      - 'tests/public/test_exact_compare.py'\n      - 'tests/public/test_exam_repair.py'\n",
        "      - 'tests/public/test_exact_compare.py'\n      - 'tests/public/test_auto_selector_v11.py'\n      - 'tests/public/test_exact_edgeless.py'\n      - 'tests/public/test_exam_repair.py'\n",
        label="exact workflow path filters",
    )
    text = replace_once(
        text,
        "            tests/public/test_exact_compare.py \\\n            tests/public/test_exam_repair.py \\\n",
        "            tests/public/test_exact_compare.py \\\n            tests/public/test_auto_selector_v11.py \\\n            tests/public/test_exact_edgeless.py \\\n            tests/public/test_exam_repair.py \\\n",
        label="Linux public selector tests",
    )
    text = replace_once(
        text,
        "python -m pytest tests/public/test_exact_api.py tests/public/test_exact_selftest.py tests/public/test_exact_compare.py tests/public/test_exam_repair.py tests/public/test_itc2007_exam.py --confcutdir=tests/public",
        "python -m pytest tests/public/test_exact_api.py tests/public/test_exact_selftest.py tests/public/test_exact_compare.py tests/public/test_auto_selector_v11.py tests/public/test_exact_edgeless.py tests/public/test_exam_repair.py tests/public/test_itc2007_exam.py --confcutdir=tests/public",
        label="Windows public selector tests",
    )
    text = replace_once(
        text,
        "          python -m pytest \\\n            tests/public/test_exact_api.py \\\n            --confcutdir=tests/public -q \\\n",
        "          python -m pytest \\\n            tests/public/test_exact_api.py \\\n            tests/public/test_auto_selector_v11.py \\\n            tests/public/test_exact_edgeless.py \\\n            --confcutdir=tests/public -q \\\n",
        label="sanitizer selector tests",
    )
    write(path, text)


def cleanup_bootstrap() -> None:
    for path in (
        ".github/scripts/sync_part3_status.py",
        ".github/workflows/sync-part3-status.yml",
    ):
        (ROOT / path).unlink()


def main() -> None:
    patch_selector_docstrings()
    patch_selector_tests()
    patch_exact_api_doc()
    write_part3_status()
    patch_current_status()
    patch_changelog()
    patch_exact_workflow()
    cleanup_bootstrap()


if __name__ == "__main__":
    main()
