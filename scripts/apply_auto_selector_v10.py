from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def replace_markers(path: str) -> None:
    target = Path(path)
    text = target.read_text()
    text = text.replace("structural-v9", "structural-v10")
    text = text.replace("Structural auto policy v7", "Structural auto policy v10")
    text = text.replace("structural policy v7", "structural policy v10")
    text = text.replace("structural policy v8", "structural policy v10")
    target.write_text(text)


def replace_block(path: str, start_marker: str, end_marker: str, replacement: str) -> None:
    target = Path(path)
    text = target.read_text()
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    target.write_text(text[:start] + replacement + text[end:])


def main() -> None:
    # Auto dispatch belongs in the native endpoint.  This removes the measurable
    # Python-only tax from raw selections without changing validation or raw.
    for path in ("spectra/exact.py", "experiments/compiled_query/exact.py"):
        start = "        native_mode = {'raw': 0, 'scc': 1, 'sparse': 2, 'route': 3, 'auto': 4}[mode]\n"
        end = "        info = {**info, 'backend': 'native-exact-binary', 'mode': mode,\n"
        replacement = (
            start
            + "        values = self._module.solve_session_packet(packet, native_mode, max_bytes)\n"
            + "        output, count, sat, unsat, proof_bytes, visits, arcs, info = values\n"
        )
        replace_block(path, start, end, replacement)

    policy = dedent(
        """\
                if(mode==4) {
                    // Structural auto policy v10. Raw handles tiny, contradictory,
                    // proof-heavy small-compressed, and one measured tile64 warmup
                    // shape. Other feasible long banks use the SCC tile64 engine.
                    const bool high_pin=bank_size
                            && W(bank_pins)>=8*W(bank_size)
                            && W(24)*W(bank_pins)>=W(n)*W(bank_size);
                    if(bank_size<=8 && n>512) {
                        condensed=false;routed=false;auto_reason="large_tiny_bank";
                    } else if(bad!=NIL) {
                        condensed=false;routed=false;auto_reason="base_infeasible";
                    } else if(high_pin && bank_size<64) {
                        condensed=false;routed=false;auto_reason="dense_pin_short";
                    } else if(high_pin && n<96 && W(nc)<2*W(n)) {
                        condensed=false;routed=false;auto_reason="dense_small_compressed";
                    } else if(bank_size==64 && W(nc)==2*W(n)
                            && W(p.edges.size())>=4*W(n)) {
                        condensed=false;routed=false;auto_reason="dense_uncompressed_tile64";
                    } else if(bank_size>=64) {
                        condensed=true;routed=false;auto_reason="scc_long_bank";
                    } else if(bank_size>=8 && W(2)*W(nc)<=W(n)) {
                        condensed=true;routed=false;auto_reason="scc_strong_compression";
                    } else {
                        condensed=false;routed=false;auto_reason="short_weak_compression";
                    }
                }
        """
    )
    for path in (
        "spectra/_native/exact_binary.cpp",
        "experiments/compiled_query/exact_native.cpp",
    ):
        replace_block(
            path,
            "        if(mode==4) {\n",
            "        if(condensed && bad==NIL) {\n",
            policy,
        )

    benchmark = Path("experiments/compiled_query/auto_selector_benchmark.py")
    text = benchmark.read_text()
    text = text.replace("structural auto policy v7", "structural auto policy v10")
    text = text.replace(
        "spectra.auto_selector_benchmark.v6",
        "spectra.auto_selector_benchmark.v7",
    )
    text = text.replace('POLICY = "structural-v9"', 'POLICY = "structural-v10"')
    text = text.replace("auto-v9-", "auto-v10-")
    text = text.replace(
        "SPECTRA-auto-selector-v9-confirmation",
        "SPECTRA-auto-selector-v10-confirmation",
    )
    text = text.replace(
        "SPECTRA-auto-selector-v7-schedule",
        "SPECTRA-auto-selector-v10-schedule",
    )
    old_regimes = dedent(
        """\
            regimes = (
                ("strong-sparse", 336, 3, True, False, 0.0),
                ("dense-sub96-small", 80, 12, False, False, 0.0),
                ("dense-sub96-boundary", 95, 12, False, False, 0.0),
                ("dense-at96", 96, 12, False, False, 0.0),
                ("dense-above96", 112, 12, False, False, 0.0),
                ("dense-medium", 176, 12, False, False, 0.0),
                ("edge-heavy-warmup", 736, 12, False, False, 5.0),
                ("base-infeasible", 432, 4, False, True, 0.0),
            )
        """
    )
    new_regimes = dedent(
        """\
            regimes = (
                ("strong-sparse", 352, 3, True, False, 0.0),
                ("dense-small-uncompressed", 84, 12, False, False, 0.0),
                ("dense-boundary-uncompressed", 92, 12, False, False, 0.0),
                ("dense-at96", 96, 12, False, False, 0.0),
                ("dense-above96", 108, 12, False, False, 0.0),
                ("dense-medium", 184, 12, False, False, 0.0),
                ("edge-heavy-warmup", 752, 12, False, False, 5.0),
                ("base-infeasible", 440, 4, False, True, 0.0),
            )
        """
    )
    if text.count(old_regimes) != 1:
        raise RuntimeError("v9 confirmation regimes differ")
    text = text.replace(old_regimes, new_regimes)
    start = text.index("def expected_policy(")
    end = text.index("\n\ndef run(", start)
    expected = dedent(
        """\
        def expected_policy(*, base: bool, pins: int, count: int, vertices: int,
                            edges: int, components: int) -> tuple[str, str]:
            high_pin = bool(count and pins >= 8 * count
                            and 24 * pins >= vertices * count)
            if count <= 8 and vertices > 512:
                return "raw", "large_tiny_bank"
            if not base:
                return "raw", "base_infeasible"
            if high_pin and count < 64:
                return "raw", "dense_pin_short"
            if high_pin and vertices < 96 and components < 2 * vertices:
                return "raw", "dense_small_compressed"
            if count == 64 and components == 2 * vertices and edges >= 4 * vertices:
                return "raw", "dense_uncompressed_tile64"
            if count >= 64:
                return "scc", "scc_long_bank"
            if count >= 8 and 2 * components <= vertices:
                return "scc", "scc_strong_compression"
            return "raw", "short_weak_compression"
        """
    ).rstrip()
    benchmark.write_text(text[:start] + expected + text[end:])

    old_test = Path("tests/public/test_auto_selector_v9.py")
    new_test = Path("tests/public/test_auto_selector_v10.py")
    text = old_test.read_text()
    text = text.replace("selector-v9-runtime", "selector-v10-runtime")
    text = text.replace("structural-v9", "structural-v10")
    text = text.replace(
        "test_large_tiny_bank_uses_the_direct_raw_path",
        "test_large_tiny_bank_uses_the_native_raw_path",
    )
    old_dense = dedent(
        """\
        def test_dense_long_bank_below_96_vertices_stays_raw(exact_runtime):
            problem = independent_problem(95)
            row = tuple((vertex, 3) for vertex in range(8))
            answer = solve(exact_runtime, problem, (row,) * 1024)
            assert answer.info["selected_mode"] == "raw"
            assert answer.info["auto_policy_reason"] == "dense_sub96_high_pin"
        """
    )
    new_dense = dedent(
        """\
        def test_dense_long_uncompressed_bank_below_96_uses_scc(exact_runtime):
            problem = independent_problem(95)
            row = tuple((vertex, 3) for vertex in range(8))
            answer = solve(exact_runtime, problem, (row,) * 1024)
            assert answer.info["components"] == 2 * problem.n
            assert answer.info["selected_mode"] == "scc"
            assert answer.info["auto_policy_reason"] == "scc_long_bank"


        def test_dense_small_compressed_bank_stays_raw(exact_runtime):
            problem = path_problem(78)
            row = tuple((vertex, 3) for vertex in range(8))
            answer = solve(exact_runtime, problem, (row,) * 1024)
            assert answer.info["components"] < 2 * problem.n
            assert answer.info["selected_mode"] == "raw"
            assert answer.info["auto_policy_reason"] == "dense_small_compressed"
        """
    )
    if text.count(old_dense) != 1:
        raise RuntimeError("v9 dense public selector test differs")
    new_test.write_text(text.replace(old_dense, new_dense))
    old_test.unlink()

    exact_tests = Path("experiments/compiled_query/test_exact.py")
    text = exact_tests.read_text().replace("structural-v9", "structural-v10")
    text = text.replace(
        "# The bank averages well above eight pins per request, so structural-v10\n"
        "    # intentionally selects raw rather than paying for tile64 propagation.",
        "# This small compressed bank is proof-heavy, so structural-v10 keeps raw.",
    )
    text = text.replace("'dense_sub96_high_pin'", "'dense_small_compressed'", 1)
    start = text.index("def test_auto_structural_v2_freezes_the_eight_pin_density_boundary")
    end = text.index("\n\n@pytest.mark.parametrize('count'", start)
    boundary_test = dedent(
        """\
        def test_auto_distinguishes_small_compressed_and_uncompressed_dense_banks(exact_runtime):
            independent = BinaryListProblem.create(16, 2, (), (3,) * 16)
            path = BinaryListProblem.create(
                16, 2, tuple((vertex, vertex + 1) for vertex in range(15)), (3,) * 16
            )
            rows = tuple(tuple((vertex, 3) for vertex in range(8)) for _ in range(64))
            uncompressed = exact_runtime.solve_session_packet(
                pack_session_packet(independent, rows), mode='auto')
            compressed = exact_runtime.solve_session_packet(
                pack_session_packet(path, rows), mode='auto')
            assert uncompressed.info['components'] == 2 * independent.n
            assert uncompressed.info['selected_mode'] == 'scc'
            assert uncompressed.info['auto_policy_reason'] == 'scc_long_bank'
            assert compressed.info['components'] < 2 * path.n
            assert compressed.info['selected_mode'] == 'raw'
            assert compressed.info['auto_policy'] == 'structural-v10'
            assert compressed.info['auto_policy_reason'] == 'dense_small_compressed'
        """
    ).rstrip()
    text = text[:start] + boundary_test + text[end:]
    old_policy = dedent(
        """\
            if structure == 'odd_cycle':
                selected, reason = 'raw', 'base_infeasible'
            elif (pins >= 8 * count and 24 * pins >= n * count
                  and (count < 64 or n < 96)):
                selected = 'raw'
                reason = 'dense_sub96_high_pin' if n < 96 else 'dense_pin_short'
            else:
                components = {'empty': 0, 'independent': 18, 'chain': 2}[structure]
                if count == 64 and components == 2 * n and len(edges) >= 4 * n:
                    selected, reason = 'raw', 'dense_uncompressed_tile64'
                elif count >= 64:
                    selected, reason = 'scc', 'scc_long_bank'
                elif count >= 8 and 2 * components <= n:
                    selected, reason = 'scc', 'scc_strong_compression'
                else:
                    selected, reason = 'raw', 'short_weak_compression'
        """
    )
    new_policy = dedent(
        """\
            if structure == 'odd_cycle':
                selected, reason = 'raw', 'base_infeasible'
            else:
                components = {'empty': 0, 'independent': 18, 'chain': 2}[structure]
                high_pin = bool(count and pins >= 8 * count
                                and 24 * pins >= n * count)
                if high_pin and count < 64:
                    selected, reason = 'raw', 'dense_pin_short'
                elif high_pin and n < 96 and components < 2 * n:
                    selected, reason = 'raw', 'dense_small_compressed'
                elif count == 64 and components == 2 * n and len(edges) >= 4 * n:
                    selected, reason = 'raw', 'dense_uncompressed_tile64'
                elif count >= 64:
                    selected, reason = 'scc', 'scc_long_bank'
                elif count >= 8 and 2 * components <= n:
                    selected, reason = 'scc', 'scc_strong_compression'
                else:
                    selected, reason = 'raw', 'short_weak_compression'
        """
    )
    if text.count(old_policy) != 1:
        raise RuntimeError("exact packet reference policy differs")
    exact_tests.write_text(text.replace(old_policy, new_policy))

    for path in (
        "spectra/exact.py",
        "experiments/compiled_query/exact.py",
        "experiments/compiled_query/itc_exam_suite.py",
        "experiments/compiled_query/itc_exam_transfer.py",
        "experiments/compiled_query/test_itc_exam_transfer.py",
        "experiments/compiled_query/auto_selector_wap_guard.py",
        "experiments/compiled_query/auto_selector_wap_overhead.py",
        "tests/public/test_exact_api.py",
    ):
        replace_markers(path)

    for path in (
        "experiments/compiled_query/itc_exam_suite.py",
        "experiments/compiled_query/itc_exam_transfer.py",
        "experiments/compiled_query/test_itc_exam_transfer.py",
    ):
        target = Path(path)
        target.write_text(
            target.read_text().replace("dense_sub96_high_pin", "dense_small_compressed")
        )

    protocol = dedent(
        """\
        # Automatic exact-engine selector v10 — frozen protocol

        ## Question

        Can a query-blind structural dispatcher stay within five percent of the fastest
        explicit exact engine on at least 95% of a fresh confirmation panel, while
        preserving the established WAP SCC win and the ITC raw fallback?

        ## Frozen correction

        The v9 panel is now development evidence. It identified two causes rather than a
        random failure: Python-side raw dispatch added measurable overhead to tiny calls,
        and every small dense bank was forced to raw even when it had no SCC compression.

        V10 therefore makes exactly two executable changes:

        1. `auto` always enters native mode 4; Python no longer duplicates dispatch or
           rewrites selector metadata.
        2. Dense high-pin banks below 96 vertices remain raw only when they are actually
           compressed (`components < 2 * vertices`). Uncompressed long banks may use SCC.

        All other structural boundaries, exact formats, decisions, witnesses, proofs,
        resource limits and explicit engines remain unchanged.

        ## Fresh confirmation

        The confirmation namespace, generated cases and timing schedule are new. The
        panel contains eight query-blind regimes across all eight established topology
        families and request counts 1, 8, 32, 64, 128 and 1,024. It is project-owned
        prospective confirmation, not outside replication.

        The unchanged confirmation gates are:

        - global realized-regret geometric mean <= 1.04;
        - selected-engine regret geometric mean <= 1.025;
        - every family/count group geometric mean <= 1.06;
        - at least 95% of cells <= 1.05 regret;
        - worst cell <= 1.10;
        - every family auto/raw geometric mean <= 1.03;
        - zero structural-rule, output, replay or decision failures.

        Separate frozen guards must also pass on x86-64 and ARM64:

        - WAP long banks select SCC and retain the Issue 1 advantage;
        - ITC set9/set10/set12 preserve the correct engine choices;
        - automatic dispatch overhead remains within the existing paired limits;
        - full public, native, Windows, sanitizer, package and regression matrices pass.

        No threshold or case may be changed after the v10 timing run begins.
        """
    )
    Path("experiments/compiled_query/AUTO_SELECTOR_V10_PROTOCOL_20261018.md").write_text(protocol)


if __name__ == "__main__":
    main()
