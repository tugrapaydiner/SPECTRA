from __future__ import annotations

from pathlib import Path


def replace(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = Path(path)
    text = target.read_text()
    found = text.count(old)
    if found != count:
        raise RuntimeError(
            f"{path}: expected {count} occurrences, found {found}: {old!r}"
        )
    target.write_text(text.replace(old, new))


def main() -> None:
    benchmark = "experiments/compiled_query/auto_selector_benchmark.py"
    replace(
        benchmark,
        'SCHEMA = "spectra.auto_selector_benchmark.v5"\nPOLICY = "structural-v8"',
        'SCHEMA = "spectra.auto_selector_benchmark.v6"\nPOLICY = "structural-v9"',
    )
    replace(
        benchmark,
        'f"auto-v8-{family}-{regime}-n{n}"',
        'f"auto-v9-{family}-{regime}-n{n}"',
    )
    replace(
        benchmark,
        'namespace = "SPECTRA-auto-selector-v8-confirmation"',
        'namespace = "SPECTRA-auto-selector-v9-confirmation"',
    )
    replace(
        benchmark,
        '''    regimes = (
        ("strong-sparse", 320, 3, True, False, 0.0),
        ("dense-small", 96, 12, False, False, 0.0),
        ("dense-boundary-low", 120, 12, False, False, 0.0),
        ("dense-boundary-high", 128, 12, False, False, 0.0),
        ("dense-medium", 192, 12, False, False, 0.0),
        ("edge-heavy-warmup", 768, 12, False, False, 5.0),
        ("base-infeasible", 448, 4, False, True, 0.0),
    )''',
        '''    regimes = (
        ("strong-sparse", 336, 3, True, False, 0.0),
        ("dense-sub96-small", 80, 12, False, False, 0.0),
        ("dense-sub96-boundary", 95, 12, False, False, 0.0),
        ("dense-at96", 96, 12, False, False, 0.0),
        ("dense-above96", 112, 12, False, False, 0.0),
        ("dense-medium", 176, 12, False, False, 0.0),
        ("edge-heavy-warmup", 736, 12, False, False, 5.0),
        ("base-infeasible", 432, 4, False, True, 0.0),
    )''',
    )
    replace(
        benchmark,
        '''    if (pins >= 8 * count and 24 * pins >= vertices * count
            and (count < 64 or vertices < 128)):
        return "raw", "dense_tiny_high_pin" if vertices < 128 else "dense_pin_short"''',
        '''    if (pins >= 8 * count and 24 * pins >= vertices * count
            and (count < 64 or vertices < 96)):
        return "raw", "dense_sub96_high_pin" if vertices < 96 else "dense_pin_short"''',
    )

    for path in (
        "experiments/compiled_query/auto_selector_wap_guard.py",
        "experiments/compiled_query/auto_selector_wap_overhead.py",
    ):
        replace(path, 'POLICY = "structural-v8"', 'POLICY = "structural-v9"')

    exact = "spectra/exact.py"
    replace(
        exact,
        "The default ``auto`` mode uses structural policy v7.",
        "The default ``auto`` mode uses structural policy v9.",
    )
    replace(
        exact,
        "stays raw before 64 requests and at every length below 256 vertices.",
        "stays raw before 64 requests and at every length below 96 vertices.",
    )
    replace(
        exact,
        '''            magic, vertices, _palette, _edges, rows, _pins, _r0, _r1 = (
                SESSION_PACKET_HEADER.unpack_from(packet)
            )
            if magic == SESSION_PACKET_MAGIC and rows <= 8 and vertices > 512:
                native_mode = 0
                direct_reason = 'large_tiny_bank' ''',
        '''            magic, vertices, _palette, _edges, rows, pins, _r0, _r1 = (
                SESSION_PACKET_HEADER.unpack_from(packet)
            )
            if magic == SESSION_PACKET_MAGIC:
                if rows <= 8 and vertices > 512:
                    native_mode = 0
                    direct_reason = 'large_tiny_bank'
                elif (rows and pins >= 8 * rows
                      and 24 * pins >= vertices * rows
                      and (rows < 64 or vertices < 96)):
                    native_mode = 0
                    direct_reason = (
                        'dense_sub96_high_pin' if vertices < 96
                        else 'dense_pin_short'
                    )''',
    )
    replace(
        exact,
        "'auto_policy': 'structural-v8'",
        "'auto_policy': 'structural-v9'",
    )

    for path in (
        "spectra/_native/exact_binary.cpp",
        "experiments/compiled_query/exact_native.cpp",
    ):
        replace(
            path,
            "&& (bank_size<64 || n<128))",
            "&& (bank_size<64 || n<96))",
        )
        replace(
            path,
            'auto_reason=n<128?"dense_tiny_high_pin":"dense_pin_short";',
            'auto_reason=n<96?"dense_sub96_high_pin":"dense_pin_short";',
        )
        replace(
            path,
            'which==4?"structural-v8":"explicit"',
            'which==4?"structural-v9":"explicit"',
        )

    suite = Path("experiments/compiled_query/itc_exam_suite.py")
    text = suite.read_text()
    text = text.replace("structural-v8 selector", "structural-v9 selector")
    text = text.replace(
        '"selector_policy": "structural-v8"',
        '"selector_policy": "structural-v9"',
    )
    text = text.replace(
        '"structural-v7", "structural-v8")',
        '"structural-v7", "structural-v8", "structural-v9")',
    )
    suite.write_text(text)

    old_test = Path("tests/public/test_auto_selector_v8.py")
    new_test = Path("tests/public/test_auto_selector_v9.py")
    old_test.rename(new_test)
    text = new_test.read_text()
    text = text.replace("selector-v8-runtime", "selector-v9-runtime")
    text = text.replace("structural-v8", "structural-v9")
    replace_block = '''def test_dense_long_bank_below_128_vertices_stays_raw(exact_runtime):
    problem = independent_problem(120)
    row = tuple((vertex, 3) for vertex in range(8))
    answer = solve(exact_runtime, problem, (row,) * 1024)
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "dense_tiny_high_pin"


def test_dense_long_bank_switches_to_scc_at_128_vertices(exact_runtime):
    problem = independent_problem(128)
    row = tuple((vertex, 3) for vertex in range(8))
    answer = solve(exact_runtime, problem, (row,) * 64)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_long_bank"'''
    replacement = '''def test_dense_long_bank_below_96_vertices_stays_raw(exact_runtime):
    problem = independent_problem(95)
    row = tuple((vertex, 3) for vertex in range(8))
    answer = solve(exact_runtime, problem, (row,) * 1024)
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "dense_sub96_high_pin"


def test_dense_long_bank_switches_to_scc_at_96_vertices(exact_runtime):
    problem = independent_problem(96)
    row = tuple((vertex, 3) for vertex in range(8))
    answer = solve(exact_runtime, problem, (row,) * 64)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_long_bank"


def test_dense_long_bank_at_120_vertices_uses_scc(exact_runtime):
    problem = independent_problem(120)
    row = tuple((vertex, 3) for vertex in range(8))
    answer = solve(exact_runtime, problem, (row,) * 1024)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_long_bank"'''
    if text.count(replace_block) != 1:
        raise RuntimeError("dense-boundary selector tests differ")
    new_test.write_text(text.replace(replace_block, replacement))

    Path("experiments/compiled_query/AUTO_SELECTOR_V9_PROTOCOL_20261018.md").write_text(
        """# Automatic selector v9 protocol — 18 October 2026

## Status

The v8 prospective panel is now development evidence because its failures are
used here. The policy, thresholds, fresh namespace, and v9 confirmation panel
below are frozen before v9 timing.

## Correction

V8 kept every dense high-pin bank below 128 vertices on raw. The retained v8
result showed a stable crossover instead: at 64 or more requests, the 96- and
120-vertex banks preferred SCC across all eight graph families on x86-64 and
ARM64, while the fixed ITC set12 holdout at 78 vertices still preferred raw.

V9 therefore keeps dense high-pin work on raw when the bank has fewer than 64
requests or the graph has fewer than 96 vertices. Dense banks at 96 vertices or
more use the ordinary SCC long-bank rule. Obvious structural raw decisions are
also dispatched before native SCC construction, without consulting answers or
measured timings.

## Fresh confirmation panel

The new namespace is `SPECTRA-auto-selector-v9-confirmation`. It uses the same
eight graph generators with new seeds and these regimes: strong sparse n=336;
dense high-pin n=80, n=95, n=96, n=112, and n=176; edge-heavy n=736; and base
infeasible n=432.

Each case is tested at 1, 8, 32, 64, 128, and 1,024 requests with 31 paired
repetitions on CPython 3.13 x86-64 and native ARM64.

## Unchanged gates

- at least 95% of cells within 5% of the measured oracle;
- selection-regret geomean <= 1.025;
- realized-regret geomean <= 1.04;
- worst group regret <= 1.06;
- worst cell <= 1.10;
- no graph family slower than explicit raw by more than 3%; and
- every structural-rule assertion passes.

The existing WAP long-bank, WAP overhead, and ITC set9/set10/set12 guards must
also pass unchanged. Set12 must remain raw; set9 and set10 must remain SCC.

This is project-owned prospective confirmation, not outside replication or a
universal solver claim.
"""
    )


if __name__ == "__main__":
    main()
