from __future__ import annotations

from pathlib import Path
import subprocess


def replace(path: str, old: str, new: str, *, count: int | None = None) -> None:
    target = Path(path)
    text = target.read_text()
    found = text.count(old)
    if count is not None and found != count:
        raise RuntimeError(f"{path}: expected {count} occurrences, found {found}: {old!r}")
    if count is None and found == 0:
        raise RuntimeError(f"{path}: missing expected text: {old!r}")
    target.write_text(text.replace(old, new))


# Keep public and research native mirrors byte-identical.
for path in ("spectra/_native/exact_binary.cpp", "experiments/compiled_query/exact_native.cpp"):
    replace(
        path,
        "&& (bank_size<64 || n<256)) {",
        "&& (bank_size<64 || n<128)) {",
        count=1,
    )
    replace(
        path,
        'auto_reason=n<256?"dense_small_high_pin":"dense_pin_short";',
        'auto_reason=n<128?"dense_tiny_high_pin":"dense_pin_short";',
        count=1,
    )

# Promote executable policy metadata to v8 while preserving historical documents.
matches = subprocess.check_output(
    ["git", "grep", "-l", "structural-v7", "--", "*.py", "*.cpp"],
    text=True,
).splitlines()
for path in matches:
    if path == "scripts/apply_auto_selector_v8.py":
        continue
    target = Path(path)
    target.write_text(target.read_text().replace("structural-v7", "structural-v8"))

# Fresh v8 benchmark panel and matching reference policy.
benchmark = Path("experiments/compiled_query/auto_selector_benchmark.py")
text = benchmark.read_text()
text = text.replace(
    'SCHEMA = "spectra.auto_selector_benchmark.v4"',
    'SCHEMA = "spectra.auto_selector_benchmark.v5"',
)
text = text.replace('name = f"auto-v7-', 'name = f"auto-v8-')
text = text.replace(
    "SPECTRA-auto-selector-v7-confirmation",
    "SPECTRA-auto-selector-v8-confirmation",
)
start = text.index("def confirmation_cases() -> tuple[PanelCase, ...]:")
end = text.index("\ndef panel_cases", start)
confirmation = '''def confirmation_cases() -> tuple[PanelCase, ...]:
    result: list[PanelCase] = []
    namespace = "SPECTRA-auto-selector-v8-confirmation"
    regimes = (
        ("strong-sparse", 320, 3, True, False, 0.0),
        ("dense-small", 96, 12, False, False, 0.0),
        ("dense-boundary-low", 120, 12, False, False, 0.0),
        ("dense-boundary-high", 128, 12, False, False, 0.0),
        ("dense-medium", 192, 12, False, False, 0.0),
        ("edge-heavy-warmup", 768, 12, False, False, 5.0),
        ("base-infeasible", 448, 4, False, True, 0.0),
    )
    for family in FAMILIES:
        for regime, n, width, strong, infeasible, edge_factor in regimes:
            case = generated_case(
                family,
                regime,
                n=n,
                width=width,
                strong_contraction=strong,
                base_infeasible=infeasible,
                namespace=namespace,
                minimum_edge_factor=edge_factor,
            )
            result.append(PanelCase(case, family, regime, CONFIRMATION_COUNTS))
    return tuple(result)
'''
text = text[:start] + confirmation + text[end:]
replace_policy = (
    'and (count < 64 or vertices < 256)):\n'
    '        return "raw", "dense_small_high_pin" if vertices < 256 else "dense_pin_short"'
)
new_policy = (
    'and (count < 64 or vertices < 128)):\n'
    '        return "raw", "dense_tiny_high_pin" if vertices < 128 else "dense_pin_short"'
)
if replace_policy not in text:
    raise RuntimeError("benchmark reference policy differs")
benchmark.write_text(text.replace(replace_policy, new_policy))

# Replace the old boundary contract with explicit tests around 128 vertices.
old_test = Path("tests/public/test_auto_selector_v7.py")
new_test = Path("tests/public/test_auto_selector_v8.py")
old_test.rename(new_test)
text = new_test.read_text().replace("selector-v7-runtime", "selector-v8-runtime")
start = text.index("def test_small_high_pin_bank_stays_raw_even_when_long")
end = text.index("\ndef test_large_high_pin_bank_uses_scc_at_tile64", start)
new_tests = '''def test_dense_long_bank_below_128_vertices_stays_raw(exact_runtime):
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
    assert answer.info["auto_policy_reason"] == "scc_long_bank"


def test_dense_medium_long_bank_uses_scc(exact_runtime):
    problem = independent_problem(192)
    row = tuple((vertex, 3) for vertex in range(8))
    answer = solve(exact_runtime, problem, (row,) * 1024)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_long_bank"

'''
new_test.write_text(text[:start] + new_tests + text[end + 1 :])

# Keep old evidence readable while admitting new v8 manifests.
suite = Path("experiments/compiled_query/itc_exam_suite.py")
text = suite.read_text()
text = text.replace(
    '"structural-v5", "structural-v6", "structural-v8")',
    '"structural-v5", "structural-v6", "structural-v7", "structural-v8")',
)

# Freeze the expected mode for each public ITC case without teaching the selector names.
text = text.replace(
    "    solution_path: str\n\n    def validate",
    "    solution_path: str\n    expected_mode: str | None = None\n\n    def validate",
)
text = text.replace(
    '        if self.role not in ("development", "holdout", "replication"):\n'
    '            raise ValueError("suite case role differs")',
    '        if self.role not in ("development", "holdout", "replication"):\n'
    '            raise ValueError("suite case role differs")\n'
    '        if self.expected_mode not in (None, "raw", "scc"):\n'
    '            raise ValueError("suite expected mode differs")',
)
for solution, mode in (
    ("InitMethods/SolStore/set9_OBSIFeas_0.sol", "scc"),
    ("InitMethods/SolStore/set10_OBSIFeas_0.sol", "scc"),
    ("InitMethods/SolStore/set12_OBSIFeas_0.sol", "raw"),
):
    old = f'        "{solution}",\n    ),'
    new = f'        "{solution}", "{mode}",\n    ),'
    if old not in text:
        raise RuntimeError(f"missing ITC case {solution}")
    text = text.replace(old, new)
text = text.replace(
    '                "solution_path": case.solution_path,\n'
    '                "manifest_sha256":',
    '                "solution_path": case.solution_path,\n'
    '                "expected_mode": case.expected_mode,\n'
    '                "manifest_sha256":',
)
text = text.replace(
    '        item.get("instance_path"), item.get("solution_path"),\n'
    '    ) for item in records if type(item) is dict)',
    '        item.get("instance_path"), item.get("solution_path"),\n'
    '        item.get("expected_mode"),\n'
    '    ) for item in records if type(item) is dict)',
)
old_gate = '''        auto_elapsed = child_run["modes"]["auto"]["elapsed_ns"]
        raw_elapsed = child_run["modes"]["raw"]["elapsed_ns"]
        paired_median = statistics.median(
            auto / raw for auto, raw in zip(auto_elapsed, raw_elapsed)
        )
        selected = result["modes"]["auto"]["selected_mode"]
        reason = result["modes"]["auto"]["auto_policy_reason"]
        gate_pass = True
        if item["role"] == "holdout" and threshold is not None:
            gate_pass = paired_median <= threshold
            if not gate_pass:
                holdout_failures.append(item["name"])
'''
new_gate = '''        auto_elapsed = child_run["modes"]["auto"]["elapsed_ns"]
        raw_elapsed = child_run["modes"]["raw"]["elapsed_ns"]
        scc_elapsed = child_run["modes"]["scc"]["elapsed_ns"]
        paired_median = statistics.median(
            auto / raw for auto, raw in zip(auto_elapsed, raw_elapsed)
        )
        auto_scc_paired = statistics.median(
            auto / scc for auto, scc in zip(auto_elapsed, scc_elapsed)
        )
        scc_raw_paired = statistics.median(
            scc / raw for scc, raw in zip(scc_elapsed, raw_elapsed)
        )
        selected = result["modes"]["auto"]["selected_mode"]
        reason = result["modes"]["auto"]["auto_policy_reason"]
        expected_mode = item.get("expected_mode")
        gate_pass = True
        if expected_mode == "scc":
            gate_pass = (
                selected == "scc"
                and auto_scc_paired <= 1.03
                and scc_raw_paired <= 0.98
            )
        elif expected_mode == "raw":
            gate_pass = selected == "raw" and paired_median <= (threshold or 1.05)
        elif item["role"] == "holdout" and threshold is not None:
            gate_pass = paired_median <= threshold
        if not gate_pass:
            holdout_failures.append(item["name"])
'''
if old_gate not in text:
    raise RuntimeError("ITC audit gate block differs")
text = text.replace(old_gate, new_gate)
text = text.replace(
    '            "auto_selected_mode": selected,\n'
    '            "auto_policy_reason": reason,\n'
    '            "auto_vs_raw_median_ratio": result["auto_vs_raw_median_ratio"],\n'
    '            "auto_vs_raw_paired_median_ratio": paired_median,',
    '            "expected_mode": expected_mode,\n'
    '            "auto_selected_mode": selected,\n'
    '            "auto_policy_reason": reason,\n'
    '            "auto_vs_raw_median_ratio": result["auto_vs_raw_median_ratio"],\n'
    '            "auto_vs_raw_paired_median_ratio": paired_median,\n'
    '            "auto_vs_scc_paired_median_ratio": auto_scc_paired,\n'
    '            "scc_vs_raw_paired_median_ratio": scc_raw_paired,',
)
text = text.replace(
    '        "holdout_gate_pass": not holdout_failures,\n'
    '        "holdout_failures": holdout_failures,',
    '        "holdout_gate_pass": not holdout_failures,\n'
    '        "selector_guard_pass": not holdout_failures,\n'
    '        "holdout_failures": holdout_failures,',
)
suite.write_text(text)

# Ensure no active executable code still advertises v7, apart from compatibility literals.
remaining = subprocess.run(
    ["git", "grep", "-n", "structural-v7", "--", "*.py", "*.cpp"],
    text=True,
    capture_output=True,
)
allowed = "experiments/compiled_query/itc_exam_suite.py"
if remaining.returncode == 0:
    unexpected = [line for line in remaining.stdout.splitlines() if not line.startswith(allowed + ":")]
    if unexpected:
        raise RuntimeError("executable v7 references remain:\n" + "\n".join(unexpected))
elif remaining.returncode != 1:
    raise RuntimeError(remaining.stderr)
