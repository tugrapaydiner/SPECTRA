from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace(path: str, old: str, new: str, count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text()
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"{path}: expected {count} matches, found {actual}")
    target.write_text(text.replace(old, new))


path = "experiments/compiled_query/test_exact.py"
replace(path, "assert answer.info['auto_policy_reason'] == 'dense_pin_fraction'",
        "assert answer.info['auto_policy_reason'] == 'dense_small_high_pin'", 1)
replace(path, "assert low.info['auto_policy_reason'] == 'scc_tile64'",
        "assert low.info['auto_policy_reason'] == 'scc_long_bank'")
replace(path, "assert high.info['auto_policy_reason'] == 'dense_pin_fraction'",
        "assert high.info['auto_policy_reason'] == 'dense_small_high_pin'")
replace(
    path,
'''    if structure == 'odd_cycle':
        selected, reason = 'raw', 'base_infeasible'
    elif pins >= 8 * count and 24 * pins >= n * count:
        selected, reason = 'raw', 'dense_pin_fraction'
    else:
        components = {'empty': 0, 'independent': 18, 'chain': 2}[structure]
        if count <= 128 and components == 2 * n and len(edges) >= 4 * n:
            selected, reason = 'raw', 'dense_uncompressed_warmup'
        elif count >= 64:
            selected, reason = 'scc', 'scc_tile64'
        elif count >= 8 and 2 * components <= n:
            selected, reason = 'scc', 'scc_strong_compression'
        else:
            selected, reason = 'raw', 'short_weak_compression'
''',
'''    if structure == 'odd_cycle':
        selected, reason = 'raw', 'base_infeasible'
    elif (pins >= 8 * count and 24 * pins >= n * count
          and (count < 64 or n < 256)):
        selected = 'raw'
        reason = 'dense_small_high_pin' if n < 256 else 'dense_pin_short'
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
''')

# Historical evidence must remain readable after the policy version advances.
for path in (
    "experiments/compiled_query/itc_exam_suite.py",
    "experiments/compiled_query/itc_exam_transfer.py",
):
    target = ROOT / path
    text = target.read_text()
    text = text.replace(
        '"structural-v2", "structural-v3", "structural-v4", "structural-v5", "structural-v7"',
        '"structural-v2", "structural-v3", "structural-v4", "structural-v5", "structural-v6", "structural-v7"',
    )
    target.write_text(text)

# The exposed WAP guard follows the current policy identity and a new schedule.
path = ROOT / "experiments/compiled_query/auto_selector_wap_guard.py"
text = path.read_text()
text = text.replace("structural automatic engine selection v4", "structural automatic engine selection v7")
text = text.replace('POLICY = "structural-v6"', 'POLICY = "structural-v7"')
text = text.replace("SPECTRA-auto-wap-v6", "SPECTRA-auto-wap-v7")
path.write_text(text)

# Ensure the intended active code no longer makes stale v6-only assertions.
for path in (
    "experiments/compiled_query/test_exact.py",
    "tests/public/test_auto_selector_v7.py",
    "tests/public/test_exact_api.py",
    "experiments/compiled_query/auto_selector_wap_guard.py",
):
    text = (ROOT / path).read_text()
    if "structural-v6" in text or "dense_pin_fraction" in text or "scc_tile64" in text:
        raise RuntimeError(f"stale selector contract remains in {path}")
