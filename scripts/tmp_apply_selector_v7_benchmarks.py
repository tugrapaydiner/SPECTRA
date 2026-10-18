from __future__ import annotations

from pathlib import Path
import textwrap


# ---------------------------------------------------------------------------
# Structural benchmark: retain the rejected v6 panel as retrospective
# development evidence and create a fresh, larger v7 confirmation panel.
# ---------------------------------------------------------------------------
path = Path("experiments/compiled_query/auto_selector_benchmark.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    '"""Frozen development and confirmation benchmark for structural auto policy v4."""',
    '"""Frozen development, retrospective, and confirmation benchmark for structural auto policy v7."""',
)
text = text.replace('SCHEMA = "spectra.auto_selector_benchmark.v4"',
                    'SCHEMA = "spectra.auto_selector_benchmark.v7"')
text = text.replace('POLICY = "structural-v6"', 'POLICY = "structural-v7"')
text = text.replace('"SPECTRA-auto-selector-v6-schedule"',
                    '"SPECTRA-auto-selector-v7-schedule"')

old_expected = '''def expected_policy(*, base: bool, pins: int, count: int, vertices: int,
                    edges: int, components: int) -> tuple[str, str]:
    if count <= 8 and vertices > 512:
        return "raw", "large_tiny_bank"
    if not base:
        return "raw", "base_infeasible"
    if pins >= 8 * count and 24 * pins >= vertices * count:
        return "raw", "dense_pin_fraction"
    if count <= 128 and components == 2 * vertices and edges >= 4 * vertices:
        return "raw", "dense_uncompressed_warmup"
    if count >= 64:
        return "scc", "scc_tile64"
    if count >= 8 and 2 * components <= vertices:
        return "scc", "scc_strong_compression"
    return "raw", "short_weak_compression"
'''
new_expected = '''def expected_policy(*, base: bool, pins: int, count: int, vertices: int,
                    edges: int, components: int) -> tuple[str, str]:
    if count <= 8 and vertices > 512:
        return "raw", "large_tiny_bank"
    if not base:
        return "raw", "base_infeasible"
    if pins >= 8 * count and 24 * pins >= vertices * count:
        if count >= 64 and vertices >= 128:
            return "scc", "dense_long_tile64"
        return "raw", "dense_pin_fraction"
    if count <= 128 and components == 2 * vertices and edges >= 4 * vertices:
        return "raw", "dense_uncompressed_warmup"
    if count >= 64:
        return "scc", "scc_tile64"
    if count >= 8 and 2 * components <= vertices:
        return "scc", "scc_strong_compression"
    return "raw", "short_weak_compression"
'''
if text.count(old_expected) != 1:
    raise RuntimeError("v6 expected-policy block differs")
text = text.replace(old_expected, new_expected)

start = text.index("def confirmation_cases() -> tuple[PanelCase, ...]:")
end = text.index("\n\ndef panel_cases(panel: str)", start)
old_confirmation = text[start:end]
old_confirmation = old_confirmation.replace(
    "def confirmation_cases()", "def v6_failed_confirmation_cases()", 1
)
new_confirmation = '''def confirmation_cases() -> tuple[PanelCase, ...]:
    result: list[PanelCase] = []
    namespace = "SPECTRA-auto-selector-v7-confirmation"
    regimes = (
        ("dense-small-96", 96, 8, False, False, 0.0),
        ("dense-boundary-127", 127, 8, False, False, 0.0),
        ("dense-boundary-128", 128, 8, False, False, 0.0),
        ("dense-medium-192", 192, 10, False, False, 0.0),
        ("strong-sparse-384", 384, 2, True, False, 0.0),
        ("edge-heavy-warmup-768", 768, 12, False, False, 5.0),
        ("low-density-large-1024", 1024, 16, False, False, 0.0),
        ("base-infeasible-512", 512, 4, False, True, 0.0),
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
new_panel = '''def panel_cases(panel: str) -> tuple[PanelCase, ...]:
    if panel == "development":
        return development_cases()
    if panel == "retrospective":
        return v6_failed_confirmation_cases()
    if panel == "confirmation":
        return confirmation_cases()
    raise ValueError("panel must be development, retrospective, or confirmation")
'''
text = text[:start] + old_confirmation + "\n\n" + new_confirmation.rstrip() + "\n\n" + new_panel.rstrip() + text[end + 2 + len("def panel_cases(panel: str) -> tuple[PanelCase, ...]:"):]
# Remove the old panel function body left after replacing its signature.
leftover = '''
    if panel == "development":
        return development_cases()
    if panel == "confirmation":
        return confirmation_cases()
    raise ValueError("panel must be development or confirmation")
'''
if leftover not in text:
    raise RuntimeError("old panel body differs")
text = text.replace(leftover, "", 1)

text = text.replace(
    'if panel == "development":\n        limits = {',
    'if panel in ("development", "retrospective"):\n        limits = {',
    1,
)
text = text.replace(
    '"exposed development evidence" if panel == "development"\n            else "fresh project-owned prospective confirmation; not external replication"',
    '"exposed development evidence" if panel == "development"\n            else "retained failed-v6 confirmation; v7 development evidence"\n            if panel == "retrospective"\n            else "fresh project-owned prospective confirmation; not external replication"',
)
text = text.replace(
    'parser.add_argument("--panel", choices=("development", "confirmation"), required=True)',
    'parser.add_argument("--panel", choices=("development", "retrospective", "confirmation"), required=True)',
)

if "structural-v6" in text:
    # The only allowed v6 occurrence is the retrospective namespace/function text.
    disallowed = [line for line in text.splitlines()
                  if "structural-v6" in line and "v6-confirmation" not in line]
    if disallowed:
        raise RuntimeError("unexpected v6 policy identity remains: " + repr(disallowed))
path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# WAP guard: same cases and gates, new immutable policy identity.
# ---------------------------------------------------------------------------
path = Path("experiments/compiled_query/auto_selector_wap_guard.py")
text = path.read_text(encoding="utf-8")
text = text.replace("structural automatic engine selection v4", "structural automatic engine selection v7")
text = text.replace('spectra.auto_selector_wap_guard.v4', 'spectra.auto_selector_wap_guard.v7')
text = text.replace('POLICY = "structural-v6"', 'POLICY = "structural-v7"')
text = text.replace('SPECTRA-auto-wap-v6/', 'SPECTRA-auto-wap-v7/')
path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# ITC suite: v6 already opened set10/set12, so all three cases are now an
# explicitly retrospective v7 guard. Preserve the schema and raw receipts.
# ---------------------------------------------------------------------------
path = Path("experiments/compiled_query/itc_exam_suite.py")
text = path.read_text(encoding="utf-8")
text = text.replace("The structural-v6 selector", "The structural-v7 selector")
text = text.replace(
    "``set10`` and ``set12`` are fixed holdout\ninstances: their execution results must not be used to retune the policy.",
    "``set10`` and ``set12`` rejected or informed earlier selector versions and\nare therefore retrospective development evidence for v7, not holdouts.",
)
text = text.replace(
    'if self.role not in ("development", "holdout", "replication"):',
    'if self.role not in ("development", "retrospective", "holdout", "replication"):',
)
text = text.replace('"set9-development", "development",',
                    '"set9-retrospective", "retrospective",')
text = text.replace('"set10-holdout", "holdout",',
                    '"set10-retrospective", "retrospective",')
text = text.replace('"set12-holdout", "holdout",',
                    '"set12-retrospective", "retrospective",')
text = text.replace(
    '    if not any(case.role == "holdout" for case in checked):\n'
    '        raise ValueError("suite requires at least one holdout")\n',
    '',
)
text = text.replace(
    '"prospective multi-instance bounded exam-repair transfer; set9 is "\n'
    '                "development and set10/set12 are fixed execution holdouts"',
    '"retrospective multi-instance bounded exam-repair selector guard; "\n'
    '                "no case remains an untouched v7 holdout"',
)
text = text.replace('"selector_policy": "structural-v6"',
                    '"selector_policy": "structural-v7"')
text = text.replace(
    '"retained failed-v3 confirmation and v4 density sweep",\n'
    '                "set9 development transfer",',
    '"retained failed-v3 and failed-v6 confirmations",\n'
    '                "opened set9/set10/set12 transfer results",',
)
text = text.replace(
    '"set10 and set12 results may falsify the frozen selector but may not "\n'
    '                "change it inside this suite"',
    '"all three cases are retrospective development guards for v7"',
)
text = text.replace('"structural-v5", "structural-v6")',
                    '"structural-v5", "structural-v6", "structural-v7")')
text = text.replace('if item["role"] == "holdout" and threshold is not None:',
                    'if item["role"] in ("holdout", "retrospective") and threshold is not None:')
text = text.replace(
    '"set9 is development evidence; set10 and set12 are prospective execution holdouts",',
    '"set9, set10, and set12 are retrospective development evidence for v7",',
)
path.write_text(text, encoding="utf-8")
