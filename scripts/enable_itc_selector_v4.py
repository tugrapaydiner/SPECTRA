from __future__ import annotations

from pathlib import Path

transfer = Path("experiments/compiled_query/itc_exam_transfer.py")
text = transfer.read_text(encoding="utf-8")
replacements = {
    'or current["auto_policy"] not in ("explicit", "structural-v2")':
        'or current["auto_policy"] not in ("explicit", "structural-v4")',
    'or (mode == "auto" and current["auto_policy"] != "structural-v2")':
        'or (mode == "auto" and current["auto_policy"] != "structural-v4")',
    'or policy not in ("explicit", "structural-v2")':
        'or policy not in ("explicit", "structural-v2", "structural-v3", "structural-v4")',
    'or (mode == "auto" and policy != "structural-v2")':
        'or (mode == "auto" and policy not in ("structural-v2", "structural-v3", "structural-v4"))',
}
for old, new in replacements.items():
    if text.count(old) != 1:
        raise RuntimeError(f"expected one transfer policy clause: {old}")
    text = text.replace(old, new)
transfer.write_text(text, encoding="utf-8")

suite = Path("experiments/compiled_query/itc_exam_suite.py")
text = suite.read_text(encoding="utf-8")
text = text.replace(
    "The selector-v2 rule is developed from the exposed synthetic panel and the\n"
    "published ``set9`` transfer only.",
    "The structural-v4 selector is developed from the exposed synthetic panel,\n"
    "the retained failed-v3 confirmation, and the published ``set9`` transfer.",
)
if text.count("DEFAULT_HOLDOUT_MAX_RATIO = 1.15") != 1:
    raise RuntimeError("ITC holdout default differs")
text = text.replace("DEFAULT_HOLDOUT_MAX_RATIO = 1.15", "DEFAULT_HOLDOUT_MAX_RATIO = 1.05")
if text.count('"selector_policy": "structural-v2"') != 1:
    raise RuntimeError("ITC suite selector identity differs")
text = text.replace('"selector_policy": "structural-v2"', '"selector_policy": "structural-v4"')
if text.count('or manifest.get("selector_policy") != "structural-v2"') != 1:
    raise RuntimeError("ITC suite manifest policy check differs")
text = text.replace(
    'or manifest.get("selector_policy") != "structural-v2"',
    'or manifest.get("selector_policy") not in ("structural-v2", "structural-v3", "structural-v4")',
)
text = text.replace(
    '"exposed synthetic graph panel",\n                "set9 development transfer",',
    '"exposed synthetic graph panel",\n                "retained failed-v3 confirmation and v4 density sweep",\n                "set9 development transfer",',
)
suite.write_text(text, encoding="utf-8")

contract = Path("experiments/compiled_query/test_itc_exam_transfer.py")
text = contract.read_text(encoding="utf-8")
if text.count('== "structural-v2"') != 1:
    raise RuntimeError("ITC transfer contract policy identity differs")
contract.write_text(text.replace('== "structural-v2"', '== "structural-v4"'), encoding="utf-8")
