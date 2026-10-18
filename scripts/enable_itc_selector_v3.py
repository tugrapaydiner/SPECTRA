from pathlib import Path

path = Path("experiments/compiled_query/itc_exam_transfer.py")
text = path.read_text(encoding="utf-8")

old_workflow = '    Path(".github/workflows/itc-exam-transfer.yml"),\n'
if text.count(old_workflow) != 1:
    raise RuntimeError("stale ITC transfer workflow inventory entry differs")
text = text.replace(old_workflow, "")

replacements = {
    'or current["auto_policy"] not in ("explicit", "structural-v2")':
        'or current["auto_policy"] not in ("explicit", "structural-v2", "structural-v3")',
    'or (mode == "auto" and current["auto_policy"] != "structural-v2")':
        'or (mode == "auto" and current["auto_policy"] not in ("structural-v2", "structural-v3"))',
    'or policy not in ("explicit", "structural-v2")':
        'or policy not in ("explicit", "structural-v2", "structural-v3")',
    'or (mode == "auto" and policy != "structural-v2")':
        'or (mode == "auto" and policy not in ("structural-v2", "structural-v3"))',
}
for old, new in replacements.items():
    if text.count(old) != 1:
        raise RuntimeError(f"ITC selector-policy check differs: {old}")
    text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
