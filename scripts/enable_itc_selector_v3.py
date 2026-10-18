from pathlib import Path

path = Path("experiments/compiled_query/itc_exam_transfer.py")
text = path.read_text(encoding="utf-8")
old_allowed = 'or policy not in ("explicit", "structural-v2")'
new_allowed = 'or policy not in ("explicit", "structural-v2", "structural-v3")'
old_auto = 'or (mode == "auto" and policy != "structural-v2")'
new_auto = 'or (mode == "auto" and policy not in ("structural-v2", "structural-v3"))'
if text.count(old_allowed) != 1 or text.count(old_auto) != 1:
    raise RuntimeError("ITC selector-policy audit block differs")
text = text.replace(old_allowed, new_allowed).replace(old_auto, new_auto)
path.write_text(text, encoding="utf-8")
