"""Aggregate edge-report JSON files into a comparison table (BLUEPRINT section 27).

    python scripts/make_report.py outputs/*.json --out outputs/summary.md

Builds the baseline/ablation comparison and the compute-optimal frontier summary
the paper needs (accuracy per measured joule, per MB, per ms).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import json  # noqa: E402

from common import get_logger  # noqa: E402

log = get_logger("make_report")

_COLUMNS = [
    ("name", "Model"),
    ("accuracy", "Acc"),
    ("latency_ms", "Latency(ms)"),
    ("peak_ram_mb", "RAM(MB)"),
    ("model_size_mb", "Size(MB)"),
    ("joules_per_problem", "J/prob"),
    ("accuracy_per_joule", "Acc/J"),
    ("accuracy_per_ms", "Acc/ms"),
]


def _fmt(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate edge reports into a table.")
    ap.add_argument("reports", nargs="+", help="edge report JSON files")
    ap.add_argument("--out", default="outputs/summary.md")
    args = ap.parse_args()

    rows = []
    for path in args.reports:
        p = Path(path)
        if not p.exists():
            log.warning("skip missing %s", p)
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        data.setdefault("name", p.stem)
        rows.append(data)

    if not rows:
        log.warning("no reports found")
        return

    header = "| " + " | ".join(label for _, label in _COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in _COLUMNS) + " |")

    table = "\n".join(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# SPECTRA edge report\n\n" + table + "\n", encoding="utf-8")
    log.info("Wrote %s\n%s", out, table)


if __name__ == "__main__":
    main()
