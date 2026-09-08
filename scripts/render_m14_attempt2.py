#!/usr/bin/env python3
"""Render M14 Attempt-2 result table and Pareto plot from raw JSONL evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.m14_primary_experiment import all_aggregates


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=".m14a2/experiment")
    ap.add_argument("--out", default=".m14a2/rendered")
    args = ap.parse_args()
    exp = Path(args.experiment); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary_path = exp / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    surface = "confirmation" if summary.get("confirmation_opened") else "development"
    raw_path = exp / f"{surface}_rows.jsonl"
    rows = read_jsonl(raw_path)
    aggs = all_aggregates(rows)

    table_rows = []
    for r in sorted(aggs, key=lambda x: (float(x.get("latency_median_ms") or 1e30), str(x["config_id"]))):
        table_rows.append({
            "surface": surface,
            "config_id": r["config_id"],
            "family": r.get("family"),
            "n": r.get("n"),
            "n_training_seeds": r.get("n_training_seeds"),
            "semantic_validity": r.get("semantic_validity"),
            "exact_reference_match": r.get("exact_reference_match"),
            "blank_cell_accuracy": r.get("blank_cell_accuracy"),
            "latency_median_ms": r.get("latency_median_ms"),
            "latency_p95_ms": r.get("latency_p95_ms"),
            "selected_candidate": r["config_id"] == summary.get("selected_candidate"),
            "primary_baseline": r["config_id"] == summary.get("primary_baseline"),
        })

    csv_path = out / "result_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table_rows[0].keys()))
        writer.writeheader(); writer.writerows(table_rows)

    md_path = out / "result_table.md"
    lines = [
        "# M14 Attempt 2 result table",
        "",
        f"Evidence surface: **{surface}**",
        "",
        "| config | semantic validity | median complete-solve ms | p95 ms | role |",
        "|---|---:|---:|---:|---|",
    ]
    for r in table_rows:
        role = "candidate" if r["selected_candidate"] else ("primary baseline" if r["primary_baseline"] else "context")
        lines.append(
            f"| {r['config_id']} | {float(r['semantic_validity']):.6f} | "
            f"{float(r['latency_median_ms']):.6f} | {float(r['latency_p95_ms']):.6f} | {role} |"
        )
    lines += [
        "",
        "Latency is measured complete-solve latency. A fast Path-B point is not called iso-latency; the primary gate is defined in the preregistered protocol.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for r in table_rows:
        x = float(r["latency_median_ms"]); y = float(r["semantic_validity"])
        marker = "*" if r["selected_candidate"] else ("s" if r["primary_baseline"] else "o")
        size = 130 if r["selected_candidate"] else 65
        ax.scatter([x], [y], marker=marker, s=size)
        ax.annotate(str(r["config_id"]), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Median complete-solve latency (ms)")
    ax.set_ylabel("Strict semantic solve rate")
    ax.set_title(f"M14 Attempt 2 — {surface} quality / latency")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plot_path = out / "pareto.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    prov = {
        "milestone": 14,
        "attempt": 2,
        "surface": surface,
        "raw_rows": str(raw_path),
        "raw_rows_sha256": sha256(raw_path),
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "result_table": str(csv_path),
        "result_table_sha256": sha256(csv_path),
        "result_table_markdown": str(md_path),
        "result_table_markdown_sha256": sha256(md_path),
        "pareto_plot": str(plot_path),
        "pareto_plot_sha256": sha256(plot_path),
        "plot_source_of_truth": "raw JSONL rows; pixels are presentation only",
        "primary_candidate": summary.get("selected_candidate"),
        "primary_baseline": summary.get("primary_baseline"),
        "thresholds_changed": False,
        "adaptive_development_attempts_before_confirmation": 2,
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(prov, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
