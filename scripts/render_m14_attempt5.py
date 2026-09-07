#!/usr/bin/env python3
"""Render M14 Attempt-5 result table and Pareto plot from raw evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.m14_attempt5_dual_stream_semantic_exit import aggregate_with_work


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=".m14a5/experiment")
    ap.add_argument("--out", default=".m14a5/rendered")
    args = ap.parse_args()
    exp = Path(args.experiment); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary_path = exp / "summary.json"; summary = json.loads(summary_path.read_text())
    surface = "confirmation" if summary.get("confirmation_opened") else "development"
    raw_path = exp / f"{surface}_rows.jsonl"; rows = read_jsonl(raw_path); aggs = aggregate_with_work(rows)
    table = []
    for r in sorted(aggs, key=lambda x: (float(x.get("latency_median_ms") or 1e30), str(x["config_id"]))):
        table.append({
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
            "executed_steps_mean": r.get("executed_steps_mean"),
            "block_applications_mean": r.get("block_applications_mean"),
            "semantic_early_exit_fraction": r.get("semantic_early_exit_fraction"),
            "selected_candidate": r["config_id"] == summary.get("selected_candidate"),
            "primary_baseline": r["config_id"] == summary.get("primary_baseline"),
        })
    csv_path = out / "result_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
    md = out / "result_table.md"
    lines = [
        "# M14 Attempt 5 result table", "", f"Evidence surface: **{surface}**", "",
        "Candidate: original dim-64 dual-stream FP TRM with reference-free semantic early exit.", "",
        "| config | semantic validity | median complete-solve ms | mean supervision steps | mean shared-block apps | early-exit fraction | role |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in table:
        role = "candidate" if r["selected_candidate"] else ("primary baseline" if r["primary_baseline"] else "context")
        steps = "" if r["executed_steps_mean"] is None else f"{float(r['executed_steps_mean']):.4f}"
        blocks = "" if r["block_applications_mean"] is None else f"{float(r['block_applications_mean']):.4f}"
        stop = "" if r["semantic_early_exit_fraction"] is None else f"{float(r['semantic_early_exit_fraction']):.4f}"
        lines.append(
            f"| {r['config_id']} | {float(r['semantic_validity']):.6f} | {float(r['latency_median_ms']):.6f} | {steps} | {blocks} | {stop} | {role} |"
        )
    lines += ["", "Complete-solve latency includes every executed dual-stream recurrence step and semantic validity check. Unmatched points are not called iso-budget.", ""]
    md.write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.8, 5.9))
    for r in table:
        x = float(r["latency_median_ms"]); y = float(r["semantic_validity"])
        marker = "*" if r["selected_candidate"] else ("s" if r["primary_baseline"] else "o")
        size = 140 if r["selected_candidate"] else 65
        ax.scatter([x], [y], marker=marker, s=size)
        ax.annotate(str(r["config_id"]), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Median complete-solve latency (ms)")
    ax.set_ylabel("Strict semantic solve rate")
    ax.set_title(f"M14 Attempt 5 — {surface} quality / latency")
    ax.grid(True, alpha=.25); fig.tight_layout()
    plot = out / "pareto.png"; fig.savefig(plot, dpi=180); plt.close(fig)

    prov = {
        "milestone": 14,
        "attempt": 5,
        "surface": surface,
        "raw_rows": str(raw_path), "raw_rows_sha256": sha256(raw_path),
        "summary": str(summary_path), "summary_sha256": sha256(summary_path),
        "result_table": str(csv_path), "result_table_sha256": sha256(csv_path),
        "result_table_markdown": str(md), "result_table_markdown_sha256": sha256(md),
        "pareto_plot": str(plot), "pareto_plot_sha256": sha256(plot),
        "plot_source_of_truth": "raw JSONL rows; pixels are presentation only",
        "primary_candidate": summary.get("selected_candidate"),
        "primary_baseline": summary.get("primary_baseline"),
        "adaptive_development_attempts_before_confirmation": 5,
        "candidate_architecture": "original_dual_stream_fp64_trm_n1_t1",
        "candidate_training_weights": [0.1,0.2,0.3,0.4],
        "semantic_stop_reference_target_used": False,
        "thresholds_changed": False,
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(prov, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
