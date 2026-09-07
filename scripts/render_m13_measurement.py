#!/usr/bin/env python3
"""Regenerate the single M13 figure from retained raw full-solve timing rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timings", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provenance", required=True)
    args = ap.parse_args()
    timings = Path(args.timings); summary_path = Path(args.summary)
    out = Path(args.out); prov = Path(args.provenance)
    rows = list(csv.DictReader(timings.open()))
    if not rows: raise ValueError("timing CSV is empty")
    summary = json.loads(summary_path.read_text())
    if summary.get("operation_convention") != "1_MAC_equals_2_arithmetic_operations":
        raise ValueError("M13 summary operation convention mismatch")
    if any(r.get("workload") != "actual_sequential_recurrence_complete_solve" for r in rows):
        raise ValueError("figure accepts only M13 actual sequential full-solve rows")

    by_example: dict[int, list[float]] = {}
    for r in rows:
        by_example.setdefault(int(r["example_index"]), []).append(float(r["latency_ms"]))
    examples = sorted(by_example)
    med = [statistics.median(by_example[i]) for i in examples]
    overall = statistics.median([float(r["latency_ms"]) for r in rows])

    plt.rcParams.update({"font.size": 10, "figure.dpi": 180})
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(range(len(examples)), med, marker="o", linewidth=1.25, markersize=4)
    ax.axhline(overall, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Distinct held-out instance (fixed M13 order)")
    ax.set_ylabel("Warm complete-solve latency (ms; median of 3 runs)")
    ax.set_title("M13 — actual sequential recurrence, complete B=1 solve path")
    ax.grid(True, alpha=0.25)
    ax.set_xticks(range(len(examples)))
    ax.set_xticklabels([str(i) for i in examples], rotation=90, fontsize=7)
    cold = summary["timing"].get("cold_start_ms")
    text = (
        f"warm median={overall:.3f} ms  |  cold start={cold:.1f} ms\n"
        f"24 distinct held-out examples; decode + semantic validation included; inference_mode=True"
    )
    ax.text(0.01, 0.99, text, transform=ax.transAxes, va="top", ha="left", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", metadata={"Software": "SPECTRA M13"})
    plt.close(fig)

    record = {
        "figure": str(out),
        "figure_sha256": sha256(out),
        "raw_timing_csv": str(timings),
        "raw_timing_csv_sha256": sha256(timings),
        "measurement_summary": str(summary_path),
        "measurement_summary_sha256": sha256(summary_path),
        "git_sha": os.environ.get("GITHUB_SHA", "local"),
        "workload": "actual_sequential_recurrence_complete_solve",
        "operation_convention": "1_MAC_equals_2_arithmetic_operations",
        "units": {"latency": "milliseconds", "throughput_if_referenced": "GOP/s arithmetic operations"},
        "figure_source_of_truth": "raw CSV and JSON; pixels are presentation only",
        "renderer": "scripts/render_m13_measurement.py",
        "matplotlib": matplotlib.__version__,
        "cache_residency_claim": False,
        "bandwidth_bottleneck_claim": False,
    }
    prov.parent.mkdir(parents=True, exist_ok=True)
    prov.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
