#!/usr/bin/env python3
"""Audit retained M14 predictions and regenerate tables/figures without fitting."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.controlled_comparison import digest, read_jsonl, wilson, write_json


LABELS = {"fp_recursive": "Small FP32 recursive", "ternary_recursive": "Small ternary · PyTorch",
          "ternary_native": "Small ternary · packed C++", "single_pass": "Larger FP32 single-pass",
          "single_int8": "Larger mixed INT8 single-pass", "symbolic": "Symbolic MRV reference"}
COLORS = {"fp_recursive": "#2C5985", "ternary_recursive": "#C17430", "ternary_native": "#A34228",
          "single_pass": "#496A4E", "single_int8": "#769260", "symbolic": "#6B6474"}


def independently_check(puzzle: list[int], target: list[int], pred: list[int]) -> dict:
    if len(puzzle) != 81 or len(target) != 81 or len(pred) != 81:
        raise ValueError("prediction/task geometry changed")
    if any(type(v) is not int for v in pred):
        raise ValueError("prediction contains noninteger values")
    domain = set(range(1, 10))
    groups = [pred[9 * r:9 * r + 9] for r in range(9)]
    groups += [pred[c::9] for c in range(9)]
    groups += [[pred[(3 * br + dr) * 9 + 3 * bc + dc] for dr in range(3) for dc in range(3)]
               for br in range(3) for bc in range(3)]
    clue_errors = sum(x != p for x, p in zip(puzzle, pred) if x != 0)
    success = int(all(set(g) == domain for g in groups) and clue_errors == 0)
    return {"success": success, "exact": int(pred == target),
            "blank_correct": sum(p == y for x, p, y in zip(puzzle, pred, target) if x == 0),
            "blank_count": sum(x == 0 for x in puzzle), "clue_errors": clue_errors,
            "invalid_digits": sum(p not in domain for p in pred)}


def audit_rows(root: Path, path: Path, split: str, rounds: int) -> tuple[list[dict], int]:
    manifest = json.loads((root / "data/manifest.json").read_text())
    part = manifest["partitions"][split]
    data_path = root / f"data/{split}.npz"
    if digest(data_path) != part["array_sha256"]:
        raise ValueError("data hash does not match manifest")
    if split == "confirmation" and not (root / "confirmation_consumed.json").exists():
        raise RuntimeError("confirmation report cannot bypass the sealed evaluation")
    with np.load(data_path, allow_pickle=False) as ds:
        index = {m["id"]: (x.tolist(), y.tolist()) for m, x, y in zip(part["examples"], ds["inputs"], ds["targets"])}
    rows = read_jsonl(path); samples = {}
    freeze_path = path.parent / f"{split}_freeze.json"
    frozen = json.loads(freeze_path.read_text()) if freeze_path.exists() else None
    expected_runs = {(r["lane"], r["seed"]): r["run_id"] for r in frozen["lanes"]} if frozen else None
    if frozen and {r["lane"] for r in rows} != {r["lane"] for r in frozen["lanes"]}:
        raise ValueError("raw lanes differ from the frozen comparison")
    for r in rows:
        if expected_runs and expected_runs.get((r["lane"], r["seed"])) != r["run_id"]:
            raise ValueError("raw run identity differs from the frozen checkpoint mapping")
        if r["example_id"] not in index:
            raise ValueError("raw row refers to an undeclared example")
        expected = independently_check(*index[r["example_id"]], r["prediction"])
        if any(r[k] != v for k, v in expected.items()):
            raise ValueError("raw success/cell counts disagree with independent semantic checker")
        if not math.isfinite(r["latency_ms"]) or r["latency_ms"] <= 0:
            raise ValueError("invalid measured latency")
        samples.setdefault((r["lane"], r["seed"], r["example_id"]), []).append(r)
    expected_seeds = json.loads((root / "config.json").read_text())["seeds"]
    for lane in {r["lane"] for r in rows}:
        for seed in expected_seeds:
            for eid in index:
                sample = samples.get((lane, seed, eid), [])
                if sorted(r["round"] for r in sample) != list(range(rounds)):
                    raise ValueError("missing or duplicate paired timing rows")
                if len({tuple(r["prediction"]) for r in sample}) != 1:
                    raise ValueError("timing repetitions do not agree on prediction")
    # Statistical units are one training seed and one puzzle, after reducing rounds.
    reduced = []
    for sample in samples.values():
        reduced.append({**sample[0], "latency_ms": float(np.median([r["latency_ms"] for r in sample]))})
    return reduced, len(rows)


def summarize(rows: list[dict], phase: str, split: str) -> list[dict]:
    summaries = []
    for lane in sorted({r["lane"] for r in rows}):
        selected = [r for r in rows if r["lane"] == lane]
        seeds = sorted({r["seed"] for r in selected})
        per_seed = []
        for seed in seeds:
            ss = [r for r in selected if r["seed"] == seed]
            successes = sum(r["success"] for r in ss)
            per_seed.append({"seed": seed, "successes": successes, "examples": len(ss),
                             "solve_rate": successes / len(ss), "solve_rate_wilson95": wilson(successes, len(ss)),
                             "blank_accuracy": sum(r["blank_correct"] for r in ss) / sum(r["blank_count"] for r in ss),
                             "mean_latency_ms": float(np.mean([r["latency_ms"] for r in ss]))})
        costs = [r["latency_ms"] for r in selected]
        summaries.append({"phase": phase, "split": split, "lane": lane, "label": LABELS[lane],
                          "training_seeds": 0 if lane == "symbolic" else len(seeds),
                          "paired_examples": len(selected) // len(seeds),
                          "mean_solve_rate": float(np.mean([s["solve_rate"] for s in per_seed])),
                          "mean_blank_accuracy": float(np.mean([s["blank_accuracy"] for s in per_seed])),
                          "mean_latency_ms": float(np.mean(costs)), "median_latency_ms": float(np.median(costs)),
                          "p95_latency_ms": float(np.quantile(costs, .95)),
                          "mean_clue_errors_per_board": float(np.mean([r["clue_errors"] for r in selected])),
                          "mean_invalid_digits_per_board": float(np.mean([r["invalid_digits"] for r in selected])),
                          "energy_joules_per_solve": None, "per_seed": per_seed})
    return summaries


def render(root: Path) -> dict:
    cfg = json.loads((root / "config.json").read_text())
    rows_used, summaries, raw_count = {}, [], 0
    for phase in ["initial", "blank_only"]:
        for split in ["development", "confirmation"]:
            p = root / phase / f"{split}_rows.jsonl"
            if p.exists():
                reduced, count = audit_rows(root, p, split, cfg["timing_rounds"])
                summaries.extend(summarize(reduced, phase, split)); raw_count += count
                rows_used[str(p.relative_to(root))] = digest(p)
    p = root / "decode_control/rows.jsonl"
    if p.exists():
        reduced, count = audit_rows(root, p, "development", cfg["timing_rounds"])
        summaries.extend(summarize(reduced, "decode_only", "development")); raw_count += count
        rows_used[str(p.relative_to(root))] = digest(p)
    if not summaries:
        raise RuntimeError("no completed measured comparisons to render")
    fits = {r["run_id"]: r for r in [json.loads(p.read_text()) for p in (root / "fits").glob("*/fit.json")]}
    for s in summaries:
        if s["lane"] == "symbolic":
            s.update(selected_lr=None, checkpoint_step=0, inference_recursion_steps=0, trainable_params=0)
            continue
        source_phase = "initial" if s["phase"] == "decode_only" else s["phase"]
        selection = json.loads((root / source_phase / "selection.json").read_text())["selected"]
        lane = "ternary_recursive" if s["lane"] == "ternary_native" else s["lane"]
        sel = selection[lane]
        s.update(selected_lr=sel["lr"], checkpoint_step=sel["step"],
                 inference_recursion_steps=sel["depth"] if "recursive" in lane else 0,
                 trainable_params=fits[sel["profiles"][0]["run_id"]]["trainable_params"])
    report = root / "report"; report.mkdir(exist_ok=True)
    write_json(report / "summary.json", summaries)
    fields = [k for k in summaries[0] if k != "per_seed"]
    with (report / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in summaries)
    lines = ["# M14 measured development results", "", "Generated from independently checked saved predictions and raw complete-solve timing rows.", "",
             "Timing rounds are reduced to per-example medians. Three training seeds share paired examples; they do not multiply the unique test sample count. The symbolic solver is untrained.", ""]
    for phase in ["initial", "decode_only", "blank_only"]:
        selected = [r for r in summaries if r["phase"] == phase and r["split"] == "development"]
        if not selected:
            continue
        lines += [f"## {phase.replace('_', ' ')}", "",
                  "| System | Solve rate | Blank accuracy | Mean solve ms | p95 ms |", "|---|---:|---:|---:|---:|"]
        for s in selected:
            lines.append(f"| {s['label']} | {100*s['mean_solve_rate']:.2f}% | {100*s['mean_blank_accuracy']:.2f}% | {s['mean_latency_ms']:.3f} | {s['p95_latency_ms']:.3f} |")
        lines.append("")
    lines += ["Energy per solve: unavailable; these are latency measurements.", "",
              "Inspect `per_seed` in `summary.json` for individual successes, Wilson intervals and training-seed variation.", ""]
    (report / "RESULTS.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    phases = [p for p in ["initial", "blank_only"] if any(s["phase"] == p for s in summaries)]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), layout="constrained")
    frontiers = []
    for ax, metric, title in zip(axes, ["mean_solve_rate", "mean_blank_accuracy"],
                                ["Complete Sudoku solves", "Blank-cell accuracy (secondary)"]):
        for phase in phases:
            candidates = [s for s in summaries if s["phase"] == phase and s["split"] == "development" and s["lane"] != "symbolic"]
            frontier = [s for s in candidates if not any(
                p["mean_latency_ms"] <= s["mean_latency_ms"] and p[metric] >= s[metric]
                and (p["mean_latency_ms"] < s["mean_latency_ms"] or p[metric] > s[metric]) for p in candidates)]
            frontier.sort(key=lambda s: s["mean_latency_ms"])
            frontiers.append({"phase": phase, "metric": metric, "scope": "empirical_neural_means_not_significance",
                              "lanes": [s["lane"] for s in frontier]})
            ax.plot([s["mean_latency_ms"] for s in frontier], [100*s[metric] for s in frontier],
                    color="#A9AEB3", linestyle="-" if phase == "initial" else "--", linewidth=1, zorder=1)
            for s in summaries:
                if s["phase"] != phase or s["split"] != "development":
                    continue
                marker = "o" if phase == "initial" else "s"
                ax.scatter(s["mean_latency_ms"], s[metric] * 100, marker=marker, s=60,
                           facecolors=COLORS[s["lane"]] if phase == "initial" else "white",
                           edgecolors=COLORS[s["lane"]], linewidths=1.5, zorder=3)
                # Individual seeds show variability without pretending timings are independent tasks.
                per_metric = "solve_rate" if metric == "mean_solve_rate" else "blank_accuracy"
                ys = [ps[per_metric] * 100 for ps in s["per_seed"]]
                ax.vlines(s["mean_latency_ms"], min(ys), max(ys), colors=COLORS[s["lane"]], linewidth=1.3)
        ax.set_xscale("log"); ax.set_xlabel("Mean complete-solve latency (ms, log scale)")
        ax.set_ylabel("Accuracy (%)"); ax.set_ylim(-4, 104); ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#DFE3E6", linewidth=.6); ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[l], markeredgecolor=COLORS[l], label=LABELS[l])
               for l in LABELS if any(s["lane"] == l for s in summaries)]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("SPECTRA M14 · measured development comparison", x=.02, ha="left", fontsize=14, fontweight="bold")
    fig.text(.02, -.005, "Filled circles: initial. Open squares: blank-only training + clue-preserving decode. Bars: training-seed range.\n"
             "Generated 9×9 Sudoku · 192 paired development examples · 3 training seeds · CPU only · no measured joules", fontsize=8)
    fig.savefig(report / "accuracy_latency.png", dpi=180, bbox_inches="tight")
    fig.savefig(report / "accuracy_latency.svg", bbox_inches="tight")
    plt.close(fig)
    write_json(report / "pareto_frontiers.json", frontiers)
    gate_files = list(root.glob("*/development_gate.json")) + list(root.glob("*/confirmation_gate.json"))
    provenance = {"raw_files": rows_used, "audited_raw_rows": raw_count,
                  "independent_checker": "standard_python_row_column_box_and_clue_checks",
                  "renderer_sha256": digest(Path(__file__)),
                  "gates": {str(p.relative_to(root)): digest(p) for p in gate_files},
                  "confirmation_evaluated": (root / "confirmation_consumed.json").exists(),
                  "table_sha256": digest(report / "results.csv"),
                  "figure_sha256": digest(report / "accuracy_latency.svg"),
                  "statistical_unit": "training_seed_and_paired_example; timing rounds collapsed"}
    write_json(report / "provenance.json", provenance)
    print(json.dumps({"independently_audited_rows": raw_count, "confirmation_evaluated": provenance["confirmation_evaluated"]}))
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", default="results/m14/latency_v1")
    render(Path(parser.parse_args().out))
