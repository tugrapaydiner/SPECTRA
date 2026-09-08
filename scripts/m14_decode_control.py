#!/usr/bin/env python3
"""Post-failure development control: change decoder, freeze every model weight."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.controlled_comparison import append_json, digest, full_solve, load_partition, outcome, write_json
from scripts.m14_controlled_experiment import build_lanes, configure


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/m14/latency_v1")
    args = ap.parse_args(); root = Path(args.out)
    cfg = json.loads((root / "config.json").read_text()); configure(cfg)
    initial_gate = json.loads((root / "initial/development_gate.json").read_text())
    if initial_gate["passed"]:
        raise RuntimeError("failure analysis must not replace a successful independent confirmation")
    output = root / "decode_control/rows.jsonl"
    if output.exists():
        raise RuntimeError("preserve the existing decoder control")
    selected = json.loads((root / "initial/selection.json").read_text())["selected"]
    lanes = build_lanes(root, selected, "blank_only")
    x, y, examples = load_partition(root, "development", "development_evaluation")
    warm, _, _ = load_partition(root, "tuning", "warmup")
    for lane in lanes:
        for puzzle in warm[:cfg["warmup_examples"]]:
            full_solve(lane["model"], puzzle, constrained=True)
    rng = np.random.default_rng(cfg["data_seed"] + 3)
    for repetition in range(cfg["timing_rounds"]):
        for i in rng.permutation(len(x)):
            for j in rng.permutation(len(lanes)):
                lane = lanes[j]
                pred, _, ms = full_solve(lane["model"], x[i], constrained=True)
                append_json(output, {"phase": "decode_only", "split": "development", "lane": lane["lane"],
                    "seed": lane["seed"], "run_id": lane["run_id"], "checkpoint_sha256": lane["checkpoint"]["sha256"],
                    "round": repetition, "example_id": examples[i]["id"], "latency_ms": ms,
                    "prediction": pred.tolist(), **outcome(x[i], y[i], pred)})
    write_json(root / "decode_control/provenance.json", {"initial_selection_sha256": digest(root / "initial/selection.json"),
        "rows_sha256": digest(output), "scope": "reused_development_decoder_only_control_same_trained_checkpoints",
        "weight_updates": 0, "milestone_gate": False})
    print("Decoder-only control retained; no training or confirmation access.", flush=True)


if __name__ == "__main__":
    main()
