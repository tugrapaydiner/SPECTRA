#!/usr/bin/env python3
"""Verify full M14 evidence and replay saved checkpoints on development only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.controlled_comparison import digest, full_solve, read_jsonl
from scripts.m14_controlled_experiment import build_lanes, configure, training_records


def verify(root: Path) -> dict:
    cfg = json.loads((root / "config.json").read_text()); configure(cfg)
    manifest = json.loads((root / "data/manifest.json").read_text())
    for name, part in manifest["partitions"].items():
        if digest(root / f"data/{name}.npz") != part["array_sha256"]:
            raise ValueError(f"{name} array hash mismatch")
    checked = 0
    records = training_records(root)
    for record in records:
        if not record["completed"] or record["steps_completed"] != cfg["steps"]:
            raise ValueError("incomplete fit in full completed-fit evidence")
        for checkpoint in record["checkpoints"]:
            if digest(root / checkpoint["path"]) != checkpoint["sha256"]:
                raise ValueError("checkpoint hash mismatch")
            checked += 1
    inventory = root / "inventory.json"
    checked_files = 0
    if inventory.exists():
        for name, expected in json.loads(inventory.read_text())["files"].items():
            if digest(root / name) != expected["sha256"]:
                raise ValueError(f"inventory mismatch: {name}")
            checked_files += 1
    # Hashing the sealed confirmation bytes above does not load or evaluate them.
    with np.load(root / "data/development.npz", allow_pickle=False) as ds:
        inputs = ds["inputs"].copy()
    ids = [r["id"] for r in manifest["partitions"]["development"]["examples"]]
    indices = [0, len(ids) // 3, 2 * len(ids) // 3, len(ids) - 1]
    replayed = []
    for phase in ["initial", "blank_only"]:
        selected = json.loads((root / phase / "selection.json").read_text())["selected"]
        rows = read_jsonl(root / phase / "development_rows.jsonl")
        expected = {(r["lane"], r["seed"], r["example_id"]): r for r in rows if r["round"] == 0}
        for lane in build_lanes(root, selected, phase):
            for i in indices:
                pred, success, _ = full_solve(lane["model"], inputs[i], constrained=lane["constrained"])
                old = expected[lane["lane"], lane["seed"], ids[i]]
                if pred.tolist() != old["prediction"] or int(success) != old["success"]:
                    raise ValueError(f"checkpoint replay mismatch: {phase}/{lane['lane']}/{lane['seed']}/{ids[i]}")
                replayed.append({"phase": phase, "lane": lane["lane"], "seed": lane["seed"], "example_id": ids[i]})
    return {"passed": True, "fits": len(records), "checkpoint_hashes_verified": checked,
            "inventory_files_verified": checked_files, "checkpoint_prediction_replays": len(replayed),
            "replayed_split": "development", "confirmation_model_evaluations": 0,
            "native_scope": "native fidelity reports are separate; this replay covers all selected PyTorch/INT8 lanes",
            "replays": replayed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", default="results/m14/latency_v1")
    print(json.dumps(verify(Path(parser.parse_args().out)), indent=2))
