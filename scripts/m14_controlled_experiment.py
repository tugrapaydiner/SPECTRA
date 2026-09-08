#!/usr/bin/env python3
"""Execute only M14, retaining negative results and sealing confirmation data."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.energy_counters import measure_energy
from common.measurement_env import measurement_environment
from data.splits import build_reproducible_splits
from deploy.m10_artifact import export_cpu_artifact, load_cpu_artifact, module_tensor_state_sha256
from deploy.m10_runtime import CPURecursiveRuntime
from eval.controlled_comparison import (
    answer_loss, append_json, assert_hard_quantized, budget_label, convert_int8,
    digest, full_solve, load_partition, make_model, object_digest, outcome,
    paired_bounds, read_jsonl, set_quant_strength, verify_confirmation_freeze, write_json,
)

FAMILIES = ["fp_recursive", "ternary_recursive", "single_pass"]
LANES = ["fp_recursive", "ternary_recursive", "single_pass", "single_int8"]


def log(message: str) -> None:
    print(message, flush=True)


def configure(cfg: dict) -> None:
    torch.set_num_threads(cfg["threads"])
    torch.set_num_interop_threads(1)


def prepare(root: Path, config_path: Path) -> None:
    if root.exists():
        raise FileExistsError("experiment directory exists; use explicit stages, never overwrite")
    cfg = json.loads(config_path.read_text())
    configure(cfg)
    root.mkdir(parents=True)
    write_json(root / "config.json", cfg)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    source_files = {p: digest(Path(p)) for p in tracked
                    if p.endswith((".py", ".json", ".toml")) or p == "docs/M14_PROTOCOL.md"}
    write_json(root / "provenance.json", {"source_git_sha": source, "config_sha256": digest(root / "config.json"),
               "source_files": source_files, "source_inventory_sha256": object_digest(source_files),
               "created_time_ns": time.time_ns(), "protocol_sha256": digest(Path("docs/M14_PROTOCOL.md"))})
    write_json(root / "environment.json", measurement_environment(backend="CPU_PyTorch_reference_and_explicit_deployments"))
    write_json(root / "energy_probe.json", measure_energy(lambda: None))
    log("Generating fresh training/tuning/development/confirmation manifests; no models are evaluated.")
    sizes = cfg["sizes"]
    ds, original = build_reproducible_splits("sudoku", {
        "train": sizes["train"], "validation": sizes["tuning"] + sizes["development"],
        "test": sizes["confirmation"]}, cfg["data_seed"], generator_kwargs={
            "box": 3, "min_clues": 30, "max_clues": 35, "require_unique": True,
            "num_tokens": 10, "seq_len": 81, "solution_method": "random_backtracking", "augment": True},
        task_scope="generated_unique_9x9_sudoku_not_official_benchmark", official_benchmark=False)
    write_json(root / "data/m03_manifest.json", original)
    selection = {"train": ("train", 0, sizes["train"]), "tuning": ("validation", 0, sizes["tuning"]),
                 "development": ("validation", sizes["tuning"], sizes["tuning"] + sizes["development"]),
                 "confirmation": ("test", 0, sizes["confirmation"])}
    parts, seen_input, seen_pair, seen_group, seen_ids = {}, set(), set(), set(), set()
    for name, (old, start, stop) in selection.items():
        x, y = ds[old].inputs[start:stop], ds[old].targets[start:stop]
        examples = original["splits"][old]["examples"][start:stop]
        groups = {row["group_id"] for row in examples}
        inputs = {object_digest(row.tolist()) for row in x}
        pairs = {row["fingerprint"] for row in examples}
        ids = {row["id"] for row in examples}
        if len(inputs) != len(x) or len(ids) != len(x) or inputs & seen_input or pairs & seen_pair or groups & seen_group or ids & seen_ids:
            raise RuntimeError("fresh four-way split overlap/uniqueness audit failed")
        seen_input |= inputs; seen_pair |= pairs; seen_group |= groups; seen_ids |= ids
        path = root / f"data/{name}.npz"
        np.savez_compressed(path, inputs=x, targets=y)
        parts[name] = {"array_sha256": digest(path), "count": len(x), "examples": examples}
    write_json(root / "data/manifest.json", {"schema_version": 1, "data_seed": cfg["data_seed"],
               "partitions": parts, "cross_partition_exact_input_pair_group_overlap": 0,
               "unique_inputs_across_all_partitions": len(seen_input),
               "confirmation_model_evaluation": "sealed"})
    write_json(root / "preparation.json", {"completed": True, "data_manifest_sha256": digest(root / "data/manifest.json")})
    log("Data prepared and hashed; confirmation model evaluation remains sealed.")


def training_records(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((root / "fits").glob("*/fit.json"))]


def train(root: Path, cfg: dict, phase: str) -> None:
    if not (root / "preparation.json").exists():
        raise RuntimeError("prepare the data first")
    selected = None
    if phase == "blank_only":
        gate_path = root / "initial/development_gate.json"
        if not gate_path.exists() or json.loads(gate_path.read_text())["passed"]:
            raise RuntimeError("corrective training is only authorized by a failed initial development gate")
        selected = json.loads((root / "initial/selection.json").read_text())["selected"]
    tx, ty, _ = load_partition(root, "train", "training")
    x = torch.from_numpy(tx).long(); y = torch.from_numpy(ty).long()
    provenance = json.loads((root / "provenance.json").read_text())
    # Prespecified deterministic order; no family trains concurrently with another.
    tasks = []
    for seed in cfg["seeds"]:
        for family in FAMILIES:
            lrs = [selected[family]["lr"]] if selected else cfg["learning_rates"]
            tasks.extend((family, seed, lr) for lr in lrs)
    for family, seed, lr in tasks:
        identity = {"phase": phase, "family": family, "seed": seed, "lr": lr,
                    "config_sha256": provenance["config_sha256"], "source_git_sha": provenance["source_git_sha"]}
        run_id = object_digest(identity)[:24]
        directory = root / "fits" / run_id
        if (directory / "fit.json").exists():
            prior = json.loads((directory / "fit.json").read_text())
            if prior.get("completed"):
                log(f"Already retained completed fit {run_id}; leaving it unchanged.")
                continue
            raise RuntimeError(f"incomplete fit {run_id} retained; no silent overwrite")
        if directory.exists():
            raise RuntimeError(f"interrupted fit directory exists: {run_id}; preserve before a new declared attempt")
        elapsed_before = sum(r["training_seconds"] for r in training_records(root))
        if elapsed_before >= cfg["training_wall_budget_seconds"]:
            write_json(root / phase / "resource_stop.json", {"reason": "cumulative_fitting_budget_exhausted", "seconds": elapsed_before})
            return
        directory.mkdir(parents=True)
        model = make_model(family, seed)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=cfg["weight_decay"])
        generator = torch.Generator().manual_seed(seed + 1000000)
        record = {**identity, "run_id": run_id, "total_params": sum(p.numel() for p in model.parameters()),
                  "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "completed": False, "checkpoints": [], "training_seconds": 0., "steps_completed": 0}
        start = time.perf_counter()
        log(f"TRAIN {phase} {family} seed={seed} lr={lr} run={run_id}")
        error = None
        try:
            model.train()
            for step in range(1, cfg["steps"] + 1):
                if elapsed_before + time.perf_counter() - start >= cfg["training_wall_budget_seconds"]:
                    raise TimeoutError("cumulative fitting budget reached")
                set_quant_strength(model, min(1., step / cfg["quant_warmup_steps"]))
                idx = torch.randint(len(x), (cfg["batch_size"],), generator=generator)
                optimizer.zero_grad(set_to_none=True)
                loss = answer_loss(model, x[idx], y[idx], blank_only=phase == "blank_only")
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_norm"])
                if not bool(torch.isfinite(grad)) or float(grad) == 0:
                    raise FloatingPointError("nonfinite or zero gradient")
                optimizer.step()
                record["steps_completed"] = step
                append_json(directory / "curve.jsonl", {"step": step, "loss": float(loss.detach()),
                            "grad_norm_before_clip": float(grad), "elapsed_seconds": time.perf_counter() - start,
                            "quant_strength": min(1., step / cfg["quant_warmup_steps"]) if family == "ternary_recursive" else None})
                if step in cfg["validation_steps"]:
                    path = directory / f"step_{step}.pt"
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "minibatch_generator": generator.get_state(), "torch_rng": torch.get_rng_state(),
                                "identity": identity, "step": step, "quant_strength": 1.,
                                "data_manifest_sha256": digest(root / "data/manifest.json")}, path)
                    record["checkpoints"].append({"step": step, "path": str(path.relative_to(root)), "sha256": digest(path)})
                if step % 100 == 0:
                    log(f"  {family} seed={seed} step={step}/{cfg['steps']} loss={float(loss.detach()):.4f}")
            record["completed"] = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "minibatch_generator": generator.get_state(), "torch_rng": torch.get_rng_state(),
                        "identity": identity, "step": record["steps_completed"]}, directory / "partial.pt")
        record["training_seconds"] = time.perf_counter() - start
        record["error"] = error
        write_json(directory / "fit.json", record)
        if error:
            raise RuntimeError(f"fit failed and was retained: {error}")
    write_json(root / phase / "training_complete.json", {"completed": True,
        "phase": phase, "fits": len([r for r in training_records(root) if r["phase"] == phase]),
        "cumulative_training_seconds": sum(r["training_seconds"] for r in training_records(root))})


def load_model(root: Path, fit: dict, checkpoint: dict, lane: str, depth: int) -> tuple[object, dict]:
    path = root / checkpoint["path"]
    if digest(path) != checkpoint["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(fit["family"], fit["seed"])
    model.load_state_dict(saved["model"], strict=True)
    set_quant_strength(model, saved["quant_strength"])
    assert_hard_quantized(model)
    model.eval()
    if fit["family"] != "single_pass":
        model.N_sup = depth
    meta = {"backend": "pytorch_fp32", "precision": "FP32"}
    if fit["family"] == "ternary_recursive":
        meta = {"backend": "pytorch_fake_quantized_fp32", "precision": "ternary_weights_A8_state_simulated_with_FP32_arithmetic"}
    if lane == "single_int8":
        model, meta = convert_int8(model)
    return model, meta


def tune(root: Path, cfg: dict, phase: str) -> None:
    directory = root / phase
    if not (directory / "training_complete.json").exists():
        raise RuntimeError("all declared fits must complete before selection")
    if (directory / "tuning_rows.jsonl").exists():
        raise RuntimeError("tuning output exists; do not silently append a second selection experiment")
    x, y, examples = load_partition(root, "tuning", "selection")
    profiles = []
    log(f"TUNING {phase}: only tuning data are loaded.")
    for fit in training_records(root):
        if fit["phase"] != phase:
            continue
        for checkpoint in fit["checkpoints"]:
            lanes = [fit["family"]] + (["single_int8"] if fit["family"] == "single_pass" else [])
            depths = cfg["inference_steps"] if fit["family"] != "single_pass" else [1]
            for lane in lanes:
                for depth in depths:
                    model, backend = load_model(root, fit, checkpoint, lane, depth)
                    key = {"lane": lane, "lr": fit["lr"], "step": checkpoint["step"], "depth": depth}
                    profile_id = object_digest({**key, "run_id": fit["run_id"]})[:24]
                    cold = full_solve(model, x[0], constrained=phase == "blank_only")[2]
                    for puzzle in x[:cfg["warmup_examples"]]:
                        full_solve(model, puzzle, constrained=phase == "blank_only")
                    rows = []
                    for i, (puzzle, target) in enumerate(zip(x, y)):
                        pred, success, latency = full_solve(model, puzzle, constrained=phase == "blank_only")
                        row = {"profile_id": profile_id, "run_id": fit["run_id"], "seed": fit["seed"],
                               "example_id": examples[i]["id"], "latency_ms": latency,
                               **outcome(puzzle, target, pred), "prediction": pred.tolist()}
                        rows.append(row); append_json(directory / "tuning_rows.jsonl", row)
                    cost_rows = rows[cfg["warmup_examples"]:cfg["warmup_examples"] + cfg["tuning_cost_examples"]]
                    profiles.append({**key, "seed": fit["seed"], "profile_id": profile_id,
                        "run_id": fit["run_id"], "checkpoint": checkpoint, "backend": backend,
                        "success": float(np.mean([r["success"] for r in rows])),
                        "blank_accuracy": sum(r["blank_correct"] for r in rows) / sum(r["blank_count"] for r in rows),
                        "mean_latency_ms": float(np.mean([r["latency_ms"] for r in cost_rows])),
                        "cold_first_solve_ms": cold})
        log(f"  tuned {fit['family']} seed={fit['seed']} lr={fit['lr']}")
    grouped = {}
    for p in profiles:
        key = (p["lane"], p["lr"], p["step"], p["depth"])
        grouped.setdefault(key, []).append(p)
    selected = {}
    for lane in LANES:
        candidates = []
        for key, ps in grouped.items():
            if key[0] != lane:
                continue
            if sorted(p["seed"] for p in ps) != sorted(cfg["seeds"]):
                raise RuntimeError("selection must include every declared seed")
            candidates.append({"lane": lane, "lr": key[1], "step": key[2], "depth": key[3],
                "mean_success": float(np.mean([p["success"] for p in ps])),
                "mean_blank_accuracy": float(np.mean([p["blank_accuracy"] for p in ps])),
                "mean_latency_ms": float(np.mean([p["mean_latency_ms"] for p in ps])), "profiles": ps})
        selected[lane] = sorted(candidates, key=lambda c: (-c["mean_success"], -c["mean_blank_accuracy"],
                               c["mean_latency_ms"], c["lr"], c["step"], c["depth"]))[0]
    write_json(directory / "tuning_profiles.json", profiles)
    selection = {"phase": phase, "selected": selected, "selection_split": "tuning",
                 "tuning_rows_sha256": digest(directory / "tuning_rows.jsonl"),
                 "baseline_budget_caps_ms": {lane: selected[lane]["mean_latency_ms"] * (1 + cfg["cost_tolerance"])
                                              for lane in ["single_pass", "single_int8"]}}
    write_json(directory / "selection.json", selection)
    log("Selection frozen: " + json.dumps({k: {s: v[s] for s in ["lr", "step", "depth", "mean_success", "mean_blank_accuracy", "mean_latency_ms"]} for k, v in selected.items()}))


def build_lanes(root: Path, selected: dict, phase: str) -> list[dict]:
    fits = {r["run_id"]: r for r in training_records(root)}
    lanes = []
    for lane, config in selected.items():
        for profile in config["profiles"]:
            fit = fits[profile["run_id"]]
            start = time.perf_counter()
            model, metadata = load_model(root, fit, profile["checkpoint"], lane, config["depth"])
            lanes.append({"lane": lane, "seed": fit["seed"], "model": model, "metadata": metadata,
                          "run_id": fit["run_id"], "checkpoint": profile["checkpoint"], "depth": config["depth"],
                          "load_conversion_seconds": time.perf_counter() - start, "native": False,
                          "constrained": phase == "blank_only", "symbolic": False})
    return lanes


def native_lanes(root: Path, cfg: dict, phase: str, lanes: list[dict], warm_x: np.ndarray) -> tuple[list[dict], dict]:
    native = []
    reports = []
    for lane in lanes:
        if lane["lane"] != "ternary_recursive":
            continue
        path = root / phase / f"native_seed{lane['seed']}.pt"
        start = time.perf_counter()
        try:
            export_cpu_artifact(lane["model"], path, height=9, width=9, box=3,
                source_checkpoint_sha256=lane["checkpoint"]["sha256"],
                source_checkpoint_tensor_sha256=module_tensor_state_sha256(lane["model"]),
                training_seed=lane["seed"], training_step=lane["checkpoint"]["step"],
                data_provenance={"m14_manifest_sha256": digest(root / "data/manifest.json")},
                export_git_sha=json.loads((root / "provenance.json").read_text())["source_git_sha"])
            runtime = CPURecursiveRuntime(load_cpu_artifact(path))
            nr = {**lane, "lane": "ternary_native", "model": runtime, "native": True,
                  "checkpoint": {"path": str(path.relative_to(root)), "sha256": digest(path), "step": lane["checkpoint"]["step"]}}
            discrepancies = 0; errors = []; timings = []
            for i, puzzle in enumerate(warm_x[:cfg["warmup_examples"] + cfg["tuning_cost_examples"]]):
                x = torch.from_numpy(puzzle.copy()).long()[None]
                with torch.inference_mode():
                    reference = lane["model"](x, height=9, width=9)[0]
                    actual = runtime.forward(x).logits
                errors.append(float((reference - actual).abs().max()))
                if not torch.allclose(reference, actual, atol=1e-4, rtol=1e-4):
                    discrepancies += 1
                ref, _, _ = full_solve(lane["model"], puzzle, constrained=nr["constrained"])
                pred, _, ms = full_solve(runtime, puzzle, constrained=nr["constrained"], native=True)
                if not np.array_equal(ref, pred):
                    discrepancies += 1
                if i >= cfg["warmup_examples"]:
                    timings.append(ms)
            report = {"seed": lane["seed"], "eligible": discrepancies == 0,
                      "discrepancies": discrepancies, "max_logit_absolute_error": max(errors),
                      "mean_tuning_latency_ms": float(np.mean(timings)),
                      "artifact_sha256": digest(path), "export_build_validation_seconds": time.perf_counter() - start,
                      "backend": runtime.backend_report()}
            nr["metadata"] = report["backend"]
            if report["eligible"]:
                native.append(nr)
        except Exception as exc:
            report = {"seed": lane["seed"], "eligible": False, "reason": f"{type(exc).__name__}: {exc}"}
        reports.append(report)
    selection = json.loads((root / phase / "selection.json").read_text())
    selected_backend = "ternary_recursive"
    if len(native) == len(cfg["seeds"]) and np.mean([r["mean_tuning_latency_ms"] for r in reports]) < selection["selected"]["ternary_recursive"]["mean_latency_ms"]:
        selected_backend = "ternary_native"
    if len(native) != len(cfg["seeds"]):
        native = []  # An incomplete native seed set is not a comparable lane.
    return native, {"reports": reports, "primary_candidate": selected_backend, "selection_data": "tuning_only"}


def evaluate(root: Path, cfg: dict, phase: str, split: str) -> None:
    if split == "confirmation":
        verify_confirmation_freeze(root, phase)
    directory = root / phase
    rows_path = directory / f"{split}_rows.jsonl"
    if rows_path.exists():
        raise RuntimeError("evaluation already exists; preserve it rather than appending duplicate runs")
    selected = json.loads((directory / "selection.json").read_text())["selected"]
    lanes = build_lanes(root, selected, phase)
    warm_x, _, _ = load_partition(root, "tuning", "warmup")
    if split == "development":
        native, backend = native_lanes(root, cfg, phase, lanes, warm_x)
        write_json(directory / "native_selection.json", backend)
        lanes += native
    else:
        backend = json.loads((directory / "native_selection.json").read_text())
        eligible_reports = [r for r in backend["reports"] if r["eligible"]]
        for report in eligible_reports if len(eligible_reports) == len(cfg["seeds"]) else []:
                parent = next(l for l in lanes if l["lane"] == "ternary_recursive" and l["seed"] == report["seed"])
                path = directory / f"native_seed{report['seed']}.pt"
                if digest(path) != report["artifact_sha256"]:
                    raise ValueError("frozen native artifact changed")
                lanes.append({**parent, "lane": "ternary_native", "model": CPURecursiveRuntime(load_cpu_artifact(path)),
                              "native": True, "metadata": report["backend"],
                              "checkpoint": {"path": str(path.relative_to(root)), "sha256": digest(path), "step": parent["checkpoint"]["step"]}})
    for seed in cfg["seeds"]:
        lanes.append({"lane": "symbolic", "seed": seed, "model": None, "metadata": {"backend": "python_MRV_backtracking", "training_seeds": 0},
                      "run_id": "symbolic_untrained_reference", "checkpoint": None, "depth": 0,
                      "native": False, "symbolic": True, "constrained": False, "load_conversion_seconds": 0.})
    for lane in lanes:
        for puzzle in warm_x[:cfg["warmup_examples"]]:
            full_solve(lane["model"], puzzle, constrained=lane["constrained"], symbolic=lane["symbolic"], native=lane["native"])
    write_json(directory / f"{split}_freeze.json", {"selection_sha256": digest(directory / "selection.json"),
         "native_selection_sha256": digest(directory / "native_selection.json"), "primary_candidate": backend["primary_candidate"],
         "lanes": [{k: v for k, v in lane.items() if k != "model"} for lane in lanes]})
    purpose = "development_evaluation" if split == "development" else "confirmation_evaluation"
    x, y, examples = load_partition(root, split, purpose)
    rng = np.random.default_rng(cfg["data_seed"] + (1 if split == "development" else 2))
    log(f"EVALUATION {phase}/{split}: {len(lanes)} systems/seeds, {len(x)} paired examples, {cfg['timing_rounds']} rounds.")
    for repetition in range(cfg["timing_rounds"]):
        for i in rng.permutation(len(x)):
            for j in rng.permutation(len(lanes)):
                lane = lanes[j]
                pred, success, latency = full_solve(lane["model"], x[i], constrained=lane["constrained"],
                                    symbolic=lane["symbolic"], native=lane["native"])
                append_json(rows_path, {"schema_version": 1, "phase": phase, "split": split,
                    "lane": lane["lane"], "seed": lane["seed"], "run_id": lane["run_id"],
                    "example_id": examples[i]["id"], "round": repetition, "latency_ms": latency,
                    "prediction": pred.tolist(), **outcome(x[i], y[i], pred),
                    "energy_joules": None, "energy_missing_reason": "individual_solve_energy_not_measured"})
        log(f"  timing round {repetition + 1}/{cfg['timing_rounds']} retained")
    energy = measure_energy(lambda: None)
    write_json(directory / f"{split}_energy.json", {"probe": energy,
        "per_example_energy_available": False, "reason": energy["failure_reason"] or "latency_protocol_no_per_example_energy",
        "cost_endpoint": "complete_solve_latency"})
    analyze_gate(root, cfg, phase, split)


def analyze_gate(root: Path, cfg: dict, phase: str, split: str) -> dict:
    directory = root / phase
    rows_path = directory / f"{split}_rows.jsonl"
    rows = read_jsonl(rows_path)
    manifest = json.loads((root / "data/manifest.json").read_text())
    ids = [r["id"] for r in manifest["partitions"][split]["examples"]]
    seeds = cfg["seeds"]
    frozen = json.loads((directory / f"{split}_freeze.json").read_text())
    groups = {}
    for r in rows:
        groups.setdefault((r["lane"], r["seed"], r["example_id"]), []).append(r)
    arrays = {}
    lane_names = sorted({r["lane"] for r in frozen["lanes"]})
    for lane in lane_names:
        correct = np.empty((len(seeds), len(ids)))
        cost = np.empty_like(correct)
        for s, seed in enumerate(seeds):
            for i, item in enumerate(ids):
                sample = groups.get((lane, seed, item), [])
                if sorted(r["round"] for r in sample) != list(range(cfg["timing_rounds"])):
                    raise RuntimeError("missing/duplicate measurement rounds invalidate the gate")
                if len({tuple(r["prediction"]) for r in sample}) != 1:
                    raise RuntimeError("nondeterministic predictions across timing rounds")
                correct[s, i] = sample[0]["success"]
                cost[s, i] = float(np.median([r["latency_ms"] for r in sample]))
        arrays[lane] = (correct, cost)
    candidate = frozen["primary_candidate"]
    comparisons = {}
    for baseline in ["single_pass", "single_int8"]:
        a, ca = arrays[candidate]; b, cb = arrays[baseline]
        result = paired_bounds(a, b, ca, cb, alpha=cfg["one_sided_alpha_per_baseline"],
                               draws=cfg["bootstrap_draws"], seed=cfg["bootstrap_seed"])
        result["budget_label"] = budget_label(result["latency_ratio"], cfg["cost_tolerance"])
        result["accuracy_pass"] = result["accuracy_lower_bound"] > cfg["minimum_accuracy_gain"]
        result["cost_pass"] = result["latency_ratio_upper_bound"] <= 1 + cfg["cost_tolerance"]
        comparisons[baseline] = result
    passed = all(r["accuracy_pass"] and r["cost_pass"] for r in comparisons.values())
    gate = {"schema_version": 1, "phase": phase, "split": split, "passed": passed,
            "milestone_complete": passed and split == "confirmation", "primary_candidate": candidate,
            "comparisons": comparisons, "raw_rows_sha256": digest(rows_path),
            "freeze_sha256": digest(directory / f"{split}_freeze.json"),
            "status": "PASS" if passed else "INCOMPLETE_TARGET_MISSED",
            "confirmation_evaluated": split == "confirmation"}
    gate_path = directory / f"{split}_gate.json"
    write_json(gate_path, gate)
    append_json(root / "attempts.jsonl", {"phase": phase, "split": split, "passed": passed,
                "gate_sha256": digest(gate_path), "raw_rows_sha256": digest(rows_path)})
    if passed and split == "development":
        if (root / "confirmation_authorization.json").exists():
            raise RuntimeError("a confirmation attempt has already been authorized")
        write_json(root / "confirmation_authorization.json", {
            "development_gate_path": str(gate_path.relative_to(root)), "development_gate_sha256": digest(gate_path),
            "phase": phase, "maximum_attempts": 1})
    log(f"GATE {phase}/{split}: {gate['status']}")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/m14/latency_v1")
    parser.add_argument("--config", default="config/m14_comparison.json")
    parser.add_argument("--stage", choices=["prepare", "train", "tune", "evaluate"], required=True)
    parser.add_argument("--phase", choices=["initial", "blank_only"], default="initial")
    parser.add_argument("--split", choices=["development", "confirmation"], default="development")
    args = parser.parse_args(); root = Path(args.out)
    if args.stage == "prepare":
        prepare(root, Path(args.config)); return
    cfg = json.loads((root / "config.json").read_text()); configure(cfg)
    provenance = json.loads((root / "provenance.json").read_text())
    if digest(root / "config.json") != provenance["config_sha256"]:
        raise RuntimeError("frozen experiment configuration changed")
    execution = {"stage": args.stage, "phase": args.phase, "split": args.split,
        "start_time_ns": time.time_ns(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {p: digest(Path(p)) for p in ["scripts/m14_controlled_experiment.py", "eval/controlled_comparison.py"]},
        "config_sha256": digest(root / "config.json"), "completed": False}
    execution_path = root / "stage_executions" / f"{execution['start_time_ns']}_{args.stage}_{args.phase}.json"
    write_json(execution_path, execution)
    start = time.perf_counter()
    try:
        if args.stage == "train":
            train(root, cfg, args.phase)
        elif args.stage == "tune":
            tune(root, cfg, args.phase)
        else:
            evaluate(root, cfg, args.phase, args.split)
        execution["completed"] = True
    except Exception as exc:
        execution["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        execution["elapsed_seconds"] = time.perf_counter() - start
        write_json(execution_path, execution)


if __name__ == "__main__":
    main()
