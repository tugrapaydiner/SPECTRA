#!/usr/bin/env python3
"""M14 post-failure cost localization and native fidelity counterexamples."""
from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.m10_artifact import load_cpu_artifact
from deploy.m10_runtime import CPURecursiveRuntime
from eval.controlled_comparison import append_json, digest, full_solve, load_partition, write_json
from scripts.m14_controlled_experiment import build_lanes, configure


def quant_trace(reference, runtime, puzzle):
    ref_records, native_records = [], []
    def ref_hook(module, args, output):
        ref_records.append((args[0].clone(), output.clone()))
    handle = reference.act_quant.register_forward_hook(ref_hook)
    original = runtime._a8
    def wrapped(x):
        y = original(x); native_records.append((x.clone(), y.clone())); return y
    runtime._a8 = wrapped
    try:
        with torch.inference_mode():
            x = torch.from_numpy(puzzle.copy()).long()[None]
            ref = reference(x, height=9, width=9)[0]; nat = runtime.forward(x).logits
    finally:
        handle.remove(); runtime._a8 = original
    traces = []
    for j, ((a, aq), (b, bq)) in enumerate(zip(ref_records, native_records)):
        sa = a.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
        sb = b.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
        ac = torch.clamp(torch.round(a / sa), -128, 127)
        bc = torch.clamp(torch.round(b / sb), -128, 127)
        changed = ac != bc
        location = changed.nonzero()[0].tolist() if changed.any() else None
        record = {"a8_call": j, "input_max_abs_difference": float((a - b).abs().max()),
                  "output_max_abs_difference": float((aq - bq).abs().max()),
                  "integer_code_differences": int(changed.sum()), "first_changed_index": location}
        if location:
            index = tuple(location)
            record.update(reference_scaled_value=float((a / sa)[index]), native_scaled_value=float((b / sb)[index]),
                          reference_code=int(ac[index]), native_code=int(bc[index]))
        traces.append(record)
    return {"a8_trace": traces, "reference_prediction": ref.argmax(-1)[0].tolist(),
            "native_prediction": nat.argmax(-1)[0].tolist(), "max_logit_absolute_error": float((ref - nat).abs().max())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/m14/latency_v1")
    args = ap.parse_args(); root = Path(args.out)
    cfg = json.loads((root / "config.json").read_text()); configure(cfg)
    gate = json.loads((root / "initial/development_gate.json").read_text())
    if gate["passed"]:
        raise RuntimeError("post-failure analysis requires a failed development gate")
    directory = root / "failure_analysis"
    if directory.exists():
        raise RuntimeError("preserve the existing failure analysis")
    directory.mkdir()
    selected = json.loads((root / "initial/selection.json").read_text())["selected"]
    lanes = build_lanes(root, selected, "initial")
    x, _, examples = load_partition(root, "tuning", "selection")
    profiles = []
    for lane in lanes:
        if lane["seed"] != cfg["seeds"][0]:
            continue
        model = lane["model"]
        for puzzle in x[:8]:
            full_solve(model, puzzle)
        direct = [full_solve(model, x[i % 8])[2] for i in range(32)]
        profiler = cProfile.Profile()
        start = time.perf_counter(); profiler.enable()
        for i in range(32):
            full_solve(model, x[i % 8])
        profiler.disable(); wall = time.perf_counter() - start
        stats = pstats.Stats(profiler)
        functions = []
        for (filename, line, name), (primitive, calls, self_time, cumulative, _) in stats.stats.items():
            f = Path(filename)
            try:
                label = str(f.relative_to(Path.cwd()))
            except ValueError:
                label = f.name
            functions.append({"file": label, "line": line, "function": name, "calls": calls,
                              "self_seconds": self_time, "cumulative_seconds": cumulative})
        functions.sort(key=lambda r: r["self_seconds"], reverse=True)
        profiles.append({"lane": lane["lane"], "seed": lane["seed"], "checkpoint_sha256": lane["checkpoint"]["sha256"],
                         "uninstrumented_mean_ms": float(np.mean(direct)), "profiled_mean_ms": wall * 1000 / 32,
                         "scope": "diagnostic_cProfile_32_solves_8_tuning_inputs_not_primary_latency",
                         "functions": functions[:35],
                         "quantization_functions": [r for r in functions if r["function"] in ["ternarize", "_ternarize_hard"]]})
    write_json(directory / "cost_profiles.json", profiles)
    failed_examples = []
    for lane in lanes:
        if lane["lane"] != "ternary_recursive":
            continue
        artifact = root / "initial" / f"native_seed{lane['seed']}.pt"
        runtime = CPURecursiveRuntime(load_cpu_artifact(artifact))
        for i, puzzle in enumerate(x[:cfg["warmup_examples"] + cfg["tuning_cost_examples"]]):
            with torch.inference_mode():
                tx = torch.from_numpy(puzzle.copy()).long()[None]
                reference = lane["model"](tx, height=9, width=9)[0]
                actual = runtime.forward(tx).logits
            same_answer = bool((reference.argmax(-1) == actual.argmax(-1)).all())
            close = torch.allclose(reference, actual, atol=1e-4, rtol=1e-4)
            row = {"seed": lane["seed"], "example_id": examples[i]["id"], "same_answer": same_answer,
                   "logits_within_declared_tolerance": close, "max_logit_absolute_error": float((reference - actual).abs().max()),
                   "source_checkpoint_sha256": lane["checkpoint"]["sha256"], "native_artifact_sha256": digest(artifact)}
            append_json(directory / "native_fidelity_rows.jsonl", row)
            if not close or not same_answer:
                trace = quant_trace(lane["model"], runtime, puzzle)
                failed_examples.append({**row, "puzzle": puzzle.tolist(), **trace})
    write_json(directory / "native_counterexamples.json", failed_examples)
    write_json(directory / "provenance.json", {"script_sha256": digest(Path(__file__)),
              "selection_sha256": digest(root / "initial/selection.json"),
              "native_fidelity_rows_sha256": digest(directory / "native_fidelity_rows.jsonl"),
              "scope": "post_failure_tuning_analysis_no_confirmation_access",
              "counterexample_count": len(failed_examples),
              "no_native_tolerance_change": True})
    print(json.dumps({"native_fidelity_examples": len(x[:cfg["warmup_examples"] + cfg["tuning_cost_examples"]]) * 3,
                      "counterexamples": len(failed_examples), "profiled_lanes": len(profiles)}), flush=True)


if __name__ == "__main__":
    main()
