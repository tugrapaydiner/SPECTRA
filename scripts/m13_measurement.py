#!/usr/bin/env python3
"""M13 retained measurement: real sequential recurrence, raw evidence first.

The primary performance workload is CPURecursiveRuntime.forward on distinct held-out
4x4 Sudoku instances plus semantic validation. The old K-input native benchmark is
run separately by scripts/bench_kernel.py and is labelled precomputed-input reuse.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.energy_counters import discover_energy_domains, measure_energy, read_energy_snapshot
from common.measurement_env import measurement_environment
from common.seed import set_seed
from data.datasets import build_sudoku_arrays
from deploy.m10_artifact import (
    export_cpu_artifact,
    load_cpu_artifact,
    module_tensor_state_sha256,
)
from deploy.m10_runtime import CPURecursiveRuntime
from eval.latent_mcts import LatentNativeMCTS
from eval.memory import (
    linux_process_hwm_mb,
    measure_peak_ram,
    model_state_bytes,
    packed_weight_bytes,
    process_rss_mb,
    search_tree_tensor_bytes,
)
from model.energy import LatentEnergyVerifier
from model.latent_action import LatentActionCodebook
from model.trm import TRM
from model.verifier import sudoku_correct, sudoku_score
from train.losses import deep_supervision_loss

SEED = 20260913
TRAIN_STEPS = 120
BATCH = 64
REPRESENTATIVE_N = 24
WARMUP_N = 4
TIMING_REPS = 3
ENERGY_PASSES = 3


def plain(v: Any) -> Any:
    if torch.is_tensor(v):
        if v.ndim == 0: return v.detach().cpu().item()
        return v.detach().cpu().tolist()
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, Path): return str(v)
    if isinstance(v, dict): return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [plain(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v): return None
    return v


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    c = t.detach().cpu().contiguous(); h = hashlib.sha256()
    h.update(str(c.dtype).encode()); h.update(np.asarray(c.shape, dtype="<i8").tobytes())
    h.update(c.numpy().tobytes()); return h.hexdigest()


def build_model() -> TRM:
    return TRM(dim=32, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1, N_sup=4,
               heads=4, max_grid_size=8, ternary=True, act8=True)


def train_model(model: TRM, tx: torch.Tensor, ty: torch.Tensor) -> dict[str, Any]:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    gen = torch.Generator().manual_seed(SEED + 1); curve = []
    model.train(); t0 = time.perf_counter()
    for step in range(1, TRAIN_STEPS + 1):
        idx = torch.randint(0, tx.shape[0], (BATCH,), generator=gen)
        _, rows = model(tx[idx], height=4, width=4)
        loss = deep_supervision_loss(rows, ty[idx])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step in {1, 30, 60, 90, 120}:
            curve.append({"step": step, "loss": float(loss.detach())})
    model.eval()
    return {"elapsed_seconds": time.perf_counter() - t0, "curve": curve,
            "steps": TRAIN_STEPS, "excluded_from_inference_measurements": True}


def full_solve(runtime: CPURecursiveRuntime, x: torch.Tensor) -> tuple[Any, bool]:
    """Complete declared M13 solve: recurrent runtime + decode + semantic validation."""
    with torch.inference_mode():
        result = runtime.forward(x)
        success = bool(sudoku_correct(x, result.answer, box=2)[0])
    return result, success


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s: return float("nan")
    idx = min(len(s) - 1, max(0, int(math.ceil(q * len(s)) - 1)))
    return float(s[idx])


def modelled_sequential_traffic(runtime: CPURecursiveRuntime, work: dict[str, Any]) -> dict[str, Any]:
    """Declared logical tensor/interface byte model for actual sequential recurrence."""
    L = int(runtime.arch["seq_len"]); D = int(runtime.arch["dim"])
    N = int(runtime.arch["N_sup"]); T = int(runtime.arch["T"]); n = int(runtime.arch["n"])
    cycles = N * T
    native_boundary = 0
    scale_bias = 0
    for name, entry in runtime.linears.items():
        calls = int(work["native_calls_by_layer"].get(name, 0))
        in_f, out_f = int(entry["in_features"]), int(entry["out_features"])
        native_boundary += calls * L * (in_f + out_f) * 4  # FP32 call input/output tensor bytes
        for key in ("scale", "bias"):
            t = entry.get(key)
            if torch.is_tensor(t): scale_bias += t.numel() * t.element_size()
    fp_tensor_bytes = sum(t.numel() * t.element_size() for t in runtime.fp.values())
    packed = packed_weight_bytes(runtime.artifact) or 0
    # Lower-bound recurrent state interface: read+write y/z once per recursive cycle.
    yz_state = 2 * L * D * 4
    recurrent_state_rw_lower_bound = cycles * 2 * yz_state
    input_embedding_output = L * D * 4
    token_input = L * 8  # torch.long ids
    logits_decode = N * L * int(runtime.arch["num_tokens"]) * 4 + L * 8
    # Algorithmic attention score storage if materialized. PyTorch SDP may fuse/stream,
    # so it is disclosed but excluded from the denominator below.
    f_calls = N * T * (n + 1)
    heads = int(runtime.arch["heads"])
    attention_score_if_materialized = f_calls * heads * L * L * 4
    accounted = (packed + scale_bias + fp_tensor_bytes + native_boundary
                 + recurrent_state_rw_lower_bound + input_embedding_output + token_input + logits_decode)
    macs = int(work["native_scalar_products"]); operations = 2 * macs
    return {
        "workload": "actual_sequential_recurrence_complete_solve",
        "operation_convention": "1_MAC_equals_2_arithmetic_operations",
        "native_linear_macs": macs,
        "native_linear_operations": operations,
        "persistent_packed_weight_bytes": int(packed),
        "persistent_scale_bias_bytes": int(scale_bias),
        "persistent_fp_tensor_bytes": int(fp_tensor_bytes),
        "native_fp32_input_output_boundary_bytes": int(native_boundary),
        "recurrent_yz_read_write_lower_bound_bytes": int(recurrent_state_rw_lower_bound),
        "input_token_bytes": int(token_input),
        "input_embedding_output_bytes": int(input_embedding_output),
        "logits_and_decode_bytes": int(logits_decode),
        "attention_score_bytes_if_materialized": int(attention_score_if_materialized),
        "attention_score_bytes_included_in_accounted_total": False,
        "accounted_logical_bytes": int(accounted),
        "native_linear_ops_per_accounted_logical_byte": operations / max(1, accounted),
        "traffic_scope": "logical_tensor_and_interface_bytes_not_measured_cache_or_dram_transactions",
        "memory_levels": {
            "persistent_storage": "packed weights/scales/FP tensors; cache level not asserted",
            "activation_state": "logical FP32 tensor boundary/state bytes; cache/DRAM residency not asserted",
            "attention": "algorithmic score bytes disclosed conditionally; PyTorch implementation may fuse/stream",
        },
        "cache_residency_established": False,
        "bandwidth_bottleneck_established": False,
    }


def measure_instrumentation_overhead(runtime: CPURecursiveRuntime, x: torch.Tensor) -> dict[str, Any]:
    direct = []
    for _ in range(8):
        t0 = time.perf_counter_ns(); full_solve(runtime, x); direct.append((time.perf_counter_ns()-t0)/1e6)
    sampled = []
    for _ in range(8):
        m = measure_peak_ram(lambda: full_solve(runtime, x), interval_s=0.0005)
        sampled.append(float(m["elapsed_seconds"]) * 1000.0)
    snap_us = []
    for _ in range(32):
        t0 = time.perf_counter_ns(); read_energy_snapshot(); snap_us.append((time.perf_counter_ns()-t0)/1e3)
    return {
        "direct_full_solve_mean_ms": statistics.mean(direct),
        "rss_sampled_full_solve_mean_ms": statistics.mean(sampled),
        "rss_sampler_observed_overhead_ms": statistics.mean(sampled) - statistics.mean(direct),
        "rss_sampler_interval_ms": 0.5,
        "energy_snapshot_read_mean_us": statistics.mean(snap_us),
        "energy_snapshot_read_p95_us": percentile(snap_us, .95),
        "note": "overhead observation on this host; sampler thread scheduling and sysfs cost are environment-dependent",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/m13_measurement")
    args = ap.parse_args(); out = Path(args.out); raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(2, os.cpu_count() or 1)); set_seed(SEED, deterministic=True)

    rng = np.random.default_rng(SEED)
    tx, ty, _, _ = build_sudoku_arrays(2, 512, 8, rng, require_unique=True, augment=True)
    hx, hy, _, _ = build_sudoku_arrays(2, 96, 8, rng, require_unique=True, augment=True)
    tx = torch.from_numpy(tx).long(); ty = torch.from_numpy(ty).long()
    hx = torch.from_numpy(hx).long(); hy = torch.from_numpy(hy).long()
    model = build_model(); train = train_model(model, tx, ty)
    state_sha = module_tensor_state_sha256(model)
    write_json(raw / "training_excluded.json", train)

    artifact_path = raw / "m13_cpu_artifact.pt"
    t_export = time.perf_counter()
    export_cpu_artifact(model, artifact_path, height=4, width=4, box=2,
        source_checkpoint_sha256=state_sha, source_checkpoint_tensor_sha256=state_sha,
        training_seed=SEED, training_step=TRAIN_STEPS,
        data_provenance={"task": "generated_4x4_sudoku", "seed": SEED,
                         "heldout_reference_targets_used_for_timing": False},
        export_git_sha=os.environ.get("GITHUB_SHA", "local"))
    export_seconds = time.perf_counter() - t_export

    # Cold start: artifact load + runtime construction + native build/load on first
    # linear + one complete solve/semantic decision. CI uses a fresh TORCH_EXTENSIONS_DIR.
    t0 = time.perf_counter_ns()
    loaded = load_cpu_artifact(artifact_path); runtime = CPURecursiveRuntime(loaded)
    cold_result, cold_success = full_solve(runtime, hx[:1])
    cold_ms = (time.perf_counter_ns() - t0) / 1e6
    cold = {"cold_start_ms": cold_ms, "includes": ["artifact_load_and_validation",
        "runtime_construction", "native_extension_build_or_load", "complete_sequential_solve",
        "argmax_decode", "semantic_sudoku_validation"], "success": cold_success,
        "work": cold_result.work, "export_seconds_excluded": export_seconds}
    write_json(raw / "cold_start.json", cold)

    backend = runtime.backend_report()
    env = measurement_environment(backend=backend, compiler_flags=backend.get("compile_flags"), extra={
        "git_sha": os.environ.get("GITHUB_SHA", "local"),
        "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR"),
        "measurement_threads_requested": min(2, os.cpu_count() or 1),
        "workload": "CPURecursiveRuntime actual sequential recurrence B=1",
    })
    write_json(raw / "environment.json", env)
    inventory = discover_energy_domains().record(); write_json(raw / "energy_inventory.json", inventory)

    for i in range(1, 1 + WARMUP_N): full_solve(runtime, hx[i:i+1])
    representative = list(range(8, 8 + REPRESENTATIVE_N))
    rep_x = hx[representative]
    rep_hash = tensor_sha256(rep_x)
    timing_rows: list[dict[str, Any]] = []
    for rep in range(TIMING_REPS):
        order = representative[rep:] + representative[:rep]
        for order_index, idx in enumerate(order):
            xi = hx[idx:idx+1]
            t0 = time.perf_counter_ns(); result, success = full_solve(runtime, xi)
            latency_ms = (time.perf_counter_ns() - t0) / 1e6
            structural = float(sudoku_score(xi, result.answer, box=2)[0])
            macs = int(result.work["native_scalar_products"])
            timing_rows.append({"example_index": idx, "round": rep, "order_index": order_index,
                "workload": "actual_sequential_recurrence_complete_solve", "batch_size": 1,
                "latency_ms": latency_ms, "success": int(success), "structural_score": structural,
                "native_linear_macs": macs, "native_linear_operations": 2 * macs,
                "operation_convention": "1_MAC_equals_2_arithmetic_operations",
                "full_solve_native_linear_equivalent_gops": (2 * macs) / (latency_ms / 1000.0) / 1e9,
                "includes_semantic_validation": True})
    with (raw / "full_solve_timings.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(timing_rows[0])); w.writeheader(); w.writerows(timing_rows)
    lat = [r["latency_ms"] for r in timing_rows]
    timing_summary = {"distinct_heldout_examples": REPRESENTATIVE_N, "timed_rows": len(timing_rows),
        "heldout_input_tensor_sha256": rep_hash, "warmup_distinct_examples": WARMUP_N,
        "mean_ms": statistics.mean(lat), "median_ms": statistics.median(lat),
        "p95_ms": percentile(lat, .95), "min_ms": min(lat), "max_ms": max(lat),
        "cold_start_ms": cold_ms, "workload": "actual_sequential_recurrence_complete_solve",
        "inference_mode": True, "batch_size": 1, "reference_targets_used": False}
    write_json(raw / "full_solve_summary.json", timing_summary)

    # Physical energy is one repeated window over all 24 distinct solves. No idle
    # subtraction. Invalid/unavailable counters remain null with explicit reason.
    def energy_pass():
        for idx in representative: full_solve(runtime, hx[idx:idx+1])
    physical = measure_energy(energy_pass, n_runs=ENERGY_PASSES)
    physical["distinct_examples_per_pass"] = REPRESENTATIVE_N
    physical["total_complete_solves"] = REPRESENTATIVE_N * ENERGY_PASSES
    physical["joules_per_complete_solve"] = (
        physical["energy_joules"] / physical["total_complete_solves"]
        if physical["available"] else None)
    physical["idle_subtraction_performed"] = False
    write_json(raw / "physical_energy.json", physical)

    def representative_memory_pass():
        for idx in representative: full_solve(runtime, hx[idx:idx+1])
    memory = measure_peak_ram(representative_memory_pass, interval_s=0.0005)
    memory.update({"packed_weights_bytes": packed_weight_bytes(loaded),
                   "full_model_state_bytes": model_state_bytes(model),
                   "process_rss_current_mb_after_measurement": process_rss_mb(),
                   "linux_process_lifetime_hwm_mb": linux_process_hwm_mb(),
                   "packed_weight_scope": "packed ternary tensors only",
                   "full_model_state_scope": "parameters plus persistent buffers actual dtypes"})

    # Search-tree storage is measured on the real node structure but random auxiliary
    # modules: it is a memory-structure observation, not a search-quality result.
    verifier = LatentEnergyVerifier(num_tokens=5, dim=model.dim, n_layers=1, max_grid_size=8).eval()
    codebook = LatentActionCodebook(model.dim, n_actions=3).eval()
    mcts = LatentNativeMCTS(model.eval(), verifier, codebook, 4, 4, n_rollouts=8)
    with torch.inference_mode(): mcts.search(hx[representative[0]:representative[0]+1])
    tree_mem = search_tree_tensor_bytes(mcts.root)
    tree_mem.update({"measurement_kind": "controlled_reference_tree_structure_memory",
                     "trained_reasoner": True, "trained_search_auxiliaries": False,
                     "quality_claim": False})
    memory["search_tree"] = tree_mem; write_json(raw / "memory.json", memory)

    overhead = measure_instrumentation_overhead(runtime, hx[representative[1]:representative[1]+1])
    write_json(raw / "instrumentation_overhead.json", overhead)

    seq_result, _ = full_solve(runtime, hx[representative[2]:representative[2]+1])
    traffic = modelled_sequential_traffic(runtime, seq_result.work)
    traffic["representative_heldout_input_sha256"] = tensor_sha256(hx[representative[2]:representative[2]+1])
    write_json(raw / "sequential_recurrence.json", traffic)
    with (raw / "sequential_recurrence.csv").open("w", newline="") as f:
        flat = {k: v for k, v in traffic.items() if not isinstance(v, dict)}
        w = csv.DictWriter(f, fieldnames=list(flat)); w.writeheader(); w.writerow(flat)

    summary = {
        "milestone": 13, "status": "measurement_raw_ready",
        "primary_workload": "actual_sequential_recurrence_complete_solve",
        "operation_convention": "1_MAC_equals_2_arithmetic_operations",
        "timing": timing_summary,
        "physical_energy": {k: physical[k] for k in ("available", "energy_joules",
            "joules_per_complete_solve", "failure_reason", "scope", "package_domain_ids")},
        "memory": {"packed_weights_bytes": memory["packed_weights_bytes"],
            "full_model_state_bytes": memory["full_model_state_bytes"],
            "process_rss_current_mb": memory["process_rss_current_mb_after_measurement"],
            "process_rss_peak_sampled_mb": memory["process_rss_peak_sampled_mb"],
            "linux_process_lifetime_hwm_mb": memory["linux_process_lifetime_hwm_mb"],
            "search_tree_tensor_storage_bytes": tree_mem["tensor_storage_bytes"]},
        "instrumentation_overhead": overhead,
        "traffic_claims": {"cache_residency_established": False,
            "bandwidth_bottleneck_established": False,
            "scope": traffic["traffic_scope"]},
        "raw_files": {},
    }
    for p in sorted(raw.iterdir()):
        if p.is_file(): summary["raw_files"][p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    write_json(out / "measurement_summary.json", summary)
    print(json.dumps(plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
