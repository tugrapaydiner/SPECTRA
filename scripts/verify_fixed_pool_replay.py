#!/usr/bin/env python3
"""Replay every retained M17 pool from frozen CPU checkpoints and input arrays.

Already-published Sudoku confirmation is replayed, not regenerated or reused for
model selection. The unopened maze confirmation stays unopened. No fit, parameter
update, new dataset, inference-speed claim, or promotion of a failed gate occurs.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback

# Dispatch is selected before importing Torch or NumPy. This is a reproduction
# profile for the retained AMD EPYC 7763 run, NOT a performance recommendation.
# --cpu-profile host leaves dispatch untouched and can expose numerical drift.
HISTORICAL_AVX2_ENV = {
    "ATEN_CPU_CAPABILITY": "avx2", "MKL_ENABLE_INSTRUCTIONS": "AVX2",
    "ONEDNN_MAX_CPU_ISA": "AVX2", "DNNL_MAX_CPU_ISA": "AVX2",
}
if __name__ == "__main__":
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--cpu-profile", choices=("historical-avx2", "host"), default="historical-avx2")
    profile, _ = early.parse_known_args()
    if profile.cpu_profile == "historical-avx2":
        if platform.machine().lower() not in {"x86_64", "amd64"}:
            early.error("the historical AVX2 profile requires an x86-64 CPU")
        os.environ.update(HISTORICAL_AVX2_ENV)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

from data.ancestry import digest
from data.splits import example_fingerprint
from eval.checkable_tasks import MAZE11, SUDOKU_SHIFT
from eval.fixed_pool import summarize_pools
from eval.fixed_pool_replay import compare_records, reconstruct_pool, replay_rows
from eval.verified_search import ValueTarget
from model.task_value import load_task_value
from scripts.m16_cpu_experiment import source_identity
from scripts.m16_evidence import Evidence, write_json
from scripts.m17_models import load_maze_core, tensor_digest
from scripts.m17_sources import AcceptedM16
from scripts.verify_retained_results import M17_INVENTORY_SHA, M17_ZIP_SHA, json_rows, read_bound_archive, verify_m17

FAMILY_SPECS = {"sudoku_shift": (SUDOKU_SHIFT, 2026091703, 2026091704),
                "maze": (MAZE11, 2026091701, 2026091702)}


def frozen_inputs(members: dict[str, bytes], prefix: str, seed: int, spec):
    """Bind IDs and consumed inputs to the original manifest and target bytes.

    Reference targets are used ONLY to validate the existing content fingerprint,
    then discarded. They are never passed to reconstruction or the value heads.
    """
    name = prefix + f"manifests/seed{seed}"
    raw = members[name + ".json"]
    manifest = json.loads(raw)
    if manifest["seed"] != seed or manifest["task"] != spec.task:
        raise ValueError("retained manifest task/seed identity mismatch")
    entry = manifest["splits"]["test"]
    rows = entry["examples"]
    ids = [row["id"] for row in rows]
    if (type(entry["count"]) is not int or entry["count"] != len(rows) or not rows
        or any(not isinstance(eid, str) or not eid for eid in ids) or len(set(ids)) != len(ids)):
        raise ValueError("retained manifest has missing, duplicate or malformed examples")
    with np.load(io.BytesIO(members[name + "_arrays.npz"]), allow_pickle=False) as arrays:
        inputs, targets = arrays["test_inputs"], arrays["test_targets"]
        shape = (len(rows), spec.height * spec.width)
        if any(a.dtype != np.int64 or a.shape != shape for a in (inputs, targets)):
            raise ValueError("retained input/target array contract mismatch")
        for row, x, y in zip(rows, inputs, targets):
            if example_fingerprint(spec.task, x, y, spec.height, spec.width) != row["fingerprint"]:
                raise ValueError("retained input/target fingerprint mismatch")
        tensor = torch.from_numpy(inputs.copy())
    return tensor, ids, digest(raw)


def load_sources(family, members, historical, accepted, out):
    """Strict-load only the archive-bound core/value slots for this family."""
    spec, seed, _ = FAMILY_SPECS[family]
    prefix = f"experiment/{family}/"
    record = json.loads(members[prefix + "model_sources_and_training.json"])
    training_sha = digest(members[prefix + f"manifests/seed{seed}.json"])
    expected_seeds = {1401, 2402} if family == "sudoku_shift" else {1701, 2702}
    if set(record["cores"]) != {str(s) for s in expected_seeds}:
        raise ValueError("retained core-seed inventory mismatch")
    result = {}
    for core_seed in sorted(expected_seeds):
        saved = record["cores"][str(core_seed)]
        core_sha = saved["sha256"]
        work = out / f"{family}-{core_seed}"
        work.mkdir(parents=True)
        if family == "sudoku_shift":
            core, actual_sha = historical.load_model("fp_recursive_dim64", core_seed, work)
            if actual_sha != core_sha:
                raise ValueError("retained Sudoku core identity mismatch")
            values, _, _ = accepted.load_values(core_seed, work / "values")
        else:
            name = f"maze_core_{core_seed}.pt"
            path = work / name
            path.write_bytes(members[prefix + "checkpoints/" + name])
            core = load_maze_core(path, expected_sha=core_sha, expected_seed=core_seed, manifest_sha=training_sha)
            values = {}
            for vr in record["values"][str(core_seed)]:
                target = ValueTarget(vr["target"])
                if target in values:
                    raise ValueError("duplicate retained value target")
                path = work / Path(vr["path"]).name
                path.write_bytes(members[prefix + "checkpoints/" + path.name])
                values[target], _ = load_task_value(path, expected_sha256=vr["sha256"],
                    expected_core_sha256=core_sha, expected_training_manifest_sha256=training_sha,
                    spec=spec, diagnostic_improvement=target is ValueTarget.IMPROVEMENT)
        result[core_seed] = (core, values)
    return result


def verify(args) -> dict:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("fixed-pool replay requires CPU-only PyTorch")
    historical = Evidence(args.historical_dir)
    accepted = AcceptedM16(args.m16, historical)
    members = read_bound_archive(args.m17, inventory_sha=M17_INVENTORY_SHA, zip_sha=M17_ZIP_SHA)
    # Independent inventory/ancestry/scientific-status checks precede inference.
    existing = verify_m17(members, accepted.exclusion().forbidden)
    report = {"status": "RUNNING", "families": {}, "model_example_pools": 0,
              "candidate_states": 0, "head_scores_compared": 0, "head_scores_exact": 0,
              "max_score_absolute_error": 0., "pool_tensor_hashes_checked": 0,
              "all_pool_tensor_hashes_exact": True, "independent_candidate_decodes": 0,
              "independent_continuation_decodes": 0, "new_confirmation_generated": False,
              "training_updates": 0, "source": source_identity(),
              "m17_container_sha256": digest(args.m17.read_bytes()),
              "m17_inventory_sha256": M17_INVENTORY_SHA, "m16_identity": accepted.identity(),
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "torch": str(torch.__version__), "numpy": np.__version__,
                              "device": "cpu", "torch_threads": torch.get_num_threads(),
                              "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                              "dispatch_environment": {k: os.environ.get(k) for k in HISTORICAL_AVX2_ENV},
                              "cpu_model": next((line.split(":", 1)[1].strip() for line in
                                  Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown")
                                  if Path("/proc/cpuinfo").exists() else platform.processor()},
              "scientific_status": existing["scientific_status"]}
    with tempfile.TemporaryDirectory(prefix="spectra-pool-replay-") as temp:
        for family, (spec, development_seed, confirmation_seed) in FAMILY_SPECS.items():
            prefix = f"experiment/{family}/"
            sources = load_sources(family, members, historical, accepted, Path(temp))
            family_summary = json.loads(members[prefix + "summary.json"])
            family_out = {}
            for surface, seed in (("development", development_seed), ("confirmation", confirmation_seed)):
                name = prefix + surface + "_pools.jsonl"
                if name not in members:
                    if surface != "confirmation" or family_summary["confirmation_opened"]:
                        raise ValueError("required retained fixed pools are missing")
                    family_out[surface] = {"status": "UNOPENED_NOT_GENERATED"}
                    continue
                x, ids, manifest_sha = frozen_inputs(members, prefix, seed, spec)
                recorded = json_rows(members[name])
                declared = family_summary[surface + "_fixed_pool"]
                actual = []
                diagnostics = {}
                for core_seed, (core, values) in sources.items():
                    pool, diagnostic = reconstruct_pool(core, x, spec)
                    observed_hash = tensor_digest(pool)
                    expected_hash = declared["pool_tensor_sha256_by_core"][str(core_seed)]
                    if observed_hash != expected_hash:
                        raise ValueError(f"fixed-pool tensor hash mismatch: {family}/{surface}/{core_seed}; "
                                         f"observed={observed_hash} expected={expected_hash}")
                    report["pool_tensor_hashes_checked"] += 1
                    replayed = replay_rows(core_seed, ids, pool, values, spec, surface)
                    actual.extend(replayed)
                    diagnostics[str(core_seed)] = {"tensor_sha256": observed_hash, **diagnostic}
                    report["independent_candidate_decodes"] += diagnostic["candidate_decodes"]
                    report["independent_continuation_decodes"] += diagnostic["continuation_decodes"]
                    del pool
                    print(f"replayed {family}/{surface}/core{core_seed}: {len(replayed)} pools", flush=True)
                comparison = compare_records(recorded, actual)
                recomputed = summarize_pools(actual)
                if any(declared.get(key) != value for key, value in recomputed.items()):
                    raise ValueError("inference-recomputed scientific gate differs from retained result")
                for field in ("model_example_pools", "candidate_states", "head_scores_compared", "head_scores_exact"):
                    report[field] += comparison[field]
                report["max_score_absolute_error"] = max(report["max_score_absolute_error"], comparison["max_score_absolute_error"])
                with (args.out / f"{family}_{surface}_rows.jsonl").open("x") as stream:
                    for row in actual:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                family_out[surface] = {**comparison, "manifest_sha256": manifest_sha,
                    "recorded_rows_sha256": digest(members[name]), "diagnostics": diagnostics,
                    "coverage": recomputed["coverage"], "quality_minus_improvement": recomputed["quality_minus_improvement"],
                    "gate_pass": recomputed["gate_pass"]}
            report["families"][family] = family_out
    report["status"] = "PASS"
    report["scope"] = ("Frozen input/core/value inference, independent NumPy restoration and targets, exact pool tensor hashes, "
                       "exact choices and gates; no training, new data, confirmation selection or timing claim")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-profile", choices=("historical-avx2", "host"), default="historical-avx2",
                        help="historical-avx2 pins CPU dispatch before imports; host exposes default numerical drift")
    parser.add_argument("--historical-dir", type=Path, default=Path("results/m16/sources"))
    parser.add_argument("--m16", type=Path, default=Path("results/m16/runs/accepted-34522192590.tar.gz"))
    parser.add_argument("--m17", type=Path, default=Path("results/m17/runs/34534009702-33e548d3e0f52d0664c00929316b2198d5fcffe4.tar.gz"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite a fixed-pool verification record")
    args.out.mkdir(parents=True)
    start = time.perf_counter()
    try:
        report = verify(args)
    except Exception as exc:
        write_json(args.out / "failure.json", {"status": "FAIL", "error_type": type(exc).__name__,
                   "error": str(exc), "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter()-start})
        raise
    report["elapsed_seconds"] = time.perf_counter()-start
    write_json(args.out / "summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"source", "families"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
