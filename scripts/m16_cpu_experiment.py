#!/usr/bin/env python3
"""M16 CPU fidelity/speed experiment; see docs/M16_EXPERIMENT_PROTOCOL.md."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from common.measurement_env import measurement_environment
from data.ancestry import ManifestSource, digest
from data import sudoku
from deploy.m10_native import load_extension
from deploy.semantic_exit import native_semantic_exit
from scripts._common import build_data_splits
from scripts.m14_primary_experiment import make_cfg, clamp_givens
from scripts.m14_attempt5_dual_stream_semantic_exit import dual_stream_semantic_exit_solve
from scripts.m16_evidence import Evidence, CORE_SHA, write_json

PROTOCOL_COMMIT = "a879107cf5e5fbf3a07003d254a32b0c71835aa6"


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    files = {str(p.relative_to(root)): digest(p.read_bytes()) for directory in
             ["common", "data", "deploy", "eval", "model", "scripts", "train"]
             for p in sorted((root/directory).rglob("*")) if p.suffix in {".py", ".cpp", ".h"}}
    return {"github_commit": os.environ.get("GITHUB_SHA"), "files": files,
            "source_sha256": digest(json.dumps(files, sort_keys=True).encode())}


def setup_cpu() -> dict:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {available[0]})
        affinity = sorted(os.sched_getaffinity(0))
    return {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
            "platform": platform.platform(), "affinity": affinity, "threads": torch.get_num_threads(),
            "device": "cpu", "package_energy_joules": None,
            "energy_unavailable_reason": "not_measured_in_this_experiment",
            "host_provenance": measurement_environment(
                backend="pytorch_fp32_plus_native_exact_checker",
                compiler_flags=(["/O2", "/std:c++20"] if sys.platform == "win32" else ["-O3", "-std=c++20"]))}


def fresh_data(evidence, out, *, seed, train, validation, test, additional=frozenset()):
    index = evidence.exclusion()
    datasets, manifest = build_data_splits(make_cfg(seed), train, validation, test, seed=seed,
                                          forbidden_fingerprints=index.forbidden | additional,
                                          require_unique_examples=True)
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    source = ManifestSource.from_bytes(f"m16_seed{seed}.json", raw, role="development", expected_sha256=digest(raw))
    audit = index.require_disjoint(source)
    if (source.fingerprints | source.groups) & additional:
        raise ValueError("new data overlap an earlier M16 split")
    write_json(out/f"manifests/seed{seed}.json", manifest)
    write_json(out/f"manifests/seed{seed}_audit.json", {**audit, "earlier_m16_disjoint": True,
                                                       "earlier_m16_exclusion_count": len(additional)})
    np.savez_compressed(out/f"manifests/seed{seed}_arrays.npz",
                        **{f"{split}_{field}": getattr(ds, field) for split, ds in datasets.items()
                           for field in ["inputs", "targets"]})
    return datasets, source


def effect(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["seed"], r["example_index"], r["arm"])].append(r["latency_ms"])
    seeds = sorted({r["seed"] for r in rows})
    ids = sorted({r["example_index"] for r in rows})
    ref = np.asarray([[np.median(by[(s, i, "reference")]) for i in ids] for s in seeds])
    cand = np.asarray([[np.median(by[(s, i, "native")]) for i in ids] for s in seeds])
    ratio = float(cand.mean()/ref.mean())
    rng = np.random.default_rng(16092)
    samples = []
    for _ in range(2000):
        si = rng.integers(len(seeds), size=len(seeds))
        ii = rng.integers(len(ids), size=len(ids))
        samples.append(float(cand[np.ix_(si, ii)].mean()/ref[np.ix_(si, ii)].mean()))
    lo, hi = np.quantile(samples, [.025, .975]).tolist()
    summary = {}
    for arm in sorted({r["arm"] for r in rows}):
        rr = [r for r in rows if r["arm"] == arm]
        times = np.asarray([r["latency_ms"] for r in rr])
        summary[arm] = {"timing_rows": len(rr), "valid_fraction": float(np.mean([r["valid"] for r in rr])),
                        "mean_ms": float(times.mean()), "median_ms": float(np.median(times)),
                        "p95_ms": float(np.quantile(times, .95))}
    return {"paired_mean_ratio_after_round_medians": ratio, "ci95": [lo, hi],
            "model_seeds": seeds, "unique_examples": len(ids), "crossed_bootstrap_replicates": 2000,
            "gate_pass": ratio <= .80 and hi < 1.0, "arms": summary,
            "claim_boundary": "faithful speedup over old implementation, not solver or learned-search superiority"}


@torch.inference_mode()
def timing_surface(cores, students, ds, out: Path, surface: str):
    extension = load_extension()
    xs = torch.from_numpy(ds.inputs).long()
    rng = np.random.default_rng(16091)
    rows = []
    path = out/f"{surface}_timings.jsonl"
    if path.exists():
        raise FileExistsError("refusing to overwrite an already observed experimental surface")
    with path.open("w") as stream:
        for seed, core in cores.items():
            student = students[seed]
            def single(x):
                checker = extension.SudokuProblem(x, 2)
                logits, _ = student(x, height=4, width=4)
                answer, valid = checker.decode(logits.contiguous())
                return answer, {"final_semantic": valid, "executed_steps": 1}
            def exact(x):
                checker = extension.SudokuProblem(x, 2)
                solved = sudoku.solve(x.numpy().reshape(4, 4), 2)
                if solved is None:
                    raise RuntimeError("unique Sudoku generator supplied an unsatisfiable input")
                answer = torch.from_numpy(solved.reshape(1, -1))
                return answer, {"final_semantic": checker.check(answer), "executed_steps": 0}
            arms = {"reference": lambda x: dual_stream_semantic_exit_solve(core, x, 4),
                    "native": lambda x: native_semantic_exit(core, x, 4),
                    "single_pass_native_checker": single, "symbolic_native_checker": exact}
            for x in xs[:4]:
                for solve in arms.values(): solve(x[None])
            for i, x in enumerate(xs):
                xi = x[None]
                for round_id in range(3):
                    observed = {}
                    for name in rng.permutation(list(arms)):
                        start = time.perf_counter_ns()
                        answer, work = arms[name](xi)
                        elapsed = (time.perf_counter_ns()-start)/1e6
                        a = answer.numpy().reshape(4, 4)
                        independent = sudoku.is_solved(a, 2) and sudoku.respects_clues(ds.inputs[i].reshape(4, 4), a, 2)
                        if bool(work["final_semantic"]) != independent:
                            raise AssertionError("native/Torch decision disagrees with independent NumPy checker")
                        observed[name] = (answer.clone(), work)
                        row = {"surface": surface, "seed": seed, "example_id": ds.ids[i], "example_index": i,
                               "round": round_id, "arm": str(name), "latency_ms": elapsed,
                               "valid": independent, "steps": int(work["executed_steps"])}
                        stream.write(json.dumps(row, allow_nan=False)+"\n")
                        rows.append(row)
                    before, after = observed["reference"], observed["native"]
                    if not torch.equal(before[0], after[0]) or before[1]["executed_steps"] != after[1]["executed_steps"]:
                        raise AssertionError("native optimization changed answer or recurrence execution")
    result = effect(rows)
    result.update(exact_answer_step_fidelity=True, raw_rows_sha256=digest(path.read_bytes()))
    write_json(out/f"{surface}_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", action="store_true", help="also run the separately specified target-development study")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    env = setup_cpu()
    write_json(args.out/"environment.json", env)
    write_json(args.out/"source_identity.json", source_identity())
    evidence = Evidence(args.evidence_dir)
    write_json(args.out/"retrospective_audit.json", evidence.retrospective())
    start = time.perf_counter_ns()
    load_extension()
    cores, students = {}, {}
    for seed in CORE_SHA:
        cores[seed], _ = evidence.load_model("fp_recursive_dim64", seed, args.out/"source_checkpoints")
        students[seed], _ = evidence.load_model("single_pass", seed, args.out/"source_checkpoints")
    write_json(args.out/"cold_setup.json", {"native_load_and_four_checkpoints_ms": (time.perf_counter_ns()-start)/1e6,
                                           "scope": "setup only; not a full cold first solve"})
    datasets, dev_source = fresh_data(evidence, args.out, seed=2026091601, train=256, validation=64, test=128)
    dev = timing_surface(cores, students, datasets["test"], args.out, "development")
    summary = {"protocol_commit": PROTOCOL_COMMIT, "development": dev, "confirmation_opened": False,
               "status": "DEVELOPMENT_GATE_FAILED", "scope": "bounded faithful CPU optimization"}
    if dev["gate_pass"]:
        freeze = {"protocol_commit": PROTOCOL_COMMIT, "core_hashes": CORE_SHA, "source": source_identity(),
                  "development": dev, "confirmation_seed": 2026091602, "reserve_seed_unopened": 2026091603}
        write_json(args.out/"confirmation_freeze.json", freeze)
        frozen_sha = digest((args.out/"confirmation_freeze.json").read_bytes())
        confirmation, _ = fresh_data(evidence, args.out, seed=2026091602, train=0, validation=0, test=256,
                                      additional=dev_source.fingerprints | dev_source.groups)
        conf = timing_surface(cores, students, confirmation["test"], args.out, "confirmation")
        summary.update(confirmation_opened=True, confirmation=conf, freeze_sha256=frozen_sha,
                       status="FIDELITY_SPEED_GATE_PASS" if conf["gate_pass"] else "CONFIRMATION_GATE_FAILED")
    write_json(args.out/"summary.json", summary)
    if args.targets:
        from scripts.m16_target_experiment import run_targets
        run_targets(cores, datasets, args.out)
    print(json.dumps(summary, indent=2))
    # Scientific negatives are retained outcomes, not integrity errors.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
