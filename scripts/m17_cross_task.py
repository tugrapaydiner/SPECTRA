#!/usr/bin/env python3
"""Frozen M17 cross-task study, entirely on CPU.

Exit 0 means the preregistered experiment ran with valid evidence, NOT that its
scientific hypothesis passed. The explicit summary status and family gates are
mandatory when reporting outcomes. No failed development family opens confirmation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from common import load_config
from data.ancestry import ArtifactAncestry, ManifestSource, digest
from deploy.m10_native import load_extension
from eval.checkable_tasks import MAZE11, SUDOKU_SHIFT, TaskSpec, action_directions, require_core, semantic_exit
from eval.fixed_pool import GATE
from eval.measurement_contracts import validate_timing_rows
from eval.verified_search import BudgetedVerifiedSearch, ValueContract, ValueTarget
from scripts._common import build_data_splits
from scripts.m16_cpu_experiment import setup_cpu, source_identity
from scripts.m16_evidence import CORE_SHA, Evidence, write_json
from scripts.m17_models import MAZE_SEEDS, candidate_pool, fit_maze_value, fixed_surface, tensor_digest, train_maze_core
from scripts.m17_sources import AcceptedM16, TARGETS

PROTOCOL_COMMIT = "fe15894d1d57afb99a9235462c76acde3fa28d89"
FAMILIES = {
    "sudoku_shift": {"development_seed": 2026091703, "confirmation_seed": 2026091704,
                     "sizes": (0, 64, 128), "spec": SUDOKU_SHIFT},
    "maze": {"development_seed": 2026091701, "confirmation_seed": 2026091702,
             "sizes": (1024, 128, 128), "spec": MAZE11},
}
UNIFORM_SHA = digest(b"spectra.m17.uniform_priority.constant_0.5.v1")


def fresh_data(sources, out: Path, family: str, *, confirmation=False):
    declared = FAMILIES[family]
    seed = declared["confirmation_seed" if confirmation else "development_seed"]
    if family == "sudoku_shift":
        cfg = load_config("config/sudoku.yaml", overrides=[f"seed={seed}", "device=cpu",
            "data.box=2", "data.num_tokens=5", "data.seq_len=16", "data.height=4", "data.width=4",
            "data.min_clues=4", "data.max_clues=5", "data.require_unique=true", "data.augment=false",
            "data.solution_method=random_backtracking"])
    else:
        cfg = load_config("config/maze.yaml", overrides=[f"seed={seed}", "device=cpu",
            "data.height=11", "data.width=11", "data.seq_len=121", "data.num_tokens=5",
            "data.min_path_len=8", "data.max_path_len=null", "data.require_optimal=true", "data.augment=false"])
    counts = (0, 0, 256) if confirmation else declared["sizes"]
    index = sources.exclusion()
    datasets, manifest = build_data_splits(cfg, *counts, seed=seed,
        forbidden_fingerprints=index.forbidden, require_unique_examples=True)
    if any(any(d.values()) for d in manifest["duplicate_audit"].values()):
        raise ValueError("new dataset has within/cross-split duplicates")
    raw = (json.dumps(manifest, indent=2, sort_keys=True)+"\n").encode()
    manifest_path = out/f"manifests/seed{seed}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("xb") as f:
        f.write(raw)
    source = ManifestSource.from_bytes(str(manifest_path), raw,
        role="previous_confirmation" if confirmation else "development", expected_sha256=digest(raw))
    audit = index.require_disjoint(source)
    write_json(out/f"manifests/seed{seed}_audit.json", audit)
    np.savez_compressed(out/f"manifests/seed{seed}_arrays.npz",
        **{f"{split}_{field}": getattr(ds, field) for split, ds in datasets.items() for field in ("inputs", "targets")})
    sources.additional.append(source)
    return datasets, source, manifest_path


@torch.inference_mode()
def checked_search(core, value_model, contract, x, spec: TaskSpec, core_sha):
    require_core(core, spec)
    problem = spec.native_problem(x)  # BFS/constraint setup is inside the caller's timed solve.
    delta = action_directions(spec.dim)
    context = {}

    def initial():
        context["xemb"] = core.token_embed(x)+core.encode_positions(x, spec.height, spec.width)
        return torch.zeros_like(context["xemb"]), torch.zeros_like(context["xemb"])

    def transition(state, action):
        return core.recursive_cycle(context["xemb"], state[0], state[1]+delta[action])

    # The native decoder has already checked this exact returned tensor. Passing
    # the pair through the generic engine reuses that decision, rather than
    # running an uncharged or duplicate validator. Independent checks follow the
    # timed solve in evaluate_closed_loop. These flags never come from a model.
    def decode(state):
        answer, valid = problem.decode(core.out_head(state[0]).contiguous())
        return answer, bool(valid)

    def value(state):
        if value_model is None:
            return .5
        return float(value_model.value_state(x, state[0], state[1], width=spec.width).item())

    engine = BudgetedVerifiedSearch(initial=initial, actions=[0, 1, 2, 3], transition=transition,
        decode=decode, checker=lambda pair: pair[1], value=value, contract=contract,
        model_sha256=core_sha, transition_id=spec.transition_id, state_schema=spec.state_schema)
    result = engine.solve(max_transitions=24, max_depth=4, identity_prefix=4)
    answer = None if result.answer is None else result.answer[0]
    work = {**result.work, "valid": result.valid, "checker_constructions": 1, "input_embeddings": 1,
            "native_exact_check_in_decode": True, "charged_identity_prefix": 4,
            "native_semantic_checks": result.work["decodes"],
            "block_applications": result.work["transitions"]*2*len(core.blocks)}
    return answer, work


@torch.inference_mode()
def symbolic_solve(x, spec, native=False):
    problem = spec.native_problem(x)
    answer = problem.shortest_solution() if native else spec.symbolic_answer(x)
    valid = bool(problem.check(answer))
    return answer, {"valid": valid, "transitions": 0, "decodes": 0, "checks": 1,
                    "checker_constructions": 1, "input_embeddings": 0, "value_calls": 0,
                    "symbolic_solver": "native_bfs" if native else ("python_bfs" if spec.task == "maze" else "python_mrv"),
                    "bfs_preprocessing_charged": spec.task == "maze", "target_used": False}


def closed_summary(rows):
    validate_timing_rows(rows, seed_key="core_seed", example_key="example_id",
                         required_arms=("native_k4",))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["core_seed"], row["example_id"], row["arm"])].append(row)
    collapsed = []
    for key, rr in groups.items():
        if {r["round"] for r in rr} != {0, 1, 2} or len(rr) != 3:
            raise ValueError("closed-loop timing group must have exactly three distinct rounds")
        if any(r["answer"] != rr[0]["answer"] or r["valid"] != rr[0]["valid"] for r in rr[1:]):
            raise ValueError("deterministic solve answers changed across timing rounds")
        collapsed.append({**rr[0], "latency_ms": float(np.median([r["latency_ms"] for r in rr]))})
    baseline = {(r["core_seed"], r["example_id"]): r for r in collapsed if r["arm"] == "native_k4"}
    summary = {}
    for arm in sorted({r["arm"] for r in collapsed}):
        selected = [r for r in collapsed if r["arm"] == arm]
        if {(r["core_seed"], r["example_id"]) for r in selected} != set(baseline):
            raise ValueError("closed-loop arms are not paired on the same model-examples")
        times = np.asarray([r["latency_ms"] for r in selected])
        regressions = sum(baseline[(r["core_seed"], r["example_id"])]["valid"] and not r["valid"] for r in selected)
        newly_solved = sum(r["valid"] and not baseline[(r["core_seed"], r["example_id"])]["valid"] for r in selected)
        if arm.startswith("checked_") or arm == "identity_24":
            if regressions:
                raise AssertionError("charged identity continuation/prefix lost a valid baseline answer")
        raw_times = [r["latency_ms"] for r in rows if r["arm"] == arm]
        summary[arm] = {"model_example_pairs": len(selected), "timing_rows": len(raw_times),
            "strict_success": sum(r["valid"] for r in selected)/len(selected),
            "valid_answers": sum(r["valid"] for r in selected), "new_solves_vs_native_k4": newly_solved,
            "regressions_vs_native_k4": regressions, "mean_ms_after_round_medians": float(times.mean()),
            "median_ms_after_round_medians": float(np.median(times)),
            "p95_ms_after_round_medians": float(np.quantile(times, .95)),
            "raw_mean_ms": float(np.mean(raw_times)), "raw_median_ms": float(np.median(raw_times)),
            "raw_p95_ms": float(np.quantile(raw_times, .95)),
            "mean_transitions": float(np.mean([r["work"]["transitions"] for r in selected])),
            "mean_value_calls": float(np.mean([r["work"]["value_calls"] for r in selected]))}
    return {"arms": summary, "unique_examples": len({r["example_id"] for r in collapsed}),
            "core_seeds": sorted({r["core_seed"] for r in collapsed}), "rounds": 3,
            "scope": "descriptive complete-solve CPU controls; not the fixed-pool confirmation gate",
            "physical_energy_joules": None, "energy_reason": "not_measured"}


def evaluate_closed_loop(cores, values, contracts, core_shas, dataset, warmup, spec, out):
    rows = []
    rng = np.random.default_rng(17092)
    with (out/"development_closed_rows.jsonl").open("x") as stream:
        for seed, core in cores.items():
            uniform = ValueContract(ValueTarget.QUALITY, core_shas[seed], UNIFORM_SHA,
                                   spec.transition_id, state_schema=spec.state_schema)
            solvers = {"reference_k4": lambda x: semantic_exit(core, x, spec, 4, native=False),
                       "native_k4": lambda x: semantic_exit(core, x, spec, 4, native=True),
                       "identity_24": lambda x: semantic_exit(core, x, spec, 24, native=True),
                       "checked_quality": lambda x: checked_search(core, values[seed][ValueTarget.QUALITY],
                            contracts[seed][ValueTarget.QUALITY], x, spec, core_shas[seed]),
                       "checked_terminal": lambda x: checked_search(core, values[seed][ValueTarget.TERMINAL],
                            contracts[seed][ValueTarget.TERMINAL], x, spec, core_shas[seed]),
                       "checked_uniform": lambda x: checked_search(core, None, uniform, x, spec, core_shas[seed]),
                       "symbolic_python": lambda x: symbolic_solve(x, spec)}
            if spec.task == "maze":
                solvers["symbolic_native_bfs"] = lambda x: symbolic_solve(x, spec, native=True)
            for i in range(min(4, len(warmup))):
                xi = torch.from_numpy(warmup.inputs[i:i+1]).long()
                for solve in solvers.values():
                    solve(xi)
            for repeat in range(3):
                for i, example_id in enumerate(dataset.ids):
                    x = torch.from_numpy(dataset.inputs[i:i+1]).long()
                    parity = {}
                    for arm in rng.permutation(list(solvers)):
                        t0 = time.perf_counter_ns()
                        answer, work = solvers[arm](x)
                        elapsed = (time.perf_counter_ns()-t0)/1e6
                        valid = spec.independent_correct(x, answer)
                        if valid != bool(work["valid"]):
                            raise AssertionError("native or model-side decision disagrees with independent checker")
                        if arm in ("reference_k4", "native_k4"):
                            parity[arm] = (answer.clone(), work["transitions"])
                        row = {"core_seed": seed, "example_id": example_id, "round": repeat, "arm": str(arm),
                               "valid": valid, "latency_ms": elapsed, "work": work,
                               "answer": None if answer is None else answer.flatten().tolist(),
                               "reference_target_used_inside_solver": False}
                        rows.append(row)
                        stream.write(json.dumps(row, allow_nan=False)+"\n")
                    a, b = parity["native_k4"], parity["reference_k4"]
                    if a[1] != b[1] or not torch.equal(a[0], b[0]):
                        raise AssertionError("native/reference semantic-exit parity failed")
                stream.flush()
                print(f"closed-loop task={spec.task} core={seed} round={repeat} complete", flush=True)
    summary = closed_summary(rows)
    write_json(out/"development_closed_summary.json", summary)
    return summary


def register_maze_sources(sources, source, manifest_path, core_shas, value_records):
    training = ManifestSource.read(manifest_path, role="training", expected_sha256=source.sha256, splits=("train",))
    for seed, core_sha in core_shas.items():
        sources.registry[core_sha] = ArtifactAncestry(core_sha, manifests=(training,))
        sources.roots.append(core_sha)
        for record in value_records[seed]:
            sha = record["sha256"]
            sources.registry[sha] = ArtifactAncestry(sha, parents=(core_sha,), manifests=(training,))
            sources.roots.append(sha)


def run_family(family, sources, historical, out, frozen_source):
    spec = FAMILIES[family]["spec"]
    out.mkdir(exist_ok=False)
    datasets, manifest_source, manifest_path = fresh_data(sources, out, family)
    cores, values, contracts, core_shas, source_records, value_records = {}, {}, {}, {}, {}, {}
    if family == "sudoku_shift":
        for seed in CORE_SHA:
            cores[seed], core_shas[seed] = historical.load_model("fp_recursive_dim64", seed, out/"checkpoints")
            if sources.member_sha(f"experiment/source_checkpoints/fp_recursive_dim64_seed{seed}.pt") != core_shas[seed]:
                raise ValueError("M16 did not use the pinned M14 core")
            values[seed], contracts[seed], value_records[seed] = sources.load_values(seed, out/"checkpoints")
            source_records[seed] = {"sha256": core_shas[seed], "trained_here": False}
    else:
        for seed in MAZE_SEEDS:
            core, sha, record = train_maze_core(seed, datasets, manifest_source.sha256, out/"checkpoints")
            cores[seed], core_shas[seed], source_records[seed] = core, sha, record
            # The protocol names the first 256 training inputs, not a chosen subset.
            pool = candidate_pool(core, torch.from_numpy(datasets["train"].inputs[:256]).long(), spec)
            pool_sha = tensor_digest(pool)
            values[seed], contracts[seed], value_records[seed] = {}, {}, []
            for index, target in enumerate(TARGETS):
                model, contract, fitting = fit_maze_value(pool, index, seed, sha, manifest_source.sha256, out/"checkpoints")
                fitting["training_pool_sha256"] = pool_sha
                values[seed][target], contracts[seed][target] = model, contract
                value_records[seed].append(fitting)
            recs = value_records[seed]
            if len({r["initial_state_sha256"] for r in recs}) != 1 or len({r["sampled_index_sha256"] for r in recs}) != 1:
                raise AssertionError("matched-target initialization/minibatch contract failed")
            del pool
        register_maze_sources(sources, manifest_source, manifest_path, core_shas, value_records)
    write_json(out/"model_sources_and_training.json", {"cores": source_records, "values": value_records})
    fixed = fixed_surface(cores, values, datasets["test"], spec, "development", out)
    print(f"fixed-pool family={family} coverage={fixed['coverage']:.5f} effect={fixed['quality_minus_improvement']:.5f} pass={fixed['gate_pass']}", flush=True)
    closed = evaluate_closed_loop(cores, values, contracts, core_shas, datasets["test"], datasets["validation"], spec, out)
    summary = {"task": spec.metadata(), "development_fixed_pool": fixed, "development_closed_loop": closed,
               "confirmation_opened": False, "confirmation_gate_pass": False,
               "status": "DEVELOPMENT_GATE_NOT_MET", "source_records": source_records}
    if fixed["gate_pass"]:
        if source_identity()["source_sha256"] != frozen_source["source_sha256"]:
            raise RuntimeError("executable source changed during the scientific run")
        freeze = {"protocol_commit": PROTOCOL_COMMIT, "source": frozen_source,
                  "primary_rule": "learned_current_structural_quality_minus_learned_one_cycle_improvement",
                  "gate": dict(GATE), "model_shas": core_shas,
                  "value_checkpoints": {s: [r["sha256"] for r in rr] for s, rr in value_records.items()},
                  "decode_and_task": spec.metadata(), "development_manifest_sha256": manifest_source.sha256,
                  "development_gate": fixed, "no_post_confirmation_selection": True}
        write_json(out/"confirmation_freeze.json", freeze)
        confirmation, _, _ = fresh_data(sources, out, family, confirmation=True)
        confirmed = fixed_surface(cores, values, confirmation["test"], spec, "confirmation", out)
        summary.update(confirmation_opened=True, confirmation_fixed_pool=confirmed,
                       confirmation_gate_pass=confirmed["gate_pass"],
                       status="CONFIRMED_FIXED_POOL_EFFECT" if confirmed["gate_pass"] else "CONFIRMATION_GATE_NOT_MET")
    write_json(out/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-m16", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        environment = setup_cpu()
        frozen_source = source_identity()
        write_json(args.out/"environment.json", environment)
        write_json(args.out/"source_identity.json", frozen_source)
        historical = Evidence(args.evidence_dir)
        sources = AcceptedM16(args.accepted_m16, historical)
        write_json(args.out/"accepted_m16_identity.json", sources.identity())
        t0 = time.perf_counter_ns()
        load_extension()
        write_json(args.out/"cold_native_setup.json", {"compile_or_load_ms": (time.perf_counter_ns()-t0)/1e6,
            "scope": "extension build/load only; per-input problem construction remains inside solve timing"})
        results = {}
        for family in FAMILIES:
            results[family] = run_family(family, sources, historical, args.out/family, frozen_source)
            write_json(args.out/"partial_status.json", {k: r["status"] for k, r in results.items()})
        passed = all(r["confirmation_gate_pass"] for r in results.values())
        if source_identity()["source_sha256"] != frozen_source["source_sha256"]:
            raise RuntimeError("executable source changed during the scientific run")
        write_json(args.out/"summary.json", {"milestone": 17, "protocol_commit": PROTOCOL_COMMIT,
            "execution_status": "COMPLETE", "scientific_status": "TWO_FAMILY_FIXED_POOL_EFFECT_CONFIRMED" if passed else "TWO_FAMILY_CLAIM_NOT_ESTABLISHED",
            "two_family_gate_pass": passed, "families": results, "device": "cpu",
            "source_identity": frozen_source, "reference_targets_used_in_search": False,
            "limitations": ["small separately trained task models; not zero-shot transfer",
                            "two core seeds per family", "exact/group exclusion is not symmetry-family exclusion",
                            "no energy, general learned-search-superiority or hiring claim"]})
        return 0
    except Exception as exc:
        write_json(args.out/"execution_failure.json", {"execution_status": "FAILED", "error_type": type(exc).__name__,
                   "error": str(exc), "traceback": traceback.format_exc(), "no_claim_of_scientific_pass": True})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
