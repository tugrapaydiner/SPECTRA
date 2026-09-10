"""Matched-target development study: labels differ, states/architecture/work do not.

This is a bounded two-core-seed 4x4 study, not a held-out cross-task result.
The experiment never accepts reference answers into target construction/search.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.seed import set_seed
from deploy.m10_native import load_extension
from deploy.semantic_exit import native_semantic_exit
from eval.verified_search import BudgetedVerifiedSearch, ValueContract, ValueTarget, pool_metrics
from model.typed_value import TypedStateValue, load_m16_value
from model.verifier import sudoku_correct, sudoku_score
from scripts.m14_primary_experiment import clamp_givens
from scripts.m16_evidence import CORE_SHA, digest, write_json

TARGETS = [ValueTarget.IMPROVEMENT, ValueTarget.TERMINAL, ValueTarget.QUALITY]


def directions():
    rng = torch.Generator().manual_seed(16111)
    d = torch.randn(3, 64, generator=rng)
    return torch.cat([torch.zeros(1, 64), .5*d/d.norm(dim=1, keepdim=True)], dim=0)


@torch.no_grad()
def candidate_pool(core, ds):
    x = torch.from_numpy(ds.inputs).long()
    xemb = core.token_embed(x) + core.encode_positions(x, 4, 4)
    y, z = torch.zeros_like(xemb), torch.zeros_like(xemb)
    delta = directions()
    parts = []
    for depth in range(1, 5):
        identity = None
        for action in range(4):
            yn, zn = core.recursive_cycle(xemb, y, z+delta[action])
            if action == 0:
                identity = yn, zn
            answer = clamp_givens(x, core.out_head(yn).argmax(-1))
            valid = sudoku_correct(x, answer, 2).float()
            quality = sudoku_score(x, answer, 2)
            next_y, _ = core.recursive_cycle(xemb, yn, zn)
            next_answer = clamp_givens(x, core.out_head(next_y).argmax(-1))
            next_quality = sudoku_score(x, next_answer, 2)
            parts.append({"x": x, "y": yn, "z": zn,
                          "labels": torch.stack([(next_quality > quality+1e-6).float(), valid, quality], 1),
                          "example_index": torch.arange(len(ds)),
                          "depth": torch.full((len(ds),), depth),
                          "action": torch.full((len(ds),), action)})
        y, z = identity
    return {key: torch.cat([p[key] for p in parts], 0) for key in parts[0]}


def fit(pool, target_index, core_seed, out):
    # Same initial weights and minibatch indices for all targets within a core.
    seed = core_seed+16000
    set_seed(seed, deterministic=True)
    model = TypedStateValue(TARGETS[target_index], num_tokens=5, dim=64, n_layers=1, heads=4,
                                  max_grid_size=8, act_bits=8, include_y=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    rng = np.random.default_rng(seed)
    curve = []
    start = time.perf_counter_ns()
    model.train()
    for step in range(180):
        idx = torch.from_numpy(rng.integers(len(pool["x"]), size=64))
        logits = model.forward_logits(pool["x"][idx], pool["y"][idx], pool["z"][idx], width=4)
        loss = F.binary_cross_entropy_with_logits(logits, pool["labels"][idx, target_index])
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite verifier loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad):
            raise RuntimeError("non-finite verifier gradient")
        optimizer.step()
        if step == 0 or (step+1) % 30 == 0:
            curve.append({"step": step+1, "loss": float(loss.detach()), "grad_norm": float(grad)})
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    target = TARGETS[target_index]
    path = out/f"verifier_{core_seed}_{target.value}.pt"
    training_manifest = out.parent / "manifests" / "seed2026091601.json"
    manifest_sha = digest(training_manifest.read_bytes())
    ancestry = {"parent_checkpoints": [CORE_SHA[core_seed]],
                "consumed_manifests": [{"sha256": manifest_sha, "splits": ["train"], "role": "training"}]}
    torch.save({"format": "spectra.m16_typed_verifier.v1", "data_ancestry": ancestry, "target": target.value,
                "core_sha256": CORE_SHA[core_seed], "model_state": model.state_dict(),
                "architecture": {"dim": 64, "n_layers": 1, "heads": 4, "act_bits": 8}}, path)
    model, contract = load_m16_value(
        path, expected_sha256=digest(path.read_bytes()), expected_core_sha256=CORE_SHA[core_seed],
        expected_training_manifest_sha256=manifest_sha,
        diagnostic_improvement=target is ValueTarget.IMPROVEMENT)
    return model, contract, {"target": target.value, "steps": 180, "batch": 64, "seed": seed,
                             "train_ms": (time.perf_counter_ns()-start)/1e6, "curve": curve,
                             "contract": contract.metadata(), "data_ancestry": ancestry}


@torch.inference_mode()
def probabilities(model, pool):
    chunks = []
    for start in range(0, len(pool["x"]), 128):
        end = start+128
        chunks.append(model.value_state(pool["x"][start:end], pool["y"][start:end], pool["z"][start:end], width=4))
    return torch.cat(chunks).cpu().numpy()


@torch.inference_mode()
def closed_solve(core, model, contract, x, *, core_sha256):
    start = time.perf_counter_ns()
    problem = load_extension().SudokuProblem(x, 2)
    delta = directions()
    context = {}
    def initial():
        context["xemb"] = core.token_embed(x) + core.encode_positions(x, 4, 4)
        return torch.zeros_like(context["xemb"]), torch.zeros_like(context["xemb"])
    def transition(state, action):
        return core.recursive_cycle(context["xemb"], state[0], state[1]+delta[action])
    engine = BudgetedVerifiedSearch(initial=initial, actions=[0, 1, 2, 3], transition=transition,
                                    decode=lambda state: clamp_givens(x, core.out_head(state[0]).argmax(-1)),
                                    checker=problem.check,
                                    value=lambda state: float(model.value_state(x, state[0], state[1], width=4).item()),
                                    contract=contract, model_sha256=core_sha256,
                                    transition_id="fp64_cycle_action_v1")
    result = engine.solve(max_transitions=24, max_depth=4, identity_prefix=4)
    return result, (time.perf_counter_ns()-start)/1e6


def run_targets(cores, datasets, out: Path):
    target_dir = out/"targets_development"
    target_dir.mkdir(exist_ok=False)
    summary = {"scope": "development only; same-state matched-target intervention", "cores": {}}
    for seed, core in cores.items():
        train = candidate_pool(core, datasets["train"])
        models, contracts, fitting = {}, {}, []
        for index, target in enumerate(TARGETS):
            model, contract, record = fit(train, index, seed, target_dir)
            models[target] = model
            contracts[target] = contract
            fitting.append(record)
        train_hash = digest(train["labels"].numpy().tobytes() + train["y"].numpy().tobytes() + train["z"].numpy().tobytes())
        del train
        dev = candidate_pool(core, datasets["test"])
        probs = {target: probabilities(model, dev) for target, model in models.items()}
        groups = {target.value: [] for target in TARGETS}
        groups["exact_first_valid"] = []
        with (target_dir/f"fixed_pools_seed{seed}.jsonl").open("w") as stream:
            for i, example_id in enumerate(datasets["test"].ids):
                indices = np.flatnonzero(dev["example_index"].numpy() == i)
                flags = [bool(v) for v in dev["labels"][indices, 1].tolist()]
                record = {"core_seed": seed, "example_id": example_id, "validity": flags,
                          "depths": dev["depth"][indices].tolist(), "actions": dev["action"][indices].tolist(),
                          "scores": {t.value: probs[t][indices].tolist() for t in TARGETS}}
                for target in TARGETS:
                    groups[target.value].append({"pool_id": f"{seed}/{example_id}", "candidate_validity": flags,
                                                  "selected_index": int(np.argmax(probs[target][indices]))})
                groups["exact_first_valid"].append({"pool_id": f"{seed}/{example_id}", "candidate_validity": flags,
                                                    "selected_index": flags.index(True) if any(flags) else 0})
                stream.write(json.dumps(record, allow_nan=False)+"\n")
        fixed = {target: pool_metrics(rows) for target, rows in groups.items()}
        rng = np.random.default_rng(16093)
        closed = []
        with (target_dir/f"closed_loop_seed{seed}.jsonl").open("w") as stream:
            for i, example_id in enumerate(datasets["test"].ids):
                x = torch.from_numpy(datasets["test"].inputs[i:i+1]).long()
                for arm in rng.permutation(["identity", ValueTarget.TERMINAL.value, ValueTarget.QUALITY.value]):
                    if arm == "identity":
                        start = time.perf_counter_ns()
                        _, work = native_semantic_exit(core, x, 4)
                        elapsed = (time.perf_counter_ns()-start)/1e6
                        valid = work["final_semantic"]
                        transitions = work["executed_steps"]
                    else:
                        target = ValueTarget(arm)
                        result, elapsed = closed_solve(core, models[target], contracts[target], x, core_sha256=CORE_SHA[seed])
                        work, valid, transitions = result.work, result.valid, result.work["transitions"]
                    row = {"core_seed": seed, "example_id": example_id, "arm": str(arm),
                           "valid": bool(valid), "latency_ms": elapsed, "transitions": transitions, "work": work}
                    closed.append(row)
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
        baseline_valid = {r["example_id"]: r["valid"] for r in closed if r["arm"] == "identity"}
        regressions = sum(baseline_valid[r["example_id"]] and not r["valid"] for r in closed if r["arm"] != "identity")
        if regressions:
            raise AssertionError("charged identity prefix lost a previously valid answer")
        closed_summary = {}
        for arm in sorted({r["arm"] for r in closed}):
            rows = [r for r in closed if r["arm"] == arm]
            closed_summary[arm] = {"examples": len(rows), "valid": sum(r["valid"] for r in rows),
                                   "mean_ms": float(np.mean([r["latency_ms"] for r in rows])),
                                   "mean_transitions": float(np.mean([r["transitions"] for r in rows]))}
        summary["cores"][str(seed)] = {"training": fitting, "train_state_label_sha256": train_hash,
                                      "fixed_pool": fixed, "closed_loop": closed_summary, "search_regressions": regressions}
        write_json(target_dir/"summary.json", summary)
        del dev, models
    return summary
