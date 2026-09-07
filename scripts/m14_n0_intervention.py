#!/usr/bin/env python3
"""M14 Development Amendment A: one-block recurrent (n=0) intervention.

The frozen amendment is docs/M14_DEVELOPMENT_AMENDMENT_A.md.  This runner does
not alter the original M14 practical-effect gate.  It retrains the primary
single-pass baseline, trains two n=0 recurrent candidates across all five seeds,
selects width on validation only, evaluates on a fresh development reserve, and
opens the untouched original confirmation set only after a development pass.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.m14_primary_experiment as m
from scripts._common import build_data_splits

AMENDMENT_PREREG_SHA = "8d76523ede76cdabf609351d721bd9e67867d624"
NEW_DEVELOPMENT_SEED = 2026091404
CANDIDATE_KINDS = ("n0_dim64", "n0_dim96")
CANDIDATE_DIMS = {"n0_dim64": 64, "n0_dim96": 96}
PRIMARY_BASELINE = m.PRIMARY_BASELINE


def make_n0_model(kind: str, seed: int) -> m.TRM:
    if kind not in CANDIDATE_DIMS:
        raise ValueError(kind)
    m.set_seed(int(seed), deterministic=True)
    return m.TRM(
        dim=CANDIDATE_DIMS[kind],
        num_tokens=5,
        seq_len=16,
        n_layers=1,
        n=0,
        T=1,
        N_sup=m.RECURSIVE_TRAIN_NSUP,
        heads=4,
        alpha_y=0.1,
        alpha_z=0.1,
        max_grid_size=8,
        ternary=False,
        act8=False,
    ).cpu()


def n0_checkpoint(out: Path, kind: str, seed: int) -> Path:
    return out / "checkpoints" / f"{kind}_seed{seed}.pt"


def train_n0(kind: str, seed: int, train_ds: m.GridDataset, val_ds: m.GridDataset, out: Path) -> dict[str, Any]:
    model = make_n0_model(kind, seed)
    arch = m.family_record(kind, model)
    if int(arch["n"]) != 0 or int(arch["block_apps_per_training_example"]) != m.RECURSIVE_TRAIN_NSUP:
        raise m.ExperimentStop(f"unexpected n=0 architecture accounting: {arch}")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=m.LR, weight_decay=m.WEIGHT_DECAY)
    x = torch.from_numpy(train_ds.inputs).long()
    y = torch.from_numpy(train_ds.targets).long()
    rng = np.random.default_rng(int(seed) + 141400)
    curve = out / "curves" / f"{kind}_seed{seed}.jsonl"
    curve.parent.mkdir(parents=True, exist_ok=True)
    if curve.exists():
        curve.unlink()
    snapshots = {1, m.TRAIN_STEPS // 2, 3 * m.TRAIN_STEPS // 4, m.TRAIN_STEPS}
    losses: list[float] = []
    max_grad = 0.0
    train_seconds = 0.0
    model.train()
    for step in range(1, m.TRAIN_STEPS + 1):
        idx = torch.from_numpy(rng.integers(0, len(train_ds), size=m.BATCH_SIZE, dtype=np.int64))
        t0 = time.perf_counter()
        loss = m.training_loss(model, kind, x[idx], y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, m.CLIP_NORM)
        opt.step()
        train_seconds += time.perf_counter() - t0
        lv = float(loss.detach())
        gv = float(torch.as_tensor(grad).detach())
        if not math.isfinite(lv) or not math.isfinite(gv):
            raise m.ExperimentStop(f"nonfinite Amendment-A training state {kind}/seed{seed}/step{step}")
        losses.append(lv)
        max_grad = max(max_grad, gv)
        if step == 1 or step % 100 == 0 or step in snapshots:
            val = None
            if step in snapshots:
                model.eval()
                val = m.quick_metrics(model, kind, val_ds)
                model.train()
            m.append_jsonl(
                curve,
                {
                    "kind": kind,
                    "seed": seed,
                    "step": step,
                    "train_blank_ce": lv,
                    "grad_norm_preclip": gv,
                    "validation": val,
                    "cumulative_train_seconds": train_seconds,
                    "n": 0,
                },
            )
    model.eval()
    window = min(50, len(losses))
    first = float(np.mean(losses[:window]))
    last = float(np.mean(losses[-window:]))
    record = {
        "kind": kind,
        "seed": int(seed),
        "steps": m.TRAIN_STEPS,
        "batch_size": m.BATCH_SIZE,
        "examples_sampled": m.TRAIN_STEPS * m.BATCH_SIZE,
        "train_seconds": train_seconds,
        "optimizer": {"name": "AdamW", "lr": m.LR, "weight_decay": m.WEIGHT_DECAY},
        "clip_grad_norm": m.CLIP_NORM,
        "objective": "blank_cell_cross_entropy",
        "initial_loss_mean": first,
        "final_loss_mean": last,
        "relative_loss_improvement": (first - last) / max(abs(first), 1e-12),
        "max_grad_norm_preclip": max_grad,
        "architecture": arch,
        "training_block_applications": m.TRAIN_STEPS * m.BATCH_SIZE * arch["block_apps_per_training_example"],
        "validation_final": m.quick_metrics(model, kind, val_ds),
        "tensor_state_sha256": m.tensor_state_sha256(model),
        "ternary_report": None,
    }
    ckpt = n0_checkpoint(out, kind, seed)
    m.save_checkpoint(ckpt, model, kind, seed, record)
    record["checkpoint"] = str(ckpt)
    record["checkpoint_sha256"] = m.sha256_file(ckpt)
    m.append_jsonl(out / "training_runs.jsonl", record)
    return record


def load_n0_checkpoint(path: Path) -> tuple[m.TRM, str, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "spectra.m14_model" or payload.get("version") != 1:
        raise m.ExperimentStop(f"invalid Amendment-A checkpoint {path}")
    kind = str(payload["kind"])
    seed = int(payload["seed"])
    model = make_n0_model(kind, seed)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if int(model.n) != 0:
        raise m.ExperimentStop("Amendment-A checkpoint restored with n != 0")
    return model, kind, payload


def n0_bundle(kind: str, seed: int, out: Path) -> m.SolverBundle:
    ck = n0_checkpoint(out, kind, seed)
    model, loaded_kind, payload = load_n0_checkpoint(ck)
    model.N_sup = 1

    def solve(x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            logits, _ = model(x, height=4, width=4)
            return m.clamp_givens(x, logits.argmax(-1))

    def work() -> dict[str, Any]:
        return {
            "recursive_cycle_calls": 1,
            "block_applications": 1,
            "verifier_calls": 0,
            "router_calls": 0,
            "action_calls": 0,
            "conversion_calls": 0,
        }

    return m.SolverBundle(
        config_id=kind,
        family="fp_n0",
        seed=int(seed),
        solve=solve,
        work=work,
        setup={
            "checkpoint": str(ck),
            "checkpoint_sha256": m.sha256_file(ck),
            "kind": loaded_kind,
            "dim": int(model.dim),
            "n": 0,
            "T": 1,
            "N_sup": 1,
            "complete_solve_auxiliary_work": {
                "verifier": "none",
                "router": "none",
                "action_policy": "none",
                "conversion": "none",
            },
            "trainable_params": int(payload["family"]["trainable_params"]),
        },
    )


def audit_disjoint(prior_manifest: dict[str, Any], later_manifest: dict[str, Any], *, later_seed: int) -> dict[str, Any]:
    prior = m.manifest_fingerprints(prior_manifest)
    later = m.manifest_fingerprints(later_manifest)
    overlap = sorted(prior & later)
    return {
        "prior_fingerprints": len(prior),
        "later_fingerprints": len(later),
        "overlap_count": len(overlap),
        "overlap_examples": overlap[:10],
        "later_seed": int(later_seed),
    }


def build_new_development(out: Path, prior_manifest: dict[str, Any]) -> tuple[m.GridDataset, dict[str, Any], dict[str, Any]]:
    cfg = m.make_cfg(NEW_DEVELOPMENT_SEED)
    ds, manifest = build_data_splits(
        cfg,
        0,
        0,
        m.DEV_N,
        seed=NEW_DEVELOPMENT_SEED,
        manifest_path=out / "manifests" / "amendment_a_development_reserve.json",
    )
    audit = audit_disjoint(prior_manifest, manifest, later_seed=NEW_DEVELOPMENT_SEED)
    m.write_json(out / "manifests" / "amendment_a_development_overlap_audit.json", audit)
    if audit["overlap_count"]:
        raise m.ExperimentStop("Amendment-A development reserve overlaps original development hierarchy")
    return ds["test"], manifest, audit


def build_confirmation(
    out: Path,
    prior_manifest: dict[str, Any],
    dev2_manifest: dict[str, Any],
) -> tuple[m.GridDataset, dict[str, Any], dict[str, Any]]:
    cfg = m.make_cfg(m.CONFIRM_DATA_SEED)
    ds, manifest = build_data_splits(
        cfg,
        0,
        0,
        m.CONFIRM_N,
        seed=m.CONFIRM_DATA_SEED,
        manifest_path=out / "manifests" / "confirmation_1.json",
    )
    used = m.manifest_fingerprints(prior_manifest) | m.manifest_fingerprints(dev2_manifest)
    conf = m.manifest_fingerprints(manifest)
    overlap = sorted(used & conf)
    audit = {
        "prior_plus_amendment_development_fingerprints": len(used),
        "confirmation_fingerprints": len(conf),
        "overlap_count": len(overlap),
        "overlap_examples": overlap[:10],
        "confirmation_seed": m.CONFIRM_DATA_SEED,
        "reserve_confirmation_seed_unopened": m.RESERVE_CONFIRM_DATA_SEED,
    }
    m.write_json(out / "manifests" / "confirmation_overlap_audit.json", audit)
    if overlap:
        raise m.ExperimentStop("Amendment-A confirmation overlaps development evidence")
    return ds["test"], manifest, audit


def clear_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def train_all(train_ds: m.GridDataset, val_ds: m.GridDataset, out: Path) -> list[dict[str, Any]]:
    clear_if_exists(out / "training_runs.jsonl")
    runs: list[dict[str, Any]] = []
    for seed in m.MODEL_SEEDS:
        runs.append(m.train_one("single_pass", seed, train_ds, val_ds, out))
    for kind in CANDIDATE_KINDS:
        for seed in m.MODEL_SEEDS:
            runs.append(train_n0(kind, seed, train_ds, val_ds, out))
    return runs


def evaluate_validation(val_ds: m.GridDataset, out: Path, training: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows_path = out / "validation_rows.jsonl"
    clear_if_exists(rows_path)
    rows: list[dict[str, Any]] = []
    for seed in m.MODEL_SEEDS:
        for kind in CANDIDATE_KINDS:
            rows.extend(
                m.evaluate_bundle(
                    n0_bundle(kind, seed, out),
                    val_ds,
                    "validation_amendment_a",
                    cost_n=m.VALIDATION_COST_N,
                    repeats=m.LATENCY_REPEATS,
                    out_rows=rows_path,
                )
            )
        rows.extend(
            m.evaluate_bundle(
                m.single_pass_bundle(seed, out, int8=False),
                val_ds,
                "validation_amendment_a",
                cost_n=m.VALIDATION_COST_N,
                repeats=m.LATENCY_REPEATS,
                out_rows=rows_path,
            )
        )
    agg = m.all_aggregates(rows)
    params: dict[str, int] = {}
    for rec in training:
        params.setdefault(str(rec["kind"]), int(rec["architecture"]["trainable_params"]))
    selection = select_validation_candidate(agg, params)
    m.write_json(out / "validation_operating_points.json", agg)
    m.write_csv(out / "validation_operating_points.csv", agg)
    m.write_json(out / "candidate_selection.json", selection)
    return rows, agg, selection


def select_validation_candidate(aggregates: list[dict[str, Any]], params: dict[str, int]) -> dict[str, Any]:
    by = {str(r["config_id"]): dict(r) for r in aggregates}
    base = by.get(PRIMARY_BASELINE)
    if base is None or base.get("latency_median_ms") is None:
        raise m.ExperimentStop("Amendment-A baseline validation point missing")
    base_q = float(base["semantic_validity"])
    base_lat = float(base["latency_median_ms"])
    base_params = int(params.get("single_pass", -1))
    if base_params <= 0:
        raise m.ExperimentStop("Amendment-A baseline parameter count missing")
    candidates: list[dict[str, Any]] = []
    for cid in CANDIDATE_KINDS:
        row = by.get(cid)
        if row is None or row.get("latency_median_ms") is None:
            raise m.ExperimentStop(f"Amendment-A validation point missing for {cid}")
        kind_params = int(params.get(cid, -1))
        if kind_params <= 0 or kind_params >= base_params:
            raise m.ExperimentStop(
                f"Amendment-A candidate must remain smaller than baseline: {cid}={kind_params}, baseline={base_params}"
            )
        row["validation_quality_difference_vs_baseline"] = float(row["semantic_validity"]) - base_q
        row["validation_latency_ratio_vs_baseline"] = float(row["latency_median_ms"]) / base_lat
        row["trainable_params"] = kind_params
        row["parameter_ratio_vs_baseline"] = kind_params / base_params
        row["preferred_pool_eligible"] = (
            row["validation_quality_difference_vs_baseline"] >= -0.02
            and row["validation_latency_ratio_vs_baseline"] <= 0.75
        )
        candidates.append(row)

    preferred = [r for r in candidates if r["preferred_pool_eligible"]]
    if preferred:
        chosen = sorted(
            preferred,
            key=lambda r: (
                -float(r["semantic_validity"]),
                float(r["latency_median_ms"]),
                str(r["config_id"]),
            ),
        )[0]
        pool = "q_ge_-0.02_and_latency_le_0.75"
    else:
        chosen = sorted(
            candidates,
            key=lambda r: (
                max(0.0, -0.02 - float(r["validation_quality_difference_vs_baseline"])),
                max(0.0, float(r["validation_latency_ratio_vs_baseline"]) - 0.65),
                -float(r["semantic_validity"]),
                float(r["latency_median_ms"]),
                str(r["config_id"]),
            ),
        )[0]
        pool = "preregistered_lexicographic_fallback"
    return {
        "selected_config_id": str(chosen["config_id"]),
        "selection_pool": pool,
        "selected_validation": chosen,
        "primary_baseline_validation": base,
        "all_candidate_validation": candidates,
        "selection_used_development": False,
        "selection_used_confirmation": False,
        "selection_rule": "docs/M14_DEVELOPMENT_AMENDMENT_A.md#5-validation-only-candidate-selection",
    }


def bundle_for_amendment(cid: str, seed: int, out: Path) -> m.SolverBundle | None:
    if cid in CANDIDATE_KINDS:
        return n0_bundle(cid, seed, out)
    if cid == PRIMARY_BASELINE:
        return m.single_pass_bundle(seed, out, int8=False)
    if cid == "single_pass_int8":
        return m.single_pass_bundle(seed, out, int8=True, audit_sink=out / "int8_conversion_audit.jsonl")
    if cid == "symbolic_exact":
        return m.symbolic_bundle()
    raise ValueError(cid)


def evaluate_surface(
    candidate: str,
    ds: m.GridDataset,
    out: Path,
    *,
    split: str,
    rows_filename: str,
    cost_n: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    path = out / rows_filename
    clear_if_exists(path)
    rows: list[dict[str, Any]] = []
    for seed in m.MODEL_SEEDS:
        for cid in (candidate, PRIMARY_BASELINE, "single_pass_int8"):
            bundle = bundle_for_amendment(cid, seed, out)
            if bundle is not None:
                rows.extend(
                    m.evaluate_bundle(
                        bundle,
                        ds,
                        split,
                        cost_n=cost_n,
                        repeats=m.LATENCY_REPEATS,
                        out_rows=path,
                    )
                )
    rows.extend(
        m.evaluate_bundle(
            m.symbolic_bundle(),
            ds,
            split,
            cost_n=cost_n,
            repeats=m.LATENCY_REPEATS,
            out_rows=path,
        )
    )
    cand = [r for r in rows if r["config_id"] == candidate]
    base = [r for r in rows if r["config_id"] == PRIMARY_BASELINE]
    effect = m.paired_effect(cand, base)
    gate = m.practical_gate(effect)
    return rows, effect, gate


def write_operating_points(path: Path, rows: list[dict[str, Any]]) -> None:
    m.write_json(path, m.all_aggregates(rows))


def measure_representative_energy(candidate: str, ds: m.GridDataset, out: Path, *, split: str) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "scope": "cpu_package_rapl_not_gpu_not_whole_system",
        "split": split,
        "configs": {},
    }
    xs = torch.from_numpy(ds.inputs[:32]).long()
    for cid in (candidate, PRIMARY_BASELINE):
        bundle = bundle_for_amendment(cid, m.MODEL_SEEDS[0], out)
        assert bundle is not None

        def run() -> None:
            for i in range(len(xs)):
                ans = bundle.solve(xs[i : i + 1])
                _ = m.sudoku_correct(xs[i : i + 1], ans, 2)

        e = m.measure_energy_record(run, n_runs=1)
        rec["configs"][cid] = {
            **e,
            "problems_per_window": len(xs),
            "joules_per_complete_solve": (
                float(e["energy_joules"]) / len(xs)
                if e.get("available") and e.get("energy_joules") is not None
                else None
            ),
        }
    m.write_json(out / f"{split}_energy.json", rec)
    return rec


def freeze_candidate(candidate: str, out: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for seed in m.MODEL_SEEDS:
        cp = n0_checkpoint(out, candidate, seed)
        bp = out / "checkpoints" / f"single_pass_seed{seed}.pt"
        files.append(
            {"role": "candidate_reasoner", "seed": seed, "path": str(cp), "sha256": m.sha256_file(cp)}
        )
        files.append(
            {"role": "primary_baseline", "seed": seed, "path": str(bp), "sha256": m.sha256_file(bp)}
        )
    rec = {
        "candidate_config_id": candidate,
        "files": files,
        "frozen_before_confirmation": True,
        "runtime": {"n": 0, "T": 1, "N_sup": 1},
        "git_sha": m.git_sha(),
        "timestamp_unix": time.time(),
        "amendment_prereg_sha": AMENDMENT_PREREG_SHA,
    }
    m.write_json(out / "confirmation_freeze_manifest.json", rec)
    return rec


def runtime_contract(out: Path, training: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        str(r["kind"]): int(r["architecture"]["trainable_params"])
        for r in training
        if str(r["kind"]) in {*CANDIDATE_KINDS, "single_pass"}
    }
    rec = {
        "candidate_runtime": {
            "n": 0,
            "T": 1,
            "N_sup": 1,
            "block_applications_per_complete_model_solve": 1,
            "verifier_calls": 0,
            "router_calls": 0,
            "action_calls": 0,
            "conversion_calls": 0,
        },
        "primary_baseline_runtime": {
            "class": "System1Student",
            "dim": 96,
            "blocks": 2,
            "block_applications_per_complete_model_solve": 2,
            "verifier_calls": 0,
            "router_calls": 0,
            "action_calls": 0,
            "conversion_calls": 0,
        },
        "trainable_params": counts,
        "cost_note": (
            "block applications are structural counters; acceptance uses measured complete-solve "
            "latency and validated CPU-package energy when available"
        ),
    }
    m.write_json(out / "runtime_contract.json", rec)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".m14a/experiment")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    m.write_json(
        out / "environment.json",
        m.measurement_environment(
            backend="pytorch_cpu_complete_solve",
            compiler_flags=None,
            extra={
                "milestone": 14,
                "development_amendment": "A",
                "git_sha": m.git_sha(),
                "amendment_prereg_sha": AMENDMENT_PREREG_SHA,
                "primary_cost": "complete_solve_latency",
            },
        ),
    )
    m.write_json(out / "energy_inventory.json", m.energy_counter_inventory())

    train_ds, val_ds, _consumed_dev1, prior_manifest = m.build_development_data(out)
    training = train_all(train_ds, val_ds, out)
    runtime_contract(out, training)
    _val_rows, val_agg, selection = evaluate_validation(val_ds, out, training)
    candidate = str(selection["selected_config_id"])

    dev2_ds, dev2_manifest, dev2_audit = build_new_development(out, prior_manifest)
    dev_rows, dev_effect, dev_gate = evaluate_surface(
        candidate,
        dev2_ds,
        out,
        split="development_amendment_a",
        rows_filename="development_rows_amendment_a.jsonl",
        cost_n=m.DEVELOPMENT_COST_N,
    )
    m.write_json(out / "development_effect_amendment_a.json", dev_effect)
    m.write_json(out / "development_decision_amendment_a.json", dev_gate)
    write_operating_points(out / "development_operating_points_amendment_a.json", dev_rows)
    measure_representative_energy(candidate, dev2_ds, out, split="development_amendment_a")

    confirmation_opened = False
    confirmation_effect = None
    confirmation_gate = None
    confirmation_audit = None
    if bool(dev_gate["pass"]):
        freeze_candidate(candidate, out)
        conf_ds, _conf_manifest, confirmation_audit = build_confirmation(out, prior_manifest, dev2_manifest)
        confirmation_opened = True
        conf_rows, _initial_effect, _initial_gate = evaluate_surface(
            candidate,
            conf_ds,
            out,
            split="confirmation",
            rows_filename="confirmation_rows.jsonl",
            cost_n=m.CONFIRMATION_COST_N,
        )
        cand = [r for r in conf_rows if r["config_id"] == candidate]
        base = [r for r in conf_rows if r["config_id"] == PRIMARY_BASELINE]
        confirmation_effect = m.paired_effect(cand, base, seed=m.BOOTSTRAP_SEED + 1)
        confirmation_gate = m.practical_gate(confirmation_effect)
        m.write_json(out / "confirmation_effect.json", confirmation_effect)
        m.write_json(out / "confirmation_decision.json", confirmation_gate)
        write_operating_points(out / "confirmation_operating_points.json", conf_rows)
        measure_representative_energy(candidate, conf_ds, out, split="confirmation")

    status = (
        "COMPLETE"
        if confirmation_opened and confirmation_gate is not None and bool(confirmation_gate["pass"])
        else "INCOMPLETE"
    )
    summary = {
        "milestone": 14,
        "development_amendment": "A",
        "status": status,
        "claim_confirmed": status == "COMPLETE",
        "amendment_preregistered_before_results": True,
        "amendment_prereg_sha": AMENDMENT_PREREG_SHA,
        "prior_attempt": {
            "run": 34159474745,
            "head": "66acbc62f48b3e5b1e81080897b89562edfd8bd8",
            "artifact_id": 10032762057,
            "artifact_sha256": "04a5b810d697227c4ccf88a365cf529647734ea0d95deec363f84fcaae885522",
            "status": "INCOMPLETE",
            "confirmation_opened": False,
        },
        "primary_metric": "strict_sudoku_semantic_validity",
        "primary_baseline": PRIMARY_BASELINE,
        "candidate_architectures": list(CANDIDATE_KINDS),
        "selected_candidate": candidate,
        "selection_used_validation_only": True,
        "validation_operating_points": val_agg,
        "development_seed": NEW_DEVELOPMENT_SEED,
        "development_examples": m.DEV_N,
        "development_overlap_audit": dev2_audit,
        "development_effect": dev_effect,
        "development_gate": dev_gate,
        "confirmation_opened": confirmation_opened,
        "confirmation_seed": m.CONFIRM_DATA_SEED if confirmation_opened else None,
        "confirmation_examples": m.CONFIRM_N if confirmation_opened else 0,
        "confirmation_overlap_audit": confirmation_audit,
        "confirmation_effect": confirmation_effect,
        "confirmation_gate": confirmation_gate,
        "reserve_confirmation_seed_unopened": m.RESERVE_CONFIRM_DATA_SEED,
        "confirmation_becomes_development_evidence_if_future_changes_follow": bool(
            confirmation_opened and status != "COMPLETE"
        ),
        "n_training_seeds": len(m.MODEL_SEEDS),
        "training_seeds": list(m.MODEL_SEEDS),
        "training_steps": m.TRAIN_STEPS,
        "train_examples": m.TRAIN_N,
        "validation_examples": m.VAL_N,
        "batch_size": m.BATCH_SIZE,
        "training_runs": len(training),
        "thresholds_changed_after_results": False,
        "baseline_weakened": False,
        "favorable_seed_filtering": False,
        "scaling_law_claim": False,
        "primary_gate_unchanged_from_m14": True,
        "energy_scope": "cpu_package_if_valid_otherwise_unavailable_not_gpu_not_whole_system",
        "sequential_selection_note": (
            "Attempt 1 and its dim64 reserve are development evidence. Amendment A adds two "
            "validation-selected n=0 widths and a fresh development reserve. Independent "
            "confirmation is required for any final claim."
        ),
    }
    m.write_json(out / "summary.json", summary)
    return 0 if status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
