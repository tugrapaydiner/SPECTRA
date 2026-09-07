"""Checkpoint-backed scaling/evaluation harness.

Research rows are built only from trained versioned checkpoints plus one immutable
evaluation manifest. Randomly initialized parameter-budget models are retained only
behind the explicit ``smoke_random_init=True`` contract for plumbing tests.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import torch

from eval.checkpoint_eval import EvaluationContractError, LoadedAuxiliary, LoadedTRMCheckpoint
from eval.edge_energy import measure_energy_joules, rapl_available
from eval.evaluation_manifest import LoadedEvaluationManifest
from eval.latency import measure_latency
from eval.metrics import board_accuracy
from eval.research_eval import (
    CountingLatentNativeMCTS,
    InferenceSetting,
    aggregate_example_records,
    assert_unique_realized_settings,
    evaluate_setting,
    realized_setting,
)
from model.trm import TRM


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Explicit random-init smoke mode (never a research result)
# --------------------------------------------------------------------------- #
def make_param_budget_model(
    target_params: int,
    num_tokens: int,
    seq_len: int,
    n_layers: int = 2,
    heads: int = 8,
    max_grid_size: int = 32,
    **trm_kwargs: Any,
) -> tuple[TRM, int]:
    """Construct an untrained TRM near a parameter target. SMOKE TEST ONLY."""
    approx_dim = int(math.sqrt(max(1, target_params) / (n_layers * 12)))
    approx_dim = max(heads, (approx_dim // heads) * heads)
    best_model: TRM | None = None
    best_count = -1
    for dim in range(max(heads, approx_dim - 6 * heads), approx_dim + 7 * heads, heads):
        model = TRM(
            dim=dim, num_tokens=num_tokens, seq_len=seq_len,
            n_layers=n_layers, heads=heads, max_grid_size=max_grid_size, **trm_kwargs,
        )
        count = count_params(model)
        if best_model is None or abs(count - target_params) < abs(best_count - target_params):
            best_model, best_count = model, count
    assert best_model is not None
    return best_model, best_count


MCTSFactory = Callable[[TRM, int], Any]


@torch.no_grad()
def _greedy_accuracy(model, x, y, height, width):
    model.eval()
    return board_accuracy(model(x, height=height, width=width)[0].argmax(-1), y)


@torch.no_grad()
def _mcts_accuracy(controller, x, y):
    answers = torch.cat([controller.search_and_decode(x[i : i + 1]) for i in range(x.shape[0])])
    return board_accuracy(answers, y)


def run_scaling_grid(
    param_targets: list[int],
    depths: list[int],
    rollouts_list: list[int],
    x: torch.Tensor,
    y: torch.Tensor,
    height: int,
    width: int,
    num_tokens: int,
    seq_len: int,
    device: torch.device | str = "cpu",
    n_latency_runs: int = 10,
    mcts_factory: MCTSFactory | None = None,
    n_layers: int = 2,
    max_grid_size: int = 32,
    *,
    smoke_random_init: bool = False,
) -> list[dict[str, Any]]:
    """Legacy untrained grid, now impossible to confuse with research output."""
    if not smoke_random_init:
        raise EvaluationContractError(
            "random-init scaling is smoke-test-only; research evaluation requires trained checkpoints"
        )
    rows: list[dict[str, Any]] = []
    x, y = x.to(device), y.to(device)
    for target in param_targets:
        model, pcount = make_param_budget_model(
            target, num_tokens, seq_len, n_layers=n_layers, max_grid_size=max_grid_size
        )
        model = model.to(device).eval()
        for depth in depths:
            model.N_sup = int(depth)
            for rollouts in rollouts_list:
                if rollouts == 0:
                    acc = _greedy_accuracy(model, x, y, height, width)
                    infer = lambda: model(x[:1], height=height, width=width)
                else:
                    if mcts_factory is None:
                        raise ValueError("rollouts > 0 requires an mcts_factory")
                    controller = mcts_factory(model, rollouts)
                    acc = _mcts_accuracy(controller, x, y)
                    infer = lambda c=controller: c.search_and_decode(x[:1])
                latency = measure_latency(infer, n_runs=n_latency_runs, warmup=3)
                energy = measure_energy_joules(infer, n_runs=n_latency_runs)
                micro_j = energy["joules_per_run"] * 1e6 if energy is not None else None
                rows.append({
                    "result_kind": "smoke_random_init",
                    "research_result": False,
                    "target_params": int(target),
                    "param_count": int(pcount),
                    "recursion_depth": int(depth),
                    "mcts_rollouts": int(rollouts),
                    "accuracy": float(acc),
                    "latency_ms": float(latency["latency_ms_mean"]),
                    "microjoules_per_infer": micro_j,
                    "joules_per_problem": (micro_j / 1e6) if micro_j is not None else None,
                    "energy_measured": energy is not None,
                })
    return rows


# --------------------------------------------------------------------------- #
# Research checkpoint grid
# --------------------------------------------------------------------------- #
def _measure_setting_cost(
    core: LoadedTRMCheckpoint,
    manifest: LoadedEvaluationManifest,
    setting: InferenceSetting,
    verifier: LoadedAuxiliary | None,
    action_policy: LoadedAuxiliary | None,
    n_runs: int,
) -> dict[str, Any]:
    if n_runs <= 0 or len(manifest.dataset) == 0:
        return {"latency_ms": None, "microjoules_per_infer": None, "energy_measured": False}
    setting.validate()
    ds = manifest.dataset
    x1 = torch.from_numpy(ds.inputs[:1]).to(core.device)
    original_n_sup = int(core.model.N_sup)

    def infer():
        with torch.inference_mode():
            core.model.eval()
            if setting.mode == "greedy":
                core.model.N_sup = int(setting.ordinary_n_sup)
                return core.model(x1, height=ds.height, width=ds.width)[0]
            controller = CountingLatentNativeMCTS(
                core.model,
                verifier.module,
                action_policy.module,
                ds.height,
                ds.width,
                n_rollouts=int(setting.mcts_rollouts),
                c_puct=float(setting.c_puct),
                uncertainty_beta=float(setting.uncertainty_beta),
            )
            return controller.search_and_decode(x1)

    try:
        latency = measure_latency(infer, n_runs=n_runs, warmup=min(2, n_runs))
        energy = measure_energy_joules(infer, n_runs=n_runs)
    finally:
        core.model.N_sup = original_n_sup
    micro_j = energy["joules_per_run"] * 1e6 if energy is not None else None
    return {
        "latency_ms": float(latency["latency_ms_mean"]),
        "microjoules_per_infer": micro_j,
        "energy_measured": energy is not None,
    }


def run_checkpoint_scaling_grid(
    cores: list[LoadedTRMCheckpoint],
    manifest: LoadedEvaluationManifest,
    *,
    greedy_n_sup: list[int] | None = None,
    search_rollouts: list[int] | None = None,
    auxiliary_pairs: list[tuple[LoadedAuxiliary, LoadedAuxiliary] | None] | None = None,
    search_seed: int | None = None,
    c_puct: float = 1.5,
    uncertainty_beta: float = 0.0,
    n_latency_runs: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate actual checkpoint families/parameter counts on one fixed snapshot.

    ``greedy_n_sup=None`` evaluates the checkpoint's recorded N_sup. An explicit
    empty list means no greedy rows. Current LatentNativeMCTS is deterministic;
    a non-None ``search_seed`` is rejected by ``InferenceSetting`` rather than
    being recorded as a decorative configuration difference.
    """
    if not cores:
        raise EvaluationContractError("research scaling requires at least one trained checkpoint")
    search_rollouts = [int(v) for v in (search_rollouts or [])]
    if any(v <= 0 for v in search_rollouts):
        raise EvaluationContractError("research search_rollouts must contain only positive budgets")
    if auxiliary_pairs is None:
        auxiliary_pairs = [None] * len(cores)
    if len(auxiliary_pairs) != len(cores):
        raise EvaluationContractError("auxiliary checkpoint pairs must align one-to-one with core checkpoints")

    summary_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for core_idx, core in enumerate(cores):
        pair = auxiliary_pairs[core_idx]
        verifier, action_policy = (None, None) if pair is None else pair
        depths = (
            [int(core.model.N_sup)]
            if greedy_n_sup is None
            else [int(v) for v in greedy_n_sup]
        )
        settings = [InferenceSetting(mode="greedy", ordinary_n_sup=v) for v in depths]
        settings += [
            InferenceSetting(
                mode="latent_mcts",
                ordinary_n_sup=None,
                mcts_rollouts=v,
                c_puct=float(c_puct),
                uncertainty_beta=float(uncertainty_beta),
                search_seed=search_seed,
            )
            for v in search_rollouts
        ]
        if not settings:
            raise EvaluationContractError("research evaluation requires at least one inference setting")
        realized = [realized_setting(core, s, action_policy=action_policy) for s in settings]
        assert_unique_realized_settings(realized)

        for setting in settings:
            if setting.mode == "latent_mcts" and pair is None:
                raise EvaluationContractError(
                    "learned search requested without compatible trained verifier/action checkpoints"
                )
            records = evaluate_setting(
                core,
                manifest,
                setting,
                verifier=verifier if setting.mode == "latent_mcts" else None,
                action_policy=action_policy if setting.mode == "latent_mcts" else None,
            )
            aggregate = aggregate_example_records(records)
            cost = _measure_setting_cost(
                core,
                manifest,
                setting,
                verifier if setting.mode == "latent_mcts" else None,
                action_policy if setting.mode == "latent_mcts" else None,
                int(n_latency_runs),
            )
            summary_rows.append({
                **aggregate,
                # Summary rows are research rows too: preserve the exact trained
                # artifact identities and runtime metadata already present on each
                # per-example record instead of dropping them during aggregation.
                "checkpoint_format": records[0]["checkpoint_format"],
                "checkpoint_version": records[0]["checkpoint_version"],
                "checkpoint_global_step": records[0]["checkpoint_global_step"],
                "checkpoint_n": records[0]["checkpoint_n"],
                "checkpoint_T": records[0]["checkpoint_T"],
                "checkpoint_N_sup": records[0]["checkpoint_N_sup"],
                "evaluation_manifest_file_sha256": records[0]["evaluation_manifest_file_sha256"],
                "evaluation_split": records[0]["evaluation_split"],
                "task": records[0]["task"],
                "task_scope": records[0]["task_scope"],
                "verifier_checkpoint_sha256": records[0]["verifier_checkpoint_sha256"],
                "action_policy_checkpoint_sha256": records[0]["action_policy_checkpoint_sha256"],
                "eval_parameter_dtype": records[0]["eval_parameter_dtype"],
                "training_runtime": records[0]["training_runtime"],
                **cost,
                "energy_available_on_host": bool(rapl_available()),
            })
            example_rows.extend(records)
    return summary_rows, example_rows


def to_dataframe(rows: list[dict[str, Any]]):
    """Flatten summary rows enough for CSV while preserving JSON provenance fields."""
    import json
    import pandas as pd

    flat: list[dict[str, Any]] = []
    for row in rows:
        out = {k: v for k, v in row.items() if k not in {"metrics", "realized_compute_totals", "inference_setting", "realized_setting", "stopping_reasons", "training_runtime"}}
        for k, v in row.get("metrics", {}).items():
            out[f"metric_{k}"] = v
        for k, v in row.get("realized_compute_totals", {}).items():
            out[f"compute_{k}"] = v
        out["inference_setting_json"] = json.dumps(row.get("inference_setting", {}), sort_keys=True)
        out["realized_setting_json"] = json.dumps(row.get("realized_setting", {}), sort_keys=True)
        out["stopping_reasons_json"] = json.dumps(row.get("stopping_reasons", {}), sort_keys=True)
        flat.append(out)
    return pd.DataFrame(flat)


# --------------------------------------------------------------------------- #
# Historical FLOP helpers retained for explicit smoke analysis only.
# --------------------------------------------------------------------------- #
def transformer_forward_flops(
    params: int, seq_len: int, dim: int, n_layers: int, applications: int = 1
) -> float:
    linear = 2.0 * params * seq_len
    attn = 2.0 * n_layers * seq_len * seq_len * dim
    return (linear + attn) * applications


def spectra_inference_flops(
    model: TRM, seq_len: int, mcts_rollouts: int = 0, n_children: int = 3
) -> float:
    applications = model.N_sup * model.T * (model.n + 1)
    base = transformer_forward_flops(
        count_params(model), seq_len, model.dim, len(model.blocks), applications
    )
    search_factor = 1.0 + mcts_rollouts * n_children / max(1, model.N_sup)
    return base * search_factor


def build_dense_baseline_for_flops(
    target_flops: float,
    num_tokens: int,
    seq_len: int,
    max_grid_size: int = 32,
    n_layers: int = 3,
    heads: int = 8,
):
    from model.system1_student import System1Student
    approx_params = target_flops / (2.0 * seq_len)
    approx_dim = max(heads, int((approx_params / (n_layers * 12)) ** 0.5))
    approx_dim = max(heads, (approx_dim // heads) * heads)
    best = None
    best_flops = -1.0
    for dim in range(max(heads, approx_dim - 6 * heads), approx_dim + 7 * heads, heads):
        m = System1Student(
            dim=dim, num_tokens=num_tokens, seq_len=seq_len,
            n_layers=n_layers, max_grid_size=max_grid_size,
        )
        f = transformer_forward_flops(count_params(m), seq_len, dim, n_layers, 1)
        if best is None or abs(f - target_flops) < abs(best_flops - target_flops):
            best, best_flops = m, f
    return best, best_flops


def run_null_hypothesis(*args, smoke_random_init: bool = False, **kwargs):
    """The old random dense-vs-SPECTRA null test is no longer a research result."""
    if not smoke_random_init:
        raise EvaluationContractError(
            "random-init dense/null-hypothesis evaluation is smoke-only; "
            "research comparison requires separately trained compatible checkpoints"
        )
    raise EvaluationContractError(
        "legacy random-init null-hypothesis rows were intentionally disabled in M05; "
        "use checkpoint-backed research evaluation or dedicated smoke tests"
    )
