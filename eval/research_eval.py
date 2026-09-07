"""Checkpoint-backed research evaluation and realized compute accounting."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from common.seed import derive_seed, isolated_seed
from eval.checkpoint_eval import (
    EvaluationContractError,
    LoadedAuxiliary,
    LoadedTRMCheckpoint,
    validate_checkpoint_manifest_compatibility,
)
from eval.evaluation_manifest import LoadedEvaluationManifest
from eval.latent_mcts import LatentNativeMCTS
from eval.metrics import task_metrics


@dataclass(frozen=True)
class InferenceSetting:
    """Only knobs that the selected inference implementation actually consumes."""

    mode: str = "greedy"  # greedy | latent_mcts
    ordinary_n_sup: int | None = None
    mcts_rollouts: int = 0
    c_puct: float = 1.5
    uncertainty_beta: float = 0.0
    search_seed: int = 20260907

    def validate(self) -> None:
        if self.mode == "greedy":
            if self.ordinary_n_sup is None or int(self.ordinary_n_sup) <= 0:
                raise EvaluationContractError("greedy evaluation requires ordinary_n_sup >= 1")
            if self.mcts_rollouts != 0:
                raise EvaluationContractError("greedy evaluation cannot specify MCTS rollouts")
        elif self.mode == "latent_mcts":
            # N_sup is not consumed by LatentNativeMCTS._step. Rejecting it avoids
            # apparently different search depths that execute identical search transitions.
            if self.ordinary_n_sup is not None:
                raise EvaluationContractError(
                    "ordinary_n_sup/N_sup is not a latent-MCTS transition knob; "
                    "search uses checkpoint T and n plus the rollout/action budgets"
                )
            if int(self.mcts_rollouts) <= 0:
                raise EvaluationContractError("latent_mcts requires mcts_rollouts > 0")
        else:
            raise EvaluationContractError(f"unsupported inference mode {self.mode!r}")
        if not np.isfinite(self.c_puct) or self.c_puct < 0:
            raise EvaluationContractError("c_puct must be finite and non-negative")
        if not np.isfinite(self.uncertainty_beta) or self.uncertainty_beta < 0:
            raise EvaluationContractError("uncertainty_beta must be finite and non-negative")


class CountingLatentNativeMCTS(LatentNativeMCTS):
    """Instrumentation wrapper; search mathematics remain in LatentNativeMCTS."""

    def reset_compute_stats(self) -> None:
        self._transition_calls = 0
        self._expansions = 0
        self._verifier_calls = 0
        self._stopping_reason = "not_started"
        self._rollouts_completed = 0

    def _step(self, x_emb, node, action):
        self._transition_calls += 1
        return super()._step(x_emb, node, action)

    def _expand(self, node, x_emb):
        self._expansions += 1
        return super()._expand(node, x_emb)

    def _value(self, x, node):
        self._verifier_calls += 1
        return super()._value(x, node)

    @torch.no_grad()
    def search(self, x: torch.Tensor):
        self.reset_compute_stats()
        try:
            out = super().search(x)
        except Exception:
            self._stopping_reason = "error"
            raise
        self._rollouts_completed = self._verifier_calls
        self._stopping_reason = "rollout_budget_exhausted"
        return out

    def compute_stats(self) -> dict[str, Any]:
        return {
            "search_transition_calls": int(self._transition_calls),
            "search_expansions": int(self._expansions),
            "verifier_calls": int(self._verifier_calls),
            "rollouts_requested": int(self.n_rollouts),
            "rollouts_completed": int(self._rollouts_completed),
            "stopping_reason": str(self._stopping_reason),
        }


def realized_setting(
    core: LoadedTRMCheckpoint,
    setting: InferenceSetting,
    *,
    action_policy: LoadedAuxiliary | None = None,
) -> dict[str, Any]:
    setting.validate()
    model = core.model
    if setting.mode == "greedy":
        n_sup = int(setting.ordinary_n_sup)
        return {
            "mode": "greedy",
            "ordinary_n_sup": n_sup,
            "checkpoint_T": int(model.T),
            "checkpoint_inner_n": int(model.n),
            "ordinary_recursive_cycle_calls_per_example": n_sup * int(model.T),
            "ordinary_shared_operator_applications_per_example": n_sup * int(model.T) * (int(model.n) + 1),
            "mcts_rollouts": 0,
            "N_sup_consumed_by_search_transition": False,
        }
    if action_policy is None:
        raise EvaluationContractError("latent_mcts realized settings require an action policy")
    n_actions = int(action_policy.module.n_actions)
    return {
        "mode": "latent_mcts",
        "ordinary_n_sup": None,
        "checkpoint_N_sup": int(model.N_sup),
        "N_sup_consumed_by_search_transition": False,
        "transition_T_cycles": int(model.T),
        "transition_inner_n": int(model.n),
        "n_actions": n_actions,
        "mcts_rollouts": int(setting.mcts_rollouts),
        "c_puct": float(setting.c_puct),
        "uncertainty_beta": float(setting.uncertainty_beta),
        "search_seed": int(setting.search_seed),
    }


def realized_signature(realized: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Canonical effective-compute signature used to detect ignored knob sweeps."""
    return tuple(sorted((str(k), v) for k, v in realized.items()))


def assert_unique_realized_settings(settings: list[dict[str, Any]]) -> None:
    seen: dict[tuple[tuple[str, Any], ...], int] = {}
    for i, row in enumerate(settings):
        sig = realized_signature(row)
        if sig in seen:
            raise EvaluationContractError(
                f"inference settings {seen[sig]} and {i} execute the same realized computation; "
                "ignored/decorative compute knobs are forbidden"
            )
        seen[sig] = i


def _task_metric_kwargs(manifest: LoadedEvaluationManifest) -> dict[str, Any]:
    cfg = dict(manifest.payload.get("task_config", {}))
    return {
        "box": cfg.get("box"),
        "height": int(manifest.payload["height"]),
        "width": int(manifest.payload["width"]),
        "pad_token": manifest.payload.get("pad_token"),
        "require_optimal": bool(cfg.get("require_optimal", True)),
    }


def _empty_compute(setting: InferenceSetting, core: LoadedTRMCheckpoint) -> dict[str, Any]:
    if setting.mode == "greedy":
        n_sup = int(setting.ordinary_n_sup)
        cycles = n_sup * int(core.model.T)
        return {
            "ordinary_forward_calls": 1,
            "ordinary_recursive_cycle_calls": cycles,
            "ordinary_shared_operator_applications": cycles * (int(core.model.n) + 1),
            "search_transition_calls": 0,
            "search_expansions": 0,
            "verifier_calls": 0,
            "rollouts_requested": 0,
            "rollouts_completed": 0,
            "stopping_reason": "greedy_forward_complete",
        }
    return {
        "ordinary_forward_calls": 0,
        "ordinary_recursive_cycle_calls": 0,
        "ordinary_shared_operator_applications": 0,
    }


def _provenance_base(
    core: LoadedTRMCheckpoint,
    manifest: LoadedEvaluationManifest,
    setting: InferenceSetting,
    realized: dict[str, Any],
    verifier: LoadedAuxiliary | None,
    action_policy: LoadedAuxiliary | None,
) -> dict[str, Any]:
    return {
        **core.provenance(),
        "evaluation_manifest_path": str(manifest.path),
        "evaluation_manifest_sha256": manifest.canonical_sha256,
        "evaluation_manifest_file_sha256": manifest.file_sha256,
        "evaluation_split": str(manifest.payload["split"]),
        "task": str(manifest.payload["task"]),
        "task_scope": str(manifest.payload.get("task_scope", "")),
        "official_benchmark": bool(manifest.payload.get("official_benchmark", False)),
        "inference_setting": asdict(setting),
        "realized_setting": realized,
        "verifier_checkpoint_sha256": None if verifier is None else verifier.sha256,
        "action_policy_checkpoint_sha256": None if action_policy is None else action_policy.sha256,
    }


@torch.inference_mode()
def evaluate_setting(
    core: LoadedTRMCheckpoint,
    manifest: LoadedEvaluationManifest,
    setting: InferenceSetting,
    *,
    verifier: LoadedAuxiliary | None = None,
    action_policy: LoadedAuxiliary | None = None,
) -> list[dict[str, Any]]:
    """Evaluate fixed manifest examples and emit one provenance-rich record each."""
    setting.validate()
    validate_checkpoint_manifest_compatibility(core, manifest)
    if torch.is_grad_enabled():
        raise EvaluationContractError("research evaluation must run under inference_mode")
    core.model.eval()
    if core.model.training:
        raise EvaluationContractError("core model is not in eval mode")

    if setting.mode == "latent_mcts":
        if verifier is None or action_policy is None:
            raise EvaluationContractError(
                "learned latent MCTS cannot run without trained verifier/action checkpoints"
            )
        verifier.module.eval()
        action_policy.module.eval()
        if verifier.module.training or action_policy.module.training:
            raise EvaluationContractError("learned search auxiliaries must be in eval mode")
    elif verifier is not None or action_policy is not None:
        raise EvaluationContractError("greedy evaluation must not attach unused learned auxiliaries")

    realized = realized_setting(core, setting, action_policy=action_policy)
    base = _provenance_base(core, manifest, setting, realized, verifier, action_policy)
    ds = manifest.dataset
    kwargs = _task_metric_kwargs(manifest)
    original_n_sup = int(core.model.N_sup)
    rows: list[dict[str, Any]] = []
    try:
        if setting.mode == "greedy":
            core.model.N_sup = int(setting.ordinary_n_sup)
        for i in range(len(ds)):
            x = torch.from_numpy(ds.inputs[i : i + 1]).to(core.device)
            y = torch.from_numpy(ds.targets[i : i + 1]).to(core.device)
            compute = _empty_compute(setting, core)
            example_seed = None
            if setting.mode == "greedy":
                logits, _ = core.model(x, height=ds.height, width=ds.width)
                pred = logits.argmax(-1)
            else:
                example_seed = derive_seed(int(setting.search_seed), f"m05-search:{ds.ids[i]}")
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
                with isolated_seed(example_seed):
                    pred = controller.search_and_decode(x)
                search_stats = controller.compute_stats()
                compute.update(search_stats)
                compute["ordinary_recursive_cycle_calls"] = (
                    int(search_stats["search_transition_calls"]) * int(core.model.T)
                )
                compute["ordinary_shared_operator_applications"] = (
                    compute["ordinary_recursive_cycle_calls"] * (int(core.model.n) + 1)
                )

            metrics = task_metrics(
                str(ds.task), x, pred, y,
                box=kwargs["box"], height=kwargs["height"], width=kwargs["width"],
                pad_token=kwargs["pad_token"], require_optimal=kwargs["require_optimal"],
            )
            rows.append({
                **base,
                "data_id": str(ds.ids[i]),
                "group_id": str(ds.group_ids[i]),
                "example_index": i,
                "search_example_seed": example_seed,
                "prediction": pred[0].detach().cpu().tolist(),
                "correct": bool(torch.equal(pred[0].cpu(), y[0].cpu())),
                "metrics": metrics,
                "realized_compute": compute,
            })
    finally:
        core.model.N_sup = original_n_sup
    return rows


def aggregate_example_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise EvaluationContractError("research evaluation produced no example records")
    metric_keys = sorted({k for r in records for k in r["metrics"]})
    metrics: dict[str, float] = {}
    for key in metric_keys:
        vals = [float(r["metrics"][key]) for r in records if key in r["metrics"]]
        metrics[key] = float(np.mean(vals))
    compute_keys = (
        "ordinary_forward_calls", "ordinary_recursive_cycle_calls",
        "ordinary_shared_operator_applications", "search_transition_calls",
        "search_expansions", "verifier_calls", "rollouts_requested", "rollouts_completed",
    )
    totals = {
        key: int(sum(int(r["realized_compute"].get(key, 0)) for r in records))
        for key in compute_keys
    }
    stopping: dict[str, int] = {}
    for r in records:
        reason = str(r["realized_compute"]["stopping_reason"])
        stopping[reason] = stopping.get(reason, 0) + 1
    first = records[0]
    return {
        "result_kind": "research_checkpoint",
        "checkpoint_sha256": first["checkpoint_sha256"],
        "evaluation_manifest_sha256": first["evaluation_manifest_sha256"],
        "weight_identity": first["weight_identity"],
        "model_family": first["model_family"],
        "param_count": int(first["param_count"]),
        "ternary": bool(first["ternary"]),
        "act8": bool(first["act8"]),
        "eval_backend": first["eval_backend"],
        "eval_device": first["eval_device"],
        "inference_setting": first["inference_setting"],
        "realized_setting": first["realized_setting"],
        "examples": len(records),
        "metrics": metrics,
        "realized_compute_totals": totals,
        "stopping_reasons": stopping,
    }
