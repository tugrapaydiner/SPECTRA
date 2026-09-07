#!/usr/bin/env python3
"""M12 real router/halter actor-critic training entry point.

This command requires strict reasoner + grounded-verifier checkpoints.  It freezes
both, optimizes only the router, halter and online critic, writes a versioned M12
checkpoint, strict-reloads it, and compares held-out learned control with fixed
and heuristic controls under one symbolic-success protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.seed import set_seed
from eval.adaptive_rl_checkpoint import (
    load_adaptive_rl_checkpoint,
    save_adaptive_rl_checkpoint,
)
from eval.checkpoint_eval import load_research_trm_checkpoint
from eval.grounded_checkpoint import load_grounded_verifier_checkpoint
from eval.grounded_targets import tensor_state_sha256
from model.halting import DEVICE_DIM, HaltingPolicy
from model.lazy_router import ConfidenceRouter, LatentValueHead, RLTokenRouter
from model.verifier import sudoku_correct, sudoku_score
from train.rl import (
    compute_gae,
    grounded_actor_critic_loss,
    grounded_step_rewards,
    make_target_critic,
    soft_update,
)

SEED = 20260912
GAMMA = 0.99
GAE_LAMBDA = 0.95
SUCCESS_REWARD = 1.0
HALT_FAILURE_PENALTY = 0.25
LAMBDA_STEP = 0.01
LAMBDA_TOKEN = 0.02
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
TARGET_TAU = 0.02
GRAD_CLIP = 1.0
LR = 2e-3
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
HEURISTIC_THRESHOLD = 0.90
HEURISTIC_WARMUP = 1
HEURISTIC_MIN_ACTIVE = 0.25


def _plain(v: Any) -> Any:
    if torch.is_tensor(v):
        if v.ndim == 0:
            return v.detach().cpu().item()
        return v.detach().cpu().tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_plain(payload), sort_keys=True) + "\n")


def load_inputs(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("RL input payload must be a mapping")
    if payload.get("format") != "spectra.m12_rl_inputs" or payload.get("version") != 1:
        raise ValueError("unsupported M12 RL input format/version")
    if payload.get("task") != "sudoku" or bool(payload.get("reference_targets_included", True)):
        raise ValueError("M12 pilot requires target-free Sudoku input payload")
    for key in ("train_inputs", "heldout_inputs"):
        x = payload.get(key)
        if not torch.is_tensor(x) or x.ndim != 2 or x.dtype not in {torch.int64, torch.int32}:
            raise ValueError(f"{key} must be an integer [N,L] tensor")
    return payload


def device_batch(batch: int, device: torch.device) -> torch.Tensor:
    # One fixed neutral/high-budget device condition for this policy-training pilot.
    row = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 0.1], dtype=torch.float32, device=device)
    return row.unsqueeze(0).expand(batch, DEVICE_DIM)


def ownership_audit(
    *,
    router: RLTokenRouter,
    halter: HaltingPolicy,
    critic: LatentValueHead,
    target: LatentValueHead,
    reasoner: torch.nn.Module,
    verifier: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    groups = {
        "router": list(router.named_parameters()),
        "halter": list(halter.named_parameters()),
        "critic": list(critic.named_parameters()),
        "target_critic": list(target.named_parameters()),
        "reasoner": list(reasoner.named_parameters()),
        "verifier": list(verifier.named_parameters()),
    }
    ids = {name: {id(p) for _, p in rows} for name, rows in groups.items()}
    intended = ids["router"] | ids["halter"] | ids["critic"]
    optimizer_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    forbidden = ids["target_critic"] | ids["reasoner"] | ids["verifier"]
    report = {
        "parameter_counts": {
            name: int(sum(p.numel() for _, p in rows)) for name, rows in groups.items()
        },
        "optimizer_parameter_tensor_count": len(optimizer_ids),
        "intended_parameter_tensor_count": len(intended),
        "optimizer_exactly_intended": optimizer_ids == intended,
        "optimizer_forbidden_overlap": len(optimizer_ids & forbidden),
        "reasoner_requires_grad_false": all(not p.requires_grad for _, p in groups["reasoner"]),
        "verifier_requires_grad_false": all(not p.requires_grad for _, p in groups["verifier"]),
        "target_requires_grad_false": all(not p.requires_grad for _, p in groups["target_critic"]),
        "owned_names": {
            name: [n for n, _ in rows] for name, rows in groups.items()
        },
    }
    if not report["optimizer_exactly_intended"] or report["optimizer_forbidden_overlap"]:
        raise RuntimeError(f"optimizer ownership violation: {report}")
    if not (
        report["reasoner_requires_grad_false"]
        and report["verifier_requires_grad_false"]
        and report["target_requires_grad_false"]
    ):
        raise RuntimeError(f"frozen ownership violation: {report}")
    return report


def _zero_row(batch: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(batch, dtype=torch.float32, device=device)


def rollout_batch(
    *,
    core,
    grounded,
    router: RLTokenRouter,
    halter: HaltingPolicy,
    critic: LatentValueHead,
    target_critic: LatentValueHead,
    x: torch.Tensor,
    height: int,
    width: int,
    box: int,
    training: bool,
) -> dict[str, Any]:
    """One M12 logical episode batch over the real M11 execution state."""
    model = core.model
    b, length = x.shape
    k_steps = int(model.N_sup)
    d = device_batch(b, x.device)

    model.eval(); grounded.module.eval()
    if training:
        router.train(); halter.train(); critic.train()
    else:
        router.eval(); halter.eval(); critic.eval()
    target_critic.eval()

    with torch.no_grad():
        state = model.init_execution_state(x, height=height, width=width)

    alive = torch.ones(b, dtype=torch.bool, device=x.device)
    router_lp: list[torch.Tensor] = []
    router_ent: list[torch.Tensor] = []
    router_decision: list[torch.Tensor] = []
    halter_lp: list[torch.Tensor] = []
    halter_ent: list[torch.Tensor] = []
    halter_decision: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    terminated_rows: list[torch.Tensor] = []
    truncated_rows: list[torch.Tensor] = []
    success_rows: list[torch.Tensor] = []
    bad_halt_rows: list[torch.Tensor] = []
    density_rows: list[torch.Tensor] = []
    online_values: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    potentials: list[torch.Tensor] = []
    final_answer = torch.zeros_like(x)
    final_reason = ["not_executed"] * b

    with torch.no_grad():
        target_values.append(target_critic(state.z.detach(), d))
        potentials.append(grounded.module.value_state(x, state.y, state.z, width))

    for k in range(k_steps):
        valid = alive.clone()
        valid_rows.append(valid)
        online_values.append(critic(state.z.detach(), d))

        if not bool(valid.any()):
            router_lp.append(_zero_row(b, x.device))
            router_ent.append(_zero_row(b, x.device))
            router_decision.append(torch.zeros(b, dtype=torch.bool, device=x.device))
            halter_lp.append(_zero_row(b, x.device))
            halter_ent.append(_zero_row(b, x.device))
            halter_decision.append(torch.zeros(b, dtype=torch.bool, device=x.device))
            terminated_rows.append(torch.zeros(b, dtype=torch.bool, device=x.device))
            truncated_rows.append(torch.zeros(b, dtype=torch.bool, device=x.device))
            success_rows.append(torch.zeros(b, dtype=torch.bool, device=x.device))
            bad_halt_rows.append(torch.zeros(b, dtype=torch.bool, device=x.device))
            density_rows.append(_zero_row(b, x.device))
            with torch.no_grad():
                target_values.append(target_values[-1].clone())
                potentials.append(potentials[-1].clone())
            continue

        if k == 0:
            # Preregistered environment warm-up: forced all-active, zero PG credit.
            route_mask = valid[:, None, None].expand(b, length, 1).to(torch.float32)
            rlp = _zero_row(b, x.device)
            rent = _zero_row(b, x.device)
            rdecision = torch.zeros(b, dtype=torch.bool, device=x.device)
        else:
            mask_sample, token_lp, token_ent = router.act(
                k, state.y.detach(), state.z.detach(), None, d
            )
            route_mask = mask_sample * valid[:, None, None].to(mask_sample.dtype)
            rlp = token_lp.sum(dim=1)
            rent = token_ent.mean(dim=1)
            rdecision = valid.clone()
        router_lp.append(rlp)
        router_ent.append(rent)
        router_decision.append(rdecision)
        density = route_mask.mean(dim=(1, 2))
        density_rows.append(density)

        with torch.no_grad():
            out = model.run_execution_step(
                state, active_mask=route_mask, reactivation_policy="allow"
            )
            candidate = out["logits"].argmax(dim=-1)
            exact = sudoku_correct(x, candidate, box=box).bool() & valid

        last_budget = k + 1 >= k_steps
        budget_mask = torch.full_like(valid, last_budget, dtype=torch.bool)
        eligible = valid & ~exact & ~budget_mask
        if bool(eligible.any()):
            action, hlp, hent = halter.act(out["y"].detach(), d, sample=training)
            voluntary = action.bool() & eligible
        else:
            hlp = _zero_row(b, x.device)
            hent = _zero_row(b, x.device)
            voluntary = torch.zeros(b, dtype=torch.bool, device=x.device)
        hdecision = eligible
        halter_lp.append(hlp)
        halter_ent.append(hent)
        halter_decision.append(hdecision)

        terminated_now = exact | voluntary
        truncated_now = valid & ~terminated_now & budget_mask
        terminated_rows.append(terminated_now)
        truncated_rows.append(truncated_now)
        success_rows.append(exact)
        bad_halt_rows.append(voluntary)

        ending = terminated_now | truncated_now
        if bool(ending.any()):
            final_answer = torch.where(ending[:, None], candidate, final_answer)
            for idx in torch.nonzero(exact, as_tuple=False).flatten().tolist():
                final_reason[int(idx)] = "exact_success"
            for idx in torch.nonzero(voluntary, as_tuple=False).flatten().tolist():
                final_reason[int(idx)] = "policy_halt_unsolved"
            for idx in torch.nonzero(truncated_now, as_tuple=False).flatten().tolist():
                final_reason[int(idx)] = "budget_truncated"

        alive = valid & ~ending
        with torch.no_grad():
            target_values.append(target_critic(state.z.detach(), d))
            potentials.append(grounded.module.value_state(x, state.y, state.z, width))

    valid_t = torch.stack(valid_rows)
    term_t = torch.stack(terminated_rows)
    trunc_t = torch.stack(truncated_rows)
    success_t = torch.stack(success_rows)
    bad_halt_t = torch.stack(bad_halt_rows)
    density_t = torch.stack(density_rows)
    potential_t = torch.stack(potentials)
    target_t = torch.stack(target_values)
    online_t = torch.stack(online_values)

    rewards, reward_parts = grounded_step_rewards(
        potential_t, density_t, valid_t, term_t, trunc_t, success_t, bad_halt_t,
        gamma=GAMMA, success_reward=SUCCESS_REWARD,
        halt_failure_penalty=HALT_FAILURE_PENALTY,
        lambda_step=LAMBDA_STEP, lambda_token=LAMBDA_TOKEN,
    )
    advantages, returns = compute_gae(
        rewards, target_t, gamma=GAMMA, lam=GAE_LAMBDA,
        valid_mask=valid_t, terminated=term_t, truncated=trunc_t,
    )
    loss, components = grounded_actor_critic_loss(
        router_logprobs=torch.stack(router_lp),
        router_entropies=torch.stack(router_ent),
        router_decisions=torch.stack(router_decision),
        halter_logprobs=torch.stack(halter_lp),
        halter_entropies=torch.stack(halter_ent),
        halter_decisions=torch.stack(halter_decision),
        advantages=advantages,
        returns=returns,
        critic_values=online_t,
        valid_mask=valid_t,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF,
    )

    realized_steps = valid_t.float().sum(dim=0)
    active_sum = (density_t * valid_t.float()).sum(dim=0)
    mean_density = active_sum / realized_steps.clamp_min(1.0)
    terminal_success = success_t.any(dim=0)
    return {
        "loss": loss,
        "components": components,
        "rewards": rewards.detach(),
        "reward_parts": reward_parts,
        "advantages": advantages.detach(),
        "returns": returns.detach(),
        "valid": valid_t.detach(),
        "terminated": term_t.detach(),
        "truncated": trunc_t.detach(),
        "success": success_t.detach(),
        "bad_halt": bad_halt_t.detach(),
        "density": density_t.detach(),
        "router_decisions": torch.stack(router_decision).detach(),
        "halter_decisions": torch.stack(halter_decision).detach(),
        "realized_steps": realized_steps.detach(),
        "mean_active_density": mean_density.detach(),
        "terminal_success": terminal_success.detach(),
        "final_answer": final_answer.detach(),
        "final_reason": final_reason,
        "potential": potential_t.detach(),
        "target_values": target_t.detach(),
    }


def train_policies(
    *,
    core,
    grounded,
    train_inputs: torch.Tensor,
    steps: int,
    out: Path,
    seed: int,
) -> tuple[RLTokenRouter, HaltingPolicy, LatentValueHead, LatentValueHead, dict[str, Any]]:
    device = core.device
    dim = int(core.architecture["dim"])
    router = RLTokenRouter(dim=dim, device_dim=DEVICE_DIM).to(device)
    halter = HaltingPolicy(dim=dim, device_dim=DEVICE_DIM).to(device)
    critic = LatentValueHead(dim=dim, device_dim=DEVICE_DIM).to(device)
    target = make_target_critic(critic).to(device)

    params = list(router.parameters()) + list(halter.parameters()) + list(critic.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    ownership = ownership_audit(
        router=router, halter=halter, critic=critic, target=target,
        reasoner=core.model, verifier=grounded.module, optimizer=optimizer,
    )

    before = {
        "router": tensor_state_sha256(router),
        "halter": tensor_state_sha256(halter),
        "critic": tensor_state_sha256(critic),
        "target_critic": tensor_state_sha256(target),
        "reasoner": tensor_state_sha256(core.model),
        "verifier": tensor_state_sha256(grounded.module),
    }
    log_path = out / "training_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    gen = torch.Generator(device="cpu").manual_seed(seed + 91)
    rng_state_before = torch.random.get_rng_state().clone()
    t0 = time.perf_counter()
    aggregate = []

    height, width = int(core.task["height"]), int(core.task["width"])
    box = int(round(math.sqrt(height)))
    for update in range(1, int(steps) + 1):
        idx = torch.randint(0, train_inputs.shape[0], (BATCH_SIZE,), generator=gen)
        xb = train_inputs.index_select(0, idx).to(device)
        rollout = rollout_batch(
            core=core, grounded=grounded, router=router, halter=halter,
            critic=critic, target_critic=target, x=xb,
            height=height, width=width, box=box, training=True,
        )
        loss = rollout["loss"]
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite M12 loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Forbidden families must never receive gradients.
        forbidden_grad = {
            "reasoner": any(p.grad is not None for p in core.model.parameters()),
            "verifier": any(p.grad is not None for p in grounded.module.parameters()),
            "target_critic": any(p.grad is not None for p in target.parameters()),
        }
        if any(forbidden_grad.values()):
            raise RuntimeError(f"gradient ownership violation: {forbidden_grad}")

        grad_norm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise RuntimeError(f"non-finite M12 gradient at update {update}")
        optimizer.step()
        soft_update(target, critic, tau=TARGET_TAU)

        valid = rollout["valid"].float()
        step_proxy = rollout["reward_parts"]["step_proxy_cost"]
        token_proxy = rollout["reward_parts"]["token_proxy_cost"]
        row = {
            "update": update,
            "task_success": float(rollout["terminal_success"].float().mean()),
            "realized_steps": float(rollout["realized_steps"].mean()),
            "active_density": float(rollout["mean_active_density"].mean()),
            "router_entropy": rollout["components"]["router_entropy"],
            "halter_entropy": rollout["components"]["halter_entropy"],
            "router_policy_loss": rollout["components"]["router_policy"],
            "halter_policy_loss": rollout["components"]["halter_policy"],
            "value_loss": rollout["components"]["value"],
            "total_loss": rollout["components"]["total"],
            "step_proxy_cost": float(step_proxy.sum() / valid.sum().clamp_min(1.0)),
            "token_proxy_cost": float(token_proxy.sum() / valid.sum().clamp_min(1.0)),
            "total_proxy_cost": float((step_proxy + token_proxy).sum() / valid.sum().clamp_min(1.0)),
            "terminated_fraction": float(rollout["terminated"].any(dim=0).float().mean()),
            "truncated_fraction": float(rollout["truncated"].any(dim=0).float().mean()),
            "router_decisions": rollout["components"]["router_decisions"],
            "halter_decisions": rollout["components"]["halter_decisions"],
            "grad_norm_preclip": float(grad_norm),
            "cost_kind": "logical_step_token_proxy_v1",
            "measured_energy_joules": None,
        }
        append_jsonl(log_path, row)
        aggregate.append(row)

    elapsed = time.perf_counter() - t0
    after = {
        "router": tensor_state_sha256(router),
        "halter": tensor_state_sha256(halter),
        "critic": tensor_state_sha256(critic),
        "target_critic": tensor_state_sha256(target),
        "reasoner": tensor_state_sha256(core.model),
        "verifier": tensor_state_sha256(grounded.module),
    }
    changed = {name: before[name] != after[name] for name in before}
    required_updates = changed["router"] and changed["halter"] and changed["critic"] and changed["target_critic"]
    frozen_ok = not changed["reasoner"] and not changed["verifier"]
    if not required_updates or not frozen_ok:
        raise RuntimeError(f"M12 tensor ownership/update gate failed: changed={changed}")

    report = {
        "steps": int(steps),
        "elapsed_seconds": elapsed,
        "before_hashes": before,
        "after_hashes": after,
        "changed": changed,
        "optimizer_ownership": ownership,
        "forbidden_gradients_observed": False,
        "rng_state_changed": not torch.equal(rng_state_before, torch.random.get_rng_state()),
        "mean_last_10": {
            key: float(np.mean([r[key] for r in aggregate[-10:]]))
            for key in (
                "task_success", "realized_steps", "active_density", "router_entropy",
                "halter_entropy", "router_policy_loss", "halter_policy_loss",
                "value_loss", "total_loss", "total_proxy_cost",
            )
        },
    }
    write_json(out / "ownership.json", report)
    return router, halter, critic, target, report


@torch.no_grad()
def evaluate_one(
    *,
    core,
    x: torch.Tensor,
    mode: str,
    router: RLTokenRouter | None,
    halter: HaltingPolicy | None,
    heuristic: ConfidenceRouter | None,
) -> dict[str, Any]:
    model = core.model
    h, w = int(core.task["height"]), int(core.task["width"])
    box = int(round(math.sqrt(h)))
    state = model.init_execution_state(x, h, w)
    d = device_batch(1, x.device)
    densities = []
    final = None
    reason = "budget_truncated"

    for k in range(int(model.N_sup)):
        if mode == "fixed_depth":
            mask = torch.ones(1, x.shape[1], 1, device=x.device)
        elif mode == "heuristic":
            assert heuristic is not None
            prior_logits = model.out_head(state.y)
            mask, _ = heuristic(k, state.y, state.z, prior_logits, d)
        elif mode == "learned_rl":
            assert router is not None and halter is not None
            if k == 0:
                mask = torch.ones(1, x.shape[1], 1, device=x.device)
            else:
                mask, _, _ = router.act(k, state.y, state.z, None, d, )
        else:
            raise ValueError(mode)

        densities.append(float(mask.mean()))
        out = model.run_execution_step(state, active_mask=mask, reactivation_policy="allow")
        answer = out["logits"].argmax(dim=-1)
        exact = bool(sudoku_correct(x, answer, box=box)[0])
        final = answer
        if exact:
            reason = "exact_success"
            break
        if mode == "learned_rl" and k + 1 < int(model.N_sup):
            p = halter.halt_prob(out["y"], d)
            if bool((p > 0.5)[0]):
                reason = "policy_halt_unsolved"
                break

    assert final is not None
    steps = len(densities)
    mean_density = float(np.mean(densities)) if densities else 0.0
    success = bool(sudoku_correct(x, final, box=box)[0])
    structural = float(sudoku_score(x, final, box=box)[0])
    step_cost = LAMBDA_STEP * steps
    token_cost = LAMBDA_TOKEN * sum(densities)
    return {
        "success": success,
        "structural_score": structural,
        "realized_steps": steps,
        "mean_active_density": mean_density,
        "step_proxy_cost": step_cost,
        "token_proxy_cost": token_cost,
        "total_proxy_cost": step_cost + token_cost,
        "stop_reason": reason,
    }


def evaluate_controllers(*, core, loaded_rl, heldout: torch.Tensor, out: Path) -> dict[str, Any]:
    heuristic = ConfidenceRouter(
        threshold=HEURISTIC_THRESHOLD,
        warmup_steps=HEURISTIC_WARMUP,
        min_active_frac=HEURISTIC_MIN_ACTIVE,
    )
    modes = ("learned_rl", "fixed_depth", "heuristic")
    rows: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}
    for i in range(heldout.shape[0]):
        x = heldout[i:i+1].to(core.device)
        for mode in modes:
            row = evaluate_one(
                core=core, x=x, mode=mode,
                router=loaded_rl.router if mode == "learned_rl" else None,
                halter=loaded_rl.halter if mode == "learned_rl" else None,
                heuristic=heuristic if mode == "heuristic" else None,
            )
            row["index"] = i
            rows[mode].append(row)

    summary: dict[str, Any] = {}
    for mode in modes:
        r = rows[mode]
        summary[mode] = {
            "examples": len(r),
            "task_success_rate": float(np.mean([x["success"] for x in r])),
            "mean_structural_score": float(np.mean([x["structural_score"] for x in r])),
            "mean_realized_steps": float(np.mean([x["realized_steps"] for x in r])),
            "mean_active_density": float(np.mean([x["mean_active_density"] for x in r])),
            "mean_step_proxy_cost": float(np.mean([x["step_proxy_cost"] for x in r])),
            "mean_token_proxy_cost": float(np.mean([x["token_proxy_cost"] for x in r])),
            "mean_total_proxy_cost": float(np.mean([x["total_proxy_cost"] for x in r])),
            "stop_reasons": {
                reason: sum(x["stop_reason"] == reason for x in r)
                for reason in sorted({x["stop_reason"] for x in r})
            },
        }
    write_json(out / "heldout_rows.json", rows)
    write_json(out / "heldout_comparison.json", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoner-checkpoint", required=True)
    ap.add_argument("--verifier-checkpoint", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", default="outputs/m12_adaptive_rl")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    set_seed(args.seed, deterministic=True)

    core = load_research_trm_checkpoint(args.reasoner_checkpoint, device="cpu", weight_identity="recorded")
    grounded = load_grounded_verifier_checkpoint(args.verifier_checkpoint, core)
    for p in core.model.parameters():
        p.requires_grad_(False)
    for p in grounded.module.parameters():
        p.requires_grad_(False)
    core.model.eval(); grounded.module.eval()

    inputs = load_inputs(Path(args.inputs))
    if int(inputs["height"]) != int(core.task["height"]) or int(inputs["width"]) != int(core.task["width"]):
        raise ValueError("RL inputs geometry does not match reasoner checkpoint")
    if int(inputs["num_tokens"]) != int(core.architecture["num_tokens"]):
        raise ValueError("RL inputs vocabulary does not match reasoner checkpoint")
    train_inputs = inputs["train_inputs"].long()
    heldout_inputs = inputs["heldout_inputs"].long()

    router, halter, critic, target, training = train_policies(
        core=core, grounded=grounded, train_inputs=train_inputs,
        steps=args.steps, out=out, seed=args.seed,
    )

    objective = {
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "success_reward": SUCCESS_REWARD,
        "halt_failure_penalty": HALT_FAILURE_PENALTY,
        "lambda_step": LAMBDA_STEP,
        "lambda_token": LAMBDA_TOKEN,
        "value_coef": VALUE_COEF,
        "entropy_coef": ENTROPY_COEF,
        "target_tau": TARGET_TAU,
        "router_step0_forced_all_active": True,
        "forced_actions_policy_gradient_credit": False,
        "terminal_bootstrap": False,
        "truncation_bootstrap": True,
        "potential_shaping": "gamma*Phi(next_effective)-Phi(current)",
        "terminal_potential": 0.0,
        "cost_kind": "logical_step_token_proxy_v1",
        "measured_energy_used": False,
        "measured_energy_joules": None,
    }
    checkpoint = out / "adaptive_rl.pt"
    save_adaptive_rl_checkpoint(
        checkpoint,
        router=router, halter=halter, critic=critic, target_critic=target,
        core=core, verifier=grounded, trained_steps=args.steps, seed=args.seed,
        objective=objective, ownership=training["optimizer_ownership"],
        tensor_hashes={
            "before": training["before_hashes"],
            "after": training["after_hashes"],
        },
        training_summary=training["mean_last_10"],
    )
    loaded_rl = load_adaptive_rl_checkpoint(checkpoint, core=core, verifier=grounded)
    strict_reload = {
        "checkpoint": str(checkpoint),
        "sha256": loaded_rl.sha256,
        "router_hash_matches": tensor_state_sha256(loaded_rl.router) == tensor_state_sha256(router),
        "halter_hash_matches": tensor_state_sha256(loaded_rl.halter) == tensor_state_sha256(halter),
        "critic_hash_matches": tensor_state_sha256(loaded_rl.critic) == tensor_state_sha256(critic),
        "target_hash_matches": tensor_state_sha256(loaded_rl.target_critic) == tensor_state_sha256(target),
    }
    if not all(v for k, v in strict_reload.items() if k.endswith("_matches")):
        raise RuntimeError(f"strict adaptive RL reload mismatch: {strict_reload}")
    write_json(out / "checkpoint_reload.json", strict_reload)

    comparison = evaluate_controllers(
        core=core, loaded_rl=loaded_rl, heldout=heldout_inputs, out=out,
    )
    learned = comparison["learned_rl"]
    fixed = comparison["fixed_depth"]
    heuristic = comparison["heuristic"]
    quality = {
        "learned_success_minus_fixed": learned["task_success_rate"] - fixed["task_success_rate"],
        "learned_success_minus_heuristic": learned["task_success_rate"] - heuristic["task_success_rate"],
        "learned_score_minus_fixed": learned["mean_structural_score"] - fixed["mean_structural_score"],
        "learned_score_minus_heuristic": learned["mean_structural_score"] - heuristic["mean_structural_score"],
        "learned_proxy_cost_minus_fixed": learned["mean_total_proxy_cost"] - fixed["mean_total_proxy_cost"],
        "learned_proxy_cost_minus_heuristic": learned["mean_total_proxy_cost"] - heuristic["mean_total_proxy_cost"],
    }
    benefit = bool(
        learned["task_success_rate"] >= fixed["task_success_rate"]
        and learned["mean_structural_score"] >= fixed["mean_structural_score"]
        and learned["mean_total_proxy_cost"] < fixed["mean_total_proxy_cost"]
    )
    summary = {
        "milestone": 12,
        "status": "complete",
        "scope": "grounded_router_halter_rl_training_path",
        "reasoner_checkpoint_sha256": core.sha256,
        "grounded_verifier_checkpoint_sha256": grounded.sha256,
        "rl_checkpoint_sha256": loaded_rl.sha256,
        "reference_targets_used_by_rl": False,
        "objective": objective,
        "training": training,
        "strict_reload": strict_reload,
        "heldout_comparison": comparison,
        "heldout_deltas": quality,
        "learned_control_benefit_observed": benefit,
        "interpretation": (
            "bounded pilot shows a quality-noninferior lower-proxy-cost learned controller"
            if benefit else
            "bounded pilot does not establish learned-control superiority; training-path evidence remains valid"
        ),
        "acceptance": {
            "real_training_updates_router": bool(training["changed"]["router"]),
            "real_training_updates_halter": bool(training["changed"]["halter"]),
            "real_training_updates_critic": bool(training["changed"]["critic"]),
            "target_polyak_updates": bool(training["changed"]["target_critic"]),
            "reasoner_frozen": not bool(training["changed"]["reasoner"]),
            "verifier_frozen": not bool(training["changed"]["verifier"]),
            "strict_checkpoint_reload": all(
                v for k, v in strict_reload.items() if k.endswith("_matches")
            ),
            "heldout_baselines_compared": set(comparison) == {"learned_rl", "fixed_depth", "heuristic"},
            "cost_is_proxy_not_energy": objective["cost_kind"] == "logical_step_token_proxy_v1" and not objective["measured_energy_used"],
        },
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())