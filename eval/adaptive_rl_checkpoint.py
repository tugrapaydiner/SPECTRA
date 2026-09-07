"""Strict M12 router/halter actor-critic checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from eval.checkpoint_eval import EvaluationContractError, LoadedTRMCheckpoint, sha256_file
from eval.grounded_checkpoint import LoadedGroundedVerifier
from model.halting import DEVICE_DIM, HaltingPolicy
from model.lazy_router import LatentValueHead, RLTokenRouter
from train.checkpoint import atomic_torch_save

ADAPTIVE_RL_FORMAT = "spectra.adaptive_rl"
ADAPTIVE_RL_VERSION = 1
ADAPTIVE_RL_KIND = "router_halter_actor_critic"


@dataclass
class LoadedAdaptiveRL:
    path: Path
    sha256: str
    router: RLTokenRouter
    halter: HaltingPolicy
    critic: LatentValueHead
    target_critic: LatentValueHead
    payload: dict[str, Any]


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _is_tensor_state(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and all(isinstance(k, str) for k in value)
        and all(torch.is_tensor(v) for v in value.values())
    )


def adaptive_rl_payload(
    *,
    router: RLTokenRouter,
    halter: HaltingPolicy,
    critic: LatentValueHead,
    target_critic: LatentValueHead,
    core: LoadedTRMCheckpoint,
    verifier: LoadedGroundedVerifier,
    trained_steps: int,
    seed: int,
    objective: Mapping[str, Any],
    ownership: Mapping[str, Any],
    tensor_hashes: Mapping[str, Any],
    training_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if int(trained_steps) <= 0:
        raise EvaluationContractError("adaptive RL checkpoint must record trained_steps > 0")
    if int(router.device_dim) != DEVICE_DIM or int(halter.device_dim) != DEVICE_DIM:
        raise EvaluationContractError("M12 checkpoint requires the declared DEVICE_DIM")
    if int(router.policy[0].in_features - router.device_dim) != int(core.architecture["dim"]):
        raise EvaluationContractError("router dimension is incompatible with reasoner")
    if int(halter.net[0].in_features - halter.device_dim) != int(core.architecture["dim"]):
        raise EvaluationContractError("halter dimension is incompatible with reasoner")
    if bool(objective.get("measured_energy_used", True)):
        raise EvaluationContractError("M12 pilot checkpoint must not claim measured energy")
    if objective.get("cost_kind") != "logical_step_token_proxy_v1":
        raise EvaluationContractError("M12 cost proxy identity mismatch")

    return {
        "format": ADAPTIVE_RL_FORMAT,
        "version": ADAPTIVE_RL_VERSION,
        "kind": ADAPTIVE_RL_KIND,
        "trained_steps": int(trained_steps),
        "seed": int(seed),
        "architecture": {
            "router_class": "RLTokenRouter",
            "halter_class": "HaltingPolicy",
            "critic_class": "LatentValueHead",
            "dim": int(core.architecture["dim"]),
            "device_dim": DEVICE_DIM,
        },
        "compatibility": {
            "reasoner_checkpoint_sha256": core.sha256,
            "grounded_verifier_checkpoint_sha256": verifier.sha256,
            "task": str(core.task["task"]),
            "dim": int(core.architecture["dim"]),
            "num_tokens": int(core.architecture["num_tokens"]),
            "seq_len": int(core.architecture["seq_len"]),
        },
        "objective": dict(objective),
        "ownership": dict(ownership),
        "tensor_hashes": dict(tensor_hashes),
        "training_summary": dict(training_summary),
        "states": {
            "router": _state(router),
            "halter": _state(halter),
            "critic": _state(critic),
            "target_critic": _state(target_critic),
        },
    }


def save_adaptive_rl_checkpoint(
    path: str | Path,
    *,
    router: RLTokenRouter,
    halter: HaltingPolicy,
    critic: LatentValueHead,
    target_critic: LatentValueHead,
    core: LoadedTRMCheckpoint,
    verifier: LoadedGroundedVerifier,
    trained_steps: int,
    seed: int,
    objective: Mapping[str, Any],
    ownership: Mapping[str, Any],
    tensor_hashes: Mapping[str, Any],
    training_summary: Mapping[str, Any],
) -> None:
    atomic_torch_save(
        adaptive_rl_payload(
            router=router, halter=halter, critic=critic, target_critic=target_critic,
            core=core, verifier=verifier, trained_steps=trained_steps, seed=seed,
            objective=objective, ownership=ownership, tensor_hashes=tensor_hashes,
            training_summary=training_summary,
        ),
        path,
    )


def load_adaptive_rl_checkpoint(
    path: str | Path,
    *,
    core: LoadedTRMCheckpoint,
    verifier: LoadedGroundedVerifier,
) -> LoadedAdaptiveRL:
    p = Path(path)
    digest = sha256_file(p)
    try:
        payload = torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(p, map_location="cpu")
    except (OSError, RuntimeError) as exc:
        raise EvaluationContractError(f"cannot load adaptive RL checkpoint {p}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise EvaluationContractError("adaptive RL checkpoint root must be a mapping")
    if payload.get("format") != ADAPTIVE_RL_FORMAT or payload.get("version") != ADAPTIVE_RL_VERSION:
        raise EvaluationContractError("unsupported adaptive RL checkpoint format/version")
    if payload.get("kind") != ADAPTIVE_RL_KIND or int(payload.get("trained_steps", 0)) <= 0:
        raise EvaluationContractError("adaptive RL checkpoint is not a trained M12 policy state")

    comp = payload.get("compatibility")
    if not isinstance(comp, Mapping):
        raise EvaluationContractError("adaptive RL checkpoint lacks compatibility metadata")
    expected = {
        "reasoner_checkpoint_sha256": core.sha256,
        "grounded_verifier_checkpoint_sha256": verifier.sha256,
        "task": str(core.task["task"]),
        "dim": int(core.architecture["dim"]),
        "num_tokens": int(core.architecture["num_tokens"]),
        "seq_len": int(core.architecture["seq_len"]),
    }
    bad = {k: (comp.get(k), v) for k, v in expected.items() if comp.get(k) != v}
    if bad:
        raise EvaluationContractError(f"adaptive RL checkpoint incompatible with inputs: {bad}")

    objective = payload.get("objective")
    if not isinstance(objective, Mapping):
        raise EvaluationContractError("adaptive RL checkpoint lacks objective metadata")
    if objective.get("cost_kind") != "logical_step_token_proxy_v1":
        raise EvaluationContractError("adaptive RL checkpoint cost proxy mismatch")
    if bool(objective.get("measured_energy_used", True)):
        raise EvaluationContractError("adaptive RL checkpoint makes an unsupported measured-energy claim")

    arch = payload.get("architecture")
    if not isinstance(arch, Mapping):
        raise EvaluationContractError("adaptive RL checkpoint lacks architecture metadata")
    if (
        arch.get("router_class") != "RLTokenRouter"
        or arch.get("halter_class") != "HaltingPolicy"
        or arch.get("critic_class") != "LatentValueHead"
        or int(arch.get("dim", -1)) != int(core.architecture["dim"])
        or int(arch.get("device_dim", -1)) != DEVICE_DIM
    ):
        raise EvaluationContractError("adaptive RL architecture metadata mismatch")

    states = payload.get("states")
    if not isinstance(states, Mapping):
        raise EvaluationContractError("adaptive RL checkpoint lacks tensor states")
    for name in ("router", "halter", "critic", "target_critic"):
        if not _is_tensor_state(states.get(name)):
            raise EvaluationContractError(f"adaptive RL {name} state is malformed")

    dim = int(core.architecture["dim"])
    router = RLTokenRouter(dim=dim, device_dim=DEVICE_DIM)
    halter = HaltingPolicy(dim=dim, device_dim=DEVICE_DIM)
    critic = LatentValueHead(dim=dim, device_dim=DEVICE_DIM)
    target = LatentValueHead(dim=dim, device_dim=DEVICE_DIM)
    try:
        router.load_state_dict(states["router"], strict=True)
        halter.load_state_dict(states["halter"], strict=True)
        critic.load_state_dict(states["critic"], strict=True)
        target.load_state_dict(states["target_critic"], strict=True)
    except RuntimeError as exc:
        raise EvaluationContractError(f"adaptive RL state_dict incompatible: {exc}") from exc

    dev = core.device
    router = router.to(dev).eval()
    halter = halter.to(dev).eval()
    critic = critic.to(dev).eval()
    target = target.to(dev).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    return LoadedAdaptiveRL(p, digest, router, halter, critic, target, dict(payload))
