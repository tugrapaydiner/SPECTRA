"""Independent M07 target construction from frozen reasoner search states.

No function in this module accepts a retained reference solution for evaluation.
Labels come from the puzzle, decoded model state, one frozen deterministic
continuation cycle, and the validated symbolic Sudoku scorer.
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

import torch

from model.grounded_verifier import GROUNDED_TARGET_ID, STATE_REPRESENTATION_VERSION
from model.trm import TRM
from model.verifier import sudoku_score

LABEL_MARGIN = 1e-6


def tensor_state_sha256(module: torch.nn.Module) -> str:
    """Deterministic hash of a module's tensor state, independent of serialization."""
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        name_b = name.encode("utf-8")
        h.update(len(name_b).to_bytes(8, "little")); h.update(name_b)
        cpu = tensor.detach().cpu().contiguous()
        dtype_b = str(cpu.dtype).encode("ascii")
        h.update(len(dtype_b).to_bytes(8, "little")); h.update(dtype_b)
        h.update(torch.tensor(list(cpu.shape), dtype=torch.int64).numpy().tobytes())
        h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def assert_frozen_reasoner(model: TRM) -> None:
    if model.training:
        raise RuntimeError("M07 target reasoner must be in eval mode")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("M07 target reasoner must have requires_grad=False for all parameters")


def decode_state(model: TRM, y_state: torch.Tensor) -> torch.Tensor:
    """Decode the answer accumulator without consulting a reference target."""
    return model.out_head(y_state).argmax(dim=-1)


@torch.no_grad()
def oracle_structural_score(
    model: TRM,
    puzzle: torch.Tensor,
    y_state: torch.Tensor,
    *,
    box: int = 3,
) -> torch.Tensor:
    """Validated Sudoku structural score of the state decode; reference-free."""
    return sudoku_score(puzzle, decode_state(model, y_state), box)


@torch.no_grad()
def one_cycle_improvement_target(
    model: TRM,
    puzzle: torch.Tensor,
    y_state: torch.Tensor,
    z_state: torch.Tensor,
    *,
    height: int = 9,
    width: int = 9,
    box: int = 3,
    margin: float = LABEL_MARGIN,
) -> dict[str, torch.Tensor]:
    """Grounded binary event for one deterministic frozen continuation cycle.

    The returned label is independent of dataset reference answers:
    ``1[ sudoku_score(next_decode) > sudoku_score(current_decode) + margin ]``.
    """
    assert_frozen_reasoner(model)
    if puzzle.ndim != 2 or y_state.ndim != 3 or z_state.ndim != 3:
        raise ValueError("expected puzzle[B,L], y[B,L,D], z[B,L,D]")
    if y_state.shape != z_state.shape or puzzle.shape[:2] != y_state.shape[:2]:
        raise ValueError("grounded target state shapes disagree")

    before = oracle_structural_score(model, puzzle, y_state, box=box)
    x_emb = model.token_embed(puzzle) + model.encode_positions(puzzle, height, width)
    y_next, z_next = model.recursive_cycle(x_emb, y_state, z_state)
    after = oracle_structural_score(model, puzzle, y_next, box=box)
    label = (after > before + float(margin)).to(torch.float32)
    return {
        "label": label,
        "score_before": before,
        "score_after": after,
        "y_next": y_next,
        "z_next": z_next,
    }


@torch.no_grad()
def generate_trajectory_states(
    model: TRM,
    puzzles: torch.Tensor,
    ids: Sequence[str],
    *,
    max_depth: int,
    height: int = 9,
    width: int = 9,
    box: int = 3,
    margin: float = LABEL_MARGIN,
) -> list[dict[str, Any]]:
    """Generate real deterministic frozen-reasoner states and independent labels."""
    assert_frozen_reasoner(model)
    if len(ids) != puzzles.shape[0]:
        raise ValueError("ids must align with puzzle batch")
    x_emb = model.token_embed(puzzles) + model.encode_positions(puzzles, height, width)
    y = torch.zeros(puzzles.shape[0], puzzles.shape[1], model.dim, device=puzzles.device)
    z = torch.zeros_like(y)
    rows: list[dict[str, Any]] = []
    for depth in range(int(max_depth)):
        before = oracle_structural_score(model, puzzles, y, box=box)
        y_next, z_next = model.recursive_cycle(x_emb, y, z)
        after = oracle_structural_score(model, puzzles, y_next, box=box)
        labels = (after > before + float(margin)).to(torch.float32)
        for i in range(puzzles.shape[0]):
            rows.append({
                "id": str(ids[i]),
                "depth": int(depth),
                "x": puzzles[i].detach().cpu().clone(),
                "y": y[i].detach().cpu().clone(),
                "z": z[i].detach().cpu().clone(),
                "label": float(labels[i].item()),
                "score_before": float(before[i].item()),
                "score_after": float(after[i].item()),
                "target_id": GROUNDED_TARGET_ID,
                "state_representation": STATE_REPRESENTATION_VERSION,
                "reference_target_used": False,
            })
        y, z = y_next, z_next
    return rows


def grounding_metadata() -> dict[str, Any]:
    return {
        "target_id": GROUNDED_TARGET_ID,
        "state_representation": STATE_REPRESENTATION_VERSION,
        "oracle": "model.verifier.sudoku_score",
        "reference_target_used": False,
        "bootstrapped": False,
        "continuation_policy": "frozen_trm_one_recursive_cycle_no_action_no_noise",
        "label_rule": "score_after > score_before + 1e-6",
        "margin": LABEL_MARGIN,
        "score_inputs": "puzzle and decoded candidate only",
    }
