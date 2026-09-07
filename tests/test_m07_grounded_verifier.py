"""Milestone 07 grounded-verifier acceptance contracts."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from eval.checkpoint_eval import EvaluationContractError, LoadedTRMCheckpoint
from eval.grounded_checkpoint import (
    grounded_verifier_payload,
    load_grounded_verifier_checkpoint,
    save_grounded_verifier_checkpoint,
)
from eval.grounded_mcts import GroundedLatentNativeMCTS
from eval.grounded_targets import (
    assert_frozen_reasoner,
    grounding_metadata,
    one_cycle_improvement_target,
    tensor_state_sha256,
)
from eval.latent_mcts import _LatentNode
from model.grounded_verifier import (
    GROUNDED_TARGET_ID,
    STATE_REPRESENTATION_VERSION,
    EnsembleGroundedStateVerifier,
    GroundedStateVerifier,
)
from model.trm import TRM
from train.distill import MCTS_BOOTSTRAP_TARGET_KIND, grounded_improvement_bce_loss


def _frozen_reasoner() -> TRM:
    torch.manual_seed(1)
    model = TRM(
        dim=16, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1, N_sup=2,
        heads=4, max_grid_size=8,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _core(model: TRM, sha: str = "core-sha") -> LoadedTRMCheckpoint:
    payload = {
        "architecture": {
            "class": "TRM", "dim": model.dim, "num_tokens": model.num_tokens,
            "seq_len": model.seq_len, "n_layers": len(model.blocks), "n": model.n,
            "T": model.T, "N_sup": model.N_sup, "max_grid_size": model.max_grid_size,
            "ternary": model.ternary, "act8": model.act8,
        },
        "task": {"task": "sudoku", "num_tokens": model.num_tokens},
    }
    return LoadedTRMCheckpoint(
        Path("core.pt"), sha, payload, model, "ema",
        sum(p.numel() for p in model.parameters()), torch.device("cpu"), "pytorch_eager",
    )


def test_grounded_target_api_has_no_reference_answer_argument():
    sig = inspect.signature(one_cycle_improvement_target)
    forbidden = {"target", "answer", "solution", "reference", "y_target"}
    assert forbidden.isdisjoint(sig.parameters)
    meta = grounding_metadata()
    assert meta["target_id"] == GROUNDED_TARGET_ID
    assert meta["reference_target_used"] is False
    assert meta["bootstrapped"] is False


def test_grounded_target_keeps_reasoner_frozen_and_binary():
    model = _frozen_reasoner()
    before = tensor_state_sha256(model)
    x = torch.randint(0, 5, (3, 16))
    y = torch.zeros(3, 16, 16)
    z = torch.zeros_like(y)
    out = one_cycle_improvement_target(model, x, y, z, height=4, width=4, box=2)
    after = tensor_state_sha256(model)
    assert before == after
    assert set(out["label"].tolist()) <= {0.0, 1.0}
    assert not any(p.requires_grad for p in model.parameters())
    assert_frozen_reasoner(model)


def test_target_rejects_unfrozen_reasoner():
    model = _frozen_reasoner()
    next(model.parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError, match="requires_grad"):
        one_cycle_improvement_target(
            model, torch.zeros(1, 16, dtype=torch.long),
            torch.zeros(1, 16, 16), torch.zeros(1, 16, 16),
            height=4, width=4, box=2,
        )


def test_full_state_verifier_requires_y_and_versions_representation():
    model = GroundedStateVerifier(5, 16, n_layers=1, heads=4, max_grid_size=8)
    assert model.include_y is True
    assert model.state_representation == STATE_REPRESENTATION_VERSION
    x = torch.randint(0, 5, (2, 16))
    y = torch.randn(2, 16, 16)
    z = torch.randn_like(y)
    p = model(x, y, z, width=4)
    assert p.shape == (2,) and ((p >= 0) & (p <= 1)).all()
    with pytest.raises(TypeError):
        model(x, z, width=4)  # y cannot be silently omitted


def test_grounded_loss_rejects_nonbinary_targets():
    model = GroundedStateVerifier(5, 16, n_layers=1, heads=4, max_grid_size=8)
    x = torch.randint(0, 5, (2, 16)); y = torch.randn(2, 16, 16); z = torch.randn_like(y)
    with pytest.raises(ValueError, match="binary"):
        grounded_improvement_bce_loss(model, x, y, z, torch.tensor([0.0, 0.5]), width=4)


def test_checkpoint_binds_grounding_representation_and_exact_core(tmp_path):
    reasoner = _frozen_reasoner()
    core = _core(reasoner)
    verifier = EnsembleGroundedStateVerifier(
        5, 16, n_members=2, n_layers=1, heads=4, max_grid_size=8
    )
    path = tmp_path / "verifier.pt"
    grounding = grounding_metadata()
    save_grounded_verifier_checkpoint(
        verifier, path, core=core, trained_steps=3, grounding=grounding,
        member_seeds=[1, 2], data_provenance={"test_reference_target_used": False},
    )
    loaded = load_grounded_verifier_checkpoint(path, core)
    assert isinstance(loaded.module, EnsembleGroundedStateVerifier)
    assert loaded.payload["grounding"]["reference_target_used"] is False
    assert loaded.payload["architecture"]["state_representation"] == STATE_REPRESENTATION_VERSION

    wrong_core = _core(reasoner, sha="different-core")
    with pytest.raises(EvaluationContractError, match="incompatible"):
        load_grounded_verifier_checkpoint(path, wrong_core)


def test_checkpoint_refuses_circular_or_leaky_grounding():
    reasoner = _frozen_reasoner(); core = _core(reasoner)
    verifier = GroundedStateVerifier(5, 16, n_layers=1, heads=4, max_grid_size=8)
    bad = grounding_metadata(); bad["bootstrapped"] = True
    with pytest.raises(EvaluationContractError, match="grounding"):
        grounded_verifier_payload(
            verifier, core=core, trained_steps=1, grounding=bad,
            member_seeds=[1], data_provenance={},
        )
    bad2 = grounding_metadata(); bad2["reference_target_used"] = True
    with pytest.raises(EvaluationContractError, match="grounding"):
        grounded_verifier_payload(
            verifier, core=core, trained_steps=1, grounding=bad2,
            member_seeds=[1], data_provenance={},
        )


def test_z_only_ablation_cannot_be_saved_as_accepted_grounded_verifier():
    reasoner = _frozen_reasoner(); core = _core(reasoner)
    zonly = GroundedStateVerifier(
        5, 16, n_layers=1, heads=4, max_grid_size=8, include_y=False
    )
    with pytest.raises(EvaluationContractError, match="must include y"):
        grounded_verifier_payload(
            zonly, core=core, trained_steps=1, grounding=grounding_metadata(),
            member_seeds=[1], data_provenance={},
        )


def test_grounded_mcts_value_receives_node_y_and_z():
    class Recorder:
        def __init__(self): self.seen = None
        def value_state(self, x, y, z, width):
            self.seen = (x.clone(), y.clone(), z.clone(), width)
            return torch.tensor([0.75])

    mcts = GroundedLatentNativeMCTS.__new__(GroundedLatentNativeMCTS)
    mcts.verifier = Recorder(); mcts.width = 4; mcts.uncertainty_beta = 0.0
    y = torch.randn(1, 16, 8)
    codes = torch.zeros(1, 16, 8, dtype=torch.int8)
    scale = torch.ones(1, 16, 1)
    node = _LatentNode(y, codes, scale)
    x = torch.randint(0, 5, (1, 16))
    value = mcts._value(x, node)
    assert value == pytest.approx(0.75)
    assert torch.equal(mcts.verifier.seen[1], y)
    assert torch.equal(mcts.verifier.seen[2], node.latent())


def test_grounded_mcts_refuses_legacy_batched_z_only_path():
    mcts = GroundedLatentNativeMCTS.__new__(GroundedLatentNativeMCTS)
    with pytest.raises(RuntimeError, match="incomplete state"):
        mcts._value_batch(torch.zeros(1, 16, dtype=torch.long), torch.zeros(1, 16, 8))


def test_mcts_backup_provenance_is_explicitly_bootstrapped():
    mcts = GroundedLatentNativeMCTS.__new__(GroundedLatentNativeMCTS)
    meta = mcts.prm_target_metadata()
    assert meta["target_kind"] == MCTS_BOOTSTRAP_TARGET_KIND
    assert meta["independent_ground_truth"] is False
