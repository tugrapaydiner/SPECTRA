"""Additional M08 failure-path invariant: virtual loss must clean on expansion errors."""
from __future__ import annotations

import pytest
import torch

from eval.latent_mcts import LatentNativeMCTS
from tests.test_m08_mcts_reference import TinyModel, TinyVerifier, x1


class FailDuringLeafExpansionCodebook:
    n_actions = 2

    def __init__(self):
        self.calls = 0

    def apply_action(self, z, action):
        self.calls += 1
        # Initial root expansion consumes calls 1 and 2. During the first selected
        # leaf expansion action 0 succeeds (call 3), action 1 fails (call 4).
        if self.calls == 4:
            raise RuntimeError("intentional transition failure")
        return z if action == 0 else z + 0.5

    def priors(self):
        return torch.tensor([0.5, 0.5], dtype=torch.float32)


def test_virtual_loss_cleanup_when_leaf_expansion_raises():
    search = LatentNativeMCTS(
        TinyModel(),
        TinyVerifier(),
        FailDuringLeafExpansionCodebook(),
        height=2,
        width=2,
        n_rollouts=2,
        max_depth=3,
    )
    with pytest.raises(RuntimeError, match="intentional transition failure"):
        search.search_batched(x1(), leaf_batch=2, virtual_loss=0.75)

    exported = search.export_tree()
    work = exported["work"]
    assert work["status"] == "error"
    assert work["completed_rollouts"] == 0
    assert work["virtual_loss_applications"] > 0
    assert work["virtual_loss_applications"] == work["virtual_loss_cleanups"]
    assert work["virtual_loss_outstanding"] == 0
    # Initial expansion (2 transitions) + two attempted leaf transitions. The
    # failing action is counted because it was actual attempted work.
    assert work["transition_calls"] == 4
    assert all(row["visits"] == 0 for row in exported["nodes"])
    assert all(row["value_sum"] == pytest.approx(0.0) for row in exported["nodes"])
