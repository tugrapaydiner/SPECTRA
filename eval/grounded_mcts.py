"""M07-native search adapter for complete-state grounded verifiers.

The legacy ``LatentNativeMCTS`` is preserved for historical z-only checkpoints.
This adapter refuses to silently omit ``y`` when the verifier target is defined on
``search_state_xyz_v1``.
"""
from __future__ import annotations

import torch

from eval.latent_mcts import LatentNativeMCTS
from train.distill import MCTS_BOOTSTRAP_TARGET_KIND


class GroundedLatentNativeMCTS(LatentNativeMCTS):
    """Native MCTS whose verifier receives the complete ``(x,y,z)`` state."""

    def _value(self, x, node):
        z = node.latent()
        if self.uncertainty_beta > 0.0:
            if not hasattr(self.verifier, "value_with_uncertainty_state"):
                raise RuntimeError(
                    "grounded MCTS uncertainty_beta requires value_with_uncertainty_state; "
                    "z-only uncertainty is not a substitute"
                )
            value, std = self.verifier.value_with_uncertainty_state(
                x, node.y, z, self.width
            )
            return float((value - self.uncertainty_beta * std).detach().mean())
        if not hasattr(self.verifier, "value_state"):
            raise RuntimeError(
                "grounded native MCTS requires verifier.value_state(x,y,z,width)"
            )
        return float(self.verifier.value_state(x, node.y, z, self.width).detach().mean())

    def _value_batch(self, x_rep, z_batch):  # pragma: no cover - defensive API boundary
        raise RuntimeError(
            "legacy search_batched supplies z without the corresponding y batch; "
            "M07 grounded verifier refuses this incomplete state representation"
        )

    @torch.no_grad()
    def search_batched(self, *args, **kwargs):  # pragma: no cover - explicit unsupported path
        raise RuntimeError(
            "GroundedLatentNativeMCTS.search_batched is disabled until batched leaves "
            "carry y and z together under search_state_xyz_v1"
        )

    def prm_target_metadata(self) -> dict[str, object]:
        """Label inherited MCTS backups as bootstrapped, never independent truth."""
        return {
            "target_kind": MCTS_BOOTSTRAP_TARGET_KIND,
            "independent_ground_truth": False,
            "source": "LatentNativeMCTS q=W/N backup",
        }
