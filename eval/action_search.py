"""M09 search adapters for trained state-conditioned latent actions.

Milestone 08's :class:`LatentNativeMCTS` remains unchanged.  This module adds the
small policy-prior hook needed by M09 and a transparent symbolic-oracle evaluator
used only to isolate action quality in the acceptance experiment.
"""
from __future__ import annotations

import torch

from eval.latent_mcts import LatentNativeMCTS, _LatentNode
from model.verifier import sudoku_score


class StateConditionedLatentNativeMCTS(LatentNativeMCTS):
    """M08 serial/batched semantics with optional ``P(a|x,y,z)`` priors.

    Codebooks exposing ``priors_for_state`` receive the complete current search
    state. Legacy codebooks still use their historical global ``priors()`` path.
    """

    def _reset_search_state(self, mode: str) -> None:
        super()._reset_search_state(mode)
        self.last_search_stats["state_conditioned_prior_calls"] = 0
        self.last_search_stats["global_prior_calls"] = 0
        self.last_search_stats["policy_forward_calls"] = 0

    def _action_priors(self, node: _LatentNode) -> torch.Tensor:
        if hasattr(self.codebook, "priors_for_state"):
            x = getattr(self, "_m09_search_x", None)
            if x is None:
                raise RuntimeError("state-conditioned action search has no active problem context")
            priors = self.codebook.priors_for_state(x, node.y, node.latent(), width=self.width)
            self.last_search_stats["state_conditioned_prior_calls"] = int(
                self.last_search_stats["state_conditioned_prior_calls"]
            ) + 1
            self.last_search_stats["policy_forward_calls"] = int(
                self.last_search_stats["policy_forward_calls"]
            ) + 1
            if priors.ndim == 2:
                if priors.shape[0] != 1:
                    raise RuntimeError("per-example native search expects one prior row")
                priors = priors[0]
        else:
            priors = self.codebook.priors()
            self.last_search_stats["global_prior_calls"] = int(
                self.last_search_stats["global_prior_calls"]
            ) + 1
        priors = torch.as_tensor(priors)
        if priors.ndim != 1 or priors.numel() != int(self.codebook.n_actions):
            raise RuntimeError("action prior shape does not match n_actions")
        if not torch.isfinite(priors).all() or (priors < 0).any():
            raise RuntimeError("action priors must be finite and non-negative")
        total = float(priors.sum())
        if total <= 0.0:
            raise RuntimeError("action priors must have positive mass")
        # Search consumes normalized probabilities even for custom policy adapters.
        return priors / priors.sum()

    def _expand(self, node: _LatentNode, x_emb, *, initial: bool = False) -> bool:
        if node.children or node.depth >= self.max_depth:
            return False
        priors = self._action_priors(node)
        before = int(self.last_search_stats["transition_calls"])
        for action in range(int(self.codebook.n_actions)):
            y, z_codes, z_scale = self._step(x_emb, node, action)
            node.children.append(
                _LatentNode(
                    y,
                    z_codes,
                    z_scale,
                    prior=float(priors[action]),
                    depth=node.depth + 1,
                    action_from_parent=action,
                    path=node.path + (action,),
                )
            )
        self.last_search_stats["expansion_calls"] = int(self.last_search_stats["expansion_calls"]) + 1
        self.last_search_stats["max_depth_reached"] = max(
            int(self.last_search_stats["max_depth_reached"]), node.depth + 1
        )
        if initial:
            self.last_search_stats["initial_expansion_calls"] = int(
                self.last_search_stats["initial_expansion_calls"]
            ) + 1
            self.last_search_stats["initial_expansion_transition_calls"] = int(
                self.last_search_stats["initial_expansion_transition_calls"]
            ) + (int(self.last_search_stats["transition_calls"]) - before)
        return True

    @torch.no_grad()
    def search(self, x: torch.Tensor):
        self._m09_search_x = x
        try:
            return super().search(x)
        finally:
            self._m09_search_x = None

    @torch.no_grad()
    def search_batched(self, x: torch.Tensor, leaf_batch: int = 8, virtual_loss: float = 1.0):
        self._m09_search_x = x
        try:
            return super().search_batched(x, leaf_batch=leaf_batch, virtual_loss=virtual_loss)
        finally:
            self._m09_search_x = None


class OracleScoreActionMCTS(StateConditionedLatentNativeMCTS):
    """Transparent M09 action-isolation evaluator.

    This deliberately decodes during evaluation and returns the validated symbolic
    Sudoku structural score. It is an experiment harness, not the production
    no-decode native verifier path.
    """

    def _value(self, x: torch.Tensor, node: _LatentNode) -> float:
        answer = self.model.out_head(node.y).argmax(dim=-1)
        return float(sudoku_score(x, answer, box=3).detach().mean())

    def _value_batch(self, x_rep: torch.Tensor, z_batch: torch.Tensor):  # pragma: no cover
        del x_rep, z_batch
        raise RuntimeError(
            "M09 oracle action-isolation evaluation is serial; batched symbolic evaluation "
            "would require the corresponding y state for every leaf"
        )
