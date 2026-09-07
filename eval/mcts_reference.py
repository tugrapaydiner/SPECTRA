"""Small deterministic MCTS reference for Milestone 08.

This module intentionally has no dependency on torch, TRM, latent actions, or neural
verifiers.  It exists so selection/expansion/evaluation/backup semantics can be
checked against hand-computed tree statistics before comparing the neural search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable


Path = tuple[int, ...]
ValueFn = Callable[[Path], float]


@dataclass
class ReferenceNode:
    path: Path
    prior: float = 1.0
    action_from_parent: int | None = None
    visits: int = 0
    value_sum: float = 0.0
    children: list["ReferenceNode"] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.path)

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class DeterministicReferenceMCTS:
    """Pure reference implementation matching the M08 serial search contract."""

    def __init__(
        self,
        *,
        priors: list[float],
        value_fn: ValueFn,
        n_rollouts: int,
        max_depth: int,
        c_puct: float = 1.0,
    ) -> None:
        if not isinstance(n_rollouts, int) or isinstance(n_rollouts, bool) or n_rollouts < 0:
            raise ValueError("n_rollouts must be an integer >= 0")
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if not priors or any((not math.isfinite(float(p)) or float(p) < 0.0) for p in priors):
            raise ValueError("priors must be a non-empty list of finite non-negative values")
        total = float(sum(priors))
        if total <= 0.0:
            raise ValueError("priors must have positive total mass")
        if not math.isfinite(float(c_puct)) or float(c_puct) < 0.0:
            raise ValueError("c_puct must be finite and non-negative")
        self.priors = [float(p) / total for p in priors]
        self.value_fn = value_fn
        self.n_rollouts = n_rollouts
        self.max_depth = max_depth
        self.c_puct = float(c_puct)
        self.root: ReferenceNode | None = None
        self.best_path: Path = ()
        self.best_value = -math.inf
        self.work: dict[str, int | float | str | list[int]] = {}

    def _reset(self) -> None:
        self.root = ReferenceNode(())
        self.best_path = ()
        self.best_value = -math.inf
        self.work = {
            "requested_rollouts": self.n_rollouts,
            "completed_rollouts": 0,
            "expansion_calls": 0,
            "transition_calls": 0,
            "verifier_evaluations": 0,
            "selection_edges": 0,
            "max_depth_reached": 0,
            "status": "running",
        }

    def _expand(self, node: ReferenceNode) -> None:
        if node.depth >= self.max_depth or node.children:
            return
        node.children = [
            ReferenceNode(
                node.path + (action,),
                prior=prior,
                action_from_parent=action,
            )
            for action, prior in enumerate(self.priors)
        ]
        self.work["expansion_calls"] = int(self.work["expansion_calls"]) + 1
        self.work["transition_calls"] = int(self.work["transition_calls"]) + len(self.priors)
        self.work["max_depth_reached"] = max(
            int(self.work["max_depth_reached"]), node.depth + 1
        )

    def _puct(self, parent: ReferenceNode, child: ReferenceNode) -> float:
        explore = (
            self.c_puct
            * child.prior
            * math.sqrt(parent.visits + 1)
            / (1 + child.visits)
        )
        return child.q + explore

    def _select(self) -> tuple[list[ReferenceNode], ReferenceNode]:
        assert self.root is not None
        path = [self.root]
        node = self.root
        while node.children and node.depth < self.max_depth:
            # Children are action ordered; max keeps the first exact maximum.
            node = max(node.children, key=lambda c: self._puct(path[-1], c))
            path.append(node)
            self.work["selection_edges"] = int(self.work["selection_edges"]) + 1
        return path, node

    def search(self) -> ReferenceNode:
        self._reset()
        assert self.root is not None
        if self.n_rollouts == 0:
            self.best_path = ()
            self.best_value = float("nan")
            self.work["status"] = "complete"
            return self.root

        self._expand(self.root)  # initial expansion is real work
        for _ in range(self.n_rollouts):
            path, leaf = self._select()
            if leaf.depth < self.max_depth and not leaf.children:
                self._expand(leaf)
            value = float(self.value_fn(leaf.path))
            if not math.isfinite(value):
                raise ValueError("reference value_fn must return a finite scalar")
            self.work["verifier_evaluations"] = int(self.work["verifier_evaluations"]) + 1
            self.work["completed_rollouts"] = int(self.work["completed_rollouts"]) + 1
            if value > self.best_value:
                self.best_value = value
                self.best_path = leaf.path
            for node in path:
                node.visits += 1
                node.value_sum += value
        self.work["status"] = "complete"
        return self._find(self.best_path)

    def _find(self, path: Path) -> ReferenceNode:
        assert self.root is not None
        node = self.root
        for action in path:
            node = node.children[action]
        return node

    def export_tree(self) -> dict[str, object]:
        if self.root is None:
            raise RuntimeError("run search first")
        nodes: list[dict[str, object]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            nodes.append(
                {
                    "path": list(node.path),
                    "depth": node.depth,
                    "action_from_parent": node.action_from_parent,
                    "prior": node.prior,
                    "visits": node.visits,
                    "value_sum": node.value_sum,
                    "q": node.q,
                    "children": [list(child.path) for child in node.children],
                }
            )
            stack.extend(reversed(node.children))
        return {
            "nodes": nodes,
            "best_path": list(self.best_path),
            "best_value": self.best_value,
            "work": dict(self.work),
        }


def frozen_reference_value(path: Path) -> float:
    """Hand-checkable M08 fixture values fixed in docs/M08_PROTOCOL.md."""
    return {
        (0,): 0.2,
        (1,): 0.8,
        (1, 0): 0.9,
    }.get(path, 0.0)


def run_frozen_reference() -> dict[str, object]:
    search = DeterministicReferenceMCTS(
        priors=[0.5, 0.5],
        value_fn=frozen_reference_value,
        n_rollouts=4,
        max_depth=2,
        c_puct=1.0,
    )
    search.search()
    return search.export_tree()
