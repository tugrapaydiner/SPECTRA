"""Latent MCTS implementations.

`LatentMCTS` is the older decoded/noise reference retained for compatibility.
`LatentNativeMCTS` is the INT8-latent search path. Milestone 08 makes the native
path's state, horizon, rollout budget, batching, tie-breaking, tree export, and
actual-work accounting explicit and testable.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from model.energy import LatentEnergyVerifier
from model.latent_action import LatentActionCodebook
from model.trm import TRM

# value_fn(x[1, L], answer[1, L]) -> float (higher is better)
ValueFn = Callable[[torch.Tensor, torch.Tensor], float]


class _Node:
    __slots__ = ("y", "z", "prior", "children", "visits", "value_sum")

    def __init__(self, y: torch.Tensor, z: torch.Tensor, prior: float = 1.0):
        self.y = y
        self.z = z
        self.prior = prior
        self.children: list[_Node] = []
        self.visits = 0
        self.value_sum = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class LatentMCTS:
    """Historical decoded/noise search retained unchanged for compatibility."""

    def __init__(
        self,
        model: TRM,
        value_fn: ValueFn,
        height: int,
        width: int,
        n_rollouts: int = 32,
        n_children: int = 3,
        c_puct: float = 1.5,
        noise_std: float = 0.3,
    ):
        self.model = model
        self.value_fn = value_fn
        self.height = height
        self.width = width
        self.n_rollouts = n_rollouts
        self.n_children = n_children
        self.c_puct = c_puct
        self.noise_std = noise_std

    def _step_latent(
        self, x_emb: torch.Tensor, y: torch.Tensor, z: torch.Tensor, noisy: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noisy:
            z = z + self.noise_std * torch.randn_like(z)
        for _ in range(self.model.T):
            y, z = self.model.recursive_cycle(x_emb, y, z)
        return y, z

    def _expand(self, node: _Node, x_emb: torch.Tensor) -> None:
        for c in range(self.n_children):
            y, z = self._step_latent(x_emb, node.y, node.z, noisy=(c > 0))
            node.children.append(_Node(y, z, prior=1.0 / self.n_children))

    def _puct(self, parent: _Node, child: _Node) -> float:
        explore = self.c_puct * child.prior * math.sqrt(parent.visits + 1) / (1 + child.visits)
        return child.q + explore

    def _evaluate(self, x: torch.Tensor, node: _Node) -> tuple[float, torch.Tensor]:
        logits = self.model.out_head(node.y)
        answer = logits.argmax(dim=-1)
        return self.value_fn(x, answer), answer

    @torch.no_grad()
    def search(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.model.token_embed(x) + self.model.encode_positions(
            x, self.height, self.width
        )
        root = _Node(torch.zeros_like(x_emb), torch.zeros_like(x_emb))
        self._expand(root, x_emb)
        best_value, best_answer = self._evaluate(x, root)

        for _ in range(self.n_rollouts):
            path = [root]
            node = root
            while node.children:
                node = max(node.children, key=lambda c: self._puct(node, c))
                path.append(node)
            self._expand(node, x_emb)
            value, answer = self._evaluate(x, node)
            if value > best_value:
                best_value, best_answer = value, answer
            for nd in path:
                nd.visits += 1
                nd.value_sum += value
        return best_answer


def _quantize_int8(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric INT8 quantize: returns `(codes int8, scale float)`."""
    scale = z.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    codes = torch.clamp(torch.round(z / scale), -128, 127).to(torch.int8)
    return codes, scale


def _dequantize_int8(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return codes.to(scale.dtype) * scale


class _LatentNode:
    """Inspectable native-search node.

    Environment state is `(y, z_codes, z_scale)`; `depth`, `action_from_parent`,
    and `path` make trajectory identity explicit. Search statistics remain FP/int.
    """

    __slots__ = (
        "y", "z_codes", "z_scale", "prior", "children", "visits", "value_sum",
        "depth", "action_from_parent", "path",
    )

    def __init__(
        self,
        y: torch.Tensor,
        z_codes: torch.Tensor,
        z_scale: torch.Tensor,
        prior: float = 1.0,
        *,
        depth: int = 0,
        action_from_parent: int | None = None,
        path: tuple[int, ...] = (),
    ):
        self.y = y
        self.z_codes = z_codes
        self.z_scale = z_scale
        self.prior = float(prior)
        self.children: list[_LatentNode] = []
        self.visits = 0
        self.value_sum = 0.0
        self.depth = int(depth)
        self.action_from_parent = action_from_parent
        self.path = tuple(int(a) for a in path)

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def latent(self) -> torch.Tensor:
        return _dequantize_int8(self.z_codes, self.z_scale)


class LatentNativeMCTS:
    """INT8-latent MCTS with explicit Milestone-08 semantics.

    Positive-budget search never decodes during selection/evaluation. `n_rollouts`
    means the exact number of real verifier evaluations. Initial expansion is real
    transition work and is counted separately. `n_rollouts == 0` is the explicit
    ordinary greedy TRM baseline, not a decode of the all-zero root state.
    """

    def __init__(
        self,
        model: TRM,
        energy_verifier: LatentEnergyVerifier,
        action_codebook: LatentActionCodebook,
        height: int,
        width: int,
        n_rollouts: int = 32,
        c_puct: float = 1.5,
        uncertainty_beta: float = 0.0,
        latent_vq=None,
        max_depth: int = 32,
    ):
        self.model = model
        self.verifier = energy_verifier
        self.codebook = action_codebook
        self.height = height
        self.width = width
        self.n_rollouts = n_rollouts
        self.c_puct = c_puct
        self.uncertainty_beta = uncertainty_beta
        self.latent_vq = latent_vq
        self.max_depth = max_depth
        self.root: _LatentNode | None = None
        self.best_node: _LatentNode | None = None
        self.best_value: float | None = None
        self.last_search_stats: dict[str, object] = {}
        self._validate_configuration()

    @staticmethod
    def _is_int(value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def _validate_configuration(self) -> None:
        if not self._is_int(self.height) or self.height <= 0:
            raise ValueError("height must be a positive integer")
        if not self._is_int(self.width) or self.width <= 0:
            raise ValueError("width must be a positive integer")
        if not self._is_int(self.n_rollouts) or self.n_rollouts < 0:
            raise ValueError("n_rollouts must be an integer >= 0")
        if not self._is_int(self.max_depth) or self.max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if not math.isfinite(float(self.c_puct)) or float(self.c_puct) < 0.0:
            raise ValueError("c_puct must be finite and non-negative")
        if not math.isfinite(float(self.uncertainty_beta)) or float(self.uncertainty_beta) < 0.0:
            raise ValueError("uncertainty_beta must be finite and non-negative")
        if not self._is_int(self.codebook.n_actions) or int(self.codebook.n_actions) <= 0:
            raise ValueError("action codebook must expose n_actions >= 1")

    def _validate_x(self, x: torch.Tensor) -> None:
        self._validate_configuration()  # also catches invalid mutations after init
        if not torch.is_tensor(x) or x.ndim != 2 or x.shape[0] != 1:
            raise ValueError("native MCTS requires x with shape [1, L]")
        if x.shape[1] != self.height * self.width:
            raise ValueError(
                f"x sequence length must equal height*width={self.height * self.width}"
            )

    def _reset_search_state(self, mode: str) -> None:
        self.root = None
        self.best_node = None
        self.best_value = None
        self.last_search_stats = {
            "mode": mode,
            "status": "running",
            "requested_rollouts": int(self.n_rollouts),
            "completed_rollouts": 0,
            "max_depth": int(self.max_depth),
            "max_depth_reached": 0,
            "initial_expansion_calls": 0,
            "initial_expansion_transition_calls": 0,
            "expansion_calls": 0,
            "transition_calls": 0,
            "recursive_cycle_calls": 0,
            "latent_vq_calls": 0,
            "verifier_calls": 0,
            "verifier_evaluations": 0,
            "selection_edges": 0,
            "leaf_batches": 0,
            "batch_sizes": [],
            "virtual_loss_applications": 0,
            "virtual_loss_cleanups": 0,
            "greedy_forward_calls": 0,
            "decode_calls": 0,
            "evaluated_paths": [],
            "best_path": None,
            "best_value": None,
            "error_type": None,
            "error_message": None,
        }

    def _record_error(self, exc: Exception) -> None:
        self.last_search_stats["status"] = "error"
        self.last_search_stats["error_type"] = type(exc).__name__
        self.last_search_stats["error_message"] = str(exc)

    def _make_zero_root(self, x_emb: torch.Tensor) -> _LatentNode:
        z0_codes, z0_scale = _quantize_int8(torch.zeros_like(x_emb))
        return _LatentNode(torch.zeros_like(x_emb), z0_codes, z0_scale, depth=0, path=())

    def _greedy_root(self, x: torch.Tensor) -> _LatentNode:
        _, steps = self.model(x, height=self.height, width=self.width)
        if not steps:
            raise RuntimeError("TRM greedy forward produced no supervision steps")
        final = steps[-1]
        z_codes, z_scale = _quantize_int8(final["z"])
        root = _LatentNode(final["y"], z_codes, z_scale, depth=0, path=())
        self.root = root
        self.best_node = root
        self.last_search_stats["greedy_forward_calls"] = 1
        self.last_search_stats["max_depth_reached"] = 0
        self.last_search_stats["best_path"] = []
        self.last_search_stats["status"] = "complete"
        return root

    def _step(self, x_emb, node: _LatentNode, action: int):
        """One action transition; all actual transition work is counted."""
        if not self._is_int(action) or not 0 <= action < int(self.codebook.n_actions):
            raise ValueError(f"action must be in [0, {int(self.codebook.n_actions) - 1}]")
        self.last_search_stats["transition_calls"] = int(self.last_search_stats["transition_calls"]) + 1
        z = self.codebook.apply_action(node.latent(), action)
        y = node.y
        for _ in range(self.model.T):
            y, z = self.model.recursive_cycle(x_emb, y, z)
            self.last_search_stats["recursive_cycle_calls"] = int(
                self.last_search_stats["recursive_cycle_calls"]
            ) + 1
        if self.latent_vq is not None:
            z = self.latent_vq.snap(z)
            self.last_search_stats["latent_vq_calls"] = int(self.last_search_stats["latent_vq_calls"]) + 1
        z_codes, z_scale = _quantize_int8(z)
        return y, z_codes, z_scale

    def _expand(self, node: _LatentNode, x_emb, *, initial: bool = False) -> bool:
        """Materialize all action children exactly once, unless at the horizon."""
        if node.children or node.depth >= self.max_depth:
            return False
        priors = self.codebook.priors()
        if len(priors) != int(self.codebook.n_actions):
            raise RuntimeError("codebook prior count does not match n_actions")
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

    def _puct(self, parent: _LatentNode, child: _LatentNode) -> float:
        explore = self.c_puct * child.prior * math.sqrt(parent.visits + 1) / (1 + child.visits)
        return child.q + explore

    def _select(self) -> tuple[list[_LatentNode], _LatentNode]:
        if self.root is None:
            raise RuntimeError("search root is not initialized")
        path = [self.root]
        node = self.root
        while node.children and node.depth < self.max_depth:
            parent = node
            # Children are action ordered; max retains the first exact maximum.
            node = max(parent.children, key=lambda c: self._puct(parent, c))
            path.append(node)
            self.last_search_stats["selection_edges"] = int(
                self.last_search_stats["selection_edges"]
            ) + 1
        return path, node

    def _value(self, x, node):
        z = node.latent()
        if self.uncertainty_beta > 0.0 and hasattr(self.verifier, "value_with_uncertainty"):
            value, std = self.verifier.value_with_uncertainty(x, z, self.width)
            return float((value - self.uncertainty_beta * std).detach().mean())
        return float(self.verifier.value(x, z, self.width).detach().mean())

    def _value_batch(self, x_rep, z_batch):
        if self.uncertainty_beta > 0.0 and hasattr(self.verifier, "value_with_uncertainty"):
            value, std = self.verifier.value_with_uncertainty(x_rep, z_batch, self.width)
            return value - self.uncertainty_beta * std
        return self.verifier.value(x_rep, z_batch, self.width)

    def _consider_best(self, node: _LatentNode, value: float) -> None:
        # Strict > preserves earliest-evaluated deterministic ties.
        if self.best_value is None or value > self.best_value:
            self.best_value = float(value)
            self.best_node = node
            self.last_search_stats["best_path"] = list(node.path)
            self.last_search_stats["best_value"] = float(value)

    def _backup(self, path: list[_LatentNode], value: float) -> None:
        for node in path:
            node.visits += 1
            node.value_sum += float(value)

    def _record_successful_evaluation(self, node: _LatentNode, value: float) -> None:
        if not math.isfinite(float(value)):
            raise RuntimeError("verifier produced a non-finite value")
        self.last_search_stats["verifier_evaluations"] = int(
            self.last_search_stats["verifier_evaluations"]
        ) + 1
        self.last_search_stats["completed_rollouts"] = int(
            self.last_search_stats["completed_rollouts"]
        ) + 1
        self.last_search_stats["evaluated_paths"].append(list(node.path))
        self._consider_best(node, float(value))

    @torch.no_grad()
    def search(self, x: torch.Tensor) -> _LatentNode:
        """Serial M08 search. One requested rollout = one real verifier evaluation."""
        self._validate_x(x)
        self._reset_search_state("serial" if self.n_rollouts > 0 else "greedy_zero_search")
        try:
            if self.n_rollouts == 0:
                return self._greedy_root(x)
            x_emb = self.model.token_embed(x) + self.model.encode_positions(x, self.height, self.width)
            self.root = self._make_zero_root(x_emb)
            self._expand(self.root, x_emb, initial=True)
            for _ in range(self.n_rollouts):
                path, leaf = self._select()
                if leaf.depth < self.max_depth and not leaf.children:
                    self._expand(leaf, x_emb)
                self.last_search_stats["verifier_calls"] = int(
                    self.last_search_stats["verifier_calls"]
                ) + 1
                value = self._value(x, leaf)
                self._record_successful_evaluation(leaf, value)
                self._backup(path, value)
            self.last_search_stats["status"] = "complete"
            assert self.best_node is not None
            return self.best_node
        except Exception as exc:
            self._record_error(exc)
            raise

    def _apply_virtual_loss(self, path: list[_LatentNode], virtual_loss: float) -> None:
        for node in path:
            node.visits += 1
            node.value_sum -= virtual_loss
            self.last_search_stats["virtual_loss_applications"] = int(
                self.last_search_stats["virtual_loss_applications"]
            ) + 1

    def _cleanup_virtual_loss(
        self,
        entries: list[tuple[list[_LatentNode], _LatentNode]],
        virtual_loss: float,
    ) -> None:
        for path, _ in entries:
            for node in path:
                node.visits -= 1
                node.value_sum += virtual_loss
                if node.visits < 0:
                    raise RuntimeError("virtual-loss cleanup produced negative visits")
                self.last_search_stats["virtual_loss_cleanups"] = int(
                    self.last_search_stats["virtual_loss_cleanups"]
                ) + 1

    @torch.no_grad()
    def search_batched(
        self, x: torch.Tensor, leaf_batch: int = 8, virtual_loss: float = 1.0
    ) -> _LatentNode:
        """Virtual-loss leaf batching with exact rollout-budget semantics.

        `leaf_batch == 1` is required to agree with serial search under equivalent
        deterministic settings. Larger batches can follow different trajectories
        because multiple selections occur before their real values are known.
        """
        self._validate_x(x)
        if not self._is_int(leaf_batch) or leaf_batch <= 0:
            raise ValueError("leaf_batch must be a positive integer")
        if not math.isfinite(float(virtual_loss)) or float(virtual_loss) < 0.0:
            raise ValueError("virtual_loss must be finite and non-negative")
        self._reset_search_state("batched" if self.n_rollouts > 0 else "greedy_zero_search")
        self.last_search_stats["leaf_batch"] = int(leaf_batch)
        self.last_search_stats["virtual_loss"] = float(virtual_loss)
        try:
            if self.n_rollouts == 0:
                return self._greedy_root(x)
            x_emb = self.model.token_embed(x) + self.model.encode_positions(x, self.height, self.width)
            self.root = self._make_zero_root(x_emb)
            self._expand(self.root, x_emb, initial=True)

            remaining = int(self.n_rollouts)
            while remaining > 0:
                current = min(int(leaf_batch), remaining)
                entries: list[tuple[list[_LatentNode], _LatentNode]] = []
                try:
                    for _ in range(current):
                        path, leaf = self._select()
                        self._apply_virtual_loss(path, float(virtual_loss))
                        entries.append((path, leaf))
                        if leaf.depth < self.max_depth and not leaf.children:
                            self._expand(leaf, x_emb)

                    leaves = [leaf for _, leaf in entries]
                    z_batch = torch.cat([leaf.latent() for leaf in leaves], dim=0)
                    x_rep = x.expand(len(leaves), -1)
                    self.last_search_stats["verifier_calls"] = int(
                        self.last_search_stats["verifier_calls"]
                    ) + 1
                    raw_values = self._value_batch(x_rep, z_batch)
                    values_tensor = torch.as_tensor(raw_values).detach().reshape(-1)
                    if values_tensor.numel() != current:
                        raise RuntimeError(
                            f"batched verifier returned {values_tensor.numel()} values for {current} leaves"
                        )
                    values = [float(v) for v in values_tensor.cpu().tolist()]
                    if not all(math.isfinite(v) for v in values):
                        raise RuntimeError("batched verifier produced a non-finite value")
                except Exception:
                    self._cleanup_virtual_loss(entries, float(virtual_loss))
                    raise

                # Virtual statistics never become real/exported statistics.
                self._cleanup_virtual_loss(entries, float(virtual_loss))
                self.last_search_stats["leaf_batches"] = int(
                    self.last_search_stats["leaf_batches"]
                ) + 1
                self.last_search_stats["batch_sizes"].append(current)
                for (path, leaf), value in zip(entries, values):
                    self._record_successful_evaluation(leaf, value)
                    self._backup(path, value)
                remaining -= current

            self.last_search_stats["status"] = "complete"
            assert self.best_node is not None
            return self.best_node
        except Exception as exc:
            self._record_error(exc)
            raise

    def export_tree(self) -> dict[str, object]:
        """Export stable tree/search statistics without serializing latent tensors."""
        if self.root is None:
            raise RuntimeError("run search first")
        nodes: list[dict[str, object]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            nodes.append(
                {
                    "path": list(node.path),
                    "depth": int(node.depth),
                    "action_from_parent": node.action_from_parent,
                    "prior": float(node.prior),
                    "visits": int(node.visits),
                    "value_sum": float(node.value_sum),
                    "q": float(node.q),
                    "children": [list(child.path) for child in node.children],
                    "y_shape": list(node.y.shape),
                    "z_codes_shape": list(node.z_codes.shape),
                    "z_codes_dtype": str(node.z_codes.dtype).replace("torch.", ""),
                    "z_scale_shape": list(node.z_scale.shape),
                    "y_norm": float(node.y.detach().float().norm().cpu()),
                }
            )
            stack.extend(reversed(node.children))
        stats = dict(self.last_search_stats)
        stats["node_count"] = len(nodes)
        stats["virtual_loss_outstanding"] = int(stats.get("virtual_loss_applications", 0)) - int(
            stats.get("virtual_loss_cleanups", 0)
        )
        return {"nodes": nodes, "work": stats}

    def root_visit_policy(self, temperature: float = 1.0) -> torch.Tensor:
        if self.root is None or not self.root.children:
            raise RuntimeError("root visit policy requires a positive-budget completed search")
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise ValueError("temperature must be finite and > 0")
        visits = torch.tensor([c.visits for c in self.root.children], dtype=torch.float32)
        if visits.sum() == 0:
            return torch.full_like(visits, 1.0 / len(visits))
        powered = visits.pow(1.0 / temperature)
        return powered / powered.sum()

    def root_value(self) -> float:
        if self.root is None or self.root.visits <= 0:
            raise RuntimeError("root value requires at least one completed search rollout")
        return float(self.root.q)

    @torch.no_grad()
    def prm_targets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return MCTS-bootstrapped q targets; these are not independent truth."""
        if self.root is None:
            raise RuntimeError("run search first")
        nodes, stack = [], [self.root]
        while stack:
            node = stack.pop()
            if node.visits > 0:
                nodes.append(node)
            stack.extend(node.children)
        if not nodes:
            raise RuntimeError("PRM targets require at least one completed rollout")
        z = torch.cat([n.latent() for n in nodes], dim=0)
        values = torch.tensor([n.q for n in nodes], dtype=torch.float32)
        visits = torch.tensor([float(n.visits) for n in nodes], dtype=torch.float32)
        return z, values, visits

    @torch.no_grad()
    def decode(self, node: _LatentNode) -> torch.Tensor:
        if not isinstance(node, _LatentNode):
            raise TypeError("decode expects a native MCTS node")
        if self.last_search_stats:
            self.last_search_stats["decode_calls"] = int(
                self.last_search_stats.get("decode_calls", 0)
            ) + 1
        return self.model.out_head(node.y).argmax(dim=-1)

    @torch.no_grad()
    def search_and_decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.search(x))
