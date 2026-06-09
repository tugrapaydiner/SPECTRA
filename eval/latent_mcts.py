"""Cache-Resident Latent MCTS (BLUEPRINT section 9.6).

Instead of generating many full text candidates (memory-prohibitive on the edge),
SPECTRA searches *inside* the latent reasoning state ``z``. The whole ~1.4 MB core
fits in L3 cache, so AlphaZero-style search runs cache-resident:

  1. Selection: descend the tree by PUCT.
  2. Expansion: step the recursive core ``f_theta`` to new latent states ``z``
     (the first child is the deterministic step; the rest add latent noise to
     branch).
  3. Evaluation: score the node with a value function -- the neural energy
     verifier ``-E_psi(x, .)`` or a symbolic verifier. For grid tasks decoding is
     a cheap argmax, so we evaluate the decoded answer while the *search* stays in
     latent space.
  4. Backpropagation: update node value/visit statistics.

Search is per-example (B=1), matching the edge deployment regime (section 26.2).
``search`` returns the best answer (token grid) found, which is always at least as
good as greedy decoding because the deterministic trajectory is in the tree.
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
    """AlphaZero-style search over the recursive core's latent states."""

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
        """Run one supervision step (T cycles) of the core, optionally branching."""
        if noisy:
            z = z + self.noise_std * torch.randn_like(z)
        for _ in range(self.model.T):
            y, z = self.model.recursive_cycle(x_emb, y, z)
        return y, z

    def _expand(self, node: _Node, x_emb: torch.Tensor) -> None:
        """Add children: first deterministic, the rest latent-noised branches."""
        for c in range(self.n_children):
            y, z = self._step_latent(x_emb, node.y, node.z, noisy=(c > 0))
            node.children.append(_Node(y, z, prior=1.0 / self.n_children))

    def _puct(self, parent: _Node, child: _Node) -> float:
        explore = self.c_puct * child.prior * math.sqrt(parent.visits + 1) / (1 + child.visits)
        return child.q + explore

    def _evaluate(self, x: torch.Tensor, node: _Node) -> tuple[float, torch.Tensor]:
        """Decode the node's answer and score it (cheap for grids)."""
        logits = self.model.out_head(node.y)
        answer = logits.argmax(dim=-1)
        return self.value_fn(x, answer), answer

    @torch.no_grad()
    def search(self, x: torch.Tensor) -> torch.Tensor:
        """Search from input ``x`` ``[1, L]`` and return the best answer ``[1, L]``."""
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
    """Per-token symmetric INT8 quantize: returns ``(codes int8, scale float)``."""
    scale = z.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    codes = torch.clamp(torch.round(z / scale), -128, 127).to(torch.int8)
    return codes, scale


def _dequantize_int8(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reconstruct the float-on-grid latent from INT8 codes + scale."""
    return codes.to(scale.dtype) * scale


class _LatentNode:
    """MCTS node with an explicit INT8/FP casting boundary.

    The *environment dynamics* (the reasoning latent ``z``) are stored as literal
    INT8 codes + an FP32 per-token scale -- 1 byte/element, the cache-resident
    representation. The *search statistics* (``visits`` N, ``value_sum`` W, ``q``,
    ``prior`` P) are kept in FP32/int so PUCT accumulation never suffers INT8
    quantization noise. ``y`` (the answer accumulator, decoded exactly once) stays
    FP for decode precision.
    """

    __slots__ = ("y", "z_codes", "z_scale", "prior", "children", "visits", "value_sum")

    def __init__(self, y, z_codes, z_scale, prior=1.0):
        self.y = y                       # FP answer accumulator (decoded once)
        self.z_codes = z_codes           # INT8 latent codes (env state)
        self.z_scale = z_scale           # FP32 per-token dequant scale
        self.prior = float(prior)        # FP search prior
        self.children: list[_LatentNode] = []
        self.visits = 0                  # int visit count N
        self.value_sum = 0.0             # FP64 value accumulator W

    @property
    def q(self) -> float:                # FP mean action value
        return self.value_sum / self.visits if self.visits else 0.0

    def latent(self) -> torch.Tensor:
        return _dequantize_int8(self.z_codes, self.z_scale)


class LatentNativeMCTS:
    """Cache-Resident Latent MCTS done right (BLUEPRINT section 9.6).

    Fixes the flaws of :class:`LatentMCTS`:
      * **No decoding during search** -- nodes are scored by the latent energy
        verifier ``E_psi(x, z)`` directly; ``out_head`` is called exactly *once*,
        at the very end, to decode the chosen latent.
      * **Principled actions** -- expansion uses a learned discrete
        :class:`LatentActionCodebook` (residual directions), not ``torch.randn``.
      * **Literal INT8 latent / FP search stats** -- the latent is stored as INT8
        codes (the cache-resident state); PUCT's Q/N/prior live in FP32, so search
        statistics never accumulate inside the quantized space.

    Search returns the best latent node; call :meth:`decode` for the single decode.
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
    ):
        self.model = model
        self.verifier = energy_verifier
        self.codebook = action_codebook
        self.height = height
        self.width = width
        self.n_rollouts = n_rollouts
        self.c_puct = c_puct
        # Pessimism weight on epistemic uncertainty (requires an ensemble verifier).
        self.uncertainty_beta = uncertainty_beta
        # Optional latent VQ: bounds compounding recursion error to the covering
        # radius (constant), so deep rollouts do not shred the latent topology.
        self.latent_vq = latent_vq
        self.root: _LatentNode | None = None  # last search root (for visit-count distill)

    def _step(self, x_emb, node, action):
        """Dequantize -> apply action -> recurse (FP) -> re-quantize to INT8."""
        z = node.latent()  # INT8 codes -> FP-on-grid for the FP recursion math
        z = self.codebook.apply_action(z, action)
        y = node.y
        for _ in range(self.model.T):
            y, z = self.model.recursive_cycle(x_emb, y, z)
        if self.latent_vq is not None:
            z = self.latent_vq.snap(z)  # project onto codebook -> bounded error
        z_codes, z_scale = _quantize_int8(z)  # store latent as literal INT8
        return y, z_codes, z_scale

    def _expand(self, node, x_emb):
        priors = self.codebook.priors()
        for a in range(self.codebook.n_actions):
            y, z_codes, z_scale = self._step(x_emb, node, a)
            node.children.append(_LatentNode(y, z_codes, z_scale, prior=float(priors[a])))

    def _puct(self, parent, child):
        explore = self.c_puct * child.prior * math.sqrt(parent.visits + 1) / (1 + child.visits)
        return child.q + explore  # all-FP search statistics

    def _value(self, x, node):
        """Latent value via the energy verifier -- NO decoding of ``z``.

        With an ensemble verifier and ``uncertainty_beta > 0``, returns the
        pessimistic lower-confidence bound ``mean_value - beta * epistemic_std``,
        so OOD latents (high member disagreement) are penalised rather than chased.
        """
        z = node.latent()
        if self.uncertainty_beta > 0.0 and hasattr(self.verifier, "value_with_uncertainty"):
            value, std = self.verifier.value_with_uncertainty(x, z, self.width)
            return float((value - self.uncertainty_beta * std).detach().mean())
        return float(self.verifier.value(x, z, self.width).detach().mean())

    @torch.no_grad()
    def search(self, x: torch.Tensor) -> _LatentNode:
        """Search in INT8 latent space; return the best latent node (undecoded)."""
        x_emb = self.model.token_embed(x) + self.model.encode_positions(x, self.height, self.width)
        z0_codes, z0_scale = _quantize_int8(torch.zeros_like(x_emb))
        root = _LatentNode(torch.zeros_like(x_emb), z0_codes, z0_scale)
        self._expand(root, x_emb)

        best_value, best_node = -math.inf, root
        for _ in range(self.n_rollouts):
            path, node = [root], root
            while node.children:
                node = max(node.children, key=lambda c: self._puct(node, c))
                path.append(node)
            self._expand(node, x_emb)
            value = self._value(x, node)
            if value > best_value:
                best_value, best_node = value, node
            for nd in path:
                nd.visits += 1
                nd.value_sum += value
        self.root = root
        return best_node

    def root_visit_policy(self, temperature: float = 1.0) -> torch.Tensor:
        """AlphaZero policy target ``pi(a) ∝ N(root, a)^{1/tau}`` over the codebook.

        The FULL search signal -- the visit distribution over actions, not just the
        final answer -- that System 1 distils, so it cannot collapse to behavioral
        cloning.
        """
        assert self.root is not None and self.root.children, "run a search first"
        visits = torch.tensor([c.visits for c in self.root.children], dtype=torch.float32)
        if visits.sum() == 0:
            return torch.full_like(visits, 1.0 / len(visits))
        powered = visits.pow(1.0 / temperature)
        return powered / powered.sum()

    def root_value(self) -> float:
        """MCTS root value (visit-weighted mean child Q) -- the value head target."""
        assert self.root is not None and self.root.children, "run a search first"
        total_v = sum(c.value_sum for c in self.root.children)
        total_n = sum(c.visits for c in self.root.children)
        return total_v / max(1, total_n)

    @torch.no_grad()
    def prm_targets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-step process-reward labels from the search tree (NO human labels).

        Every visited node's *backed-up* value ``q = W/N`` is a self-supervised
        process reward for its latent state: the search's own credit assignment to
        an intermediate reasoning step. The backup (a node's value is the mean of
        its subtree's leaf evaluations) is what turns a raw outcome score into a
        per-step *process* signal. Training the energy verifier to regress these is
        MCTS-bootstrapped Latent PRM training (``train/distill.latent_prm_loss``).

        Returns ``(Z [M, L, D], values [M], visits [M])`` over all visited nodes.
        """
        assert self.root is not None, "run a search first"
        nodes, stack = [], [self.root]
        while stack:
            node = stack.pop()
            if node.visits > 0:
                nodes.append(node)
            stack.extend(node.children)
        z = torch.cat([n.latent() for n in nodes], dim=0)
        values = torch.tensor([n.q for n in nodes], dtype=torch.float32)
        visits = torch.tensor([float(n.visits) for n in nodes], dtype=torch.float32)
        return z, values, visits

    def _value_batch(self, x_rep, z_batch):
        """Batched latent value (leaf parallelism). Returns ``[batch]`` tensor."""
        if self.uncertainty_beta > 0.0 and hasattr(self.verifier, "value_with_uncertainty"):
            value, std = self.verifier.value_with_uncertainty(x_rep, z_batch, self.width)
            return value - self.uncertainty_beta * std
        return self.verifier.value(x_rep, z_batch, self.width)

    @torch.no_grad()
    def search_batched(
        self, x: torch.Tensor, leaf_batch: int = 8, virtual_loss: float = 1.0
    ) -> _LatentNode:
        """Virtual-loss leaf-parallel search (defeats the B=1 MCTS latency death).

        Serial MCTS pays one verifier forward per rollout. Here each iteration
        selects ``leaf_batch`` leaves under **virtual loss** -- a transient penalty
        ``visits += 1, W -= vloss`` applied along each selected path so subsequent
        in-batch selections steer to *different* leaves -- then evaluates all of
        them in ONE batched verifier call (a B=8/16 SIMD payload), and backs up the
        real values while undoing the virtual loss (``W += vloss + value``).

        Same tree semantics as :meth:`search`, far fewer (batched) verifier calls.
        """
        x_emb = self.model.token_embed(x) + self.model.encode_positions(x, self.height, self.width)
        z0_codes, z0_scale = _quantize_int8(torch.zeros_like(x_emb))
        root = _LatentNode(torch.zeros_like(x_emb), z0_codes, z0_scale)
        self._expand(root, x_emb)

        best_value, best_node = -math.inf, root
        iterations = max(1, self.n_rollouts // leaf_batch)
        for _ in range(iterations):
            batch_paths: list[tuple[list[_LatentNode], _LatentNode]] = []
            for _ in range(leaf_batch):
                path, node = [root], root
                while node.children:
                    node = max(node.children, key=lambda c: self._puct(node, c))
                    path.append(node)
                for nd in path:  # apply virtual loss so the next pick diverges
                    nd.visits += 1
                    nd.value_sum -= virtual_loss
                self._expand(node, x_emb)
                batch_paths.append((path, node))

            leaves = [leaf for _, leaf in batch_paths]
            z_batch = torch.cat([leaf.latent() for leaf in leaves], dim=0)  # [batch, L, D]
            x_rep = x.expand(len(leaves), -1)
            values = self._value_batch(x_rep, z_batch).tolist()

            for (path, leaf), value in zip(batch_paths, values):
                if value > best_value:
                    best_value, best_node = value, leaf
                for nd in path:  # undo virtual loss, back up the real value
                    nd.value_sum += virtual_loss + value
        return best_node

    @torch.no_grad()
    def decode(self, node: _LatentNode) -> torch.Tensor:
        """The single decode: map the chosen latent's answer state to tokens."""
        return self.model.out_head(node.y).argmax(dim=-1)

    @torch.no_grad()
    def search_and_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: search the latent space, then decode the winner once."""
        return self.decode(self.search(x))
