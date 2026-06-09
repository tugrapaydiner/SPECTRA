"""Open-ended self-play flywheel (BLUEPRINT section 10).

The old ``data.generate.self_play_flywheel`` was a static ``for`` loop at fixed
difficulty. This is the real thing: a curriculum that samples difficulty from the
**frontier of competence** -- the zone where System 1 *fails* but System 2 *solves*
-- and an asynchronous-style producer/consumer loop where System 2 mines verified
trajectories on hard tasks and System 1 continuously distills them into reflexes.

Open-endedness comes from the moving frontier: as System 1 masters a difficulty,
its frontier weight collapses and the curriculum automatically shifts to harder
tasks, with no static dataset and no fixed difficulty (cf. learning-progress /
ZPD curricula, POET-style auto-curricula).

Components
---------
* :class:`CurriculumState` -- per-difficulty competence tracker + frontier sampler.
* :class:`FlywheelBuffer` -- replay of verified ``(input, answer, z*)`` trajectories.
* :class:`SelfPlayFlywheel` -- the ``produce`` (System 2) / ``consume`` (System 1)
  loop; ``produce`` and ``consume`` are decoupled through the buffer, so a thread
  can run the night-shift producer while the day-shift consumer trains.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from data.augment import augment_sudoku_pair
from model.verifier import sudoku_correct


# --------------------------------------------------------------------------- #
# Curriculum state (the moving frontier of competence)
# --------------------------------------------------------------------------- #
class CurriculumState:
    """Tracks System 1 / System 2 success per difficulty and samples the frontier.

    Frontier weight for difficulty ``d``:
        ``w(d) ∝ (1 - s1(d)) * s2(d) + floor``
    -- high where System 1 fails but System 2 succeeds (learnable & not yet
    learned), low where System 1 already wins (mastered) or System 2 also fails
    (currently unsolvable). The ``floor`` keeps unexplored difficulties sampled.
    """

    def __init__(self, difficulties: list[float], ema: float = 0.85, floor: float = 0.05):
        self.difficulties = list(difficulties)
        self.n = len(difficulties)
        self.ema = ema
        self.floor = floor
        self.s1 = np.zeros(self.n)  # EMA System 1 success rate per bucket
        self.s2 = np.zeros(self.n)  # EMA System 2 success rate per bucket
        self.seen = np.zeros(self.n, dtype=np.int64)

    def update(self, idx: int, s1_solved: float, s2_solved: float) -> None:
        a = self.ema if self.seen[idx] > 0 else 0.0  # first sample seeds directly
        self.s1[idx] = a * self.s1[idx] + (1 - a) * float(s1_solved)
        self.s2[idx] = a * self.s2[idx] + (1 - a) * float(s2_solved)
        self.seen[idx] += 1

    def frontier_weights(self) -> np.ndarray:
        # Unexplored buckets get pure-floor exploration; explored use the ZPD score.
        s2_eff = np.where(self.seen > 0, self.s2, 1.0)  # assume solvable until shown otherwise
        w = (1.0 - self.s1) * s2_eff + self.floor
        return w / w.sum()

    def sample_index(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self.n, p=self.frontier_weights()))

    def mean_sampled_difficulty(self) -> float:
        return float(np.dot(self.frontier_weights(), self.difficulties))


# --------------------------------------------------------------------------- #
# Replay buffer of verified trajectories
# --------------------------------------------------------------------------- #
class FlywheelBuffer:
    """Anti-forgetting replay of verified ``(input, answer, difficulty)`` trajectories.

    Two mechanisms guard the base policy against catastrophic forgetting when the
    curriculum drifts to hard tasks:

      * **Reservoir sampling** on insertion -- once full, a new item replaces a
        uniformly-random existing slot with probability ``capacity/seen``. This
        keeps a *uniform sample of the entire history*, so the easy tasks System 1
        mastered early are never simply evicted (unlike a ring buffer's FIFO trim).
      * **Difficulty-stratified sampling** on read -- a training batch is drawn
        evenly across difficulty bins, guaranteeing every System 1 update still
        sees easy tasks alongside the new hard ones (mixing old + new).
    """

    def __init__(self, capacity: int = 20000, n_bins: int = 4, seed: int = 0):
        self.capacity = capacity
        self.n_bins = n_bins
        self.inputs: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.difficulties: list[float] = []
        self.seen = 0
        self._rng = np.random.default_rng(seed)

    def add(self, inputs: np.ndarray, targets: np.ndarray, difficulty: float) -> None:
        for x, y in zip(inputs, targets):
            self.seen += 1
            if len(self.inputs) < self.capacity:
                self.inputs.append(x)
                self.targets.append(y)
                self.difficulties.append(difficulty)
            else:
                # Reservoir replacement -- preserves a uniform sample of all history.
                j = int(self._rng.integers(0, self.seen))
                if j < self.capacity:
                    self.inputs[j] = x
                    self.targets[j] = y
                    self.difficulties[j] = difficulty

    def __len__(self) -> int:
        return len(self.inputs)

    def _bin_indices(self) -> list[np.ndarray]:
        diffs = np.asarray(self.difficulties)
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        bins = np.digitize(diffs, edges)
        return [np.where(bins == b)[0] for b in range(self.n_bins)]

    def sample(self, batch: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Difficulty-stratified batch: mixes easy + hard to prevent forgetting."""
        n = len(self.inputs)
        if n == 0:
            raise ValueError("cannot sample from an empty buffer")
        non_empty = [ix for ix in self._bin_indices() if len(ix) > 0]
        per_bin = max(1, batch // max(1, len(non_empty)))

        picks: list[int] = []
        for ix in non_empty:
            picks.extend(rng.choice(ix, size=min(per_bin, len(ix)), replace=False).tolist())
        # Top up uniformly to reach the requested batch size.
        if len(picks) < batch:
            picks.extend(rng.integers(0, n, size=batch - len(picks)).tolist())
        picks = picks[:batch]

        return (
            np.stack([self.inputs[i] for i in picks]),
            np.stack([self.targets[i] for i in picks]),
        )


# --------------------------------------------------------------------------- #
# Adversarial validator -- defeats the Clever Hans effect
# --------------------------------------------------------------------------- #
class AdversarialValidator:
    """Tests whether the student *reasons* or exploits generator artifacts (GT #4).

    A genuine Sudoku solver is invariant to the Sudoku symmetry group (digit
    relabelling, band/stack permutations, transpose): these change every surface
    statistic -- digit frequencies, positions, sequence patterns -- while leaving
    the reasoning identical. So if the student solves a board but FAILS its
    symmetry image, its success was a Clever Hans shortcut, not reasoning.

    ``probe`` measures invariance (the consistency the student should have) and
    returns the transformed boards it failed as *counterexamples* to train on,
    actively driving the loss toward generalised reasoning.
    """

    def __init__(self, box: int, device: str | torch.device = "cpu"):
        self.box = box
        self.side = box * box
        self.device = device

    def _symmetry_images(self, puzzles_np, solutions_np, rng):
        p_aug = np.empty_like(puzzles_np)
        s_aug = np.empty_like(solutions_np)
        for i in range(len(puzzles_np)):
            pg = puzzles_np[i].reshape(self.side, self.side)
            sg = solutions_np[i].reshape(self.side, self.side)
            pa, sa = augment_sudoku_pair(pg, sg, self.box, rng)
            p_aug[i] = pa.reshape(-1)
            s_aug[i] = sa.reshape(-1)
        return p_aug, s_aug

    @torch.no_grad()
    def probe(
        self,
        puzzles: torch.Tensor,
        solutions: torch.Tensor,
        predict_fn: Callable[[torch.Tensor], torch.Tensor],
        rng: np.random.Generator,
    ) -> dict:
        """Measure symmetry-invariance and mine failed symmetry images."""
        base_ok = sudoku_correct(puzzles, predict_fn(puzzles), self.box)  # [N]

        p_aug_np, s_aug_np = self._symmetry_images(
            puzzles.cpu().numpy(), solutions.cpu().numpy(), rng
        )
        p_aug = torch.from_numpy(p_aug_np).to(self.device)
        s_aug = torch.from_numpy(s_aug_np).to(self.device)
        aug_ok = sudoku_correct(p_aug, predict_fn(p_aug), self.box)

        solved = base_ok
        n_solved = int(solved.sum())
        # Invariance: of the boards it solved, what fraction survive the symmetry?
        invariance = float(aug_ok[solved].float().mean()) if n_solved else 1.0
        brittle = solved & ~aug_ok  # right originally, wrong on the symmetry image
        return {
            "invariance": invariance,
            "n_solved": n_solved,
            "n_counterexamples": int(brittle.sum()),
            "counter_puzzles": p_aug[brittle],
            "counter_solutions": s_aug[brittle],
        }


# --------------------------------------------------------------------------- #
# The flywheel
# --------------------------------------------------------------------------- #
class SelfPlayFlywheel:
    """Night-shift producer (System 2) + day-shift consumer (System 1).

    Args:
        generator: A task generator exposing ``generate(difficulty, rng)`` ->
            object with ``.puzzle`` / ``.solution`` flat arrays and ``.box``.
        system1: The feed-forward student (``System1Student``).
        system2_solve: ``(puzzles [B,L] long) -> answers [B,L] long`` -- the slow
            solver (recursive TRM / Latent MCTS / symbolic oracle).
        verify: ``(puzzles, answers, box) -> bool [B]`` correctness check.
        curriculum: A :class:`CurriculumState`.
        buffer: A :class:`FlywheelBuffer`.
        height / width / box: Grid geometry.
        device: Torch device.
    """

    def __init__(
        self, generator, system1, system2_solve, verify, curriculum, buffer,
        height, width, box, device="cpu", adversarial=None,
    ):
        self.adversarial = adversarial if adversarial is not None else AdversarialValidator(box, device)
        self.generator = generator
        self.system1 = system1
        self.system2_solve = system2_solve
        self.verify = verify
        self.curriculum = curriculum
        self.buffer = buffer
        self.height = height
        self.width = width
        self.box = box
        self.device = device

    @torch.no_grad()
    def _system1_solves(self, puzzles: torch.Tensor) -> torch.Tensor:
        """Boolean mask of which puzzles System 1 already solves (reflex)."""
        self.system1.eval()
        ans, _ = self.system1.predict(puzzles, self.height, self.width)
        return self.verify(puzzles, ans, self.box)

    def produce(self, rng: np.random.Generator, n_tasks: int = 32) -> dict:
        """Sample a frontier difficulty, mine verified System-2 trajectories.

        System 2 is invoked only on the tasks System 1 fails (the productive
        frontier); verified solutions are pushed to the buffer.
        """
        idx = self.curriculum.sample_index(rng)
        difficulty = self.curriculum.difficulties[idx]
        tasks = [self.generator.generate(difficulty, rng) for _ in range(n_tasks)]
        puzzles = torch.from_numpy(np.stack([t.puzzle for t in tasks])).to(self.device)

        s1_mask = self._system1_solves(puzzles)
        s1_rate = float(s1_mask.float().mean())

        # Run System 2 only where System 1 failed.
        hard = ~s1_mask
        s2_rate = 0.0
        admitted = 0
        if hard.any():
            hard_puzzles = puzzles[hard]
            s2_answers = self.system2_solve(hard_puzzles)
            s2_ok = self.verify(hard_puzzles, s2_answers, self.box)
            s2_rate = float(s2_ok.float().mean())
            if s2_ok.any():
                self.buffer.add(
                    hard_puzzles[s2_ok].cpu().numpy(),
                    s2_answers[s2_ok].cpu().numpy(),
                    difficulty=difficulty,
                )
                admitted = int(s2_ok.sum())

        self.curriculum.update(idx, s1_rate, s2_rate)
        return {
            "difficulty": difficulty, "s1_rate": s1_rate, "s2_rate": s2_rate,
            "admitted": admitted, "buffer": len(self.buffer),
        }

    def consume(self, optimizer, rng: np.random.Generator, batch: int = 64) -> float:
        """One System 1 distillation step on buffered verified trajectories."""
        if len(self.buffer) == 0:
            return float("nan")
        xb, yb = self.buffer.sample(batch, rng)
        x = torch.from_numpy(xb).to(self.device)
        y = torch.from_numpy(yb).to(self.device)
        self.system1.train()
        logits, conf_logit = self.system1(x, self.height, self.width)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        correct = (logits.argmax(-1) == y).all(dim=1).float()
        conf_bce = F.binary_cross_entropy_with_logits(conf_logit, correct)
        loss = ce + 0.5 * conf_bce
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def adversarial_mine(self, rng: np.random.Generator, n: int = 32) -> dict:
        """Probe buffered boards for Clever Hans shortcuts; mine counterexamples.

        Samples verified boards, checks the student's symmetry-invariance, and adds
        any failed symmetry images back to the buffer so the next System 1 update is
        forced to learn the invariant reasoning instead of the artifact.
        """
        if len(self.buffer) == 0:
            return {"invariance": 1.0, "n_counterexamples": 0}
        xb, yb = self.buffer.sample(n, rng)
        x = torch.from_numpy(xb).to(self.device)
        y = torch.from_numpy(yb).to(self.device)

        self.system1.eval()

        def predict_fn(p):
            return self.system1.predict(p, self.height, self.width)[0]

        rep = self.adversarial.probe(x, y, predict_fn, rng)
        if rep["n_counterexamples"] > 0:
            self.buffer.add(
                rep["counter_puzzles"].cpu().numpy(),
                rep["counter_solutions"].cpu().numpy(),
                difficulty=1.0,  # symmetry-broken boards are the hardest signal
            )
        return {"invariance": rep["invariance"], "n_counterexamples": rep["n_counterexamples"]}

    def step(self, optimizer, rng: np.random.Generator, n_tasks: int = 32, batch: int = 64) -> dict:
        """One full flywheel turn: produce hard trajectories, then distill them."""
        stats = self.produce(rng, n_tasks)
        stats["s1_loss"] = self.consume(optimizer, rng, batch)
        stats.update(self.adversarial_mine(rng, n_tasks))
        stats["frontier_difficulty"] = self.curriculum.mean_sampled_difficulty()
        return stats
