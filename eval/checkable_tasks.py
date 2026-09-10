"""Task-bound decode/check adapters for CPU experiments.

A task-specific checker is a dependency, not a learned guarantee. Maze native
construction includes BFS preprocessing; benchmark callers must charge it. The
learned recurrence remains PyTorch FP32. No reference answer enters these APIs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import torch

from data import maze, sudoku
from deploy.m10_native import load_extension
from model.trm import TRM
from model.verifier import maze_correct, maze_score, sudoku_correct, sudoku_score


@dataclass(frozen=True)
class TaskSpec:
    task: str
    height: int
    width: int
    dim: int
    max_grid_size: int
    box: int | None = None

    def __post_init__(self):
        for value in (self.height, self.width, self.dim, self.max_grid_size):
            if type(value) is not int or value < 1:
                raise ValueError("task dimensions must be positive integers")
        if max(self.height, self.width) > self.max_grid_size:
            raise ValueError("task exceeds the positional-embedding capacity")
        if self.task == "sudoku":
            if type(self.box) is not int or self.box < 1 or self.height != self.box**2 or self.width != self.height:
                raise ValueError("Sudoku needs consistent square-box geometry")
        elif self.task != "maze" or self.box is not None:
            raise ValueError("only explicit Sudoku or maze contracts are supported")

    @property
    def num_tokens(self) -> int:
        return self.width + 1 if self.task == "sudoku" else 5

    @property
    def decode_id(self) -> str:
        return "argmax_then_restore_givens_v1" if self.task == "sudoku" else "argmax_then_open_path_restore_v1"

    @property
    def transition_id(self) -> str:
        # Preserve the existing, narrowly defined M16 Sudoku binding.
        if self == SUDOKU_SHIFT:
            return "fp64_cycle_action_v1"
        return f"m17_{self.task}_{self.dim}_cycle_action_seed16111_norm05_v1"

    @property
    def state_schema(self) -> str:
        if self == SUDOKU_SHIFT:
            return "search_state_xyz_v1"
        return f"m17_{self.task}_{self.height}x{self.width}_xyz_{self.decode_id}"

    def metadata(self) -> dict:
        return {**asdict(self), "decode_id": self.decode_id,
                "transition_id": self.transition_id, "state_schema": self.state_schema,
                "num_tokens": self.num_tokens, "reference_target_used": False}

    def validate_input(self, x: torch.Tensor, *, single: bool = False) -> None:
        if not isinstance(x, torch.Tensor) or x.device.type != "cpu" or x.dtype != torch.int64:
            raise ValueError("task input must be a CPU int64 tensor")
        if x.ndim != 2 or x.shape[1] != self.height*self.width or (single and x.shape[0] != 1):
            raise ValueError("input shape does not match the task contract")

    def restore(self, x: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        if candidate.shape != x.shape or candidate.device != x.device or candidate.dtype != torch.int64:
            raise ValueError("candidate must have the input's shape, CPU device and int64 dtype")
        if self.task == "sudoku":
            return torch.where(x != 0, x, candidate)
        # Do not compare two selected logits: argmax over the original five
        # classes is part of this frozen decoder, including first-maximum ties.
        return torch.where((x == maze.OPEN) & (candidate == maze.PATH), maze.PATH, x)

    def correct(self, x: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        if self.task == "sudoku":
            return sudoku_correct(x, answer, self.box)
        return maze_correct(x, answer, self.height, self.width, require_optimal=True)

    def quality(self, x: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
        if self.task == "sudoku":
            return sudoku_score(x, answer, self.box)
        return maze_score(x, answer, self.height, self.width)

    def native_problem(self, x: torch.Tensor):
        self.validate_input(x, single=True)
        extension = load_extension()
        if self.task == "sudoku":
            return extension.SudokuProblem(x.contiguous(), self.box)
        return extension.MazeProblem(x.contiguous(), self.height, self.width, True)

    def independent_correct(self, x: torch.Tensor, answer: torch.Tensor | None) -> bool:
        """NumPy checker for assertions outside the measured solve window."""
        self.validate_input(x, single=True)
        if answer is None:
            return False
        if answer.shape != x.shape or answer.device.type != "cpu" or answer.dtype != torch.int64:
            return False
        inp = x.detach().numpy().reshape(self.height, self.width)
        cand = answer.detach().numpy().reshape(self.height, self.width)
        if self.task == "sudoku":
            return sudoku.is_solved(cand, self.box) and sudoku.respects_clues(inp, cand, self.box)
        return maze.candidate_success(inp, cand, self.height, self.width, require_optimal=True)

    def symbolic_answer(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x, single=True)
        inp = x.numpy().reshape(self.height, self.width)
        if self.task == "sudoku":
            answer = sudoku.solve(inp, self.box)
            return x.clone() if answer is None else torch.from_numpy(answer.reshape(1, -1))
        starts, goals = np.argwhere(inp == maze.START), np.argwhere(inp == maze.GOAL)
        if len(starts) != 1 or len(goals) != 1:
            return x.clone()
        start, goal = tuple(starts[0]), tuple(goals[0])
        path = maze.shortest_path(inp, start, goal)
        answer = inp.copy() if path is None else maze.make_target(inp, start, goal, path)
        return torch.from_numpy(answer.reshape(1, -1))


SUDOKU_SHIFT = TaskSpec("sudoku", 4, 4, 64, 8, 2)
MAZE11 = TaskSpec("maze", 11, 11, 48, 16)


def action_directions(dim: int) -> torch.Tensor:
    if type(dim) is not int or dim < 1:
        raise ValueError("dimension must be a positive integer")
    generator = torch.Generator(device="cpu").manual_seed(16111)
    directions = torch.randn(3, dim, generator=generator, device="cpu")
    return torch.cat([torch.zeros(1, dim, device="cpu"), 0.5*directions/directions.norm(dim=1, keepdim=True)])


def require_core(core: TRM, spec: TaskSpec) -> None:
    if not isinstance(core, TRM) or core.dim != spec.dim or core.n != 1 or core.T != 1 or core.ternary or core.act8:
        raise ValueError("task runtime requires its declared FP n=T=1 TRM")
    if core.num_tokens != spec.num_tokens or core.seq_len != spec.height*spec.width or core.max_grid_size != spec.max_grid_size:
        raise ValueError("core vocabulary/geometry does not match the task")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in core.parameters()):
        raise ValueError("the experiment accepts only CPU FP32 core parameters")
    if core.training or any(p.requires_grad for p in core.parameters()):
        raise ValueError("the inference core must be frozen and in eval mode")


@torch.inference_mode()
def semantic_exit(core: TRM, x: torch.Tensor, spec: TaskSpec, cycles: int = 4, *, native: bool = True):
    """Complete B=1 solve; cycles beyond N_sup are explicit depth extrapolation."""
    require_core(core, spec)
    spec.validate_input(x, single=True)
    if type(cycles) is not int or not 1 <= cycles <= 256:
        raise ValueError("cycles must be an integer in [1,256]")
    problem = spec.native_problem(x) if native else None
    xemb = core.token_embed(x) + core.encode_positions(x, spec.height, spec.width)
    y, z = torch.zeros_like(xemb), torch.zeros_like(xemb)
    for step in range(1, cycles+1):
        y, z = core.recursive_cycle(xemb, y, z)
        logits = core.out_head(y).contiguous()
        if native:
            answer, valid = problem.decode(logits)
        else:
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("logits must be finite")
            answer = spec.restore(x, logits.argmax(-1))
            valid = bool(spec.correct(x, answer).item())
        if valid:
            break
    return answer, {"valid": bool(valid), "transitions": step, "decodes": step, "checks": step,
                    "block_applications": step*2*len(core.blocks), "value_calls": 0,
                    "checker_constructions": int(native), "input_embeddings": 1,
                    "requested_cycles": cycles, "trained_cycles": core.N_sup,
                    "depth_extrapolation": cycles > core.N_sup, "target_used": False,
                    "stop_reason": "valid_answer" if valid else "transition_budget"}
