"""Dataset builders with explicit task/data contracts.

``GridDataset`` remains backward compatible with the training code (getitem still
returns only input/target tensors) while retaining contract metadata, stable IDs,
and content masks for evaluation/manifests.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from data import arc as arc_task
from data import augment as aug
from data import babyai as babyai_task
from data import maze as mz
from data import smarthome as sh_task
from data import sudoku as sk


GENERATOR_VERSION = "spectra-m03-data-v1"


class GridDataset(Dataset):
    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        height: int,
        width: int,
        *,
        task: str | None = None,
        num_tokens: int | None = None,
        pad_token: int | None = None,
        input_mask: np.ndarray | None = None,
        target_mask: np.ndarray | None = None,
        ids: list[str] | None = None,
        group_ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ):
        x = np.asarray(inputs)
        y = np.asarray(targets)
        if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
            raise ValueError(f"inputs/targets must have identical [M,L] shape, got {x.shape}, {y.shape}")
        if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
            raise ValueError("height/width must be positive integers")
        if x.shape[1] != height * width:
            raise ValueError(
                f"flattened length {x.shape[1]} disagrees with height*width={height*width}"
            )
        self.inputs = np.ascontiguousarray(x, dtype=np.int64)
        self.targets = np.ascontiguousarray(y, dtype=np.int64)
        self.height = height
        self.width = width
        self.task = task
        self.num_tokens = num_tokens
        self.pad_token = pad_token

        if num_tokens is not None:
            if not isinstance(num_tokens, int) or num_tokens <= 0:
                raise ValueError("num_tokens must be a positive integer")
            for name, arr in (("inputs", self.inputs), ("targets", self.targets)):
                if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= num_tokens):
                    raise ValueError(
                        f"{name} token range [{int(arr.min())},{int(arr.max())}] exceeds [0,{num_tokens})"
                    )
        if pad_token is not None and num_tokens is not None and not 0 <= pad_token < num_tokens:
            raise ValueError("pad_token must lie inside the declared vocabulary")

        shape = self.inputs.shape
        self.input_mask = np.ones(shape, dtype=bool) if input_mask is None else np.asarray(input_mask, dtype=bool)
        self.target_mask = np.ones(shape, dtype=bool) if target_mask is None else np.asarray(target_mask, dtype=bool)
        if self.input_mask.shape != shape or self.target_mask.shape != shape:
            raise ValueError("input_mask/target_mask must match dataset array shape")
        self.input_mask = np.ascontiguousarray(self.input_mask)
        self.target_mask = np.ascontiguousarray(self.target_mask)

        n = len(self.inputs)
        self.ids = ids if ids is not None else [f"row-{i}" for i in range(n)]
        self.group_ids = group_ids if group_ids is not None else list(self.ids)
        self.metadata = metadata if metadata is not None else [{} for _ in range(n)]
        if len(self.ids) != n or len(self.group_ids) != n or len(self.metadata) != n:
            raise ValueError("ids/group_ids/metadata lengths must equal dataset length")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.targets[idx])


def _default_num_tokens(task: str, kwargs: dict[str, Any]) -> int:
    if task == "sudoku":
        return sk.grid_size(int(kwargs.get("box", 3))) + 1
    if task == "maze":
        return 5
    if task == "arc":
        return int(kwargs.get("pad_token", 10)) + 1
    if task == "babyai":
        return babyai_task.NUM_TOKENS
    if task == "smarthome":
        return sh_task.NUM_TOKENS
    raise ValueError(f"Unknown task: {task!r}")


def generate_example(
    task: str, rng: np.random.Generator, **kwargs: Any
) -> tuple[np.ndarray, np.ndarray, int, int, dict[str, Any]]:
    """Generate one *unaugmented* example plus difficulty/provenance metadata."""
    if task == "sudoku":
        box = int(kwargs.get("box", 3))
        side = sk.grid_size(box)
        min_clues = int(kwargs.get("min_clues", kwargs.get("num_clues", 40)))
        max_clues = int(kwargs.get("max_clues", kwargs.get("num_clues", min_clues)))
        if not 0 < min_clues <= max_clues <= side * side:
            raise ValueError("invalid Sudoku clue range")
        requested = int(rng.integers(min_clues, max_clues + 1))
        puzzle, solution = sk.generate_pair(
            box,
            requested,
            rng,
            require_unique=bool(kwargs.get("require_unique", True)),
            solution_method=str(kwargs.get("solution_method", "random_backtracking")),
        )
        actual = int((puzzle != 0).sum())
        return (
            puzzle.reshape(-1), solution.reshape(-1), side, side,
            {
                "difficulty": {"clues": actual, "blanks": side * side - actual},
                "requested_clues": requested,
                "box": box,
                "solution_method": str(kwargs.get("solution_method", "random_backtracking")),
            },
        )

    if task == "maze":
        h = int(kwargs.get("height", 15)); w = int(kwargs.get("width", 15))
        x, y = mz.generate_pair(
            h, w, rng,
            min_path_len=int(kwargs.get("min_path_len", 0)),
            max_path_len=kwargs.get("max_path_len", None),
        )
        path_len = int((y == mz.PATH).sum()) + 2
        wall_fraction = float((x == mz.WALL).mean())
        return x.reshape(-1), y.reshape(-1), h, w, {
            "difficulty": {"path_length": path_len, "wall_fraction": wall_fraction},
            "require_optimal": bool(kwargs.get("require_optimal", True)),
        }

    if task == "arc":
        h = int(kwargs.get("canvas_h", kwargs.get("height", 10)))
        w = int(kwargs.get("canvas_w", kwargs.get("width", 10)))
        pad = int(kwargs.get("pad_token", 10))
        transform = str(kwargs.get("transform", "flip_h"))
        x, y = arc_task.generate_pair(
            transform, rng,
            canvas_h=h, canvas_w=w,
            max_h=int(kwargs.get("max_h", min(6, h))),
            max_w=int(kwargs.get("max_w", min(6, w))),
            n_colors=int(kwargs.get("n_colors", 5)), pad_token=pad,
        )
        grid = x.reshape(h, w)
        loc = np.argwhere(grid != pad)
        source_h = int(loc[:, 0].max() + 1) if len(loc) else 0
        source_w = int(loc[:, 1].max() + 1) if len(loc) else 0
        return x, y, h, w, {
            "difficulty": {"source_height": source_h, "source_width": source_w},
            "transform": transform,
            "scope": "synthetic_local_arc_style_not_official_arc",
        }

    if task == "babyai":
        h = int(kwargs.get("height", 8)); w = int(kwargs.get("width", 8))
        x, y = babyai_task.generate_pair(h, w, rng, float(kwargs.get("wall_prob", 0.2)))
        xin = x.reshape(h, w); yout = y.reshape(h, w)
        agent_in = np.argwhere(np.isin(xin, list(babyai_task._DELTAS)))
        agent_out = np.argwhere(yout == babyai_task.AGENT)
        moved = bool(len(agent_in) == 1 and len(agent_out) == 1 and not np.array_equal(agent_in[0], agent_out[0]))
        return x, y, h, w, {
            "difficulty": {"walls": int((xin == babyai_task.WALL).sum()), "moved": int(moved)},
            "scope": "synthetic_local_babyai_style_not_official_babyai",
        }

    if task == "smarthome":
        x, y = sh_task.generate_pair(rng)
        return x, y, 1, sh_task.SEQ_LEN, {
            "difficulty": {
                "action": int(y[0]),
                "nonpad_sensor_tokens": int((x != sh_task.PAD).sum()),
            }
        }

    raise ValueError(f"Unknown task: {task!r}")


def augment_example(
    task: str,
    x: np.ndarray,
    y: np.ndarray,
    height: int,
    width: int,
    rng: np.random.Generator,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment only after split/group assignment so derived examples cannot leak."""
    if not bool(kwargs.get("augment", False)):
        return np.asarray(x).copy(), np.asarray(y).copy()
    if task == "sudoku":
        xx, yy = aug.augment_sudoku_pair(
            np.asarray(x).reshape(height, width), np.asarray(y).reshape(height, width),
            int(kwargs.get("box", 3)), rng,
        )
        return xx.reshape(-1), yy.reshape(-1)
    if task == "maze":
        if height != width:
            raise ValueError("D4 maze augmentation currently requires a square retained grid")
        xx, yy = aug.augment_maze_pair(
            np.asarray(x).reshape(height, width), np.asarray(y).reshape(height, width), rng,
        )
        return xx.reshape(-1), yy.reshape(-1)
    raise ValueError(f"augmentation is not defined for task {task!r}")


def _content_masks(
    task: str, inputs: np.ndarray, targets: np.ndarray, pad_token: int | None
) -> tuple[np.ndarray, np.ndarray]:
    if task == "arc":
        if pad_token is None:
            raise ValueError("ARC-style dataset requires pad_token")
        return inputs != pad_token, targets != pad_token
    if task == "smarthome":
        return inputs != sh_task.PAD, np.ones(targets.shape, dtype=bool)
    return np.ones(inputs.shape, dtype=bool), np.ones(targets.shape, dtype=bool)


def build_dataset(task: str, n: int, rng: np.random.Generator, **kwargs: Any) -> GridDataset:
    """Build one dataset and enforce declared shape/vocabulary/mask invariants."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    examples = [generate_example(task, rng, **kwargs) for _ in range(n)]
    if n:
        h, w = examples[0][2], examples[0][3]
        if any((e[2], e[3]) != (h, w) for e in examples):
            raise RuntimeError("generator returned inconsistent spatial dimensions")
        xs, ys, meta = [], [], []
        for x0, y0, _, _, m in examples:
            x, y = augment_example(task, x0, y0, h, w, rng, **kwargs)
            xs.append(x); ys.append(y); meta.append(m)
        inputs = np.stack(xs).astype(np.int64)
        targets = np.stack(ys).astype(np.int64)
    else:
        h = int(kwargs.get("height", kwargs.get("canvas_h", 1)))
        w = int(kwargs.get("width", kwargs.get("canvas_w", kwargs.get("seq_len", 1))))
        if task == "sudoku":
            h = w = sk.grid_size(int(kwargs.get("box", 3)))
        if task == "smarthome":
            h, w = 1, sh_task.SEQ_LEN
        inputs = np.empty((0, h * w), dtype=np.int64)
        targets = np.empty_like(inputs)
        meta = []

    expected_seq_len = int(kwargs.get("seq_len", h * w))
    if expected_seq_len != h * w:
        raise ValueError(f"declared seq_len={expected_seq_len} disagrees with generated {h*w}")
    num_tokens = int(kwargs.get("num_tokens", _default_num_tokens(task, kwargs)))
    pad_token = kwargs.get("pad_token", None)
    pad_token = None if pad_token is None else int(pad_token)
    input_mask, target_mask = _content_masks(task, inputs, targets, pad_token)
    return GridDataset(
        inputs, targets, h, w,
        task=task, num_tokens=num_tokens, pad_token=pad_token,
        input_mask=input_mask, target_mask=target_mask, metadata=meta,
    )


def build_sudoku_arrays(
    box: int,
    n: int,
    num_clues: int,
    rng: np.random.Generator,
    require_unique: bool = True,
    augment: bool = False,
    *,
    min_clues: int | None = None,
    max_clues: int | None = None,
    solution_method: str = "random_backtracking",
) -> tuple[np.ndarray, np.ndarray, int, int]:
    ds = build_dataset(
        "sudoku", n, rng, box=box,
        min_clues=num_clues if min_clues is None else min_clues,
        max_clues=num_clues if max_clues is None else max_clues,
        require_unique=require_unique, augment=augment,
        solution_method=solution_method,
        num_tokens=sk.grid_size(box) + 1, seq_len=sk.grid_size(box) ** 2,
    )
    return ds.inputs, ds.targets, ds.height, ds.width


def build_maze_arrays(
    h: int,
    w: int,
    n: int,
    rng: np.random.Generator,
    min_path_len: int = 0,
    augment: bool = False,
    max_path_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    ds = build_dataset(
        "maze", n, rng, height=h, width=w, seq_len=h*w, num_tokens=5,
        min_path_len=min_path_len, max_path_len=max_path_len, augment=augment,
        require_optimal=True,
    )
    return ds.inputs, ds.targets, ds.height, ds.width


def build_smarthome_arrays(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int, int]:
    ds = build_dataset(
        "smarthome", n, rng, height=1, width=sh_task.SEQ_LEN,
        seq_len=sh_task.SEQ_LEN, num_tokens=sh_task.NUM_TOKENS, pad_token=sh_task.PAD,
    )
    return ds.inputs, ds.targets, ds.height, ds.width


def build_arc_arrays(
    n: int,
    rng: np.random.Generator,
    transform: str = "flip_h",
    canvas_h: int = 10,
    canvas_w: int = 10,
    n_colors: int = 5,
    pad_token: int = 10,
    max_h: int = 6,
    max_w: int = 6,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    ds = build_dataset(
        "arc", n, rng, transform=transform, canvas_h=canvas_h, canvas_w=canvas_w,
        height=canvas_h, width=canvas_w, seq_len=canvas_h*canvas_w,
        n_colors=n_colors, pad_token=pad_token, num_tokens=pad_token+1,
        max_h=max_h, max_w=max_w,
    )
    return ds.inputs, ds.targets, ds.height, ds.width


def build_babyai_arrays(
    n: int,
    rng: np.random.Generator,
    height: int = 8,
    width: int = 8,
    wall_prob: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    ds = build_dataset(
        "babyai", n, rng, height=height, width=width, seq_len=height*width,
        num_tokens=babyai_task.NUM_TOKENS, wall_prob=wall_prob,
    )
    return ds.inputs, ds.targets, ds.height, ds.width
