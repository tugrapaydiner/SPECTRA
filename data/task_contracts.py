"""Explicit task/data contracts for SPECTRA milestone 03.

The YAML file is the experiment declaration; this module is the executable
contract.  A retained configuration must agree with the generator on sequence
shape, spatial dimensions, token vocabulary, padding/mask policy, and the exact
local task variant being evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from data import arc, babyai, smarthome, sudoku


@dataclass(frozen=True)
class TaskContract:
    task: str
    scope: str
    official_benchmark: bool
    num_tokens: int
    seq_len: int
    height: int
    width: int
    pad_token: int | None
    input_mask_policy: str
    target_mask_policy: str
    generator_kwargs: dict[str, Any]


def _req_int(data: Mapping[str, Any], key: str) -> int:
    if key not in data:
        raise ValueError(f"data.{key} is required")
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"data.{key} must be an integer, got {type(value).__name__}")
    return int(value)


def _common(data: Mapping[str, Any]) -> tuple[int, int, int, int]:
    num_tokens = _req_int(data, "num_tokens")
    seq_len = _req_int(data, "seq_len")
    height = _req_int(data, "height")
    width = _req_int(data, "width")
    if num_tokens <= 0 or height <= 0 or width <= 0:
        raise ValueError("num_tokens/height/width must be positive")
    if seq_len != height * width:
        raise ValueError(
            f"data.seq_len={seq_len} must equal height*width={height * width}"
        )
    return num_tokens, seq_len, height, width


def contract_from_config(task: str, data: Mapping[str, Any]) -> TaskContract:
    """Resolve and validate the executable contract for one task config."""
    num_tokens, seq_len, height, width = _common(data)

    if task == "sudoku":
        box = _req_int(data, "box")
        side = sudoku.grid_size(box)
        if box <= 0 or height != side or width != side or seq_len != side * side:
            raise ValueError(
                f"Sudoku box={box} requires {side}x{side} and seq_len={side * side}"
            )
        if num_tokens != side + 1:
            raise ValueError(
                f"Sudoku vocabulary must be blank 0 + digits 1..{side}: num_tokens={side + 1}"
            )
        min_clues = int(data.get("min_clues", side * side // 2))
        max_clues = int(data.get("max_clues", min_clues))
        if not 0 < min_clues <= max_clues <= side * side:
            raise ValueError("Sudoku clue range must satisfy 0 < min <= max <= side^2")
        return TaskContract(
            task=task,
            scope="generated Sudoku with randomized completed-board construction",
            official_benchmark=False,
            num_tokens=num_tokens,
            seq_len=seq_len,
            height=height,
            width=width,
            pad_token=None,
            input_mask_policy="all_cells",
            target_mask_policy="all_cells",
            generator_kwargs={
                "box": box,
                "min_clues": min_clues,
                "max_clues": max_clues,
                "require_unique": bool(data.get("require_unique", True)),
                "augment": bool(data.get("augment", True)),
                "solution_method": str(data.get("solution_method", "random_backtracking")),
                "num_tokens": num_tokens,
                "seq_len": seq_len,
            },
        )

    if task == "maze":
        if num_tokens != 5:
            raise ValueError("Maze vocabulary is exactly 0=wall,1=open,2=start,3=goal,4=path")
        if height < 3 or width < 3 or height % 2 == 0 or width % 2 == 0:
            raise ValueError("Maze height and width must be odd integers >= 3")
        min_path_len = int(data.get("min_path_len", 2))
        max_path_len = data.get("max_path_len", None)
        max_path_len = None if max_path_len is None else int(max_path_len)
        if min_path_len < 2 or (max_path_len is not None and max_path_len < min_path_len):
            raise ValueError("Maze path-length contract is invalid")
        return TaskContract(
            task=task,
            scope="synthetic perfect-maze shortest-path overlay task",
            official_benchmark=False,
            num_tokens=num_tokens,
            seq_len=seq_len,
            height=height,
            width=width,
            pad_token=None,
            input_mask_policy="all_cells",
            target_mask_policy="all_cells",
            generator_kwargs={
                "height": height,
                "width": width,
                "min_path_len": min_path_len,
                "max_path_len": max_path_len,
                "augment": bool(data.get("augment", True)),
                "require_optimal": bool(data.get("require_optimal", True)),
                "num_tokens": num_tokens,
                "seq_len": seq_len,
            },
        )

    if task == "arc":
        pad_token = _req_int(data, "pad_token")
        n_colors = int(data.get("n_colors", 9))
        max_h = int(data.get("max_h", min(6, height)))
        max_w = int(data.get("max_w", min(6, width)))
        transform = str(data.get("transform", "flip_h"))
        if transform not in set(arc.TRANSFORMS) | {"recolor"}:
            raise ValueError(f"Unsupported synthetic ARC-style transform: {transform!r}")
        if not 1 <= n_colors < pad_token < num_tokens:
            raise ValueError("ARC-style contract requires colors < pad_token < num_tokens")
        if max_h < 2 or max_w < 2 or max_h > height or max_w > width:
            raise ValueError("ARC-style max grid size must fit inside the declared canvas")
        return TaskContract(
            task=task,
            scope="synthetic local ARC-style single-transform task; NOT official ARC/ARC-AGI",
            official_benchmark=False,
            num_tokens=num_tokens,
            seq_len=seq_len,
            height=height,
            width=width,
            pad_token=pad_token,
            input_mask_policy="non_padding",
            target_mask_policy="non_padding",
            generator_kwargs={
                "transform": transform,
                "canvas_h": height,
                "canvas_w": width,
                "max_h": max_h,
                "max_w": max_w,
                "n_colors": n_colors,
                "pad_token": pad_token,
                "num_tokens": num_tokens,
                "seq_len": seq_len,
            },
        )

    if task == "babyai":
        if num_tokens != babyai.NUM_TOKENS:
            raise ValueError(
                f"Synthetic BabyAI-style next-state generator emits {babyai.NUM_TOKENS} tokens; "
                f"config declares {num_tokens}"
            )
        wall_prob = float(data.get("wall_prob", 0.2))
        if not 0.0 <= wall_prob < 1.0:
            raise ValueError("BabyAI-style wall_prob must be in [0,1)")
        return TaskContract(
            task=task,
            scope="synthetic local BabyAI-style one-step gridworld; NOT official BabyAI",
            official_benchmark=False,
            num_tokens=num_tokens,
            seq_len=seq_len,
            height=height,
            width=width,
            pad_token=None,
            input_mask_policy="all_cells",
            target_mask_policy="all_cells",
            generator_kwargs={
                "height": height,
                "width": width,
                "wall_prob": wall_prob,
                "num_tokens": num_tokens,
                "seq_len": seq_len,
            },
        )

    if task == "smarthome":
        if height != 1 or width != smarthome.SEQ_LEN or seq_len != smarthome.SEQ_LEN:
            raise ValueError(
                f"Smart-home generator requires shape 1x{smarthome.SEQ_LEN}"
            )
        if num_tokens != smarthome.NUM_TOKENS:
            raise ValueError(
                f"Smart-home generator vocabulary is {smarthome.NUM_TOKENS}, got {num_tokens}"
            )
        pad_token = int(data.get("pad_token", smarthome.PAD))
        if pad_token != smarthome.PAD:
            raise ValueError(f"Smart-home PAD token must be {smarthome.PAD}")
        return TaskContract(
            task=task,
            scope="synthetic deterministic smart-home policy snapshot task",
            official_benchmark=False,
            num_tokens=num_tokens,
            seq_len=seq_len,
            height=height,
            width=width,
            pad_token=pad_token,
            input_mask_policy="non_padding",
            target_mask_policy="all_cells",
            generator_kwargs={
                "num_tokens": num_tokens,
                "seq_len": seq_len,
                "pad_token": pad_token,
            },
        )

    raise ValueError(f"Unsupported task configuration: {task!r}")
