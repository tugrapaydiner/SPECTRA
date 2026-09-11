"""Independent reference-free Sudoku checks and exact CPU enumeration.

Solved(answer) AND preserved_givens implies partial_validity(puzzle). Under the
strict integer/shape contract, rechecking the puzzle's groups is redundant.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Iterable
from .identity import integral_tuple, positive_int


def _geometry(box: int) -> tuple[int, int]:
    box = positive_int(box, "box"); n = box * box
    if n > 63: raise ValueError("side lengths <= 63 supported")
    return n, n * n


def valid(puzzle: Iterable[int], answer: Iterable[int], box: int) -> bool:
    n, cells = _geometry(box); x = integral_tuple(puzzle); y = integral_tuple(answer)
    if len(x) != cells or len(y) != cells: raise ValueError("puzzle/answer must have box**4 cells")
    row = [0] * n; col = [0] * n; boxes = [0] * n
    for i, v in enumerate(y):
        if not 1 <= v <= n or (x[i] != 0 and x[i] != v): return False
        r, c = divmod(i, n); b = (r // box) * box + c // box; bit = 1 << (v - 1)
        if (row[r] | col[c] | boxes[b]) & bit: return False
        row[r] |= bit; col[c] |= bit; boxes[b] |= bit
    return True


def structural_score(puzzle: Iterable[int], answer: Iterable[int], box: int) -> float:
    """Absolute structural quality diagnostic, not a solve probability."""
    n, cells = _geometry(box); x = integral_tuple(puzzle); y = integral_tuple(answer)
    if len(x) != cells or len(y) != cells: raise ValueError("geometry mismatch")
    if _initial(x, box) is None: return 0.0
    target = set(range(1, n + 1)); ok = 0
    for r in range(n): ok += set(y[r*n:(r+1)*n]) == target
    for c in range(n): ok += {y[r*n+c] for r in range(n)} == target
    for br in range(box):
        for bc in range(box):
            ok += {y[(br*box+r)*n+bc*box+c] for r in range(box) for c in range(box)} == target
    givens = [i for i, value in enumerate(x) if value]
    kept = sum(y[i] == x[i] for i in givens) / len(givens) if givens else 1.0
    return 0.25 * (ok / n + kept)


def _initial(puzzle: tuple[int, ...], box: int):
    n, cells = _geometry(box)
    if len(puzzle) != cells: raise ValueError("geometry mismatch")
    row = [0] * n; col = [0] * n; boxes = [0] * n; empties = []
    for i, value in enumerate(puzzle):
        if not 0 <= value <= n: return None
        if value == 0:
            empties.append(i); continue
        r, c = divmod(i, n); b = (r // box) * box + c // box; bit = 1 << (value - 1)
        if (row[r] | col[c] | boxes[b]) & bit: return None
        row[r] |= bit; col[c] |= bit; boxes[b] |= bit
    return row, col, boxes, empties


@dataclass(frozen=True)
class SolveStats:
    nodes: int
    backtracks: int
    complete: bool


def solutions(puzzle: Iterable[int], box: int, *, limit: int = 2, node_limit: int = 1_000_000,
              rng: random.Random | None = None) -> tuple[list[tuple[int, ...]], SolveStats]:
    n, _ = _geometry(box); x = integral_tuple(puzzle)
    positive_int(limit, "limit"); positive_int(node_limit, "node_limit")
    initial = _initial(x, box)
    if initial is None: return [], SolveStats(0, 0, True)
    row, col, boxes, empty = initial; grid = list(x); found = []; nodes = 0; backtracks = 0; complete = True
    full = (1 << n) - 1
    def dfs():
        nonlocal nodes, backtracks, complete
        if nodes >= node_limit:
            complete = False; return True
        nodes += 1
        if not empty:
            found.append(tuple(grid)); return len(found) >= limit
        best = []; size = n + 1
        for j, i in enumerate(empty):
            r, c = divmod(i, n); b = (r // box) * box + c // box
            mask = full & ~(row[r] | col[c] | boxes[b]); count = mask.bit_count()
            if count < size: best = [(j, mask)]; size = count
            elif count == size: best.append((j, mask))
            if size == 0:
                backtracks += 1; return False
        j, mask = best[rng.randrange(len(best))] if rng is not None else best[0]
        i = empty.pop(j); r, c = divmod(i, n); b = (r // box) * box + c // box
        bits = []
        while mask:
            bit = mask & -mask; mask ^= bit; bits.append(bit)
        if rng is not None: rng.shuffle(bits)
        for bit in bits:
            row[r] |= bit; col[c] |= bit; boxes[b] |= bit; grid[i] = bit.bit_length()
            stop = dfs()
            row[r] ^= bit; col[c] ^= bit; boxes[b] ^= bit; grid[i] = 0
            if stop:
                empty.insert(j, i); return True
        empty.insert(j, i); backtracks += 1; return False
    dfs()
    return found, SolveStats(nodes, backtracks, complete)


def all_four_by_four() -> tuple[tuple[int, ...], ...]:
    boards, stats = solutions((0,) * 16, 2, limit=1000)
    if not stats.complete or len(boards) != 288 or len(set(boards)) != 288:
        raise RuntimeError("4x4 universe failed independent enumeration")
    return tuple(boards)


def generate_unique(box: int, clues: int, rng: random.Random, *, attempts: int = 128) -> tuple[tuple[int, ...], tuple[int, ...]]:
    n, cells = _geometry(box); positive_int(attempts, "attempts")
    if isinstance(clues, bool) or not isinstance(clues, int) or not 0 <= clues <= cells: raise ValueError("clues out of range")
    for _ in range(attempts):
        boards, stats = solutions((0,) * cells, box, limit=1, rng=rng)
        if not stats.complete or not boards: continue
        target = boards[0]; puzzle = list(target); order = list(range(cells)); rng.shuffle(order); remaining = cells
        for index in order:
            if remaining == clues: break
            value = puzzle[index]; puzzle[index] = 0
            ss, proof = solutions(puzzle, box, limit=2)
            if not proof.complete or len(ss) != 1: puzzle[index] = value
            else: remaining -= 1
        if remaining == clues: return tuple(puzzle), target
    raise RuntimeError("requested unique puzzle not generated within retry limit")
