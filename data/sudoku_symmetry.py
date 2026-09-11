"""Exact, input-only isomorphism keys for 4x4 Sudoku (not a 9x9 heuristic).

The 128 spatial maps are band/stack permutations, row/column permutations
within bands/stacks, and optional transpose. First-occurrence relabeling removes
all 24 digit permutations exactly, including puzzles with absent digits.
No reference solution is used to compute a key. See docs/SYMMETRY_AUDIT.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools

import numpy as np

POLICY = "sudoku4_isomorphism_v1"


def _board(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.shape not in ((16,), (4, 4)) or value.dtype.kind not in "iu":
        raise ValueError("Sudoku4 requires an integer (16,) or (4,4) board")
    if np.any(value < 0) or np.any(value > 4):
        raise ValueError("Sudoku4 symbols must be in [0,4], with 0 blank")
    return value.reshape(16).astype(np.uint8, copy=True)


@lru_cache(maxsize=1)
def _spatial_maps() -> np.ndarray:
    within = tuple(itertools.permutations(range(2)))
    orders = [tuple(2*b+i for b in bands for i in rows[b])
              for bands in within for rows in itertools.product(within, repeat=2)]
    grid = np.arange(16).reshape(4, 4)
    maps = np.array([g[np.ix_(r, c)].ravel() for g in (grid, grid.T)
                     for r in orders for c in orders], dtype=np.int64)
    maps.setflags(write=False)
    return maps


def _normalize(rows: np.ndarray) -> np.ndarray:
    # Rank nonzero symbols by first appearance. Unseen symbols need no rank in
    # the canonical board; a complete invertible labeling is built for witnesses.
    first = np.stack([np.where(rows == v, np.arange(16), 16).min(axis=1)
                      for v in range(1, 5)], axis=1)
    ranks = 1 + (first[:, :, None] > first[:, None, :]).sum(axis=2)
    labels = np.column_stack((np.zeros(len(rows), dtype=np.int64), ranks))
    return labels[np.arange(len(rows))[:, None], rows].astype(np.uint8)


@dataclass(frozen=True)
class CanonicalSudoku4:
    board: tuple[int, ...]
    positions: tuple[int, ...]  # canonical position -> original position
    labels: tuple[int, ...]  # original symbol -> canonical symbol (0 fixed)

    @property
    def key(self) -> str:
        return hashlib.sha256(b"spectra.sudoku4.isomorphism.v1\0" + bytes(self.board)).hexdigest()

    def transform(self, board: np.ndarray) -> np.ndarray:
        x = _board(board)
        return np.asarray(self.labels, dtype=np.uint8)[x[list(self.positions)]].astype(np.int64)

    def restore(self, canonical: np.ndarray) -> np.ndarray:
        x = _board(canonical)
        inverse = np.argsort(self.labels)
        result = np.empty(16, dtype=np.int64)
        result[list(self.positions)] = inverse[x]
        return result


def canonical_sudoku4(board: np.ndarray) -> CanonicalSudoku4:
    """Return the lexicographically minimal representative and a bijective witness."""
    x = _board(board)
    maps = _spatial_maps()
    variants = x[maps]
    normalized = _normalize(variants)
    index = min(range(len(maps)), key=lambda i: normalized[i].tobytes())
    order = list(dict.fromkeys(int(v) for v in variants[index] if v))
    order.extend(v for v in range(1, 5) if v not in order)
    labels = [0]*5
    for new, old in enumerate(order, 1):
        labels[old] = new
    return CanonicalSudoku4(tuple(map(int, normalized[index])),
                           tuple(map(int, maps[index])), tuple(labels))


def sudoku4_orbit_key(board: np.ndarray) -> str:
    return canonical_sudoku4(board).key


class Sudoku4OrbitLookup:
    """Training-only exact symmetry lookup; abstain on unseen puzzle orbits.

    This is a diagnostic comparator, not a novel neural reasoning algorithm.
    Any stored answer is independently validated before insertion. Keys contain
    only puzzle inputs. Solving never sees an evaluation reference answer.
    """
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        from data import sudoku
        x, y = np.asarray(inputs), np.asarray(targets)
        if x.ndim != 2 or x.shape[1] != 16 or y.shape != x.shape:
            raise ValueError("lookup training arrays must have matching [N,16] shapes")
        self._answers: dict[str, np.ndarray] = {}
        for inp, target in zip(x, y):
            inp, target = _board(inp), _board(target)
            if not sudoku.is_solved(target.reshape(4, 4), 2) or not sudoku.respects_clues(inp.reshape(4, 4), target.reshape(4, 4), 2):
                raise ValueError("lookup training target is not a valid clue-respecting solution")
            canonical = canonical_sudoku4(inp)
            self._answers.setdefault(canonical.key, canonical.transform(target))

    @property
    def orbit_count(self) -> int:
        return len(self._answers)

    def solve(self, puzzle: np.ndarray) -> np.ndarray | None:
        canonical = canonical_sudoku4(puzzle)
        answer = self._answers.get(canonical.key)
        return None if answer is None else canonical.restore(answer)
