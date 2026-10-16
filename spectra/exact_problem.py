"""Immutable original-address inputs for SPECTRA's exact-query API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class ExactProblem:
    """A fixed simple conflict graph with one or two colours per vertex."""

    n: int
    palette: int
    edges: tuple[tuple[int, int], ...]
    masks: tuple[int, ...]

    @classmethod
    def create(cls, n: int, palette: int, edges: Iterable[Sequence[int]],
               masks: Iterable[int]) -> 'ExactProblem':
        if type(n) is not int or not 0 <= n <= 100_000:
            raise ValueError('n must be an exact integer in [0, 100000]')
        if type(palette) is not int or not 1 <= palette <= 64:
            raise ValueError('palette must be an exact integer in [1, 64]')
        snapshot = tuple(masks)
        if len(snapshot) != n:
            raise ValueError('mask count differs from vertex count')
        palette_mask = (1 << palette) - 1 if palette < 64 else (1 << 64) - 1
        for mask in snapshot:
            if (type(mask) is not int or mask <= 0 or mask & ~palette_mask
                    or mask.bit_count() not in (1, 2)):
                raise ValueError('each original domain must contain one or two palette colours')
        copied: list[tuple[int, int]] = []
        for edge in edges:
            row = tuple(edge)
            if (len(row) != 2 or any(type(v) is not int or not 0 <= v < n for v in row)
                    or row[0] == row[1]):
                raise ValueError('edge must contain two distinct in-range exact integers')
            copied.append((min(row), max(row)))
        if len(copied) > 2_000_000:
            raise ValueError('edge count exceeds native bound')
        return cls(n, palette, tuple(sorted(set(copied))), snapshot)

    def colors(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(c for c in range(self.palette) if mask >> c & 1)
                     for mask in self.masks)

    def freeze_query(self, restrictions: Iterable[Sequence[int]]) -> tuple[tuple[int, int], ...]:
        copied: list[tuple[int, int]] = []
        for item in restrictions:
            row = tuple(item)
            if (len(row) != 2 or type(row[0]) is not int or not 0 <= row[0] < self.n
                    or type(row[1]) is not int or not 0 <= row[1] < 1 << 64):
                raise ValueError('query must contain in-range vertex/uint64-mask pairs')
            copied.append((row[0], row[1]))
        if len(copied) > 1_000_000:
            raise ValueError('query exceeds one million pins')
        return tuple(copied)

    def check(self, labels: bytes, query: Iterable[Sequence[int]] = ()) -> bool:
        """Check a complete label vector against original inputs only."""
        try:
            frozen = self.freeze_query(query)
            if type(labels) is not bytes or len(labels) != self.n:
                return False
            allowed = list(self.masks)
            for vertex, mask in frozen:
                allowed[vertex] &= mask
            if any(color >= self.palette or not ((allowed[v] >> color) & 1)
                   for v, color in enumerate(labels)):
                return False
            return all(labels[left] != labels[right] for left, right in self.edges)
        except (TypeError, ValueError, OverflowError):
            return False
