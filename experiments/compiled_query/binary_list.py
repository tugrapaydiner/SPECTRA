"""Original-address binary-list adapter for the generic compilation control.

Encoding, Boolean witness decoding, and the independent original-edge checker
are intentionally separate.  This contract is feasibility, not the stronger
minimum-migration/UNSAT-path contract of experiments.real_traffic.PacketService.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable, Iterable, Sequence

from experiments.compiled_query.bdd import CompiledCNF


@dataclass(frozen=True)
class BinaryListProblem:
    n: int
    palette: int
    edges: tuple[tuple[int, int], ...]
    masks: tuple[int, ...]

    @classmethod
    def create(cls, n: int, palette: int, edges: Iterable[Sequence[int]],
               masks: Iterable[int]) -> 'BinaryListProblem':
        if type(n) is not int or not 0 <= n <= 1_000_000:
            raise ValueError('n must be an exact nonnegative bounded integer')
        if type(palette) is not int or not 1 <= palette <= 64:
            raise ValueError('palette must be an exact integer in [1, 64]')
        snapshot = tuple(masks)
        if len(snapshot) != n:
            raise ValueError('mask count differs from vertex count')
        for mask in snapshot:
            if (type(mask) is not int or mask <= 0 or mask >> palette
                    or mask.bit_count() not in (1, 2)):
                raise ValueError('each original domain must contain one or two palette colours')
        copied = []
        for edge in edges:
            row = tuple(edge)
            if (len(row) != 2 or any(type(v) is not int or not 0 <= v < n for v in row)
                    or row[0] == row[1]):
                raise ValueError('edge must contain two distinct in-range exact integers')
            copied.append(tuple(sorted(row)))
        return cls(n, palette, tuple(sorted(set(copied))), snapshot)

    def colors(self) -> tuple[tuple[int, int], ...]:
        return tuple(((mask & -mask).bit_length() - 1, mask.bit_length() - 1)
                     for mask in self.masks)

    def cnf(self) -> tuple[tuple[int, ...], ...]:
        choices = self.colors()
        clauses: list[tuple[int, ...]] = [(-(v + 1),)
                    for v, (low, high) in enumerate(choices) if low == high]
        for left, right in self.edges:
            shared = self.masks[left] & self.masks[right]
            while shared:
                bit = shared & -shared
                shared ^= bit
                color = bit.bit_length() - 1
                # If selecting c is literal L, forbid simultaneous selections
                # with (~L_left OR ~L_right). Singleton vertices are fixed low.
                ll = left + 1 if color == choices[left][1] and choices[left][0] != color else -(left + 1)
                rr = right + 1 if color == choices[right][1] and choices[right][0] != color else -(right + 1)
                clauses.append((-ll, -rr))
        return tuple(clauses)

    def freeze_query(self, restrictions: Iterable[Sequence[int]]) -> tuple[tuple[int, int], ...]:
        copied: list[tuple[int, int]] = []
        for item in restrictions:
            row = tuple(item)
            if (len(row) != 2 or type(row[0]) is not int or not 0 <= row[0] < self.n
                    or type(row[1]) is not int or not 0 <= row[1] < (1 << 64)):
                raise ValueError('query must contain in-range vertex/uint64-mask pairs')
            copied.append(row)
        return tuple(copied)

    def assumptions(self, query: Sequence[tuple[int, int]]) -> tuple[int, ...] | None:
        domains: dict[int, int] = {}
        for vertex, allowed in query:
            domains[vertex] = domains.get(vertex, self.masks[vertex]) & allowed
        literals = []
        for vertex, remaining in sorted(domains.items()):
            if remaining == 0:
                return None
            if remaining.bit_count() == 1:
                original = self.masks[vertex]
                high = original ^ (original & -original)
                literals.append(vertex + 1 if remaining == high else -(vertex + 1))
        return tuple(literals)

    def check(self, labels: bytes, query: Sequence[tuple[int, int]]) -> bool:
        """Scalar observer: original masks, original edges, current restrictions."""
        if type(labels) is not bytes or len(labels) != self.n:
            return False
        if any(color >= self.palette or not ((self.masks[v] >> color) & 1)
               for v, color in enumerate(labels)):
            return False
        if any(labels[left] == labels[right] for left, right in self.edges):
            return False
        return all((allowed >> labels[vertex]) & 1 for vertex, allowed in query)


@dataclass(frozen=True)
class CompiledAnswer:
    status: str
    labels: bytes
    unsat_certificate: None = None


class CompiledListSession:
    """Full original-vertex SAT output; a separately owned observer checks it.

    ``UNSAT`` records the compiler's exact decision but supplies no portable proof.
    It must never be presented as ``UNSAT_VERIFIED`` or as a packet-service answer.
    """

    def __init__(self, problem: BinaryListProblem, *, backend: str = 'cudd',
                 order: str = 'bfs', reordering: bool = True,
                 checker: Callable[[bytes, Sequence[tuple[int, int]]], bool] | None = None,
                 max_nodes: int = 2_000_000, memory_bytes: int = 512 << 20,
                 construction: str = 'balanced64'):
        # Revalidate even a caller-constructed dataclass and own immutable inputs.
        self.problem = BinaryListProblem.create(problem.n, problem.palette,
                                               problem.edges, problem.masks)
        self._choices = self.problem.colors()
        self._checker = self.problem.check if checker is None else checker
        self._lock = threading.RLock()
        self._engine = CompiledCNF(problem.n, self.problem.cnf(), backend=backend,
                                  order=order, reordering=reordering,
                                  max_nodes=max_nodes, memory_bytes=memory_bytes,
                                  construction=construction)

    def solve(self, restrictions: Iterable[Sequence[int]] = ()) -> CompiledAnswer:
        query = self.problem.freeze_query(restrictions)
        with self._lock:
            if self._engine is None:
                raise RuntimeError('compiled list session is closed')
            assumptions = self.problem.assumptions(query)
            if assumptions is None:
                return CompiledAnswer('UNSAT', b'')
            model = self._engine.model(assumptions)
            if model is None:
                return CompiledAnswer('UNSAT', b'')
            labels = bytes(self._choices[v][int(value)] for v, value in enumerate(model))
            if not self._checker(labels, query):
                raise AssertionError('compiled model failed the original-input observer')
            return CompiledAnswer('SAT_VERIFIED', labels)

    @property
    def info(self) -> dict:
        with self._lock:
            if self._engine is None:
                raise RuntimeError('compiled list session is closed')
            return self._engine.info

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None

    def __enter__(self) -> 'CompiledListSession':
        with self._lock:
            if self._engine is None:
                raise RuntimeError('compiled list session is closed')
        return self

    def __exit__(self, *_args) -> None:
        self.close()
