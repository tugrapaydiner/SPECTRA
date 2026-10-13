"""Generic exact CNF reduction followed by an independently maintained BDD.

This comparator computes its own equivalences from original binary clauses,
substitutes signed variables, propagates units, and retains a total original
witness map. It never imports SPECTRA's quotient or accepts prepared answers.
These are established preprocessing techniques, not a novelty claim.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Iterable

from .bdd import CompiledCNF, freeze_cnf
from .preprocess import binary_equivalences


@dataclass(frozen=True)
class CNFReduction:
    original_nvars: int
    nvars: int
    clauses: tuple[tuple[int, ...], ...]
    mapping: tuple[int, ...]
    fixed: tuple[bool, ...]
    equivalent_variables_removed: int
    fixed_classes: int
    input_clauses: int
    contradiction: bool

    def assumptions(self, literals: Iterable[int]) -> tuple[int, ...] | None:
        """Validate every original literal, even after a known contradiction."""
        result = []
        contradiction = self.contradiction
        for literal in literals:
            if (type(literal) is not int or literal == 0
                    or abs(literal) > self.original_nvars):
                raise ValueError('assumption must be a nonzero in-range exact integer')
            index = abs(literal) - 1
            target = self.mapping[index]
            if target:
                result.append(target if literal > 0 else -target)
            elif self.fixed[index] != (literal > 0):
                contradiction = True
        return None if contradiction else tuple(result)

    def lift(self, model: tuple[bool, ...]) -> tuple[bool, ...]:
        if (type(model) is not tuple or len(model) != self.nvars
                or any(type(value) is not bool for value in model)):
            raise ValueError('partial or non-Boolean reduced model')
        return tuple(self.fixed[i] if target == 0 else
                     model[abs(target)-1] == (target > 0)
                     for i, target in enumerate(self.mapping))


def reduce_cnf(nvars: int, clauses: Iterable[Iterable[int]]) -> CNFReduction:
    """Lossless signed substitution and unit propagation of a generic CNF.

    The implication pass uses only clauses of width at most two. Longer clauses
    are substituted and propagated, never used as binary implications. Every
    remaining equivalence class is declared, including unconstrained classes,
    so later assumptions can constrain otherwise absent original variables.
    """
    original = freeze_cnf(nvars, clauses)
    contradiction, pairs = binary_equivalences(nvars, original)
    if contradiction:
        return CNFReduction(nvars, 0, ((),), (0,) * nvars, (False,) * nvars,
                            0, 0, len(original), True)
    parent = list(range(nvars + 1))
    parity = [0] * (nvars + 1)  # value[v] XOR value[parent[v]]

    def root(vertex: int) -> tuple[int, int]:
        chain = []
        current = vertex
        difference = 0
        while parent[current] != current:
            chain.append(current)
            difference ^= parity[current]
            current = parent[current]
        remaining = difference
        for member in chain:
            old = parity[member]
            parent[member] = current
            parity[member] = remaining
            remaining ^= old
        return current, difference

    for left, right in pairs:
        a, pa = root(left)
        b, pb = root(abs(right))
        relation = pa ^ pb ^ int(right < 0)
        if a == b:
            if relation:
                raise AssertionError('inconsistent equivalence inference')
        elif a < b:
            parent[b], parity[b] = a, relation
        else:
            parent[a], parity[a] = b, relation
    resolved = [root(v) for v in range(1, nvars + 1)]
    representatives = sorted({r for r, _ in resolved})
    compact = {r: i + 1 for i, r in enumerate(representatives)}
    signed = tuple(compact[r] * (-1 if sign else 1) for r, sign in resolved)
    substituted = freeze_cnf(len(representatives),
        (tuple(signed[abs(lit)-1] * (1 if lit > 0 else -1) for lit in row)
         for row in original))
    # Event-driven unit propagation: each assigned literal visits only its
    # incidence lists; every clause is scanned for its final unit at most once.
    k = len(representatives)
    occurrences: dict[int, list[int]] = {}
    remaining = [len(row) for row in substituted]
    satisfied = bytearray(len(substituted))
    values = [0] * (k + 1)
    pending = deque()
    contradiction = False
    for index, row in enumerate(substituted):
        if not row:
            contradiction = True
        elif len(row) == 1:
            pending.append(row[0])
        for literal in row:
            occurrences.setdefault(literal, []).append(index)
    while pending and not contradiction:
        literal = pending.popleft()
        v, sign = abs(literal), 1 if literal > 0 else -1
        if values[v]:
            contradiction = values[v] != sign
            continue
        values[v] = sign
        for index in occurrences.get(literal, ()):
            satisfied[index] = 1
        for index in occurrences.get(-literal, ()):
            if satisfied[index]:
                continue
            remaining[index] -= 1
            if remaining[index] == 0:
                contradiction = True
                break
            if remaining[index] == 1:
                unit = next(x for x in substituted[index] if not values[abs(x)])
                pending.append(unit)
    removed = nvars - k
    fixed_count = sum(bool(value) for value in values)
    if contradiction:
        return CNFReduction(nvars, 0, ((),), (0,) * nvars, (False,) * nvars,
                            removed, fixed_count, len(original), True)
    free = {v: i+1 for i, v in enumerate(v for v in range(1, k+1) if not values[v])}
    residual = freeze_cnf(len(free),
        (tuple(free[abs(lit)] * (1 if lit > 0 else -1)
               for lit in row if not values[abs(lit)])
         for index, row in enumerate(substituted) if not satisfied[index]))
    mapping = tuple(0 if values[abs(v)] else free[abs(v)] * (1 if v > 0 else -1)
                    for v in signed)
    fixed = tuple(bool(values[abs(v)] * (1 if v > 0 else -1) > 0) for v in signed)
    return CNFReduction(nvars, len(free), residual, mapping, fixed, removed,
                        fixed_count, len(original), False)


class ReducedCNF:
    """Compile the independently reduced CNF and lift every complete model.

    The original input and every request are validated. UNSAT is an exact
    backend decision, not an externally checkable proof. The same explicit
    native dependency and resource guards apply as to CompiledCNF.
    """
    def __init__(self, nvars: int, clauses: Iterable[Iterable[int]], *,
                 backend: str = 'cudd', order: str = 'bfs', reordering: bool = True,
                 max_nodes: int = 2_000_000, memory_bytes: int = 512 << 20,
                 construction: str = 'reduced'):
        if construction != 'reduced':
            raise ValueError('ReducedCNF requires explicit reduced construction')
        if backend not in ('cudd', 'autoref') or order not in ('natural', 'bfs', 'occurrence'):
            raise ValueError('explicit backend and query-blind variable order required')
        if type(reordering) is not bool:
            raise ValueError('reordering must be an exact bool')
        if type(max_nodes) is not int or max_nodes < 1:
            raise ValueError('max_nodes must be a positive exact integer')
        if type(memory_bytes) is not int or memory_bytes < (1 << 20):
            raise ValueError('memory_bytes must be an exact integer of at least 1 MiB')
        self._lock = threading.RLock()
        self._engine = None
        self.nvars = nvars
        self.reduction = reduce_cnf(nvars, clauses)
        self._engine = CompiledCNF(self.reduction.nvars, self.reduction.clauses,
            backend=backend, order=order, reordering=reordering, max_nodes=max_nodes,
            memory_bytes=memory_bytes, construction='frontier')

    def model(self, assumptions: Iterable[int] = ()) -> tuple[bool, ...] | None:
        with self._lock:
            if self._engine is None:
                raise RuntimeError('reduced CNF is closed')
            reduced = self.reduction.assumptions(assumptions)
            if reduced is None:
                return None
            model = self._engine.model(reduced)
            return None if model is None else self.reduction.lift(model)

    @property
    def info(self) -> dict:
        with self._lock:
            if self._engine is None:
                raise RuntimeError('reduced CNF is closed')
            return {**self._engine.info,
                'construction': 'reduced', 'bdd_construction': 'frontier',
                'original_variables': self.nvars,
                'compiled_variables': self.reduction.nvars,
                'original_normalized_clauses': self.reduction.input_clauses,
                'equivalent_variables_removed': self.reduction.equivalent_variables_removed,
                'unit_fixed_classes': self.reduction.fixed_classes,
                'generic_equivalence_elimination': True,
                'generic_unit_propagation': True,
                'full_original_witness_reconstruction': True,
                'preprocessing_contradiction': self.reduction.contradiction}

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None

    def __enter__(self) -> 'ReducedCNF':
        with self._lock:
            if self._engine is None:
                raise RuntimeError('reduced CNF is closed')
        return self

    def __exit__(self, *_args) -> None:
        self.close()
