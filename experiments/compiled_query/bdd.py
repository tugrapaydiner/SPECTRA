"""Generic CNF compilation through the independently maintained ``dd`` package.

The native arm *requires* dd.cudd.  dd.autoref is a separately named diagnostic,
never an automatic fallback.  No SPECTRA quotient, SCC, answer cache or prepared
solution is supplied to either compiler.  Resource checks between CUDD operations
are soft guards; a caller must use process isolation for a hard wall/RSS limit.
"""
from __future__ import annotations

from collections import Counter, deque
import importlib
import importlib.metadata
import threading
from typing import Iterable, Sequence


class BackendUnavailable(RuntimeError):
    """The explicitly requested independent backend cannot be imported."""


class CompilationLimit(RuntimeError):
    """A soft compilation guard was exceeded; this is not a solver result."""


def freeze_cnf(nvars: int, clauses: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    """Validate all original literals before eliminating generic tautologies.

    The empty formula denotes TRUE; an empty clause denotes FALSE.  Literal zero,
    booleans, non-integers and out-of-range variables are rejected, even inside a
    tautological clause.  Deduplication is purely propositional, not domain aware.
    """
    if type(nvars) is not int or not 0 <= nvars <= 1_000_000:
        raise ValueError('nvars must be an exact integer in [0, 1000000]')
    rows: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for clause in clauses:
        row = tuple(clause)
        for literal in row:
            if type(literal) is not int or literal == 0 or abs(literal) > nvars:
                raise ValueError('CNF literal must be a nonzero in-range exact integer')
        values = set(row)
        if any(-literal in values for literal in values):
            continue
        normalized = tuple(sorted(values, key=lambda value: (abs(value), value)))
        if normalized not in seen:
            rows.append(normalized)
            seen.add(normalized)
    return tuple(rows)


def variable_order(nvars: int, clauses: Sequence[Sequence[int]], mode: str) -> tuple[int, ...]:
    """Deterministic query-blind orders based only on generic CNF incidence.

    BFS uses a clause-incidence traversal rather than expanding a long clause
    into a quadratic clique.  Every isolated variable is retained.  Nothing in
    this function examines answers, query timings or SPECTRA's native index.
    """
    if mode == 'natural':
        return tuple(range(1, nvars + 1))
    if mode not in ('occurrence', 'bfs'):
        raise ValueError('order must be natural, occurrence or bfs')
    counts = Counter(abs(literal) for clause in clauses for literal in clause)
    roots = sorted(range(1, nvars + 1), key=lambda var: (-counts[var], var))
    if mode == 'occurrence':
        return tuple(roots)
    incidence: list[list[int]] = [[] for _ in range(nvars + 1)]
    for index, clause in enumerate(clauses):
        for literal in clause:
            incidence[abs(literal)].append(index)
    visited: set[int] = set()
    visited_clauses: set[int] = set()
    result: list[int] = []
    for root in roots:
        if root in visited:
            continue
        queue = deque([root])
        visited.add(root)
        while queue:
            variable = queue.popleft()
            result.append(variable)
            neighbors: set[int] = set()
            for index in incidence[variable]:
                if index in visited_clauses:
                    continue
                visited_clauses.add(index)
                neighbors.update(abs(literal) for literal in clauses[index])
            for neighbor in sorted(neighbors, key=lambda var: (-counts[var], var)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return tuple(result)


class CompiledCNF:
    """Compile once, condition each current request, extract a full Boolean model.

    SAT models are returned in original variable order, including unconstrained
    and conditioned-away variables.  ``None`` means this backend returned FALSE;
    it is not an independently checkable UNSAT certificate.  Lifecycle and native
    manager access are serialized.  All backend Function references stay private.
    """

    def __init__(self, nvars: int, clauses: Iterable[Iterable[int]], *,
                 backend: str = 'cudd', order: str = 'bfs', reordering: bool = True,
                 max_nodes: int = 2_000_000, memory_bytes: int = 512 << 20):
        if backend not in ('cudd', 'autoref'):
            raise ValueError('backend must be explicitly cudd or autoref')
        if type(reordering) is not bool:
            raise ValueError('reordering must be an exact bool')
        if type(max_nodes) is not int or max_nodes < 1:
            raise ValueError('max_nodes must be a positive exact integer')
        if type(memory_bytes) is not int or memory_bytes < (1 << 20):
            raise ValueError('memory_bytes must be an exact integer of at least 1 MiB')
        frozen = freeze_cnf(nvars, clauses)
        permutation = variable_order(nvars, frozen, order)
        try:
            module = importlib.import_module('dd.' + backend)
        except ImportError as exc:
            raise BackendUnavailable(
                f'dd.{backend} is required; no fallback was used. '
                'Install the pinned research dependency with this backend enabled.'
            ) from exc
        self._lock = threading.RLock()
        self._bdd = None
        self._root = None
        self.nvars = nvars
        self.backend = backend
        self.order = order
        self.reordering = reordering
        self._names = tuple(f'x{index}' for index in range(1, nvars + 1))
        self._care_vars = set(self._names)
        self._max_nodes = max_nodes
        self._clauses = len(frozen)
        try:
            self._bdd = (module.BDD(memory_estimate=memory_bytes)
                         if backend == 'cudd' else module.BDD())
            self._bdd.configure(reordering=reordering)
            self._bdd.declare(*(self._names[index - 1] for index in permutation))
            self._root = self._bdd.true
            self._guard()
            # No expression parser and no Python string-CNF parsing in the arm.
            # Small balanced blocks prevent holding every clause BDD alive.
            for offset in range(0, len(frozen), 64):
                block = []
                for row in frozen[offset:offset + 64]:
                    clause = self._bdd.false
                    for literal in row:
                        atom = self._bdd.var(self._names[abs(literal) - 1])
                        clause |= atom if literal > 0 else ~atom
                    block.append(clause)
                    self._guard()
                while len(block) > 1:
                    block = [block[index] & block[index + 1]
                             if index + 1 < len(block) else block[index]
                             for index in range(0, len(block), 2)]
                    self._guard()
                if block:
                    self._root &= block[0]
                block.clear()
                self._guard()
                if self._root == self._bdd.false:
                    break
        except BaseException:
            # Locals holding Functions are unwound with this frame.  The manager
            # remains referenced by those Functions until their final decref.
            self.close()
            raise

    def _guard(self) -> None:
        if self._bdd is not None and len(self._bdd) > self._max_nodes:
            raise CompilationLimit('BDD live-node soft limit exceeded')

    def model(self, assumptions: Iterable[int] = ()) -> tuple[bool, ...] | None:
        """Apply *all* assumptions to the immutable original compiled root.

        Repeated opposite literals are contradictory.  We deliberately do not
        mutate the root, so one query cannot constrain the next query.  A partial
        assignment obtained after conditioning must restore the fixed variables.
        """
        literals = tuple(assumptions)
        fixed: dict[str, bool] = {}
        contradictory = False
        for literal in literals:
            if (type(literal) is not int or literal == 0
                    or abs(literal) > self.nvars):
                raise ValueError('assumption must be a nonzero in-range exact integer')
            name = self._names[abs(literal) - 1]
            value = literal > 0
            if name in fixed and fixed[name] != value:
                contradictory = True
            fixed[name] = value
        with self._lock:
            if self._bdd is None:
                raise RuntimeError('compiled CNF is closed')
            if contradictory:
                return None
            conditioned = self._bdd.let(fixed, self._root) if fixed else self._root
            if conditioned == self._bdd.false:
                return None
            picked = self._bdd.pick(conditioned, care_vars=self._care_vars)
            if picked is None:
                raise AssertionError('non-FALSE BDD has no model')
            # ``{}`` is a valid model of TRUE, not a failure.  A conditioned-away
            # variable returned as a don't-care by pick must not override fixed.
            picked.update(fixed)
            result = tuple(bool(picked.get(name, False)) for name in self._names)
            self._guard()
            return result

    @property
    def info(self) -> dict:
        with self._lock:
            if self._bdd is None:
                raise RuntimeError('compiled CNF is closed')
            try:
                version = importlib.metadata.version('dd')
            except importlib.metadata.PackageNotFoundError:
                version = 'unknown'
            return {
                'backend': 'dd.' + self.backend,
                'dd_version': version,
                'variables': self.nvars,
                'normalized_clauses': self._clauses,
                'initial_order': self.order,
                'dynamic_reordering': self.reordering,
                'manager_live_nodes': len(self._bdd),
                'root_dag_nodes': self._root.dag_size,
                'statistics': self._bdd.statistics(),
                'limits_are_soft': True,
                'uses_spectra_quotient': False,
                'query_answer_cache': False,
            }

    def close(self) -> None:
        with self._lock:
            self._root = None
            self._bdd = None

    def __enter__(self) -> 'CompiledCNF':
        with self._lock:
            if self._bdd is None:
                raise RuntimeError('compiled CNF is closed')
        return self

    def __exit__(self, *_args) -> None:
        self.close()
