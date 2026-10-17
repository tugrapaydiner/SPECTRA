"""Pinned strong-competitor adapters for the development champion gauntlet.

These wrappers deliberately preserve the original-address query contract.  Every
SAT model is lifted to a complete original-vertex colouring and checked by a
separately owned native observer.  Generic-solver UNSAT decisions do not acquire
SPECTRA's portable SBP1 proof contract and must be reported as such.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Iterable

from .binary_list import BinaryListProblem, CompiledAnswer


def _library(path: Path | str, *, name: str) -> ctypes.CDLL:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f'{name} shared library not found: {candidate}')
    return ctypes.CDLL(str(candidate.resolve()))


class _CheckedSession:
    """Own canonical original inputs and a separately constructed checker."""

    def __init__(self, problem: BinaryListProblem, runtime, max_bytes: int):
        self.problem = BinaryListProblem.create(
            problem.n, problem.palette, problem.edges, problem.masks)
        self._choices = self.problem.colors()
        self._checker = runtime.checker(
            problem.n, problem.palette, problem.edges, problem.masks, max_bytes)

    def _labels(self, values: Iterable[bool], query) -> bytes:
        snapshot = tuple(values)
        if len(snapshot) != self.problem.n or any(type(v) is not bool for v in snapshot):
            raise AssertionError('competitor returned a partial or non-Boolean model')
        labels = bytes(self._choices[v][int(value)] for v, value in enumerate(snapshot))
        if not self._checker.check(labels, query):
            raise AssertionError('competitor SAT model failed the original-input observer')
        return labels

    def _close_checker(self) -> None:
        checker, self._checker = self._checker, None
        if checker is not None:
            checker.close()


class CaDiCaL3Session(_CheckedSession):
    """Persistent CaDiCaL 3 IPASIR session with arbitrary future assumptions.

    All original variables are frozen before the first solve.  This is required by
    the service contract: any original vertex may be restricted by a later query.
    Learned clauses are retained across calls.  UNSAT is an exact solver decision,
    but this adapter does not export a portable contradiction proof.
    """

    def __init__(self, problem, runtime, library: Path | str, max_bytes: int):
        super().__init__(problem, runtime, max_bytes)
        self._lib = _library(library, name='CaDiCaL 3')
        lib = self._lib
        lib.ccadical_signature.restype = ctypes.c_char_p
        lib.ccadical_init.restype = ctypes.c_void_p
        lib.ccadical_release.argtypes = (ctypes.c_void_p,)
        lib.ccadical_add.argtypes = (ctypes.c_void_p, ctypes.c_int)
        lib.ccadical_assume.argtypes = (ctypes.c_void_p, ctypes.c_int)
        lib.ccadical_solve.argtypes = (ctypes.c_void_p,)
        lib.ccadical_solve.restype = ctypes.c_int
        lib.ccadical_val.argtypes = (ctypes.c_void_p, ctypes.c_int)
        lib.ccadical_val.restype = ctypes.c_int
        lib.ccadical_freeze.argtypes = (ctypes.c_void_p, ctypes.c_int)
        lib.ccadical_set_option.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int)
        self._solver = lib.ccadical_init()
        if not self._solver:
            self._close_checker()
            raise MemoryError('CaDiCaL 3 initialization failed')
        try:
            lib.ccadical_set_option(self._solver, b'quiet', 1)
            for variable in range(1, self.problem.n + 1):
                lib.ccadical_freeze(self._solver, variable)
            for clause in self.problem.cnf():
                for literal in clause:
                    lib.ccadical_add(self._solver, literal)
                lib.ccadical_add(self._solver, 0)
        except BaseException:
            self.close()
            raise
        self._signature = (lib.ccadical_signature() or b'unknown').decode('utf-8', 'replace')
        self._solves = 0

    def solve(self, restrictions=()) -> CompiledAnswer:
        if self._solver is None:
            raise RuntimeError('CaDiCaL 3 session is closed')
        query = self.problem.freeze_query(restrictions)
        assumptions = self.problem.assumptions(query)
        if assumptions is None:
            return CompiledAnswer('UNSAT', b'')
        for literal in assumptions:
            self._lib.ccadical_assume(self._solver, literal)
        status = self._lib.ccadical_solve(self._solver)
        self._solves += 1
        if status == 20:
            return CompiledAnswer('UNSAT', b'')
        if status != 10:
            raise RuntimeError(f'CaDiCaL 3 returned nonterminal status {status}')
        values = []
        for variable in range(1, self.problem.n + 1):
            value = self._lib.ccadical_val(self._solver, variable)
            if value == 0:
                raise AssertionError('CaDiCaL 3 returned a partial original model')
            values.append(value > 0)
        return CompiledAnswer('SAT_VERIFIED', self._labels(values, query))

    @property
    def info(self) -> dict:
        return dict(backend='cadical3-ctypes', signature=self._signature,
                    persistent_learned_state=True, original_variables_frozen=True,
                    rebuild_per_query=False, solves=self._solves,
                    unsat_certificate=False)

    def close(self) -> None:
        solver, self._solver = getattr(self, '_solver', None), None
        try:
            if solver:
                self._lib.ccadical_release(solver)
        finally:
            self._close_checker()


class Kissat4RebuildSession(_CheckedSession):
    """Kissat 4 one-shot control, rebuilt from the original CNF for every query.

    Kissat intentionally exposes no incremental-assumption interface.  Keeping it
    in a separately labelled rebuild track prevents a misleading comparison with
    compile-once/persistent services while still measuring the current solver core.
    """

    def __init__(self, problem, runtime, library: Path | str, max_bytes: int):
        super().__init__(problem, runtime, max_bytes)
        self._lib = _library(library, name='Kissat 4')
        lib = self._lib
        lib.kissat_signature.restype = ctypes.c_char_p
        lib.kissat_version.restype = ctypes.c_char_p
        lib.kissat_init.restype = ctypes.c_void_p
        lib.kissat_release.argtypes = (ctypes.c_void_p,)
        lib.kissat_add.argtypes = (ctypes.c_void_p, ctypes.c_int)
        lib.kissat_solve.argtypes = (ctypes.c_void_p,)
        lib.kissat_solve.restype = ctypes.c_int
        lib.kissat_value.argtypes = (ctypes.c_void_p, ctypes.c_int)
        lib.kissat_value.restype = ctypes.c_int
        lib.kissat_reserve.argtypes = (ctypes.c_void_p, ctypes.c_int)
        lib.kissat_set_option.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int)
        self._clauses = self.problem.cnf()
        self._signature = (lib.kissat_signature() or b'unknown').decode('utf-8', 'replace')
        self._version = (lib.kissat_version() or b'unknown').decode('utf-8', 'replace')
        self._solves = 0

    def solve(self, restrictions=()) -> CompiledAnswer:
        if self._checker is None:
            raise RuntimeError('Kissat 4 session is closed')
        query = self.problem.freeze_query(restrictions)
        assumptions = self.problem.assumptions(query)
        if assumptions is None:
            return CompiledAnswer('UNSAT', b'')
        solver = self._lib.kissat_init()
        if not solver:
            raise MemoryError('Kissat 4 initialization failed')
        try:
            self._lib.kissat_set_option(solver, b'quiet', 1)
            self._lib.kissat_reserve(solver, self.problem.n)
            for clause in self._clauses:
                for literal in clause:
                    self._lib.kissat_add(solver, literal)
                self._lib.kissat_add(solver, 0)
            for literal in assumptions:
                self._lib.kissat_add(solver, literal)
                self._lib.kissat_add(solver, 0)
            status = self._lib.kissat_solve(solver)
            self._solves += 1
            if status == 20:
                return CompiledAnswer('UNSAT', b'')
            if status != 10:
                raise RuntimeError(f'Kissat 4 returned nonterminal status {status}')
            values = []
            for variable in range(1, self.problem.n + 1):
                value = self._lib.kissat_value(solver, variable)
                if value == 0:
                    raise AssertionError('Kissat 4 returned a partial original model')
                values.append(value > 0)
            return CompiledAnswer('SAT_VERIFIED', self._labels(values, query))
        finally:
            self._lib.kissat_release(solver)

    @property
    def info(self) -> dict:
        return dict(backend='kissat4-ctypes', signature=self._signature,
                    version=self._version, persistent_learned_state=False,
                    rebuild_per_query=True, solves=self._solves,
                    unsat_certificate=False)

    def close(self) -> None:
        self._close_checker()


class ORToolsCPSATSession(_CheckedSession):
    """Single-worker OR-Tools CP-SAT feasibility control using assumptions.

    The immutable Boolean model is retained, but each ``solve`` call starts a new
    CP-SAT search.  The adapter disables interleaved search and uses one worker so
    timing and assumption semantics remain deterministic and directly auditable.
    """

    def __init__(self, problem, runtime, max_bytes: int):
        super().__init__(problem, runtime, max_bytes)
        from ortools.sat.python import cp_model
        self._cp_model = cp_model
        self._model = cp_model.CpModel()
        self._variables = tuple(self._model.new_bool_var(f'x{v}')
                                for v in range(self.problem.n))
        for clause in self.problem.cnf():
            terms = [self._variables[abs(lit)-1] if lit > 0
                     else self._variables[abs(lit)-1].Not() for lit in clause]
            self._model.add_bool_or(terms)
        self._solver = cp_model.CpSolver()
        self._solver.parameters.num_workers = 1
        self._solver.parameters.random_seed = 0
        self._solver.parameters.log_search_progress = False
        # Explicitly avoid the assumption/interleave interaction reported upstream.
        if hasattr(self._solver.parameters, 'interleave_search'):
            self._solver.parameters.interleave_search = False
        self._solves = 0

    def solve(self, restrictions=()) -> CompiledAnswer:
        if self._model is None:
            raise RuntimeError('OR-Tools session is closed')
        query = self.problem.freeze_query(restrictions)
        assumptions = self.problem.assumptions(query)
        if assumptions is None:
            return CompiledAnswer('UNSAT', b'')
        self._model.clear_assumptions()
        self._model.add_assumptions([
            self._variables[abs(lit)-1] if lit > 0
            else self._variables[abs(lit)-1].Not() for lit in assumptions])
        status = self._solver.solve(self._model)
        self._solves += 1
        if status == self._cp_model.INFEASIBLE:
            return CompiledAnswer('UNSAT', b'')
        if status not in (self._cp_model.FEASIBLE, self._cp_model.OPTIMAL):
            raise RuntimeError(f'OR-Tools CP-SAT returned nonterminal status {status}')
        values = [bool(self._solver.value(variable)) for variable in self._variables]
        return CompiledAnswer('SAT_VERIFIED', self._labels(values, query))

    @property
    def info(self) -> dict:
        try:
            import ortools
            version = ortools.__version__
        except Exception:
            version = 'unknown'
        return dict(backend='ortools-cp-sat', version=version,
                    workers=1, interleave_search=False,
                    persistent_model=True, persistent_learned_state=False,
                    rebuild_per_query=False, solves=self._solves,
                    unsat_certificate=False)

    def close(self) -> None:
        self._model = None
        self._solver = None
        self._variables = ()
        self._close_checker()
