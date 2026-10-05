"""Opt-in bounded classical DPLL with watched literals and an iterative trail.

No clause learning, neural weights, UNSAT certificate or hard time limit.
Existing search implementations and defaults remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import time

from data.cnf import CNF


def _index(literal):
    return 2*(abs(literal)-1)+int(literal < 0)


@dataclass(frozen=True)
class DPLLResult:
    status: str
    witness: tuple[bool, ...]
    unsatisfied: tuple[int, ...]
    reason: str
    decisions: int
    propagations: int
    conflicts: int
    backtracks: int
    max_decisions: int
    path_sha256: str
    elapsed_ns: int

    def record(self):
        return {**asdict(self), 'witness': list(self.witness), 'unsatisfied': list(self.unsatisfied),
                'schema': 'spectra.cnf.dpll.v1', 'algorithm': 'watched_dpll', 'learned': False}


class _Search:
    def __init__(self, problem):
        self.values = [-1]*problem.nvars
        self.clauses, self.positions = [], []
        self.watches = [[] for _ in range(2*problem.nvars)]
        self.trail, self.head = [], 0
        self.decisions = self.propagations = self.conflicts = self.backtracks = 0
        self.digest = hashlib.sha256(b'spectra.dpll.v1\0')
        self.contradiction = False
        for original in problem.clauses:
            # Binary clauses dominate pairwise encodings. Avoid allocating a dict
            # and set for them while preserving literal order and normalization.
            if len(original) == 2:
                a, b = original
                if a == -b:
                    continue
                clause = (a,) if a == b else original
            elif len(original) < 2:
                clause = original
            else:
                clause = tuple(dict.fromkeys(original))
                members = set(clause)
                if any(-lit in members for lit in clause):
                    continue
            if not clause:
                self.contradiction = True
                continue
            cid = len(self.clauses)
            self.clauses.append(clause)
            second = int(len(clause) > 1)
            self.positions.append([0, second])
            self.watches[_index(clause[0])].append(cid)
            if second:
                self.watches[_index(clause[second])].append(cid)
            elif not self.assign(clause[0]):
                self.contradiction = True
        self.positive = [c for c in self.clauses if all(lit > 0 for lit in c)]

    def value(self, literal):
        value = self.values[abs(literal)-1]
        return -1 if value == -1 else value if literal > 0 else 1-value

    def assign(self, literal):
        index, value = abs(literal)-1, int(literal > 0)
        if self.values[index] != -1:
            return self.values[index] == value
        self.values[index] = value
        self.trail.append(literal)
        self.digest.update(str(literal).encode('ascii')+b';')
        return True

    def propagate(self):
        while self.head < len(self.trail):
            false_literal = -self.trail[self.head]
            self.head += 1
            self.propagations += 1
            watching, i = self.watches[_index(false_literal)], 0
            while i < len(watching):
                cid = watching[i]
                clause, positions = self.clauses[cid], self.positions[cid]
                slot = 0 if clause[positions[0]] == false_literal else 1
                other = positions[1-slot]
                if self.value(clause[other]) == 1:
                    i += 1
                    continue
                replacement = next((j for j, lit in enumerate(clause)
                                    if j != other and j != positions[slot] and self.value(lit) != 0), None)
                if replacement is not None:
                    positions[slot] = replacement
                    watching[i] = watching[-1]
                    watching.pop()
                    self.watches[_index(clause[replacement])].append(cid)
                elif self.value(clause[other]) == 0 or not self.assign(clause[other]):
                    self.conflicts += 1
                    return False
                else:
                    i += 1
        return True

    def choose(self):
        # Positive clauses expose small remaining domains in one-hot encodings.
        # This is a fixed syntactic heuristic, not a learned Sudoku classifier.
        best_count, best_literal = len(self.values)+1, None
        for clause in self.positive:
            count, first = 0, None
            for lit in clause:
                value = self.values[lit-1]
                if value == 1:
                    break
                if value == -1:
                    count += 1
                    if first is None:
                        first = lit
            else:
                if 0 < count < best_count:
                    best_count, best_literal = count, first
                    # Propagation is at a fixed point: no unresolved unit exists.
                    if count == 2:
                        return first
        if best_literal is not None:
            return best_literal
        best = None
        for clause in self.clauses:
            free, satisfied = [], False
            for lit in clause:
                value = self.value(lit)
                if value == 1:
                    satisfied = True
                    break
                if value == -1:
                    free.append(lit)
            if not satisfied and (best is None or len(free) < len(best)):
                best = free
        return best[0] if best else None

    def undo(self, cutoff):
        for literal in self.trail[cutoff:]:
            self.values[abs(literal)-1] = -1
        del self.trail[cutoff:]
        self.head = len(self.trail)
        self.backtracks += 1
        self.digest.update(b'B'+str(cutoff).encode('ascii')+b';')

    def run(self, limit):
        if self.contradiction:
            self.conflicts += 1
            return 'exhausted'
        stack = []
        while True:
            if not self.propagate():
                while stack:
                    cutoff, literal, retried = stack.pop()
                    self.undo(cutoff)
                    if not retried:
                        if self.decisions == limit:
                            return 'budget'
                        stack.append((cutoff, literal, True))
                        self.decisions += 1
                        self.assign(-literal)
                        break
                else:
                    return 'exhausted'
                continue
            literal = self.choose()
            if literal is None:
                return 'satisfied'
            if self.decisions == limit:
                return 'budget'
            stack.append((len(self.trail), literal, False))
            self.decisions += 1
            self.assign(literal)


def solve_dpll(problem: CNF, *, max_decisions: int = 2048) -> DPLLResult:
    """Bound branch attempts (both polarities); propagation adds input-dependent work.

    UNKNOWN on budget exhaustion or exhausted unsatisfiable search: no independently
    checkable UNSAT certificate is emitted. max_decisions is not a flip/time cap.
    """
    if not isinstance(problem, CNF):
        raise TypeError('CNF required')
    if type(max_decisions) is not int or max_decisions < 0:
        raise ValueError('max_decisions must be a nonnegative integer')
    start = time.perf_counter_ns()
    state = _Search(problem)
    reason = state.run(max_decisions)
    witness = tuple(value == 1 for value in state.values)
    unsatisfied = problem.violated(witness)
    if reason == 'satisfied' and unsatisfied:
        raise AssertionError('DPLL witness differs from original formula')
    counters = state.decisions, state.propagations, state.conflicts, state.backtracks
    digest = state.digest.hexdigest()
    del state
    return DPLLResult('UNKNOWN' if unsatisfied else 'SAT_VERIFIED', witness, unsatisfied,
                      reason, *counters, max_decisions, digest, time.perf_counter_ns()-start)
