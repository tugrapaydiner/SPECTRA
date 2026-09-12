"""Opt-in, trajectory-equivalent indexed focused search.

Immutable clause preparation is reusable; mutable assignments, count/XOR state,
break scores and ranked residual sets belong to one solve. The old full repair
API and the historical solve driver remain untouched. Warm calls explicitly
exclude preparation; cold calls include construction and destruction of it.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import time
from data.cnf import CNF, SplitMix64
from .ranked import RankedSet
from .search import SolveResult
from .state import CompactCNFRepairState


def _settings(problem: CNF, seed: int, max_flips: int) -> None:
    if not isinstance(problem, CNF):
        raise TypeError("CNF required")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if type(max_flips) is not int or max_flips < 0:
        raise ValueError("max_flips must be a nonnegative integer")


@dataclass(frozen=True, init=False)
class PreparedCNF:
    """Owned immutable formula index. Safe to reuse across independent searches.

    This is not a transformed logical problem: original clause IDs, duplicate
    clauses and original signed literals remain in ``problem`` for verification.
    Duplicate literals are collapsed only inside the execution index. Tautology
    sentinels never enter the occurrence graph or unsatisfied set.
    """
    problem: CNF
    literals: tuple[tuple[int, ...] | None, ...]
    variables: tuple[tuple[int, ...], ...]
    occurrences: tuple[tuple[int, ...], ...]
    weights: tuple[float, ...]

    def __init__(self, problem: CNF):
        if not isinstance(problem, CNF):
            raise TypeError("CNF required")
        literals, variables = [], []
        occurrences: list[list[int]] = [[] for _ in range(problem.nvars)]
        for index, clause in enumerate(problem.clauses):
            unique = set(clause)
            if any(-literal in unique for literal in unique):
                literals.append(None)
                variables.append(())
                continue
            ordered = tuple(sorted(unique, key=abs))
            literals.append(ordered)
            variables.append(tuple(abs(lit) - 1 for lit in ordered))
            for lit in ordered:
                occurrences[abs(lit)-1].append(index+1 if lit > 0 else -index-1)
        # A variable can break no more clauses than its non-tautological degree.
        # Each float uses the SAME scalar expression as the reference table.
        degree = max((len(o) for o in occurrences), default=0)
        object.__setattr__(self, "problem", problem)
        object.__setattr__(self, "literals", tuple(literals))
        object.__setattr__(self, "variables", tuple(variables))
        object.__setattr__(self, "occurrences", tuple(tuple(o) for o in occurrences))
        object.__setattr__(self, "weights", tuple((1+b)**-2.3 for b in range(degree+1)))

    def solve(self, *, seed: int = 0, max_flips: int = 1024) -> SolveResult:
        """Warm search. elapsed_ns excludes this immutable index's preparation."""
        _settings(self.problem, seed, max_flips)
        start = time.perf_counter_ns()
        rng = SplitMix64(seed)
        state = _BreakState(self, tuple(bool(rng.below(2)) for _ in range(self.problem.nvars)))
        values = _walk(self.problem, state, rng, max_flips, self.weights, self.variables)
        del state
        return _result(values, time.perf_counter_ns()-start, seed, max_flips)


class _BreakState:
    """Search-only cache; deliberately NOT a full make/break repair interface."""
    __slots__ = ("index", "assignment", "counts", "xors", "breaks", "residuals")

    def __init__(self, index: PreparedCNF, witness: tuple[bool, ...]):
        index.problem.validate_witness(witness)
        self.index = index
        self.assignment = list(witness)
        self.counts, self.xors = [], []
        self.breaks = [0] * index.problem.nvars
        residuals = []
        for i, clause in enumerate(index.literals):
            if clause is None:
                self.counts.append(-1)
                self.xors.append(0)
                continue
            count = support = 0
            for lit in clause:
                if witness[abs(lit)-1] == (lit > 0):
                    count += 1
                    support ^= abs(lit)
            self.counts.append(count)
            self.xors.append(support)
            if count == 0:
                residuals.append(i)
            elif count == 1:
                self.breaks[support-1] += 1
        self.residuals = RankedSet(len(index.problem.clauses), residuals)

    @property
    def witness(self) -> tuple[bool, ...]:
        return tuple(self.assignment)

    def break_count(self, variable: int) -> int:
        return self.breaks[variable]

    def flip(self, variable: int) -> None:
        # Only validated canonical variables from this index reach this private API.
        value = self.assignment[variable]
        counts, supports, breaks = self.counts, self.xors, self.breaks
        for occurrence in self.index.occurrences[variable]:
            clause = abs(occurrence)-1
            old_count, old_support = counts[clause], supports[clause]
            new_count = old_count + (-1 if value == (occurrence > 0) else 1)
            new_support = old_support ^ (variable+1)
            counts[clause], supports[clause] = new_count, new_support
            if old_count == 0:
                self.residuals.remove(clause)
                breaks[variable] += 1
            elif new_count == 0:
                breaks[variable] -= 1
                self.residuals.add(clause)
            elif old_count == 1:
                breaks[old_support-1] -= 1
            elif new_count == 1:
                breaks[new_support-1] += 1
        self.assignment[variable] = not value


class _RankedCompactState(CompactCNFRepairState):
    """Rank-only ablation: preserve the complete old make/break cache."""
    def __init__(self, problem, witness):
        super().__init__(problem, witness)
        self._unsatisfied = RankedSet(len(problem.clauses), self._unsatisfied)
        self.residuals = self._unsatisfied

    def break_count(self, variable: int) -> int:
        return self.make_break(variable)[1]


def _walk(problem, state, rng, max_flips, weights, variables=None):
    trace = hashlib.sha256()
    flips = queries = 0
    while flips < max_flips and state.residuals:
        clause_id = state.residuals.select(rng.below(len(state.residuals)))
        candidates = (variables[clause_id] if variables is not None else
                      sorted({abs(lit)-1 for lit in problem.clauses[clause_id]}))
        if not candidates:
            break
        scores = [weights[state.break_count(v)] for v in candidates]
        queries += len(candidates)
        threshold = (rng.below(2**53) / 2**53) * sum(scores)
        picked = candidates[-1]
        for variable, weight in zip(candidates, scores):
            threshold -= weight
            if threshold < 0:
                picked = variable
                break
        state.flip(picked)
        trace.update(str(picked).encode("ascii") + b";")
        flips += 1
    witness = state.witness
    violated = problem.violated(witness)
    if violated != tuple(state.residuals):
        raise AssertionError("indexed state differs from original formula")
    return witness, violated, flips, queries, trace.hexdigest()


def _result(values, elapsed_ns, seed, max_flips):
    witness, violated, flips, queries, digest = values
    return SolveResult("UNKNOWN" if violated else "SAT_VERIFIED", witness, violated,
                       flips, queries, digest, elapsed_ns, seed, max_flips)


def solve_indexed(problem: CNF, *, seed: int = 0, max_flips: int = 1024) -> SolveResult:
    """Cold indexed search; elapsed_ns includes preparation and its release.

    Imports and parsed CNF construction are outside, as in the historical solve.
    The same formula, seed and flip cap preserve every non-timing result field.
    There is no hard time deadline and UNKNOWN is not an UNSAT proof.
    """
    _settings(problem, seed, max_flips)
    start = time.perf_counter_ns()
    index = PreparedCNF(problem)
    rng = SplitMix64(seed)
    state = _BreakState(index, tuple(bool(rng.below(2)) for _ in range(problem.nvars)))
    values = _walk(problem, state, rng, max_flips, index.weights, index.variables)
    del state
    del index
    return _result(values, time.perf_counter_ns()-start, seed, max_flips)


def _solve_ranked(problem: CNF, *, seed: int, max_flips: int) -> SolveResult:
    """Internal rank-only ablation; cold timing and historical cache semantics."""
    _settings(problem, seed, max_flips)
    start = time.perf_counter_ns()
    rng = SplitMix64(seed)
    state = _RankedCompactState(problem, tuple(bool(rng.below(2)) for _ in range(problem.nvars)))
    weights = [(1+b)**-2.3 for b in range(len(problem.clauses)+1)]
    values = _walk(problem, state, rng, max_flips, weights)
    del state
    del weights
    return _result(values, time.perf_counter_ns()-start, seed, max_flips)
