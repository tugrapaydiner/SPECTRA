"""Experimental classical move policies with constant-time residual sampling.

The historical indexed solver remains unchanged. Clause-pool order changes the
seeded trajectory, even with identical polynomial variable-selection weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

from data.cnf import CNF, SplitMix64
from .indexed import PreparedCNF, _BreakState, _settings
from .search import SolveResult


class _DenseResiduals:
    """Private swap-delete set; iteration is deliberately not sorted."""
    __slots__ = ('items', 'positions')

    def __init__(self, size, values):
        self.items = list(values)
        self.positions = [-1]*size
        for i, value in enumerate(self.items):
            self.positions[value] = i

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def select(self, rank):
        return self.items[rank]

    def add(self, value):
        if self.positions[value] == -1:
            self.positions[value] = len(self.items)
            self.items.append(value)

    def remove(self, value):
        position = self.positions[value]
        if position < 0:
            raise KeyError(value)
        last = self.items.pop()
        if position < len(self.items):
            self.items[position] = last
            self.positions[last] = position
        self.positions[value] = -1


@dataclass(frozen=True)
class FocusedResult(SolveResult):
    policy: str
    restarts: int
    restart_interval: int

    def record(self):
        return {**super().record(), 'algorithm': 'dense_focused_search',
                'policy': self.policy, 'restarts': self.restarts,
                'restart_interval': self.restart_interval}


def _state(index, rng):
    state = _BreakState(index, tuple(bool(rng.below(2)) for _ in range(index.problem.nvars)))
    state.residuals = _DenseResiduals(len(index.problem.clauses), state.residuals)
    return state


def _pick(candidates, breaks, weights, rng, policy, previous=None, ages=None):
    if policy == 'novelty_break':
        ordered = sorted(candidates, key=lambda v: (breaks[v], ages[v], v))
        best = ordered[0]
        if breaks[best] != 0 and len(ordered) > 1 and ages[best] == max(ages[v] for v in candidates) and rng.below(2):
            return ordered[1]
        return best
    if policy in ('freebie', 'minbreak', 'anti_reverse'):
        minimum = min(breaks[v] for v in candidates)
        if minimum == 0 or policy == 'minbreak':
            if minimum != 0 and rng.below(2):
                return candidates[rng.below(len(candidates))]
            best = [v for v in candidates if breaks[v] == minimum]
            return best[rng.below(len(best))]
    if policy == 'anti_reverse' and previous in candidates and len(candidates) > 1:
        candidates = tuple(v for v in candidates if v != previous)
    scores = [weights[breaks[v]] for v in candidates]
    threshold = (rng.below(2**53)/2**53)*sum(scores)
    for variable, weight in zip(candidates, scores):
        threshold -= weight
        if threshold < 0:
            return variable
    return candidates[-1]


def solve_focused(problem: CNF, *, seed: int = 0, max_flips: int = 1024,
                  policy: str = 'novelty_break', restart_interval: int = 0) -> FocusedResult:
    """Bound total flips across all restarts; UNKNOWN is not an UNSAT proof.

    Complete timing includes new index/state setup, verification and disposal.
    There is no elapsed-time cap. Nondefault policies are explicit research controls.
    """
    _settings(problem, seed, max_flips)
    if policy not in ('poly', 'freebie', 'minbreak', 'sharp', 'anti_reverse', 'novelty_break'):
        raise ValueError('unknown focused search policy')
    if type(restart_interval) is not int or restart_interval < 0:
        raise ValueError('restart_interval must be a nonnegative integer')
    start = time.perf_counter_ns()
    index, rng = PreparedCNF(problem), SplitMix64(seed)
    weights = (tuple((0.1+b)**-2.3 for b in range(len(index.weights)))
               if policy == 'sharp' else index.weights)
    state = _state(index, rng)
    ages = [-1]*problem.nvars if policy == 'novelty_break' else None
    previous = None
    trace = hashlib.sha256(b'spectra.focused.v1\0')
    flips = queries = restarts = 0
    # Empty clauses cannot be repaired; do not spin until one is randomly drawn.
    impossible = any(c == () for c in index.literals)
    while flips < max_flips and state.residuals and not impossible:
        if restart_interval and flips and flips % restart_interval == 0:
            state = _state(index, rng)
            restarts += 1
            previous = None
            if ages is not None:
                ages = [-1]*problem.nvars
            trace.update(b'R;')
            if not state.residuals:
                break
        clause_id = state.residuals.select(rng.below(len(state.residuals)))
        candidates = index.variables[clause_id]
        variable = _pick(candidates, state.breaks, weights, rng, policy, previous, ages)
        queries += len(candidates)
        state.flip(variable)
        trace.update(str(variable).encode('ascii')+b';')
        flips += 1
        previous = variable
        if ages is not None:
            ages[variable] = flips
    witness = state.witness
    violated = problem.violated(witness)
    if violated != tuple(sorted(state.residuals)):
        raise AssertionError('focused state differs from original formula')
    del state, index, weights
    return FocusedResult('UNKNOWN' if violated else 'SAT_VERIFIED', witness, violated,
                         flips, queries, trace.hexdigest(), time.perf_counter_ns()-start,
                         seed, max_flips, policy, restarts, restart_interval)
