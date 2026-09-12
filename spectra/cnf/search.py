"""Bounded focused classical search with complete original-input verification.

No neural model is used. No UNSAT conclusion or hard time deadline is claimed.
Unknown capped executions are returned explicitly, never as successful decisions.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import time
from data.cnf import CNF, SplitMix64
from .state import CompactCNFRepairState


@dataclass(frozen=True)
class SolveResult:
    status: str
    witness: tuple[bool, ...]
    unsatisfied: tuple[int, ...]
    flips: int
    queries: int
    path_sha256: str
    elapsed_ns: int
    seed: int
    max_flips: int

    def record(self) -> dict:
        return {"schema": "spectra.cnf.solve.v1", "status": self.status,
                "witness": list(self.witness), "unsatisfied": list(self.unsatisfied),
                "flips": self.flips, "queries": self.queries,
                "path_sha256": self.path_sha256, "elapsed_ns": self.elapsed_ns,
                "seed": self.seed, "max_flips": self.max_flips,
                "algorithm": "focused_classical_local_search", "learned": False}


def solve(problem: CNF, *, seed: int = 0, max_flips: int = 1024) -> SolveResult:
    """Try to find a witness; same seed/input/cap gives the same non-timing result.

    Timing includes initial assignment, state construction, weight-table creation,
    every search step, original-formula validation and state release. Parsed input
    and module imports are outside. An empty clause stops as UNKNOWN (no UNSAT
    proof interface). The returned witness on UNKNOWN is only a candidate.
    """
    return _execute(problem, seed=seed, max_flips=max_flips, state_class=CompactCNFRepairState)


def _execute(problem: CNF, *, seed: int, max_flips: int, state_class) -> SolveResult:
    if not isinstance(problem, CNF):
        raise TypeError("CNF required")
    if type(max_flips) is not int or max_flips < 0:
        raise ValueError("max_flips must be a nonnegative integer")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    start = time.perf_counter_ns()
    rng = SplitMix64(seed)
    state = state_class(problem, tuple(bool(rng.below(2)) for _ in range(problem.nvars)))
    weights = [(1 + b)**-2.3 for b in range(len(problem.clauses) + 1)]
    trace = hashlib.sha256()
    flips = queries = 0
    while flips < max_flips and not state.valid:
        violated = state.unsatisfied
        clause = problem.clauses[violated[rng.below(len(violated))]]
        candidates = sorted({abs(lit) - 1 for lit in clause})
        if not candidates:
            break
        scores = [weights[state.make_break(v)[1]] for v in candidates]
        queries += len(candidates)
        threshold = (rng.below(2**53) / 2**53) * sum(scores)
        picked = candidates[-1]
        for v, weight in zip(candidates, scores):
            threshold -= weight
            if threshold < 0:
                picked = v
                break
        state.flip(picked)
        trace.update(str(picked).encode("ascii") + b";")
        flips += 1
    witness = state.witness
    # Independently recompute against ORIGINAL signed literals, not cached counts.
    unsatisfied = problem.violated(witness)
    if unsatisfied != state.unsatisfied or (not unsatisfied) != state.valid:
        raise AssertionError("incremental state differs from original formula")
    del state
    del weights
    elapsed = time.perf_counter_ns() - start
    return SolveResult("SAT_VERIFIED" if not unsatisfied else "UNKNOWN", witness,
                       unsatisfied, flips, queries, trace.hexdigest(), elapsed, seed, max_flips)
