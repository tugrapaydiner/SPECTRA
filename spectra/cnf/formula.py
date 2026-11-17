"""Solver-independent Boolean CNF witnesses and deterministic development inputs.

Identity hashes preserve either exact solver order or clause/literal order-normalized
identity. Neither claims complete logical, variable-renaming or sign equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import comb


@dataclass(frozen=True)
class CNF:
    nvars: int
    clauses: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.nvars) is not int or self.nvars < 0:
            raise ValueError("nvars must be a nonnegative integer")
        if type(self.clauses) is not tuple or any(type(c) is not tuple for c in self.clauses):
            raise TypeError("clauses must be immutable tuples")
        if any(type(lit) is not int or lit == 0 or abs(lit) > self.nvars
               for clause in self.clauses for lit in clause):
            raise ValueError("literals must be nonzero integers within the variable range")

    def validate_witness(self, witness: tuple[bool, ...]) -> None:
        if (type(witness) is not tuple or len(witness) != self.nvars
                or any(type(value) is not bool for value in witness)):
            raise ValueError("a complete immutable Boolean assignment is required")

    def violated(self, witness: tuple[bool, ...]) -> tuple[int, ...]:
        self.validate_witness(witness)
        return tuple(i for i, clause in enumerate(self.clauses)
                     if not any(witness[abs(lit)-1] == (lit > 0) for lit in clause))

    def satisfied(self, witness: tuple[bool, ...]) -> bool:
        self.validate_witness(witness)
        return all(any(witness[abs(lit)-1] == (lit > 0) for lit in clause)
                   for clause in self.clauses)

    def record(self) -> dict:
        return {"nvars": self.nvars, "clauses": [list(c) for c in self.clauses]}

    @classmethod
    def from_record(cls, record: dict) -> CNF:
        if (type(record) is not dict or set(record) != {"nvars", "clauses"}
                or type(record["clauses"]) is not list
                or any(type(c) is not list for c in record["clauses"])):
            raise ValueError("invalid CNF record")
        return cls(record["nvars"], tuple(tuple(c) for c in record["clauses"]))

    def sha256(self, *, normalize_order: bool = False) -> str:
        if type(normalize_order) is not bool:
            raise TypeError("normalize_order must be a Boolean")
        clauses = self.clauses
        if normalize_order:
            # Preserve the declared variable range; only deduplicate and reorder.
            clauses = tuple(sorted(set(tuple(sorted(set(c))) for c in clauses)))
        payload = json.dumps([self.nvars, clauses], separators=(",", ":")).encode("ascii")
        return hashlib.sha256(b"spectra.cnf.v1\0" + payload).hexdigest()


class SplitMix64:
    """Explicit generator avoids changes in library sampling implementations."""

    def __init__(self, seed: int):
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        self.state = seed

    def below(self, stop: int) -> int:
        if type(stop) is not int or not 1 <= stop <= 2**64:
            raise ValueError("invalid sampling range")
        threshold = 2**64 % stop
        while True:
            self.state = (self.state + 0x9E3779B97F4A7C15) & (2**64-1)
            z = self.state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & (2**64-1)
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & (2**64-1)
            z ^= z >> 31
            if z >= threshold:
                return z % stop


def random_3sat(nvars: int, nclauses: int, seed: int, *, planted: bool = False
                ) -> tuple[CNF, tuple[bool, ...] | None]:
    """Generate distinct non-tautological clauses without solver-based filtering.

    Planting is explicitly a different distribution, not a source of hard cases.
    Uniform formulas can be SAT, UNSAT or unresolved within the solver budget.
    """
    if type(nvars) is not int or not 3 <= nvars <= 2**64 or type(planted) is not bool:
        raise ValueError("at least three variables and an explicit planting flag required")
    maximum = comb(nvars, 3) * (7 if planted else 8)
    if type(nclauses) is not int or not 0 <= nclauses <= maximum:
        raise ValueError("clause count exceeds the distinct-clause inventory")
    rng = SplitMix64(seed)
    witness = tuple(bool(rng.below(2)) for _ in range(nvars)) if planted else None
    clauses, seen = [], set()
    while len(clauses) < nclauses:
        variables = []
        while len(variables) < 3:
            v = 1 + rng.below(nvars)
            if v not in variables:
                variables.append(v)
        clause = tuple(sorted((v if rng.below(2) else -v for v in variables), key=abs))
        if clause in seen or (witness is not None and
                not any(witness[abs(lit)-1] == (lit > 0) for lit in clause)):
            continue
        seen.add(clause)
        clauses.append(clause)
    result = CNF(nvars, tuple(clauses))
    if witness is not None and not result.satisfied(witness):
        raise AssertionError("invalid planted construction")
    return result, witness
