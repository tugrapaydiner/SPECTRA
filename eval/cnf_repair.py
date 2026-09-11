"""Exact failure residuals for future SAT repair policies, not a learned solver.

Flips touch only occurrences of that variable. Make/break counts remain exact
with repeated literals, tautologies, empty clauses and unused variables.
"""
from __future__ import annotations

from data.cnf import CNF


class CNFRepairState:
    def __init__(self, problem: CNF, witness: tuple[bool, ...]):
        if not isinstance(problem, CNF):
            raise TypeError("CNF required")
        problem.validate_witness(witness)
        self.problem = problem
        self._assignment = list(witness)
        self._counts = [sum(witness[abs(lit)-1] == (lit > 0) for lit in clause)
                        for clause in problem.clauses]
        self._unsatisfied = {i for i, count in enumerate(self._counts) if count == 0}
        self._occurrences: list[list[tuple[int, bool]]] = [[] for _ in range(problem.nvars)]
        for i, clause in enumerate(problem.clauses):
            for lit in clause:
                self._occurrences[abs(lit)-1].append((i, lit > 0))
        self.flips = 0
        self.literal_updates = 0
        self.feature_literal_visits = 0

    @property
    def witness(self) -> tuple[bool, ...]:
        return tuple(self._assignment)

    @property
    def unsatisfied(self) -> tuple[int, ...]:
        return tuple(sorted(self._unsatisfied))

    @property
    def valid(self) -> bool:
        return not self._unsatisfied

    def _changes(self, variable: int) -> dict[int, int]:
        if type(variable) is not int or not 0 <= variable < self.problem.nvars:
            raise ValueError("variable must be a zero-based integer in range")
        value = self._assignment[variable]
        changes: dict[int, int] = {}
        for clause, positive in self._occurrences[variable]:
            changes[clause] = changes.get(clause, 0) + (-1 if value == positive else 1)
        return changes

    def make_break(self, variable: int) -> tuple[int, int]:
        """Clauses made satisfied / broken by a flip; charge feature work separately.

        U(a xor e_v) = U(a) - make_v(a) + break_v(a).
        These exact features do not predict how many later flips solve the task.
        """
        changes = self._changes(variable)
        self.feature_literal_visits += len(self._occurrences[variable])
        made = sum(self._counts[c] == 0 and self._counts[c]+d > 0 for c, d in changes.items())
        broken = sum(self._counts[c] > 0 and self._counts[c]+d == 0 for c, d in changes.items())
        return made, broken

    def flip(self, variable: int) -> None:
        changes = self._changes(variable)
        for clause, delta in changes.items():
            self._counts[clause] += delta
            if self._counts[clause] == 0:
                self._unsatisfied.add(clause)
            else:
                self._unsatisfied.discard(clause)
        self._assignment[variable] = not self._assignment[variable]
        self.flips += 1
        self.literal_updates += len(self._occurrences[variable])
