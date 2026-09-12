"""Optional exact CNF score cache; historical CNFRepairState remains unchanged.

For each non-tautological clause, a bit set records the variables whose literals
are true. A zero set contributes one make to every incident variable; a singleton
contributes one break to its sole member. Changes remove the old contribution and
add the new one. Duplicated literals are collapsed, tautologies are always true,
and original clause identities (including duplicate clauses) are preserved.

Single-variable queries are O(1). Patches touch variable incidences and traverse
clause variables only when a clause enters/leaves the unsatisfied set. This is an
exact classical data structure, NOT a predictor of eventual search success.
"""
from __future__ import annotations

from data.cnf import CNF


class CachedCNFRepairState:
    """Owned mutable state with immutable public observations and atomic patches.

    Work counters count canonical variable/clause incidences, not CPU operations
    or physical energy. Tautologies and duplicate literals do not generate work
    after construction. ``cache_literal_visits`` charges traversals used to
    maintain make counts; score queries report their own incidence visits.
    """

    def __init__(self, problem: CNF, witness: tuple[bool, ...]):
        if not isinstance(problem, CNF):
            raise TypeError("CNF required")
        problem.validate_witness(witness)
        self.problem = problem
        self._assignment = list(witness)
        self._variables: list[tuple[int, ...]] = []
        self._supports: list[int] = []
        self._occurrences: list[list[int]] = [[] for _ in range(problem.nvars)]
        self._make = [0] * problem.nvars
        self._break = [0] * problem.nvars
        self._unsatisfied: set[int] = set()
        self.flips = 0
        self.patches = 0
        self.literal_updates = 0
        self.feature_literal_visits = 0
        self.cache_literal_visits = 0
        self.score_queries = 0
        for index, clause in enumerate(problem.clauses):
            literals = set(clause)
            if any(-lit in literals for lit in literals):
                # Never enters the occurrence graph. Sentinel is not a variable.
                self._variables.append(())
                self._supports.append(-1)
                continue
            variables = tuple(sorted(abs(lit) - 1 for lit in literals))
            support = 0
            for lit in literals:
                variable = abs(lit) - 1
                self._occurrences[variable].append(index)
                if witness[variable] == (lit > 0):
                    support |= 1 << variable
            self._variables.append(variables)
            self._supports.append(support)
            self._contribute(index, support, 1)
            if support == 0:
                self._unsatisfied.add(index)
        # Initialization is inside full execution timing, not search-work counts.
        self.initialization_cache_visits = self.cache_literal_visits
        self.cache_literal_visits = 0

    @property
    def witness(self) -> tuple[bool, ...]:
        return tuple(self._assignment)

    @property
    def unsatisfied(self) -> tuple[int, ...]:
        return tuple(sorted(self._unsatisfied))

    @property
    def valid(self) -> bool:
        return not self._unsatisfied

    def _variable(self, variable: int) -> None:
        if type(variable) is not int or not 0 <= variable < self.problem.nvars:
            raise ValueError("variable must be a zero-based integer in range")

    def _patch(self, variables: tuple[int, ...]) -> None:
        if type(variables) is not tuple or not variables:
            raise ValueError("patch must be a nonempty tuple of distinct variables")
        for variable in variables:
            self._variable(variable)
        if len(set(variables)) != len(variables):
            raise ValueError("patch variables must be distinct")

    def _contribute(self, clause: int, support: int, sign: int) -> None:
        if support == 0:
            variables = self._variables[clause]
            for variable in variables:
                self._make[variable] += sign
            self.cache_literal_visits += len(variables)
        elif support > 0 and support & (support - 1) == 0:
            self._break[support.bit_length() - 1] += sign

    def make_break(self, variable: int) -> tuple[int, int]:
        self._variable(variable)
        self.score_queries += 1
        return self._make[variable], self._break[variable]

    def _toggles(self, variables: tuple[int, ...]) -> tuple[dict[int, int], int]:
        changes: dict[int, int] = {}
        visits = 0
        for variable in variables:
            bit = 1 << variable
            for clause in self._occurrences[variable]:
                changes[clause] = changes.get(clause, 0) ^ bit
                visits += 1
        return changes, visits

    def make_break_patch(self, variables: tuple[int, ...]) -> tuple[int, int]:
        """Score the simultaneous patch, never the sum of individual scores."""
        self._patch(variables)
        if len(variables) == 1:
            return self.make_break(variables[0])
        changes, visits = self._toggles(variables)
        made = broken = 0
        for clause, toggles in changes.items():
            old = self._supports[clause]
            new = old ^ toggles
            made += old == 0 and new != 0
            broken += old != 0 and new == 0
        self.feature_literal_visits += visits
        self.score_queries += 1
        return made, broken

    def flip(self, variable: int) -> None:
        self._variable(variable)
        # The common single-flip path needs no temporary clause-change map.
        # A single toggle cannot move directly from one nonzero singleton to
        # another, so these four transitions exhaust all cache changes.
        bit = 1 << variable
        occurrences = self._occurrences[variable]
        supports = self._supports
        made, broken = self._make, self._break
        for clause in occurrences:
            old = supports[clause]
            new = old ^ bit
            supports[clause] = new
            if old == 0:
                for member in self._variables[clause]:
                    made[member] -= 1
                self.cache_literal_visits += len(self._variables[clause])
                self._unsatisfied.remove(clause)
                broken[variable] += 1
            elif new == 0:
                broken[variable] -= 1
                for member in self._variables[clause]:
                    made[member] += 1
                self.cache_literal_visits += len(self._variables[clause])
                self._unsatisfied.add(clause)
            elif old & (old - 1) == 0:
                broken[old.bit_length() - 1] -= 1
            elif new & (new - 1) == 0:
                broken[new.bit_length() - 1] += 1
        self._assignment[variable] = not self._assignment[variable]
        self.literal_updates += len(occurrences)
        self.flips += 1
        self.patches += 1

    def flip_patch(self, variables: tuple[int, ...]) -> None:
        """Validate the entire patch before any mutation, then commit atomically.

        Intermediate sequential states are not scored or exposed. Python process
        failures/MemoryError are not transactional rollback guarantees.
        """
        self._patch(variables)
        self._apply(variables)

    def _apply(self, variables: tuple[int, ...]) -> None:
        changes, visits = self._toggles(variables)
        for clause, toggles in changes.items():
            old = self._supports[clause]
            new = old ^ toggles
            self._contribute(clause, old, -1)
            self._contribute(clause, new, 1)
            self._supports[clause] = new
            if new == 0:
                self._unsatisfied.add(clause)
            else:
                self._unsatisfied.discard(clause)
        for variable in variables:
            self._assignment[variable] = not self._assignment[variable]
        self.literal_updates += visits
        self.flips += len(variables)
        self.patches += 1
