"""Exact clause-local count/XOR state; optional, not a learned algorithm.

The historical cache uses a bit for each GLOBAL variable index in every clause.
Here count[c] is the number of true literals, xor[c] the XOR of their one-based
variable IDs. Only when count[c] == 1 is xor[c] interpreted as a sole variable.
Counts distinguish zero from nonempty supports whose XOR happens to be zero.
Python integer sizes now depend on log(nvars), not nvars, per clause. This trades
an extra count array for avoiding high-index sparse bitsets. Old source stays
unchanged for historical replay. Common validation/public observations are reused.
"""
from __future__ import annotations
from data.cnf import CNF
from eval.cnf_cached import CachedCNFRepairState


class CompactCNFRepairState(CachedCNFRepairState):
    """Same exact score/patch interface as the historical cache, smaller supports.

    Work counters retain the old canonical-incidence semantics, not CPU cycles.
    All caller validation completes before a patch changes the assignment.
    """
    def __init__(self, problem: CNF, witness: tuple[bool, ...]):
        if not isinstance(problem, CNF):
            raise TypeError("CNF required")
        problem.validate_witness(witness)
        self.problem = problem
        self._assignment = list(witness)
        self._variables: list[tuple[int, ...]] = []
        self._counts: list[int] = []
        self._xors: list[int] = []
        self._occurrences: list[list[int]] = [[] for _ in range(problem.nvars)]
        self._make = [0] * problem.nvars
        self._break = [0] * problem.nvars
        self._unsatisfied: set[int] = set()
        self.flips = self.patches = self.literal_updates = 0
        self.feature_literal_visits = self.cache_literal_visits = self.score_queries = 0
        for index, clause in enumerate(problem.clauses):
            literals = set(clause)
            if any(-lit in literals for lit in literals):
                self._variables.append(())
                self._counts.append(-1)  # tautology; absent from occurrence graph
                self._xors.append(0)
                continue
            variables = tuple(sorted(abs(lit) - 1 for lit in literals))
            count = support = 0
            for lit in literals:
                v = abs(lit) - 1
                self._occurrences[v].append(index + 1 if lit > 0 else -index - 1)
                if witness[v] == (lit > 0):
                    count += 1
                    support ^= v + 1
            self._variables.append(variables)
            self._counts.append(count)
            self._xors.append(support)
            self._contribution(index, count, support, 1)
            if count == 0:
                self._unsatisfied.add(index)
        self.initialization_cache_visits = self.cache_literal_visits
        self.cache_literal_visits = 0

    def _contribution(self, clause: int, count: int, support: int, sign: int) -> None:
        if count == 0:
            variables = self._variables[clause]
            for v in variables:
                self._make[v] += sign
            self.cache_literal_visits += len(variables)
        elif count == 1:
            self._break[support - 1] += sign

    def flip(self, variable: int) -> None:
        self._variable(variable)
        value = self._assignment[variable]
        key = variable + 1
        counts, supports = self._counts, self._xors
        made, broken = self._make, self._break
        occurrences = self._occurrences[variable]
        for occurrence in occurrences:
            clause = abs(occurrence) - 1
            old_count = counts[clause]
            new_count = old_count + (-1 if value == (occurrence > 0) else 1)
            old_support = supports[clause]
            new_support = old_support ^ key
            counts[clause] = new_count
            supports[clause] = new_support
            if old_count == 0:
                for member in self._variables[clause]:
                    made[member] -= 1
                self.cache_literal_visits += len(self._variables[clause])
                self._unsatisfied.remove(clause)
                broken[variable] += 1
            elif new_count == 0:
                broken[variable] -= 1
                for member in self._variables[clause]:
                    made[member] += 1
                self.cache_literal_visits += len(self._variables[clause])
                self._unsatisfied.add(clause)
            elif old_count == 1:
                broken[old_support - 1] -= 1
            elif new_count == 1:
                broken[new_support - 1] += 1
        self._assignment[variable] = not value
        self.literal_updates += len(occurrences)
        self.flips += 1
        self.patches += 1

    def _changes(self, variables: tuple[int, ...]) -> tuple[dict[int, tuple[int, int]], int]:
        changes: dict[int, tuple[int, int]] = {}
        visits = 0
        for v in variables:
            value = self._assignment[v]
            for occurrence in self._occurrences[v]:
                clause = abs(occurrence) - 1
                delta, support = changes.get(clause, (0, 0))
                delta += -1 if value == (occurrence > 0) else 1
                changes[clause] = (delta, support ^ (v + 1))
                visits += 1
        return changes, visits

    def make_break_patch(self, variables: tuple[int, ...]) -> tuple[int, int]:
        self._patch(variables)
        if len(variables) == 1:
            return self.make_break(variables[0])
        changes, visits = self._changes(variables)
        made = broken = 0
        for clause, (delta, _) in changes.items():
            old = self._counts[clause]
            new = old + delta
            made += old == 0 and new != 0
            broken += old != 0 and new == 0
        self.feature_literal_visits += visits
        self.score_queries += 1
        return made, broken

    def flip_patch(self, variables: tuple[int, ...]) -> None:
        self._patch(variables)
        if len(variables) == 1:
            self.flip(variables[0])
            return
        changes, visits = self._changes(variables)
        for clause, (delta, toggles) in changes.items():
            old_count, old_support = self._counts[clause], self._xors[clause]
            new_count, new_support = old_count + delta, old_support ^ toggles
            self._contribution(clause, old_count, old_support, -1)
            self._contribution(clause, new_count, new_support, 1)
            self._counts[clause], self._xors[clause] = new_count, new_support
            if new_count == 0:
                self._unsatisfied.add(clause)
            else:
                self._unsatisfied.discard(clause)
        for v in variables:
            self._assignment[v] = not self._assignment[v]
        self.literal_updates += visits
        self.flips += len(variables)
        self.patches += 1
