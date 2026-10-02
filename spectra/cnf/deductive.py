"""Optional classical deduction before bounded local search.

Unit propagation and iterative implication-graph SCCs add no learned parameters.
SAT answers are checked against original clauses; contradictions remain UNKNOWN
because this API does not emit an independently checkable UNSAT certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

from data.cnf import CNF
from .indexed import _settings, solve_indexed
from .search import SolveResult


@dataclass(frozen=True)
class DeductiveResult(SolveResult):
    strategy: str
    propagated_variables: int
    implication_edges: int

    def record(self) -> dict:
        return {**super().record(), "algorithm": "deduction_then_indexed_search",
                "strategy": self.strategy, "propagated_variables": self.propagated_variables,
                "implication_edges": self.implication_edges}


def _node(literal: int) -> int:
    return 2 * (abs(literal) - 1) + int(literal > 0)


def _propagate(nvars, clauses):
    values = [-1] * nvars
    occurrences = [[] for _ in range(2 * nvars)]
    counts, remaining, satisfied, queue = [], [], [False] * len(clauses), []

    def assign(literal):
        variable, value = abs(literal) - 1, int(literal > 0)
        if values[variable] == -1:
            values[variable] = value
            queue.append(literal)
        return values[variable] == value

    for index, clause in enumerate(clauses):
        counts.append(len(clause))
        bits = 0
        for literal in clause:
            occurrences[_node(literal)].append(index)
            bits ^= literal
        remaining.append(bits)
        if not clause or (len(clause) == 1 and not assign(clause[0])):
            return values, (), True
    for literal in queue:
        for index in occurrences[_node(literal)]:
            satisfied[index] = True
        false_literal = -literal
        for index in occurrences[_node(false_literal)]:
            if satisfied[index]:
                continue
            counts[index] -= 1
            remaining[index] ^= false_literal
            if counts[index] == 0 or (counts[index] == 1 and not assign(remaining[index])):
                return values, (), True
    residual = tuple(tuple(lit for lit in clause if values[abs(lit)-1] == -1)
                     for index, clause in enumerate(clauses) if not satisfied[index])
    return values, residual, False


def _binary(nvars, clauses):
    """Kosaraju SCCs with explicit stacks: no Python recursion depth dependency."""
    graph, reverse = [[] for _ in range(2*nvars)], [[] for _ in range(2*nvars)]
    edges = 0
    for clause in clauses:
        a, b = _node(clause[0]), _node(clause[-1])
        for source, target in ((a ^ 1, b), (b ^ 1, a)):
            graph[source].append(target)
            reverse[target].append(source)
            edges += 1
    visited, order = bytearray(2*nvars), []
    for root in range(2*nvars):
        if visited[root]:
            continue
        visited[root] = 1
        stack = [(root, 0)]
        while stack:
            vertex, position = stack[-1]
            if position == len(graph[vertex]):
                order.append(vertex)
                stack.pop()
            else:
                stack[-1] = (vertex, position + 1)
                child = graph[vertex][position]
                if not visited[child]:
                    visited[child] = 1
                    stack.append((child, 0))
    component, number = [-1] * (2*nvars), 0
    for root in reversed(order):
        if component[root] != -1:
            continue
        component[root] = number
        stack = [root]
        while stack:
            for child in reverse[stack.pop()]:
                if component[child] == -1:
                    component[child] = number
                    stack.append(child)
        number += 1
    if any(component[2*v] == component[2*v+1] for v in range(nvars)):
        return None, edges
    return tuple(component[2*v+1] > component[2*v] for v in range(nvars)), edges


def solve_deductive(problem: CNF, *, seed: int = 0, max_flips: int = 1024) -> DeductiveResult:
    """Deduce forced values, solve binary residuals, then cap residual local search.

    The flip cap applies only to local search. Deduction is additional linear
    work in the input literal inventory, not a hard wall-clock deadline. General
    formulas without initial units retain the indexed backend's seeded search.
    """
    _settings(problem, seed, max_flips)
    start = time.perf_counter_ns()
    # Three distinct variables in every clause rule out initial units and a
    # binary residual. Avoid allocating a normalized copy on this common path.
    general = all(len(c) >= 3 and abs(c[0]) != abs(c[1]) and
                  abs(c[0]) != abs(c[2]) and abs(c[1]) != abs(c[2])
                  for c in problem.clauses)
    clauses = []
    if not general:
        for original in problem.clauses:
            unique = set(original)
            if not any(-literal in unique for literal in unique):
                clauses.append(tuple(sorted(unique, key=abs)))
    has_units = any(len(clause) <= 1 for clause in clauses)
    if general or (not has_units and any(len(clause) > 2 for clause in clauses)):
        # Do not change the baseline trajectory or retain unused preprocessing.
        del clauses
        result = solve_indexed(problem, seed=seed, max_flips=max_flips)
        return DeductiveResult(result.status, result.witness, result.unsatisfied,
                               result.flips, result.queries, result.path_sha256,
                               time.perf_counter_ns()-start, seed, max_flips, "indexed", 0, 0)
    if has_units:
        values, residual, contradiction = _propagate(problem.nvars, clauses)
    else:
        values, residual, contradiction = [-1] * problem.nvars, tuple(clauses), False
    propagated = sum(v != -1 for v in values)
    witness = [bool(v) if v != -1 else False for v in values]
    strategy, edges, flips, queries, path = "propagation", 0, 0, 0, ""
    if contradiction:
        strategy = "contradiction"
    elif residual:
        if all(len(clause) <= 2 for clause in residual):
            strategy = "2sat"
            answer, edges = _binary(problem.nvars, residual)
            if answer is None:
                strategy = "contradiction"
            else:
                for variable, value in enumerate(answer):
                    if values[variable] == -1:
                        witness[variable] = value
        else:
            strategy = "propagation+indexed"
            active = sorted({abs(lit) for clause in residual for lit in clause})
            mapping = {variable: i+1 for i, variable in enumerate(active)}
            reduced = CNF(len(active), tuple(tuple(mapping[abs(lit)] * (1 if lit > 0 else -1)
                                                   for lit in clause) for clause in residual))
            result = solve_indexed(reduced, seed=seed, max_flips=max_flips)
            for variable, value in zip(active, result.witness):
                witness[variable-1] = value
            flips, queries, path = result.flips, result.queries, result.path_sha256
    witness = tuple(witness)
    unsatisfied = problem.violated(witness)
    digest = hashlib.sha256(b"spectra.deductive.v1\0" + bytes(witness) + path.encode("ascii")).hexdigest()
    return DeductiveResult("UNKNOWN" if unsatisfied else "SAT_VERIFIED", witness, unsatisfied,
                           flips, queries, digest, time.perf_counter_ns()-start, seed, max_flips,
                           strategy, propagated, edges)
