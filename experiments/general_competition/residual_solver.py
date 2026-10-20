"""Correctness-first proof-carrying search for the residual general CNF.

The binary core removes only replayable unit/binary structure. This module solves
what remains with deterministic DPLL-style search, complete SAT models, and a
portable UNSAT search tree. The search tree is deliberately simple: every unit
step names its input clause, every conflict names a falsified input clause, and
every branch covers both values of one unassigned variable. A separate verifier
replays the tree without trusting the search procedure.

This establishes the Phase 4 correctness boundary. It is not yet the native CDCL
engine or the competition-performance claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from experiments.general_competition.binary_core import (
    BinaryCore,
    Clause,
    build_binary_core,
    canonical_cnf,
    canonical_sha256,
    model_satisfies,
    verify_binary_core,
)


@dataclass(frozen=True)
class PropagationStep:
    literal: int
    reason: Clause


@dataclass(frozen=True)
class SearchProofNode:
    propagations: tuple[PropagationStep, ...]
    kind: str
    conflict_clause: Clause | None = None
    variable: int = 0
    false_child: int = -1
    true_child: int = -1


@dataclass(frozen=True)
class ResidualProof:
    schema: str
    nvars: int
    canonical_sha256: str
    nodes: tuple[SearchProofNode, ...]
    root: int


@dataclass(frozen=True)
class GeneralResult:
    schema: str
    status: str
    reason: str
    model: tuple[bool, ...]
    residual_model: tuple[bool, ...]
    binary_core: BinaryCore
    residual_proof: ResidualProof | None
    decisions: int
    propagations: int
    conflicts: int
    proof_nodes: int


class _Budget(Exception):
    pass


def _limit(value: int, name: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} exact integer")
    return value


class _Search:
    def __init__(
        self,
        nvars: int,
        clauses: tuple[Clause, ...],
        *,
        max_decisions: int,
        max_propagations: int,
        max_proof_nodes: int,
        max_depth: int,
    ) -> None:
        self.nvars = nvars
        self.clauses = clauses
        self.max_decisions = max_decisions
        self.max_propagations = max_propagations
        self.max_proof_nodes = max_proof_nodes
        self.max_depth = max_depth
        self.decisions = 0
        self.propagations = 0
        self.conflicts = 0
        self.nodes: list[SearchProofNode] = []

    @staticmethod
    def _value(literal: int, values: list[int]) -> int:
        value = values[abs(literal)]
        return value if literal > 0 else -value

    def _append(self, node: SearchProofNode) -> int:
        if len(self.nodes) >= self.max_proof_nodes:
            raise _Budget("proof-node limit")
        self.nodes.append(node)
        return len(self.nodes) - 1

    def _propagate(
        self, values: list[int]
    ) -> tuple[tuple[PropagationStep, ...], Clause | None]:
        steps: list[PropagationStep] = []
        while True:
            changed = False
            for row in self.clauses:
                unassigned = 0
                unit = 0
                satisfied = False
                for literal in row:
                    value = self._value(literal, values)
                    if value > 0:
                        satisfied = True
                        break
                    if value == 0:
                        unassigned += 1
                        unit = literal
                if satisfied:
                    continue
                if unassigned == 0:
                    return tuple(steps), row
                if unassigned == 1:
                    variable = abs(unit)
                    sign = 1 if unit > 0 else -1
                    if values[variable] == 0:
                        if self.propagations >= self.max_propagations:
                            raise _Budget("propagation limit")
                        values[variable] = sign
                        steps.append(PropagationStep(unit, row))
                        self.propagations += 1
                        changed = True
                        break
            if not changed:
                return tuple(steps), None

    def _satisfied(self, values: list[int]) -> bool:
        return all(
            any(self._value(literal, values) > 0 for literal in row)
            for row in self.clauses
        )

    def _choose(self, values: list[int]) -> int:
        occurrences = [0] * (self.nvars + 1)
        for row in self.clauses:
            if any(self._value(literal, values) > 0 for literal in row):
                continue
            for literal in row:
                if values[abs(literal)] == 0:
                    occurrences[abs(literal)] += 1
        highest = max(occurrences, default=0)
        if highest == 0:
            return 0
        return next(
            variable
            for variable in range(1, self.nvars + 1)
            if occurrences[variable] == highest
        )

    def _solve(
        self, values: list[int], depth: int
    ) -> tuple[str, tuple[bool, ...] | None, int]:
        if depth > self.max_depth:
            raise _Budget("depth limit")
        local = values.copy()
        steps, conflict = self._propagate(local)
        if conflict is not None:
            self.conflicts += 1
            node = self._append(
                SearchProofNode(
                    propagations=steps,
                    kind="conflict",
                    conflict_clause=conflict,
                )
            )
            return "UNSAT", None, node
        if self._satisfied(local):
            model = tuple(
                local[variable] > 0 if local[variable] else False
                for variable in range(1, self.nvars + 1)
            )
            return "SAT", model, -1

        variable = self._choose(local)
        if variable == 0:
            raise AssertionError("unresolved residual CNF has no branch variable")
        if self.decisions >= self.max_decisions:
            raise _Budget("decision limit")
        self.decisions += 1

        low = local.copy()
        low[variable] = -1
        low_status, low_model, low_node = self._solve(low, depth + 1)
        if low_status == "SAT":
            return low_status, low_model, -1

        high = local.copy()
        high[variable] = 1
        high_status, high_model, high_node = self._solve(high, depth + 1)
        if high_status == "SAT":
            return high_status, high_model, -1

        node = self._append(
            SearchProofNode(
                propagations=steps,
                kind="branch",
                variable=variable,
                false_child=low_node,
                true_child=high_node,
            )
        )
        return "UNSAT", None, node

    def run(
        self,
    ) -> tuple[str, tuple[bool, ...] | None, ResidualProof | None, str]:
        try:
            status, model, root = self._solve([0] * (self.nvars + 1), 0)
        except _Budget as error:
            return "UNKNOWN", None, None, str(error)
        if status == "SAT":
            return "SAT", model, None, "model"
        proof = ResidualProof(
            schema="spectra.general_residual_proof.v1",
            nvars=self.nvars,
            canonical_sha256=canonical_sha256(self.nvars, self.clauses),
            nodes=tuple(self.nodes),
            root=root,
        )
        return "UNSAT", None, proof, "exhausted"


def solve_general(
    nvars: int,
    clauses: Iterable[Iterable[int]],
    *,
    max_decisions: int = 1_000_000,
    max_propagations: int = 10_000_000,
    max_proof_nodes: int = 2_000_000,
    max_depth: int = 800,
) -> GeneralResult:
    """Solve one arbitrary in-memory CNF or return an explicit resource UNKNOWN."""
    max_decisions = _limit(max_decisions, "max_decisions")
    max_propagations = _limit(max_propagations, "max_propagations")
    max_proof_nodes = _limit(max_proof_nodes, "max_proof_nodes", positive=True)
    max_depth = _limit(max_depth, "max_depth")

    canonical = canonical_cnf(nvars, clauses)
    core = build_binary_core(nvars, canonical)
    if core.contradiction:
        result = GeneralResult(
            schema="spectra.general_result.v1",
            status="UNSAT_VERIFIED",
            reason=f"binary-core-{core.conflict_kind}",
            model=(),
            residual_model=(),
            binary_core=core,
            residual_proof=None,
            decisions=0,
            propagations=len(core.unit_trace),
            conflicts=1,
            proof_nodes=0,
        )
        if not verify_general_result(nvars, canonical, result):
            raise AssertionError("binary-core result failed independent replay")
        return result

    search = _Search(
        core.reduced_nvars,
        core.residual_clauses,
        max_decisions=max_decisions,
        max_propagations=max_propagations,
        max_proof_nodes=max_proof_nodes,
        max_depth=max_depth,
    )
    status, residual_model, proof, reason = search.run()
    if status == "SAT":
        assert residual_model is not None
        result = GeneralResult(
            schema="spectra.general_result.v1",
            status="SAT_VERIFIED",
            reason=reason,
            model=core.lift(residual_model),
            residual_model=residual_model,
            binary_core=core,
            residual_proof=None,
            decisions=search.decisions,
            propagations=search.propagations + len(core.unit_trace),
            conflicts=search.conflicts,
            proof_nodes=0,
        )
    elif status == "UNSAT":
        assert proof is not None
        result = GeneralResult(
            schema="spectra.general_result.v1",
            status="UNSAT_VERIFIED",
            reason=reason,
            model=(),
            residual_model=(),
            binary_core=core,
            residual_proof=proof,
            decisions=search.decisions,
            propagations=search.propagations + len(core.unit_trace),
            conflicts=search.conflicts,
            proof_nodes=len(proof.nodes),
        )
    else:
        result = GeneralResult(
            schema="spectra.general_result.v1",
            status="UNKNOWN",
            reason=reason,
            model=(),
            residual_model=(),
            binary_core=core,
            residual_proof=None,
            decisions=search.decisions,
            propagations=search.propagations + len(core.unit_trace),
            conflicts=search.conflicts,
            proof_nodes=0,
        )

    if result.status != "UNKNOWN" and not verify_general_result(
        nvars, canonical, result
    ):
        raise AssertionError("general result failed independent replay")
    return result


def verify_residual_proof(
    nvars: int,
    clauses: Iterable[Iterable[int]],
    proof: ResidualProof,
) -> bool:
    """Check a complete branch-and-propagation UNSAT tree independently."""
    if type(proof) is not ResidualProof:
        return False
    try:
        canonical = canonical_cnf(nvars, clauses)
    except (TypeError, ValueError):
        return False
    if (
        proof.schema != "spectra.general_residual_proof.v1"
        or proof.nvars != nvars
        or proof.canonical_sha256 != canonical_sha256(nvars, canonical)
        or not proof.nodes
        or proof.root != len(proof.nodes) - 1
    ):
        return False

    clause_set = set(canonical)
    indegree = [0] * len(proof.nodes)
    for index, node in enumerate(proof.nodes):
        if type(node) is not SearchProofNode or node.kind not in (
            "conflict",
            "branch",
        ):
            return False
        if node.kind == "conflict":
            if (
                node.conflict_clause not in clause_set
                or node.variable != 0
                or node.false_child != -1
                or node.true_child != -1
            ):
                return False
        else:
            if (
                node.conflict_clause is not None
                or not 1 <= node.variable <= nvars
                or not 0 <= node.false_child < index
                or not 0 <= node.true_child < index
                or node.false_child == node.true_child
            ):
                return False
            indegree[node.false_child] += 1
            indegree[node.true_child] += 1
    if indegree[proof.root] != 0 or any(
        count != 1
        for index, count in enumerate(indegree)
        if index != proof.root
    ):
        return False

    reached: set[int] = set()
    stack: list[tuple[int, list[int]]] = [
        (proof.root, [0] * (nvars + 1))
    ]
    while stack:
        index, values = stack.pop()
        if index in reached:
            return False
        reached.add(index)
        node = proof.nodes[index]
        local = values.copy()
        for step in node.propagations:
            if type(step) is not PropagationStep or step.reason not in clause_set:
                return False
            literal = step.literal
            if (
                type(literal) is not int
                or literal == 0
                or abs(literal) > nvars
                or literal not in step.reason
                or local[abs(literal)] != 0
            ):
                return False
            free: list[int] = []
            for candidate in step.reason:
                signed = (
                    local[abs(candidate)]
                    if candidate > 0
                    else -local[abs(candidate)]
                )
                if signed > 0:
                    return False
                if signed == 0:
                    free.append(candidate)
            if free != [literal]:
                return False
            local[abs(literal)] = 1 if literal > 0 else -1

        if node.kind == "conflict":
            assert node.conflict_clause is not None
            if any(
                (
                    local[abs(literal)]
                    if literal > 0
                    else -local[abs(literal)]
                )
                >= 0
                for literal in node.conflict_clause
            ):
                return False
            continue

        variable = node.variable
        if local[variable] != 0:
            return False
        low = local.copy()
        high = local.copy()
        low[variable] = -1
        high[variable] = 1
        stack.append((node.true_child, high))
        stack.append((node.false_child, low))
    return reached == set(range(len(proof.nodes)))


def verify_general_result(
    nvars: int,
    clauses: Iterable[Iterable[int]],
    result: GeneralResult,
) -> bool:
    """Check the binary reduction and final SAT/UNSAT result from original CNF."""
    if type(result) is not GeneralResult or result.schema != "spectra.general_result.v1":
        return False
    if any(
        type(value) is not int or value < 0
        for value in (
            result.decisions,
            result.propagations,
            result.conflicts,
            result.proof_nodes,
        )
    ):
        return False
    try:
        canonical = canonical_cnf(nvars, clauses)
    except (TypeError, ValueError):
        return False
    core = result.binary_core
    if not verify_binary_core(nvars, canonical, core):
        return False

    if result.status == "SAT_VERIFIED":
        return (
            not core.contradiction
            and result.residual_proof is None
            and len(result.residual_model) == core.reduced_nvars
            and core.lift(result.residual_model) == result.model
            and model_satisfies(
                core.reduced_nvars,
                core.residual_clauses,
                result.residual_model,
            )
            and model_satisfies(nvars, canonical, result.model)
            and result.proof_nodes == 0
        )
    if result.status == "UNSAT_VERIFIED":
        if result.model or result.residual_model:
            return False
        if core.contradiction:
            return result.residual_proof is None and result.proof_nodes == 0
        return (
            result.residual_proof is not None
            and result.proof_nodes == len(result.residual_proof.nodes)
            and verify_residual_proof(
                core.reduced_nvars,
                core.residual_clauses,
                result.residual_proof,
            )
        )
    if result.status == "UNKNOWN":
        return (
            not result.model
            and not result.residual_model
            and result.residual_proof is None
            and result.proof_nodes == 0
        )
    return False
