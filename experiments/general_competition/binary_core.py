"""Proof-replayable unit/binary core for arbitrary CNF development.

The core uses only generic propositional operations: tautology/duplicate removal,
unit and binary implication SCCs, signed-equivalence substitution, and unit
propagation. Clauses of width three or greater are never treated as binary; they
remain in the residual formula after safe substitution and propagation.

This is a Phase 4 development component, not a general SAT solver and not a
novelty claim. The retained component trees and unit trace let a separate replay
check every reduction step from the original CNF without trusting a solver answer.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from itertools import chain
from typing import Iterable

Clause = tuple[int, ...]
Arc = tuple[int, int, Clause]


@dataclass(frozen=True)
class ComponentProof:
    """Two spanning trees proving mutual reachability of signed literals."""

    nodes: tuple[int, ...]
    root: int
    outward: tuple[Arc, ...]
    inward: tuple[Arc, ...]


@dataclass(frozen=True)
class UnitStep:
    literal: int
    reason: Clause


@dataclass(frozen=True)
class BinaryCore:
    schema: str
    original_nvars: int
    canonical_sha256: str
    normalized_clause_count: int
    unit_input_clauses: int
    binary_input_clauses: int
    long_input_clauses: int
    class_count: int
    class_literals: tuple[int, ...]
    component_proofs: tuple[ComponentProof, ...]
    unit_trace: tuple[UnitStep, ...]
    conflict_kind: str
    conflict_variable: int
    conflict_clause: Clause | None
    reduced_nvars: int
    residual_clauses: tuple[Clause, ...]
    mapping: tuple[int, ...]
    fixed: tuple[bool, ...]
    equivalent_variables_removed: int
    propagated_classes: int

    @property
    def contradiction(self) -> bool:
        return bool(self.conflict_kind)

    def lift(self, model: tuple[bool, ...]) -> tuple[bool, ...]:
        """Lift one complete residual model into original variable order."""
        if self.contradiction:
            raise ValueError("a contradictory core has no model to lift")
        if (type(model) is not tuple or len(model) != self.reduced_nvars
                or any(type(value) is not bool for value in model)):
            raise ValueError("model must be a complete tuple of exact bool values")
        return tuple(
            self.fixed[index]
            if target == 0
            else model[abs(target) - 1] == (target > 0)
            for index, target in enumerate(self.mapping)
        )

    def project_assumptions(self, literals: Iterable[int]) -> tuple[int, ...] | None:
        """Map original assumptions to the residual variables.

        ``None`` means the assumptions contradict a propagated fixed value or one
        another. Every original literal is still validated before that result.
        """
        if self.contradiction:
            for literal in literals:
                _validate_literal(literal, self.original_nvars, "assumption")
            return None
        projected: dict[int, int] = {}
        for literal in literals:
            _validate_literal(literal, self.original_nvars, "assumption")
            index = abs(literal) - 1
            target = self.mapping[index]
            requested = literal > 0
            if target == 0:
                if self.fixed[index] != requested:
                    return None
                continue
            reduced = target if requested else -target
            variable = abs(reduced)
            sign = 1 if reduced > 0 else -1
            previous = projected.get(variable)
            if previous is not None and previous != sign:
                return None
            projected[variable] = sign
        return tuple(variable * projected[variable] for variable in sorted(projected))


def _validate_literal(literal: object, nvars: int, label: str = "CNF literal") -> int:
    if type(literal) is not int or literal == 0 or abs(literal) > nvars:
        raise ValueError(f"{label} must be a nonzero in-range exact integer")
    return literal


def canonical_cnf(
    nvars: int, clauses: Iterable[Iterable[int]]
) -> tuple[Clause, ...]:
    """Validate and propositionally canonicalize a CNF.

    Tautological clauses, duplicate literals and duplicate clauses are removed.
    This does not assign a variable, inspect a solver result, or discard any
    non-tautological constraint.
    """
    if type(nvars) is not int or not 0 <= nvars <= 10_000_000:
        raise ValueError("nvars must be an exact integer in [0, 10000000]")
    rows: set[Clause] = set()
    try:
        iterator = iter(clauses)
    except TypeError as exc:
        raise ValueError("clauses must be iterable") from exc
    for clause in iterator:
        try:
            row = tuple(clause)
        except TypeError as exc:
            raise ValueError("each clause must be iterable") from exc
        values: set[int] = set()
        tautology = False
        for literal in row:
            value = _validate_literal(literal, nvars)
            if -value in values:
                tautology = True
            values.add(value)
        if tautology:
            continue
        rows.add(tuple(sorted(values, key=lambda value: (abs(value), value))))
    return tuple(sorted(rows, key=lambda row: (len(row), row)))


def canonical_sha256(nvars: int, clauses: tuple[Clause, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(f"p cnf {nvars} {len(clauses)}\n".encode("ascii"))
    for row in clauses:
        if row:
            digest.update(b" ".join(str(value).encode("ascii") for value in row))
            digest.update(b" ")
        digest.update(b"0\n")
    return digest.hexdigest()


def _node(literal: int) -> int:
    return 2 * (abs(literal) - 1) + int(literal < 0)


def _literal(node: int) -> int:
    variable = node // 2 + 1
    return -variable if node & 1 else variable


def _implication_graph(
    nvars: int, clauses: tuple[Clause, ...]
) -> tuple[list[list[tuple[int, Clause]]], list[list[tuple[int, Clause]]]]:
    graph: list[list[tuple[int, Clause]]] = [[] for _ in range(2 * nvars)]
    reverse: list[list[tuple[int, Clause]]] = [[] for _ in graph]

    def add(source_literal: int, target_literal: int, reason: Clause) -> None:
        source = _node(source_literal)
        target = _node(target_literal)
        graph[source].append((target, reason))
        reverse[target].append((source, reason))

    for row in clauses:
        if len(row) == 1:
            add(-row[0], row[0], row)
        elif len(row) == 2:
            left, right = row
            add(-left, right, row)
            add(-right, left, row)
    for adjacency in chain(graph, reverse):
        adjacency.sort(key=lambda item: (item[0], item[1]))
    return graph, reverse


def _strong_components(
    graph: list[list[tuple[int, Clause]]],
    reverse: list[list[tuple[int, Clause]]],
) -> tuple[list[int], tuple[tuple[int, ...], ...]]:
    size = len(graph)
    seen = bytearray(size)
    finish: list[int] = []
    for root in range(size):
        if seen[root]:
            continue
        seen[root] = 1
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            vertex, offset = stack[-1]
            if offset == len(graph[vertex]):
                finish.append(vertex)
                stack.pop()
                continue
            neighbor = graph[vertex][offset][0]
            stack[-1] = (vertex, offset + 1)
            if not seen[neighbor]:
                seen[neighbor] = 1
                stack.append((neighbor, 0))

    component = [-1] * size
    groups: list[tuple[int, ...]] = []
    for root in reversed(finish):
        if component[root] != -1:
            continue
        index = len(groups)
        component[root] = index
        members: list[int] = []
        stack = [root]
        while stack:
            vertex = stack.pop()
            members.append(vertex)
            for neighbor, _reason in reverse[vertex]:
                if component[neighbor] == -1:
                    component[neighbor] = index
                    stack.append(neighbor)
        groups.append(tuple(sorted(members)))
    return component, tuple(groups)


def _component_proof(
    group: tuple[int, ...],
    graph: list[list[tuple[int, Clause]]],
    reverse: list[list[tuple[int, Clause]]],
) -> ComponentProof:
    allowed = set(group)
    root = min(group)

    outward: list[Arc] = []
    seen = {root}
    queue = deque([root])
    while queue:
        source = queue.popleft()
        for target, reason in graph[source]:
            if target in allowed and target not in seen:
                seen.add(target)
                queue.append(target)
                outward.append((_literal(source), _literal(target), reason))
    if seen != allowed:
        raise AssertionError("SCC outward proof did not span its component")

    inward: list[Arc] = []
    seen = {root}
    queue = deque([root])
    while queue:
        target = queue.popleft()
        for source, reason in reverse[target]:
            if source in allowed and source not in seen:
                seen.add(source)
                queue.append(source)
                inward.append((_literal(source), _literal(target), reason))
    if seen != allowed:
        raise AssertionError("SCC inward proof did not span its component")

    signed_nodes = tuple(sorted((_literal(node) for node in group), key=lambda x: (abs(x), x)))
    return ComponentProof(
        nodes=signed_nodes,
        root=_literal(root),
        outward=tuple(outward),
        inward=tuple(inward),
    )


def _equivalence_classes(
    nvars: int, component: list[int]
) -> tuple[int, tuple[int, ...]]:
    by_pair: dict[tuple[int, int], list[int]] = {}
    for variable in range(1, nvars + 1):
        positive = component[_node(variable)]
        negative = component[_node(-variable)]
        key = (min(positive, negative), max(positive, negative))
        by_pair.setdefault(key, []).append(variable)

    class_literals = [0] * nvars
    groups = sorted(by_pair.values(), key=lambda members: min(members))
    for class_index, members in enumerate(groups, 1):
        representative = min(members)
        positive_component = component[_node(representative)]
        for variable in members:
            if component[_node(variable)] == positive_component:
                class_literals[variable - 1] = class_index
            elif component[_node(-variable)] == positive_component:
                class_literals[variable - 1] = -class_index
            else:
                raise AssertionError("complementary SCC pairing is inconsistent")
    return len(groups), tuple(class_literals)


def _substitute(
    class_count: int,
    clauses: tuple[Clause, ...],
    class_literals: tuple[int, ...],
) -> tuple[Clause, ...]:
    return canonical_cnf(
        class_count,
        (
            tuple(
                class_literals[abs(literal) - 1] * (1 if literal > 0 else -1)
                for literal in row
            )
            for row in clauses
        ),
    )


def _unit_propagate(
    nvars: int, clauses: tuple[Clause, ...]
) -> tuple[tuple[int, ...], tuple[UnitStep, ...], Clause | None]:
    values = [0] * (nvars + 1)
    trace: list[UnitStep] = []
    if () in clauses:
        return tuple(values), (), ()

    occurrences: dict[int, list[int]] = {}
    remaining = [len(row) for row in clauses]
    satisfied = bytearray(len(clauses))
    pending: deque[tuple[int, Clause]] = deque()
    for index, row in enumerate(clauses):
        if len(row) == 1:
            pending.append((row[0], row))
        for literal in row:
            occurrences.setdefault(literal, []).append(index)

    while pending:
        literal, reason = pending.popleft()
        variable = abs(literal)
        sign = 1 if literal > 0 else -1
        if values[variable] == sign:
            continue
        if values[variable] == -sign:
            return tuple(values), tuple(trace), reason
        values[variable] = sign
        trace.append(UnitStep(literal, reason))

        for index in occurrences.get(literal, ()):
            satisfied[index] = 1
        for index in occurrences.get(-literal, ()):
            if satisfied[index]:
                continue
            remaining[index] -= 1
            if remaining[index] == 0:
                return tuple(values), tuple(trace), clauses[index]
            if remaining[index] == 1:
                unit = next(
                    value for value in clauses[index]
                    if values[abs(value)] == 0
                )
                pending.append((unit, clauses[index]))
    return tuple(values), tuple(trace), None


def _empty_result(
    *,
    nvars: int,
    digest: str,
    clauses: tuple[Clause, ...],
    proofs: tuple[ComponentProof, ...],
    kind: str,
    variable: int,
    conflict_clause: Clause | None,
    class_count: int = 0,
    class_literals: tuple[int, ...] = (),
    unit_trace: tuple[UnitStep, ...] = (),
    equivalent_variables_removed: int = 0,
    propagated_classes: int = 0,
) -> BinaryCore:
    return BinaryCore(
        schema="spectra.general_binary_core.v1",
        original_nvars=nvars,
        canonical_sha256=digest,
        normalized_clause_count=len(clauses),
        unit_input_clauses=sum(len(row) == 1 for row in clauses),
        binary_input_clauses=sum(len(row) == 2 for row in clauses),
        long_input_clauses=sum(len(row) >= 3 for row in clauses),
        class_count=class_count,
        class_literals=class_literals,
        component_proofs=proofs,
        unit_trace=unit_trace,
        conflict_kind=kind,
        conflict_variable=variable,
        conflict_clause=conflict_clause,
        reduced_nvars=0,
        residual_clauses=((),),
        mapping=(0,) * nvars,
        fixed=(False,) * nvars,
        equivalent_variables_removed=equivalent_variables_removed,
        propagated_classes=propagated_classes,
    )


def build_binary_core(
    nvars: int, clauses: Iterable[Iterable[int]]
) -> BinaryCore:
    """Build a deterministic proof-replayable binary core and residual CNF."""
    canonical = canonical_cnf(nvars, clauses)
    digest = canonical_sha256(nvars, canonical)
    graph, reverse = _implication_graph(nvars, canonical)
    component, groups = _strong_components(graph, reverse)
    proofs = tuple(
        _component_proof(group, graph, reverse)
        for group in groups if len(group) > 1
    )

    conflict_variable = next(
        (
            variable for variable in range(1, nvars + 1)
            if component[_node(variable)] == component[_node(-variable)]
        ),
        0,
    )
    if conflict_variable:
        return _empty_result(
            nvars=nvars,
            digest=digest,
            clauses=canonical,
            proofs=proofs,
            kind="binary",
            variable=conflict_variable,
            conflict_clause=None,
        )

    class_count, class_literals = _equivalence_classes(nvars, component)
    substituted = _substitute(class_count, canonical, class_literals)
    values, trace, conflict_clause = _unit_propagate(class_count, substituted)
    removed = nvars - class_count
    propagated = sum(value != 0 for value in values[1:])
    if conflict_clause is not None:
        return _empty_result(
            nvars=nvars,
            digest=digest,
            clauses=canonical,
            proofs=proofs,
            kind="unit",
            variable=0,
            conflict_clause=conflict_clause,
            class_count=class_count,
            class_literals=class_literals,
            unit_trace=trace,
            equivalent_variables_removed=removed,
            propagated_classes=propagated,
        )

    free = {
        variable: index
        for index, variable in enumerate(
            (value for value in range(1, class_count + 1) if values[value] == 0),
            1,
        )
    }
    residual_rows: list[Clause] = []
    for row in substituted:
        if any(values[abs(literal)] == (1 if literal > 0 else -1) for literal in row):
            continue
        residual_rows.append(tuple(
            free[abs(literal)] * (1 if literal > 0 else -1)
            for literal in row if values[abs(literal)] == 0
        ))
    residual = canonical_cnf(len(free), residual_rows)

    mapping: list[int] = []
    fixed: list[bool] = []
    for target in class_literals:
        class_variable = abs(target)
        relation = 1 if target > 0 else -1
        if values[class_variable]:
            mapping.append(0)
            fixed.append(values[class_variable] * relation > 0)
        else:
            mapping.append(free[class_variable] * relation)
            fixed.append(False)

    return BinaryCore(
        schema="spectra.general_binary_core.v1",
        original_nvars=nvars,
        canonical_sha256=digest,
        normalized_clause_count=len(canonical),
        unit_input_clauses=sum(len(row) == 1 for row in canonical),
        binary_input_clauses=sum(len(row) == 2 for row in canonical),
        long_input_clauses=sum(len(row) >= 3 for row in canonical),
        class_count=class_count,
        class_literals=class_literals,
        component_proofs=proofs,
        unit_trace=trace,
        conflict_kind="",
        conflict_variable=0,
        conflict_clause=None,
        reduced_nvars=len(free),
        residual_clauses=residual,
        mapping=tuple(mapping),
        fixed=tuple(fixed),
        equivalent_variables_removed=removed,
        propagated_classes=propagated,
    )


def _arc_is_justified(source: int, target: int, reason: Clause) -> bool:
    if len(reason) == 1:
        return source == -reason[0] and target == reason[0]
    if len(reason) == 2:
        left, right = reason
        return ((source == -left and target == right)
                or (source == -right and target == left))
    return False


def _tree_spans(
    nodes: set[int], root: int, edges: tuple[Arc, ...], *, inward: bool,
    clause_set: set[Clause], nvars: int,
) -> bool:
    if len(edges) != len(nodes) - 1:
        return False
    adjacency: dict[int, list[int]] = {}
    for source, target, reason in edges:
        if (source not in nodes or target not in nodes or reason not in clause_set
                or not _arc_is_justified(source, target, reason)):
            return False
        _validate_literal(source, nvars, "proof literal")
        _validate_literal(target, nvars, "proof literal")
        if inward:
            adjacency.setdefault(target, []).append(source)
        else:
            adjacency.setdefault(source, []).append(target)
    reached = {root}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, ()):
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return reached == nodes


def _verify_component_proofs(
    nvars: int,
    canonical: tuple[Clause, ...],
    proofs: tuple[ComponentProof, ...],
) -> dict[int, int] | None:
    clause_set = set(canonical)
    owner: dict[int, int] = {}
    for index, proof in enumerate(proofs):
        if type(proof) is not ComponentProof or not proof.nodes:
            return None
        nodes = set(proof.nodes)
        if len(nodes) != len(proof.nodes) or proof.root not in nodes:
            return None
        for literal in nodes:
            try:
                _validate_literal(literal, nvars, "proof literal")
            except ValueError:
                return None
            if literal in owner:
                return None
            owner[literal] = index
        if not _tree_spans(
            nodes, proof.root, proof.outward, inward=False,
            clause_set=clause_set, nvars=nvars,
        ):
            return None
        if not _tree_spans(
            nodes, proof.root, proof.inward, inward=True,
            clause_set=clause_set, nvars=nvars,
        ):
            return None
    return owner


def _replay_units(
    nvars: int,
    clauses: tuple[Clause, ...],
    trace: tuple[UnitStep, ...],
) -> tuple[tuple[int, ...], bool]:
    values = [0] * (nvars + 1)
    clause_set = set(clauses)
    for step in trace:
        if type(step) is not UnitStep or step.reason not in clause_set:
            return tuple(values), False
        literal = step.literal
        try:
            _validate_literal(literal, nvars, "unit literal")
        except ValueError:
            return tuple(values), False
        if literal not in step.reason or values[abs(literal)] != 0:
            return tuple(values), False
        unassigned: list[int] = []
        for candidate in step.reason:
            value = values[abs(candidate)]
            if value == 0:
                unassigned.append(candidate)
            elif value == (1 if candidate > 0 else -1):
                return tuple(values), False
        if unassigned != [literal]:
            return tuple(values), False
        values[abs(literal)] = 1 if literal > 0 else -1
    return tuple(values), True


def _expected_remainder(
    class_count: int,
    substituted: tuple[Clause, ...],
    values: tuple[int, ...],
) -> tuple[int, tuple[Clause, ...], dict[int, int]]:
    free = {
        variable: index
        for index, variable in enumerate(
            (value for value in range(1, class_count + 1) if values[value] == 0),
            1,
        )
    }
    rows: list[Clause] = []
    for row in substituted:
        if any(values[abs(literal)] == (1 if literal > 0 else -1) for literal in row):
            continue
        rows.append(tuple(
            free[abs(literal)] * (1 if literal > 0 else -1)
            for literal in row if values[abs(literal)] == 0
        ))
    return len(free), canonical_cnf(len(free), rows), free


def verify_binary_core(
    nvars: int,
    clauses: Iterable[Iterable[int]],
    core: BinaryCore,
) -> bool:
    """Replay a retained binary-core transformation from original CNF inputs."""
    if type(core) is not BinaryCore:
        return False
    try:
        canonical = canonical_cnf(nvars, clauses)
    except (TypeError, ValueError):
        return False
    if (
        core.schema != "spectra.general_binary_core.v1"
        or core.original_nvars != nvars
        or core.canonical_sha256 != canonical_sha256(nvars, canonical)
        or core.normalized_clause_count != len(canonical)
        or core.unit_input_clauses != sum(len(row) == 1 for row in canonical)
        or core.binary_input_clauses != sum(len(row) == 2 for row in canonical)
        or core.long_input_clauses != sum(len(row) >= 3 for row in canonical)
    ):
        return False

    owner = _verify_component_proofs(nvars, canonical, core.component_proofs)
    if owner is None:
        return False

    if core.conflict_kind == "binary":
        variable = core.conflict_variable
        if not 1 <= variable <= nvars:
            return False
        if owner.get(variable) is None or owner.get(variable) != owner.get(-variable):
            return False
        return (
            core.class_count == 0
            and core.class_literals == ()
            and core.unit_trace == ()
            and core.conflict_clause is None
            and core.reduced_nvars == 0
            and core.residual_clauses == ((),)
            and core.mapping == (0,) * nvars
            and core.fixed == (False,) * nvars
            and core.equivalent_variables_removed == 0
            and core.propagated_classes == 0
        )

    if core.conflict_kind not in ("", "unit") or core.conflict_variable != 0:
        return False
    if len(core.class_literals) != nvars or type(core.class_count) is not int:
        return False
    used = {abs(value) for value in core.class_literals}
    if used != set(range(1, core.class_count + 1)):
        return False

    for class_index in range(1, core.class_count + 1):
        members = [
            variable for variable, target in enumerate(core.class_literals, 1)
            if abs(target) == class_index
        ]
        if not members:
            return False
        representative = min(members)
        if core.class_literals[representative - 1] != class_index:
            return False
        for variable in members:
            target = core.class_literals[variable - 1]
            related_literal = variable if target > 0 else -variable
            if related_literal == representative:
                continue
            if (owner.get(representative) is None
                    or owner.get(representative) != owner.get(related_literal)):
                return False

    try:
        substituted = _substitute(core.class_count, canonical, core.class_literals)
    except (ValueError, IndexError):
        return False
    values, trace_ok = _replay_units(core.class_count, substituted, core.unit_trace)
    if not trace_ok:
        return False
    propagated = sum(value != 0 for value in values[1:])
    if (
        core.equivalent_variables_removed != nvars - core.class_count
        or core.propagated_classes != propagated
    ):
        return False

    if core.conflict_kind == "unit":
        if core.conflict_clause not in set(substituted):
            return False
        assert core.conflict_clause is not None
        for literal in core.conflict_clause:
            value = values[abs(literal)]
            if value == 0 or value == (1 if literal > 0 else -1):
                return False
        return (
            core.reduced_nvars == 0
            and core.residual_clauses == ((),)
            and core.mapping == (0,) * nvars
            and core.fixed == (False,) * nvars
        )

    if core.conflict_clause is not None:
        return False
    for row in substituted:
        if any(values[abs(literal)] == (1 if literal > 0 else -1) for literal in row):
            continue
        if sum(values[abs(literal)] == 0 for literal in row) < 2:
            return False

    reduced_nvars, residual, free = _expected_remainder(
        core.class_count, substituted, values
    )
    expected_mapping: list[int] = []
    expected_fixed: list[bool] = []
    for target in core.class_literals:
        class_variable = abs(target)
        relation = 1 if target > 0 else -1
        if values[class_variable]:
            expected_mapping.append(0)
            expected_fixed.append(values[class_variable] * relation > 0)
        else:
            expected_mapping.append(free[class_variable] * relation)
            expected_fixed.append(False)
    return (
        core.reduced_nvars == reduced_nvars
        and core.residual_clauses == residual
        and core.mapping == tuple(expected_mapping)
        and core.fixed == tuple(expected_fixed)
    )


def model_satisfies(
    nvars: int,
    clauses: Iterable[Iterable[int]],
    model: tuple[bool, ...],
) -> bool:
    """Independently check one complete Boolean model."""
    if (type(model) is not tuple or len(model) != nvars
            or any(type(value) is not bool for value in model)):
        raise ValueError("model must be a complete tuple of exact bool values")
    canonical = canonical_cnf(nvars, clauses)
    return all(
        any(model[abs(literal) - 1] == (literal > 0) for literal in row)
        for row in canonical
    )
