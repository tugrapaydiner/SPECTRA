"""Proof-carrying bounded exam rescheduling on top of :mod:`spectra.exact`.

This adapter models a deliberately narrow operational problem: every exam has one
or two pre-approved candidate periods, and exams that share a student (or an
explicit exclusion) may not use the same period.  Period closures and exam pins
become current-query restrictions.  The exact backend can then return either a
complete checked repair or a portable contradiction proof.

Room capacities, room exclusivity, precedence, coincidence, soft penalties and
candidate generation are outside this model.  They must be handled before
admission or checked separately; the module never silently drops them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .exact_problem import ExactProblem
from .exact_proof import decode_proof, verify_answer

MAX_IDENTIFIER_CHARS = 512
MAX_ENROLLMENTS = 4_000_000
MAX_CONFLICTS = 2_000_000
MAX_PAIR_EMISSIONS = 8_000_000


@dataclass(frozen=True, slots=True)
class ExamConflictReason:
    """One deterministic witness for an admitted conflict edge."""

    kind: str
    student: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("shared_student", "explicit_exclusion"):
            raise ValueError("unsupported exam conflict reason")
        if self.kind == "shared_student":
            if type(self.student) is not int or not 0 <= self.student < 1 << 63:
                raise ValueError("shared-student reason needs a bounded student id")
        elif self.student is not None:
            raise ValueError("explicit exclusion must not carry a student id")


@dataclass(frozen=True, slots=True)
class ExamRepairModel:
    """A checked reduction from bounded exam repair to exact list colouring.

    ``problem`` uses exam positions as vertices and period positions as colours.
    ``conflict_reasons`` is aligned with ``problem.edges`` and is explanatory
    metadata only; the exact solver and verifier receive the original graph.
    """

    problem: ExactProblem
    exam_ids: tuple[str, ...]
    period_ids: tuple[str, ...]
    students: tuple[tuple[int, ...], ...]
    conflict_reasons: tuple[ExamConflictReason, ...]

    @classmethod
    def create(
        cls,
        period_ids: Iterable[str],
        exam_ids: Iterable[str],
        students: Iterable[Iterable[int]],
        candidate_periods: Iterable[Iterable[str]],
        *,
        exclusions: Iterable[Sequence[str]] = (),
    ) -> "ExamRepairModel":
        periods = cls._bounded_snapshot(period_ids, 64, "period identifier")
        exams = cls._bounded_snapshot(exam_ids, 100_000, "exam identifier")
        student_rows: list[tuple[int, ...]] = []
        try:
            for row in students:
                if len(student_rows) >= len(exams):
                    raise ValueError("exam, student and candidate row counts differ")
                if isinstance(row, (str, bytes)):
                    raise ValueError("student rows must be integer iterables, not strings")
                student_rows.append(tuple(row))
        except TypeError as error:
            raise ValueError("student rows must be iterable") from error
        candidate_rows: list[tuple[str, ...]] = []
        try:
            iterator = iter(candidate_periods)
        except TypeError as error:
            raise ValueError("candidate rows must be iterable") from error
        for row in iterator:
            if len(candidate_rows) >= len(exams):
                raise ValueError("exam, student and candidate row counts differ")
            if isinstance(row, (str, bytes)):
                raise ValueError("candidate rows must be period iterables, not strings")
            try:
                candidate_rows.append(tuple(row))
            except TypeError as error:
                raise ValueError("candidate rows must be iterable") from error
        if not 1 <= len(periods) <= 64:
            raise ValueError("period count must be in [1, 64]")
        if len(student_rows) != len(exams) or len(candidate_rows) != len(exams):
            raise ValueError("exam, student and candidate row counts differ")
        candidates = tuple(candidate_rows)
        cls._check_identifiers(periods, "period")
        cls._check_identifiers(exams, "exam")
        if len(set(periods)) != len(periods):
            raise ValueError("period identifiers must be unique")
        if len(set(exams)) != len(exams):
            raise ValueError("exam identifiers must be unique")

        period_at = {name: index for index, name in enumerate(periods)}
        masks: list[int] = []
        for row in candidates:
            if not 1 <= len(row) <= 2 or any(type(value) is not str for value in row):
                raise ValueError("each exam needs one or two exact period identifiers")
            if len(set(row)) != len(row):
                raise ValueError("candidate periods must be unique within an exam")
            try:
                indices = tuple(period_at[value] for value in row)
            except KeyError as error:
                raise ValueError("candidate names an unknown period") from error
            masks.append(sum(1 << index for index in indices))

        normalized_students: list[tuple[int, ...]] = []
        enrollment_total = 0
        roster: dict[int, list[int]] = {}
        for vertex, row in enumerate(student_rows):
            if any(type(value) is not int or not 0 <= value < 1 << 63 for value in row):
                raise ValueError("student identifiers must be bounded nonnegative integers")
            normalized = tuple(sorted(set(row)))
            enrollment_total += len(normalized)
            if enrollment_total > MAX_ENROLLMENTS:
                raise ValueError("enrollment count exceeds adapter bound")
            normalized_students.append(normalized)
            for student in normalized:
                roster.setdefault(student, []).append(vertex)

        reasons: dict[tuple[int, int], ExamConflictReason] = {}
        pair_emissions = 0
        for student in sorted(roster):
            vertices = roster[student]
            # A single very large cohort can otherwise spend unbounded time before
            # the native edge limit is reached.  This is a conservative admission
            # check; duplicate pairs from other students do not earn extra budget.
            pairs = len(vertices) * (len(vertices) - 1) // 2
            pair_emissions += pairs
            if pairs > MAX_CONFLICTS or pair_emissions > MAX_PAIR_EMISSIONS:
                raise ValueError("student cohorts exceed bounded conflict-generation work")
            for offset, left in enumerate(vertices):
                for right in vertices[offset + 1:]:
                    edge = (left, right)
                    reasons.setdefault(edge, ExamConflictReason("shared_student", student))
                    if len(reasons) > MAX_CONFLICTS:
                        raise ValueError("conflict edge count exceeds native bound")

        exam_at = {name: index for index, name in enumerate(exams)}
        for item in exclusions:
            if isinstance(item, (str, bytes)):
                raise ValueError("exclusion must contain two exact exam identifiers")
            row = tuple(item)
            if len(row) != 2 or any(type(value) is not str for value in row):
                raise ValueError("exclusion must contain two exact exam identifiers")
            try:
                left, right = exam_at[row[0]], exam_at[row[1]]
            except KeyError as error:
                raise ValueError("exclusion names an unknown exam") from error
            if left == right:
                raise ValueError("an exam cannot exclude itself")
            edge = (min(left, right), max(left, right))
            reasons.setdefault(edge, ExamConflictReason("explicit_exclusion"))
            if len(reasons) > MAX_CONFLICTS:
                raise ValueError("conflict edge count exceeds native bound")

        edges = tuple(sorted(reasons))
        problem = ExactProblem.create(len(exams), len(periods), edges, masks)
        return cls(
            problem=problem,
            exam_ids=exams,
            period_ids=periods,
            students=tuple(normalized_students),
            conflict_reasons=tuple(reasons[edge] for edge in problem.edges),
        )

    @staticmethod
    def _bounded_snapshot(values: Iterable[str], limit: int, name: str) -> tuple[str, ...]:
        try:
            iterator = iter(values)
        except TypeError as error:
            raise ValueError(f"{name}s must be iterable") from error
        copied: list[str] = []
        for value in iterator:
            if len(copied) >= limit:
                raise ValueError(f"{name} count exceeds adapter bound")
            copied.append(value)
        return tuple(copied)

    @staticmethod
    def _check_identifiers(values: tuple[str, ...], kind: str) -> None:
        for value in values:
            if type(value) is not str or not value or len(value) > MAX_IDENTIFIER_CHARS:
                raise ValueError(
                    f"{kind} identifiers must be nonempty exact strings of at most "
                    f"{MAX_IDENTIFIER_CHARS} characters"
                )

    @property
    def candidates(self) -> tuple[tuple[str, ...], ...]:
        """Return each exam's admitted candidate periods in colour order."""
        return tuple(
            tuple(self.period_ids[index] for index in range(len(self.period_ids)) if mask >> index & 1)
            for mask in self.problem.masks
        )

    @property
    def info(self) -> dict:
        shared = sum(reason.kind == "shared_student" for reason in self.conflict_reasons)
        return {
            "schema": "spectra.exam_repair.model.v1",
            "exams": self.problem.n,
            "periods": self.problem.palette,
            "conflicts": len(self.problem.edges),
            "shared_student_conflicts": shared,
            "explicit_exclusions": len(self.problem.edges) - shared,
            "enrollments": sum(map(len, self.students)),
            "scope": "one/two pre-approved periods; shared students and explicit exclusions only",
        }

    def freeze_event(
        self,
        *,
        closed_periods: Iterable[str] = (),
        fixed: Iterable[Sequence[str]] = (),
        allowed: Iterable[Sequence[object]] = (),
    ) -> tuple[tuple[int, int], ...]:
        """Translate current closures/pins into one canonical exact query.

        ``fixed`` rows are ``(exam_id, period_id)``.  ``allowed`` rows are
        ``(exam_id, iterable_of_period_ids)``.  Multiple sources of restriction
        are intersected.  A zero resulting mask is retained so the exact service
        can return a portable empty-domain proof instead of raising.
        """
        period_at = {name: index for index, name in enumerate(self.period_ids)}
        exam_at = {name: index for index, name in enumerate(self.exam_ids)}
        closed = tuple(closed_periods)
        if any(type(name) is not str for name in closed) or len(set(closed)) != len(closed):
            raise ValueError("closed periods must be unique exact strings")
        try:
            closed_mask = sum(1 << period_at[name] for name in closed)
        except KeyError as error:
            raise ValueError("closed-period event names an unknown period") from error

        effective = [mask & ~closed_mask for mask in self.problem.masks]
        fixed_seen: set[int] = set()
        for item in fixed:
            if isinstance(item, (str, bytes)):
                raise ValueError("fixed rows must be exact exam/period string pairs")
            row = tuple(item)
            if len(row) != 2 or any(type(value) is not str for value in row):
                raise ValueError("fixed rows must be exact exam/period string pairs")
            try:
                vertex, period = exam_at[row[0]], period_at[row[1]]
            except KeyError as error:
                raise ValueError("fixed row names an unknown exam or period") from error
            if vertex in fixed_seen:
                raise ValueError("an exam appears more than once in fixed rows")
            fixed_seen.add(vertex)
            effective[vertex] &= 1 << period

        allowed_seen: set[int] = set()
        for item in allowed:
            if isinstance(item, (str, bytes)):
                raise ValueError("allowed rows must contain an exam and a period iterable")
            row = tuple(item)
            if len(row) != 2 or type(row[0]) is not str or isinstance(row[1], (str, bytes)):
                raise ValueError("allowed rows must contain an exam and a period iterable")
            try:
                vertex = exam_at[row[0]]
            except KeyError as error:
                raise ValueError("allowed row names an unknown exam") from error
            if vertex in allowed_seen:
                raise ValueError("an exam appears more than once in allowed rows")
            allowed_seen.add(vertex)
            try:
                choices = tuple(row[1])
            except TypeError as error:
                raise ValueError(
                    "allowed rows must contain an exam and a period iterable"
                ) from error
            if any(type(name) is not str for name in choices) or len(set(choices)) != len(choices):
                raise ValueError("allowed period identifiers must be unique exact strings")
            try:
                mask = sum(1 << period_at[name] for name in choices)
            except KeyError as error:
                raise ValueError("allowed row names an unknown period") from error
            effective[vertex] &= mask

        return tuple(
            (vertex, mask)
            for vertex, (original, mask) in enumerate(zip(self.problem.masks, effective))
            if mask != original
        )

    def assignment_from_labels(
        self,
        labels: bytes,
        query: Iterable[Sequence[int]] = (),
    ) -> tuple[tuple[str, str], ...]:
        """Decode a complete checked label vector into stable exam/period rows."""
        frozen = self.problem.freeze_query(query)
        if not self.problem.check(labels, frozen):
            raise ValueError("labels do not satisfy the admitted repair problem and event")
        return tuple((exam, self.period_ids[labels[index]])
                     for index, exam in enumerate(self.exam_ids))

    def check_assignment(
        self,
        assignment: Mapping[str, str] | Iterable[Sequence[str]],
        query: Iterable[Sequence[int]] = (),
    ) -> bool:
        """Check a named assignment against original adapter inputs only."""
        try:
            rows = tuple(assignment.items()) if isinstance(assignment, Mapping) else tuple(
                tuple(row) for row in assignment
            )
            exam_at = {name: index for index, name in enumerate(self.exam_ids)}
            period_at = {name: index for index, name in enumerate(self.period_ids)}
            labels = [-1] * self.problem.n
            for row in rows:
                if len(row) != 2 or any(type(value) is not str for value in row):
                    return False
                vertex = exam_at[row[0]]
                if labels[vertex] != -1:
                    return False
                labels[vertex] = period_at[row[1]]
            if any(value < 0 for value in labels):
                return False
            return self.problem.check(bytes(labels), query)
        except (KeyError, TypeError, ValueError, OverflowError):
            return False

    def explain_unsat(
        self,
        query: Iterable[Sequence[int]],
        certificate: bytes,
    ) -> dict:
        """Verify and translate an SBP1 contradiction into named repair steps."""
        frozen = self.problem.freeze_query(query)
        if not verify_answer(self.problem, frozen, "UNSAT_VERIFIED", b"", certificate):
            raise ValueError("certificate does not prove this repair event infeasible")
        kind, left, right = decode_proof(certificate, self.problem.n)
        effective = list(self.problem.masks)
        for vertex, mask in frozen:
            effective[vertex] &= mask
        if kind == 1:
            vertex = left[0]
            return {
                "schema": "spectra.exam_repair.proof.v1",
                "kind": "EMPTY_DOMAIN",
                "exam": self.exam_ids[vertex],
                "original_candidates": list(self.candidates[vertex]),
                "remaining_candidates": [],
            }
        return {
            "schema": "spectra.exam_repair.proof.v1",
            "kind": "FORCED_OPPOSITES" if kind == 2 else "BASE_CONTRADICTION",
            "left": self._explain_path(left, effective),
            "right": self._explain_path(right, effective),
        }

    def _literal(self, literal: int) -> dict:
        vertex, side = divmod(literal, 2)
        colors = tuple(index for index in range(self.problem.palette)
                       if self.problem.masks[vertex] >> index & 1)
        period = self.period_ids[colors[side]] if side < len(colors) else None
        return {
            "literal": literal,
            "exam": self.exam_ids[vertex],
            "exam_index": vertex,
            "period": period,
            "side": side,
        }

    def _explain_path(self, path: tuple[int, ...], effective: list[int]) -> dict:
        reasons = {edge: reason for edge, reason in zip(
            self.problem.edges, self.conflict_reasons
        )}
        nodes = [self._literal(literal) for literal in path]
        steps = []
        for before, after in zip(path, path[1:]):
            left, right = before // 2, after // 2
            if left == right:
                steps.append({
                    "kind": "SINGLETON_DOMAIN",
                    "exam": self.exam_ids[left],
                    "therefore_period": self._literal(after)["period"],
                })
                continue
            edge = (min(left, right), max(left, right))
            reason = reasons[edge]
            before_node = self._literal(before)
            after_node = self._literal(after)
            step = {
                "kind": "SHARED_STUDENT_CONFLICT" if reason.kind == "shared_student"
                        else "EXPLICIT_EXCLUSION",
                "from_exam": before_node["exam"],
                "from_period": before_node["period"],
                "to_exam": after_node["exam"],
                "therefore_period": after_node["period"],
            }
            if reason.student is not None:
                step["student"] = reason.student
            steps.append(step)
        root = nodes[0]
        root_mask = effective[root["exam_index"]]
        root["forced_by_event"] = (
            root["period"] is not None
            and root_mask == 1 << self.period_ids.index(root["period"])
        )
        return {"nodes": nodes, "steps": steps}


__all__ = ["ExamConflictReason", "ExamRepairModel"]
