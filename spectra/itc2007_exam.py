"""Strict ITC 2007 examination-timetabling reader and bounded-repair adapter.

The reader preserves the complete hard/soft section inventory.  Conversion to
:class:`spectra.exam_repair.ExamRepairModel` is intentionally explicit about the
smaller problem being solved: shared-student and EXCLUSION conflicts over one or
two pre-approved periods per exam.  AFTER, EXAM_COINCIDENCE, room assignment,
room exclusivity, and soft objectives are never silently interpreted as solved.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable

from .exam_repair import ExamRepairModel

_HEADER = re.compile(r"^\[([A-Za-z]+)(?::([0-9]+))?\]$")
_REQUIRED = (
    "Exams", "Periods", "Rooms", "PeriodHardConstraints",
    "RoomHardConstraints", "InstitutionalWeightings",
)


@dataclass(frozen=True, slots=True)
class ITCExam:
    duration: int
    students: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ITCPeriod:
    date: str
    time: str
    duration: int
    penalty: int

    @property
    def identifier(self) -> str:
        return f"{self.date} {self.time}"


@dataclass(frozen=True, slots=True)
class ITCRoom:
    capacity: int
    penalty: int


@dataclass(frozen=True, slots=True)
class ITCConflictCore:
    """One exact conflict-core model plus an immutable omitted-constraint receipt."""

    model: ExamRepairModel
    source_sha256: str
    ignored_constraints: tuple[tuple[str, int], ...]

    @property
    def info(self) -> dict:
        return {
            **self.model.info,
            "schema": "spectra.itc2007.conflict_core.v1",
            "source_sha256": self.source_sha256,
            "ignored_constraints": dict(self.ignored_constraints),
            "claim_boundary": (
                "bounded candidate-period repair only; not a complete ITC 2007 "
                "timetable, room assignment, precedence solution or soft-score result"
            ),
        }


@dataclass(frozen=True, slots=True)
class ITCExamInstance:
    exams: tuple[ITCExam, ...]
    periods: tuple[ITCPeriod, ...]
    rooms: tuple[ITCRoom, ...]
    after: tuple[tuple[int, int], ...]
    exclusions: tuple[tuple[int, int], ...]
    coincidences: tuple[tuple[int, int], ...]
    room_exclusive: tuple[int, ...]
    institutional_weightings: tuple[tuple[str, ...], ...]
    source_sha256: str

    @property
    def ignored_constraint_counts(self) -> tuple[tuple[str, int], ...]:
        values = (
            ("after", len(self.after)),
            ("exam_coincidence", len(self.coincidences)),
            ("room_exclusive", len(self.room_exclusive)),
            ("room_assignment", len(self.rooms)),
            ("period_penalties", sum(period.penalty != 0 for period in self.periods)),
            ("room_penalties", sum(room.penalty != 0 for room in self.rooms)),
            ("institutional_weightings", len(self.institutional_weightings)),
        )
        return tuple((name, count) for name, count in values if count)

    def conflict_core(
        self,
        candidate_periods: Iterable[Iterable[int]],
        *,
        acknowledge_ignored_constraints: bool = False,
    ) -> ITCConflictCore:
        """Build the exact shared-student/EXCLUSION repair core.

        Candidate rows contain one or two period indices and are assumed to have
        been pre-approved by the operator.  Period duration is still checked here.
        Any unmodelled ITC section requires an explicit acknowledgement flag and
        remains listed in the returned receipt.
        """
        ignored = self.ignored_constraint_counts
        if ignored and not acknowledge_ignored_constraints:
            names = ", ".join(name for name, _ in ignored)
            raise ValueError("ITC conflict-core conversion would ignore: " + names)
        copied_rows: list[tuple[int, ...]] = []
        try:
            for row in candidate_periods:
                if len(copied_rows) >= len(self.exams):
                    raise ValueError("candidate row count differs from exam count")
                if isinstance(row, (str, bytes)):
                    raise ValueError("candidate rows must be period-index iterables")
                copied_rows.append(tuple(row))
        except TypeError as error:
            raise ValueError("candidate rows must be period-index iterables") from error
        rows = tuple(copied_rows)
        if len(rows) != len(self.exams):
            raise ValueError("candidate row count differs from exam count")
        names = tuple(
            f"{index:02d}:{period.identifier}"
            for index, period in enumerate(self.periods)
        )
        candidates: list[tuple[str, ...]] = []
        for exam_index, (exam, row) in enumerate(zip(self.exams, rows)):
            if not 1 <= len(row) <= 2 or any(type(value) is not int for value in row):
                raise ValueError("every exam needs one or two exact period indices")
            if len(set(row)) != len(row) or any(not 0 <= value < len(self.periods) for value in row):
                raise ValueError("candidate period indices must be unique and in range")
            if any(self.periods[value].duration < exam.duration for value in row):
                raise ValueError(f"exam {exam_index} does not fit a candidate period")
            candidates.append(tuple(names[value] for value in row))
        exam_ids = tuple(f"exam-{index:04d}" for index in range(len(self.exams)))
        exclusions = tuple((exam_ids[left], exam_ids[right])
                           for left, right in self.exclusions)
        model = ExamRepairModel.create(
            names,
            exam_ids,
            (exam.students for exam in self.exams),
            candidates,
            exclusions=exclusions,
        )
        return ITCConflictCore(model, self.source_sha256, ignored)


def parse_itc2007_exam(text: str, *, max_chars: int = 64 << 20) -> ITCExamInstance:
    """Parse one canonical ITC 2007 examination instance without dependencies."""
    if type(text) is not str:
        raise ValueError("ITC input must be an exact string")
    if type(max_chars) is not int or not 1 <= max_chars <= 1 << 30:
        raise ValueError("max_chars must be a bounded positive exact integer")
    if len(text) > max_chars:
        raise ValueError("ITC input exceeds max_chars")
    normalized = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    sections: dict[str, tuple[int | None, list[tuple[int, str]]]] = {}
    current: str | None = None
    for line_number, raw in enumerate(normalized.split("\n"), 1):
        line = raw.strip()
        if not line:
            continue
        match = _HEADER.fullmatch(line)
        if match:
            name, count_text = match.groups()
            if name not in _REQUIRED:
                raise ValueError(f"line {line_number}: unknown ITC section {name}")
            if name in sections:
                raise ValueError(f"line {line_number}: duplicate ITC section {name}")
            expected = int(count_text) if count_text is not None else None
            if name in ("Exams", "Periods", "Rooms") and expected is None:
                raise ValueError(f"line {line_number}: {name} requires a count")
            if name not in ("Exams", "Periods", "Rooms") and expected is not None:
                raise ValueError(f"line {line_number}: {name} must not carry a count")
            sections[name] = (expected, [])
            current = name
            continue
        if current is None:
            raise ValueError(f"line {line_number}: data appears before a section header")
        sections[current][1].append((line_number, line))
    if tuple(sections) != _REQUIRED:
        raise ValueError("ITC sections are missing, duplicated or out of canonical order")
    for name in ("Exams", "Periods", "Rooms"):
        expected, rows = sections[name]
        if expected != len(rows):
            raise ValueError(f"{name} row count differs: expected {expected}, found {len(rows)}")

    exams: list[ITCExam] = []
    for line_number, line in sections["Exams"][1]:
        fields = _fields(line, line_number)
        if len(fields) < 1:
            raise ValueError(f"line {line_number}: empty exam row")
        duration = _nonnegative(fields[0], line_number, "exam duration", positive=True)
        student_values = tuple(_nonnegative(value, line_number, "student id")
                               for value in fields[1:])
        if len(set(student_values)) != len(student_values):
            raise ValueError(f"line {line_number}: duplicate student within an exam")
        exams.append(ITCExam(duration, tuple(sorted(student_values))))

    periods: list[ITCPeriod] = []
    for line_number, line in sections["Periods"][1]:
        fields = _fields(line, line_number)
        if len(fields) != 4 or not fields[0] or not fields[1]:
            raise ValueError(f"line {line_number}: period row must have date,time,duration,penalty")
        periods.append(ITCPeriod(
            fields[0], fields[1],
            _nonnegative(fields[2], line_number, "period duration", positive=True),
            _nonnegative(fields[3], line_number, "period penalty"),
        ))
    identifiers = tuple(period.identifier for period in periods)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("period date/time identifiers must be unique")

    rooms: list[ITCRoom] = []
    for line_number, line in sections["Rooms"][1]:
        fields = _fields(line, line_number)
        if len(fields) != 2:
            raise ValueError(f"line {line_number}: room row must have capacity,penalty")
        rooms.append(ITCRoom(
            _nonnegative(fields[0], line_number, "room capacity"),
            _nonnegative(fields[1], line_number, "room penalty"),
        ))

    after: list[tuple[int, int]] = []
    exclusions: list[tuple[int, int]] = []
    coincidences: list[tuple[int, int]] = []
    seen_hard: set[tuple[int, str, int]] = set()
    for line_number, line in sections["PeriodHardConstraints"][1]:
        fields = _fields(line, line_number)
        if len(fields) != 3:
            raise ValueError(f"line {line_number}: period hard constraint must have three fields")
        left = _exam_index(fields[0], line_number, len(exams))
        right = _exam_index(fields[2], line_number, len(exams))
        relation = fields[1]
        if relation not in ("AFTER", "EXCLUSION", "EXAM_COINCIDENCE"):
            raise ValueError(f"line {line_number}: unsupported period relation {relation}")
        if left == right:
            raise ValueError(f"line {line_number}: hard constraint repeats one exam")
        key = ((min(left, right), relation, max(left, right))
               if relation in ("EXCLUSION", "EXAM_COINCIDENCE")
               else (left, relation, right))
        if key in seen_hard:
            raise ValueError(f"line {line_number}: duplicate period hard constraint")
        seen_hard.add(key)
        if relation == "AFTER":
            after.append((left, right))
        elif relation == "EXCLUSION":
            exclusions.append((min(left, right), max(left, right)))
        else:
            coincidences.append((min(left, right), max(left, right)))

    room_exclusive: list[int] = []
    for line_number, line in sections["RoomHardConstraints"][1]:
        fields = _fields(line, line_number)
        if len(fields) != 2 or fields[1] != "ROOM_EXCLUSIVE":
            raise ValueError(f"line {line_number}: invalid room hard constraint")
        exam = _exam_index(fields[0], line_number, len(exams))
        if exam in room_exclusive:
            raise ValueError(f"line {line_number}: duplicate room-exclusive exam")
        room_exclusive.append(exam)

    weightings: list[tuple[str, ...]] = []
    for line_number, line in sections["InstitutionalWeightings"][1]:
        fields = _fields(line, line_number)
        if not fields[0] or any(not value for value in fields):
            raise ValueError(f"line {line_number}: empty institutional weighting field")
        # The exact weighting vocabulary is not interpreted by the conflict-core
        # adapter, but every numeric payload must still be canonical and bounded.
        for value in fields[1:]:
            _nonnegative(value, line_number, "institutional weighting")
        weightings.append(fields)

    return ITCExamInstance(
        exams=tuple(exams),
        periods=tuple(periods),
        rooms=tuple(rooms),
        after=tuple(after),
        exclusions=tuple(sorted(set(exclusions))),
        coincidences=tuple(sorted(set(coincidences))),
        room_exclusive=tuple(room_exclusive),
        institutional_weightings=tuple(weightings),
        source_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    )


def _fields(line: str, line_number: int) -> tuple[str, ...]:
    values = tuple(value.strip() for value in line.split(","))
    if any(not value for value in values):
        raise ValueError(f"line {line_number}: empty comma-separated field")
    return values


def _nonnegative(value: str, line_number: int, name: str, *, positive: bool = False) -> int:
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"line {line_number}: {name} must be a canonical nonnegative integer")
    number = int(value)
    if number >= 1 << 63 or (positive and number == 0):
        raise ValueError(f"line {line_number}: {name} is outside the supported bound")
    return number


def _exam_index(value: str, line_number: int, count: int) -> int:
    index = _nonnegative(value, line_number, "exam index")
    if index >= count:
        raise ValueError(f"line {line_number}: exam index outside declared exam bank")
    return index


__all__ = [
    "ITCExam", "ITCPeriod", "ITCRoom", "ITCExamInstance", "ITCConflictCore",
    "parse_itc2007_exam",
]
