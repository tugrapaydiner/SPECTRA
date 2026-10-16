"""Strict ITC 2007 parser and bounded conflict-core adapter contracts."""
from __future__ import annotations

import pytest

from spectra.itc2007_exam import parse_itc2007_exam


FIXTURE = """[Exams:3]
120, 1, 2
180, 2, 3
120, 9
[Periods:3]
01:01:2030, 09:00:00, 180, 0
01:01:2030, 14:00:00, 120, 5
02:01:2030, 09:00:00, 240, 0
[Rooms:1]
100, 0
[PeriodHardConstraints]
0, EXCLUSION, 2
1, AFTER, 0
[RoomHardConstraints]
2, ROOM_EXCLUSIVE
[InstitutionalWeightings]
TWOINAROW, 30
"""


def test_parser_preserves_complete_constraint_inventory():
    instance = parse_itc2007_exam(FIXTURE)
    assert len(instance.exams) == 3
    assert instance.exams[0].students == (1, 2)
    assert len(instance.periods) == 3
    assert instance.after == ((1, 0),)
    assert instance.exclusions == ((0, 2),)
    assert instance.room_exclusive == (2,)
    assert dict(instance.ignored_constraint_counts) == {
        "after": 1,
        "room_exclusive": 1,
        "room_assignment": 1,
        "period_penalties": 1,
        "institutional_weightings": 1,
    }
    assert len(instance.source_sha256) == 64


def test_conflict_core_requires_explicit_scope_acknowledgement():
    instance = parse_itc2007_exam(FIXTURE)
    candidates = ((0, 1), (0, 2), (1, 2))
    with pytest.raises(ValueError, match="would ignore"):
        instance.conflict_core(candidates)
    core = instance.conflict_core(candidates, acknowledge_ignored_constraints=True)
    # Exam 0/1 share student 2; exam 0/2 have explicit EXCLUSION.
    assert core.model.problem.edges == ((0, 1), (0, 2))
    assert core.model.candidates[0] == (
        "00:01:01:2030 09:00:00", "01:01:01:2030 14:00:00"
    )
    assert core.info["schema"] == "spectra.itc2007.conflict_core.v1"
    assert "not a complete ITC 2007 timetable" in core.info["claim_boundary"]


def test_duration_check_rejects_unapproved_short_period():
    instance = parse_itc2007_exam(FIXTURE)
    # Exam 1 lasts 180 minutes; period 1 lasts only 120.
    with pytest.raises(ValueError, match="does not fit"):
        instance.conflict_core(((0,), (1,), (2,)), acknowledge_ignored_constraints=True)


def test_conflict_core_rejects_noniterable_candidate_rows():
    instance = parse_itc2007_exam(FIXTURE)
    with pytest.raises(ValueError, match="period-index iterables"):
        instance.conflict_core(((0,), 1, (2,)), acknowledge_ignored_constraints=True)


@pytest.mark.parametrize("damage, message", [
    (FIXTURE.replace("[Exams:3]", "[Exams:4]"), "row count differs"),
    (FIXTURE.replace("[Periods:3]", "[Unknown:3]"), "unknown ITC section"),
    (FIXTURE.replace("120, 1, 2", "120, 1, 1"), "duplicate student"),
    (FIXTURE.replace("0, EXCLUSION, 2", "0, OTHER, 2"), "unsupported period relation"),
    (FIXTURE.replace("0, EXCLUSION, 2", "0, EXCLUSION, 2\n2, EXCLUSION, 0"),
     "duplicate period hard constraint"),
])
def test_parser_fails_closed_on_malformed_instances(damage, message):
    with pytest.raises(ValueError, match=message):
        parse_itc2007_exam(damage)
