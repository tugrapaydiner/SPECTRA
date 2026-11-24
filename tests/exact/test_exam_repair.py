"""Public bounded exam-repair adapter contracts."""
from __future__ import annotations

import shutil
import struct
import sys

import pytest

from spectra.exam_repair import ExamRepairModel
from spectra.exact import ExactRuntime, build_exact_runtime


def path_model() -> ExamRepairModel:
    return ExamRepairModel.create(
        period_ids=("P0", "P1"),
        exam_ids=("A", "B", "C"),
        students=((1,), (1, 2), (2,)),
        candidate_periods=(("P0", "P1"),) * 3,
    )


def test_adapter_builds_deterministic_conflicts_and_named_metadata():
    model = ExamRepairModel.create(
        period_ids=("AM", "PM", "EVENING"),
        exam_ids=("MATH", "CHEM", "HIST"),
        students=((4, 2, 4), (2,), (9,)),
        candidate_periods=(("AM", "PM"), ("PM",), ("AM", "EVENING")),
        exclusions=(("CHEM", "HIST"),),
    )
    assert model.problem.edges == ((0, 1), (1, 2))
    assert tuple(reason.kind for reason in model.conflict_reasons) == (
        "shared_student", "explicit_exclusion"
    )
    assert model.conflict_reasons[0].student == 2
    assert model.candidates == (("AM", "PM"), ("PM",), ("AM", "EVENING"))
    assert model.info == {
        "schema": "spectra.exam_repair.model.v1",
        "exams": 3,
        "periods": 3,
        "conflicts": 2,
        "shared_student_conflicts": 1,
        "explicit_exclusions": 1,
        "enrollments": 4,
        "scope": "one/two pre-approved periods; shared students and explicit exclusions only",
    }


def test_event_intersects_closures_pins_and_allowed_periods():
    model = path_model()
    query = model.freeze_event(
        closed_periods=("P1",),
        fixed=(("B", "P1"),),
        allowed=(("C", ("P0",)),),
    )
    assert query == ((0, 0b01), (1, 0), (2, 0b01))
    with pytest.raises(ValueError, match="unknown period"):
        model.freeze_event(closed_periods=("P2",))
    with pytest.raises(ValueError, match="more than once"):
        model.freeze_event(fixed=(("A", "P0"), ("A", "P1")))
    with pytest.raises(ValueError, match="period iterable"):
        model.freeze_event(allowed=(("A", 7),))


def test_named_assignment_is_checked_against_event_and_conflicts():
    model = path_model()
    query = model.freeze_event(fixed=(("A", "P0"), ("C", "P0")))
    labels = b"\x00\x01\x00"
    assert model.assignment_from_labels(labels, query) == (
        ("A", "P0"), ("B", "P1"), ("C", "P0")
    )
    assert model.check_assignment({"A": "P0", "B": "P1", "C": "P0"}, query)
    assert not model.check_assignment({"A": "P0", "B": "P0", "C": "P1"}, query)
    assert not model.check_assignment({"A": "P0", "B": "P1"}, query)


def test_portable_proof_translates_to_shared_student_chain():
    model = ExamRepairModel.create(
        period_ids=("P0", "P1"),
        exam_ids=("A", "B"),
        students=((17,), (17,)),
        candidate_periods=(("P0", "P1"), ("P0", "P1")),
    )
    query = model.freeze_event(fixed=(("A", "P0"), ("B", "P0")))
    # A=P0 implies B=P1; the independently forced B=P0 reaches the opposite.
    certificate = struct.pack("<4sIII3I", b"SBP1", 2, 2, 1, 0, 3, 2)
    record = model.explain_unsat(query, certificate)
    assert record["kind"] == "FORCED_OPPOSITES"
    assert record["left"]["nodes"][0]["forced_by_event"] is True
    assert record["left"]["steps"] == [{
        "kind": "SHARED_STUDENT_CONFLICT",
        "from_exam": "A",
        "from_period": "P0",
        "to_exam": "B",
        "therefore_period": "P1",
        "student": 17,
    }]
    with pytest.raises(ValueError, match="does not prove"):
        model.explain_unsat((), certificate)


def test_empty_domain_proof_names_the_closed_exam():
    model = path_model()
    query = model.freeze_event(closed_periods=("P0", "P1"))
    certificate = struct.pack("<4sIIII", b"SBP1", 1, 1, 0, 0)
    assert model.explain_unsat(query, certificate) == {
        "schema": "spectra.exam_repair.proof.v1",
        "kind": "EMPTY_DOMAIN",
        "exam": "A",
        "original_candidates": ["P0", "P1"],
        "remaining_candidates": [],
    }


def test_adapter_rejects_ambiguous_or_out_of_scope_input_shapes():
    with pytest.raises(ValueError, match="one or two"):
        ExamRepairModel.create(("P",), ("A",), ((),), ((),))
    with pytest.raises(ValueError, match="unique"):
        ExamRepairModel.create(("P", "P"), (), (), ())
    with pytest.raises(ValueError, match="cannot exclude itself"):
        ExamRepairModel.create(("P",), ("A",), ((),), (("P",),), exclusions=(("A", "A"),))
    with pytest.raises(ValueError, match="period identifier count exceeds"):
        ExamRepairModel.create((f"P{i}" for i in range(65)), (), (), ())


def _compiler_available() -> bool:
    return bool(shutil.which("cl" if sys.platform == "win32" else "g++"))


@pytest.mark.skipif(sys.platform not in ("linux", "win32") or not _compiler_available(),
                    reason="supported native compiler unavailable")
def test_adapter_runs_through_public_native_api(tmp_path):
    model = path_model()
    runtime = ExactRuntime(build_exact_runtime(tmp_path / "native"))
    sat_query = model.freeze_event(fixed=(("A", "P0"), ("C", "P0")))
    unsat_query = model.freeze_event(fixed=(("A", "P0"), ("B", "P0")))
    with runtime.prepare(model.problem, mode="scc") as session:
        sat = session.solve(sat_query)
        unsat = session.solve(unsat_query)
    assert sat.status == "SAT_VERIFIED"
    assert model.check_assignment(model.assignment_from_labels(sat.labels, sat_query), sat_query)
    assert unsat.status == "UNSAT_VERIFIED"
    explained = model.explain_unsat(unsat_query, unsat.proof)
    assert explained["kind"] == "FORCED_OPPOSITES"
    assert any(step.get("student") == 1
               for side in ("left", "right") for step in explained[side]["steps"])
