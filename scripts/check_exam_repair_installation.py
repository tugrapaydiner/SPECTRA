"""Exercise the installed exam-repair API without repository imports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from spectra.exam_repair import ExamRepairModel
from spectra.exact import ExactRuntime, build_exact_runtime, verify_answer
from spectra.itc2007_exam import parse_itc2007_exam

FIXTURE = """[Exams:3]
120, 7
120, 7, 9
120, 9
[Periods:2]
01:01:2030, 09:00:00, 180, 0
01:01:2030, 14:00:00, 180, 0
[Rooms:1]
100, 0
[PeriodHardConstraints]
[RoomHardConstraints]
[InstitutionalWeightings]
TWOINAROW, 1
"""


def check() -> dict:
    instance = parse_itc2007_exam(FIXTURE)
    core = instance.conflict_core(((0, 1), (0, 1), (0, 1)),
                                  acknowledge_ignored_constraints=True)
    model: ExamRepairModel = core.model
    sat_query = model.freeze_event(fixed=(("exam-0000", model.period_ids[0]),))
    unsat_query = model.freeze_event(fixed=(
        ("exam-0000", model.period_ids[0]),
        ("exam-0001", model.period_ids[0]),
    ))
    with tempfile.TemporaryDirectory(prefix="spectra-exam-install-") as directory:
        library = build_exact_runtime(Path(directory) / "native")
        runtime = ExactRuntime(library)
        with runtime.prepare(model.problem, mode="scc") as session:
            sat = session.solve(sat_query)
            unsat = session.solve(unsat_query)
            if sat.status != "SAT_VERIFIED" or unsat.status != "UNSAT_VERIFIED":
                raise AssertionError("installed exact runtime returned unexpected status")
            if not verify_answer(model.problem, sat_query, sat.status, sat.labels, sat.proof):
                raise AssertionError("installed SAT result failed replay")
            if not verify_answer(
                    model.problem, unsat_query, unsat.status, unsat.labels, unsat.proof):
                raise AssertionError("installed UNSAT result failed replay")
            assignment = model.assignment_from_labels(sat.labels, sat_query)
            explanation = model.explain_unsat(unsat_query, unsat.proof)
            info = session.info
    if tuple(exam for exam, _period in assignment) != model.exam_ids:
        raise AssertionError("installed repair omitted or reordered exams")
    if not model.check_assignment(assignment, sat_query):
        raise AssertionError("installed named assignment failed adapter replay")
    if explanation.get("kind") not in ("EMPTY_DOMAIN", "FORCED_OPPOSITES"):
        raise AssertionError("installed repair returned an unknown proof explanation")
    return {
        "schema": "spectra.exam_repair.installation_check.v1",
        "exams": model.problem.n,
        "periods": model.problem.palette,
        "conflicts": len(model.problem.edges),
        "sat_status": sat.status,
        "unsat_status": unsat.status,
        "proof_kind": explanation["kind"],
        "assignment_complete": True,
        "native_abi": info["native_abi"],
        "exact_api_version": info["exact_api_version"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = check()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
