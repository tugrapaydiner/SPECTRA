# Proof-carrying bounded exam repair

`spectra.exam_repair` maps a narrow operational rescheduling problem to the
versioned `spectra.exact` API:

- each exam has one or two pre-approved candidate periods;
- exams sharing a student may not use the same period;
- explicit exam exclusions are also supported;
- current period closures, fixed moves, and allowed-period restrictions are
  intersected into an exact query;
- SAT returns a complete named assignment;
- UNSAT returns an independently verifiable proof that can be translated into
  named exams, periods, and a shared-student/exclusion chain.

```python
from spectra.exam_repair import ExamRepairModel
from spectra.exact import ExactRuntime, build_exact_runtime

model = ExamRepairModel.create(
    period_ids=("Monday AM", "Monday PM"),
    exam_ids=("MATH101", "CHEM101", "HIST101"),
    students=((11, 20), (20, 31), (31,)),
    candidate_periods=(("Monday AM", "Monday PM"),) * 3,
)
query = model.freeze_event(
    closed_periods=(),
    fixed=(("MATH101", "Monday AM"), ("CHEM101", "Monday AM")),
)

runtime = ExactRuntime(build_exact_runtime("exact-native"))
with runtime.prepare(model.problem, mode="scc") as session:
    answer = session.solve(query)

if answer.status == "SAT_VERIFIED":
    print(model.assignment_from_labels(answer.labels, query))
else:
    print(model.explain_unsat(query, answer.proof))
```

## Scope boundary

This is **bounded repair**, not full timetabling.  The adapter does not generate
candidate periods or model room capacity, room assignment, precedence,
coincidence, or soft objectives.  Those constraints must be incorporated before
admission or checked separately.

`spectra.itc2007_exam` provides a strict reader for the ITC-2007 examination
format.  Its `conflict_core` conversion preserves an immutable receipt of every
constraint class outside the bounded model and refuses conversion unless the
caller explicitly acknowledges those omissions.

The research transfer protocol is documented in
[`ITC_EXAM_TRANSFER_PROTOCOL.md`](../experiments/compiled_query/ITC_EXAM_TRANSFER_PROTOCOL.md).

## Prospective multi-instance evaluation

The selector-v2 policy and its `set10`/`set12` execution holdouts are fixed in
[`ITC_EXAM_SUITE_PROTOCOL.md`](../experiments/compiled_query/ITC_EXAM_SUITE_PROTOCOL.md).
The completed cross-platform measurements, retained negative repetitions and
artifact digests are published in the
[`2026-10-16 suite results`](https://github.com/tugrapaydiner/SPECTRA/blob/8f7cee1a9c82e7da34cc2704ff7012290a554a91/experiments/compiled_query/ITC_EXAM_SUITE_RESULTS_20261016.md).
The suite rotates measured engine order, preserves every failed cell, and gates
only the predeclared auto/raw overhead on holdout cases.  Passing that suite is
application-transfer evidence, not an operator deployment or full ITC-2007
optimization result.

An outside runner can reproduce the source, installed-wheel, freeze, execution and
offline-audit boundaries using the
[independent replication guide](../experiments/compiled_query/ITC_EXAM_REPLICATION.md).
Negative or conflicting replications are explicitly valid results.
