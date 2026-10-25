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

## Historical transfer studies

The [frozen selector-v10 tree](https://github.com/tugrapaydiner/SPECTRA/tree/9e1037f2b46780dc943804f8726cee042108d15a)
retains the transfer runners, native source, dedicated tests and manual workflow.
Use that complete checkout with its
[suite protocol](https://github.com/tugrapaydiner/SPECTRA/blob/9e1037f2b46780dc943804f8726cee042108d15a/experiments/compiled_query/ITC_EXAM_SUITE_PROTOCOL.md)
and [timing amendment](https://github.com/tugrapaydiner/SPECTRA/blob/9e1037f2b46780dc943804f8726cee042108d15a/experiments/compiled_query/ITC_EXAM_TRANSFER_REVALIDATION_20261019.md).
Its frozen `set10`/`set12` holdouts cannot be reused to tune the current v11
selector or treated as a new confirmation.

The earlier [2026-10-16 results](https://github.com/tugrapaydiner/SPECTRA/blob/8f7cee1a9c82e7da34cc2704ff7012290a554a91/experiments/compiled_query/ITC_EXAM_SUITE_RESULTS_20261016.md)
retain negative repetitions and artifact identities under their original policy.
Keep every failed cell and use each study's declared gates. These studies concern
bounded conflict-core repair, not full timetabling or operator deployment.
