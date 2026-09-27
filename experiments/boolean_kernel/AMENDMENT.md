# Default-path overhead correction, before replacement full measurement

The first partial native comparison completed all six smaller models and one HAR
repetition. Its command hit a 20-second tool limit while a later HAR repetition
was in progress. Completed records and the original runtime/source remain under
initial-partial evidence; uncompleted repetition timings were not available.
Those records are not pooled with the final complete rerun.

Inspection also found avoidable domain branches in the unselected generic
per-kernel hot path. The corrected implementation dispatches once per pair to a
separate Boolean preparation loop; the ordinary distance/exp loops retain their
original arithmetic without per-kernel Boolean-cache checks. No model, predicate,
threshold, generated C, external compiler flag or acceptance criterion changes.
The complete seven-model matrix will be rerun after this source is frozen. This
is an adaptive performance correction, not independent confirmation.

The corrected full HAR run also exceeded a 60-second tool call after repeats 0
and 1 completed. The interrupted repeat 2 had not published a record file. The
same frozen source/binaries resumed repeats 2..10, without replacing any completed
record. The final matrix contains every planned repeat exactly once. Interrupted
partial-repeat times were not recoverable and are not fabricated or counted.

The first local full regression invocation hit a separate 120-second tool limit
at roughly half its tests. Its partial log remains. The complete suite was launched
again with a retained exit-status file; only that completed run is acceptance.
These orchestration interruptions did not change the models or test assertions.

The restarted full suite initially waited on a torch-extension build lock left
by the terminated compiler. After confirming no compiler remained, the unchanged
recorded Ninja build was completed manually, then its orphan lock was removed.
The waiting suite continued normally and exited with 2,220 passes, one Windows-only
skip and 16 historical slow exclusions. This repairs a disposable compiler cache,
not a source/test change. The original interrupted log and recovery command remain.

The original numerical-audit attempt assumed every HAR file appeared in the inner
model-freeze map; HAR feature/label identities actually live in the original outer
evidence manifest. The checker was corrected to use those unchanged identities.
No model, features or expected predictions changed. Complete final replay passed.

Primitive exhaustive/generation-wrap validation and the final installed-wheel
check finished after the timing run, rather than all finishing before measurement.
The 62 source contracts and complete real-data numerical replay preceded the final
matrix; later acceptance used the same numerical source. This ordering deviation
is explicit and is not external preregistration or fresh confirmation.
