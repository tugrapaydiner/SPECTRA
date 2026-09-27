# Native ownership and bound-receipt continuation — September 27, 2026

Baseline: e6f3d6b44ec53c158b9fa8021ddbadded482f1bb, source tree
101314c0cedea8d3ee1a0b3103419a1158e288df. The baseline's eleven cloud workflows
passed; new safe dispatch tests nevertheless reproduced twelve lifecycle failures.
They intercept dispatch before dereferencing a freed pointer; they are not crash
or exploit measurements. Preserve their original failed log and test snapshot.

Change Python ownership, not any native numerical source, model, scheduler default
or preprocessing arithmetic. Retain legacy, shared and fused APIs. Replay decision
receipts with a separate wire decoder, input/model binding, ordered binary64/libm
pair calculation and the existing integer vote condition. This extends verification;
it does not make the old deliberately narrower vote predicate an incorrect API.

Before measuring cost, require focused lifecycle/receipt/SVM contracts and actual
installed-wheel replay. Use all 18 unchanged retained pipeline bundles, no fitting.
Check all 4,281 model/input pairs, every class label, input precision and receipt.
Retain signed-zero, false-but-vote-valid witnesses, malformed models, wrong inputs,
wrong models, precision changes, work limits and missing numerical dependencies.

For performance, compare the exact baseline Python package and changed package
in one process against the SAME native SVM and preprocessing binaries. Every input
and output is identical. Fixed scopes: complete all-row jobs with prepared buffers
in size-1 and size-32 batches, plus full raw-input traces cycling 1/1/8/32/128.
Fifteen shuffled before/after repetitions, all 18 models, pin one core. Include
fresh result lists, new guards/capsules and ordinary API overhead. Never call a
positive outcome from the new optional verifier an inference speedup. Report
regressions without discarding or reweighting them to pass a performance gate.
This is a correctness change; no predeclared speedup is required. No tuning after
this measured comparison. Measurements use consumed data, not fresh accuracy.

Run portable and AVX2 regressions, the unchanged historical fast suite, a real
sdist-built wheel, and final-head CI. If a check is unfinished, say so. Do not
publish changed version-0.7.1 assets. Merge only after current-head acceptance.
