# M16 — Verifier-aligned CPU search: frozen investigation plan

Base: `1e29cb10662cb83ba9e28e4d30164fc373a547a5`.
Work branch: `research/verifier-aligned-cpu-search`. Never update main automatically.

## Purpose and claim discipline
Repair cross-milestone data separation and search-value semantics, isolate candidate coverage from selection quality, then test useful CPU-only improvements. Existing M14/M15 evidence remains historical and immutable; new work does not retrospectively turn development into confirmation. Code volume, passing tests and milestone completion do not establish research importance or a hiring level.

## Mandatory repairs
1. Ancestor-wide content-fingerprint exclusion with manifest identity, training/selection roles, and adversarial duplicate-ID tests. Frozen cores carry their consumed-data ancestry. Fail closed for missing ancestry. Audit M14/M15 overlap and retain corrected sensitivity calculations.
2. Explicit evaluator contracts: one-cycle improvement, current state quality, terminal validity and budget-conditioned solve probability are different targets. Reject semantically incompatible search objectives; legacy experimental reproduction must opt in explicitly.
3. Separate candidate-pool coverage, conditional selection reliability, and returned-answer correctness. Compare selectors on exactly the same pools before testing closed-loop search.
4. Reference-free exact validity is terminal authority when available. Preserve a validated incumbent; charge all checks, transitions, baseline computation and fallback work. No uncharged baseline and no claim that a guard alone proves a speedup.
5. Preserve actual failed results, environment, source/checkpoint/data hashes, complete-solve timing, mean/median/p95, peak-memory scope and unavailable physical-energy status.

## Development sequence
A. Reproduce the existing audit from exact retained source archives; run baseline tests.
B. Implement and adversarially test the repairs above. Use the already inspected M14/M15 data only as development/retrospective evidence.
C. Build fixed-pool selection and closed-loop experiments on new ancestor-disjoint data. Preregister concrete model/checkpoint identities, candidate policies, task families, training recipes and effect gates in a separate experiment freeze before opening any new confirmation set.
D. Investigate CPU performance only on faithful, useful execution paths. Preserve checked operator interfaces; avoid recurring immutable-weight validation in a validated handle. Benchmark against the strongest applicable simple method, not only our reference.
E. Attempt a second task family and harder instances with classical, simple restart/selection and competitive neural controls. Failed or incomplete cross-task experiments remain failed or incomplete.

## Resource and verification constraints
CPU only for training, inference and evaluation. Do not provision paid hardware or use GPU/API teachers. No GPU energy claims. Package energy stays null unless measured using a valid counter. Use bounded explicit per-command timeouts as failure safeguards, not as success gates. Avoid uncontrolled CI fan-out: one branch-specific CPU workflow, no automatic PR or merge. Save all executed experiment outputs, including failures.

## Advancement gate (not yet satisfied)
A meaningful positive capability/efficiency claim needs a frozen candidate, untouched family/instance split, strengthened alternatives, multiple model seeds where learned models are involved, and complete cost accounting. Target at least 20% lower mean CPU solve cost at a predeclared quality noninferiority margin, or a predeclared strict-success improvement at fixed deadlines; disclose p95 and memory. Two distinct task families and independently reproducible CPU-host measurements are required before making a general claim. Selecting a threshold after observing confirmation is prohibited.

## Status
PREREGISTERED INVESTIGATION PLAN. No M16 result, generalization claim, native speedup or 100/100 assessment has been earned yet.
