# M16 — CPU-only verifier alignment and evidence repair

Status: protocol frozen before new M16 data/results. Base: 1e29cb10662cb83ba9e28e4d30164fc373a547a5.

## Objective and claim boundary

Repair ancestral data separation, make evaluator semantics explicit, and test whether candidate selection or candidate discovery explains search failures. A passing software test is not a scientific superiority result. This milestone cannot by itself establish frontier-lab seniority, hiring outcomes, cross-task usefulness, or a 100/100 project.

All computation is CPU-only. No GPU training/inference, external model API, or paid teacher is authorized by this protocol. Preserve M14/M15 historical evidence unchanged; corrections and replay results are new records.

## Integrity gates

1. Build content-based exclusion indexes covering all consumed M14 and M15 train, validation, development, confirmation and shift manifests, with source hashes and explicit roles. IDs alone cannot establish independence.
2. Detect the previously reported M15 confirmation/M14 training collision independently. Publish sensitivity analysis, never silently replace historical rows.
3. Generate new M16 sets only after applying ancestral exclusions; require exact input/target fingerprints and input-only identity, within-split uniqueness, and pairwise disjointness. State explicitly that exact separation is not symmetry-family/OOD separation.
4. Bind source checkpoint file and tensor hashes; freeze reasoner parameters. Do not retrain a substitute and claim byte identity.
5. Validate incoming records strictly. Corrupt/missing ancestral manifests, unknown evaluator targets, invalid budgets and incompatible model identities must fail closed.

## Experiment hierarchy

Development is explicitly exploratory; every attempt/result is retained. Use two exact accepted M14 FP64 cores (1401,2402) and the accepted M15 action directions/policies where available. All older data are development-only in M16.

The first new diagnostic uses ancestor-disjoint generated 4x4 unique Sudoku. Dataset seeds: fit 160101; validation 160102; development 160103; confirmation 160104; lower-clue shift 160105. Target sizes: fit 1024, validation 256, development 256, confirmation 512, shift 256. Clues: 6-10 except shift 4-5. No label-dependent deletion. Rejection for exact ancestral overlap/duplicate/invalid generation is logged and occurs before prediction.

Compare fixed candidate pools before closed-loop interventions. Pool metrics: exact-validity coverage, returned success, and selection regret conditional on coverage. Record per-core/per-instance rows and separate unique puzzle counts from repeated model evaluations. Reusing a pool is an analysis experiment, not a timing comparison.

Mandatory controls: original semantic exit K=4; fixed four-cycle recurrence; original learned-improvement proxy; symbolic terminal retention; absolute-validity/quality evaluation; equal-direction uniform versus learned-prior search where supported. A known valid incumbent is never discarded by the new checked selector. Charge all validation and baseline work to any runtime comparison.

Evaluator targets are distinct: one-cycle improvement, current validity/absolute quality, and success under a named continuation policy and remaining horizon. Do not silently interchange them. Any learned alternative must record its target, core hash, continuation policy, horizon, fit-state distribution, calibration data and fitting compute. Train-only data may include action-perturbed states; held-out labels may not train or select a candidate.

If auxiliary target fitting is undertaken, freeze its recipe and selected candidate before confirmation; retain all negative development results. The initial fitting cap is 300 AdamW steps, batch 64, learning rate 0.002, weight decay 0.01; no confirmation-driven tuning.

## Confirmation and measurements

Open fresh confirmation only after code/config/checkpoint selection is written into an immutable hash-bound freeze record. Persist the frozen record before generating confirmation. Reserve no hidden favorable seed. Any implementation repair after confirmation requires a new explicitly named confirmation and disclosure, not relabeling the old set untouched.

Timing is CPU B=1 complete solve, including decode, immutable-given restoration, all checks and control overhead. Separate warm-up, warm mean/median/p95, and cold start; interleave comparator order with a fixed random seed. No end-to-end claim from precomputed-input microbenchmarks. Pin CPU threads and report affinity/environment. Report physical energy as null when unavailable. No latency win inferred from logical work counts.

Use paired per-instance differences and account for shared puzzles and model seeds; two seeds are a bounded diagnostic, not reliable population-level seed inference. Preserve raw rows and deterministic summary regeneration.

## Decision rules

Integrity acceptance requires all lineage/contracts and fresh regression tests to pass. The bounded mechanism claim requires independently reproduced candidate-pool/selection decomposition and consistency on fresh confirmation. A useful CPU superiority claim additionally requires a competitive comparator, matched quality (within 1 percentage point), at least 20% lower mean complete-solve latency with a confidence interval excluding no improvement, and explicit tail disclosure. This numeric gate is a proposed local result threshold, not a hiring rubric.

4x4 Sudoku remains diagnostic. Cross-task, harder-family, external-replication and practical-importance claims remain OPEN until separately executed and evidenced. Strengthen baselines, reject noncompetitive mechanisms, and retain failures rather than manufacture a pass.

## Reproduction and durable evidence

Include executable audit/tests, exact environments, source/protocol hashes, dataset manifests, prediction/work/timing rows and freeze records. Public research artifacts must not depend only on short-lived Actions retention. Secrets, local credentials, and unrelated user data are never included in source/evidence archives.
