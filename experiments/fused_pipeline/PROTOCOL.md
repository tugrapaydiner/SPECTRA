# Tiled fused pipeline execution — prospective local protocol

Base: GitHub PR30 ca7b39f992a86d40540d18ccad15c57102715a47, tree
86a33a8ad48d9e4145df53d640fe33742beecaf8. No training/model selection.

## Hypothesis and implementation boundary

An optional compiled entry point can apply a validated preprocessing plan and call
the unchanged SVM worker directly, avoiding intermediate Python buffer wrappers
and bounding the transformed feature tile independently of the full batch size.
Keep existing transform/predict_many/certificate APIs and defaults unchanged.
Use at most 128 rows and 16,384 binary64 transformed elements per tile (128 KiB).
A caller may lower the row tile size, not increase the element bound.
The original 65,536-row / eight-million-element input and output caps still apply.

Use only exact built-in containers/scalars on the native route. Otherwise fall
back to the existing reference route without consuming iterators or invoking
custom scalar conversions twice. No native SVM numerical sources change. Native
calls may release the GIL only while operating on private tile/output storage.
Raw inputs must not be mutated until the request returns. Recheck raw structures
before dereference after a GIL release; a mutation must not yield unsafe indexing.
Each worker is locked for the complete request including fallbacks and close.
No partial prediction list is returned on any failure. This is not transactional
rollback of internal counters or a sandbox for callers supplying invalid pointers.

## Verification before measurement

Bitwise transform equality and exact label fidelity on all 18 existing frozen
pipelines. Mixed built-in/custom scalars, generator consumption, invalid schemas,
late invalid inputs, owner/worker close, interrupts and concurrent independent
workers require tests. Numerical fallbacks and first-index voting are unchanged.
An installed sdist-built wheel must execute the new path without ML dependencies.
Keep failed attempts. Do not call this new accuracy or independent replication.

## Fixed complete-request performance panel

Use all 18 frozen six-task pipelines. No model selection or retraining. For each
model, generate a deterministic request trace from every saved test row in seeded
shuffled order. Request sizes cycle 1, 1, 8, 32, 128. Repeat that trace five times
per arm, randomize arm order per repetition; preserve every per-request time.
Compare Python preprocessing/native, existing compiled two-stage/native, the new
fused path (128-row default), and specialized NumPy preprocessing/same native.
Raw Python rows to fresh labels include preprocessing, copying/packing, prediction,
and output materialization. Exclude file parsing/build/loading/domain extraction.
Match and record every output. No other local workload during timing; pin one CPU.
Report ratios of complete trace costs (including large and small requests),
separate size-one distributions, paired model summaries, all regressions. Empirical
quantiles of replayed local requests are NOT production service-latency guarantees.
Primary promotion gate: >=1.10x vs compiled two-stage on panel geometric mean,
no task-level >1.10 latency ratio, exact fidelity. If it fails, do not rewrite it.

## Memory stress, separately scoped

On one documented synthetic numeric plan/model use 512 features and 15,000 raw rows
within the existing element cap. This is a resource stress test, not accuracy or
representative latency data. Compare separate fresh subprocess peak self VmHWM,
Python/native allocation peaks if observable, expected output, and calculated
transformed-tile bytes. Store setup and teardown outside the timed prediction.
Include result-list/input storage in process RSS. Never describe a 128 KiB tile
as 128 KiB total process memory. No cross-host guarantee from one local host.

## Delivery

Commit tested implementation to the existing feature PR with real remote ancestry;
no force-push/version replacement. Final-head CI must be separately observed.
Candidate binary modules remain ABI-specific and require a trusted native runtime.
