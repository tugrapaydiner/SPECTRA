# Native-to-native deployment admission — frozen September 27, 2026

Base is merged main f98167b44610158d107164aee295656015a0ad89. The latest local stream-codec work 986613a remains a separate unchanged candidate. This protocol does not promote a new runtime or assign a hiring score.

## Questions and fixed panel

Compare the same fitted RBF SVC using unmodified upstream LIBSVM 3.37 (commit 6b907139084abf2da4d6d3cb10dc3b7eaffa2fbb), m2cgen 0.10.0-generated C, and existing SPECTRA execution. No Python-import-overhead advantage in the primary comparison. First use the original seed101 models for Wine, WDBC, Chess, Penguins, Titanic and Zoo; never refit or reselect those models. Add UCI Human Activity Recognition Using Smartphones, dataset240, using its original subject-disjoint training/test partitions and all561 provided features. These are precomputed features: raw sensor filtering/windowing/feature extraction is outside this experiment, and no smartphone or production deployment is claimed.

Fit exactly one new SVC(C=10,gamma='scale',kernel='rbf',probability=False,break_ties=False,cache_size=256), on the full provided training partition without added scaling or tuning. Also fit a fixed LinearSVC(C=1,dual='auto',max_iter=10000,random_state=20260927) as a task-level low-cost control. Save trained parameters and source/data identities before any final-test predictions. No adaptation based on test accuracy. Record training costs and convergence warnings separately. No GPU/pretrained model.

## Export and numerical contracts

No retraining to suit a backend. Export exact binary64 values, class ordering and compatible binary/multiclass tie conventions. Build LIBSVM from its unmodified source, with model input conversion handled in a separately identified wrapper. Reuse buffers outside prediction when their contents are not input-specific. All input-specific dense-to-node conversion is charged in the public complete-call comparison; additionally record prepared-node native-only scope where useful. Preserve LIBSVM's ordinary internal allocations, not an artificially slower subprocess per row. Generated C returns pair scores: decode votes using the fitted model's documented conventions. Compile native sources with the same strict noncontracted floating-point flags and O3. Explicit portable/AVX2 SPECTRA runs must not be pooled.

Generated-source/compiler costs are separate. Bound one m2cgen generation to120 seconds and one compilation to180 seconds, source64MiB and process RSS4GiB where enforceable. Keep any unsupported/failed/timed-out generated model as missing, not defeated. An unchanged model with different floating-point operation order can have slightly different margins; report every class disagreement and maximum score differences. Do not call task-output agreement bitwise or exact-real arithmetic.

## Measurement

On one pinned CPU core, after all validation and source freeze, compare whole prediction jobs at batch sizes1,32,256 over all retained/test examples, seven repetitions with fixed randomized backend order. Include one symmetric Python/native call per batch, input validation/conversion required by each backend and fresh label outputs. No loading, compilation or feature extraction in warm timers; report preparation/build size/cost separately. No hidden free baseline preprocessing. Validate every timed prediction against its already recorded backend result and compare all backends against frozen sklearn predictions. Record default SPECTRA, exhaustive and binary_stream explicitly; no posthoc best-profile promotion. Where generated code cannot build, continue complete other comparisons but mark admission incomplete.

## Interpretation and promotion

Report each model and workload separately, all slow cases and missing comparators. Useful native-speed admission requires at least1.20x over the fastest measured comparable native alternative on the new HAR workload, with no observed class change. Task-level admission additionally requires a one-percentage-point quality/cost comparison against the linear control, with all subject-level differences disclosed; a favorable speed ratio alone is not proof of task usefulness. These point gates are not statistical guarantees. No external reproduction or broad population claim follows from our runs. Do not change runtime defaults based on a failed or incomplete gate.
