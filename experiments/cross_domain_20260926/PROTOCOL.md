# Frozen cross-domain executor panel

Base: PR30 head `6a1a76249badc6c1875fbdf5f54f24ff5f714312`. This protocol is published BEFORE acquiring the new datasets, fitting models or inspecting test predictions. No new performance, generalization or hiring result is claimed. Earlier failed scheduling gates remain failed. Main and published release bytes are not changed.

## Fixed panel

Five official UCI datasets, no replacement of unfavorable tasks: Wine (109, chemical classification), Vehicle Silhouettes (149, image-shape descriptors), Landsat Satellite (146, remote sensing), Human Activity Recognition Using Smartphones (240, inertial features), Sensorless Drive Diagnosis (325, motor-current features). Expected dimensions 13,18,36,561,48. Record actual file row counts rather than trusting contradictory webpage counts. These are modest public workloads, not a broad claim about all deployed classifiers.

Use provided train/test partitions for Landsat and HAR. HAR validation holds out 20% of training subjects via GroupShuffleSplit(random_state=20260926); never split overlapping subject windows across fitting/validation. Preserve subject-disjoint official test. Wine/Vehicle: stratified25% test, then stratified20% of remaining rows for validation, random_state=20260926. Sensorless: within each class, keep original order and use first60% fitting, next20% validation, final20% test (floor boundaries). This is not independently held-out motor/operating-condition testing: group identities are not supplied. Preserve original indices and duplicates. Describe spatial/temporal correlation and small test size, not just row-disjointness.

Cap fitting at4096 and validation at2048 rows with stratified sampling and random_state=20260926 AFTER partitioning; keep full test sets. StandardScaler fitted only on fitting rows, use the same binary64 transformed inputs across executors. Keep all original feature dimensions and class labels.

## Model selection before tests

Fit four RBF SVC configurations in order C={1,10}, gamma={1/D,0.25/D}; cache_size256, probability=False, break_ties=False. Select highest validation accuracy, earliest tie, no refit. Fit separate task controls: LogisticRegression(C=1,max_iter=2000); MLPClassifier(hidden_layer_sizes=(64,64),max_iter=200,batch_size=128,solver=adam,learning_rate_init=0.001,alpha=0.0001,early_stopping=False,tol=0,n_iter_no_change=201,random_state=20260926). Fixed-budget control, not an optimized state-of-the-art baseline or equal compute claim. Retain convergence warnings and per-fit CPU/wall costs. No GPU, pretrained model, test-guided tuning or resumed training. Freeze selected models, preprocessing, exports and timing implementation BEFORE test model calls.

## Executor arms and correctness

Primary is unchanged `beretta_cert` with tables requested (lossless fallback is allowed). Matched controls: same SPECTRA kernel/exhaustive with direct distances, same exact schedule/direct distances, exhaustive/tables, certified-knockout/direct. Add actual native LIBSVM3.37 and scikit-learn SVC on the SAME selected RBF model and same binary64 inputs. Export to LIBSVM text using round-trip decimal17 precision, same label ordering and coefficients. An exporter failure is a recorded failure, not a reason to drop the comparator. Preserve upstream source/COPYRIGHT and SHA256.

Full-test class-fidelity gate: candidate matches same-kernel exhaustive; separately compare all executor results to original sklearn and LIBSVM. Certificates validate discrete partial votes only; no proof of real-valued exp or labels. All output differences remain. No precision tolerance can excuse a changed class for an exact-executor claim. Diagnostic numerical fixes must be separately logged and not silently substituted into the frozen primary experiment.

## Timing and storage

For each task, choose min(256,Ntest) test positions by `default_rng(20260926).choice` without replacement, sort indices. All cases retained for accuracy. Eleven fixed-randomized repeats. Primary native batch-one comparison and secondary public complete API/batch64 are separate scopes. Native harness includes input validation, state reset, context/table work, scheduler, class output; model load, initial preprocessing, allocation of persistent harness buffers excluded. Public complete API includes identical StandardScaler transformation from original feature rows and fresh returned labels. Record no-op overhead without subtraction. Every timed result checked outside timer. No concurrent local training/compilation while timing. Measure preprocessing/cold setup separately; counted prepared storage is not RSS.

Compute per-task same-run ratios against BOTH native LIBSVM and strengthened same-kernel exhaustive/direct-schedule controls. Do not multiply older gains or select the faster representation from these test timings. Paired-input bootstrap95% intervals condition on fixed selected model, inputs and host. Eleven observations per case are not production P95. Keep all regressions; no arithmetic mean of unrelated speedups presented as universal performance. Multiworker/RSS is secondary if actually completed, otherwise missing.

## Gates and preservation

Complete primary gate: zero class mismatches on all tasks, >=1.25x median aggregate speedup over strengthened exhaustive on at least4/5 tasks, no >10% regression on the remaining task. Report native LIBSVM comparison separately; do not count gains only over Python. Task-control accuracy may demonstrate a different better operating point; do not confuse executor fidelity and model superiority. A failed gate remains failed.

Keep every attempt, frozen source identity, data/license/partition receipt, selected/rejected model result, raw timing and independent audit. A second checker must verify inventory, full-test labels, copied model identities, schedule order, medians and ratios without importing the benchmark analyzer. Do not change runtime/model selection after seeing formal-test timings. Development and future fixes must be clearly separated. One tested host is not cross-hardware or external reproduction. Use bounded CPU compute and explicit costs. No automatic90/100 or novelty claim.
