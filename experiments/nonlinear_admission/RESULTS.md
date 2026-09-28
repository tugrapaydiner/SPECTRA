# Two-domain nonlinear admission — September 27, 2026 (Toronto)

## Decision

**ISOLET fails the predeclared one-percentage-point test margin. Gas drift passes
that comparative research screen, but is not deployment-ready.** The existing
SPECTRA engine executes both selected SVMs faster than unmodified native LIBSVM.
The optimized neural controls remain faster at batch32, so higher SVM quality
has a measured cost; this is not general classifier/runtime superiority.

No new model architecture, production inference change, release/default change,
external adoption or high-90s project grade follows. Both tasks remain reported.
Generated-C comparisons remain incomplete after all fixed attempts failed to
produce usable libraries. These failures are not treated as defeated competitors.

## Frozen selection and test boundaries

Protocol d98cc128de88a8765203560ee87ce9b39f34429f preceded acquisition and fitting.
ISOLET uses6238 original development rows and1559 official separate-speaker test
rows,617 supplied features,26 classes. Its80/20 internal stratified validation
can share speakers. No individual speaker IDs are inferred; test group0 in the
raw record denotes the aggregate official file, not a speaker identity.

Gas uses128 supplied features and6 classes: fit batches1..6 (5933 rows), validate
7..8 (3907), final refit1..8 (9840), test9..10 (4070). Batch9 has470 test rows;
batch10 has3600. Raw acoustic/sensor feature extraction is outside the study.
No exact test feature vector repeats a refit vector on either task, but that does
not eliminate speaker, concentration, batch or acquisition correlations.

The fixed per-task search has4 linear configurations,6 RBF SVM configurations,
2 MLP widths x3 seeds and2 boosted-tree configurations. Width is chosen by the
three-seed MLP mean, never the best seed. Every family uses training-fitted scaling.
All twelve final refits were frozen before either final-test prediction; no
post-test model/threshold/refit change occurred. Two ISOLET linear fits (C1,C10)
hit the unchanged120-second process cap. They remain failed attempts, not silently
excluded successes. The best-completed-linear comparison is therefore budget-limited.

| Validation-selected family | ISOLET validation | Gas validation |
|---|---:|---:|
| LinearSVC |95.1122%|85.5388%|
| RBF SVC |97.5962%|87.7911%|
| MLP256, three-seed mean |96.0470%|82.9195%|
| Boosted trees |95.1122%|75.2496%|

Selected C/gamma: ISOLET linearC0.01, SVMC10/gamma0.25/617; gas linearC1,
SVMC10/gamma1/128. Both choose MLP256 and15-leaf boosted trees. All warnings,
iterations, validation predictions and unsuccessful attempts are retained.

## Held-out quality: every selected model

| Model | ISOLET correct/1559 | Accuracy | Gas correct/4070 | Accuracy |
|---|---:|---:|---:|---:|
| Linear |1472|94.4195%|2767|67.9853%|
| SVM |1499|96.1514%|2998|73.6609%|
| Boosted trees |1475|94.6119%|2521|61.9410%|
| MLP seed101 |1491|95.6382%|2827|69.4595%|
| MLP seed202 |1489|95.5099%|2836|69.6806%|
| MLP seed303 |1486|95.3175%|2877|70.6880%|
| MLP mean, not ensemble |—|95.4886%|—|69.9427%|

ISOLET SVM's margin over the stronger linear/mean-MLP control is0.6628 points,
below the frozen1.0-point screen. Gas's margin is3.7183 points, passing the
validation and final-test screen; SVM also exceeds the pooled tree result.
The threshold is our research-admission rule, not a customer-derived quality
requirement. Standard SVC training is not a new SPECTRA learning contribution.

Gas results are not uniformly dominant. In batch9, SVM gets363/470 (77.2340%),
while trees get365/470 (77.6596%). In batch10, SVM gets2635/3600 (73.1944%),
linear2409/3600 (66.9167%) and MLPs2467/2475/2515 correct. Most pooled improvement
comes from batch10. All per-batch records and macro-F1 scores are in QUALITY.json.
Pooled SVM macro-F1 is0.961447 on ISOLET and0.740793 on gas.

73.66% absolute gas accuracy does not establish a useful field detector, safety
readiness or commercial rights. Two future batches, one official speaker-test
group and three MLP seeds do not establish population-wide statistical superiority.
No row-bootstrap interval is relabelled independent-speaker/batch uncertainty.

## Complete native comparison

One pinned AMD EPYC9V74 CPU core, Linux,Python3.13.5,NumPy2.3.5,sklearn1.8.0,
GCC14.2. Native components useO3/AVX2/no-fast-math/no-contraction. LIBSVM3.37 source
commit6b907139084abf2da4d6d3cb10dc3b7eaffa2fbb is unchanged. The existing SPECTRA
runtime/default and its exhaustive control share the same rebuilt library.
Our separate native linear/MLP loops are also compared with their original
sklearn/BLAS execution; custom C++ is not assumed fastest. The tree timing is
explicitly a framework control, not an optimized native-tree implementation.

All final rows, batch sizes1/32/256 plus scaler-inclusive batch32, twelve arms,
seven randomized whole-job repetitions:672 timing records,1891344 repeated
predictions from5629 distinct task test rows. Every timed output is checked.
Training, generation and compilation finished before timing. No discarded cells,
selective reruns, per-model fastest-scheduler selection or previous-host pooling.

| Prepared inputs, batch32 | ISOLET microseconds/row | Gas microseconds/row |
|---|---:|---:|
| Native LIBSVM, same SVM |2077.221|90.661|
| SPECTRA exhaustive, same SVM |603.907|33.246|
| SPECTRA default, same SVM |169.553|17.876|
| Native linear |4.711|0.473|
| sklearn linear |2.166|1.051|
| Native MLP, three independent models |126.470–175.366|17.434–26.238|
| sklearn/BLAS MLP, same three models |11.938–12.273|5.125–5.216|
| sklearn boosted trees |910.380|199.544|

SPECTRA is12.251x/5.072x faster than LIBSVM on the corresponding ISOLET/gas SVM.
It is3.562x/1.860x faster than its own exhaustive control. All seven batch32 paired
repetitions improve against those two SVM controls. This does not defeat the missing
generated-C competitor or other untested native engines. The cheaper neural model
still gives up some measured accuracy; speeds are not attached to another model's
quality. The large slowdown of our native MLP relative to BLAS remains in the table.

With fitted scaling charged at batch32, SPECTRA costs183.472/20.948 microseconds
per row, versus LIBSVM2073.681/93.284:11.302x/4.453x. The MLP controls cost
14.440–14.893/6.733–6.904. Other scopes and all repetitions remain in SUMMARY.json
and timings.jsonl. These are whole-job costs divided by row count, not service
latency, production P95, concurrency, sensor feature-extraction or full-file I/O.
Loading/build costs are outside warm timers; the two scopes are not subtracted
to infer negative preprocessing overhead. Seven repeats are not tail guarantees.

## Generated-code failures and bounded stronger controls

Unmodified m2cgen0.10.0 ISOLET generation times out at120 seconds. Gas generation
completes in56.47 process seconds, producing4447998 source bytes; O3 compilation
is killed after105.59 seconds. Before runtime timing, the comparator amendment
adds the already tested PR36 guard and a fixed O1 fallback, without changing models.

Guarded ISOLET generation completes in111.59 process seconds (110.15 inside export),
producing98275446 C bytes. Its O3 and O1 compilers are killed after40.14/39.87
seconds. Gas O1 is killed after105.24 seconds. All compiler caps remain180 seconds,
with4GiB address-space and a separate4GiB container limit. A killed compiler is
not a generation timeout, and a memory limit is not measured peak process RAM.

No generated library exists for either final SVM. The fixed feasibility order
never uses prediction timing to select a compiler variant. Original and fallback
failures, source text, package identities and compiler stderr are retained. We do
not claim universal inability to compile these models, nor replace failed original
m2cgen with an unrelated successful custom emitter. No native-all-alternatives
admission claim can be made from this incomplete comparison.

## Training, representation and verification

Thirty-four of36 development fits and all12 final refits completed. Completed
fit CPU time totals523.75 seconds; the two timed-out fits each consume their120s
wall allowance, with their CPU time not separately measured. Summed child wall
cost across all48 attempts is836.13 seconds including imports, validation and
serialization. Acquisition, tests, native/code generation and the660.29 seconds
of recorded runtime measurements are additional. No GPU/pretrained weights.

ISOLET's SVM has3512 support vectors and an18040457-byte export; gas has773
supports and822680 bytes. Linear controls have16068/774 parameters, and each
selected MLP164890/34566 parameters. These are stored FP64 model payloads and
file sizes, not parameter compression or whole-process RAM measurements.

Twenty-seven protocol/native tests pass (17 study/freeze,10 dense-control cases).
Four warnings arise from deliberately short synthetic MLP test fits; they are
retained. The independent audit verifies all672 timing records, frozen model/data
identities, selection rules, MLP means, test metrics, group boundaries and ratios.
Twelve corrupted copies are rejected. It never loads pickles or calls the model;
this is separate checking code, not outside-researcher replication or clock
attestation. All ten native final model instances preserve their original labels.

Eight independent decision receipts replay four fixed rows per SVM. This is NOT
an independent replay of every pair calculation for every example. Native LIBSVM
versus sklearn margins differ by at most2.35e-13 on this host; dense scores differ
by at most5.69e-14. No changed label is excused by numerical tolerance. All raw
model predictions and probes are retained. No fresh historical full-suite,
Windows/ARM execution, sanitizer, field trial or external adoption is claimed.

The installed production source is unchanged; all689 baseline files are preserved.
This additive research branch is separate from PR36 and earlier local exporter/
codec work. The original dataset/source attribution and ambiguous gas licensing
are in README.md. No release publication or automatic high-grade promotion.
