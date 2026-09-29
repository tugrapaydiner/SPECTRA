# Learned integer-metric classifiers — September 28, 2026 (Toronto)

## Decision

This round changes the learned classifier, not just the speed of an old function.
The positive diagonal neighborhood learner improves Letter from 3911/4000 to
3928/4000 against an independently tuned, equally refitted uniform-kernel control.
That is 17 fewer errors, +0.425 percentage points, at roughly unchanged measured
native inference cost. Pendigits and the pre-acquisition-frozen Satellite transfer
are slightly worse. **All original >=0.5-point / <=1.25x cost gates remain FAILED.**
Keep the learner and runtime experimental; do not change the production default.
No general-intelligence, world-first, or universal advantage claim follows.

## What was built

Learn positive diagonal weights from fitting labels using an NCA-style neighborhood
probability objective. Project to a bounded integer budget, then fit the SVM using
that projected kernel, not a continuous model later approximated at inference.
The query and support vectors are original uint8 feature codes. Their exact weighted
integer distance indexes a static RBF table. The distinct SPLMET01 format prevents
silently loading this classifier as an ordinary unweighted SVM.

The independent reference repeats coordinates according to the integer weights
and uses unchanged original binary64 RBF code. Bounded integer differences, squares
and sums are exactly representable, giving the same computed kernel arguments and
ordered pair scores in the tested FP/libm environment. See CONTRACT.md. This is
fidelity to the NEW learned function, not ground-truth correctness or exact-real
arithmetic on all platforms. Metric learning, NCA and integer kernels are established
techniques, not inventions attributed to this project.

## Selection fairness and test boundaries

Letter and Pendigits retain their original official partitions, but these tests
were already consumed by earlier SPECTRA research. They are held out from THIS
fitting/selection, not fresh independent confirmation. Same-feature duplicates are
grouped within selection folds; all official evaluation rows remain. Letter has
380 exact feature vectors repeated from development; the nonoverlap stratum is
reported separately. Pendigits has none. No lost prior model/result is counted.

Each arm (uniform, inverse variance, learned) receives the same fitting/validation
rows and nine C/gamma choices over three fixed training-only splits. Selection uses
pooled validation accuracy, fewer supports, then grid order. Every selected arm is
refitted once on all original training rows. This equalizes data and hyperparameter
search opportunities, not total CPU time: metric learning is additional work.
Fixed linear and MLP controls are retained, not given retrospective test tuning.
All final models and exports are hashed before their first evaluation predictions.

The first large-margin pilot suppressed important Letter features and failed badly.
Its six fits remain. The revised positive neighborhood objective was selected using
training/validation only. No parameter, metric rule or threshold was tuned after
these evaluation predictions. DEVELOPMENT.md preserves the chronology.

Satellite uses official UCI146 integer sensor features and the original4435/2000
split. Its additional domain/table projection, all settings and controls were
committed as4d151102482ef8f90b8dc2fb09b2f61e936e7e84 BEFORE successful acquisition,
fitting and test predictions. It is new to this continuation, but the public data
cannot reconstruct scene coordinates, so no geographic/time-independent deployment
claim is made. All36 features stay alive; the positive nonuniform weight budget64
was fixed in that prospective protocol. Uniform uses36 unit weights. Satellite has
no exact train/test feature-row duplicates, which does not establish spatial independence.

## Complete evaluation outcomes

| Classifier | Letter /4000 | Pendigits /3498 | Satellite /2000 |
|---|---:|---:|---:|
| Matched tuned uniform RBF | 3911 (97.775%) | 3435 (98.199%) | 1829 (91.450%) |
| Inverse-variance metric | 3903 (97.575%) | 3424 (97.885%) | 1836 (91.800%) |
| Label-trained integer metric | 3928 (98.200%) | 3432 (98.113%) | 1828 (91.400%) |
| Fixed linear control | 2787 (69.675%) | 3145 (89.909%) | 1632 (81.600%) |
| Fixed MLP control | 3786 (94.650%) | 3391 (96.941%) | 1686 (84.300%) |
| Fixed QDA control | not run | not run | 1616 (80.800%) |

Letter's candidate fixes31 baseline errors and introduces14: 89 errors become72,
a19.1% relative error reduction. The paired discordance test gives p=0.0161 and
the descriptive row-bootstrap95% interval is [0.10,0.75] percentage points.
These are exploratory statistics on a repeatedly used benchmark, not independent
confirmation or a multiple-testing-adjusted universal claim. The exact-feature
cluster interval is similar; feature grouping does not identify writers/fonts.

Pendigits fixes14 and creates17 errors (-0.0858 points); Satellite fixes15 and
creates16 (-0.05 points). Satellite inverse variance scores1836, above both learned
and uniform. Every outcome remains. The original0.5-point threshold is not rounded
down to turn Letter's0.425-point result into a pass.

The old historical Letter SVM scored3831/4000, but attributing the whole gain to the
new metric would be unfair: the new uniform model already scores3911 after stronger
selection/full-data refitting. The isolated metric difference is3911->3928 only.
The older Pendigits model still scores3444/3498, above ALL newly fitted nonlinear
models here. Its advantage remains; the new learned classifier is not promoted.

## Matched native runtime

One pinned AMD EPYC9V74 core, Linux, CPython3.13.5, GCC14.2, strict noncontracted
AVX2. Tables, models and weights are prepared outside the warm timer. Timed calls
start with original integer feature buffers and include shape/domain checks,
query kernels, all selected comparisons, crossing the Python/native interface,
and fresh labels. Original feature extraction, loading, fitting and compilation
are excluded. Comparisons are complete all-row jobs, not isolated kernel timings.

| Task, batch32 | Uniform compiled us/row | Learned compiled us/row | Learned/uniform |
|---|---:|---:|---:|
| letter | 25.626405 | 25.305312 | 0.987470 |
| pendigits | 2.963265 | 3.656797 | 1.234043 |
| satellite | 11.834334 | 14.264435 | 1.205343 |

Letter's1.25% point speed difference is small; three of seven paired repetitions
are slower. Call it roughly tied, not a robust speedup. Pendigits costs23.4% more
and Satellite20.5% more. All accuracy/cost gates remain false.

Letter's earlier lower-accuracy finite model takes10.770us/row in the same run;
the new98.2%-accuracy model costs25.305us, about2.35x as much. That quality/speed
tradeoff is not hidden behind the equal-quality training comparison. Pendigits's
old model is also both faster and more accurate than the new learned one.

The original matrix contained unfair NPZ-table reloads inside the sklearn timing
path. That partial run was stopped, marked REJECTED and retained. Tables were then
materialized with model loading; the ENTIRE unchanged eleven-arm matrix was rerun.
Only the complete corrected run supplies ratios. Query kernel construction remains
inside the sklearn control's timer. No candidate/model was changed in response.
See BASELINE_CORRECTION.md. This comparison is not unmodified native LIBSVM/m2cgen
on the new weighted models; those external alternatives were NOT remeasured.

Primary matrix462 cells, Satellite231:693 complete timing observations. There are
9498 underlying evaluation rows, not2.19million independent test examples. Seven
repetitions are not a production-tail or population-generalization guarantee.
The fixed MLP/linear controls use native arithmetic and fresh labels, not Python
layer dispatch. They are fixed controls, not exhaustive tuning of all competing
model families. Full ablations, all batch sizes and class confusions are retained.

## Training and resource costs

Two first-task pilot sweeps plus the162-choice selection, ten final primary fits,
81-choice Satellite selection and six final Satellite fits used263.615 CPU seconds
in aggregate (about4.39 CPU minutes), including the procedures' metric/matrix and
validation work. Imports, data acquisition, compilation, numerical audits, timing
and synthetic unit-test fits are additional. One numerical thread; no GPU,
pretrained weights, teacher or label API. Training still constructs quadratic Gram
matrices; disk backing avoids retaining extra copies but does not change complexity.
The largest scratch Gram file was2.048GB and was removed after use.

Letter support vectors fall7902->6988 (11.57%). The sixteen learned metric entries
are additional parameters; they do not make this a16-parameter classifier. Original
support values, real-valued coefficients, labels, table and runtime state all count.

27 fresh-process memory/setup observations (three per model/arm) use no numerical
frameworks and read their own Linux VmHWM. Letter uniform is25284-25412KiB versus
24380-24392KiB learned. Pendigits grows18660-18668 to19912-19964KiB. Satellite grows
36936-36964 to51068-51080KiB, principally because the weighted table has4161601
entries instead of2340901. Additional table storage is a genuine deployment cost,
not weight compression. Setup includes table construction; three samples with
warm filesystem caches do not establish universal setup advantages or RSS limits.

## Validation and what was not tested

64 unique focused tests pass, including54 native/model/buffer/domain contracts and
10 actual learner-gradient/projection tests. They repeat on portable/AVX2 and with
the changed native engine under UBSan. Counts are not added across builds. Separate
copies with16 altered selection, model, data, prediction and timing/gate records
are rejected by independent standard-library auditors; no original evidence changes.

All28494 SVM model/input pairs across the nine models match their own precomputed
sklearn predictions. A separately compiled unchanged-original observer matches
4462230 computed pairwise margins bit-for-bit. The observer uses repeated integer
coordinates, not the candidate's signature/table calculation. The full margin
arrays are not retained as multi-gigabyte artifacts: reproduction code, assertions,
counted digests and execution receipts are retained. Both extracted SDK variants
replay113976 SVM predictions (four modes) and20996 simpler-control predictions,
over the same9498 rows. They import no numerical framework.

No new full historical SPECTRA suite, installed-production-wheel, Windows/ARM,
AddressSanitizer, real deployment or outside researcher reproduction was executed
for this experiment. Prior unrelated CI acceptance is not assigned to this module.
The new dedicated CI runs the64 focused tests on Linux CPython3.11/3.13, portable
and UBSan builds; actual exact-head outcomes are recorded separately in the delivery.

## Interpretation and preservation

This is a working bridge from supervised learning to the finite-kernel compiler:
learned geometry can improve decisions while preserving efficient integer execution.
It is not yet a robust transferable learner. No model is made the default after
seeing the one favorable dataset; all three failed gates remain. The next change
needs a new diagnosis and fresh evaluation boundaries, not retuning these tests.

The complete local research checkout preserves all730 prior files byte-for-byte.
The focused GitHub integration instead starts from689-file main and adds the new
experiment plus four unchanged prior comparison prerequisites. Those are distinct
source views; neither silently overwrites the other. The old m2cgen/codec branches
and reports remain separate. No release/version/default or prior outcome is changed.

Primary references (not novelty claims):
- Goldberger, Hinton, Roweis, Salakhutdinov (2004), Neighbourhood Components Analysis:
  https://papers.nips.cc/paper_files/paper/2004/hash/42fe880812925e520249e808937738d2-Abstract.html
- UCI Statlog Landsat Satellite, Srinivasan1993, DOI10.24432/C55887, CC BY4.0:
  https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite
- UCI Letter, Frey1991, and Pen-Based Recognition of Handwritten Digits, Alimoglu1996:
  https://archive.ics.uci.edu/dataset/59/letter+recognition
  https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits
MIT applies to implementation, not a relicensing of external data or earlier work.
