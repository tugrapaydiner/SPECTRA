# Integer feature interactions: complete result

## Decision

The primary full-NCA feature-mixing hypothesis FAILS: its final classifier is less
accurate than the newly matched diagonal model on all three exposed test partitions.
The secondary local-covariance model improves Letter by0.5 percentage points versus
that new diagonal control at about1.8% more cost, but does not improve Pendigits or
Satellite. The neighbor-label-agnostic control reaches the SAME Letter correct count.
Neither the primary nor secondary two-task quality/cost gate passes. No default,
release model or old result is replaced; this is not a breakthrough learning claim.

Compared with the PREVIOUS round's strongest learned Letter model,3928/4000, the
new local model3929/4000 adds only ONE net correct prediction. Calling the full20-case
new-diagonal difference a20-case improvement over our prior best would be wrong.
This round mainly establishes what larger metric capacity does and does not solve,
and removes the flat-table allocation barrier to compiled feature mixing.

## What changed

A full linear map learns feature relationships instead of only individual weights.
The map is rounded to small signed integers, an identity block is appended, and the
SVM is fitted using that projected map. A new SPINT001 model stores the transform,
raw support codes and original real-valued classifier coefficients. Inputs are not
silently snapped. No pretrained teacher, GPU, inference label lookup or ensemble.

The product-table kernel is part of TRAINING: K(S)=H[S>>b]*L[S&(2^b-1)] on an exact
integer distance S. This is not claimed bitwise equivalent to one direct exp call.
The two tables need O(sqrt(S_max)) entries, rather than O(S_max). Matrix construction,
query projection, shape checks, integer distances and all voting remain charged.
Diagonal controls get an exact model-only shortcut, with a forced-projection arm
retained. Unused double support-feature storage is released after validation; peak
preparation allocations and counted persistent storage remain distinct.

## Candidate selection and dependence

The original four-arm pilot has24 completed fits; an earlier21-cell interrupted
attempt is retained. The matched complete search contains432 choices: four families,
three tasks, three grouped training-only splits, twelve C/gamma settings. Full NCA
was inferior on all three validation panels. A separately frozen local-covariance
amendment adds two controls and216 choices, not a replacement for those failures.
An additional12-fit continuous-map pilot shows the main NCA failure is not obviously
caused solely by integer rounding. That is only a development diagnostic, not a
complete final float-model comparison. Do not infer that every continuous alternative
was excluded by these probes.

Uniform, diagonal NCA, full NCA, pooled within-class whitening, local same-label
covariance and local all-label covariance each receive equal SVM search opportunities.
Local all-label covariance still samples anchors/gallery with label stratification;
it is not wholly label-free. Each of18 selected models is refitted once on all
original training rows and hashed before its first evaluation prediction. Full
Satellite NCA reaches its100-iteration cap; it is recorded as nonconverged, not
silently optimized further. Selection and final source snapshots remain byte-bound.
The old inherited final-fit scope string says all12, but the actual inventory is18;
the count comes from the stored model/fit inventory, not that stale description.

All three official test partitions were ALREADY used in SPECTRA. Grouping exact
feature duplicates inside fitting/validation avoids that overlap, but creates no
new independent test set. Letter's380 development/test exact-feature overlaps remain,
with the nonoverlap stratum reported. No font/writer/geographic independence can be
inferred solely from nonidentical feature rows. No parameter or family was selected
using these final test predictions. Public protocol recording followed local
commits and some ongoing selection, so it is not external preregistration.

## All final model quality

| Classifier | Letter /4000 | Pendigits /3498 | Satellite /2000 |
|---|---:|---:|---:|
|Uniform|3911 (97.775%)|3435 (98.199%)|1829 (91.450%)|
|Diagonal NCA|3909 (97.725%)|3434 (98.170%)|1836 (91.800%)|
|Full NCA (primary)|3860 (96.500%)|3425 (97.913%)|1823 (91.150%)|
|Pooled whitening|3893 (97.325%)|3405 (97.341%)|1767 (88.350%)|
|Local same-label covariance (secondary)|3929 (98.225%)|3428 (97.999%)|1798 (89.900%)|
|Local all-label covariance (control)|3929 (98.225%)|3430 (98.056%)|1807 (90.350%)|

On Letter the secondary fixes39 new-diagonal errors and introduces19: +20/4000,
+0.5 points; exploratory paired p=.01193. Exact-feature group-bootstrap95% interval
is approximately[+.125,+.899] points. These are descriptive exposed-data statistics,
not independent confirmation, source-group guarantees or a multiple-comparison
adjusted causal conclusion. Relative to its all-label neighbor control it fixes11
and introduces11, for zero net gain. Thus labels in neighbor choice have not
established a benefit on the one improved task.

The original full-NCA primary worsens Letter by1.225 points, Pendigits by0.2573 and
Satellite by0.65 relative to its matched diagonal model. A secondary Letter point
success does not convert this into a passed primary or a transferable learner.

## Complete native cost comparison

One pinned AMD EPYC9V74 core, Linux, Python3.13.5, GCC14.2, strict noncontracted
AVX2. All rows, raw uint8 codes through fresh labels; projection, validity checks,
kernels, sparse sums, class scheduling and Python/native crossing are timed.
Model loading, table construction, fitting, compiler and original sensor/image
feature extraction are excluded. The old fixed linear/MLP controls are recompiled
and receive the same raw codes; no Python neural-layer dispatch handicaps them.

| Arm, batch32 us/row | Letter | Pendigits | Satellite |
|---|---:|---:|---:|
|uniform|29.940103|2.650562|12.035105|
|diagonal|23.872461|2.682945|11.461200|
|full|23.808493|2.951944|8.168852|
|whitening|19.904649|8.258921|9.719077|
|local_supervised|24.303203|4.494224|9.862118|
|local_unsupervised|25.776851|4.155496|9.914371|
|local_scalar|28.238511|5.022482|13.735391|
|local_exhaustive|116.323751|10.529284|18.703389|
|local_direct_exp|32.369089|5.746660|12.063737|
|diagonal_projected|35.340526|2.754695|11.822145|
|parent_uniform|24.712761|2.821889|11.221350|
|parent_nca|23.606321|3.374210|12.992720|
|linear|0.226802|0.218033|0.218309|
|mlp|6.827558|4.093544|4.558079|

The secondary Letter model costs24.303us versus23.872 for the new diagonal control
(ratio1.01804). It is not faster than the previous learned model23.606us, whose
3928 correct answers differ from its3929 by one. On Pendigits the secondary is
1.675x the new-diagonal cost with worse quality. Satellite secondary is faster but
loses1.9 accuracy points versus the newly selected diagonal. Report quality AND
latency rather than selecting whichever improves.

The scalar and exhaustive local arms, direct-exp diagnostic and forced projected
diagonal are retained causal execution controls. No ordinary m2cgen/native-LIBSVM
comparison for the newly trained product-kernel models was run here. The old fixed
MLP/linear models are useful controls, not an exhaustive tuning campaign proving
superiority to all competing learners. The prior old Pendigits model3444/3498 is
still more accurate than every newly fitted model in this round.

The complete matrix has882 recorded jobs and2,792,412 repeated checked predictions,
not that many independent samples. Seven repetitions do not establish production
P95, concurrent service latency or universal throughput. The first foreground run
was interrupted after261 jobs by tool timeout. It has no completed acceptance;
the entire identical882-job matrix was rerun separately and all cells kept. No
selective outlier removal, partial pooling or tuning after timing occurred.

## Table and whole-process memory costs

The local same-label models' actual compact table inventories are:

| Task | Actual entries | Actual table bytes | Counterfactual flat-table entries |
|---|---:|---:|---:|
|letter|2,769|22,152|1,474,876|
|pendigits|21,721|173,768|87,440,001|
|satellite|113,744|909,952|3,159,304,651|

The flat-table numbers are mathematical inventory requirements, NOT an allocated
baseline or measured gigabytes of RAM saved. The product definition changes the
rounding function compared with an old one-exp table. Table capacity is not total
classifier size; support vectors, transform, coefficients and worker state count.

Forty-five fresh isolated processes (three per listed task/model) measured their
own Linux VmHWM and preparation time, with no numerical Python packages loaded.
Filesystems may be warm. Full table preparation and transient validation buffers
are included in peak memory. The previous kernel's peak is a different trained
model, not a matched same-model memory proof.

| Task / arm | Peak self KiB range | Median preparation ms |
|---|---:|---:|
|letter / uniform|30244–30380|65.322|
|letter / diagonal|27172–27184|50.182|
|letter / full|27040–27184|50.765|
|letter / local_supervised|28300–28324|56.757|
|letter / parent_nca|23840–23968|37.639|
|pendigits / uniform|17416–17420|5.161|
|pendigits / diagonal|17412–17428|5.335|
|pendigits / full|17412–17556|5.335|
|pendigits / local_supervised|17668–17800|6.592|
|pendigits / parent_nca|19512–19528|5.425|
|satellite / uniform|20228–20356|16.690|
|satellite / diagonal|19972–20100|16.303|
|satellite / full|19844–19852|13.961|
|satellite / local_supervised|20356–20488|15.902|
|satellite / parent_nca|50656–50664|53.871|

Letter local counted persistent storage is lower than the prior learned model,
but peak memory is HIGHER because preparation allocates temporary validation
representations. Satellite's compact tables give much lower measured footprint,
but the selected local model is less accurate. Neither tradeoff is hidden.

## Validation, costs and boundaries

All62 unique runtime/learner tests pass on the final AVX2, portable and UBSan
builds. Sixteen additional copied-evidence corruptions are rejected by a separate
standard-library auditor. These are62 repeated tests across builds, not186 new
cases. Both extracted SDK builds replay284940 predictions across18 models/five
execution modes on56988 model/input pairs over9498 underlying rows. It includes
empty and closed-session checks and imports no numerical framework.

The independent scalar-coordinate observer, using original ordered distance and
coefficient logic but direct calls for both product factors, matches8,924,460 pair
scores bit-for-bit. It does not use the candidate projection/table lookup code.
All final predicted labels match precomputed sklearn on the new fitted kernel.
The direct-exp diagnostic happens to change no retained labels, but that observation
is not universal equivalence and its floating scores need not match. Same-environment
FP/libm assumptions remain; no ground-truth-perfect, cross-library or real-exp proof.

The independent audit checks648 complete selection cells,882 timing cells, source
snapshots, input/group identities, chosen parameters, model locks, labels, confusions,
paired discordances and all timing/gate arithmetic. It does not unpickle a model,
import the learner, authenticate clocks, independently validate bootstrap dependence
assumptions or establish another research group's reproduction.

Completed pilots, the648-choice selection, eighteen final fits and twelve continuous
pilot controls consumed660.458 CPU seconds (11.008 CPU minutes), including their
recorded preparation/validation procedures. The interrupted21-cell pilot consumed
additional work with no complete total receipt. Imports, acquisition recovery,
compilation, tests, score audits, benchmarks and packaging are extra. No GPU or
pretrained model. The largest quadratic Gram scratch remains2.048GB; disk backing
is not linear-time training. The fixed model capacity is not merely a few learned
matrix coefficients: all support vectors/classifier parameters are part of it.

No new full historical SPECTRA suite, installed-production-wheel acceptance,
Windows/ARM execution, AddressSanitizer, new official benchmark or external
maintainer/user review is claimed. Dedicated GitHub CI checks the new contracts;
its exact-head completed outcome must be observed separately from local tests.
The repository's old data/results and production defaults are unchanged.

## Interpretation

The richer full NCA model failed for all tasks; its intended learning benefit is
not established. Local covariance has a small exposed-Letter result but is not
better than the matched all-label-neighbor control and adds only one correct answer
over the prior learned Letter model. This is not a substantial intelligence jump.
The useful systems capability is compilation of integer feature interactions with
bounded compact product tables, retaining exact execution of the NEW kernel.
Neither contribution is presented as a first invention. NCA and covariance/local
metric approaches have prior art; see README for primary references. Future work
needs a different learning diagnosis and fresh evidence, not retuning these tests.
