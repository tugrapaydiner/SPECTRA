# Certified integer-tree execution — September 29, 2026

## Decision

The checkpoint now has a compact binary runtime, complete real-model replay and
an official native CatBoost comparator. All11,295 source decisions are preserved
by a source-bound owning fallback pipeline. Compact16-bit execution alone certifies
11,284 (99.9026%) and returns11 unresolved entries. All39 incorrect naive8-bit
decisions are refused by the certificate. These are source-model fidelity results,
not correct ground-truth classification of every row or a new intelligence gain.

Full-coverage performance is mixed: about26.9% lower cost on Letter versus native
CatBoost, but about5.6–6.6% higher on the other three tasks. The equal-task geometric
cost ratio is0.966756, only3.32% lower. Do NOT describe that as universal superiority.
No production default or release changes. Earlier model-quality failures remain.

## New source panel and exact verification

The prior trained CatBoost weights were not recovered. Instead, the four fixed
models are new, clearly identified fits under protocol commit7e65205 before fitting:
256 trees, depth6, learning rate.08, L2=3, no bootstrap, random strength0,
seed20260929, one CPU thread. All exports are locked before test predictions.
Source fitting consumed28.432088 CPU seconds total; compilation, evaluation,
verification and measurements are extra. No GPU/teacher/pretrained model.
All four test partitions were exposed by previous SPECTRA research. This is an
execution compatibility panel, not competitive hyperparameter tuning or a new holdout.
The source models score3735/4000,3369/3498,1794/2000,1730/1797 respectively; earlier
SPECTRA models have stronger accuracy. We do not replace them with these sources.

The unchanged exact-rational checkpoint encloses both source floating-point
roundoff and quantization residuals. No empirical threshold is trained. The native
engine accepts an index only when the resulting interval proves the source winner
under CONTRACT.md's numeric assumptions. Otherwise it emits-1. Full refinement
uses the original official CPU library on unresolved rows. The12 compiled objects
(8-bit,16-bit and original-float across four tasks) are rebuilt byte-for-byte by
the final auditor from their original JSON and exact compiler source.

| Task | Rows | q16 certified | q16 unresolved | Naive q8 incorrect, all refused |
|---|---:|---:|---:|---:|
|letter|4000|3991|9|29|
|pendigits|3498|3497|1|4|
|satellite|2000|2000|0|1|
|optdigits|1797|1796|1|5|

The optional correlated class-pair oracle and safe early exit remain tested.
They do not imply a faster universal path: checkpoint16 is often slower in this
panel. The compact model uses classwise residual bounds rather than a large pair
matrix. Original CPU/exported-C++ scores and exact rational sampled bounds were
checked separately; matching scores on samples does not prove every library backend.

## Three whole native measurements, all retained

The original row-major matrix has1092 cells and an equal-task full-refinement cost
ratio2.001312 versus the official library: an actual failure. A predeclared32-row
tree-major layout then produced a full1260-cell run with ratio1.022443. Finally a
single owning native composite, replacing two Python ownership leases but not any
fallback work, produced the complete1344-cell matrix reported below. No source
weights, quantization rule or certificate bound changed. No cells were pooled
across runs, and the original source/binaries remain for direct comparison.

The timer begins at original uint8 feature buffers and includes validation,
conversion, temporary allocation, predicates, tree computation, certification,
all actual fallback and fresh returned indices. Loading, certificate compilation,
preparation and original feature extraction are excluded. Seven fixed shuffled
repetitions, batches1/32/256, one pinned AMD EPYC9V74 core, strict arithmetic,
CPython3.13.5 and GCC14.2. Native CatBoost1.2.8 is an unmodified official library.
The C++ export is unmodified source with only a boundary wrapper, not the fastest
CatBoost implementation. No Python CatBoost import overhead is in the control.

### Final batch32 microseconds per row

|Task|Official CatBoost|C++ export|Old row-major q16|Tiled compact q16|Full owning refinement|
|---|---:|---:|---:|---:|---:|
|letter|2.315391|3.078635|2.965543|1.579823|1.693295|
|pendigits|1.225160|2.202535|2.518635|1.196055|1.299797|
|satellite|1.246064|2.226796|3.020193|1.231702|1.316109|
|optdigits|1.385566|2.389358|2.728586|1.257561|1.476905|

Compact-only timing has unresolved outputs and is NOT full-coverage classification.
The full owning column includes their packing, native reference calls and merged
outputs. At batch1 and256 the relationships differ; all cells and repeats are
retained. Whole-dataset per-row averages are not request-tail latency, production
traffic, energy measurements or a statistically general speed guarantee.

## Storage: compact model versus complete pipeline

|Task|Compact q16 file bytes|Original CBM bytes|CBM/compact ratio|
|---|---:|---:|---:|
|letter|859304|3569904|4.154|
|pendigits|335772|1473768|4.389|
|satellite|204948|950096|4.636|
|optdigits|334718|1471616|4.397|

Compact prepared state additionally holds validated topology and suffix bounds.
Full refinement retains BOTH files/models and the upstream shared library. It is
therefore not a4x full-pipeline storage or RAM reduction. The official shared
library is12,579,096 bytes; runtime mappings and interpreter allocations are separate.

Actual fresh-process high-water memory and setup (3 processes per task/mode,36 total):

|Task|Compact peak KiB|Official peak KiB|Full-coverage peak KiB|Compact setup ms|Full setup ms|
|---|---:|---:|---:|---:|---:|
|letter|18996–19000|38376–38640|40076–40384|13.318|22.494|
|pendigits|17332–17336|28432–28600|29028–29332|5.962|11.031|
|satellite|17104–17268|25712–25972|26512–26620|4.055|7.858|
|optdigits|17332–17336|28384–28532|29152–29180|5.886|10.574|

Setup includes full model construction and byte validation, followed by one fixed
verified query before reading the process's own VmHWM. Input is tiny; filesystem
caches may be warm. These measurements are neither worst-case batch RAM nor cold-
cache/power-loss guarantees. Full coverage uses slightly more memory, as expected.

## Validation and boundaries

Thirteen original rational-oracle tests pass unchanged. The new native suite has
58 tests,6 actual CatBoost-owning-pipeline tests and8 evidence-parser tests:85
unique top-level cases including the inherited oracle, not a sum across builds.
The72 native/pipeline/evidence cases pass on final portable and AVX2 UBSan runtime
builds; repeated builds and16 subtests are not extra top-level cases. The native
certificate fails closed on the explicit wrong8-bit case and tests ties, cancelled
sums, depth0 trees, generic geometries, tile tails, corrupted bytes, buffer leases,
reentrant close, concurrent calls and invalid rounding modes.

The final standard-library empirical audit checks every source/artifact identity,
all recorded decisions, complete1092/1260/1344 grids and derived ratios. It also
executes the literal hashed compiler bytes to rebuild all12 binaries. Sixteen
corrupted disposable record copies are rejected; initial copy setup omitted the
upstream binary and was fixed without changing any scientific observation. An
initial auditor rejected a legitimate2D Fortran-order saved score array; the audited
parser now converts its logical ordering explicitly, with C/Fortran equivalence
tests. Initial failures remain in the evidence rather than relabelled as passes.

Fresh extracted-style SDK checks replay both precisions and all full-coverage
outputs on portable and AVX2. SDK wrappers are relinked with relative dependencies
and tested; timing wrappers keep their original bytes/receipts. No numerical Python
framework is imported by deployment replay. No manual full historical suite,
Windows/ARM, ASan, old physical CPU, independent researcher, real application or
first-invention claim. Offline model verification and trusted native code remain
required. Hashes establish identities, not authenticity or the truth of clocks.

The source-audit mismatch from the older prototype experiment remains a separate
historical issue; this new source-bound tree delivery does not reconstruct the lost
local audit or its old empirical weights. No release/default/model is replaced.

Primary references: CatBoost evaluation library and C++ export documentation;
quantized tree research including FQTree(arXiv2608.12140). These are not claims
that this experiment is first or has defeated unmeasured hardware-specific systems.
