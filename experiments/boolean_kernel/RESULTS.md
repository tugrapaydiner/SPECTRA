# Exact Boolean-domain execution: completed local result

September 27, 2026. Base: c9f8bb6a2ca30d6840c69b463fd6dc1ea046ad6f,
now merged through PR34. This is opt-in execution of existing SVMs, not a new
classifier, training result, general native superiority or high-90s hiring claim.

## Result and mechanism

Only Chess-101 in the fixed seven-model panel meets the exact support-coordinate
predicate. All 73 coordinates are zero or one. Its 377 supports occupy two packed
64-bit words each. XOR/population counts replace 73 ordered scalar squared-distance
terms; the integer count is exactly the old binary64 result. The original gamma,
system scalar exponential and coefficient summation order remain. Inputs outside
the exact domain fall back without rounding. Signed zero is accepted, the smallest
nonzero subnormal is not. Explicit float32 input conversion still occurs first.

The additional lookup mode evaluates each encountered exp(-gamma*h) once WITHIN
one input. It does not cache across different or repeated input rows. On the799
Chess rows the normal exhaustive path evaluates377 kernels per row; lookup makes
11–21 scalar exponential calls, mean16.03755, with the rest exact reuse. The usual
support/term counters retain their original semantics; new diagnostic counters
separate exponential calls, lookup hits and754 packed-word comparisons per row.
Hamming/population count and memoization are established techniques. The contribution
here is a checked domain specialization with original-output fidelity, not a novel
distance identity. The method is not applicable to arbitrary continuous features.

## Full prepared-input comparison

AMD EPYC9V74, GCC14.2, Python3.13.5, one pinned core. Same strict O3/AVX2 settings
for fresh original SPECTRA, candidate, unmodified LIBSVM3.37 and identical previous
m2cgen C sources. All seven models, all4,374 model/input pairs, chunks1/32/256 and
11 randomized repetitions: **2,310 timing cells /1,443,420 repeated predictions**.
Each complete job includes checked buffer native calls and fresh result lists.
Preprocessing, model preparation, compilation and file I/O are outside this timer.

| Chess-101, batch32 | Median complete-job cost per row |
|---|---:|
| Unmodified native LIBSVM |40.597 us|
| Unmodified SPECTRA default |10.628 us|
| Unmodified SPECTRA binary_stream |8.778 us|
| Same-build off, default schedule |10.723 us|
| Same-build off, exhaustive |10.321 us|
| Matched generated C |7.975 us|
| Packed distance only, default schedule |4.210 us|
| Packed distance only, exhaustive |4.094 us|
| Packed plus per-input lookup, default schedule |**2.601 us**|
| Packed plus per-input lookup, exhaustive |2.635 us|

Lookup/default is **3.066x generated C**, **4.123x same-build off/default** and
**3.375x the prior explicit binary_stream path**. The locally fixed primary point
gate (at least1.20x vs both generated C and same-build off, with numerical fidelity)
passes. Do not multiply these ratios or present exhaustive/default as independent
trained models. There are zero paired Chess batch32 repeat regressions against
generated C or same-build off. Eleven repetitions are not production-tail evidence.

## Negative controls and costs

| Noneligible model, batch32 | Lookup option / same-build off ratio |
|---|---:|
| Wine |0.9470|
| WDBC |1.0261|
| Penguins |1.0086|
| Titanic |0.9791|
| Zoo |0.9818|
| HAR |1.0076|

All use the original arithmetic, not an approximation. Timing changes reflect
code layout, dispatch and noise, not new mathematical savings. HAR lookup/default
is74.565us versus the fixed DIFFERENT linear model's3.063us. The previous finding
that linear is the preferable measured HAR choice remains. Generated C still wins
Wine/Penguins/Titanic/Zoo; WDBC already favored SPECTRA and does not gain from this
specialization. HAR generated C remains missing from the prior bounded attempt.
No general superiority claim follows from one deliberately targeted known model.

For Chess, same-build shared prepared storage is227,664 bytes with off and233,696
with packed/lookup: **6,032 extra bytes**, not compression. Worker scratch is6,452
bytes off,6,468 packed and7,652 lookup. The original support representation remains
for arbitrary-query fallback. Added object fields also exist in off mode. These
are counted buffers/capacities, not peak RSS; no new process-memory reduction claim.
Preparation scanning/packing is additional work and has not received a complete
cold-start timing study in this round. No loading-time or energy benefit is claimed.

## Secondary raw-input pipeline

A separately fixed experiment uses all799 raw Chess rows, the SAME existing
compiled preprocessing for every backend, chunks1/32/256 and11 randomized repeats
(165 cells). The generated-C path receives the same transformed binary64 buffer;
no Python per-feature baseline or hidden uncharged preprocessing is used.

| Raw-input complete cost, us/row | Batch1 | Batch32 | Batch256 |
|---|---:|---:|---:|
| Original SPECTRA |15.797|10.384|10.230|
| New same-build off |16.038|10.567|10.417|
| Packed only |10.040|4.462|4.353|
| Packed plus lookup |**8.427**|**2.948**|**2.722**|
| Generated C, same preprocessor |11.515|7.931|7.908|

At batch32, lookup is **2.690x generated C**; single-row improvement is **1.366x**.
The timer includes preprocessing, allocation, checked calls and fresh labels but
not input-file parsing, initialization or compilation. This is secondary adaptive
engineering, not a new untouched corpus or a pooled extension to the primary run.

## Correctness and acceptance

The separately compiled original-source observer and new observer compare all
**4,071,918 distances,4,071,918 kernels and46,414 pair margins** bit-for-bit across
all4,374 model/input pairs. Six mode/table variants are repeated checks of the
same values, not six independent corpora. Only Chess activates the fast domain.
All reference predictions remain unchanged across the modes and schedules.

A separate integer/Python arithmetic probe checks75,602 population inputs, every
pair of six-bit vectors (4,096 pairs),64 complete ordered synthetic scores, and a
forced generation-marker wrap with deliberately poisoned cache entries. Tests
cover fractional/subnormal fallbacks, tails, class ties, invalid inputs, ownership,
raw-pipeline integration and up to4,096 features/128 classes.

- Full local fast suite: **2,220 passed**, one Windows-only skip,16 historical slow
  exclusions,two prior warnings. Includes573 SVM tests and62 new cases.
- The same573 SVM tests pass on portable and with the changed native SVM runtime
  instrumented under UBSan. The CPython preprocessing extension is unchanged and
  was not newly sanitizer-instrumented. No fresh ASan campaign is claimed.
- Actual wheel built through its source distribution: **58 framework-free installed
  checks** and146 package-source matches. The new Boolean path and general-input
  fallback both execute from installed source; old-library opt-in rejection passes.
- Independent standard-library primary audit checks all2,310 planned records,
  model/input identities, source/binary bindings and ratios.15 corrupted copies
  are rejected. Separate raw audit checks all165 secondary records.
- Original449-file history preservation audit passes. Frozen model/data bytes and
  prior unfavorable results are unchanged. Native generic shared code DOES change;
  the four historical numerical implementations are not rewritten.

Original protocol and implementation were committed locally before formal runs;
they were not externally preregistered. The first partial experiment exposed
unnecessary checks in the ordinary inner loop; those checks were moved to a
separate Boolean path, and the full final matrix was rerun. Tool-interrupted
partial repetitions are separately described in AMENDMENT.md, never pooled with
completed final records. A killed historical compiler left an orphan build lock;
its unchanged Ninja build was finished and the lock released before the full
suite completed. No tolerance or assertion was relaxed.

Current outside-researcher reproduction, real downstream use, new Windows/ARM
acceptance and production-tail tests must be reported separately when actually
observed. No new release/version/default or model accuracy is established here.
