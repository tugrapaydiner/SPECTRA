# Source-verified integer tree inference — September 29, 2026

## Decision and scope

A source-verified 16-bit-first pipeline preserves all **11,295** recorded source
class decisions and improves batch32 complete-call cost by **1.36–2.01x** against
the faster of the two measured official CatBoost CPU libraries. It certifies
**11,284** inputs without a full-precision model call and sends the remaining
**11** to unmodified CatBoost. Single-row calls lose against the original exported
C++ control. This is an experimental execution result, not a better learned model,
new independent holdout, universal speedup, or production-default replacement.

The earlier interrupted native workspace and fitted trees were not recovered.
GitHub f816bb8 contained only three protocols. This continuation preserves the
uploaded exact-rational reference verbatim and performs a NEW fixed-recipe fit
and measured replay. The missing earlier 1092/1260-cell comparisons and old
5150/5151 count are not reassigned to the new files. New public protocols are
0d35521 (replay/acquisition) and e0bfe9f (model-frozen vector implementation).

## What was implemented

The compact compiler subtracts a common class-zero leaf offset, stores signed8/16-bit
class contrasts and power-of-two quantization scale, and computes exact-rational
outward bounds on quantization AND source binary64 arithmetic. A complete or
partial prediction is certified only if its score interval settles the source
winner. Otherwise its result is UNRESOLVED, not the unguarded quantized guess.
Optional early checkpoints include extrema of every unevaluated tree.

`VerifiedCompact` reconstructs the exact packed bytes from their original JSON,
not just a checksum. The binary loader separately checks numerical envelopes,
indices, geometry and buffer caps. No calibration examples, test-fitted confidence
threshold, distillation or changed leaf values define the proof. Certificates
are conditional on the explicit source arithmetic contract in CONTRACT.md; they
are not true-label proofs or a guarantee for every CatBoost backend and future
compiler. Source verification is expensive preparation, outside warm timing.

A single native owning pipeline can use compact16 then official fallback, or
compact8 then compact16 then fallback. The8-to16 path reuses leaf routes. Both
compact models must have identical source identity and routing. The trusted
original CBM is separately bound to its JSON in the source-binding experiment and
SDK manifest. The generic constructor does not independently convert arbitrary
CBM files back into JSON: supplying a different same-shaped original model violates
the fallback contract. File hashes prove identity only relative to trusted receipts.

Routing uses bounded32-row tiles. The optimized implementation transposes those
rows, vectorizes predicates and leaf-index formation, and keeps end-only integer
class sums in registers. Arithmetic and all certificate bounds are unchanged.
Repeated/constant predicates, depth>8 fallback, partial tiles and trailing vector
loads are covered. Extra guard storage is counted. General portable/scalar/early
paths remain available; source code and actual pre-optimization binaries are retained.

## Actual new source models

Four original training partitions, CatBoost1.2.8, 256 depth6 oblivious trees,
learning_rate.08, l2_leaf_reg3, bootstrap_type No, random_strength0,
random_seed20260929, thread_count1, MultiClass. Raw integer features, no selection,
early stopping, feature engineering or test-based changes. All four CBM/JSON/C++
exports are locked before the new test predictions. These are compatibility
workloads, not carefully tuned best classifiers.

| Task | Source correct / rows | Accuracy |
| letter | 3,735 / 4,000 | 93.375% |
| pendigits | 3,369 / 3,498 | 96.312% |
| satellite | 1,794 / 2,000 | 89.700% |
| optdigits | 1,730 / 1,797 | 96.272% |

All tests were exposed by earlier SPECTRA work. Prior stronger SVM/prototype
accuracy remains unchanged. Compression cannot improve these frozen predictions.
The completed four fit calls consumed **42.010 CPU seconds** in aggregate. An
initial process was interrupted after two models; the missing two were fitted
without replacing the completed ones. Unrecorded interrupted overhead, exports,
imports, data acquisition, compilers, proofs and testing are additional;42 seconds
is not a claim about total session compute. No GPU or teacher was used.

## Certificate outcomes and nontrivial safeguards

| Task | Inputs | Certified8 | Certified16 | Official fallback after16 | Wrong unguarded8 decisions refused |
| letter | 4,000 | 1,752 | 3,991 | 9 | 29 |
| pendigits | 3,498 | 3,246 | 3,497 | 1 | 4 |
| satellite | 2,000 | 1,771 | 2,000 | 0 | 1 |
| optdigits | 1,797 | 1,589 | 1,796 | 1 | 5 |

The39 actual 8-bit changes to source decisions are ALL rejected by the certificate.
No observed16-bit quantized class changes occur, yet eleven conservative bounds
remain unresolved; those are not silently promoted. The8-bit-only certified
coverage is8,358/11,295. The16-bit certified fraction is99.9026%, not unconditional
100%. Full fallback coverage includes real official computation for those eleven.

Every certified answer matches official CPU libraries1.2.8 AND1.2.10. The original
CBM and JSON contain851,648 bit-identical leaf scalar values, equal split borders
and scale/bias;2,891,520 retained tree routes also match. The exported C++ control
matches all class indices. Some source score sums differ by numerical reduction
order; the proof includes source roundoff rather than assuming bitwise scores.
Synthetic tests separately exercise wrong8-bit decisions, exact ties, cancellation,
same-class residual correlations, early stopping and different source sum orders.

## Two complete execution comparisons, no pooled cells

The first15-arm run contains1,260 cells and is correct but generally slower than
the official library at batch32. The entire original result remains. After a
public model-frozen protocol, the second run repeats all15 arms and adds seven
vector/register/refinement variants:22 arms x4 tasks x3 batch sizes x7 repetitions
= **1,848 cells**. All output arrays, model/library/source identities and random
orders are retained. No favorable rows were selected between the two runs.

Each timed whole-dataset job starts from original writable contiguous uint8 codes
and returns a fresh class-index list. Validation, required float conversion,
routing, accumulation, certificate checks, refinement and official fallback are
included. Model preparation, exact-rational verification, compilation and original
image/sensor feature extraction are excluded. One pinned Linux x86-64 CPU core,
no simultaneous fitting or compilation. These costs are not individual-request
latency distributions or production tails.

### Final batch32 microseconds per row

| Task | Official1.2.8 | Official1.2.10 | Exported C++ | 16-first + official | Speedup vs faster official |
| letter | 3.2716 | 3.3778 | 6.3957 | **1.6263** | 2.012x |
| pendigits | 1.6748 | 1.9909 | 2.7941 | **0.9647** | 1.736x |
| satellite | 1.4837 | 1.4174 | 3.5640 | **0.8124** | 1.745x |
| optdigits | 1.8171 | 2.0655 | 3.0466 | **1.3365** | 1.360x |

The same16-first profile is used on every task; the faster of both official
versions is the deliberately conservative comparator. The8-first shared-route
pipeline is not uniformly better: its batch32 costs are2.0915/1.0645/0.9995/1.2386
microseconds (Letter/Pendigits/Satellite/OptDigits). It remains a measured alternative,
not an automatic policy selected from each test set. Early-exit versions are also
retained; counting fewer leaf additions does not imply less routing in tiled mode.

At batch256,16-first full coverage is also faster on all four measured tasks.
At batch1 it is SLOWER than exported C++ on every task: approximately13.629 vs9.193,
11.621 vs7.391,11.491 vs6.466,15.086 vs6.541 microseconds respectively. Native library
and interface overhead matter. This is a batched efficiency result, not universal
low-latency superiority. Pairwise medians and all seven repetitions remain available.

## File size versus full deployment cost

| Task | Original CBM bytes | Compact8 bytes | Compact16 bytes | CBM / compact16 |
| letter | 3,569,888 | 433,324 | 859,308 | 4.154x |
| pendigits | 1,473,752 | 171,936 | 335,776 | 4.389x |
| satellite | 950,080 | 106,648 | 204,952 | 4.636x |
| optdigits | 1,471,600 | 171,202 | 334,722 | 4.396x |

The4.15–4.64x file reduction belongs to the compact16 representation, which may
abstain. A local full-coverage pipeline retains original CatBoost model state AND
the compact model AND an official model library (the1.2.10 shared object is
12,316,552 bytes). Do not label it a4x memory reduction. The packed metadata,
integer leaves, residual bounds, suffix bounds, pointers and scratch all count.
No new whole-process RSS, setup-time or energy result is claimed. Exact Python
proof reconstruction can dominate cold preparation. Its costs are excluded from
warm timings equally and must be amortized in a real deployment.

## Tests and evidence acceptance

The13 original exact-rational tests were freshly rerun, not inferred from an old
receipt. There are60 new native/ownership/refinement cases plus13 independent
auditor cases,73 combined tests. Both original and optimized implementations have
real-model coverage, and the final portable and AVX2 implementations preserve all
recorded indices/statuses/work against the reference. The optimized AVX2 runtime
also passes undefined-behavior instrumentation; repeated builds do not multiply
the unique test count. Two native cases invoke actual official libraries on a
constructed unresolved model, not a mocked callback.48 whole-corpus layout
comparisons preserve indices, per-row leaf-work and refinement status exactly.

The independent standard-library auditor reconstructs all3,108 timing cells and
retained outputs, checks source/model/library bindings and recomputes summary
arithmetic without running the classifier or unpickling a model. Sixteen altered
actual evidence copies are rejected. This audits recorded bytes and arithmetic,
not trustworthy clocks, statistical independence, unrecorded human adaptation or
malicious wholesale evidence replacement. Source-binding and mathematical proof
reconstruction have separate roles and receipts.

Fresh acceptance scope: Linux x86-64, explicit portable/AVX2, local Python3.13.5,
strict noncontracted arithmetic and UBSan. Final remote CI is recorded separately.
No Windows/ARM, ASan, free-threaded Python, full historical SPECTRA suite, external
research-group reproduction or live deployment result is inherited from prior work.
No source-model defaults, original numerical files, main branch or release changes.

## Interpretation

This moves the checkpoint into a real deployable, source-verified native executor,
with batched improvements against two official CatBoost libraries and actual
fallback costs included. It preserves classifier capability; it does not add
learned intelligence. Input-specific certification, exact pruning, quantization
and vectorized tree evaluation have prior art. No world-first, universal best
implementation or high-90s score is inferred. A useful next application decision
must account for batch size, verification amortization, residual fallback and the
complete dependency inventory rather than only the compact leaf file.

Primary sources: official CatBoost C API1.2.8/1.2.10 and exported C++ (Apache-2.0),
https://catboost.ai/docs/en/concepts/c-plus-plus-api_dynamic-c-pluplus-wrapper
https://github.com/catboost/catboost/releases
Prior exact tree-ensemble compression: Vidal et al., ICML2020,
https://proceedings.mlr.press/v119/vidal20a.html
Original UCI data retain their separate attribution and CC BY4.0 terms.
