# Native tree replay and deployable artifacts — September 29, 2026

## What this continuation completes

Baseline: `0253da7df471b18c2f3017fad5d9b2023362c381`. The exact imported source was
recovered through actual Git bundles and checked against its CI file hashes. This
continuation does not invent or recover absent earlier empirical model files. It
creates a separately identified reproduction of the already fixed four-model
CatBoost 1.2.8 recipe. Every new model is saved and hashed before test prediction.
The numerical executor and the exact-rational oracle are **unchanged**.

The additions are a reproducible SDK assembler, complete CBM-to-JSON source-pair
verification, an independent bounded output-file checker, a real file-job comparison,
and explicit preparation/memory measurements. Two transport defects are corrected:
boolean native counters cannot masquerade as integer counts, and failed temporary
cleanup cannot turn an already published complete output into apparent inference
failure. Cleanup remains best effort in a trusted stable destination directory.

The source-pair check re-exports the original CBM using CatBoost and compares every
interpreted split, leaf, scale, bias and class mapping with the certificate source.
It rejects a different same-shaped fallback model. This depends on the trusted
exporter, input files and native library; hashes are not signatures. The generic
TreeSession constructor's documented caller-supplied fallback contract is unchanged.
The SDK assembler enforces that relationship before publishing a manifest.

## What it does not establish

There is no new learning algorithm, accuracy increase, world-first claim or default
replacement. These benchmark tests were already exposed. Certificates establish
agreement with the specified source arithmetic, not ground-truth correctness or
universal equivalence across CatBoost implementations. Full coverage retains the
original CBM and the official library. A compact leaf file is not the full memory
or deployment footprint. Production, Windows, ARM, GPU, free-threaded Python and
outside-researcher acceptance are not implied.

## Newly frozen models and complete replay

The fixed recipe remains 256 depth-6 oblivious trees, learning rate 0.08, l2=3,
bootstrap_type=No, random_strength=0, seed 20260929, one numerical thread. This run
used 28.129 CPU seconds for the four fits (imports, export, verification, testing
and measurement are additional). CatBoost model metadata means new model hashes
are recorded even when numerical behavior reproduces an earlier result.

| Model | Exposed test rows | Source correct | 16-bit certified | Official fallback |
|---|---:|---:|---:|---:|
| Letter | 4,000 | 3,735 | 3,991 | 9 |
| Pendigits | 3,498 | 3,369 | 3,497 | 1 |
| Satellite | 2,000 | 1,794 | 2,000 | 0 |
| OptDigits | 1,797 | 1,730 | 1,796 | 1 |
| Total | 11,295 | 10,628 | 11,284 | 11 |

All accepted compact indices and all full-coverage indices match official
CatBoost 1.2.8 and 1.2.10. Unguarded 8-bit quantization disagrees on real examples;
those disagreements are rejected by the interval test (see SOURCE_BINDING.json).
Scalar, tiled, early-checkpoint, refinement and optimized-layout paths are checked.
The independent rational oracle also checks the fixed 33-row-per-task samples,
including logical evaluated-tree counts. Do not describe this sample as an
independent rational run on every real row.

The source classifiers here are not the strongest models from previous SPECTRA
learning studies. Preserving their outputs is an execution result, not evidence
that these models should replace the stronger SVM/prototype classifiers.

## Matched native-call comparison

The original 15-arm matrix (1,260 cells) and complete 22-arm layout matrix (1,848
cells) were rerun without parameter changes or selective replacement. Seven
shuffled repetitions, batches 1/32/256, one pinned AMD EPYC 9V74 core. Costs include
input checks, conversion, routing, certificates, refinement/fallback and fresh
indices. Model loading and exact source verification are outside these warm calls.
Source snapshots, model identities, official libraries and every timing row are
retained. The standard-library auditor independently reconstructs the matrices,
checks outputs and recomputes all medians. It does not attest to clock truth.

| Batch 32, microseconds per row | Faster official 1.2.8/1.2.10 | Verified 16-bit + fallback | Ratio: official / candidate |
|---|---:|---:|---:|
| Letter | 2.4848 | 1.2152 | 2.04x |
| Pendigits | 1.3385 | 0.8879 | 1.51x |
| Satellite | 1.2248 | 0.8201 | 1.49x |
| OptDigits | 1.4503 | 0.9489 | 1.53x |

These are fresh measurements of the already implemented executor, not a new
optimization invented in this continuation. For batch-one calls, original exported
C++ is faster on all four models. The original unoptimized packed executor also
loses in important comparisons; its results remain in both matrices.

Official CatBoost documentation identifies its C/C++ evaluation library as its
fast deployment interface. Both unmodified CPU assets are included in comparison,
not only Python prediction or generated-source controls:
https://catboost.ai/docs/en/concepts/c-plus-plus-api_dynamic-c-pluplus-wrapper

## Full-file results are much smaller than native-call gains

A separate fixed 84-cell comparison uses the same JSONL decoder, output formatting,
hashing and publication for both official libraries and the candidate. All source
rows are processed, chunk size 128, seven shuffled repetitions per task/arm.
Timing includes reading, JSON conversion, native calls, encoding, writing, file
fsync and create-if-absent publication. Proof/model preparation remains outside.
Each completed output is checked independently against source indices and labels.
This measures complete files, not a production service's request distribution.

| Microseconds per input row | Faster official file runner | Candidate file runner | Candidate / official |
|---|---:|---:|---:|
| Letter | 9.7199 | 8.9610 | 0.9219 |
| Pendigits | 8.9650 | 8.6526 | 0.9652 |
| Satellite | 11.4260 | 11.1130 | 0.9726 |
| OptDigits | 15.0223 | 15.0473 | 1.0017 |

The approximately 3–8% improvements and optical tie must not be advertised as an
end-to-end 1.49–2.04x speedup. JSON and filesystem work limit the practical benefit.
No statistical generality or performance guarantee follows from seven repetitions.
Compact-only files are tested separately: their 11 unresolved rows contain null
labels and explicit UNRESOLVED status, never an arbitrary final class.

## Preparation is a substantial cost, not hidden compression

Twenty-four fresh processes cover four models x two initialization policies x
three repetitions. They read their own Linux VmHWM and run without numerical Python
frameworks. Exact source verification and native loading are included below;
imports and whole-SDK inventory hashing are outside this phase timer. Filesystem
caches may be warm. This is not a worst-case batch-RAM or cold-disk guarantee.

| Model | Official load median ms | Verify + native load median ms | Official peak MiB | Verified peak MiB |
|---|---:|---:|---:|---:|
| Letter | 8.79 | 3,657.36 | 37.56–37.71 | 180.63–180.77 |
| Pendigits | 6.04 | 1,248.19 | 27.43–27.62 | 81.70–81.85 |
| Satellite | 4.37 | 745.72 | 25.25–25.46 | 54.23–54.25 |
| OptDigits | 6.37 | 1,170.37 | 27.41–27.56 | 72.28–72.41 |

The exact rational verifier dominates cold preparation, especially for Letter.
This implementation is suitable for evaluating *reused prepared workers*, not for
claiming fast one-shot initialization or a smaller full-process memory footprint.
The SDK CLI re-verifies on every launch. A future cached/trusted-verification mode
would change the trust boundary and is not silently assumed here.

## Acceptance and delivery

The prior 101 tests pass before changes. The final 131-case suite adds 30 source-
pair/output tests, including a same-shaped wrong CBM, wrong labels, self-consistent
forged output, truncation, duplicate keys, bool counters and cleanup failure.
Portable and AVX2 UBSan runs repeat the same cases; they are not 262 unique tests.
The preserved exact-rational reference has 13 separate tests. Small synthetic
training in tests is not a new quality study. Native arithmetic and oracle file
hashes are unchanged from the imported implementation.

The SDK includes the four exact frozen CBM/JSON/compact models, official 1.2.10 CPU
library and Apache license, both native targets, inert inputs and source indices,
source-pair receipts, rebuild source and two runnable wrappers. Portable and AVX2
self-tests replay the entire corpus without numerical Python imports. Each actual
isolated CLI processes an entire original file; its outputs are audited separately.
Final extraction and source-archive checks are recorded in the delivery manifest,
not inferred from a test on an earlier source version.

The large official library and fallback CBMs remain necessary for full coverage.
Python itself, operating-system libraries and compilers are not bundled. Dataset
attribution accompanies the inputs. No main merge or release is performed.
