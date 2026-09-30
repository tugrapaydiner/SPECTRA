# Total specified-source tree execution — September 29, 2026

## Decision and change

The supported-input execution policy no longer abstains when the compact proof fails. It reuses tree routes and performs the original ordered binary64 calculation from a lossless leaf bank. No CatBoost inference library, original CBM, retraining or test-calibrated confidence threshold is required. The guarantee is the specified JSON-source arithmetic under the admitted numerical/domain contract, not every CatBoost backend or ground-truth accuracy. This closes general-input abstention; the previous residual model already resolved the retained natural corpus.

The complete performance gate passes, with geometric total/prior-full16 cost 0.971994. The worst task ratio is 1.043585. This is roughly tied throughput, not a major new speedup. Exact original leaves add storage relative to the residual-only abstaining model. Neither production defaults nor any learned model is changed.

## Fixed-corpus and model-only storage

| Task | Rows | Exact fallback rows | Flat total bytes | Interned total bytes | Interned prepared bytes | Residual-only bytes |
|---|---:|---:|---:|---:|---:|---:|
| letter | 4,000 | 9 | 4,267,500 | 3,025,020 | 3,078,752 | 1,285,830 |
| pendigits | 3,498 | 1 | 1,646,688 | 1,192,176 | 1,213,012 | 499,882 |
| satellite | 2,000 | 0 | 991,544 | 679,384 | 691,996 | 303,454 |
| optdigits | 1,797 | 1 | 1,643,074 | 900,258 | 921,094 | 498,508 |

Interning is lossless whole-vector deduplication by binary64 bytes, not pruning or quantization. Every leaf keeps its route mapping and every selected tree contributes in the original sequence. Signed zero is retained. The source JSON is additionally required for verification. The format remains larger than residual-only data; removing an external engine does not make full source precision free.

## Source arithmetic and independent checks

All 11,295 retained rows and 16,384 deterministic uniform/boundary-domain stress rows match the independent JSON-source calculation bit-for-bit in full class scores. The stress contains135 exact escapes. These generated rows are correctness stress, not new generalization data. In total381,942 saved source score values are checked; repeated layouts/policies are not independent examples.

The slower preserved Fraction oracle reconstructs all8 flat/interned artifacts byte-for-byte, including the original compact proof. The exact dyadic analyzer was copied unchanged from the preceding delivery. A separate scalar reader evaluates original source leaves; neither the native fallback nor the source-bank dictionary is reused by that reader.

The exported original C++ scores are not bit-identical to the JSON-source definition. The following differences are retained rather than described as universal backend equivalence:

| Task | Different exported-C++ score values | Maximum absolute difference | Retained label disagreements |
|---|---:|---:|---:|
| letter | 60,604 | 3.55271e-15 | 0 |
| pendigits | 18,457 | 3.55271e-15 | 0 |
| satellite | 5,833 | 2.66454e-15 | 0 |
| optdigits | 10,935 | 2.66454e-15 | 0 |

All measured official1.2.8/1.2.10 predictions also match on the natural corpus. This empirical agreement cannot prove all-input equality across different operation orders. Synthetic cancellation tests explicitly distinguish the source floating answer from the exact-real answer; the former is the contract. Totality follows from exact routing, preserved leaf operands and bounded finite source loops, assuming the existing certificate argument and the documented environment. It is not immunity to runtime errors or malicious native code.

## Complete warm comparison

One pinned CPU, nine arms, four models, batches1/32/256, seven shuffled repetitions and ten whole jobs per cell:756 observations and21,347,550 repeated prediction checks. Each job includes input conversion/validation, routing, all actual escape work, fresh labels and identical output comparison. Verification, loading, file I/O and original feature extraction are excluded. The actual CPU and versions are in LOCK.json.

| Task, batch32 microseconds/row | Faster official API | Prior full16 + official | Residual adaptive | New interned total | Force exact interned |
|---|---:|---:|---:|---:|---:|
| letter | 2.3427 | 0.9936 | 0.9712 | 0.9884 | 2.4810 |
| pendigits | 1.2477 | 0.8113 | 0.7906 | 0.7902 | 1.4527 |
| satellite | 1.0385 | 0.7271 | 0.7321 | 0.7588 | 1.4855 |
| optdigits | 1.3750 | 0.9687 | 0.9023 | 0.8551 | 1.5986 |

Satellite regresses4.36% against prior full16. Paired repetitions and all batch sizes are retained; seven repetitions do not establish production tails or universal gains. Much of the advantage against official APIs belongs to the inherited vectorized executor. Pure exact evaluation is a necessary ablation: the selective fast path, not merely removing a library, preserves throughput.

Two command-time limits interrupted timing after Letter176 and Satellite134 cells. Their exact completed prefixes are preserved and only missing suffixes were appended. The source amendment changes resume bookkeeping only, not models, implementation, timer scope or job order. No completed observation was replaced or selectively rerun. The independent auditor checks both prefixes and both script hashes.

## Isolated startup and memory

The first34-process resource prefix was rejected because environment sitecustomize imported NumPy. The entire48-process matrix was rerun under -I -S, three fresh processes per task/policy. Imports are outside the setup timer; source/model reading, full proof reconstruction and native loading are inside. VmHWM includes interpreter imports and transient verification storage. VmRSS is read while the native session is still alive after releasing the verified Python object; allocator retention still makes it more than model storage. Filesystem caches can be warm.

| Task | Interned setup ms | Residual setup ms | Official-only load ms | Interned peak KiB | Residual peak KiB |
|---|---:|---:|---:|---:|---:|
| letter | 980.72 | 1029.47 | 8.27 | 133,676–133,932 | 114,052–114,076 |
| pendigits | 388.95 | 393.53 | 4.81 | 66,132–66,596 | 57,256–57,392 |
| satellite | 246.64 | 245.41 | 3.88 | 48,008–48,052 | 40,512–40,540 |
| optdigits | 339.98 | 389.87 | 4.77 | 61,468–61,596 | 53,268–53,292 |

Source verification still costs roughly0.25–0.98seconds, whereas ordinary official loading costs roughly4–8milliseconds. The total format is NOT a cold-start advantage over CatBoost. Verification peak memory rises compared with residual-only execution. All flat-layout, live RSS and per-process observations remain available. No hard RAM or worst-case service bound is implied.

## Delivery and acceptance boundaries

122 unique runtime and JSONL tests pass locally on portable, AVX2 and AVX2 UBSan. The122 comprises the105 numerical/source/ownership cases and17 file-runner cases. An initial file-test expectation incorrectly assigned the tie result to a different leaf and was corrected; its failed log remains. No numerical implementation changed for that fix. Twelve deliberately damaged evidence copies are rejected by the independent recorded-data auditor.

The kit reconstructs source proofs at load and executes without CatBoost or numerical Python frameworks. The normal API and JSONL path return source class indices; the explicitly diagnostic certificate_only mode still abstains. File publication is no-replacement, after complete successful input processing, with a trusted stable directory and hard-link requirement. Forced process death can leave a private partial file.

The four classifiers are unchanged and do not become more intelligent. Their source true-label counts remain3735/4000,3369/3498,1794/2000,1730/1797; earlier research includes stronger learned models. No model fitting, GPU, pretrained teacher, new holdout, world-first algorithm, official upstream acceptance, external researcher reproduction, Windows/ARM, full historical suite, ASan or production release is claimed.

The remote branch is additive on inspected20e36d4, preserving concurrent source. The full local research archive additionally retains the prior852-file residual delivery; it is a separate source view with the same tested numerical dependencies. Exact-head cloud and extracted-package outcomes are recorded in the final manifest/report instead of being inferred from a workflow file.
