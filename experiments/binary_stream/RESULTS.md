# Retained-panel binary streaming result — September 27, 2026

## Decision

Promote `binary_stream` only as an explicit optional execution profile, NOT a new
default or a more accurate classifier. It preserves the old computed binary64
margin and meets the locally predeclared batch32 performance gate. It cannot
promise a universal advantage on unmeasured models, CPUs or input distributions.

The same fitted models, scalar system exp, feature order and coefficient order
are retained. Binary models have one pair, so no other classifier can reuse their
request-local kernel values. Streaming removes that cache/centroid work. In
AVX2 direct batches, four independent inputs occupy four lanes; there is no
horizontal reduction, fused multiply-add or approximate kernel. Partial packets,
feature-table models and portable builds retain ordered support-packet execution.
Multiclass mode delegates to `beretta_cert`. No support pruning, retraining or GPU.

## Fixed prepared-input batch experiment

AMD EPYC 9V74, one pinned core, Linux, GCC14.2, Python3.13.5. All18 retained models
and all4,281 model/input pairs. Four batch sizes1/4/32/128;31 shuffled whole-job
repetitions; three same-library controls: exhaustive, beretta_cert, binary_stream.
6,696 timing cells;1,592,532 checked repeated predictions. Public checked-buffer
calls and fresh labels are charged, not loading/preprocessing/compilation. Inputs
and numerical source were fixed before formal timing. No observations removed.

| Binary task, batch32 | Existing default us/row | Streaming us/row | Matched streaming/default ratio |
|---|---:|---:|---:|
| Chess |9.268|7.580|0.813852|
| Titanic |5.148|4.556|0.890327|
| WDBC |1.570|1.191|0.756262|

Displayed times are task medians of per-model complete-job median cost/rows.
Ratios are geometric means of matched models, not necessarily ratios of displayed
medians. Primary binary geometric ratio **0.818319** (18.17% less cost,1.222x).
All nine binary batch32 model medians improve. The required <=1/1.20 panel ratio
and no binary task ratio>1.10 pass. Against optimized exhaustive execution, the
binary task ratios are0.761739/0.760215/0.733065. These controls share the same
kernel primitives; this is not just a comparison against Python SVM code.

Multiclass batch32 task ratios are0.999251 (Penguins),1.001291 (Wine),1.001456 (Zoo).
Those paths use the unchanged scheduler; slight measured differences/regressions
remain rather than being described as a new multiclass gain. The corresponding
batch1 ratios on binary tasks are0.938000/0.912245/0.952149. These are whole jobs of
size-one calls, not independent service-latency distributions.

## Separately measured raw-input mixed requests

The same full panel, deterministic shuffled requests cycling1/1/8/32/128, five
repetitions, four arms: compiled-default/stream and fused-default/stream. All
preprocessing, input conversion, SVM execution and fresh labels are charged.
3,240 cells;85,620 checked repeated predictions. No earlier host/run is pooled.

| Task | Fused streaming/default complete-trace ratio |
|---|---:|
| Chess |0.803899|
| Titanic |0.720184|
| WDBC |0.776073|
| Penguins |1.000549|
| Wine |1.014070|
| Zoo |1.001837|

Pooled model trace median ratio **0.792312** (20.77% lower time). Equal-task
geometric ratio **0.877557** (12.24% lower cost). Both weightings are reported,
neither is observed production traffic. All nine binary model trace medians
improve. Five multiclass model medians regress; the underlying multiclass
computation is unchanged. There is no production-tail, queueing, concurrency,
cold-start, current external-runtime, energy or fresh process-RAM claim.

## Numerical and delivery checks

- Full fast suite: **1,999 passes**,16 historical slow exclusions,two warnings.
  This includes the352-case SVM suite, not352 extra cases. The same352 tests also
  pass on portable and with the changed SVM runtime instrumented under UBSan.
- All4,281 frozen panel predictions match on portable/AVX2, direct/tables and
  fused execution. These overlap the retained dataset, not new accuracy evidence.
- Independent original-source actual margins and separately decoded Python
  arithmetic match **3,810 natural binary margins** and **2,754 synthetic margins**
  from162 fixed fixtures. Repeated native paths yield39,384 comparisons; there
  are6,564 distinct model/input margins in this audit, not39,384 independent cases.
- The independent standard-library timing auditor reconstructs both grids,
  outcome digests, identities and all aggregates. All9,936 cells are covered.
  Eighteen deliberately corrupted copied receipts are rejected. This is an
  independent checking implementation, not outside researcher replication.
- An actual wheel built from the sdist passes **44 installed-only checks** in a
  fresh environment without numerical frameworks. Both native components build
  from installed source. **142 packaged source members** match the checkout.

Four-input feature packing introduces additional temporary storage (4*d doubles,
at most128KiB), allocated once per native batch. Combined with the separately
bounded fused feature tile, the two temporary feature buffers can reach256KiB.
Existing worker/model buffers and Python objects are additional. Persistent
kernel-cache storage is retained for compatibility with other schedules. There
is no new memory-reduction or allocation-free claim.

The preprocessing operation ABI changes to4; rebuild both native components.
The original narrow Session and the four retained historical numerical source
files are unchanged, but the generic shared native runtime has changed. Linux
x86-64 is tested; Windows/ARM and external use are unestablished. CPU numerical
success is not a comprehensive security proof or exact-real exponential result.

The six datasets are previously consumed small convenience benchmarks with the
original overlapping splits/repeated features. WDBC is only a software benchmark,
not clinical validation. These results do not establish a high-90s hiring grade,
a new learned capability or new SIMD theory. No existing release is replaced.
