# Binary streaming execution: fixed local implementation/evaluation scope

Base: PR30 4faa21280587fa9c4bf13cc2eb465613b5a95d4c, tree
210a21eafd982258f27642e0a1f33f6bc2e738a4. CPU only, no new fitting or labels.

## Mechanism

Binary SVMs have a single pair: their evaluated kernel values cannot be reused by
another classifier. Test an opt-in `binary_stream` execution schedule which avoids
centroid ordering and materializing a per-request kernel cache for this case.
Stream four-support calculations for scalar inputs; for direct (non-table) AVX2
batches evaluate four independent input rows in SIMD lanes, retaining each row's
feature accumulation order and each classifier's original coefficient order.
Use the exact same scalar system exp, binary64 parameters, separate operations,
zero/tie convention and input precision as the existing exhaustive reference.
No quantization, altered exponentials, reordered reductions or new classifier.
Multiclass requests fall back to the existing beretta_cert schedule. Old defaults
and specialized Session remain unchanged. Mode numbers 0..6 retain their meaning.

A preliminary numerical headroom screen using approximate NumPy kernels and
coefficient-tail triangle bounds is retained as diagnostic only. It is NOT a
native/fidelity benchmark or an implemented arithmetic certificate. It suggested
little prefix savings on the expensive binary cases; do not promote those bounds.

## Acceptance before timing

Check all frozen six-task pipelines (all 18 models/all saved rows) against the
existing exhaustive reference and safe expected labels, both precisions where
applicable, direct/tables, scalar/batch/tiled-fused, portable/AVX2. Test empty batches,
2–128 classes, odd dimensions, incomplete SIMD packets, zero coefficients/ties,
subnormal/extreme finite inputs, invalid late inputs and prior certificate reset.
Independently compare actual computed binary margins bit-for-bit, not only labels,
using an original-source native observer and a separate ordered interpreter.
Build/install the actual wheel outside the source without numerical frameworks.

## Fixed performance panel

All 18 unchanged models, float64, tables=False primary. Report three binary task
summaries separately and retain all three multiclass negative controls. 31 shuffled
repeats of whole jobs over every saved test row, one pinned CPU core. Batch sizes
1, 4, 32, 128 (last packet shortened). Pair exhaustive, existing beretta_cert,
opt-in binary_stream in the SAME rebuilt library. Time checked borrowed-buffer
public calls with fresh result lists; separately compare complete raw-row compiled
and fused pipelines on the existing 1/1/8/32/128 mixed trace. No discarded cases,
retiming until green, model selection, previous-host pooling or overhead subtraction.
All outputs checked after each call. Preparation/compilation and original domain
feature extraction outside warm timers. No concurrency during measurement.

Primary admission: binary batch32 geometric ratio <= 1/1.20 vs beretta_cert, no
binary task ratio >1.10; all fidelity passes. Record batch1 regressions and all
model medians regardless of the gate. A failed gate remains failed. Full panel
ratios and pooled time are secondary, not a universal claim. Timing paths and
numerical source are frozen before the accepted complete experiment.
