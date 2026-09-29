# Exact dyadic verification — source-frozen continuation

Base: 20e36d4851ac37d33748c89ffbf496b8cd8358b5 (live tree-native-replay),
with the separately preserved 04173ea local delivery retained as a comparison
branch rather than silently overwritten. Reuse that delivery's four frozen source
models and compact files. No fitting, model choice, label change or new holdout.

Target the expensive exact-rational source verification, not inference arithmetic.
Represent stored binary64 leaves on a common power-of-two integer grid. Compute
contrasts, nearest-even quantization, per-tree extrema and correlated residual
extrema exactly with Python integers. Retain Fraction arithmetic where division
by the positive source scale genuinely produces a non-dyadic rational. Keep the
original oracle unchanged as an independent differential control.

Acceptance requires IDENTICAL complete oracle objects, canonical hashes, packed
bytes and validation outcomes, including 8/16-bit, binary and multiclass, pairwise
bounds, cancellation, subnormals, unusual scales and invalid source/packed files.
No tolerance, calibrated threshold, cached trusted proof or skipped check. Both
implementations must reject unsupported inputs. No claim of new arithmetic theory.

Before promotion, run existing tests and the new adversarial tests, replay every
retained input through both verified objects and the unchanged native engine,
and compare official CatBoost 1.2.8/1.2.10 outputs and unresolved/fallback counts.
Keep provenance and all failed attempts. Warm model accuracy and runtime are not
claimed improved by a compiler/verification-only change.

After implementation is frozen, measure complete fresh-process verification and
full native-session setup, separately: four frozen models, 8/16-bit, both exact
verifiers, three shuffled fresh processes per cell, one pinned CPU. No child runs
concurrently. Verify output byte hashes on every completed cell. Include parsing,
reconstruction and packing in verification time; include actual original-model
fallback loading in setup. Record elapsed/CPU time, own VmHWM, interpreter startup
separately, source/model identities. No selective reruns. Repetition is not new
sample size. Full inference and file-I/O speed are outside this experiment.

Keep native C++, oracle source, model bytes, compiler numerical contract, original
labels and official fallback unchanged. New deployment default permitted only
inside this experimental branch after exact parity; reference backend remains
selectable. No production/main merge, release, Windows/ARM or external reproduction.
