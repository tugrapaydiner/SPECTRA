# Certified tree capacity frontier — prospective protocol

Base: research/tree-native-replay at 20e36d4851ac37d33748c89ffbf496b8cd8358b5.
The current compatibility models use a deliberately small fixed recipe
(256 depth-6 oblivious trees, learning_rate .08) and are materially less accurate
than several earlier SPECTRA classifiers. This study changes the learned source
model itself while preserving the same numeric-only oblivious-tree semantics needed
by the independent certificate compiler. It is not a claim that quantization or
verification makes a classifier more accurate.

## Data boundary

Use only the official UCI Letter Recognition, Pendigits, Statlog Satellite and
OptDigits original training partitions for model selection. Their official test
partitions were already exposed by earlier research, so the final comparison is
exploratory compatibility/quality evidence, not fresh independent confirmation.
Acquire the official archives by immutable URL and retain SHA256/raw-row receipts.
Exact duplicate feature rows stay together inside validation splits.

Two fixed grouped train/validation splits, seeds 611 and 977. No test predictions,
test confusion matrices or old test scores are inputs to candidate selection.

## Candidate family

CatBoost 1.2.8 CPU, MultiClass, bootstrap_type=No, random_strength=0,
l2_leaf_reg=3, random_seed fixed per split, thread_count=1,
allow_writing_files=False, no early stopping.

Reference control:
- 256 trees, depth 6, learning_rate .08.

Capacity grid, identical on all four tasks:
- trees in {384, 512}
- depth in {7, 8}
- learning_rate in {.04, .06}

Eight candidates plus the control. No post-test capacity expansion.

Selection per task:
1. pooled validation accuracy across the two fixed splits;
2. fewer tree-predicate evaluations (trees * depth);
3. smaller serialized CBM;
4. fixed enumeration order above.

Refit each selected recipe once on the complete official training partition using
random_seed 20260929. Freeze CBM, JSON and exported C++ identities before any
official-test prediction.

## Accuracy/efficiency decision

Primary quality target: selected source model improves the 256x6 control by at
least 0.75 percentage points on at least two tasks. This is deliberately stronger
than the previous 0.5-point exploratory metric-learning gate.

Then compile both control and selected models using the existing exact
source-verified residual-tree compiler. Compare:
- source-model correct counts;
- compact certification coverage;
- compact model/source bytes;
- proof verification time and memory;
- warm compact execution at batches 1/32/256;
- official CatBoost 1.2.10 CPU C API on the same selected source model.

A useful frontier point must improve source accuracy and not exceed 2.0x the
control's batch-32 compact execution cost on the same task. Report every task and
all regressions. Do not multiply unrelated speedups or compare new accuracy to an
old model with different training/search budget as if it isolated architecture.

## Follow-up architecture gate

Only if this capacity frontier still leaves a clear accuracy/cost gap, a separate
prospectively locked specialist/cascade experiment may be started. No conditional
routing, specialist training, margin threshold or post-hoc ensemble is selected
from official-test outcomes in this protocol.

## Preservation and claims

No production default, release, main merge or previous evidence is replaced.
All fitting is CPU-only; record wall/CPU time and model sizes. The exact certificate
still proves agreement with the newly fitted source model under its documented
binary64 arithmetic contract; it does not prove ground-truth correctness.

CatBoost, boosted oblivious trees and capacity scaling are established techniques.
Any contribution here is an observed accuracy/certified-execution frontier, not a
first-invention or general-intelligence claim.
