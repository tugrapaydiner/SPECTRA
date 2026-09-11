# Frozen fixed-pool inference replay and CPU numerical identity

## What was missing

The PR #18 consolidation independently checked retained complete-solve answers
and reran all 3,840 frozen development model/example/solver cases. The scientific
fixed-pool selection result, however, was only reaggregated from stored labels.
It did not independently reconstruct the candidate states, labels and predicted
value vectors behind that result.

`eval/fixed_pool_replay.py` now reconstructs the documented four-depth/four-action
pools from the exact frozen cores. It does not call the original pool builder,
Torch structural scorers, TaskSpec restoration, or native checkers. Restoration
uses NumPy first-maximum argmax, and exact validity uses the reference task
validators. The neural core primitives and frozen value-head implementation are
shared intentionally: this audits the experiment, not an independently written
neural network library.

The replay checks every already-published M17 surface: Sudoku development,
Sudoku confirmation, and maze development. It reads hash-bound input arrays and
checkpoint bytes; it never regenerates a task, retrains a model, tunes an
estimator, or opens the previously unused maze confirmation seed.

The expected inventory is **1,024 model-example pools**, **16,384 candidate states**,
**16,384 additional one-cycle continuations**, and **49,152 learned-head scores**.
It checks six exact combined pool-tensor hashes, all discrete labels, candidate
order, selections, successes and scientific gates. A changed near-tie selection
fails even when its score difference is within the numerical allowance.
Missing, duplicate, extra, mislabeled, retargeted or tampered records fail closed.

## Reproduction command

```bash
python scripts/verify_fixed_pool_replay.py \
  --cpu-profile historical-avx2 --out outputs/fixed-pool-replay
```

The output directory must not already exist. Success creates `summary.json` and
all reconstructed pool rows. An exception creates `failure.json`, exits nonzero,
and never writes a success summary. CPU CI requires this replay on Python 3.11
and 3.13, alongside the existing fast tests, archive audit and complete-solve replay.

`historical-avx2` is the CLI default. Before importing PyTorch or NumPy it sets:

```text
ATEN_CPU_CAPABILITY=avx2
MKL_ENABLE_INSTRUCTIONS=AVX2
ONEDNN_MAX_CPU_ISA=AVX2
DNNL_MAX_CPU_ISA=AVX2
```

These settings reproduce the instruction-set scope of the recorded AMD EPYC 7763
run. They do not promise universal bitwise reproducibility or improve performance.
PyTorch, MKL and oneDNN use separate dispatch controls, so capping only one library
is not treated as a complete numerical contract. Exact tensor hashes remain
mandatory; no fingerprint or scientific threshold was relaxed to obtain a pass.
The score allowance was fixed at absolute 2e-6 before the dispatch diagnostic;
observed successful replay scores were all exactly equal (maximum error zero).

An explicit `--cpu-profile host` leaves dispatch untouched for portability checks.
On the local AVX-512-capable AMD EPYC 9V74, the first host-profile replay failed its
pool tensor hash check. A separate diagnostic on the first Sudoku development
core found unchanged labels and selected candidates but probability differences
up to approximately 1.334e-4, exceeding the declared allowance. The historical
AVX2 profile then reproduced all six pool tensor hashes and all 49,152 scores
exactly. The failed attempt remains evidence, not a deleted run. Only CPU dispatch
changed; source checkpoints, input arrays, labels and acceptance tests did not.
This is a bounded portability finding, not a universal attribution to one kernel.

Dispatch documentation: Intel oneMKL, "Instruction Set-Specific Dispatching on
Intel Architectures"; oneDNN, "CPU Dispatcher Control"; PyTorch 2.10 CPU dispatcher
(`ATEN_CPU_CAPABILITY`). Environment values and the observed ATen CPU capability
are recorded with each new replay.

## A precise limitation of the maze structural target

For a valid maze input, the frozen decoder restores walls and endpoints and
allows PATH only on originally OPEN cells. Write `P(a)` for the event that a
restored candidate has at least one PATH cell, and `V(a)` for exact maze validity.
The existing structural score therefore simplifies exactly to

```text
Q(a) = 1/2 + 1/4 * 1[P(a)] + 1/4 * 1[V(a)].
```

Proof: walls-kept and endpoints-kept are each one after restoration. Path-on-open
is one for a nonempty overlay and zero for an empty overlay. The fourth term is
the exact semantic bit; the implementation averages the four terms. Consequently,
all nonempty invalid candidates have score 0.75, regardless of how disconnected,
misdirected or otherwise incorrect their paths are. On the retained maze problems,
start and goal are nonadjacent, so every valid answer needs a nonempty PATH
overlay. For a nonempty current candidate on these problems, the improvement
label is exactly

```text
1[Q(next) > Q(current) + 1e-6] = 1[not V(current) and V(next)].
```

It is thus a solve-transition event on this subset, not graded progress toward a
solution. This event identity must not be generalized to adjacent endpoints: an
empty valid route can score 0.75, equal to a nonempty invalid route. A dedicated
counterexample test retains this boundary. The new replay verifies both identities on the real checkpoint outputs,
not merely on a hand-built counterexample. Small tests also include empty, invalid
nonempty and valid overlays and compare independent and production reconstruction.

The retained maze development pools contain **4,096 candidates**. All have
nonempty overlays; **3,707 are invalid and all of those receive exactly 0.75**.
Only **38 of 256 model-example pools** contain any valid candidate, so even perfect
selection cannot exceed 14.84375% success on these fixed pools. Better ranking
alone cannot solve that coverage deficit. The simplification is an algebraic
property of this particular decoder/scorer pair, not a new universal theorem
about learned search or a proof that it caused the entire maze failure.

## Claim boundary

This finishes a missing independent inference check and identifies a precise
limitation of the maze target. It does not produce new model capability, establish
search superiority, or convert M17 into a two-family positive result. Historical
quality thresholds, checkpoints, source evidence and negative statuses remain
unchanged. The long legacy retraining tests remain a separate validation scope.
