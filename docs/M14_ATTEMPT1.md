# Milestone 14 — Attempt 1 Result

**Decision: VALID NEGATIVE DEVELOPMENT RESULT. M14 remains INCOMPLETE.**

Attempt 1 executed the preregistered protocol in [`M14_PROTOCOL.md`](M14_PROTOCOL.md), committed at `99102c30f87661f6767c586ba6f81971e32a654c` before any M14 result. The executable head was `66acbc62f48b3e5b1e81080897b89562edfd8bd8`.

## Provenance

```text
Actions run      34159474745
job              101858086130
artifact         m14-primary-controlled-experiment-evidence
artifact id      10032762057
artifact SHA256  04a5b810d697227c4ccf88a365cf529647734ea0d95deec363f84fcaae885522
experiment exit  2 (valid INCOMPLETE)
focused          17 passed
full fast        291 passed, 16 deselected, 1 pre-existing M10 warning
```

Every integrity stage passed: preregistration ancestry, compile, focused contracts, experiment artifact production, direct-from-raw rendering, evidence hierarchy, and the full regression suite. Only the positive M14 claim-enforcement step failed, as intended for a valid miss.

## Data and repeated units

```text
train                 4096
validation              512
development             512
training seeds            5: 1401, 2402, 3403, 4404, 5405
paired quality rows    512 per seed
paired latency rows    128 per seed
bootstrap replicates  5000
```

The independent confirmation manifest was **not generated**. Confirmation seed `2026091402` and reserve confirmation seed `2026091403` remain untouched.

## Main dim-48 validation selection

The larger trained FP single-pass baseline:

```text
semantic validity   0.841016
median solve         1.067034 ms
```

The latency-eligible validation winner was `fp_n1`:

```text
semantic validity   0.793750
median solve         1.138551 ms
latency ratio        1.0670
```

Higher-recursion FP operating points were much more accurate but not cost matched:

```text
fp_n2   validity 0.921094   1.734206 ms
fp_n3   validity 0.965625   2.323187 ms
fp_n4   validity 0.974609   2.906702 ms
```

This establishes an important failure structure: the trained recursive trajectory contains high-quality later states, but first-step quality is substantially lower and each additional recurrent step costs roughly another 0.6 ms on this implementation.

The W1.58A8 trajectory showed the same qualitative pattern at a worse latency/quality point:

```text
ternary_n1   0.758594   1.969253 ms
ternary_n2   0.881641   3.401135 ms
ternary_n3   0.927344   4.778690 ms
ternary_n4   0.944922   6.212366 ms
```

The trained-reasoner/trained-verifier/fixed-action search variants did not recover the quality gap and added large cost. For FP:

```text
fp_search_r1   validity 0.793750   4.389116 ms
fp_search_r2   validity 0.793359   6.551035 ms
fp_search_r4   validity 0.791016  10.781479 ms
```

This is consistent with the earlier M09 negative action-mechanism evidence: M14 does not relabel the fixed action codebook as learned.

## Main development gate

Frozen candidate: `fp_n1`.

```text
quality difference candidate - baseline   -0.05078125
95% hierarchical-bootstrap CI             [-0.162890625, +0.032421875]
latency ratio candidate / baseline          1.079275
95% ratio CI                                [1.077121, 1.081362]
```

Both preregistered paths failed. The candidate was slower and materially less accurate on development.

## Preregistered reserve intervention

The protocol permitted exactly one initial intervention: increase FP recursive hidden width `48 -> 64` at fixed data, steps, optimizer, objective, and baseline.

The five dim-64 models trained successfully and had final four-step validation validity between `0.9766` and `0.9883`.

Validation operating points:

```text
fp64_n1   validity 0.837891   1.154915 ms   ratio 1.0881  (matched)
fp64_n2   validity 0.942188   1.765067 ms   ratio 1.6629  (unmatched)
fp64_n3   validity 0.975000   2.368587 ms   ratio 2.2315  (unmatched)
fp64_n4   validity 0.981641   2.963897 ms   ratio 2.7924  (unmatched)
baseline  validity 0.841016   1.061416 ms
```

Validation therefore froze `fp64_n1`.

Reserve development result:

```text
candidate semantic validity                0.832812
baseline semantic validity                 0.837891
quality difference                         -0.005078125
95% hierarchical-bootstrap CI              [-0.04453125, +0.033203125]
latency ratio                               1.094335
95% ratio CI                                [1.088790, 1.099829]
```

The width increase removed most of the first-step quality deficit but did not produce the required +3-point superiority, and it remained slower than the larger baseline. The fast-tradeoff path was also impossible at a 1.09x latency ratio.

## Conventional INT8 and symbolic context

PyTorch dynamic INT8 conversion was genuinely supported on this runner: six `nn.Linear` modules in the larger baseline were converted (`ff0/ff2` in both blocks plus output/confidence heads). Attention internals remained floating point and were disclosed. The INT8 row was slightly slower and nearly identical in quality:

```text
validation FP baseline     0.841016   1.067034 ms
validation INT8 baseline   0.839844   1.233279 ms
```

The exact MRV/backtracking symbolic solver was a contextual reference with a different inductive bias:

```text
validation semantic validity   1.000000
median solve                    0.491984 ms
```

It is not used to satisfy or weaken the learned-model claim.

## Failure diagnosis

Attempt 1 falsifies the simple **capacity-only** hypothesis. Increasing recursive width from 48 to 64 raised one-step quality near baseline, but did not create either quality superiority or a latency advantage.

The strongest actionable signal is temporal:

- dim-48 `N_sup=1 -> 4`: `0.7938 -> 0.9746` validity;
- dim-64 `N_sup=1 -> 4`: `0.8379 -> 0.9816` validity;
- training used later-heavy deep-supervision weights `0.1/0.2/0.3/0.4`.

Therefore the next falsifiable intervention is **early-exit credit allocation**, not another width increase: retain a small recursive graph but train the first supervised state much more aggressively so the high-quality information currently reached after multiple recurrent steps is moved toward the cost-matched first step.

No threshold, baseline, seed-filtering rule, or claim definition is changed. Attempt 1 remains permanent development evidence and cannot be reclassified as a pass.
