# Milestone 14 — Attempt 2 Protocol: Early-Exit Credit Allocation

**Status: preregistered before any Attempt 2 training, validation, development, or confirmation result.**

Attempt 1 is permanent development evidence; see [`M14_ATTEMPT1.md`](M14_ATTEMPT1.md) and [`../results/m14/attempt1/`](../results/m14/attempt1/). Attempt 1 falsified the simple capacity-only intervention. It showed that high-quality recurrent states exist at later supervision steps, while the latency-matched first recurrent state is under-optimized.

Attempt 2 tests one specific mechanism: **move supervised credit toward the first recurrent output without changing the practical-effect target or weakening the baseline.**

## 1. Frozen claim target

The primary claim, practical thresholds, cost-matching rule, uncertainty rule, and final confirmation hierarchy are unchanged from [`M14_PROTOCOL.md`](M14_PROTOCOL.md).

The primary comparator remains a trained larger floating-point single-pass `System1Student`.

The candidate must still satisfy one of the original paths:

### Path A — quality superiority at comparable cost

```text
observed semantic-validity difference >= +0.03
paired hierarchical-bootstrap 95% CI lower > 0
candidate/baseline median complete-solve latency <= 1.15
bootstrap 95% latency-ratio upper <= 1.20
```

### Path B — declared quality/cost trade-off

```text
observed semantic-validity difference >= -0.02
paired hierarchical-bootstrap 95% CI lower >= -0.03
candidate/baseline median complete-solve latency <= 0.65
bootstrap 95% latency-ratio upper <= 0.75
```

No Attempt 1 or Attempt 2 result may weaken these values.

## 2. Mechanistic hypothesis

Attempt 1 used recursive deep-supervision weights proportional to:

```text
[0.10, 0.20, 0.30, 0.40]
```

and found:

```text
dim48 FP: N_sup=1 validity 0.7938 -> N_sup=4 validity 0.9746
dim64 FP: N_sup=1 validity 0.8379 -> N_sup=4 validity 0.9816
```

while `N_sup=1` was the only recursive operating point near the larger baseline's measured latency.

Attempt 2 therefore freezes the recursive supervision weights to:

```text
[0.70, 0.10, 0.10, 0.10]
```

normalized to sum to one.

**Falsifiable prediction:** concentrating 70% of supervised answer loss on the first recurrent state will improve the cost-matched `N_sup=1` semantic-validity point enough to meet Path A, without increasing its inference graph or latency materially.

If that prediction fails on the fresh development hierarchy, Attempt 2 is a negative result; the weights are not retuned inside Attempt 2.

## 3. Fresh development hierarchy

Attempt 2 uses a new generated unique-4×4-Sudoku hierarchy:

```text
data seed       2026091411
train           4096
validation       512
development      512
box                2
clues             6..10
unique          true
augmentation    false
```

Before any Attempt 2 optimization, the runner regenerates Attempt 1's hierarchy from seed `2026091401` and verifies zero exact puzzle fingerprint overlap with the new Attempt 2 hierarchy.

Roles remain strict:

- train: optimizer gradients only;
- validation: `N_sup` selection only;
- development: one frozen validation-selected candidate versus unchanged baseline;
- confirmation: unavailable until development passes.

## 4. Fresh training seeds

Attempt 2 uses five new training seeds:

```text
1411, 2412, 3413, 4414, 5415
```

No Attempt 1 seed is reused for the primary Attempt 2 effect estimate.

## 5. Candidate family

Only the falsifiable intervention is retrained:

```text
TRM
floating point
dim                  64
blocks                 1
heads                  4
n / T                 1 / 1
training N_sup         4
ternary              false
act8                 false
trainable graph       unchanged across candidate inference budgets
```

Training objective:

- blank-cell cross entropy;
- four recurrent supervision outputs;
- fixed normalized weights `[0.70, 0.10, 0.10, 0.10]`.

Training budget is unchanged from Attempt 1:

```text
AdamW
lr              1e-3
weight decay     0.01
batch size          64
steps             1800
grad clip           1.0
```

Inference-budget validation sweep remains:

```text
N_sup in {1, 2, 3, 4}
```

No search/action variant is added in Attempt 2. Attempt 1 already measured the available trained-reasoner/trained-verifier/fixed-action search variants and found no quality gain with large added latency. Those rows remain part of the M14 development record rather than being rerun until favorable.

## 6. Primary baseline

The primary baseline is retrained from scratch on the **same Attempt 2 training rows** and five Attempt 2 seeds:

```text
System1Student
dim              96
blocks             2
heads              4
single pass       true
confidence head   frozen/excluded from trainable count
```

Its training budget/objective remains exactly the Attempt 1 baseline:

```text
AdamW lr 1e-3
weight decay 0.01
batch 64
1800 steps
blank-cell cross entropy
```

The baseline architecture, step count, data count, seed count, and evaluation procedure are not reduced after seeing Attempt 1.

A dynamic-INT8 form of each frozen Attempt 2 baseline is again attempted as contextual deployment evidence only when the PyTorch runtime creates real quantized linear modules. The exact symbolic backtracking solver remains contextual only.

## 7. Validation selection

For each of the five candidate checkpoints, validation evaluates `N_sup={1,2,3,4}`. The baseline is measured on the same paired validation examples.

The deterministic selection rule is unchanged:

1. aggregate semantic validity and complete-solve latency over all five seeds/examples;
2. among candidate `N_sup` points with median latency `<= 1.15 ×` the larger FP baseline, select highest semantic validity;
3. tie break by lower latency then lexical id;
4. if no point satisfies the cost eligibility rule, select highest quality only as an explicitly `unmatched` diagnostic point; an unmatched point cannot be described as iso-budget.

Development cannot select the `N_sup`.

## 8. Complete-solve cost

Timing remains batch-size-one under `torch.inference_mode()` and includes:

- full model forward for the selected recurrent/single-pass graph;
- output head;
- argmax decode;
- deterministic given-clue preservation;
- symbolic semantic-validity check.

Setup/loading is excluded from warm per-problem latency and reported separately when recorded.

Attempt 2 again records CPU-package energy only if the M13 counter reader returns a valid package window; unavailable energy remains `null` and is not converted to zero.

## 9. Repeated units and uncertainty

Paired development uses all 512 examples for quality and 128 paired examples per training seed for complete-solve latency.

The hierarchical bootstrap rule is unchanged:

- resample the five training seeds with replacement;
- within each sampled seed, resample paired examples with replacement;
- 5000 replicates;
- percentile 95% intervals.

Attempt 2 bootstrap seed: `2026091519` (new RNG stream, same estimator/rule).

Five seeds are the primary repeated training units; no favorable seed subset may be reported as the claim.

## 10. Independent confirmation remains sealed

If and only if Attempt 2 passes development:

1. freeze/hash all five candidate and all five primary-baseline checkpoints plus selected `N_sup`;
2. then generate confirmation seed **`2026091402`**, 1024 examples;
3. cross-audit confirmation fingerprints against regenerated Attempt 1 hierarchy **and** the Attempt 2 hierarchy;
4. run one paired confirmation comparison using the same practical gate and a fresh bootstrap RNG stream;
5. include Attempt 2 dynamic-INT8 baseline and exact symbolic solver as contextual rows.

Attempt 1 never opened confirmation, so confirmation seed `2026091402` remains the first independent confirmation attempt. Development adaptivity across Attempt 1→Attempt 2 is disclosed, but no confirmation evidence has influenced the intervention.

If Attempt 2 development fails, confirmation remains unopened.

If Attempt 2 development passes but confirmation fails, `2026091402` becomes development evidence. Any later M14 claim must preregister a new intervention and use the already-named reserve confirmation seed `2026091403` (or another genuinely independent set) with sequential-selection interpretation stated before that later confirmation.

## 11. Attempt 2 outputs

Required retained artifacts:

- this protocol commit before Attempt 2 result;
- regenerated Attempt 1 fingerprint audit;
- Attempt 2 train/validation/development manifest and overlap audit;
- all 10 training curves/checkpoints/cost records (5 candidate + 5 baseline);
- validation `N_sup` table and selection record;
- raw paired development rows;
- practical-effect + bootstrap decision record;
- dynamic INT8 audit and symbolic context;
- measured complete-solve cost rows;
- confirmation freeze/manifest/raw rows only if development passes;
- result table and Pareto plot directly regenerated from raw rows;
- attempt ledger recording Attempt 1 and Attempt 2 without deleting misses.

## 12. Decision

Attempt 2 completes M14 **only** if the early-exit-weighted candidate passes the unchanged practical gate on the fresh development hierarchy and then passes the same gate again on untouched confirmation seed `2026091402`.

A clean Attempt 2 miss remains development evidence and M14 remains **INCOMPLETE**.
