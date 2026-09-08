# Milestone 14 Attempt 5 Protocol — Dual-Stream FP Recursive Semantic Exit

**Status: preregistered after Attempts 1–4 failed development, but before any Attempt 5 training, validation, development, timing, or confirmation result.**

Attempt 5 is a new adaptive-development attempt inside Milestone 14. It does not rewrite any prior miss. Confirmation seed `2026091402` remains unopened at preregistration time.

## 1. Prior adaptive evidence and why Attempt 5 is different

M14 has accumulated four development attempts under the same numerical claim gate.

### Attempt 1 — original trained families and dim-64 FP reserve

Run `34159474745`, head `66acbc62f48b3e5b1e81080897b89562edfd8bd8`, artifact `10032762057`, ZIP SHA256 `04a5b810d697227c4ccf88a365cf529647734ea0d95deec363f84fcaae885522`.

The strongest retained recursive operating points were the **ordinary dual-stream floating-point TRM**, especially the preregistered dim-64 reserve:

```text
validation, dim-64 FP recursive
N_sup=1     semantic 0.837890625    median 1.15491475 ms
N_sup=2     semantic 0.942187500    median 1.76506700 ms
N_sup=3     semantic 0.975000000    median 2.36858675 ms
N_sup=4     semantic 0.981640625    median 2.96389675 ms

same-run larger FP single-pass baseline
             semantic 0.841015625    median 1.06141575 ms
```

The one-step dim-64 candidate was essentially quality-tied with the baseline but slower on development:

```text
quality difference          -0.005078125
95% CI                      [-0.04453125, +0.033203125]
latency ratio                1.094335291
95% CI                      [1.088790090, 1.099828683]
```

Attempt 1 therefore missed the quality-superiority threshold and stayed **INCOMPLETE**. Confirmation was not opened.

### Attempt 2 — single-stream structural simplification

Run `34168430915`, protocol commit `1282d026c89a6af696a225947aa67bd00e970917`, artifact `10035110496`. One shared block per recurrent step was cheaper but first-step quality collapsed. Development quality difference was `-0.640625`; confirmation remained sealed.

### Attempt 3 — semantic early exit on the single-stream model

Run `34169488811`, protocol commit `41e23c69fd6a0664ce1f5e6cd4712f95898294d8`, artifact `10035348365`, ZIP SHA256 `fb653a307d918f8e010d6648694ef2bdeb08654bee9faaf093f1f71ff07eaaac`.

Reference-free semantic stopping reduced unnecessary later steps, but only about 20% of validation examples were valid after the first one-block step. K=4 reached `0.910546875` semantic validity, but the deeper hybrid path remained about `1.385x` the baseline latency. Development stayed negative and confirmation remained sealed.

### Attempt 4 — fixed front-loaded anytime supervision

Run `34170096945`, protocol commit `7b2ec6fb0c3264307b9623603431072c9c62b769`, artifact `10035628899`, ZIP SHA256 `1d7a346a47f82b360d25d97b6650a555216385b8ce523a0867302a98002438ee`.

The sole candidate training change was the fixed reverse supervision schedule `[0.4,0.3,0.2,0.1]`. It did not improve early correctness: validation K=1 fell to `0.176953125`, K=4 to `0.640625`. Development selected the diagnostic K=1 point:

```text
quality difference          -0.680078125
95% CI                      [-0.716796875, -0.646484375]
latency ratio                0.775202180
95% CI                      [0.771741396, 0.778782654]
```

Attempt 4 was therefore **INCOMPLETE** and confirmation remained sealed. Its workflow also contained a non-scientific unit-test bug: exact binary floating-point equality was used for `sum([0.4,0.3,0.2,0.1]) == 1.0`. The experiment, renderer and evidence-hierarchy checks completed; the assertion has been corrected on the Attempt-5 branch with an absolute-tolerance comparison. No experimental threshold or result changed.

## 2. Attempt-5 hypothesis

The prior evidence points back to the original SPECTRA recurrence rather than the single-stream simplification.

The dim-64 dual-stream FP model already solves roughly `84%` after one supervision step and `94%` after two. Therefore most examples may be able to stop after the first supervision step if the system checks independent Sudoku validity online; only the hard tail needs the second expensive supervision step.

Attempt 5 tests the predeclared hypothesis:

> **A trained dim-64 dual-stream FP recursive SPECTRA model with reference-free semantic early exit can retain the strong two-step quality while reducing average complete-solve work enough to satisfy the unchanged M14 practical-improvement gate against the trained larger FP single-pass baseline.**

This is not a new width search. `dim=64` is fixed because it was the one preregistered reserve intervention already evaluated in Attempt 1 and is now being carried forward as adaptive development evidence.

## 3. Frozen primary claim and numerical gate

The primary comparator remains the credibly trained larger FP `System1Student`, dim 96, two transformer blocks, confidence head frozen and excluded from trainable parameter count.

The primary endpoint remains strict symbolic Sudoku semantic validity after immutable-given restoration.

The numerical gate is **identical to all previous M14 attempts**.

### Path A — quality superiority at comparable latency

- semantic-validity difference `candidate - baseline >= +0.03`;
- paired hierarchical-bootstrap 95% CI lower bound `> 0`;
- median complete-solve latency ratio `candidate / baseline <= 1.15`;
- latency-ratio bootstrap 95% CI upper bound `<= 1.20`.

### Path B — quality/cost trade-off

- semantic-validity difference `candidate - baseline >= -0.02`;
- quality 95% CI lower bound `>= -0.03`;
- median complete-solve latency ratio `<= 0.65`;
- latency-ratio bootstrap 95% CI upper bound `<= 0.75`.

No threshold is changed in response to Attempts 1–4.

## 4. Sequential-selection accounting

Attempt 5 is informed by four prior development attempts. If it reaches confirmation, the final interpretation must state that candidate development was adaptive across **five** attempts before the independent confirmation set was opened.

The same train/validation/development hierarchy may continue to serve development. Confirmation seed `2026091402` remains unopened and is eligible as the first independent confirmation set.

- Confirmation can be generated only after Attempt 5 passes the unchanged development gate.
- Before confirmation, all five candidate checkpoints, all five primary-baseline checkpoints, the selected maximum supervision budget, semantic termination rule, architecture identity, training objective, and file hashes are frozen.
- If confirmation `2026091402` is opened and fails, it is consumed as development evidence. Any subsequent claim must use reserve confirmation seed `2026091403` or another independently preregistered set.
- All five training seeds remain included; no favorable-seed filtering is permitted.

## 5. Data hierarchy

Attempt 5 uses the exact M14 generated unique 4x4 Sudoku development hierarchy:

```text
development data seed       2026091401
train / validation / dev    4096 / 512 / 512
box                          2
clues                        uniform 6..10
augmentation                 false
solution method              random_backtracking
```

Only training data receives gradients. Validation selects the maximum semantic-exit budget. Development is evaluated once for that selected point.

If development passes:

```text
confirmation seed            2026091402
confirmation examples        1024
```

with explicit fingerprint-overlap audit against the complete development hierarchy.

## 6. Models and training budgets

Attempt 5 retrains from scratch exactly two learned families under the same manifest and five seeds:

```text
seeds                        1401, 2402, 3403, 4404, 5405
optimizer                    AdamW
learning rate                1e-3
weight decay                 0.01
batch size                   64
optimizer steps              1800 per model per seed
grad clip                    1.0
```

### Candidate — original dual-stream FP recursive TRM

```text
class                        TRM
dim                          64
blocks                       1 shared transformer block
heads                        4
n                            1
T                            1
training N_sup               4
precision                    FP32
ternary / A8                 false / false
alpha_y / alpha_z            0.1 / 0.1
```

Each supervision step therefore performs the original two shared-block applications:

```text
z <- f(x_emb + y + z)
y <- f(y + z)
```

with the existing recurrent residual/norm semantics.

Candidate training objective is unchanged from Attempt 1:

```text
four blank-cell CE losses
weights [0.1,0.2,0.3,0.4] / sum
```

No loss reweighting is permitted in Attempt 5.

### Primary baseline — unchanged larger FP single-pass

```text
System1Student
dim                          96
blocks                       2
heads                        4
precision                    FP32
confidence head              frozen / excluded
objective                    blank-cell CE
```

The baseline is not weakened, shortened, quantized for the primary comparison, or seed-selected.

The candidate must remain smaller in trainable parameter count than the baseline; this is asserted before result acceptance.

Training cost is reported separately from inference cost: train wall time, steps, examples sampled, trainable parameters, and block applications.

## 7. Attempt-5 inference intervention — semantic exit on original dual-stream recurrence

The candidate uses a specialized batch-size-one online inference loop. It does **not** call the compatibility `TRM.forward` wrapper and does **not** execute the learned halt head, because M14 Attempt 5 is testing independently verifiable task termination rather than learned halting.

For a puzzle `x` and validation-selected maximum supervision budget `K`:

1. compute token + positional embedding once;
2. initialize `y=z=0`;
3. execute one complete original dual-stream supervision step:
   - one input-conditioned `z` shared-block update;
   - one `y` shared-block update;
   - recurrent norms/residual arithmetic;
4. run output head and argmax;
5. restore immutable givens from `x`;
6. call the independent symbolic Sudoku semantic validator;
7. if the board is a complete valid Sudoku respecting givens, stop immediately and return it;
8. otherwise continue with the next full supervision step until `K`;
9. at `K`, return the final candidate even if invalid.

The semantic stop receives no target/reference solution. This is a task-specific hybrid inductive bias and must be reported as such. A positive result supports the **trained dual-stream recursive SPECTRA + symbolic-validity termination system** on this task; it does not establish generic neural halting or cross-task transfer.

## 8. Exactly-once complete-solve cost

The specialized candidate timing window includes:

- input embedding once;
- every actually executed dual-stream recurrent update;
- exactly two shared-block applications per executed supervision step;
- recurrent norm/residual arithmetic;
- output head and argmax after every tested supervision step;
- immutable-given restoration after every tested step;
- one semantic-validity check after every tested step;
- Python/control overhead of the online early-exit loop.

The final candidate semantic decision is reused by the evaluator and is not charged twice.

The primary baseline timing includes:

- two feed-forward transformer blocks;
- output head and argmax;
- immutable-given restoration;
- exactly one final semantic-validity check.

Raw rows record actual executed supervision steps, actual shared-block applications, output-head calls, semantic-check calls, stop reason, and measured latency.

CPU-package energy is measured only if the M13 validated powercap reader returns a complete valid package window. Otherwise energy is unavailable/null, never zero. CPU package is neither GPU nor whole-system energy.

## 9. Validation-only maximum-budget selection

The same trained dim-64 checkpoint is evaluated at:

```text
K in {1,2,3,4}
```

No retraining occurs per inference budget.

Validation records semantic validity, exact reference match, blank-cell accuracy, median/p95 complete-solve latency, actual executed-step/block distribution, and semantic-stop fraction.

Selection is deterministic and unchanged in spirit from Attempts 3–4:

1. Path-B point eligible: quality difference `>= -0.02` and latency ratio `<= 0.65`.
2. Path-A point eligible: quality difference `>= +0.03` and latency ratio `<= 1.15`.
3. If any Path-B point is eligible, choose highest semantic validity; ties lower latency, then lower K.
4. Else if any Path-A point is eligible, choose highest semantic validity; same tie breaks.
5. Else choose highest-semantic-validity point with latency ratio `<=1.15`; if none exists, choose the lowest-latency point. This fallback is diagnostic only and cannot alter the development gate.

Validation selection uses no development or confirmation data and no bootstrap uncertainty.

## 10. Development and uncertainty

The validation-selected point is evaluated once on all 512 development examples for each of the five training seeds, paired example-by-example with the unchanged primary baseline.

The primary uncertainty rule remains the M14 hierarchical bootstrap:

- resample the five training seeds with replacement;
- within each sampled seed, resample paired examples with replacement;
- 5000 bootstrap replicates;
- development bootstrap seed `2026091499`;
- percentile 95% intervals.

Latency uncertainty uses the paired measured complete-solve subset and bootstraps the median candidate/baseline ratio.

If the unchanged gate fails, Attempt 5 is **INCOMPLETE** and confirmation remains sealed.

## 11. Independent confirmation

Only after development passes:

1. freeze and hash all five dim-64 FP candidate checkpoints;
2. freeze and hash all five primary FP baseline checkpoints;
3. freeze candidate architecture and original `[0.1,0.2,0.3,0.4]` training objective identity;
4. freeze selected max budget K;
5. freeze semantic termination/timing implementation identity;
6. generate confirmation seed `2026091402`, 1024 examples;
7. audit fingerprints against the entire development hierarchy;
8. execute exactly one paired final comparison;
9. use confirmation bootstrap seed `2026091500`.

M14 completes only if the **same unchanged numerical gate** independently passes confirmation.

## 12. Context baselines and claim boundary

Attempt 5 also reports where supported:

- dynamic INT8 conversion of the trained larger single-pass baseline, clearly labelled partial/conventional dynamic INT8 where unsupported attention remains FP;
- exact MRV/backtracking as a task-specific symbolic contextual reference.

Neither can replace the primary FP comparator.

The earlier trained ternary/search variants remain preserved in Attempt-1 evidence rather than being rerun merely to increase the number of operating points.

No scaling-law claim is permitted. Even a positive result is limited to the generated unique 4x4 Sudoku distribution, this adaptive five-attempt development sequence, and the measured CPU complete-solve environment.

## 13. Required evidence

Attempt 5 must retain:

- this preregistration commit before any Attempt-5 result;
- provenance and failed-gate summaries for Attempts 1–4;
- exact train/validation/development manifest;
- all ten trained checkpoints and curves;
- candidate/baseline parameter and training-cost audit;
- validation raw rows and deterministic selection record;
- actual dual-stream step/block work counts;
- development raw rows, paired effect, uncertainty, and decision;
- physical-energy availability record;
- INT8 and symbolic contextual rows;
- confirmation freeze manifest and independent confirmation files only after a development pass;
- table and Pareto figure rendered directly from raw artifacts with hashes;
- full regression result;
- M14 acceptance/state documentation.

**Acceptance remains the original user-specified M14 gate: only an independently confirmed practical improvement over the credible trained baseline completes Milestone 14.**