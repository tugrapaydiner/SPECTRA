# Milestone 14 Attempt 3 Protocol — Semantic Early Exit

**Status: preregistered after Attempts 1–2 failed development, but before any Attempt 3 training, validation, development, timing, or confirmation result.**

Attempt 3 is a new adaptive-development attempt inside Milestone 14. It does not rewrite or erase either earlier miss.

## 1. Prior adaptive evidence

### Attempt 1

Actions run `34159474745`, head `66acbc62f48b3e5b1e81080897b89562edfd8bd8`, artifact `10032762057`.

The preregistered dim-64 FP recursive reserve reached strong validation quality, but its one-step development point was essentially tied in quality and slower than the unchanged larger FP baseline:

```text
quality difference          -0.005078125
95% CI                      [-0.04453125, +0.033203125]
latency ratio                1.094335291
95% CI                      [1.088790090, 1.099828683]
```

Attempt 1 remained **INCOMPLETE** and did not open confirmation.

### Attempt 2

Attempt-2 protocol commit `1282d026c89a6af696a225947aa67bd00e970917`. The input-conditioned single-stream recurrence reduced one recursive step to one shared block application while keeping the puzzle in the first update.

Attempt-2 run `34168430915` produced a valid scientific miss. Its reporting renderer later hit an infrastructure-only import-path bug; the raw experiment artifact remains authoritative for the scientific result.

Validation:

```text
larger FP single pass      semantic 0.841016   median 1.207073 ms
single-stream N=1          semantic 0.203125   median 1.026614 ms
single-stream N=2          semantic 0.806250   median 1.469636 ms
single-stream N=3          semantic 0.876953   median 1.913003 ms
single-stream N=4          semantic 0.907813   median 2.341106 ms
```

The validation-selected diagnostic fallback was `single_stream_n1`. Development:

```text
quality difference          -0.640625
95% CI                      [-0.669921875, -0.6109375]
latency ratio                0.855750596
95% CI                      [0.852578487, 0.858563651]
```

Attempt 2 remained **INCOMPLETE** and did not open confirmation seed `2026091402`.

The observed structure is falsifiable: later recurrent steps recover large amounts of quality, while many examples are already valid before the maximum depth. Attempt 3 tests whether a reference-free semantic stop can preserve later-step quality while avoiding unnecessary recurrent steps.

## 2. Frozen claim and numerical gate

The primary comparator remains the unchanged credibly trained larger FP `System1Student` (`dim=96`, two blocks). The primary endpoint remains strict symbolic Sudoku semantic validity after given-cell preservation.

The numerical M14 gate is **unchanged**.

### Path A — quality superiority at comparable latency

- `candidate - baseline >= +0.03` semantic-validity points;
- paired hierarchical-bootstrap 95% CI lower bound `> 0`;
- median complete-solve latency ratio `<= 1.15`;
- latency-ratio 95% CI upper bound `<= 1.20`.

### Path B — predeclared quality/cost trade-off

- `candidate - baseline >= -0.02`;
- quality 95% CI lower bound `>= -0.03`;
- median complete-solve latency ratio `<= 0.65`;
- latency-ratio 95% CI upper bound `<= 0.75`.

No result from Attempts 1–2 changes these thresholds.

## 3. Sequential-selection accounting and final-test hierarchy

Attempt 3 is explicitly informed by two prior development attempts. Therefore any final interpretation must state that model/system development was adaptive across three attempts before confirmation.

The same train/validation/development hierarchy may be reused for development. Confirmation seed `2026091402` has never been generated/opened and remains the independent confirmation set.

- Confirmation is generated only after Attempt 3 passes development.
- Before confirmation, candidate checkpoints, primary baseline checkpoints, maximum recurrent budget, termination rule, and all hashes are frozen.
- If confirmation `2026091402` is opened and fails, it becomes development evidence. Any subsequent M14 claim must use reserve confirmation seed `2026091403` or another independently preregistered set.
- No favorable seed filtering is allowed.

## 4. Data and training

Attempt 3 uses the exact existing M14 generated unique 4×4 Sudoku hierarchy:

```text
development seed            2026091401
train / validation / dev    4096 / 512 / 512
box                          2
clues                        uniform 6..10
augmentation                 false
solution_method              random_backtracking
```

The candidate and primary baseline are retrained from scratch in the Attempt-3 workflow with the same five seeds:

```text
1401, 2402, 3403, 4404, 5405
```

Training hyperparameters remain:

```text
AdamW
lr                           1e-3
weight decay                 0.01
batch size                   64
steps                        1800 per model per seed
grad clip                    1.0
```

Candidate architecture/training is exactly Attempt 2's `InputConditionedSingleStreamTRM`:

```text
dim                          96
one shared transformer block
heads                        4
FP32
training N_sup               4
four-step blank-cell CE weights [0.1,0.2,0.3,0.4] / sum
```

The primary baseline remains the two-block dim-96 FP `System1Student` trained with blank-cell CE. It is not weakened or shortened.

No new hidden-width, learning-rate, data-size, loss-weight, ternary, search-action, or training-step sweep is permitted in Attempt 3.

## 5. Attempt-3 inference intervention

The only new intervention is **online semantic early exit** over the trained single-stream recurrence.

For a problem `x`, encode the puzzle once and start `y=0`. For each recurrent step up to a validation-selected maximum budget `K`:

1. execute exactly one input-conditioned shared-block recurrent update;
2. execute the output head and argmax;
3. copy immutable givens from `x` into the candidate board;
4. call the independent symbolic Sudoku semantic validator on `(x, candidate)`;
5. if the candidate is a valid full Sudoku completion respecting the givens, stop immediately and return it;
6. otherwise continue until `K`; at the maximum budget return the final candidate even if invalid.

The semantic stop uses **no reference target solution**. For this generated unique-Sudoku task, a complete valid board respecting the givens is independently verifiable from the puzzle constraints alone.

This termination rule has a task-specific inductive bias and must be described as such. A positive result supports the **hybrid trained-recursive + symbolic-validity-stop solver**, not a generic claim that neural halting is solved or that the method transfers unchanged to tasks without cheaply verifiable answers.

The larger FP baseline is also passed through the same given-cell projection and one final semantic validity check. It has no recurrent continuation to exploit after an invalid answer.

## 6. Cost accounting

The candidate uses a specialized inference loop rather than the compatibility `TRM.forward` wrapper. The loop may omit the unused learned halt head and step-dictionary construction, because Attempt 3 does not use learned halting. It must not omit any computation required by the declared recurrence, output, projection, or semantic stop.

Complete-solve latency includes:

- token and positional embedding once;
- every executed shared transformer block;
- recurrent RMSNorm/residual update;
- output head and argmax at every tested step;
- given-cell preservation at every tested step;
- symbolic semantic-validity check at every tested step;
- Python/control overhead of the early-exit loop.

Per-example raw rows record actual executed recurrent steps and block applications. The maximum-budget candidate is not credited with unexecuted later steps.

The primary baseline complete solve includes its two transformer blocks, output head/argmax, given-cell projection, and semantic validity check.

CPU package energy is recorded only if M13's validated powercap reader reports a complete valid package window; otherwise energy remains unavailable/null and is not converted to zero. Package energy is not GPU or whole-system energy.

## 7. Validation-only maximum-budget selection

Attempt 3 evaluates semantic-early-exit maximum budgets:

```text
K in {1, 2, 3, 4}
```

The same trained checkpoint is used for all four; there is no retraining per budget.

Validation records semantic validity, exact reference match, blank-cell accuracy, complete-solve latency, executed-step distribution, and mean/median block applications.

The deterministic selection rule is:

1. Compute point quality difference and latency ratio versus the larger FP baseline.
2. Path-B point eligible: quality difference `>= -0.02` and latency ratio `<= 0.65`.
3. Path-A point eligible: quality difference `>= +0.03` and latency ratio `<= 1.15`.
4. If any Path-B point is eligible, choose highest semantic validity; tie break lower median latency, then lower `K`.
5. Else if any Path-A point is eligible, choose highest semantic validity; same tie break.
6. Else choose the highest-semantic-validity point with latency ratio `<=1.15`; if none, choose lowest-latency point. This is diagnostic fallback only and does not change the development gate.

Validation does not use bootstrap uncertainty for configuration selection. The unchanged hierarchical-bootstrap rule is used on development and confirmation only.

## 8. Development and uncertainty

The validation-selected maximum budget is evaluated exactly once on all 512 development examples for all five training seeds, paired with the unchanged larger FP baseline.

The primary effect uses the existing M14 paired hierarchical bootstrap:

- resample five training seeds with replacement;
- resample paired examples within each sampled seed;
- at least 5000 replicates;
- development bootstrap seed `2026091499`;
- percentile 95% intervals.

Latency uses paired complete-solve observations on the preregistered measured subset and bootstraps the median ratio.

If the unchanged practical gate fails, Attempt 3 is **INCOMPLETE** and confirmation remains sealed.

## 9. Confirmation

Only after a development pass:

1. hash/freeze all five candidate checkpoints;
2. hash/freeze all five primary-baseline checkpoints;
3. freeze `InputConditionedSingleStreamTRM` semantics id;
4. freeze selected max budget `K`;
5. freeze semantic termination rule and measurement code identity;
6. generate independent confirmation seed `2026091402`, 1024 examples;
7. audit fingerprints against the development hierarchy;
8. run the same paired comparison once;
9. use bootstrap seed `2026091500`.

M14 completes only if the same numerical gate independently passes confirmation.

## 10. Context baselines and claim boundary

Attempt 3 also reports when supported:

- dynamic INT8 conversion of the trained larger single-pass model, clearly labelled as a conventional partial dynamic-INT8 deployment baseline where unsupported attention remains floating point;
- exact MRV/backtracking as a task-specific symbolic contextual reference.

Neither replaces the primary FP baseline.

No scaling-law claim is permitted. A positive Attempt-3 result would establish only the predeclared quality/cost claim for this trained hybrid solver on this generated 4×4 Sudoku distribution and measured CPU environment.

## 11. Required evidence

Attempt 3 must retain:

- this preregistration commit before any Attempt-3 result;
- compact provenance for Attempts 1 and 2 and their failed gates;
- development manifest;
- all ten training runs/checkpoints/curves;
- validation raw rows and deterministic selection record;
- raw executed-step/block counts;
- development raw rows/effect/decision;
- energy availability record;
- INT8 conversion audit and symbolic context;
- confirmation freeze manifest and independent confirmation only after development pass;
- directly generated result table/Pareto figure plus hashes;
- full regression result;
- updated M14/state documentation.

**M14's acceptance gate remains unchanged: only an independently confirmed practical improvement completes the milestone.**