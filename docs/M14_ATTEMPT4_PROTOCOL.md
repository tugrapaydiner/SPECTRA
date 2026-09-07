# Milestone 14 Attempt 4 Protocol — Front-Loaded Anytime Supervision

**Status: preregistered after Attempts 1–3 failed development, but before any Attempt 4 training, validation, development, timing, or confirmation result.**

Attempt 4 is a new adaptive-development attempt inside Milestone 14. Attempts 1–3 remain immutable failed/incomplete evidence and are not rewritten.

## 1. Prior adaptive evidence

### Attempt 1 — ordinary recursive family / dim-64 reserve

Run `34159474745`, head `66acbc62f48b3e5b1e81080897b89562edfd8bd8`, artifact `10032762057`.

The best reserve one-step FP recursive candidate was quality-tied but slower than the larger FP single-pass baseline on development:

```text
quality difference          -0.005078125
95% CI                      [-0.04453125, +0.033203125]
latency ratio                1.094335291
95% CI                      [1.088790090, 1.099828683]
```

Status: **INCOMPLETE**. Confirmation not opened.

### Attempt 2 — one-block input-conditioned single-stream recurrence

Protocol commit `1282d026c89a6af696a225947aa67bd00e970917`; run `34168430915`; artifact `10035110496`.

The structural intervention made one step cheaper but first-step quality collapsed. Development selected the diagnostic `single_stream_n1` point:

```text
quality difference          -0.640625
95% CI                      [-0.669921875, -0.6109375]
latency ratio                0.855750596
95% CI                      [0.852578487, 0.858563651]
```

Status: **INCOMPLETE**. Confirmation not opened.

### Attempt 3 — reference-free semantic early exit

Protocol commit `41e23c69fd6a0664ce1f5e6cd4712f95898294d8`; run `34169488811`; head `d1e491f412494c0756753bfa4b78b8002133d39c`; artifact `10035348365`, ZIP SHA256 `fb653a307d918f8e010d6648694ef2bdeb08654bee9faaf093f1f71ff07eaaac`.

Attempt 3 used the same trained single-stream recurrence but stopped online whenever the independent symbolic Sudoku validity condition became true. Validation showed:

```text
larger FP baseline     semantic 0.851563   median 0.737865 ms
semantic_exit K=1      semantic 0.203125   median 0.601685 ms   mean blocks 1.000
semantic_exit K=2      semantic 0.808203   median 1.018277 ms   mean blocks 1.797
semantic_exit K=3      semantic 0.880469   median 1.020919 ms   mean blocks 1.989
semantic_exit K=4      semantic 0.910547   median 1.021901 ms   mean blocks 2.108
```

Thus later steps had enough task capability, but too few examples were correct after step 1 to make the hybrid solver latency-competitive. K=3 was only `+0.02890625` above the baseline—just below the frozen Path-A `+0.03` minimum—and had latency ratio `1.383612`. K=4 exceeded the quality target by `+0.058984375` but had latency ratio `1.384942`.

Validation selected K=1 only as the deterministic diagnostic fallback. Development:

```text
quality difference          -0.637890625
95% CI                      [-0.6625, -0.612890625]
latency ratio                0.812743047
95% CI                      [0.806199172, 0.822727758]
```

Status: **INCOMPLETE**. Confirmation seed `2026091402` remains unopened.

The falsifiable diagnosis for Attempt 4 is therefore not “more capacity.” The same one-block shared recurrence eventually solves many examples, but the existing deep-supervision objective gives the first step only 10% of the supervised weight.

## 2. Frozen primary claim and numerical gate

The primary comparator remains the same credibly trained larger FP `System1Student` (`dim=96`, two transformer blocks). The primary endpoint remains strict symbolic Sudoku semantic validity after immutable-given preservation.

**The M14 practical gate is unchanged from Attempt 1.**

### Path A — quality superiority at comparable latency

- semantic-validity difference `candidate - baseline >= +0.03`;
- paired hierarchical-bootstrap 95% CI lower bound `> 0`;
- median complete-solve latency ratio `candidate / baseline <= 1.15`;
- latency-ratio 95% CI upper bound `<= 1.20`.

### Path B — predeclared quality/cost trade-off

- semantic-validity difference `candidate - baseline >= -0.02`;
- quality 95% CI lower bound `>= -0.03`;
- median complete-solve latency ratio `<= 0.65`;
- latency-ratio 95% CI upper bound `<= 0.75`.

Attempts 1–3 do not modify or relax these thresholds.

## 3. Sequential-selection accounting

Attempt 4 is explicitly informed by three prior development attempts. A final claim, if reached, must state that development was adaptive across **four** attempts before confirmation.

The train/validation/development hierarchy can be reused for development. Confirmation seed `2026091402` has never been generated/opened and remains the first independent confirmation set.

- Confirmation is generated only after Attempt 4 passes development.
- Candidate checkpoints, primary-baseline checkpoints, semantic-stop rule, selected maximum budget, architecture id, and hashes are frozen before confirmation.
- If confirmation `2026091402` is opened and fails, it is consumed as development evidence; any later claim requires reserve seed `2026091403` or another preregistered independent set.
- All five training seeds remain included. No seed filtering is allowed.

## 4. Data hierarchy

Attempt 4 uses the exact M14 generated unique 4×4 Sudoku development hierarchy:

```text
development data seed       2026091401
train / validation / dev    4096 / 512 / 512
box                          2
clues                        uniform 6..10
augmentation                 false
solution method              random_backtracking
```

Only train rows receive gradients. Validation selects the Attempt-4 maximum early-exit budget. Development is used exactly once for the selected point.

If development passes:

```text
confirmation seed            2026091402
confirmation examples        1024
```

with an explicit fingerprint overlap audit.

## 5. Models and fixed training budgets

Attempt 4 retrains from scratch:

1. the exact Attempt-2/3 `InputConditionedSingleStreamTRM` candidate;
2. the unchanged primary larger FP `System1Student` baseline.

Five model seeds:

```text
1401, 2402, 3403, 4404, 5405
```

Candidate architecture remains:

```text
dim                          96
one shared transformer block
heads                        4
FP32
training N_sup               4
input-conditioned y-only recurrence
z compatibility state        frozen zero
```

Primary baseline remains:

```text
System1Student
dim                          96
blocks                        2
heads                         4
FP32
confidence head               frozen / excluded from trainable count
```

The candidate must remain smaller in trainable parameters than the primary baseline.

Training hyperparameters are unchanged:

```text
optimizer                     AdamW
learning rate                 1e-3
weight decay                  0.01
batch size                    64
optimizer steps               1800 per model per seed
grad clip                     1.0
```

No hidden-width, LR, data-size, training-step, ternary, architecture, router, halter, search-action, or baseline sweep is permitted in Attempt 4.

## 6. The sole Attempt-4 intervention: fixed front-loaded deep supervision

The previous candidate objective used four blank-cell CE losses with normalized weights:

```text
[0.1, 0.2, 0.3, 0.4]
```

Attempt 4 changes **only** this weighting for the single-stream candidate to the exact reverse schedule:

```text
[0.4, 0.3, 0.2, 0.1]
```

The weights still sum to one. The schedule is not chosen by a search; it is the symmetric reversal of the historical later-step weighting and directly tests the hypothesis that weak early supervision, rather than insufficient representational capacity, is the limiting factor for adaptive latency.

The primary single-pass baseline retains the exact same blank-cell CE objective as before.

No other optimizer or training behavior changes.

## 7. Inference: same semantic early exit as Attempt 3

Attempt 4 uses the exact reference-free task-specific semantic early-exit rule preregistered in Attempt 3.

For max budget `K`:

1. encode puzzle once;
2. execute one input-conditioned shared block update;
3. decode with output head/argmax;
4. restore immutable givens;
5. call the independent symbolic Sudoku validity checker;
6. stop immediately if the board is a valid full completion respecting givens;
7. otherwise continue until `K`.

The target/reference solution is never used inside inference or the stop decision. The learned halt head is not used.

The task-specific bias is explicit: a positive result is for the **hybrid trained recursive solver + symbolic validity stop**, not generic neural halting.

## 8. Validation-only maximum-budget selection

The same trained candidate is evaluated at:

```text
K in {1,2,3,4}
```

Validation records semantic validity, exact reference match, blank-cell accuracy, complete-solve latency, actual executed-step distribution, actual block applications, and semantic-stop fraction.

Selection is unchanged from Attempt 3:

1. Path-B point eligible: quality difference `>= -0.02` and latency ratio `<=0.65`.
2. Path-A point eligible: quality difference `>= +0.03` and latency ratio `<=1.15`.
3. Prefer any Path-B eligible point, highest semantic validity; ties lower latency then lower K.
4. Else prefer any Path-A eligible point, same tie breaks.
5. Else highest-semantic point with latency ratio `<=1.15`; if none, lowest-latency point. This fallback is diagnostic only and cannot pass development by changing the gate.

No development or confirmation information enters selection.

## 9. Complete-solve cost

The complete-solve timing path is the exactly-once semantic timing contract from Attempt 3:

Candidate includes:

- embedding once;
- every executed recurrent block;
- recurrent norm/residual update;
- output head and argmax at every executed step;
- given restoration at every executed step;
- one semantic validity check per executed step, with the final internal decision reused rather than checked twice;
- Python/control overhead.

Primary baseline includes:

- two feed-forward transformer blocks;
- output head and argmax;
- given restoration;
- one final semantic validity check.

Raw rows record actual steps, blocks, output-head calls, semantic checks, stop reason, and latency.

CPU-package energy is used only when the M13 validated RAPL reader reports a complete valid package window. Otherwise energy remains unavailable/null. Package energy is neither GPU nor whole-system energy.

## 10. Development uncertainty

The validation-selected point is evaluated once on all 512 development examples for every one of the five training seeds, paired with the primary baseline.

The unchanged M14 hierarchical bootstrap is used:

- resample training seeds with replacement;
- within each sampled seed, resample paired examples with replacement;
- 5000 replicates;
- development bootstrap seed `2026091499`;
- percentile 95% intervals.

Latency uncertainty uses the paired measured complete-solve subset and bootstraps the median ratio.

If the unchanged practical gate fails, Attempt 4 is **INCOMPLETE** and confirmation stays sealed.

## 11. Independent confirmation

Only after a development pass:

1. hash/freeze all five candidate checkpoints;
2. hash/freeze all five primary-baseline checkpoints;
3. freeze architecture semantics;
4. freeze front-loaded loss schedule identity;
5. freeze max budget K and semantic-stop rule;
6. generate independent confirmation seed `2026091402`, 1024 rows;
7. audit exact/group fingerprints against the development hierarchy;
8. run one paired final comparison;
9. use confirmation bootstrap seed `2026091500`.

M14 completes only if the **same unchanged gate** independently passes confirmation.

## 12. Context baselines and limits

Attempt 4 continues to report when supported:

- dynamic INT8 conversion of the trained larger single-pass baseline, explicitly labelled partial/conventional dynamic INT8 where unsupported attention remains FP;
- exact MRV/backtracking as a task-specific symbolic contextual reference.

Neither replaces the primary FP baseline.

No scaling-law claim is permitted. Even a positive result is limited to this generated 4×4 Sudoku distribution, this adaptive development sequence, and the measured complete-solve CPU environment.

## 13. Required evidence

Attempt 4 must retain:

- this protocol commit predating any Attempt-4 result;
- provenance for all three prior failed attempts;
- exact development manifest;
- ten trained checkpoints/curves and training-cost rows;
- explicit fixed `[0.4,0.3,0.2,0.1]` objective identity in raw evidence;
- validation raw rows/selection;
- actual executed-step/block distributions;
- development raw rows/effect/decision;
- physical-energy availability record;
- INT8/symbolic context;
- confirmation freeze and independent confirmation files only after dev pass;
- result table and Pareto plot generated directly from raw artifacts;
- full regression result;
- M14 acceptance/state documentation.

**Acceptance remains the original user-specified M14 gate: only an independently confirmed practical improvement completes the milestone.**