# Milestone 14 Development Amendment B — Validator-Gated Adaptive Depth

**Status: preregistered before Amendment-B training, validation, fresh-development, or confirmation results.**

This amendment preserves the primary M14 hypothesis, trained primary baseline, semantic-validity endpoint, practical-effect thresholds, uncertainty rule, and independent-confirmation requirement from [`M14_PROTOCOL.md`](M14_PROTOCOL.md). It is a new development attempt after two preserved misses; it does not rewrite either result.

## 1. Development history retained before this amendment

### Attempt 1

The first M14 hierarchy (`run 34159474745`, artifact `10032762057`, ZIP SHA-256 `04a5b810d697227c4ccf88a365cf529647734ea0d95deec363f84fcaae885522`) was valid but incomplete. Its validation-selected `n=1` FP recursive operating point was close in quality to the larger single-pass FP baseline but slower. The preregistered `dim=64` reserve remained quality-competitive on development:

```text
semantic-validity difference   -0.005078125
95% CI                          [-0.04453125, +0.033203125]
latency ratio                   1.0943352911
95% CI                          [1.0887900899, 1.0998286826]
```

The same trained recursive family at fixed depth 2 had materially higher validation semantic validity than the baseline, but fixed depth 2 was too slow. Search variants were substantially slower and did not recover a favorable operating point. No confirmation set was opened.

### Amendment A

Amendment A (`run 34164642690`, artifact `10034038517`, ZIP SHA-256 `943c05666b86fe8e1567e552602a593d7aacbdce7935903f51b1f053c503c6c3`) tested the falsifiable hypothesis that the inner `z` update was expendable. It was not.

Validation-selected `n=0_dim64` result:

```text
semantic validity     0.0000 across all five seeds
median latency        0.87045 ms
single-pass baseline  0.84102 semantic validity / 1.04049 ms
```

Fresh-development effect:

```text
quality difference    -0.8484375
95% CI                 [-0.86640625, -0.8296875]
latency ratio           0.8344037
95% CI                  [0.8326569, 0.8362158]
```

Thus removing the `z` update produced only a modest latency reduction and catastrophically destroyed task capability. The fresh development-reserve seed `2026091404` is now development evidence. Confirmation seeds `2026091402` and `2026091403` remain unopened.

## 2. Falsifiable diagnosis

The evidence now supports a narrower runtime hypothesis:

1. `n=1` recurrence is required for capability;
2. one recurrent supervision step can already solve many boards;
3. a second recurrent step materially improves solve rate;
4. paying the second step for **every** board makes the fixed-depth point too slow.

Amendment B therefore preserves the successful `n=1` state transition and changes only **when another supervision step is executed**.

The task has an exact, reference-free semantic validator: `model.verifier.sudoku_correct(puzzle, candidate, box=2)`. It checks completion, row/column/box validity, and clue preservation without consulting the retained target solution. This validator already participates in M14 complete-solve timing. Amendment B uses it online: if the current decode is already a valid Sudoku solution, later recurrent steps are skipped; otherwise execution continues up to the fixed maximum depth.

This is a task-specific verifier-gated compute policy, not a claim that arbitrary tasks have cheap exact validators. The final claim, if confirmed, must say so.

## 3. Candidate family

Train floating-point recursive TRMs with the original successful recurrence:

```text
n = 1
T = 1
training N_sup = 4
n_layers = 1
heads = 4
alpha_y = 0.1
alpha_z = 0.1
ternary = false
act8 = false
```

Widths:

```text
32
48
64
```

Each trained checkpoint is evaluated with two adaptive execution policies:

```text
semantic_stop_max2
semantic_stop_max3
```

Execution semantics for a batch-size-one solve:

1. initialize the explicit M11 recurrent state once;
2. execute exactly one supervision step;
3. decode, clamp givens, and run `sudoku_correct`;
4. stop immediately on semantic validity;
5. otherwise execute the next supervision step and repeat until `max_steps`;
6. after the final allowed step, return that decode whether valid or not.

There is no target/reference-solution access in the stopping rule. No learned verifier, router, action policy, search codebook, or post-hoc repair is used in the primary candidate. Their work counts are zero; the exact symbolic validity checks are counted and timed.

Candidate IDs:

```text
fp32_semstop2
fp32_semstop3
fp48_semstop2
fp48_semstop3
fp64_semstop2
fp64_semstop3
```

## 4. Training budget and baseline

The primary baseline is retrained in the Amendment-B run using the unchanged M14 definition:

```text
System1Student
FP32
dim = 96
2 transformer blocks
heads = 4
confidence head frozen
```

Candidate and baseline training use the original M14 train/validation hierarchy and exactly:

```text
training/data hierarchy seed  2026091401
train examples                4096
validation examples            512
training seeds                 1401, 2402, 3403, 4404, 5405
optimizer                      AdamW
learning rate                  0.001
weight decay                   0.01
batch size                     64
optimizer steps                1800
clip global norm               1.0
objective                      blank-cell cross entropy
```

No baseline budget may be reduced. Candidate trainable parameters must remain below the trained single-pass baseline. Training cost and inference cost are reported separately.

## 5. Validation-only selection

All six adaptive candidate configurations and the newly trained single-pass FP baseline are evaluated on the existing 512-example validation split across all five training seeds.

Complete-solve timing includes:

- recurrent state initialization;
- every executed recursive step;
- every intermediate decode;
- clue clamping;
- every online `sudoku_correct` call;
- final returned-answer validation.

Cost sample:

```text
64 distinct validation examples per seed
2 timing repeats per measured example
```

For each configuration record aggregate semantic validity, median/p95 complete-solve latency, mean/median realized recurrent steps, fraction stopping after step 1, and structural block applications.

Define validation point effects against `single_pass_fp`:

```text
q = semantic-validity difference
r = median complete-solve latency ratio
```

Selection rule frozen now:

1. reject any candidate whose trainable parameter count is not smaller than the primary baseline;
2. define Path-A point eligibility as `q >= +0.03` and `r <= 1.15`;
3. define Path-B point eligibility as `q >= -0.02` and `r <= 0.65`;
4. if any candidate is point-eligible for either path, choose the candidate with highest semantic validity, then lower latency ratio, then lower width, then lower max depth;
5. otherwise choose the candidate minimizing the following deterministic violation score, then higher semantic validity, then lower latency:

```text
A_violation = max(0, 0.03-q)/0.03 + max(0, r-1.15)/0.15
B_violation = max(0, -0.02-q)/0.02 + max(0, r-0.65)/0.35
score = min(A_violation, B_violation)
```

Development and confirmation results cannot affect this choice.

## 6. Fresh development reserve

The original M14 development split and Amendment-A seed `2026091404` have both informed development and cannot be reused for the next gate.

Fresh Amendment-B development seed:

```text
2026091405
```

Size:

```text
512 paired examples
```

Its fingerprints must be audited against the complete original train/validation/development manifest and Amendment-A development manifest. Any overlap aborts the attempt.

All five selected-candidate checkpoints are paired with the corresponding five newly trained single-pass baseline checkpoints.

Primary quality uses all 512 examples per seed. Complete-solve latency uses 128 paired examples per seed, two repeats. Hierarchical bootstrap resamples training seeds and paired examples with 5,000 replicates and seed `2026091499`.

The **unchanged M14 practical gate** is required:

### Path A — quality superiority at comparable cost

```text
quality difference >= +0.03
quality 95% CI lower bound > 0
latency ratio <= 1.15
latency-ratio 95% CI upper bound <= 1.20
```

### Path B — practical quality/cost trade-off

```text
quality difference >= -0.02
quality 95% CI lower bound >= -0.03
latency ratio <= 0.65
latency-ratio 95% CI upper bound <= 0.75
```

A miss preserves all results and leaves M14 incomplete. It does not open confirmation.

## 7. Independent confirmation

If and only if the fresh development gate passes:

1. freeze SHA-256 hashes of all five selected candidate checkpoints and all five corresponding primary-baseline checkpoints;
2. freeze the selected width, max-depth policy, and exact runtime implementation identity;
3. generate the still-untouched confirmation set with seed `2026091402`, 1024 examples;
4. audit its fingerprints against all prior development evidence, including seeds `2026091401`, `2026091404`, and `2026091405`;
5. evaluate the exact frozen candidate and baseline pair.

Confirmation uses all 1024 paired examples per seed for quality, 256 paired examples per seed for complete-solve timing, two timing repeats, and hierarchical bootstrap seed `2026091500`.

The same numerical M14 gate must pass. No threshold, backend, width, max depth, stopping rule, baseline, or seed can change after confirmation is opened.

If confirmation is opened and fails, `2026091402` becomes development evidence for any later change. The still-unopened reserve confirmation seed `2026091403` is then the only already-declared final reserve.

## 8. Contextual baselines and cost reporting

On validation/development/confirmation surfaces, report when supported:

- the newly trained conventional dynamic-INT8 single-pass model;
- exact MRV/backtracking Sudoku solver as a contextual task-specific symbolic reference.

The symbolic solver is not the primary learned baseline and has a different inductive bias.

Retain Attempt-1 floating/ternary/search results rather than rerunning and cherry-picking them. Amendment B is specifically an adaptive-depth development experiment.

Report per training run:

- trainable parameters;
- optimizer steps and examples sampled;
- train seconds;
- structural training block applications.

Report per evaluation configuration:

- semantic validity;
- paired complete-solve latency;
- CPU-package energy only when the M13 validated counter path is available;
- realized recurrent steps;
- semantic-validator calls;
- block applications;
- all completed seed/example rows.

Step counts and block applications are compute counters, not joules. Unavailable package energy remains unavailable.

## 9. Sequential selection and claim boundary

M14 has now used multiple development attempts. The final statistical interpretation must disclose this sequential model development. No p-value from the development stages is interpreted as if only one hypothesis was ever tried. Independent untouched confirmation is mandatory for any positive M14 claim.

No scaling-law claim is permitted from these operating points.
