# Milestone 14 Development Amendment A — One-Block Recurrent Intervention

**Status: preregistered before Amendment-A training, validation, development-reserve, or confirmation results.**

This amendment does **not** change the primary M14 claim, practical-effect thresholds, uncertainty rule, primary baseline, task metric, or final confirmation hierarchy in [`M14_PROTOCOL.md`](M14_PROTOCOL.md). It records the next falsifiable development intervention after the first valid M14 experiment missed the frozen gate.

## 1. Prior attempt is preserved as a miss

Authoritative prior execution:

```text
head            66acbc62f48b3e5b1e81080897b89562edfd8bd8
Actions run     34159474745
job             101858086130
artifact        m14-primary-controlled-experiment-evidence
artifact id     10032762057
artifact SHA256 04a5b810d697227c4ccf88a365cf529647734ea0d95deec363f84fcaae885522
```

The experiment/evidence hierarchy was valid; the positive acceptance wrapper failed because the practical-improvement claim did not pass.

The preregistered `dim=64, n=1` reserve intervention produced, on the 512-example paired development split across all five training seeds:

```text
quality difference vs single-pass FP   -0.005078125
quality 95% bootstrap CI                [-0.04453125, +0.033203125]
latency ratio candidate / baseline      1.0943352911
latency-ratio 95% bootstrap CI          [1.0887900899, 1.0998286826]
```

It therefore missed both unchanged M14 paths. The first confirmation set (`2026091402`) and reserve confirmation set (`2026091403`) were not generated or evaluated.

## 2. Falsifiable diagnosis

The selected recursive candidate is no longer primarily limited by optimization or width: the `dim=64` training runs reached `97.7–98.8%` validation semantic validity at four training supervision steps. At the validation-selected one-step inference point, however, the `n=1` TRM still executes two shared-block applications per supervision step:

1. one inner `z` update;
2. one `y` update.

The development result was quality-competitive but slower than the two-block `dim=96` single-pass baseline. Amendment A therefore targets **recurrent step structure**, not the baseline, LR, data difficulty, or claim threshold.

## 3. Intervention

Train two floating-point TRM candidates that set:

```text
n = 0
T = 1
training N_sup = 4
inference N_sup = 1 only
n_layers = 1
heads = 4
alpha_y = 0.1
alpha_z = 0.1  # retained parameter; z has no inner update when n=0
ternary = false
act8 = false
```

Candidate widths:

```text
n0_dim64
n0_dim96
```

`n=0` is an existing faithful TRM semantic: `recursive_cycle` skips the inner `z` loop and still executes the recurrent `y` update. The four training supervision steps therefore remain recurrent through `y`, while one-step inference executes exactly one shared transformer block. No new inference operator or approximation is introduced.

The intervention is successful only if measured complete-solve behavior satisfies the **existing** M14 practical gate. Halving a structural work counter alone is not success.

## 4. Training budget and seeds

The primary baseline is retrained in the Amendment-A run rather than reused from a favorable prior seed.

All three families use the same train split, optimizer, training steps, batch size, and seeds:

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

The single-pass baseline remains exactly the M14 baseline:

```text
System1Student
FP32
dim=96
2 transformer blocks
heads=4
confidence head frozen
```

No baseline hyperparameter, data allocation, or training budget may be reduced after seeing Amendment-A results.

## 5. Validation-only candidate selection

Both `n0_dim64` and `n0_dim96` are evaluated across all five seeds on the existing validation split. Complete-solve latency includes model execution, decode, clue clamping, and symbolic semantic validation.

Validation cost sample:

```text
64 distinct examples per seed
2 timing repeats per measured example
```

For each candidate define:

```text
q = aggregate validation semantic-validity difference vs single-pass FP
r = aggregate validation median-latency ratio candidate / single-pass FP
```

Selection rule, frozen now:

1. require candidate trainable parameters `<` single-pass baseline trainable parameters;
2. form a preferred pool with `q >= -0.02` and `r <= 0.75`;
3. if the preferred pool is non-empty, choose highest semantic validity, then lower latency, then lexical config id;
4. otherwise choose the candidate minimizing the lexicographic tuple
   `(max(0,-0.02-q), max(0,r-0.65), -semantic_validity, latency, config_id)`.

Development and confirmation data may not affect this choice.

## 6. New development reserve

The previous M14 development split has already informed model development and is not reused as the Amendment-A claim gate.

Fresh development-reserve seed:

```text
2026091404
```

Size:

```text
512 paired examples
```

Before evaluation, its fingerprints must be audited against the entire original train/validation/development manifest. Any overlap aborts Amendment A.

All five seed-specific selected-candidate models are compared with the five corresponding newly trained single-pass baselines. Primary quality uses all 512 paired examples per seed. Complete-solve latency uses 128 paired examples per seed, 2 repeats. Hierarchical bootstrap resamples training seeds and paired examples with 5,000 replicates and seed `2026091499`.

The development-reserve gate is **unchanged** from M14:

### Path A — quality superiority at comparable cost

```text
quality difference >= +0.03
quality CI lower bound > 0
latency ratio <= 1.15
latency-ratio CI upper bound <= 1.20
```

### Path B — practical quality/cost trade-off

```text
quality difference >= -0.02
quality CI lower bound >= -0.03
latency ratio <= 0.65
latency-ratio CI upper bound <= 0.75
```

Failure on this fresh development reserve leaves M14 **INCOMPLETE** and does not open confirmation.

## 7. Confirmation remains untouched until a pass

If and only if the new development reserve passes, freeze SHA-256 hashes for:

- all five selected-candidate checkpoints;
- all five newly trained single-pass baseline checkpoints;
- selected config id and runtime settings.

Then generate the original untouched confirmation set:

```text
seed      2026091402
examples  1024
```

Audit it against original train/validation/development plus Amendment-A development-reserve fingerprints. Any overlap aborts confirmation.

Confirmation uses the same paired five-seed comparison, the same practical gate, 256 paired latency examples per seed, 2 repeats, and bootstrap seed `2026091500` (`2026091499 + 1`). No configuration change is allowed after confirmation is opened.

If confirmation fails and later informs another intervention, `2026091402` becomes development evidence. Any subsequent claim must use the still-unopened reserve confirmation seed `2026091403` or another preregistered independent set.

## 8. Cost and contextual reporting

Training cost and inference cost remain separate. Report per run:

- train seconds;
- optimizer steps/examples sampled;
- structural training block applications;
- trainable parameters.

Report complete-solve inference latency per example. CPU-package energy is reported only when M13's validated reader returns a valid measurement; unavailable energy remains unavailable.

On the development-reserve/confirmation surface, also report the newly trained conventional dynamic-INT8 single-pass baseline when supported and the exact symbolic Sudoku solver as contextual references. The symbolic solver has a different task-specific inductive bias and cannot serve as the learned primary baseline.

The already-retained first M14 attempt remains the evidence for floating recursive, ternary recursive, and available search variants. Amendment A does not erase or rerun-favorably-filter those negative/slow search results.

## 9. Sequential-selection interpretation

M14 now has one completed primary development attempt plus one preregistered reserve intervention before this amendment. Amendment A adds two architecture candidates selected on validation and one new development-reserve gate. This sequential development must be disclosed in the final interpretation; the independent confirmation requirement is what protects the final claim from the accumulated development selection.

No scaling-law claim is permitted.
