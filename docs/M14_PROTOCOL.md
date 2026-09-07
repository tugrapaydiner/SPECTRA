# Milestone 14 Protocol — Primary Controlled Experiment

**Status: preregistered before any M14 training, validation, development, timing, or confirmation result.**

M14 is the first SPECTRA milestone whose completion requires a positive, independently confirmed practical improvement. A technically valid negative or inconclusive experiment remains evidence but **does not complete M14**.

## 1. Primary hypothesis and claim target

On a fresh generated 4×4 unique-Sudoku distribution, a **trained small recursive SPECTRA model**, chosen only from preregistered recursive configurations using validation data, will provide a practically better complete-solve quality/cost point than a credibly trained larger floating-point single-pass Transformer baseline.

The primary task metric is strict symbolic Sudoku semantic validity after a deterministic given-cell preservation projection applied identically to every neural solver. A board counts as successful only if the symbolic validator accepts the full completed grid.

The larger FP single-pass model is the sole primary claim comparator. A conventional dynamic-INT8 form of the same trained baseline is an additional deployment baseline if the installed PyTorch build supports a real quantized conversion. A task-specific exact backtracking solver is a contextual reference with a different inductive bias and is never used to satisfy the learned-model claim.

## 2. Practical improvement gate

The candidate must satisfy **one** of these two frozen paths against the larger FP single-pass baseline.

### A. Quality superiority at comparable cost

- semantic-validity point improvement `candidate - baseline >= +0.03`;
- paired hierarchical-bootstrap 95% CI lower bound on that improvement `> 0`;
- median complete-solve latency ratio `candidate / baseline <= 1.15`;
- bootstrap 95% upper bound on the latency ratio `<= 1.20`.

### B. Predeclared quality/cost trade-off

- semantic-validity point difference `candidate - baseline >= -0.02`;
- paired hierarchical-bootstrap 95% CI lower bound on the difference `>= -0.03`;
- median complete-solve latency ratio `candidate / baseline <= 0.65`;
- bootstrap 95% upper bound on the latency ratio `<= 0.75`.

Path B therefore permits at most a two-point observed quality loss only in exchange for at least a 35% observed complete-solve latency reduction, with an uncertainty bound that excludes losses worse than three points.

No threshold may be weakened after development or confirmation results are observed.

## 3. Cost matching and terminology

Latency is the primary physical cost because GitHub-hosted runners may not expose readable RAPL package counters. CPU-package energy is also recorded when `common.energy_counters` reports a valid complete measurement; unavailable/invalid energy remains `null` and is never converted to zero.

A point is called **latency matched** only when its validation median is within ±15% of the larger FP baseline. Points outside that tolerance are labelled `unmatched`; they may still satisfy the explicitly preregistered Pareto trade-off path B, but are never called iso-latency or iso-budget.

Training cost is reported separately from inference cost. Training wall time, optimizer steps, examples sampled, parameter count, and recursive block applications are not substituted for inference latency or physical energy.

## 4. Data hierarchy and contamination control

Task contract for every M14 neural comparison:

```text
task                 generated unique Sudoku
box                  2
height x width        4 x 4
vocabulary            0 blank + digits 1..4
clues                  uniform 6..10
augmentation           false
solution method        random_backtracking
```

### Development hierarchy

Primary development manifest seed: `2026091401`.

```text
train        4096
validation    512
development   512   (the manifest's test split; not the final confirmation set)
```

- Train rows are the only rows used for optimizer gradients.
- Validation rows select candidate architecture/inference budget and any preregistered reserve intervention.
- Development rows evaluate whether the frozen validation-selected candidate reaches the practical gate. Development results may diagnose failures and trigger only the reserve intervention below.

### Independent confirmation

Confirmation manifest seed: `2026091402`, `1024` test rows, generated/opened only after a candidate passes the development gate. It must be cross-audited against every train/validation/development fingerprint before use.

Reserve confirmation seed `2026091403` is named now but is **not used in the initial M14 run**. If confirmation-1 ever informs later model changes, confirmation-1 becomes development evidence; a later claim must use this genuinely independent reserve (or another preregistered independent set) and account for sequential selection. The initial M14 workflow does not automatically recycle a failed confirmation.

Earlier M01–M13 task rows cannot consume M14 confirmation because M14 uses new seeds and performs explicit fingerprint overlap checks against its own hierarchy. The M14 confirmation rows are never supplied to training, verifier fitting, configuration selection, budget selection, early stopping, or development diagnosis.

## 5. Training seeds and neural families

Main model seeds:

```text
1401, 2402, 3403, 4404, 5405
```

All main neural fits use the same train manifest, batch size, optimizer-step budget, and optimizer hyperparameters.

### Small recursive FP

```text
TRM
dim             48
blocks            1
heads             4
n / T             1 / 1
training N_sup    4
ternary          false
act8             false
```

### Small recursive W1.58A8

Same recursive architecture and schedule, with hard ternary projection path plus recurrent A8. Quantization strength is warmed linearly for the first quarter of training and must reload at exactly `1.0` for evaluation.

### Larger single-pass FP baseline

```text
System1Student
dim             96
blocks            2
heads             4
single pass       true
confidence head   frozen / excluded from training count
```

This baseline is deliberately larger and is never reduced after results are observed.

### Conventional INT8 baseline

After each larger FP baseline is trained and frozen, M14 attempts PyTorch dynamic INT8 quantization of supported `nn.Linear` modules. The result is included only if conversion produces at least one actual quantized linear module and the complete solve executes successfully. Unsupported attention/projection portions remain explicitly FP; a partial conversion is labelled as such. If the runtime cannot produce a real quantized model, INT8 is marked unsupported rather than replaced with fake quantization.

### Symbolic contextual reference

A deterministic exact 4×4 Sudoku backtracking solver is measured on the same evaluation rows. It is expected to exploit hard Sudoku constraints and therefore has a fundamentally different inductive bias and training cost (`0` learned optimizer steps). Its result is context, not evidence that a learned SPECTRA model passed the primary claim.

## 6. Training objective and budget

Main fits:

```text
optimizer          AdamW
learning rate      1e-3
weight decay       0.01
batch size         64
steps              1800 per model per seed
grad clip          1.0
quant warmup       450 steps (ternary only)
```

The learned objective focuses on unknown cells. Given cells are excluded from answer cross-entropy because they are already observed inputs.

- Single-pass: blank-cell cross entropy.
- Recursive: weighted blank-cell cross entropy over all four supervision outputs with monotonically increasing later-step weights `[0.1, 0.2, 0.3, 0.4] / sum`.

The same deterministic output projection is applied to every neural family before evaluation: positions containing givens are copied from the input; only predicted blank cells can change.

No test/reference target is used during inference.

## 7. Validation-only inference-budget selection

For each trained recursive checkpoint, validation evaluates ordinary recursion budgets:

```text
N_sup in {1, 2, 3, 4}
```

The checkpoint is not retrained for those inference budgets.

M14 also reports a trained-small-model search variant where the existing accepted machinery can be used faithfully:

- trained recursive reasoner;
- a train/validation-fitted full-state one-cycle-improvement verifier;
- serial M08 tree semantics with a deterministic fixed global action codebook;
- action 0 identity plus two fixed residual directions;
- rollout budgets `{1, 2, 4}`;
- maximum search depth `4`.

Because M09 did not establish a useful trained action mechanism, M14 does **not** pretend the fixed action codebook is learned. The search row is labelled `trained_reasoner_trained_verifier_fixed_actions`. It may be selected as the primary candidate only if its full complete-solve path passes the same validation selection rule and subsequent development/confirmation gates.

Validation selection is deterministic:

1. compute per-seed semantic validity and complete-solve latency for every recursive operating point;
2. form the validation aggregate across all five seeds/examples;
3. if any candidate has median latency `<= 1.15 ×` the larger FP baseline, choose the candidate with highest semantic validity among those eligible; ties are broken by lower median latency, then lexical configuration id;
4. otherwise choose the highest-semantic-validity recursive point and label it unmatched.

No development or confirmation row may choose the configuration.

## 8. Reserve development intervention

If and only if the validation-frozen dim-48 candidate fails the development practical gate, M14 may run one preregistered falsifiable intervention before touching confirmation:

```text
family              FP recursive only
dim                 64
blocks               1
heads                4
n / T                1 / 1
training N_sup       4
training steps       1800
same five seeds/data/objective/optimizer
inference N_sup      {1,2,3,4}
```

Hypothesis: the development failure is candidate capacity-limited rather than an intrinsic recursion/cost failure. The larger dim-96 two-block single-pass baseline remains exactly unchanged; no baseline weakening or favorable-seed filtering is permitted.

The dim-64 operating point is selected using **validation only**, then evaluated once on development. If it also misses, M14 is incomplete. No other post-hoc architecture, LR, training-step, data-size, loss, or threshold sweep is allowed in this milestone run.

Parameter sweep (dim 48→64), training-step budget (fixed 1800), and inference-budget sweep (`N_sup`) are therefore distinct dimensions.

## 9. Complete-solve inference measurement

All neural latency rows are batch-size one and wrap the entire loaded-model solve path under `torch.inference_mode()`:

- input/device preparation already resident in CPU memory;
- recursive cycles or single-pass blocks;
- any verifier call;
- action application;
- latent quantize/dequantize/conversion used by search;
- output head;
- argmax decode;
- deterministic given-cell preservation;
- symbolic semantic-validity check.

Model/checkpoint loading and compilation are recorded separately as cold/setup cost, not mixed into warm per-problem inference latency.

Latency is measured on paired examples with warmup and repeated solves. Environment provenance records CPU model, affinity, thread counts, frequency policy, backend/compiler identity, OS/kernel, and power context using the M13 measurement helpers.

Where valid package RAPL is available, energy scope is CPU package only — not GPU or whole-system energy.

## 10. Statistics and uncertainty

The five training seeds and paired evaluation examples are the repeated units.

Primary quality effect:

```text
per row = semantic_success(candidate) - semantic_success(baseline)
```

M14 uses a paired hierarchical bootstrap with seed resampling followed by paired example resampling within sampled seeds. Bootstrap seed is `2026091499`; at least `5000` replicates are used. The reported interval is percentile 95%.

Latency ratio uses the same seed/example pairing, bootstrapping the median complete-solve ratio. Per-example raw latency and success records are retained.

Reported surfaces include:

- point effect size and 95% CI;
- number of training seeds and examples;
- seed-specific metrics;
- per-example task outcomes and costs;
- all completed training and evaluation runs, including failures;
- exact checkpoint/config identities.

No scaling-law claim is permitted from these operating points.

## 11. Development and confirmation decision

1. Train all main families and fit permitted search verifiers using train/validation only.
2. Select exactly one primary candidate using validation only.
3. Evaluate that frozen candidate and the unchanged baselines on development.
4. If development passes path A or B, freeze/hash the candidate checkpoint(s), inference configuration, verifier/action state if any, and baseline checkpoint set.
5. Only then generate/open confirmation seed `2026091402` and run one final paired comparison.
6. M14 completes only if the same predeclared gate passes on independent confirmation.

A development pass followed by confirmation failure is a failed confirmation, not a near-pass. Confirmation is never used to lower thresholds or select a different seed/configuration.

## 12. Required artifacts

M14 must retain:

- this protocol committed before results;
- train/validation/development manifest and final confirmation manifest when opened;
- cross-manifest overlap audit;
- every training curve/checkpoint and training-cost record;
- validation operating-point table;
- raw development rows and, only after dev pass, raw confirmation rows;
- complete-solve latency/energy records and environment provenance;
- INT8 support/conversion audit;
- symbolic reference rows;
- bootstrap replicate summary and decision record;
- result table generated directly from raw rows;
- Pareto plot generated directly from raw rows with unmatched points labelled;
- figure/table provenance hashes;
- failure diagnosis and reserve-intervention result if triggered;
- updated `docs/RESEARCH_STATE.md`.

## 13. Acceptance

**PASS only if** a reproducible trained small-recursive candidate satisfies the frozen practical-improvement rule on development and then independently satisfies it again on untouched confirmation, with measured complete-solve latency (and energy when validly available), hierarchical uncertainty, credible trained baselines, all completed runs retained, and no post-hoc weakening.

A scientifically valid miss remains useful evidence but leaves M14 **INCOMPLETE**.
