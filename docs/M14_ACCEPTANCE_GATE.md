# Milestone 14 Acceptance Gate — Primary Controlled Experiment

**Decision: PASS, with a narrow claim boundary.**

M14 completes on Attempt 5. The accepted system is a trained dim-64 dual-stream FP recursive TRM with a reference-free, task-specific Sudoku semantic-validity early exit. It is compared against the unchanged trained dim-96, two-block FP single-pass `System1Student` baseline.

## Predeclared hierarchy and sequential selection

The original M14 protocol was committed before M14 results. Attempts 1–4 were retained as development failures and did not open confirmation. Attempt 5 was preregistered at commit `abbf24fc4166eab71be3d590e9754dd1b54612b7` after those misses and before Attempt-5 training/results. The same numerical gate was retained.

Adaptive development attempts before confirmation: **5**. No favorable-seed filtering, baseline weakening, threshold lowering, LR sweep, loss-weight sweep, or post-confirmation candidate selection occurred.

Attempt-5 candidate/baseline checkpoints and the selected `K=4` semantic-exit rule were hashed before confirmation. Confirmation seed `2026091402` was then generated exactly once. The overlap audit found **0** shared fingerprints between the 1,024 confirmation rows and the 5,120 train/validation/development fingerprints. Reserve confirmation seed `2026091403` remains unused.

## Training and models

Both candidate and primary baseline were trained from scratch on the same 4×4 unique-Sudoku train manifest with five seeds: `1401, 2402, 3403, 4404, 5405`. Each used 1,800 AdamW optimizer steps, batch size 64, LR `1e-3`, weight decay `0.01`, and gradient clipping at 1.0.

Candidate trainable parameters: **51,720**. Baseline trainable parameters: **225,893**.

The candidate retains the original dual-stream recurrence (`n=1`, `T=1`, one shared block) and original recursive supervision weights `[0.1, 0.2, 0.3, 0.4]`. The semantic stop is not the M12 learned halter. It tests predicted-board validity after each full supervision step and never receives a reference solution.

## Development gate

On five seeds × 512 paired development examples:

- strict semantic-validity effect, candidate minus baseline: **+0.145703125**
- hierarchical-bootstrap 95% CI: **[+0.121484375, +0.17734375]**
- paired complete-solve **median** latency ratio: **0.9830553023**
- latency-ratio 95% CI: **[0.9788213036, 0.9871616551]**

This passed the unchanged preregistered Path A gate before confirmation was opened.

## Independent confirmation

On five seeds × 1,024 paired untouched confirmation examples:

- candidate semantic validity: **0.9845703125**
- primary-baseline semantic validity: **0.8332031250**
- effect: **+0.1513671875** (+15.14 percentage points)
- hierarchical-bootstrap 95% CI: **[+0.1361328125, +0.1691406250]**
- paired complete-solve **median** latency ratio: **0.9842012360**
- latency-ratio 95% CI: **[0.9808512693, 0.9873359073]**
- confirmation decision: **Path A / quality superiority PASS**

The predeclared Path-A requirements were `quality >= +0.03`, quality CI lower bound `> 0`, median latency ratio `<= 1.15`, and latency-ratio CI upper bound `<= 1.20`. All four were satisfied without changing the gate.

## Complete-solve work and latency boundary

The candidate timing window includes embedding, every actually executed dual-stream recurrent update, two shared-block applications per executed supervision step, recurrent normalization/residual arithmetic, output head, argmax, immutable-given restoration, the independent semantic-validity check after each attempted step, and Python/control overhead. The final semantic decision is reused rather than charged twice. The baseline includes both transformer blocks, output head, argmax, immutable-given restoration, and one final semantic-validity check.

**Important distributional caveat:** the gate was preregistered on the paired **median** latency statistic. The adaptive candidate is not equal-cost under every latency summary. From the retained raw confirmation rows:

- candidate mean / median / p95: **1.466297 / 1.215085 / 3.206238 ms**
- baseline mean / median / p95: **1.238605 / 1.234590 / 1.276632 ms**
- mean latency ratio: **1.18383×**
- p95 latency ratio: **2.51147×**
- candidate mean supervision steps: **1.2484**
- step distribution over 5,120 candidate rows: **4,260 one-step; 569 two-step; 170 three-step; 121 four-step**

So M14 establishes **substantially higher strict solve accuracy at comparable median complete-solve latency**. It does **not** establish matched mean cost, matched p95/tail latency, universal speedup, or a 35% latency reduction. The hard tail is more expensive because unsolved examples continue recurrence.

## Context baselines

The accepted confirmation table also retains:

- dynamic INT8 single-pass baseline: semantic validity **0.83359375**, median **1.403155 ms**; this is contextual and does not replace the FP primary comparator;
- exact symbolic MRV/backtracking: semantic validity **1.0**, median **0.560654 ms**; this has a fundamentally different task-specific inductive bias and zero learned training steps.

Earlier trained ternary and trained-reasoner/search variants remain preserved in Attempt-1 evidence. They were not rerun merely to create favorable additional operating points.

## Energy

The accepted host exposed no valid CPU-package powercap domain. Candidate and baseline physical energy are therefore **unavailable/null** with `failure_reason=no_package_domain`. No CPU-package, GPU, whole-system, or iso-energy claim is made.

## Provenance

Authoritative accepted execution:

```text
branch        research/m14-attempt5-dual-stream-semantic-exit
head          12f911b1abfa9710c7aa14a3f4ba2e691f84ac1c
run           34171128736
job           101891478657
artifact      m14-attempt5-dual-stream-semantic-exit-evidence
artifact id   10035946017
ZIP SHA256    0306f64efb48c264aa7b387ff6b449def011fb64ad02aa74eb03ee3b4e58e314
size          6,670,698 bytes
```

Accepted workflow outcome:

```text
prereg        0
compile       0
focused       0   (42 focused tests passed)
experiment    0
render        0
evidence      0
fast          0   (316 passed, 16 deselected, 1 pre-existing M10 warning)
status        COMPLETE
```

The accepted full raw rows remain in Actions artifact `10035946017` and are bound by the recorded raw-row SHA-256. Git-retained derived evidence, the result table, deterministic SVG Pareto regeneration, hashes, environment, training audits, prior-attempt ledger, and the post-hoc latency-tail disclosure are retained under `results/m14/`.

## Claim boundary

**PASS — M14 primary controlled experiment.** The accepted evidence supports a trained recursive SPECTRA + symbolic-validity termination system on the generated unique 4×4 Sudoku distribution. The result survived an independently generated confirmation set after five adaptive development attempts.

This does **not** establish generic learned halting, cross-task transfer, a scaling law, energy superiority, matched average/tail cost, a generic advantage of ternary inference, or a generic search benefit. The exact symbolic solver remains faster and perfect on this task because it encodes Sudoku constraints directly.

**Stop here for M14. Do not begin M15 automatically.**
