# SPECTRA Research State

This is the live milestone register. Detailed accepted history is preserved in the milestone acceptance/protocol files and the archived state snapshots.

## Milestone index

| Milestone | Scope | State |
|---|---|---|
| M01 | trustworthy baseline | accepted / merged |
| M02 | native-kernel correctness/input contracts | accepted / merged |
| M03 | task/data/evaluation contracts | accepted / merged |
| M04 | reproducible training/checkpoint state | accepted / merged through PR #4 |
| M05 | checkpoint-backed evaluation | accepted / merged through PR #5 |
| M06 | controlled trained baseline | accepted / merged through PR #6; gate clarification PR #7 |
| M07 | grounded verifier training/evaluation | accepted / merged through PR #8 |
| M08 | correct inspectable MCTS reference | accepted / merged through PR #9 |
| M09 | trained search action mechanism | **INCOMPLETE**; negative result preserved / merged through PR #10 |
| M10 | faithful CPU deployment for one trained configuration | accepted / merged through PR #11 |
| M11 | real adaptive execution | accepted / merged through PR #12 |
| M12 | grounded router/halter RL training path | accepted / merged through PR #13; learned-control quality **NEGATIVE / COLLAPSED** |
| M13 | defensible measurement protocol | accepted / merged through PR #14 |
| M14 | primary controlled experiment | **COMPLETE on Attempt 5; independently confirmed quality-superiority gate** |

M09 remains scientifically incomplete. M12 established a real RL training path but not useful learned control. M13 establishes measurement integrity, not physical joules on hosts without a valid package counter. M14 establishes a narrow task-specific practical improvement for recursive FP + symbolic-validity termination on generated 4×4 Sudoku.

---

# Milestone 14 — primary controlled experiment

**Stage status: COMPLETE on `research/m14-attempt5-dual-stream-semantic-exit`.**

Authoritative protocols:

- initial M14 preregistration: [`M14_PROTOCOL.md`](M14_PROTOCOL.md)
- final adaptive Attempt-5 preregistration: [`M14_ATTEMPT5_PROTOCOL.md`](M14_ATTEMPT5_PROTOCOL.md)
- accepted evidence/claim boundary: [`M14_ACCEPTANCE_GATE.md`](M14_ACCEPTANCE_GATE.md)
- Git-retained evidence summary/provenance: [`../results/m14/`](../results/m14/)
- full accepted raw rows/checkpoints: Actions artifact `10035946017`, ZIP SHA256 `0306f64efb48c264aa7b387ff6b449def011fb64ad02aa74eb03ee3b4e58e314`

## Sequential development history

Four earlier M14 development attempts failed the unchanged gate and did not open confirmation. Attempt 5 was preregistered after those failures and before its own training/results. The final interpretation accounts for **five adaptive development attempts before confirmation**.

The final candidate is the original dim-64 dual-stream FP TRM (`n=1`, `T=1`, one shared block), trained with the original recursive supervision weights `[0.1, 0.2, 0.3, 0.4]`, plus a reference-free task-specific Sudoku semantic-validity early exit. The primary comparator remains the unchanged trained dim-96, two-block FP `System1Student`.

Five training seeds were retained: `1401, 2402, 3403, 4404, 5405`. No favorable-seed filtering or post-result threshold weakening occurred.

## Development pass

On five seeds × 512 paired development examples:

```text
quality difference        +0.145703125
95% CI                    [+0.121484375, +0.177343750]
median latency ratio       0.983055302
95% CI                     [0.978821304, 0.987161655]
path                        quality_superiority
decision                    PASS
```

Only after this pass were candidate/baseline checkpoints and the semantic-exit configuration frozen and confirmation generated.

## Independent confirmation pass

Confirmation seed `2026091402` generated 1,024 untouched examples. Fingerprint overlap with the 5,120 train/validation/development examples was **0**.

Across all five training seeds:

```text
candidate semantic validity       0.9845703125
baseline semantic validity        0.8332031250
quality difference               +0.1513671875
quality 95% CI                   [+0.1361328125, +0.1691406250]
paired median latency ratio        0.9842012360
latency-ratio 95% CI              [0.9808512693, 0.9873359073]
path                               quality_superiority
decision                           PASS
```

Reserve confirmation seed `2026091403` remains unused.

## Cost-distribution boundary

The M14 gate was preregistered on **median complete-solve latency**. The adaptive candidate has a more expensive hard tail:

```text
                              candidate       baseline
mean latency ms                1.466297       1.238605
median latency ms              1.215085       1.234590
p95 latency ms                 3.206238       1.276632
```

Candidate mean/baseline mean ratio is about **1.184×**; p95 ratio is about **2.511×**. Mean executed candidate supervision steps are **1.2484**; 16.8% of confirmation rows require more than one step.

Therefore the accepted M14 claim is **higher strict solve accuracy at comparable median latency**, not equal average cost, equal tail latency, universal speedup, or iso-energy.

## Task-specific boundary

The semantic stop checks predicted-board Sudoku validity and immutable givens; it never sees the reference solution and does not use the learned M12 halt policy. This is a task-specific hybrid inductive bias. M14 does not establish generic learned halting or cross-task transfer.

The exact symbolic MRV/backtracking reference remains perfect and faster on this task, as expected from its hard-coded Sudoku constraints. Dynamic INT8 is contextual only. Earlier ternary/search variants remain preserved in Attempt-1 evidence.

## Energy

The accepted runner exposed no valid CPU-package RAPL domain. Physical energy is unavailable/null with `failure_reason=no_package_domain`; no GPU or whole-system energy claim is made.

## Accepted execution

```text
head          12f911b1abfa9710c7aa14a3f4ba2e691f84ac1c
run           34171128736
job           101891478657
artifact id   10035946017
ZIP SHA256    0306f64efb48c264aa7b387ff6b449def011fb64ad02aa74eb03ee3b4e58e314
focused       42 passed
full fast     316 passed, 16 deselected, 1 pre-existing M10 warning
```

## M14 decision

**PASS — primary controlled experiment, narrowly scoped.** The predeclared quality-superiority criterion passed on development and then independently passed on untouched confirmation with complete-solve median latency within the declared tolerance and all adaptive development attempts accounted for.

**Stop here for M14. Do not begin M15 automatically.**
