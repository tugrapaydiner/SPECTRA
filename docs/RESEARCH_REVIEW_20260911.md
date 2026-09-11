# SPECTRA: harsh research review, September 11, 2026

**Scoring correction after clarification of the user's standard:** the original
55/100 below overweights engineering and auditability for a goal of exceptional
research impact that attracts aggressive recruitment. On that stricter subjective
scale, the current artifact is approximately **35/100**. This is neither a hiring
probability nor a compensation/level prediction. The original rubric remains
below as a record of the earlier assessment; more tests alone do not raise the
recalibrated score.

## Assessment: 55/100 against the requested frontier-impact standard

This is an explicit subjective artifact rubric, not an OpenAI hiring rubric or
a mapping from a repository to an employee level. The reviewed base is main
`a6d2e18a30467f10d01539d680e7b30776bd50fc`, PR20 at
`9939e1b0ac9977216541050a221a78f80a2979e3`, and the recovered source/evidence now
integrated with this audit. Additional software tests do not earn originality points.

| Dimension | Earned / available | Reason |
|---|---:|---|
| Engineering and correctness | 18 / 20 | Real native kernels, immutable search ownership, strict model contracts, frozen inference replay; broad portability and deployment remain bounded |
| Experimental integrity | 18 / 20 | Pinned raw evidence, independent checkers, negative results preserved, ancestry and symmetry audits; adaptive development and missing later evidence remain limits |
| Capability and practical value | 12 / 25 | A confirmed small Sudoku result and useful measured implementation work; no competitive application regime or robust general search win |
| Original scientific contribution | 7 / 20 | Bounded target-mismatch diagnosis and detailed arithmetic/overlap investigations; major ingredients and restart/selection ideas already have close prior work |
| Independent replication and impact | 0 / 15 | No reviewed independent external replication, adoption, or demonstrated influence on other teams |
| **Total** | **55 / 100** | **Strong research instrument; insufficient evidence of frontier-level impact** |

OpenAI's public [Research Engineer description](https://openai.com/careers/research-engineer-san-francisco/)
emphasizes building new capabilities/performance, strong ML engineering, and
experience with large distributed systems. It does not publish a project-to-L7
checklist. This repository demonstrates some engineering strengths, but neither
its test count nor completion of the roadmap certifies a role or level.

## What is actually strong

M14 confirmed a +15.14 percentage-point Sudoku advantage over its declared
single-pass FP baseline. The correct claim is comparable **median** latency:
mean and p95 costs were worse. The accepted model is the small FP dual-stream
core with task-specific validity termination. This result is not evidence for
the whole ternary/learned-router/MCTS design.

M15 explains a real failure: a probability of improving a structural proxy was
used as though it measured absolute state quality. Quantization was not the main
cause in those controls. M16 improves a faithful native checker path. M17 has a
positive Sudoku fixed-pool comparison and a failed maze gate. These are useful,
bounded results. Retaining the failed hypotheses is a strength.

PR20 adds an exact Sudoku4 input-symmetry audit and the explicit arithmetic
needed to reproduce historical CPU pool tensors. The symmetry audit finds
351/512 consumed development puzzles in training orbits. This is not proof of
neural memorization, and the unseen-orbit stratum retains a neural advantage.
It rules out an unqualified structural-generalization interpretation.

## Newly integrated positive result and its limits

Recovered source implements eight identity cycles followed by three transformed
views of four cycles each, stopping on exact validity. It is optional and leaves
historical default inference intact. Original complete runs, exact executable
sources, and the incomplete first-run log remain inspectable. Its missing 37
timing rows were not invented; a separately retained exact-source replay supplies
new timing observations and agrees on all 7,643 surviving original outcomes.

The following new statistical audit uses original refinement timings. Ratios are
candidate/comparator. Three rounds are collapsed per model/example before analysis.
Crossed percentile intervals resample the two model seeds and 128 shared puzzles;
they do not correct adaptive selection or multiple comparisons.

| Family / comparator | Candidate vs comparator solves | Gain, pp (95% interval) | Mean latency ratio | p95 ratio (95% interval) |
|---|---:|---:|---:|---:|
| Sudoku / identity 32 | 255 vs 241 of 256 | +5.47 [2.34, 10.16] | 0.784 | 0.389 [0.372, 1.033] |
| Maze / identity 32 | 53 vs 37 of 256 | +6.25 [1.56, 11.72] | 0.660 | 0.676 [0.661, 0.693] |
| Sudoku / identity 20 | 255 vs 240 of 256 | +5.86 [2.34, 10.16] | 0.930 | 0.616 [0.587, 1.041] |
| Maze / identity 20 | 53 vs 36 of 256 | +6.64 [1.56, 13.28] | 1.041 | 1.079 [1.061, 1.108] |

This narrows the earlier apparent tail-speed claim: the Sudoku p95 interval
crosses 1. Zero observed regressions against identity 32 is not a dominance
theorem. The guaranteed shared-prefix protection applies only through cycle 8.
The equal-20-cycle maze comparison exchanges more time for additional solves.
Cycles are not milliseconds, and these are not matched wall-clock-budget trials.

Classical solvers solve 256/256 in both families. The restart policy's mean time
is about **11.8x the Sudoku comparator** and **443.9x the maze comparator**.
These are historical local descriptive ratios, not estimates for other hardware.
The maze validator itself runs BFS during construction and stores predecessor
information for `shortest_solution()`. Optimality checking already performs the
central task-solving computation. This makes the present maze a particularly
poor flagship demonstration of a useful neural speed advantage.

Reproduce the analysis with:

```bash
python scripts/audit_research_frontier.py --out outputs/frontier.json \
  --verify-report results/reliability/research_frontier_audit.json
```

The implementation rejects missing models/examples/arms/rounds, duplicate records,
changed deterministic answers/work, and invalid timings. Expected inventories come
from the pinned source manifests. It reports all four controls per family. Passing
this audit verifies the analysis, not the independent-confirmation hypothesis.

## Novelty audit: a serious obstacle

These primary papers were inspected on September 11, 2026. This is a targeted
related-work check, not an exhaustive literature clearance:

| Prior work | What overlaps | Consequence for SPECTRA |
|---|---|---|
| [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871) | Compact shared recursive networks on reasoning tasks | Small recursive models alone are not the new contribution |
| [Bitnet.cpp](https://arxiv.org/abs/2502.11880) | Efficient ternary CPU inference | Packing and CPU execution need a measured improvement over appropriate implementations |
| [POMO](https://arxiv.org/abs/2010.16011) | Symmetries and augmentation-based inference in neural combinatorial optimization | Geometric restart diversity alone cannot carry a novelty claim |
| [TRM on ARC-AGI-1](https://arxiv.org/abs/2512.11847v2) | Separates augmentation/selection benefits from recursion | SPECTRA needs the same causal separation on its own task/model scope |
| [Speed is Confidence](https://arxiv.org/abs/2601.19085v2) | Halt-first TRM ensemble selection and training-time multiple latent states on Sudoku | A directly relevant comparator for proposals involving first-success selection or diverse trajectories |

These papers do not establish that every proposed SPECTRA variant has been done.
They do establish that a combination of recursion, augmentation and early stopping
needs a sharper claim and direct comparison. Their reported benchmark percentages
must not be compared numerically with SPECTRA's much smaller Sudoku4 task.

## Highest-value research direction

Study **which additional trajectory is useful after the trajectories already
tried have failed, at the remaining measured budget**. A high unconditional
success rate can be redundant with the incumbent trajectory. The relevant
quantity for a proposal action a is its conditional additional solve probability:

    q(a | h, b) = P(a produces a verified answer within b | observed failure history h).

Measure the corresponding real transform/reset/inference/checking cost c(a).
The heuristic q/c can prioritize proposals, but is not asserted optimal when
actions are dependent, costs vary, or budgets are finite. The existing deterministic
schedule remains a strong simple baseline. Conditioned allocation, failure
diversity and budget-aware values also have prior art; novelty must be established
for the eventual specific mechanism/result, not asserted for this formula.

The initial causal experiment should distinguish:

1. Extra total compute from better allocation: identical transition caps and
   separate complete-solve wall-clock curves.
2. Different presentations from redundant restarts: repeated identity, uniform
   views, best frozen static ordering and failure-conditioned ordering.
3. Useful candidate coverage from selection: valid-pool upper bound, first valid
   return, and learned selection on the **same** candidate inventory.
4. Learned transfer from dataset symmetry: all consumed ancestor orbits excluded,
   a held-out generator/difficulty family, and independently checked task semantics.

Use Sudoku4 for protocol diagnostics. Move the flagship to a task where checking
a supplied witness is substantially cheaper than solving, with a real user budget
and a strong classical comparator. Structured SAT/scheduling is a candidate, not
a promised win: first profile the tuned classical solver and reject trivial
instances. Do not spend a large training budget before this workload gate passes.

## Gates that could materially change the score

These are proposed criteria for a **new** protocol. They do not replace or relax
any historical gate, and numerical thresholds are proposed effect sizes rather
than official standards or hiring promises.

| Step | Required evidence before advancing | Fail / pivot condition |
|---|---|---|
| Workload feasibility | Relevant workload; checker cost separately measured; no ancestor/equivalence overlap; predeclared baseline tuning; development headroom and a power/sensitivity analysis | Classical preprocessing already solves the task cheaply, or the chosen gain is mathematically unattainable |
| Causal pilot | Best static/random/identity/classical controls; at least 5 model seeds; all outcomes retained; allocation benefit survives equal-work and full-cost accounting | Benefit vanishes after matching compute or conditioning on task difficulty |
| Independent confirmation | Lock code, task generator, model selection and primary operating point; >=5 pp gain with positive paired lower bound and mean/p95 ratios <=1 on two appropriate families; report uncertainty and every failure | Either family misses its frozen gate; no test-set recycling or post-hoc threshold adjustment |
| Specific mechanism contribution | Implement closest published method under a comparable budget; decisive ablations and explicit counterexamples/limits | Existing augmentation or halt-first selection explains the complete effect |
| Systems usefulness | Preserve the capability result through deployment; complete-solve latency/memory at representative scales on two hosts; precision ablation; physical energy only when measurable | Kernel wins disappear in the full solve, quality drops, or serving costs exceed the useful alternative |
| External reproducibility and impact | A separate researcher can reproduce the main tables from one documented command; a real downstream use or independent scientific follow-up | Only the original author/environment can reproduce the result |

For n paired model/example cases, a baseline with k valid solves leaves at most
n-k net additional solves. A required gain g needs `ceil(g*n)` net additional
solves. The new `quality_headroom` function computes this with exact rational
threshold arithmetic. This is an inventory ceiling, not a statistical power
calculation. If a baseline exceeds 95%, a +5 pp criterion is impossible on that
inventory. Inspect this on development before locking a new confirmation design;
if discovered afterwards, retain the original failed gate rather than lowering it.

## Outstanding evidence and compute scope

The existing final-confirmation protocol is preserved byte-for-byte. A prior
session reported an execution beyond the recovered development package, but its
runner and raw evidence are absent from the retrieved package and current branch.
This review does not certify those reported numbers. Its seeds are potentially
consumed: recover the original records or label any reconstruction a reproduction.

The user now permits available GPU or CPU compute. This workspace exposes 8 CPU
cores under its execution quota, 20 GiB memory, and no GPU device. Historical CPU
protocols and their source hashes are preserved. New hardware permission does not
retroactively alter the scope of their experiments.

Current merge validation covers the full fast suite, historical stored evidence,
actual frozen checkpoint/pool/restart inference, the exact symmetry report, and
the new paired analysis. The 16 legacy slow retraining tests are a separate scope.
CI archives the complete tested source and all validation outputs; local source
and result records remain attached to the exact experiment that produced them.
