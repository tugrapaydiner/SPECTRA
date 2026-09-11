# CPU progress: verified symmetry restarts and reproducible evidence

> Historical package report, preserved with its original validation and timing
> scope. The recovered code/evidence is now integrated through PR20; the current
> assessment and new paired analysis are in [the September 11 review](RESEARCH_REVIEW_20260911.md).
> The local-only publication statements below describe the original delivery.

## Status and scope

This source integrates the upstream PR20 snapshot at
`a55bb75411d99afcf19d80b73600d0de5dd7e6d9`
(tree `39d6ceabe39129c1017cca5a6e441544dd93a6c7`) with a new, optional
CPU inference policy, counterfactual scheduling, strict evidence verification,
and three legacy CI dependency corrections. No checkpoint, historical expected
pool hash, scientific threshold, or historical negative result was changed.

**The new quality result is an adaptively selected development result, not
independent confirmation, a novelty claim, a matched-wall-clock-budget result,
or a victory over the exact classical solvers.** Both tasks use existing frozen
floating-point models. No GPU, training update, API model, reference solution in
inference, or newly generated confirmation is involved.

The delivery is a local source-and-evidence package and an exact patch. Its
integration validation does not constitute remote CI or a GitHub merge. The
connected GitHub actions available during this work exposed reads but no writes.

## Actual complete-solve result

The original complete refinement is retained in
[`results/cpu_progress/attempt2`](../results/cpu_progress/attempt2/).
Each task has 128 distinct already-consumed M17 development puzzles and two
frozen model seeds, giving 256 model-example comparisons. Three timing rounds
are repetitions, not additional independent examples. For each model-example-arm,
the timing median across the three rounds is computed first; the table reports
the mean and p95 of those per-pair medians. All arms use the same puzzle inventory.

| Task | Method | Valid / 256 | Mean ms | p95 ms |
|---|---|---:|---:|---:|
| Sudoku shift | Identity, at most 4 cycles | 226 | 1.0384 | 1.6764 |
| Sudoku shift | Identity, at most 20 cycles | 240 | 1.4348 | 6.9367 |
| Sudoku shift | Identity, at most 32 cycles | 241 | 1.7020 | 10.9778 |
| Sudoku shift | **Identity 8 + three views × 4** | **255** | **1.3346** | **4.2719** |
| Sudoku shift | Exact symbolic comparator | 256 | 0.1132 | 0.1392 |
| Maze | Identity, at most 4 cycles | 31 | 3.7236 | 4.3315 |
| Maze | Identity, at most 20 cycles | 36 | 16.2637 | 19.4444 |
| Maze | Identity, at most 32 cycles | 37 | 25.6467 | 31.0112 |
| Maze | **Identity 8 + three views × 4** | **53** | **16.9277** | **20.9747** |
| Maze | Exact native BFS | 256 | 0.0381 | 0.0447 |

Relative to 32-cycle continuation, the policy adds 14 Sudoku solves (+5.47
percentage points) and 16 maze solves (+6.25 percentage points), with zero
observed lost solves in either task. Mean latency falls by about 21.6% and 34.0%,
and p95 falls by about 61.1% and 32.4%, respectively. This is a measured local
quality/latency improvement over that neural control, not an equal-cost theorem.

The equal-20-cycle-cap comparison is different: maze improves from 36 to 53
solves but costs about 4.1% more mean time and 7.9% more p95 time. The new policy
also costs more than the 4-cycle baseline. There is no universal speedup. The
classical comparators remain perfect and much faster on both tasks.

Host: AMD EPYC 9V74, process affinity CPU 0, one PyTorch/BLAS thread,
Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu, declared AVX2 profile. Raw rows
retain the complete measured windows. Transform construction, input transforms,
embedding, state resets, decode, native validation, answer inversion, and original
input verification are online. Native extension first load is separately recorded
and excluded from warm timing. Physical energy was not measured and is null.

## What changed in inference

`eval/symmetry_search.py` implements a bounded B=1 restart policy. It first runs
up to eight ordinary recursive cycles. If no answer is valid, it restarts on a
transposed grid, a 180-degree-rotated grid, then an anti-transposed grid, allowing
up to four cycles on each. A valid answer stops execution immediately. At most
20 recursive cycles are executed, with only the current trajectory stored.

For the maze rotation and anti-transpose, START and GOAL are also exchanged.
This preserves the generator's diagonal endpoint convention while remaining a
valid task transformation. The coordinate and symbol permutations are inverted
before returning an answer. Any answer accepted in a transformed coordinate
system is checked again against the original input. No reference answer or
learned quality score is used to accept it.

The models are not assumed equivariant: equivalent inputs can produce distinct
prediction errors. The initial repeated-identity control adds no solves over
four cycles, whereas changed views add useful candidates. This separates restart
alone from changing the coordinate view on these deterministic frozen cores.
It is a diagnostic explanation, not a claim that all models behave this way.

### Exact validity and prefix properties

For an allowed task transformation g and its inverse, correctness is preserved:
`valid(x, a) = valid(g(x), g(a))`. Tests check the finite spatial maps and answer
round trips, including maze endpoint swaps; runtime still validates the inverse
answer rather than relying solely on that proof.

The identity prefix is executed first and valid answers are retained. Therefore
any answer found by the same deterministic identity execution within the shared
prefix is retained by the restart policy. This does **not** guarantee dominance
over an identity execution that continues beyond the prefix. A counterexample
test retains that limitation. Zero regressions against 32 cycles is an empirical
result on these 512 model-example cases, not a universal guarantee.

## Adaptive selection and executable counterfactuals

The first pilot used 8 views × 4 cycles, plus identity and repeated-identity
controls. It reached 255/256 Sudoku and 49/256 maze solves. It lost four maze
solves relative to the longer identity trajectory. The losses motivated the
longer initial identity prefix; all examined prefix lengths 4 through 32 are
retained, not just the selected operating point.

Let t0 be the first valid identity time, and tj the first valid time within
transformed view j. For a prefix of length p, the stopping time is t0 when
`t0 <= p`; otherwise the first successful later view costs
`p + 4*(j-1) + tj`, subject to its view/cycle limits. The calculation uses the
recorded deterministic trajectories and verifies shared-prefix consistency.
It predicts validity and transition counts, not elapsed time.

The chosen schedule was locked after examining this consumed-development
curve, before its actual refinement execution. All 512 predicted validity and
transition outcomes matched execution. `eval/symmetry_schedule.py` implements
and tests the calculation. Selection records and both local protocol files are
retained. No claim of external preregistration or fresh held-out selection is made.

The result should next face a separately locked experiment with orbit-disjoint
Sudoku training ancestry, a new maze development/confirmation design, more model
seeds, and stronger simple/classical controls. Do not relabel the present cases
as untouched confirmation, and do not reopen the old unused maze confirmation
as part of this adaptive development.

## Artifact failure: preserved, not papered over

The first pilot's original summary declared 7,680 rows, but later verification
found only 7,643 in its raw log. Its expected and observed hashes differ. The
cause of the 37 missing timing rows is not established. The original incomplete
log, original summary, exact executable source, and failure record remain under
`results/cpu_progress/attempt1/`.

All first-round model-example-arm outcomes were present, so the original
counterfactual selection is inspectable. An exact-source replay was then executed
once with the same frozen checkpoints and inputs. It produced a separately
retained complete 7,680-row run and matched answers and all work fields for all
7,643 available original records. These are new timings, **not recovered or
fabricated original timings**. The complete replay is in `attempt1_replay/`.

The original refinement log is intact at 7,680 rows. A subsequent writer change
now compares all in-memory rows with the closed on-disk serialization and writes
and fsyncs a compressed mirror before declaring completion. Truncation tests
require rejection, not a silently repaired summary. Post-publication hashes
remain necessary because an external mutation can occur after any sealing check.

| Evidence | SHA-256 of uncompressed rows |
|---|---|
| Original first pilot, incomplete | `40b5ecaa8a87ac3c45488e7142cabbd597cb6ef204e32a121445867432fdea7a` |
| Complete exact-source first-pilot replay | `18835b6ae54ea48bfb2dadbc3ab771c99d5d8132dec3a3504a177462aa73df74` |
| Original complete refinement | `123bbc137d7ea92a429853ca963c464a25f37652b32727f5dc4187207c975e68` |

The complete first-pilot replay uses executable-source SHA-256
`a180d429fa0c7147a31809b49d0f355dd6f90bd1031b610582d637807b7b8e5e`.
The original refinement uses
`c10c8226a6aa1e2e67e2d2de9df91415f2c345adf1a79e3b1901b9f9075ac23b`.
Each source archive contains the exact executable inventory for that run. Later
integration edits are not silently assigned the older measured timing results.

## Numerical replay and concurrent upstream integration

The read-only paired trace comparator in `eval/replay_trace.py` verifies source,
artifact and tensor inventories before localizing chronological divergence. It
distinguishes bitwise identity from numerical equality and rejects missing,
duplicate, malformed or nonfinite trace fields. JSON key order is not treated
as execution order.

On the inspected Intel/AMD traces, embeddings agree. The first observed Sudoku
module divergence is the second feed-forward linear, and the first maze module
divergence is attention. Both the original and independent pool constructors
agree with each other on each host. A successful replay on AMD alone does not
establish bitwise portability to Intel.

While this work ran, upstream PR20 independently added explicit ordered FMA
replay arithmetic. This package preserves that work rather than replacing it
with the restart policy. See [ORDERED_REPLAY.md](ORDERED_REPLAY.md). Its scope is
frozen B=32 historical replay, not the B=1 inference benchmark here.

Both inspected dedicated CPU CI artifacts for upstream commit `3ecac995` passed
561 fast tests and the exact fixed-pool replay, on AMD EPYC 7763 and AMD EPYC 9V74.
Their immutable source reconstructed the declared upstream tree. These inspected
artifacts do **not** establish successful ordered-profile execution on Intel.
Raw old-profile Intel failures remain failures; no expected hash was changed.

The final integrated upstream snapshot is `a55bb754`, which additionally fixes
the remaining maze core linear reductions after an intermediate Intel failure.
Its five changed files were read at the exact commit and their Git blob hashes
verified before reconstructing the complete upstream tree. See
[the upstream follow-through](ORDERED_REPLAY_FOLLOWUP.md). The final local
integration reruns all seven validation stages on this later source, not only
the earlier passing `3ecac995` snapshot.

The legacy baseline CI artifact installed unconstrained PyTorch 2.14.0+cpu and
failed 13 ordered-replay tests at the explicit 2.10.0+cpu version guard. The patch
pins the three legacy full-fast-suite workflows to the same CPU 2.10 environment
and `requirements-cpu-research.txt`, without removing tests or weakening the
version contract. Upstream also independently corrected the default requirements at `a55bb754`;
that correction is preserved. Pinning the initial workflow install avoids first
installing and then downgrading an incompatible Torch release. A fresh remote
run of the delivered workflow changes is still required.

## Verification commands and claim boundaries

Use a clean Python process and the pinned CPU environment:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r requirements-cpu-research.txt
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
python -m pytest -m 'not slow' -ra
python scripts/verify_retained_results.py --out outputs/original-integrity.json
python scripts/verify_checkpoint_replay.py --out outputs/original-checkpoint-replay
python scripts/verify_fixed_pool_replay.py --cpu-profile historical-ordered --out outputs/original-fixed-pools
python scripts/audit_sudoku_symmetry.py --out outputs/sudoku-orbit-audit --verify-report results/reliability/sudoku4_symmetry_audit.json
python scripts/verify_cpu_progress.py --out outputs/cpu-progress-integrity.json
python scripts/verify_symmetry_checkpoint_replay.py --out outputs/refinement-checkpoint-replay
```

Output locations must not already exist. The fixed-pool replay faithfully
reconstructs already-published Sudoku confirmation; it does not generate it or
use it to select this policy. Unopened maze confirmation remains unopened.
The new policy replay uses only consumed development.

### Completed delivery validation

The exact delivered executable source has SHA-256
`73c0cdbe0606a55f2dfd2190026cdfb8560049bdd164b3f5bd1d54d2188d4dc0`.
All seven local validation commands completed successfully and executable source
identity was unchanged across the run. The final fast suite passed **642 tests**,
with **16 slow tests deselected** and two retained warnings. The latest base
contributes 563 tests; this patch adds 79. These are software tests, not 79 new
scientific experiments.

Historical verification checked 1,728 manifest rows and 11,520 stored answers,
then actually replayed 3,840 complete-solve model-example-arm comparisons.
The explicit-order replay reconstructed 16,384 candidate states and their
continuations: **six original pool hashes and all 49,152 evaluator scores match
exactly**, with maximum score error zero. The original 5,120-example Sudoku
symmetry audit also reproduced its pinned report. The new integrated inference
replay checked **all 2,560 distinct refinement model-example-arm comparisons**,
with exact answers, validity, and every recorded work field.

Logs, JUnit, source identities, summaries, and compressed comparison rows are in
[`results/cpu_progress/validation`](../results/cpu_progress/validation/).
The local validation is on AMD EPYC 9V74 / Python 3.13.5 / PyTorch 2.10.0+cpu.
The inspected newer upstream Python 3.11 artifact also passes its 563-test and
historical-replay scope on AMD EPYC 9V74, but does not include this feature patch.
No new measured timings are attributed to the integrated reproduction profile.

The stored-result verifier checks 15,360 complete-run answer rows, the preserved
7,643 original partial outcomes, exact source inventories, paired summaries and
512 counterfactual outcomes. This is separate from actually re-running all
2,560 distinct refinement model-example-arm comparisons through the integrated
code. Timing repetitions and repeated verification are never counted as new
independent scientific evidence. Final delivery verification files state what
actually executed; the 16 long legacy training tests are outside the fast scope.

The local environment's normal Python startup preloads NumPy through
sitecustomize. The old replay correctly rejected that process before numerical
profile selection. The successful clean-process validation uses `python -S`
with the same installed site-packages and no automatic host preload. No project
safety check or arithmetic is modified to avoid the guard. The rejected
invocation and its log are retained in delivery verification.

**Result boundary:** a useful two-task CPU development improvement over specified
neural controls, with exact answer checks and better evidence handling. Not a
newly confirmed generalization result, a novel-search claim, energy superiority,
a proven low-bit capability advantage, classical solver superiority, a completed
GitHub merge, or an L7/100-point achievement.
