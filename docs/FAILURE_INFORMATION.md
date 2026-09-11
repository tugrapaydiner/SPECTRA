# Failure information, candidate coverage and static continuation

This follow-up starts from merged PR20 (`b6e7ddc`) and the user's clarified
research-impact standard, approximately **35/100** for the current artifact.
Better auditing does not itself establish scientific originality or hiring value.

## Question and scope

Can observed wrong answers improve allocation beyond the input, an identical
failed prefix and the remaining action inventory? Before fitting a controller,
we measured the entire candidate inventory rather than guessing from censored
first-success logs.

The experiment uses only the 128 previously consumed M17 development inputs per
family and two frozen model seeds per family. No new confirmation inputs or core
training updates are involved. Reference targets are used by the existing loader
only to check pinned manifest fingerprints; inference and controller features
never receive them. Every recorded decoded answer is independently checked
against the original problem input.

## The decisive coverage result

Collection retains every cycle of identity continuation through 32, and all
seven alternative views through four cycles, even after an earlier valid answer.
That is **30,720 independently checked answers across 512 model/example cases**.

| Family | Identity 8 | Identity 32 | Existing restart order | Oracle over the complete measured inventory |
|---|---:|---:|---:|---:|
| Sudoku | 237/256 | 241/256 | 255/256 | 255/256 |
| Maze | 36/256 | 37/256 | 53/256 | 53/256 |

For case i and candidate action a, let S(i,a) indicate that a valid answer occurs
within its measured trajectory cap. Every selector returning a member of this
inventory has success at most `max_a S(i,a)`, even with free advance knowledge of
all outcomes. The existing order already attains this ceiling on every consumed
case. **There are zero additional solves available through selection alone on
this inventory.** This is an exact finite-data statement, not an impossibility
claim about other depths, generators, datasets or models.

After a failed eight-cycle prefix, transpose supplies all 17 additional maze
solves; none of the other six four-cycle views adds a solve. The remaining 203
maze cases have no valid candidate in the measured inventory. Sudoku has 18
rescuable prefix failures and one case unsolved by every measured trajectory.

## Why a failure bit alone is insufficient here

Fix the input, budget and any random seed. Suppose a policy sees only the actions
already attempted and their success/failure bits, and execution stops at success.
Along every nonterminal path, all those bits are failures. Recursively evaluating
the policy on that all-failure path therefore produces a fixed action order.
Actual successful execution is a prefix of that order.

This elementary observation assumes fixed attempt caps and excludes observed
answer content, variable elapsed times and other online state. It is not a new
algorithm-selection theorem. It explains why our experiment distinguishes the
contents of wrong answers from merely recording that an attempt failed.

## Controlled cross-fitting

All policies share an eight-cycle identity prefix and at most three four-cycle
alternative views. At the primary 20-cycle cap, the controls are the existing
order, an exhaustively selected training-fold static order, deterministic random
order, an input-only linear selector, a selector that also sees the failed prefix
answer, and a selector that additionally sees later wrong answers. Identity 20,
identity 32 and a classical solver are also timed.

Five folds keep equivalent inputs together across both model seeds. Sudoku uses
the full exact spatial/digit equivalence group: the 128 inputs contain **64
distinct groups**. Maze uses all D4 maps with either endpoint labeling and has
128 groups. This protects controller fitting across folds; it does not erase
the frozen cores' historical training overlap or previous adaptive development.

The linear scores use fixed 4x4 spatial bins of symbol frequencies, remaining
action masks, and, where allowed, the failed prefix and mean later-answer
features. Ridge penalty 10 is fixed. Training enumerates reachable subsets of up
to two failed alternative views. Each eligible model/example contributes total
weight one per action. No hyperparameter sweep or held-out policy selection is
performed. The 30 fitted selectors and every fold's membership are retained.

| Policy at 20 cycles | Sudoku solves | Maze solves |
|---|---:|---:|
| Existing order, previously selected on these development cases | 255/256 | 53/256 |
| Static order selected within each training fold | 253/256 | 53/256 |
| Deterministic random order | 253/256 | 43/256 |
| Input-only selector | 253/256 | 53/256 |
| Prefix-answer selector | 253/256 | 53/256 |
| Failure-aware selector | 253/256 | 53/256 |

The primary five-percentage-point feedback gain is absent. Equal solve counts
alone need not mean identical solved cases; the retained paired report includes
gains and regressions separately. These results do not rule out nonlinear or
better trained feedback policies elsewhere. They do remove a reason to build a
larger selector for this already exhausted candidate inventory.

The historical existing order was selected on the full consumed development
surface, whereas the new selectors use excluded folds. Its advantage is not an
unbiased comparison of learning algorithms. Every result here remains a pilot.

## Cost and statistical accounting

Live B=1 solves include selection, feature construction, transforms, checker
construction, embeddings, state resets, inference, decoding and original-input
verification. Loading frozen coefficients is outside the solve window; data
collection and controller fitting have separate recorded costs. Equal recursive
cycle caps are not equal milliseconds. Three timing rounds are collapsed per
model/example before analysis.

Bootstrap draws resample entire input-equivalence groups and model seeds,
preserving pairing and group multiplicities. Intervals are descriptive and
conditional on the fitted cross-fit policies. They do not refit the controllers,
undo adaptive data use, or adjust for multiple comparisons. With two model seeds,
they do not establish broad training-run robustness.

An added unequal-group test exposed a variable-name collision in the first
grouped bootstrap implementation. The initial benchmark completed all timing
rows but failed during aggregation. Its source, raw rows, log and exception are
retained in `benchmark_attempt1`. The corrected implementation is tested against
an independent draw oracle, and the complete benchmark was rerun; no failed
attempt is represented as accepted evidence.

## Follow-up: change the trajectory inventory

The coverage result motivated two explicitly adaptive static experiments:

1. Stop after identity 8 plus transpose 4 (12 cycles). This tests whether pruning
   unproductive maze views reduces complete cost at unchanged observed validity.
2. Extend the productive transpose trajectory. Identity 8 plus transpose 12 uses
   the original 20-cycle cap but can generate answers beyond the old four-cycle
   view inventory. Transpose-only 20 is included to distinguish extra depth from
   the shared-prefix protection.

The follow-up retains transpose through 32 cycles, independently checks all
16,384 new decoded answers, requires exact agreement with the original first four
cycles, and measures every declared arm. Its selection basis and timing order
are recorded before execution in `TRANSPOSE_CONTINUATION_PROTOCOL.json` and
`FAILURE_PRUNING_FOLLOWUP.json`. Neither changes the original failed feedback gate.

The measured outcomes against the original 20-cycle restart order are:

| Family / configuration | Solves | New / regressed cases | Mean time ratio | p95 time ratio |
|---|---:|---:|---:|---:|
| Maze, prefix 8 + transpose 4 | 53/256 | 0 / 0 | 0.6063 | 0.5875 |
| Maze, prefix 8 + transpose 12 | 58/256 | 5 / 0 | 0.9779 | 0.9977 |
| Maze, transpose alone 20 | 32/256 | 6 / 27 | 0.9546 | 0.8833 |
| Sudoku, prefix 8 + transpose 4 | 251/256 | 0 / 4 | 0.9771 | 1.0180 |
| Sudoku, prefix 8 + transpose 12 | 252/256 | 0 / 3 | 1.0095 | 0.9694 |
| Sudoku, transpose alone 20 | 236/256 | 0 / 19 | 1.2547 | 1.5699 |

Maze pruning reduces mean time by **39.37%** and p95 by **41.25%**, with identical
per-case validity on these consumed cases. Its grouped 95% mean-ratio interval is
[0.5966, 0.6180] and p95 interval [0.5690, 0.5993]. It still takes about **222 times
the classical comparator's mean time**, and solves 53 rather than 256 cases.

Longer transpose continuation adds five maze solves, all from model seed **1701**;
seed **2702** gains none. Its quality-gain interval includes zero [0, 6.25 pp],
and its p95 ratio interval [0.9564, 1.0898] crosses 1. This is a candidate-generation
lead with limited seed support, not a robust quality/cost superiority claim.
Discarding the identity prefix causes 27 maze regressions, showing why a raw
transpose-only accuracy total would hide important complementary behavior.

All three changes regress Sudoku. Keep the existing Sudoku schedule. For an
explicit maze development configuration, the existing checked runtime supports:

```python
from eval.checkable_tasks import MAZE11
from eval.symmetry_search import symmetry_solve

# Lower-cost measured maze configuration; no learned selector needed.
answer, work = symmetry_solve(
    frozen_maze_core, input_tokens, MAZE11,
    identity_cycles=8, cycles_per_view=4, view_limit=2,
    transition_budget=12,
)
# The quality-oriented development alternative uses cycles_per_view=12 and
# transition_budget=20. Neither configuration is a new default or a guarantee.
```

Reproduce and independently validate the retained experiments with:

```bash
python scripts/verify_failure_information.py --refit --replay \
  --out outputs/failure-information-verification.json
```

This verifies the complete evidence inventory and source archives, rechecks
stored answers, refits every excluded-fold controller, reconstructs all 47,104
trajectory answers, replays 8,192 complete solves, and regenerates the paired
reports. It consumes no new task data and makes no new timing claim. CI runs this
on both declared Python versions. Source archives and the failed initial
aggregation attempt remain inspectable under `results/failure_information/`.

## Related work and next decision

Per-instance portfolio selection is established by
[SATzilla](https://arxiv.org/abs/1111.2249). Runtime prediction with censored
observations and risk-sensitive selection are studied by
[Run2Survive](https://arxiv.org/abs/2007.02816). Selection using features observed
during execution is also established work, including
[dynamic selection for differential evolution](https://arxiv.org/abs/2403.02131).
[Speed is Confidence](https://arxiv.org/abs/2601.19085v2) is a close recursive-model
comparison for halt-first selection and diverse latent states. These references
were checked as relevant precedents, not replicated on their original benchmarks.

The next substantial research investment should create useful candidate coverage
on a workload where verification is cheaper than solving. Require positive
coverage headroom against strong static and classical controls before scaling an
allocator. A static depth-allocation improvement on these puzzles can justify a
bounded implementation improvement; it does not establish that broader thesis.
