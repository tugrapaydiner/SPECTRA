# Residual-patch experimental overhaul

**An implemented, trained and measured research fork; not a demonstrated breakthrough.**
All 550 legacy files and their modes from `e73dc05379b38ab2689e0e3054f6eee74e0290c0`
remain unchanged. Nothing here replaces the default SPECTRA model or runtime.

## What changed

This path discards the grid-token transformer, fixed random latent action vectors,
MCTS and quantized-runtime restrictions for this experiment. It operates directly
on a Boolean assignment and its exact violated constraints. A standalone C++ loop
maintains clause counts and applies simultaneous one- or two-variable repairs.
A bounded native generator constructs up to 24 residual-conditioned patches; a
small trained scorer chooses among them. This is a learned patch selector, **not**
an unrestricted neural patch generator or a new recurrent network architecture.

The learned scorer is a 16 -> 16 ReLU -> 1 network, **289 double-precision
parameters**, trained with PyTorch and executed directly in native C++. Feature
normalization is folded into the weights. There is no PyTorch call in the solve
loop. Recurrence is in the evolving solver state, not an invented neural design.
The native arithmetic is not ternary or an energy-saving result.

Atomic make/break changes are exact, including interacting flips. Adding two
single-variable gains is generally wrong. Tests exhaust small assignments and
patches, including duplicate literals, tautologies, empty clauses and unused
variables. The custom stochastic controls are probSAT-style and WalkSAT-style
implementations, **not the official or most optimized solvers**. Random and greedy
patch controls use the identical patch generator but skip unnecessary neural
feature/model work. Two external default CDCL solvers provide additional context.

## Executed local experiment

Inputs are unfiltered uniform/community 3-SAT at clause ratio 4.2. These are two
distributions of **one task**, not cross-domain generalization. Training uses
192 formulas at 64/128 variables; validation uses 32 at 128/256; development uses
96 at 128/256. Seed ranges and order-normalized formula hashes are disjoint.
No stored solution, planting witness or solver-filtered input selection is used.

The first attempt retained all 768 source snapshots, including already solved
ones; 591 were unsolved. It collected **8,211 patch labels and 16,422 actual
512-move counterfactual continuations**. Each target is the success fraction in
two fixed baseline continuations, not probability of general SAT solvability.
Only **208/591** failed snapshots have unequal empirical labels across patches.

Six models (three seeds each, full versus coarse feedback) were trained with
absolute-success binary cross-entropy. The initial validation was negative.
The separate [follow-up declaration](FOLLOWUP.md) changes only the training
objective to within-state ranking and/or the call interval from 32 to 128 moves.
Six further models were trained from the same complete retained training data.
Rank scores are not calibrated success probabilities. The primary follow-up
candidate `rank128` was fixed before its development execution.

### Local development result: the promotion gate FAILS

At the primary 10ms complete-witness deadline, averaged over 96 formulas, three
model seeds, two search seeds and two timing rounds:

| Variant | Within-budget witness success |
|---|---:|
| Validation-selected nonlearned control, by stratum | 37.9340% |
| Custom probSAT-style control | 40.2778% |
| Custom WalkSAT-style control | 38.7153% |
| Random patch, every 128 moves | 37.1528% |
| Greedy exact-residual patch, every 128 moves | 41.6667% |
| Absolute-success model, every 32 moves | 32.8125% |
| Absolute-success model, every 128 moves | 39.2361% |
| Within-state rank model, every 32 moves | 33.6806% |
| **Primary rank model, every 128 moves** | **38.3681%** |
| Coarse rank model, every 128 moves | 38.3681% |

The primary gain is **+0.4340 percentage points**, descriptive crossed-bootstrap
95% interval **[-3.6480, +4.6875] points**. The +10-point gate fails. The primary
model is not superior to the globally best controls. Reducing call frequency
improves the rank model by 4.6875 points, but does not establish learned-search
superiority. Full residual and coarse rank models have equal aggregate success,
not necessarily identical solved trials. All comparisons and per-seed effects
are retained; the best-looking development variant is not retrospectively primary.

The baseline was selected separately in four strata on only eight validation
formulas each. Its suboptimal development performance illustrates selection
noise, not a reason to suppress the stronger global controls. Confidence intervals
resample formulas within strata and model seeds after collapsing search/timing
repetitions. They are descriptive adaptive-development intervals, not an external
confirmation, multiplicity correction, or proof of broad robustness.

## Exact cost and protocol limitations

The complete solve window starts with initial assignment generation and includes
native array conversion, state construction, search, features, model calls,
returned-answer conversion, original-formula verification and native destruction.
Creating the immutable parsed CNF and its trial seed hash occur before that
window. Model-file loading is outside the window. Deadline overrun is retained;
a valid answer delivered late receives no within-deadline credit.

**Clarifications/deviations from the broad initial protocol:** the first solve
in each evaluation process also includes lazy native-library setup (including
compiler identity/cache/load work); those observations are retained, not removed
or silently relabeled. Later trials reuse the loaded library. Build/cache metadata
are recorded, but model-file loading is not separately timed and no complete
cold-start measurement is established. The measured coarse arm still sees global
residual information, and both pools depend on exact failed clauses; it is not
an input-only ablation. Only order-normalized duplicate identities are excluded;
no general variable-renaming/sign-switching isomorphism audit was performed.
These limitations preclude stronger claims even apart from the failed quality gate.

The two teacher continuations per candidate are noisy, the action pool is local
and restricted to two flips, and the 16 aggregate features omit the full clause
incidence structure. The baseline continuation used for labels is not the same
as repeated learned control online. This experiment does not resolve which of
these limitations dominates, or whether a larger generator would succeed.

External comparisons use CaDiCaL195 and Glucose4 through the existing pinned
PySAT dependency. They retain all 96 formulas and both rounds. Their 2/10ms numbers
are retrospective full-witness delivery cutoffs, not enforced native deadlines.
A 2,000-conflict request and a separate three-second process watchdog are recorded;
actual conflict overshoot is not censored. UNSAT is a solver report without an
independently checked proof; no-witness stochastic outcomes remain unknown.
Do not compare local-host latency directly with CI-host latency. External results
must be interpreted alongside the hybrid rerun on that same CI host.

## Verification and retention

Local Python 3.13.5 / CPU torch 2.10.0:
- Combined **891 tests passed**, zero failures or selected-test skips; 16 existing
  slow retraining tests deselected. This is 808 legacy and 83 new tests.
- **57,510 original-formula answer checks**, including failures, training source
  snapshots, every teacher continuation and all timed/fixed-work evaluation rows.
- **25,254 exact native non-timing reconstructions**: 17,190 training/source paths
  and 8,064 fixed-work model/search evaluations. All 8,211 feature rows replayed.
- All **12 model refits exact**, maximum coefficient difference zero within the
  predeclared 1e-10 refit allowance. Native/torch exported scores were checked too.
- The initial `_RNG` import failure occurred before collection; it was corrected
  to the existing public `SplitMix64` API, and its traceback is retained.

Native replay shares the solver/model implementation. Original-formula witness
checking and exhaustive small-formula flip checks provide independent semantics;
this is not independent external scientific replication. Passing integrity never
promotes a failed scientific gate. No legacy thresholds or checkpoints change.

The read-only `residual-patch-overhaul` workflow repeats the whole bounded pilot,
all model fits, native replays, and external solver context. It retains complete
source, all raw observations, failed attempts, model weights, manifests and logs
as a **90-day CI artifact**. The complete original local source/evidence/model ZIP
is also supplied in the conversation. Raw local timing archives are not silently
claimed to be permanently tracked Git blobs; keep that ZIP for long-term retention.
CI reruns on the same inputs are reproducibility checks, not new confirmation.

## Reproduce from the repository root

Install the CPU requirements and, for external context, `requirements-sat-research.txt`.
A compatible C++17 compiler and Linux shared-library loader are needed. Outputs
must be new paths; do not overwrite an existing experiment.

```bash
python -m pytest experiments/residual_patch -q
python -m experiments.residual_patch.experiment freeze --out outputs/repair-v1
python -m experiments.residual_patch.experiment collect --out outputs/repair-v1
python -m experiments.residual_patch.experiment train --out outputs/repair-v1
python -m experiments.residual_patch.experiment validation --out outputs/repair-v1
python -m experiments.residual_patch.followup train --original outputs/repair-v1 --out outputs/repair-v2
python -m experiments.residual_patch.followup validation --original outputs/repair-v1 --out outputs/repair-v2
python -m experiments.residual_patch.followup development --original outputs/repair-v1 --out outputs/repair-v2
python -m experiments.residual_patch.audit --original outputs/repair-v1 --followup outputs/repair-v2 --replay --refit --out outputs/repair-audit.json
python -m experiments.residual_patch.external run --followup outputs/repair-v2 --out outputs/repair-external
python -m experiments.residual_patch.external verify --followup outputs/repair-v2 --out outputs/repair-external
```

## Research disposition

Keep this branch experimental; do not replace the working legacy runtime merely
because the new path compiles. The next meaningful architectural experiment would
need a richer residual representation or a broader learned repair generator, plus
strong classical controls and full-cost accounting. Larger models are hypotheses,
not promised solutions. Removing legacy constraints alone did not make this
particular learned method competitive. No research-impact promotion is supported
by this bounded result.

Related primary sources (not novelty certification): [NLocalSAT](https://arxiv.org/abs/2001.09398),
[Large Neighborhood Search meets Iterative Neural Constraint Heuristics](https://arxiv.org/abs/2603.20801),
and the [PySAT solver API](https://pysathq.github.io/docs/html/api/solvers.html).
Neural constraint repair, stochastic local-search guidance and LNS are established
research directions; any future originality claim must be substantially narrower.

## Retained first CI failure and correction

Run `34646255794` completed all training, native replays, model refits and all
384 external solver observations, then failed external summary generation. The
worker sent a tuple witness through multiprocessing while the strict summary
expected the JSON list representation. Archived JSON witnesses were valid lists;
independent replay checked every original row successfully. The original failed
run and all its records remain retained, not relabeled as a passing run.

The correction converts the witness to a list before sending it. Three worker
round-trip regression cases cover SAT, UNSAT-report and budget-unknown outputs.
No solver algorithm, measured timer, threshold or model is changed. The corrected
head reruns all measurements and requires both live and archived verification.
The Python module commands above also correct an earlier documentation typo.
