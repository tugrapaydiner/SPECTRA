# Deductive CNF solving: capability and cost, 2026-10-02

The optional candidate solves **124/160 formula/seed cases**, compared with
**97/160** for existing indexed search, on 80 fresh synthetic formulas. Mean
complete CPU wall time falls from **12.520 to 8.588 ms** (31.4%); p95 rises from
24.908 to 25.483 ms (2.3%). The additional successes are 26 equivalence-ring cases
and one forced mixed case. This supports an optional structured-CNF solver path,
not a change to the default or a claim of improved learned/general reasoning.

## Identity and boundaries

- Draft [PR #50](https://github.com/tugrapaydiner/SPECTRA/pull/50), based on PR #46.
- Candidate and generator published before evaluation generation at commit
  `9fd2265bac0c2aa18fd711b47dc75dbc75a6248f`, tree
  `b01b2791f9605d7c906e7ab5fce2559c87fe19e6`; see [FROZEN.json](FROZEN.json).
  No candidate/generator edits followed evaluation. Later changes save evidence,
  check retained observations and document results.
- [Protocol](PROTOCOL.md) and [pre-evaluation amendment](AMENDMENT-01.md): five
  families, 128/512 variables, eight formulas per cell, seeds 17/73, 2,048 flips,
  three alternated-order rounds. There are 1,440 timed observations, **80 distinct
  formulas**, and 160 formula/seed cases. Repetitions are not independent inputs.
- AMD EPYC 9V74, Linux x86-64, Python 3.12.14, affinity CPU 0. Timers measure
  elapsed wall time on the CPU, not process CPU counters or energy. Complete calls
  include preparation, solving, result construction and an independent check of
  every SAT witness against original clauses. Parsing/imports are excluded;
  native setup, model extraction and destruction are included. Inputs are already
  in memory. Cold here means newly prepared solver state, not a fresh interpreter.
- Glucose4 through pinned python-sat 1.9.dev15 is the mature native reference.
  Its requested 2,000-conflict budget differs from 2,048 flips and is not a hard
  time bound: 120/480 native timed calls exceed that budget, up to 7,859 conflicts.
  Search seeds do not randomize this native arm; its repeated seed calls are
  matching observations, not independent native trials.
- Every declared formula is retained, including failures. No historical sealed
  confirmation set, old model, default, raw observation or tolerance was changed.

## Fresh evaluation

Each row has 16 formula/seed cases per solver and 48 timed calls per solver. Means
include both successful and unsuccessful calls; lower time alone is not success.
`I` is indexed, `D` deductive and `G` native Glucose4.

| Family, variables | SAT I / D / G | Mean ms I | Mean ms D | Mean ms G | D / I mean |
|---|---:|---:|---:|---:|---:|
| Planted binary, 128 | 16 / 16 / 16 | 1.932 | 1.151 | 0.385 | 0.596 |
| Planted binary, 512 | 16 / 16 / 16 | 8.929 | 5.482 | 1.533 | 0.614 |
| Forced mixed, 128 | 16 / 16 / 16 | 3.635 | 1.241 | 0.473 | 0.341 |
| Forced mixed, 512 | 15 / 16 / 16 | 18.458 | 5.144 | 1.922 | 0.279 |
| Planted 3-SAT, 128 | 15 / 15 / 16 | 5.491 | 5.546 | 0.804 | 1.010 |
| Planted 3-SAT, 512 | 11 / 11 / 2 | 21.760 | 22.069 | 53.465 | 1.014 |
| Equivalence ring, 128 | 6 / 16 / 16 | 9.299 | 0.687 | 0.258 | 0.074 |
| Equivalence ring, 512 | 0 / 16 / 16 | 15.260 | 2.896 | 0.850 | 0.190 |
| Uniform 3-SAT, 128 | 2 / 2 / 6 | 15.966 | 15.951 | 8.391 | 0.999 |
| Uniform 3-SAT, 512 | 0 / 0 / 0 | 24.472 | 25.709 | 41.107 | 1.051 |

| Aggregate | Indexed | Deductive | Glucose4 |
|---|---:|---:|---:|
| SAT-verified formula/seed cases | 97/160 | 124/160 | 120/160 |
| Distinct formulas solved by at least one seed | 54/80 | 66/80 | 60/80 |
| UNKNOWN formula/seed cases | 63 | 36 | 30 |
| UNSAT reports, not proof-checked | 0 | 0 | 10 |
| Mean complete ms | 12.520 | 8.588 | 10.919 |
| p95 complete ms | 24.908 | 25.483 | 62.525 |
| SAT within observed 5 ms | 42 | 82 | 116 |
| SAT within observed 20 ms | 89 | 118 | 120 |
| SAT within observed 100 ms | 97 | 124 | 120 |

Deadline counts use each formula/seed's median complete call. They are retrospective
censoring, not enforced online limits. All 192 paired general-3-SAT observations
have identical non-timing results, including witnesses, flips and trajectory
hashes. General-3-SAT quality does not improve; mean overhead reaches 5.1% and the
overall tail is slightly slower. The candidate solves all 96 structured cases.

Native Glucose4 is substantially faster on the structured cells and reports some
UNSAT formulas that these SPECTRA interfaces cannot certify. The planted 512-variable
cell favors local search under these particular, unequal budgets. This panel does
not establish superiority to native SAT solving. Rings have an intentionally easy
implication structure and only two complementary assignments; they diagnose a
failure of capped local walks, not a new hard benchmark or a novel algorithm.

## Memory and measurement correction

Memory runs use seed 17 once per formula/arm and are separate from timing.
Tracemalloc measures Python peak allocations after the input exists. Native
allocations are not included in those Python peaks.

| Family, variables | Indexed Python peak KiB, mean | Deductive Python peak KiB, mean |
|---|---:|---:|
| Binary, 128 | 86.2 | 68.1 |
| Binary, 512 | 461.7 | 490.7 |
| Forced, 128 | 102.9 | 61.2 |
| Forced, 512 | 549.9 | 295.2 |
| Planted 3-SAT, 128 | 113.1 | 113.2 |
| Planted 3-SAT, 512 | 620.8 | 609.2 |
| Ring, 128 | 40.0 | 57.2 |
| Ring, 512 | 227.0 | 334.0 |
| Uniform 3-SAT, 128 | 113.2 | 113.3 |
| Uniform 3-SAT, 512 | 621.6 | 610.3 |

The implication graph adds about 107 KiB on 512-variable rings. Allocator reuse
and object caching can influence these peaks; they are not model size or total RAM.

The first process-RSS records inherit the larger benchmark supervisor's high-water
floor (about 44 MiB). **Those RSS values are rejected for comparison and preserved**
in `evaluation/memory.jsonl`; their separate Python-allocation measurements remain
reported. [rss_check.py](rss_check.py) inserts a fresh small interpreter before
launching each worker and repeats only memory measurements on the consumed inputs.
The candidate and timed evaluation remain unchanged.

| Corrected worker RSS | Indexed | Deductive | Glucose4 |
|---|---:|---:|---:|
| Mean peak MiB | 20.943 | 21.051 | 20.971 |
| Observed range MiB | 20.105–22.000 | 20.215–21.984 | 20.102–22.441 |

All workers import the native comparator, even for Python arms, so these values
include a common interpreter/import baseline. They do not measure the minimal
dependency-free installation. Small RSS differences are not a strong ranking.

## Development history and incomplete evidence

1. The original four-family development run reported no binary/forced quality
   headroom and lower candidate cost. Its original summary/log are retained.
   **Only 407 of 432 expected raw observations are present**: 25 final-round
   observations are missing. The cause is not established. Do not reconstruct
   those timings from summaries or treat that initial summary as fully replayable.
   The frozen amendment's earlier description of a complete retained first run
   must be read with this later retention correction.
2. Before evaluation, a declared ring challenge was added. Development-02 retains
   all 540 observations: candidate 12/12 versus indexed 4/12 on rings. Initial
   normalization added up to 12.7% mean general-3-SAT overhead.
3. Development-03 reuses these development inputs to test the cheap structural
   screen; all 540 observations remain. General quality stays unchanged, mean
   overhead drops to 2.4–3.7%, and rings stay 12/12 versus 4/12. These consumed
   development inputs are not independent confirmation.
4. The final candidate/generator was published before generating the fresh
   80-formula evaluation. Its 1,440 timed observations and both 240-row memory
   records are complete. Every SAT witness passes independent original-clause
   checking. Development incompleteness does not supply missing evaluation data.

Initial package installation hit a network/proxy failure; pinned python-sat was
then installed successfully into an experiment-only directory. An initial
installed-probe invocation used a wrong relative path and failed before running;
the corrected absolute-path invocation passes. Neither failed attempt is a solver
result or omitted evaluation case.

## Retention and reproduction

[EVIDENCE.json](EVIDENCE.json) records member hashes and byte counts for
[evidence-20261002.zip](evidence-20261002.zip). The archive contains all available
raw formulas, observations, summaries, source snapshots and the original RSS
records. Development source snapshots match their original locks, including the
earlier amendment text. Full source is available at the frozen commit. No old
record was regenerated to make it match the final source.

From this checkout, verify retained bytes, formula/row identities, complete
evaluation coverage, independent SAT witnesses, memory statuses and general-path
equivalence using only Python's standard library:

```bash
python -I -S experiments/deductive_search/check_evidence.py
```

The output deliberately reports `INCOMPLETE_FIRST_RAW_RUN` for development while
checking fresh evaluation separately. It does not certify native UNSAT results,
establish provenance beyond the committed record, or reproduce old elapsed times.
The original runnable benchmark is retained; rerunning its evaluation seeds is a
replay of consumed inputs, not fresh confirmation. Further candidate selection
requires a new declared input range and a freeze before looking at those results.

## Validation and next decision

All 350 public tests pass, including 16 new contracts with exhaustive two-variable
binary truth-table checks and independent small-CNF witnesses. An actual sdist-built
wheel passes all 11 existing installed checks, matches 147 packaged source files,
and passes 512 exhaustive binary cases plus CLI solve/check outside the checkout
without Torch, NumPy, pytest or PySAT. The workspace audit preserves 449 historical
files and the recovered helper. Details and exclusions are in the
[checkpoint](../../maintenance/reviews/deductive-capability-20261002.md).
Small validation logs and installation receipts, including the first development
stdout log, are retained in [validation-20261002.zip](validation-20261002.zip)
with hashes in [VALIDATION.json](VALIDATION.json).

Keep this an optional structured-CNF capability; do not promote it as a universal
default. The remaining general-3-SAT failures warrant a separate search study with
strong native controls and new declared development inputs. No neural training or
real-world performance gain is established by this experiment.
