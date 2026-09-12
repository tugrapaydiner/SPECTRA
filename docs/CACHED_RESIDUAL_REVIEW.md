# SPECTRA: exact-cache improvement and research decision

Review date: September 11, 2026 (America/Toronto).
Reviewed main: `e73dc05379b38ab2689e0e3054f6eee74e0290c0`.
Reviewed experimental PR23: `6c0d8695a1def36ef112d8864df700a2bc4eb3ae`.
The optional cache is developed separately from PR23. None of the 550 original
main files, modes, checkpoints, runtime defaults or scientific gates is changed.

## Assessment: approximately 35/100 on the frontier-impact scale

This is a subjective assessment of demonstrated research impact, not an employee
level, hiring probability or compensation prediction. More test cases and a
faster Python data structure do not establish a new learned capability.

The strongest feature of the project is now its ability to expose its own false
hypotheses: independent witness checks, complete-source receipts, retained failed
experiments, replayed checkpoints and explicit train/development boundaries.
The main weakness is still the absence of a practically compelling, causally
isolated learned advantage over strong classical alternatives.

The [earlier review](RESEARCH_REVIEW_20260911.md) records the small Sudoku result,
failed cross-task gate, train/development symmetry overlap and the maze checker's
BFS computation. Those limitations are not repaired by a cache. PR22's six SAT
admission gates fail. PR23's removal of the old transformer/MCTS/quantized-runtime
restrictions does not rescue its learned selector: the CI development result is
42.9688% for the primary learned arm versus 47.2222% for greedy residual repair.
The same-input local result is also below greedy repair. Keep that PR draft;
passing correctness tests is not a scientific promotion.

PR23's supplementary training-only diagnosis is especially important. Selecting
and evaluating the best patch on the same rollout samples gives misleadingly
large apparent headroom; split-selection evaluation leaves a formula-balanced
advantage of only +0.1695 percentage points, with an interval crossing zero.
That does not prove all repair policies fail. It says this restricted action pool,
teacher and representation have not demonstrated a stable learnable advantage.
The supplementary raw data are not in PR23's source tree; its PR receipt and
conversation bundle describe their separate retention.

## Implemented improvement

`eval/cnf_cached.py` adds `CachedCNFRepairState`, an optional pure-Python exact
residual state. It does not replace `CNFRepairState` or change PR23's C++ engine.
A separate benchmark uses the optional class as a drop-in state backend.

For each non-tautological clause C, let S_C be the set of variables whose signed
literal is currently true. Then:

- make(v) counts clauses with S_C empty that contain v;
- break(v) counts clauses with S_C equal to the singleton {v}.

These counts are maintained at clause transitions rather than recomputed at
every query. Single-variable score queries read two cached integers. For a patch
P, the truth-support bitset changes by XOR with the variables of P occurring in
that clause. The joint score is evaluated on the resulting support; it is not
the sum of individual scores. For example, two true literals can jointly break
a clause even though neither single flip breaks it.

The common single-flip path handles 0-to-1, 1-to-0, 1-to-2 and 2-to-1 support
transitions directly, avoiding the temporary patch-change map. Tautological
clauses stay true; repeated literals are collapsed; duplicate clauses retain
separate original identities; empty clauses remain unsatisfied. Patches are
fully validated before mutation. Process failure and MemoryError are not
transactional rollback guarantees.

This is an engineering optimization, not a new SAT algorithm. It trades cache
storage/update work for cheaper queries. Python bitset operations scale with the
highest variable index; flips are not constant-time. No lower-memory or universal
large-instance speed claim is established.

```python
from data.cnf import CNF
from eval.cnf_cached import CachedCNFRepairState

state = CachedCNFRepairState(CNF(2, ((1, 2),)), (True, True))
assert state.make_break(0) == (0, 0)
assert state.make_break_patch((0, 1)) == (0, 1)
state.flip_patch((0, 1))
assert state.unsatisfied == (0,)
```

## Verified local result: same decisions, less time

The [protocol](CACHED_RESIDUAL_PROTOCOL.md) was committed remotely as
`f4e743dd59510537cf10f8829747e39427d34172` before the retained measurement below.
Both arms use the same Python focused probSAT-style policy, not the authors'
optimized implementation. There are 64 unfiltered formulas, uniform/planted
families, four sizes from 64 to 512 variables, two search seeds and three timing
rounds: **768 observations**. The cap is 1,024 flips, without restarts. Actual RNG
initialization XORs the declared case seed with each declared search seed. Both
engines solve exactly **53 of 128** formula/search-seed pairs; other results are
UNKNOWN, not UNSAT. All 128 paired trajectories agree exactly.

| Verified local measurement | Reference | Cached |
|---|---:|---:|
| Mean complete execution | 14.87567 ms | 9.06444 ms |
| Solved formula/search-seed pairs | 53/128 | 53/128 |

The cached/reference mean-time ratio is **0.60935**, with descriptive paired
formula-cluster 95% interval **[0.59778, 0.61977]**. That is **39.1% lower mean
execution time** (about 1.64x reciprocal speed), not a 39.1% improvement in the
neural solver. Each of the eight family/size cells passes the unchanged gate.
The p95 point ratio is **0.70693**. This p95 is computed over formula/search-seed
costs after averaging the three timing repetitions, not over individual requests
in a production-serving latency distribution.

Timing includes random assignment creation, state construction, every scoring
and search step, final original-formula checking and state release. Parsed input
generation, immutable weight-table creation, source hashing and serialization are
outside. Runs hitting the flip cap remain included. This is not a complete SAT
decision procedure or a cold-start, physical-energy, native-learner or optimized
external-portfolio comparison. The local host has CPU-only PyTorch, a four-core
CPU quota and 4 GiB memory limit; no GPU was used.

### Evidence correction

The initial draft of this review and PR24 included a supposed earlier generic
update timing attempt, different local final timing numbers, and a stale-build-lock
recovery account. Their original local source/log/result files were not present
when checked. Those statements are **withdrawn as unverified**, not represented
as retained experiments or reproduced results. The measured table above comes
from the actual bundled `local_run/` raw inputs and 768 observations. Historical
Git commits preserve the draft; this correction supersedes it. No performance
threshold, score implementation or input was changed to obtain this result.

The downloaded initial CI run **34670591162** is a separate, verified failure:
both new cache jobs stopped at test collection because shared `tests/conftest.py`
imports PyTorch and the new workflow installed only NumPy and pytest. No cache
benchmark ran in those failed jobs. The correction installs the existing declared
CPU environment, including CPU-only PyTorch; shared test setup remains unchanged.
The first failed CI artifact/source is retained in the delivery bundle.

## Validation and retention

The **67 new tests pass locally**. They include exhaustive small truth tables,
optimized and atomic flip paths, randomized long patch sequences, invalid-input
atomicity, coupled patches and evidence tampering. All **768** local answers were
independently checked against the original formulas. All **256** engine-specific
deterministic paths and work records replay exactly. The existing retained-result
verifier also passes with its negative two-family scientific outcome unchanged.

Full legacy fast-regression outcomes and final-head CI receipts are recorded in
PR24. The **16 historical slow retraining tests** remain a separately reported
scope; neither a cache test nor an integrity audit establishes those training
gates. No claim is made that historical training runs were repeated.

The new read-only workflow checks Python 3.11 and 3.13, preserves the exact source,
runs all new contracts, records every comparison observation, replays all paths
and requires the unchanged bounded systems gate. Existing full regression and
checkpoint/evidence workflows are untouched. Merge requires all final-head checks
to pass and their source/evidence archives to be verified.

The verified local raw inputs, observations, source identity, replay receipts and
logs are supplied in the conversation evidence bundle, alongside the failed CI
archive. They are not falsely described as permanent raw Git blobs. CI archives
have a 90-day retention setting. Retain the delivered bundle for long-term
reproduction. The code deterministically regenerates inputs and all non-timing
observations; wall-clock durations cannot be regenerated exactly. CI measurements
repeat the declared inputs and are reproduction, not untouched statistical
confirmation or independent external-team replication.

```bash
python -m pytest tests/test_cnf_cached.py tests/test_cnf_cache_benchmark.py -q
python -m eval.cnf_cache_benchmark run --out cache-run
python -m eval.cnf_cache_benchmark verify --out cache-run --replay
```

## What would justify a major research jump

First establish useful action-space headroom before fitting a larger selector.
Use training-only failed states, connected multi-variable neighborhoods and a
bounded repair teacher. Select patches on one rollout group and evaluate their
ranking on disjoint continuations. Count teacher cost. Expand the action space
only when this cross-fitted estimate shows reproducible headroom; a same-sample
best patch is not an oracle upper bound.

Then test a signed, permutation-equivariant clause/variable model that proposes
structured repairs rather than only ranking a fixed pool of one/two-bit patches.
This is a hypothesis, not an implemented or guaranteed contribution here. Gather
states from the deployed repeated policy as well as the baseline; a target for
one baseline continuation need not rank repeated learned-policy actions.

Use a generator-by-selector factorial control: classical/classical,
learned/classical, classical/learned and learned/learned. Hold native execution,
initialization, randomness and full wall-clock accounting fixed where meaningful.
Compare with tuned native SLS/CDCL and relevant existing neural repair methods,
not only this Python reference. A gain over weak controls cannot identify a
learned contribution.

Freeze a validation-selected system and evaluate untouched instance families,
larger sizes and multiple training seeds. Report paired formula-level uncertainty,
all negative strata, tail cost, memory and verifier overhead. A proposed promotion
bar is a lower confidence bound above zero and a practically material point gain
at equal complete execution cost over the strongest selected comparator, plus
survival of the factorial controls. The exact practical margin and workload must
be declared before that experiment, not chosen after results.

Only after a useful learned advantage exists should quantization, native fusion,
cache residency and energy become the flagship story. Preserve the toy Sudoku
and maze tasks as regression fixtures, but do not let their convenience define
the impact claim. Independent reproduction and external use would then strengthen
the artifact far more than another architecture name or larger test count.

## Relevant primary prior work and role context

This is a focused novelty check, not proof that no related method exists.
[NLocalSAT](https://arxiv.org/abs/2001.09398) already combines neural solution
prediction with stochastic local-search initialization.
[Large Neighborhood Search meets Iterative Neural Constraint Heuristics](https://arxiv.org/abs/2603.20801)
explicitly studies neural destroy/repair operators and classical alternatives.
[TRM](https://arxiv.org/abs/2510.04871) establishes small recursive reasoning as
prior work. Combining recursion, repair and learning is therefore not itself a
first-of-its-kind contribution. The new mechanism and measured advantage must
be specific.

OpenAI's public [Research Engineer role](https://openai.com/careers/research-engineer-san-francisco/)
emphasizes new capabilities/performance, strong ML systems engineering and large
distributed systems. This document is not an official OpenAI L7 rubric. The cache
is a defensible bounded engineering improvement; it does not fill the remaining
research-impact, scale or independent-adoption gaps.
