# Exact ranked search: bounded systems protocol

Base: `46fa23a0165ea9d61415389fdfee900285006d33`.

The target is lower complete execution cost with exactly the same seeded focused
local-search trajectory. This is a classical implementation experiment, not a
new SAT algorithm or learned capability claim. Historical source, experiments,
scientific gates and numerical tolerances remain unchanged.

## Proposed mechanisms and controls

1. Reference: the existing compact state and historical `_execute` driver.
2. Ranked-only: the existing compact state, with order-statistic selection instead
   of sorting all unsatisfied clause IDs on each flip. Ascending-rank semantics,
   random-number consumption, variable ordering and floating weights stay exact.
3. Indexed cold: immutable formula preparation, a separate break-only search state,
   ranked sampling and all preparation/release inside the execution timer.
4. Indexed warm: reuse the same immutable preparation across independent searches.
   Report preparation cost separately; never compare warm setup exclusion as a
   cold-start improvement. Mutable assignments and caches must never be shared.

The public historical `solve` and replay path remain available and unchanged.
New optimized interfaces are explicit opt-ins. The repair API retains full joint
make/break semantics; the specialized search state is not exposed as that API.

## Frozen workload

Generate 24 unfiltered formulas: sizes 512, 4096 and 16384; uniform and planted
3-SAT; four formulas per size/family cell; 4.2 clauses per variable rounded down.
Use the existing `random_3sat`, formula seeds 91326000 + sequential case index,
search seeds 19101 and 29102, caps 128 and 4096, and three timing rounds. Each
formula appears under every engine/seed/cap/round: 1152 measured executions.
Record the complete formulas, deterministic outputs, work, full path digests,
execution order and environment. Use a deterministic shuffled schedule with
SplitMix64 seed 91326. No solver filtering, favorable-case replacement, retiming
of completed observations, or suppression of failed hypotheses.

Initialize a warm prepared index once per formula and record preparation time.
Cold timing includes preparation, assignment, caches, weights, all search,
original-literal checking and release. Warm timing includes every per-search
operation and release but excludes reusable preparation. Imports, input parsing,
report serialization and compilation are outside both. These are fixed-flip
executions, not hard deadlines or matched-success external-solver comparisons.

Measure peak Python allocations separately with tracemalloc for each formula and
engine at seed 19101, cap 128. Preexisting formula storage is outside; shared
warm index storage is reported separately, not silently hidden. This is not
RSS, physical energy, or native memory. Formula preparation allocation is also
recorded separately.

## Verification and analysis

Independently evaluate each complete assignment against original signed literals.
Require exact agreement across engines/rounds for witness, residual clause IDs,
status, flip count, score-query count and path digest. Replay each unique
engine/case/seed/cap once. Check all declared input/output inventories and source
hashes. Negative and corrupted-evidence tests must reject missing/duplicate rows,
wrong seeds, changed outcomes, nonpositive timings and mismatched sources.

Collapse rounds and search seeds within each formula before comparing mean full
execution costs. Report every size/family/cap cell, paired formula-level bootstrap
intervals (4000 draws; seed 91327), pooled long-run ratios and the memory tradeoff.
Repeated timings are not independent formulas. Do not claim useful SAT superiority
from UNKNOWN trajectories or correctness tests.

The primary performance hypothesis is an indexed-cold/reference mean time ratio
at most 0.70 on the pooled 4096-flip cases with at least 4096 variables, with the
paired formula-bootstrap upper 95% endpoint below 1. Report all smaller/shorter
regressions; there is no universal speedup claim. A failed performance hypothesis
does not invalidate a correct optional implementation or relax the gate.

## Output-only neural deployment, separate scope

An additional opt-in runtime may omit diagnostic trajectory retention and heads
whose outputs cannot affect the final recurrence. Preserve the historical
`CPURecursiveRuntime.forward` byte-for-byte. Require bitwise-equal final logits,
answers and final halt outputs to its original execution on deterministic CPU
artifacts and inputs, including multiple batch sizes and supervision depths.
Report changed head-call counts and actual retained tensor storage separately
from RSS. This is an inference implementation result, not improved model quality.
Time both APIs after native warmup; do not count compilation as warm inference.

## Integration

Run the complete fast suite, explicit native/output-only contracts, installed
wheel checks outside the checkout, historical evidence verification and CI. Do
not count excluded slow retraining tests as passes. Publish raw records and a
reproduction command with the exact executable source. The protocol precedes
implementation; diagnostic tests are development, not held-out model confirmation.
