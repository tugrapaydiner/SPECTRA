# Exact CPU efficiency: SPECTRA 0.7.1

SPECTRA's indexed search is an **optional implementation of the same seeded
classical search**, not a new SAT algorithm or a learned reasoner. It addresses
a measured bottleneck: the historical driver sorted every unsatisfied clause ID
on every flip. The original `solve`, full repair state and neural runtime remain
unchanged for historical replay.

## Use the new path

```bash
python -m pip install .
spectra cnf solve examples/tiny.cnf --backend indexed --seed 7 --max-flips 4096 --out answer.json
spectra cnf check examples/tiny.cnf answer.json
```

The default backend is still `compact`. Existing output files are not overwritten.
`UNKNOWN` is a bounded-search outcome, not an UNSAT proof. There is no hard
wall-clock deadline.

```python
from spectra.cnf import CNF, PreparedCNF, solve_indexed

problem = CNF(3, ((1, 2), (-1, 3), (-2, 3)))
cold = solve_indexed(problem, seed=7, max_flips=4096)
prepared = PreparedCNF(problem)  # pay once; immutable and reusable
warm = prepared.solve(seed=7, max_flips=4096)
assert cold.witness == warm.witness
assert cold.path_sha256 == warm.path_sha256
```

`solve_indexed.elapsed_ns` includes preparation and release. A prepared object's
`solve` includes every per-search operation but excludes reusable preparation.
Benchmark tables use the **outer complete-call timer**, also including result
construction. Parsed CNF input, imports and serialization are outside both.
The index's memory is not free: report it alongside warm-call allocations.

## What changes and what stays exact

A Fenwick prefix tree over fixed 256-bit blocks selects the kth unsatisfied
clause in ascending ID order. Updates change membership without rebuilding a
sorted tuple. Each instance owns its blocks and counters. This is a classical
order-statistic data structure.

The prepared index holds canonical literals, variable order, signed occurrences
and the same break-score weights. Separate search states own assignments, counts,
XORs, break scores and residual sets. The focused policy consumes break scores,
so its specialized state does not maintain unused make scores. The full joint
make/break repair API is neither weakened nor replaced.

For identical original formula, seed and flip cap, induction over flips gives
trajectory equivalence: equal assignment implies equal violated clause IDs;
rank selection and the next RNG call therefore choose the same clause. Sorted
variables and scalar weight expressions give identical weighted choice, followed
by the same flip and path-hash update. Clause count plus XOR preserves exact
sole-true-variable break contributions. Tautologies cannot become unsatisfied,
duplicate literals are canonicalized, and duplicate original clauses keep their
identities. Original signed literals independently check the returned assignment.
This argument is backed by exhaustive small-set tests, finite-difference score
checks and complete seeded execution comparisons, not by timings alone.

## Frozen local experiment

The [protocol](INDEXED_SEARCH_PROTOCOL.md) was published at commit
`506510cf89ee463d3e3fbf7a7c52299a9cbae046` before implementation/measurement.
The executable snapshot and hashes were frozen locally before execution; its
later remote publication is not claimed as external executable preregistration.

Twenty-four unfiltered formulas span uniform/planted 3-SAT, sizes 512, 4096 and
16384, and four formulas per cell. Two search seeds, two caps, three randomized
rounds and four implementations yield **1,152 measured calls**. The verifier
checks every original-literal answer and replays **384 distinct backend paths**.
Every compared path, witness and work count agrees. Repetitions are not additional
independent formulas.

Local Python 3.13.5, single pinned CPU, cold indexed/reference ratios:

| Variables | Family | 128 flips | 4096 flips |
|---|---|---:|---:|
| 512 | planted | 1.008 | 1.059 |
| 512 | uniform | 1.003 | 1.063 |
| 4096 | planted | 0.816 | 0.487 |
| 4096 | uniform | 0.906 | 0.445 |
| 16384 | planted | 0.948 | 0.210 |
| 16384 | uniform | 0.888 | 0.199 |

The frozen primary, pooling the 16 larger formulas at 4096 flips, measures
**1005.795 ms reference versus 238.881 ms indexed cold**, a ratio of **0.237505**
and approximately **4.21x lower execution time**. The paired formula-bootstrap
95% ratio interval is **[0.216128, 0.281891]**; the predeclared ratio <=0.70 and
upper interval <1 gates pass. This is a bounded local result, not a forecast for
other hardware. CI reruns have their own raw evidence and statistics.

The rank-only ablation explains most of the large-input improvement. Prepared
warm execution additionally avoids rebuilding the formula index. Neither is a
new learned capability. On the small long-run cells the candidate is about 6%
slower, and rank-only is worse still; the historical default is retained.

**The large measured cases all return UNKNOWN.** Each engine has 24 SAT timing
observations (8 distinct case/seed/cap outcomes, repeated three times) out of
288 calls. Exact bounded search is cheaper; no increase in success rate, useful
SAT superiority or external-solver speedup follows.

## Memory is a disclosed tradeoff

Mean measured peak Python allocations, MB, averaged over eight formulas per size:

| Variables | Reference cold | Indexed cold | Indexed warm per call | Retained shared index |
|---|---:|---:|---:|---:|
| 512 | 0.735 | 0.821 | 0.092 | 0.689 |
| 4096 | 6.997 | 7.380 | 1.016 | 6.279 |
| 16384 | 28.260 | 29.762 | 4.080 | 25.448 |

Warm and index columns have distinct lifetime scopes, not a free total-memory
reduction. These are tracemalloc allocations, not RSS, native memory or energy.
Cold peak allocation increases. This release's search result is a speed result.

## Output-only neural deployment

```python
from spectra.inference import predict_final
# runtime is the stock deploy.m10_runtime.CPURecursiveRuntime
prediction = predict_final(runtime, input_tokens)
```

This requires the research/native extras. It keeps all recurrent computation,
only omitting intermediate heads that cannot feed back into this graph and the
returned diagnostic trajectory. Final logits, answer and halt output must match
historical `forward` bitwise. It is not early stopping, a changed training loss,
or a new model. Custom runtime subclasses are rejected; a runtime instance is
not shared concurrently between workers.

Six deterministic **untrained** artifacts, dimensions 16/64, supervision depths
1/4/16, batches 1/4 and five rounds produce 120 measured calls. At dim64, depth16,
batch1, reachable returned tensor storage falls from **136,384 to 452 bytes**.
This is storage reachable from the returned object, **not peak inference RAM or
RSS**. The local mean latency ratio in that cell is 0.964; some cells regress.
No universal inference speedup or improved answer quality is claimed.

## Reproduce and inspect

```bash
python -m pytest -m 'not slow'
python scripts/bench_indexed_search.py run --out outputs/indexed-new
python scripts/bench_indexed_search.py verify --out outputs/indexed-new --replay
python scripts/bench_output_only.py --out outputs/output-only-new
```

New output directories are required. Use a single pinned core when available;
compilation and model loading precede warm inference timing. Source hashes,
complete input formulas, raw timing rows, all outcomes, allocation/preparation
records and analysis travel together. The versioned GitHub release workflow
publishes source, installed wheel and selected literal CI evidence only after
matching source trees, test results, original-formula checks and replays pass.
It refuses to replace an existing version/tag. Each publication result must be
checked; a configured workflow alone is not a published release.

Old slow retraining tests remain separate and are never counted as passes. The
historical failed learned selector and workload-admission gates stay failed.
The next research step is still useful-workload admission and cross-fitted repair
headroom, not another efficiency claim based solely on a microbenchmark.
