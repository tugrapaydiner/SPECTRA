# Compact CNF state comparison

Base main: 181e236528b5b58263c98b09925f0d7612c919d3. Cleanup branch: maintenance/consolidate-research-workspace.

Hypothesis: replacing per-clause global-index truth bitsets with a true-literal count and XOR of true variable identifiers reduces Python allocation footprint at larger variable counts while preserving every decision of the same search policy. This is a classical representation optimization, not a learned capability or originality claim. The old cache and all historical evidence remain unchanged.

Inputs: 12 unfiltered random 3-SAT formulas. Sizes 512, 4096, 16384 variables; two uniform and two planted formulas per size; floor(4.2*n) clauses. Seed 91226000 plus case index in size/family/index order. No witness is given to search. Search seeds 17001 and 27002; 128-flip cap; three alternating-order timing rounds; no restart. Compare old CachedCNFRepairState and the new compact backend through exactly the same executable search loop. Every timed input and observation is retained, including UNKNOWN outcomes and the first invocation. Input generation, parsed inputs, import and JSON serialization are outside; initialization, construction, search, final original-formula checking and release are inside. No hard time limit is claimed.

Memory: separate instrumented constructor runs for every formula at search seed 17001, one per backend, use tracemalloc current and peak Python allocation bytes before state release. Do not treat these as RSS, native memory, energy or timings. Tracemalloc is not enabled for the speed runs.

Correctness: all corresponding paths, witnesses, statuses and flips must agree; exhaustive small-clause/patch and randomized-sequence tests use original-formula truth semantics. Retain all raw values. Report paired formula-cluster bootstrap intervals (4000 draws, seed 91226), collapsing seeds and timing repetitions before inference. Intervals are descriptive same-workload measurements, not untouched confirmation.

Promotion to the public optional backend requires zero semantic mismatches, no family/size mean-time point ratio above 1.10, and pooled peak Python allocation ratio at sizes >=4096 no greater than 0.70. Report failed conditions without retuning inputs, caps or thresholds. This is not a gate for changing default neural inference or old benchmark implementations. Standard-library public installation, complete packaged native sources, legacy fast regression and retained-evidence audits are independent software acceptance requirements.
