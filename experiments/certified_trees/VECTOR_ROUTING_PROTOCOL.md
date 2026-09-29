# Model-frozen routing and end-only accumulator experiment

The first new native replay has completed all1260 predetermined timing cells.
All source models, packed8/16-bit coefficients, mathematical bounds and outputs
remain frozen. Compact-only16-bit certifies11284/11295 inputs; eleven require
original-model fallback. No certified disagreement was observed against official
CatBoost1.2.8/1.2.10, but full-coverage pipelines are generally slower than the
optimized official library at batch32. Preserve the complete first result.

Hypothesis: the implementation's scalar predicate/tree routing and dynamically
sized per-row additions hide the memory savings of compact leaves. Test two
fixed same-function implementations: (a) transpose a32-row tile and form leaf
indices with byte-vector predicate operations; (b) additionally use fixed-width
register accumulators for end-only certification. Accumulated class contrasts
and every source-error bound remain EXACTLY the same. Unsupported depths and
portable hosts retain general routing. Padding may protect vector loads but
must be counted. Inactive SIMD lanes are not new examples or skipped work.

Keep the literal old source/library. Match new versus old labels, certificate
status, evaluated-tree counts and shared-route refinement on all synthetic and
real tests BEFORE timing. Add both new full-coverage variants to the entire
15-arm baseline matrix, plus new compact8/16 end-only and8-to16 refinement arms;
freeze the full concrete inventory before the new run. Four models, three batch
sizes, seven shuffled whole jobs, one pinned CPU, no concurrent fitting/building.
No selective reruns, model fitting, quantization change, lower precision control
or source-model accuracy claim. End-only specialization is not safe early-exit
innovation; vectorized tree evaluation has established prior art.

A useful implementation gain must survive official CatBoost1.2.10, not merely
our initial slow implementation. Keep all regressions and compact/fallback
storage separate. This continuation does not inherit the missing historical
1092/1260-cell runs or their absent model bytes.
