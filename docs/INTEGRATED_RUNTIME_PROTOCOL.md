# Integrated CPU runtime protocol

Base: `9fd6682345b34e992b3cab7e55750e9ef4b04e87` (0.7.1).
Branch: `perf/integrated-final-runtime`.

## Hypothesis and scope

Combine the existing once-validated packed-weight runtime and final-output-only
execution without changing recurrent arithmetic or final numerical outputs.
Use profiles to decide whether another optimization is justified. This is an
implementation experiment, not a learned-capability or external-solver claim.
Historical FP-only M16 checkpoints are not silently converted into packed models.
Historical implementations, results, protocols and tolerances remain recoverable.

## Required controls

Measure all four combinations: checked weights with trace, checked weights with
final-only output, validated owned weights with trace, and validated owned weights
with final-only output. Compare a candidate against the fastest supported existing
configuration as well as the historical baseline. Arbitrary runtime subclasses
and unsupported graphs must still be rejected. The full diagnostic path remains
available. No timing observation may be discarded because it is unfavorable.

## Correctness gate

Require bitwise-identical logits, final halt output and decoded answers for the
same artifact and input. Check all work counters, nested recurrence budgets,
batch geometry, invalid inputs, repeated/interleaved calls, separate runtime
instances, and absence of retained recurrent trajectories. Include exported
trained hard-ternary/A8 fixtures, not just randomly initialized models. Any new
fixture training is a declared cost/equivalence fixture, not new task capability.
The original fast suite, retained-evidence audit and installed-wheel checks must
pass. Excluded slow tests are reported as excluded, never counted as passes.

## Measurement gate

Freeze the complete executable configuration, source hashes, artifact hashes,
inputs and environment before timing. Use CPU-only execution and one pinned core
where available. Randomize complete arm order in balanced repeated measurements.
Keep full raw integer-nanosecond observations and exact output digests. Native
compilation, model/weight construction and first execution are recorded separately
from warmed API latency. Returned tensor storage, retained owned weights,
tracemalloc and process RSS are distinct scopes. Do not label returned storage
as peak memory, and do not infer energy from latency. Profiling is a separate
instrumented pass, never substituted for uninstrumented timing.

Report every case, regressions, medians and tails. Repeated timings are not
independent artifacts. No universal speedup, novelty, quality improvement or
hiring-level conclusion follows from a passing implementation check. Until the
final frozen matrix runs, this document specifies evaluation safeguards rather
than a preregistered numerical success threshold. A later benchmark configuration
freeze must be distinguished from this protocol's publication.

## Repository maintenance

Fix executable documentation rather than rewriting historical receipts. Add a
version-independent wheel-selection path with rejection of ambiguous builds,
exercise documented commands, and keep the public base package free of mandatory
Torch/NumPy imports. Add focused CI for the integrated runtime and evidence.
No main merge, release publication, branch deletion or repository permission
change is implied by this experimental branch.
