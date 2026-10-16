# Prospective ITC-2007 exam-repair suite protocol

**Status:** frozen before executing the `set10` and `set12` result panels.  The
selector may be falsified by these holdouts, but it must not be retuned inside the
same suite.

## Purpose

The first application transfer used `set9` and exposed an important limitation:
the original automatic policy selected the SCC engine even though the explicit raw
engine was faster on that dense repair bank.  The structural-v2 policy adds one
answer-independent rule:

> For a bank of at least 64 requests, use raw traversal when the immutable request
> packet contains at least eight pins per request on average.

The threshold was chosen from the already exposed synthetic development panel and
the `set9` development transfer.  It depends only on packet geometry, not SAT/UNSAT
answers, proof sizes, or measured times.  Short banks, base-infeasible graphs and
the prior SCC/component rules retain their previous ordering.

## Fixed external source

Every workflow checks out `davidvanbulck/ITC-2007-CETT` at commit
`ed2d4c0cf5302b77a8b321cd0380de0bb579d538` and retains its `LICENSE` in each
case directory.  No upstream data is committed to SPECTRA.

The fixed cases are:

| Case | Role | Instance | Published feasible schedule |
|---|---|---|---|
| `set9-development` | Development | `examtt-instances/Instances/set9.exam` | `InitMethods/SolStore/set9_OBSIFeas_0.sol` |
| `set10-holdout` | Prospective holdout | `examtt-instances/Instances/set10.exam` | `InitMethods/SolStore/set10_OBSIFeas_0.sol` |
| `set12-holdout` | Prospective holdout | `examtt-instances/Instances/set12.exam` | `InitMethods/SolStore/set12_OBSIFeas_0.sol` |

The bytes are public, so “holdout” means **execution holdout**, not secret data:
no selector-v2 answer or timing from `set10` or `set12` may influence the committed
policy or acceptance threshold.

## Per-case construction

Each case uses the exact construction in
[`ITC_EXAM_TRANSFER_PROTOCOL.md`](ITC_EXAM_TRANSFER_PROTOCOL.md): validate the
published schedule against all original hard constraints, freeze exams involved in
unsupported hard constraints, assign one deterministic duration-compatible
alternative to eligible exams, build the shared-student/`EXCLUSION` conflict core,
and generate 1,024 unique deterministic repair events.

Every event is answered by `auto`, `raw`, `scc`, and `route`.  Completed answers
must include a full assignment or portable contradiction proof and must replay from
the original `SBC1` packet without native code.

## Timing schedule and prospective gate

Each engine receives one untimed warmup.  In each of five measured repetitions the
engine order rotates, preventing one mode from always running first or last:

1. `auto, raw, scc, route`
2. `raw, scc, route, auto`
3. `scc, route, auto, raw`
4. `route, auto, raw, scc`
5. repeat from the first order

For every holdout, pair `auto` and `raw` by repetition and compute the median of the
five `auto_elapsed / raw_elapsed` ratios.  The predeclared guardrail is:

```text
paired-median auto/raw ratio <= 1.15
```

This is a bounded overhead guardrail, not a speedup claim.  Development-case timing
is reported but cannot pass or fail the prospective gate.  Failed holdouts remain
in the artifact and may not be removed, renamed, rerun under a changed threshold,
or converted to development cases.

## Evidence and audit

The suite manifest binds:

- source commit and per-case source inventory;
- upstream repository and commit;
- exact instance and solution paths and SHA-256 identities;
- development/holdout role;
- selector policy and threshold;
- per-case manifest hashes.

The suite audit reruns every child audit, checks every result and proof, checks the
rotated timing schedule, computes paired ratios from recorded repetitions, and
publishes all failures.  A successful result is still project-owned execution, not
outside-team replication, operator adoption, full ITC-2007 feasibility, or
algorithmic-priority evidence.
