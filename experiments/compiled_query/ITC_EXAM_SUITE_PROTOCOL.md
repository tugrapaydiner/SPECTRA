# Prospective ITC-2007 exam-repair suite protocol

**Status:** the `set10` and `set12` executions are frozen holdouts.
Their outcomes may falsify the selector but may not retune it inside
the same suite. `set9` is development evidence and is reported
separately from the holdout gate.

## Fixed cases

| Case | Role | Expected engine |
|---|---|---|
| `set9-development` | Development | `scc` |
| `set10-holdout` | Prospective holdout | `scc` |
| `set12-holdout` | Prospective holdout | `raw` |

Every case uses the published ITC-2007 instance and feasible schedule
from `davidvanbulck/ITC-2007-CETT` at commit
`ed2d4c0cf5302b77a8b321cd0380de0bb579d538`. The original hard
constraints are checked before deriving the bounded conflict-core
repair problem. No upstream data is committed to SPECTRA.

## Frozen selector identity

The suite evaluates `structural-v10`. Selection depends only on the
immutable graph and request geometry. It never observes answers,
proof sizes, or measured times. The two holdout results cannot alter
the policy, expected engines, cases, query banks, or thresholds.

## Timing schedule

Each mode receives eight untimed warmups. There are 32 measured
repetitions. The four modes rotate through complete four-position
blocks, so every mode appears first, second, third, and fourth an
equal number of times.

Per repetition, the suite pairs `auto` with the relevant explicit
control and reports medians of paired ratios. The predeclared limits
are:

- an expected `scc` case must select `scc`, have paired
  `auto/scc <= 1.03`, and demonstrate `scc/raw <= 0.98`;
- an expected `raw` holdout must select `raw` and have paired
  `auto/raw <= 1.05`.

The strict `selector_guard_pass` covers every case. The
`holdout_gate_pass` covers only true holdouts. Development failures
are retained under `development_failures`; they are never mislabeled
as holdout failures.

## Evidence boundary

Every completed answer contains a full assignment or portable
contradiction proof and is replayed from the original `SBC1` packet.
The evidence remains a bounded conflict-core repair study, not full
ITC-2007 feasibility, optimization, outside replication, or adoption.
