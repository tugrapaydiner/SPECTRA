# Prospective ITC exam-repair suite — 2026-10-16

## Result

The prospectively fixed bounded exam-repair suite completed on all three declared
environments. Both execution holdouts passed the unchanged paired-median
automatic/raw guardrail of **1.15**. Every admitted result was independently
replayed from the original pinned ITC bytes, and no failed cell or output was
removed.

This is application-transfer evidence for the binary-domain exam-conflict core. It
is **not** a full ITC-2007 timetabling result, an operator deployment, outside-team
replication, universal performance evidence, or an algorithmic-priority claim.

## Frozen boundary

| Item | Value |
|---|---|
| Measured SPECTRA commit | `ff77d802e9a2bb3c1d97c5acf70d8871dc114f7c` |
| Published implementation commit | `3036a3c97cf5706d60ab9e277708081b4f241954` |
| Upstream ITC repository commit | `ed2d4c0cf5302b77a8b321cd0380de0bb579d538` |
| Selector policy | `structural-v2` |
| Requests per case | 1,024 |
| Warmup | one untimed call per mode |
| Measured repetitions | five |
| Mode order | rotated each repetition |
| Holdout gate | paired median of `auto_elapsed / raw_elapsed <= 1.15` |
| Development case | `set9` |
| Execution holdouts | `set10`, `set12` |

The implementation, policy, case roles, event generator, repetition count and gate
were committed before any holdout execution. The automatic engine used packet
structure only. All three cases triggered `high_pin_density` and selected `raw`.

## Functional outcomes

The decisions and counts were identical on x86 Python 3.11, x86 Python 3.13 and
native ARM64 Python 3.13.

| Case | Role | Exams | Periods | Enrolments | Conflicts | Explicit exclusions | SAT | UNSAT | Pins |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `set9` | development | 169 | 25 | 2,532 | 1,114 | 1 | 268 | 756 | 9,698 |
| `set10` | holdout | 214 | 32 | 7,853 | 1,133 | 0 | 327 | 697 | 9,311 |
| `set12` | holdout | 78 | 12 | 3,685 | 560 | 6 | 24 | 1,000 | 10,361 |

For `set10`, 178 UNSAT answers used an empty-domain proof and 519 used a
forced-opposites path proof. For `set12`, those counts were 269 and 731. The native
engines agreed on every SAT/UNSAT decision, and the standard-library audit checked
every complete assignment or contradiction proof against the original fixed graph
and current restrictions.

## Prospective guardrail

Values below are the median of the five paired repetition ratios, not a ratio of
independently aggregated medians.

| Environment | `set9` development | `set10` holdout | `set12` holdout | Holdout gate |
|---|---:|---:|---:|---|
| Linux x86, Python 3.11 | 0.995442 | **0.999355** | **1.003860** | PASS |
| Linux x86, Python 3.13 | 1.073799 | **1.019084** | **1.026623** | PASS |
| Linux ARM64, Python 3.13 | 1.017422 | **0.999036** | **1.000705** | PASS |

The largest holdout paired-median ratio was **1.026623**, leaving approximately
10.7 percentage points of margin to the fixed 1.15 limit.

Individual repetitions were intentionally retained. In particular, the first
`set12` repetition on x86 Python 3.13 had ratio **1.168387**, and the first
`set10` repetition on x86 Python 3.11 had ratio **1.158769**. Both exceed the gate
when viewed alone; the predeclared gate is the paired median over five rotating
repetitions, so these are negative retained observations rather than deleted or
retried cells.

### Paired ratios, all repetitions

| Environment | Case | Five `auto/raw` ratios |
|---|---|---|
| x86 / Python 3.11 | `set10` | 1.158769, 0.999355, 1.000486, 0.984425, 0.995438 |
| x86 / Python 3.11 | `set12` | 1.123582, 1.004558, 1.000192, 1.003860, 0.997922 |
| x86 / Python 3.13 | `set10` | 1.098272, 1.066995, 1.018905, 0.998810, 1.019084 |
| x86 / Python 3.13 | `set12` | 1.168387, 1.012663, 1.026623, 0.978225, 1.040015 |
| ARM64 / Python 3.13 | `set10` | 1.112409, 0.999036, 0.987604, 1.004978, 0.985442 |
| ARM64 / Python 3.13 | `set12` | 1.099958, 0.985422, 0.994436, 1.000705, 1.030142 |

The automatic and explicit raw modes execute the same native engine after policy
selection. Ratios close to one therefore measure selection/dispatch noise and host
variation rather than an algorithmic speedup. The purpose of this gate is to stop
the automatic policy from imposing material overhead on unseen application cases.

## Complete-call medians

Elapsed time includes native construction, all 1,024 decisions, complete witnesses
or portable proofs, independent native checking, `SCQ2` construction, disposal and
hashing. Candidate generation, packet creation, compilation and process startup are
outside the measured call.

| Environment | Case | Auto | Raw | SCC | Route |
|---|---|---:|---:|---:|---:|
| x86 / Python 3.11 | `set9` | 1.286 ms | 1.292 ms | 1.288 ms | 1.657 ms |
| x86 / Python 3.11 | `set10` | 1.163 ms | 1.161 ms | 1.144 ms | 1.358 ms |
| x86 / Python 3.11 | `set12` | 0.888 ms | 0.887 ms | 0.987 ms | 1.199 ms |
| x86 / Python 3.13 | `set9` | 0.823 ms | 0.765 ms | 0.788 ms | 0.952 ms |
| x86 / Python 3.13 | `set10` | 0.697 ms | 0.666 ms | 0.693 ms | 0.795 ms |
| x86 / Python 3.13 | `set12` | 0.536 ms | 0.523 ms | 0.605 ms | 0.725 ms |
| ARM64 / Python 3.13 | `set9` | 1.114 ms | 1.095 ms | 1.052 ms | 1.402 ms |
| ARM64 / Python 3.13 | `set10` | 1.004 ms | 1.004 ms | 0.972 ms | 1.172 ms |
| ARM64 / Python 3.13 | `set12` | 0.772 ms | 0.771 ms | 0.824 ms | 1.050 ms |

SCC remains slightly faster than raw on some cells. Selector-v2 deliberately
prioritises a conservative no-material-overhead rule for heavily pinned application
banks rather than claiming a per-cell optimum. The development result does not
justify retuning after seeing the holdouts.

## CI and artifacts

### Development transfer

Workflow run: `38951373279`

- Linux x86 / Python 3.11: PASS
- Linux x86 / Python 3.13: PASS
- Native Linux ARM64 / Python 3.13: PASS

Artifacts:

| Environment | Artifact ID | Artifact SHA-256 |
|---|---:|---|
| x86 / Python 3.11 | `12068467826` | `3afc63b70f8499c03fe97df78eff288c8316accaf714bace3585be938cab1fe4` |
| x86 / Python 3.13 | `12068077883` | `32823ece7589b6f76901469c141210ca3796d489f8a90eaa29aaac0e933d69d1` |
| ARM64 / Python 3.13 | `12068222627` | `d5de0a96b0a8347df58296e6bad01e4cd24e49cb908e952c47056609f2acf762` |

### Prospective suite

Workflow run: `38951398495`

- Linux x86 / Python 3.11: PASS
- Linux x86 / Python 3.13: PASS
- Native Linux ARM64 / Python 3.13: PASS
- Source contracts: PASS on all cells
- Installed-wheel exam-repair execution: PASS on all cells
- Freeze-before-execution: PASS on all cells
- Independent source-only audit: PASS on all cells

Artifacts:

| Environment | Artifact ID | Artifact SHA-256 |
|---|---:|---|
| x86 / Python 3.11 | `12068498169` | `fb100a67983b02c7969809efb680deaf5b4002c6f2242122d8cca8ff4a415c23` |
| x86 / Python 3.13 | `12068552843` | `8f86935fa6e6811c56fc4156d8dc9ea422c4a14b5ef0cba356b496d3ef4208e7` |
| ARM64 / Python 3.13 | `12068023095` | `35247cef687fc26ccbfec69d6903ed421a4a3fc9df8682b7143afa0600b055f4` |

## Preserved failed attempts

The first transfer run (`38951148893`) and first suite run (`38951172493`) failed
during test collection because the dependency-light public-test command loaded the
repository-wide ML `tests/conftest.py` without NumPy. They stopped before evidence
freezing or any holdout execution. The correction isolated public tests with
`--confcutdir=tests/public`; it did not change the selector, cases, event bank,
repetitions, timing order or guardrail. The failed runs remain visible.

## Replication

Follow [`ITC_EXAM_REPLICATION.md`](ITC_EXAM_REPLICATION.md) from a clean checkout.
An independent disagreement or negative result is valid evidence and should be
published with the complete artifacts rather than normalised away.
