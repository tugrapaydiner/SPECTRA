# Direct 2-SAT challenge — development results

**Date:** 2026-10-20  
**Classification:** exposed development evidence, not a holdout or general state-of-the-art claim

## What was challenged

The missing close competitor was a purpose-built 2-SAT service under the same
`SBC1 -> SCQ2` contract as SPECTRA. The challenger is separately implemented C++
with its own packet parser, implication graph, SCC decomposition, proof builder,
serializer and original-input observer. It exposes three independently checked
engines:

- dense component-closure;
- scalar condensation-DAG propagation; and
- tile-64 propagation across request banks.

For every measured cell the comparison gives the challenger the faster of those
three engines after timing. SPECTRA is reported both as its public `auto` policy
and as the faster explicit SPECTRA arm. This is deliberately conservative toward
the challenger.

## Correctness before timing

All three challenger engines passed:

- every three-vertex graph;
- every one/two-colour domain over a three-colour palette;
- every per-vertex restriction on those domains;
- random mixed SAT/UNSAT banks checked against brute force and SPECTRA raw;
- all three portable `SBP1` proof kinds;
- independent standard-library `SCQ2` replay;
- receipt and malformed-packet rejection;
- Linux x86-64 and native ARM64; and
- UBSan plus ASan+UBSan.

The challenger remains project-authored. Passing these contracts makes it a
serious matched control, not an externally independent implementation or proof
that no faster 2-SAT service exists.

## Exposed synthetic matrix

Five disclosed graph constructions were measured at 1, 8, 64 and 1,024 requests.
Each cell used calibrated batches and 24 repetitions in complete six-position
blocks. `direct / best SPECTRA` above 1 favours SPECTRA.

| Requests | x86-64 direct / best SPECTRA | ARM64 direct / best SPECTRA | SPECTRA cell wins |
|---:|---:|---:|---:|
| 1 | 1.8414 | 1.7549 | 10 / 10 |
| 8 | 1.7215 | 1.6139 | 10 / 10 |
| 64 | 1.9468 | 1.8471 | 10 / 10 |
| 1,024 | 1.9489 | 1.9604 | 10 / 10 |

These are geometric means over five exposed synthetic cases per architecture.
They are useful implementation diagnostics, not prospective evidence.

## WAP05/06 development matrix

The second matrix used the exact pinned WAP05/06 graph bytes from
`marijnheule/clicolcom@4932048642da2144f387961b595112277afff82f` and the historical
1,024-query development generator. Those queries are SAT-filtered and were
already exposed before this challenge.

### Public `auto` versus the fastest direct engine

The following values are `SPECTRA auto / best direct`; below 1 favours SPECTRA.

| Requests | x86-64 | ARM64 | Interpretation |
|---:|---:|---:|---|
| 1 | 0.7290 | **1.0561** | SPECTRA wins x86; direct 2-SAT wins both ARM cells by about 5% |
| 8 | 0.7211 | 0.9358 | SPECTRA wins in aggregate; the ARM margin is small |
| 64 | 0.4624 | 0.4777 | SPECTRA is about 2.16× / 2.09× faster |
| 1,024 | 0.2952 | 0.2596 | SPECTRA is about 3.39× / 3.85× faster |

The ARM single-query loss is a real negative result. It is not pooled with long
banks or renamed as noise. The current evidence supports sustained-bank
leadership on these exposed inputs, not universal first-query leadership.

## Identities

- Source and WAP workflow commit:
  `d4bb068ede57668d269224b89dd5600cf5d3bd86`
- Six-arm correctness/development run: `39352838532`
- WAP05/06 run: `39353268332`
- WAP x86 artifact digest:
  `sha256:9f75814563fe5bb94728448b1aef838b93442286d37494858ed4d4cb93f442f0`
- WAP ARM artifact digest:
  `sha256:3b825e5a394577cc45d101042965a052576d827e06dfdf9440b5b7520599bdff`

## What this establishes

SPECTRA's long-bank advantage is not an artifact of comparing only with general
CDCL or BDD systems. It survives a separately implemented, proof-carrying direct
2-SAT challenger with closure, scalar and tile-64 paths on both synthetic mixed
banks and realistic-size WAP development graphs.

## What remains open

- all WAP05/06 restrictions in this matrix were SAT-filtered;
- the topology and workload were already exposed;
- only two realistic-size graphs were measured;
- the challenger was written inside this project;
- SPECTRA lost the ARM single-query cells;
- no operator-authored restrictions or external replication are present; and
- no general SAT, graph-colouring, asymptotic or production superiority follows.

The next legitimate step is a prospectively frozen, mixed SAT/UNSAT study on
unopened graph bytes from several topology families. Its protocol and generator
must be committed before the graph contents are acquired.