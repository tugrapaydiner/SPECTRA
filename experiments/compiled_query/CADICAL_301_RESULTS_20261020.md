# Official CaDiCaL 3.0.1 challenge — results

**Date:** 2026-10-20  
**Verdict:** PASS on x86-64 and native ARM64  
**SPECTRA source:** `8ec74965b1c4e99518376c2e6781a59488e6088c`  
**Official solver source:** `arminbiere/cadical@c60730422e758ef1cebe7aeddf2dda31c996bf04` (`rel-3.0.1`)  
**GitHub Actions run:** `39358409851`

## What was tested

This challenge compared SPECTRA's public `auto` exact service against the official
CaDiCaL 3.0.1 solver core under one matched proof-carrying first-use contract.
Both arms received the same immutable `SBC1` packet and had to return complete
`SCQ2` output containing either every original-vertex colour or a portable `SBP1`
contradiction proof.

The CaDiCaL instance was persistent across each request bank, so learned clauses
remained available between rows. The timed boundary charged both systems for:

- packet parsing and validation;
- formula or implication-graph construction;
- solver/observer construction;
- every SAT/UNSAT decision;
- complete original-colour reconstruction;
- portable proof construction;
- independent original-input checking of every answer;
- `SCQ2` serialization and immutable output allocation; and
- all native state disposal.

Packet generation, native compilation, workflow startup, and later Python replay
were excluded symmetrically and retained separately.

The workload contained four disclosed graph families, each with a fixed 1,024-row
bank containing exactly 512 SAT and 512 UNSAT requests. Prefixes of 1, 8, 64, and
1,024 rows were measured. Because these banks had already been opened before this
external-source protocol, this is development evidence, not a new prospective
holdout.

## Correctness before speed

The official source identity was pinned before compilation. The receipt binds 91
source files, including the C-based Kitten core, under the shared source-manifest
hash:

```text
408be158c9bce59aec9b00582b221e58c42ff45f09fb78b99c8676ac6459ca7c
```

The linked library reported the runtime signature `cadical-3.0.1`. Before timing,
the adapter passed:

- every labelled three-vertex graph;
- every one/two-colour domain over a three-colour palette;
- every per-vertex restriction on those domains;
- random mixed banks checked against brute force and SPECTRA;
- all three portable `SBP1` proof kinds;
- complete standard-library `SCQ2` replay;
- declared-but-unused variable reconstruction;
- receipt, packet, and output mutation rejection;
- Ubuntu x86-64 and native ARM64; and
- UBSan plus ASan+UBSan.

No crash, `UNKNOWN`, malformed output, resource error, proof failure, or sanitizer
finding was converted into a mathematical answer or speed result.

## Result

The table reports how many times longer the official CaDiCaL arm took than SPECTRA.
Values are geometric means over the four graph families.

| Requests | x86-64 speedup | Native ARM64 speedup | Slowest SPECTRA win on any graph |
|---:|---:|---:|---:|
| 1 | **9.86×** | **9.26×** | **9.04×** |
| 8 | **10.23×** | **9.76×** | **9.52×** |
| 64 | **13.03×** | **13.12×** | **12.17×** |
| 1,024 | **23.67×** | **25.82×** | **15.78×** |

SPECTRA won all four bank sizes on all four graphs on both architectures:
**32 of 32 measured cells**. CaDiCaL did not win or tie a cell.

### Frozen large-gap gates

Ratios below are `SPECTRA / CaDiCaL`; lower is better for SPECTRA. All numerical
limits were committed before the official adapter was timed.

| Endpoint | Frozen gate | x86-64 | ARM64 | Verdict |
|---:|---|---:|---:|---|
| 8 | geometric mean ≤ 0.50 | 0.0977 | 0.1025 | PASS |
| 8 | every graph ≤ 0.75 | 0.0996 max | 0.1050 max | PASS |
| 64 | geometric mean ≤ 0.30 | 0.0767 | 0.0762 | PASS |
| 64 | every graph ≤ 0.45 | 0.0790 max | 0.0821 max | PASS |
| 1,024 | geometric mean ≤ 0.20 | 0.0422 | 0.0387 | PASS |
| 1,024 | every graph ≤ 0.30 | 0.0491 max | 0.0634 max | PASS |

### Full-bank graph results

| Graph family | Graph | x86-64 speedup | ARM64 speedup |
|---|---|---:|---:|
| random DIMACS | `DSJC250.5.col` | 24.47× | 29.18× |
| flat/planted | `flat300_28_0.col` | 26.31× | 32.85× |
| Leighton | `le450_15c.col` | 23.95× | 29.37× |
| geometric | `miles1000.col` | 20.37× | 15.78× |

Even the weakest 1,024-row result remained a 15.78× win. The strongest individual
cell was 32.85× on the ARM64 flat/planted graph.

## Evidence identities

### Shared source identities

- Official CaDiCaL commit:
  `c60730422e758ef1cebe7aeddf2dda31c996bf04`
- Official source-manifest SHA-256:
  `408be158c9bce59aec9b00582b221e58c42ff45f09fb78b99c8676ac6459ca7c`
- Adapter source SHA-256:
  `cf173fb4668267abf69802263e50d7addd021d7e9a7fd4004a05689d9b101d0a`
- Portable-proof support SHA-256:
  `37836a873e9a0418a663875d4ac1dee0f337135bee7ae7251303bf8e5cc8d748`

### x86-64

- Workflow artifact:
  `sha256:f0b9f2bce00f6fcb268c518888ecd030566e8a16f21cd0e362c2e3b1b872a709`
- Complete results:
  `2074cafbaf4c556a8775ca0b6217bc64fa297cd387df68f7a899edb6fe19c740`
- Summary:
  `cc58e66fcb62c27a73ddc94588bb93f88e3b234781d5a6bb4a2616a0e7336a6d`
- Combined CaDiCaL adapter library:
  `b0a21a95427a634b565cea64e86a537cd5f90359aa423e143fba1d494eabb8ab`

### Native ARM64

- Workflow artifact:
  `sha256:b05c11feefd499a3fe858e75453bc58fd894b20ec952c156d462f2a0d4d93a26`
- Complete results:
  `f1847cfa2ec905e3cfdaf399939ff7f8cdfd2ccc9a83cee82f4ad703671eda4b`
- Summary:
  `73e31c57b62e5b675cbb4d9de2c6a0be7e9cf73f0b04d9c1e95b023c28faa034`
- Combined CaDiCaL adapter library:
  `af0c1f3c45639d2449d0e449e0e4c5f031b6f743f6a264d28c828a9381529128`

The adjacent `CADICAL_301_RESULTS_20261020.json` retains the exact architecture,
per-case, gate, ratio, and hash values in machine-readable form.

## What this establishes

Under this narrow repeated binary-list exact-service contract, SPECTRA's public
`auto` path did not merely edge past the official CaDiCaL 3.0.1 core. It produced a
large and architecture-consistent gap, including a roughly 9× advantage on the
one-query diagnostic and a 20–33× advantage on three of the four full-bank graph
families.

This makes the current competitive case materially stronger than comparisons only
against a project-authored baseline. The decision engine is the official external
CaDiCaL implementation, built from a hash-bound unmodified source inventory.

## What this does not establish

CaDiCaL is a broad general-purpose SAT solver; SPECTRA is specialized for fixed
one/two-colour graph domains and repeated restrictions. The result therefore does
**not** prove that SPECTRA is a faster general SAT solver or that it wins ordinary
SAT Competition workloads.

The packet adapter, original-colour reconstruction, portable proof translation,
and benchmark harness were written inside this project. Although all of that work
was charged and independently replayed, the adapter was not authored or optimized
by the CaDiCaL team. The four workload banks were also already open before this
protocol. Consequently this is not outside-team replication or fresh prospective
confirmation.

The remaining high-value gates are:

1. a fresh transfer suite frozen before new graph bytes and outcomes are opened;
2. execution by an outside team from the retained source and protocol;
3. an independently written external adapter or challenger; and
4. real operator-authored restrictions and deployment measurements.

The defensible statement now is:

> On the disclosed proof-carrying repeated binary-list workload, SPECTRA beat the
> official CaDiCaL 3.0.1 solver core in every measured x86-64 and ARM64 cell, with
> geometric-mean gains from 9.26× to 25.82× depending on bank size.
