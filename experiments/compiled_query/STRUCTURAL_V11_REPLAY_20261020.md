# Structural-v11 exact-service replay

**Date:** 2026-10-20  
**Classification:** opened mixed-holdout development replay; not prospective confirmation  
**Runtime commit:** `be9530baf2508bb8f01dc304efc817c70d686d09`  
**Self-test commit:** `a43481e58461619c9d974130d0997817de6912f3`  
**Complete matrix workflow commit:** `d93e670c3b0e09e803b2f97ef8d84512d001dadb`  
**GitHub Actions run:** `39356700487`

## Change under test

`structural-v11` keeps banks of at least 64 requests on the raw implication engine
when the SCC quotient has exactly `2*n` components and the original graph has at
least `4*n` edges. In that regime the quotient removes no literal, so constructing
and traversing it only adds work. Compressed cases keep the prior SCC/tile path.

The runtime also retains the independently checked direct-forced-conflict fast path:
when two current forced literals already contradict on one original implication, it
emits a minimal portable kind-2 proof without replaying a full graph closure.

## Correctness result

The final matrix passed on:

- Ubuntu 24.04 x86-64 with CPython 3.13;
- native Ubuntu 24.04 ARM64 with CPython 3.13;
- UBSan;
- ASan + UBSan;
- installed self-test generation and fresh no-native audit;
- exhaustive and randomized exact-service contracts; and
- independent standard-library replay of every measured `SCQ2` bank.

No correctness, sanitizer, proof, parser, self-test, or architecture failure was
converted into a speed result.

## Performance result

The comparison gives the independently implemented direct 2-SAT challenger the
fastest median of its closure, scalar DAG, and tile-64 engines for every cell.
Ratios are `SPECTRA auto / fastest direct`; lower is better for SPECTRA.

| Requests | x86-64 ratio | x86-64 speedup | ARM64 ratio | ARM64 speedup |
|---:|---:|---:|---:|---:|
| 1 | 0.167599 | 5.97× | 0.181989 | 5.49× |
| 8 | 0.167972 | 5.95× | 0.182767 | 5.47× |
| 64 | 0.170797 | 5.85× | 0.186261 | 5.36× |
| 1,024 | 0.224501 | 4.45× | 0.230510 | 4.33× |

Using the full-precision retained summaries, the direct challenger required:

- **5.9777×**, **5.9627×**, **5.8655×**, and **4.4661×** the best SPECTRA time
  on x86-64 at 1, 8, 64, and 1,024 requests; and
- **5.5025×**, **5.4817×**, **5.3694×**, and **4.3531×** the best SPECTRA time
  on ARM64 at the same bank sizes.

Public `auto` was itself the best SPECTRA arm at all four 1,024-query graph cells
on each architecture. SPECTRA won all **32 of 32** architecture/graph/bank cells.

## Evidence identities

- x86-64 artifact:
  `sha256:58d286400df57bf484823a0f08c707b5d962c0074da74ba2457e77347f92e4b5`
- ARM64 artifact:
  `sha256:805cc40f9469e29dc05705a0c0e66a060e02ee6af58aabb0759cd7ebc89ffa4a`
- x86-64 benchmark results:
  `b13d60882e4371906abfd6d2f1d63833de1d62834032556428eaf10666861477`
- x86-64 summary:
  `4bcece5aa53fd74c766b7099c6d90d9f6b9145fd84cfb57d47eb9272c7ef6ee1`
- ARM64 benchmark results:
  `e36d0964755b4cf36570c64b4e94dad37a66f86d781760a4a88dbed92d98e042`
- ARM64 summary:
  `bc30ccff7f329cbaa9764c3e82fc3c5c0210f410f2a2c950c12df527659e661f`

## Claim boundary

This replay demonstrates a large implementation gap against the strongest of three
project-authored, separately implemented direct 2-SAT engines under the same
proof-carrying first-use `SBC1 -> SCQ2` contract. It strengthens the engineering
case for the selector update.

It does **not** create new prospective evidence because the four graph banks were
already opened before `structural-v11` was written. The original frozen
`structural-v10` holdout remains the prospective result. This replay also does not
establish general SAT leadership, ordinary graph-colouring leadership, independent
external implementation superiority, operator value, production readiness, or
outside replication.

The next competitive gate is an external-source implementation and a fresh transfer
suite frozen before either source or workload outcomes are inspected.
