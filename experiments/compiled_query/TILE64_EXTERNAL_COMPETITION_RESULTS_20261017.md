# Tile64 external-competition audit — 17 October 2026

## Verdict

**External competition: decisive pass.**  On both tested CPU architectures,
`exact-bank-scc` is about 2.8–2.9× faster than persistent MiniCard and
persistent CaDiCaL under the frozen complete-cost, all-SAT witness contract.  On
x86-64 it is about 3.94× faster than reduced native CUDD.  Every individual
graph clears the predeclared per-graph margin; this is not a geomean rescued by
one easy instance.

**Frozen composite audit: fail.**  The same run also required SCC to beat the
current optimized raw engine by at least 25% on the complete cold-session
boundary.  SCC beat raw on both graphs and both architectures, but only by about
5.5–7.6% complete-session time.  The predeclared gate is therefore red and is
not weakened after measurement.

The correct status is consequently:

- the requirement “beat serious external competition by a large margin” is
  satisfied on this disclosed development scope;
- the requirement “also show a large cold-session SCC/raw gap on these WAP
  graphs” is not satisfied;
- Issue 2 automatic selection remains paused; and
- this audit branch is not merged as a passing checkpoint.

## Frozen protocol

The protocol was committed before timing.  It used the exposed WAP development
graphs `wap05a.col` and `wap06a.col`, reconstructed at the pinned upstream commit
`4932048642da2144f387961b595112277afff82f`.  Each graph supplied exactly 1,024
historically SAT-accepted restrictions.  Nine paired complete sessions were run
per graph and arm, cycling forward, reverse and fixed-shuffle query order.

Every arm began from the same canonical case JSON.  Complete time charged input
read/validation, deterministic ordering, solver construction, all 1,024
queries, complete original-vertex witness materialisation, independently owned
original-input witness checking, output serialisation, diagnostics and explicit
disposal.  Process launch/import was retained separately as a cold-wall receipt.

Arms:

1. current merged `exact-bank-scc` candidate;
2. current optimized `exact-bank-raw` internal control;
3. persistent MiniCard from pinned `python-sat==1.9.dev15`;
4. persistent CaDiCaL 1.9.5 from the same pinned distribution; and
5. reduced native CUDD from pinned `tulip-control/dd` commit
   `4442d498ddea1bec0071315f538cb5ef37443985` on x86-64.

Because every query was SAT, each arm had the same relevant certificate
obligation: return and independently validate a complete original-vertex
witness.  Exact SCQ2 output was additionally replayed through the independent
exact audit.

## Correctness and completeness

| Environment | Scheduled complete sessions | Audited SAT answers | Incomplete / UNKNOWN / false result / crash / timeout |
|---|---:|---:|---:|
| x86-64, CPython 3.13 | 90/90 | 92,160 | 0 |
| ARM64, CPython 3.13 | 72/72 | 73,728 | 0 |
| **Combined** | **162/162** | **165,888** | **0** |

Before timing, both environments passed:

- source-tree and source-hash freezing;
- byte identity of packaged and research exact C++ sources;
- the frozen gate evaluator's rejection tests;
- the 65-row, two-tile exact bank contract; and
- byte-level verification of both acquired WAP graphs against the retained
  upstream inventory.

## Complete-cost competition results

Ratios are `SPECTRA exact-bank-scc complete time / baseline complete time`;
lower is better for SPECTRA.  The frozen external gates were geomean `<= 0.50`
and every graph `<= 0.67`.

### x86-64 / CPython 3.13

| Baseline | Graph A | Graph B | Geomean ratio | SPECTRA speedup | Frozen external gate |
|---|---:|---:|---:|---:|---:|
| Persistent MiniCard | 0.353067 | 0.359621 | **0.356329** | **2.806×** | PASS |
| Persistent CaDiCaL 1.9.5 | 0.344791 | 0.353595 | **0.349165** | **2.864×** | PASS |
| Reduced native CUDD | 0.248679 | 0.258566 | **0.253574** | **3.944×** | PASS |

Across all paired repetitions, complete ratios stayed within:

- MiniCard: `0.345940–0.369528`;
- CaDiCaL: `0.340047–0.365326`; and
- CUDD: `0.241639–0.266489`.

### Native ARM64 / CPython 3.13

| Baseline | Graph A | Graph B | Geomean ratio | SPECTRA speedup | Frozen external gate |
|---|---:|---:|---:|---:|---:|
| Persistent MiniCard | 0.350601 | 0.357292 | **0.353931** | **2.825×** | PASS |
| Persistent CaDiCaL 1.9.5 | 0.338207 | 0.344505 | **0.341342** | **2.930×** | PASS |

Across all paired repetitions, complete ratios stayed within:

- MiniCard: `0.348208–0.360291`; and
- CaDiCaL: `0.334389–0.347726`.

This is a substantial external advantage, not a marginal result: the weakest
single paired external ratio across both architectures was still below `0.370`.

## Why the composite raw gate failed

The frozen internal-control gates were geomean complete ratio `<= 0.75` and
every graph `<= 0.85`.

| Environment | Raw graph A | Raw graph B | Complete geomean | Complete speedup | Gate |
|---|---:|---:|---:|---:|---:|
| x86-64 | 0.923786 | 0.934761 | **0.929257** | 1.076× | FAIL |
| ARM64 | 0.946753 | 0.949122 | **0.947937** | 1.055× | FAIL |

This failure is real under the frozen complete boundary.  It must not be turned
into a pass by changing the denominator after measurement.

However, the retained stage receipts identify what the failure means.  The SCC
query loop itself remains substantially faster than raw:

| Environment | SCC/raw query-stage geomean | Query-stage speedup | Paired range |
|---|---:|---:|---:|
| x86-64 | **0.399304** | **2.504×** | 0.394469–0.404834 |
| ARM64 | **0.412833** | **2.422×** | 0.408973–0.415504 |

The complete boundary is dominated by work shared by both exact arms:

- x86 raw query time is only about 10.7–11.6% of complete time;
- ARM raw query time is only about 8.4–9.0% of complete time;
- JSON ingestion, validation and exact-graph construction are charged afresh in
  every session and account for the remainder.

Consequently, even a hypothetical zero-time SCC query loop would have a
complete SCC/raw lower bound of approximately:

- `0.8837–0.8926` on x86; and
- `0.9095–0.9159` on ARM,

if all shared cold work remained unchanged.  The frozen `0.75` complete ratio
was therefore unattainable through tile64 query-engine optimization alone on
these two cases.  This diagnosis does not erase the failed gate; it identifies
that closing it requires reducing or amortising cold input/setup work, changing
the actual product endpoint, or predeclaring a different scope on fresh cases.

## Claim boundary

This audit supports the following narrow statement:

> On two disclosed, all-SAT WAP development banks of 1,024 restrictions each,
> the merged SPECTRA exact SCC bank endpoint returned and independently verified
> all full original-vertex witnesses while taking approximately 25–36% of the
> complete time of persistent MiniCard, persistent CaDiCaL and reduced native
> CUDD on CPython 3.13 CPU hosts.

It does **not** establish a new holdout, outside-team replication, universal SAT
or colouring dominance, one-query superiority, high-pin ITC superiority,
CPython 3.11 parity, or a production application trace.  The earlier frozen WAP
holdout remains the stronger untouched-instance result.  This development audit
adds current-main, cross-architecture and multi-competitor evidence.

## Retained evidence

Workflow run: `39069192983`

- x86 artifact `12114203731`, ZIP digest
  `sha256:004a779c97c96ad9a4de16944059dce36e116d09c682ee86f30279ae44e081d6`;
- ARM artifact `12113843750`, ZIP digest
  `sha256:d9ef2b0eb04f7f8018858fe6fdf411be647af362b97c08f29c0d76b4586c940e`.

Each artifact retains canonical inputs, native build receipts, all session
records and output banks, controller timings, independent audit output, frozen
gate output, environment and CPU receipts, source identities, graph acquisition
receipts and a SHA-256 manifest.
