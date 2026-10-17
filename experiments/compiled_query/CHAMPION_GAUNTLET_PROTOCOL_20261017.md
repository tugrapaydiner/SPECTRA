# Strong-competitor development gauntlet — frozen protocol

**Status:** exposed development diagnosis. This is not a new WAP holdout, an
independent replication, or permission to advertise universal solver leadership.
All losses, timeouts, crashes, unsupported contracts, and incomplete cells remain
in the evidence and in any generated chart.

## Question

For exact repeated feasibility queries on a fixed simple conflict graph whose
original vertices have one- or two-colour domains, where does SPECTRA currently
win and lose against strong current general and specialized alternatives when the
entire checked session is charged?

The study is deliberately split into two classes.

1. **Persistent/compile-once repeated-query engines.** The fixed formula or
   compiled representation is built once and all current restrictions are issued
   against that owner. This class contains SPECTRA raw/SCC/route, current CaDiCaL,
   MiniCard, the retained CaDiCaL 1.9.5 control, reduced CUDD, and OR-Tools CP-SAT
   with one retained model and a fresh single-worker search per query.
2. **Rebuild-per-query solver cores.** Kissat does not expose an incremental
   assumption interface. Its original CNF plus current query units are rebuilt for
   every request. It is reported separately and is never described as a fair
   persistent-service substitute.

The direct specialized threat is SPECTRA `exact-raw`, not a broad SAT solver. A
large win over a general solver and a tie with raw traversal is reported as a tie
with the direct threat, not as field dominance.

## Frozen external implementations

- **CaDiCaL 3.0.1:** `arminbiere/cadical` commit
  `c60730422e758ef1cebe7aeddf2dda31c996bf04`, shared C API. Every original
  variable is frozen before the first solve, and learned clauses remain across
  assumption calls.
- **Kissat 4.0.4:** `arminbiere/kissat` commit
  `8af8e56f174b778aef3aa45af9f739b2a5f492c2`, shared C API, rebuilt per query.
- **OR-Tools CP-SAT 9.15.6755:** retained Boolean model, one worker, seed zero,
  interleaved search disabled, fresh `solve()` per current assumption bank.
- **PySAT 1.9.dev15:** persistent MiniCard and retained CaDiCaL 1.9.5 controls.
- **CUDD:** independently maintained `tulip-control/dd` commit
  `4442d498ddea1bec0071315f538cb5ef37443985`, generic reduction followed by a
  fixed BFS variable order.

Build receipts bind source commit, compiler identity, command, and shared-library
SHA-256. The immutable study copies the exact libraries and rejects changed bytes.

The 2026 SAT Competition winner `satsuma-iter+kissat` is not silently treated as
incremental. It belongs in a later cold/rebuild track using the official competition
`build.sh`/`run.sh`, because changing its interface would no longer test the
submitted solver. Current Kissat is the first reproducible strong one-shot core in
this development matrix.

## Frozen development cases

Six already exposed deterministic structural cases are generated without consulting
any solver or answer:

| Family | Replica | Vertices | Domain regime |
|---|---:|---:|---|
| random tree | 0 | 128 | two colours |
| geometric | 1 | 128 | two colours from palette 8 |
| preferential attachment | 2 | 128 | mixed singleton/two-colour |
| rewired ring | 3 | 512 | two colours |
| communities | 4 | 512 | two colours from palette 8 |
| random bipartite | 5 | 512 | mixed singleton/two-colour |

Each case contains 1,024 deterministic unfiltered requests. Empty intersections,
SAT requests, and structural contradictions are retained. No request is repaired,
rejected, or selected by any candidate answer.

## Schedule and measurement

- Request-bank lengths: **1, 64, 1,024**.
- Paired repetitions: **3**: forward, reverse, and one fixed shuffled order.
- Arm order is shuffled inside each paired group from seed `20261017`.
- One case is one independent descriptive cluster. Queries and repetitions are not
  counted as independent graphs.
- Per-session wall deadline: **120 seconds**.
- Address-space limit: **3 GiB**; native admission bound: **512 MiB**.
- CUDD node limit: **2,000,000**.

`complete_ns` includes immutable JSON input reading/validation, solver or compiled
owner construction, every query, complete original-label materialization, one
separately owned native original-input check for every SAT witness, output-bank
construction and hashing, diagnostics, and disposal. Source checkout, native build,
case generation, process startup, and the post-run Python replay are separate.

Every non-SAT decision from a generic engine is independently recomputed from the
original graph, domains, and exact current query after the timed call. Generic
engines do not receive SPECTRA's portable UNSAT-certificate label. SPECTRA exact
arms produce and check their ordinary portable proof contract.

## Reporting

For every fixed SPECTRA mode and every competitor, report:

- geometric mean of within-case complete-time ratios;
- per-case ratios;
- wins below 0.99, ties in [0.99, 1.01], and losses above 1.01;
- all incomplete/error cells;
- timing-stage decomposition;
- output/certificate bytes and solver metadata.

The primary public diagnostic chart uses the predeclared `exact-scc` mode. A second
chart may show all fixed SPECTRA modes. A post-hoc best-of-raw/SCC/route number may
appear only under the explicit label **oracle diagnostic — not deployable**.

No `world-leading`, `beats SAT`, or `fastest` statement is allowed from this panel.
A future dominance claim requires a separately committed final policy, untouched
application families, a current SAT-competition cold track, peak-RSS and energy
measurements, and outside-team reproduction.
