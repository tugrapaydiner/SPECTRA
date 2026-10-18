# Automatic engine selection v6 — frozen protocol

## Why v6 exists

Structural-v5 passed every aggregate development criterion and both real-workload
guards, but failed the frozen per-cell ceiling on one micro-latency cell:

- 720 development cells, 17 paired repetitions;
- global realised-regret geometric mean: `1.002645` — pass;
- global selection-regret geometric mean: `1.002622` — pass;
- worst family/count geometric mean: `1.017768` — pass;
- cells within 5% of the explicit oracle: `99.17%` — pass;
- worst realised regret: `1.156867` — fail versus `1.10`.

The failing cell was `preferential/density-low-large`, 1,536 vertices and eight
requests. Auto selected raw, raw was the explicit oracle, and structural
selection regret was exactly `1.0`; the red value came from paying the generic
automatic path and from independent microsecond-scale medians. The complete v5
failure remains retained as workflow `39093492099`, artifact `12123703350`.
No threshold is changed or reinterpreted.

Across the complete exposed 720-cell panel, every eight-request cell above 512
vertices already selects raw under v5, and no such cell has SCC as the selected
structural engine. The real WAP guard starts at 64 requests. The real ITC guard
uses 1,024 requests. The fresh strong-compression confirmation case has 320
vertices. Therefore a direct header-only raw dispatch for this exact region is a
query-blind implementation optimization rather than a timing-derived answer
cache.

## Frozen structural-v6 policy

Before native index construction, the public `ExactRuntime.solve_session_packet`
reads only the validated SBC1 header geometry. If the requested mode is `auto`,
the packet has SBC1 geometry, the bank has at most eight requests, and the graph
has more than 512 original vertices, it invokes the native **explicit raw** mode
and rewrites only the returned dispatch metadata to:

- `auto_policy = structural-v6`;
- `selected_mode = raw`;
- `auto_policy_reason = large_tiny_bank`.

The same rule is present in the native policy for callers that invoke native auto
directly. The native raw solver, observer, complete labels/proofs, SCQ2 bytes,
resource accounting and disposal path are otherwise identical to explicit raw.
Malformed or resource-invalid packets are still rejected by the native raw
entry point; header inspection does not accept or repair bytes.

For all other banks, v6 retains the v5 order:

1. base-infeasible fixed graph: `raw`;
2. at least eight pins per request and average pin density at least `1/24`:
   `raw`;
3. at most 128 requests, no SCC contraction, and at least four original edges
   per vertex: `raw`;
4. feasible bank of at least 64 requests: `scc`;
5. feasible bank of at least eight requests whose SCC count is at most one
   quarter of the original literal count: `scc`;
6. otherwise: `raw`.

Selection uses immutable packet/graph/request geometry only. It may not inspect
answers, trial solves, measured timings, proof paths, competitor output or cached
query results. Automatic route selection remains disabled.

## Unchanged contracts

Exact API 1, native ABI 3, SBC1/SBQ1/SCQ2/SBP1 bytes, explicit raw/SCC/route
semantics, complete original assignments, portable contradiction proofs,
independent replay, resource limits and the public API shape remain unchanged.
The only added public behaviour is an exact raw fast dispatch for the frozen
large-tiny region and a versioned reason string.

## Development gate

The exact v5 development panel is rerun from scratch on x86-64 with **31 paired
repetitions** after two warmups. Increasing repetitions strengthens the noisy
micro-cell check; no threshold is relaxed.

Acceptance remains:

- global realised-regret geometric mean `<= 1.025`;
- global selection-regret geometric mean `<= 1.015`;
- every family/count geometric mean `<= 1.05`;
- at least 97% of cells within 5% of the explicit oracle;
- every cell `<= 1.10`;
- all structural rules exactly followed;
- zero correctness, replay, timeout, crash or completeness failure.

The real guards are rerun on the final v6 source:

- WAP05/WAP06 at 64, 128 and 1,024 requests on x86-64 and ARM64; all select SCC;
  1,024-request auto/raw geometric mean `<= 0.75`, every graph `<= 0.80`;
- ITC set9/set10/set12 at 1,024 requests on x86-64 and ARM64; all select raw for
  `dense_pin_fraction`, every paired auto/raw median `<= 1.05`.

## Fresh prospective confirmation

Only after v6 development and both real guards pass may cases be generated under
the new namespace `SPECTRA-auto-selector-v6-confirmation`. The unopened v5
namespace is retired unused and may not be reused.

The same query-blind topology/regime design frozen for v5 is retained:

1. strong two-colour contraction with sparse pins;
2. pin density just below `1/24`;
3. pin density just above `1/24`;
4. high-edge, uncompressed graphs exercising the 64/128-request raw guard;
5. large graphs with high absolute pins but low relative density; and
6. prospectively base-infeasible graphs.

Eight topology families are crossed with counts `1, 8, 32, 64, 128, 1024`.
Each cell runs explicit raw, SCC and route plus auto for **31 paired repetitions**
after two warmups on GitHub x86-64 and native ARM64.

Confirmation acceptance, independently on each architecture, remains:

- global realised-regret geometric mean `<= 1.04`;
- global selection-regret geometric mean `<= 1.025`;
- every family/count geometric mean `<= 1.06`;
- at least 95% of cells within 5% of the explicit oracle;
- every cell `<= 1.10`;
- no family aggregate more than 3% slower than raw;
- all structural rules exactly followed; and
- zero incorrect decision, invalid witness/proof, incomplete cell, timeout or
  crash.

Every workflow uses `set -euo pipefail`, uploads failed evidence, and rereads the
machine result to enforce `summary.pass`.

## Product and repository gates

Before merge, the final source must also pass:

- Issue 1 first-use SCC/raw gates on x86-64 and ARM64;
- WAP flagship regression;
- exact API on Linux x86-64, native ARM64 and Windows/MSVC;
- Python 3.11/3.13 installed APIs and compiler-free wheels;
- UBSan and ASan+UBSan;
- public-package, full CPU, repository and SVM portability matrices.

The exploratory branch is archived, then the successful change is rewritten as
one human-readable commit based directly on `main`. Temporary workflows and
v3/v4/v5 helper scripts do not enter the active tree.

## Interpretation

A pass establishes a reliable dispatcher for the tested exact-query scope and
removes the remaining measured auto overhead in a structurally obvious tiny-bank
region. It does not establish universal SAT/colouring dominance, outside-team
replication, operator adoption, production p99 behaviour or one-query leadership
against every external solver.
