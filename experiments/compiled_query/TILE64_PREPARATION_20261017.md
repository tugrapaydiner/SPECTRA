# Tile64 request-preparation checkpoint — 17 October 2026

## Verdict

The first targeted performance issue is resolved on the exposed 48-graph structural development panel: the explicit SCC tile64 engine now beats the matched optimized raw implication engine by more than 10% on every base-satisfiable 1,024-request graph on both Linux x86-64 and native Linux ARM64.

This is not a universal solver claim, a fresh holdout, an automatic-selector result, or external replication. The 33 base-infeasible graphs remain expected raw/SCC ties, and structural auto policy already chooses raw for those graphs.

## Scope

This checkpoint changes one thing only: request preparation inside the existing 64-lane SCC bank. It does not change API 1, native ABI 3, SBQ1/SBC1/SCQ2/SBP1 formats, proof rules, default selection, the frozen WAP result, or the raw engine.

## Implementation

The prior path allocated and copied one `vector<Pin>` per lane, sorted all pins, built a vector-of-vectors for roots, and repeatedly allocated tile scratch. The new path:

1. reads pins directly from the independently parsed immutable packet;
2. intersects duplicates with generation-stamped per-vertex scratch;
3. sorts only unique touched vertices, preserving scalar root/proof order;
4. stores roots in one flat array plus 65 offsets;
5. reuses reached, conflict, assignment, replay and parent buffers; and
6. creates a per-lane root vector only when an UNSAT proof must be replayed.

Every call still emits all original labels or portable contradiction proofs, runs the separately owned native observer, serializes complete SCQ2, and returns immutable Python bytes.

## Correctness

- Packaged and research native sources are byte-identical at SHA-256 `570c5fe9204342535adb766d01a923001c49881e31b7bd2da42fc972a27c73d3`.
- A 65-row duplicate-pin bank exercises two tile64 blocks, SAT, path contradictions, empty domains, and noncanonical pin order.
- Every bank row is byte-compared with scalar SCC output.
- Persistent SBQ1 and one-call SBC1 paths produce byte-identical SCQ2.
- Candidate and previous-main SCC outputs were byte-identical on all 48 panel graphs locally.
- Both CI environments independently replayed every raw and SCC output from original inputs.
- No decision disagreement, invalid witness, invalid proof, crash, or timeout occurred.

## Cross-architecture paired result

Each environment ran all 48 fixed graphs. Raw and SCC sessions were prepared once; five warmups preceded 31 randomized paired complete-bank calls per engine and graph. Timings include request-bank parsing, solving, all witnesses/proofs, independent native checking, SCQ2 construction, and immutable Python-byte ownership. Session construction is outside this prepared-bank timer.

| Quantity | x86-64 / Python 3.13 | ARM64 / Python 3.13 |
|---|---:|---:|
| Graphs completed and replayed | 48/48 | 48/48 |
| Base-satisfiable speed-gate graphs | 15 | 15 |
| Base-infeasible expected ties | 33 | 33 |
| Feasible SCC/raw geometric mean | **0.553647** | **0.529067** |
| Graph-bootstrap upper 95% ratio | **0.637479** | **0.594739** |
| Worst feasible SCC/raw ratio | **0.855100** | **0.778360** |
| Feasible graphs below 0.90 | **15/15** | **15/15** |
| Base-infeasible geometric ratio | 0.997918 | 1.001015 |

Worst feasible x86 case: `preferential-r2-n128-d2` at `0.855100`.  
Worst feasible ARM64 case: `preferential-r2-n128-d2` at `0.778360`.

Thus the weakest feasible graph retains at least a **14.5%** complete-call advantage over raw across the two CI architectures. The geometric-mean advantage is approximately **44.6%** on x86 and **47.1%** on ARM64.

## Evidence identities

- Base `main`: `65f9981a90d0b0c8fb4a7ce9f4c1fda71aaaeffd`
- Measurement commit: `e4793d2834934966e18304453126c5230e1d6624`
- Workflow run: `39064845872`
- x86 artifact: `12112812502` — `sha256:4e1085578a501fc71f6692811470575367671d8c63f85a9b2f5d5f692be60524`
- ARM64 artifact: `12112837344` — `sha256:d6311d41439f88ea95147ca598643935b06b2d7e80a976452d82222e55c65c31`

The artifacts retain every timing sample, all case-level ratios, output hashes, build receipt, CPU/platform identity, source/tree identities, logs, and SHA-256 manifests.

## Gate

The fixed development gate passes on both architectures:

- all 48 graphs complete and independently replay;
- every one of the 15 base-satisfiable graphs has SCC/raw median ratio below 0.90; and
- the graph-bootstrap upper 95% geometric-mean ratio is below 0.75.

## Claim boundary and next issue

This resolves **Issue 1: explicit SCC tile64 versus matched raw on feasible long banks**. It does not show that auto selects SCC correctly, that short/cold calls beat raw, or that SPECTRA beats every external solver. The next issue must be separate: improve and freeze structural auto selection so qualifying banks receive this winning SCC path while high-pin/base-infeasible/short banks remain on raw.
