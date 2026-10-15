# SPECTRA: continuation from the user's faster-queries branch

**15 October 2026. Exposed synthetic development, not new confirmation.**

## Preserved starting point

User checkpoint: `098a8477f6cc8bb696423e78c4d6b6a26a95a0e1` on `faster-queries`.
This continuation preserves the user's automatic engine selection, prepared-root
proof reconstruction, direct witness serialization, reusable proof-checking
scratch, Windows/MSVC support and seven-arm comparison. The user's branch and
main are unchanged; earlier PRs are not merged or replaced.

The user's latest run, `38691730508`, already passes the 10% development target
on x86 Python 3.11, x86 Python 3.13 and ARM64 Python 3.13. Its long-bank auto/raw
ratios are 0.885672, 0.896982 and 0.889297 respectively. The older failed target
remains historical evidence. Those improvements belong to the user's checkpoint,
not to the new optimization reported below.

## New code

Published implementation: `8d28a63707a1e3d56d5d16688924cd3985ee80d7`.
Implementation tree: `0753490153b4e68036e30be738f305eed96e57f8`.
Native SHA-256: `67a8b5c8c55dd4fdc5070f2308cabf37a463a7a1294f12a8dd496d792852fb25`.
Branch: `research/faster-queries-review-20261015`.

Three core files change: `exact_native.cpp`, `test_linear_construction.py` and
`LINEAR_CONSTRUCTION.md`. Two redundant global sorts are removed: implication
arc sorting/deduplication and observer forbidden-pair sorting/deduplication.
The input paths already supply sorted, unique, canonical original edges.

Stable forward and reverse CSR construction consequently produces exactly the
same adjacency arrays as the old sorted implementation, including singleton unit
arcs. The proof is in `LINEAR_CONSTRUCTION.md`. Diagnostic-only native builds
construct both versions and compare all four arrays and the arc count before
SCC processing. The diagnostic macro is never enabled in benchmark builds.

Every original constraint remains in the independently owned observer. Raw mode
receives the same applicable improvement. No ABI, packet schema, memory cap,
automatic-selection threshold, default or public-package behavior changes.

Implication emission and both CSR builds take O(n+a) after canonical admission,
where a is the implication count. Downstream condensation/routing sorts and tuple
normalization still have their own costs. This is not a linear-time claim for the
whole routed constructor, a new 2-SAT algorithm, or an algorithmic-priority claim.

## Matched complete-call results

Baseline: the immediate user checkpoint **098a847**, not an older slower version.
The unchanged seven-arm schedule retains 48 exposed graph realizations, eight
families, five bank sizes, nine paired repetitions, two warmups, fixed order
permutations, process limits and original-input replay. Parent raw/route/auto and
current raw/SCC/route/auto are all retained, with matching compiler/options.

The timed boundary starts at ready immutable SBC1 input and includes input
hashing, native construction, solving, full witnesses/proofs, independent
serialized-output checking, SCQ2 production, native disposal and output hashing.
Packet preparation, compilation, process startup, evidence writes and post-run
Python replay are outside it. It is not a production end-to-end latency claim.

Ratios are **new automatic-engine time / parent automatic-engine time**; lower is
better. Aggregation uses graph medians and geometric means, not pooled queries.

| Queries/call | Local x86 Py3.13 | CI x86 Py3.11 | CI x86 Py3.13 | CI ARM64 Py3.13 |
|---|---:|---:|---:|---:|
| 1 | 0.547348 | 0.552124 | 0.621886 | 0.663518 |
| 63 | 0.692583 | 0.688031 | 0.730851 | 0.763819 |
| 64 | 0.665698 | 0.682101 | 0.717376 | 0.754774 |
| 65 | 0.681992 | 0.674030 | 0.723055 | 0.756958 |
| 1,024 | 0.922051 | 0.931766 | 0.960653 | 0.945146 |

CI therefore shows **33.6-44.8% less single-request time** and **3.9-6.8% less
1,024-request time** against the user's latest auto engine. Construction cost,
not a new propagation rule, accounts for the change being tested.

| CI environment | New/parent auto, 1,024 | Graph-bootstrap 95% interval | New auto/new raw, 1,024 | Development target |
|---|---:|---:|---:|---|
| x86 Py3.11 | 0.931766 | [0.913470, 0.948929] | 0.878810 | PASS |
| x86 Py3.13 | 0.960653 | [0.945421, 0.974489] | 0.897270 | PASS |
| ARM64 Py3.13 | 0.945146 | [0.930190, 0.958338] | 0.882249 | PASS |

The unchanged target requires long-bank auto/raw <= 0.90, its upper 95% bound < 1,
every family <= 1.10, and single-request aggregate <= 1.10. Local also passes.
Intervals use the existing 4,000 family-stratified graph resamples and describe
this fixed exposed synthetic panel, not real-world application populations.

Each environment completes **15,120 timed calls and 48 workers**. Each audit
checks **1,636 distinct packet/output pairs containing 484,546 decisions**.
Only post-run replay deduplicates repeated outputs; native checking still occurs
on every call. Repetitions, prefixes and environments are not independent problems.

A separate comparison of all five modes, 48 graphs and five lengths confirms
**1,200 entire outputs and all returned metadata identical to the parent**.
The timing audit also enforces same-mode raw/route byte equality and auto equality
with its reported selected engine.

## Verification

| Check | Result |
|---|---:|
| All local compiled-query modules, in recorded groups/fresh processes | 455 passed |
| Focused local subset, included above | 152 passed |
| Focused CI: x86 Py3.11, x86 Py3.13, ARM64 Py3.13 | 152 passed each |
| Windows x64/MSVC Py3.13 | 156 passed |
| CI exact contracts under UBSan | 112 passed |
| CI exact contracts under ASan + UBSan | 112 passed |
| Three downloaded CI audits replayed with Python -S, no native solver | All identical |
| Stable spectra/, tests/, results/, data/ against user checkpoint | Unchanged |

All seven jobs in run **38781025834** completed successfully. Its trigger commit
is 512ecc64; the dependent jobs check out actual published implementation 8d28a637.
Local timing commit c3b602f8 has the same executable/test subtree and native hash;
its workflow-only ancestry differs. Later reporting and read-only workflow cleanup
commits do not alter tested native code.

ASan used detect_leaks=0: no LeakSanitizer claim. Windows is functional/build
coverage, not a Windows timing panel. The remaining local compiler modules used
actual native CUDD. This step does not claim a rerun of the entire model/training
repository suite or promote the experiment into the stable installed API.

## Negative observations retained

An initial combined local compiled-query run ended without a final result. Its
partial log is retained and not counted as a pass. All modules subsequently passed
in recorded groups/fresh processes, with JUnit results. Separately run module and
benchmark/replay controllers retain their explicit exit statuses.

The x86/Py3.13 CI has one single-request **new-auto/new-raw** ratio of 1.313129 on
`percolated_grid-r4-n512-d1`. Local retains two 63-request new-auto/new-raw ratios
above 1.25: 1.423691 on `geometric-r4-n512-d1` and 1.396579 on
`random_bipartite-r3-n512-d0`. These are not regressions against parent auto, but
rule out a claim that auto is never slower than raw on individual measurements.
None were dropped or retried until favorable. No auto/parent-auto cell exceeds
1.25 in the four runs; this is not a universal no-regression guarantee.

Explicit route remains substantially slower than raw on many short-call cells.
The user's automatic policy, all seven arms, all slow cells and per-family
results are retained. No historical frozen speed failure is reclassified or
retuned. No new holdout, external workload, outside-team replication, adoption,
priority determination, universal memory bound or production latency is claimed.

## Assessment and next gap

My provisional whole-program judgment is **78/100**, not a measured statistic.
Relative to the earlier 75, credit comes primarily from the user's demonstrated
platform coverage and publication, plus stronger current execution evidence.
Removing redundant sorts does not increase the novelty score.

Rubric: importance 9/12; correctness 15/15; experimental rigor 17/18;
comparators/attribution 14/15; novelty/depth 8/15; generalization/portability 8/10;
repository/release coherence 7/8; independent replication/impact 0/7.

The next larger gaps are a versioned installed exact-query API and an
operator-defined evaluation. Any new prospective test needs a newly frozen
endpoint/protocol and new workload, not a relabeling of these exposed 48 graphs.

Publication is isolated from the user's branch. Temporary patch transport is
removed and the retained workflow is read-only. No merge/default promotion is
implied. Completed workflow:
https://github.com/tugrapaydiner/SPECTRA/actions/runs/38781025834

The recovery handoff retains original CI ZIPs with GitHub SHA-256 receipts, full
local raw evidence, three replay receipts, the five-mode equivalence comparison,
test logs including the incomplete attempt, and exact published Git source.
Hosted artifacts have their stated 30-day retention; the handoff is the durable copy.
