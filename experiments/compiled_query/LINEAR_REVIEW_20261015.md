# Continuing the user's faster-queries checkpoint

**15 October 2026. Exposed synthetic development, not a new confirmation.**

## Starting point and preserved work

This continuation starts at the user's `faster-queries` commit
`098a8477f6cc8bb696423e78c4d6b6a26a95a0e1`, not the older prepared-transfer branch.
The user's automatic engine policy, prepared-root contradiction reconstruction,
direct SAT witness serialization, reusable per-call proof scratch, Windows/MSVC
support and seven-arm comparison are retained. The original `faster-queries`
branch is not rewritten. Main, PR #60 and PR #61 are not merged or replaced.

The user's latest completed seven-arm run, `38691730508`, already passes the
10% development target in all three measured environments. At 1,024 queries,
its auto/equally improved raw ratios are 0.885672 (x86 Python 3.11), 0.896982
(x86 Python 3.13), and 0.889297 (ARM64 Python 3.13). Its older failed target at
`2afaf00` remains historical evidence. This review does not take credit for the
user's improvements or treat that older failure as the current outcome.

## New implementation

Tested, published implementation: **`8d28a63707a1e3d56d5d16688924cd3985ee80d7`**.
Tree: `0753490153b4e68036e30be738f305eed96e57f8`.
Native source SHA-256:
`67a8b5c8c55dd4fdc5070f2308cabf37a463a7a1294f12a8dd496d792852fb25`.

The core delta is three files: `exact_native.cpp`,
`test_linear_construction.py`, and [the construction proof](LINEAR_CONSTRUCTION.md).
Two unnecessary global sorts are removed: the implication arc sort/deduplication
and the observer's forbidden-pair sort/deduplication. Both were redundant after
canonical original-edge admission. Every original observer constraint is retained.
The raw control gets the same applicable improvement as the optimized modes.

Stable forward and reverse CSR scattering now consumes the already ordered
per-vertex incident subsequences. Both resulting adjacencies are byte-for-byte
identical to the previous sorted construction, including singleton unit arcs.
A diagnostic-only native build actually constructs both versions and compares
all four CSR arrays and arc counts before SCC processing. It is never timed.
No ABI, packet format, resource cap, engine-selection threshold or default changes.

Implication emission plus both CSR builds are O(n+a) after canonical admission,
where a is the implication count. This is not a linear-time claim for the entire
routed constructor: downstream condensation and routing-bridge sorts remain.
It is an implementation specialization, not a new 2-SAT algorithm or a priority
claim. Canonical tuple normalization also retains its own cost.

## Matched complete-call results against the immediate user checkpoint

Baseline is **098a847**, not an earlier slower implementation. The existing
seven-arm harness, 48 exposed graph realizations, eight families, five bank sizes,
nine paired repetitions, two warmups, order permutations, process limits and
independent original-input replay are unchanged. Both versions use the same
compiler and options on each host. Raw, routed and automatic parent engines are
all retained; current raw, SCC, routed and automatic engines are all reported.

The timed boundary is ready immutable SBC1 input through input hashing, native
construction, all solving, complete witnesses/proofs, independent serialized-output
verification, SCQ2 creation, disposal, and output hashing. Packet preparation,
compilation, process startup, evidence writes and post-run Python replay are
outside it. Process wall time and RSS remain separate; this is not a production
latency or total-input-ingestion claim.

Ratios below are **new automatic-engine time / immediate-parent automatic-engine
time**. Lower is better. They aggregate per-graph median complete-call times
geometrically, rather than pooling individual requests as independent samples.

| Queries per prepared call | Local x86 / Python 3.13 | CI x86 / Python 3.11 | CI x86 / Python 3.13 | CI ARM64 / Python 3.13 |
|---|---:|---:|---:|---:|
| 1 | 0.547348 | 0.552124 | 0.621886 | 0.663518 |
| 63 | 0.692641 | 0.688031 | 0.730851 | 0.763819 |
| 64 | 0.665747 | 0.682101 | 0.717376 | 0.754774 |
| 65 | 0.682029 | 0.674030 | 0.723055 | 0.756958 |
| 1,024 | 0.922051 | 0.931766 | 0.960653 | 0.945146 |

The three CI environments therefore show **33.6-44.8% less single-request time**
and **3.9-6.8% less 1,024-request time** than the user's latest automatic engine.
The improvement is predominantly construction cost, not a new propagation rule.

| CI environment | New/parent auto at 1,024 | Stratified graph-bootstrap 95% interval | New auto / new raw at 1,024 | Existing development target |
|---|---:|---:|---:|---|
| x86 / Python 3.11 | 0.931766 | [0.913470, 0.948929] | 0.878810 | PASS |
| x86 / Python 3.13 | 0.960653 | [0.945421, 0.974489] | 0.897270 | PASS |
| ARM64 / Python 3.13 | 0.945146 | [0.930190, 0.958338] | 0.882249 | PASS |

The original target is not relaxed: long-bank auto/raw <= 0.90, its upper
95% bound < 1, every family <= 1.10, and single-request aggregate <= 1.10.
The local run also passes. These intervals describe this fixed exposed synthetic
panel, not a population of real applications or independently selected families.

Every environment completes **15,120 scheduled timed calls and 48 workers**.
Each audit independently checks **1,636 distinct packet/output pairs containing
484,546 decisions**; repeated outputs are deduplicated only in post-run replay.
Native verification still occurs on every call. Repetitions, environments and
prefixes must not be counted as independent problem instances.

A separate five-mode comparison across all 48 graphs and five lengths verifies
**1,200 complete outputs and all returned metadata identical to the parent**.
The production benchmark separately enforces same-mode raw/routed output equality
and automatic-output equality with its reported selected engine.

## Tests and independent replay

| Check | Observed outcome |
|---|---:|
| All compiled-query test modules locally, in recorded groups/fresh processes | 455 passed |
| Focused local subset, included in the 455 | 152 passed |
| Focused CI on x86 Python 3.11 and 3.13 and native ARM64 Python 3.13 | 152 passed on each |
| Actual Windows x64/MSVC, Python 3.13 | 156 passed |
| CI exact-service contracts under UBSan | 112 passed |
| CI exact-service contracts under ASan + UBSan | 112 passed |
| Downloaded final CI audits replayed using Python -S without a native solver | All three identical |
| Stable spectra/, tests/, results/ and data/ versus user checkpoint | Unchanged |

All seven jobs in workflow `38781025834` finished successfully. The completed
implementation is 8d28a637; its workflow trigger commit is 512ecc64. Each dependent
job checked out the actual published implementation, not the staging commit.
The local timing commit is c3b602f8; its executable/test subtree matches 8d28a637,
and both native source hashes are the value above. Their workflow-only ancestry
differs. Later workflow cleanup/reporting commits do not change that tested code.

ASan uses detect_leaks=0, so no LeakSanitizer result is claimed. Windows establishes
build and functional coverage here, not a Windows timing-panel result. Local
remaining compiler modules require actual native CUDD; they are not Python-only
substitutes. This step does not claim a rerun of the full model/training repository
suite. The stable public source directories are unchanged.

## Retained failures and limitations

An initial combined local compiled-query run ended without a final result. Its
partial log is retained and is not counted as passed. All modules were subsequently
run successfully in the recorded groups/fresh processes above, with JUnit results
and explicit exit statuses for the separately run groups and comparison workers.

The x86/Python 3.13 CI retains one single-request **new-auto/new-raw** ratio of
1.313129 on `percolated_grid-r4-n512-d1`. The local run retains two 63-request
new-auto/new-raw ratios above 1.25: 1.423691 on `geometric-r4-n512-d1` and 1.396579
on `random_bipartite-r3-n512-d0`. They are not regressions against the parent auto
engine; they still prevent a claim that automatic selection is never slower than
raw on individual measurements. None were removed or rerun until favorable.
No auto/parent-auto cell exceeds 1.25 in these four runs. This is not a universal
no-regression guarantee. Explicit routed mode still loses substantially to raw
on many short-call cells, which is why the user's automatic policy is preserved.

All exposed cases, all arms, every stored timing, per-family summary and slow
cell remain in the full audits/artifacts. No frozen prior speed failure has been
relabelled or retuned. There is no new outside workload, outside-team replication,
operator adoption, stable installed API promotion or algorithmic-priority result.

## Assessment and publication

A provisional whole-program review judgment is **78/100**, not a measured score.
The earlier 75 is improved primarily by the user's now-demonstrated platform
coverage and clean publication, with the new construction optimization adding
execution evidence. It is not a novelty-score increase for removing sorts.

| Category | Score |
|---|---:|
| Importance and focus | 9/12 |
| Correctness and contracts | 15/15 |
| Experimental rigor | 17/18 |
| Competitors and attribution | 14/15 |
| Novelty and scientific depth | 8/15 |
| Generalization and portability | 8/10 |
| Repository/release coherence | 7/8 |
| Independent replication and impact | 0/7 |

The code remains opt-in research. A versioned installed exact-query API and an
operator-defined evaluation would address larger remaining gaps than another
exposed-panel tuning cycle. The next prospective test must use a newly frozen
endpoint/protocol and new workload, not rename these already exposed 48 graphs.

This continuation is isolated on `research/faster-queries-review-20261015`, based
on `faster-queries`. The temporary source patch is removed and the publication
workflow is replaced by read-only validation. Main and the user's branch are
unchanged. No default promotion or merge is implied.

Completed cross-platform workflow:
https://github.com/tugrapaydiner/SPECTRA/actions/runs/38781025834

The recovery handoff contains immutable original CI ZIPs with their GitHub SHA-256
receipts, full local evidence, all three separate replay receipts, the 1,200-output
comparison, test logs (including the incomplete combined attempt), and the exact
published Git source. Hosted artifacts retain their stated 30-day lifetime; the
recovery archive is the durable copy.
