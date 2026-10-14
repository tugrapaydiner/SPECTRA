# Exact mixed outcomes, with independently checked proofs

**13 October 2026, America/Toronto. Opt-in development checkpoint.**

SPECTRA now has an exact binary-domain support service that returns a complete
checked colouring or a checked contradiction proof. This closes the observed
mixed-outcome completeness gap without relabelling the legacy API's `UNKNOWN`
answers. It is a functional and verification advance, not a large speed or
algorithmic-novelty breakthrough. The default solver and frozen WAP result stay
unchanged; draft PR #60 is not a release or merge approval.

## Source identity and implementation

Starting checkpoint: `19a423adb8ede4e830476112046886bcc6b924a8`.
The first exact service is `5ec4e57b6cb618969c1b211375fd99b745691f77`.
The final routed implementation is **`1556309acf130b9091c5980d8d1aed48de6d32b8`**,
with source tree **`469906fd28c7ee32970cbfbd3ee92fd1165d54eb`**.
Final native source SHA-256 is
`1047e34e42a829fe2a6a700d4d83673185e1946836c75c09fbb1e84059d117c6`.

Both commits are on `research/compiled-query-validation`. The final code delta
from the starting checkpoint is 14 files, 2,043 insertions and 29 deletions,
before this reporting-only addition. These counts are not research-quality metrics.
The independently merged migration core on `main` is not silently imported into
this branch or confused with this different feasibility contract.

The implementation contains:

- `exact_native.cpp` and `exact.py`: immutable native solver and separately owned
  original-input observer, explicit construction, bounded admission and lifecycle.
- `proof.py`: standard-library original-input verification of SBP1 certificates.
  `proof_replay.py` retains only validated original data for efficient post-run
  checking; it never imports a candidate index or trusts a saved answer.
- Four named controls: raw original-graph traversal, SCC propagation, sparse proof
  traversal, and direct routed proofs. They share the same sparse request preparation,
  complete answer contract, native observer and output accounting.
- Strict SCQ2 proof-carrying output, isolated worker receipts, source/binary hashes,
  independent replay, unfiltered requests and a common-mask projection for fair
  comparisons inside every included control's input domain.

The [mathematical and transport contract](EXACT_BINARY_CONTRACT.md) specifies the
reduction, extension argument, proof rules, ownership and limitations. It is not a
machine-checked proof of the C++ implementation, compiler or hardware.

## What the new service guarantees

The admitted input has a fixed simple conflict graph, one or two original colours
per vertex, and current original-vertex allowed-colour masks. Duplicate pins are
intersected. Unused mask bits cannot add colours. Every pin is validated, even
after an earlier contradiction. Invalid input and exhausted resources are errors,
not mathematical UNSAT decisions. Wider original domains are outside this API.

For every successfully completed admitted call, the result is either
`SAT_VERIFIED`, with every original vertex labelled, or `UNSAT_VERIFIED`, with a
portable original-constraint proof. The three proof kinds cover an empty effective
domain, two current-forced implication paths ending at opposite literals, and a
base contradiction with paths from a literal to its complement and back.
The in-call observer checks the original graph, domains and current request,
not the solver's SCC map, propagation flags or proof-routing index.

The `route` mode removes the second proof-graph BFS. It retains inward/outward
original-edge trees inside each SCC and an original representative for each DAG
edge. Current propagation predecessors connect these into original-literal paths.
The separate observer still holds and checks the original graph. No original
adjacency in the solver does not mean no retained proof information or zero memory.

Tree walks, ancestor-chain sorting and potentially longer certificates remain real
costs. This implementation is neither shortest-proof nor guaranteed output-linear
routing. All four exact modes receive the same request-preparation improvements;
the raw control is not intentionally left on a slower preparation path.

A native ABI hazard found during review was fixed before publication: ABI 2 also
uses `solver.v2` and `observer.v2` capsule names. Foreign ABI-1 handles are rejected
before dereferencing. Regression tests exercise that boundary with invalid addresses.

## Unfiltered requests, not accepted-only answers

The primary common-mask WAP bank contains **2,048 request positions** on two already
exposed graph/list instances. It has **1,206 SAT and 842 UNSAT** decisions. Of the
842 contradictions, **644 require implication-path evidence** and 198 have an empty
effective domain. Routed certificates have median size **60 bytes** and maximum
**112 bytes** in this bank. These are development observations, not universal bounds.

The two graphs have 905 and 947 vertices, and 43,081 and 43,571 original edges.
The generated lists and policy are inherited. New requests are generated without
solver-based rejection, repair or selection. There is no new topology holdout.

The wide-mask robustness bank includes bit 63 outside the declared palette. Older
quotient-based checkers reject those inputs. Their retained errors are API-domain
mismatches, not false mathematical answers or numerical speedup denominators.
`common_masks.py` removes only unused palette bits: every pin's intersection with
its original domain, every query, and the query order are preserved. No difficult
request disappears. The wide and common banks are mathematically equivalent and
must not be counted as independent evidence.

Each final CI environment ran all of the following pre-scheduled development cells:

| Bank | Scheduled sessions | Complete | Incomplete | API errors |
|---|---:|---:|---:|---:|
| Five synthetic structures, 1 and 128 requests | 270 | 240 | 30 | 0 |
| Previously SAT-filtered WAP, 1 and 1,024 requests | 108 | 108 | 0 | 0 |
| Wide-mask mixed WAP, 1 and 1,024 requests | 108 | 78 | 0 | 30 |
| Common-mask mixed WAP, 1 and 1,024 requests | 108 | 96 | 12 | 0 |

Every scheduled exact-mode session completed. The old arms retain **942 UNKNOWN
responses** on the synthetic bank and **5,052 UNKNOWN responses** on the common WAP
bank, counted across their repeated executions. The new exact modes do not silently
change those older results. The synthetic bank has 640 request positions: 483 SAT
and 157 UNSAT, of which 122 require implication paths and 35 have empty domains.

For the common bank alone, each of local Python 3.13, GitHub Python 3.11 and GitHub
Python 3.13 has **24,600 checked exact-mode decisions: 14,496 SAT and 10,104 UNSAT**.
This includes four implementations, three repetitions and overlapping prefixes;
it is not 24,600 independent problems. All 842 common-bank contradictions are
verified by each exact mode, with additional independent Python replay.

## Complete-cost results: modest against the cheapest exact control

The following are geometric means of within-graph mean complete-session ratios,
**routed exact service time / comparator time**, for 1,024 common-mask requests.
Lower is better for the routed service. Three paired repetitions use forward,
reverse and shuffled request orders. These are descriptive development ratios,
not confidence bounds, population estimates or outside replication.

| Comparator | Local Python 3.13 | GitHub Python 3.11 | GitHub Python 3.13 |
|---|---:|---:|---:|
| Matched exact raw traversal | 0.941200 | 0.980156 | 0.983952 |
| Matched exact SCC + proof search | 0.952353 | 1.003389 | 0.998482 |
| Persistent MiniCard | 0.602160 | 0.491429 | 0.517295 |
| Persistent CaDiCaL | 0.549184 | 0.498716 | 0.511640 |
| Reduced native CUDD, BFS order | 0.629284 | 0.420726 | 0.440983 |
| Legacy SPECTRA SCC / noncontracting arms | Incomplete | Incomplete | Incomplete |

Only comparisons among the four exact modes have the same portable-certificate
contract. SAT and CUDD controls return exact decisions and checked SAT witnesses,
but do not generate the new portable UNSAT proofs. This difference is explicitly
marked in the machine-readable audit. The current mixed matrix includes reduced
BFS CUDD; the earlier study of multiple CUDD orders remains separate evidence.

The common-bank benefit over matched raw traversal is **5.88% locally and only
1.98% / 1.60% on the GitHub runners**. Direct routing does not consistently beat
the existing exact SCC implementation. On the synthetic long bank, routed/raw
ratios are 0.986236 and 1.010087 on Python 3.11 and 3.13: no robust speed win.
This checkpoint closes completeness and proof boundaries, not a major execution
advantage over the strongest cheap exact method.

The full session includes input decoding, validation and hashing; solver and
independent-observer construction; every query, witness/proof construction and
check; SCQ2 output materialization and hashing; diagnostics and disposal. Import,
native build and post-run Python audit are outside the warm endpoint. Cold process
wall cost is separately retained. The native observer is never sampled or omitted.
With only three repetitions, reported p95 is the maximum order statistic, not a
well-estimated production tail.

### Negative results that remain important

On the historically SAT-only WAP bank, routed service / older SAT-focused SCC
ratios are **1.256240 on Python 3.11** and **1.285807 on Python 3.13** at 1,024 queries.
That is a 25.6% / 28.6% regression, not a reason to redefine the endpoint. At one
common-bank query, the ratios to the old SCC endpoint are 1.641455 and 1.752092.
The exact companion service is therefore not promoted to the default.

Proof routing does not universally reduce retained memory either. On WAP05,
raw and routed solver arrays are 33,631 and 34,159 bytes; on WAP06 they are 35,277
and 35,621 bytes. Both additionally require the same original observer, 353,698
and 358,038 bytes respectively. A dense synthetic graph shows larger solver-array
savings, but that does not transfer to every graph or imply lower process RSS.

The first exact-service Python 3.11 CI job was cancelled during its slow final
post-run audit. All worker trials had finished; their unchanged artifacts were
retained. The final prepared-original replay audited those bytes successfully,
without remeasuring them. Its original missing audit stays missing in the archive,
and the recovery audit is separately named. The final routed CI run passed both
Python versions with the improved audit. This does not erase the initial cancelled
job or the older frozen Python 3.11 strict-performance-gate failure.

## Verification and replay

| Check on final implementation | Observed result |
|---|---|
| Local Python 3.13 compiler/service/harness/replay tests | 338 passed |
| GitHub Python 3.11, actual native CUDD required | 338 passed |
| GitHub Python 3.13, actual native CUDD required | 338 passed |
| Changed native contracts under local UBSan | 58 passed |
| Changed native contracts under local ASan + UBSan | 58 passed |
| Same two sanitizer configurations on GitHub | 58 passed each |
| Release/navigation audit, local and both CI versions | PASS |
| Protected historical files | 449 byte-identical |
| Downloaded final CI audit replay | All eight audits matched exactly |
| Existing PR workflows for implementation commit `1556309` | All 13 passed |

ASan used `detect_leaks=0` and preloaded `libasan` plus `libstdc++`. This is not a
LeakSanitizer claim. Tests include all three-vertex labelled graphs with all
one/two-colour domains from three colours and all effective-domain restrictions,
random differential MiniCard checks, 10,001-vertex paths and odd cycles, singleton
and bit-63 cases, malformed later pins, stale/forged proofs, lifecycle, concurrency,
ABI handles and injected observer failure. Four modes execute 186,624 exhaustive
small original-domain queries; they are not independent benchmark samples.

All final study source hashes match commit `1556309`; common input bytes match
across the three measured environments. The final CI uses Python 3.11.17 and
3.13.16 on AMD EPYC 7763 runners; local execution uses Python 3.13.5. Project-owned
CI and container replays do not count as independent-team scientific replication.
The stable public CPU target has its own PR workflows; this report does not turn
an earlier memory-limited unfiltered local suite into a passing full-suite claim.

Final development and sanitizer workflow:
https://github.com/tugrapaydiner/SPECTRA/actions/runs/38502087775

First exact milestone, including the cancelled Python 3.11 audit:
https://github.com/tugrapaydiner/SPECTRA/actions/runs/38499063421

Original artifact SHA-256 receipts, full audits, source/build identities, logs and
all negative cells are preserved in the recovery archive. Actions retention is
30 days; the archive is the durable handoff, not a promise about hosted retention.

## Reproduce or use the exact companion

Use Linux, a C++17 compiler and the pinned native dependencies in
[the workflow](../../.github/workflows/exact-binary-development.yml). From the
repository root, a minimal proof example needs no SAT or CUDD backend:

```python
from pathlib import Path
from experiments.compiled_query.binary_list import BinaryListProblem
from experiments.compiled_query.exact import ExactRuntime, build_exact_runtime
from experiments.compiled_query.proof import verify_answer

library = build_exact_runtime(Path("my-exact-build"))  # fresh directory
runtime = ExactRuntime(library)
problem = BinaryListProblem.create(2, 2, ((0, 1),), (3, 3))
request = ((0, 1), (1, 1))  # adjacent vertices both required to use colour 0
with runtime.prepare(problem, mode="route") as service:
    answer = service.solve(request)
    assert answer.status == "UNSAT_VERIFIED"
    assert verify_answer(problem, request, answer.status, answer.labels, answer.proof)
```

Run the complete native-backed contracts with
`SPECTRA_REQUIRE_CUDD=1 python -m pytest experiments/compiled_query -q`.
Build an explicit quotient library and exact library before freezing a comparative
study; the committed workflow supplies all four input banks and exact commands.
A successful evidence audit does not assert that every old arm completed or that
the candidate won. Study resumption requires identical source, inputs, binaries
and environment; failed cells are not silently retried until they succeed.

## Harsh reassessment and next gates

A provisional whole-program judgment is **about 71/100**, up from the original
67 baseline across the CUDD and exact-service checkpoints. This is an evaluative
rubric, not a measured score, confidence interval or hiring prediction.

| Category | Earlier | Now | Reason |
|---|---:|---:|---|
| Importance and focus / 12 | 9 | 9 | Useful, still narrow repeated-query service. |
| Correctness and contracts / 15 | 14 | 15 | Exact mixed outcomes and checked original-input contradictions. |
| Experimental rigor / 18 | 16 | 16 | Strong receipts and preserved failures, but exposed development data. |
| Competitors and attribution / 15 | 10 | 12 | Working reduced CUDD and matched proof-carrying raw control. |
| Novelty and scientific depth / 15 | 7 | 7 | Established 2-SAT, closure and SCC-tree techniques. |
| Generalization and portability / 10 | 5 | 6 | Mixed requests and both Python versions; no broad deployment evidence. |
| Repository/release coherence / 8 | 6 | 6 | Opt-in branch and preserved history; integration/identity work remains. |
| Independent replication and impact / 7 | 0 | 0 | No outside-team reproduction or adoption. |

The next engineering gate is a substantial complete-cost advantage over
`exact-raw`, not merely over general solvers. First isolate full-output checking,
input/setup and proof serialization costs; retain the one-query and SAT-only
regressions. A claimed speed milestone needs a fixed protocol, more than three
repetitions, graph-level uncertainty and a clear no-regression deployment boundary.
Optimizing just a path walk or omitting proof/output cost would not close that gate.

The next research gate is an output-sensitive certificate construction or another
clearly justified algorithmic contribution that survives closest-work review and
strong native alternatives. The current route is explicitly not that theorem.
Then freeze the final endpoint and comparison choices before opening a genuinely
unseen topology/application workload. Exposed WAP lists, common-mask projections
and reused synthetic banks must never be relabelled holdouts.

A default or package change requires installable API integration and its own
compatibility, resource and portability evidence. Independent workload definition,
outside-team reproductions, maintained adoption and a credible novelty determination
remain necessary for the original high-90s flagship ambition. Neither more tests
nor relabelling this implementation as a general reasoner substitutes for those.
