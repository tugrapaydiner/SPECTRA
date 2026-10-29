# Compiled-query development protocol

This is a **new development study**, not a modification, repeat confirmation or
extension of the frozen WAP headline. Passing internal contracts does not
establish independent replication or a broader scientific claim.

## Question and contract

Compare a complete original-address binary-list feasibility session against
independently maintained CUDD, persistent native MiniCard, persistent native
CaDiCaL and the noncontracting SPECTRA control. The CUDD arms use natural,
occurrence and incidence-BFS orders, with dynamic reordering enabled. No arm is
chosen retrospectively as the only reported competitor. The BDD receives the
original CNF, not SPECTRA's quotient, cached answers or a supplied solution.

A SAT answer contains every original vertex's colour. All arms use a separately
owned native original-edge/list/restriction observer. The audit additionally
checks full SAT witnesses directly in Python. Its independently written
binary-domain implication checker challenges UNSAT decisions from original
constraints without calling the benchmark's CNF encoder or solver. This does
not create a portable UNSAT certificate.

The existing SPECTRA API returns `UNKNOWN` for unsuccessful searches, including
infeasible requests. This study **does not reinterpret those results as UNSAT**.
They are retained as `INCOMPLETE`, and incomplete cells cannot earn a latency
win. The migration packet service has a stronger minimum-change/proof-path
contract and is not compared as though it were this feasibility endpoint.

## Cost boundary

The input to every measured session contains **exactly the requested number of
queries**. A one-query session does not decode a 1,024-query bank and discard
most of it. Prefixes overlap and are not independent samples.

The complete timer includes bounded file read, JSON decoding, validation and
hashing; fresh formula/index/BDD and independent-checker construction; all
query conditioning/solving, full witness creation and checking; SCQ1 output-bank
construction and its SHA-256; backend diagnostics; native disposal and release
of non-output Python input/staging owners. The six disjoint stage durations
must sum exactly to `complete_ns`; per-query durations are nested inside the
query-loop duration and are not added twice.

The fixed SCQ1 bank has a 16-byte header (`<4sIQ`: magic, original vertex count,
query count), followed by one decision byte plus n colour bytes per request.
Decisions are 0=UNSAT, 1=SAT, 2=UNKNOWN. Non-SAT rows contain zero padding, not a
witness. This output format and the leaner input schema differ from the old
frozen JSON endpoint; ratios across the two studies are not interchangeable.

Process startup, source/dependency integrity checks and imports are excluded
from this warm endpoint but INCLUDED in separately retained `cold_wall_ns`.
Native build, graph acquisition, case generation, evidence-file writing and
the independent post-run audit are outside both the warm endpoint and its
scientific claim. Runtime compilation has an explicit build receipt. The
returned measurement-record construction is instrumentation, not a separately
claimed user service. All cases and binaries are replayed from local snapshots;
filesystem caching and the host environment remain part of the experiment.

## Scheduling, limits and interrupted work

Before any trial, `new` writes source hashes, interpreter/backend-code/host
identities, exact per-prefix case snapshots, native library and build receipt,
configuration and the entire randomized paired schedule. All arms in a paired
case/prefix/repetition group use the SAME request order and seed. Repetitions
cycle through forward, reverse and shuffled order. The controller runs one
worker at a time, with a fixed CPU affinity in each worker.

Workers use a hard Linux address-space limit before backend loading, and the
parent enforces a wall deadline by killing the whole worker process group.
`RLIMIT_AS` is **not an RSS cap**. Peak process RSS is reported, not claimed as
a hard cap. CUDD's memory estimate and between-operation node checks remain
soft guards. A timeout, crash, unavailable backend or compilation limit is a
failure record, never an UNSAT decision or an infinite-speedup denominator.

Raw output and worker/controller receipts precede any aggregate analysis.
Resume verifies input, source, interpreter, direct backend-code, host and
native-library identities, as well as prior result/output hashes. It does not
rerun a failed trial until one succeeds. An interrupted trial is marked
`INTERRUPTED`; a retry requires a separately identified study. A controller
lock prevents concurrent writers. Retained negative outcomes are immutable.

## Workloads and interpretation

The built-in synthetic cases cover unconstrained full output, many small
components, a long component, mixed binary lists and a dense bipartite graph.
Requests are generated without a solver or an acceptance filter; contradictory
and infeasible restrictions are kept. These are disclosed constructions, not
operator workloads or evidence of broad real-world transfer.

The WAP exporter reads **only wap05a.col and wap06a.col**, the previously
identified development graphs, from the pinned upstream inventory. Their
conflict topologies are public; their lists are synthetic and the historical
query generator retains SAT requests using a SAT solver. The original graph
hashes and generation receipts are preserved. No new holdout is opened and no
outcome-blind confirmation is claimed.

Reports retain every scheduled cell. An aggregate comparison is withheld if
any requested case has an incomplete/failed cell. Complete-session ratios are
first combined within the same graph cluster and then geometrically averaged
across graph clusters. Requests, repetitions and overlapping prefixes are not
counted as additional graphs. Summaries are descriptive: no new statistical
significance or flagship gate is claimed. The cheapest exact applicable
competitor, preparation costs, negative cases and p95 remain visible.

## Reproduce

Install the pinned native dependencies using the workflows in this branch.
The direct upstream dd source pin is
`4442d498ddea1bec0071315f538cb5ef37443985`; native SAT is
`python-sat==1.9.dev15`. Native CUDD must import successfully; there is no
silent fallback to the Python reference implementation.

```bash
SPECTRA_REQUIRE_CUDD=1 python -m pytest experiments/compiled_query -q
python -c "from spectra.cnf.quotient_query import build_quotient_runtime; print(build_quotient_runtime('cq-build'))"
python -m experiments.compiled_query.study synthetic --output cq-cases --n 96 --queries 128
python -m experiments.compiled_query.study new --cases cq-cases --library cq-build/_spectra_quotient_query.cpython-313-x86_64-linux-gnu.so --output cq-study --prefixes 1 128 --repeats 3
python -m experiments.compiled_query.study run --output cq-study
python -m experiments.compiled_query.study audit --output cq-study > cq-audit.json
```

Use the actual library path printed by the build command for the local Python
ABI. `run --max-new N` stops after N additional scheduled jobs and can be
continued with the identical source and environment. Neither `new` nor the
case generators overwrite an existing destination.
