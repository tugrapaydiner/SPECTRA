# Exact queries

`spectra.exact` answers restriction queries over a fixed conflict graph in which
every vertex has one or two allowed colours. Each completed call returns either a
complete colouring of the original vertices or a portable proof that the request is
contradictory. There is no `UNKNOWN` outcome for an admitted call, no fallback
solver, and no compilation at import time.

## Formats and versions

| Layer | Identifier | Contents |
|---|---|---|
| Python API | `EXACT_API_VERSION = 1` | names, validation, result objects |
| Native ABI | `EXACT_NATIVE_ABI = 3` | CPython extension entry points |
| Query bank | `SBQ1` | restrictions for a prepared graph |
| Session packet | `SBC1` | graph, domains and query bank in one immutable buffer |
| Answer bank | `SCQ2` | per query: full labels (SAT) or a complete proof (UNSAT) |
| Proof | `SBP1` | original-literal implication paths, checkable without solver state |

## Building the runtime

```python
from spectra.exact import ExactRuntime, build_exact_runtime

runtime = ExactRuntime(build_exact_runtime("build/exact"))
```

The builder compiles the packaged C++ source with the local compiler (`g++` on
Linux, MSVC on Windows x64) and publishes the extension next to a `build.json`
receipt recording the source hash, compiler command and version, Python ABI, and
binary hash. `ExactRuntime` checks the receipt before loading the library; pass
`verify_receipt=False` only for a separately managed, trusted build. Requires a
GIL-enabled CPython.

## Problems and sessions

```python
from spectra.exact import ExactProblem, verify_answer

problem = ExactProblem.create(
    n=4, palette=3,
    edges=((0, 1), (1, 2), (2, 3)),
    masks=(0b011, 0b110, 0b101, 0b011),
)

with runtime.prepare(problem, mode="scc") as session:
    query = ((0, 0b001), (3, 0b010))
    answer = session.solve(query)
    assert session.check(query, answer)
    assert verify_answer(problem, query, answer.status, answer.labels, answer.proof)
```

`ExactProblem.create` validates and canonicalises its inputs: edges become sorted,
unique `(low, high)` pairs and each mask must select one or two colours from a
palette of at most 64. Limits are 100,000 vertices and 2,000,000 edges. A query is a
sequence of `(vertex, allowed_mask)` pairs; repeated pins on a vertex intersect.

Sessions own a solver and an independently constructed observer that checks every
answer. Modes `raw`, `scc`, `sparse` and `route` share the same answer contract and
differ only in internal data structures. Closing is idempotent; use after close
raises.

## One-call session packets

```python
from spectra.exact import pack_session_packet, verify_session_output

packet = pack_session_packet(problem, ((), ((0, 0b001),), ((0, 0b001), (1, 0b001))))
result = runtime.solve_session_packet(packet, mode="auto")
assert verify_session_output(packet, result.output)
```

The solver and observer parse the packet independently. One call builds both,
answers and checks every query, serialises `SCQ2` and frees native state.

`mode="auto"` chooses a strategy from the graph's structure, the SCC geometry, the
bank length and pin counts only; it never looks at answers or timings. The chosen
mode and the reason are reported in `result.info` (`selected_mode`,
`auto_policy_reason`). Edgeless graphs use a direct-domain kernel, and queries that
directly force a conflict on an original edge receive a minimal proof without a
full closure replay.

`pack_validated_session_packet` is a faster packer for inputs that are already
admitted; it still rechecks types, bounds and ordering. The decoder rejects
reversed, duplicate, unsorted, truncated or overlong packets instead of repairing
them.

## Independent verification

`verify_answer`, `verify_certified_output` and `verify_session_output` are pure
Python. They read only the original inputs and the answer bytes; they never load
the native extension or reuse solver state.

```bash
spectra exact verify request.sbc1 answer.scq2
```

## Self-test bundles

```bash
spectra exact selftest bundle/      # build, run SAT/UNSAT/replay paths, write a manifest
spectra exact audit bundle/         # re-verify a bundle without compiling or loading native code
spectra exact compare a/ b/ c/      # check bundles from several machines agree
```

A bundle contains the source used for compilation, the request packets, the
certified outputs and a hash manifest. Bundles from different platforms must agree
on everything except the compiled binary and environment metadata.

## Resource limits

`max_bytes` bounds native owners, scratch arrays, proofs and serialised output. It
is an admission bound, not a process-memory limit or a deadline; run untrusted
workloads under operating-system limits.

## Quotient runtime

`spectra.cnf.quotient_query` is the compiler behind the WAP result in the README.
It builds the binary implication relation of a list-colouring problem, merges
strongly connected (implication-equivalent) choices, and keeps an exact mapping
from every original vertex and colour to the quotient. Queries are expressed in
the original address space and every successful answer is lifted back to a full
colouring and checked against the original graph.

```python
from spectra.cnf.quotient_query import QuotientRuntime, build_quotient_runtime

quotient_runtime = QuotientRuntime(build_quotient_runtime("build/quotient"))
prepared = quotient_runtime.prepare(
    n=4, k=3, edges=((0, 1), (1, 2), (2, 3)),
    masks=(0b011, 0b110, 0b101, 0b011), mode="scc",
)
try:
    result = prepared.solve(restrictions=((0, 0b001), (3, 0b010)), max_work=1_000_000)
    if result.status == "SAT_VERIFIED":
        assert len(result.labels) == 4
finally:
    prepared.close()
```

Exhausting `max_work` or reaching a contradiction returns a non-solution status;
this runtime does not emit UNSAT proofs (use `spectra.exact` for that).
`spectra.cnf.quotient_certificate` independently verifies an integer-only
whole-relation certificate, rebuilding the original implication graph with the
standard library only.

## Exam repair

`spectra.exam_repair` maps bounded exam rescheduling onto `spectra.exact`. Each
exam has one or two candidate periods; exams that share a student, or are
explicitly excluded from each other, may not share a period. An event (closed
periods, fixed moves, restricted periods) becomes an exact query.

```python
from spectra.exam_repair import ExamRepairModel

model = ExamRepairModel.create(
    period_ids=("Monday AM", "Monday PM"),
    exam_ids=("MATH101", "CHEM101", "HIST101"),
    students=((11, 20), (20, 31), (31,)),
    candidate_periods=(("Monday AM", "Monday PM"),) * 3,
)
query = model.freeze_event(
    closed_periods=(),
    fixed=(("MATH101", "Monday AM"), ("CHEM101", "Monday AM")),
)
with runtime.prepare(model.problem, mode="scc") as session:
    answer = session.solve(query)

if answer.status == "SAT_VERIFIED":
    print(model.assignment_from_labels(answer.labels, query))
else:
    print(model.explain_unsat(query, answer.proof))   # names exams, periods, student
```

This is repair within pre-approved candidates, not full timetabling: room
capacity, precedence and soft objectives are out of scope.
`spectra.itc2007_exam` reads the ITC-2007 examination format; its `conflict_core`
conversion records every constraint class it omits and requires the caller to
acknowledge them.
