# ITC-2007 bounded exam-repair transfer protocol

**Status:** fixed before opening the external instance and schedule in CI.  This is
an application-format transfer of the versioned proof-carrying exact-query API,
not a complete capacitated examination-timetabling result.

## External source and attribution

The workflow checks out `davidvanbulck/ITC-2007-CETT` at commit
`ed2d4c0cf5302b77a8b321cd0380de0bb579d538`.  The selected files are:

- `examtt-instances/Instances/set9.exam`
- `InitMethods/SolStore/set9_OBSIFeas_0.sol`
- `LICENSE`

The upstream repository states that its solver and instances originate from the
ITC-2007 examination-timetabling setting and requests citation of Van Bulck,
Goossens and Schaerf.  Every evidence package retains the upstream license and
this acknowledgment:

> This product includes software/data distributed by David Van Bulck, Dries
> Goossens, and Andrea Schaerf, Ghent University, Belgium
> (https://github.com/davidvanbulck/ITC-2007-CETT.git).

No upstream source or data is committed to SPECTRA.

## Admitted question

Start from the published feasible timetable.  Each examination is assigned one
or two **pre-approved candidate periods**:

1. The published period is always retained.
2. Exams appearing in `AFTER`, `EXAM_COINCIDENCE`, or `ROOM_EXCLUSIVE` hard
   constraints are frozen to the published period.
3. Every other exam receives one deterministic alternative when available.  The
   alternative must have enough duration and the instance must contain a room
   large enough for the exam in isolation.  A SHA-256 ordering of
   `(input hash, exam, period)` chooses among eligible alternatives.
4. Shared-student conflicts and explicit `EXCLUSION` constraints become graph
   edges.  The palette contains the instance's declared periods.

The exact question is therefore:

> Given current period closures and requested moves within these fixed
> one/two-period domains, does a conflict-free period assignment exist?

A completed call must return either a full original-exam assignment or a portable
original-literal contradiction proof.  The standard-library verifier replays all
answers from the frozen `SBC1` packet and `SCQ2` output.

## Deliberately unmodelled constraints

The reduction does **not** solve or score:

- room reassignment or simultaneous room-capacity packing;
- soft institutional penalties;
- changes to `AFTER`, `EXAM_COINCIDENCE`, or `ROOM_EXCLUSIVE` exams;
- candidate-period generation outside the frozen deterministic rule;
- the full ITC-2007 optimization problem.

The parser retains counts for every omitted constraint class.  Conversion requires
an explicit `acknowledge_ignored_constraints=True`, and the omission receipt is
stored in every manifest and audit.  A conflict-core SAT answer is not a complete
ITC timetable certificate.

## Frozen event bank

The bank contains 1,024 unique events in this order:

1. the unchanged published baseline;
2. every single-period closure;
3. every single flexible-exam move to its alternative;
4. deterministic multi-exam move sets of width 2 through 12;
5. deterministic period-closure plus multi-move events.

The seed is the first 64 bits of SHA-256 over the exact instance bytes, a NUL byte,
and the exact solution bytes.  Widths, exam rankings and closed periods are then
derived directly from labelled SHA-256 words rather than Python's version-sensitive
sampling helpers.  Duplicate exact queries are removed before the bank is
truncated.  Failure to create 1,024 distinct events is a protocol failure, not
permission to alter the policy after observing results.

## Engines and measured boundary

The same immutable packet is run through `auto`, `raw`, `scc`, and `route`.  Each
mode receives eight untimed warmups; 32 measured repetitions rotate through
complete four-position blocks, so every mode occupies every timing position
equally.  The measured boundary starts with ready
`SBC1` bytes and includes:

- input validation and native construction;
- every query decision;
- complete SAT witnesses or portable UNSAT proofs;
- the separately owned native observer;
- `SCQ2` materialization;
- native disposal and the Python call boundary.

External checkout, parsing, candidate generation, packet packing, native
compilation, evidence writes, and offline replay are outside the timing boundary.
Timing is descriptive for this single development transfer; there is no speed-win
gate here.  The separate prospective multi-instance suite declares its holdout
threshold before execution.

## Freeze, execution and audit

`freeze` validates everything before atomically publishing an evidence directory.
It writes external bytes, the upstream license, candidate rows, canonical queries,
packet, source inventory, commit/tree identities and hashes with exclusive
creation.  Every listed source blob must match the named commit.

`run` explicitly builds the bundled ABI-3 runtime, records its receipt hash, runs
all engines/repetitions, requires byte-stable repeated output per engine, and
verifies every output before publication.

`audit` is rerunnable without a native solver.  It reparses the original instance
and solution, rechecks all baseline hard constraints, regenerates the seed,
candidate rows and complete event bank, reconstructs the packet, checks recorded
metadata against the certified bytes, replays every output, requires all modes to
agree on SAT/UNSAT decisions, counts proof kinds, and refuses changed evidence or
a changed prior audit.  Stored candidate/query JSON is evidence, not an oracle.

## Interpretation

A successful run demonstrates that the exact API transfers from exposed synthetic
graph panels to a published timetabling-format repair core while retaining full
proof boundaries.  It remains project-owned execution on a deterministic derived
problem.  It is not operator adoption, outside-team replication, a claim of
algorithmic priority, or evidence that SPECTRA solves the full ITC-2007 task.
