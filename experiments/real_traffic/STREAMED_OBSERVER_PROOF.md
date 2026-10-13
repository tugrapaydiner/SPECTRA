# Checking the same answer with less stored state

This is an implementation derivation, not a claim to have invented implication
reasoning, word-parallel reachability, or certifying computation. The supported
contract is the existing two-valid-plan migration problem; no general-CNF or
operational network-control guarantee is added.

## Original-input contract

Let G=(V,E) be a fixed conflict graph and let A and B be two proper colourings.
Each colour is an integer in [0,63]. Every answer assigns vertex v either A(v)
or B(v). A query supplies additional allowed-colour masks at original vertices;
duplicates mean conjunction, not replacement. An empty intersection is infeasible.

The observer independently validates both original colourings against **every
supplied edge** during preparation. It does not accept a solver-generated SCC map,
condensation graph, reachability table, proof graph, or bank of reference answers.

## Exact edge elimination, not sampled checking

For an assignment define x(v)=1 exactly when its colour differs from A(v).
A vertex with A(v)=B(v) has x(v)=0.

For each original edge {u,v}, proper A and B exclude both AA and BB collisions.
The only possible remaining collisions are:

* B(u)=A(v), corresponding to x(u)=1 and x(v)=0;
* B(v)=A(u), corresponding to x(v)=1 and x(u)=0.

Consequently the entire original relation, on this supported domain, is equivalent
to the directed implications u -> v for B(u)=A(v), and v -> u for B(v)=A(u).
No such implication can touch an unchanged vertex: the requisite equality would
contradict proper A or proper B. An edge generating no implication is satisfied
by every permitted local assignment and can be discarded exactly.

The observer constructs and owns all these **original direct implications** in
sorted sparse rows. It performs no SCC contraction or transitive reduction. The
larger undirected edge bank is temporary and is actually released; retained vector
capacity is included in the payload measurement. Every SAT response is explicitly
checked against the retained exact relation, even when minimum-change checking is
not requested. This is equivalent checking of every original constraint, not a
literal rescan of every original edge on every query.

## Minimum necessary change

Let P be the query-pinned migrations and F the query-pinned non-migrations. Let
R be the least set containing P and closed under the original direct implications.
Every feasible answer must move all of R. If no input mask is empty and R and F
are disjoint, moving exactly R is itself valid. It is therefore the unique
inclusion-minimal migration set. It minimizes any nonnegative additive cost of
migration; additional zero-cost moves may tie in cost but not in inclusion.

This is not a guarantee of a safe live reconfiguration sequence or an optimal
choice of target plan. The input two-plan restriction remains essential.

## Independent 64-request verification

For each tile of at most 64 requests, each original vertex has a machine word.
Bit j represents one SAT request. The observer constructs two independent word
banks: `actual`, decoded from the submitted full assignments, and `required`,
seeded from the original query pins. UNSAT lanes are absent from both banks.

Membership in {A(v),B(v)} is checked before any answer colour is used as a mask
shift. The original implication u -> v is checked by `actual[u] & ~actual[v] == 0`.
The observer then computes the least fixed point of required bits using its own
work queue on the original graph, including cycles. Only newly arriving bits cause
a vertex to be re-enqueued. At most one queue entry per vertex exists concurrently.
When popped, its full required word is propagated; old bits cannot create a new
arrival because they were propagated on the preceding pop. A second pending-word
array is therefore unnecessary.

Inductively, every propagated bit is reachable from that request's original pins
(soundness). When the queue empties, the set is closed and contains all its pins,
so it contains the least reachable set (completeness). No bit is removed, and each
vertex gains at most 64 bits, which proves termination. Equality of `actual` and
`required` at every original vertex proves minimum necessary change separately
for every SAT lane. A closed but unseeded migrated cycle is rejected, not mistaken
for the least solution.

The scalar comparison uses ordinary per-request BFS over the **same independently
constructed original dependency rows**. It is an ablation, not a deliberately
weaker semantic contract.

## Fresh UNSAT proof checking

A normal contradiction is a simple original-vertex path beginning at a pinned
migration and ending at a pinned non-migration. Every directed step is checked
against the independently reconstructed original direct implications; transitive
shortcuts are not accepted. A length-one path can prove contradictory duplicate
pins at a vertex. An empty-domain proof must identify a vertex with an individually
empty request mask, not an unrelated constraint.

Status values, proof kinds, offsets, little-endian vertex decoding, path length,
repeated vertices, full assignment sizes and zero UNSAT-slot padding are validated.
No stored proof is looked up. Every returned proof is generated by the solver on
the current request and checked again.

## Immutable transport is not a trust bypass

The new service snapshots mutable query iterables once. Fully immutable, exact
tuples containing pairs of exact Python integers are reused. The freezing operation
has hard query and pin-count limits and propagates iterator errors. Query values
are **still parsed and range-checked independently** by the solver and observer.
The explicit `solve_frozen`/`verify_frozen` methods reject mutable bank structures;
existing iterable-taking methods remain available.

The old solver and checker receive these same immutable entry points in the matched
benchmark. Thus reduced Python allocation is measured separately from checker
algorithm changes. Complete output materialization and checking remain charged.

## Costs and limits

If d is the number of original direct implications (d <= 2|E|), retained observer
arrays are O(|V|+d), without storing the additional 8-byte-per-edge original bank.
Preprocessing still checks every original edge and sorts sparse rows; this cost
is inside every fresh-session measurement.

Work storage is O(|V|+pins+requests). For minimality-enabled tiled checking the
vertex work arrays occupy 33 bytes per original vertex in this implementation;
scalar checking uses 20 bytes per vertex. The tiled method is **not** always lower
scratch memory than scalar BFS. Both reuse the same retained observer index.
The fixed 64-request tile bounds work storage independently of bank length.

All full output bytes are read, so O(|V|*requests) output inspection is not avoided.
Propagation can still take O(64*(|V|+d)) per tile in an adverse cyclic graph; no
universal speedup follows from word-parallel execution. The `auto` threshold of
64 requests is a simple heuristic, not an oracle or a universal optimal choice.

Reported array payloads exclude Python objects, allocator metadata, shared-library
mappings, compiler memory and caller-retained inputs/outputs. They are not process
RSS guarantees. Resource tests, sanitizers and differential testing support this
implementation; they do not constitute a machine-checked proof of C++, the compiler
or hardware.

Related foundation: George C. Necula, *Proof-Carrying Code*, POPL 1997,
DOI 10.1145/263699.263712. This is background on separating production from checking,
not a claimed direct predecessor or an exhaustive novelty review of this service.
