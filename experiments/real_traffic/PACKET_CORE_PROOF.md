# Exact answers with fewer copies

This is an opt-in systems implementation of the established two-valid-plan
migration problem, not a new general SAT algorithm or a scientific-priority claim.

## What is trusted

The caller supplies a graph G, two proper colourings A and B, and a bank of
original-vertex uint64 allowed-colour masks. Both components validate the graph
and plans independently. Every mixed-plan conflict is the implication u -> v
when B(u)=A(v). Proper A and B exclude all other collision types.

The SPQ1 transport contains only requests, not a solver index, proof, or answer.
The solver and observer have separate native parsers. Each checks the schema,
reserved bytes, total lengths, count limits, complete monotone offsets and every
vertex bound before releasing the GIL or reading a query in its worker.

## SPQ1 exact byte format

All integers are little-endian, with no alignment requirement.

* 4 bytes: ASCII `SPQ1`.
* uint32: request count Q.
* uint32: total pins P.
* uint32: reserved, must equal zero.
* Q+1 uint32 offsets, starting at zero and ending at P.
* P uint32 original vertex IDs.
* P uint64 original masks.

Exact length is 16 + 4(Q+1) + 12P. No trailing data is accepted. Repeated pins
are conjunctions; row and pin order are preserved, including empty rows and masks.
A raw uint64 mask is retained exactly, including bits unused by A or B.
The immutable Python bytes object remains strongly referenced during native work.
No pointer into caller-mutable memory is accepted.

Packing is not solving. A bank can be constructed without knowing a graph or its
plans. `solve` charges packing; `solve_packet` is a separate ready-input endpoint.

## Independent direct minimum verification

For a SAT request, let P be vertices whose pin disallows A and allows B, and F
vertices whose pin disallows B. Let R be the least set containing P and closed
under ORIGINAL implications. An individually empty mask is infeasible. A request
is satisfiable exactly when it has no empty mask and R avoids F.

Every satisfying assignment must migrate R. Setting precisely R to B and all
other vertices to A is valid when that intersection is empty. This is the unique
inclusion-minimal migration set and minimizes any nonnegative additive migration
cost. Zero-cost superfluous changes can tie on cost, but not on inclusion.

The direct checker computes R independently from the original graph. It constructs
an expected full answer one row at a time and compares every submitted output
byte with that answer. It then checks every raw pin against the already
membership-validated submitted bytes. Equality proves original feasibility and
minimality together; no separate decoding of the answer into a second bit matrix
is necessary. An unseeded migrated cycle is rejected, not treated as a minimum.

The observer's cyclic work queue is deliberately different from the solver's
SCC-condensed topological propagation. It accepts no solver-produced SCC map,
reachability bits, prevalidated answer template or proof cache.

The low-scratch `rows` mode uses ordinary original-graph BFS and compares a fresh
row. The `direct` mode shares forced query bits across at most 64 requests before
constructing row expectations. Both are exact. `auto` is a fixed 64-request
heuristic, not an oracle or a guaranteed fastest selector.

## Contradictions

As before, each UNSAT output contains a fresh original-edge path from a required
migration to a protected vertex, or an original vertex with an individually empty
mask. Duplicate contradictory pins can have a one-vertex path. The observer
checks every path edge, endpoints, simple-path requirement, offsets, kinds,
output lengths and zero-filled UNSAT answer slots.

## Answer allocation and lifetime

Full labels are written into a newly allocated private Python bytes object. It
is not exposed to Python until solving is finished, and the checked API does not
return it until the independent observer accepts it. The object is never mutated
after publication. All exception paths restore the GIL before Python reference
cleanup. Empty and single-byte output cases are tested explicitly.

This uses the documented `PyBytes_FromStringAndSize(NULL, size)` construction
contract. Status and proof banks still use ordinary checked copying. The solver
therefore avoids one full Q*|V|-byte native staging buffer; it does not claim
zero total copying or zero full-output memory.

## Accounting and scope

Persistent solver and observer indexes are unchanged in size. Packets replace
per-component parsed query arrays, but remain live themselves. Count packet bytes
alongside checker scratch: reporting the reduced checker-owned scratch alone would
overstate total memory savings. Full returned assignments remain O(Q*|V|) bytes.

The supplied guards cover conservative array/byte-buffer admission, not total
process RSS, Python object metadata, compiler memory or execution time.
The protocol provides no safe live reconfiguration schedule, dynamic graph-update
support, general-CNF claim, fresh unseen-network evidence, or outside replication.

Sources consulted for implementation semantics:
- CPython 3.13 Bytes Objects API: https://docs.python.org/3.13/c-api/bytes.html
- GCC auto-vectorization documentation: https://gcc.gnu.org/projects/tree-ssa/vectorization.html
These support implementation choices, not novelty priority.
