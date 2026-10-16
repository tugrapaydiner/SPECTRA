# Exact mixed-outcome support, with original-input proofs

Status: opt-in development implementation. No frozen WAP result, legacy API,
release version, or default solver is replaced. No algorithmic-priority claim.

## Contract

The input is a fixed simple conflict graph with 0..100,000 original vertices,
1..64 colours, and one or two allowed colours per vertex. At most 2,000,000 input
edges are admitted natively. A query contains at most 1,000,000 original-vertex /
uint64-mask pairs. Repeated pins are intersections, not last-write-wins updates;
bits outside the original domain are harmless and cannot add colours. Malformed
inputs raise errors even when an earlier pin already establishes a contradiction.
Original zero/wider domains and self-loops are outside this admitted API.

For every successfully completed call, the service returns either a complete
immutable original-vertex colouring (`SAT_VERIFIED`) or an independently checked
original-literal contradiction (`UNSAT_VERIFIED`). A resource or admission error
is an exception, never a mathematical UNSAT answer. There is no search budget and
no `UNKNOWN` decision for a completed admitted call. This is exact feasibility,
not minimum-change optimization, arbitrary-width SAT, or live migration safety.

## Static formula and exact extension

Write literal `2*v` for the lower available colour and `2*v+1` for the higher.
For a singleton, the second literal is a virtual unavailable alternative; include
its unit implication `2*v+1 -> 2*v`. Complement is XOR with one.

For every shared colour on an original conflict edge, forbid its simultaneous
selection. This gives the two contraposed implications `a -> complement(b)` and
`b -> complement(a)`. The graph is exactly the original binary-list relation.

Compute strongly connected components once. A component containing a literal and
its complement establishes base infeasibility. Otherwise the SCC topological order
provides a base satisfying assignment. This is standard 2-SAT compilation.

A current query either empties an original domain, or supplies forced literals S.
Let C be forward reachability from S in the static implication graph. The query
is infeasible exactly when C contains a complementary pair. Necessity follows
from implication closure. For sufficiency, set C true, its complement false, and
leave every remaining variable at its base value.

To check sufficiency, consider any implication u -> v. A forced-true u forces v
true. If v is forced false, complement(v) is in C; the contraposed implication
forces complement(u), so u is false. The remaining case inherits satisfaction
from the base model. Thus the completed assignment satisfies every original
binary constraint and current pin. Singleton units are included in this argument.

The implementation offers four explicitly named modes, not an outcome-selected
portfolio:

- `raw`: propagate original literals and retain original implication edges.
- `scc`: propagate the SCC DAG; reconstruct contradictions on original edges.
- `sparse`: the same DAG executor, with a reachability-preserving original-edge
  proof subgraph instead of the complete original solver adjacency.
- `route`: retain rooted inward/outward SCC trees and one original representative
  per DAG edge. Reconstruct current-query paths directly from those trees and
  the current propagation predecessors, without a second original-graph BFS.

Every mode performs a separately owned original-input check. `raw` is the cheap
matched exact control, not a general solver forced into a harder problem.

## Smaller proof graph

Within every SCC, keep an outward tree from one root and an inward tree toward
that root. Keep one actual original arc for each distinct condensation edge.
Every retained arc is an original implication. Every pair of vertices in the
same SCC remains mutually reachable via its root; every inter-component step
remains possible. Consequently the entire reachability relation is preserved.

For N=2n literal vertices, K components, and D condensation edges, this retains at
most 2(N-K)+D arcs, and never more than the original implication bank. The native
solver releases its full adjacency in `sparse` mode. The separate checker still
owns the original graph: smaller solver storage is NOT smaller total input or
process RSS. Paths in the reduced proof graph need not be shortest original paths.

When a DAG contradiction is found, a fresh traversal of this proof graph produces
current-root paths. There is no answer cache, saved query decision, or proof cache.
A base contradiction is reconstructed as two paths on every call, not asserted
from a stored Boolean flag. All proof construction and verification are timed.

## SBP1 certificate format

The header is `<4sIII`: magic `SBP1`, kind, left count, right count. Following it
are exactly those two counts of little-endian uint32 values, with no trailing data.

| Kind | Evidence | Independently verified condition |
|---|---|---|
| 1 | One original vertex; no right sequence | Original domain intersected with every current pin is empty. |
| 2 | Two simple original-literal paths | Each root is forced by the current effective domain; endpoints are complements. |
| 3 | Two simple original-literal paths | One runs x to complement(x), the other back, using only static original implications. |

A same-vertex hop is legal only for the original singleton unit. A cross-vertex hop
requires a real original edge and equality of the source colour with the colour
excluded at the destination. The checker does not trust a compiler clause number,
SCC identifier, quotient map, or claimed intermediate reachability.

Both paths are nonempty and individually bounded by 2n literals. SAT answers have
no proof bytes; UNSAT answers have no labels. A proof may remain valid under stronger
restrictions. Fresh checking means rechecking against the CURRENT original inputs,
not demanding a different byte string for the same contradiction.

## Two independent checking implementations

The in-call native observer parses and owns the original graph and lists separately.
It receives no solver adjacency, SCC/DAG, base model, propagation parents or flags.
Its SAT branch checks every vertex/domain, original edge and current pin. Its UNSAT
branch checks every submitted proof hop and endpoint. The solver and observer share
only input admission helpers, not semantic indices. The Python `proof.py` replay
verifier implements admission and path semantics separately and imports no solver.

These are implementation checks, not a machine-checked C++/compiler/hardware proof.
Wrong checks could still be a defect; exhaustive original-label enumeration,
differential tests, adversarial mutations and sanitizers address that risk.

## Costs, transport, and limits

Static construction includes sorting and SCC analysis. Retained storage is linear
in vertices plus retained arcs, with an additional original-graph observer. A raw
query can traverse O(n+m) implication data; condensed propagation traverses its
reachable DAG. UNSAT proof reconstruction may require another traversal. SAT output
and checking still cost O(n+m+r). This is not a universal asymptotic speedup.

New arms produce SCQ2: the old `<4sIQ` header with magic `SCQ2`, followed by each
status byte, n label/padding bytes, a uint32 proof length, and its complete SBP1
bytes. UNSAT padding is all zero. UNKNOWN is not accepted in SCQ2. The complete
session timer includes input parsing/validation/hashing, both native constructions,
every decision, proof creation, original check, output construction, output hash,
diagnostics and disposal. Native build/import and post-run Python audit remain
outside the warm endpoint; cold process wall cost is retained separately.

### Prepared SBC1 session packet

ABI 3 adds an explicitly separate ready-input endpoint. It does **not** redefine
the validated-JSON service or pretend that deployment preparation is free. The
little-endian SBC1 layout is:

1. `<4s7I>`: magic `SBC1`, vertex count n, palette size, edge count m,
   query count q, total pin count p, and two zero reserved words;
2. n uint64 original masks;
3. m sorted unique `(uint32,uint32)` original edges, each with `left < right`;
4. q+1 monotone uint32 pin offsets starting at zero and ending at p;
5. p uint32 original vertex identifiers;
6. p uint64 current allowed masks, with no trailing bytes.

The solver and observer each parse the complete immutable packet into separately
owned graph/domain arrays. Both independently validate schema, counts, reserved
words, exact length, mask domains, edge ordering, all offsets and every current
vertex. The solver builds its own SCC/routing index; the observer builds no such
index. One native call then solves the complete bank, constructs every witness or
SBP1 proof, emits SCQ2, and has the observer reparse the request bank and final
serialized output before publication. Native construction, decisions, proof work,
checking, serialization and disposal are all inside that call.

Packet creation from already validated original objects is recorded separately at
the immutable study-freeze stage. Runtime file read and SHA-256 are charged to the
ready-packet endpoint, but the earlier JSON decode and packet preparation are not.
Therefore a ready-SBC1 result is a deployment-format result, not evidence that first
use from arbitrary JSON has the same latency. Matched prepared `raw`, `scc`,
`sparse` and `route` arms share exactly the same packet and certificate contract.

Old SCQ1 controls are NOT upgraded silently: their UNSAT decisions have no portable
certificate, and the old SPECTRA modes may return UNKNOWN. Comparisons between
SCQ1 and SCQ2 have different certificate contracts and are flagged accordingly.
Only exact modes share the same proof-carrying contract. The Python audit also
independently challenges feasibility decisions from both formats.

Admission checks bound conservative array/proof payload, not Python caller objects,
allocator metadata, total RSS, or deadlines. The process-isolated harness separately
uses RLIMIT_AS and an external wall deadline; RLIMIT_AS is not an RSS bound.
Linux builds, selected CPython versions and tested hardware do not establish ARM,
Windows, laptop or production portability. This experiment namespace is not part of
the stable installed wheel API.

## Prior art and scope of contribution

2-SAT SCC solving, implication propagation, signed equivalences and reachability
spanners built from SCC trees are established. The base mathematical reference is
Aspvall, Plass and Tarjan (1979), *A linear-time algorithm for testing the truth of
certain quantified Boolean formulas*, DOI 10.1016/0020-0190(79)90002-4.

The contribution here is the implemented exact mixed-result service, original-input
proof boundary, accounting and evidence—not a claim to have invented those tools.
New unfiltered banks reuse exposed graph/list inputs; they are development tests,
not fresh WAP confirmation. Independent replication and adoption remain separate
open research-program gates.

## Direct proof routing (retained under ABI 3)

The `route` implementation retains two original-literal parent trees per SCC and
one actual original bridge per condensation edge. It releases the solver's full
original adjacency after construction. Zero retained adjacency arcs does NOT
mean zero retained original constraint information: the parent/bridge arrays and
the separately owned original-graph observer remain and are counted.

Propagation records the actual current seed, predecessor SCC and entering bridge
for each reached SCC. Once complementary components are reached, the executor
lifts two paths from those seeds to a representative literal and its complement.
Inside an SCC, follow an inward tree until its first intersection with the
outward root-to-target chain. Joining there gives a simple path; distinct SCCs
along a DAG predecessor chain cannot repeat. Every emitted hop is an original
implication, and the independent verifier rechecks that hop and the current roots.

This avoids a second search of the proof graph, but is NOT a guaranteed
output-linear or shortest-proof algorithm. Forming/sorting a target ancestor
chain and probing it can cost more than the final returned path, and routed
proofs can be longer. Setup, tree arrays, propagation predecessors, path walks,
serialization and every verification hop remain charged. Base contradictions
are freshly routed in the same trees; empty-domain contradictions retain kind 1.

All four exact modes receive the same sparse pin preparation and root-domain
checking improvement. The raw control is not left on an artificially slower
preparation path. Conservative native query admission now includes an allowance
for routing scratch; it is still not a process RSS bound. The Python/native ABI
is explicitly bumped to 2, so old shared libraries fail admission rather than
being mixed with the new wrapper.

Post-run replay now retains validated ORIGINAL masks and edges. At n <= 4,096 it
checks original conflict edges by exact colour-class/adjacency bitset intersection;
above that fixed threshold it checks original edge pairs without an n-squared
bit matrix. Every submitted colouring and proof is checked, not sampled. The
independent feasibility replay retains only its own original implication adjacency;
current query arcs and SCC decisions are recomputed. The slower scalar verifier
remains available as `proof.verify_answer`; differential tests challenge both.
This audit optimization is outside the timed service and cannot create a service
speedup. Old measured bytes/times are preserved when replayed by the new checker.

ABI 3 capsule names are versioned (`solver.v3`, `observer.v3`). Old ABI 2 handles
are rejected before dereferencing, not merely by a wrapper check. The bump is
required by the new batched and prepared-session entry points; it does not change
the mathematical single-query answer contract.

## Common-contract mixed bank

The wide-mask robustness bank deliberately includes bit 63 even when it lies
outside the declared palette. The older quotient-based checker rejects such pins;
those cells are API-domain errors, not false mathematical answers or numerical
speedup denominators. They remain in their own retained study.

For the matched mixed-outcome comparison, `common_masks.py` removes only unused
palette bits, preserving every pin's intersection with the ORIGINAL domain.
No query is removed, resampled, repaired for satisfiability, or selected by an
answer. The common and wide banks therefore encode exactly the same mathematical
requests, while the common bank stays within all included controls' mask range.
This development correction does not turn exposed cases into a new holdout.
