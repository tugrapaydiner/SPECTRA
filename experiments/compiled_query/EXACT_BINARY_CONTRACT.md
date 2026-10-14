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

The implementation offers three explicitly named modes, not an outcome-selected
portfolio:

- `raw`: propagate original literals and retain original implication edges.
- `scc`: propagate the SCC DAG; reconstruct contradictions on original edges.
- `sparse`: the same DAG executor, with a reachability-preserving original-edge
  proof subgraph instead of the complete original solver adjacency.

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
