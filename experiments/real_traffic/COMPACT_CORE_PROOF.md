# Sparse, checked two-plan migration

## Supported question

Inputs are a fixed conflict graph G=(V,E), two already proper colourings A and B,
and requests restricting original vertices to colours in {A[v],B[v]}. Colours are
integers 0..63. Duplicate request pins are intersected. An empty allowed domain is
infeasible. This is a final-state support service, not a safe live transition
schedule, a general SAT solver, or an optimizer that chooses B.

Let x[v]=1 mean that a changed vertex uses B[v]. Unchanged vertices need no Boolean
choice. Since A and B are both proper, an original edge {u,v} can fail only in a
mixed state. If B[u]=A[v], it imposes x[u] => x[v]. If B[v]=A[u], it imposes
x[v] => x[u]. These are the complete constraints; there are no other failing
combinations. An unchanged vertex cannot be incident to such a dependency: either
A or B would already violate the corresponding conflict.

A request gives a set R of required migrations and a set P of protected vertices.
Let C(R) be forward reachability from R, including R itself. The request is feasible
exactly when it has no empty original domain and C(R) is disjoint from P. In that
case use B on C(R) and A elsewhere. Every feasible solution must contain C(R), so
this is the unique inclusion-minimal migration set. It minimizes any nonnegative
additive migration cost. Zero-cost extra changes can tie in cost; do not claim a
unique cost optimizer.

## Removing the dense closure table

Every vertex of a strongly connected dependency component is forced together.
Contract those components and number them in topological order. Associate one
64-bit word with each component for a tile of at most 64 requests. Bit j is set
when request j forces that component. Seed required components, then OR each
component's word into its successors in topological order.

By induction along that order, a bit reaches a component exactly when the
corresponding request has an original dependency path to it. Thus no transitive
closure table is needed. A conflict bit can stop propagating after meeting a
protected component: it has already established infeasibility, and its proof is
subsequently constructed in the original-address proof graph. Successful lanes
are lifted through retained component members to complete original colourings.

Small request banks can use iterative SCC traversal on the same index. Neither
mode reads an answer bank. `auto` uses traversal below 64 requests and tiles at
64 or more. This is an explicit engineering heuristic, not a theorem about speed.

## Smaller original-edge proofs

For each nontrivial SCC choose a root. Retain an outward search tree from the root
and an inward search tree to the root, using original dependency edges. Keep one
original representative of each condensation edge. Remove duplicate retained edges.

Any two vertices within an SCC remain mutually reachable via its root. Every
inter-component transition remains possible. Therefore the retained graph has the
same reachability relation as the full dependency graph. Conversely, no false
reachability is introduced, because every retained edge is an original edge.

With h changed vertices, k SCCs and D condensation edges, at most 2(h-k)+D proof
edges are needed; the implementation also checks that it never retains more edges
than the original dependency bank. These are paths, not necessarily shortest paths.
When a request is impossible, ordinary traversal of this retained graph produces a
fresh original-address path from a required migration to a protected vertex.

## Independent observer

`migration_observer.cpp` accepts only the original graph, original plans, current
requests, and returned bytes. It does not load the candidate's SCCs, condensation,
proof forest, membership map, or computed closure.

For SAT it checks every colour belongs to the original two-plan domain, every
request pin, and every original conflict. It separately reconstructs the required
migration closure from the full original dependency graph and verifies exact
minimum change. This check is inside the measured service timer.

For dense original graphs it checks conflict equalities in 64-output tiles. After
domain membership and proper A/B have been verified, {u,v} fails exactly when
B[u]=A[v] and x[u]=1,x[v]=0, or B[v]=A[u] and x[v]=1,x[u]=0. Wordwise tests of these
two conditions check the same original edge relation without a quadratic matrix.
For sparse original graphs it directly compares returned endpoint colours.

For UNSAT it verifies either a genuinely empty original allowed domain or every
edge and both endpoints of a required-to-protected path. Unknown statuses, malformed
buffers, out-of-range colours, repeated proof vertices, and unconsumed proof bytes
are rejected. UNSAT label padding must be zero. No benchmark-supplied answer or
contradiction is used as an oracle.

## Complexity and boundaries

Retained solver arrays are O(n+h+D+d'), where d' is the retained original-edge proof
graph. The independent observer retains O(n+m) arrays, including the full original
conflict graph and independently reconstructed dependencies. Neither retains an
n-by-n reachability/implication matrix. Construction still reads and sorts the
original graph; sparse retained storage does not imply sublinear preprocessing.

The propagation kernel visits each condensation edge at most once per 64-request
tile. Original output construction costs at least Omega(nQ), since complete
answers are requested. Proof creation can require a graph traversal for each
infeasible request. Independent minimum-change verification can also take O(n+d)
per successful request. Accordingly no universal end-to-end asymptotic speedup
is claimed. The whole service must be measured, not just the wordwise kernel.

Admission caps describe conservative native array payload, not total process RSS,
Python object storage, allocator overhead, thread stacks, or application-owned
inputs. Read the measured RSS separately. Only Linux x86-64 / CPython 3.13 was
exercised in this continuation.

## Relationship to established work

SCC condensation, graph traversal, inward/outward reachability trees and bit-parallel
multi-source processing are established foundations. This checkpoint is an exact
implementation and measured engineering improvement, not a claim of algorithmic
priority. Relevant primary/background documentation:

- NetworkX's condensation documentation: contraction yields a DAG.
  https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.condensation.html
- Wang, Blelloch, Gu and Sun, *Parallel Cluster-BFS and Applications to Shortest
  Paths* (2024), describes established bit-parallel multi-source BFS as background.
  https://arxiv.org/abs/2410.17226

The arguments above specify the intended mathematics. Exhaustive tests, differential
checks and sanitizers support implementation correspondence but are not a
machine-checked proof of the C++ program, compiler or hardware.
