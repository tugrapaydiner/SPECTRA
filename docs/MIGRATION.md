# The smallest necessary change between two valid plans

This result concerns **final-state support**, not a safe sequence of live network
reconfiguration operations. Both input plans must be proper colourings of the
same original conflict graph. Every vertex can use its reference or target colour,
and requests may restrict any original vertex. Traffic values and the plan policy
remain separate from this mathematical contract.

Reachability, Horn implication and reference-first model selection are established
techniques. The implementation and proof below do not establish algorithmic novelty.

## Exact reduction

Let A and B be the two proper plans. Ignore unchanged vertices when discussing
migration choices: they always retain their one colour. For each changed vertex v,
let x_v=1 mean using B_v instead of A_v. Construct a directed dependency u -> v
exactly when {u,v} is an original conflict edge and B_u=A_v.

For any original edge, the A/A assignment is legal because A is proper, and the
B/B assignment is legal because B is proper. Therefore a violation can only be a
mixed B/A assignment. The equality B_u=A_v forbids x_u=1,x_v=0, which is precisely
x_u -> x_v. The reverse mixed assignment contributes v -> u when appropriate.
No other original edge restrictions remain. Edges incident to unchanged vertices
cannot introduce these dependencies: the equality would make one original plan
improper, which input admission rejects.

**Lemma.** A set M of changed vertices yields a proper final colouring if and only
if it is forward-closed under these directed dependencies.

A request becomes a set S of vertices that must move and a set T that must stay.
A request allowing neither original colour is an immediate contradiction.
Repeated restrictions are intersections, not last-write-wins overrides.

Let R=Reach(S), including S itself.

**Feasibility theorem.** A request has a supported final colouring if and only if
there is no empty restriction and R intersects T trivially.

Necessity follows because every supported migration set is forward-closed and
contains S, hence contains R. For sufficiency, choose B on R and A elsewhere.
The set R is closed, respects the positive and negative pins, and satisfies the
original edge relation by the lemma.

## Minimum-change guarantee

Every feasible migration set contains R. Thus the solution above is the unique
**inclusion-minimal** migration set. For every collection of nonnegative vertex
costs w_v, it minimizes sum_{v in M} w_v. In particular it minimizes the number
of changed original vertices. With zero costs, other sets can have the same cost;
uniqueness of the minimum numerical cost is not claimed.

This guarantee requires no traffic-cost fitting or objective-specific search.
It does not optimize routes, spectrum width, capacity, arbitrary colours outside
the two plans, physical-layer constraints, or the choice of target plan.

## Why an impossible request can carry a small proof

When R intersects T, a directed path starts at a member of S and ends at a member
of T. A verifier checks its two endpoint pins and, for every hop u -> v, checks
an actual original edge and B_u=A_v. That proves the contradiction without trusting
a compiled closure matrix or a stored answer. The native BFS returns a shortest
such dependency path. A zero-edge path covers incompatible duplicate pins on one
vertex. An empty-domain restriction has a separate one-vertex proof.

For compatibility with the existing independent native checker, paths are also
serialized into two original signed-literal implication paths. If L_i denotes
selecting the target colour of path vertex i, the long path is
L_0 -> ... -> L_t -> not L_t -> ... -> not L_0; the root pin gives the reverse
not L_0 -> L_0. Each step must be justified by an original edge or current request.
Serialization itself never consults a saved SAT/UNSAT status or reference proof.

SAT answers are complete immutable byte arrays. The timed service checks every
original vertex list, edge and current pin. Independent minimum-change checking
is also performed in the evaluation, but is **outside** the service timer. Do not
advertise the timings as including an additional per-call optimality audit.

## The strong cheap controls

Uncompiled BFS solves this special contract directly; it is not a deliberately
slow control. The API defaults to BFS. The compiled alternative precomputes
reachability, unions the rows addressed by current pins, and still constructs a
real path when it must explain an impossible request.

MiniCard is also given reference-first polarities with phase saving disabled,
while retaining learned clauses. A CUDD control descends its actual BDD using
reference-first choices. Because R extends every consistent reference-first partial
choice, these controls also return R for this two-plan relation. They are not being
asked to solve a harder general optimization problem than the candidate.

The benchmark keeps both BFS and compilation. A small speed difference over BFS
cannot establish a novel research advantage, even when general SAT or BDD execution
costs more.

## Complexity and limits

With n original vertices and d directed dependencies, a fresh BFS query costs
O(n+d+r) before full original-edge checking, where r is the number of request pins.
Full immutable output costs O(n), and original-edge checking costs O(n+m+r).
The optional precomputation uses O(n ceil(n/64)) machine words (O(n²) bits) and
O(n(n+d)) construction work; this is charged in setup. Its query unions cost
O(|S| ceil(n/64)), plus output and any requested contradiction-path construction.
It is not a free preparation step or a guaranteed latency improvement.

Native array admission caps do not bound allocator metadata, Python inputs, the
process RSS or hard real-time deadlines. There is no deployment, independent-team
replication, transient reconfiguration-safety, or general reasoning claim here.

## Measured request transport

The lower native API supports arbitrary allowed masks and repeated pins. The timed
service adapter in this study admits available singleton pins, matching the retained
X01 request bank. Wider or empty masks are covered by API correctness tests, not by
this performance claim.
