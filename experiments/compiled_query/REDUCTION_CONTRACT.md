# Generic reduction control: contract and proof obligations

This is an additional **development comparator**, not a new SPECTRA algorithm,
new confirmation, or a claim of independent-team replication. Its input is the
original generic CNF. Neither SPECTRA's quotient nor its prepared solution is
passed to the reducer. Its CUDD engine is independently maintained upstream;
the reduction and adapter code are implemented in this repository.

## Why this control is necessary

A comparison against an OBDD that retains hundreds of logically equivalent
variables can be much weaker than a comparison against a generic compiler with
ordinary preprocessing. Frontier clause construction and redundant equivalence
seeding alone do not eliminate that gap. The `cudd-reduced-*` arms therefore
substitute equivalent literals and propagate units before BDD construction.
All original-variable witnesses are reconstructed and independently checked.
This tests the end-to-end systems result, not whether a specialized compiler can
beat an unnecessarily redundant representation.

The earlier `cudd-*`, `cudd-frontier-*`, and `cudd-seeded-*` arms remain available.
Their timeouts and negative outcomes are not overwritten. Each changed source
requires a new study manifest; old timing evidence cannot inherit new code.

## Exact transformation

Let F be the original CNF on n Boolean variables. For every original literal l,
the binary-clause implication graph has an edge from -l to r for clause (l OR r),
and from -r to l. A unit (l) contributes -l -> l. Longer clauses are not used as
binary implications. They remain in the formula through substitution.

Within a strongly connected component every literal implies every other. Thus
F entails the equality of their truth values. If a variable and its complement
lie in the same component, the binary subset is inconsistent and so is F.
Otherwise a signed disjoint-set map assigns each original variable either a
representative z or its complement. No query participates in this construction.

Write sigma for this total original-variable substitution. Because all imposed
equalities are consequences of F, substituting them is lossless:

    Models(F) = { lift_sigma(a) : a satisfies F[sigma] }.

Tautological clauses can be removed and identical literals/clauses deduplicated.
An empty clause remains FALSE; an empty formula remains TRUE. Validation happens
before these operations, so malformed literals cannot disappear in a tautology
or after an already discovered contradiction.

Unit propagation assigns only entailed literals of the current substituted
formula. Removing satisfied clauses and false literals preserves exactly the
models that extend those fixed assignments. The implementation uses incidence
lists and an event queue; it does not scan the whole formula repeatedly for each
unit. A conflicting unit or emptied clause means FALSE, not UNKNOWN.

After propagation, every unfixed equivalence class receives a dense Boolean
variable, including classes that appear in no remaining clause. This last point
is necessary: a future request can constrain an otherwise unconstrained original
variable. The retained map contains a signed dense variable or a fixed Boolean
value for **every** original variable.

## Query and reconstruction argument

A current original-literal request is mapped through the same total signed map.
A request that conflicts with a fixed value is infeasible. Repeated or opposing
mapped literals are retained for the compiled engine to reject consistently.
Every original request literal is validated even after a conflict is known.
No request mutates the compiled root or the original map.

The residual formula and mapped assumptions are solved by the explicitly
requested BDD backend. Every successful result is lifted to all n original
Boolean variables; the graph adapter then emits all original vertex colours.
The separate original-input observer checks original domains, every original
edge, and the complete current restriction list. It does not receive the
reduction map. Therefore a faulty mapping cannot be excused by the solver's own
internal consistency check.

Conversely, every original satisfying assignment induces a residual assignment
consistent with the mapped request. Hence a false residual formula means the
original request is infeasible under this reduction. This is a mathematical
contract supported by exhaustive and randomized regression checks, **not** a
machine-checked proof of every Python or CUDD instruction.

## Independence and evidence boundaries

The post-run SAT audit directly checks the original graph in Python. For UNSAT,
it separately constructs the original binary-domain implication relation without
calling the generic CNF encoder, reducer, or BDD. It does not turn a solver
response into a portable proof certificate. The historical SPECTRA arm's UNKNOWN
responses remain UNKNOWN and make a mixed-query session incomplete.

No new holdout is opened. The WAP05/06 case banks are already exposed development
inputs with synthetic domains and SAT-filtered requests. Prefixes and repeated
sessions are not additional graph clusters. Strong performance on those banks
would not establish superiority on unfiltered, changing, or operator-defined
workloads.

## Costs and limits

The complete session timer includes original input decoding and validation,
generic encoding, equivalence construction, substitution, unit propagation,
BDD build, independent-checker build, conditioning every query, both levels of
witness reconstruction, original checking, full SCQ1 output materialization,
output hashing, diagnostics and disposal. Reduced compilation is not given a
precomputed quotient for free. No completed answer is cached.

CUDD is pinned through `tulip-control/dd` at
`4442d498ddea1bec0071315f538cb5ef37443985`. The `frontier` constructor is used for
the residual CNF, with explicit natural, incidence-BFS or occurrence order and
dynamic reordering enabled. The node diagnostic is a soft guard sampled every
64 operation checkpoints, forced before/after construction. It can overshoot
between checks. The study's external wall deadline and Linux address-space
limit are separate; neither the node diagnostic nor RLIMIT_AS is a hard RSS cap.

The earlier implementation's `len(bdd)` calls `Cudd_CheckZeroRef` in the pinned
upstream `dd/cudd.pyx`, rather than reading an O(1) node counter. Per-operation
reference scans therefore cannot be mistaken for unavoidable BDD query cost.
The per-answer original-input checker is never sampled or disabled.

## Remaining challenge

This is a serious generic baseline, not a declaration that every strongest
implementation has been exhausted. Native witness reconstruction, other
knowledge-compilation families, dynamic graph/domain updates, authentic traces,
non-x86 hardware, outside replication and external novelty review remain useful
challenges. A fast Python/CUDD control alone does not close those gates.
