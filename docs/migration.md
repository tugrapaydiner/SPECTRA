# Two-plan migration

`spectra.migration` answers repeated support queries between two valid colourings
of the same conflict graph. Given proper colourings `A` and `B`, every vertex may
keep `A(v)` or move to `B(v)`; a query additionally requires some vertices to move
and others to stay. For each query the service returns either

- a complete assignment containing the **unique inclusion-minimal set of moves**, or
- an **original-edge implication path** proving the request contradictory.

Typical use: moving lightpaths (or any conflict-constrained resources) from a
current plan to a target plan while some must change and others must not.

```python
from spectra.migration import PacketService

service = PacketService.build("build/migration")   # explicit native build
with service.prepare(edges=((0, 1),), plan_a=(0, 1), plan_b=(1, 0)) as prepared:
    result = prepared.solve((((0, 2),), ((0, 2), (1, 2))))
    assert result.audit.sat == 1 and result.audit.unsat == 1
    assert result.audit.minimum_checked
```

Native builds use `g++` and are supported on Linux.

Every returned result has been checked by an independent native observer for
request compliance, the exact two-plan relation, minimality and every
contradiction path.

| Service | Engine | Observer |
|---|---|---|
| `CompactService` | sparse SCC + BFS over the dependency graph | sparse original-input checker |
| `StreamedService` | same engine | streamed checker sharing one immutable request bank |
| `PacketService` | engine over immutable request packets | two independent native packet parsers |

## Why the minimum is unique

Only vertices with `A(v) ≠ B(v)` have a choice. Let `x_v = 1` mean vertex `v` moves
to `B(v)`. Add a directed dependency `u → v` whenever `{u, v}` is an edge and
`B(u) = A(v)`.

On any edge, `A/A` is legal because `A` is proper and `B/B` is legal because `B` is
proper, so only a mixed assignment can conflict. `B(u) = A(v)` forbids
`x_u = 1, x_v = 0`, which is exactly the implication `x_u → x_v`.

**Lemma.** A set `M` of moved vertices gives a proper colouring if and only if `M`
is closed under the dependencies.

A request is a set `S` that must move and a set `T` that must stay. Let
`R = Reach(S)` (including `S`).

**Theorem.** The request is feasible if and only if no restriction is empty and
`R ∩ T = ∅`. When feasible, `R` is the unique inclusion-minimal migration set.

Every feasible move set is closed and contains `S`, so it contains `R`. Conversely,
moving exactly `R` respects both pins and, by the lemma, every edge. Because every
feasible set contains `R`, `R` also minimises any nonnegative per-vertex cost, in
particular the number of changed vertices.

## Why contradictions have small proofs

If `R ∩ T ≠ ∅`, a dependency path leads from a vertex in `S` to a vertex in `T`. A
verifier checks the two endpoint pins and, for each hop `u → v`, that `{u, v}` is an
original edge with `B(u) = A(v)`. No compiled closure or stored answer is trusted.
The native BFS returns a shortest such path; incompatible pins on one vertex and
empty restrictions have one-vertex proofs.

## Complexity

With `n` vertices, `m` edges, `d` dependencies and `r` request pins, a BFS query
costs `O(n + d + r)`, full output `O(n)` and original-edge checking `O(n + m + r)`.
The optional precomputed reachability uses `O(n·⌈n/64⌉)` words and
`O(n(n + d))` setup; it is charged as setup, not hidden.

This solves final-state support between two given plans. It does not choose the
target plan, optimise routes or spectrum, or sequence live reconfiguration steps.
