# Certified critical-clique compiler: development protocol

## Exact reduction

Vertices with equal closed neighborhoods form a clique module (a maximal true-twin, or critical-clique, class). Contract every such class `C` to one quotient vertex with demand `w(C)=|C|`. Two quotient vertices are adjacent exactly when their classes are completely joined in the original graph.

For a fixed palette `[k]`, an original proper coloring is equivalent to assigning every quotient vertex `C` a set `S(C) subseteq [k]` such that:

1. `|S(C)| = w(C)`;
2. `S(C) intersection S(D) = empty` for every quotient edge `CD`.

Forward direction: collect the distinct colors used by each clique class. Reverse direction: biject each selected color set to the original vertices of its class. Complete joins make disjointness exactly the original cross-edge constraint. The compiler emits and audits the full partition and reconstruction certificate.

## Frozen development instances

Only `wap01a.col`, `wap05a.col`, and `wap06a.col` are development data. The other WAP-A graphs remain unopened holdouts. Exact chromatic numbers and maximum-clique witnesses are pinned to `marijnheule/clicolcom@4932048642da2144f387961b595112277afff82f` in `KNOWN_WAP_OPTIMA.json`.

## Arms

All arms use the same pinned `python-sat==1.9.dev15` wheel and the same CaDiCaL 1.9.5 backend.

- `direct_assignment`: classical one-hot assignment encoding on the original graph.
- `direct_pop`: compact partial-order encoding on the original graph; this is the strong encoding control.
- `critical_clique_assignment`: exact-cardinality assignment encoding on the audited weighted quotient.

The complete maximum clique is pinned to colors `0..k-1` in every arm. Because an exact maximum clique contains every member of any critical clique it intersects, the quotient receives the identical original-vertex symmetry fixing.

## Primary cost boundary

Every repetition begins with graph bytes and charges parsing, quotient detection/audit, formula generation, native solver construction, solving, model extraction, full witness lifting, original-edge verification, and solver disposal. Arms alternate order between repetitions. Peak process RSS, clauses, variables, solver statistics, source hashes, data hashes, and witness hashes are retained.

## Admission rule

Development can justify a prospective holdout only if:

- every arm returns a fully verified coloring on every graph and repetition;
- the quotient beats the faster of the two direct controls on every development graph;
- the graph-clustered upper 95% confidence bound of the candidate/best-control full-boundary ratio is at most `0.70`;
- no source, threshold, solver, clique witness, or analysis code changes after the freeze.

A numerical pass is a strong systems result, not by itself a flagship claim. Prior-art analysis, unopened holdout replication, and independent external reproduction remain separate gates.
