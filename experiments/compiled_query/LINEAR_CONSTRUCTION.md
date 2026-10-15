# Linear implication construction from canonical original edges

Continuation base: `098a8477f6cc8bb696423e78c4d6b6a26a95a0e1` (`faster-queries`).
The user's automatic policy, prepared-root proof reconstruction, direct witness
serialization, proof scratch reuse and Windows support are preserved.

## Invariant and proof

Both admitted input paths supply a simple undirected graph as strictly increasing
lexicographic pairs `(u,v)` with `u < v`. The tuple path normalizes and deduplicates
before construction. Both independent SBC1 parsers reject noncanonical edges.
Each vertex has one or two distinct original colours.

For any fixed vertex `v`, its incident subsequence of the original edge stream
visits neighbours less than `v` in increasing order, followed by neighbours
greater than `v` in increasing order. Each colour shared by an edge's endpoints
produces two implications: choosing that colour at either endpoint forbids it
at the other. For a fixed source literal and a fixed neighbour, there is at most
one outgoing implication; the analogous statement holds for incoming implications.
Thus the edge-generated targets in every forward CSR row and sources in every
reverse CSR row are already strictly increasing by literal number.

A singleton has a synthetic unused high literal and a unit arc from that literal
to its real low literal. No conflict implication leaves its synthetic high literal
or enters its real low literal. Consequently that unit arc is the only entry of
its forward source row and reverse target row; prepending the unit arcs does not
break either ordering. There are no self edges and no duplicate implications.

Stable counting-and-scattering CSR construction on the unsorted global stream
therefore yields arrays *identical* to construction after lexicographic arc sort
and deduplication, for both forward and reverse graphs. This preserves SCC IDs,
base assignments, path traversal order, routing trees and serialized certificates.
The downstream condensation/bridge sorts remain: this is not a claim that the
entire routed constructor is linear-time.

The independent observer separately derives one forbidden literal pair for every
original edge/shared colour. Those pairs are unique; its verifier scans all pairs
rather than binary-searching them. Dropping that list's sort does not drop any
constraint. Its independently owned original edge list remains sorted for proof
hop membership tests. No solver state is passed to the observer.

For `n` vertices, `m` canonical edges and `a <= 4m+n` implications, implication
emission and both CSR builds now take `O(n+a)` time and `O(n+a)` storage. The removed
comparison sort was `O(a log a)`; no claim is made about an empirical speedup until
complete-call measurements finish. This is an implementation specialization of
established 2-SAT machinery, not a new 2-SAT algorithm or priority claim. See
Aspvall, Plass and Tarjan (1979), DOI 10.1016/0020-0190(79)90002-4.

## Validation, fixed before the new comparison

A diagnostic-only `SPECTRA_EXACT_VALIDATE_LINEAR_CSR` build reconstructs the old
sorted arrays and compares *all four native arrays and the arc count* on every
construction. It must never be used for timing. The focused tests include all
three-vertex graphs/domain combinations, singleton and bit-63 domains, randomized
larger graphs, noncanonical tuple inputs, and malformed binary packet rejection.
Existing exhaustive, MiniCard, proof-mutation, memory, and concurrency contracts
remain in force. Compare complete SBC1/SCQ2 outputs against the immediate parent
for all five explicit/automatic modes. No format, ABI, resource cap, default,
engine-selection threshold, or admission rule changes.

Reuse the user's seven-arm harness with baseline commit `098a847...` in full,
48 exposed structural graphs, bank lengths 1/63/64/65/1024, nine paired repetitions,
two warmups, identical host/compiler flags, and original-input replay. Keep the
existing 10% auto-versus-candidate-raw target unchanged and report its outcome even
if common raw improvements make that relative target harder. Also report each
candidate against its immediate-parent counterpart, including previous auto.
All cases, errors, regressions and nonpassing runs are retained. These are exposed
development measurements, not a fresh confirmation or outside replication.

```bash
python -m experiments.compiled_query.query_latency run --out linear-comparison \
  --baseline-commit 098a8477f6cc8bb696423e78c4d6b6a26a95a0e1 --include-baseline-auto
python -m experiments.compiled_query.query_latency audit --out linear-comparison
```
