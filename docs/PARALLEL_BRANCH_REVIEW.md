# Concurrent alternative implementation: disposition

During integration, `research/m16-verifier-aligned-cpu` advanced from the reviewed
`2910d0c8447d03e353b515bc3a57c1cfe5bda839` to
`d0f6c44a41acdfbd163625cd728b290a0107a999`. The branch-head guard stopped the
first consolidation attempt, before publication. The validated patch itself
applied and its compressed/uncompressed checksums matched.

The new commit adds a parallel `spectra_reliability` package rather than changes
to the evidence-backed M16/M17 implementation. Its runtime and native code were
reviewed, including the output-channel AVX2 tiling and additional transposed INT8
weight layout. That alternative may be useful, but its speed and fidelity cannot
be inferred from the different M16/M17 executable sources and records.

The alternative search currently passes parent/root state directly to callbacks
and transitions. An in-place transition can therefore affect sibling paths or
the original root. The integrated `eval/verified_search.py` explicitly isolates
those states and has adversarial regression tests. Maintaining two overlapping
search/value/ancestry APIs without a demonstrated integration benefit is not the
goal of this consolidation.

Decision: recover the already-reviewed telemetry changes at `2910d0c...`; keep
the new alternative implementation on its existing branch, unmodified and
unmerged. Do not delete that concurrently active branch, adopt its AVX2 claims,
or mark its new commit as integrated. The current PR deliberately chooses one
validated implementation instead of blindly merging every open branch.
