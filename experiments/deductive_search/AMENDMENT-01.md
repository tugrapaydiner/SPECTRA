# Additional development challenge, before evaluation

The first declared 128-variable development run completed. Both SPECTRA paths
solved all 12 formula/seed cases in each binary/forced family; quality headroom
was zero there. Candidate mean costs were 0.515 and 0.283 times indexed cost.
General 3-SAT had identical outcomes and 2.4–6.8% mean preprocessing overhead.
All original inputs, rows and source hashes remain under development/.

Before any evaluation inputs are generated, add a fifth family: signed,
variable-permuted equivalence rings. Two clauses per edge require neighboring
literal values to agree; two complementary witnesses exist, and no unit starts
propagation. This is a declared structured 2-SAT challenge, not a real-world
workload or a hardness claim against native CDCL. Its purpose is to test a complete
implication method against a capped local walk with actual quality headroom.

Run a second development view with the same first four families/seeds and six new
ring formulas; repeated old inputs are explicitly reused development. Evaluation
retains seed base 202610022000, sizes 128/512, eight formulas/cell and adds eight
ring formulas per size after the original four families. All other protocol
settings remain fixed. Candidate and generator hashes must be frozen before that
80-formula evaluation. Do not change them after viewing evaluation outcomes.

Memory probes use search seed 17 once per case/arm. Timed repetitions use both
17 and 73. RSS worker imports the native comparator for every arm, giving a common
process baseline; report this overhead and do not interpret RSS as standalone
minimal-package RAM. No memory probe is included in timing summaries.

The added development ring cell has headroom: indexed solves 4/12 formula/seed
cases; deduction solves 12/12. Initial full normalization also costs up to 12.7%
mean overhead on planted3 development. Before evaluation, add a cheap three-
distinct-variable screen: if every clause already contains at least three distinct
variables, no root unit or binary-only formula exists, so reuse the indexed path
without constructing a normalized copy. All first three development attempts,
including the slower normalization candidate, remain retained. Final development
has unchanged solve outcomes with 2.4%/3.7% mean general-3-SAT overhead.

Complexity clarification: propagation and SCC passes are linear in their variable/
occurrence/edge inventories. Canonicalization uses per-clause sorting, so total
preprocessing is not claimed strictly linear for arbitrary unbounded clause widths.
All preparation is included in measured complete-call costs.
