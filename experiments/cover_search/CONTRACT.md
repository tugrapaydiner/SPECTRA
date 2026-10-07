# Exact contract and scope

## Input class
`ChoiceProblem(nvars, covers, exclusive)` uses immutable groups of distinct
one-based Boolean variable indices. Each cover requires at least one true
variable; each exclusive group permits at most one. Empty covers are false and
empty exclusive groups are true. A group can appear in both collections, imposing
exactly one. Groups need not partition the variables. This can express exact cover,
Boolean one-hot graph colouring and Sudoku, but is not a general CP/SAT language.

The optional CNF input path admits positive clauses and negative clauses containing
at most two distinct variables after normalization. Literal repeats have Boolean
idempotent semantics. A clause containing a literal and its complement is a
true tautology and is dropped. Other mixed-sign or wider negative clauses are
rejected explicitly, not interpreted approximately or silently ignored.

Variable, group/clause and literal inventory limits are 65,536, 1,000,000 and
16,000,000 respectively. Python and native validation both enforce their relevant
boundaries. Supplied native libraries and the Python process itself are trusted
executable code. The C ABI requires valid accessible buffers of the declared sizes;
it does not make arbitrary pointers or malicious host code safe.

## State invariant and solver argument
Let S be selected variables, A the available variables, and U the uncovered positive
groups. Native indexes store variable conflicts and variable/group membership.
A contains only unselected variables that have not been rejected by a false branch,
a negative unit, or a conflict with S. U contains exactly groups disjoint from S.
For every group g in U, an incremental count equals |g intersect A|. Counts of
already covered groups are irrelevant and are never consulted.

If a remaining group has zero choices, this branch has no satisfying extension.
If it has one, every extension must select that variable. Otherwise choose a
variable in a smallest group. Its true and false branches partition all extensions:
the alternative is a deep state copy with that variable removed from A. The true
branch adds it to S, removes its conflicting variables from A and covers every
group it meets. Removing an available variable decrements precisely the still
uncovered groups containing it. Removing an already absent variable does nothing.
Restoring a saved state restores its corresponding counts as well.

When U is empty, setting exactly S true satisfies every cover and preserves all
exclusions and negative units. Search stops at that point: no subsequent branch
can overwrite this incumbent. Every exposed SAT result is independently checked
against the original input representation. The experiment additionally checks
original graph edges or Sudoku clues/rows/columns/boxes without reusing the encoder.

Each true or false branch removes at least one available variable; depth is finite.
Without resource limits the binary search is complete for the admitted class.
In the implementation, resource limits can terminate it early. Exhaustion is
reported as UNKNOWN, not a proof-certified UNSAT result. This argument is not a
machine-checked verification of the C++ implementation, compiler or hardware.
Truth-table, differential, ownership, malformed-input and UBSan tests supply
implementation evidence within their actual tested scope.

## Dense-count ablation
`incremental=False` recomputes group counts with exact bitset intersections. Both
modes use identical group order, minimum-count rule, variable order, branch order
and stopping conditions. When resource stopping does not differ, their selected
witness, node/decision/propagation/backtrack counts and diagnostic path fingerprint
must agree. The incremental mode stores additional 32-bit counters; it is not a
free memory improvement. The fingerprint is diagnostic FNV64, not a cryptographic
certificate or authentication mechanism.

## Resource and ownership limits
Index admission checks the bit-matrix payload before those matrices are allocated.
The state limit bounds explicitly counted simultaneous active state/stack payload,
including 32-bit counters. It does not bound allocator overhead, spare vector
capacity, original Python inputs, temporary preparation buffers, loaded libraries,
thread stacks or process RSS. `max_nodes` counts search-state inspections, not CPU
instructions, all propagation work, or elapsed time. A supervisor deadline and
separately measured process memory are required for deployment guarantees.

C++ owns its index arrays; temporary Python buffers are not retained. Every search
has its own state. Close and solve share a lock, preventing concurrent destruction
of an active index. Repeated close is harmless and solving a closed index is refused.
The retained native geometry, not a reassignable Python input property, determines
output allocation. Deliberately mutating a frozen source input remains unsupported
and is not an avenue for shrinking the native output buffer.

The builder is explicit and Linux-only. Importing/installing the package does not
invoke a compiler. Output publication is create-if-absent in a trusted directory;
no crash-atomic two-file transaction or hostile-directory guarantee is claimed.
Only the tested Linux hosts are covered by empirical execution/performance claims.

## Relationship to prior work and claims
This combines established ideas: Boolean constraint compilation, Algorithm X-like
minimum-choice branching, bitsets and incremental counts. Relevant primary sources:
Knuth, *Dancing Links*, arXiv:cs/0011047; PySAT's solver API and the native MiniCard
and GlueCard implementations. Exact mathematical constraint preservation does not
establish a new search algorithm, learned reasoning, general SAT superiority,
independent replication or application adoption.

Prebuilt PySAT wheels are practical native controls, not a source-retuned upper
bound on every competitor. There is no compiled DLX or CP-SAT comparison in this
study. Keep all unfavorable native-cardinality comparisons visible.
