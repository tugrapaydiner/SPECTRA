# Two-choice kernel checkpoint — correct, no cold-call promotion

The candidate now compiles vertices with at most two allowed colours into an exact
2-SAT implication graph, checks strongly connected components, and propagates
multi-choice decisions reversibly through its component DAG. Only the original
multi-choice core is branched on. Every completed answer is checked against the
original edge and list tuples; exhausted/contradictory results remain UNKNOWN.
This is an implementation of established implication/backdoor techniques, not a
new learned model or an algorithmic-first claim.

The original finite-domain and sparse engines, models, defaults, old source files
and prior failed gates remain unchanged. The kernel's max_search_work is explicitly
a SEARCH counter: construction, SCC preprocessing, checking and output have real
additional cost. All are included in complete-call measurements. Native array
payload guards are not process RSS bounds or hard real-time guarantees.

## Actual local validation

48 kernel tests pass normally and under the actual UBSan-built native module.
They include all five-vertex graphs in four search modes, 4,000 random mixed-list
problems checked by exhaustive enumeration, all triangle list combinations,
several-thousand-vertex binary kernels, mixed cores, iterative depth, strict
resource/type checks, original-checker rejection and repeated/concurrent ownership.
The initial compiler diagnostics were fixed without relaxing -Werror. They remain
in the delivery with the pre-fix source.

MiniCard now also has a stronger exact compact encoding: singleton choices are
constants, binary lists are opposite literals of ONE Boolean, and wider lists
use one-hot/native-cardinality constraints. The solver core remains unmodified
at 79776615ddc8803dd86803ea6f00838fc74349b1. Both original and compact bulk adapters
pass 28 tests including exhaustive graphs, random lists, constants and high bits.
No timing from an invalid adapter is used.

## Completed negative development study

36 new, explicitly DEVELOPMENT planted mixed-list graphs: 128/512/2048 vertices,
0/4/16/64 multi-choice core vertices, three fixed cases per cell, expected degree
6 (exactly 3*n sampled distinct edges), four colours. All other vertices have two
choices, including the hidden planted colour. Edges join unequal planted colours;
no solver filters or replaces a graph. This is synthetic structural research,
not a real compiler workload or independent confirmation.

All 756 calls (36 cases x seven arms x three shuffled rounds) are retained, with
source hashes, exact inputs, every witness, common one-second supervisor deadlines,
four-GiB address-space guard and complete original-graph-to-checked-answer timing.
Imports, file I/O and process creation are outside this in-memory API timer.
All exact arms solve all cases; no errors occurred. Cheap greedy controls fail
some cases, but the kernel does NOT establish a speed advantage over the ordinary
domain engine or compact MiniCard. At n=2048/core=16, for example, kernel mean
1.6620 ms exceeds ordinary no-binary-closure domain 1.2520 ms; compact MiniCard
is 2.0702 ms. These three underlying cases are not a statistical headline.

Decision: no cold-call promotion. The precomputation is not free, and the tested
one-shot instances need too little search to amortize it. Study more demanding
residual structure or persistent-query reuse before spending confirmation data.
The prior register-graph gain is also superseded by the compiled degree-greedy
control (all 13 tasks, ~0.50 ms versus ~0.82 ms domain); retain that failure.
No new confirmation set has been generated or opened.
