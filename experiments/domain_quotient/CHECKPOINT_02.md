# Domain quotient checkpoint 02 — development, not promotion

The candidate now stores one <=64-bit domain per vertex, admits original graph/list
constraints directly, and uses reversible singleton propagation with optional exact
binary residual closure. A bounded three-level priority bitset preserves the scan's
minimum-domain, static-degree and vertex-id ordering exactly. It adds 8*n index bytes
for rank permutations and an explicitly counted priority payload. This is a measured
implementation of established methods, not a new learning algorithm or novelty claim.

## Completed local development

Baseline: 679004e, plus the remote c8b65d0 application-control acquisition. All 242
cases are exposed development or public application-admission: 85 Sudoku, 72 planted
and 72 uniform old graphs, and 13 register-allocation graphs in four source families.
No new performance-confirmation input has been generated or opened.

Two complete studies are retained: 7,476 observations in the first full matrix and
7,620 in the five-round priority study. Every SAT output is independently checked
from original edges and allowed masks. Native MiniCard is pinned unmodified core
79776615ddc8803dd86803ea6f00838fc74349b1 with a bulk original-graph adapter, as well as
its ordinary PySAT interface. The anchor control fixes a maximum-degree vertex to
zero only for unrestricted lists. The igraph DSATUR control is original igraph 1.0.0.
The old arm label `igraph_mcs` actually means igraph's `colored_neighbors` method,
NOT its separate maximum-cardinality-search API. Preserve raw labels and disclose
this correction; use an unambiguous name in subsequent studies.

Priority study: mean complete in-memory graph-to-checked-answer milliseconds:

| Panel | Domain buckets | Same policy scan | Bulk MiniCard | Native igraph DSATUR |
|---|---:|---:|---:|---:|
| Register graphs (all 13 solved) |0.821033|1.207741|17.543127|2.026363|
| Sudoku (all 85 solved) |2.714449|2.009652|1.233159|not list-capable|
| Planted (all exact arms solve 72) |0.193743|0.200722|0.205143|49/72 solved|
| Uniform (22 SAT; other reports unproved) |0.229609|0.219662|0.182434|7/72 solved|

Register p95 is 1.664120 ms versus scan 3.536665 and igraph DSATUR 4.769497.
These are single-server DEVELOPMENT measurements, not population superiority,
clean confirmation or field adoption. The four dependent register-source families
cannot be treated as thirteen unrelated applications. File parsing, compilation,
module imports and process launch are outside this timer. All preparation, search,
original-constraint checking, fresh labels and disposal are included. A shared
one-second supervisor deadline and four-GiB address-space guard apply. Failed calls
remain in the matrix. Native work counters are not equivalent compute units.

Buckets improve this register panel, but worsen Sudoku. On small planted graphs,
MiniCard's anchor control is slightly faster than the candidate (0.189645 ms), so
PR54's demanding external speed gate is NOT overturned. Binary closure has not
established a useful benefit here. Extra complexity requires further justification.
A compiled cheap greedy/MCS control remains an important next comparison.

The first study used a 1,000,000-work candidate guard and solved only 79/85 Sudoku.
The second development study uses 100,000,000 and retains all previous failed calls;
it is not a selective confirmation rerun or a changed wall deadline. Work-cap failure
is not an invalid witness. Public API defaults remain explicit and unchanged.

## Validation and actual limits

647 public tests PASS locally. All 63 new tests PASS with the actual native library
under UBSan. The bulk MiniCard adapter separately passes 13 contracts. Tests include
all five-vertex graphs, random vertex lists, all three-vertex two-choice assignments,
exact scan/bucket path matching, hierarchy boundaries, 64 colours, malformed inputs,
work/payload guards, original checker rejection and concurrent ownership.

One pre-existing test expectation for the new module initially failed because rank
permutations intentionally raised index storage; it now tests the exact 20*n+8*m+4
payload formula. The original failure is retained. No solver tolerance was relaxed.
No full historical neural/slow suite, ASan, consumer-host, new installed-wheel or
outside researcher result is claimed at this checkpoint. No model/default/release
or historical source is replaced. Local source/evidence archives and Git commits
retain both studies and the failed comparison arms.
