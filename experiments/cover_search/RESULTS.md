# Native cover search: substantial internal gain, competitive gate failed

**Retain as an experimental backend. Do not promote it as the flagship or claim
native-solver superiority.** Main, defaults, models and all 764 PR53 files remain
unchanged. This tests direct native covering/exclusion constraints and incremental
counts, not learned intelligence or algorithmic novelty.

## Exact study and audit correction

The base is `eabcc9c5a8ebe0beee364da3ecd1a68bedeb7253`. A public hash commitment at
`b2ea0625bf6767933560e2ed292b45c58ea8217a` and complete protocol at
`ee1454a7562561f442f2c674b3535c3c65c7851f` preceded generation of the new graph
inventory. Complete source files were uploaded afterward under those same hashes;
this is a prior public hash commitment, not prior outside source review.

The native source SHA256 is
`0020c52ca4c378328518180766c9e305f48b212a73d8158ef943d623b561251d`;
Python binding SHA256 is
`198e943a28c5e97431fe77eab8a7218ac8a578e7d302912990d4d89000040702`.
All 17 original source/data bindings remain unchanged.

All 6,438 timing jobs and 158 resource probes completed. The frozen analyser then
refused Python tuples versus JSON lists in reconstructed graph edges. Its original
failure is retained. `audit_corrected.py` proves canonical input equality and digest
identity, supplies JSON-native expected inputs temporarily to the UNCHANGED analyser,
then restores its function. Four added regression tests cover nonempty edges,
genuine input changes, restoration on failure and nonfinite JSON refusal.

No measurement, solver, task, threshold or statistical calculation is edited. The
following are **corrected descriptive measurements, not a retroactively clean
confirmation**. The original protocol's strict clean-confirmation requirement is
not marked passed. The primary external gate also fails numerically after correction.
A new clean confirmation would require a genuinely new prospective study.

## Complete latency, milliseconds

One pinned AMD EPYC 9V74 CPU, Linux, Python 3.13.5, GCC 14.2, portable C++17 `-O3`
(no `-march=native`), one numerical thread and a four-GiB address-space guard.
PySAT 1.9.dev15 supplies the five unmodified native controls. MiniCard and GlueCard4
receive native at-most-one groups, not forced pairwise CNF. The candidate is untrained.

There are 85 already exposed public Sudoku compatibility puzzles, plus 144 graphs
new to this continuation: planted/uniform generators, 18/36/72 vertices, 24 per cell.
Graphs are synthetic, not a scheduling deployment. Distinct colour-refinement
signatures exclude graph isomorphism across development/evaluation and within the
new inventory, subject to hash collision resistance. This is not a complete
canonicalizer; a matching signature would refuse, not silently regenerate a case.

Three shuffled rounds are repeated measurements, not additional independent tasks.
Each complete job includes encoding, validation, fresh preparation, search, witness
construction, original constraint/task checks, disposal and diagnostic result creation.
Imports, process creation, file I/O and compilation are outside primary timing;
startup and process memory have separate probes. All arms share a one-second
supervisor deadline. Additional custom-node/DPLL-decision guardrails have different
units, so equal instructions are not claimed. Failures and slow calls are retained.

### Sudoku: every arm solves all 85 tasks in every round

| Arm | Mean ms | p95 ms | Maximum ms |
|---|---:|---:|---:|
| Native MiniCard | 2.8651 | 3.4768 | 8.5904 |
| Native GlueCard4 | 2.9538 | 3.6350 | 5.4394 |
| SPECTRA direct + incremental | 3.0232 | 3.5670 | 9.7280 |
| Same engine, dense counts | 3.5226 | 5.0312 | 7.7435 |
| Original Norvig | 5.9147 | 13.6982 | 31.4405 |
| Same engine, expanded CNF | 12.1181 | 14.2198 | 19.3225 |
| Native Glucose4.2 | 12.4095 | 14.4307 | 28.7419 |
| Native Kissat404 | 13.8429 | 22.7111 | 43.2240 |
| Native CaDiCaL300 | 16.3840 | 21.1351 | 31.9588 |
| Previous SPECTRA Python DPLL | 29.2889 | 42.0956 | 69.3525 |

### Planted graphs: every arm solves all 72 tasks in every round

| Arm | Mean ms | p95 ms | Maximum ms |
|---|---:|---:|---:|
| Native MiniCard | 0.9664 | 1.5928 | 2.5565 |
| Native GlueCard4 | 1.0466 | 1.5810 | 6.6801 |
| Native Glucose4.2 | 1.2377 | 1.9704 | 2.3492 |
| SPECTRA direct + incremental | 1.3486 | 1.8994 | 4.9264 |
| Same engine, dense counts | 1.4090 | 1.9905 | 9.2067 |
| Same engine, expanded CNF | 1.6733 | 2.7417 | 5.8420 |
| Native Kissat404 | 2.3364 | 5.7491 | 9.4379 |
| Native CaDiCaL300 | 3.0191 | 7.0707 | 9.6092 |
| Previous SPECTRA Python DPLL | 7.8397 | 7.0298 | 422.8684 |

On uniform graphs, every arm finds verified SAT on the same 22/72 tasks. The five
native controls report UNSAT on the remaining 50 without proof checking; custom
arms and DPLL return UNKNOWN after exhaustion. These are not certified UNSAT cases
or 50 established algorithmic errors. Mean complete-attempt costs are 1.3485 ms for
cover, 0.9145 for MiniCard and 8.6730 for DPLL. All uniform observations, maxima,
statuses and six size/generator strata are retained in the machine-readable summary.
Uniform return time is not time to a solution on all inputs.

## Effect attribution and uncertainty

Case-stratified paired bootstrap: 2,000 draws, retaining all three rounds and both
arms within each sampled problem. These are single-host descriptive 95% intervals,
not a production tail guarantee or multiple-comparison discovery claim.

| Candidate / baseline | Mean ratio | Descriptive 95% interval | p95 ratio | Descriptive 95% interval |
|---|---:|---:|---:|---:|
| Sudoku cover / DPLL | 0.103220 | [0.097915, 0.109021] | 0.084736 | [0.060801, 0.098773] |
| Sudoku cover / expanded CNF | 0.249479 | [0.243475, 0.256910] | 0.250848 | [0.229697, 0.289218] |
| Sudoku cover / dense counts | 0.858226 | [0.826157, 0.894191] | 0.708984 | [0.606231, 0.890559] |
| Sudoku cover / MiniCard | 1.055169 | [1.024239, 1.091011] | 1.025940 | [0.959754, 1.191106] |
| Planted cover / MiniCard | 1.395482 | [1.308545, 1.522111] | 1.192508 | [0.953291, 2.002232] |
| Planted cover / DPLL | 0.172024 | [0.074613, 0.615368] | 0.270196 | [0.210123, 0.399748] |
| Planted cover / dense | 0.957178 | [0.881752, 1.022169] | 0.954236 | [0.692159, 1.149295] |

Direct constraints cost about one-quarter of the identical engine's expanded-CNF
pipeline on Sudoku. Incremental counts additionally reduce mean about 14.2% versus
dense counts, at the cost of extra counters. Their benefit on the smaller planted
graphs is not clearly separated from zero. This is representation and exact execution,
not learning. The corrected numerical internal criterion passes; that does not erase
the original audit failure. The fixed primary planted MiniCard gate FAILS. Neither a
different baseline nor an adjusted success threshold substitutes for that failure.

## Memory and installed execution

158 fresh-exec probes cover five public puzzles and the first two graphs per stratum.
Their own `/proc` high-water marks avoid inherited supervisor maxima; no probe fails.

| Arm | Probe tasks | Median peak MiB | Maximum peak MiB | Median cold-process ms |
|---|---:|---:|---:|---:|
| SPECTRA cover | 17 | 17.809 | 17.973 | 61.523 |
| Python DPLL | 17 | 17.051 | 20.438 | 61.755 |
| Native MiniCard through PySAT | 17 | 23.430 | 23.730 | 72.565 |
| Native GlueCard4 through PySAT | 17 | 23.375 | 23.730 | 71.187 |
| Original Norvig | 5 | 16.945 | 16.949 | 68.573 |

These are fixed probe samples, not worst-case deployment memory. PySAT loads a
multi-solver library; no claim is made against a standalone tuned MiniCard build.
Cold process time includes imports and one solve, but filesystem caches can be warm.
Index/state caps bound counted payload, not all temporary buffers or process RSS.
No laptop, Windows/ARM, energy, concurrent-service or real-time tail guarantee follows.

## Verification and remaining gates

Local public suite: **467 passed**, including 64 new contracts. The same 64 pass
under UBSan. Coverage includes all 512 small formulas, 500 random noncanonical
formulas, 500 direct-group instances with exhaustive CNF correspondence, strict
budgets, malformed buffers, iterative depth above 1,000, concurrent ownership,
close/use-after-close, and refusal when the original checker rejects a witness.

All **687 custom paths** replay exactly: witness, status, work counts, state/index
payloads and diagnostic fingerprint. Timings are not replayed. The source distribution
was extracted to build the wheel; all **151 shipped source files** byte-match.
Eleven inherited installed checks, 512 cover truth-table cases and three direct
lifecycle checks pass in a clean outside-checkout environment without numerical
frameworks. Compilation is explicit; install/import does not compile. No version,
default backend, model or release is promoted.

The local historical Git-object audit cannot run from the imported source archive;
its missing-object failure is retained. Direct checking verifies all 764 base files.
Metadata/navigation checks pass. Full-checkout CI checks history separately; consult
that exact run's receipt before asserting success. CI also repeats the fixed study
on its own host; that is automated reproduction, not outside-researcher replication.
The original failed analysis remains visible alongside the corrected analysis.

Open gates include: primary competitive advantage, a clean new confirmation,
compiled DLX/CP-SAT controls where relevant, ordinary-hardware measurements,
real application usefulness, learned-model seed/transfer evidence, multi-budget
scaling, independent replication and adoption. The full neural/slow/native-SVM
suites and ASan were not run locally. Five training seeds do not apply to this
untrained mechanism; they remain necessary for future learned-model claims.

Prior art: Knuth, *Dancing Links*, arXiv:cs/0011047, and the official PySAT solver
API. Algorithm X-like branching, bitsets and incremental cardinality counts are
established techniques, not invented here. No new general reasoning claim follows.

Reproduction commands and scope are in README.md and CONTRACT.md. The accompanying
downloadable packet retains complete raw local data, witnesses, source/input hashes,
original audit failure, corrected results, negative development attempts and build/
test/installation receipts. CI artifacts are an additional expiring copy, not proof
of permanent public raw-data archival or independent adoption.
