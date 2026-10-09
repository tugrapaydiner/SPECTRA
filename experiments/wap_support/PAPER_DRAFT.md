# Certified Original-Address Compilation for Repeated List-Colouring Support Queries

**Draft status:** result-complete technical manuscript outline. Numerical values are bound to the frozen one-shot holdout. Literature positioning and exposition remain open to external review.

## Abstract

Repeated constraint queries over a fixed structure create a familiar offline/online trade-off: a one-time compilation can reduce online latency, but only if it preserves the caller’s original variables, produces complete checkable answers, and amortizes against strong persistent solvers. We study this trade-off for repeated list-colouring support queries on optical-network conflict graphs. SPECTRA constructs a quotient of binary colour choices using strongly connected components of the implication relation, retains a lift from quotient choices to every original vertex, and answers restrictions addressed to arbitrary original vertices. Each query returns a complete immutable original-graph colouring and is checked by an independently owned original-input observer. The compiler can also emit an integer-only certificate that a separate standard-library verifier uses to reconstruct and validate the entire rewritten relation.

We prospectively froze five previously unopened WAP-A conflict graphs, 1,024 deterministic satisfiable support queries per graph, seven implementations, three query orders, complete cost accounting, and graph-clustered acceptance thresholds. The primary control is persistent native MiniCard with a compact one-Boolean encoding for every original binary list and learned-clause retention across the complete query sequence. All 105 sessions and 107,520 full answers completed and passed audit. SPECTRA used 36.35% of MiniCard’s mean complete-session time (exact whole-graph 95% interval 34.68–37.29%) and 36.47% of its p95 cost (33.82–38.32%), winning on all five graphs. Mean complete 1,024-query latency was 569.3 ms versus 1,566.1 ms. The result is narrow: availability lists and restrictions are synthetic, all graphs belong to one public benchmark family, the method loses for short sessions, and equivalent-literal substitution and knowledge compilation are established foundations. The evidence nevertheless establishes a substantial prospectively confirmed systems advantage for an exact, certified, original-address repeated-query compiler.

## 1. Introduction

A solver invoked repeatedly on the same structural instance faces a choice. It can rebuild or incrementally reuse a general representation, or it can compile invariant structure into a specialized online representation. The latter can be much faster, but it is scientifically easy to overstate: a reduced representation may silently change the query language, omit answer reconstruction, compare against a solver that discards learned state, or report only an inner kernel rather than complete checked outcomes.

This work imposes a stricter contract. A fixed graph has an allowed colour list at each original vertex. A query adds restrictions at arbitrary original vertices. The system must answer satisfiability, return a full original-vertex colouring, and pass an independent checker over every original edge, list, and query restriction. Compilation may merge equivalent choices internally, but the external address space cannot change.

SPECTRA’s compiler targets instances dominated by binary lists. It builds the implication relation induced by those choices, identifies equivalent literals through strongly connected components, and constructs a quotient with an explicit lift to every original variable. When the compiled quotient has no remaining cross-component implications, a native support executor answers query restrictions directly. General quotient search remains available for non-arc-free cases. A separate verifier can reconstruct the original implication graph and check the complete quotient relation without executing the native compiler.

The principal experiment asks one question:

> On fixed WAP optical-conflict graphs with synthetic two-colour availability lists, does exact original-address SCC compilation reduce the complete cost of answering 1,024 distinct satisfiable support queries relative to persistent native MiniCard?

The study was frozen before acquiring five holdout graphs. It retains all outputs, costs, failures, build identities, and optional certificate audits. Its main contribution is therefore both a compiler/runtime contract and a prospectively confirmed complete-pipeline result.

## 2. Contributions

1. **Original-address implication quotient.** The compiler merges implication-equivalent binary choices while retaining exact translation of restrictions addressed to any original vertex.
2. **Complete answer lifting.** Every successful query produces an immutable colour byte for every original vertex. No quotient-only or lazy answer is counted as solved.
3. **Independent relation certificate.** An integer-only certificate exposes original-to-quotient mapping, palettes, restrictions, and rewritten arcs. A separate verifier rebuilds the original implication SCCs and checks the entire relation.
4. **Native arc-free support executor.** For a certified quotient with no residual cross-quotient implications, the online runtime evaluates support restrictions over compiled component choices and reconstructs complete answers.
5. **Strong complete-cost comparison.** Persistent MiniCard and CaDiCaL controls retain solver state across queries; MiniCard uses one Boolean per original binary choice rather than a handicapped one-hot encoding.
6. **Prospective WAP holdout.** Five graph blobs, source, workload generation, schedules, budgets, analysis, and gates were committed before graph acquisition. The frozen gate passed on every graph.
7. **Negative and boundary evidence.** SPECTRA loses at short query sequences and previously lost on a compiler register-interference transfer panel. These failures remain part of the claim boundary.

## 3. Problem definition

Let \(G=(V,E)\) be a fixed undirected graph, with an allowed list \(L_v\subseteq C\) for each original vertex \(v\in V\). A query \(R_t\) supplies further allowed-colour masks for a small set of original vertices. The task is to find a total colouring \(f_t:V\to C\) such that

\[
 f_t(v)\in L_v\cap R_t(v),\qquad
 f_t(u)\ne f_t(v)\ \text{for every }\{u,v\}\in E,
\]

where \(R_t(v)=C\) for unmentioned vertices. A valid system must either return such a complete colouring or an honest non-solution status. The experimental workload contains satisfiable queries by construction, so every primary call is expected to return a checked witness.

The persistent-session objective is

\[
T_{\mathrm{session}} = T_{\mathrm{decode}} + T_{\mathrm{prepare}}
 + \sum_{t=1}^{1024}
 (T_{\mathrm{query},t}+T_{\mathrm{materialise},t}+T_{\mathrm{check},t})
 + T_{\mathrm{dispose}}.
\]

Process launch and imports are measured separately as cold wall time. Graph download, case generation, source compilation, evidence compression, and optional independent certificate verification are disclosed but excluded from the primary persistent-session contract.

## 4. Exact compilation

### 4.1 Binary choice encoding

A vertex with two allowed colours is represented by one Boolean choice. An edge sharing one of those colours induces a binary exclusion, which becomes an implication pair. Singleton lists become fixed literals. Wider lists remain explicit core choices.

### 4.2 SCC equivalence

If two literals imply each other in the binary implication graph, they are logically equivalent. The compiler assigns a quotient choice to each complementary SCC pair, rejects a component containing both a literal and its complement, and stores how each original vertex’s colours map onto quotient sides.

This step uses established equivalent-literal substitution. The contribution is not a new SCC algorithm. The relevant contract is that arbitrary later restrictions on original variables can be translated exactly and complete original assignments can be reconstructed.

### 4.3 Rewritten constraints

Original conflicts crossing quotient choices become forbidden target masks indexed by a selected source atom. Conflicts internal to one quotient choice remove inconsistent sides from its initial domain. Wider vertices retain their original palettes. The resulting relation is searched through reversible domain reductions or, when it contains no residual arcs, through the specialized support executor.

### 4.4 Full witness lifting

After a quotient assignment is selected, the stored original lift maps every quotient side to the appropriate original colour. A separately owned observer checks:

- output length and colour range;
- every original list;
- every original edge;
- every current original-address restriction.

The observer does not reuse the quotient adjacency, native search state, or a previous validity result.

## 5. Independent compilation audit

The certificate verifier is deliberately implemented separately from the native compiler. It:

1. validates the original graph and lists;
2. reconstructs the original binary implication graph;
3. computes SCCs iteratively;
4. checks that every merged original choice belongs to the claimed SCC;
5. checks complement orientation and invertible lifting;
6. independently rewrites every original edge into quotient constraints;
7. compares the complete palettes, initial domains, atom offsets, arc offsets, and forbidden masks.

The certificate therefore validates the whole compiled relation, not merely sampled solver outputs. It does not prove the correctness of the search implementation, compiler, Python runtime, or hardware. Those are supported by exhaustive small-system tests, differential tests, sanitizer runs, and original-input witness checking.

Certificate generation and independent verification averaged materially more than a normal service session in earlier development. It is consequently reported as an optional audit cost rather than silently included or advertised as free.

## 6. Experimental design

### 6.1 Data

The WAP graphs are public optical-conflict benchmark topologies pinned to `marijnheule/clicolcom@4932048642da2144f387961b595112277afff82f`.

Development: WAP01a, WAP05a, WAP06a.

Prospective holdout: WAP02a, WAP03a, WAP04a, WAP07a, WAP08a.

All five holdouts form one benchmark family and count as five independent graph clusters, not thousands of independent queries.

### 6.2 Synthetic lists and queries

A deterministic DSATUR colouring partitions each graph. Consecutive triples of DSATUR colour classes are assigned cyclic binary lists `{a,b}`, `{b,c}`, and `{c,a}`. A fixed generator selects eight original vertices and one allowed colour per selected vertex. Persistent MiniCard admits the first 1,024 unique satisfiable restrictions within a frozen attempt budget. Using MiniCard for admission is conservative in the primary MiniCard comparison.

These lists and restrictions are synthetic. The experiment does not claim to reproduce operational optical-network availability traces.

### 6.3 Systems

- **SCC:** exact implication-equivalence quotient, complete lifting, original checking.
- **Hybrid:** parity contraction followed by SCC compilation.
- **Parity:** cheap same-list parity contraction.
- **None:** same native framework without quotient contraction.
- **MiniCard:** persistent native MiniCard with compact original binary-list encoding.
- **CaDiCaL:** persistent native CaDiCaL control.
- **Cache + MiniCard:** fixed 16-model cache with fully checked hits and MiniCard fallback.

Query orders are forward, reverse, and one fixed shuffle. Every arm starts from a fresh case and runs all queries persistently within a session.

### 6.4 Hardware and budgets

The one-shot run used one pinned CPU from an AMD EPYC 9V45 host, Ubuntu 24.04, Python 3.13.16, GCC 13.3, one numerical thread, a 4 GiB address-space limit, admitted native payload limits of 512 MiB, and a 60-second outer session deadline.

### 6.5 Prospective gate

The frozen result passes only if:

- all 105 sessions and 107,520 answers complete and audit;
- the exact graph-clustered upper 95% mean ratio bound is at most 0.50;
- the upper p95 ratio bound is at most 1.10;
- SCC has lower mean complete cost on every graph.

All \(5^5=3,125\) resamples of whole graphs are enumerated exactly. Queries and timing orders are never resampled as independent evidence.

## 7. Results

### 7.1 Primary complete-session result

| Quantity | SCC | MiniCard | Ratio |
|---|---:|---:|---:|
| Mean complete session | 569.290 ms | 1,566.119 ms | **0.363503** |
| p95 complete session | — | — | **0.364735** |

Exact graph-clustered intervals:

- mean ratio: **[0.346789, 0.372944]**;
- p95 ratio: **[0.338157, 0.383226]**.

The predeclared gate passed.

### 7.2 Per-graph results

| Holdout graph | SCC mean | MiniCard mean | SCC / MiniCard |
|---|---:|---:|---:|
| WAP02a | 376.220 ms | 1,105.848 ms | **0.340209** |
| WAP03a | 910.548 ms | 2,411.080 ms | **0.377652** |
| WAP04a | 944.368 ms | 2,572.329 ms | **0.367126** |
| WAP07a | 297.408 ms | 848.619 ms | **0.350461** |
| WAP08a | 317.905 ms | 892.720 ms | **0.356109** |

SPECTRA won on every unopened graph.

### 7.3 Other strong controls

Aggregate complete means:

| Arm | Mean complete session |
|---|---:|
| SCC | **569.290 ms** |
| Hybrid | 575.860 ms |
| Cache + MiniCard | 1,457.250 ms |
| MiniCard | 1,566.119 ms |
| Parity | 1,568.495 ms |
| No contraction | 1,581.250 ms |
| CaDiCaL | 1,609.582 ms |

The fixed 16-model cache did not explain the gain. Query-order changes also left the principal result stable: forward, reverse, and shuffled complete means differed only modestly within each arm.

### 7.4 Internal versus complete time

The in-worker service mean was approximately 154.3 ms for SCC and 1,134.4 ms for MiniCard. The frozen headline remains the stricter complete-process measurement, where process/import overhead reduces the apparent advantage to 2.75×. This distinction prevents an inner-loop speedup from being presented as the deployed result.

### 7.5 Structural compression

Independent certificates reported:

| Graph | Original binary vertices | Quotient vertices | Cross-quotient arcs |
|---|---:|---:|---:|
| WAP02a | 2,464 | 183 | 92 |
| WAP03a | 4,730 | 227 | 198 |
| WAP04a | 5,231 | 283 | 256 |
| WAP07a | 1,809 | 94 | 120 |
| WAP08a | 1,870 | 97 | 112 |

The result is therefore associated with substantial implication-equivalence structure rather than a universally trivial query bank.

## 8. Robustness, failures, and ablations

### 8.1 Query-order robustness

All three fixed query orders preserve the large gap. No favorable order is selected for the headline.

### 8.2 Short-session crossover

Development measurements showed that compilation does not amortize immediately. SPECTRA lost to the strongest control for one query and remained weaker or marginal at short sequences before winning decisively at 64–1,024 queries. The claim is explicitly a persistent repeated-query claim.

### 8.3 Negative transfer

On 13 public compiler register-interference graphs with artificial lists, MiniCard previously beat the SCC implementation by approximately 3.9×. An exact vacuous-search repair narrowed another configuration’s deficit but did not produce a win. Those results motivated the WAP-specific hypothesis and remain evidence against general superiority.

### 8.4 Certificate overhead

An application requiring a fresh independent compiler audit for every compiled graph may not amortize at 1,024 queries. The certificate audit is a verification option, not an uncharged part of the primary online result.

### 8.5 One-family evidence

Five holdout graphs provide genuine unopened problem instances but only five independent clusters from one benchmark family. The exact interval is rigorous for that fixed family; it is not a universal population claim.

## 9. Related work and novelty boundary

Equivalent-literal substitution through binary implication SCCs is established SAT preprocessing. Incremental SAT under assumptions is established. Knowledge compilation explicitly studies paying offline cost to accelerate online queries. Wavelength assignment through graph colouring is established.

SPECTRA does not claim those primitives. Its demonstrated contribution is the integration of:

- exact original-address restrictions after quotienting;
- complete witness lifting;
- independent certificate of the entire rewritten relation;
- specialized native support execution;
- strong persistent controls;
- prospectively frozen complete-cost evidence.

A systematic review has not yet proven that no prior system exposes the same end-to-end contract. The safest scientific description is a novel systems integration and empirical result, pending expert comparison with any closer predecessor.

## 10. Reproducibility

Immutable identifiers:

- freeze commit: `0af6fda36c60a110584e7db836ee5724e530fbd4`;
- freeze SHA-256: `31997d38cced5e5611f3f17dc6bc3929b21dcc92a6b7bc81a78db31aab949020`;
- open commit: `b7bca7fb7f62fcb0a43e6b33705cb834cd64494c`;
- one-shot workflow: `37883619073`;
- artifact ID: `11595284861`;
- artifact ZIP SHA-256: `5410db4d5832544c3ec55b19c1449395020319516d20823038de66216b45527c`.

The artifact contains source, raw graph bytes, deterministic cases, complete outputs, schedule, build/environment receipts, certificate audits, analysis, and manifests. A post-run independent manifest check verified all 382 entries and recomputed the published statistics.

Project-owned cross-environment reruns are labelled reproduction, not new confirmation. GitHub issue #55 defines what would count as independent-person/team replication.

## 11. Limitations and future work

1. Replace synthetic lists with operator- or simulator-derived wavelength availability traces.
2. Replicate on ordinary workstation/laptop CPUs and another architecture.
3. Obtain independent-team reproduction with raw public receipts.
4. Compare against any prior compiler preserving arbitrary original-address support queries.
5. Measure compile/use lifecycle and cache persistence inside an actual network-planning service.
6. Establish admission rules predicting when the quotient will be sufficiently arc-free to amortize.
7. Investigate proof-producing compilation correspondence beyond the current independent semantic certificate.

## 12. Conclusion

The study establishes one important narrow result: exact implication-equivalence compilation can materially reduce the complete cost of a persistent original-address list-colouring support workload. The result survives unopened instances, strong persistent native controls, complete witness materialisation, original-input checking, order changes, full cost accounting, and whole-graph uncertainty. It is appropriate as SPECTRA’s flagship research result when reported with this exact scope. It is not evidence that the broader SPECTRA architecture, general graph colouring, or SAT solving is universally superior.
