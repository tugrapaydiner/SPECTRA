# Prior-art audit for certified original-address support compilation

**Status:** focused, reproducible literature audit—not proof that no equivalent system exists. The correct novelty claim remains an end-to-end systems contribution until domain experts and peer review identify or exclude closer predecessors.

## Question

Has prior work already demonstrated the following complete contract?

1. a fixed list-colouring instance is compiled once;
2. binary choices may be quotient-merged through implication equivalence;
3. later restrictions may still address **any original vertex**;
4. every query returns a complete original-variable witness;
5. a separately implemented verifier checks the **entire compiled relation**, not only output samples;
6. the complete persistent-query pipeline is compared prospectively against a strong solver that retains learning.

A paper need not use SPECTRA's terminology to qualify. Work on CSP compilation, SAT preprocessing, graph/list colouring, graph quotients, incremental SAT, wavelength assignment, and model reconstruction was considered.

## Search procedure

Searches used combinations of:

```text
list coloring repeated queries compilation
list homomorphism knowledge compilation
constraint satisfaction partial compilation online queries
incremental SAT graph coloring assumptions
modular decomposition list coloring
true twins false twins list coloring quotient
wavelength assignment graph coloring repeated requests
SCC equivalent literal substitution model reconstruction
original variables knowledge compilation conditioning
support queries graph coloring
```

Primary or author-hosted sources were prioritized. Citation chains from the closest sources were inspected conceptually; this repository does not claim a systematic-review guarantee over every non-English, unpublished, proprietary, or uncatalogued implementation.

## Closest foundations

### 1. General knowledge compilation

Darwiche and Marquis, **A Knowledge Compilation Map**, *Journal of Artificial Intelligence Research* 17 (2002), 229–264, DOI `10.1613/jair.989`.

This establishes the offline/online viewpoint and evaluates target languages by succinctness and supported queries/transformations. It is direct conceptual prior art for paying compilation cost to answer repeated queries. It does not present SPECTRA's graph/list-colouring quotient, original-address lift, or WAP experiment.

**Implication:** SPECTRA cannot claim invention of the offline/online compilation idea.

### 2. Partial compilation of repeated constraint problems

Andrea Balogh, **Partial Compilation of Constraint Problems**, PhD thesis, University College Cork (2024), URI `hdl.handle.net/10468/18385`.

The thesis explicitly studies repeatedly queried constraint instances and selective/partial compilation when complete compilation is too large. Its focus includes retaining important subsets of solutions and evaluating compactness on circuit, planning, and generated model-counting instances.

SPECTRA differs in the confirmed result by preserving the exact complete relation for its supported contract, retaining arbitrary original-address restrictions, reconstructing every full witness, and independently checking the canonical quotient certificate. These are differences of contract and implementation, not evidence that the general partial-compilation research direction is new.

### 3. Equivalent-literal substitution and SAT preprocessing

Armin Biere, Matti Järvisalo, and Benjamin Kiesl, **Preprocessing in SAT Solving**, in *Handbook of Satisfiability*, 2nd ed. (2021), pp. 391–435, DOI `10.3233/FAIA200992`.

Binary implication graphs, SCC-based equivalence detection, substitution, simplification, and reconstruction are established SAT techniques. Modern CaDiCaL source also contains SCC/equivalent-literal machinery. SPECTRA uses this foundation rather than inventing it.

The distinctive question is whether equivalences can be exposed as a persistent original-address list-colouring service with arbitrary later restrictions, complete lifts, a whole-relation certificate, and a measured end-to-end advantage. This audit did not locate that full contract in the preprocessing literature.

**Implication:** no claim of a new SCC algorithm, new equivalent-literal theorem, or general SAT preprocessor is defensible.

### 4. Static list colouring and list homomorphism algorithms

Jessica Enright, Lorna Stewart, and Gábor Tardos, **On List Coloring and List Homomorphism of Permutation and Interval Graphs**, *SIAM Journal on Discrete Mathematics* 28(4) (2014), DOI `10.1137/13090465X`.

This gives polynomial algorithms for static list colouring/list homomorphism on particular graph classes. Related work studies counting list homomorphisms, fixed targets, bounded treewidth, flexibility under requests, and reconfiguration.

These are important problem-domain predecessors. The located papers do not appear to compile one arbitrary fixed WAP graph into an original-address repeated support-query service or report the same certificate/runtime contract.

### 5. Flexible list-colouring requests

Hemanshu Kaul, Rogers Mathew, Jeffrey A. Mudrock, and Michael J. Pelsmajer, **Flexible list colorings: Maximizing the number of requests satisfied**, arXiv `2211.09048` (2022).

This literature uses “requests” for desired vertex colours and studies what fraction can be satisfied under structural assumptions. It is mathematically adjacent to restrictions addressed to original vertices, but its objective and algorithms differ from exact repeated satisfiability with a complete witness.

### 6. Dynamic graph colouring

Long Yuan, Lu Qin, Xuemin Lin, Lijun Chang, and Wenjie Zhang, **Effective and Efficient Dynamic Graph Coloring**, *Proceedings of the VLDB Endowment* 11(3) (2017), DOI `10.14778/3157794.3157802`.

This incrementally maintains colourings as the graph changes. It demonstrates that repeated graph-colouring workloads and state reuse are established topics. SPECTRA's graph remains fixed while list restrictions change; it answers exact support queries rather than maintaining a heuristic/minimum-colour solution under edge updates.

### 7. SAT-based graph colouring

André Schidler and Stefan Szeider, **SAT-boosted Tabu Search for Coloring Massive Graphs**, *ACM Journal of Experimental Algorithmics* 28 (2023), DOI `10.1145/3603112`.

SAT has long been used inside colouring systems, including repeated local subproblems. This is relevant practical prior art for comparing against strong solver reuse. It does not establish SPECTRA's exact query contract.

### 8. Wavelength/frequency assignment as colouring

Optical wavelength assignment and wireless frequency assignment are established colouring applications. The public WAP graph topologies and their interpretation are not new contributions. Prior algorithms solve offline assignment, topology-specific assignment, or conflict-free colouring variants.

The confirmed SPECTRA workload uses authentic conflict topologies but **synthetic two-colour availability lists and query restrictions**. No operational-network novelty or deployment claim follows.

## Candidate predecessor classes still requiring expert review

The following areas may contain a closer system and should be explicitly invited during peer review:

- constraint-programming explanation/diagnosis systems that compile once and answer many restrictions;
- interactive product configuration with original-variable conditioning and model reconstruction;
- symbolic graph-colouring systems based on BDD, d-DNNF, ZDD, or MDD representations;
- incremental list-colouring/wavelength-assignment software with persistent learned state;
- SAT preprocessors that expose eliminated-variable assumptions after inprocessing;
- CSP quotienting by interchangeable values, neighbourhood substitution, twins, modules, or automorphisms;
- assumption-preserving preprocessing and proof/log formats for repeated SAT queries.

A paper matching all six contract items above would materially narrow or remove SPECTRA's novelty claim even if its performance differs.

## Findings matrix

| Work family | Offline/online compilation | Original-address restrictions after reduction | Complete witness lift | Independent whole-relation certificate | Prospectively frozen strong-baseline result |
|---|---:|---:|---:|---:|---:|
| Knowledge compilation map/languages | Yes | Sometimes, abstractly | Language-dependent | Language-dependent | No WAP result |
| Partial CSP compilation | Yes | Variable-query dependent | Often possible | Not the same certificate located | No matching WAP result |
| SAT SCC/equivalence preprocessing | Preprocessing | Assumptions can be delicate after elimination | Reconstruction established | Proof/reconstruction tools exist | No matching service result located |
| Static list-colouring algorithms | No persistent service in located work | Original vertices | Yes | No matching compiler certificate | No |
| Dynamic graph colouring | State reuse | Different update model | Colouring maintained | No matching relation certificate | No |
| SAT-based colouring systems | Often incremental/local | Encoding-dependent | Yes | Solver checking/proofs possible | No matching contract located |
| SPECTRA WAP result | Yes | **Yes, any original vertex** | **Yes, every query** | **Yes, canonical integer-only relation audit** | **Yes, five unopened graphs** |

The table is a research aid, not a legal novelty opinion or exhaustive bibliometric result.

## Defensible novelty statement

> SPECTRA presents an exact systems integration for repeated list-colouring support queries: implication-equivalent binary choices are compiled while preserving restrictions at arbitrary original vertices; every answer is lifted and checked in the original address space; a separate verifier audits the complete compiled relation; and a prospectively frozen WAP study establishes a large complete-pipeline advantage over persistent native MiniCard.

## Statements that remain excluded

- “We invented SCC equivalence substitution.”
- “We invented knowledge compilation for repeated queries.”
- “This is the first incremental graph-colouring system.”
- “No prior implementation has the same contract.”
- “This is a new asymptotic graph-colouring algorithm.”
- “This solves deployed wavelength assignment faster.”

## Novelty verdict

**Result-level novelty:** credible as a distinctive exact systems contract plus prospectively confirmed empirical result.

**Primitive-algorithm novelty:** not established; its central primitives are prior art.

**Absolute priority:** open pending broader literature review and external expert challenge.
