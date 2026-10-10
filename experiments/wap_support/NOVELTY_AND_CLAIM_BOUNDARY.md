# Novelty and claim boundary

## One-sentence contribution

SPECTRA compiles repeated list-colouring support queries over a fixed graph by quotienting implication-equivalent original choices, preserves arbitrary original-address restrictions and complete witness lifting, emits an independently checkable whole-relation certificate, and demonstrates a prospectively confirmed complete-pipeline advantage over a persistent native cardinality solver on five unopened WAP optical-conflict graphs.

That sentence is the claim. It is intentionally narrower than “new SAT algorithm,” “new SCC algorithm,” “faster graph colouring,” or “general reasoning.”

## Established foundations

The following ingredients are prior art and are not claimed as inventions:

1. **Implication graphs, strongly connected components, and equivalent-literal substitution.** SAT preprocessors identify equivalent literals and substitute representatives; this is covered in Biere, Järvisalo, and Kiesl, *Preprocessing in SAT Solving*, Handbook of Satisfiability, 2nd ed., 2021, DOI 10.3233/FAIA200992.
2. **Incremental SAT and retained learning across related calls.** Reusing a solver under assumptions is an established way to answer sequences of related SAT queries. The MiniCard and CaDiCaL controls in this study retain their solver state across all 1,024 queries.
3. **Knowledge compilation for repeated online queries.** Darwiche and Marquis, *A Knowledge Compilation Map*, JAIR 17 (2002), DOI 10.1613/jair.989, formalized the offline/online trade-off; Balogh, *Partial Compilation of Constraint Problems* (University College Cork, 2024), studies partial compilation for repeated constraint queries.
4. **Graph colouring formulations of wavelength assignment.** Wavelength-conflict instances represented as graph-colouring problems are established in optical-network research. The public WAP topologies are not newly introduced here.
5. **Compact Boolean encodings for two-choice domains and native cardinality solving.** One Boolean per binary choice and persistent native MiniCard are practical baseline techniques, not handicapped comparators.

## What is demonstrated here

The confirmed result is a systems contribution with five linked properties:

- **Original-address preservation.** Restrictions can target any original lightpath even when its binary choice has been merged into a quotient representative.
- **Complete immutable witnesses.** Every query returns a full assignment, not a lazy handle or a quotient-only answer.
- **Independent whole-relation checking.** A standard-library verifier reconstructs the original binary implication relation and checks the quotient mapping and rewritten constraints without executing the native compiler.
- **Specialized native support execution.** When the certified quotient has no remaining cross-quotient implications, the runtime answers support restrictions through compiled component choices while still materializing and checking the original witness.
- **Prospective complete-cost evidence.** On five unopened WAP-A graphs, 105 sessions and 107,520 answers passed audit; the candidate/MiniCard mean ratio was 0.363503 with an exact graph-clustered upper 95% bound of 0.372944, and the candidate won on every graph.

The contribution is therefore not one isolated primitive. It is the exact, checkable, original-address compiler/runtime contract plus evidence that the contract has substantial value on a frozen repeated-query workload.

## Novelty status

**Established:** the individual SCC, equivalence-substitution, incremental-solving, and offline/online compilation ideas.

**Plausibly original engineering/scientific contribution:** their particular integration into an exact original-address quotient for repeated list-colouring support queries; independent certificate of the entire rewritten relation; full witness lifting and original checking; native arc-free support specialization; and a prospectively confirmed complete-cost comparison against persistent native MiniCard on WAP instances.

**Not yet established by this repository alone:** that no prior system implements an equivalent end-to-end contract, or that the composition is patent-level or theorem-level novel. A publication should include a systematic literature review and invite domain experts to identify the closest predecessor.

## Flagship wording

Defensible:

> SPECTRA’s flagship research result is a certified repeated-query compiler for a narrow list-colouring support workload. On five prospectively unopened WAP optical-conflict graphs, it used 36.4% of persistent native MiniCard’s mean complete-session time across 1,024 fully materialized and independently checked queries, with an exact graph-clustered upper 95% ratio bound of 37.3%.

Not defensible:

- “SPECTRA is 2.75x faster at graph colouring.”
- “SPECTRA beats SAT solvers generally.”
- “SPECTRA invents SCC quotienting.”
- “SPECTRA proves faster wavelength assignment in deployed networks.”
- “SPECTRA is a new general reasoning architecture.”

## Remaining scientific and impact gates

1. Independent replication by a person or team outside the project.
2. Evaluation on authentic operational availability/restriction traces, or a partner-defined workload.
3. Measurements on ordinary non-server hardware and another architecture.
4. Comparison with any identified prior system that preserves the same original-address repeated-query contract.
5. External use showing that the compile cost amortizes in a maintained application.

These gates affect generality, adoption, and confidence. They do not erase the already passed prospectively frozen result, but they constrain how broadly it may be advertised.
