# Direct 2-SAT competition challenge

**Status:** frozen before the first timing run  
**Scope:** exposed development evidence; not a new holdout or outside replication

## Question

Can SPECTRA's public `auto` endpoint beat a separately implemented, purpose-built
2-SAT service when both consume the same immutable `SBC1` packet and return a
complete independently replayable `SCQ2` bank?

This is the closest missing algorithmic control. Every admitted SPECTRA exact
problem has one or two colours per original vertex and therefore admits a direct
2-CNF implication representation. General CDCL, cardinality and BDD controls remain
important, but they do not replace this comparison.

## Independent challenger

`twosat_baseline.cpp` is a separate implementation. It does not call or copy a
SPECTRA runtime object, SCC identifier, closure matrix, cached decision, witness or
proof search. Its only shared boundary is the public byte contract.

For each packet it:

1. parses and validates `SBC1` independently;
2. constructs the original-literal implication graph;
3. computes SCCs and the condensation DAG;
4. uses a dense bitset closure when admitted by the memory cap and a sparse DAG
   traversal otherwise;
5. treats original singleton domains and current singleton restrictions as units;
6. returns full original-colour labels for SAT;
7. returns `SBP1` empty-domain, current-unit path, or static-contradiction evidence
   for UNSAT; and
8. serializes a complete `SCQ2` bank that the standard-library verifier replays.

The implementation is project-owned. A favorable result is therefore a matched
algorithmic challenge, not independent scientific confirmation.

## Frozen development inputs

The first matrix uses only the already exposed `wap05a.col` and `wap06a.col` case
files. Their topologies are public WAP graphs; their two-colour lists and accepted
restrictions are synthetic and already consumed. Prefixes of 1, 8, 64 and 1,024
requests are measured. No result from this matrix may be described as a new
holdout.

## Arms

- `auto`: public structural-v10 selector;
- `raw`: explicit SPECTRA raw engine;
- `scc`: explicit SPECTRA SCC engine; and
- `twosat`: the independent direct 2-SAT challenger.

Every arm receives identical packet bytes. Every first output and every retained
output identity must replay from the original packet. Decision sequences must
match. SAT witnesses may differ when multiple colourings exist.

## Timing boundary

One timed call starts immediately before the Python-to-native call and ends after a
complete immutable output byte string is owned by Python. It includes packet
parsing, implication/SCC/closure construction, all decisions, full labels or
proofs, serialization and native disposal. It excludes packet creation, native
compilation, process startup and offline Python replay; those costs are retained
separately.

Very short calls are calibrated into repeated complete calls until a sample lasts
at least 2 ms, capped at 128 calls. Per-call time is the batch duration divided by
the batch size. Thirty-two repetitions use complete four-position blocks: one
seeded shuffle per block followed by cyclic rotations, so every arm occupies every
position equally. Ratios are paired by repetition; the reported cell ratio is the
median paired `SPECTRA / twosat` ratio.

## Frozen gates

All correctness, replay and decision-agreement checks are mandatory. Performance
gates apply to public `auto`:

- geometric mean `auto / twosat <= 0.90` across the four 64/1,024-query cells;
- every 64/1,024-query cell `<= 1.10`;
- every 1/8-query diagnostic cell `<= 1.25`; and
- zero direct-baseline assignment fallbacks.

A failed gate remains a failed result. The baseline, cases, thresholds, schedule or
negative cells must not be edited after timing to manufacture a pass. Any later
optimization requires a new protocol and must retain this first result.

## Required environments

The first execution must run on Linux x86-64 and native Linux ARM64 with CPython
3.13. Functional contracts must also pass under Windows/MSVC and under UBSan and
ASan+UBSan. Cross-platform timing ratios are reported separately and are not pooled.

## Claim boundary

A pass would establish only that SPECTRA beats this direct project-owned 2-SAT
implementation on the exposed fixed-graph request banks and timing boundary above.
It would not establish universal 2-SAT, SAT, graph-colouring, one-query, production,
novelty or external-adoption superiority. A failure identifies the strongest next
optimization target and must remain visible.
