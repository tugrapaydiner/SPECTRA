# SPECTRA: evidence-led development from the 67/100 review

Date: 2026-10-13. Branch: `research/compiled-query-validation`.

## Baseline and preservation

Public base: `5edab3a14b7e9645cc33c1568e55450f2b15a790`.
The recovered checked-core bundle has local merge `09dbaa9fb9b7ff9f67cae84e21d799ba9266e609`, tree `0ba02630103342a7ed2a4f3b65481ec7853e6719`, and bundle SHA-256 `b3131524feb78b751dddfa6ea47ba0bb5d1de0f4ab4bd79963ee4819ba1697aa`. Recovery is not evidence that those additions were pushed to GitHub.

The starting assessment is 67/100 for the whole research program, 84/100 for the curated engineering track, and 80/100 for the narrow frozen WAP result. These are judgment-based review scores, not measurements or promised future grades.

Do not edit the WAP freeze, relabel exposed graphs as holdouts, hide a failed comparison, weaken a checker, promote the packet service to the default, or erase protected historical evidence. New comparisons belong to a separately named development study.

## Work packages and acceptance criteria

| Priority | Work package | Required evidence before calling it done |
|---|---|---|
| P0 | Audit and recover the actual baseline | Exact source identity, protected-file audit, public regression suite, scoped review of native ownership, contracts, experiment accounting and release/navigation. |
| P1 | Add an independently maintained generic compiled-query comparator | CUDD-backed OBDD on the original CNF, no SPECTRA SCC/quotient or cached answers, explicit unavailable-backend failure, full original-vertex witnesses, separately owned original checker, compilation/query/output/check/disposal charged, exhaustive semantic tests and actual native-backend execution. |
| P1 | Build an auditable development harness | Raw records before aggregate claims; pinned source/environment; output hashes and original checks; timeout/resource failures retained; interleaved paired scheduling; graphs, not queries, treated as independent clusters. |
| P2 | Challenge the cheapest exact alternatives | Matched sparse traversal for migration; persistent native SAT and multiple query-blind BDD orders for WAP; compile-once amortization from one request to long banks; preparation, p95 and dense-case regressions included. |
| P2 | Isolate deployment-format overhead | A new explicitly defined endpoint, not a rewrite of the confirmed JSON endpoint; byte-format validation, full accounting and Python-version tests; fresh freeze only after development choices are fixed. |
| P2 | Broaden final-endpoint evidence | Second topology with outcome-blind workload selection; structural transfer, density and restriction-width variation; unseen confirmation opened once after a committed protocol; no prefix reuse presented as independent data. |
| P3 | Harden portability and resources | Python 3.11/3.13, ordinary laptop and ARM where available; clean installs; sanitizer checks for changed native code; resource caps distinguished from estimates; no broad portability claim from one machine. |
| P3 | Clarify scientific contribution | A focused theorem/contract and closest-work comparison; distinguish established SCC/2-SAT/OBDD techniques from genuinely new contributions; compare against strongest applicable exact methods. |
| P3 | Consolidate the research story | One front-door claim, track-specific limitations, executable replication guide and preserved negative evidence. Review stale branches/PRs before any archival or closure. |
| External | Independent replication | Outside researchers run the committed package and publish full raw evidence. Project-owned CI does not satisfy this gate. |
| External | Operator-defined utility and adoption | An application owner defines the task before seeing outcomes and independently measures a useful improvement in a maintained system. |

## First implementation target

Implement a generic CNF compiler through the independently maintained `tulip-control/dd` CUDD backend. Keep the binary-list encoder, query conditioning, witness reconstruction and original-input observer separate. Test constants, singleton domains, repeated/conflicting restrictions, unconstrained and conditioned-away variables, malformed inputs, lifecycle and failed compilation. The native CUDD arm must never silently become a Python reference arm.

This addresses a missing comparator class; merely adding it does not establish a performance win or close the novelty gate. The two-plan migration service has a stronger minimum-change/proof-path contract than the WAP feasibility study and must not be conflated with this baseline.

## Progress through 14 October 2026

- The independent native CUDD comparator, generic lossless reduction, complete-cost
  harness, exact mixed SAT/UNSAT companion, original-input SBP1 proofs, and
  Python/native replay are implemented and retained with their negative results.
- SBC1 now isolates deployment-format overhead. A ready packet is parsed separately
  by solver and observer and executed as one checked SCQ2 bank. JSON decoding and
  packet preparation remain separately disclosed rather than moved outside the
  narrative.
- The cheap exact `raw` arm remains the primary matched control. Direct proof
  routing is not promoted as a speed result when it ties or loses to raw traversal.
- The next internally actionable gate is cross-platform/ordinary-hardware transfer
  of the final endpoint and an outcome-blind second problem family. The score ceiling
  after that remains external replication, operator-defined utility and adoption.

## Progress rules

Each milestone must retain its exact code, tests, costs and negative outcomes. A failed strong baseline is a result to investigate, not a reason to change the denominator. New development measurements cannot inherit the old frozen confirmation. No numerical score increase is automatic. High-90s claims require independent impact and replication as well as internal engineering.
