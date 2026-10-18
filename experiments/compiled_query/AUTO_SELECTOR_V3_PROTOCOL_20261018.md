# Automatic engine selection v3 — frozen protocol

**Frozen:** 18 October 2026  
**Base:** `main@ac71b3a7777f4869424247df6481ef32f9996a6b`

## Question

Can `mode="auto"` select the right exact engine from immutable problem and request-bank geometry, without consulting answers or measured runtimes, while staying close to the fastest explicit engine across short, long, compressible, weakly compressible, base-infeasible and heavily pinned banks?

Issue 1 has already established that explicit SCC is substantially faster than explicit raw on long, feasible WAP banks and that raw remains the correct conservative choice for short, base-infeasible and heavily pinned banks. This protocol covers only the selection problem. It does not reopen or relabel the Issue 1 measurements.

## Allowed selector inputs

The selector may use only values available before solving the current bank:

- request count and total pin count;
- original vertices, edges, implication arcs and palette size;
- whether the fixed graph is already contradictory;
- SCC count and the resulting contraction ratio;
- condensation-DAG arc count, depth or width computed during normal construction;
- singleton-domain count;
- maximum and mean pins per request; and
- deterministic integer arithmetic over those values.

It may not use query answers, SAT/UNSAT counts, proof sizes, measured timings, machine identity, environment variables, retry results or graph names.

## Explicit oracle

The comparison oracle is the fastest paired median among the explicit modes:

- `raw`;
- `scc`;
- `sparse`; and
- `route`.

All explicit modes and `auto` must agree on every decision and every SAT label. Different explicit modes may emit different but independently valid UNSAT paths. `auto` must be byte-identical to the explicit mode reported by `info['selected_mode']` for the same immutable packet.

The oracle is an evaluation device only. It is not available to the runtime selector.

## Development panel

The exposed 48-case structural panel remains the tuning set:

- eight topology families;
- six fixed realizations per family;
- 128 or 512 vertices;
- three domain regimes, including base-infeasible cases;
- bank sizes `1, 63, 64, 65, 256, 1024`;
- five paired repetitions after two warmups; and
- deterministic mode-order rotation.

These cases are already exposed and may be used to choose a simple structural rule. Their results cannot be called confirmation.

## Prospective confirmation panel

Confirmation cases use a new seed namespace and are generated only after the selector implementation and thresholds are committed.

- eight topology families;
- three new realizations per family;
- 256 or 1024 vertices;
- feasible low-pin, feasible high-pin and deliberately base-infeasible regimes;
- bank sizes `1, 63, 64, 65, 256, 1024`;
- seven paired repetitions after three warmups; and
- deterministic mode-order rotation.

The confirmation generator may not reject, repair, filter or regenerate a case based on outcomes or timings. Every scheduled case remains in the audit.

## Application guardrails

The already fixed application roles remain unchanged:

1. WAP05 and WAP06, 1,024 requests: `auto` must choose `scc` and remain within 3% of explicit SCC on each architecture.
2. ITC set9, set10 and set12, 1,024 requests: `auto` must not be more than 5% slower than the fastest of raw and SCC by paired median on any case. A selector change may choose SCC if it is structurally justified, but it may not use case names or outcomes.
3. Fixed-graph contradictions: `auto` must choose raw and reuse the fixed contradiction proof.
4. Banks below 64 requests: `auto` must choose raw unless a separately frozen short-bank rule is introduced before confirmation.

## Correctness gate

Before any timing statement:

- every scheduled process must complete;
- every explicit mode and `auto` must agree on every SAT/UNSAT decision;
- every complete label and portable proof must pass independent original-input replay;
- `auto` output bytes must equal the output bytes of its reported explicit mode;
- source, native binary, cases, packets, outputs and timing records must be hash-bound; and
- no timeout, crash, UNKNOWN, malformed packet, invalid witness or invalid proof may be omitted.

Any correctness failure blocks the performance claim.

## Performance gates

Ratios are `auto complete time / explicit-oracle complete time`; lower is better.

### Prospective confirmation

- geometric mean across all confirmation cases and counts: `<= 1.03`;
- geometric mean within every topology family: `<= 1.05`;
- worst individual case/count paired-median ratio: `<= 1.10`;
- no count bucket may have a geometric mean above `1.05`; and
- at least 90% of case/count cells must select the actual median-fastest explicit mode, with ties defined as explicit medians within 2%.

### Development diagnostic

The same statistics are reported on the exposed tuning set, but a development pass alone cannot close the issue.

### Selection overhead

When `auto` selects the same native engine as an explicit arm, its paired-median ratio to that arm must be `<= 1.03` on every application guardrail. This prevents dispatch or metadata work from erasing the selected engine's gain.

## Interpretation boundary

A pass establishes a robust automatic selector for the exact binary-domain API across the declared structural and application banks. It does not establish universal solver dominance, one-query superiority beyond the fixed raw choice, production p99, outside-team replication, or transfer to unrelated constraint classes.

Thresholds, case roles, seed namespaces, counts and repetitions are not relaxed after confirmation is opened. Negative cells remain in the final report.
