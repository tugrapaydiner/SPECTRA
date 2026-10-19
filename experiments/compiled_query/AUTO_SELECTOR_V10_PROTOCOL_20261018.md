# Automatic exact-engine selector v10 — frozen protocol

## Question

Can a query-blind structural dispatcher stay within five percent of the fastest
explicit exact engine on at least 95% of a fresh confirmation panel, while
preserving the established WAP SCC win and the ITC raw fallback?

## Frozen correction

The v9 panel is now development evidence. It identified two causes rather than a
random failure: Python-side raw dispatch added measurable overhead to tiny calls,
and every small dense bank was forced to raw even when it had no SCC compression.

V10 therefore makes exactly two executable changes:

1. `auto` always enters native mode 4; Python no longer duplicates dispatch or
   rewrites selector metadata.
2. Dense high-pin banks below 96 vertices remain raw only when they are actually
   compressed (`components < 2 * vertices`). Uncompressed long banks may use SCC.

All other structural boundaries, exact formats, decisions, witnesses, proofs,
resource limits and explicit engines remain unchanged.

## Fresh confirmation

The confirmation namespace, generated cases and timing schedule are new. The
panel contains eight query-blind regimes across all eight established topology
families and request counts 1, 8, 32, 64, 128 and 1,024. It is project-owned
prospective confirmation, not outside replication.

The unchanged confirmation gates are:

- global realized-regret geometric mean <= 1.04;
- selected-engine regret geometric mean <= 1.025;
- every family/count group geometric mean <= 1.06;
- at least 95% of cells <= 1.05 regret;
- worst cell <= 1.10;
- every family auto/raw geometric mean <= 1.03;
- zero structural-rule, output, replay or decision failures.

Separate frozen guards must also pass on x86-64 and ARM64:

- WAP long banks select SCC and retain the Issue 1 advantage;
- ITC set9/set10/set12 preserve the correct engine choices;
- automatic dispatch overhead remains within the existing paired limits;
- full public, native, Windows, sanitizer, package and regression matrices pass.

No threshold or case may be changed after the v10 timing run begins.

The original run and its failures remain archived. Revalidation changes
only measurement power and position balance as specified in
`AUTO_SELECTOR_V10_REVALIDATION_20261019.md`; cases, selector rules, and
acceptance thresholds remain unchanged.
