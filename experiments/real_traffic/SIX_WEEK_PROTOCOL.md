# Six-week temporal confirmation

This is the prospective follow-up to the exposed X01 development work.

## Frozen evidence

The six traffic weeks are `X04`, `X08`, `X12`, `X16`, `X20`, and `X24`. They may be opened only after this protocol and the one-week workflow are public on the `real-traffic` branch.

Every week must use the exact same route parser, plan construction, outcome-blind request generator, native SPECTRA service, persistent native MiniCard control, native CUDD control, independent original-input checker, and contradiction-proof checker that passed on X01. No solver, query, ordering, or threshold change is permitted after the first later week is downloaded.

Each week is evaluated in chronological, reverse-chronological, and fixed shuffled order. The primary comparison uses the slowest SPECTRA order and the fastest MiniCard order for that week. This is deliberately adverse to SPECTRA.

## Fixed gates

All six weeks must complete without missing requests, invalid witnesses, missing contradiction proofs, or session deadline overruns.

The candidate passes only if all of the following hold:

- SPECTRA has lower mean complete-session cost than MiniCard on every week.
- The upper exact week-clustered 95% bound on the pooled mean-cost ratio is at most `0.50`.
- The upper fixed-seed week-clustered 95% bound on the pooled p95 ratio is at most `0.90`.
- The result survives all three query orders and every chronological quarter.

The exact mean interval enumerates all `6^6 = 46,656` week resamples. The p95 interval uses 20,000 whole-week draws with seed `118603`.

## Claim boundary

The Abilene topology, fixed routes, and traffic values are measured public data. The wavelength plans and protection/migration policy are disclosed research constructions, not operator-deployed configurations. Passing this study would support a narrow repeated-support-query systems claim, not general SAT superiority, general AI reasoning, production adoption, or physical optical-network feasibility.

A failed gate remains a failed gate. These six weeks will not be retuned or replaced.
