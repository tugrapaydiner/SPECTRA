# Prospective SPECTRA quotient-query confirmation commitment

**Committed before generating any new confirmation graph or query.** This is a hash-bound commitment to an unpublished local source snapshot, not a claim that the remote branch has a complete runnable checkout.

- Local implementation commit: `22547102c2da8528171024d29bd6c9b15f061eeb`.
- Local freeze commit: `181a14dc932d2089752c8b609fb41773162031c4`.
- SHA-256 of exact `experiments/quotient_query/FREEZE.json`: `132e8b12164f9bee108a78d14c9248205790827cdffbdae6d4c405757861030e`.
- Prospective master seed: `3581664344273647893`.
- Dataset: **72 fresh** graph sources, two synthetic wire families (cycle and triangle-tree), lengths 64/192/576, 12 per family/length; 64 independent restrictions per source.
- Controls: direct-domain, cheap parity quotient, no compilation, hybrid ablation, exact native MiniCard, native CaDiCaL, simplifying CaDiCaL, and sixteen-model CaDiCaL cache. Native SAT solvers retain per-session learning and receive the future query-address universe.
- Primary: each system must independently verify every complete SAT witness on all 64-query sessions, in three shuffled rounds; no dropped timeouts or failures.
- Numerical criterion: upper stratified paired 95% case-bootstrap bound on candidate / per-case-fastest conventional mean complete 64-query session **<=0.50**, and upper p95 session ratio **<=1.10**. Both also required against every fixed conventional arm. 2,000 bootstrap draws with seed 931405; resample original graphs inside family/size strata, keep repeats together.
- One CPU thread, 4-GiB address space, 64-MiB declared native array bounds, two-second session deadline, original verification, preparation, all 64 checked full materialized answers and disposal charged.
- No tuning, adjustment, solver replacement, or selected retry is permitted after opening prospective evidence.
- Previously consumed graphs are development. The source registry includes 243 known prior cases and 12 previous repeated-query IDs whose graphs are unrecoverable. We do not claim a comprehensive ancestor-isomorphism proof.
- Passing these numerical tests would demonstrate **a narrow synthetic specialist result**. It would *not* establish novel algorithms, general learned reasoning, ordinary-host portability, external researcher replication, or field adoption.

The precise local Git commits, full source and freeze, saved preflight, pinned external upstream hashes and test receipts must accompany any results. This document is a public hash commitment only, not publication of those bytes or an independent preregistration authority.