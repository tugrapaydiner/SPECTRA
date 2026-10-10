# Quotient application continuation v2

Parent commit: `ad3efe269c6a81df394dcf01ce01f407621769a0` on
`research/quotient-query-20261007`.

This branch replaces the invalid application continuation that was based on the
wrong source ancestry. The failed setup run `37825612437` is preserved: its ancestry
guard correctly stopped before any solver or performance experiment. No guard is
weakened here.

## Upstream inventory correction

Pinned upstream repository: `marijnheule/clicolcom`, commit
`4932048642da2144f387961b595112277afff82f`.

That exact tree contains eight WAP files, all in the A family:
`wap01a.col` through `wap08a.col`. It does **not** contain a WAP-B family. Every
previous WAP-B plan, result, workflow description, or flagship inference is retired
as unsupported. The failed data run `37846001615` is preserved: it fetched
`wap01a.col`, then correctly failed on the nonexistent `wap01b.col` before any
solver was run.

Because the `wap01a.col` bytes were already opened during that failed acquisition,
WAP01a is development data. WAP05a and WAP06a are also development data by the
pre-existing protocol. Subject to a successful byte-identity audit and no evidence
of prior access, WAP02a, WAP03a, WAP04a, WAP07a and WAP08a remain prospective
application holdouts.

## Scientific boundary

The target claim is narrow: exact repeated list-colouring support queries on fixed
optical-network conflict graphs. It is not a claim of general reasoning, universal
graph-colouring superiority, a learned model, algorithmic novelty, or application
adoption.

All preparation, retained solver state, every query, complete immutable witness
materialization, independent original-graph/list/query checking, and disposal must
be charged in session cost. Process startup, imports, file reads and graph parsing
are measured separately. Query repetitions and timing rounds are nested within a
graph and are never counted as independent graph evidence.

Development may use only WAP01a, WAP05a and WAP06a. Candidate source, competitor
source identities, query generator, budgets, analysis, failure taxonomy and gates
must be frozen before any remaining holdout bytes are opened by a solver workflow.
Any change inspired by holdout observations requires a genuinely new application
family or a separately justified new prospective study.

Strong controls retain learning where their public API permits it and receive the
same original query universe. At minimum: the uncompiled exact executor, the cheap
direct-domain solver, native MiniCard, native CaDiCaL, simplifying CaDiCaL, and any
predeclared portfolio used in the headline comparison. An optimistic post-hoc oracle
cannot serve as the primary comparator.

With at most five independent holdout graphs, report every graph individually and
use graph-clustered uncertainty. Do not manufacture precision by resampling queries
or rounds as independent problems. A numerical pass is at most a strong application
result. Independent-team replication, precise prior-art positioning and measured
external use remain separate flagship gates.
