# Active result summaries

This directory contains compact, human-auditable summaries and small fixtures that
are still exercised by the current test suite. Bulky nested source archives,
workflow downloads, generated figures, and superseded evidence ZIPs are not kept in
the normal checkout.

Their exact pre-cleanup bytes and SHA-256 identities are listed in
[`HISTORICAL_ARCHIVES.json`](HISTORICAL_ARCHIVES.json). They remain reachable from:

- branch `archive/research-history-20261016`;
- the original immutable commits named by each result report; and
- the GitHub Actions artifacts cited by the relevant protocol, subject to the
  artifact retention period.

The sole retained archive in the active tree is
[`sat_workload/pilot.zip`](sat_workload/pilot.zip). It is a small negative-result
fixture whose witnesses and corruption paths are independently checked by current
unit tests. New large evidence must be published as a release or workflow artifact,
not committed under `results/`. Recover an old archive only in a detached or
isolated worktree, for example with `git show archive/research-history-20261016:<path>`.
