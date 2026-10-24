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

## Retired CPU-development replay files

The `retired_development_scaffolding` section of
[`HISTORICAL_ARCHIVES.json`](HISTORICAL_ARCHIVES.json) records 27 raw replay logs,
JUnit reports, comparison-row files and local drivers retired on 2026-10-21.
Every file was checked against the already-existing archive's Git blob identity
before removal. The manifest retains its exact original path, byte count, Git
blob SHA and SHA-256 digest. Compact summaries, source inventories, failures,
and substantive research reports remain in the active checkout.

Recover these files from the immutable archive commit, independently of future
branch moves:

```bash
git fetch origin archive/research-history-20261016
git show 0b68898f3e64063ac3dc8904ae42f0b33590e7af:results/cpu_progress/validation/pytest_fast.log > /tmp/spectra-pytest-fast.log
```

Use the same `git show <archive_commit>:<path>` form for every entry in the
manifest and verify its SHA-256 before replay. This is archival of prior
development evidence, including negative runs; it changes no research result.

## Historical validation logs

The `retired_validation_logs` section of the same manifest identifies nine
execution logs, JUnit reports and environment snapshots removed from the active
tree on 2026-10-24. Their exact bytes were matched to archive commit
`0b68898f3e64063ac3dc8904ae42f0b33590e7af` before removal. Use the recovery command
above and check the recorded SHA-256 when inspecting an original run.

The corresponding outcome summaries and failure records remain available.
Historical inventories still describe their original complete source trees;
reproduce those runs from the pinned tree rather than mixing an old inventory
with the current checkout. Current source, regression tests and benchmark inputs
are unaffected by this archival step.
