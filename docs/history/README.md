# Historical research and evidence

The active `main` checkout intentionally contains only current source, tests,
documentation, compact summaries, and eight permanent workflows. The full
pre-cleanup research histories are preserved without merging their alternate file
trees into `main`.

## Canonical archive

```text
branch: archive/research-history-20261016
head:   0b68898f3e64063ac3dc8904ae42f0b33590e7af
```

The archive preserves the pre-cleanup file tree and history links to retired
research branches. Individual old branch refs were removed only after their tips
were proven reachable from the archive. Earlier receipts may name the prior
archive commit `abd10d3d0ef24300c3e43b8432886c8203fd465b`; retain those immutable
identities when reproducing the corresponding historical result.

Use an isolated checkout when inspecting or rerunning retired work:

```bash
git fetch origin archive/research-history-20261016
git worktree add --detach ../spectra-history origin/archive/research-history-20261016
```

Large evidence archives and generated figures removed from the active checkout are
also present in the historical commits that originally introduced them. Reports in
`docs/`, `experiments/`, and `results/` retain the exact commit, artifact, and hash
identities needed to locate those bytes.

## Boundary

Archival reachability is not promotion. Failed gates, exposed datasets, negative
comparisons, and unsupported claims remain historical evidence. They must not be
copied into current defaults or advertised as results of the proof-carrying exact
query API or the frozen WAP flagship study.

## Historical neural documentation

The old milestone registers, acceptance reports, reviews, research proposals and
completed M17 run request are preserved at Part 3 ancestor
`45b7fba65d0eb702616b1f617b3e4f0017fb4438`. The `retired_research_documents` section
of [the archive manifest](../../results/HISTORICAL_ARCHIVES.json) lists all 13
original paths with their byte counts, Git blobs and SHA-256 digests.

Start with the [historical milestone register](https://github.com/tugrapaydiner/SPECTRA/blob/45b7fba65d0eb702616b1f617b3e4f0017fb4438/docs/RESEARCH_STATE.md)
or [September research review](https://github.com/tugrapaydiner/SPECTRA/blob/45b7fba65d0eb702616b1f617b3e4f0017fb4438/docs/RESEARCH_REVIEW_20260911.md).
Their relative links resolve within that complete historical tree. Recorded
output paths in old protocols also refer to that tree. No milestone decision,
failed result or numerical gate was changed by retiring these active copies.
Use [current status](../STATUS.md) for the present project scope.
