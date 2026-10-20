# Part 3 direct 2-SAT consolidation history

Date: 2026-10-20

This archive commit preserves both histories that were consolidated into the clean
Part 3 branch.

- `f36183dd3c859abfa6b2eccf859a3b2599875b3b` — the mature native 2-SAT challenge,
  including the complete 28-cell x86-64/ARM64 development result and cross-platform
  contracts.
- `80f904b3d655ab16c2079b8e0b8d776600522d25` — the discarded duplicate challenger,
  temporary repair workflows and branch-cleanup attempts.

The active branch was rebuilt from clean Part 3 commit
`65dc9ff1f7c32e3528e1635af58a4f761ef514af` as one additional commit:

`b033e13382b8bf7955c65906b234d2962c315369` — Challenge the exact service with direct 2-SAT

The active result intentionally retains the negative pure-service outcome. The
purpose-built challenger was faster overall, while SPECTRA retained strong wins on
dense-compression cells. No losing cells or failed attempts were deleted from Git
history; the redundant branch names may be removed after this archive ref is
verified.
