# Part 2 automatic selector v10 — consolidation revalidation

Date: 2026-10-19

This is the single canonical continuation of Part 2. The v10 source
was rebuilt on the released `main` baseline, while every superseded
branch head was retained in the recovery archive before branch
cleanup.

## Corrections

- Public and research API documentation now describes the executable
  v10 policy and its 96-vertex boundary consistently.
- ITC development failures and true holdout failures are reported
  separately; the strict all-case selector guard remains unchanged.
- The confirmation benchmark retains every original case and gate.
  Sub-10-microsecond cells now use calibrated batches, complete
  four-position timing blocks, and paired per-repetition ratios.
- The original failed evidence remains archived and is not relabelled
  or deleted.

## Acceptance rule

Part 2 remains unmerged until the complete x86-64 and ARM64 selector,
WAP, overhead, ITC, public API, sanitizer, packaging, platform, and
repository matrices pass on one source identity. No threshold may be
relaxed to obtain a pass.
