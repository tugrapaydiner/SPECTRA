# Changelog

## Unreleased — release maintenance

### Final review and README

- Bound JSON read requests to 64 KiB chunks; reject unrepresentable byte caps as
  settings errors instead of exposing an OverflowError traceback. Preserve the
  byte limit, strict decoding rules and solver/model arithmetic.
- Add regression tests for extreme caps, exact multichunk boundaries, Unicode,
  duplicate keys and at-most-limit-plus-one consumption.
- Rebuild the README around current interfaces and source-bound, reproducible
  SVG charts. Retain regressions, missing controls and failed research gates.
- Correct stale merged/unmerged status for PR27/28 and native-platform guidance.
  Include README figure assets in the source distribution.

### Earlier maintenance

- Reject ambiguous or nonfinite witness/manifest JSON, invalid UTF-8, excessive
  nesting and input above an explicit byte limit; preserve solver trajectories.
- Select and validate one current compiler-free wheel rather than hard-coding a
  version in test/publication commands. Retain installed-only package checks.
- Gate publication on an actual package-version change, using the isolated
  version-intent fix from PR27, without importing its runtime experiments.
- Check source-version agreement and current-guide local navigation in CI.
- Separate API, development, release and security guidance; label old test
  receipts and open experiments explicitly. Include guides/examples in the sdist.
- Retain research implementations, original tests, configurations, protocols,
  raw evidence, negative outcomes and historical import paths.

This is not a new release, retraining result or scientific promotion. The source
version remains unchanged until an explicit versioned release is authorized.

## 0.7.1 — 2026-09-12

Published as [SPECTRA 0.7.1](https://github.com/tugrapaydiner/SPECTRA/releases/tag/v0.7.1).
Added opt-in indexed/prepared CNF search with preserved seeded paths and an
output-only CPU inference interface with checked final-output equivalence.
Measurements retain small-case regressions, increased cold allocation and the
`UNKNOWN` large-case outcomes. No external SAT superiority or learned-capability
improvement was established. See [the efficiency guide](docs/EFFICIENCY_GUIDE.md).

Earlier packaging and compact-state work is described in
[current status](docs/STATUS.md) and the [history index](docs/history/README.md).
