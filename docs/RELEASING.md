# Releasing SPECTRA

A clean source tree, passing tests, a successful benchmark, and a published release
are separate states. This checklist prepares a maintenance release; it does not
approve broader scientific claims.

## Source and version

Release from the exact reviewed commit on `main`. The existing v0.7.1 tag and assets
must not be moved or replaced. Change both `[project].version` and
`spectra.__version__` in one intentional release commit.

## Required local acceptance

```bash
python -m pytest tests/public --confcutdir=tests/public
python scripts/check_active_tree.py --out active-tree.json
python scripts/check_release_readiness.py --out release-readiness.json
python scripts/audit_workspace.py --out workspace-audit.json
python -m build --outdir dist/current
python scripts/check_current_installation.py \
  --dist dist/current --out install-check
spectra exact selftest release-selftest
spectra exact audit release-selftest
spectra evidence release-selftest release-selftest/SHA256.json
```

Run the self-test in a fresh disposable directory, then audit it in a separate
fresh process. Generation compiles the bundled exact runtime, covers SAT and all
three portable UNSAT proof kinds, independently replays the answer banks, rejects
a mutation, retains the exact source used for compilation, and binds every artifact
in a portable manifest. Audit refuses an already-loaded native module, recomputes
the source/binary identities and replays the frozen packets without native
execution. These are semantic release checks, not a performance result or an
independent scientific replication.

A release containing `spectra.exact` must also pass the dedicated Linux x86,
Linux ARM64, Windows, and sanitizer matrix, plus the installed exact and exam-repair
checks. Each installed environment must retain its complete self-test bundle. The
matrix must then run `spectra exact compare` over those bundles and retain the
cross-platform receipt. Platform-specific native binaries may differ; the exact
source, packets, certified outputs, proof kinds and selector receipts may not.
Report excluded slow tests and disabled leak detection explicitly.

## Repository boundary

The active tree is capped by `scripts/check_active_tree.py`: no large historical
archives, retired transport directories, stale branch-trigger workflows, or files
larger than the declared limit may be added. Durable research evidence belongs in a
release or a hash-bound workflow artifact. Historical bytes remain reachable from
`archive/research-history-20261016` and their original commits.

## Publication

Require successful checks for the exact final merge source. Retain the commit, tree,
wheel, sdist, environment, complete per-platform self-test bundles, cross-platform
comparison receipt, and SHA-256 receipts. Creating a tag or draft release is not
proof that publication completed; download and verify every final asset before
announcing it.

The latest published package remains
[v0.7.1](https://github.com/tugrapaydiner/SPECTRA/releases/tag/v0.7.1) until a
separate versioned release is intentionally made.
