# Releasing SPECTRA

A clean repository, passing tests, a successful benchmark and a published release
are separate states. This guide prepares a maintenance release; it does not
approve unsupported learned-capability, memory, L3-residency or frontier claims.

## Choose the source and scope

Use the intended commit on `main`, after reviewing its exact diff and CI. Keep
experimental PRs separate until independently accepted; [current status](STATUS.md)
records the prepared-runtime/compiler stack. Do not silently fold it into a
cleanup release. Review the [changelog](../CHANGELOG.md) and
[security boundary](../SECURITY.md). Preserve all prior raw evidence and failures.

The existing v0.7.1 tag and assets must not be moved or replaced. This maintenance
branch retains the version until an intentional release-version commit is made.
Update both `[project].version` and `spectra.__version__` together in that commit;
never upload changed bytes under an already published version.

## Local acceptance

Run the complete [development test/build procedure](DEVELOPMENT.md), including
public contracts, fast research regressions, a wheel built from the sdist,
11 installed-only checks, navigation/version checks and the history-based
workspace audit. Retain exact commands, exit statuses, environments and logs.
Native execution and historical evidence replays are additional checks, not
implied by successful wheel installation. Report excluded slow tests explicitly.

```bash
python scripts/check_release_readiness.py
python scripts/audit_workspace.py --out release-workspace-audit.json
```

The first command checks two source versions, nine current guides' local links
and the current-wheel CI entry points. External links and all historical prose
are outside this offline check. Neither command is a security certification or
an approval of the scientific claims.

## GitHub acceptance and publication

Require successful checks for the **exact final PR/merge source**, not an earlier
head. Retain source archives and Git tree hashes alongside workflow receipts.
The full CPU workflow includes checkpoint, fixed-pool, restart, controller and
SAT audits. The indexed-efficiency matrix and output-only checks feed the
versioned evidence packager. Do not relax an existing failed gate to release.

The publishing workflow only acts on a proven package-version transition in a
push to `main`. Editing package metadata without changing the version does not
request publication. It rejects an unknown previous commit, existing tag or
release, mixed source evidence, incomplete checks and ambiguous wheel selection.
It creates a new exact tag, uploads a draft, downloads/checks asset hashes, and
only then publishes. A failed upload can leave a tag/draft: inspect and resolve
that state deliberately; do not force-push or silently overwrite it.

After publication, inspect the real tag, release state, five expected assets and
SHA256SUMS. Save the publication receipt. Workflow configuration or a green PR is
not evidence that a new release exists. PyPI publication is a separate operation
and is not configured by this cleanup.

## Repository presentation and governance

Before presenting the next release, correct GitHub's About description to:

> Verified CNF search and reproducible CPU reasoning research.

The older “Edge-native o1” and “entirely inside the L3 cache” wording is not
supported by the retained evidence. About metadata is separate from README files.
Also review protection for `main`, required successful checks and review policy.
Record actual settings changes; this source cleanup does not itself configure
repository administration, retire branches or certify unmerged experiments.
