# Development and reproduction

## Public tools

Use Python 3.10 or newer in a virtual environment:

```bash
python -m pip install ".[test]"
python -m spectra doctor
python -m pytest tests/public --confcutdir=tests/public
python scripts/check_active_tree.py
python scripts/check_release_readiness.py
python scripts/audit_workspace.py --out workspace-audit.json
```

The base package has no mandatory third-party dependencies; the `test` extra adds
pytest and the pinned SAT wrapper required by public exact-query contracts. Native
builds are explicit and never occur during import.

## Build and test the distribution

Use fresh output directories; the wheel checker rejects stale or ambiguous builds.

```bash
python -m pip install build
python -m build --outdir .release-check/dist
python scripts/check_current_installation.py \
  --dist .release-check/dist --out .release-check/install
.release-check/install/venv/bin/python -I scripts/check_exact_installation.py \
  --out .release-check/install/exact.json
.release-check/install/venv/bin/python -I scripts/check_exam_repair_installation.py \
  --out .release-check/install/exam-repair.json
```

The concise current-build form is:

```bash
python scripts/check_current_installation.py --dist dist/current --out install-check
```

The canonical fast CPU matrix runs on Python 3.11 and 3.13. It executes the active
test suite, native builds, the optimizer/energy smoke test, release checks, and the
active-tree budget. It no longer replays multi-megabyte historical archives on every
pull request.

### Permanent workflow dependencies

All eight permanent workflows pin external Actions to full immutable commit SHAs.
The official major-version refs were verified through GitHub's `git/ref/tags`
API on 2026-10-21:

| Action | Verified major ref | Pinned source |
|---|---|---|
| `actions/checkout` | `v4` | [`11d5960a326750d5838078e36cf38b85af677262`](https://github.com/actions/checkout/commit/11d5960a326750d5838078e36cf38b85af677262) |
| `actions/setup-python` | `v5` | [`a26af69be951a213d495a4c3e4e4022e16d87065`](https://github.com/actions/setup-python/commit/a26af69be951a213d495a4c3e4e4022e16d87065) |
| `actions/upload-artifact` | `v4` | [`ea165f8d65b6e75b540449e92b4886f43607fa02`](https://github.com/actions/upload-artifact/commit/ea165f8d65b6e75b540449e92b4886f43607fa02) |
| `actions/download-artifact` | `v4` | [`d3f86a106a0bac45b974a628896c90dbdf5c8093`](https://github.com/actions/download-artifact/commit/d3f86a106a0bac45b974a628896c90dbdf5c8093) |

The active-tree audit rejects mutable action refs. Updating a pin requires an
explicit source review; a later movement of a major-version tag does not change
the workflow's code. Local actions inherit the checked-out source identity.

## Exact-query development

```bash
python -m spectra exact build exact-native-build
SPECTRA_EXACT_LIBRARY=exact-native-build/_spectra_exact_binary*.so \
  python -m pytest \
    tests/public/test_exact_api.py \
    tests/public/test_exam_repair.py \
    tests/public/test_itc2007_exam.py
```

`ExactRuntime.solve_session_packet` defaults to the structural `auto` policy.
Explicit `raw`, `scc`, `sparse`, and `route` modes remain available for controlled
comparisons. Selection uses admitted structure only, never cached answers or timing
feedback.

The full ITC suite is intentionally manual because it downloads pinned upstream
inputs and performs a long cross-platform evaluation. Trigger
`itc-exam-suite.yml` only when changing the exam adapter, exact engine, or protocol.

## General-CNF reproduction

Follow the frozen inventory, solver field and limits in the
[2026-10-21 exposed campaign](https://github.com/tugrapaydiner/SPECTRA/blob/8f7cee1a9c82e7da34cc2704ff7012290a554a91/experiments/general_competition/EXPOSED_CAMPAIGN_20261021.md).
Use its recorded source commit, acquire the original inputs with matching hashes,
and build source components with the recorded commands. Where the field uses a
bundled prebuilt checker, retrieve that exact artifact and verify its hash; do not
describe it as a local source build. Rebase absolute paths in a separate field
receipt and record the resulting binary hashes; a rebuild is a reproduction with
its own identity. The
[campaign runner](../experiments/general_competition/campaign.py) accepts:

```bash
python -m experiments.general_competition.campaign run \
  --inventory /absolute/path/acquired/inventory.json \
  --solvers /absolute/path/reproduction-solvers.json \
  --out /absolute/path/new-evidence-directory \
  --timeout 5 --checker-timeout 30 --memory-mib 4096 \
  --output-mib 64 --dimacs-mib 64 --compress-evidence \
  --checker /absolute/path/drat-trim
```

Use a new output directory and retain every declared cell, including admission,
timeout, resource, model and proof failures. The field selects separate DSR and
pseudo-Boolean checkers where required; `drat-trim` does not check those formats.
Keep standalone SPECTRA-G results separate from upstream solvers and frontend or
portfolio results. This 96-formula, eight-entry, shared-host development campaign
has short limits and an incomplete roadmap field. It does not establish hidden
confirmation, independent replication, official competition rank, or the 100/100
gate. Preserve evidence beyond expiring CI artifacts as described in
[the release guide](RELEASING.md#repository-boundary).

Admission bounds each decoded physical-line read by the remaining DIMACS byte
budget plus one overflow-probe byte. This prevents unbounded physical-line reads;
gzip may buffer a limited amount ahead. Inputs exactly at the limit remain valid,
with or without a final newline; accepted receipts are unchanged. This is an
input-read bound, not a total process-RSS guarantee.

XZ staging creates its temporary output exclusively and removes only an output
created by that call when reading, decoding or the byte limit fails. An existing
destination is preserved, including when it is the source file itself. Successful
admission receipts and solver behavior are unchanged by this cleanup repair.

The competition loader also binds replay to the admitted decoded byte count and
SHA-256, not just equivalent clauses. It rejects comment/whitespace changes and
growth between admission and replay, then rechecks the original source bytes so
changed gzip/XZ containers with identical decoded content cannot inherit an old
source receipt. The semantic hash definition is unchanged. These checks detect
persistent source changes, not every possible race with a hostile writer; use
immutable inputs for a scientific run. The additional source read is part of the
current loader's work, not a retrofit of any frozen timing measurement.

Native result replay checks both the direct and binary-core paths against the
native ABI's unsigned 64-bit budgets and counters. A conflict-budget `UNKNOWN`
must report exactly one conflict beyond its limit; learned-literal exhaustion
requires a positive conflict count within that limit. A core that already settles
the formula cannot yield a native-budget `UNKNOWN`. Malformed model and DRUP
containers are rejected before counting or replaying their payloads. These are
receipt-consistency checks, not an independent reconstruction of interrupted
search or a solution to the unresolved failure cases. Native solver code, frozen
experiments and retained observations are unchanged.

### Scoring policy and offline evidence verification

New campaigns default to `--scoring-policy declared-inventory-v2`. This keeps
metadata-unknown formulas in their predeclared stratum, reports actual verified
answers separately, and refuses a ranking if checked evidence contradicts the
inventory metadata. Recognized checker allocation failures earn no solve; they
do not become proof disqualifications. Unrecognized proof rejections remain
invalid evidence.

The frozen 2026-10-21 baseline predates this policy field. Its original receipts
must be replayed as `legacy-v1`, including the historical inference of unknown
strata. A corrected table is a separately identified derivation, not a rewrite of
the original campaign. Explicit `--scoring-policy legacy-v1` is available when
reproducing that policy with the current runner.

Verify a downloaded, extracted evidence directory with a separately retained
SHA-256 of its `evidence-sha256.json`:

```bash
python -m experiments.general_competition.campaign_verify /absolute/path/evidence \
  --expected-manifest-sha256 EXPECTED_SHA256 \
  --recheck-models
```

The default verification path executes no commands from the archive. It checks
the exact file inventory, original CNFs, lossless compression receipts, complete
solver/formula rectangle, process records, and recomputed scores under the
recorded policy. Fresh model checks use temporary copies. Fresh proof checks
require explicitly supplied trusted local checker paths; DSR and pseudo-Boolean
formats need their declared interfaces through `TrustedProofChecker`.

Archive consistency does not establish independent replication, hidden-set
integrity, source-to-binary rebuild linkage, or independently measured timings.
Without an external manifest pin, verification establishes internal consistency
only. Keep the verifier report outside the evidence directory so the archived
file inventory remains unchanged.

## Flagship reproduction

The WAP confirmation result is immutable. Follow
[the replication guide](../experiments/wap_support/REPLICATION.md) from the named
frozen source commit. A later run is reproduction, not a second confirmation and
not permission to retune the protocol.

## Historical experiments

Bulky evidence archives and one-shot workflows are stored in Git history and on
`archive/research-history-20261016`, not in the active checkout. See the
[history index](history/README.md). Do not restore them merely to make routine CI
replay old results; use an isolated historical worktree instead. Exact archived
paths and hashes are recorded in [`results/HISTORICAL_ARCHIVES.json`](https://github.com/tugrapaydiner/SPECTRA/blob/38c13990a1ba92229cf5cf2a57b1a600b218889e/results/HISTORICAL_ARCHIVES.json).

## README figures

The active README uses compact SVG assets. Historical PNG figures are archived.
`python scripts/render_readme_charts.py --check` validates the current SVG/chart
sources without running benchmarks.
