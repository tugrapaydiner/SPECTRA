# Native cover/exclusion search

**Decision: experimental, not a competitive promotion.** Direct native constraints
substantially improve SPECTRA's earlier structured solver. MiniCard still wins the
primary external comparison. The original frozen analyser refused JSON tuple/list
identity; the explicitly separated correction verifies the records without changing
source, observations or statistical criteria. This is corrected descriptive evidence,
not a clean confirmation or an improvement in learned intelligence.

Start with [results](RESULTS.md), the [supported contract](CONTRACT.md), and the
unchanged [prospective protocol](PROTOCOL.md). The protocol's original analyser is
preserved, including its defect. Use `audit_corrected.py` for the published corrigendum.

## Run the installed solver

Build and install the source wheel using ordinary Python packaging. It has no
numerical-framework runtime dependency. Then, on tested Linux with a C++17 compiler:

```python
from spectra.cnf.cover import ChoiceProblem, build_cover_runtime, solve_cover

# Each covering group needs >=1 choice. Each exclusion group allows <=1.
problem = ChoiceProblem(
    nvars=4,
    covers=((1, 2), (3, 4)),
    exclusive=((1, 3), (2, 4)),
)
library = build_cover_runtime('/tmp/spectra-cover-build')  # explicit, new path
result = solve_cover(problem, library, max_nodes=100_000)
print(result.status, result.witness)
```

`PreparedCover` offers reusable indexing with independently owned search states.
The default SPECTRA backend is unchanged. An exhausted search returns UNKNOWN,
not an unverified UNSAT certificate. Unsupported CNF structure raises an explicit
error; it is not silently rounded, discarded or delegated to a hidden solver.

## Reproduce the complete fixed comparison

Use the actual research branch, Python 3.13, Linux and a C++17 compiler. Install
`python-sat==1.9.dev15` for the five external controls; it is not a deployment dependency.
A full Git checkout is needed only to reconstruct the pinned ancestor inventory.
The downloadable complete source already includes that inventory.

```bash
python -m pip install 'python-sat==1.9.dev15'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -I -S \
  experiments/cover_search/reproduce.py --out /tmp/spectra-cover-reproduction
```

This one command obtains the hash-bound base inventory from commit
`eabcc9c5a8ebe0beee364da3ecd1a68bedeb7253`, checks every source hash, reconstructs
all task inputs, compiles, executes all 6,438 timing jobs and 158 resource probes,
and derives a separately identified corrected summary. Existing output is refused.
No remote data download, fitting, selective retry or parameter tuning occurs.
Re-running these exposed fixed inputs is replication, not new confirmation.

For retained evidence extracted from the accompanying delivery:

```bash
python -I -S experiments/cover_search/reproduce.py --prepare-only
python -I -S experiments/cover_search/audit_corrected.py EVIDENCE_DIRECTORY \
  --out /tmp/cover-retained-summary.json
python -I -S experiments/cover_search/replay.py EVIDENCE_DIRECTORY \
  --native-dir /tmp/cover-replay-native --out /tmp/cover-replay.json
```

Raw observations, source/data hashes, original audit failure, initial prototypes,
complete build/test receipts and the installed wheel accompany the downloadable
source/evidence packet. CI artifacts are an additional copy, not a promise of
permanent archival availability or external researcher replication.

## Validation boundary

The completed local run passes 467 public tests, including 64 new contracts. The
same 64 pass with the changed native library under UBSan. All 687 custom paths
(229 problems times three implementations) replay exactly, excluding timings.
An actual sdist-built wheel byte-matches all 151 shipped source files and passes
11 inherited installed checks plus 512 exhaustive cover formulas and three direct
lifecycle checks in a clean environment without numerical frameworks.

The local Git-history audit could not run against the imported source archive,
because old Git objects are absent. A direct check verifies all 764 base files
byte-for-byte. The complete-history CI job checks that historical audit separately.
No full neural/slow/native-SVM suite, Windows/ARM, ASan, outside researcher,
production deployment, new release or main merge is claimed by these local checks.
