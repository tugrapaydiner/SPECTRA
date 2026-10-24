# Active results and archived evidence

Keep compact result summaries, required evidence and small regression fixtures in
the active tree. The retained [SAT pilot](sat_workload/pilot.zip) is a negative-result
fixture whose witnesses and corruption paths are checked by current tests.

[HISTORICAL_ARCHIVES.json](HISTORICAL_ARCHIVES.json) records the exact original
paths, byte counts and hashes of material retired from the active checkout:

| Manifest section | Material | Immutable recovery source |
|---|---|---|
| `files` | 22 bulky source/evidence archives | `archive_commit` at the manifest root |
| `retired_development_scaffolding` | 27 CPU replay logs, raw rows and local drivers | The section's `archive_commit` |
| `retired_validation_logs` | Nine execution logs, JUnit reports and environment snapshots | The section's `archive_commit` |
| `retired_research_documents` | 13 historical neural reports, registers and run requests | The section's `archive_commit` |

Recover each file with `git show <archive_commit>:<path> > <destination>` in a
separate historical workspace and verify its recorded SHA-256. The
[history guide](../docs/history/README.md) explains the archive branch and links
to the complete historical documentation.

Frozen inventories describe their original source trees. Reproduce those runs
from the pinned tree; an old inventory is not an inventory of the current checkout.
Archival preserves negative results, original gates and claim limits. It changes
neither the measured outcomes nor the retained source and regression coverage.

Keep new temporary logs, test output and benchmark dumps in ignored local storage.
Deliberately published research evidence needs a durable archive with exact hashes;
workflow artifacts alone expire. Do not commit bulk evidence under this directory.
