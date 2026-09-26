# Contributing

Start with [Development](docs/DEVELOPMENT.md) and open a focused pull request.
State whether the change is maintenance, a protocol, an experiment or a proposed
promotion. Include the exact tested commit/tree, commands, results and exclusions.
A faster implementation is not automatically a better learner.

## Evidence and compatibility

Keep old raw observations, failed attempts, split decisions, checkpoints,
configurations and numerical tolerances unchanged. Never use a cleanup to remove
unfavorable data or reopen sealed confirmation inputs. Preserve old module paths
because saved checkpoints and replay code use them. Add narrowly scoped modules
instead of mass-formatting or renaming evidence-bound implementations.

Run the history-based workspace audit before proposing deletion or relocation.
Already archived files have immutable receipts in [the history index](docs/history/README.md).
Do not force-push over advanced branches or delete unmerged experimental work.

## Tests and claims

Public changes need dependency-light tests and an actual installed-wheel check.
Model/native changes additionally need their declared CPU environment, semantic
and numerical contracts, and applicable retained-evidence replays. New tests are
included in the overall count, not added twice. State skipped/slow exclusions and
keep initial failures with subsequent fixes.

Use original-clause/task checkers and strong applicable classical/compiler
baselines. Separate preparation, cold execution, warm calls and total charged
cost. Payload bytes, Python allocations and peak RSS are different measurements.
Keep synthetic tests distinct from benchmark observations. See
[Releasing](docs/RELEASING.md) before changing a version or requesting publication.
