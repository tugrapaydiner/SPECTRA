# Documentation

## Flagship research result

| Task | Document |
|---|---|
| Read the confirmed result and exact numerical scope | [Frozen WAP holdout](../experiments/wap_support/HOLDOUT_RESULTS_20261009.md) |
| Inspect the predeclared protocol | [Protocol](../experiments/wap_support/PROTOCOL.md) |
| Review every flagship gate and remaining limitation | [Gate ledger](../experiments/wap_support/FLAGSHIP_GATE_LEDGER.md) |
| Check the novelty wording and excluded claims | [Novelty boundary](../experiments/wap_support/NOVELTY_AND_CLAIM_BOUNDARY.md) |
| Inspect the closest-work audit | [Prior-art audit](../experiments/wap_support/PRIOR_ART_AUDIT_20261009.md) |
| Review portability passes and failures | [Cross-environment reproduction](../experiments/wap_support/CROSS_ENVIRONMENT_REPRODUCTION_20261009.md) |
| Reproduce the canonical compiler certificates | [Certificate reproduction](../experiments/wap_support/CERTIFICATE_REPRODUCTION_20261009.md) |
| Run or independently replicate the exposed study | [Replication guide](../experiments/wap_support/REPLICATION.md) |
| Read the paper draft | [Paper draft](../experiments/wap_support/PAPER_DRAFT.md) |

The WAP result is a narrow empirical systems result. It does not transfer to the
historical neural, MCTS, low-bit, general-SAT, or universal CPU-performance claims.

## Use and maintain SPECTRA

| Task | Guide |
|---|---|
| Install and run the public tools | [Quick start](../README.md#base-toolkit-quick-start) |
| Understand API, input formats and return values | [API and CLI](API.md) |
| Check released, merged, experimental and failed work | [Current status](STATUS.md) |
| Run tests, build a wheel or reproduce experiments | [Development](DEVELOPMENT.md) |
| Prepare a package release | [Release checklist](RELEASING.md) |
| Submit a change | [Contributing](../CONTRIBUTING.md) |
| Handle external inputs and native code | [Security](../SECURITY.md) |
| Review unreleased changes | [Changelog](../CHANGELOG.md) |

## Other measured systems work

[Indexed efficiency](EFFICIENCY_GUIDE.md),
[matched native alternatives](../experiments/native_baselines/RESULTS.md),
[prepared FP32](TRAINED_FP_GUIDE.md), [packed runtime](RUNTIME_GUIDE.md), and
[compiler controls](COMPILER_BASELINE_GUIDE.md) are separate tracks with separate
inputs, hardware, cost boundaries, and conclusions. Do not multiply their speedups
or use the WAP result to promote them.

## Preserved history

The [history index](history/README.md) links retired documents and workflows at
immutable commits. Raw evidence and negative outcomes remain in Git history and
under their experiment directories. Transport-only patch chunks and one-shot
application workflows are intentionally absent from the current working tree after
merge; their original bytes remain recoverable from the research commits.
