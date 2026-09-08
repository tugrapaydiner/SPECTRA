# Reproducing the M14 latency pilot

Read `M14_PROTOCOL.md` before running this experiment. The configuration is
`config/m14_comparison.json`. This is a bounded CPU pilot on generated unique
9×9 Sudoku, with a separate, guarded confirmation partition. A completed run is
not automatically a passing research milestone.

## Tested CPU environment

Use Python 3.12 on Linux. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-m14.txt
export MAX_JOBS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
```

Activating the environment matters: the C++ extension locates the installed
Ninja executable through PATH. The native lane also needs a working C++20
compiler. An unavailable native lane is reported, not relabelled as optimized.
The actual experiment environment, including transitive versions, is in
`environment.json` and `environment.lock.txt` inside the evidence directory.
Neither GPU training nor Windows deployment is validated by this pilot.

## Execute the experiment in order

Use a NEW output directory. Existing experiment directories are never cleared
or overwritten by these commands.

```bash
python scripts/m14_controlled_experiment.py --out outputs/m14_reproduction --stage prepare
python scripts/m14_controlled_experiment.py --out outputs/m14_reproduction --stage train
python scripts/m14_controlled_experiment.py --out outputs/m14_reproduction --stage tune
python scripts/m14_controlled_experiment.py --out outputs/m14_reproduction --stage evaluate
```

Preparation produces all four data partitions and their hashes. Training reads
only the training partition. Tuning chooses a common configuration across every
declared seed. Development evaluates paired examples with shuffled system order.
It stores all predictions and all timing rounds before computing the gate.

Only if `initial/development_gate.json` has `passed: false`, run the predeclared
development control and intervention:

```bash
python scripts/m14_decode_control.py --out outputs/m14_reproduction
python scripts/m14_controlled_experiment.py --out outputs/m14_reproduction --stage train --phase blank_only
python scripts/m14_controlled_experiment.py --out outputs/m14_reproduction --stage tune --phase blank_only
python scripts/m14_controlled_experiment.py --out outputs/m14_reproduction --stage evaluate --phase blank_only
```

The corrective fit uses each family's previously selected learning rate and the
same data, three seeds, step cap and cumulative fitting budget. It does not lower
the practical improvement target or weaken the baseline. Its development data
have been reused, so this is explicitly development evidence.

Only a passing development gate writes `confirmation_authorization.json`.
Inspect that file's `phase` and use the authorized phase in:

```bash
python scripts/m14_controlled_experiment.py --out outputs/m14_reproduction --stage evaluate --split confirmation --phase initial
```

Substitute `blank_only` only if that is the authorized phase. The confirmation
loader verifies the successful gate hash and records its single consumption.
There is no command-line override to use confirmation for training or tuning.
A failed confirmation does not permit another confirmation attempt in this
protocol. Interrupted confirmation is also consumed; retain it and declare a
new independent attempt instead of silently running it again.

## Regenerate the retained table and figure

Extract the evidence ZIP so that its experiment directory is available locally.
Then pass that directory as `--out` (the path is independent of the source tree):

```bash
python scripts/render_m14_comparison.py --out /absolute/path/to/latency_v1
```

This runs a separate Python row/column/box/clue checker against every saved
prediction, verifies the data hashes, frozen run identities and paired timing
coverage, and regenerates `report/results.csv`, `report/RESULTS.md`, the SVG/PNG
accuracy–latency plot and explicit empirical neural Pareto frontiers. It does
not fit a model or evaluate a sealed confirmation set. Its provenance records
all raw-row input hashes. Figure error bars are observed seed ranges, not
confidence intervals; inferential gate bounds are in the gate JSON files.

Checkpoints contain model weights, optimizer state, minibatch-generator state,
torch RNG state, dataset hash, representation strength and training step.
Tuning and measured comparisons always reload the declared checkpoint.
All fits, including unselected learning rates and earlier checkpoints, remain
in the evidence. INT8 weights are deterministically converted from their
identified FP32 checkpoint using the recorded x86 backend; its converted layer
list and exact quantized weight coverage are recorded.

## Correctness gate

```bash
python -m pytest tests/test_m14_controlled_comparison.py -q
python -m pytest -m 'not slow' -q
```

The focused tests verify guarded confirmation access, tamper detection, real
INT8 execution, paired seed/example inference, the complete symbolic solve,
clue-aware decoding, quantization strength and live training gradients. The
full fast suite includes existing native and measurement contracts. These
correctness tests are separate from evidence of research improvement.

## Retained evidence replay

The Git-backed `results/m14/development_reproduction.zip` is sufficient to regenerate all public development tables and the figure using the commands in its README. It excludes checkpoint weights.

For the separate full `SPECTRA_M14_evidence.zip`, extract it and run:

```bash
python scripts/verify_m14_evidence.py --out /absolute/path/to/latency_v1
```

This checks the evidence inventory, all data and checkpoint hashes, and 96 selected development predictions. Confirmation bytes are hashed only, never loaded for inference. Use the immutable reproduction code commit recorded in the evidence README. This is an artifact replay, not an independent rerun of training or a cross-platform deployment certification.
