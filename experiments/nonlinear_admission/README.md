# Nonlinear usefulness before runtime optimization

This optional experiment asks whether retaining a nonlinear classifier is justified
before measuring its execution. It does not introduce a new learning architecture
or alter SPECTRA inference defaults. Start with `PROTOCOL.md`, the comparator
amendment and `RESULTS.md` (the final result is published only after measurement).

## Reproduce the ordered study

Use Python3.13, NumPy2.3.5, SciPy1.17.0, scikit-learn1.8.0, threadpoolctl3.6.0 and
GCC14.2 on a little-endian Linux x86-64 machine supporting AVX2. These are the
measured environment versions, not claims about all versions or architectures.
Training children run on one CPU with120-second wall and4GiB address-space limits.
The dependency environment, acquisition and code generation are separate costs.

The evidence archive supplies the original acquisition ZIP contents and all frozen
artifacts. The acquisition workflow retrieves only the two official UCI archives.
Use the supplied `SPECTRA_nonlinear_admission_inputs.zip` or reconstruct that ZIP
from the workflow artifact including its `ACQUISITION.json` receipt.

```bash
python experiments/nonlinear_admission/study.py prepare --archive /inputs.zip --out /new/study
python experiments/nonlinear_admission/study.py develop --root /new/study
python experiments/nonlinear_admission/study.py refit --root /new/study
python experiments/nonlinear_admission/study.py evaluate --root /new/study
```

Never overwrite a stage directory, change a timeout because a result is unfavorable,
or refit based on the final test. `SELECTED.json` and `MODEL_FREEZE.json` bind all
choices and model/data bytes before either final-test prediction. The three MLP
seeds determine a mean width-selection score, not an ensemble or lucky-seed choice.
The provided pickles are locally produced training artifacts; do not load unknown
pickles. The independent auditor never deserializes a model pickle.

## Build and measure native controls

Use unmodified upstream LIBSVM3.37 sources at commit
`6b907139084abf2da4d6d3cb10dc3b7eaffa2fbb`, with its COPYRIGHT preserved. The
existing exact-value exporter and adapter are reused. The new dense adapter covers
only the frozen multiclass linear and one-hidden-ReLU graphs. It preserves every
parameter and uses strict ordered arithmetic; it is not a new production API or a
thread-safe general model-serving framework. Compiled numerical differences from
BLAS are checked explicitly and never excuse a changed class.

```bash
python experiments/nonlinear_admission/native.py --root /new/study --vendor /libsvm --out /new/builds
python experiments/nonlinear_admission/generated.py run --root /new/study --builds /new/builds --package /original-m2cgen
# Apply the existing PR36 single-lookup patch to a separate copy before fallback.py.
python experiments/nonlinear_admission/fallback.py --root /new/study --builds /new/builds --package /patched-m2cgen
python experiments/nonlinear_admission/fidelity.py --root /new/study --builds /new/builds --out /new/fidelity
python experiments/nonlinear_admission/bench.py --root /new/study --builds /new/builds --out /new/timing
python experiments/nonlinear_admission/audit.py --root /new/study --builds /new/builds --run /new/timing --source . --out /new/audit.json
python experiments/nonlinear_admission/check_negative.py --root /new/study --builds /new/builds --run /new/timing --source . --out /new/negative.json
```

The exporter fallback preserves original timeouts and compiler failures. It tries
fixed feasibility options, not a timing-based best implementation. Selected-model
quality is already frozen. The fallback helper and applied-source receipt for the
existing PR36 patch are retained in evidence. All compiled sources/libraries are
trusted local code; neither these experiments nor resource limits are a sandbox.

The complete timing grid includes native SPECTRA default/exhaustive, upstream
LIBSVM, feasible generated C, native linear/MLP and their sklearn implementations,
plus an explicitly framework-only boosted-tree control. Full test inputs and fresh
labels are charged. Batch sizes1/32/256 are complete-job throughput measurements;
they are not production request-latency distributions. Scaler-inclusive batch32
charges the same frozen StandardScaler. Raw acoustic/sensor feature extraction,
loading and build costs are outside those warm timers. No performance cells are
pooled with earlier hosts or assigned to a different model's quality.

## Meaning of the admission result

One percentage point above the best linear and mean MLP control on validation and
final test is a predeclared **research screen**, not a user-derived product quality
requirement. The tree control remains visible. Two selected datasets, one official
speaker test group, three MLP seeds and two future sensor batches do not justify
population-wide generality or independent-group bootstrap inference. Binary files
are artifact sizes, not process memory. A passing comparative screen is not safety,
commercial or field-deployment validation.

## Sources and dataset terms

ISOLET: Cole and Fanty, UCI54, DOI10.24432/C51G69. The original final file uses a
separate group of30 speakers; individual speaker IDs are not inferred from row
positions. Internal random development validation may share speakers. UCI lists
CC BY4.0. Source: https://archive.ics.uci.edu/dataset/54/isolet

Gas Sensor Array Drift: Vergara, UCI224, DOI10.24432/C5RP6W. Fit batches1..6,
validate7..8,refit1..8,test9..10. These are chronological batches, not random
train/test partitions. The UCI page contains BOTH research-only/commercial-exclusion
wording in its description and a CC BY4.0 footer. Retain that ambiguity; this
experiment does not resolve or grant commercial rights. Source:
https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset

MIT applies to SPECTRA code, not automatically the original datasets, LIBSVM or
m2cgen. Original bytes, attribution, version identifiers and acquisition hashes
are preserved. No production gas-identification or safety-critical use is claimed.
