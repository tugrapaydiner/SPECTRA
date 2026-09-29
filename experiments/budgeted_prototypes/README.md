# Fixed-budget learned prototype experiment

Read [RESULTS.md](RESULTS.md): the two-task learning gate FAILED; Satellite has a
useful measured compact-model tradeoff, while the optical transfer loses to SVC.
This folder is an additive research experiment, not a production SPECTRA default.
The learned architecture and numerical limits are in [CONTRACT.md](CONTRACT.md).

## Reproduction roles

The evidence delivery contains the original248 selection fits,36 optical-width
amendment fits, all24 frozen final classifiers, four mechanical FP32 MLP conversions,
three complete timing matrices, stage-specific source snapshots, original data,
local and remote test receipts, independent observer, setup/memory observations,
and the failed pilots. Models pickled during selection are trusted reproduction
inputs only. Do not load an untrusted pickle. Deployment uses bounded inert formats.

Use supplied frozen files for exact replay. Repeating fitting in another environment
may produce different arrays and must be recorded as a NEW run, not retroactively
assigned the original hashes or test independence. The OptDigits official test is
already open after this experiment. Do not tune on it and call it a fresh holdout.

## Build and tests

Linux x86-64, Python3.11+ and a C++17 compiler are the accepted experimental scope.
Training/tests used Python3.13.5, PyTorch2.10.0+cpu, NumPy2.3.5, SciPy1.17.0,
scikit-learn1.8.0 and pytest9.0.2. Prototype inference needs no numerical framework.
All output build directories must be fresh; imports never compile/download.

```bash
python experiments/budgeted_prototypes/build.py --out /new/native --target portable
python -m pytest experiments/budgeted_prototypes/test_prototypes.py \
  --import-mode=importlib --confcutdir=experiments/budgeted_prototypes
# Explicitly use only on compatible AVX2 x86-64 hardware:
python experiments/budgeted_prototypes/build.py --out /new/register \
  --target avx2 --layout register
```

`PrototypeSession(model.spp, library)` accepts writable contiguous uint8 raw-code
buffers and returns fresh labels. `scores()` exposes own-model scores for inspection;
`direct_exp` is only a different-function diagnostic. Input buffers must not be
mutated concurrently. See the contract for limits and trusted-native boundaries.

## Training reconstruction

`study.py initialize/worker/select/refit/finish` implements the fixed initial
selection and full-data refits. `optical_amendment.py` adds the declared36 wider
optical kernels before selecting final settings. The required original parsed
training inputs are under `prior/datasets`, `prior/satellite/data`, and `new_data`
in the evidence. The original archives and their attribution are retained there.
`SELECTION_LOCK.md` and `OPTICAL_RANGE_AMENDMENT.md` define the intended order.
Use a new evidence directory for a new training run, never overwrite prior records.

For example, initialize the original selection with:

```bash
python experiments/budgeted_prototypes/study.py initialize \
  --prior /evidence/prior --new /evidence/new_data --out /new/selection
python experiments/budgeted_prototypes/study.py worker \
  --prior /evidence/prior --new /evidence/new_data --out /new/selection \
  --workers 1 --worker 0
python experiments/budgeted_prototypes/study.py select --out /new/selection
```

The complete commands and source snapshots in the packet bind each earlier stage.
The source folder contains later report/audit files which did not exist at every
training stage; their presence must not rewrite earlier source inventory receipts.

## Strong baselines and complete timing

`controls.py` explicitly builds ordinary native SVC and ordered/scaled MLP controls.
`strong_controls.py` builds the previous finite/adaptive SVC raw-input wrapper and
a single-thread OpenBLAS DGEMM network. `float_control.py` casts the same selected
MLP parameters/scaler to FP32 and builds an SGEMM implementation. No prototype
parameter changes occur during any of these baseline amendments.

The BLAS builders require the recorded LP64 SciPy OpenBLAS library exposing
`scipy_cblas_dgemm`, `scipy_cblas_sgemm` and its thread/config symbols. Supply its
actual local path; it is not embedded in the prototype runtime. The recorded
library is OpenBLAS0.3.30, DYNAMIC_ARCH, one numerical thread for these experiments.
External BLAS/OS libraries, Python itself and compilers are not bundled in the
small framework-free kit. Plain native MLP replay in that kit is a correctness
check, NOT reproduction of the faster BLAS/FP32 timing without its dependencies.

The final benchmark requires models/evaluation, three prototype-library paths,
ordinary controls, BLAS64, finite-SVC and FP32 libraries, FP32 model folder and a
fresh output folder. Run `bench.py --help` for its exact arguments. It includes
all17 arms, not a subset of favorable models. Stage snapshots reproduce the earlier
14- and16-arm matrices; none is pooled with the final17-arm result.

## Independent evidence checks

```bash
python -I -S /matching/source/experiments/budgeted_prototypes/audit.py \
  --root /extracted/evidence --out /new/audit.json
python experiments/budgeted_prototypes/negative_audit.py \
  --root /extracted/evidence --out /new/corruption-test
```

The primary auditor uses only the standard library and inert JSON/NPY data; it does
not import PyTorch/sklearn, execute pickles, retrain or call the candidate runtime.
It checks actual stage identities, all selection outcomes, final locks, official
OptDigits rows, prediction/confusion counts, every timing cell, the mechanical
FP32 conversion and all gate arithmetic. The corruption test makes disposable
copies; retain only its report when space matters, not redundant copied datasets.
No file-hash audit can authenticate a hostile wholesale replacement of evidence
or prove the truth of elapsed clocks and statistical independence.

The standalone delivery verifier additionally checks archive/source manifests,
local test XML and the downloaded exact-head CI records. `resource_probe.py` and
`resource_float.py` measure own-process high-water memory with one validated row;
they do not establish worst-case service memory or cold-filesystem performance.

## Prior art and attribution

RBF networks and discriminatively learned prototypes are established techniques.
ProtoNN is a directly relevant prior method, not a competitor we measured or beat:
https://proceedings.mlr.press/v70/gupta17a.html

Official UCI sources: Letter59 (David Slate), Pendigits81 (Ethem Alpaydin and Fevzi
Alimoglu), Statlog Satellite146 (Ashwin Srinivasan), and OptDigits80 (Ethem Alpaydin
and Cevdet Kaynak). Preserve their dataset attribution and CC BY4.0 terms; the
implementation license does not relicense those data. No pretrained/teacher models,
GPU, external maintainer review, published release or model-default change.
