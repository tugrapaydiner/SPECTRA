# Source-verified compact tree execution

Read `RESULTS.md` for completed measurements and limitations. This is an additive
experimental classifier executor, not a new learner or production default.
It resumes the delivered exact-rational checkpoint with actual native code,
frozen-model replay, two official CatBoost C API comparators and complete records.
The older missing native workspace was not recovered or relabelled as this run.

## Build and test

Use Python3.11+ and a C++17 compiler on Linux x86-64. Inference and proof checking
need only the standard library plus compiled libraries. Model reproduction and
some synthetic compatibility tests use CatBoost1.2.8 and NumPy. Native AVX2 is an
explicit target for compatible hardware, not automatic CPU dispatch.

```bash
python -m experiments.certified_trees.build --out /new/native --target portable
python -m experiments.certified_trees.build --out /new/fast --target avx2 --layout register
python -m pytest experiments/certified_trees/test_native.py \
  experiments/certified_trees/test_audit.py --import-mode=importlib \
  --confcutdir=experiments/certified_trees
cd experiments/certified_trees/reference && python -m unittest -v test_oracle.py
```

The two actual-official-fallback synthetic tests require `TREE_UPSTREAM` pointing
to a folder with the pinned1.2.8/1.2.10 libraries. A missing external-library fixture
is an explicit skip, not acceptance. `TREE_LIBRARY` can name the specific compiled
candidate, including a sanitizer build. Build directories must be new.

## Compact-only versus full-coverage API

```python
from experiments.certified_trees.session import VerifiedCompact, TreeSession
from array import array

proof = VerifiedCompact.from_files('model.json', 'model-16.sct')
with TreeSession('trees.so', first=proof) as runner:
    indices = runner.predict_buffer(array('B', raw_feature_codes))
    # -1 means UNRESOLVED. It must never be interpreted as a predicted class.
```

To require full coverage, explicitly supply the matching trusted original CBM and
an official CatBoost library, then use `fallback=True`. Refinement additionally
loads verified8-bit and16-bit models and sets `refine=True`. Source identities and
routing must agree. `inspect_buffer()` returns counts and per-row tree work.
See CONTRACT.md: certificates are conditional source-index agreement, not proof
of real-world correctness or universal backend numerical semantics.

## Reproduce the recorded experiment

The evidence/SDK delivery contains the actual four fitted CBM/JSON/C++ models,
original evaluation codes/labels and source identities. Use those frozen files
for byte-exact replay. The original data roles are Letter16000/4000,
Pendigits7494/3498, Satellite4435/2000 and OptDigits3823/1797. These public tests
were already exposed; no new generalization result is claimed.

`reproduce.py fit` records the fixed recipe and input hashes before fitting, saves
all four models before `open_tests` can predict, and performs no selection. An
interrupted initial process completed two models; `resume_fit.py` verified and
retained those files, then fitted the missing two. All surviving model records
are in MODEL_LOCK.json. A repeated fit creates new model identities, not recovery
of the missing earlier models or a new held-out study.

```bash
python -m experiments.certified_trees.packed --models /frozen/models --out /new/compact
python -m experiments.certified_trees.replay --models /frozen/models \
  --compiled /new/compact --evaluation /frozen/evaluation --library /new/fast/trees.so \
  --upstream /official/libraries --out /new/replay
```

`benchmark.py` is the complete original15-arm matrix. `benchmark_layouts.py`
retains every original arm and adds seven vector/register paths (22 arms).
Use `--help` for the exact library/model arguments. The original library and its
source snapshot must remain a real control, not be replaced by the optimized
implementation. Both full matrices, seeds and all unfavorable outcomes are retained.

Official libraries are unmodified assets from the CatBoost1.2.8/1.2.10 releases.
Their acquisition receipts include size and SHA256, with upstream digest checked
when supplied. CatBoost identifies its evaluation library as its fast deployment
interface. The separately compiled exported C++ is another control, not a substitute
for the official optimized library.

## Evidence verification

```bash
python -I -S experiments/certified_trees/audit.py --root /extracted/evidence \
  --source /matching/source/experiments/certified_trees --out /new/audit.json
```

The audit validates complete source/model/output bindings, source quality counts,
all retained indices and timing rows, and recalculates medians without importing
the runtime or unpickling models. It does not authenticate clocks or research
independence. `corruptions.py` tests actual evidence copies; originals are untouched.
The exact-rational proof reconstruction is separately exercised by `VerifiedCompact`.

Both measured implementations use the same classifier. To improve learned quality
requires a separate training/selection/evaluation study; compression alone cannot
do that. Prior stronger SVM/prototype accuracy results remain unchanged.

## Attribution

CatBoost source/API and official model libraries are Apache-2.0:
https://github.com/catboost/catboost
https://catboost.ai/docs/en/concepts/c-plus-plus-api_dynamic-c-pluplus-wrapper

Original numerical datasets: UCI Letter59 (David Slate), Pendigits81 (Ethem
Alpaydin and Fevzi Alimoglu), Statlog Satellite146 (Ashwin Srinivasan), and
OptDigits80 (Ethem Alpaydin and Cevdet Kaynak). Preserve their separate data
attribution and applicable CC BY4.0 terms. SPECTRA implementation licensing does
not relicense external libraries or data. No third-party issue/PR was submitted.
