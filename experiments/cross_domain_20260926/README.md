# Frozen cross-domain executor panel

This is a separately scoped experiment on PR30's unchanged generic SVM runtime,
not a new default classifier or a replacement release. Read `PROTOCOL.md` before
interpreting results. `RESULTS.md` records the local five-task result; the failed
four-of-five speed gate is retained.

## Reproduce the complete experiment

From the repository root, use Linux x86-64 with AVX2 and a C++17 compiler. Install
the declared CPU environment (no PyTorch is needed):

```bash
python -m pip install numpy==2.3.5 scipy==1.17.0 scikit-learn==1.8.0 pytest==9.0.2
python experiments/cross_domain_20260926/run_panel.py \
  --acquisition /path/to/acquisition --out /path/to/fresh-panel
```

`acquisition/` is the literal public-input directory from workflow36279357914
or the accompanying evidence packet. It contains five original UCI ZIPs, pinned
LIBSVM3.37 source and its COPYRIGHT, and `acquisition.json` with exact hashes.
It is not an editable personal dataset location. The runner refuses an existing
output directory and uses a single available CPU core.

The runner fits only the fixed development portions, checks validation exports,
seals all models, then performs full test replay, complete native/API timing,
frozen-control replay and an independent standard-library audit. It does not
select models from final-test predictions or tune the runtime from timing results.
A new run is a reproduction, not a new independent test set. Model fitting and
floating-point results may differ with library/compiler/hardware changes.

```bash
SPECTRA_PANEL_BUILD=/path/to/fresh-panel/build python -m pytest \
  experiments/cross_domain_20260926/test_panel.py \
  experiments/cross_domain_20260926/test_audit_panel.py \
  --confcutdir=experiments/cross_domain_20260926
python -S experiments/cross_domain_20260926/audit_panel.py \
  --evidence /path/to/fresh-panel --source . --out /path/to/new-audit.json
```

The 13 parser/export tests and 16 independent-auditor tests are distinct from
PR30's 112 SVM contracts. Compilation must precede the six native export fixtures;
without a build, those fixtures are skipped, not passed.

`setup_probe.py` is a separate secondary warm-cache preparation/preprocessing
measurement. It is not included in the frozen native timing matrix or interpreted
as cold process startup. The initial contaminated probe is preserved in the
local evidence alongside the corrected one.

## Scope and licenses

Model inference is stock dense RBF SVC with ordinary pairwise voting. The certificate
checks the winner relative to computed pair signs, not label truth, authenticity,
real-valued exponentials or agreement with every numerical build.

SPECTRA source is MIT. UCI data remain CC BY4.0: Wine (Aeberhard/Forina,
DOI10.24432/C5PC7J); Vehicle (Mowforth/Shepherd, DOI10.24432/C5HG6N); Landsat
(Srinivasan, DOI10.24432/C55887); HAR (Reyes-Ortiz et al., DOI10.24432/C54S4K);
Sensorless (Bator, DOI10.24432/C5VP5F). Original archive bytes are unchanged;
partition/scaled derivatives are labelled in the evidence. UCI's Vehicle webpage
lists946 instances; its provided .dat files in this acquisition actually contain846.
No additional100 examples were invented.

LIBSVM is comparison-only upstream code under its BSD-style three-clause COPYRIGHT,
retained verbatim with `svm.cpp` and `svm.h`. This is not a copied replacement
implementation or a benchmark against an intentionally slow Python SVM.

The evidence pickles were created by this experiment and are trusted-source
reproduction artifacts. Do not deserialize arbitrary third-party pickles. Public
SPECTRA inference uses the bounded non-pickle export.
