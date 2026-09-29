# Integer feature interactions: learning and compact execution

Read RESULTS.md and CONTRACT.md before using this experiment. Full learned NCA
fails its three-task quality comparison; local covariance helps Letter only, and
a no-label-neighbor control matches its correct count. Neither two-task promotion
gate passes. The runtime enables integer feature mixing with two compact kernel
tables, but no production default or previous trained model is replaced.

This is an additive research module on top of PR38. Development and fitting use
NumPy2.3.5, SciPy1.17.0, scikit-learn1.8.0, threadpoolctl and a C++17 compiler;
inference uses only the standard library and an explicitly built native library.
Linux x86-64 portable and optional AVX2 are the new accepted targets. No automatic
compilation, downloads, pretrained models, teacher API or silent input quantization.

## Reproduce execution, not a new training claim

The supplied kit includes18 frozen classifiers and all9498 original test rows.
From the extracted kit folder:

    python -I -S selftest.py --target portable --out replay.json
    python -I -S selftest.py --target avx2 --out replay-avx2.json

AVX2 requires compatible x86-64 hardware. The two libraries were compiled on the
reported Linux environment; they are not universal OS/Python deployment binaries.
The retained source permits explicit local rebuilding. Tests check output fidelity
to these trained functions, not new accuracy or exact-real arithmetic.

## Reproduce the study

The evidence parent/ directory contains the original parsed data, provenance and
prior fixed controls needed by the scripts. New fitting runs can produce different
bytes on another numerical stack; retain them as new results. Do not reuse a result
name or tune settings on these already opened test partitions. Trusted pickle
artifacts are only for reproducing the supplied models; never load arbitrary pickles.

    python experiments/integer_interactions/study.py select --data /evidence/parent --out /new/selection
    python experiments/integer_interactions/local_study.py --data /evidence/parent --out /new/secondary
    python experiments/integer_interactions/refit_all.py --data /evidence/parent --selection /new/selection --secondary /new/secondary --combined /new/combined --out /new/models
    python experiments/integer_interactions/build.py --out /new/native --target portable
    python experiments/integer_interactions/evaluate.py --data /evidence/parent --models /new/models --library /new/native/interaction.so --out /new/evaluation

These are explicit quadratic SVM fits. A full Letter Gram file is2.048GB on disk;
metric optimization and validation add CPU/RAM costs. The two-table representation
improves deployment storage, not training complexity. Original pilot commands and
the full-float diagnostic are in pilot.py/continuous_probe.py and their receipts.

The complete cost comparison is bench.py, with --parent, --models, --evaluation,
--controls, --library, --parent-library and --out paths. All supplied model-specific
expectations and fourteen arms must stay. Projection is inside the request timer.
The direct_exp arm is a different rounding convention, not assumed bit-identical.
Separate resource_probe.py measurements charge setup and self process high-water RAM.

## Test and audit

    python -m pytest experiments/integer_interactions/test_interactions.py --import-mode=importlib --confcutdir=experiments/integer_interactions
    python experiments/integer_interactions/audit.py --root /evidence --source /matching/source --out /new/audit.json

The audit reads only JSON/NPY data, verifies exact-feature split boundaries,
complete hyperparameter selection, final model locks, all predictions and timing
arithmetic. The original source snapshots required by different fitting phases
are included. It imports no numerical framework and never unpickles a model.
negative_audit.py modifies disposable copies to test explicit failure paths.
Clock authenticity, spatial/writer independence and external reproduction are
outside its scope. The raw data and model sources retain their original attribution.

Primary references, not novelty claims:
- Goldberger et al., Neighbourhood Components Analysis (NeurIPS2004):
  https://papers.nips.cc/paper_files/paper/2004/hash/42fe880812925e520249e808937738d2-Abstract.html
- Shi, Bellet, Sha, Sparse Compositional Metric Learning (AAAI2014):
  https://ojs.aaai.org/index.php/AAAI/article/view/8968
- UCI Letter, Pendigits and Statlog Landsat Satellite metadata and prior receipt
  are retained in the parent delivery. MIT code licensing does not relicense data.
