# Native alternatives and held-out activity comparison

Read [RESULTS.md](RESULTS.md) before interpreting any speed ratio. This is an
optional Linux CPU research experiment, not a change to SPECTRA's deployment API.
Native-speed admission is incomplete and task usefulness failed on the new HAR
workload. No model is retrained to suit a backend and no prior result is removed.

## Inputs

Use pinned NumPy2.3.5, SciPy1.17.0, pandas2.2.3, scikit-learn1.8.0, joblib1.5.3,
threadpoolctl3.6.0 and m2cgen0.10.0. The source includes the acquisition workflow
for official UCI HAR data and unmodified LIBSVM3.37 commit
6b907139084abf2da4d6d3cb10dc3b7eaffa2fbb. The recorded artifact is10939445657 from
run36341344110, SHA25689b78284fdaf1b0c44c11da15d4e019f6c4e9a3eccf8de020707071cdba36c46.
Acquisition is not a performance result. Inspect original licences before reuse.

The six old models can be reconstructed from the versioned previous corpus,
without ChatGPT attachment history. The reconstruction MUST match its old hashes.
Use a new absolute working directory W, with acquired files under W/external.
Commands run from a repository checkout; replace the explicit paths below.

```bash
python experiments/native_baselines/reconstruct_controls.py --out /W/old-controls
python experiments/native_baselines/prepare.py --old-inputs /W/old-controls \
  --har /W/external/har --out /W/models --trust-frozen-pickles
python experiments/native_baselines/build.py --vendor /W/external/libsvm \
  --models /W/models --out /W/builds-final
python experiments/native_baselines/evaluate.py --models /W/models \
  --har /W/external/har --builds /W/builds-final --out /W/validation
```

Preparation performs only the two predeclared HAR fits and freezes their identities
before evaluation. It reads verified benchmark pickles; do not use unknown inputs.
Generation and compilation run in bounded subprocesses. A timeout remains in the
result inventory and does not make a comparator defeated. Do not discard it or
silently extend the reported experiment's budget. The original six SVC and the
new fixed SVC's parameters are shared by all nonlinear backends.

## Measure without another numerical job running

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python experiments/native_baselines/bench.py --models /W/models \
  --validation /W/validation --builds /W/builds-final --out /W/benchmark
python experiments/native_baselines/matched_isa.py --root /W
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python experiments/native_baselines/bench.py --models /W/models \
  --validation /W/matched_isa/validation --builds /W/matched_isa/builds-final \
  --out /W/matched_isa/benchmark
```

The second entire run adds AVX2 to both external competitors and revalidates their
classes. It uses the same already generated C, model parameters and test inputs.
Never pool the two runs or select the favorable one for each case. The linear
control remains unchanged. Native libraries are host-specific executable artifacts;
there is no implied Windows or ARM acceptance for these experiment wrappers.

```bash
NATIVE_BASELINE_LIBSVM=/W/matched_isa/builds-final/libsvm/libnative_svm.so \
  python -m pytest experiments/native_baselines/test_adapters.py \
  --confcutdir=experiments/native_baselines -q
python experiments/native_baselines/audit.py --root /W/matched_isa \
  --source /path/to/exact-measured-source --out /W/matched-audit.json
```

The standard-library auditor uses frozen source identities. For the delivered
runs, use supplied timed_source/ for the first run and matched_timed_source/ for
the stronger run, rather than substituting this later documentation/test state.
Report all results, including generated-code failures, quality controls, repeated
examples and subject-level uncertainty. Reading this README is not a claim of
external reproduction; the experiment was executed by the project itself.
