# Independent replication: proof-carrying ITC exam repair

This guide is for an outside runner who has not changed the selector, event bank,
acceptance threshold, or result interpretation. A successful run reproduces a
bounded exam-repair transfer. It does **not** establish full ITC-2007 timetabling,
operator adoption, universal performance, or algorithmic priority.

## 1. Pin both repositories

Record the exact SPECTRA commit under review and use the fixed upstream revision:

```bash
git clone https://github.com/tugrapaydiner/SPECTRA.git
cd SPECTRA
git checkout <exact-spectra-commit>

git clone https://github.com/davidvanbulck/ITC-2007-CETT.git upstream/ITC-2007-CETT
git -C upstream/ITC-2007-CETT checkout ed2d4c0cf5302b77a8b321cd0380de0bb579d538
```

Do not substitute newer upstream bytes. Preserve the upstream `LICENSE` in every
shared evidence package.

## 2. Record the environment before execution

```bash
mkdir -p replication-receipts
git rev-parse HEAD > replication-receipts/spectra-commit.txt
git rev-parse HEAD^{tree} > replication-receipts/spectra-tree.txt
git -C upstream/ITC-2007-CETT rev-parse HEAD > replication-receipts/upstream-commit.txt
python -VV > replication-receipts/python.txt
uname -a > replication-receipts/platform.txt
lscpu > replication-receipts/cpu.txt 2>/dev/null || true
c++ --version > replication-receipts/compiler.txt 2>&1
```

Use a GIL-enabled CPython 3.11 or 3.13 and a C++17 compiler. Do not install or load
an alternative solver for the measured calls.

## 3. Check source and installed-package contracts

```bash
python -m pip install 'build==1.3.0' 'pytest==8.4.2'
python -m pytest \
  tests/public/test_exam_repair.py \
  tests/public/test_itc2007_exam.py \
  --confcutdir=tests/public -q
python -m pytest \
  experiments/compiled_query/test_exact.py \
  experiments/compiled_query/test_itc_exam_transfer.py \
  experiments/compiled_query/test_itc_exam_suite.py \
  -q -k 'auto or tiled_witnesses or itc or suite'

python -m build --wheel --outdir replication-wheel
python -m venv replication-install
replication-install/bin/python -m pip install --no-deps replication-wheel/*.whl
replication-install/bin/python -I scripts/check_exam_repair_installation.py \
  --out replication-receipts/installed-exam-repair.json
```

On Windows, use `replication-install/Scripts/python.exe` and a configured MSVC x64
developer environment.

## 4. Freeze before running any case

The freeze must complete before inspecting any holdout answer or timing:

```bash
python -m experiments.compiled_query.itc_exam_suite freeze \
  --root itc-suite-evidence \
  --upstream-root upstream/ITC-2007-CETT \
  --queries 1024 \
  --holdout-max-auto-raw-ratio 1.15
```

Confirm that `suite-manifest.json` records, in order:

1. `set9-development` with role `development`;
2. `set10-holdout` with role `holdout`;
3. `set12-holdout` with role `holdout`.

Do not change the roles, query count, five-repetition schedule, selector-v2 rule or
1.15 paired-median guardrail after this point.

## 5. Execute once and audit without native code

```bash
python -m experiments.compiled_query.itc_exam_suite run \
  --root itc-suite-evidence
python -S -m experiments.compiled_query.itc_exam_suite audit \
  --root itc-suite-evidence
```

The audit must independently regenerate every derived input, replay every `SCQ2`
answer from the original `SBC1` packets, require cross-engine decision agreement,
and publish all holdout failures. Do not delete negative cells or repeat a failed
measurement under a changed rule.

## 6. Report every result

Publish the complete `itc-suite-evidence` directory plus the receipts above. At a
minimum, report:

- SPECTRA commit and tree;
- upstream commit;
- OS, CPU, Python and compiler;
- all source/public/installed contract outcomes;
- each case's role, exam count, conflict count, query count, SAT/UNSAT counts,
  selected automatic engine and policy reason;
- all five paired `auto_elapsed / raw_elapsed` ratios per holdout;
- paired-median ratio and pass/fail against 1.15;
- every execution or audit failure, including resource failures.

A useful independent report explicitly says whether the bytes and conclusions
match the project-owned CI artifact. A disagreement is a valid replication result
and should not be hidden or relabelled as an environment problem without evidence.
