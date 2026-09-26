# Cross-domain panel: fidelity passes, primary speed gate fails

The frozen protocol (`2ed5eab`) preceded acquisition. Implementation (`803ece2`)
and independent audit/runner (`b5a8421`) are committed on this branch. All617
original PR30 files remain unchanged. No model was selected or retrained from
test results. The runtime is unchanged; this is a transfer/limitations study.

## Native result

Five fixed UCI tasks span13,18,36,561 and48 features. All16,908 test inputs retain
the selected SVC's class under five SPECTRA arms and native upstream LIBSVM3.37.
The LIBSVM text export preserves binary64 parameters through17-digit decimals;
an independent checker compares every exported support/coefficient/bias value.

Local Intel Xeon Platinum8573C, explicit AVX2, one pinned core. Mean per-case
median complete native microseconds; includes finite checks, per-request reset,
context/table work and class output, not preprocessing/loading/compilation.

| Task | Certified SPECTRA | Same-kernel exhaustive | Native LIBSVM | Speedup/exhaustive | Speedup/LIBSVM |
|---|---:|---:|---:|---:|---:|
| Wine |1.053|1.016|1.038|0.965x|0.985x|
| Vehicle |4.278|5.222|6.937|1.221x|1.622x|
| Landsat |11.695|22.577|36.692|1.930x|3.137x|
| HAR |77.088|181.442|487.186|2.354x|6.320x|
| Sensorless |37.164|73.859|119.416|1.987x|3.213x|

Only three tasks meet1.25x versus strengthened exhaustive; the frozen gate requires
four of five. **FAIL**, not rescued by changing baselines or removing small models.
Tables activate on Vehicle/Landsat only. Vehicle's table path is4.5% slower than
the same scheduler/direct distances; storage suitability is not a latency oracle.

Public complete-call preprocessing/conversion changes the comparison. SPECTRA /
native-LIBSVM Python adapter in us: Wine33.214/21.751; Vehicle39.135/29.325;
Landsat58.395/64.105; HAR277.822/554.865; Sensorless93.876/152.586. Small-model
regressions remain. Do not assign native times to the full Python application.

## Task quality, not just faithful execution

| Task | Test rows | SVC accuracy | Logistic accuracy | Fixed-budget MLP accuracy |
|---|---:|---:|---:|---:|
| Wine |45|95.56%|91.11%|95.56%|
| Vehicle |212|81.60%|77.83%|82.55%|
| Landsat |2000|89.95%|84.25%|90.15%|
| HAR |2947|93.59%|93.52%|92.81%|
| Sensorless |11704|45.40%|54.23%|34.24%|

Sensorless is a poor classifier result under the predeclared original-order block
split, despite faster inference. HAR's linear control is nearly as accurate and
faster through its public API. No universal task-level dominance or state-of-the-art
accuracy claim. Thirty fixed fits consumed26.1788 CPU seconds locally, including
validation predictions; compilation/measurement/audits are additional. No GPU.

Wine/Vehicle are stratified row splits; Landsat retains its provided split but is
not independent-scene testing. HAR fitting/validation/test subjects are disjoint.
Sensorless lacks motor/condition group IDs; this is not unseen-motor evaluation.
All original rows are retained. Vehicle's actual downloaded files contain846 rows,
not the946 listed on its webpage. MLP convergence warnings and Wine batch-size
clipping remain. No repeated test feature vectors were found, which does not prove
statistical independence or remove spatial/temporal correlation.

## Checks and second-host reproduction

Locally:29 panel/auditor tests and112 existing SVM tests pass. The separate
standard-library audit checks124,190 native/public/batch timing records and101,448
executor decisions across16,908 unique inputs, plus every supplied vote certificate,
sealed preprocessing, source/model/library hashes and derived comparisons. It does
not import a numerical framework or the timing analyzer. Initial malformed-NPY
parser failures and an invalid preprocessing probe are retained with corrected
reruns; runtime/model bytes and original timings did not change.

GitHub workflow36280443872 on `b5a8421` completed full fitting, exports, held-out
fidelity, timings, audit and29 tests on AMD EPYC7763. Its five selected SVM files
are byte-identical to the local Intel models. Independent re-audit of the downloaded
artifact passes. CI speedups versus strong exhaustive are1.006x,1.142x,1.803x,2.590x,
2.040x; versus LIBSVM1.035x,1.611x,3.454x,6.309x,4.771x. **The primary gate fails
on both hosts.** Automated second-host execution is not outside researcher replication.

Artifact10918678637, SHA256
`7556a780a698d5c8da15b4c5a104c7a52ce4993854774aeb72b3e166314e1ebe`, has verified
ZIP CRC, exact tree `fdb1310fc992c1958163fb2b00705039e6bba2be` and frozen-member
hashes. CI receipts and local raw evidence are in the accompanying delivery.

Intervals condition on fixed models/inputs/hosts. Eleven repeats are not production
P95 guarantees; all regressions stay in the records. The1759-test historical suite,
new sanitizer runs, peakRSS/energy, multiworker throughput and Windows/ARM were not
rerun here. Certificates settle computed votes, not label truth or exact-real exp.

See README.md and PROTOCOL.md for reproduction and attribution. UCI data are
CC BY4.0; upstream LIBSVM COPYRIGHT is retained. SPECTRA code remains MIT. No release,
main merge, new default scheduler or90/100 claim follows from this experiment.
