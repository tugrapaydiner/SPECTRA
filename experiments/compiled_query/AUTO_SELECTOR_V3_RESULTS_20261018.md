# Automatic selector v3 — retained result, 18 October 2026

## Verdict

**Development passed. Prospective confirmation failed. The policy is not eligible
for merge.**

The failure is narrow and reproducible: structural-v3 treated every bank with at
least eight pins per request as a raw-traversal workload. On the prospective
1,024-vertex cases, 15.09 pins per request represented only 1.47% of the graph.
SCC remained substantially faster at 128 and 1,024 requests, but automatic mode
selected raw.

No threshold was changed after opening confirmation. These results are retained
as failed evidence and structural-v3 is superseded rather than relabelled as a
pass.

## Exposed development panel

The 48-case, 384-cell development matrix passed all frozen gates on x86-64:

| Metric | Result | Gate |
|---|---:|---:|
| Global realised-regret geometric mean | **1.005101** | <= 1.03 |
| Global selection-regret geometric mean | **1.004323** | <= 1.02 |
| Worst family/count geometric mean | **1.019694** | <= 1.05 |
| Cells within 5% of the explicit oracle | **98.44%** | >= 95% |
| Worst cell | **1.078111** | <= 1.10 |

This panel was tuning evidence, not confirmation.

Workflow run: `39088861752`; artifact `12122076171`.

## Prospective confirmation

The fixed panel contained 32 generated cases and 160 case/count cells on each
architecture. All answers completed and independently replayed. The selector code,
thresholds and seed namespace were committed before the cases were generated.

### x86-64

| Metric | Result | Gate | Verdict |
|---|---:|---:|---|
| Global realised regret | **1.011453** | <= 1.05 | PASS |
| Global selection regret | **1.010937** | <= 1.03 | PASS |
| Worst family/count group | **1.033493** | <= 1.07 | PASS |
| Cells within 5% | **90.625%** | >= 90% | PASS |
| Worst cell | **1.130612** | <= 1.12 | **FAIL** |
| Worst family auto/raw ratio | **0.817935** | <= 1.05 | PASS |

The worst cell was `confirm-random_tree-r2-n1024` at 1,024 requests. Automatic
mode selected raw under `high_pin_density`; explicit SCC was about 13.0% faster.
Seven other 1,024-request cells in the same density regime showed approximately
10.0–12.7% regret.

### Native ARM64

| Metric | Result | Gate | Verdict |
|---|---:|---:|---|
| Global realised regret | **1.016322** | <= 1.05 | PASS |
| Global selection regret | **1.014754** | <= 1.03 | PASS |
| Worst family/count group | **1.037124** | <= 1.07 | PASS |
| Cells within 5% | **87.5%** | >= 90% | **FAIL** |
| Worst cell | **1.150248** | <= 1.12 | **FAIL** |
| Worst family auto/raw ratio | **0.837099** | <= 1.05 | PASS |

The same low-density/high-absolute-pin regime caused the failures. Eight
1,024-request cells had 13.1–15.0% regret because automatic mode selected raw.

Workflow run: `39089238933`.

- x86 artifact `12122306101`, ZIP digest
  `sha256:317d6b60306db5ea30c43dbbc4f661994dff0a4e0a8672ef91b3c82467580b65`;
- ARM artifact `12122157779`, ZIP digest
  `sha256:d2f3ea4f73202c616c0437e1f6302461cde96c03e5ced99dbafed4b39158a9c7`.

## Real WAP guard

The independent WAP guard also rejected v3 before timing could complete:
`wap05a.col` at 64 requests selected raw instead of SCC. The WAP request bank has
a high absolute pin count but low density relative to its graph, the same defect
exposed by confirmation.

Workflow run: `39089501658`. Both x86-64 and ARM64 retained the failure artifacts.

## Workflow-status correction

The confirmation jobs appeared green because the benchmark command was piped to
`tee` without shell `pipefail`; the benchmark itself wrote `pass: false` and
raised. The retained result JSON, not the outer badge, is authoritative. All
subsequent selector workflows must use `set -euo pipefail` and explicitly assert
the stored `summary.pass` field after measurement.

## Diagnosis and next policy

Absolute pins/request is not a sufficient fallback feature. The relevant
quantity is pin density relative to graph size.

The failed panel, existing ITC evidence and WAP evidence separate the regimes:

- failed confirmation: 15.09 pins / 1,024 vertices = **1.47%**; SCC should run;
- ITC set9: 9.47 pins / 169 exams = **5.60%**; raw remains conservative;
- ITC set10: 9.09 pins / 214 exams = **4.25%**; raw remains conservative;
- ITC set12: 10.12 pins / 78 exams = **12.97%**; raw remains conservative.

Structural-v4 therefore replaces the absolute high-pin rule with a frozen density
rule. This failed confirmation becomes development evidence for v4. A new seed
namespace and new cases are required for any later confirmation claim.
