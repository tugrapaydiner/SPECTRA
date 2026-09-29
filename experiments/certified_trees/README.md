# Certified integer trees — experimental native continuation

Read [RESULTS.md](RESULTS.md) and [CONTRACT.md](CONTRACT.md). Compact-only16-bit
execution settles11,284/11,295 source decisions with11 explicit unresolved rows.
The complete owning pipeline preserves all source decisions, but beats the actual
native CatBoost control decisively only on Letter in this panel. It is not a
production default, a learning advance or a broadly superior inference engine.

This is a new fixed-protocol panel because the prior interrupted tree investigation's
trained weights were not recovered. The prior exact-rational checkpoint is preserved
unchanged; unsupported old certification counts are not reused. All four official
evaluation partitions were exposed previously by SPECTRA research. No new independent
accuracy confirmation or state-of-the-art classification claim is made.

## Build and source verification

Python3.11+, a C++17 compiler and Linux x86-64 are required for the accepted scope.
Build commands never run on import and require fresh output directories.

```bash
python -m experiments.certified_trees.build --out /new/trees --target portable
python -m experiments.certified_trees.packing compile --source source.json \
  --maximum 15 --bits 16 --out model.sct
python -m experiments.certified_trees.packing verify --source source.json --out model.sct
```

Use the original feature maximum and count, not the example maximum for another
model. A matching verification receipt supplies the expected model hash. The
`TreeSession` buffer API accepts writable contiguous uint8 data and returns class
indices or -1. Preserve unresolved entries. Class labels are mapped by the source
export's declared class order, retained in the corpus's `fit.json`.

For full coverage, build `controls.py --upstream /official-library-directory
--tree-library /new/trees/trees.so --out /new/pipeline`, then use `RefinementSession`
with trusted compact/CBM/source-JSON identities. It cannot infer that unrelated
JSON and CBM files represent the same model. The four source files are exported
and bound together before evaluation in this study.

## Reproducing the study

`panel.py` fits the four fixed original training partitions without evaluation-based
early stopping, then hashes all models before its test phase. Inputs are the retained
UCI Letter, Pendigits, Satellite and OptDigits partitions; the evidence includes
exact parsed data and original archive provenance. Model-specific random metadata
or library differences may change a repeated fit's bytes: treat that as a new run,
not a recovery of the frozen model hashes. Exact inference replay uses the supplied
CBM/JSON and compact binaries. No model pickle is needed by deployment or the auditor.

`compile_panel.py` compiles8/16-bit and original full-precision representations.
`fidelity.py` checks every decision and sampled exact-rational intervals against
our literal source evaluator, official native CatBoost and unchanged exported C++.
The official library/header/license are acquired by the bounded pinned-input workflow.

`benchmark.py --help` lists every final library/model argument. The full run retains
16 arms, four tasks, batches1/32/256 and seven fixed shuffled repetitions. Earlier
13/15-arm matrices and their exact source snapshots remain in the evidence. They
are separate complete runs, not pooled favorable timing cells. Tree traversal and
ownership refinements were protocol-locked before their full measurements. No
source model, quantization level or certificate bound was fitted to test outcomes.

## Verification commands

```bash
python -m pytest experiments/certified_trees/test_native.py \
  experiments/certified_trees/test_pipeline.py experiments/certified_trees/test_evidence.py \
  --import-mode=importlib --confcutdir=experiments/certified_trees
# From experiments/certified_trees:
python -m unittest -v test_oracle
# Standard-library complete empirical audit, optionally rebuilding every binary:
python -I -S /matching/source/experiments/certified_trees/audit.py \
  --root /extracted/evidence --source /matching/source/experiments/certified_trees \
  --out /new/audit.json --recompile
```

The6 real C-API pipeline tests require `TREE_PIPELINE_LIBRARY` naming the built
owning wrapper; absent that variable, those6 are explicitly skipped. `TREE_LIBRARY`
selects the tested compact library. The original13 oracle tests are a separate
suite and the8 evidence-unit tests use synthetic parsing fixtures, not empirical
accuracy observations. `negative_audit.py` alters disposable copies only and keeps
all16 rejection results. Hash audits verify bytes and derived arithmetic, not clock
truth, unknown human adaptation, signatures or statistical independence.

The ready kit offers `python -I -S selftest.py --target portable --out ../replay.json`.
Its wrappers are relinked with relative dependency paths and replayed after copying.
That is not the same byte identity as the absolute-path timing wrappers; both build
receipts are retained. CatBoost's official shared library/license is bundled unchanged;
Python, compilers, OS libraries and font files are not bundled. Older glibc may
require rebuilding. Full coverage stores the original model in addition to compact
leaves; do not use its compact-only model-size ratio as full-pipeline RAM savings.

## Prior work and attribution

CatBoost is upstream software, not a SPECTRA invention. Its optimized native CPU
library is the principal external timing control; its simpler C++ export is kept
separately. Integer tree quantization, exact pruning and mixed-precision certification
have related prior work; this experiment does not establish first invention.
See CatBoost's native evaluation library and model-export documentation, and
FQTree (arXiv2608.12140) for a recent tree-quantization example. No direct FQTree,
TreeLUT or other hardware-specific comparison is claimed.

Official UCI sources: Letter59 (David Slate), Pendigits81 (Ethem Alpaydin and Fevzi
Alimoglu), Satellite146 (Ashwin Srinivasan), OptDigits80 (Ethem Alpaydin and Cevdet
Kaynak). Preserve CC BY4.0 data attribution separately from code licensing.
