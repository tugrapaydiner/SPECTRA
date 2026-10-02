# Total source-arithmetic trees

Read CONTRACT.md for the precise guarantee. This experiment closes the residual
engine's general-input abstention by retaining lossless original leaves and reusing
routes for source-order fallback. It changes no learned function or production
SPECTRA default. The performance result is roughly tied with the previous complete
pipeline; the main change is that an external CatBoost runtime is unnecessary.

## Native verification and execution

```bash
python -m experiments.tree_total.build --out /new/native --target portable
python -m pytest experiments/tree_total --import-mode=importlib \
  --confcutdir=experiments/tree_total
```

Compilation is explicit and needs a new directory. The Python API takes a
VerifiedTotal(source_json_bytes, compiled_bytes) and a trusted native library.
Compilation uses compiler.compile_bytes(source, maximum, features=..., layout='interned').
Each successful verification reconstructs the proof and every original-leaf byte.
`predict_buffer` accepts writable contiguous uint8 values. `scores` always computes
all original source scores; `policy='exact'` is a forced-fallback control.

```python
from experiments.tree_total.compiler import VerifiedTotal
from experiments.tree_total.session import TotalSession

proof = VerifiedTotal.from_files('model.json', 'model.sctt')
with TotalSession(proof, '/new/native/total.so') as model:
    indices = model.predict_buffer(bytearray([0] * model.features))
```

Indices agree with the specified sequential source arithmetic, not necessarily
with a backend that changes its rounding order at a near tie. This is not a new
accuracy claim. Invalid input fails rather than being snapped onto a domain.

## Supplied replay

The standalone kit includes the four literal source models, inputs, compiled
interned models and both Linux libraries. Run its selftest.py under isolated Python
and -S. The kit loads neither CatBoost nor numerical Python frameworks. Its run.py
publishes a new complete JSONL result outside the immutable bundle. Source proof
verification takes substantial startup work, included in separate measurements.

Before loading a model or library, the runner and self-test check the complete
four-model inventory and every required source, native library, build receipt and
replay file. The files on disk must exactly match the manifest inventory; missing,
unlisted, changed or symlinked members are rejected. The UTF-8 manifest is limited
to 1 MiB, rejects duplicate fields and nonfinite numbers, and validates replay
dimensions and class mappings. Self-test reports must also be new files outside
the kit. An empty or incomplete kit cannot report a successful replay.
Both entry points compare the selected model's feature width, input maximum and
class count with its reconstructed source proof before loading the native model.
A syntactically valid manifest cannot override these source-defined dimensions.

The kit tests include an isolated-Python replay of four tiny synthetic fixtures.
They check assembly, both native targets when supported, prediction output and
unchanged bundle contents. These fixtures do not reproduce the retained benchmark
or establish dataset accuracy.

`independent.py` reconstructs all eight flat/interned artifacts with the preserved
Fraction oracle, independently of the fast dyadic analyzer copied unchanged from
the previous local delivery. `replay.py` compares full source scores with a separate
scalar source reader, preserves any exported-C++ score differences, and executes
a fixed uniform/boundary stress set without model fitting.

Replay acceptance uses explicit errors, so `python -O` cannot disable source-score
or label checks. Prediction counts must match the complete reference inventory;
a matching prefix is insufficient. Failed replay leaves diagnostic partial files
but does not publish its final `report.json` success record.

## Reproducing the comparison

The evidence contains all original model/native-library bytes, original licenses,
initial and resumed benchmark scripts, source snapshots and a PATHS.json relocation
map. Full benchmark dependencies include the previous local residual experiment;
they are under evidence/source/experiments/tree_residual, not part of the new
production module. Use the full supplied research source, or copy that unchanged
baseline folder alongside this checkout before running benchmark.py. This is an
explicit baseline dependency, not a newly implemented competitor.

The measured script's parameters select the prior SDK, prior evidence, residual
SDK/library and new model/library directories. The original absolute paths in
LOCK.json record what ran; PATHS.json binds their portable evidence equivalents.
For a new run, reconstruct these model folders from the map or use the supplied
previous SDK. Never reuse old timing receipts for a new execution.

The complete756-cell experiment fixes four models, nine arms, three batch sizes,
seven repetitions and ten whole jobs per cell. Two tool-command interruptions
preserved exact prefixes and resumed only missing suffixes. No numerical code or
completed observation was replaced. The initial resource prefix was separately
rejected because sitecustomize imported NumPy. The accepted48-process resource
matrix ran under -I -S. Those exclusions are explicitly retained.

`audit.py --root /evidence --out /new/audit.json` checks fixed schedules, source and
artifact identities, source-score labels, proof receipts, preserved prefixes,
resource records and every derived timing ratio without loading a native library.
`negative_audit.py` damages disposable copies only. Neither tool authenticates
clocks or proves outside reproduction. The independent mathematical constructor
is separate from the recorded-data auditor.

The auditor requires all 41 benchmark artifact roles: nine per task and five
shared libraries, using the delivered model/library filename conventions and the
`baseline_sdk` path component. Each role must have a unique binding; deleting an
entry from both the lock and relocation map cannot remove the requirement.
Retained indices are selected from verified bindings, not arbitrary map entries.
The seven original total-tree execution/build sources and both baseline C++
runtimes must be bound in the timing lock; replay must bind those seven total-tree
sources. Extra bound artifacts and sources remain allowed and hash-checked.
These source minima do not certify the full transitive Python dependency closure
or require tests/docs added after the original run. Replay-versus-timing identity
relationships beyond these inventories are not established by this check.

The resource auditor requires the complete four-task, four-policy matrix with
three distinct repeats per cell. It rejects empty, shortened, extended or duplicated
schedules before comparing all 48 literal process receipts. Each receipt must
contain the expected identities and valid setup/memory measurements; empty output
is not a successful measurement. Synthetic regression fixtures exercise this
bookkeeping and the 16 current corruption cases; they are
separate from the 12 historical damaged-copy checks reported in RESULTS.md.

Recorded retained and stress score arrays must contain finite values with exact
declared dimensions. The auditor independently derives first-index argmax labels
and compares them with every stored stress label, in addition to checking hashes.
It also checks integer coverage totals, input domains and the declared fixed
stress settings. These are consistency checks on recorded evidence; the auditor
does not rerun native inference or prove that a recorded score came from its model.

No new learning, world-first mechanism, Windows/ARM, historical full-suite,
production release or external researcher acceptance is claimed. Earlier
classifiers with higher task accuracy remain unchanged.
