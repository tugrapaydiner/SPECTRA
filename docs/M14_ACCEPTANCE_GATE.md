# M14 acceptance gate

**INCOMPLETE_TARGET_MISSED.** The bounded comparison and predeclared correction are finished; the scientific improvement gate failed. M15–M20 have not started.

The protocol required a semantic solve-rate improvement lower bound above five percentage points against both FP32 and mixed INT8, with an upper paired latency ratio at most 1.05 and frozen validation latency caps. Neither attempt passed. All selected neural systems solved zero of the 192 development boards at each of three seeds. The untrained symbolic reference solved all 192.

| Attempt | Ternary blank accuracy | Ternary / FP32 single-pass latency | Ternary / INT8 latency | Gate |
|---|---:|---:|---:|---|
| Initial | 32.77% | 2.672× | 2.466× | failed |
| Blank-only correction | 31.33% | 2.706× | 2.472× | failed |

FP32 single-pass blank accuracy was 56.48% initially and 55.05% after correction. The clue-preserving decoder alone changed no blank predictions and produced no complete solves. This correction did not establish a capability improvement. Ratios use complete-solve means; measured latency includes input/output conversion and independent validation. The small FP32 recursive baseline is also retained in the full table.

## Evidence and scope

- 27 completed fits, three seeds, two initial learning rates, 54 retained checkpoints including optimizer/RNG state; cumulative fitting time 1,490.17 seconds under the 2,700-second cap. Fitting overlapped some tests, so these times are bookkeeping rather than clean training-throughput comparisons.
- 24,192 measured prediction rows independently checked for Sudoku rows, columns, boxes and clues. Three timing rounds collapse to paired-example medians; repeated seeds do not create new puzzles.
- 303 fast tests passed; 16 slow tests were deselected. One existing M10 tensor-to-scalar warning remains. Checkpoint verification covered all 54 hashes and 96 exact prediction replays.
- The 512 confirmation examples remain unevaluated. No confirmation authorization or consumption was created. Zero observed successes and degenerate empirical differences are not population certainty; per-seed Wilson intervals and paired bootstrap/seed-t bounds remain in the evidence. This is a small three-seed pilot.
- Host: AMD EPYC 9V74, Linux CPU, PyTorch 2.8.0+cpu, two intra-op threads. No GPU, measured energy, memory benefit or cross-host performance claim.
- Real x86 dynamic INT8 covered 65.5% of weight elements; attention and other operations remained FP32. It is explicitly a mixed INT8 baseline.
- Native ternary was excluded by frozen numerical fidelity checks: initially 5 of 168 checks failed despite identical decoded answers on all 168 checks. Tiny differences near an A8 rounding boundary changed activation codes. The correction also failed native eligibility. Earlier M10 correctness remains scoped to its tested configuration; it does not certify this 9×9 model.
- Profiling found 26 ternarization calls per solve. Caching effective weights is a future cost hypothesis, not an implemented improvement or a solution to the zero-solve deficit. Profile timings include instrumentation overhead.
- Prior M09 search and M12 learned-control results were negative and used incompatible configurations; they were not presented as successful trained 9×9 baselines.

A final implementation audit tightened enforcement of already frozen absolute latency caps. Original gate files, the two-attempt ledger and the revision record are retained. No measurements were rerun and neither outcome changed. See [execution notes](M14_EXECUTION_NOTES.md).

See [protocol](M14_PROTOCOL.md), [reproduction instructions](M14_REPRODUCE.md), and [raw evidence and tables](../results/m14/README.md). Further improvement needs a newly declared development experiment and independent confirmation; acceptance thresholds must not be weakened to pass this result.
