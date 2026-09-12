# Compiler baseline protocol

Status at creation: prospective protocol, not an experimental result.
Base: b28e01b85bd95b78f291e466c25c13bfa2fc5b46 (PR27).

## Question

Does the prepared C++/ATen execution advantage survive a functioning PyTorch
Inductor comparator, including CPU max-autotune and frozen weights? Prefer the
best demonstrated implementation over protecting the custom runtime. This is a
systems experiment on previously observed development data, not new capability.

## Fixed workload and controls

Use the existing `scripts.bench_trained_fp.sources()` workload without filtering:
accepted FP32 seeds 1401 and 2402; 128 ordinary and 128 shifted development
puzzles; 256 unique arrays and 512 model/puzzle cases. No training, checkpoint
selection, confirmation/reserve evaluation or puzzle-answer cache is permitted.

Four planned arms: matched-preparation eager, existing PreparedFPSudoku,
Inductor default regional compilation, and Inductor max-autotune regional
compilation with freezing enabled. All use frozen copied FP32 models, fixed
input-independent positions, fresh zero recurrent state, and the original exact
native Sudoku checker. The trained budget is four cycles, with decoding and
verification after each cycle. An independently compiled recurrent cycle plus
output head is an appropriate region: do not give a competitor a disadvantage
by insisting that its Python/C++ checker or data-dependent stopping loop compile.

Attempt a strict full-solve capture separately as a diagnostic; retain an
unsupported-operation failure rather than treating it as proof that regional
compilation cannot work. Regional functions use fullgraph=True and dynamic=False;
no suppressed compiler errors or unreported eager fallback. If compilation fails,
record exception, configuration and elapsed setup cost. A failed arm makes the
planned comparison incomplete; it is not silently omitted as a slow competitor.
A corrected lowering, extra configuration, or scope amendment must be identified
as a separate follow-up, retaining original errors and observations.

The tested environment is the project's PyTorch 2.10.0 CPU stack, not a claim
about the latest compiler. Record precise versions, CPU/ISA, affinity, flags,
compiler counters, cache locations and generated-source identities. Default
Inductor options and max-autotune/freezing options are explicit. Warmup includes
all four recurrent steps so initial-state and recurrent-state specializations are
covered. Report compilation/recompilation counts and reject a supposedly warm
measurement that causes new captured graphs. Never set global MHA dispatch just
to disadvantage the historical reference.

## Two independent correctness tracks

Exact fidelity: compare complete answers, validity and work to the historical
FP reference and compare every full-budget recurrent state/logit bit pattern.
Existing exact APIs and tolerances are unchanged. A compiler that changes bits
is ineligible for an exact claim even when it is faster.

Task-output fidelity: independently verify each answer against original clues,
rows, columns and boxes; record validity, answer/depth/work disagreement,
finite-state checks, maximum absolute state/logit differences, and every lost or
gained valid answer. A bounded output-preservation result requires zero lost
valid answers in this fixed workload, all states finite, and disclosed answer or
stopping-depth changes. This is not a population noninferiority theorem. Do not
reject a faster compiler solely for nonbitwise outputs in this separate track,
and do not silently grant it an exact label.

## Measurements

Single pinned CPU where affinity is available; one intra-op and inter-op thread.
Warm complete request begins with a pretokenized CPU int64 [1,16] input and ends
with the returned answer/work record. It includes input embedding, fresh checker
and state, every executed recurrence and validity check. Additional independent
answer audit, serialization, input creation, compilation and model preparation
are outside warm timing and must be reported separately. No state is shared
between requests. Seven rounds per case, with all four arms randomized in each
round using seed 9161227: 14,336 planned timing rows. Retain every observation,
including failures and regressions. Flush results during the run.

Report sums of per-case medians, all four family/model strata, per-case ratios,
charged total time per verified answer (including failed attempts), descriptive
P95 regressions, and paired problem-cluster bootstrap intervals retaining both
fixed checkpoints. Seven repetitions are not production-tail evidence. Ratios
compare real comparable work; do not multiply these results by PR27 speedups.
No hard speedup acceptance threshold is used to select observations or modes.

Compile in a fresh declared Inductor cache for the main run, with native-library
cache scope separately recorded. Retain construction, first-call compilation,
full-budget warmup, and post-warmup counters. Report amortization under explicitly
stated cached/cold scopes, not an inflated warm-only deployment claim. Compiler
cache source files and identities are retained; generated code is not portable.
No peak-memory or physical-energy advantage is inferred from tensor payload.

## Evidence and promotion

Freeze executable/configuration/fixture digests before measuring. Keep raw rows,
source snapshots, setup/error logs, compiler reports, checksums, quality diagnostics
and exact within-environment replays. Verification must reject missing/duplicate
rows, changed execution order, corrupted original clues, Boolean counters,
changed repeated work, wrong checkpoint/source identity and altered summaries.
Cross-hardware trajectory differences are recorded without weakening verification.

The compiler adapter stays experimental and opt-in. No silent default switch,
main merge, release replacement or historical gate reversal follows from this
protocol. A separate stacked PR contains the experiment; PR27 remains unchanged.

Primary implementation references: PyTorch's torch.compile API, CPU max-autotune
and regional-compilation documentation. Local behavior is verified against the
installed 2.10.0 implementation rather than assumed from newer online manuals.
