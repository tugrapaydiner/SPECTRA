# Sparse-incidence continuation — development checkpoint

Base commit: 8987562356b70a9eb1720c298b509adf8cb739d7 (PR54).
The previous primary MiniCard gate FAILED on both measured hosts. Nothing here
relabels it, changes its frozen inputs, or promotes a model/default/release.

## Hypothesis, not result

The current cover executor materializes variable-pair conflict bitsets and copies
complete search states. Investigate sparse variable/group incidence with reversible
updates, direct native input ingestion, and separately measured preparation versus
search. Require the complete original-constraint-to-checked-answer cost, not just
a faster inner loop. Identify which changes actually cause any gain.

At least MiniCard and GlueCard native-cardinality controls remain. Add original
compiled Knuth exact-cover code on supported workloads and OR-Tools CP-SAT where
applicable; do not call a homemade slow control the strongest alternative.

Sparse sets, reversible trails, Algorithm X and minimum-remaining-values branching
are established ideas. Primary references: Knuth, Dancing Links (2000), original
DLX1 and SSXCC sources at his Stanford programs page, official PySAT solver API,
and official OR-Tools documentation. No algorithmic-first or learned-reasoning
claim is made by planning this implementation.

## Experimental roles

All PR54 puzzles and graphs are exposed development/compatibility data for this
continuation. They must never be described as new confirmation. Development may
also use explicitly identified additional cases, with every consumed case retained.
No confirmation input is generated/opened until candidate, competitors, adapters,
analysis, budgets and complete source identities pass preflight and are frozen in
a public commit. Any post-confirmation algorithm change needs a new study.

Stop or narrow an unsuccessful hypothesis instead of silently moving its gate.
Do not run fresh confirmation merely to improve an unchanged verdict. Independent
external replication, consumer hardware and real application adoption remain open
unless actually observed; CI is not an outside researcher.

## Delivery and continuation

Commit useful code/protocol/evidence checkpoints during work. Preserve incomplete
or rejected attempts and source snapshots. Keep raw rows and machine-readable
source, task and environment identities. No main merge, release, historical evidence
replacement, background assistant process or automatic model promotion is authorized
by this checkpoint. This is an active development branch, not a completed result.
