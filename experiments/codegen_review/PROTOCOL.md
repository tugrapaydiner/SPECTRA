# Code-generation review: prospective validation scope

Continue from the locally delivered 3dbbacf codegen experiment and merged PR34
b9d9f90d28a3e7071080be38555cd6ca09afcccf. No benchmark-model retraining,
production-inference change, new release or upstream submission is authorized by
this experiment. Earlier compact-emitter regressions and HAR task failure remain.

## Questions

1. Does the output-preserving negative-cache guard retain upstream tests and all
   supported interpreter output fixtures, rather than only five languages?
2. Can a smaller safe cache policy remove the same needless recursive hashing
   without a fragile AST-class whitelist or a measurable cheap-model penalty?
3. Which graphs retain expensive same-type hashing? Do not infer universal
   linear-time export from the earlier no-reuse-chain experiment.

## Inputs and identity

Inspect the live upstream default head, then freeze its complete source at
9784632311986234032673cdbfd29fc4c5cb429d (reported by GitHub before this protocol).
Also retain the original m2cgen0.10.0 wheel and existing seven frozen SVM models.
The source snapshot is for analysis/tests only; installation runs no upstream
setup hooks. Acquire official upstream tests and fixtures, never a guessed rewrite.
Preserve MIT license and original source identity.

## Checks before benchmark promotion

Run original upstream interpreter/AST tests as baseline; record incompatible
optional dependencies separately rather than weakening tests. Run the identical
eligible suite with candidate patches in isolated copies. Compare every supported
language on a fixed estimator panel including SVM, tree/ensemble, linear and MLP
where supported. Unsupported exports and errors remain explicit.

Test structural-equal/distinct nodes, same/different-type hash collisions, custom
cross-type equality, cache mutation paths, reset/reuse between exports and repeated
subexpressions. The original type-guard candidate is retained as a measured arm.
No global mutable AST hash cache or identity-only semantic substitution.

Use controlled AST families to count hashing work, including no-reuse left chains,
reused same-type expressions and vector children. Instrumentation must not occur
inside timing runs. Preserve each candidate rejected on semantic or performance
criteria. Candidate selection on these development probes is disclosed.

## Measurement and limits

Freeze final candidates before their complete timing run. Original six small SVMs
plus HAR use existing model bytes; fixed fresh-process order, three small-model
repetitions and one bounded HAR attempt per arm. Match timer boundary (assembly plus
interpretation, excluding model load and output write); retain total process time
and output hashes. Limit each process to120 seconds and4GiB address space; one
pinned core, no concurrent numerical tasks. Do not rerun timed cells selectively.
For new AST/family regression benchmarks, bound graph/model sizes and publish every
cell. No general model-intelligence, output-runtime or high-90s score claim.

## Deliverable

Publish tested code, upstream-applicable patch and literal validation results on a
SPECTRA review branch. Do not submit to the third-party repository without separate
explicit authorization. No merging of failed/unverified work or rewriting main.
