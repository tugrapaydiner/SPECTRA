# Export-local structural indexing — prospective architecture experiment

Base: SPECTRA afcf25a4d5ca222b44fedb0889d5d6a94ef05314, atop merged main b9d9f90. The user requested an important architectural change rather than another small lookup refinement. This experiment changes the representation used for expression-cache identity, not numerical inference, trained parameters or code-generator arithmetic. No third-party submission or release.

## Hypothesis

Assign structural equivalence classes bottom-up once per export, with child class IDs rather than recursive expression keys. Subsequent cache operations use integer IDs. Preserve exact pinned AST equality and original visit/emission order, including equal-but-distinct expressions and branch-local cache resets. No digest-only equality, permanent hash mutation, object-identity substitution, arithmetic rewriting or unsafe cross-export reuse. Well-formed supported ASTs are stable during an export; detect unsupported/cyclic inputs conservatively and retain explicit fallback or rejection. Identity bookkeeping must own referenced nodes so temporary fallback expressions cannot cause stale-ID collisions.

The technique is related to established hash-consing (Filliatre/Conchon); no new-theory claim. An export-local memoized original-hash implementation is a simpler comparator under the same stable-graph contract. The current single-lookup implementation remains the main production-independent baseline. Account for all index construction, storage and fallback costs.

## Stages and gates

1. Correctness and bounded mechanism pilot: all pinned node types; randomized structural equality; hash collisions; signed zero/nonfinite constants; repeated DAG nodes; separate equivalent trees; custom nodes; mutation between exports; nested branch cache reset; cycle/size rejection. Original generated source must remain identical. Instrument graph work separately from timing. Test no-reuse, real reuse and same-type-polluted-cache chains at 16,32,64,128,256,512,1024 nodes. Do not claim linear total code generation merely from linear cache-index work.
2. Run upstream original non-e2e suite and fixed all-exporter compatibility panel. End-to-end foreign-language execution remains separate. No weakening original assertions.
3. Freeze implementations before one complete fresh-process benchmark: original six small SVMs and HAR from prior evidence, same model bytes, three repetitions for completed candidates, bounded original-HAR negative control. Main comparison includes assembly, structural preparation and interpretation, excludes imports/model load/output write equally. One CPU thread/core, per-process 120-second wall/4-GiB address-space caps; retain failures. Run a broader fixed synthetic non-SVM export panel as a regression control, without model-quality claims.

Promotion requires exact outputs, no missed planned cells, and at least 1.5x geometric mean full-export improvement against the single-lookup baseline on the three previously expensive SVMs (Chess, Titanic, HAR), without >20% median regressions on the other four. Otherwise retain as experimental, never rewrite the gate. Memory, absolute seconds, cheap-model regressions and compilation limitations must be explicit. Upstream original shapes remain unchanged, so prior HAR compiler failures are not solved by this experiment.

Preserve old source/evidence. Publish actual implementation, fixed run receipts, executable tests and honest findings on a separate review branch. No grade increase follows from test counts alone.
