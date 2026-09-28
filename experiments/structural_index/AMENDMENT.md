# Representation selection before the fixed confirmation matrix

2026-09-28. This amendment precedes the final seven-model export matrix. The initial
protocol and every screening attempt remain. This is adaptive engineering on known
models, not external preregistration or new accuracy confirmation.

## What the screens rejected

A Python bottom-up tuple index removed recursive AST hashing but made the HAR export
much slower and substantially increased process memory. Native exact child-ID
interning, including a compact open-addressed implementation, also increased memory
and failed to provide a convincing complete-export gain. Intermediate builds and
single-process observations are retained as development screens, not confirmation.
An initial tool timeout interrupted the first screen controller; its completed child
output is retained but lacks the controller's normal completion receipt. It is not
included in the final matrix.

## Selected representation

Instead of storing every structural key, compute a fingerprint bottom-up. Use that
fingerprint ONLY to select cache buckets. Any nonidentical objects with matching
fingerprints undergo an iterative exact structural comparison. Forced collisions
must retain the original equality result. No digest is a proof of equality.

The full-index control memoizes every visited identity. The primary candidate retains
fingerprints only for node types queried by the result cache, plus multiply-owned
nodes. Traversal-added child references are excluded from the retention heuristic.
This prevents exponential expansion of shared diamonds without retaining millions
of one-use leaf/intermediate objects. Reference counts control retention only, never
equality. This is CPython-specific, GIL-held and assumes quiescent, well-formed pinned
ASTs throughout one export. State is discarded between exports. Custom/unsupported
semantics fall back to the original dictionary for the entire cache epoch.

Graph-probe and compatibility evidence justified selecting this representation. The
core was then hardened to own attribute references while traversing them. The final
matrix uses that hardened source; earlier screen times are not relabelled as its
results. The exact same-type stress outputs match the prior exporter, including
structurally equal but separately allocated expressions.

## Fixed final matrix and decision

Three arms: previous PR36 single-lookup guard; full native verified-fingerprint
index; demand-and-alias-aware native verified-fingerprint index. Seven unchanged
frozen SVMs. Three fresh child processes PER arm PER model, including HAR:63 jobs.
Fixed seed2026092801, shuffled before any execution. No selective reruns. Same
assembly + interpretation + result-cache teardown timer; import, model loading,
UTF-8 encoding/output writes are excluded. Record whole-process time and process
high-water memory separately. The native extension's one-time build is separate.

Keep120-second child wall and4-GiB address-space caps, one pinned CPU and one
numerical thread. No other numerical work during measurement. Compare generated
bytes with original frozen small-model C; HAR compares with the previous guard's
completed source because no unmodified stock HAR source exists. Preserve all failures.

Primary claims are evaluated separately: (1) populated-same-type hash traversal
removed in controlled families with byte-identical output; (2) complete real-export
cost versus the previous guard; (3) measured native-index storage and process RAM.
A synthetic improvement is not a universal export improvement; measured regression
or no real-model gain must remain explicit. No inference or model-quality claim.
