# Upstream code-generation review — September 27, 2026 (Toronto)

## Decision

Retain the type-aware negative-cache filter and its new single-lookup refinement
as an upstream patch CANDIDATE. The new refinement gives modest extra export
improvements, not a new inference speedup or a higher-quality model. Titanic is
slightly slower than the previous guard. Do not claim universal linear-time
export: the controlled same-type-miss family remains quadratic. No production
SPECTRA runtime, model parameter, default, release version or earlier result changed.

## Mechanism and scope

Five sites across four interpreter files previously performed a membership lookup
followed by a second dictionary lookup on a hit, hashing a structural expression
twice. The refinement uses `get` with a private sentinel. Potentially equal keys
still use normal structural dictionary semantics; equal distinct nodes are not
replaced with identity keys. The type guard skips only impossible negative lookups
for pinned, well-formed exact stock AST types. Custom stored types conservatively
disable it; clear resets it; deletion retains conservative type history.

The prospective protocol is commit00f6796, and the candidate/initial upstream
harness was pushed at031144c. The later patch helper pins validated AST/interpreter
hash pairs and corrects a lookup-count comment (both source versions have five
sites). Its output package bytes match the already measured candidates exactly.
No timed implementation was changed after seeing performance. Earlier local-only
codegen3dbbacf and stream-codec986613a remain separately preserved, not claimed as
merged by this branch.

## New fixed fresh-process generation run

AMD EPYC9V74, one pinned core, Python3.13.5, NumPy2.3.5, sklearn1.8.0,
m2cgen0.10.0. Original seven frozen SVMs, no benchmark fitting or GPU. Each small
model has three fresh processes per variant; HAR has one. All57 jobs were fixed
and shuffled before timing;56 complete and stock HAR reaches120s wall limit.
No rerun or observation was selected to improve the outcome.

Assembly plus interpretation is timed equally for each arm. Import, trusted
pickle loading and output-file writing are outside that timer; total child wall,
CPU time, assembly time and process high-water RSS are separately retained. The
4-GiB address-space cap is not an RSS measurement or deployment-memory claim.

| Model | Stock seconds | Previous guard seconds | Single-lookup seconds | Single / previous guard |
|---|---:|---:|---:|---:|
| Wine |0.033002|0.016285|0.015960|0.980059|
| WDBC |0.135322|0.046734|0.044631|0.954999|
| Chess |4.957376|0.548654|0.516208|0.940862|
| Penguins |0.018316|0.009819|0.009503|0.967807|
| Titanic |1.157695|0.121399|0.121678|1.002294|
| Zoo |0.095791|0.037096|0.034564|0.931721|
| HAR |TIMEOUT|25.483788|23.889896|0.937455|

The incremental change reduces these medians by about2.0–6.8% on six models,
including the single HAR observation. Titanic regresses0.23%. Three processes are
not robust evidence of a universal small percentage benefit; HAR has only one
process per candidate. No statistical generality is asserted.

Combined guard+single versus stock spans1.93–9.60x on the six completed stock
models. Most of that is the PREVIOUS guard, not this round's additional work.
Stock HAR is censored, so no exact HAR stock speedup ratio exists. The completed
HAR candidate sources match each other byte-for-byte; no complete stock HAR
source was generated for a direct full-file comparison. All six smaller-model
outputs match every repetition and their previously frozen C source.

## Broader compatibility

The inspected current upstream default head was9784632311986234032673cdbfd29fc4c5cb429d,
not assumed equal to release0.10.0. Original current tests were acquired as a full
Git archive with commit/tree/SHA256 receipt. The initial local unmodified suite
stopped at collection because the modern local environment lacked an old optional
dependency. That failed attempt is retained; no stub or relaxed test replaced it.

A real GitHub-hosted Python3.10 environment with upstream-compatible dependencies
runs the IDENTICAL original non-e2e suite on stock, previous guard and single:
**592 pass on each, no skips/failures/errors.** Testcase inventories and all71/72
package source files were checked from the downloaded artifacts. This is592
repeated tests, not1776 unique cases. Compiled foreign-language end-to-end tests
are explicitly excluded. It is not upstream maintainers' review or outside replication.

A separate fixed19-estimator panel exercises all16 advertised exporters on release
stock/guard/single and current stock/single. In each source family the **288
successful outputs match byte-for-byte** (18 supported models ×16 languages), and
all16 attempts on a deliberately unsupported MLP retain the same rejection.
Do not call this304 successful exports or actual execution of16 languages. The
small compatibility fixtures consumed0.043 CPU seconds of fitting, with the
unsupported MLP's expected convergence warning retained. No accuracy measured.

All supplied package output hashes and final panel source inventories are checked.
The original current-source patch applies cleanly; its72 resulting package files
match the tested single copy, with24 focused regression tests additionally included.
The complete44-test local suite covers24 cache semantics,7 copy/identity guards and
13 audit fixtures (12 intentionally corrupted receipts rejected). Counts overlap
other differential checks, not new independent benchmark models.

## Controlled hashing work, not timing

For256-term graphs, measured recursive hash invocations are:

| Graph family | Stock | Empty-cache check | Previous guard | Single lookup |
|---|---:|---:|---:|---:|
| No-reuse left-deep chain |66305|0|0|0|
| Expensive reused expression,8 uses |78642|7710|7710|4112|
| Existing same-type cached expression |66831|66309|66054|66051|

All four output sources match on the complete16/32/64/128/256 grids. The empty-cache
check is a cheaper partial alternative, not a solution after populated caches.
The single lookup helps real hits; it does NOT eliminate same-type-miss traversal.
Instrumentation was separate from generation timing. No universal linear bound,
new caching theory, or changed generated-runtime behavior is claimed.

## Evidence and delivery boundaries

The standard-library auditor reconstructs all57 jobs, validates source/model/
package/output identities, retains the timeout and recomputes medians. A separate
comparison checks every generated panel file and original upstream testcase
inventory. These verify recorded bytes/arithmetic, not the truth of clocks or
malicious wholesale replacement of the evidence. Initial metadata-light panel
runs remain alongside final source-bound complete reruns; no performance cells
were replaced. The later copy-helper guard does not alter candidate source bytes.

No new full SPECTRA historical suite, installed runtime acceptance, sanitizer,
Windows/ARM performance or new inference benchmark was run here. Nothing is
submitted to the third-party repository. Current code/test acceptance is recorded
on the SPECTRA review PR; earlier head031144 CI is not relabelled as a later head.

The prior compact-emitter inference regressions, monolithic/partitioned compile
failures and stronger linear HAR task control remain unchanged. This round is a
more thoroughly checked compiler patch candidate, not evidence of high-90s research
impact. External review and broader production workloads are still required.
